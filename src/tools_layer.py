"""Tool layer between the LLM and the course-supplied mock tools.

Wraps tools.lookup_order / tools.create_ticket and *enforces in code* the rules the
documents set (verify before sharing, one retry, no secrets in tickets), adds joined
catalog facts and pre-computed date facts, and exposes small helpers the LLM is bad at
(business-day arithmetic).
"""
import csv
import json
import re
import sys
from datetime import date, datetime, timedelta

from .config import DATASET_DIR, TODAY

sys.path.insert(0, str(DATASET_DIR))
from tools import tools as _mock                # noqa: E402  (course-supplied module)
from .config import ROOT                        # noqa: E402

_mock._TICKETS = str(ROOT / "outputs" / "tickets_created.csv")   # keep generated files out of the dataset
(ROOT / "outputs").mkdir(exist_ok=True)


def _rows(name, key):
    with open(DATASET_DIR / "data" / name, encoding="utf-8", newline="") as f:
        return {r[key]: r for r in csv.DictReader(f)}


PRODUCTS = _rows("products.csv", "sku")
PINCODES = _rows("pincodes.csv", "pincode")
HOLIDAYS = {date.fromisoformat(d) for d in _rows("holidays.csv", "date")}


def is_business_day(d: date) -> bool:
    return d.weekday() != 6 and d not in HOLIDAYS          # Mon-Sat, minus holidays (KB-00.3)


def add_business_days(start: date, n: int) -> date:
    d = start
    while n > 0:
        d += timedelta(days=1)
        if is_business_day(d):
            n -= 1
    return d


def business_days_between(start: date, end: date) -> int:
    """Business days after `start` up to and including `end` (0 if end <= start)."""
    d, c = start, 0
    while d < end:
        d += timedelta(days=1)
        c += is_business_day(d)
    return c


def _d(s):
    if not s:
        return None
    return date.fromisoformat(str(s)[:10])


TOOL_SCHEMAS = [
    {"type": "function", "function": {
        "name": "lookup_order",
        "description": "Look up an order. Call with order_id alone first (returns a verification prompt only). "
                       "After the caller gives ONE verification value (last 4 digits of registered phone OR "
                       "6-digit delivery pincode), call again with verification_value. Order details are "
                       "returned only after verification succeeds.",
        "parameters": {"type": "object", "properties": {
            "order_id": {"type": "string", "description": "8-digit order id, digits only"},
            "verification_value": {"type": "string", "description": "4-digit phone suffix or 6-digit pincode"}},
            "required": ["order_id"]}}},
    {"type": "function", "function": {
        "name": "create_ticket",
        "description": "Create a human-handover ticket. Call BEFORE telling the customer a colleague will "
                       "contact them. Summary must include: the issue, detected intent, order id if any, what the "
                       "customer already said, what you checked, and the reason for escalation. Never include "
                       "verification values.",
        "parameters": {"type": "object", "properties": {
            "summary": {"type": "string"}, "order_id": {"type": "string"},
            "reason": {"type": "string", "description": "short code, e.g. high_value_damage, repeat_contact"},
            "intent": {"type": "string"}}, "required": ["summary"]}}},
    {"type": "function", "function": {
        "name": "date_calc",
        "description": "Exact date arithmetic. Business days = Mon-Sat excluding warehouse holidays. "
                       "ops: add_business_days(date,n) -> date n business days after; "
                       "business_days_between(date,end_date) -> business days after date up to and including end_date; "
                       "add_calendar_days(date,n); days_between(date,end_date). Dates are YYYY-MM-DD.",
        "parameters": {"type": "object", "properties": {
            "op": {"type": "string", "enum": ["add_business_days", "business_days_between",
                                              "add_calendar_days", "days_between"]},
            "date": {"type": "string"}, "n": {"type": "integer"}, "end_date": {"type": "string"}},
            "required": ["op", "date"]}}},
    {"type": "function", "function": {
        "name": "search_docs",
        "description": "Search ZentroMart's policy documents, product catalog and pincode table with your own "
                       "query (use English keywords). Use when the documents already shown do not cover the question.",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
]


class ToolBox:
    """One instance per conversation: holds verification state."""

    def __init__(self, kb=None):
        self.kb = kb
        self.verified = {}          # order_id -> cached verified record
        self.attempts = {}          # order_id -> failed verification attempts
        self.secrets = set()
        self.calls = []             # all calls this conversation
        self.tickets = []
        self.retrieved_extra = []   # chunk ids returned through search_docs

    # ---------- dispatch ----------
    def run(self, name, args):
        fn = getattr(self, f"_t_{name}", None)
        try:
            out = fn(**args) if fn else {"error": f"unknown tool {name}"}
        except TypeError as e:
            out = {"error": f"bad arguments: {e}"}
        except Exception as e:  # never crash the conversation on a tool bug
            out = {"error": f"{e.__class__.__name__}: {e}"}
        self.calls.append({"name": name, "args": args, "result": self._safe(out)})
        return out

    @staticmethod
    def _safe(out):   # what is logged: never persist verified order contents or secrets
        if isinstance(out, dict) and out.get("verified") and out.get("found"):
            return {"found": True, "verified": True, "order_id": out.get("order_id")}
        return out

    # ---------- tools ----------
    def _t_lookup_order(self, order_id, verification_value=None):
        oid = re.sub(r"\D", "", str(order_id))
        v = None if verification_value in (None, "") else re.sub(r"\D", "", str(verification_value))
        if oid in self.verified and not v:
            return self.verified[oid]
        if v is None:
            return _mock.lookup_order(oid)
        if self.attempts.get(oid, 0) >= 2:
            return {"verified": False, "locked": True,
                    "message": "Verification failed twice. Share no order details. Offer general policy help and "
                               "suggest contacting support from the registered phone number (IDV-3)."}
        res = _mock.lookup_order(oid, verification_value=v)
        if res.get("verified") and res.get("found"):
            self.secrets.update({v})
            rec = self._enrich(res)
            self.verified[oid] = rec
            return rec
        self.attempts[oid] = self.attempts.get(oid, 0) + 1
        left = 2 - self.attempts[oid]
        res["attempts_left"] = left
        res["message"] = ("Verification failed. " + ("One retry allowed: ask for the other value or re-check."
                                                     if left else "No retries left: share no order details."))
        return res

    def _enrich(self, o):
        o = {k: v for k, v in o.items() if v is not None}
        items = []
        for it in o.get("items", []):
            p = PRODUCTS.get(it.get("sku"), {})
            it = {k: v for k, v in it.items() if v is not None}
            it.update({"fulfilled_by": p.get("fulfilled_by"), "assured": p.get("assured"),
                       "warranty_months": p.get("warranty_months")})
            items.append(it)
        o["items"] = items
        pin = PINCODES.get(o.get("delivery_pincode"))
        if pin:
            o["delivery_area"] = {k: pin[k] for k in ("city", "metro", "cod_available", "express_available")}
        sh, dl, pr = _d(o.get("shipped_on")), _d(o.get("delivered_on")), _d(o.get("promised_date"))
        d = {"today": TODAY.isoformat()}
        if dl:
            d["days_since_delivered"] = (TODAY - dl).days
        if sh:
            d["days_since_shipped"] = (TODAY - sh).days
        if pr:
            ref = dl or TODAY
            d["days_past_promised_date"] = max((ref - pr).days, 0)
        for pre in ("return_refund", "rto_refund", "cancellation_refund"):
            ini = _d(o.get(f"{pre}_initiated_on"))
            if ini:
                d[f"business_days_since_{pre}_initiated"] = business_days_between(ini, TODAY)
        if o.get("return_pickup_on"):
            d["business_days_since_return_pickup"] = business_days_between(_d(o["return_pickup_on"]), TODAY)
        o["derived"] = d
        return o

    def _t_create_ticket(self, summary, order_id=None, reason=None, intent=None):
        for s in self.secrets:                      # IDV-4: never log verification values
            summary = summary.replace(s, "[redacted]")
        res = _mock.create_ticket(summary=summary, order_id=order_id, reason=reason, intent=intent)
        if res.get("created"):
            self.tickets.append(res["ticket_id"])
        return res

    def _t_date_calc(self, op, date, n=None, end_date=None):
        a, b = _d(date), _d(end_date) if end_date else None
        if op == "add_business_days":
            return {"result": add_business_days(a, int(n)).isoformat()}
        if op == "add_calendar_days":
            return {"result": (a + timedelta(days=int(n))).isoformat()}
        if op == "business_days_between":
            return {"result": business_days_between(a, b)}
        if op == "days_between":
            return {"result": (b - a).days}
        return {"error": "unknown op"}

    def _t_search_docs(self, query):
        if not self.kb:
            return {"error": "no knowledge base"}
        chunks = self.kb.search(query, k=5)
        self.retrieved_extra += [c.id for c in chunks]
        return {"documents": [c.render() for c in chunks]}
