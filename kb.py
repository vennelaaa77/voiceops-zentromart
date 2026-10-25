"""Knowledge base: chunk docs by [SECTION-ID], index with BM25 (+ optional dense), retrieve.

Chunks come from kb/*.md sections, plus product-catalog rows and pincode rows so that
catalog facts are retrievable and citable (ids like 'products.csv:ELE-0012').
"""
import csv
import math
import os
import re
from collections import Counter
from dataclasses import dataclass, field

from .config import DATASET_DIR

SECTION_RE = re.compile(r"^## \[([^\]]+)\]\s*(.*)$")
SKU_RE = re.compile(r"\b([A-Z]{3}-\d{4})\b")
PIN_RE = re.compile(r"(?<!\d)(\d{6})(?!\d)")

# precedence from KB-00.1: policy > agent guide > help > faq > marketing; superseded last
TYPE_WEIGHT = {"policy": 1.15, "agent_guide": 1.0, "meta": 0.9, "help": 0.95,
               "faq": 0.85, "marketing": 0.7, "data": 1.0}

# Light query expansion so casual / Hinglish / Hindi wording reaches the documents'
# vocabulary. This only widens retrieval; it contains no answers.
SYNONYMS = {
    "send it back": "return", "send back": "return", "sent back": "return", "give back": "return",
    "wapas": "return", "vapas": "return", "लौटा": "return", "वापस": "return", "रिटर्न": "return",
    "money back": "refund", "paisa": "refund", "paise": "refund", "पैसे": "refund",
    "रिफंड": "refund", "रिफ़ंड": "refund", "rifund": "refund",
    "kitne din": "days window", "कितने दिन": "days window", "kab tak": "how long timeline",
    "कब तक": "how long timeline", "kitna time": "how long timeline", "how long": "timeline days",
    "cancel": "cancel cancellation", "कैंसिल": "cancel cancellation", "radd": "cancel",
    "broken": "damaged defective", "torn": "damaged", "cracked": "damaged", "faulty": "defective damaged",
    "kharab": "damaged defective", "खराब": "damaged defective", "टूटा": "damaged", "toota": "damaged",
    "galat": "wrong item", "गलत": "wrong item", "missing": "missing item",
    "late": "late delay delivery", "delay": "late delivery", "der": "late delivery", "देर": "late delivery",
    "nobody came": "reverse pickup missed", "didn't come": "reverse pickup missed",
    "not come": "reverse pickup missed", "pickup": "reverse pickup",
    "human": "human colleague handover escalate", "insaan": "human colleague", "इंसान": "human colleague",
    "talk to a person": "human colleague handover", "real person": "human colleague handover",
    "unauthorised": "unauthorised activity escalate", "unauthorized": "unauthorised activity escalate",
    "didn't order": "unauthorised activity escalate", "did not order": "unauthorised activity escalate",
    "fraud": "unauthorised activity escalate", "court": "legal consumer court escalate",
    "police": "legal police escalate", "lawyer": "legal escalate", "legal": "legal escalate",
    "guarantee": "warranty", "गारंटी": "warranty", "वारंटी": "warranty", "assured": "assured warranty",
    "fee": "fee charge", "charge": "fee", "free": "free fee waived", "muft": "free", "मुफ़्त": "free",
    "wallet": "wallet instant", "cod": "cash on delivery", "cash on delivery": "cod",
    "कैश ऑन डिलीवरी": "cash on delivery cod", "parcel": "order delivery", "package": "order delivery",
    "exchange": "exchange size colour", "coupon": "coupon discount", "festive": "festive sale",
    "compensation": "compensation late delivery", "sale": "festive sale",
    "express": "express delivery", "plus": "plus membership", "swelling": "safety battery escalate",
    "burning": "safety escalate", "never placed": "unauthorised activity escalate",
    "never ordered": "unauthorised activity escalate", "didn't place": "unauthorised activity escalate",
    "not placed": "unauthorised activity escalate", "someone else": "another customer details never share",
    "phone number of": "another customer details never share", "already on its way": "shipped after ship cancel",
    "on its way": "shipped cancel", "शिप": "shipped after ship cancel", "stop it": "cancel shipped",
    "doesn't fit": "return change of mind size", "opened": "opened unopened seal return condition",
    "tried it": "opened unopened seal return condition", "kab tak aa": "refund timeline business days",
    "kab tak": "refund timeline business days", "by which date": "refund timeline business days",
    "warranty": "warranty assured brand", "delivered on": "delivery date return window",
    "parso": "delivered return window", "received": "delivered damaged", "smoke": "safety escalate", "sparks": "safety escalate",
}

_STOP = set("""a an the is are was were be been am i you your my me we our it its this that these those
to of in on at for from by with and or but if then so as do does did can could would should will shall
what when where which who how why not no yes please hi hello hey ok okay thanks thank
hai hain ka ki ke ko se me mein pe par kya aur ya to mujhe aap hum kaise kitne tha thi the
है हैं का की के को से में पर और या क्या मुझे आप हम कैसे कितने था थी थे""".split())


def stem(w: str) -> str:
    if len(w) > 5:
        for suf in ("ing", "ed", "es"):
            if w.endswith(suf):
                return w[: -len(suf)]
    if len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
        return w[:-1]
    return w


def tokenize(text: str):
    toks = re.findall(r"[\w₹]+(?:[-.][\w]+)*", text.lower())
    return [stem(t) for t in toks if t not in _STOP and len(t) > 0]


def expand_query(q: str) -> str:
    low = q.lower()
    extra = [v for k, v in SYNONYMS.items() if k in low]
    return q + " " + " ".join(extra)


@dataclass
class Chunk:
    id: str
    text: str
    doc_id: str
    title: str
    doc_type: str
    status: str = "current"
    effective_from: str = ""
    source: str = ""
    tokens: list = field(default_factory=list, repr=False)

    def render(self) -> str:
        meta = f"{self.title} | {self.doc_type} | status={self.status}"
        if self.effective_from:
            meta += f" | effective_from={self.effective_from}"
        return f"[{self.id}] ({meta})\n{self.text.strip()}"


def _parse_front_matter(raw: str):
    meta, body = {}, raw
    if raw.startswith("---"):
        end = raw.find("\n---", 3)
        if end != -1:
            for line in raw[3:end].strip().splitlines():
                if ":" in line:
                    k, v = line.split(":", 1)
                    meta[k.strip()] = v.strip()
            body = raw[end + 4:]
    return meta, body


def load_chunks():
    chunks = []
    for path in sorted((DATASET_DIR / "kb").glob("*.md")):
        meta, body = _parse_front_matter(path.read_text(encoding="utf-8"))
        note = " ".join(l[1:].strip() for l in body.splitlines() if l.startswith(">"))
        cur_id, cur_title, buf = None, "", []

        def flush():
            if cur_id:
                text = "\n".join(buf).strip()
                if note:
                    text = f"NOTE: {note}\n{text}"
                chunks.append(Chunk(
                    id=cur_id, text=f"{cur_title}\n{text}", doc_id=meta.get("doc_id", path.stem),
                    title=meta.get("title", path.stem), doc_type=meta.get("doc_type", "policy"),
                    status=meta.get("status", "current"),
                    effective_from=meta.get("effective_from", ""), source=path.name))

        for line in body.splitlines():
            m = SECTION_RE.match(line)
            if m:
                flush()
                cur_id, cur_title, buf = m.group(1), m.group(2), []
            elif cur_id:
                buf.append(line)
        flush()

    def rows(name):
        with open(DATASET_DIR / "data" / name, encoding="utf-8", newline="") as f:
            return list(csv.DictReader(f))

    for r in rows("products.csv"):
        txt = "; ".join(f"{k}={v}" for k, v in r.items() if v != "")
        chunks.append(Chunk(id=f"products.csv:{r['sku']}", text=f"Product catalog row: {txt}",
                            doc_id="products.csv", title="Product catalog", doc_type="data",
                            source="products.csv"))
    for r in rows("pincodes.csv"):
        txt = "; ".join(f"{k}={v}" for k, v in r.items() if v != "")
        chunks.append(Chunk(id=f"pincodes.csv:{r['pincode']}",
                            text=f"Pincode table row: {txt}", doc_id="pincodes.csv",
                            title="Serviceable pincodes", doc_type="data", source="pincodes.csv"))
    for c in chunks:
        c.tokens = tokenize(c.text + " " + c.title)
    return chunks


class KnowledgeBase:
    def __init__(self, use_dense=None):
        self.chunks = load_chunks()
        self.by_id = {c.id: c for c in self.chunks}
        self._build_bm25()
        self.dense = None
        if use_dense is None:
            use_dense = os.getenv("USE_EMBEDDINGS", "0") == "1"
        if use_dense:
            self._try_dense()

    # ---- BM25 ----
    def _build_bm25(self, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.N = len(self.chunks)
        self.avgdl = sum(len(c.tokens) for c in self.chunks) / max(self.N, 1)
        df = Counter()
        for c in self.chunks:
            df.update(set(c.tokens))
        self.idf = {t: math.log(1 + (self.N - n + 0.5) / (n + 0.5)) for t, n in df.items()}
        self.tf = [Counter(c.tokens) for c in self.chunks]

    def _bm25(self, query: str):
        q = tokenize(expand_query(query))
        scores = []
        for i, c in enumerate(self.chunks):
            tf, dl, s = self.tf[i], len(c.tokens), 0.0
            for t in q:
                if t in tf:
                    f = tf[t]
                    s += self.idf.get(t, 0) * f * (self.k1 + 1) / (
                        f + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
            w = TYPE_WEIGHT.get(c.doc_type, 1.0)
            if c.status == "superseded":
                w *= 0.6
            scores.append(s * w)
        return scores

    # ---- optional dense (needs `pip install fastembed`; downloads a model on first use) ----
    def _try_dense(self):
        try:
            import numpy as np
            from fastembed import TextEmbedding
            name = os.getenv("EMBED_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
            self._emb = TextEmbedding(model_name=name)
            vecs = np.array(list(self._emb.embed([c.text for c in self.chunks])))
            self.dense = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
            self._np = np
        except Exception as e:  # fall back silently to BM25
            print(f"[kb] dense retrieval disabled ({e.__class__.__name__}: {e})")
            self.dense = None

    def _dense_scores(self, query):
        v = self._np.array(list(self._emb.embed([query])))[0]
        v = v / self._np.linalg.norm(v)
        return list(self.dense @ v)

    # ---- public ----
    def _rank(self, query, history, pool, n):
        """Fuse BM25 (+dense) rankings restricted to chunks whose index is in `pool`."""
        ranked = {}

        def add(scores, weight=1.0):
            order = sorted(pool, key=lambda i: -scores[i])[:15]
            for r, i in enumerate(order):
                if scores[i] > 0:
                    ranked[i] = ranked.get(i, 0) + weight / (60 + r)   # reciprocal-rank fusion

        add(self._bm25(query))
        if history:
            add(self._bm25(" ".join(history[-2:] + [query])), 0.6)
        if self.dense is not None:
            add(self._dense_scores(query), 1.0)
        return [self.chunks[i] for i in sorted(ranked, key=lambda i: -ranked[i])][:n]

    @staticmethod
    def _product_hints(prod):
        """Words that link a catalog row to the policies that apply to it (two-source answers)."""
        f = dict(p.split("=", 1) for p in prod.text.replace("Product catalog row: ", "").split("; ") if "=" in p)
        hints = [f.get("return_category", ""), "return window warranty"]
        if f.get("assured") == "True":
            hints.append("assured")
        if f.get("fulfilled_by", "ZentroMart") != "ZentroMart":
            hints.append("marketplace seller")
        try:
            if int(f.get("price_inr", 0)) > 10000:
                hints.append("high-value 10000 Priority Resolution handover")
        except ValueError:
            pass
        return " ".join(hints)

    def search(self, query: str, k: int = 8, history=None):
        """Top-k chunks: exact ids -> catalog/pincode rows -> policy sections (+ companions)."""
        history = history or []
        data_idx = [i for i, c in enumerate(self.chunks) if c.doc_type == "data"]
        doc_idx = [i for i, c in enumerate(self.chunks) if c.doc_type != "data"]

        # 1) catalog / pincode rows
        data = []
        for sku in SKU_RE.findall(query):
            if f"products.csv:{sku}" in self.by_id:
                data.append(self.by_id[f"products.csv:{sku}"])
        for pin in PIN_RE.findall(query):
            if f"pincodes.csv:{pin}" in self.by_id:
                data.append(self.by_id[f"pincodes.csv:{pin}"])
        for c in self._rank(query, None, data_idx, 3):
            # keep a fuzzy product match only when the title clearly matches
            if c not in data and self._title_overlap(query, c) >= 0.6:
                data.append(c)
        data = data[:2]

        # 2) policy sections; enrich the query with what the matched product implies
        q2 = query
        for c in data:
            if c.id.startswith("products.csv:"):
                q2 += " " + self._product_hints(c)
        docs = self._rank(q2, history, doc_idx, k - len(data))

        # 3) companion sections: timelines need the business-day rule; conflicting
        #    docs (FAQ / marketing / superseded) need the precedence rule
        extra = []
        if any("business day" in c.text.lower() for c in docs) and "KB-00.3" in self.by_id:
            extra.append(self.by_id["KB-00.3"])
        if any(c.doc_type in ("faq", "marketing") or c.status == "superseded" for c in docs):
            extra.append(self.by_id["KB-00.1"])
        out = []
        for c in data + docs + extra:
            if c not in out:
                out.append(c)
        return out[: k + 1]

    @staticmethod
    def _title_overlap(query, chunk):
        m = re.search(r"title=([^;]+)", chunk.text)
        if not m:
            return 0.0
        tt = set(tokenize(m.group(1)))
        qt = set(tokenize(query))
        return len(tt & qt) / max(len(tt), 1)
