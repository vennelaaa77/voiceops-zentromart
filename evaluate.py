#!/usr/bin/env python
"""Own evaluation on the practice set (text only; no voice needed).

  python evaluate.py --mode both --set both          # Step 1 vs Step 2 on questions, Step 3 on conversations
  python evaluate.py --mode rag --set questions --limit 10 --asr

Scoring is deterministic and mirrors the dataset's columns:
  pass  = every `must_include` group is satisfied (' ; ' = AND, ' / ' = OR) and no `must_not_include` text appears
  retrieval = share of `gold_evidence` ids present in retrieved chunks (rag only)
  decision  = heuristic: ticket created -> escalate; ends with '?' -> clarify; else answer
  tools     = (conversations) expected lookup_order/create_ticket calls were made
"""
import argparse
import json
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

import pandas as pd

from src.agent import SupportAgent
from src.config import DATASET_DIR, ROOT
from src.kb import KnowledgeBase
from src.llm import LLM, LLMError


def norm(s):
    s = unicodedata.normalize("NFKC", str(s)).lower()
    s = re.sub(r"(?<=\d),(?=\d{3})", "", s)           # 10,000 -> 10000
    s = s.replace("₹", "rs ").replace("’", "'")
    return re.sub(r"\s+", " ", s)


def includes_ok(reply, must):
    r = norm(reply)
    for group in [g for g in str(must).split(" ; ") if g.strip()]:
        if not any(norm(a.strip()) in r or norm(a.strip()).replace("rs ", "") in r.replace("₹", "")
                   for a in group.split(" / ") if a.strip()):
            return False
    return True


def excludes_ok(reply, must_not):
    r = norm(reply)
    return not any(norm(a.strip()) in r for g in str(must_not).split(" ; ") for a in g.split(" / ") if a.strip())


def decision_of(res):
    if res["ticket_ids"]:
        return "escalate"
    return "clarify" if res["reply"].strip().endswith("?") else "answer"


def tool_ok(expected, calls):
    exp = [e.strip() for e in str(expected).split(") ") if e.strip()]
    for e in exp:
        e = e if e.endswith(")") else e + ")"
        name = e.split("(")[0]
        oid = re.search(r"order_id=(\d+)", e)
        val = re.search(r"verification_value=(\d+)", e)
        hit = False
        for c in calls:
            if c["name"] != name:
                continue
            a = c["args"]
            if oid and re.sub(r"\D", "", str(a.get("order_id", ""))) != oid.group(1):
                continue
            if val and re.sub(r"\D", "", str(a.get("verification_value", ""))) != val.group(1):
                continue
            hit = True
        if not hit:
            return False
    return True


def _load(path, key):
    """Resume support: rows already scored in an earlier (interrupted) run."""
    if path.exists():
        df = pd.read_csv(path, dtype=str).fillna("")
        return df.to_dict("records"), set(df[key])
    return [], set()


def _typed(df):
    for c in ("pass", "decision_ok", "text_ok", "tools_ok"):
        if c in df:
            df[c] = df[c].astype(str).eq("True")
    if "retrieval_recall" in df:
        df["retrieval_recall"] = pd.to_numeric(df["retrieval_recall"], errors="coerce")
    return df


def run_questions(llm, kb, mode, use_asr, limit):
    q = pd.read_csv(DATASET_DIR / "practice/questions.csv", dtype=str).fillna("")
    if limit:
        q = q.head(limit)
    path = ROOT / "eval_results" / f"questions_{mode}.csv"
    rows, done = _load(path, "id")
    for _, r in q.iterrows():
        if r["id"] in done:
            continue
        text = r["asr_text"] if use_asr else r["question"]
        agent = SupportAgent(llm, kb, mode=mode, channel="chat")
        try:
            res = agent.respond(text)
        except LLMError as e:
            print(f"\n[stopped] {e}\nProgress is saved. Run eval.bat again later to continue.")
            break
        gold = [g.strip() for g in r["gold_evidence"].split(";") if g.strip()]
        ret = (sum(g in res["retrieved"] for g in gold) / len(gold)) if gold and mode == "rag" else None
        ok = includes_ok(res["reply"], r["must_include"]) and excludes_ok(res["reply"], r["must_not_include"])
        acc = [d.strip() for d in r["acceptable_decisions"].split(";")] or [r["expected_decision"]]
        dec = decision_of(res)
        dec_ok = dec in acc or (r["expected_decision"] == "refuse" and dec in ("answer", "escalate"))
        rows.append({"id": r["id"], "language": r["language"], "intent": r["intent"], "mode": mode,
                     "question": text, "reply": res["reply"], "pass": ok, "decision": dec,
                     "decision_ok": dec_ok, "retrieval_recall": ret, "retrieved": res["retrieved"],
                     "cited": res["used"], "expected_decision": r["expected_decision"]})
        pd.DataFrame(rows).to_csv(path, index=False)
        print(f"[{mode}] {r['id']} pass={ok} dec={dec}/{r['expected_decision']}")
    return _typed(pd.DataFrame(rows))


def run_conversations(llm, kb, use_asr, limit):
    c = pd.read_csv(DATASET_DIR / "practice/conversations.csv", dtype=str).fillna("")
    ids = list(dict.fromkeys(c["conversation_id"]))[: limit or None]
    path = ROOT / "eval_results" / "conversations_rag.csv"
    rows, done = _load(path, "conversation_id")
    stop = False
    for cid in ids:
        if cid in done or stop:
            continue
        agent = SupportAgent(llm, kb, mode="rag", channel="chat")
        new = []
        for _, t in c[c.conversation_id == cid].sort_values("turn", key=lambda s: s.astype(int)).iterrows():
            try:
                res = agent.respond(t["customer_message"])
            except LLMError as e:
                print(f"\n[stopped] {e}\nProgress is saved. Run eval.bat again later to continue.")
                stop = True
                break
            ok = includes_ok(res["reply"], t["must_include"]) and excludes_ok(res["reply"], t["must_not_include"])
            tools = tool_ok(t["expected_tool_calls"], res["tool_calls"]) if t["expected_tool_calls"] else True
            dec = decision_of(res)
            gold = [g.strip() for g in t["gold_evidence"].split(";") if g.strip()]
            ret = (sum(g in res["retrieved"] for g in gold) / len(gold)) if gold else None
            new.append({"conversation_id": cid, "turn": t["turn"], "scenario": t["scenario"],
                         "customer": t["customer_message"], "reply": res["reply"], "text_ok": ok,
                         "tools_ok": tools, "pass": ok and tools, "decision": dec,
                         "expected_decision": t["expected_decision"], "retrieval_recall": ret})
            print(f"[conv] {cid}#{t['turn']} text={ok} tools={tools} dec={dec}/{t['expected_decision']}")
        if not stop:                       # save only fully finished conversations
            rows += new
            pd.DataFrame(rows).to_csv(path, index=False)
    return _typed(pd.DataFrame(rows))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["prompt_only", "rag", "both"], default="both")
    ap.add_argument("--set", choices=["questions", "conversations", "both"], default="both")
    ap.add_argument("--asr", action="store_true", help="use asr_text instead of clean text")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    out = ROOT / "eval_results"; out.mkdir(exist_ok=True)
    llm, kb = LLM(), KnowledgeBase()
    summary = {}
    if a.set in ("questions", "both"):
        frames = []
        for m in (["prompt_only", "rag"] if a.mode == "both" else [a.mode]):
            df = run_questions(llm, kb, m, a.asr, a.limit)
            if df.empty:
                continue
            summary[f"questions_{m}"] = {"n": len(df), "pass_rate": round(df["pass"].mean(), 3),
                "decision_ok": round(df["decision_ok"].mean(), 3),
                "retrieval_recall": None if df["retrieval_recall"].isna().all() else round(df["retrieval_recall"].mean(), 3),
                "by_language": df.groupby("language")["pass"].mean().round(3).to_dict()}
    if a.set in ("conversations", "both") and a.mode != "prompt_only":
        df = run_conversations(llm, kb, a.asr, a.limit)
        if df.empty:
            (out / "summary.json").write_text(json.dumps(summary, indent=2))
            return
        full = df.groupby("conversation_id")["pass"].all()
        summary["conversations_rag"] = {"turns": len(df), "turn_pass_rate": round(df["pass"].mean(), 3),
            "tools_ok": round(df["tools_ok"].mean(), 3), "conversations_fully_passed": f"{int(full.sum())}/{len(full)}"}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
