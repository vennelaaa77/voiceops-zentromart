"""Offline tests (no API keys needed): python -m pytest -q   or   python tests/test_offline.py"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.agent import SupportAgent
from src.kb import KnowledgeBase
from src.textnorm import detect_tts_lang, normalize_asr, speakable, strip_used
from src.tools_layer import ToolBox, add_business_days, business_days_between
from datetime import date


class ScriptedLLM:
    """Plays a fixed list of assistant messages, to exercise the tool loop without a network."""
    def __init__(self, script): self.script, self.seen = list(script), []
    def chat(self, messages, tools=None, **kw):
        self.seen.append(messages); return self.script.pop(0)


def tc(name, **args):
    return {"role": "assistant", "content": "", "tool_calls": [
        {"id": "c1", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}]}


def test_kb_loads():
    kb = KnowledgeBase()
    ids = {c.id for c in kb.chunks}
    assert {"RET-2", "REF-2", "ESC-1", "RETv3-2", "products.csv:ELE-0012"} <= ids
    assert sum(c.doc_type != "data" for c in kb.chunks) >= 80


def test_retrieval_sku_and_precedence():
    kb = KnowledgeBase()
    ids = [c.id for c in kb.search("Is SKU ELE-0012 ZentroMart Assured? how to claim warranty")]
    assert ids[0] == "products.csv:ELE-0012" and "WAR-2" in ids
    ids = [c.id for c in kb.search("How fast are refunds to the ZentroMart Wallet?")]
    assert "REF-2" in ids


def test_business_days():
    assert add_business_days(date(2026, 10, 17), 1) == date(2026, 10, 20)      # Sat -> Tue? (Mon 19 holiday)
    assert business_days_between(date(2026, 10, 17), date(2026, 10, 20)) == 1
    assert business_days_between(date(2026, 10, 1), date(2026, 10, 3)) == 1    # Oct 2 is a holiday


def test_verification_flow_and_secret_scrub():
    tb = ToolBox()
    r = tb.run("lookup_order", {"order_id": "8978 2837"})
    assert r.get("verification_required") and "status" not in r
    assert not tb.run("lookup_order", {"order_id": "89782837", "verification_value": "0000"}).get("found")
    ok = tb.run("lookup_order", {"order_id": "89782837", "verification_value": "4533"})
    assert ok["found"] and ok["verified"] and "derived" in ok
    t = tb.run("create_ticket", {"summary": "Verified caller 4533 reports damaged TV; human review needed.", "order_id": "89782837"})
    assert t["created"]
    import csv
    from src.config import ROOT
    assert "4533" not in (ROOT / "outputs" / "tickets_created.csv").read_text()
    # two failures lock the order
    tb2 = ToolBox()
    for _ in range(2):
        tb2.run("lookup_order", {"order_id": "89782837", "verification_value": "1111"})
    assert tb2.run("lookup_order", {"order_id": "89782837", "verification_value": "4533"}).get("locked")


def test_agent_tool_loop_and_memory():
    llm = ScriptedLLM([
        tc("lookup_order", order_id="89782837"),
        {"role": "assistant", "content": "Hi, I'm ZentroMart's AI assistant. Please share the last 4 digits of your phone.\nUSED: [IDV-2]"},
        tc("lookup_order", order_id="89782837", verification_value="4533"),
        tc("create_ticket", summary="Verified caller wants TV replacement above 10000 rupees; Priority Resolution must decide.", order_id="89782837", reason="high_value_damage"),
        {"role": "assistant", "content": "A colleague will contact you. \nUSED: [DWM-3], [ESC-1]"},
    ])
    a = SupportAgent(llm, KnowledgeBase(), channel="chat")
    r1 = a.respond("Order number hai 8978 2837")
    assert r1["used"] == ["IDV-2"] and "USED" not in r1["reply"]
    r2 = a.respond("Last four digits 4533 hain, replacement approve kar do")
    assert r2["ticket_ids"] and "USED" not in r2["reply"]
    # retrieved docs are not kept in history; tool trail is
    assert all("RETRIEVED DOCUMENTS" not in str(m.get("content")) for m in a.messages if m["role"] == "user")
    assert any(m["role"] == "tool" for m in a.messages)


def test_textnorm():
    s = speakable("**Refund of ₹10,000** for order 89782837 starts 2026-10-24. Ticket TKT" + "a" * 32 + " raised.\nUSED: [REF-2]")
    assert "ten thousand rupees" in s and "8978 2837" in s and "24th of October 2026" in s
    assert "TKT" not in s and "USED" not in s and "*" not in s
    assert "forty-nine rupees" in speakable("Fee is ₹49.")
    assert "24th of October 2026" in speakable("by 24 October")
    assert normalize_asr("sku ele zero zero one to pe") == "sku ELE-0012 pe"
    assert detect_tts_lang("आपका रिफंड 3 दिन में आ जाएगा") == "hi-IN"
    assert detect_tts_lang("Aapka refund 3 din mein aa jayega, kya aur help chahiye") == "hi-en"
    assert detect_tts_lang("Your refund will arrive soon.") == "en-IN"
    assert strip_used("ok\nUSED: none")[1] == []


if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("test_"):
            f(); print("ok", n)
