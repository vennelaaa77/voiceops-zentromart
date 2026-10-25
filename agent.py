"""ZentroMart support agent.

mode="prompt_only": Step 1 baseline (LLM + system prompt, no documents, no tools).
mode="rag":        Steps 2+3 (retrieval, tools, conversation memory, escalation).
"""
import json
import re
import uuid

from .config import DATASET_DIR, TODAY
from .kb import KnowledgeBase
from .textnorm import normalize_asr, strip_used
from .tools_layer import TOOL_SCHEMAS, ToolBox

BASELINE_PROMPT = (
    "You are ZentroMart's customer support assistant. ZentroMart is an Indian online store selling groceries "
    "and electronics. Answer the customer's question helpfully and briefly (2-3 short sentences). "
    "In your first reply, say you are an AI assistant."
)

RULES = """You are ZentroMart's AI customer-support assistant for an Indian online store (groceries, electronics, fashion, home). Today's date is {today} (use it for every deadline, delay and timeline).

GROUNDING
- Answer ONLY from: (a) the RETRIEVED DOCUMENTS attached to the customer's message, (b) tool results (verified order records), (c) the agent guides below. Never invent a policy, date, amount, fee or timeline.
- If documents disagree, the precedence rule in KB-00.1 decides: current policy > agent guide > help article > FAQ > marketing. Superseded documents apply only to the cases their note names (e.g. check the delivery date).
- If the documents do not cover the question, say so plainly ("Sorry, I don't have information about that") and offer to connect the customer to a human colleague. Do not guess.
- If retrieved documents look insufficient, call search_docs once with better English keywords before giving up.
- Combine sources when needed: policy + catalog row (return_category, assured, fulfilled_by, warranty_months, price) + order record.
- Use date_calc for business-day / calendar arithmetic instead of counting in your head. Return windows count calendar days from the delivery date; refund and logistics timelines count business days (Mon-Sat, minus holidays).

CONVERSATION
- Remember everything the customer already said (order id, product, problem, payment method). Resolve follow-ups like "how long will that take?" from context; do not ask again.
- If a needed detail is missing, ask exactly ONE short question.
- Reply in the customer's language and script: English -> English; Hindi in Devanagari -> Hindi in Devanagari; Romanised Hindi / Hinglish -> Hinglish; follow them if they switch. Other Indian languages: reply in that language.
- Politely refuse anything the policy does not allow (extra compensation, exceptions, approvals you cannot give). Never promise outcomes, amounts or times that a document does not state.

TOOLS
- Before sharing anything about a specific order, verify the caller: ask for the order id, call lookup_order(order_id) (it only returns a verification prompt), then ask for ONE of: last 4 digits of the registered phone, or the delivery pincode. Call lookup_order again with verification_value. Allow one retry on failure; after two failures share no order details. General policy questions need no verification.
- Never say the expected verification value, and never read out a full phone number, email or address.
- Escalate to a human as the Escalation guide (ESC-1) says: first call create_ticket (a complete summary per ESC-3, no verification values), then tell the customer a colleague will contact them. Use the order's contacts_last_7_days for the repeat-contact rule. Hand over when the customer asks for a human.
- NEVER say an action was done (ticket created, order cancelled, refund issued) unless a tool result confirms it. The assistant cannot cancel orders or approve claims itself; guide the customer as the documents say. Only say a ticket exists if create_ticket returned created=true.

STYLE
- First reply of a conversation: say you are ZentroMart's AI support assistant.
- Keep replies to 2-3 short sentences. Answer first. Dates with month names (include the year). Amounts in rupees.
{channel}
- At the very end, on its own last line, write: USED: [ID], [ID] listing the document section ids you relied on (e.g. USED: [REF-2], [KB-00.3]) or USED: none. This line is stripped before the customer sees the reply.

AGENT GUIDES (from the company knowledge base)
{guides}
"""

CHANNEL = {
    "chat": "- Plain text; no markdown tables.",
    "voice": "- This reply will be SPOKEN on a phone call: no lists, no markdown, no emoji, short sentences. Read order ids in groups of four digits. Do not read ticket ids aloud; just say a ticket has been created.",
}


def _guides():
    out = []
    for name in ("00_about_this_knowledge_base.md", "agent_conversation_style.md",
                 "agent_escalation_handoff.md", "agent_identity_verification.md"):
        txt = (DATASET_DIR / "kb" / name).read_text(encoding="utf-8")
        txt = re.sub(r"^---.*?---\s*", "", txt, flags=re.S)
        out.append(txt.strip())
    return "\n\n".join(out)


class SupportAgent:
    def __init__(self, llm, kb=None, mode="rag", channel="chat", max_steps=6):
        self.llm, self.mode, self.channel, self.max_steps = llm, mode, channel, max_steps
        self.kb = kb if kb is not None else (KnowledgeBase() if mode == "rag" else None)
        self.tb = ToolBox(self.kb)
        self.user_history, self.first = [], True
        sys_prompt = (BASELINE_PROMPT if mode == "prompt_only" else
                      RULES.format(today=TODAY.strftime("%d %B %Y"), channel=CHANNEL[channel], guides=_guides()))
        self.messages = [{"role": "system", "content": sys_prompt}]

    def reset(self):
        self.__init__(self.llm, self.kb, self.mode, self.channel, self.max_steps)

    # ------------------------------------------------------------------
    def respond(self, user_text: str) -> dict:
        text = normalize_asr(user_text.strip())
        result = {"reply": "", "used": [], "retrieved": [], "tool_calls": [], "ticket_ids": []}
        if self.mode == "prompt_only":
            msgs = self.messages + [{"role": "user", "content": text}]
            reply = self.llm.chat(msgs, max_tokens=300)["content"]
            self.messages = msgs + [{"role": "assistant", "content": reply}]
            result["reply"], _ = strip_used(reply)
            return result

        chunks = self.kb.search(text, history=self.user_history)
        result["retrieved"] = [c.id for c in chunks]
        ctx = "\n\n".join(c.render() for c in chunks)
        note = ("\n(This is your first reply: introduce yourself as ZentroMart's AI support assistant.)"
                if self.first else "")
        turn_msg = {"role": "user", "content":
                    f"{text}\n\n---\nRETRIEVED DOCUMENTS for this message (answer only from these, tool results and "
                    f"the guides):\n{ctx}{note}"}
        work = self.messages + [turn_msg]
        new_msgs, n_calls0 = [], len(self.tb.calls)
        final = None
        for _ in range(self.max_steps):
            msg = self.llm.chat(work + new_msgs, tools=TOOL_SCHEMAS, max_tokens=700)
            calls = msg.get("tool_calls") or []
            if not calls:
                final = msg["content"]
                new_msgs.append({"role": "assistant", "content": final})
                break
            for c in calls:
                c.setdefault("id", "call_" + uuid.uuid4().hex[:8])
                c.setdefault("type", "function")
            new_msgs.append(msg)
            for c in calls:
                try:
                    args = json.loads(c["function"].get("arguments") or "{}")
                except json.JSONDecodeError:
                    args = {}
                out = self.tb.run(c["function"]["name"], args)
                new_msgs.append({"role": "tool", "tool_call_id": c["id"],
                                 "content": json.dumps(out, ensure_ascii=False, default=str)})
        if not final:
            new_msgs.append({"role": "user", "content": "Give the customer your final reply now (no more tool calls)."})
            final = self.llm.chat(work + new_msgs, max_tokens=500)["content"]
            new_msgs.pop()
            new_msgs.append({"role": "assistant", "content": final})

        reply, used = strip_used(final)
        # persist: plain customer text (without the bulky retrieved docs) + this turn's tool trail + reply
        self.messages += [{"role": "user", "content": text}] + new_msgs[:-1] + [{"role": "assistant", "content": reply}]
        self.user_history.append(text)
        self.first = False
        turn_calls = self.tb.calls[n_calls0:]
        extra = [i for c in turn_calls if c["name"] == "search_docs" for i in []]
        result.update(reply=reply, used=used,
                      retrieved=result["retrieved"] + [i for i in self.tb.retrieved_extra if i not in result["retrieved"]],
                      tool_calls=turn_calls,
                      ticket_ids=[c["result"].get("ticket_id") for c in turn_calls
                                  if c["name"] == "create_ticket" and c["result"].get("created")])
        self.tb.retrieved_extra = []
        return result
