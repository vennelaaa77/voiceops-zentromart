#!/usr/bin/env python
"""Batch run: python run.py --audio-dir ./audio --out ./sample_outputs

For every audio file: Gnani STT -> agent (RAG + tools) -> Gnani TTS.
Saves per file: <stem>.reply.txt, <stem>.reply.wav, and one results.jsonl / results.csv with
transcript, reply, retrieved + cited document chunks and tool calls.
Files named <conv>_<n>.<ext> are treated as turns of one conversation (ordered by n);
all other files are independent single-turn calls.
"""
import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

from src.agent import SupportAgent
from src.kb import KnowledgeBase
from src.llm import LLM

EXTS = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".webm", ".aac", ".opus"}
RECOVER = "Sorry, I didn't catch that. Could you say it again?"


def plan(audio_dir: Path):
    files = sorted(p for p in audio_dir.iterdir() if p.suffix.lower() in EXTS)
    groups, order = defaultdict(list), []
    for p in files:
        m = re.match(r"^(.+?)[_-](\d+)$", p.stem)
        key = m.group(1) if m else p.stem
        if key not in groups:
            order.append(key)
        groups[key].append((int(m.group(2)) if m else 0, p))
    return [(k, [p for _, p in sorted(groups[k])]) for k in order]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio-dir", required=True)
    ap.add_argument("--out", default="sample_outputs")
    ap.add_argument("--stt-lang", default=None, help="Gnani language_code (default en-IN, also handles Hinglish)")
    ap.add_argument("--no-tts", action="store_true", help="skip speech output (text only)")
    a = ap.parse_args()

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    from src.voice import Voice
    voice = Voice()
    llm, kb = LLM(), KnowledgeBase()
    rows = []
    for key, paths in plan(Path(a.audio_dir)):
        agent = SupportAgent(llm, kb, mode="rag", channel="voice")      # fresh conversation per group
        for p in paths:
            row = {"file": p.name, "transcript": "", "reply": "", "spoken_reply": "", "reply_audio": "",
                   "retrieved_chunks": [], "cited_chunks": [], "tool_calls": [], "error": ""}
            try:
                row["transcript"] = voice.transcribe(p, a.stt_lang)
                if not row["transcript"].strip():
                    row["reply"] = RECOVER                                   # unclear / empty audio
                else:
                    r = agent.respond(row["transcript"])
                    row.update(reply=r["reply"], retrieved_chunks=r["retrieved"], cited_chunks=r["used"],
                               tool_calls=[{"name": c["name"], "args": c["args"]} for c in r["tool_calls"]])
            except Exception as e:                                           # never crash the batch
                row["error"] = f"{e.__class__.__name__}: {e}"
                row["reply"] = ("Sorry, I'm having trouble right now. I can connect you to a colleague who can help.")
            wav = out / f"{p.stem}.reply.wav"
            if not a.no_tts:
                row["spoken_reply"] = voice.speak(row["reply"], wav)
                row["reply_audio"] = wav.name
            (out / f"{p.stem}.reply.txt").write_text(
                f"TRANSCRIPT: {row['transcript']}\nREPLY: {row['reply']}\n"
                f"RETRIEVED: {', '.join(row['retrieved_chunks'])}\nCITED: {', '.join(row['cited_chunks'])}\n",
                encoding="utf-8")
            rows.append(row)
            print(f"[{p.name}] {row['transcript'][:60]!r} -> {row['reply'][:70]!r}")
    (out / "results.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
    with open(out / "results.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["file"])
        w.writeheader()
        for r in rows:
            w.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v for k, v in r.items()})
    print(f"Done: {len(rows)} file(s) -> {out}")


if __name__ == "__main__":
    main()
