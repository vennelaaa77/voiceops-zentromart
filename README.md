# ZentroMart VoiceOps: support agent that listens and talks

Pipeline: **customer audio → Gnani STT → agent (LLM + RAG + tools) → Gnani TTS → spoken reply.**
Speech uses **Gnani only**. The LLM is any free model on **OpenRouter** (auto-discovered, with fallback).
No answers are hardcoded: replies come from the retrieved documents, catalog rows and tool results.

## Setup
```bash
git clone <this repo> && cd <repo>
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env        # then paste OPENROUTER_API_KEY and GNANI_API_KEY (never commit .env)
unzip Capstone_Project_Dataset_FTTT_2026.zip            # next to this README (or set DATASET_DIR)
python Capstone_Project_Dataset_FTTT_2026/validate_dataset.py
```
## Run
| What | Command |
| --- | --- |
| **Batch run on a folder of audio** (organizers) | `./run.sh path/to/audio_folder [out_dir]`  or  `python run.py --audio-dir path/to/audio_folder --out sample_outputs` |
| Own evaluation, Step 1 vs Step 2 + conversations | `python evaluate.py --mode both --set both` |
| Voice demo page (record + hear reply) | `python demo_app.py` → http://127.0.0.1:5000 |
| Offline tests (no keys) | `python tests/test_offline.py` |

`run.py` output per file: `<name>.reply.wav` (speech), `<name>.reply.txt`, plus `results.jsonl` / `results.csv` with
transcript, reply, **retrieved and cited document chunk ids** and tool calls. Files named `<conv>_<n>.wav` are treated
as turns of one conversation, other files as separate calls. Unclear or silent audio gets "Sorry, I didn't catch that."

## How it works
- **Step 1 (`mode=prompt_only`)**: LLM + system prompt only (baseline for the comparison).
- **Step 2 RAG** (`src/kb.py`): `kb/*.md` split by `[SECTION-ID]` (id, status, effective_from kept) + product and pincode
  rows. Hybrid retrieval: BM25 with Hindi/Hinglish query expansion, optional multilingual embeddings
  (`USE_EMBEDDINGS=1`, needs `pip install fastembed`), exact SKU/pincode matching, a product→policy hop (category, assured,
  price), precedence weighting (policy > guide > help > FAQ > marketing; superseded down-weighted), and companion rules
  (timelines add the business-day section; FAQ/marketing hits add the precedence section).
- **Step 3 conversation** (`src/agent.py`, `src/tools_layer.py`): full history incl. tool results, one question at a time,
  follow-ups, polite refusals, escalation via `create_ticket`. Rules enforced **in code**: no order details before
  verification, one retry then lock, verification values scrubbed from tickets and logs, "ticket created" only after
  `created=True`. Helper tool `date_calc` does business-day maths (Mon–Sat minus holidays, today = 20 Oct 2026).
- **Step 4 voice** (`src/voice.py`, `src/textnorm.py`): STT `en-IN` (handles Hinglish; other languages via `--stt-lang`),
  clips over 55 s split automatically; replies rewritten for the ear (₹ → words, order ids in groups of four, dates with
  month and year, no markdown, ticket ids not read out); TTS language picked from the reply (script / Hinglish);
  model `timbre-v2.5`, voice `Kaveri` (override in `.env`).

## Notes
- Free OpenRouter models are rate-limited (about 20 requests/minute; 50/day unless the account has ≥$10 credit, then 1000/day).
  The client throttles and falls back across free models automatically.
- Tickets are written to `outputs/tickets_created.csv`. Secrets live only in `.env`.
