"""Central config. Secrets come from the environment / .env, never from code."""
import os
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:  # python-dotenv is optional
    pass


def _find_dataset() -> Path:
    env = os.getenv("DATASET_DIR")
    cands = [Path(env)] if env else []
    cands += [ROOT / "Capstone_Project_Dataset_FTTT_2026", ROOT / "dataset",
              ROOT.parent / "Capstone_Project_Dataset_FTTT_2026"]
    for c in cands:
        if (c / "kb").is_dir() and (c / "data").is_dir():
            return c.resolve()
    raise FileNotFoundError(
        "Dataset not found. Unzip Capstone_Project_Dataset_FTTT_2026.zip next to this "
        "README (or set DATASET_DIR in .env).")


DATASET_DIR = _find_dataset()
TODAY = date(2026, 10, 20)          # fixed dataset reference date (KB-00.2)

# --- LLM (OpenRouter, free models only) ---
OPENROUTER_BASE = os.getenv("OPENROUTER_BASE", "https://openrouter.ai/api/v1")
OPENROUTER_MODELS = [m.strip() for m in os.getenv("OPENROUTER_MODELS", "").split(",") if m.strip()]

# --- Gnani voice ---
TTS_MODEL = os.getenv("TTS_MODEL", "timbre-v2.5")
TTS_VOICE = os.getenv("TTS_VOICE", "Kaveri")
STT_LANG = os.getenv("STT_LANG", "en-IN")   # en-IN also accepts English+Hindi mix
