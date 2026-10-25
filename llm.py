"""OpenRouter client restricted to FREE models, with automatic fallback across them.

No key is stored in code: set OPENROUTER_API_KEY. If OPENROUTER_MODELS is empty the
list of free tool-capable models is discovered from OpenRouter's public /models endpoint.
"""
import json
import os
import re
import time

import requests

from . import config

# preference order by substring; anything else free + tool-capable follows by context size
PREFERRED = ["gpt-oss-120b", "llama-3.3-70b", "qwen3-235b", "qwen3-next", "deepseek", "gemma-3-27b",
             "mistral-small", "glm", "qwen", "llama", "gpt-oss"]
CACHE = config.ROOT / ".cache" / "free_models.json"


class LLMError(RuntimeError):
    pass


class LLM:
    def __init__(self, models=None, timeout=60, min_interval=3.2):
        gem = os.getenv("GEMINI_API_KEY", "")
        self.provider = "gemini" if gem and os.getenv("LLM_PROVIDER", "gemini") == "gemini" else "openrouter"
        if self.provider == "gemini":      # Google AI Studio free tier, OpenAI-compatible endpoint
            self.key, self.base = gem, "https://generativelanguage.googleapis.com/v1beta/openai"
            env_models = [m.strip() for m in os.getenv("GEMINI_MODELS", "").split(",") if m.strip()]
            self.models = models or env_models or self._discover_gemini()
            min_interval = max(min_interval, 4.5)           # free tier ~15 requests/minute
        else:
            self.key = os.getenv("OPENROUTER_API_KEY", "")
            self.base = config.OPENROUTER_BASE
            if not self.key:
                raise LLMError("Set GEMINI_API_KEY or OPENROUTER_API_KEY in .env")
            self.models = models or config.OPENROUTER_MODELS or self._discover()
        self.timeout = timeout
        self.min_interval = min_interval
        self._last = 0.0
        self.log = []                             # one entry per call: model, seconds, usage

    def _headers(self):
        return {"Authorization": f"Bearer {self.key}", "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/voiceops-capstone", "X-Title": "ZentroMart VoiceOps"}

    def _discover_gemini(self):
        """Ask Google which models this key can use; pick current text 'flash' models."""
        try:
            r = requests.get(f"{self.base}/models", headers=self._headers(), timeout=30)
            r.raise_for_status()
            ids = [m["id"].replace("models/", "") for m in r.json().get("data", [])]
        except Exception as e:
            raise LLMError(f"Could not list Gemini models: {e}")
        bad = ("image", "tts", "live", "audio", "embed", "robotics", "computer", "vision", "learnlm", "gemma", "preview", "exp", "thinking")
        flash = [i for i in ids if "flash" in i and not any(b in i for b in bad)]
        lite = sorted([i for i in flash if "lite" in i], reverse=True)
        full = sorted([i for i in flash if "lite" not in i], reverse=True)
        picks = (full[:2] + lite[:2]) or [i for i in ids if "gemini" in i][:3]
        if not picks:
            raise LLMError(f"No usable Gemini models for this key. Available: {ids[:10]}")
        print(f"[llm] using Gemini models: {picks}")
        return picks

    def _discover(self):
        if CACHE.exists() and time.time() - CACHE.stat().st_mtime < 86400:
            return json.loads(CACHE.read_text())
        r = requests.get(f"{self.base}/models", headers=self._headers(), timeout=30)
        r.raise_for_status()
        free = []
        for m in r.json()["data"]:
            p = m.get("pricing", {})
            if str(p.get("prompt")) in ("0", "0.0") and str(p.get("completion")) in ("0", "0.0") \
                    and "tools" in (m.get("supported_parameters") or []):
                free.append((m["id"], m.get("context_length") or 0))

        def rank(item):
            mid = item[0].lower()
            pref = next((i for i, s in enumerate(PREFERRED) if s in mid), len(PREFERRED))
            return (pref, -item[1])
        ids = [i for i, _ in sorted(free, key=rank)][:8]
        if not ids:
            raise LLMError("No free tool-capable models found on OpenRouter right now.")
        CACHE.parent.mkdir(exist_ok=True)
        CACHE.write_text(json.dumps(ids))
        return ids

    def chat(self, messages, tools=None, temperature=0.2, max_tokens=700):
        """Return the assistant message dict. Tries each free model; backs off on 429."""
        last_err = None
        for attempt in range(2):
            for model in self.models:
                wait = self.min_interval - (time.time() - self._last)
                if wait > 0:
                    time.sleep(wait)
                body = {"model": model, "messages": messages, "temperature": temperature,
                        "max_tokens": max_tokens}
                if tools:
                    body["tools"] = tools
                    body["tool_choice"] = "auto"
                t0 = time.time()
                try:
                    r = requests.post(f"{self.base}/chat/completions",
                                      headers=self._headers(), json=body, timeout=self.timeout)
                    self._last = time.time()
                    if r.status_code in (429, 502, 503, 504, 408):
                        last_err = f"{model}: HTTP {r.status_code}"
                        time.sleep(2 + 4 * attempt)
                        continue
                    if r.status_code >= 400:
                        last_err = f"{model}: HTTP {r.status_code} {r.text[:200]}"
                        continue
                    r.raise_for_status()
                    data = r.json()
                    if "error" in data:
                        last_err = f"{model}: {data['error']}"
                        continue
                    msg = data["choices"][0]["message"]
                    content = msg.get("content") or ""
                    content = re.sub(r"(?s)<think>.*?</think>", "", content).strip()
                    if not content and not msg.get("tool_calls"):
                        last_err = f"{model}: empty reply"
                        continue
                    msg = {"role": "assistant", "content": content,
                           **({"tool_calls": msg["tool_calls"]} if msg.get("tool_calls") else {})}
                    self.log.append({"model": model, "sec": round(time.time() - t0, 2),
                                     "usage": data.get("usage", {})})
                    return msg
                except (requests.RequestException, KeyError, ValueError) as e:
                    last_err = f"{model}: {e.__class__.__name__}: {e}"
            time.sleep(5)
        raise LLMError(f"All free models failed. Last error: {last_err}")
