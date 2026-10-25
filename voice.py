"""Gnani STT / TTS wrappers (the ONLY speech services used). Key: GNANI_API_KEY."""
import math
import os
import shutil
import subprocess
import tempfile
import wave
from pathlib import Path

from . import config
from .textnorm import detect_tts_lang, speakable

CLIP_SECONDS = 55     # REST STT accepts clips up to 60 s


class Voice:
    def __init__(self):
        if not os.getenv("GNANI_API_KEY"):
            raise RuntimeError("GNANI_API_KEY is not set (see .env.example)")
        from gnani.stt import GnaniSTTClient      # pip install gnani-vachana
        from gnani.tts import GnaniTTSClient
        self.stt, self.tts = GnaniSTTClient(), GnaniTTSClient()

    # ---- audio prep: wav pieces of <= 55 s ----
    @staticmethod
    def _to_wav(path: Path, tmp: Path) -> Path:
        if path.suffix.lower() == ".wav":
            return path
        if not shutil.which("ffmpeg"):
            return path           # let the API try the original container
        out = tmp / (path.stem + ".wav")
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(path), "-ar", "16000", "-ac", "1", str(out)],
                       check=True)
        return out

    @staticmethod
    def _split(path: Path, tmp: Path):
        try:
            with wave.open(str(path)) as w:
                rate, n, ch, sw = w.getframerate(), w.getnframes(), w.getnchannels(), w.getsampwidth()
                if n / rate <= CLIP_SECONDS:
                    return [path]
                step, parts = CLIP_SECONDS * rate, []
                for i in range(math.ceil(n / step)):
                    w.setpos(i * step)
                    frames = w.readframes(step)
                    p = tmp / f"{path.stem}_part{i}.wav"
                    with wave.open(str(p), "wb") as o:
                        o.setnchannels(ch); o.setsampwidth(sw); o.setframerate(rate); o.writeframes(frames)
                    parts.append(p)
                return parts
        except wave.Error:
            return [path]

    @staticmethod
    def _normalize(path: Path, tmp: Path) -> Path:
        """Boost quiet 16-bit wav recordings (laptop mics are often very low)."""
        import array
        try:
            with wave.open(str(path)) as w:
                params, frames = w.getparams(), w.readframes(w.getnframes())
            if params.sampwidth != 2:
                return path
            a = array.array("h", frames)
            peak = max((abs(x) for x in a), default=0)
            if peak == 0 or peak >= 8000:
                return path
            gain = min(20000 / peak, 40.0)
            a = array.array("h", (max(-32768, min(32767, int(x * gain))) for x in a))
            out = tmp / (path.stem + "_norm.wav")
            with wave.open(str(out), "wb") as o:
                o.setparams(params); o.writeframes(a.tobytes())
            return out
        except (wave.Error, OSError):
            return path

    def transcribe(self, audio_path, language_code=None) -> str:
        """Return the transcript ('' if nothing was heard or the call failed)."""
        language_code = language_code or config.STT_LANG
        text = []
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            for part in self._split(self._normalize(self._to_wav(Path(audio_path), tmp), tmp), tmp):
                for attempt in range(2):
                    try:
                        res = self.stt.transcribe(str(part), language_code=language_code)
                        text.append(res["transcript"])
                        if not str(res["transcript"]).strip():
                            print(f"[stt] empty transcript for {part.name}: {str(res)[:200]}")
                        break
                    except Exception as e:
                        if attempt:
                            print(f"[stt] failed on {part.name}: {e}")
        return " ".join(t.strip() for t in text if t and t.strip())

    def speak(self, text: str, out_file, lang=None) -> str:
        """Normalise `text` for the ear and synthesise it to out_file. Returns the spoken text."""
        lang = lang or detect_tts_lang(text)
        spoken = speakable(text, lang)
        for attempt in range(2):
            try:
                self.tts.synthesize(spoken, voice=config.TTS_VOICE, model=config.TTS_MODEL,
                                    language=lang, output_file=str(out_file))
                break
            except Exception as e:
                if attempt:
                    print(f"[tts] failed: {e}")
                    return spoken
        return spoken
