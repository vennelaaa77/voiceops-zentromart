"""Text helpers: make replies speakable (STY-3 / Gnani text-normalization guide),
clean ASR output, detect the reply language for TTS."""
import re
from datetime import date

from .config import TODAY

MONTHS = "January|February|March|April|May|June|July|August|September|October|November|December"
_ONES = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen " \
        "sixteen seventeen eighteen nineteen".split()
_TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()


def number_to_words(n: int) -> str:
    """Indian-style number words, e.g. 49 -> 'forty-nine', 10000 -> 'ten thousand'."""
    if n < 20:
        return _ONES[n]
    if n < 100:
        return _TENS[n // 10] + ("-" + _ONES[n % 10] if n % 10 else "")
    if n < 1000:
        return _ONES[n // 100] + " hundred" + (" " + number_to_words(n % 100) if n % 100 else "")
    for div, name in ((10_000_000, "crore"), (100_000, "lakh"), (1000, "thousand")):
        if n >= div:
            return number_to_words(n // div) + " " + name + (" " + number_to_words(n % div) if n % div else "")
    return str(n)


def ordinal(d: int) -> str:
    suf = "th" if 10 <= d % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(d % 10, "th")
    return f"{d}{suf}"


def strip_used(text: str):
    """Split the model's reply from its trailing 'USED: [ID], [ID]' line."""
    used = []
    m = re.search(r"(?im)^\s*USED\s*:\s*(.*)$", text)
    if m:
        used = [u.strip(" []`") for u in re.split(r"[,\s]+(?=\S)", m.group(1)) if u.strip(" []`")]
        used = [u for u in used if u.lower() != "none"]
        text = text[: m.start()] + text[m.end():]
    return text.strip(), used


_DEV = re.compile(r"[ऀ-ॿ]")
_SCRIPTS = {  # unicode block -> Gnani language code
    "bn-IN": r"[ঀ-৿]", "pa-IN": r"[਀-੿]", "gu-IN": r"[઀-૿]",
    "ta-IN": r"[஀-௿]", "te-IN": r"[ఀ-౿]", "kn-IN": r"[ಀ-೿]",
    "ml-IN": r"[ഀ-ൿ]", "hi-IN": r"[ऀ-ॿ]",
}
_HINGLISH = set("hai hain kya aap aapka aapki nahi nahin karein karna kijiye mein mera meri mujhe "
                "kab tak kitne din paisa paise wapas shukriya dhanyavaad abhi ho gaya gayi order ka ki ke "
                "se ko ji haan".split())


def detect_tts_lang(text: str) -> str:
    """Gnani TTS language code for a reply: by script, else Hinglish vs English."""
    best, n = None, 0
    for code, pat in _SCRIPTS.items():
        c = len(re.findall(pat, text))
        if c > n:
            best, n = code, c
    if best and n >= 3:
        return best
    words = re.findall(r"[a-z]+", text.lower())
    hits = sum(w in _HINGLISH for w in words)
    return "hi-en" if words and hits / len(words) >= 0.12 and hits >= 2 else "en-IN"


def _group(digits: str, n=4) -> str:
    return " ".join(digits[i:i + n] for i in range(0, len(digits), n))


def speakable(text: str, lang: str = "en-IN", voice_ticket: bool = True) -> str:
    """Rewrite a chat reply so TTS reads it well."""
    t, _ = strip_used(text)
    t = re.sub(r"[*_`#>]+", "", t)
    t = re.sub(r"^\s*[-•]\s+", "", t, flags=re.M)
    t = re.sub(r"[\U0001F000-\U0001FAFF☀-➿]", "", t)          # emoji
    if voice_ticket:  # a 35-character ticket id is noise on a call
        t = re.sub(r"(?i)\b(ticket(?: id| number)?(?: is| :|:)?\s*)?TKT[0-9a-f]{8,}\b", "", t)
        t = re.sub(r"\s+\(\s*\)", "", t)

    def money(m):
        n = int(m.group(1).replace(",", ""))
        return f"{n} रुपये" if lang in ("hi-IN",) else f"{number_to_words(n)} rupees"
    t = re.sub(r"(?:₹|Rs\.?|INR)\s*([\d,]+)(?:\.\d+)?", money, t)
    t = re.sub(r"([\d,]+)\s*(?:₹|rupees)\b", lambda m: money(m) if "," in m.group(1) or m.group(1).isdigit() else m.group(0), t)

    # dates: ISO and "24 October [2026]" -> "24th of October 2026"
    def iso(m):
        y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        return f"{ordinal(d)} of {date(y, mo, 1).strftime('%B')} {y}"
    t = re.sub(r"\b(20\d\d)-(\d\d)-(\d\d)\b", iso, t)
    t = re.sub(rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+({MONTHS})(?:\s+(20\d\d))?",
               lambda m: f"{ordinal(int(m.group(1)))} of {m.group(2)} {m.group(3) or TODAY.year}", t)
    t = re.sub(rf"\b({MONTHS})\s+(\d{{1,2}})(?:st|nd|rd|th)?(?:,?\s+(20\d\d))?\b",
               lambda m: f"{ordinal(int(m.group(2)))} of {m.group(1)} {m.group(3) or TODAY.year}", t)

    # long ids (order 8 digits, refund refs 12 digits): groups of four
    t = re.sub(r"(?<![\d.,])(\d{8}|\d{12})(?![\d])", lambda m: _group(m.group(1)), t)
    t = re.sub(r"(?<=\d)\s*%", " percent", t)
    t = re.sub(r"\s*\n+\s*", ". ", t)
    t = re.sub(r"\.\s*\.", ".", t)
    return re.sub(r"\s{2,}", " ", t).strip()


_W2D = {"zero": "0", "oh": "0", "one": "1", "two": "2", "to": "2", "too": "2", "three": "3", "four": "4",
        "for": "4", "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9"}
_NUMWORDS = "|".join(sorted(_W2D, key=len, reverse=True))


def normalize_asr(text: str) -> str:
    """Fix ASR habits that break ids: 'ele zero zero one to' -> 'ELE-0012'; spoken digit runs -> digits."""
    def sku(m):
        digits = "".join(_W2D[w] for w in re.findall(_NUMWORDS, m.group(2).lower()))
        trail = " " if m.group(2).endswith((" ", "-")) else ""
        return f"{m.group(1).upper()}-{digits}{trail}" if len(digits) == 4 else m.group(0)
    t = re.sub(rf"\b([A-Za-z]{{3}})[-\s]((?:(?:{_NUMWORDS})\b[\s-]*){{4}})", sku, text)

    def run(m):  # 4+ spoken digits in a row (not a lone 'to'/'for')
        words = re.findall(_NUMWORDS, m.group(0).lower())
        if len(words) >= 4 and sum(w not in ("to", "too", "for", "oh") for w in words) >= 3:
            return "".join(_W2D[w] for w in words) + (" " if m.group(0).endswith((" ", "-")) else "")
        return m.group(0)
    return re.sub(rf"\b(?:(?:{_NUMWORDS})\b[\s-]*){{4,}}", run, t)
