from langdetect import DetectorFactory, detect

DetectorFactory.seed = 0

LANGUAGE_MAP = {
    "hi": "hi",
    "en": "en",
    "bn": "bn",
    "ta": "ta",
    "te": "te",
    "kn": "kn",
    "ml": "ml",
}

SCRIPT_RANGES = {
    "hi": [(0x0900, 0x097F)],
    "bn": [(0x0980, 0x09FF)],
    "ta": [(0x0B80, 0x0BFF)],
    "te": [(0x0C00, 0x0C7F)],
    "kn": [(0x0C80, 0x0CFF)],
    "ml": [(0x0D00, 0x0D7F)],
}


def _detect_by_script(text: str) -> str:
    for char in text:
        code = ord(char)
        for language, ranges in SCRIPT_RANGES.items():
            if any(start <= code <= end for start, end in ranges):
                return language
    return ""


def detect_language(text: str) -> str:
    script_language = _detect_by_script(text)
    if script_language:
        return script_language

    try:
        lang = detect(text)
        return LANGUAGE_MAP.get(lang, "en")
    except Exception:
        return "en"
