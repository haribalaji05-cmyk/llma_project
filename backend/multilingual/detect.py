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


def detect_language(text: str) -> str:
    try:
        lang = detect(text)
        return LANGUAGE_MAP.get(lang, "en")
    except Exception:
        return "en"
