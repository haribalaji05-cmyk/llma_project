import os

try:
    from indictrans import Transliterator
except ImportError:
    Transliterator = None


def _build_transliterator(source_lang: str, target_lang: str):
    if Transliterator is None:
        return None
    return Transliterator(source_lang, target_lang)


def translate_to_english(text: str, source_lang: str = "en") -> str:
    if source_lang == "en":
        return text

    translator = _build_transliterator(source_lang, "en")
    if translator is None:
        return text

    try:
        return translator.transform(text)
    except Exception:
        return text


def translate_from_english(text: str, target_lang: str = "en") -> str:
    if target_lang == "en":
        return text

    translator = _build_transliterator("en", target_lang)
    if translator is None:
        return text

    try:
        return translator.transform(text)
    except Exception:
        return text
