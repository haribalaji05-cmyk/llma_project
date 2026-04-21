import os
from typing import Optional

import requests

try:
    from indictrans import Transliterator
except ImportError:
    Transliterator = None


LANGUAGE_NAMES = {
    "en": "English",
    "hi": "Hindi",
    "bn": "Bengali",
    "ta": "Tamil",
    "te": "Telugu",
    "kn": "Kannada",
    "ml": "Malayalam",
}


def _build_transliterator(source_lang: str, target_lang: str):
    if Transliterator is None:
        return None
    return Transliterator(source_lang, target_lang)


def _translate_via_api(text: str, source_lang: str, target_lang: str) -> Optional[str]:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        return None

    api_base = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    api_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    source_name = LANGUAGE_NAMES.get(source_lang, source_lang)
    target_name = LANGUAGE_NAMES.get(target_lang, target_lang)
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": api_model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a translation engine.\n"
                    "Translate the user text accurately.\n"
                    "Preserve meaning, service names, URLs, and numbers.\n"
                    "Return only the translated text with no explanation."
                ),
            },
            {
                "role": "user",
                "content": f"Translate this from {source_name} to {target_name}:\n\n{text}",
            },
        ],
        "temperature": 0,
    }

    try:
        response = requests.post(
            f"{api_base}/chat/completions",
            headers=headers,
            json=payload,
            timeout=45,
        )
        response.raise_for_status()
        data = response.json()
        translated = data["choices"][0]["message"]["content"].strip()
        return translated or None
    except Exception:
        return None


def _translate_via_transliterator(text: str, source_lang: str, target_lang: str) -> str:
    translator = _build_transliterator(source_lang, target_lang)
    if translator is None:
        return text

    try:
        return translator.transform(text)
    except Exception:
        return text


def translate_to_english(text: str, source_lang: str = "en") -> str:
    if source_lang == "en" or not text.strip():
        return text

    translated = _translate_via_api(text, source_lang, "en")
    if translated:
        return translated

    return _translate_via_transliterator(text, source_lang, "en")


def translate_from_english(text: str, target_lang: str = "en") -> str:
    if target_lang == "en" or not text.strip():
        return text

    translated = _translate_via_api(text, "en", target_lang)
    if translated:
        return translated

    return _translate_via_transliterator(text, "en", target_lang)
