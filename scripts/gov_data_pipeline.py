import json
import re
from pathlib import Path
from typing import Dict, List

import requests
from bs4 import BeautifulSoup


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = PROJECT_ROOT / "data" / "source_catalog.json"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed_chunks"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
}


def clean_text(text: str) -> str:
    text = text.replace("\r", "")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def split_into_chunks(text: str, chunk_min: int = 140, chunk_max: int = 260) -> List[str]:
    paragraphs = [part.strip() for part in re.split(r"\n\n+", text) if part.strip()]
    chunks: List[str] = []
    current: List[str] = []
    current_words = 0

    for paragraph in paragraphs:
        count = len(paragraph.split())
        if current and current_words + count > chunk_max and current_words >= chunk_min:
            chunks.append("\n\n".join(current))
            current = [paragraph]
            current_words = count
        else:
            current.append(paragraph)
            current_words += count

    if current:
        chunks.append("\n\n".join(current))
    return [chunk.strip() for chunk in chunks if chunk.strip()]


def fetch_page(url: str) -> str:
    response = requests.get(url, headers=HEADERS, timeout=20)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form", "noscript"]):
        tag.decompose()
    main = soup.find("main") or soup.body or soup
    return clean_text(main.get_text(separator="\n"))


def infer_topic(text: str) -> str:
    lowered = text.lower()
    if "required documents" in lowered or "proof of" in lowered:
        return "documents"
    if "fee" in lowered or "rs." in lowered or "₹" in text:
        return "fees"
    if "processing time" in lowered or "working days" in lowered:
        return "processing_time"
    if "step" in lowered or "process" in lowered:
        return "steps"
    if "faq" in lowered or "frequently asked" in lowered:
        return "faq"
    return "general"


def build_entries(service: str, state: str, source_name: str, url: str, text: str) -> List[Dict[str, str]]:
    entries = []
    for chunk in split_into_chunks(text):
        title = chunk.splitlines()[0].strip() if chunk.splitlines() else source_name
        entries.append(
            {
                "title": title,
                "text": chunk,
                "service": service,
                "state": state,
                "topic": infer_topic(chunk),
                "source_url": url,
                "source_name": source_name,
            }
        )
    return entries


def process_catalog() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    catalog = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    grouped: Dict[str, List[Dict[str, str]]] = {"passport": [], "aadhaar": [], "pf": [], "state_services": []}

    for service, sources in catalog.get("national", {}).items():
        for source in sources:
            try:
                text = fetch_page(source["url"])
                grouped.setdefault(service, []).extend(
                    build_entries(service, "national", source["label"], source["url"], text)
                )
                print(f"[OK] {service}: {source['label']}")
            except Exception as exc:
                print(f"[WARN] Failed {service}: {source['url']} -> {exc}")

    for state, sources in catalog.get("states", {}).items():
        for source in sources:
            try:
                text = fetch_page(source["url"])
                grouped["state_services"].extend(
                    build_entries("general", state.replace("_", " "), source["label"], source["url"], text)
                )
                print(f"[OK] {state}: {source['label']}")
            except Exception as exc:
                print(f"[WARN] Failed {state}: {source['url']} -> {exc}")

    for name, entries in grouped.items():
        if not entries:
            continue
        output_path = OUTPUT_DIR / f"{name}.json"
        output_path.write_text(json.dumps(entries, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[SAVED] {output_path} ({len(entries)} chunks)")


if __name__ == "__main__":
    process_catalog()
