import json
import re
from datetime import date
from pathlib import Path
from typing import Any

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

COMMON_MOJIBAKE = {
    "â€”": "—",
    "â€“": "–",
    "â†’": "→",
    "â‚¹": "₹",
    "â€™": "'",
    "â€œ": '"',
    "â€": '"',
    "â€˜": "'",
    "Â ": " ",
    "Â": "",
}

SERVICE_ALIASES = {
    "passport": ["passport", "passport seva", "psk", "popsk", "tatkal"],
    "aadhaar": ["aadhaar", "aadhar", "uidai", "eid", "baal aadhaar", "myaadhaar"],
    "pf": ["pf", "epf", "epfo", "uan", "form 19", "form 31", "form 10c", "eps"],
    "general": [],
}


def repair_text(text: str) -> str:
    fixed = str(text)
    for broken, replacement in COMMON_MOJIBAKE.items():
        fixed = fixed.replace(broken, replacement)
    fixed = fixed.replace("\r", "")
    fixed = re.sub(r"[ \t]+", " ", fixed)
    fixed = re.sub(r"\n{3,}", "\n\n", fixed)
    return fixed.strip()


def clean_text(text: str) -> str:
    text = repair_text(text)
    bad_patterns = [
        r"^\s*(home|login|register|sign in|skip to|menu|search|copyright|all rights reserved|sitemap|privacy policy|feedback|contact us)\s*$",
        r"^\s*\|+\s*$",
    ]
    lines = []
    for line in text.split("\n"):
        stripped = line.strip()
        if not stripped:
            lines.append("")
            continue
        if any(re.match(pattern, stripped, re.IGNORECASE) for pattern in bad_patterns):
            continue
        lines.append(stripped)
    return repair_text("\n".join(lines))


def split_into_chunks(text: str, chunk_min: int = 100, chunk_max: int = 220) -> list[str]:
    sections = [part.strip() for part in re.split(r"\n\s*(?:={3,}|-{3,})\s*\n", text) if part.strip()]
    paragraphs = []
    for section in sections or [text]:
        paragraphs.extend([p.strip() for p in re.split(r"\n\n+", section) if p.strip()])

    chunks = []
    current = []
    current_words = 0
    for paragraph in paragraphs:
        word_count = len(paragraph.split())
        is_heading = word_count < 14 and not paragraph.endswith(".")
        if is_heading and current:
            chunks.append("\n\n".join(current).strip())
            current = [paragraph]
            current_words = word_count
            continue
        if current and current_words + word_count > chunk_max and current_words >= chunk_min:
            chunks.append("\n\n".join(current).strip())
            current = [paragraph]
            current_words = word_count
        else:
            current.append(paragraph)
            current_words += word_count

    if current:
        chunks.append("\n\n".join(current).strip())
    return [chunk for chunk in chunks if chunk]


def fetch_page(url: str) -> str:
    response = requests.get(url, headers=HEADERS, timeout=20)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form", "noscript"]):
        tag.decompose()
    main = soup.find("main") or soup.find("article") or soup.body or soup
    return clean_text(main.get_text(separator="\n"))


def extract_title(text: str, fallback: str) -> str:
    for line in text.splitlines():
        clean = line.strip()
        if clean:
            return clean[:140]
    return fallback


def infer_topic(title: str, text: str) -> str:
    title_lower = title.lower()
    lowered = text.lower()
    if any(keyword in title_lower for keyword in ["overview", "step", "process", "procedure", "how to"]):
        return "steps"
    if any(keyword in title_lower for keyword in ["required documents", "documents", "proof of"]):
        return "documents"
    if any(keyword in title_lower for keyword in ["fee", "fees", "cost", "charges"]):
        return "fees"
    if any(keyword in title_lower for keyword in ["processing time", "timeline", "timelines"]):
        return "processing_time"
    if "eligib" in title_lower:
        return "eligibility"
    if any(keyword in title_lower for keyword in ["faq", "frequently asked"]):
        return "faq"
    if "processing time" in lowered or "working days" in lowered:
        return "processing_time"
    if "fee" in lowered or "₹" in text or "rs." in lowered:
        return "fees"
    if "step" in lowered or "process" in lowered:
        return "steps"
    if "required documents" in lowered or "proof of" in lowered:
        return "documents"
    if "eligib" in lowered:
        return "eligibility"
    if "faq" in lowered or "frequently asked" in lowered:
        return "faq"
    return "general"


def extract_keywords(service: str, title: str, text: str, topic: str) -> list[str]:
    lowered = f"{title} {text}".lower()
    keywords = set(SERVICE_ALIASES.get(service, []))
    keywords.add(topic)
    if "online" in lowered:
        keywords.add("online")
    if "apply" in lowered or "application" in lowered:
        keywords.add("application")
    if "update" in lowered:
        keywords.add("update")
    if "withdraw" in lowered or "withdrawal" in lowered:
        keywords.add("withdrawal")
    if "track" in lowered or "status" in lowered:
        keywords.add("tracking")
    if "document" in lowered:
        keywords.add("documents")
    if "fee" in lowered or "₹" in text or "rs." in lowered:
        keywords.add("fees")
    return sorted(keyword for keyword in keywords if keyword and keyword != "general")


def build_entries(service: str, state: str, source_name: str, url: str, text: str) -> list[dict[str, Any]]:
    entries = []
    for index, chunk in enumerate(split_into_chunks(text), start=1):
        title = extract_title(chunk, f"{source_name}-{index}")
        topic = infer_topic(title, chunk)
        entries.append(
            {
                "title": title,
                "text": chunk,
                "service": service,
                "state": state,
                "topic": topic,
                "keywords": extract_keywords(service, title, chunk, topic),
                "source_url": url,
                "source_name": source_name,
                "last_updated": str(date.today()),
            }
        )
    return entries


def process_catalog() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    catalog = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    grouped: dict[str, list[dict[str, Any]]] = {"passport": [], "aadhaar": [], "pf": [], "state_services": []}

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
