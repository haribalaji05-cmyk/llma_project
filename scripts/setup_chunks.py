import json
import re
from datetime import date
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = Path(r"C:\Users\harir\Downloads\llma_chunks")
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed_chunks"

SERVICE_ALIASES = {
    "passport": ["passport", "passport seva", "psk", "popsk", "tatkal"],
    "aadhaar": ["aadhaar", "aadhar", "uidai", "eid", "baal aadhaar", "myaadhaar"],
    "pf": ["pf", "epf", "epfo", "uan", "form 19", "form 31", "form 10c", "eps"],
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


def repair_text(text: str) -> str:
    fixed = str(text)
    for broken, replacement in COMMON_MOJIBAKE.items():
        fixed = fixed.replace(broken, replacement)
    fixed = fixed.replace("\r", "")
    fixed = re.sub(r"[ \t]+", " ", fixed)
    fixed = re.sub(r"\n{3,}", "\n\n", fixed)
    return fixed.strip()


def extract_title(text: str, fallback: str) -> str:
    for line in text.splitlines():
        clean = line.strip()
        if clean:
            return clean[:140]
    return fallback


def extract_source_url(text: str) -> str:
    match = re.search(r"https?://[^\s)]+", text)
    return match.group(0) if match else ""


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


def normalize_chunks(service: str, chunks: list[Any]) -> list[dict[str, Any]]:
    entries = []
    for index, chunk in enumerate(chunks, start=1):
        text = repair_text(chunk)
        if not text:
            continue
        title = extract_title(text, f"{service} chunk {index}")
        topic = infer_topic(title, text)
        entries.append(
            {
                "title": title,
                "text": text,
                "service": service,
                "state": "national",
                "topic": topic,
                "keywords": extract_keywords(service, title, text, topic),
                "source_url": extract_source_url(text),
                "source_name": f"{service}.json",
                "last_updated": str(date.today()),
            }
        )
    return entries


def write_outputs(service: str, entries: list[dict[str, Any]]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = OUTPUT_DIR / f"{service}.json"
    txt_path = OUTPUT_DIR / f"{service}.txt"

    json_path.write_text(json.dumps(entries, ensure_ascii=False, indent=2), encoding="utf-8")
    separator = "\n\n" + ("=" * 80) + "\n\n"
    txt_path.write_text(separator.join(entry["text"] for entry in entries), encoding="utf-8")
    print(f"[OK] Wrote {json_path} ({len(entries)} metadata chunks)")
    print(f"[OK] Wrote {txt_path}")


def main() -> None:
    for service in ("passport", "aadhaar", "pf"):
        source_path = SOURCE_DIR / f"{service}.json"
        if not source_path.exists():
            print(f"[SKIP] Missing source file: {source_path}")
            continue
        chunks = json.loads(source_path.read_text(encoding="utf-8"))
        entries = normalize_chunks(service, chunks)
        write_outputs(service, entries)


if __name__ == "__main__":
    main()
