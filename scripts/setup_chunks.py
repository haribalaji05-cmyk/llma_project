import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = Path(r"C:\Users\harir\Downloads\llma_chunks")
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed_chunks"


def convert_chunks(service: str, chunks: list[str]) -> list[dict]:
    entries = []
    for index, chunk in enumerate(chunks, start=1):
        text = str(chunk).strip()
        if not text:
            continue
        first_line = text.splitlines()[0].strip() if text.splitlines() else f"{service} chunk {index}"
        topic = "general"
        lowered = text.lower()
        if "required documents" in lowered or "proof of" in lowered:
            topic = "documents"
        elif "fee" in lowered or "rs." in lowered or "₹" in text:
            topic = "fees"
        elif "processing time" in lowered or "working days" in lowered:
            topic = "processing_time"
        elif "step" in lowered or "process" in lowered:
            topic = "steps"
        elif "faq" in lowered or "frequently asked" in lowered:
            topic = "faq"
        entries.append(
            {
                "title": first_line,
                "text": text,
                "service": service,
                "state": "national",
                "topic": topic,
                "source_url": next((token for token in text.split() if token.startswith("http")), ""),
                "source_name": f"{service}.json",
            }
        )
    return entries


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for service in ("passport", "aadhaar", "pf"):
        source_path = SOURCE_DIR / f"{service}.json"
        if not source_path.exists():
            print(f"[SKIP] Missing source file: {source_path}")
            continue
        chunks = json.loads(source_path.read_text(encoding="utf-8"))
        entries = convert_chunks(service, chunks)
        output_path = OUTPUT_DIR / f"{service}.json"
        output_path.write_text(json.dumps(entries, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[OK] Wrote {output_path} ({len(entries)} metadata chunks)")


if __name__ == "__main__":
    main()
