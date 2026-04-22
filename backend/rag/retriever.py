import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

from backend.rag.embedder import Embedder
from backend.rag.vector_store import FaissVectorStore


STATE_ALIASES = {
    "andhra pradesh": ["andhra pradesh", "ap"],
    "bihar": ["bihar"],
    "gujarat": ["gujarat"],
    "haryana": ["haryana"],
    "karnataka": ["karnataka"],
    "kerala": ["kerala"],
    "maharashtra": ["maharashtra"],
    "odisha": ["odisha", "orissa"],
    "rajasthan": ["rajasthan"],
    "tamil nadu": ["tamil nadu", "tn"],
    "telangana": ["telangana"],
    "uttar pradesh": ["uttar pradesh", "up"],
}


class RAGRetriever:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.embedder = Embedder(model_name)
        self.vector_store = FaissVectorStore(dimension=384)
        self.texts: List[str] = []
        self.metadata: List[Dict[str, str]] = []
        self._load_default_store()

    def _load_default_store(self) -> None:
        processed_dir = Path(__file__).resolve().parents[2] / "data" / "processed_chunks"
        documents = self._read_processed_chunks(processed_dir)
        if documents:
            self.add_documents(documents)

    def _repair_text(self, text: str) -> str:
        replacements = {
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
        fixed = str(text)
        for broken, replacement in replacements.items():
            fixed = fixed.replace(broken, replacement)
        return fixed.strip()

    def _infer_service(self, path: Path, text: str) -> str:
        combined = f"{path.stem} {text[:400]}".lower()
        if "passport" in combined:
            return "passport"
        if "aadhaar" in combined or "aadhar" in combined or "uidai" in combined:
            return "aadhaar"
        if "pf" in combined or "epfo" in combined or "uan" in combined:
            return "pf"
        return "general"

    def _infer_state(self, path: Path, text: str) -> str:
        combined = f"{path.stem} {text[:400]}".lower()
        for state, aliases in STATE_ALIASES.items():
            if any(alias in combined for alias in aliases):
                return state
        return "national"

    def _infer_topic(self, text: str, title: str = "") -> str:
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

    def _resolve_topic(self, title: str, text: str, existing_topic: str = "") -> str:
        inferred = self._infer_topic(text, title=title)
        normalized_existing = existing_topic.strip().lower()
        if not normalized_existing:
            return inferred
        if any(keyword in title.lower() for keyword in ["overview", "step", "process", "procedure"]) and normalized_existing in {"documents", "fees", "processing_time"}:
            return "steps"
        return normalized_existing

    def _extract_source_url(self, text: str) -> str:
        match = re.search(r"https?://[^\s)]+", text)
        return match.group(0) if match else ""

    def _extract_title(self, text: str, fallback: str) -> str:
        for line in text.splitlines():
            clean = line.strip()
            if clean:
                return clean[:120]
        return fallback

    def _normalize_entry(self, path: Path, item: object, index: int) -> Dict[str, str] | None:
        if isinstance(item, dict):
            text = self._repair_text(item.get("text", ""))
            if not text:
                return None
            source_url = str(item.get("source_url", "")).strip() or self._extract_source_url(text)
            title = self._repair_text(item.get("title", "")).strip() or self._extract_title(text, path.stem)
            service = str(item.get("service", "")).strip() or self._infer_service(path, text)
            state = str(item.get("state", "")).strip() or self._infer_state(path, text)
            topic = self._resolve_topic(
                title=title,
                text=text,
                existing_topic=str(item.get("topic", "")).strip(),
            )
            source_name = str(item.get("source_name", "")).strip() or path.name
            keywords = item.get("keywords", [])
            if not isinstance(keywords, list):
                keywords = []
        else:
            text = self._repair_text(item)
            if not text:
                return None
            source_url = self._extract_source_url(text)
            title = self._extract_title(text, f"{path.stem}-{index}")
            service = self._infer_service(path, text)
            state = self._infer_state(path, text)
            topic = self._resolve_topic(title=title, text=text)
            source_name = path.name
            keywords = []

        return {
            "text": text,
            "service": service,
            "state": state,
            "topic": topic,
            "title": title,
            "source_url": source_url,
            "source_name": source_name,
            "keywords": [str(keyword).strip().lower() for keyword in keywords if str(keyword).strip()],
        }

    def _split_text_chunks(self, text: str) -> List[str]:
        chunks = [chunk.strip() for chunk in re.split(r"\n\s*=+\s*\n", text) if chunk.strip()]
        return chunks if chunks else [text.strip()]

    def _read_processed_chunks(self, directory: Path) -> List[Dict[str, str]]:
        if not directory.exists():
            return []

        sibling_jsons = {path.stem for path in directory.glob("*.json")}
        chunks: List[Dict[str, str]] = []
        seen_texts = set()

        for path in sorted(directory.glob("**/*")):
            if not path.is_file():
                continue
            if path.suffix.lower() == ".txt" and path.stem in sibling_jsons:
                continue

            loaded_entries: List[object] = []
            if path.suffix.lower() in {".txt", ".md"}:
                text = path.read_text(encoding="utf-8").strip()
                if text:
                    loaded_entries = self._split_text_chunks(text)
            elif path.suffix.lower() == ".json":
                try:
                    data = json.loads(path.read_text(encoding="utf-8"))
                except Exception:
                    continue
                if isinstance(data, list):
                    loaded_entries = data

            for index, item in enumerate(loaded_entries, start=1):
                entry = self._normalize_entry(path, item, index)
                if not entry:
                    continue
                key = entry["text"].strip().lower()
                if key in seen_texts:
                    continue
                seen_texts.add(key)
                chunks.append(entry)
        return chunks

    def add_documents(self, documents: List[Dict[str, str]]) -> None:
        texts = [document["text"] for document in documents]
        embeddings = self.embedder.embed_texts(texts)
        self.vector_store.add_documents(texts, embeddings)
        self.texts.extend(texts)
        self.metadata.extend(documents)

    def detect_service(self, query: str) -> str:
        lowered = query.lower()
        if "passport" in lowered:
            return "passport"
        if "aadhaar" in lowered or "aadhar" in lowered or "uidai" in lowered:
            return "aadhaar"
        if "pf" in lowered or "epfo" in lowered or "uan" in lowered:
            return "pf"
        return "general"

    def detect_state(self, query: str) -> str:
        lowered = query.lower()
        for state, aliases in STATE_ALIASES.items():
            if any(re.search(rf"\b{re.escape(alias)}\b", lowered) for alias in aliases):
                return state
        return "national"

    def detect_topic(self, query: str) -> str:
        lowered = query.lower()
        if any(keyword in lowered for keyword in ["document", "documents", "proof", "required"]):
            return "documents"
        if any(keyword in lowered for keyword in ["fee", "fees", "cost", "charge", "price"]):
            return "fees"
        if any(keyword in lowered for keyword in ["time", "timeline", "how long", "days", "processing"]):
            return "processing_time"
        if any(keyword in lowered for keyword in ["eligible", "eligibility", "who can", "requirement"]):
            return "eligibility"
        if any(keyword in lowered for keyword in ["portal", "register", "login", "use", "apply", "process", "steps"]):
            return "steps"
        return "general"

    def _filter_metadata(self, service: str, state: str, topic: str = "general") -> List[Dict[str, str]]:
        candidates = self.metadata
        if service != "general":
            scoped = [item for item in candidates if item.get("service") == service]
            if scoped:
                candidates = scoped
        if state != "national":
            scoped = [item for item in candidates if item.get("state") in {state, "national"}]
            if scoped:
                candidates = scoped
        if topic != "general":
            scoped = [
                item
                for item in candidates
                if item.get("topic") == topic or (topic == "steps" and item.get("topic") == "general")
            ]
            if scoped:
                candidates = scoped
        return candidates

    def _rerank_candidates(
        self,
        query: str,
        candidates: List[Dict[str, str]],
        top_k: int,
        topic: str = "general",
    ) -> List[Dict[str, str]]:
        query_tokens = {
            token
            for token in re.findall(r"[a-zA-Z0-9]+", query.lower())
            if len(token) > 2 and token not in {"what", "how", "for", "the", "and", "with"}
        }

        def score(item: Dict[str, str]) -> Tuple[int, int, int, int, int, int]:
            text = item.get("text", "").lower()
            title = item.get("title", "").lower()
            topic = item.get("topic", "").lower()
            keywords = item.get("keywords", [])
            overlap = sum(token in text for token in query_tokens)
            title_boost = sum(token in title for token in query_tokens)
            topic_boost = 2 if topic and topic == requested_topic else 0
            title_phrase_boost = 1 if query.lower() in title else 0
            source_boost = 1 if item.get("source_url") else 0
            keyword_boost = sum(token in keywords for token in query_tokens)
            return (overlap, title_boost, topic_boost, title_phrase_boost, source_boost, keyword_boost)

        requested_topic = topic.lower()
        ranked = sorted(candidates, key=score, reverse=True)
        return ranked[:top_k]

    def _select_context_bundle(
        self,
        query: str,
        service: str,
        state: str,
        topic: str,
        top_k: int,
    ) -> List[Dict[str, str]]:
        primary_pool = self._filter_metadata(service, state, topic)
        fallback_pool = self._filter_metadata(service, state, "general")

        combined_candidates = primary_pool or fallback_pool
        if not combined_candidates:
            return []

        primary_ranked = self._rerank_candidates(
            query=query,
            candidates=combined_candidates,
            top_k=min(max(top_k, 2), len(combined_candidates)),
            topic=topic,
        )

        if topic == "general":
            return primary_ranked[:top_k]

        supplemental_candidates = [
            item
            for item in fallback_pool
            if item not in primary_ranked and item.get("topic") in {"general", "steps", "eligibility", "faq"}
        ]
        supplemental_ranked = self._rerank_candidates(
            query=query,
            candidates=supplemental_candidates,
            top_k=1,
            topic="general",
        )
        bundled = primary_ranked[: max(top_k - len(supplemental_ranked), 1)] + supplemental_ranked

        unique_bundle: List[Dict[str, str]] = []
        seen = set()
        for item in bundled:
            key = item.get("text", "").strip().lower()
            if key and key not in seen:
                seen.add(key)
                unique_bundle.append(item)
        return unique_bundle[:top_k]

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        service: str = "general",
        state: str = "national",
        topic: str = "general",
    ) -> List[str]:
        if not self.texts:
            return []

        candidates = self._select_context_bundle(query, service, state, topic, top_k=max(top_k, 3))
        if candidates:
            filtered_documents = [item["text"] for item in candidates]
            filtered_store = FaissVectorStore(dimension=384)
            filtered_embeddings = self.embedder.embed_texts(filtered_documents)
            filtered_store.add_documents(filtered_documents, filtered_embeddings)
            query_embedding = self.embedder.embed_query(query)
            initial_results = filtered_store.search(query_embedding, k=min(max(top_k * 2, 4), len(filtered_documents)))
            rerank_pool = [item for item in candidates if item["text"] in initial_results]
            reranked = self._rerank_candidates(
                query=query,
                candidates=rerank_pool or candidates,
                top_k=top_k,
                topic=topic,
            )
            return [item["text"] for item in reranked]

        query_embedding = self.embedder.embed_query(query)
        return self.vector_store.search(query_embedding, k=top_k)

    def retrieve_with_metadata(
        self,
        query: str,
        top_k: int = 3,
        service: str = "general",
        state: str = "national",
        topic: str = "general",
    ) -> List[Tuple[str, Dict[str, str]]]:
        texts = self.retrieve(query=query, top_k=top_k, service=service, state=state, topic=topic)
        results: List[Tuple[str, Dict[str, str]]] = []
        for text in texts:
            for item in self.metadata:
                if item.get("text") == text:
                    results.append((text, item))
                    break
        return results
