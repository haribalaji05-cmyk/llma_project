import json
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel

from backend.llm.generator import LLMGenerator
from backend.llm.prompt import build_prompt
from backend.memory.chat_memory import ChatMemory
from backend.multilingual.detect import detect_language
from backend.multilingual.translate import translate_from_english, translate_to_english
from backend.rag.retriever import RAGRetriever
from backend.safety.filter import SafetyFilter
from backend.voice.voice import SpeechToText, TextToSpeech


app = FastAPI(title="GovAssist Core Intelligence API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

retriever = RAGRetriever()
generator = LLMGenerator()
chat_memory = ChatMemory()
speech_to_text_engine = SpeechToText()
text_to_speech_engine = TextToSpeech()


class ChatRequest(BaseModel):
    query: str


class TextToSpeechRequest(BaseModel):
    text: str
    voice: Optional[str] = None
    language: Optional[str] = "en"


def build_empty_response(query: str, language: str, message: str = "") -> Dict[str, Any]:
    return {
        "query": query,
        "intent": "unknown",
        "language": language or "en",
        "answer": {
            "title": "Not available",
            "steps": [],
            "documents": [],
            "fees": "Not available",
            "processing_time": "Not available",
            "eligibility": [],
            "official_links": [],
        },
        "follow_up_questions": [],
        "safety": {
            "is_safe": True,
            "message": message,
        },
        "source_chunks": [],
    }


def translate_response_fields(payload: Dict[str, Any], target_language: str) -> Dict[str, Any]:
    if target_language == "en":
        return payload

    answer = payload.get("answer", {})
    translated = json.loads(json.dumps(payload))
    translated["answer"]["title"] = translate_from_english(answer.get("title", ""), target_language)
    translated["answer"]["steps"] = [
        translate_from_english(step, target_language) for step in answer.get("steps", [])
    ]
    translated["answer"]["documents"] = [
        translate_from_english(document, target_language)
        for document in answer.get("documents", [])
    ]
    translated["answer"]["fees"] = translate_from_english(answer.get("fees", ""), target_language)
    translated["answer"]["processing_time"] = translate_from_english(
        answer.get("processing_time", ""),
        target_language,
    )
    translated["answer"]["eligibility"] = [
        translate_from_english(item, target_language)
        for item in answer.get("eligibility", [])
    ]
    translated["answer"]["processing_time"] = translate_from_english(
        answer.get("processing_time", ""),
        target_language,
    )
    translated["follow_up_questions"] = [
        translate_from_english(question, target_language)
        for question in payload.get("follow_up_questions", [])
    ]
    translated["safety"]["message"] = translate_from_english(
        payload.get("safety", {}).get("message", ""),
        target_language,
    )
    return translated


def build_source_chunks(
    retrieved_results: List[tuple[str, Dict[str, str]]],
    fallback_chunks: List[str],
) -> List[str]:
    if not retrieved_results:
        return fallback_chunks

    formatted = []
    for text, metadata in retrieved_results:
        title = metadata.get("title", "").strip()
        source_url = metadata.get("source_url", "").strip()
        topic = metadata.get("topic", "").strip()
        preview = " ".join(text.split())[:220].strip()
        parts = [part for part in [title, topic, preview] if part]
        line = " | ".join(parts)
        if source_url:
            line = f"{line} | Source: {source_url}" if line else f"Source: {source_url}"
        if line:
            formatted.append(line)
    return formatted or fallback_chunks


def build_low_confidence_response(
    query: str,
    language: str,
    intent: str,
    source_chunks: List[str],
    official_links: List[str],
) -> Dict[str, Any]:
    return {
        "query": query,
        "intent": intent,
        "language": language,
        "answer": {
            "title": "Not available",
            "steps": [],
            "documents": [],
            "fees": "Not available",
            "processing_time": "Not available",
            "eligibility": [],
            "official_links": official_links,
        },
        "follow_up_questions": [
            "Can you mention the exact service name or state?",
            "Do you want the official portal link instead?",
        ],
        "safety": {
            "is_safe": True,
            "message": "Limited matching official data was found for this query.",
        },
        "source_chunks": source_chunks,
    }


def assess_retrieval_confidence(
    query: str,
    service: str,
    state: str,
    retrieved_results: List[tuple[str, Dict[str, str]]],
) -> Dict[str, Any]:
    if not retrieved_results:
        return {"is_confident": False, "official_links": [], "source_chunks": []}

    query_lower = query.lower()
    query_tokens = {
        token
        for token in query_lower.split()
        if len(token) > 2 and token not in {"what", "how", "for", "the", "and", "fees", "fee"}
    }

    service_matches = 0
    state_matches = 0
    overlap_hits = 0
    official_links = []
    source_chunks = build_source_chunks(retrieved_results, [text for text, _ in retrieved_results])

    for text, metadata in retrieved_results:
        meta_service = metadata.get("service", "general")
        meta_state = metadata.get("state", "national")
        text_lower = text.lower()
        if service == "general" or meta_service == service:
            service_matches += 1
        if state == "national" or meta_state in {state, "national"}:
            state_matches += 1
        if any(token in text_lower for token in query_tokens):
            overlap_hits += 1
        source_url = metadata.get("source_url", "").strip()
        if source_url and source_url not in official_links:
            official_links.append(source_url)

    portal_keywords = ["tnesevai", "esevai", "seva sindhu", "aaple sarkar", "esathi", "meeseva", "portal"]
    asks_for_portal = any(keyword in query_lower for keyword in portal_keywords)
    if asks_for_portal and overlap_hits == 0:
        return {
            "is_confident": False,
            "official_links": official_links,
            "source_chunks": source_chunks,
        }

    is_confident = service_matches > 0 and state_matches > 0 and overlap_hits > 0
    return {
        "is_confident": is_confident,
        "official_links": official_links,
        "source_chunks": source_chunks,
    }


@app.get("/health")
def health_check() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/chat")
def chat(request: ChatRequest) -> Dict[str, Any]:
    original_query = request.query.strip()
    detected_language = detect_language(original_query or "en")

    if not original_query:
        response = build_empty_response("", detected_language, "Query cannot be empty.")
        response["safety"] = {"is_safe": False, "message": "Query cannot be empty."}
        return response

    safety_result = SafetyFilter.evaluate(original_query)
    if not safety_result["is_safe"]:
        response = build_empty_response(original_query, detected_language, safety_result["message"])
        response["safety"] = safety_result
        return response

    english_query = translate_to_english(original_query, source_lang=detected_language)
    detected_service = retriever.detect_service(english_query)
    detected_state = retriever.detect_state(english_query)
    retrieved_results = retriever.retrieve_with_metadata(
        english_query,
        top_k=3,
        service=detected_service,
        state=detected_state,
    )
    retrieved_chunks = [text for text, _ in retrieved_results]
    confidence = assess_retrieval_confidence(
        query=english_query,
        service=detected_service,
        state=detected_state,
        retrieved_results=retrieved_results,
    )
    print(
        f"[RAG] service={detected_service} state={detected_state} "
        f"results={len(retrieved_results)} confident={confidence['is_confident']}"
    )
    if not confidence["is_confident"]:
        response = build_low_confidence_response(
            query=original_query,
            language=detected_language,
            intent=detected_service if detected_service != "general" else "government_service_query",
            source_chunks=confidence["source_chunks"],
            official_links=confidence["official_links"],
        )
        localized_response = translate_response_fields(response, detected_language)
        localized_response["query"] = original_query
        localized_response["language"] = detected_language
        localized_response["answer"]["official_links"] = response["answer"]["official_links"]
        localized_response["source_chunks"] = response["source_chunks"]
        return localized_response

    prompt = build_prompt(
        retrieved_context="\n\n".join(retrieved_chunks),
        user_query=english_query,
        history=chat_memory.get_history_for_prompt(),
    )
    llm_payload = generator.generate_json(prompt)
    response = generator.normalize_response(
        payload=llm_payload,
        query=original_query,
        language=detected_language,
        source_chunks=build_source_chunks(retrieved_results, retrieved_chunks),
    )
    response["safety"] = safety_result
    localized_response = translate_response_fields(response, detected_language)
    localized_response["query"] = original_query
    localized_response["language"] = detected_language
    localized_response["answer"]["official_links"] = response["answer"]["official_links"]
    localized_response["source_chunks"] = response["source_chunks"]

    chat_memory.update(
        user_query=original_query,
        assistant_response=localized_response,
    )
    return localized_response


@app.post("/speech-to-text")
async def speech_to_text(file: UploadFile = File(...), language: Optional[str] = Form(None)) -> Dict[str, Any]:
    if not file.content_type.startswith("audio/"):
        raise HTTPException(status_code=400, detail="Audio file must be provided.")

    suffix = Path(file.filename).suffix or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        audio_path = tmp.name

    try:
        result = speech_to_text_engine.transcribe(audio_path, language=language)
        return {
            "text": result.get("text", ""),
            "language": result.get("language", language or "en"),
            "error": result.get("error", ""),
        }
    finally:
        try:
            Path(audio_path).unlink()
        except Exception:
            pass


@app.post("/text-to-speech")
def text_to_speech(request: TextToSpeechRequest) -> Response:
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text for synthesis must not be empty.")

    result = text_to_speech_engine.synthesize(
        text=request.text,
        language=request.language or "en",
        voice=request.voice,
    )

    if not result["audio"]:
        print(f"[TTS] Synthesis unavailable: {result.get('error', 'unknown error')}")
        raise HTTPException(status_code=503, detail=result.get("error", "Text-to-speech service unavailable."))

    return Response(content=result["audio"], media_type="audio/wav")
