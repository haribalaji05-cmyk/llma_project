import json
from typing import Any, Dict, List

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from backend.llm.generator import LLMGenerator
from backend.llm.prompt import build_prompt
from backend.memory.chat_memory import ChatMemory
from backend.multilingual.detect import detect_language
from backend.multilingual.translate import translate_from_english, translate_to_english
from backend.rag.retriever import RAGRetriever
from backend.safety.filter import SafetyFilter


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


class ChatRequest(BaseModel):
    query: str


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
    translated["follow_up_questions"] = [
        translate_from_english(question, target_language)
        for question in payload.get("follow_up_questions", [])
    ]
    translated["safety"]["message"] = translate_from_english(
        payload.get("safety", {}).get("message", ""),
        target_language,
    )
    return translated


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
    retrieved_chunks = retriever.retrieve(
        english_query,
        top_k=3,
        service=detected_service,
        state=detected_state,
    )
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
        source_chunks=retrieved_chunks,
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
