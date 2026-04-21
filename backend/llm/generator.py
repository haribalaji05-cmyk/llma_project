import json
import os
import re
from typing import Any, Dict, List, Optional

import requests

try:
    from transformers import pipeline
except ImportError:
    pipeline = None


class LLMGenerator:
    def __init__(self, model_name: str = None):
        self.api_key = os.getenv("OPENAI_API_KEY", "").strip()
        self.api_base = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
        self.api_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.enable_local_fallback = os.getenv("ENABLE_LOCAL_LLM_FALLBACK", "false").lower() == "true"
        self.model_name = model_name or os.getenv("LLM_MODEL", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        self.generator = None
        self.load_error = None

        if self.api_key:
            print(f"[LLM] API mode enabled with model '{self.api_model}' at '{self.api_base}'")
        elif self.enable_local_fallback:
            self._load_model()
        else:
            self.load_error = "No API key configured and local fallback disabled."
            print(f"[LLM] {self.load_error}")

    def _load_model(self) -> None:
        if pipeline is None:
            self.load_error = "transformers pipeline is unavailable"
            print(f"[LLM] Failed to initialize local model '{self.model_name}': {self.load_error}")
            return
        try:
            print(f"[LLM] Loading local model '{self.model_name}'...")
            self.generator = pipeline(
                "text-generation",
                model=self.model_name,
                device_map="auto",
                trust_remote_code=True,
            )
            print(f"[LLM] Local model loaded successfully: '{self.model_name}'")
        except Exception as exc:
            self.load_error = str(exc)
            self.generator = None
            print(f"[LLM] Failed to load local model '{self.model_name}': {self.load_error}")

    def safe_parse(self, response: str) -> Dict[str, Any]:
        try:
            return json.loads(response)
        except Exception:
            cleaned = self.clean_llm_output(response)
            if cleaned is not None:
                return cleaned
            return {"error": "Invalid JSON", "raw_output": response}

    def generate_json(self, prompt: str) -> Dict[str, Any]:
        if self.api_key:
            return self._generate_via_api(prompt)
        if self.generator is not None:
            return self._generate_via_local_model(prompt)

        error_message = self.load_error or "LLM generator is not available in this environment."
        print(f"[LLM] Skipping generation because model is unavailable: {error_message}")
        return {"error": error_message}

    def _generate_via_api(self, prompt: str) -> Dict[str, Any]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.api_model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a strict JSON generator.\n"
                        "Return ONLY valid JSON.\n"
                        "Do not include explanations, prose, or markdown.\n"
                        "Follow the exact schema requested by the user prompt."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            "temperature": 0,
        }
        try:
            response = requests.post(
                f"{self.api_base}/chat/completions",
                headers=headers,
                json=payload,
                timeout=60,
            )
            response.raise_for_status()
            data = response.json()
            raw_output = data["choices"][0]["message"]["content"].strip()
            print("=== API GENERATION ===")
            print(raw_output)
            parsed = self.safe_parse(raw_output)
            if isinstance(parsed, dict) and "error" in parsed:
                parsed["raw_output"] = raw_output
            return parsed
        except Exception as exc:
            print(f"[LLM] API generation failed: {exc}")
            if hasattr(exc, "response") and getattr(exc, "response", None) is not None:
                try:
                    print(f"[LLM] API error response: {exc.response.text}")
                except Exception:
                    pass
            return {"error": f"API generation failed: {exc}"}

    def _generate_via_local_model(self, prompt: str) -> Dict[str, Any]:
        outputs = self.generator(
            prompt,
            max_new_tokens=128,
            do_sample=False,
            return_full_text=False,
        )
        raw_output = outputs[0]["generated_text"].strip()
        print("=== LOCAL GENERATION ===")
        print(raw_output)
        parsed = self.safe_parse(raw_output)
        if isinstance(parsed, dict) and "error" in parsed:
            parsed["raw_output"] = raw_output
        return parsed

    def clean_llm_output(self, text: str) -> Optional[Dict[str, Any]]:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return None
        json_str = match.group()
        try:
            return json.loads(json_str)
        except Exception:
            return None

    def normalize_response(
        self,
        payload: Dict[str, Any],
        query: str,
        language: str,
        source_chunks: List[str],
    ) -> Dict[str, Any]:
        answer = payload.get("answer", {}) if isinstance(payload, dict) else {}
        safety_payload = payload.get("safety", {}) if isinstance(payload, dict) else {}
        normalized_intent = self._normalize_intent(payload.get("intent", "unknown"), query, source_chunks)
        normalized = {
            "query": query,
            "intent": normalized_intent,
            "language": language,
            "answer": {
                "title": self._normalize_text(answer.get("title", "Not available")),
                "steps": self._dedupe_preserve(self._normalize_list(answer.get("steps"))),
                "documents": self._normalize_documents(answer.get("documents")),
                "fees": self._normalize_text(answer.get("fees", "Not available")),
                "processing_time": self._normalize_text(
                    answer.get("processing_time", "Not available")
                ),
                "eligibility": self._dedupe_preserve(self._normalize_list(answer.get("eligibility"))),
                "official_links": self._dedupe_preserve(self._normalize_list(answer.get("official_links"))),
            },
            "follow_up_questions": self._dedupe_preserve(self._normalize_list(payload.get("follow_up_questions"))),
            "safety": {
                "is_safe": bool(safety_payload.get("is_safe", True)),
                "message": self._normalize_optional_text(safety_payload.get("message", "")),
            },
            "source_chunks": self._dedupe_preserve(
                self._normalize_list(payload.get("source_chunks")) or source_chunks
            ),
        }

        if "error" in payload:
            normalized = self._build_fallback_response(normalized, query, source_chunks)
            normalized["safety"]["message"] = ""
        elif self._answer_is_empty(normalized["answer"]):
            normalized = self._build_fallback_response(normalized, query, source_chunks)

        normalized["answer"] = self._postprocess_answer(
            normalized["answer"],
            query=query,
            intent=normalized["intent"],
            source_chunks=normalized["source_chunks"],
        )
        normalized["follow_up_questions"] = self._build_follow_ups(normalized["intent"])
        return normalized

    def _normalize_text(self, value: Any) -> str:
        if value is None:
            return "Not available"
        text = str(value).strip()
        return text if text else "Not available"

    def _normalize_optional_text(self, value: Any) -> str:
        if value is None:
            return ""
        return str(value).strip()

    def _normalize_list(self, value: Any) -> List[str]:
        if not value:
            return []
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        return [str(value).strip()] if str(value).strip() else []

    def _dedupe_preserve(self, values: List[str]) -> List[str]:
        unique: List[str] = []
        seen = set()
        for value in values:
            key = value.strip().lower()
            if not key or key in seen:
                continue
            seen.add(key)
            unique.append(value.strip())
        return unique

    def _normalize_documents(self, value: Any) -> List[str]:
        if not value:
            return []
        if isinstance(value, list):
            flattened: List[str] = []
            for item in value:
                if isinstance(item, dict):
                    doc_type = str(item.get("type", "")).strip()
                    options = item.get("options", [])
                    if isinstance(options, list):
                        for option in options:
                            option_text = str(option).strip()
                            if option_text:
                                flattened.append(
                                    f"{doc_type}: {option_text}" if doc_type else option_text
                                )
                    else:
                        option_text = str(options).strip()
                        if option_text:
                            flattened.append(
                                f"{doc_type}: {option_text}" if doc_type else option_text
                            )
                else:
                    text = str(item).strip()
                    if text:
                        flattened.append(text)
            return self._dedupe_preserve(flattened)[:10]
        return self._dedupe_preserve(self._normalize_list(value))

    def _normalize_intent(self, value: Any, query: str, source_chunks: List[str]) -> str:
        text = str(value or "").strip().lower()
        exact_intents = {
            "passport_application",
            "passport_fees",
            "passport_documents",
            "passport_processing_time",
            "aadhaar_service",
            "aadhaar_enrolment",
            "aadhaar_update",
            "aadhaar_fees",
            "pf_withdrawal",
            "pf_uan_activation",
            "pf_transfer",
            "government_service_query",
            "state_eservice_portal_usage",
        }
        if text in exact_intents:
            return text
        if "passport" in text:
            if "fee" in text or "cost" in text or "tatkal" in text:
                return "passport_fees"
            if "document" in text or "verification" in text:
                return "passport_documents"
            if "time" in text or "days" in text or "processing" in text:
                return "passport_processing_time"
            return "passport_application"
        if "aadhaar" in text or "aadhar" in text or "uidai" in text:
            if "update" in text:
                return "aadhaar_update"
            if "enrol" in text or "enroll" in text or "apply" in text:
                return "aadhaar_enrolment"
            if "fee" in text or "cost" in text:
                return "aadhaar_fees"
            return "aadhaar_service"
        if "pf" in text or "epfo" in text or "uan" in text:
            if "uan" in text:
                return "pf_uan_activation"
            if "transfer" in text:
                return "pf_transfer"
            return "pf_withdrawal"
        if any(token in text for token in ["esevai", "seva", "sarkar", "portal", "esathi", "meeseva"]):
            return "state_eservice_portal_usage"
        combined = f"{query}\n" + "\n".join(source_chunks[:2])
        return self._detect_intent(combined.lower(), combined.lower())

    def _answer_is_empty(self, answer: Dict[str, Any]) -> bool:
        return (
            answer.get("title") == "Not available"
            and not answer.get("steps")
            and not answer.get("documents")
            and answer.get("fees") == "Not available"
            and answer.get("processing_time") == "Not available"
            and not answer.get("eligibility")
            and not answer.get("official_links")
        )

    def _build_fallback_response(
        self,
        normalized: Dict[str, Any],
        query: str,
        source_chunks: List[str],
    ) -> Dict[str, Any]:
        extracted = self._extract_from_chunks(query, source_chunks)
        normalized["intent"] = extracted["intent"]
        normalized["answer"] = extracted["answer"]
        normalized["follow_up_questions"] = extracted["follow_up_questions"]
        normalized["source_chunks"] = self._dedupe_preserve(source_chunks)
        return normalized

    def _extract_from_chunks(self, query: str, source_chunks: List[str]) -> Dict[str, Any]:
        combined = "\n".join(source_chunks)
        lowered = query.lower()
        intent = self._detect_intent(lowered, combined.lower())
        answer = {
            "title": self._extract_title(intent, combined),
            "steps": self._extract_steps(combined),
            "documents": self._extract_documents(combined),
            "fees": self._extract_fees(combined),
            "processing_time": self._extract_processing_time(combined),
            "eligibility": self._extract_eligibility(combined),
            "official_links": self._extract_links(combined),
        }
        return {
            "intent": intent,
            "answer": answer,
            "follow_up_questions": self._build_follow_ups(intent),
        }

    def _detect_intent(self, query: str, context: str) -> str:
        combined = f"{query}\n{context}"
        if any(token in combined for token in ["tnesevai", "e sevai", "esevai", "seva sindhu", "aaple sarkar", "esathi", "meeseva"]):
            return "state_eservice_portal_usage"
        if "passport" in combined:
            if any(token in combined for token in ["fee", "fees", "cost", "tatkal"]):
                return "passport_fees"
            if any(token in combined for token in ["document", "documents", "verification"]):
                return "passport_documents"
            if any(token in combined for token in ["time", "days", "processing"]):
                return "passport_processing_time"
            return "passport_application"
        if "aadhaar" in combined or "aadhar" in combined or "uidai" in combined:
            if "update" in combined:
                return "aadhaar_update"
            if any(token in combined for token in ["enrol", "enroll", "apply", "application"]):
                return "aadhaar_enrolment"
            if any(token in combined for token in ["fee", "fees", "cost"]):
                return "aadhaar_fees"
            return "aadhaar_service"
        if "pf" in combined or "epfo" in combined or "uan" in combined:
            if "uan" in combined:
                return "pf_uan_activation"
            if "transfer" in combined:
                return "pf_transfer"
            return "pf_withdrawal"
        return "government_service_query"

    def _extract_title(self, intent: str, combined: str) -> str:
        title_map = {
            "passport_application": "Passport Application Process (India)",
            "passport_fees": "Passport Fees",
            "passport_documents": "Passport Required Documents",
            "passport_processing_time": "Passport Processing Time",
            "aadhaar_service": "Aadhaar Service Information",
            "aadhaar_enrolment": "Aadhaar Enrolment Process",
            "aadhaar_update": "Aadhaar Update Process",
            "aadhaar_fees": "Aadhaar Fees",
            "pf_withdrawal": "PF Withdrawal Process",
            "pf_uan_activation": "UAN Activation Process",
            "pf_transfer": "PF Transfer Process",
            "state_eservice_portal_usage": "State e-Service Portal Information",
            "government_service_query": "Government Service Information",
        }
        for line in combined.splitlines():
            clean = line.strip()
            if clean and len(clean) < 100 and ("—" in clean or "-" in clean):
                return clean
        return title_map.get(intent, "Government Service Information")

    def _extract_steps(self, text: str) -> List[str]:
        steps = []
        for line in text.splitlines():
            clean = line.strip()
            if re.match(r"^(Step\s*\d+|[0-9]+\.)", clean, flags=re.IGNORECASE):
                step_text = re.sub(r"^(Step\s*\d+\s*[—\-:]?\s*|[0-9]+\.\s*)", "", clean, flags=re.IGNORECASE)
                if step_text:
                    steps.append(step_text)
        return self._dedupe_preserve(steps)[:8]

    def _extract_documents(self, text: str) -> List[str]:
        documents = []
        capture = False
        for line in text.splitlines():
            clean = line.strip()
            lower = clean.lower()
            if "required documents" in lower or "proof of" in lower:
                capture = True
                continue
            if capture and not clean:
                if documents:
                    break
                continue
            if capture and clean.startswith("-"):
                documents.append(clean.lstrip("- ").strip())
        return self._dedupe_preserve(documents)[:10]

    def _extract_fees(self, text: str) -> str:
        fee_lines = []
        for line in text.splitlines():
            clean = line.strip()
            lower = clean.lower()
            if clean.startswith("Step "):
                continue
            if "fee" in lower or "₹" in clean or "rs." in lower:
                fee_lines.append(clean)
        fee_lines = self._dedupe_preserve(fee_lines)
        if not fee_lines:
            return "Not available"
        condensed = []
        for line in fee_lines[:6]:
            lower = line.lower()
            if any(token in lower for token in ["fresh", "renewal", "tatkal", "pcc", "minor", "lost", "damaged", "fee"]):
                condensed.append(line)
        return " | ".join(condensed[:4] or fee_lines[:3])

    def _extract_processing_time(self, text: str) -> str:
        normal_line = ""
        tatkal_line = ""
        for line in text.splitlines():
            clean = line.strip()
            lower = clean.lower()
            if "processing time is typically" in lower:
                normal_line = clean
            elif "processed within" in lower and "tatkal" in lower:
                tatkal_line = clean
        if normal_line and tatkal_line:
            return f"{normal_line} | {tatkal_line}"
        if normal_line:
            return normal_line
        if tatkal_line:
            return tatkal_line
        return "Not available"

    def _extract_eligibility(self, text: str) -> List[str]:
        eligibility = []
        for line in text.splitlines():
            clean = line.strip()
            lower = clean.lower()
            if clean.startswith("Step ") or clean.startswith("Q:") or clean.startswith("A:"):
                continue
            if "required documents" in lower or "frequently asked" in lower:
                continue
            if any(token in lower for token in ["must", "required", "mandatory", "who can enrol", "allowed after", "applies to", "resident of india"]):
                eligibility.append(clean.lstrip("- ").strip())
        return self._dedupe_preserve(eligibility)[:6]

    def _extract_links(self, text: str) -> List[str]:
        matches = re.findall(r"https?://[^\s)]+", text)
        return self._dedupe_preserve(matches)[:5]

    def _build_follow_ups(self, intent: str) -> List[str]:
        follow_ups = {
            "passport_application": [
                "Do you want the Tatkal process details?",
                "Do you need the required documents list?",
            ],
            "passport_fees": [
                "Do you want the Tatkal fee details?",
                "Do you need the required documents for passport application?",
            ],
            "passport_documents": [
                "Do you want the full passport application steps?",
                "Do you want the official passport portal link?",
            ],
            "passport_processing_time": [
                "Do you want the Tatkal processing timeline too?",
                "Do you need passport tracking details?",
            ],
            "aadhaar_service": [
                "Do you want online or offline Aadhaar update steps?",
                "Do you need the required documents?",
            ],
            "aadhaar_enrolment": [
                "Do you need the Aadhaar enrolment documents?",
                "Do you want the official UIDAI status tracking link?",
            ],
            "aadhaar_update": [
                "Do you want online address update steps?",
                "Do you need the Aadhaar update fee details?",
            ],
            "aadhaar_fees": [
                "Do you want Aadhaar update fee details?",
                "Do you need the Aadhaar update process?",
            ],
            "pf_withdrawal": [
                "Do you need UAN and KYC requirements?",
                "Do you want online EPFO claim steps?",
            ],
            "pf_uan_activation": [
                "Do you need the UAN activation steps?",
                "Do you want KYC linking details too?",
            ],
            "pf_transfer": [
                "Do you need the Form 13 transfer steps?",
                "Do you want PF transfer eligibility details?",
            ],
            "government_service_query": [
                "Do you want required documents?",
                "Do you want official links for this service?",
            ],
            "state_eservice_portal_usage": [
                "Do you want portal registration steps?",
                "Do you need the official portal link and service list?",
            ],
        }
        return follow_ups.get(intent, [])

    def _postprocess_answer(
        self,
        answer: Dict[str, Any],
        query: str,
        intent: str,
        source_chunks: List[str],
    ) -> Dict[str, Any]:
        combined = "\n".join(source_chunks)
        query_lower = query.lower()

        if not answer.get("steps"):
            answer["steps"] = self._extract_steps(combined)
        else:
            answer["steps"] = self._dedupe_preserve(answer["steps"])[:8]

        if not answer.get("documents"):
            answer["documents"] = self._extract_documents(combined)
        else:
            answer["documents"] = self._dedupe_preserve(answer["documents"])[:10]

        if answer.get("fees", "Not available") == "Not available" or "step 3" in answer.get("fees", "").lower():
            answer["fees"] = self._extract_fees(combined)

        extracted_time = self._extract_processing_time(combined)
        if extracted_time != "Not available":
            answer["processing_time"] = extracted_time

        answer["eligibility"] = self._extract_eligibility(combined)
        answer["official_links"] = self._dedupe_preserve(
            answer.get("official_links") or self._extract_links(combined)
        )[:5]

        if "document" in query_lower or "verification" in query_lower or intent.endswith("_documents"):
            answer["steps"] = []
            answer["fees"] = "Not available"
            answer["processing_time"] = "Not available"
        elif any(token in query_lower for token in ["fee", "fees", "cost", "tatkal"]) or intent.endswith("_fees"):
            answer["steps"] = []
            answer["documents"] = []
            answer["eligibility"] = []
            answer["processing_time"] = "Not available"
            answer["fees"] = self._compress_fee_summary(answer["fees"])
        elif any(token in query_lower for token in ["time", "days", "processing"]) or intent.endswith("_processing_time"):
            answer["steps"] = answer["steps"][:4]
            answer["documents"] = []
            answer["fees"] = "Not available"
        elif any(token in query_lower for token in ["how", "apply", "process", "enrol", "enroll", "update"]) or intent in {"passport_application", "aadhaar_enrolment", "aadhaar_update", "pf_withdrawal", "pf_uan_activation", "pf_transfer"}:
            answer["documents"] = answer["documents"][:5]
            if len(answer["eligibility"]) > 3:
                answer["eligibility"] = answer["eligibility"][:3]

        if intent == "state_eservice_portal_usage":
            answer["documents"] = answer["documents"][:5]
            if not answer["official_links"]:
                answer["official_links"] = self._extract_links(combined)

        if answer.get("title") == "Not available":
            answer["title"] = self._extract_title(intent, combined)

        return answer

    def _compress_fee_summary(self, fee_text: str) -> str:
        if not fee_text or fee_text == "Not available":
            return "Not available"

        lines = [line.strip(" -") for line in fee_text.splitlines() if line.strip()]
        selected = []
        for line in lines:
            lower = line.lower()
            if any(token in lower for token in ["fresh / renewal", "tatkal", "pcc", "reprint", "update fee", "free of charge"]):
                selected.append(line)
        selected = self._dedupe_preserve(selected)
        if not selected:
            return fee_text
        return " | ".join(selected[:4])
