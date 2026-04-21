MASTER_PROMPT_TEMPLATE = """You are a Government Service Assistant AI for India.

Return ONLY valid JSON.

Use this exact format:
{
  "query": "",
  "intent": "",
  "language": "",
  "answer": {
    "title": "",
    "steps": [],
    "documents": [],
    "fees": "",
    "processing_time": "",
    "eligibility": [],
    "official_links": []
  },
  "follow_up_questions": [],
  "safety": {
    "is_safe": true,
    "message": ""
  },
  "source_chunks": []
}

Rules:
- Use only the provided context.
- If information is missing, write "Not available".
- Write all answer text in English only.
- Do not add any text before or after the JSON.

Context:
__RETRIEVED_CONTEXT__

Query:
__USER_QUERY__
"""


def build_prompt(retrieved_context: str, user_query: str, history: str = "") -> str:
    combined_context = retrieved_context.strip() or "Not available"
    trimmed_history = history.strip()
    if trimmed_history:
        combined_context = f"{combined_context}\n\nFollow-up memory:\n{trimmed_history}"
    return (
        MASTER_PROMPT_TEMPLATE.replace("__RETRIEVED_CONTEXT__", combined_context)
        .replace("__USER_QUERY__", user_query)
    )
