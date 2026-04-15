import json
from typing import Dict, List, Tuple


class ChatMemory:
    def __init__(self, max_turns: int = 5):
        self.max_turns = max_turns
        self.history: List[Tuple[str, Dict[str, object]]] = []

    def update(self, user_query: str, assistant_response: Dict[str, object]) -> None:
        self.history.append((user_query, assistant_response))
        self.history = self.history[-self.max_turns :]

    def get_history_for_prompt(self) -> str:
        if not self.history:
            return ""

        formatted = []
        for idx, (query, response) in enumerate(self.history[-self.max_turns :], start=1):
            compact_response = {
                "intent": response.get("intent", ""),
                "title": response.get("answer", {}).get("title", ""),
                "follow_up_questions": response.get("follow_up_questions", []),
            }
            formatted.append(
                f"Turn {idx} User Query: {query}\n"
                f"Turn {idx} Assistant Summary: {json.dumps(compact_response, ensure_ascii=False)}"
            )
        return "\n\n".join(formatted)
