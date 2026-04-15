from typing import Dict


BLOCKED_KEYWORDS = [
    "fake certificate",
    "hack",
    "illegal",
    "cheat",
    "exploit",
    "bypass",
    "forge",
    "fraud",
]


class SafetyFilter:
    @staticmethod
    def evaluate(text: str) -> Dict[str, object]:
        normalized = text.lower()
        for keyword in BLOCKED_KEYWORDS:
            if keyword in normalized:
                return {
                    "is_safe": False,
                    "message": "I cannot assist with illegal activities.",
                }
        return {"is_safe": True, "message": ""}
