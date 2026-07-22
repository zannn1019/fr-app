from dataclasses import dataclass

@dataclass(slots=True)
class VerificationResult:
    matched: bool
    user_id: str | None
    similarity: float