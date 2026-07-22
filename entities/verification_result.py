from dataclasses import dataclass
from typing import Optional

from entities.user import User

@dataclass(slots=True)
class VerificationResult:
    matched: bool
    similarity: float
    user: Optional[User] = None