from attr import dataclass

@dataclass(slots=True)
class RegistrationResult:
    success: bool
    user_id: str