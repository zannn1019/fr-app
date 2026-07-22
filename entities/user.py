from dataclasses import dataclass

@dataclass(slots=True)
class User:
    id: str
    name: str | None = None