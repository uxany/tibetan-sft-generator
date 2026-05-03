from dataclasses import dataclass, field
from typing import Optional, Dict


@dataclass
class LLMResponse:
    content: str
    model: str
    provider: str
    usage: Dict[str, int]  # prompt_tokens, completion_tokens, total_tokens
    latency_ms: float
    cost: float = 0.0
    finish_reason: Optional[str] = None

    @property
    def prompt_tokens(self) -> int:
        return self.usage.get("prompt_tokens", 0)

    @property
    def completion_tokens(self) -> int:
        return self.usage.get("completion_tokens", 0)

    @property
    def total_tokens(self) -> int:
        return self.usage.get("total_tokens", 0)
