from pydantic import BaseModel, Field, model_validator
from typing import Optional, List, Literal
from datetime import date


class Language(BaseModel):
    input: Literal["tibetan", "chinese", "english", "bilingual"]
    output: Literal["tibetan", "chinese", "english", "bilingual"]


class Metadata(BaseModel):
    L1_domain: str
    L2_task_type: str
    L3_task: str
    task_code: str = Field(pattern=r"^\d+\.\d+\.\d+$")
    language: Language
    difficulty: Literal["easy", "medium", "hard"]
    turns: int = 1
    has_system: bool = False
    source: Literal["synthetic", "human", "translated", "augmented"]
    method: Literal["M1", "M2", "M3", "M4", "M5", "M6", "M7"]
    generator_model: Optional[str] = None
    quality_score: Optional[float] = Field(None, ge=0, le=1)
    quality_tier: Optional[Literal["gold", "silver", "bronze"]] = None
    dialect: Literal["standard", "amdo", "kham", "uke"] = "standard"
    script: Literal["tibetan", "wylie", "mixed"] = "tibetan"
    domain_tags: List[str] = Field(default_factory=list)
    requires_cultural_knowledge: bool = False
    created_at: str = Field(default_factory=lambda: date.today().isoformat())
    version: str = "1.0"


class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class SFTSample(BaseModel):
    """SFT data sample using OpenAI/ChatML messages format.

    Canonical format (stored on disk):
        {
          "id": "bo_sft_000001",
          "messages": [
            {"role": "system", "content": "..."},   # optional
            {"role": "user", "content": "..."},
            {"role": "assistant", "content": "..."}
          ],
          "metadata": { ... }
        }
    """
    id: str = Field(pattern=r"^bo_sft_\d{6}$")
    messages: List[Message]
    metadata: Metadata

    # ------------------------------------------------------------------
    # Convenience constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_instruction(
        cls,
        *,
        id: str,
        instruction: str,
        output: str,
        input: str = "",
        system_prompt: Optional[str] = None,
        metadata: Metadata,
    ) -> "SFTSample":
        """Build a SFTSample from instruction/output fields (used by methods)."""
        msgs: List[Message] = []
        if system_prompt:
            msgs.append(Message(role="system", content=system_prompt))
            metadata = metadata.model_copy(update={"has_system": True})
        user_content = f"{instruction}\n{input}".strip() if input else instruction
        msgs.append(Message(role="user", content=user_content))
        msgs.append(Message(role="assistant", content=output))
        return cls(id=id, messages=msgs, metadata=metadata)

    # ------------------------------------------------------------------
    # Convenience accessors (read from messages)
    # ------------------------------------------------------------------

    @property
    def system_prompt(self) -> Optional[str]:
        for m in self.messages:
            if m.role == "system":
                return m.content
        return None

    @property
    def instruction(self) -> str:
        for m in self.messages:
            if m.role == "user":
                return m.content
        return ""

    @property
    def input(self) -> str:
        """Backward-compat: input was merged into the user message content."""
        return ""

    @property
    def output(self) -> str:
        for m in self.messages:
            if m.role == "assistant":
                return m.content
        return ""

    # ------------------------------------------------------------------
    # Format conversion
    # ------------------------------------------------------------------

    def to_sharegpt_format(self) -> dict:
        convs = []
        for m in self.messages:
            if m.role == "system":
                convs.append({"from": "system", "value": m.content})
            elif m.role == "user":
                convs.append({"from": "human", "value": m.content})
            elif m.role == "assistant":
                convs.append({"from": "gpt", "value": m.content})
        return {"id": self.id, "conversations": convs, "metadata": self.metadata.model_dump()}
