from abc import ABC, abstractmethod
from typing import Optional

import httpx

from .response import LLMResponse


class BaseLLMClient(ABC):
    """Base class for all LLM provider clients."""

    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: Optional[str] = None,
        **kwargs,
    ):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url
        self._client: Optional[httpx.AsyncClient] = None  # initialized lazily

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.8,
        max_tokens: int = 2048,
        **kwargs,
    ) -> LLMResponse:
        pass

    async def close(self):
        if self._client:
            await self._client.aclose()
            self._client = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.close()
