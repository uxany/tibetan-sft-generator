import time
import logging
from typing import Optional

import httpx

from ..base import BaseLLMClient
from ..response import LLMResponse
from ..cost_tracker import CostTracker

logger = logging.getLogger(__name__)


class OpenAIClient(BaseLLMClient):
    """OpenAI-compatible client."""

    PROVIDER = "openai"
    DEFAULT_BASE_URL = "https://api.openai.com/v1"

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-5.4",
        base_url: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(api_key, model, base_url or self.DEFAULT_BASE_URL, **kwargs)

    def _ensure_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                timeout=httpx.Timeout(120.0, connect=10.0),
            )
        return self._client

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.8,
        max_tokens: int = 2048,
        response_format: Optional[dict] = None,
        **kwargs,
    ) -> LLMResponse:
        """
        Generate text using an OpenAI-compatible chat completions endpoint.

        response_format:
            Optional OpenAI JSON mode / structured output parameter.
            Example:
                {"type": "json_object"}

        kwargs:
            Extra provider-specific parameters. They will be merged into the request body.
        """
        client = self._ensure_client()

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        body = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        if response_format is not None:
            body["response_format"] = response_format

        # Allow provider-specific options, e.g. top_p, seed, extra_body fields, etc.
        body.update(kwargs)

        start = time.time()

        try:
            resp = await client.post("/chat/completions", json=body)
            resp.raise_for_status()
            data = resp.json()
        except httpx.HTTPStatusError as exc:
            logger.error(
                "OpenAI API error %s: %s",
                exc.response.status_code,
                exc.response.text,
            )
            raise
        except httpx.RequestError as exc:
            logger.error("OpenAI request failed: %s", exc)
            raise

        latency_ms = (time.time() - start) * 1000

        choice = data["choices"][0]
        usage = data.get("usage", {})

        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)

        cost = CostTracker().estimate_cost(
            self.model,
            prompt_tokens,
            completion_tokens,
        )

        message = choice.get("message", {})
        content = message.get("content") or ""

        return LLMResponse(
            content=content,
            model=self.model,
            provider=self.PROVIDER,
            usage={
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
            latency_ms=latency_ms,
            cost=cost,
            finish_reason=choice.get("finish_reason"),
        )