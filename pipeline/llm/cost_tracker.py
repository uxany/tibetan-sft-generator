from collections import defaultdict
from typing import Dict


class CostTracker:
    """Track token usage and estimated costs across providers."""

    # Pricing per 1M tokens: (input_cost, output_cost)
    PRICING: Dict[str, tuple] = {
        "gemini-3-pro": (1.25, 5.0),
        "gemini-3.1-pro-preview": (1.25, 10.0),
        "google/gemini-3.1-pro-preview": (1.25, 10.0),
        "gpt-5.4": (2.5, 10.0),
        "claude-opus-4-6": (15.0, 75.0),
        "qwen3-plus": (0.8, 2.0),
        "auto": (1.0, 4.0),  # ZenMux estimate
    }

    def __init__(self):
        self._records: Dict[str, Dict[str, float]] = defaultdict(
            lambda: {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "cost": 0.0,
                "requests": 0,
            }
        )

    def estimate_cost(
        self, model: str, prompt_tokens: int, completion_tokens: int
    ) -> float:
        """Estimate the cost for a single request given token counts."""
        input_price, output_price = self.PRICING.get(model, (1.0, 4.0))
        cost = (prompt_tokens * input_price + completion_tokens * output_price) / 1_000_000
        return cost

    def record(self, model: str, prompt_tokens: int, completion_tokens: int):
        """Record usage for a completed request."""
        cost = self.estimate_cost(model, prompt_tokens, completion_tokens)
        entry = self._records[model]
        entry["prompt_tokens"] += prompt_tokens
        entry["completion_tokens"] += completion_tokens
        entry["total_tokens"] += prompt_tokens + completion_tokens
        entry["cost"] += cost
        entry["requests"] += 1

    def summary(self) -> dict:
        """Return cost summary broken down by model and totals."""
        by_model = dict(self._records)
        total_cost = sum(v["cost"] for v in by_model.values())
        total_prompt = sum(v["prompt_tokens"] for v in by_model.values())
        total_completion = sum(v["completion_tokens"] for v in by_model.values())
        total_requests = sum(v["requests"] for v in by_model.values())
        return {
            "by_model": by_model,
            "total": {
                "cost": total_cost,
                "prompt_tokens": total_prompt,
                "completion_tokens": total_completion,
                "total_tokens": total_prompt + total_completion,
                "requests": total_requests,
            },
        }
