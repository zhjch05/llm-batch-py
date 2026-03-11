from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelPricing:
    input_per_million: float
    output_per_million: float = 0.0


DEFAULT_PRICING: dict[str, ModelPricing] = {
    "claude-3-5-haiku-latest": ModelPricing(input_per_million=0.80, output_per_million=4.00),
    "claude-3-7-sonnet-latest": ModelPricing(input_per_million=3.00, output_per_million=15.00),
    "gpt-4o-mini": ModelPricing(input_per_million=0.15, output_per_million=0.60),
    "gpt-4o": ModelPricing(input_per_million=2.50, output_per_million=10.00),
    "text-embedding-3-small": ModelPricing(input_per_million=0.02, output_per_million=0.0),
    "text-embedding-3-large": ModelPricing(input_per_million=0.13, output_per_million=0.0),
}
