# extrai/core/cost_calculator.py
import typing
from dataclasses import dataclass

from .analytics_collector import WorkflowAnalyticsCollector
from .pricing_updater import load_pricing_data, update_prices_if_stale

if typing.TYPE_CHECKING:
    from .base_llm_client import BaseLLMClient

@dataclass
class ModelCosts:
    """A simple dataclass to store cost per million tokens for a model."""

    input_cost_per_million: float
    output_cost_per_million: float
    input_cached_cost_per_million: float | None = None


# Costs are per million tokens
update_prices_if_stale()
pricing_data = load_pricing_data()

MODEL_COSTS = {}
if pricing_data:
    for item in pricing_data.get("prices", []):
        MODEL_COSTS[item["id"]] = ModelCosts(
            input_cost_per_million=item["input"],
            output_cost_per_million=item["output"],
            input_cached_cost_per_million=item["input_cached"] if item.get("input_cached") else None,
        )


def calculate_cost(
    model_name: str, input_tokens: int, output_tokens: int, is_batch: bool = False
) -> float | None:
    """
    Calculates the cost of a single LLM call.

    Args:
        model_name: The name of the model used.
        input_tokens: The number of input tokens.
        output_tokens: The number of output tokens.
        is_batch: If True, uses the cached input cost if available.

    Returns:
        The calculated cost, or None if the model is not found.
    """
    if model_name not in MODEL_COSTS:
        return None

    costs = MODEL_COSTS[model_name]

    if is_batch and costs.input_cached_cost_per_million is not None:
        input_cost = (input_tokens / 1_000_000) * costs.input_cached_cost_per_million
    else:
        input_cost = (input_tokens / 1_000_000) * costs.input_cost_per_million

    output_cost = (output_tokens / 1_000_000) * costs.output_cost_per_million
    return input_cost + output_cost


def track_usage_from_response(
    response_dict: dict,
    client: "BaseLLMClient",
    analytics_collector: WorkflowAnalyticsCollector,
    batch_id: str,
    extra_details: dict | None = None,
) -> None:
    """
    Extracts usage from a batch response wrapper and records it via analytics collector.
    """
    # Check for OpenAI-style batch response structure: {"response": {"body": {"usage": {...}}}}
    if (
        isinstance(response_dict, dict)
        and "response" in response_dict
        and "body" in response_dict["response"]
        and "usage" in response_dict["response"]["body"]
    ):
        usage = response_dict["response"]["body"]["usage"]
        input_tokens = usage.get("prompt_tokens", 0)
        output_tokens = usage.get("completion_tokens", 0)

        cost = calculate_cost(
            client.model_name,
            input_tokens,
            output_tokens,
            is_batch=True,
        )

        details = {"batch_id": batch_id}
        if extra_details:
            details.update(extra_details)

        analytics_collector.record_llm_usage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model=client.model_name,
            cost=cost,
            details=details,
        )
