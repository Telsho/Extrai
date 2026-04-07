import logging
from typing import Any


class HierarchicalCoordinator:
    """
    Captures the model-level iteration policy (order, context passing, termination check)
    shared between standard and batch pipelines.
    """

    def __init__(self, model_registry: Any, logger: logging.Logger):
        self.model_registry = model_registry
        self.logger = logger

    def get_models(self) -> list[type]:
        """Returns the list of models in the order they should be processed."""
        return self.model_registry.models

    def is_final_step(self, index: int) -> bool:
        """Checks if the current step is the final one in the hierarchy."""
        return index >= len(self.model_registry.models) - 1

    def next_index(self, index: int) -> int:
        """Returns the index of the next step."""
        return index + 1

    def collect_previous_entities(self, completed_steps: list[dict]) -> list[dict]:
        """
        Collects entities from previous steps to be used as context for the current step.
        In the current implementation, this is a pass-through of all entities found so far.
        """
        return completed_steps
