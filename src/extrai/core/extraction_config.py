# extrai/core/extraction_config.py

from collections.abc import Callable
from dataclasses import dataclass


@dataclass
class ExtractionConfig:
    """Configuration for extraction workflows."""

    num_llm_revisions: int = 3
    num_counting_revisions: int = 3
    max_validation_retries_per_revision: int = 2
    consensus_threshold: float = 0.51
    counting_levenshtein_threshold: float = 0.85
    conflict_resolver: Callable | None = None
    use_hierarchical_extraction: bool = False
    use_structured_output: bool = False

    def __post_init__(self):
        """Validates configuration parameters."""
        if self.num_llm_revisions < 1:
            raise ValueError("num_llm_revisions must be at least 1")

        if self.max_validation_retries_per_revision < 1:
            raise ValueError("max_validation_retries_per_revision must be at least 1")

        if not (0.0 <= self.consensus_threshold <= 1.0):
            raise ValueError("consensus_threshold must be between 0.0 and 1.0")
