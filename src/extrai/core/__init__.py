"""
Core logic for the Extrai project.

Contains modules responsible for key processing tasks like
database writing, LLM interaction, and workflow orchestration.
"""

from .analytics_collector import WorkflowAnalyticsCollector
from .conflict_resolvers import (
    SimilarityClusterResolver,
    default_conflict_resolver,
    prefer_most_common_resolver,
)
from .errors import (
    ConfigurationError,
    ConsensusProcessError,
    HydrationError,
    LLMAPICallError,
    LLMConfigurationError,
    LLMInteractionError,
    LLMOutputParseError,
    LLMOutputValidationError,
    WorkflowError,
)
from .example_json_generator import ExampleJSONGenerator
from .json_consensus import JSONConsensus
from .model_registry import ModelRegistry
from .prompt_builder import generate_system_prompt, generate_user_prompt_for_docs
from .result_processor import ResultProcessor, SQLAlchemyHydrator, persist_objects
from .schema_inspector import SchemaInspector
from .sqlmodel_generator import SQLModelCodeGenerator
from .workflow_orchestrator import WorkflowOrchestrator

__all__ = [
    # Errors
    "WorkflowError",
    "LLMInteractionError",
    "ConfigurationError",
    "ConsensusProcessError",
    "HydrationError",
    "LLMConfigurationError",
    "LLMOutputParseError",
    "LLMOutputValidationError",
    "LLMAPICallError",
    "ExampleGenerationError"
    # Classes & Functions
    "BaseLLMClient",
    "WorkflowAnalyticsCollector",
    "JSONConsensus",
    "default_conflict_resolver",
    "prefer_most_common_resolver",
    "generate_system_prompt",
    "generate_user_prompt_for_docs",
    "ModelRegistry",
    "SchemaInspector",
    "ResultProcessor",
    "SQLAlchemyHydrator",
    "persist_objects",
    "WorkflowOrchestrator",
    "SQLModelCodeGenerator",
    "ExampleJSONGenerator",
    "SimilarityClusterResolver",
]
