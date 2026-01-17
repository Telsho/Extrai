"""
Core logic for the Extrai project.

Contains modules responsible for key processing tasks like
database writing, LLM interaction, and workflow orchestration.
"""

from .errors import (
    WorkflowError,
    LLMInteractionError,
    ConfigurationError,
    ConsensusProcessError,
    HydrationError,
    LLMConfigurationError,
    LLMOutputParseError,
    LLMOutputValidationError,
    LLMAPICallError,
    ExampleGenerationError,
)

from .base_llm_client import BaseLLMClient
from .analytics_collector import WorkflowAnalyticsCollector
from .json_consensus import JSONConsensus, default_conflict_resolver, prefer_most_common_resolver, SimilarityClusterResolver
from .prompt_builder import generate_system_prompt, generate_user_prompt_for_docs
from .model_registry import ModelRegistry
from .schema_inspector import SchemaInspector
from .result_processor import ResultProcessor, SQLAlchemyHydrator, persist_objects
from .workflow_orchestrator import WorkflowOrchestrator
from .sqlmodel_generator import SQLModelCodeGenerator
from .example_json_generator import ExampleJSONGenerator

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
    "ExampleGenerationError",
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
    "SimilarityClusterResolver"
]
