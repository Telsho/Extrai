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
from .json_consensus import JSONConsensus, default_conflict_resolver
from .prompt_builder import generate_system_prompt, generate_user_prompt_for_docs
from .schema_inspector import (
    generate_llm_schema_from_models,
    discover_sqlmodels_from_root,
    inspect_sqlalchemy_model,
)
from .sqlalchemy_hydrator import SQLAlchemyHydrator
from .db_writer import persist_objects
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
    "generate_system_prompt",
    "generate_user_prompt_for_docs",
    "generate_llm_schema_from_models",
    "discover_sqlmodels_from_root",
    "inspect_sqlalchemy_model",
    "SQLAlchemyHydrator",
    "persist_objects",
    "WorkflowOrchestrator",
    "SQLModelCodeGenerator",
    "ExampleJSONGenerator",
]
