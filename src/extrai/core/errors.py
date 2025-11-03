# extrai/core/errors.py
"""
Centralized custom exceptions for the project.
This module consolidates exceptions from various core components
to provide a single point of reference for error types.
"""

from typing import Any, Dict, Optional

from pydantic import ValidationError

# --- Workflow Orchestration Errors ---


class WorkflowError(Exception):
    """Base exception for errors occurring within the workflow orchestration."""

    pass


class LLMInteractionError(WorkflowError):
    """
    Custom exception for errors specifically related to LLM interaction.
    This can include issues with parsing LLM output, validation failures against
    a schema, or errors during the API call to the LLM service, typically
    wrapping a more specific LLMClientError.
    """

    pass


class ConfigurationError(WorkflowError):
    """
    Custom exception for configuration-related errors within the workflow
    orchestration, such as issues with schema discovery or setup parameters.
    (Note: This is specific to workflow configuration, distinct from LLMClient's
    LLMConfigurationError).
    """

    pass


class ConsensusProcessError(WorkflowError):
    """Custom exception for errors occurring during the JSON consensus process."""

    pass


class HydrationError(WorkflowError):
    """Custom exception for errors during SQLAlchemy object hydration from JSON."""

    pass


# --- LLM Client Errors ---


class LLMClientError(Exception):
    """Base exception for errors originating from an LLM client."""

    pass


class LLMOutputParseError(LLMClientError):
    """
    Raised when LLM output cannot be parsed as JSON after all retries.
    Carries the raw content that failed to parse and the original exception.
    """

    def __init__(
        self,
        message: str,
        raw_content: Optional[str] = None,
        original_exception: Optional[Exception] = None,
    ):
        super().__init__(message)
        self.raw_content = raw_content
        self.original_exception = original_exception

    def __str__(self) -> str:
        base_message = super().__str__()
        if self.raw_content:
            # Truncate raw_content to avoid overly long error messages
            return f"{base_message} - Raw content (first 200 chars): '{self.raw_content[:200]}...'"
        return base_message


class LLMOutputValidationError(LLMClientError):
    """
    Raised when LLM JSON output fails schema validation after all retries.
    Carries the parsed JSON that failed validation and the original Pydantic validation error.
    """

    def __init__(
        self,
        message: str,
        parsed_json: Optional[Dict[str, Any]] = None,
        validation_error: Optional[Any] = None,
    ):  # PydanticValidationError type hint can be 'Any' for simplicity here or more specific if PydanticValidationError is imported
        super().__init__(message)
        self.parsed_json = parsed_json
        self.validation_error = (
            validation_error  # Stores the original Pydantic ValidationError
        )

    def __str__(self) -> str:
        base_message = super().__str__()
        if self.parsed_json:
            # Convert parsed_json to string and truncate
            parsed_json_str = str(self.parsed_json)
            return f"{base_message} - Parsed JSON (first 200 chars): {parsed_json_str[:200]}..."
        return base_message


class LLMConfigurationError(LLMClientError):
    """
    Raised for configuration issues specific to the LLM client,
    such as invalid API keys, model names, or malformed schema strings
    passed to the client.
    """

    pass


class LLMRevisionGenerationError(LLMClientError):
    """
    Raised when all LLM revisions fail for a given prompt.
    Contains a list of all the exceptions that occurred.
    """

    def __init__(self, message: str, failures: list[Exception]):
        super().__init__(message)
        self.failures = failures

    def __str__(self) -> str:
        base_message = super().__str__()
        failure_details = "\n".join(
            [f"  - {type(e).__name__}: {e}" for e in self.failures]
        )
        return f"{base_message}\nUnderlying failures:\n{failure_details}"


class LLMAPICallError(LLMClientError):
    """
    Raised when an underlying LLM API call itself fails (e.g., network issues,
    server errors from the LLM provider, authentication failures).
    """

    pass


class SQLModelCodeGeneratorError(Exception):
    """Custom exception for SQLModelCodeGenerator errors."""

    pass


class SQLModelInstantiationValidationError(SQLModelCodeGeneratorError):
    """
    Custom exception for when a dynamically generated SQLModel class
    raises a Pydantic ValidationError upon default instantiation.
    This is often expected for models with required fields without defaults.
    """

    def __init__(
        self, model_name: str, validation_error: ValidationError, generated_code: str
    ):
        self.model_name = model_name
        self.validation_error = validation_error
        self.generated_code = generated_code
        super().__init__(
            f"Default instantiation of '{model_name}' failed with ValidationError (often expected for required fields): {validation_error}\n"
            f"Generated code for module:\n{generated_code}"
        )


class ExampleGenerationError(Exception):
    """Custom exception for errors during example JSON generation."""

    def __init__(self, message: str, original_exception: Optional[Exception] = None):
        super().__init__(message)
        self.original_exception = original_exception
