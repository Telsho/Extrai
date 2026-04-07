import asyncio
import logging
from abc import ABC, abstractmethod
from collections.abc import Callable
from contextvars import ContextVar
from enum import Enum
from typing import Any

from sqlmodel import SQLModel

from extrai.core.analytics_collector import WorkflowAnalyticsCollector
from extrai.core.errors import (
    LLMAPICallError,
    LLMOutputParseError,
    LLMOutputValidationError,
    LLMRevisionGenerationError,
)
from extrai.utils.llm_output_processing import (
    process_and_validate_llm_output,
    process_and_validate_raw_json,
)

revision_context: ContextVar[str] = ContextVar("revision_context", default="")


class ResponseMode(Enum):
    """Defines the format of the LLM response."""

    TEXT = "text"
    STRUCTURED = "structured"


class ProviderBatchStatus(Enum):
    """Standardized batch job status across providers."""

    PROCESSING = "processing"
    PENDING = "pending"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class BaseLLMClient(ABC):
    """
    Abstract base class for LLM clients.

    Handles LLM API calls with retry logic, validation, and concurrent revision generation.
    """

    def __init__(
        self,
        api_key: str,
        model_name: str,
        base_url: str | None = None,
        temperature: float | None = 0.7,
        logger: logging.Logger | None = None,
    ):
        self.api_key = api_key
        self.model_name = model_name
        self.base_url = base_url
        self.temperature = temperature
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        if not logger:
            self.logger.setLevel(logging.WARNING)

    @abstractmethod
    async def _execute_llm_call(
        self,
        system_prompt: str,
        user_prompt: str,
        response_mode: ResponseMode = ResponseMode.TEXT,
        response_model: type[Any] | None = None,
        analytics_collector: WorkflowAnalyticsCollector | None = None,
        **kwargs: Any,
    ) -> Any:
        """
        Makes the actual API call to the LLM.

        Args:
            system_prompt: The system prompt for the LLM.
            user_prompt: The user prompt for the LLM.
            response_mode: Whether to return raw text or structured output.
            response_model: The Pydantic/SQLModel class for structured responses.
            analytics_collector: Optional analytics collector for tracking costs.
            **kwargs: Additional provider-specific arguments.

        Returns:
            - TEXT mode: Raw string content from the LLM
            - STRUCTURED mode: Instance of response_model

        Raises:
            LLMAPICallError: If the API call fails.
            NotImplementedError: If the provider doesn't support the requested mode.
        """
        ...

    async def _generate_single_revision(
        self,
        system_prompt: str,
        user_prompt: str,
        max_attempts: int,
        validation_fn: Callable[[Any], dict[str, Any]] | None,
        revision_index: int,
        response_mode: ResponseMode = ResponseMode.TEXT,
        response_model: type[Any] | None = None,
        analytics_collector: WorkflowAnalyticsCollector | None = None,
    ) -> dict[str, Any]:
        """
        Generates a single revision with retry logic.

        Args:
            validation_fn: Optional function to validate/transform the response.
                          If None (structured mode), response is used as-is.
        """
        last_error: Exception | None = None

        for attempt in range(max_attempts):
            revision_info = f"Revision {revision_index + 1}, Attempt {attempt + 1}"
            token = revision_context.set(revision_info)
            try:
                # Execute LLM call
                response = await self._execute_llm_call(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    response_mode=response_mode,
                    response_model=response_model,
                    analytics_collector=analytics_collector,
                )

                # Validate/transform if needed
                if validation_fn:
                    if not response:
                        raise ValueError("LLM returned empty content")
                    result = validation_fn(response)
                else:
                    # Structured mode - convert model to dict
                    if hasattr(response, "model_dump"):
                        result = response.model_dump()
                    elif hasattr(response, "dict"):
                        result = response.dict()
                    else:
                        result = response

                if analytics_collector:
                    analytics_collector.record_llm_api_call_success()

                self.logger.debug(f"{revision_info}: Success")
                return result

            except Exception as e:
                last_error = e
                self.logger.warning(f"{revision_info}: {type(e).__name__} - {e}")

                # Record analytics for final attempt
                if attempt + 1 == max_attempts and analytics_collector:
                    if isinstance(e, LLMAPICallError):
                        analytics_collector.record_llm_api_call_failure()
                    elif isinstance(e, LLMOutputParseError):
                        analytics_collector.record_llm_output_parse_error()
                    elif isinstance(e, LLMOutputValidationError):
                        analytics_collector.record_llm_output_validation_error()

                # Retry with backoff
                if attempt + 1 < max_attempts:
                    delay = (
                        0.5
                        * (attempt + 1)
                        * (2 if isinstance(e, LLMAPICallError) else 1)
                    )
                    self.logger.info(f"Retrying in {delay:.2f}s...")
                    await asyncio.sleep(delay)
            finally:
                revision_context.reset(token)

        raise last_error or RuntimeError("Generation failed without recorded error")

    async def generate_revisions(
        self,
        system_prompt: str,
        user_prompt: str,
        num_revisions: int,
        max_attempts_per_revision: int = 3,
        validation_fn: Callable[[Any], dict[str, Any]] | None = None,
        response_mode: ResponseMode = ResponseMode.TEXT,
        response_model: type[Any] | None = None,
        analytics_collector: WorkflowAnalyticsCollector | None = None,
    ) -> list[dict[str, Any]]:
        """
        Generates multiple revisions concurrently.

        Args:
            validation_fn: Optional validation/transformation function.
                          Required for TEXT mode, not used for STRUCTURED mode.
        """
        tasks = [
            self._generate_single_revision(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_attempts=max(1, max_attempts_per_revision),
                validation_fn=validation_fn,
                revision_index=i,
                response_mode=response_mode,
                response_model=response_model,
                analytics_collector=analytics_collector,
            )
            for i in range(num_revisions)
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Separate successes from failures
        successful = []
        failures = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                failures.append(result)
                self.logger.error(f"Revision {i + 1} failed: {result}")
            elif isinstance(result, list):
                successful.extend(result)
            else:
                successful.append(result)

        self.logger.info(
            f"Generated {len(successful)}/{num_revisions} revisions successfully"
        )

        if not successful and num_revisions > 0:
            raise LLMRevisionGenerationError(
                "All LLM revisions failed.", failures=failures
            )

        if failures:
            self.logger.warning(f"{len(failures)} revision(s) failed")

        return successful

    # =========================================================================
    # HIGH-LEVEL CONVENIENCE METHODS
    # =========================================================================

    async def generate_json_revisions(
        self,
        system_prompt: str,
        user_prompt: str,
        num_revisions: int,
        model_schema_map: dict[str, type[SQLModel]],
        max_validation_retries_per_revision: int = 3,
        use_structured_output: bool = False,
        analytics_collector: WorkflowAnalyticsCollector | None = None,
    ) -> list[dict[str, Any]]:
        """
        Generates JSON revisions validated against SQLModel schemas.
        """
        response_mode = ResponseMode.TEXT
        response_model = None
        validation_fn = None

        if use_structured_output and len(model_schema_map) == 1:
            response_mode = ResponseMode.STRUCTURED
            response_model = list(model_schema_map.values())[0]
        else:
            if use_structured_output:
                self.logger.warning(
                    "Structured output with multiple schemas not supported. Using text mode."
                )

            def validation_fn(content: str) -> list[dict[str, Any]]:
                revision_info = revision_context.get()
                return process_and_validate_llm_output(
                    raw_llm_content=content,
                    model_schema_map=model_schema_map,
                    revision_info_for_error=revision_info,
                    analytics_collector=analytics_collector,
                )

        return await self.generate_revisions(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            num_revisions=num_revisions,
            max_attempts_per_revision=max_validation_retries_per_revision,
            validation_fn=validation_fn,
            response_mode=response_mode,
            response_model=response_model,
            analytics_collector=analytics_collector,
        )

    async def generate_and_validate_raw_json_output(
        self,
        system_prompt: str,
        user_prompt: str,
        num_revisions: int,
        max_validation_retries_per_revision: int = 3,
        target_json_schema: dict[str, Any] | None = None,
        attempt_unwrap: bool = True,
        analytics_collector: WorkflowAnalyticsCollector | None = None,
    ) -> list[dict[str, Any]]:
        """
        Generates JSON revisions validated against a raw JSON schema.
        """

        def validation_fn(content: str) -> dict[str, Any]:
            revision_info = revision_context.get()
            return process_and_validate_raw_json(
                raw_llm_content=content,
                revision_info_for_error=revision_info,
                target_json_schema=target_json_schema,
                attempt_unwrap=attempt_unwrap,
            )

        return await self.generate_revisions(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            num_revisions=num_revisions,
            max_attempts_per_revision=max_validation_retries_per_revision,
            validation_fn=validation_fn,
            analytics_collector=analytics_collector,
        )

    # =========================================================================
    # BATCH PROCESSING (Optional - Provider-Specific)
    # =========================================================================

    async def create_batch_job(
        self,
        requests: list[dict[str, Any]],
        endpoint: str = "/v1/chat/completions",
        completion_window: str = "24h",
        metadata: dict[str, str] | None = None,
        response_model: type[Any] | None = None,
    ) -> Any:
        """Creates a batch job. Override in subclass if supported."""
        raise NotImplementedError("Batch processing is not supported by this provider")

    async def get_batch_status(self, batch_id: str) -> "ProviderBatchStatus":
        """Retrieves a standardized batch job status. Override in subclass if supported."""
        raise NotImplementedError("Batch processing is not supported by this provider")

    async def list_batch_jobs(self, limit: int = 20, after: str | None = None) -> Any:
        """Lists batch jobs. Override in subclass if supported."""
        raise NotImplementedError("Batch processing is not supported by this provider")

    async def cancel_batch_job(self, batch_id: str) -> Any:
        """Cancels a batch job. Override in subclass if supported."""
        raise NotImplementedError("Batch processing is not supported by this provider")

    async def retrieve_batch_results(self, file_id: str) -> str:
        """Retrieves batch results. Override in subclass if supported."""
        raise NotImplementedError("Batch processing is not supported by this provider")

    def extract_content_from_batch_response(
        self, response: dict[str, Any]
    ) -> str | None:
        """Extracts content from batch response. Override in subclass if supported."""
        raise NotImplementedError("Batch processing is not supported by this provider")

    def prepare_request(
        self,
        system_prompt: str,
        user_prompt: str,
        json_schema: Any | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Prepares a request dictionary for batch processing.
        Override in subclasses to provide provider-specific formatting.
        """
        raise NotImplementedError(
            "Batch request preparation is not supported by this provider"
        )
