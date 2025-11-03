import logging
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Type
import asyncio
from sqlmodel import SQLModel

from extrai.core.errors import (
    LLMOutputParseError,
    LLMOutputValidationError,
    LLMAPICallError,
    LLMRevisionGenerationError,
)
from extrai.utils.llm_output_processing import (
    process_and_validate_llm_output,
    process_and_validate_raw_json,
)
from extrai.core.analytics_collector import (
    WorkflowAnalyticsCollector,
)


class BaseLLMClient(ABC):
    """
    Abstract base class for LLM clients.

    This class provides a common structure for interacting with various LLM providers.
    It handles the generic logic for generating multiple JSON revisions, including
    retries and validation, while delegating the actual LLM API call to subclasses.

    Attributes:
        api_key (str): The API key for authenticating with the LLM service.
        model_name (str): The specific model identifier to be used for generation.
        base_url (Optional[str]): Base URL for the API, if applicable.
        temperature (Optional[float]): The sampling temperature for generation.
    """

    def __init__(
        self,
        api_key: str,
        model_name: str,
        base_url: Optional[str] = None,
        temperature: Optional[float] = 0.7,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initializes the BaseLLMClient.

        Args:
            api_key: The API key for the LLM service.
            model_name: The model identifier.
            base_url: Optional base URL for the LLM API.
            temperature: Optional sampling temperature.
            logger: An optional logger instance. If not provided, a default logger is created.
        """
        self.api_key = api_key
        self.model_name = model_name
        self.base_url = base_url
        self.temperature = temperature
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        if not logger:
            self.logger.setLevel(logging.WARNING)

    @abstractmethod
    async def _execute_llm_call(self, system_prompt: str, user_prompt: str) -> str:
        """
        Makes the actual API call to the LLM and returns the raw string content.

        This method must be implemented by concrete subclasses to interact with
        their specific LLM provider's API.

        Args:
            system_prompt: The system prompt for the LLM.
            user_prompt: The user prompt for the LLM.

        Returns:
            The raw string content from the LLM response. Should return an empty
            string if the LLM response was empty but did not constitute an API error.

        Raises:
            LLMAPICallError: If the underlying API call fails.
            Exception: For other unexpected errors during the API call.
        """
        ...

    async def _attempt_single_generation_and_validation(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        validation_callable: Callable[[str, str], Dict[str, Any]],
        revision_info_for_error: str,
    ) -> Dict[str, Any]:
        """
        Performs one LLM call and one validation attempt.
        """
        raw_response_content = await self._execute_llm_call(
            system_prompt=system_prompt, user_prompt=user_prompt
        )

        if not raw_response_content:
            raise ValueError(f"{revision_info_for_error}: LLM returned empty content.")

        validated_data = validation_callable(
            raw_response_content, revision_info_for_error
        )
        return validated_data

    async def _generate_one_revision_with_retries(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        max_attempts: int,
        validation_callable: Callable[[str, str], Dict[str, Any]],
        analytics_collector: Optional[WorkflowAnalyticsCollector],
        revision_index: int,
    ) -> Dict[str, Any]:
        """
        Manages the retry loop for generating a single valid revision.
        """
        last_error: Optional[Exception] = None
        for attempt in range(max_attempts):
            revision_info_for_error = (
                f"Revision {revision_index + 1}, Attempt {attempt + 1}"
            )
            try:
                validated_data = await self._attempt_single_generation_and_validation(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    validation_callable=validation_callable,
                    revision_info_for_error=revision_info_for_error,
                )
                if analytics_collector:
                    analytics_collector.record_llm_api_call_success()
                self.logger.debug(
                    f"{revision_info_for_error}: Successfully generated and validated."
                )
                return validated_data
            except (LLMOutputParseError, LLMOutputValidationError, ValueError) as e:
                self.logger.warning(
                    f"{revision_info_for_error}: Validation or parsing error: {e}"
                )
                last_error = e
            except LLMAPICallError as e:
                self.logger.warning(f"{revision_info_for_error}: API call error: {e}")
                last_error = e
            except Exception as e:
                self.logger.warning(
                    f"{revision_info_for_error}: Unexpected error: {type(e).__name__} - {e}"
                )
                last_error = Exception(
                    f"{revision_info_for_error}: Unexpected error: {type(e).__name__} - {e}"
                )

            if attempt + 1 < max_attempts:
                delay_multiplier = 2 if isinstance(last_error, LLMAPICallError) else 1
                delay = 0.5 * (attempt + 1) * delay_multiplier
                self.logger.info(
                    f"{revision_info_for_error}: Retrying in {delay:.2f} seconds..."
                )
                await asyncio.sleep(delay)

        if last_error:
            if analytics_collector:
                if isinstance(last_error, LLMAPICallError):
                    analytics_collector.record_llm_api_call_failure()
                elif isinstance(last_error, LLMOutputParseError):
                    analytics_collector.record_llm_output_parse_error()
                elif isinstance(last_error, LLMOutputValidationError):
                    analytics_collector.record_llm_output_validation_error()
            raise last_error

        # This line should be unreachable, but linters might complain.
        raise RuntimeError("Revision generation failed without a recorded error.")

    async def _generate_all_revisions(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        num_revisions: int,
        max_validation_retries_per_revision: int,
        validation_callable: Callable[[str, str], Any],
        analytics_collector: Optional[WorkflowAnalyticsCollector] = None,
    ) -> List[Any]:
        """
        Orchestrates the generation of all revisions concurrently.
        """
        if max_validation_retries_per_revision < 1:
            actual_attempts_per_revision = 1
        else:
            actual_attempts_per_revision = max_validation_retries_per_revision

        tasks = [
            self._generate_one_revision_with_retries(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_attempts=actual_attempts_per_revision,
                validation_callable=validation_callable,
                analytics_collector=analytics_collector,
                revision_index=i,
            )
            for i in range(num_revisions)
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        successful_revisions = []
        failures = []
        for i, res in enumerate(results):
            if isinstance(res, Exception):
                failures.append(res)
                self.logger.error(f"Revision {i + 1} failed: {res}")
            elif isinstance(res, list):
                successful_revisions.extend(res)
            else:
                successful_revisions.append(res)

        num_successful = len(successful_revisions)
        num_failures = len(failures)
        self.logger.info(
            f"Revision generation summary: {num_successful} successful, {num_failures} failed."
        )

        if failures:
            # If all revisions failed, raise an aggregate error.
            # If some succeeded, this error could be logged or handled differently.
            if not successful_revisions:
                self.logger.error("All LLM revisions failed.")
                raise LLMRevisionGenerationError(
                    "All LLM revisions failed.", failures=failures
                )
            self.logger.warning(
                f"Partial failure in revision generation: {num_failures} revision(s) failed."
            )

        return successful_revisions

    async def generate_json_revisions(
        self,
        system_prompt: str,
        user_prompt: str,
        num_revisions: int,
        model_schema_map: Dict[str, Type[SQLModel]],
        max_validation_retries_per_revision: int,
        analytics_collector: Optional[WorkflowAnalyticsCollector] = None,
    ) -> List[Dict[str, Any]]:
        """
        Generates multiple JSON output revisions from the LLM, validating against a SQLModel.
        """

        def validation_callable(
            content: str, revision_info: str
        ) -> List[Dict[str, Any]]:
            return process_and_validate_llm_output(
                raw_llm_content=content,
                model_schema_map=model_schema_map,
                revision_info_for_error=revision_info,
                analytics_collector=analytics_collector,
            )

        return await self._generate_all_revisions(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            num_revisions=num_revisions,
            max_validation_retries_per_revision=max_validation_retries_per_revision,
            validation_callable=validation_callable,
            analytics_collector=analytics_collector,
        )

    async def generate_and_validate_raw_json_output(
        self,
        system_prompt: str,
        user_prompt: str,
        num_revisions: int,
        max_validation_retries_per_revision: int,
        target_json_schema: Optional[Dict[str, Any]] = None,
        analytics_collector: Optional[WorkflowAnalyticsCollector] = None,
    ) -> List[Dict[str, Any]]:
        """
        Generates multiple JSON output revisions, validating against a raw JSON schema.
        """

        def validation_callable(content: str, revision_info: str) -> Dict[str, Any]:
            return process_and_validate_raw_json(
                raw_llm_content=content,
                revision_info_for_error=revision_info,
                target_json_schema=target_json_schema,
            )

        return await self._generate_all_revisions(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            num_revisions=num_revisions,
            max_validation_retries_per_revision=max_validation_retries_per_revision,
            validation_callable=validation_callable,
            analytics_collector=analytics_collector,
        )
