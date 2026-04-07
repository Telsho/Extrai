# extrai/core/llm_runner.py

import asyncio
import logging
from typing import Any

from .analytics_collector import WorkflowAnalyticsCollector
from .base_llm_client import BaseLLMClient, ResponseMode
from .errors import (
    LLMAPICallError,
    LLMConfigurationError,
    LLMInteractionError,
    LLMOutputParseError,
    LLMOutputValidationError,
)
from .extraction_config import ExtractionConfig
from .model_registry import ModelRegistry
from .shared.consensus_runner import ConsensusRunner


class LLMRunner:
    """
    Manages LLM client rotation and extraction cycles.
    """

    def __init__(
        self,
        model_registry: ModelRegistry,
        llm_client: BaseLLMClient | list[BaseLLMClient],
        config: ExtractionConfig,
        analytics_collector: WorkflowAnalyticsCollector,
        logger: logging.Logger,
    ):
        self.model_registry = model_registry
        self.config = config
        self.analytics_collector = analytics_collector
        self.logger = logger
        self.clients = self._setup_clients(llm_client)
        self.client_index = 0
        self.consensus_runner = ConsensusRunner(config, analytics_collector, logger)
        self.logger.info(
            f"LLMRunner initialized with {len(self.clients)} client(s), "
            f"{config.num_llm_revisions} revisions per cycle"
        )

    def _setup_clients(
        self, llm_client: BaseLLMClient | list[BaseLLMClient]
    ) -> list[BaseLLMClient]:
        """
        Validates and normalizes LLM client input.

        Args:
            llm_client: Single client or list of clients

        Returns:
            List of validated LLM clients

        Raises:
            ValueError: If input is invalid
        """
        if isinstance(llm_client, list):
            if not llm_client:
                raise ValueError("llm_client list cannot be empty")

            if not all(isinstance(c, BaseLLMClient) for c in llm_client):
                raise ValueError(
                    "All items in llm_client list must be instances of BaseLLMClient"
                )
            clients = llm_client
        elif isinstance(llm_client, BaseLLMClient):
            clients = [llm_client]
        else:
            raise ValueError(
                "llm_client must be an instance of BaseLLMClient or a list of them"
            )

        # Set logger on all clients
        for client in clients:
            client.logger = self.logger

        return clients

    def get_next_client(self) -> BaseLLMClient:
        """
        Returns next client in rotation (round-robin).

        This enables load balancing across multiple LLM providers
        or API keys.

        Returns:
            Next BaseLLMClient in the rotation
        """
        client = self.clients[self.client_index]
        self.client_index = (self.client_index + 1) % len(self.clients)
        return client

    async def run_extraction_cycle(
        self, system_prompt: str, user_prompt: str
    ) -> list[dict[str, Any]]:
        """
        Runs a complete extraction cycle.

        Steps:
        1. Generate multiple revisions in parallel using different clients
        2. Normalize results to handle array ordering issues
        3. Run consensus mechanism to reconcile differences
        4. Return processed output

        Args:
            system_prompt: System prompt for LLM
            user_prompt: User prompt containing documents

        Returns:
            List of consensus entity dictionaries

        Raises:
            LLMInteractionError: If LLM calls fail
            ConsensusProcessError: If consensus fails
        """
        self.logger.info(
            f"Starting extraction cycle with {self.config.num_llm_revisions} revisions"
        )

        # Step 1: Generate revisions in parallel
        revisions = await self._generate_revisions(system_prompt, user_prompt)

        self.logger.debug(f"Generated {len(revisions)} revisions before consensus")

        # Step 3: Run consensus
        results = self.consensus_runner.run(revisions)

        self.logger.info(f"Extraction cycle completed with {len(results)} entities")

        return results

    async def run_structured_extraction_cycle(
        self,
        system_prompt: str,
        user_prompt: str,
        response_model: Any,
    ) -> list[dict[str, Any]]:
        """
        Runs a structured extraction cycle using response_model directly.
        """
        self.logger.info(
            f"Starting structured extraction cycle with {self.config.num_llm_revisions} revisions"
        )

        async def generate_single_revision(client_instance: BaseLLMClient) -> Any:
            """Helper to generate a single revision and extract the result."""
            results = await client_instance.generate_revisions(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                num_revisions=1,
                response_mode=ResponseMode.STRUCTURED,
                response_model=response_model,
            )
            return results[0]

        tasks = []
        for i in range(self.config.num_llm_revisions):
            client = self.get_next_client()
            tasks.append(asyncio.create_task(generate_single_revision(client)))

        try:
            results = await asyncio.gather(*tasks)
        except Exception as e:
            self.logger.error(f"Structured extraction failed: {e}")
            raise LLMInteractionError(f"Structured extraction failed: {e}") from e

        # Convert Pydantic models to dicts
        revisions = []
        for result in results:
            if hasattr(result, "model_dump"):
                revisions.append(result.model_dump(mode="json"))
            elif hasattr(result, "dict"):
                revisions.append(result.dict())
            else:
                self.logger.warning(f"Result {type(result)} is not a Pydantic model.")

        # Extract the list of entities if present
        processed_revisions = []
        for rev in revisions:
            if "entities" in rev and isinstance(rev["entities"], list):
                processed_revisions.append(rev["entities"])
            else:
                processed_revisions.append(rev)

        # Step 3: Consensus
        final_results = self.consensus_runner.run(processed_revisions)

        return final_results

    async def _generate_revisions(
        self, system_prompt: str, user_prompt: str
    ) -> list[Any]:
        """
        Generates multiple LLM revisions in parallel.

        Each revision is generated by a different client (round-robin)
        to distribute load and increase diversity of outputs.

        Args:
            system_prompt: System prompt for LLM
            user_prompt: User prompt containing documents

        Returns:
            List of revision outputs from LLM clients

        Raises:
            LLMInteractionError: If LLM interaction fails
        """
        tasks = []

        # Create parallel tasks for each revision
        for i in range(self.config.num_llm_revisions):
            client = self.get_next_client()

            self.logger.debug(
                f"Creating revision task {i + 1}/{self.config.num_llm_revisions} "
                f"with client {type(client).__name__}"
            )

            task = asyncio.create_task(
                client.generate_json_revisions(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    num_revisions=1,  # Each task generates 1 revision
                    model_schema_map=self.model_registry.model_map,
                    max_validation_retries_per_revision=self.config.max_validation_retries_per_revision,
                    analytics_collector=self.analytics_collector,
                )
            )
            tasks.append(task)

        # Execute all tasks in parallel
        try:
            revisions = await asyncio.gather(*tasks)

            # Validate we got results
            if not revisions and self.config.num_llm_revisions > 0:
                raise LLMInteractionError(
                    "LLM client returned no revisions despite being requested."
                )

            return revisions
        except (
            LLMConfigurationError,
            LLMOutputParseError,
            LLMOutputValidationError,
            LLMAPICallError,
            LLMInteractionError,
        ) as client_err:
            # Known LLM client errors
            self.logger.error(f"LLM client operation failed: {client_err}")
            raise LLMInteractionError(
                f"LLM client operation failed: {client_err}"
            ) from client_err

        except Exception as e:
            # Unexpected errors
            self.logger.error(f"Unexpected error during LLM interaction: {e}")
            raise LLMInteractionError(
                f"An unexpected error occurred during LLM interaction: {e}"
            ) from e

    def get_client_count(self) -> int:
        """
        Returns the number of LLM clients in rotation.

        Returns:
            Number of clients
        """
        return len(self.clients)

    def reset_client_rotation(self):
        """
        Resets client rotation to start from the first client.

        Useful for testing or ensuring consistent behavior.
        """
        self.client_index = 0
        self.logger.debug("Client rotation reset to index 0")

    def __repr__(self) -> str:
        """String representation of the runner."""
        return (
            f"LLMRunner(clients={len(self.clients)}, "
            f"revisions={self.config.num_llm_revisions})"
        )
