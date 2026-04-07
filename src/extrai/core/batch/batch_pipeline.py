import asyncio
import logging
from typing import Any, Union

from sqlalchemy.orm import Session
from sqlmodel import SQLModel

from extrai.core.analytics_collector import WorkflowAnalyticsCollector
from extrai.core.base_llm_client import BaseLLMClient
from extrai.core.batch_models import BatchJobStatus, BatchProcessResult
from extrai.core.client_rotator import ClientRotator
from extrai.core.entity_counter import EntityCounter
from extrai.core.extraction_config import ExtractionConfig
from extrai.core.extraction_context_preparer import ExtractionContextPreparer
from extrai.core.extraction_request_factory import ExtractionRequestFactory
from extrai.core.model_registry import ModelRegistry
from extrai.core.model_wrapper_builder import ModelWrapperBuilder
from extrai.core.prompt_builder import PromptBuilder
from extrai.core.shared.consensus_runner import ConsensusRunner
from extrai.core.shared.hierarchical_coordinator import HierarchicalCoordinator

from .batch_processor import BatchProcessor
from .batch_result_retriever import BatchResultRetriever
from .batch_status_checker import BatchStatusChecker
from .batch_submitter import BatchSubmitter


class BatchPipeline:
    """Manages batch extraction workflows."""

    def __init__(
        self,
        model_registry: ModelRegistry,
        llm_client: Union["BaseLLMClient", list["BaseLLMClient"]],
        config: ExtractionConfig,
        analytics_collector: WorkflowAnalyticsCollector,
        logger: logging.Logger,
        counting_llm_client: BaseLLMClient | None = None,
    ):
        self.model_registry = model_registry
        self.config = config
        self.analytics_collector = analytics_collector
        self.logger = logger

        self.client_rotator = ClientRotator(llm_client)
        self.prompt_builder = PromptBuilder(model_registry, logger=logger)
        c_client = counting_llm_client or llm_client
        if isinstance(c_client, list):
            c_client = c_client[0]

        self.entity_counter = EntityCounter(
            model_registry, c_client, config, analytics_collector, logger=logger
        )
        self.context_preparer = ExtractionContextPreparer(
            model_registry,
            analytics_collector,
            config.max_validation_retries_per_revision,
            logger=logger,
        )
        self.model_wrapper_builder = ModelWrapperBuilder()
        self.consensus_runner = ConsensusRunner(config, analytics_collector, logger)
        self.request_factory = ExtractionRequestFactory(
            model_registry,
            self.prompt_builder,
            self.model_wrapper_builder,
            logger=logger,
        )
        self.hierarchical_coordinator = HierarchicalCoordinator(model_registry, logger)

        # Instantiate components
        self.submitter = BatchSubmitter(
            model_registry,
            self.client_rotator,
            config,
            self.entity_counter,
            self.context_preparer,
            self.request_factory,
            logger,
        )
        self.status_checker = BatchStatusChecker(
            self.client_rotator, self.entity_counter, logger
        )
        self.retriever = BatchResultRetriever(
            model_registry, logger, analytics_collector
        )
        self.processor = BatchProcessor(
            model_registry,
            config,
            analytics_collector,
            self.client_rotator,
            self.entity_counter,
            self.submitter,
            self.status_checker,
            self.retriever,
            self.consensus_runner,
            self.hierarchical_coordinator,
            logger,
        )

    async def submit_batch(
        self,
        db_session: Session,
        input_strings: list[str],
        extraction_example_json: str = "",
        extraction_example_object: SQLModel | list[SQLModel] | None = None,
        custom_extraction_process: str | list[str] = "",
        custom_extraction_guidelines: str | list[str] = "",
        custom_final_checklist: str | list[str] = "",
        custom_context: str | list[str] = "",
        count_entities: bool = False,
        custom_counting_context: str | list[str] = "",
    ) -> str:
        """Submits a batch job and returns root_batch_id."""
        return await self.submitter.submit_batch(
            db_session,
            input_strings,
            extraction_example_json,
            extraction_example_object,
            custom_extraction_process,
            custom_extraction_guidelines,
            custom_final_checklist,
            custom_context,
            count_entities,
            custom_counting_context,
        )

    async def create_continuation_batch(
        self,
        db_session: Session,
        original_batch_id: str,
        new_config_dict: dict[str, Any],
        start_from_step_index: int,
    ) -> str:
        """
        Creates a new batch cycle continuing from a previous batch's state.
        Copies completed steps up to start_from_step_index into the new batch.
        """
        return await self.submitter.create_continuation_batch(
            db_session, original_batch_id, new_config_dict, start_from_step_index
        )

    async def get_status(
        self, root_batch_id: str, db_session: Session
    ) -> BatchJobStatus:
        return await self.status_checker.get_status(root_batch_id, db_session)

    async def process_batch(
        self, root_batch_id: str, db_session: Session
    ) -> BatchProcessResult:
        return await self.processor.process_batch(root_batch_id, db_session)

    async def monitor_batch_job(
        self, root_batch_id: str, db_session: Session, poll_interval: int = 60
    ) -> BatchProcessResult:
        """
        Polls the batch job status until it reaches a terminal state.
        """
        self.logger.info(f"Monitoring batch job {root_batch_id}...")

        while True:
            status = await self.get_status(root_batch_id, db_session)
            self.logger.info(f"Batch Status: {status}")

            if status in [
                BatchJobStatus.READY_TO_PROCESS,
                BatchJobStatus.COUNTING_READY_TO_PROCESS,
            ]:
                self.logger.info("Batch ready! Processing...")
                result = await self.process_batch(root_batch_id, db_session)

                if result.status == BatchJobStatus.COMPLETED:
                    self.logger.info("Batch workflow completed successfully.")
                    if result.hydrated_objects:
                        self.processor.result_processor.persist(
                            result.hydrated_objects, db_session
                        )
                    return result

                # Other non-terminal statuses mean we should continue polling
                elif result.status not in [
                    BatchJobStatus.COMPLETED,
                    BatchJobStatus.FAILED,
                    BatchJobStatus.CANCELLED,
                ]:
                    self.logger.info(
                        f"Intermediate step processed (new status: {result.status}). Continuing to monitor..."
                    )
                else:
                    # Processing returned a terminal status
                    self.logger.error(
                        f"Batch processing failed with terminal status: {result.status} - {result.message}"
                    )
                    return result

            elif status in [
                BatchJobStatus.COMPLETED,
                BatchJobStatus.FAILED,
                BatchJobStatus.CANCELLED,
            ]:
                self.logger.info(f"Batch job reached terminal state: {status}")
                return await self.process_batch(root_batch_id, db_session)

            # Any other status is an in-progress state, so we wait.
            self.logger.debug(f"Current status {status}, waiting for next poll.")
            await asyncio.sleep(poll_interval)
