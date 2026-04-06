# extrai/core/workflow_orchestrator.py

import logging
from typing import Any

from sqlalchemy.orm import Session
from sqlmodel import SQLModel

from extrai.core.base_llm_client import BaseLLMClient
from extrai.core.batch.batch_pipeline import BatchPipeline
from extrai.core.batch_models import BatchJobStatus, BatchProcessResult

from .analytics_collector import WorkflowAnalyticsCollector
from .extraction_config import ExtractionConfig
from .extraction_pipeline import ExtractionPipeline
from .model_registry import ModelRegistry
from .result_processor import ResultProcessor


class WorkflowOrchestrator:
    """
    Orchestrates data extraction workflows by delegating to specialized components.

    This class serves as a facade, coordinating between:
    - ModelRegistry: Schema discovery and management
    - ExtractionPipeline: Standard extraction flow
    - BatchPipeline: Batch extraction flow
    - ResultProcessor: Result hydration and persistence
    """

    def __init__(
        self,
        root_sqlmodel_class: type[SQLModel],
        llm_client: BaseLLMClient | list[BaseLLMClient],
        num_llm_revisions: int = 3,
        num_counting_revisions: int = 3,
        max_validation_retries_per_revision: int = 2,
        consensus_threshold: float = 0.51,
        counting_levenshtein_threshold: float = 0.85,
        conflict_resolver=None,
        analytics_collector: WorkflowAnalyticsCollector | None = None,
        use_hierarchical_extraction: bool = False,
        use_structured_output: bool = False,
        logger: logging.Logger | None = None,
        counting_llm_client: BaseLLMClient | None = None,
    ):
        self.logger = logger or self._create_default_logger()

        # Initialize registry first (validates root model)
        self.model_registry = ModelRegistry(root_sqlmodel_class, self.logger)

        # Create shared config
        self.config = ExtractionConfig(
            num_llm_revisions=num_llm_revisions,
            num_counting_revisions=num_counting_revisions,
            max_validation_retries_per_revision=max_validation_retries_per_revision,
            consensus_threshold=consensus_threshold,
            counting_levenshtein_threshold=counting_levenshtein_threshold,
            conflict_resolver=conflict_resolver,
            use_hierarchical_extraction=use_hierarchical_extraction,
            use_structured_output=use_structured_output,
        )

        # Initialize components
        self.analytics_collector = analytics_collector or WorkflowAnalyticsCollector(
            logger=self.logger
        )

        self.pipeline = ExtractionPipeline(
            model_registry=self.model_registry,
            llm_client=llm_client,
            config=self.config,
            analytics_collector=self.analytics_collector,
            logger=self.logger,
            counting_llm_client=counting_llm_client,
        )

        self.batch_pipeline = BatchPipeline(
            model_registry=self.model_registry,
            llm_client=llm_client,
            config=self.config,
            analytics_collector=self.analytics_collector,
            logger=self.logger,
            counting_llm_client=counting_llm_client,
        )

        self.result_processor = ResultProcessor(
            model_registry=self.model_registry,
            analytics_collector=self.analytics_collector,
            logger=self.logger,
        )

    def _create_default_logger(self) -> logging.Logger:
        logger = logging.getLogger(self.__class__.__name__)
        logger.setLevel(logging.WARNING)
        return logger

    # ==================== Standard Extraction ====================

    async def synthesize(
        self,
        input_strings: list[str],
        db_session_for_hydration: Session | None = None,
        extraction_example_json: str = "",
        extraction_example_object: SQLModel | list[SQLModel] | None = None,
        custom_extraction_process: str | list[str] = "",
        custom_extraction_guidelines: str | list[str] = "",
        custom_final_checklist: str | list[str] = "",
        custom_context: str | list[str] = "",
        count_entities: bool = False,
        custom_counting_context: str | list[str] = "",
    ) -> list[Any]:
        """Executes extraction pipeline and returns hydrated objects."""
        if not input_strings:
            raise ValueError("Input strings list cannot be empty.")

        # Extract to consensus JSON
        consensus_results = await self.pipeline.extract(
            input_strings=input_strings,
            extraction_example_json=extraction_example_json,
            extraction_example_object=extraction_example_object,
            custom_extraction_process=custom_extraction_process,
            custom_extraction_guidelines=custom_extraction_guidelines,
            custom_final_checklist=custom_final_checklist,
            custom_context=custom_context,
            count_entities=count_entities,
            custom_counting_context=custom_counting_context,
        )

        # Hydrate results
        return self.result_processor.hydrate(
            consensus_results, db_session_for_hydration
        )

    async def synthesize_and_save(
        self,
        input_strings: list[str],
        db_session: Session,
        extraction_example_json: str = "",
        extraction_example_object: SQLModel | list[SQLModel] | None = None,
        custom_extraction_process: str | list[str] = "",
        custom_extraction_guidelines: str | list[str] = "",
        custom_final_checklist: str | list[str] = "",
        custom_context: str | list[str] = "",
        count_entities: bool = False,
        custom_counting_context: str | list[str] = "",
    ) -> list[Any]:
        """Synthesizes and persists objects in a single transaction."""
        hydrated_objects = await self.synthesize(
            input_strings=input_strings,
            db_session_for_hydration=db_session,
            extraction_example_json=extraction_example_json,
            extraction_example_object=extraction_example_object,
            custom_extraction_process=custom_extraction_process,
            custom_extraction_guidelines=custom_extraction_guidelines,
            custom_final_checklist=custom_final_checklist,
            custom_context=custom_context,
            count_entities=count_entities,
            custom_counting_context=custom_counting_context,
        )

        if hydrated_objects:
            self.result_processor.persist(hydrated_objects, db_session)

        return hydrated_objects

    # ==================== Batch Extraction ====================

    async def synthesize_batch(
        self,
        input_strings: list[str],
        db_session: Session,
        extraction_example_json: str = "",
        extraction_example_object: SQLModel | list[SQLModel] | None = None,
        custom_extraction_process: str | list[str] = "",
        custom_extraction_guidelines: str | list[str] = "",
        custom_final_checklist: str | list[str] = "",
        custom_context: str | list[str] = "",
        count_entities: bool = False,
        custom_counting_context: str | list[str] = "",
        wait_for_completion: bool = False,
        poll_interval: int = 60,
    ) -> str | BatchProcessResult:
        """Submits a batch job.

        Args:
            input_strings: List of strings to process
            db_session: Database session for state persistence
            ...
            wait_for_completion: If True, waits for the batch job (and any hierarchical steps) to complete.
            poll_interval: Interval in seconds to poll for status if wait_for_completion is True.

        Returns:
            root_batch_id (str) if wait_for_completion is False.
            BatchProcessResult if wait_for_completion is True.
        """
        root_batch_id = await self.batch_pipeline.submit_batch(
            db_session=db_session,
            input_strings=input_strings,
            extraction_example_json=extraction_example_json,
            extraction_example_object=extraction_example_object,
            custom_extraction_process=custom_extraction_process,
            custom_extraction_guidelines=custom_extraction_guidelines,
            custom_final_checklist=custom_final_checklist,
            custom_context=custom_context,
            count_entities=count_entities,
            custom_counting_context=custom_counting_context,
        )

        if wait_for_completion:
            return await self.monitor_batch_job(
                root_batch_id, db_session, poll_interval
            )

        return root_batch_id

    async def create_continuation_batch(
        self,
        original_batch_id: str,
        db_session: Session,
        start_from_step_index: int,
        extraction_example_json: str = "",
        extraction_example_object: SQLModel | list[SQLModel] | None = None,
        custom_extraction_process: str | list[str] = "",
        custom_extraction_guidelines: str | list[str] = "",
        custom_final_checklist: str | list[str] = "",
        custom_context: str | list[str] = "",
        count_entities: bool = False,
        custom_counting_context: str | list[str] = "",
        wait_for_completion: bool = False,
        poll_interval: int = 60,
    ) -> str | BatchProcessResult:
        """
        Creates a new batch cycle continuing from a previous batch's state.
        Copies completed steps up to start_from_step_index into the new batch.
        Accepts all configuration parameters to update the job logic.
        """
        # Prepare example
        example_json = await self.batch_pipeline.context_preparer.prepare_example(
            extraction_example_json,
            extraction_example_object,
            self.batch_pipeline.client_rotator.get_next_client,
        )

        config_data = {
            "extraction_example_json": example_json,
            "custom_extraction_process": custom_extraction_process,
            "custom_extraction_guidelines": custom_extraction_guidelines,
            "custom_final_checklist": custom_final_checklist,
            "custom_context": custom_context,
            "count_entities": count_entities,
            "custom_counting_context": custom_counting_context,
            "schema_json": self.model_registry.llm_schema_json,
        }

        new_batch_id = await self.batch_pipeline.create_continuation_batch(
            db_session, original_batch_id, config_data, start_from_step_index
        )

        if wait_for_completion:
            return await self.monitor_batch_job(new_batch_id, db_session, poll_interval)

        return new_batch_id

    async def get_batch_status(
        self, root_batch_id: str, db_session: Session
    ) -> BatchJobStatus:
        """Retrieves current batch job status."""
        return await self.batch_pipeline.get_status(root_batch_id, db_session)

    async def process_batch(
        self,
        root_batch_id: str,
        db_session: Session,
    ) -> "BatchProcessResult":
        """Processes a completed batch job and persists results."""
        result = await self.batch_pipeline.process_batch(
            root_batch_id,
            db_session,
        )

        if result.status.name == "COMPLETED" and result.hydrated_objects and result.original_pk_map:
             self.result_processor.original_pk_map.update(result.original_pk_map)

        return result

    async def monitor_batch_job(
        self, root_batch_id: str, db_session: Session, poll_interval: int = 60
    ) -> "BatchProcessResult":
        """
        Polls the batch job status until it reaches a terminal state.
        Automatically handles hierarchical extraction steps by re-polling
        if an intermediate step is submitted.
        """
        return await self.batch_pipeline.monitor_batch_job(
            root_batch_id, db_session, poll_interval
        )

    # ==================== Analytics ====================

    def get_analytics_report(self) -> dict[str, Any]:
        """Retrieves analytics report."""
        return self.analytics_collector.get_report()

    def get_analytics_collector(self) -> WorkflowAnalyticsCollector:
        """Returns the analytics collector instance."""
        return self.analytics_collector

    def get_total_steps(self, count_entities: bool) -> int:
        """Calculates the total number of steps for a workflow."""
        num_models = len(self.model_registry.models)
        if self.config.use_hierarchical_extraction and count_entities:
            return num_models * 2
        return num_models
