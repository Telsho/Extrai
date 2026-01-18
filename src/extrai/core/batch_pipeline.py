import json
import uuid
import logging
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Union, Tuple
from sqlalchemy.orm import Session
from sqlmodel import SQLModel, select

from extrai.core.base_llm_client import BaseLLMClient
from .client_rotator import ClientRotator
from .extraction_context_preparer import ExtractionContextPreparer
from .model_registry import ModelRegistry
from .extraction_config import ExtractionConfig
from .prompt_builder import PromptBuilder
from .entity_counter import EntityCounter
from .analytics_collector import WorkflowAnalyticsCollector
from .batch_models import (
    BatchJobContext,
    BatchJobStatus,
    BatchProcessResult,
    BatchJobStep,
)
from .errors import WorkflowError
from .model_wrapper_builder import ModelWrapperBuilder
from extrai.utils.llm_output_processing import process_and_validate_llm_output
from extrai.utils.alignment_utils import normalize_json_revisions
from .json_consensus import JSONConsensus
from .extraction_request_factory import ExtractionRequestFactory


class BatchPipeline:
    """Manages batch extraction workflows."""

    def __init__(
        self,
        model_registry: ModelRegistry,
        llm_client: Union["BaseLLMClient", List["BaseLLMClient"]],
        config: ExtractionConfig,
        analytics_collector: WorkflowAnalyticsCollector,
        logger: logging.Logger,
        counting_llm_client: Optional[BaseLLMClient] = None,
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
        self.consensus = JSONConsensus(
            consensus_threshold=config.consensus_threshold,
            conflict_resolver=config.conflict_resolver,
            logger=logger,
        )
        self.request_factory = ExtractionRequestFactory(
            model_registry,
            self.prompt_builder,
            self.model_wrapper_builder,
            logger=logger,
        )

    async def submit_batch(
        self,
        db_session: Session,
        input_strings: List[str],
        extraction_example_json: str = "",
        extraction_example_object: Optional[Union[SQLModel, List[SQLModel]]] = None,
        custom_extraction_process: str = "",
        custom_extraction_guidelines: str = "",
        custom_final_checklist: str = "",
        custom_context: str = "",
        count_entities: bool = False,
        custom_counting_context: str = "",
    ) -> str:
        """Submits a batch job and returns root_batch_id."""
        if not input_strings:
            raise ValueError("input_strings cannot be empty")

        # Prepare example
        example_json = await self.context_preparer.prepare_example(
            extraction_example_json,
            extraction_example_object,
            self.client_rotator.get_next_client,
        )

        root_batch_id = str(uuid.uuid4())

        # Initialize configuration
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

        if self.config.use_hierarchical_extraction:
            config_data.update({"hierarchical": True, "current_model_index": 0})

        context = BatchJobContext(
            root_batch_id=root_batch_id,
            current_batch_id="pending",
            status=BatchJobStatus.SUBMITTED,
            input_strings=input_strings,
            config=config_data,
        )
        db_session.add(context)
        db_session.commit()

        if count_entities:
            await self._submit_counting_phase(context, db_session)
        else:
            await self._submit_extraction_phase(context, db_session, step_index=0)

        self.logger.info(f"Batch workflow initiated: {root_batch_id}")
        return root_batch_id

    async def create_continuation_batch(
        self,
        db_session: Session,
        original_batch_id: str,
        new_config: Dict[str, Any],
        start_from_step_index: int,
    ) -> str:
        """
        Creates a new batch cycle continuing from a previous batch's state.
        Copies completed steps up to start_from_step_index into the new batch.
        """
        old_context = db_session.get(BatchJobContext, original_batch_id)
        if not old_context:
            raise ValueError("Old batch not found")

        new_batch_id = str(uuid.uuid4())

        # Ensure new config has required fields
        if self.config.use_hierarchical_extraction and "hierarchical" not in new_config:
            new_config["hierarchical"] = True
            new_config["current_model_index"] = start_from_step_index

        if "expected_entity_descriptions" in old_context.config:
            new_config["expected_entity_descriptions"] = old_context.config[
                "expected_entity_descriptions"
            ]

        new_context = BatchJobContext(
            root_batch_id=new_batch_id,
            current_batch_id="pending",
            status=BatchJobStatus.SUBMITTED,
            input_strings=old_context.input_strings,
            config=new_config,
        )
        db_session.add(new_context)
        db_session.commit()

        # Copy valid steps from old batch
        if start_from_step_index > 0:
            old_steps = db_session.exec(
                select(BatchJobStep)
                .where(BatchJobStep.batch_id == original_batch_id)
                .where(BatchJobStep.step_index < start_from_step_index)
                .where(BatchJobStep.status == BatchJobStatus.COMPLETED)
            ).all()

            for step in old_steps:
                new_step = BatchJobStep(
                    batch_id=new_batch_id,
                    step_index=step.step_index,
                    status=step.status,
                    result=step.result,
                    metadata_json=step.metadata_json,
                )
                db_session.add(new_step)

            db_session.commit()

        self.logger.info(
            f"Created continuation batch {new_batch_id} from {original_batch_id}, starting at step {start_from_step_index}"
        )

        # Determine starting phase
        # If counting is enabled, we start with counting phase for the starting step
        if new_config.get("count_entities"):
            step_idx = start_from_step_index if new_config.get("hierarchical") else 0
            await self._submit_counting_phase(
                new_context, db_session, step_index=step_idx
            )
        elif new_config.get("hierarchical"):
            await self._submit_extraction_phase(
                new_context, db_session, step_index=start_from_step_index
            )
        else:
            await self._submit_extraction_phase(new_context, db_session, step_index=0)

        return new_batch_id

    async def _submit_counting_phase(
        self,
        context: BatchJobContext,
        db_session: Session,
        step_index: Optional[int] = None,
    ):
        input_strings = context.input_strings

        # Determine which models to count
        if context.config.get("hierarchical"):
            idx = (
                step_index
                if step_index is not None
                else context.config.get("current_model_index", 0)
            )
            if 0 <= idx < len(self.model_registry.models):
                model_names = [self.model_registry.models[idx].__name__]
            else:
                model_names = self.model_registry.get_all_model_names()
        else:
            model_names = self.model_registry.get_all_model_names()

        custom_counting_context = context.config.get("custom_counting_context", "")

        system_prompt, user_prompt = self.entity_counter.prepare_counting_prompts(
            input_strings, model_names, custom_counting_context
        )

        client = self.entity_counter.llm_client
        requests = self._create_batch_requests(
            system_prompt, user_prompt, num_revisions=1, override_client=client
        )

        batch_job = await client.create_batch_job(requests)
        provider_batch_id = batch_job.id if hasattr(batch_job, "id") else str(batch_job)

        context.current_batch_id = provider_batch_id
        context.status = BatchJobStatus.COUNTING_SUBMITTED
        context.updated_at = datetime.now(timezone.utc)
        db_session.add(context)
        db_session.commit()

        self.logger.info(
            f"Submitted counting batch for {context.root_batch_id}: {provider_batch_id}"
        )

    async def _submit_extraction_phase(
        self, context: BatchJobContext, db_session: Session, step_index: int = 0
    ):
        # Update current index in config if hierarchical
        if context.config.get("hierarchical"):
            # Update the config dictionary properly
            new_config = context.config.copy()
            new_config["current_model_index"] = step_index
            context.config = new_config
            db_session.add(context)
            db_session.commit()

        # Retrieve previous entities from completed steps
        previous_entities = []
        if context.config.get("hierarchical") and step_index > 0:
            steps = db_session.exec(
                select(BatchJobStep)
                .where(BatchJobStep.batch_id == context.root_batch_id)
                .where(BatchJobStep.step_index < step_index)
                .where(BatchJobStep.status == BatchJobStatus.COMPLETED)
                .order_by(BatchJobStep.step_index)
            ).all()
            for s in steps:
                previous_entities.extend(s.result)

        # Prepare request
        request = self.request_factory.prepare_request(
            input_strings=context.input_strings,
            config=self.config,
            extraction_example_json=context.config.get("extraction_example_json", ""),
            custom_extraction_process=context.config.get(
                "custom_extraction_process", ""
            ),
            custom_extraction_guidelines=context.config.get(
                "custom_extraction_guidelines", ""
            ),
            custom_final_checklist=context.config.get("custom_final_checklist", ""),
            custom_context=context.config.get("custom_context", ""),
            expected_entity_descriptions=context.config.get(
                "expected_entity_descriptions"
            ),
            previous_entities=previous_entities if previous_entities else None,
            hierarchical_model_index=step_index
            if context.config.get("hierarchical")
            else None,
        )

        requests = self._create_batch_requests(
            request.system_prompt, request.user_prompt, request.json_schema
        )

        client = self.client_rotator.get_next_client()
        batch_job = await client.create_batch_job(requests)
        provider_batch_id = batch_job.id if hasattr(batch_job, "id") else str(batch_job)

        context.current_batch_id = provider_batch_id
        context.status = BatchJobStatus.SUBMITTED
        context.updated_at = datetime.now(timezone.utc)
        db_session.add(context)
        db_session.commit()

        phase_name = (
            f"step {step_index}" if context.config.get("hierarchical") else "extraction"
        )
        self.logger.info(
            f"Submitted extraction batch ({phase_name}) for {context.root_batch_id}: {provider_batch_id}"
        )

    async def get_status(
        self, root_batch_id: str, db_session: Session
    ) -> BatchJobStatus:
        context = db_session.get(BatchJobContext, root_batch_id)
        if not context:
            raise ValueError(f"Batch job {root_batch_id} not found")

        terminal_states = [
            BatchJobStatus.COMPLETED,
            BatchJobStatus.FAILED,
            BatchJobStatus.CANCELLED,
            BatchJobStatus.READY_TO_PROCESS,
            BatchJobStatus.COUNTING_READY_TO_PROCESS,
        ]
        if context.status in terminal_states:
            return context.status

        try:
            # Determine client based on phase
            if context.status in [
                BatchJobStatus.COUNTING_SUBMITTED,
                BatchJobStatus.COUNTING_PROCESSING,
            ]:
                client = self.entity_counter.llm_client
            else:
                client = self.client_rotator.get_next_client()

            batch_job = await client.retrieve_batch_job(context.current_batch_id)
            new_provider_status = self._map_provider_status(batch_job.status)

            new_status = context.status
            # Map provider status based on current internal phase
            if context.status in [
                BatchJobStatus.COUNTING_SUBMITTED,
                BatchJobStatus.COUNTING_PROCESSING,
            ]:
                if new_provider_status == BatchJobStatus.READY_TO_PROCESS:
                    new_status = BatchJobStatus.COUNTING_READY_TO_PROCESS
                elif new_provider_status == BatchJobStatus.FAILED:
                    new_status = BatchJobStatus.FAILED
                elif new_provider_status == BatchJobStatus.CANCELLED:
                    new_status = BatchJobStatus.CANCELLED
                elif new_provider_status == BatchJobStatus.PROCESSING:
                    new_status = BatchJobStatus.COUNTING_PROCESSING
            else:
                new_status = new_provider_status

            if new_status != context.status:
                context.status = new_status
                context.updated_at = datetime.now(timezone.utc)
                db_session.add(context)
                db_session.commit()

        except Exception as e:
            self.logger.error(f"Failed to check batch status: {e}", exc_info=True)

        return context.status

    async def process_batch(
        self, root_batch_id: str, db_session: Session
    ) -> BatchProcessResult:
        status = await self.get_status(root_batch_id, db_session)
        context = db_session.get(BatchJobContext, root_batch_id)

        # 1. Already Completed
        if status == BatchJobStatus.COMPLETED and context.results:
            return await self._finalize_completion(context, db_session)

        # 2. Counting Phase Completed -> Submit Extraction
        if status == BatchJobStatus.COUNTING_READY_TO_PROCESS:
            return await self._process_counting_completion(context, db_session)

        # 3. Extraction Phase Ready -> Process Results
        if status == BatchJobStatus.READY_TO_PROCESS:
            return await self._process_extraction_completion(context, db_session)

        return BatchProcessResult(
            status=status, message="Batch not ready for processing"
        )

    async def _process_counting_completion(
        self, context: BatchJobContext, db_session: Session
    ) -> BatchProcessResult:
        try:
            # Use counting client
            client = self.entity_counter.llm_client
            results_content = await client.retrieve_batch_results(
                context.current_batch_id
            )

            # Determine expected models for validation
            if context.config.get("hierarchical"):
                current_idx = context.config.get("current_model_index", 0)
                if 0 <= current_idx < len(self.model_registry.models):
                    target_model_names = [
                        self.model_registry.models[current_idx].__name__
                    ]
                else:
                    target_model_names = self.model_registry.get_all_model_names()
            else:
                target_model_names = self.model_registry.get_all_model_names()

            # Parse descriptions
            descriptions = []

            for line in results_content.strip().split("\n"):
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                    content = client.extract_content_from_batch_response(item)
                    if content:
                        raw_json = json.loads(content)
                        if isinstance(raw_json, list) and raw_json:
                            raw_json = raw_json[0]
                        if isinstance(raw_json, dict):
                            validated_counts = self.entity_counter.validate_counts(
                                raw_json, target_model_names
                            )
                            for model_name, descs in validated_counts.items():
                                for desc in descs:
                                    descriptions.append(f"[{model_name}] {desc}")
                except Exception as e:
                    self.logger.warning(f"Failed to parse counting result: {e}")

            # Update config with descriptions
            new_config = context.config.copy()
            new_config["expected_entity_descriptions"] = descriptions
            context.config = new_config

            # Proceed to Extraction
            next_step = (
                context.config.get("current_model_index", 0)
                if context.config.get("hierarchical")
                else 0
            )
            await self._submit_extraction_phase(
                context, db_session, step_index=next_step
            )

            return BatchProcessResult(
                status=BatchJobStatus.PROCESSING,
                message="Transitioned from counting to extraction",
                retry_batch_id=context.root_batch_id,
            )

        except Exception as e:
            self.logger.error(f"Failed to process counting results: {e}", exc_info=True)
            return BatchProcessResult(
                status=BatchJobStatus.FAILED, message=f"Counting failed: {e}"
            )

    async def _process_extraction_completion(
        self, context: BatchJobContext, db_session: Session
    ) -> BatchProcessResult:
        try:
            results = await self._retrieve_and_validate_results(context)

            if results:
                consensus_output, details = self.consensus.get_consensus(results)
                if details:
                    self.analytics_collector.record_consensus_run_details(details)

                processed = self._process_consensus_output(consensus_output)

                if context.config.get("hierarchical"):
                    return await self._process_hierarchical_step(
                        context, processed, db_session
                    )

                # Finalize non-hierarchical
                context.results = processed
                context.status = BatchJobStatus.COMPLETED
                context.updated_at = datetime.now(timezone.utc)
                db_session.add(context)
                db_session.commit()

                return await self._finalize_completion(context, db_session)

            # If no valid results, maybe retry?
            return await self._handle_batch_retry(
                context, context.root_batch_id, db_session
            )

        except Exception as e:
            self.logger.error(f"Batch processing failed: {e}", exc_info=True)
            return BatchProcessResult(status=BatchJobStatus.FAILED, message=str(e))

    async def _process_hierarchical_step(
        self,
        context: BatchJobContext,
        processed_results: List[Dict],
        db_session: Session,
    ) -> BatchProcessResult:
        current_index = context.config.get("current_model_index", 0)

        # Save step result to DB
        step = BatchJobStep(
            batch_id=context.root_batch_id,
            step_index=current_index,
            status=BatchJobStatus.COMPLETED,
            result=processed_results,
            metadata_json={"timestamp": datetime.now(timezone.utc).isoformat()},
        )
        db_session.add(step)

        # Advance index
        next_index = current_index + 1

        # Update config
        new_config = context.config.copy()
        new_config["current_model_index"] = next_index
        context.config = new_config
        db_session.add(context)
        db_session.commit()

        if next_index >= len(self.model_registry.models):
            # All steps done - aggregate results for final hydration
            all_steps = db_session.exec(
                select(BatchJobStep)
                .where(BatchJobStep.batch_id == context.root_batch_id)
                .order_by(BatchJobStep.step_index)
            ).all()

            final_results = []
            for s in all_steps:
                final_results.extend(s.result)

            context.results = final_results
            context.status = BatchJobStatus.COMPLETED
            context.updated_at = datetime.now(timezone.utc)
            db_session.add(context)
            db_session.commit()
            return await self._finalize_completion(context, db_session)

        # Submit next step (counting or extraction)
        model_name = self.model_registry.models[next_index].__name__

        if context.config.get("count_entities"):
            await self._submit_counting_phase(
                context, db_session, step_index=next_index
            )
            return BatchProcessResult(
                status=BatchJobStatus.COUNTING_PROCESSING,
                message=f"Submitted counting step for {model_name}",
                retry_batch_id=context.root_batch_id,
            )
        else:
            await self._submit_extraction_phase(
                context, db_session, step_index=next_index
            )
            return BatchProcessResult(
                status=BatchJobStatus.PROCESSING,
                message=f"Submitted hierarchical step for {model_name}",
                retry_batch_id=context.root_batch_id,
            )

    async def _finalize_completion(
        self, context: BatchJobContext, db_session: Session
    ) -> BatchProcessResult:
        from .result_processor import ResultProcessor

        processor = ResultProcessor(
            self.model_registry, self.analytics_collector, self.logger
        )

        # Determine default model type for hydration
        default_model_type = None
        if self.config.use_structured_output:
            default_model_type = self.model_registry.root_model.__name__

        hydrated = processor.hydrate(
            context.results, db_session, default_model_type=default_model_type
        )
        return BatchProcessResult(
            status=BatchJobStatus.COMPLETED,
            hydrated_objects=hydrated,
            original_pk_map=processor.original_pk_map,
        )

    async def _retrieve_and_validate_results(
        self, context: BatchJobContext
    ) -> List[List[Dict]]:
        client = self.client_rotator.get_next_client()
        results_content = await client.retrieve_batch_results(context.current_batch_id)

        # Determine default model type clearly
        default_model_type = None
        if self.config.use_structured_output:
            if context.config.get("hierarchical"):
                current_idx = context.config.get("current_model_index", 0)
                if 0 <= current_idx < len(self.model_registry.models):
                    default_model_type = self.model_registry.models[
                        current_idx
                    ].__name__
            else:
                default_model_type = self.model_registry.root_model.__name__

        valid_revisions = []
        for line in results_content.strip().split("\n"):
            if not line.strip():
                continue

            try:
                item = json.loads(line)
                content = client.extract_content_from_batch_response(item)

                if content:
                    validated = process_and_validate_llm_output(
                        raw_llm_content=content,
                        model_schema_map=self.model_registry.model_map,
                        revision_info_for_error="batch_revision",
                        analytics_collector=self.analytics_collector,
                        default_model_type=default_model_type,
                    )
                    if validated:
                        valid_revisions.append(validated)
            except Exception as e:
                self.logger.warning(f"Failed to validate batch revision: {e}")

        return normalize_json_revisions(valid_revisions) if valid_revisions else []

    async def _handle_batch_retry(
        self, context: BatchJobContext, root_batch_id: str, db_session: Session
    ):
        max_retries = self.config.max_validation_retries_per_revision

        if context.retry_count < max_retries:
            context.retry_count += 1
            self.logger.info(
                f"Retrying batch {root_batch_id} ({context.retry_count}/{max_retries})"
            )

            # Resubmit current step
            if "counting" in context.status.value:
                await self._submit_counting_phase(context, db_session)
            else:
                current_idx = context.config.get("current_model_index", 0)
                await self._submit_extraction_phase(
                    context, db_session, step_index=current_idx
                )

            return BatchProcessResult(
                status=BatchJobStatus.PROCESSING,
                message="Retry submitted",
                retry_batch_id=root_batch_id,
            )

        context.status = BatchJobStatus.FAILED
        context.last_error = "Max retries exceeded"
        context.updated_at = datetime.now(timezone.utc)
        db_session.add(context)
        db_session.commit()

        return BatchProcessResult(
            status=BatchJobStatus.FAILED, message="Max retries exceeded"
        )

    def _create_batch_requests(
        self,
        system_prompt: str,
        user_prompt: str,
        json_schema: Optional[Dict] = None,
        num_revisions: Optional[int] = None,
        override_client: Optional[BaseLLMClient] = None,
    ) -> List[Dict]:
        requests = []
        client = override_client or self.client_rotator.current_client
        revisions = (
            num_revisions
            if num_revisions is not None
            else self.config.num_llm_revisions
        )

        if self.config.use_structured_output and json_schema:
            self.logger.info("Using structured output for batch requests")

        for i in range(revisions):
            body = {
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": client.temperature,
            }
            if hasattr(client, "model_name"):
                body["model"] = client.model_name

            if self.config.use_structured_output and json_schema:
                body["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "extraction_response",
                        "schema": json_schema,
                        "strict": True,
                    },
                }
            elif self.model_registry.llm_schema_json:
                body["response_format"] = {"type": "json_object"}

            requests.append({"custom_id": f"rev-{i}", "body": body})
        return requests

    def _map_provider_status(self, provider_status) -> BatchJobStatus:
        status_str = str(provider_status).lower()

        if "complete" in status_str or "succeeded" in status_str:
            return BatchJobStatus.READY_TO_PROCESS
        elif "fail" in status_str:
            return BatchJobStatus.FAILED
        elif "cancel" in status_str:
            return BatchJobStatus.CANCELLED
        elif (
            "process" in status_str or "active" in status_str or "running" in status_str
        ):
            return BatchJobStatus.PROCESSING

        return BatchJobStatus.SUBMITTED

    def _process_consensus_output(self, output) -> List[Dict[str, Any]]:
        if output is None:
            return []
        if isinstance(output, list):
            return output
        if isinstance(output, dict):
            if "results" in output and isinstance(output["results"], list):
                return output["results"]
            return [output]
        return []
