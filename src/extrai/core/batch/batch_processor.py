import logging
from datetime import UTC, datetime

from sqlalchemy.orm import Session
from sqlmodel import select

from extrai.core.analytics_collector import WorkflowAnalyticsCollector
from extrai.core.batch_models import (
    BatchJobContext,
    BatchJobStatus,
    BatchJobStep,
    BatchProcessResult,
)
from extrai.core.client_rotator import ClientRotator
from extrai.core.cost_calculator import track_usage_from_response
from extrai.core.entity_counter import EntityCounter
from extrai.core.extraction_config import ExtractionConfig
from extrai.core.model_registry import ModelRegistry
from extrai.core.result_processor import ResultProcessor
from extrai.core.shared.consensus_runner import ConsensusRunner
from extrai.core.shared.hierarchical_coordinator import HierarchicalCoordinator
from extrai.utils.alignment_utils import normalize_json_revisions

from .batch_result_retriever import BatchResultRetriever
from .batch_status_checker import BatchStatusChecker
from .batch_submitter import BatchSubmitter


class BatchProcessor:
    def __init__(
        self,
        model_registry: ModelRegistry,
        config: ExtractionConfig,
        analytics_collector: WorkflowAnalyticsCollector,
        client_rotator: ClientRotator,
        entity_counter: EntityCounter,
        submitter: BatchSubmitter,
        status_checker: BatchStatusChecker,
        retriever: BatchResultRetriever,
        consensus_runner: ConsensusRunner,
        hierarchical_coordinator: HierarchicalCoordinator,
        logger: logging.Logger,
    ):
        self.model_registry = model_registry
        self.config = config
        self.analytics_collector = analytics_collector
        self.client_rotator = client_rotator
        self.entity_counter = entity_counter
        self.submitter = submitter
        self.status_checker = status_checker
        self.retriever = retriever
        self.consensus_runner = consensus_runner
        self.hierarchical_coordinator = hierarchical_coordinator
        self.logger = logger

        # Inject ResultProcessor
        self.result_processor = ResultProcessor(
            self.model_registry, self.analytics_collector, self.logger
        )

    def _get_absolute_step_index(
        self, model_index: int, phase: str, count_entities: bool
    ) -> int:
        """Calculates the absolute workflow step index based on model index and phase."""
        if not count_entities:
            return model_index

        # If counting enabled:
        # Model 0: Count (0), Extract (1)
        # Model 1: Count (2), Extract (3)
        base = model_index * 2
        return base if phase == "counting" else base + 1

    async def process_batch(
        self, root_batch_id: str, db_session: Session
    ) -> BatchProcessResult:
        status = await self.status_checker.get_status(root_batch_id, db_session)
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

            # Debug log the raw result format
            self.logger.debug(
                f"Counting results type: {type(results_content)}, "
                f"first 200 chars: {str(results_content)[:200]}"
            )

            # Determine expected models for validation
            if context.config.hierarchical:
                current_idx = context.config.current_model_index
                if 0 <= current_idx < len(self.model_registry.models):
                    target_model_names = [
                        self.model_registry.models[current_idx].__name__
                    ]
                else:
                    target_model_names = self.model_registry.get_all_model_names()
            else:
                target_model_names = self.model_registry.get_all_model_names()

            import json

            revisions = []

            # Extract raw content from results
            # Handle both string (JSONL) and list return types
            lines = []
            if isinstance(results_content, str):
                lines = [
                    l.strip() for l in results_content.strip().split("\n") if l.strip()
                ]
            elif isinstance(results_content, list):
                lines = results_content
            else:
                raise ValueError(
                    f"Unexpected results content type: {type(results_content)}"
                )

            if not lines:
                raise ValueError("Empty results content")

            # Parse each line as a revision
            for raw_content in lines:
                try:
                    if isinstance(raw_content, str):
                        wrapper = json.loads(raw_content)
                    else:
                        wrapper = raw_content

                    # Check if it's wrapped in OpenAI batch response format
                    if "response" in wrapper and "body" in wrapper.get("response", {}):
                        body = wrapper["response"]["body"]

                        if self.analytics_collector:
                            track_usage_from_response(
                                wrapper,
                                client,
                                self.analytics_collector,
                                context.current_batch_id,
                                extra_details={"phase": "counting"},
                            )

                        if "choices" in body and body["choices"]:
                            content = body["choices"][0]["message"]["content"]
                            parsed_json = json.loads(content)
                            revisions.append(parsed_json)
                    else:
                        # Maybe it's directly the JSON string or dict
                        if isinstance(wrapper, str):
                            revisions.append(json.loads(wrapper))
                        elif isinstance(wrapper, dict):
                            revisions.append(wrapper)

                except json.JSONDecodeError as e:
                    self.logger.error(f"Failed to parse counting result as JSON: {e}")
                    continue

            self.logger.debug(f"Parsed {len(revisions)} counting revisions")

            # Recreate original prompts to use for consensus fallback if needed
            from extrai.core.prompts.counting import (
                generate_entity_counting_system_prompt,
                generate_entity_counting_user_prompt,
            )

            schema_json = self.model_registry.get_schema_for_models(target_model_names)
            system_prompt = generate_entity_counting_system_prompt(
                target_model_names,
                schema_json,
                context.config.custom_counting_context,
            )
            user_prompt = generate_entity_counting_user_prompt(context.input_strings)
            target_json_schema = (
                self.entity_counter.get_counting_model(
                    target_model_names
                ).model_json_schema()
                if self.config.use_structured_output
                else None
            )

            # Achieve consensus
            consensus_result = (
                await self.entity_counter.counting_consensus.achieve_consensus(
                    revisions=revisions,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    target_json_schema=target_json_schema,
                )
            )

            # Filter out any hallucinated models not in target_model_names
            entity_descriptions = [
                item
                for item in consensus_result
                if item.get("model") in target_model_names
            ]

            self.logger.debug(f"Extracted entity descriptions: {entity_descriptions}")

            context.config = context.config.evolve(
                expected_entity_descriptions=entity_descriptions
            )

            # Create step for counting completion
            step_index_abs = self._get_absolute_step_index(
                context.config.current_model_index,
                "counting",
                context.config.count_entities,
            )

            step = BatchJobStep(
                batch_id=context.root_batch_id,
                step_index=step_index_abs,
                status=BatchJobStatus.COMPLETED,
                result=entity_descriptions,
                metadata_json={"phase": "counting"},
            )
            db_session.add(step)
            db_session.add(context)
            db_session.commit()

            # Submit next phase
            await self.submitter._submit_extraction_phase(
                context, db_session, step_index=context.config.current_model_index
            )

            return BatchProcessResult(
                status=BatchJobStatus.SUBMITTED,
                message="Counting complete, extraction submitted.",
            )
        except Exception as e:
            self.logger.error(
                f"Error processing counting completion: {e}", exc_info=True
            )
            context.status = BatchJobStatus.FAILED
            db_session.add(context)
            db_session.commit()
            return BatchProcessResult(
                status=BatchJobStatus.FAILED, message=f"Processing failed: {e}"
            )

    async def _process_extraction_completion(
        self, context: BatchJobContext, db_session: Session
    ) -> BatchProcessResult:
        try:
            client = self.client_rotator.get_next_client()
            (
                results,
                validation_errors,
            ) = await self.retriever.retrieve_and_validate_results(context, client)

            if validation_errors:
                return await self._handle_batch_retry(
                    context, db_session, results, validation_errors
                )

            # Re-run consensus with partial results
            all_revisions = [r["revisions"] for r in results]
            all_revisions.extend(context.config.partial_results)
            all_revisions = normalize_json_revisions(all_revisions)

            processed_results = self.consensus_runner.run(all_revisions)

            return await self._process_hierarchical_step(
                context, db_session, processed_results
            )

        except Exception as e:
            self.logger.error(
                f"Error processing batch completion for {context.root_batch_id}: {e}",
                exc_info=True,
            )
            context.status = BatchJobStatus.FAILED
            db_session.add(context)
            db_session.commit()
            return BatchProcessResult(
                status=BatchJobStatus.FAILED, message=f"Processing failed: {e}"
            )

    async def _process_hierarchical_step(
        self,
        context: BatchJobContext,
        db_session: Session,
        processed_results: list[dict],
    ):
        # Save step results
        step_index_abs = self._get_absolute_step_index(
            context.config.current_model_index,
            "extraction",
            context.config.count_entities,
        )

        step = BatchJobStep(
            batch_id=context.root_batch_id,
            step_index=step_index_abs,
            status=BatchJobStatus.COMPLETED,
            result=processed_results,
        )
        db_session.add(step)
        db_session.commit()

        # Check for completion using HierarchicalCoordinator
        is_final = False
        if not context.config.hierarchical:
            is_final = True
        else:
            is_final = self.hierarchical_coordinator.is_final_step(
                context.config.current_model_index
            )

        if is_final:
            return await self._finalize_completion(context, db_session)
        else:
            # Submit next step using HierarchicalCoordinator
            next_step_index = self.hierarchical_coordinator.next_index(
                context.config.current_model_index
            )

            if context.config.count_entities:
                await self.submitter._submit_counting_phase(
                    context, db_session, step_index=next_step_index
                )
                return BatchProcessResult(
                    status=BatchJobStatus.COUNTING_SUBMITTED,
                    message=f"Step {context.config.current_model_index} complete, counting for step {next_step_index} submitted.",
                )
            else:
                await self.submitter._submit_extraction_phase(
                    context, db_session, step_index=next_step_index
                )
                return BatchProcessResult(
                    status=BatchJobStatus.SUBMITTED,
                    message=f"Step {context.config.current_model_index} complete, extraction for step {next_step_index} submitted.",
                )

    async def _handle_batch_retry(
        self,
        context: BatchJobContext,
        db_session: Session,
        validated_results: list[dict],
        validation_errors: list[dict],
    ) -> BatchProcessResult:
        if not validated_results:
            context.status = BatchJobStatus.FAILED
            db_session.add(context)
            db_session.commit()
            return BatchProcessResult(
                status=BatchJobStatus.FAILED,
                message="All revisions failed validation, cannot retry.",
                errors=validation_errors,
            )

        # Store valid partial results
        partial_results = [r["revisions"] for r in validated_results]
        context.config = context.config.evolve(
            partial_results=context.config.partial_results + partial_results
        )
        db_session.add(context)
        db_session.commit()

        # Resubmit with fewer revisions
        num_to_retry = len(validation_errors)
        await self.submitter._submit_extraction_phase(
            context,
            db_session,
            step_index=context.config.current_model_index,
            num_revisions=num_to_retry,
        )

        return BatchProcessResult(
            status=BatchJobStatus.SUBMITTED,
            message=f"Partial success. Retrying {num_to_retry} failed revisions.",
            errors=validation_errors,
        )

    async def _finalize_completion(
        self, context: BatchJobContext, db_session: Session
    ) -> BatchProcessResult:
        # Load all step results
        steps = db_session.exec(
            select(BatchJobStep)
            .where(BatchJobStep.batch_id == context.root_batch_id)
            .where(BatchJobStep.status == BatchJobStatus.COMPLETED)
            .order_by(BatchJobStep.step_index)
        ).all()
        final_results = [
            item
            for s in steps
            if not s.metadata_json or s.metadata_json.get("phase") != "counting"
            for item in s.result
        ]

        # Process final results using the injected result_processor
        processed_objects = self.result_processor.hydrate(final_results, db_session)

        # Persist objects (this links FKs and commits)
        self.result_processor.persist(processed_objects, db_session)

        context.results = [p.model_dump(mode="json") for p in processed_objects]
        context.status = BatchJobStatus.COMPLETED
        context.updated_at = datetime.now(UTC)
        db_session.add(context)
        db_session.commit()

        return BatchProcessResult(
            status=BatchJobStatus.COMPLETED,
            message="Batch processing complete.",
            hydrated_objects=processed_objects,
            original_pk_map=self.result_processor.original_pk_map,
        )
