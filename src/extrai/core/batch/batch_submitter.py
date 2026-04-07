import logging
import uuid
from datetime import UTC, datetime
from typing import Any

from sqlalchemy.orm import Session
from sqlmodel import SQLModel, select

from extrai.core.base_llm_client import BaseLLMClient
from extrai.core.batch_models import (
    BatchJobContext,
    BatchJobStatus,
    BatchJobStep,
)
from extrai.core.client_rotator import ClientRotator
from extrai.core.config.batch_job_config import BatchJobConfig
from extrai.core.entity_counter import EntityCounter
from extrai.core.extraction_config import ExtractionConfig
from extrai.core.extraction_context_preparer import ExtractionContextPreparer
from extrai.core.extraction_request_factory import ExtractionRequestFactory
from extrai.core.model_registry import ModelRegistry
from extrai.utils.serialization_utils import resolve_step_param


class BatchSubmitter:
    def __init__(
        self,
        model_registry: ModelRegistry,
        client_rotator: ClientRotator,
        config: ExtractionConfig,
        entity_counter: EntityCounter,
        context_preparer: ExtractionContextPreparer,
        request_factory: ExtractionRequestFactory,
        logger: logging.Logger,
    ):
        self.model_registry = model_registry
        self.client_rotator = client_rotator
        self.config = config
        self.entity_counter = entity_counter
        self.context_preparer = context_preparer
        self.request_factory = request_factory
        self.logger = logger

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
        config_data = BatchJobConfig(
            extraction_example_json=example_json,
            custom_extraction_process=custom_extraction_process,
            custom_extraction_guidelines=custom_extraction_guidelines,
            custom_final_checklist=custom_final_checklist,
            custom_context=custom_context,
            count_entities=count_entities,
            custom_counting_context=custom_counting_context,
            schema_json=self.model_registry.llm_schema_json,
        )

        if self.config.use_hierarchical_extraction:
            config_data = config_data.evolve(hierarchical=True, current_model_index=0)

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
        new_config_dict: dict[str, Any],
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

        new_config = BatchJobConfig(**new_config_dict)

        # Ensure new config has required fields
        if self.config.use_hierarchical_extraction and not new_config.hierarchical:
            new_config = new_config.evolve(
                hierarchical=True, current_model_index=start_from_step_index
            )

        if old_context.config.expected_entity_descriptions is not None:
            new_config = new_config.evolve(
                expected_entity_descriptions=old_context.config.expected_entity_descriptions
            )

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
                .where(BatchJobStep.status == BatchJobStatus.COMPLETED)
            ).all()

            for step in old_steps:
                effective_index = step.step_index

                if effective_index < start_from_step_index:
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

        # Determine starting phase and normalize step index
        target_step_index = start_from_step_index
        is_counting_phase = False

        if new_context.config.count_entities:
            if new_context.config.hierarchical:
                # Interleaved: Even=Count, Odd=Extract. Model = step // 2
                target_step_index = start_from_step_index // 2
                is_counting_phase = start_from_step_index % 2 == 0
            else:
                # Non-hierarchical: 0=Count, 1=Extract
                target_step_index = 0
                is_counting_phase = start_from_step_index == 0

        if is_counting_phase:
            await self._submit_counting_phase(
                new_context, db_session, step_index=target_step_index
            )
        else:
            await self._submit_extraction_phase(
                new_context, db_session, step_index=target_step_index
            )

        return new_batch_id

    async def _submit_counting_phase(
        self,
        context: BatchJobContext,
        db_session: Session,
        step_index: int | None = None,
    ):
        if context.config.hierarchical and step_index is not None:
            context.config = context.config.evolve(current_model_index=step_index)
            db_session.add(context)
            db_session.commit()

        input_strings = context.input_strings

        # Determine which models to count
        current_step_index = step_index
        if context.config.hierarchical:
            idx = (
                step_index
                if step_index is not None
                else context.config.current_model_index
            )
            current_step_index = idx
            if 0 <= idx < len(self.model_registry.models):
                model_names = [self.model_registry.models[idx].__name__]
            else:
                model_names = self.model_registry.get_all_model_names()
        else:
            model_names = self.model_registry.get_all_model_names()

        # Retrieve previous entities from completed steps
        previous_entities = []
        if (
            context.config.hierarchical
            and current_step_index is not None
            and current_step_index > 0
        ):
            step_threshold = self._get_absolute_step_index(
                current_step_index, "counting", context.config.count_entities
            )
            steps = db_session.exec(
                select(BatchJobStep)
                .where(BatchJobStep.batch_id == context.root_batch_id)
                .where(BatchJobStep.step_index < step_threshold)
                .where(BatchJobStep.status == BatchJobStatus.COMPLETED)
                .order_by(BatchJobStep.step_index)
            ).all()
            for s in steps:
                # Exclude counting phases from previous entities
                if s.metadata_json and s.metadata_json.get("phase") == "counting":
                    continue
                previous_entities.extend(s.result)

        custom_counting_context = context.config.custom_counting_context
        examples = context.config.extraction_example_json

        total_steps = (
            len(self.model_registry.models) if context.config.hierarchical else 1
        )
        resolved_context = resolve_step_param(
            custom_counting_context,
            current_step_index if current_step_index is not None else 0,
            total_steps,
        )

        system_prompt, user_prompt = self.entity_counter.prepare_counting_prompts(
            input_strings,
            model_names,
            resolved_context,
            previous_entities=previous_entities if previous_entities else None,
            examples=examples,
        )

        client = self.entity_counter.llm_client
        requests = self._create_batch_requests(
            system_prompt,
            user_prompt,
            num_revisions=self.config.num_counting_revisions,
            override_client=client,
        )

        response_model = None
        if self.config.use_structured_output:
            response_model = self.entity_counter.get_counting_model(model_names)

        batch_job = await client.create_batch_job(
            requests, response_model=response_model
        )
        if hasattr(batch_job, "id"):
            provider_batch_id = batch_job.id
        elif hasattr(batch_job, "name"):
            provider_batch_id = batch_job.name
        else:
            provider_batch_id = str(batch_job)

        context.current_batch_id = provider_batch_id
        context.status = BatchJobStatus.COUNTING_SUBMITTED
        context.updated_at = datetime.now(UTC)
        db_session.add(context)
        db_session.commit()

        self.logger.info(
            f"Submitted counting batch for the models {model_names} for {context.root_batch_id}: {provider_batch_id}"
        )

    async def _submit_extraction_phase(
        self,
        context: BatchJobContext,
        db_session: Session,
        step_index: int = 0,
        num_revisions: int | None = None,
    ):
        # Update current index in config if hierarchical
        if context.config.hierarchical:
            context.config = context.config.evolve(current_model_index=step_index)
            db_session.add(context)
            db_session.commit()

        # Retrieve previous entities from completed steps
        previous_entities = []
        if context.config.hierarchical and step_index > 0:
            step_threshold = self._get_absolute_step_index(
                step_index, "extraction", context.config.count_entities
            )
            steps = db_session.exec(
                select(BatchJobStep)
                .where(BatchJobStep.batch_id == context.root_batch_id)
                .where(BatchJobStep.step_index < step_threshold)
                .where(BatchJobStep.status == BatchJobStatus.COMPLETED)
                .order_by(BatchJobStep.step_index)
            ).all()
            for s in steps:
                # Exclude counting phases from previous entities
                if s.metadata_json and s.metadata_json.get("phase") == "counting":
                    continue
                previous_entities.extend(s.result)

        # Prepare request
        total_steps = (
            len(self.model_registry.models) if context.config.hierarchical else 1
        )

        request = self.request_factory.prepare_request(
            input_strings=context.input_strings,
            config=self.config,
            extraction_example_json=context.config.extraction_example_json,
            custom_extraction_process=resolve_step_param(
                context.config.custom_extraction_process,
                step_index,
                total_steps,
            ),
            custom_extraction_guidelines=resolve_step_param(
                context.config.custom_extraction_guidelines,
                step_index,
                total_steps,
            ),
            custom_final_checklist=resolve_step_param(
                context.config.custom_final_checklist,
                step_index,
                total_steps,
            ),
            custom_context=resolve_step_param(
                context.config.custom_context, step_index, total_steps
            ),
            expected_entity_descriptions=context.config.expected_entity_descriptions,
            previous_entities=previous_entities if previous_entities else None,
            hierarchical_model_index=step_index
            if context.config.hierarchical
            else None,
        )

        requests = self._create_batch_requests(
            request.system_prompt,
            request.user_prompt,
            request.json_schema,
            num_revisions=num_revisions,
        )

        client = self.client_rotator.get_next_client()
        batch_job = await client.create_batch_job(
            requests, response_model=request.response_model
        )
        if hasattr(batch_job, "id"):
            provider_batch_id = batch_job.id
        elif hasattr(batch_job, "name"):
            provider_batch_id = batch_job.name
        else:
            provider_batch_id = str(batch_job)

        context.current_batch_id = provider_batch_id
        context.status = BatchJobStatus.SUBMITTED
        context.updated_at = datetime.now(UTC)
        db_session.add(context)
        db_session.commit()

        phase_name = (
            f"step {step_index}" if context.config.hierarchical else "extraction"
        )
        self.logger.info(
            f"Submitted extraction batch ({phase_name}) for {context.root_batch_id}: {provider_batch_id}"
        )

    def _create_batch_requests(
        self,
        system_prompt: str,
        user_prompt: str,
        json_schema: str | None = None,
        num_revisions: int | None = None,
        override_client: BaseLLMClient | None = None,
    ) -> list[dict]:
        """Create batch requests in OpenAI batch format."""
        num_revs = num_revisions or self.config.num_llm_revisions
        client = override_client or self.client_rotator.get_next_client()

        requests = []
        for _ in range(num_revs):
            req = client.prepare_request(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                json_schema=json_schema,
            )
            # Wrap request in OpenAI batch format
            batch_req = {
                "custom_id": str(uuid.uuid4()),
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": req,
            }
            requests.append(batch_req)
        return requests
