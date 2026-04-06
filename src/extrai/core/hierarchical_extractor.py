import logging
from typing import Any

from extrai.core.extraction_config import ExtractionConfig
from extrai.utils.serialization_utils import (
    make_json_serializable,
    resolve_step_param,
)

from .entity_counter import EntityCounter
from .extraction_request_factory import ExtractionRequestFactory
from .llm_runner import LLMRunner
from .model_registry import ModelRegistry
from .model_wrapper_builder import ModelWrapperBuilder
from .prompt_builder import PromptBuilder
from .shared.hierarchical_coordinator import HierarchicalCoordinator


class HierarchicalExtractor:
    """
    Performs hierarchical extraction by processing models level-by-level.

    Uses breadth-first traversal to extract parent entities first,
    then uses them as context for extracting children.
    """

    def __init__(
        self,
        model_registry: ModelRegistry,
        prompt_builder: PromptBuilder,
        entity_counter: EntityCounter,
        llm_runner: LLMRunner,
        logger: logging.Logger,
        request_factory: ExtractionRequestFactory,
        model_wrapper_builder: ModelWrapperBuilder = None,
        use_structured_output: bool = False,
        config: ExtractionConfig | None = None,
    ):
        self.model_registry = model_registry
        self.prompt_builder = prompt_builder
        self.entity_counter = entity_counter
        self.llm_runner = llm_runner
        self.logger = logger
        self.request_factory = request_factory
        self.model_wrapper_builder = model_wrapper_builder
        self.use_structured_output = use_structured_output
        self.config = config
        self.coordinator = HierarchicalCoordinator(model_registry, logger)

    def _has_valid_descriptions(
        self, descriptions: list[dict] | None
    ) -> bool:
        """
        Check if descriptions list is valid (not None and contains non-empty dicts).
        
        Args:
            descriptions: List of description dicts from entity counting
            
        Returns:
            True if descriptions exist and have at least one valid dict
        """
        if descriptions is None:
            return False
        if not isinstance(descriptions, list):
            return False
        if len(descriptions) == 0:
            return False
        
        for item in descriptions:
            if isinstance(item, dict) and "description" in item:
                if item["description"] and item["description"].strip():
                    return True
        return False

    async def extract(
        self,
        input_strings: list[str],
        extraction_example_json: str,
        custom_extraction_process: str | list[str],
        custom_extraction_guidelines: str | list[str],
        custom_final_checklist: str | list[str],
        custom_context: str | list[str],
        count_entities: bool,
        custom_counting_context: str | list[str] = "",
    ) -> list[dict[str, Any]]:
        """Executes hierarchical extraction."""
        self.logger.info("Starting hierarchical extraction...")

        models = self.coordinator.get_models()
        num_models = len(models)
        total_steps = num_models * 2 if count_entities else num_models
        results_store: dict[tuple[str, str], dict[str, Any]] = {}
        current_step = 0

        for i, model_class in enumerate(models):
            model_name = model_class.__name__

            # Resolve parameters for this step using model index `i`
            counting_context_for_model = resolve_step_param(
                custom_counting_context, i, num_models
            )
            extraction_process_for_model = resolve_step_param(
                custom_extraction_process, i, num_models
            )
            guidelines_for_model = resolve_step_param(
                custom_extraction_guidelines, i, num_models
            )
            checklist_for_model = resolve_step_param(
                custom_final_checklist, i, num_models
            )
            context_for_model = resolve_step_param(custom_context, i, num_models)

            # Count entities if needed
            expected_entity_descriptions = None
            if count_entities:
                current_step += 1
                self.logger.info(
                    f"Step {current_step}/{total_steps}: Counting entities for {model_name}"
                )
                # Prepare previous entities for context
                previous_entities = None
                if results_store:
                    previous_entities = make_json_serializable(
                        self.coordinator.collect_previous_entities(
                            list(results_store.values())
                        )
                    )

                counts = await self.entity_counter.count_entities(
                    input_strings,
                    [model_name],
                    counting_context_for_model,
                    previous_entities=previous_entities,
                    examples=extraction_example_json,
                )
                
                # Filter counts just for this model
                expected_entity_descriptions = [
                    item for item in counts if item.get("model") == model_name
                ]

                # DEBUG: Log counting results to diagnose empty descriptions issue
                self.logger.debug(
                    f"DEBUG: Counting results for {model_name}: "
                    f"expected_entity_descriptions={expected_entity_descriptions}, "
                    f"type={type(expected_entity_descriptions)}, "
                    f"count={len(expected_entity_descriptions) if expected_entity_descriptions else 0}"
                )

            # Check if we should skip extraction due to empty descriptions from counting
            should_skip_extraction = (
                count_entities
                and not self._has_valid_descriptions(expected_entity_descriptions)
            )

            if should_skip_extraction:
                self.logger.info(
                    f"Step {current_step + 1}/{total_steps}: Skipping extraction for {model_name} "
                    f"- no valid entity descriptions found from counting"
                )
                # Skip the extraction step but still increment the step counter
                current_step += 1
                # Set entities to empty list to indicate nothing was extracted
                entities = []
            else:
                if not self.config:
                    raise ValueError(
                        "ExtractionConfig is required for HierarchicalExtractor"
                    )

                current_step += 1
                self.logger.info(
                    f"Step {current_step}/{total_steps}: Extracting entities for {model_name}"
                )
                request = self.request_factory.prepare_request(
                    input_strings=input_strings,
                    config=self.config,
                    extraction_example_json=extraction_example_json,
                    custom_extraction_process=extraction_process_for_model,
                    custom_extraction_guidelines=guidelines_for_model,
                    custom_final_checklist=checklist_for_model,
                    custom_context=context_for_model,
                    expected_entity_descriptions=expected_entity_descriptions,
                    previous_entities=self.coordinator.collect_previous_entities(
                        list(results_store.values())
                    )
                    if results_store
                    else None,
                    hierarchical_model_index=i,
                )

                if self.use_structured_output:
                    entities = await self.llm_runner.run_structured_extraction_cycle(
                        system_prompt=request.system_prompt,
                        user_prompt=request.user_prompt,
                        response_model=request.response_model,
                    )
                else:
                    entities = await self.llm_runner.run_extraction_cycle(
                        system_prompt=request.system_prompt, user_prompt=request.user_prompt
                    )

            # Store results
            for idx, entity in enumerate(entities):
                if "_type" not in entity:
                    entity["_type"] = model_name

                temp_id = entity.get("_temp_id")
                storage_id = temp_id if temp_id else f"__synthetic_{idx}__"

                if (model_name, storage_id) not in results_store:
                    results_store[(model_name, storage_id)] = entity

            self.logger.info(
                f"Completed {model_name}. Total entities: {len(results_store)}"
            )

        return list(results_store.values())

