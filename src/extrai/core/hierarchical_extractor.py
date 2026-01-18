import logging
from typing import List, Dict, Any, Tuple, Optional

from .model_registry import ModelRegistry
from .prompt_builder import PromptBuilder
from .entity_counter import EntityCounter
from .llm_runner import LLMRunner
from .model_wrapper_builder import ModelWrapperBuilder
from .extraction_request_factory import ExtractionRequestFactory
from extrai.core.extraction_config import ExtractionConfig
from extrai.utils.serialization_utils import make_json_serializable


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
        config: Optional[ExtractionConfig] = None,
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

    async def extract(
        self,
        input_strings: List[str],
        extraction_example_json: str,
        custom_extraction_process: str,
        custom_extraction_guidelines: str,
        custom_final_checklist: str,
        custom_context: str,
        count_entities: bool,
        custom_counting_context: str = "",
    ) -> List[Dict[str, Any]]:
        """Executes hierarchical extraction."""
        self.logger.info("Starting hierarchical extraction...")

        models = self.model_registry.models
        results_store: Dict[Tuple[str, str], Dict[str, Any]] = {}

        for i, model_class in enumerate(models):
            model_name = model_class.__name__
            self.logger.info(f"Processing model: {model_name}")

            # Count entities if needed
            expected_entity_descriptions = None
            if count_entities:
                # Prepare previous entities for context
                previous_entities = None
                if results_store:
                    previous_entities = make_json_serializable(
                        list(results_store.values())
                    )

                counts = await self.entity_counter.count_entities(
                    input_strings,
                    [model_name],
                    custom_counting_context,
                    previous_entities=previous_entities,
                )
                expected_entity_descriptions = counts.get(model_name)

            if not self.config:
                raise ValueError(
                    "ExtractionConfig is required for HierarchicalExtractor"
                )

            request = self.request_factory.prepare_request(
                input_strings=input_strings,
                config=self.config,
                extraction_example_json=extraction_example_json,
                custom_extraction_process=custom_extraction_process,
                custom_extraction_guidelines=custom_extraction_guidelines,
                custom_final_checklist=custom_final_checklist,
                custom_context=custom_context,
                expected_entity_descriptions=expected_entity_descriptions,
                previous_entities=list(results_store.values())
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
