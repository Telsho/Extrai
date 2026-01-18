import logging
from typing import List, Dict, Any, Tuple, Optional, NamedTuple

from extrai.core.model_registry import ModelRegistry
from extrai.core.prompt_builder import PromptBuilder
from extrai.core.model_wrapper_builder import ModelWrapperBuilder
from extrai.core.extraction_config import ExtractionConfig
from extrai.utils.serialization_utils import make_json_serializable


class ExtractionRequest(NamedTuple):
    system_prompt: str
    user_prompt: str
    json_schema: Optional[Dict[str, Any]]
    model_name: Optional[str]
    response_model: Optional[Any] = None


class ExtractionRequestFactory:
    """
    Factory to prepare extraction requests (prompts and schemas).
    Centralizes logic for Standard, Structured, and Hierarchical extraction preparation.
    """

    def __init__(
        self,
        model_registry: ModelRegistry,
        prompt_builder: PromptBuilder,
        model_wrapper_builder: ModelWrapperBuilder,
        logger: Optional[logging.Logger] = None,
    ):
        self.model_registry = model_registry
        self.prompt_builder = prompt_builder
        self.model_wrapper_builder = model_wrapper_builder
        self.logger = logger or logging.getLogger(__name__)

    def prepare_request(
        self,
        input_strings: List[str],
        config: ExtractionConfig,
        extraction_example_json: str = "",
        custom_extraction_process: str = "",
        custom_extraction_guidelines: str = "",
        custom_final_checklist: str = "",
        custom_context: str = "",
        expected_entity_descriptions: Optional[List[str]] = None,
        previous_entities: Optional[List[Dict[str, Any]]] = None,
        hierarchical_model_index: Optional[int] = None,
    ) -> ExtractionRequest:
        """
        Prepares the extraction request based on the configuration and current state.

        Args:
            input_strings: List of document strings.
            config: Extraction configuration.
            extraction_example_json: Example JSON for few-shot.
            custom_extraction_process: Custom instructions.
            custom_extraction_guidelines: Custom guidelines.
            custom_final_checklist: Custom checklist (standard mode only).
            custom_context: Additional context string.
            expected_entity_descriptions: List of descriptions (from counting).
            previous_entities: List of previously extracted entities (for hierarchical).
            hierarchical_model_index: Index of the model to extract (hierarchical only).

        Returns:
            ExtractionRequest containing prompts, schema, and target model name.
        """
        # 1. Determine Target Model
        if config.use_hierarchical_extraction:
            if hierarchical_model_index is None:
                hierarchical_model_index = 0

            if not (0 <= hierarchical_model_index < len(self.model_registry.models)):
                raise ValueError(
                    f"Invalid hierarchical_model_index: {hierarchical_model_index}"
                )

            target_model = self.model_registry.models[hierarchical_model_index]
            target_model_name = target_model.__name__
        else:
            target_model = self.model_registry.root_model
            target_model_name = None

        # 2. Serialize Previous Entities (Context)
        serializable_previous_entities = None
        if previous_entities:
            serializable_previous_entities = make_json_serializable(previous_entities)

        # 3. Generate Request
        json_schema = None
        wrapper_model = None

        if config.use_structured_output:
            # If hierarchical, we only want the shallow model for this step (include_relationships=False)
            # If standard (not hierarchical), we want deep extraction (include_relationships=True)
            include_relationships = not config.use_hierarchical_extraction

            wrapper_model = self.model_wrapper_builder.generate_wrapper_model(
                target_model, include_relationships=include_relationships
            )

            system_prompt, user_prompt = self.prompt_builder.build_structured_prompts(
                input_strings=input_strings,
                custom_extraction_process=custom_extraction_process,
                custom_extraction_guidelines=custom_extraction_guidelines,
                custom_context=custom_context,
                extraction_example_json=extraction_example_json,
                expected_entity_descriptions=expected_entity_descriptions,
                previous_entities=serializable_previous_entities,
                target_model_name=target_model_name
                if config.use_hierarchical_extraction
                else None,
            )
            json_schema = wrapper_model.model_json_schema()

        else:
            if config.use_hierarchical_extraction:
                schema_json = self.model_registry.get_schema_for_models(
                    [target_model_name]
                )
            else:
                schema_json = self.model_registry.llm_schema_json

            system_prompt, user_prompt = self.prompt_builder.build_prompts(
                input_strings=input_strings,
                schema_json=schema_json,
                extraction_example_json=extraction_example_json,
                custom_extraction_process=custom_extraction_process,
                custom_extraction_guidelines=custom_extraction_guidelines,
                custom_final_checklist=custom_final_checklist,
                custom_context=custom_context,
                expected_entity_descriptions=expected_entity_descriptions,
                previous_entities=serializable_previous_entities,
                target_model_name=target_model_name
                if config.use_hierarchical_extraction
                else None,
            )

        return ExtractionRequest(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            json_schema=json_schema,
            model_name=target_model_name,
            response_model=wrapper_model,
        )
