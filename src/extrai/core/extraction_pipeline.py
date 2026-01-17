import logging
from typing import List, Dict, Any, Optional, Union
from sqlmodel import SQLModel

from extrai.core.base_llm_client import BaseLLMClient
from .client_rotator import ClientRotator
from .extraction_context_preparer import ExtractionContextPreparer
from .model_registry import ModelRegistry
from .extraction_config import ExtractionConfig
from .prompt_builder import PromptBuilder
from .entity_counter import EntityCounter
from .llm_runner import LLMRunner
from .hierarchical_extractor import HierarchicalExtractor
from .analytics_collector import WorkflowAnalyticsCollector
from .model_wrapper_builder import ModelWrapperBuilder
from .extraction_request_factory import ExtractionRequestFactory


class ExtractionPipeline:
    """
    Manages the standard extraction pipeline.
    
    Flow:
    1. Prepare extraction example (auto-generate if needed)
    2. Count entities (optional)
    3. Run extraction (standard or hierarchical)
    4. Return consensus results
    
    This class coordinates between multiple components to execute
    the complete extraction workflow from input strings to consensus JSON.
    """
    
    def __init__(
        self,
        model_registry: ModelRegistry,
        llm_client: Union['BaseLLMClient', List['BaseLLMClient']],
        config: ExtractionConfig,
        analytics_collector: WorkflowAnalyticsCollector,
        logger: logging.Logger,
        counting_llm_client: Optional[BaseLLMClient] = None,
    ):
        """
        Initialize the extraction pipeline.
        
        Args:
            model_registry: Registry of SQLModel schemas
            llm_client: Single client or list of LLM clients for rotation
            config: Extraction configuration
            analytics_collector: Collector for tracking metrics
            logger: Logger instance
            counting_llm_client: Optional specific client for counting tasks
        """
        self.model_registry = model_registry
        self.config = config
        self.analytics_collector = analytics_collector
        self.logger = logger
        
        # Initialize sub-components
        self.client_rotator = ClientRotator(llm_client)
        self.prompt_builder = PromptBuilder(model_registry, logger)
        self.entity_counter = EntityCounter(
            model_registry, 
            counting_llm_client or llm_client, 
            config, 
            analytics_collector, 
            logger
        )
        self.context_preparer = ExtractionContextPreparer(
            model_registry,
            analytics_collector,
            config.max_validation_retries_per_revision,
            logger
        )
        self.llm_runner = LLMRunner(
            model_registry, llm_client, config, analytics_collector, logger
        )
        self.model_wrapper_builder = ModelWrapperBuilder()
        
        self.request_factory = ExtractionRequestFactory(
            model_registry,
            self.prompt_builder,
            self.model_wrapper_builder,
            logger
        )
        
        # Initialize hierarchical extractor if needed
        self.hierarchical_extractor = None
        if config.use_hierarchical_extraction:
            self.hierarchical_extractor = HierarchicalExtractor(
                model_registry=model_registry,
                prompt_builder=self.prompt_builder,
                entity_counter=self.entity_counter,
                llm_runner=self.llm_runner,
                logger=logger,
                request_factory=self.request_factory,
                model_wrapper_builder=self.model_wrapper_builder,
                use_structured_output=config.use_structured_output,
                config=config 
            )
            logger.warning(
                "Hierarchical extraction enabled. "
                "This may significantly increase LLM API calls and processing time "
                "based on model complexity and the number of entities."
            )
    
    async def extract(
        self,
        input_strings: List[str],
        extraction_example_json: str = "",
        extraction_example_object: Optional[Union[SQLModel, List[SQLModel]]] = None,
        custom_extraction_process: str = "",
        custom_extraction_guidelines: str = "",
        custom_final_checklist: str = "",
        custom_context: str = "",
        count_entities: bool = False,
        custom_counting_context: str = "",
    ) -> List[Dict[str, Any]]:
        """
        Executes extraction and returns consensus JSON.
        
        Args:
            input_strings: List of document strings to extract from
            extraction_example_json: Optional JSON string for few-shot prompting
            extraction_example_object: Optional SQLModel object(s) to use as example
            custom_extraction_process: Optional custom extraction process instructions
            custom_extraction_guidelines: Optional custom extraction guidelines
            custom_final_checklist: Optional custom final checklist
            custom_context: Optional custom contextual information
            count_entities: If True, performs entity counting before extraction
            
        Returns:
            List of dictionaries representing extracted entities (consensus output)
            
        Raises:
            WorkflowError: If extraction fails
        """
        self.logger.info(
            f"Starting extraction for {self.model_registry.root_model.__name__}..."
        )
        
        # Step 1: Prepare example
        example_json = await self.context_preparer.prepare_example(
            extraction_example_json, 
            extraction_example_object,
            self.client_rotator.get_next_client
        )
        
        # Step 2: Count entities if requested
        # Note: For hierarchical extraction, counting is handled per-model within the extractor
        expected_entity_descriptions = None
        if count_entities and not self.config.use_hierarchical_extraction:
            expected_entity_descriptions = await self._count_entities(
                input_strings, custom_counting_context
            )
            if expected_entity_descriptions is not None:
                self.logger.info(f"Entity count: {len(expected_entity_descriptions)}")
        
        # Step 3: Run extraction (hierarchical or standard)
        if self.config.use_hierarchical_extraction:
            self.logger.info("Using hierarchical extraction mode")
            # We assume self.hierarchical_extractor is initialized if config says so
            if not self.hierarchical_extractor:
                 # Should have been init in __init__, but safeguard
                 self.hierarchical_extractor = HierarchicalExtractor(
                    self.model_registry, self.prompt_builder, self.entity_counter,
                    self.llm_runner, self.logger, self.request_factory,
                    self.model_wrapper_builder, self.config.use_structured_output,
                    self.config
                 )

            results = await self.hierarchical_extractor.extract(
                input_strings=input_strings,
                extraction_example_json=example_json,
                custom_extraction_process=custom_extraction_process,
                custom_extraction_guidelines=custom_extraction_guidelines,
                custom_final_checklist=custom_final_checklist,
                custom_context=custom_context,
                count_entities=count_entities,
                custom_counting_context=custom_counting_context,
            )
        else:
            # Unified Non-Hierarchical Flow
            self.logger.info(
                f"Using {'structured' if self.config.use_structured_output else 'standard'} extraction mode"
            )
            
            request = self.request_factory.prepare_request(
                input_strings=input_strings,
                config=self.config,
                extraction_example_json=example_json,
                custom_extraction_process=custom_extraction_process,
                custom_extraction_guidelines=custom_extraction_guidelines,
                custom_final_checklist=custom_final_checklist,
                custom_context=custom_context,
                expected_entity_descriptions=expected_entity_descriptions
            )
            
            self.logger.debug(f"System prompt length: {len(request.system_prompt)} chars")
            self.logger.debug(f"User prompt length: {len(request.user_prompt)} chars")

            if request.response_model:
                results = await self.llm_runner.run_structured_extraction_cycle(
                    system_prompt=request.system_prompt,
                    user_prompt=request.user_prompt,
                    response_model=request.response_model
                )
            else:
                results = await self.llm_runner.run_extraction_cycle(
                    system_prompt=request.system_prompt, 
                    user_prompt=request.user_prompt
                )
        
        self.logger.info(f"Extraction completed. Found {len(results)} entities.")
        return results

    async def _count_entities(
        self, input_strings: List[str], custom_counting_context: str = ""
    ) -> Optional[List[str]]:
        """
        Counts entities in the input documents.
        
        Args:
            input_strings: Documents to analyze
            custom_counting_context: Custom context for counting phase
            
        Returns:
            List of descriptions of all model entities, or None if counting fails
        """
        all_model_names = self.model_registry.get_all_model_names()
        
        try:
            counts = await self.entity_counter.count_entities(
                input_strings, all_model_names, custom_counting_context
            )
            flat_descriptions = []
            for model_name, descriptions in counts.items():
                for desc in descriptions:
                    flat_descriptions.append(f"[{model_name}] {desc}")
            return flat_descriptions
        except Exception as e:
            self.logger.warning(f"Entity counting failed: {e}")
            return None
    
    def __repr__(self) -> str:
        """String representation of the pipeline."""
        mode = "hierarchical" if self.config.use_hierarchical_extraction else "standard"
        return (
            f"ExtractionPipeline(mode={mode}, "
            f"root={self.model_registry.root_model.__name__})"
        )
