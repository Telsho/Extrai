# extrai/core/workflow_orchestrator.py

import json
import logging
import asyncio
from typing import (
    List,
    Dict,
    Any,
    Type,
    Callable,
    Optional,
    Tuple,
    Union,
)

# SQLAlchemy imports
from sqlalchemy.orm import Session
from sqlalchemy import create_engine

from sqlmodel import SQLModel

# Project imports
from .prompt_builder import generate_system_prompt, generate_user_prompt_for_docs
from .json_consensus import JSONConsensus, default_conflict_resolver
from .sqlalchemy_hydrator import SQLAlchemyHydrator
from .db_writer import persist_objects, DatabaseWriterError
from .base_llm_client import BaseLLMClient
from .schema_inspector import (
    generate_llm_schema_from_models,
    discover_sqlmodels_from_root,
)
from .errors import (
    WorkflowError,
    LLMInteractionError,
    ConfigurationError,
    ConsensusProcessError,
    HydrationError,
    LLMConfigurationError,
    LLMOutputParseError,
    LLMOutputValidationError,
    LLMAPICallError,
)
from .analytics_collector import WorkflowAnalyticsCollector
from .example_json_generator import ExampleJSONGenerator, ExampleGenerationError


class WorkflowOrchestrator:
    """
    Orchestrates the data extraction workflow, handling both standard and hierarchical extraction.

    This class manages the entire process from receiving unstructured text to outputting
    structured SQLModel objects. It integrates various components like LLM clients,
    a JSON consensus mechanism, and a SQLAlchemy hydrator.

    For hierarchical data, it uses a breadth-first traversal approach, extracting entities
    level by level and using parent entities as context for extracting children. This logic
    is now fully integrated within this class, removing the need for a separate
    HierarchicalExtractor.
    """

    def __init__(
        self,
        root_sqlmodel_class: Type[SQLModel],
        llm_client: Union[BaseLLMClient, List[BaseLLMClient]],
        num_llm_revisions: int = 3,
        max_validation_retries_per_revision: int = 2,
        consensus_threshold: float = 0.51,
        conflict_resolver: Callable[
            [Tuple[int | str, ...], List[str | int | float | bool | None]],
            Optional[str | int | float | bool | None],
        ] = default_conflict_resolver,
        analytics_collector: Optional[WorkflowAnalyticsCollector] = None,
        use_hierarchical_extraction: bool = False,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initializes the WorkflowOrchestrator.

        Args:
            root_sqlmodel_class: The primary SQLModel class for extraction.
            llm_client: An instance or a list of LLM clients.
            num_llm_revisions: The number of JSON revisions to request from the LLM for consensus.
            max_validation_retries_per_revision: Max retries for LLM output validation per revision.
            consensus_threshold: The agreement threshold for the consensus mechanism (0.0 to 1.0).
            conflict_resolver: A function to resolve disagreements during the consensus process.
            analytics_collector: An optional collector for workflow analytics.
            use_hierarchical_extraction: If True, enables the hierarchical extraction workflow
                for models with nested relationships.
            logger: An optional logger instance. If not provided, a default logger is created.
        """
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        if not logger:
            self.logger.setLevel(logging.WARNING)

        self._validate_init_parameters(
            root_sqlmodel_class,
            num_llm_revisions,
            max_validation_retries_per_revision,
            consensus_threshold,
        )
        self._setup_llm_clients(llm_client)
        self._discover_models_and_generate_schema(root_sqlmodel_class)

        self.llm_client_index = 0
        self.llm_client = self.llm_clients[0]
        self.num_llm_revisions = num_llm_revisions
        self.max_validation_retries_per_revision = max_validation_retries_per_revision
        self.root_sqlmodel_class = root_sqlmodel_class
        self.use_hierarchical_extraction = use_hierarchical_extraction

        if self.use_hierarchical_extraction:
            self.logger.warning(
                "Hierarchical extraction is enabled. "
                "This may significantly increase LLM API calls and processing time "
                "based on model complexity and the number of entities."
            )

        self.json_consensus = JSONConsensus(
            consensus_threshold=consensus_threshold,
            conflict_resolver=conflict_resolver,
            logger=self.logger,
        )

        if analytics_collector is None:
            self.analytics_collector = WorkflowAnalyticsCollector(logger=self.logger)
        else:
            self.analytics_collector = analytics_collector

    def _validate_init_parameters(
        self,
        root_sqlmodel_class: Type[SQLModel],
        num_llm_revisions: int,
        max_validation_retries_per_revision: int,
        consensus_threshold: float,
    ):
        """Validates the initial parameters for the orchestrator."""
        if not root_sqlmodel_class or not issubclass(root_sqlmodel_class, SQLModel):
            raise ConfigurationError(
                "root_sqlmodel_class must be a valid SQLModel class."
            )
        if num_llm_revisions < 1:
            raise ConfigurationError("Number of LLM revisions must be at least 1.")
        if max_validation_retries_per_revision < 1:
            raise ConfigurationError(
                "Max validation retries per revision must be at least 1."
            )
        if not (0.0 <= consensus_threshold <= 1.0):
            raise ConfigurationError(
                "Extrai threshold must be between 0.0 and 1.0 inclusive."
            )

    def _setup_llm_clients(self, llm_client: Union[BaseLLMClient, List[BaseLLMClient]]):
        """Sets up the LLM clients list."""
        if isinstance(llm_client, list):
            if not all(isinstance(c, BaseLLMClient) for c in llm_client):
                raise ConfigurationError(
                    "All items in llm_client list must be instances of BaseLLMClient."
                )
            if not llm_client:
                raise ConfigurationError("llm_client list cannot be empty.")
            self.llm_clients = llm_client
        elif isinstance(llm_client, BaseLLMClient):
            self.llm_clients = [llm_client]
        else:
            raise ConfigurationError(
                "llm_client must be an instance of BaseLLMClient or a list of them."
            )
        for client in self.llm_clients:
            client.logger = self.logger

    def _discover_models_and_generate_schema(self, root_sqlmodel_class: Type[SQLModel]):
        """Discovers SQLModels and generates the JSON schema for the LLM."""
        try:
            self.sqla_model_classes = discover_sqlmodels_from_root(root_sqlmodel_class)
        except Exception as e:
            raise ConfigurationError(f"Failed to discover SQLModel classes: {e}") from e

        if not self.sqla_model_classes:
            raise ConfigurationError(
                "No SQLModel classes were discovered from the root model."
            )

        self.model_schema_map_for_hydration = {
            model_cls.__name__: model_cls for model_cls in self.sqla_model_classes
        }

        try:
            generated_prompt_schema_str = generate_llm_schema_from_models(
                initial_model_classes=self.sqla_model_classes
            )
            if not generated_prompt_schema_str:
                raise ConfigurationError(
                    "Generated target_json_schema_for_llm (prompt schema) is empty."
                )
            json.loads(generated_prompt_schema_str)
            self.target_json_schema_for_llm = generated_prompt_schema_str
        except json.JSONDecodeError as e:
            raise ConfigurationError(
                f"The internally generated LLM prompt JSON schema is not valid: {e}."
            )
        except Exception as e:
            raise ConfigurationError(
                f"Failed to generate the LLM prompt JSON schema: {e}"
            ) from e

    def _get_next_llm_client(self) -> BaseLLMClient:
        """Rotates through the list of LLM clients and returns the next one."""
        client = self.llm_clients[self.llm_client_index]
        self.llm_client_index = (self.llm_client_index + 1) % len(self.llm_clients)
        return client

    async def _prepare_extraction_example(self, extraction_example_json: str) -> str:
        """Prepares the extraction example, auto-generating it if necessary."""
        if extraction_example_json:
            return extraction_example_json

        try:
            llm_client_for_example = self._get_next_llm_client()
            example_generator = ExampleJSONGenerator(
                llm_client=llm_client_for_example,
                output_model=self.root_sqlmodel_class,
                analytics_collector=self.analytics_collector,
                max_validation_retries_per_revision=self.max_validation_retries_per_revision,
                logger=self.logger,
            )
            self.logger.info(
                f"Attempting to auto-generate extraction example for {self.root_sqlmodel_class.__name__}..."
            )
            generated_example = await example_generator.generate_example()
            if self.analytics_collector:
                self.analytics_collector.record_custom_event(
                    "example_json_auto_generation_success"
                )
            self.logger.info("Successfully auto-generated extraction example.")
            return generated_example
        except ExampleGenerationError as e:
            if self.analytics_collector:
                self.analytics_collector.record_custom_event(
                    "example_json_auto_generation_failure"
                )
            raise WorkflowError(
                f"Failed to auto-generate extraction example: {e}"
            ) from e
        except Exception as e:
            if self.analytics_collector:
                self.analytics_collector.record_custom_event(
                    "example_json_auto_generation_unexpected_failure"
                )
            raise WorkflowError(
                f"An unexpected error occurred during auto-generation of extraction example: {e}"
            ) from e

    async def synthesize(
        self,
        input_strings: List[str],
        db_session_for_hydration: Optional[Session],
        extraction_example_json: str = "",
        custom_extraction_process: str = "",
        custom_extraction_guidelines: str = "",
        custom_final_checklist: str = "",
    ) -> List[Any]:
        """
        Executes the full pipeline: input strings -> LLM -> consensus -> SQLAlchemy objects.

        Args:
            input_strings: A list of input strings for data extraction.
            db_session_for_hydration: SQLAlchemy session for the hydrator.
            extraction_example_json: Optional JSON string for few-shot prompting.
            custom_extraction_process: Optional custom instructions for LLM extraction process.
            custom_extraction_guidelines: Optional custom guidelines for LLM extraction.
            custom_final_checklist: Optional custom final checklist for LLM.

        Returns:
            A list of hydrated SQLAlchemy object instances.

        Raises:
            ValueError: If input_strings is empty.
            LLMInteractionError, ConsensusProcessError, HydrationError, WorkflowError: For pipeline failures.
        """
        if not input_strings:
            raise ValueError("Input strings list cannot be empty.")

        self.logger.info(
            f"Starting synthesis for {self.root_sqlmodel_class.__name__}..."
        )
        current_extraction_example_json = await self._prepare_extraction_example(
            extraction_example_json
        )

        if self.use_hierarchical_extraction:
            final_list_for_hydration = await self._execute_hierarchical_extraction(
                input_strings=input_strings,
                current_extraction_example_json=current_extraction_example_json,
                custom_extraction_process=custom_extraction_process,
                custom_extraction_guidelines=custom_extraction_guidelines,
                custom_final_checklist=custom_final_checklist,
            )
        else:
            final_list_for_hydration = await self._execute_standard_extraction(
                input_strings=input_strings,
                current_extraction_example_json=current_extraction_example_json,
                custom_extraction_process=custom_extraction_process,
                custom_extraction_guidelines=custom_extraction_guidelines,
                custom_final_checklist=custom_final_checklist,
            )

        return self._hydrate_results(final_list_for_hydration, db_session_for_hydration)

    def _process_consensus_output(
        self, consensus_output: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]]
    ) -> List[Dict[str, Any]]:
        """Processes the raw output from the consensus mechanism."""
        if consensus_output is None:
            return []
        if isinstance(consensus_output, list):
            return consensus_output
        if isinstance(consensus_output, dict):
            if "results" in consensus_output and isinstance(
                consensus_output["results"], list
            ):
                return consensus_output["results"]
            return [consensus_output]

        raise ConsensusProcessError(
            f"Unexpected type from json_consensus.get_consensus: {type(consensus_output)}."
        )

    def _hydrate_results(
        self,
        final_list_for_hydration: List[Dict[str, Any]],
        db_session_for_hydration: Optional[Session],
    ) -> List[Any]:
        """Hydrates the final list of dictionaries into SQLModel objects."""
        session_to_use = db_session_for_hydration
        if session_to_use is None:
            # Create a temporary, in-memory SQLite database if no session is provided.
            engine = create_engine("sqlite:///:memory:")
            SQLModel.metadata.create_all(engine)
            session_to_use = Session(engine)

        hydrator = SQLAlchemyHydrator(session=session_to_use)
        try:
            self.logger.info(
                f"Starting hydration for {len(final_list_for_hydration)} consensus objects."
            )
            hydrated_objects = hydrator.hydrate(
                final_list_for_hydration, self.model_schema_map_for_hydration
            )
            self.analytics_collector.record_hydration_success(len(hydrated_objects))
            self.logger.info(
                f"Successfully hydrated {len(hydrated_objects)} SQLModel objects."
            )
            if db_session_for_hydration is None:
                session_to_use.close()
            return hydrated_objects
        except Exception as e:
            self.analytics_collector.record_hydration_failure()
            if db_session_for_hydration is None and session_to_use:
                session_to_use.close()
            raise HydrationError(
                f"Failed during SQLAlchemy object hydration: {e}"
            ) from e

    def _generate_contextual_prompt_for_hierarchical(
        self,
        model_name: str,
        results_store: Dict[Tuple[str, str], Dict[str, Any]],
    ) -> str:
        """Generates the contextual prompt for a hierarchical extraction step."""
        custom_context = f"Your current task is to extract **only** entities of type '{model_name}'. Do not extract any other types of entities in this step."

        if results_store:
            custom_context += "\n\nSo far, the following entities have been extracted. Use them as context to establish relationships:\n"
            custom_context += json.dumps(list(results_store.values()), indent=2)

        return custom_context

    async def _execute_hierarchical_extraction(
        self,
        input_strings: List[str],
        current_extraction_example_json: str,
        custom_extraction_process: str,
        custom_extraction_guidelines: str,
        custom_final_checklist: str,
    ) -> List[Dict[str, Any]]:
        """Executes the hierarchical extraction process, processing each model type in order."""
        self.logger.info("Executing hierarchical extraction process...")
        models_to_process = discover_sqlmodels_from_root(self.root_sqlmodel_class)
        results_store: Dict[Tuple[str, str], Dict[str, Any]] = {}

        for model_class in models_to_process:
            model_name = model_class.__name__
            self.logger.info(f"Hierarchical step: Processing model '{model_name}'...")

            schema_json = generate_llm_schema_from_models([model_class])
            custom_context = self._generate_contextual_prompt_for_hierarchical(
                model_name, results_store
            )

            system_prompt = generate_system_prompt(
                schema_json=schema_json,
                extraction_example_json=current_extraction_example_json,
                custom_extraction_process=custom_extraction_process,
                custom_extraction_guidelines=custom_extraction_guidelines,
                custom_final_checklist=custom_final_checklist,
                custom_context=custom_context,
            )
            user_prompt = generate_user_prompt_for_docs(
                documents=input_strings, custom_context=custom_context
            )

            extracted_entities = await self._run_single_extraction_cycle(
                system_prompt, user_prompt
            )

            for entity in extracted_entities:
                temp_id = entity.get("_temp_id")
                if not temp_id:
                    continue
                result_key = (model_name, temp_id)
                if result_key not in results_store:
                    results_store[result_key] = entity
            self.logger.info(
                f"Hierarchical step for '{model_name}' completed. "
                f"Total entities in store: {len(results_store)}"
            )

        self.logger.info("Hierarchical extraction finished.")
        return list(results_store.values())

    async def _run_single_extraction_cycle(
        self, system_prompt: str, user_prompt: str
    ) -> List[Dict[str, Any]]:
        """Runs a single extraction cycle, including LLM revisions and consensus."""
        tasks = []
        try:
            for _ in range(self.num_llm_revisions):
                client_for_revision = self._get_next_llm_client()
                task = asyncio.create_task(
                    client_for_revision.generate_json_revisions(
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        num_revisions=1,
                        model_schema_map=self.model_schema_map_for_hydration,
                        max_validation_retries_per_revision=self.max_validation_retries_per_revision,
                        analytics_collector=self.analytics_collector,
                    )
                )
                tasks.append(task)

            llm_json_revisions = await asyncio.gather(*tasks)

            self.logger.debug(
                f"llm_json_revisions before consensus: {llm_json_revisions}"
            )

            if not llm_json_revisions and self.num_llm_revisions > 0:
                raise LLMInteractionError(
                    "LLM client returned no revisions despite being requested."
                )
        except (
            LLMConfigurationError,
            LLMOutputParseError,
            LLMOutputValidationError,
            LLMAPICallError,
        ) as client_err:
            raise LLMInteractionError(
                f"LLM client operation failed: {client_err}"
            ) from client_err
        except Exception as e:
            raise LLMInteractionError(
                f"An unexpected error occurred during LLM interaction: {e}"
            ) from e

        try:
            consensus_output, consensus_analytics_details = (
                self.json_consensus.get_consensus(llm_json_revisions)
            )
            self.logger.debug(
                f"consensus_output from get_consensus: {consensus_output}"
            )
            if consensus_analytics_details:
                self.analytics_collector.record_consensus_run_details(
                    consensus_analytics_details
                )

            processed_output = self._process_consensus_output(consensus_output)
            self.logger.debug(f"processed_output before hydration: {processed_output}")
            return processed_output
        except ConsensusProcessError:
            raise
        except Exception as e:
            raise ConsensusProcessError(
                f"Failed during JSON consensus processing: {e}"
            ) from e

    async def _execute_standard_extraction(
        self,
        input_strings: List[str],
        current_extraction_example_json: str,
        custom_extraction_process: str,
        custom_extraction_guidelines: str,
        custom_final_checklist: str,
    ) -> List[Dict[str, Any]]:
        """Executes the standard extraction process."""
        self.logger.info("Executing standard extraction...")
        system_prompt = generate_system_prompt(
            schema_json=self.target_json_schema_for_llm,
            extraction_example_json=current_extraction_example_json,
            custom_extraction_process=custom_extraction_process,
            custom_extraction_guidelines=custom_extraction_guidelines,
            custom_final_checklist=custom_final_checklist,
        )
        user_prompt = generate_user_prompt_for_docs(documents=input_strings)

        logging.info(f"System Prompt: {system_prompt}")
        logging.info(f"User Prompt: {user_prompt}")

        return await self._run_single_extraction_cycle(system_prompt, user_prompt)

    async def synthesize_and_save(
        self,
        input_strings: List[str],
        db_session: Session,
        extraction_example_json: str = "",
        custom_extraction_process: str = "",
        custom_extraction_guidelines: str = "",
        custom_final_checklist: str = "",
    ) -> List[Any]:
        """
        Synthesizes SQLAlchemy objects and persists them to the database.
        This method manages the transaction via the provided db_session.
        """
        hydrated_objects = await self.synthesize(
            input_strings=input_strings,
            db_session_for_hydration=db_session,
            extraction_example_json=extraction_example_json,
            custom_extraction_process=custom_extraction_process,
            custom_extraction_guidelines=custom_extraction_guidelines,
            custom_final_checklist=custom_final_checklist,
        )

        if hydrated_objects:
            try:
                persist_objects(
                    db_session=db_session,
                    objects_to_persist=hydrated_objects,
                    logger=self.logger,
                )
            except DatabaseWriterError:
                db_session.rollback()
                raise
            except Exception as e:
                db_session.rollback()
                raise WorkflowError(
                    f"An unexpected error occurred during database persistence phase: {e}"
                ) from e
        else:
            self.logger.info(
                "WorkflowOrchestrator: No objects were hydrated, thus nothing to persist."
            )

        return hydrated_objects

    def get_analytics_report(self) -> Dict[str, Any]:
        """
        Retrieves the analytics report from the associated collector.
        """
        return self.analytics_collector.get_report()

    def get_analytics_collector(self) -> WorkflowAnalyticsCollector:
        """
        Returns the instance of the analytics collector.
        """
        return self.analytics_collector
