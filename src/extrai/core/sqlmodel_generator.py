import logging
from typing import Any, Dict, Type, List as TypingList, Optional, Generator
import tempfile
import importlib.util
import sys
import os
import uuid
import json
from contextlib import contextmanager

from pydantic import ValidationError
from sqlmodel import SQLModel

from extrai.core.errors import (
    SQLModelCodeGeneratorError,
    SQLModelInstantiationValidationError,
    LLMInteractionError,
    ConfigurationError,
    LLMConfigurationError,
    LLMOutputParseError,
    LLMOutputValidationError,
    LLMAPICallError,
)
from extrai.core.base_llm_client import BaseLLMClient
from extrai.core.analytics_collector import (
    WorkflowAnalyticsCollector,
)
from extrai.core.prompt_builder import (
    generate_sqlmodel_creation_system_prompt,
    generate_user_prompt_for_docs,
)
from extrai.core.code_generation.python_builder import PythonModelBuilder


class SQLModelCodeGenerator:
    """
    Generates Python code for SQLModel classes.
    It can either take a direct model description or interact with an LLM
    to obtain a model description based on input documents and a task.
    The generated code is then dynamically loaded.
    """

    _sqlmodel_description_schema_cache: Optional[Dict[str, Any]] = None
    # Adjusted path to be relative to this file (sqlmodel_generator.py)
    _SCHEMA_FILE_PATH = os.path.join(
        os.path.dirname(__file__), "schemas", "sqlmodel_description_schema.json"
    )

    def __init__(
        self,
        llm_client: BaseLLMClient,
        analytics_collector: Optional[WorkflowAnalyticsCollector] = None,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initializes the SQLModelCodeGenerator.

        Args:
            llm_client: An instance of a LLM client.
            analytics_collector: Optional collector for workflow analytics.
            logger: An optional logger instance. If not provided, a default logger is created.
        """
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        if not logger:
            self.logger.setLevel(logging.WARNING)

        self.llm_client = llm_client
        self.llm_client.logger = self.logger
        if analytics_collector is None:
            self.analytics_collector = WorkflowAnalyticsCollector(logger=self.logger)
        else:
            self.analytics_collector = analytics_collector

    def _load_sqlmodel_description_schema(self) -> Dict[str, Any]:
        """
        Loads the SQLModel description JSON schema from file.
        Caches the schema after the first load.
        """
        if SQLModelCodeGenerator._sqlmodel_description_schema_cache is None:
            try:
                # Ensure the path is correct if this file moves relative to the schema
                schema_file_path = SQLModelCodeGenerator._SCHEMA_FILE_PATH
                if not os.path.exists(schema_file_path):
                    # Attempt to construct path relative to current file's directory
                    current_dir = os.path.dirname(os.path.abspath(__file__))
                    schema_file_path = os.path.join(
                        current_dir, "schemas", "sqlmodel_description_schema.json"
                    )

                with open(schema_file_path, "r") as f:
                    schema = json.load(f)
                SQLModelCodeGenerator._sqlmodel_description_schema_cache = schema
            except FileNotFoundError:
                raise ConfigurationError(
                    f"SQLModel description schema not found at {schema_file_path}"
                )
            except json.JSONDecodeError:
                raise ConfigurationError(
                    f"Invalid JSON in SQLModel description schema at {schema_file_path}"
                )
        return SQLModelCodeGenerator._sqlmodel_description_schema_cache

    def _generate_code_from_description(self, llm_json_output: Dict[str, Any]) -> str:
        """
        Delegates the code generation to the PythonModelBuilder.
        """
        model_descriptions = llm_json_output.get("sql_models", [])
        builder = PythonModelBuilder()
        return builder.generate_model_code(model_descriptions)

    @contextmanager
    def _managed_temp_module(self, code: str) -> Generator[str, None, None]:
        """A context manager for creating and cleaning up a temporary Python module file."""
        temp_dir = tempfile.mkdtemp()
        module_name = f"dynamic_sqlmodel_module_{uuid.uuid4().hex}"
        temp_file_path = os.path.join(temp_dir, f"{module_name}.py")
        try:
            with open(temp_file_path, "w", encoding="utf-8") as f:
                f.write(code)
            yield temp_file_path
        finally:
            # Cleanup logic
            module_name_from_path = os.path.splitext(os.path.basename(temp_file_path))[
                0
            ]
            if module_name_from_path in sys.modules:
                del sys.modules[module_name_from_path]
            if os.path.exists(temp_file_path):
                try:
                    os.remove(temp_file_path)
                except OSError as e:
                    self.logger.warning(
                        f"Could not remove temporary file {temp_file_path}: {e}"
                    )
            if os.path.exists(temp_dir):
                try:
                    os.rmdir(temp_dir)
                except OSError as e:
                    self.logger.warning(
                        f"Could not remove temporary directory {temp_dir}: {e}"
                    )

    def _import_module_from_path(self, module_name: str, path: str) -> Any:
        """Imports a module from a given file path."""
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise SQLModelCodeGeneratorError(
                f"Failed to create import spec for dynamically generated module: {module_name}"
            )

        dynamic_module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = dynamic_module
        spec.loader.exec_module(dynamic_module)
        return dynamic_module

    def _extract_models_from_module(
        self,
        module: Any,
        model_names: TypingList[str],
        generated_code: str,
        module_name: str,
    ) -> Dict[str, Type[SQLModel]]:
        """Extracts and validates SQLModel classes from a loaded module."""
        loaded_classes_map = {}
        for name_to_load in model_names:
            cls = getattr(module, name_to_load, None)
            if cls is None:
                raise SQLModelCodeGeneratorError(
                    f"Class '{name_to_load}' not found in dynamically loaded module '{module_name}'.\nGenerated code:\n{generated_code}"
                )
            if not (isinstance(cls, type) and issubclass(cls, SQLModel)):
                raise SQLModelCodeGeneratorError(
                    f"Loaded attribute '{name_to_load}' is not a class or not a subclass of SQLModel."
                )
            loaded_classes_map[name_to_load] = cls
        return loaded_classes_map

    def _rebuild_and_validate_models(
        self, loaded_classes: Dict[str, Type[SQLModel]], generated_code: str
    ):
        """Calls model_rebuild and validates instantiation for all loaded models."""
        for cls_to_rebuild in loaded_classes.values():
            if hasattr(cls_to_rebuild, "model_rebuild"):
                cls_to_rebuild.model_rebuild(force=True)

        for name, cls in loaded_classes.items():
            try:
                _ = cls.model_validate({})
            except ValidationError as ve:
                raise SQLModelInstantiationValidationError(
                    model_name=name,
                    validation_error=ve,
                    generated_code=generated_code,
                ) from ve
            except Exception as inst_e:
                raise SQLModelCodeGeneratorError(
                    f"Dynamically loaded class '{name}' failed instantiation with an unexpected error: {inst_e}.\nGenerated code for module:\n{generated_code}"
                ) from inst_e

    def _generate_and_load_class_from_description(
        self, model_description: Dict[str, Any]
    ) -> tuple[Dict[str, Type[SQLModel]], str]:
        """
        Generates SQLModel Python code from a given description, dynamically loads it,
        and returns the generated SQLModel classes and the generated code.
        """
        generated_code_str = ""
        try:
            generated_code_str = self._generate_code_from_description(model_description)
            self.logger.debug(f"Dynamically generated code:\n{generated_code_str}")

            models_to_load_names = [
                m["model_name"]
                for m in model_description.get("sql_models", [])
                if "model_name" in m
            ]

            if not models_to_load_names:
                raise SQLModelCodeGeneratorError(
                    "No models found in the 'sql_models' list from the LLM description."
                )

            with self._managed_temp_module(generated_code_str) as temp_file_path:
                module_name = os.path.splitext(os.path.basename(temp_file_path))[0]
                dynamic_module = self._import_module_from_path(
                    module_name, temp_file_path
                )
                loaded_classes = self._extract_models_from_module(
                    dynamic_module,
                    models_to_load_names,
                    generated_code_str,
                    module_name,
                )
                self._rebuild_and_validate_models(loaded_classes, generated_code_str)
                return loaded_classes, generated_code_str

        except (SQLModelCodeGeneratorError, SQLModelInstantiationValidationError):
            raise
        except Exception as e:
            error_message = (
                f"Failed to dynamically generate and load SQLModel class(es): {e}"
            )
            if generated_code_str:
                error_message += f"\nGenerated code:\n{generated_code_str}"
            raise SQLModelCodeGeneratorError(error_message) from e

    async def generate_and_load_models_via_llm(
        self,
        input_documents: TypingList[str],
        user_task_description: str,
        num_model_revisions: int = 1,
        max_retries_per_model_revision: int = 2,
    ) -> tuple[Dict[str, Type[SQLModel]], str]:
        """
        Generates SQLModel description(s) via LLM, then uses internal methods
        to generate Python code and dynamically load the class(es).

        Args:
            input_documents: A list of strings providing context for model structure.
            user_task_description: User's description of the desired model(s).
            num_model_revisions: Number of model descriptions to request from the LLM.
            max_retries_per_model_revision: Max LLM call retries per description revision.

        Returns:
            A tuple containing:
            - A dictionary mapping model names to the dynamically loaded SQLModel classes.
            - The generated Python code as a string.

        Raises:
            ConfigurationError: If the schema file for model descriptions is missing or invalid.
            LLMInteractionError: If the LLM fails to produce a valid description.
            SQLModelCodeGeneratorError: If code generation or dynamic loading fails.
            ValueError: If input documents or task description are empty.
        """
        if not input_documents:
            raise ValueError(
                "Input documents list cannot be empty for dynamic model generation."
            )
        if not user_task_description:
            raise ValueError(
                "User task description cannot be empty for dynamic model generation."
            )

        sqlmodel_desc_schema = self._load_sqlmodel_description_schema()

        system_prompt_for_model_gen = generate_sqlmodel_creation_system_prompt(
            schema_json=json.dumps(
                sqlmodel_desc_schema
            ),  # Schema for the *description*
            user_task_description=user_task_description,
        )
        user_prompt_for_model_gen = generate_user_prompt_for_docs(
            documents=input_documents
        )

        try:
            validated_descriptions: TypingList[
                Dict[str, Any]
            ] = await self.llm_client.generate_and_validate_raw_json_output(
                system_prompt=system_prompt_for_model_gen,
                user_prompt=user_prompt_for_model_gen,
                num_revisions=num_model_revisions,
                max_validation_retries_per_revision=max_retries_per_model_revision,
                analytics_collector=self.analytics_collector,
            )
        except (
            LLMConfigurationError,
            LLMOutputParseError,
            LLMOutputValidationError,
            LLMAPICallError,
        ) as client_err:
            self.analytics_collector.record_workflow_error(
                error_type=type(client_err).__name__,
                context="dynamic_sqlmodel_class_generation_llm",
                message=f"LLM client operation failed during dynamic SQLModel class generation: {client_err}",
            )
            raise LLMInteractionError(
                f"LLM client operation failed during dynamic SQLModel class generation: {client_err}"
            ) from client_err
        except Exception as e:
            self.analytics_collector.record_workflow_error(
                error_type="UnknownLLMError",
                context="dynamic_sqlmodel_class_generation_llm",
                message=f"An unexpected error occurred during LLM interaction for dynamic SQLModel class generation: {e}",
            )
            raise LLMInteractionError(
                f"An unexpected error occurred during LLM interaction for dynamic SQLModel class generation: {e}"
            ) from e

        if not validated_descriptions:
            self.analytics_collector.record_workflow_error(
                error_type="NoValidDescriptions",
                context="dynamic_sqlmodel_class_generation_llm",
                message="LLM did not return any valid SQLModel descriptions after all attempts.",
            )
            raise LLMInteractionError(
                "LLM did not return any valid SQLModel descriptions after all attempts."
            )

        if "sql_models" in validated_descriptions[0]:
            description_to_process = validated_descriptions[0]
        else:
            description_to_process = {"sql_models": validated_descriptions}

        self.analytics_collector.record_custom_event(
            event_name="chose_llm_generated_sqlmodel_description_for_class_load",
            details={"description_used": description_to_process},
        )

        main_model_name_for_analytics = "Unknown"
        if (
            "sql_models" in description_to_process
            and description_to_process["sql_models"]
        ):
            main_model_name_for_analytics = description_to_process["sql_models"][0].get(
                "model_name", "Unknown"
            )

        try:
            (
                loaded_classes_map,
                generated_code,
            ) = self._generate_and_load_class_from_description(description_to_process)

        except SQLModelCodeGeneratorError as e:
            self.analytics_collector.record_workflow_error(
                error_type=type(e).__name__,
                context="dynamic_sqlmodel_class_generation",
                message=f"Failed to generate and load dynamic SQLModel class: {e}",
            )
            raise
        except Exception as e:
            self.analytics_collector.record_workflow_error(
                error_type="UnexpectedDynamicClassLoadError",
                context="dynamic_sqlmodel_class_generation",
                message=f"An unexpected error occurred while generating/loading dynamic SQLModel class: {e}",
            )
            raise SQLModelCodeGeneratorError(
                f"An unexpected error occurred while generating/loading dynamic SQLModel class: {e}"
            ) from e

        self.analytics_collector.record_custom_event(
            event_name="dynamic_sqlmodel_class_generated_and_loaded_successfully",
            details={
                "model_name": main_model_name_for_analytics,
                "models_loaded": list(loaded_classes_map.keys()),
            },
        )
        return loaded_classes_map, generated_code
