import pytest
import pytest_asyncio
import json
import os
import sys
from unittest import mock
from sqlmodel import SQLModel

from extrai.core.errors import (
    LLMInteractionError,
    LLMAPICallError,
    ConfigurationError,
    SQLModelCodeGeneratorError,
)
from extrai.core.sqlmodel_generator import SQLModelCodeGenerator
from extrai.core.analytics_collector import (
    WorkflowAnalyticsCollector,
)
from tests.core.helpers.mock_llm_clients import MockLLMClientSqlGen


class TestSQLModelCodeGeneratorLLMIntegrationRefactored:
    @pytest.fixture(autouse=True)
    def clear_sqlmodel_metadata(self):
        """Fixture to clear SQLModel metadata before each test to prevent table redefinition errors."""
        original_metadata_tables = dict(SQLModel.metadata.tables)
        SQLModel.metadata.clear()
        yield
        SQLModel.metadata.clear()
        for table_obj in original_metadata_tables.values():
            table_obj.to_metadata(SQLModel.metadata)

    @pytest_asyncio.fixture(autouse=True)
    async def setup_method_async(self):
        self.mock_llm_client = MockLLMClientSqlGen()
        self.mock_analytics_collector = mock.Mock(spec=WorkflowAnalyticsCollector)
        self.generator_for_llm_tests = SQLModelCodeGenerator(
            llm_client=self.mock_llm_client,
            analytics_collector=self.mock_analytics_collector,
        )
        self.sample_input_docs = ["doc1 about entities", "doc2 with more details"]
        self.sample_user_task_desc = (
            "Create a model for various entities described in documents."
        )
        self.mock_sqlmodel_description_schema = {
            "type": "object",
            "properties": {
                "model_name": {"type": "string"},
                "table_name": {"type": "string"},
                "fields": {"type": "array", "items": {"type": "object"}},
            },
            "required": ["model_name", "fields"],
        }
        SQLModelCodeGenerator._sqlmodel_description_schema_cache = None

    @pytest.mark.parametrize(
        "test_id, mock_os_exists_side_effect, mock_open_config, expected_exception, expected_match",
        [
            (
                "fallback_path_success",
                [False, True],  # Fails first, succeeds on fallback
                {"read_data": '{"test": "schema"}'},
                None,
                None,
            ),
            (
                "file_not_found_error",
                [False, False],  # Fails on both primary and fallback
                {"side_effect": FileNotFoundError},
                ConfigurationError,
                r"SQLModel description schema not found at .*",
            ),
            (
                "invalid_json_content",
                lambda path: True,
                {"read_data": "invalid json"},
                ConfigurationError,
                r"Invalid JSON in SQLModel description schema at .*",
            ),
            (
                "json_decode_error",
                lambda path: True,
                {"side_effect": json.JSONDecodeError("mock error", "doc", 0)},
                ConfigurationError,
                r"Invalid JSON in SQLModel description schema at .*",
            ),
        ],
    )
    def test_load_sqlmodel_description_schema_scenarios(
        self,
        test_id,
        mock_os_exists_side_effect,
        mock_open_config,
        expected_exception,
        expected_match,
    ):
        original_schema_path = SQLModelCodeGenerator._SCHEMA_FILE_PATH
        SQLModelCodeGenerator._sqlmodel_description_schema_cache = None

        sut_module_file_path = os.path.abspath(
            sys.modules[SQLModelCodeGenerator.__module__].__file__
        )
        sut_module_dir = os.path.dirname(sut_module_file_path)
        expected_fallback_path = os.path.normpath(
            os.path.join(sut_module_dir, "schemas", "sqlmodel_description_schema.json")
        )

        try:
            SQLModelCodeGenerator._SCHEMA_FILE_PATH = "non_existent_primary_path.json"

            mock_os_exists = mock.patch("os.path.exists")

            # mock_open does not support side_effect, so we need to patch it differently for those cases.
            if "side_effect" in mock_open_config:
                mock_file_patch = mock.patch(
                    "builtins.open", side_effect=mock_open_config["side_effect"]
                )
            else:
                mock_file_patch = mock.patch(
                    "builtins.open", mock.mock_open(**mock_open_config)
                )

            with (
                mock_os_exists as mock_os_exists_instance,
                mock_file_patch as mock_open_instance,
            ):
                mock_os_exists_instance.side_effect = mock_os_exists_side_effect

                if expected_exception:
                    with pytest.raises(expected_exception, match=expected_match):
                        self.generator_for_llm_tests._load_sqlmodel_description_schema()
                else:
                    schema = (
                        self.generator_for_llm_tests._load_sqlmodel_description_schema()
                    )
                    assert schema == json.loads(mock_open_config["read_data"])
                    mock_open_instance.assert_called_once_with(
                        expected_fallback_path, "r"
                    )

        finally:
            SQLModelCodeGenerator._SCHEMA_FILE_PATH = original_schema_path
            SQLModelCodeGenerator._sqlmodel_description_schema_cache = None

    @pytest.mark.asyncio
    @mock.patch("extrai.core.sqlmodel_generator.generate_user_prompt_for_docs")
    @mock.patch(
        "extrai.core.sqlmodel_generator.generate_sqlmodel_creation_system_prompt"
    )
    @mock.patch(
        "extrai.core.sqlmodel_generator.SQLModelCodeGenerator._load_sqlmodel_description_schema"
    )
    async def test_generate_and_load_models_via_llm_success(
        self, mock_load_schema, mock_gen_sqlmodel_creation_prompt, mock_gen_user_prompt
    ):
        mock_gen_sqlmodel_creation_prompt.return_value = (
            "test_system_prompt_for_model_gen"
        )
        mock_gen_user_prompt.return_value = "test_user_prompt_for_model_gen"
        mock_load_schema.return_value = self.mock_sqlmodel_description_schema

        llm_returned_description = {
            "sql_models": [
                {
                    "model_name": "DynamicEntityLLM",
                    "table_name": "dynamic_entities_llm",
                    "description": "A model dynamically generated via LLM.",
                    "fields": [
                        {
                            "name": "id",
                            "type": "Optional[int]",
                            "primary_key": True,
                            "default": None,
                            "nullable": True,
                        },
                        {
                            "name": "property_name",
                            "type": "str",
                            "default": "Default Property",
                            "nullable": False,
                        },
                        {
                            "name": "property_value",
                            "type": "Optional[float]",
                            "default": 0.0,
                            "nullable": True,
                        },
                    ],
                    "imports": ["from typing import Optional"],
                }
            ]
        }
        self.mock_llm_client.set_raw_json_outputs_to_return([llm_returned_description])

        (
            loaded_classes_map,
            _,
        ) = await self.generator_for_llm_tests.generate_and_load_models_via_llm(
            input_documents=self.sample_input_docs,
            user_task_description=self.sample_user_task_desc,
        )

        assert loaded_classes_map is not None
        assert "DynamicEntityLLM" in loaded_classes_map
        loaded_class = loaded_classes_map["DynamicEntityLLM"]
        assert issubclass(loaded_class, SQLModel)
        assert loaded_class.__name__ == "DynamicEntityLLM"
        assert getattr(loaded_class, "__tablename__") == "dynamic_entities_llm"

        self.mock_analytics_collector.record_custom_event.assert_any_call(
            event_name="dynamic_sqlmodel_class_generated_and_loaded_successfully",
            details={
                "model_name": "DynamicEntityLLM",
                "models_loaded": ["DynamicEntityLLM"],
            },
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "test_id, setup_mocks, expected_exception, expected_match, expected_analytics",
        [
            (
                "llm_api_error",
                lambda client, internal_loader: client.set_should_raise_exception(
                    LLMAPICallError("LLM API is down")
                ),
                LLMInteractionError,
                "LLM client operation failed.*LLM API is down",
                {
                    "type": "LLMAPICallError",
                    "context": "dynamic_sqlmodel_class_generation_llm",
                    "message": "LLM client operation failed during dynamic SQLModel class generation: LLM API is down",
                },
            ),
            (
                "no_valid_description",
                lambda client, internal_loader: client.set_raw_json_outputs_to_return(
                    []
                ),
                LLMInteractionError,
                "LLM did not return any valid SQLModel descriptions",
                {
                    "type": "NoValidDescriptions",
                    "context": "dynamic_sqlmodel_class_generation_llm",
                    "message": "LLM did not return any valid SQLModel descriptions after all attempts.",
                },
            ),
            (
                "internal_code_gen_error",
                lambda client, internal_loader: setattr(
                    internal_loader,
                    "side_effect",
                    SQLModelCodeGeneratorError("Internal class loading failed"),
                ),
                SQLModelCodeGeneratorError,
                "Internal class loading failed",
                {
                    "type": "SQLModelCodeGeneratorError",
                    "context": "dynamic_sqlmodel_class_generation",
                    "message": "Failed to generate and load dynamic SQLModel class: Internal class loading failed",
                },
            ),
            (
                "llm_client_generic_exception",
                lambda client, internal_loader: client.set_should_raise_exception(
                    RuntimeError("Unexpected LLM client error")
                ),
                LLMInteractionError,
                "An unexpected error occurred during LLM interaction.*Unexpected LLM client error",
                {
                    "type": "UnknownLLMError",
                    "context": "dynamic_sqlmodel_class_generation_llm",
                    "message": "An unexpected error occurred during LLM interaction for dynamic SQLModel class generation: Unexpected LLM client error",
                },
            ),
            (
                "internal_load_generic_exception",
                lambda client, internal_loader: setattr(
                    internal_loader,
                    "side_effect",
                    RuntimeError("Unexpected internal loading error"),
                ),
                SQLModelCodeGeneratorError,
                "An unexpected error occurred while generating/loading.*Unexpected internal loading error",
                {
                    "type": "UnexpectedDynamicClassLoadError",
                    "context": "dynamic_sqlmodel_class_generation",
                    "message": "An unexpected error occurred while generating/loading dynamic SQLModel class: Unexpected internal loading error",
                },
            ),
        ],
    )
    @mock.patch(
        "extrai.core.sqlmodel_generator.SQLModelCodeGenerator._generate_and_load_class_from_description"
    )
    @mock.patch(
        "extrai.core.sqlmodel_generator.SQLModelCodeGenerator._load_sqlmodel_description_schema"
    )
    async def test_generate_and_load_models_via_llm_error_handling(
        self,
        mock_load_schema,
        mock_internal_loader,
        test_id,
        setup_mocks,
        expected_exception,
        expected_match,
        expected_analytics,
    ):
        self.mock_analytics_collector.reset_mock()
        mock_load_schema.return_value = self.mock_sqlmodel_description_schema

        # For tests that don't raise an exception in the loader, ensure the LLM returns a description to trigger it.
        if "internal" in test_id:
            self.mock_llm_client.set_raw_json_outputs_to_return(
                [{"model_name": "ErrorModel", "fields": []}]
            )

        setup_mocks(self.mock_llm_client, mock_internal_loader)

        with pytest.raises(expected_exception, match=expected_match):
            await self.generator_for_llm_tests.generate_and_load_models_via_llm(
                self.sample_input_docs, self.sample_user_task_desc
            )

        self.mock_analytics_collector.record_workflow_error.assert_called_once_with(
            error_type=expected_analytics["type"],
            context=expected_analytics["context"],
            message=expected_analytics["message"],
        )

    @pytest.mark.asyncio
    async def test_generate_and_load_models_via_llm_empty_input_args(self):
        with pytest.raises(ValueError, match="Input documents list cannot be empty"):
            await self.generator_for_llm_tests.generate_and_load_models_via_llm(
                input_documents=[], user_task_description="A valid task description."
            )
        with pytest.raises(ValueError, match="User task description cannot be empty"):
            await self.generator_for_llm_tests.generate_and_load_models_via_llm(
                input_documents=["Some document content."], user_task_description=""
            )

    @pytest.mark.asyncio
    @mock.patch(
        "extrai.core.sqlmodel_generator.SQLModelCodeGenerator._load_sqlmodel_description_schema"
    )
    async def test_generate_and_load_models_via_llm_multiple_models_in_description(
        self, mock_load_schema
    ):
        mock_load_schema.return_value = self.mock_sqlmodel_description_schema
        llm_multi_model_desc = {
            "sql_models": [
                {"model_name": "Model1", "fields": []},
                {"model_name": "Model2", "fields": []},
            ]
        }

        with mock.patch.object(
            self.generator_for_llm_tests,
            "_generate_and_load_class_from_description",
            return_value=({"Model1": SQLModel, "Model2": SQLModel}, "mock_code"),
        ) as mock_internal_load:
            self.mock_llm_client.set_raw_json_outputs_to_return([llm_multi_model_desc])
            (
                result,
                _,
            ) = await self.generator_for_llm_tests.generate_and_load_models_via_llm(
                self.sample_input_docs, self.sample_user_task_desc
            )
            mock_internal_load.assert_called_once_with(llm_multi_model_desc)
            assert "Model1" in result and "Model2" in result

        self.mock_analytics_collector.record_custom_event.assert_any_call(
            event_name="dynamic_sqlmodel_class_generated_and_loaded_successfully",
            details={"model_name": "Model1", "models_loaded": ["Model1", "Model2"]},
        )
