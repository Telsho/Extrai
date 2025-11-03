import pytest
import pytest_asyncio
import json
import os
import re
import sys
import ast
from unittest import mock
from sqlalchemy import inspect as sqlalchemy_inspect
from sqlmodel import SQLModel, create_engine as sqlmodel_create_engine

from extrai.core.errors import (
    LLMInteractionError,
    LLMAPICallError,
    ConfigurationError,
    SQLModelCodeGeneratorError,
    SQLModelInstantiationValidationError,
)
from extrai.core.sqlmodel_generator import SQLModelCodeGenerator
from extrai.core.analytics_collector import (
    WorkflowAnalyticsCollector,
)
from tests.core.helpers.mock_llm_clients import MockLLMClientSqlGen
from tests.core.helpers.sqlmodel_generator_test_utils import (
    is_valid_python_code,
    get_field_call_kwargs,
    get_annotation_str,
    get_class_def_node,
)

# This file will contain the refactored tests for both
# test_sqlmodel_generator_llm.py and test_sqlmodel_generator_codegen.py.


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


class TestSQLModelCodeGeneratorCodeGenRefactored:
    @pytest.fixture(autouse=True)
    def clear_sqlmodel_metadata(self):
        """Fixture to clear SQLModel metadata before each test to prevent table redefinition errors."""
        original_metadata_tables = dict(SQLModel.metadata.tables)
        SQLModel.metadata.clear()
        yield
        SQLModel.metadata.clear()
        for table_obj in original_metadata_tables.values():
            table_obj.to_metadata(SQLModel.metadata)

    def setup_method(self):
        self.mock_llm_client = MockLLMClientSqlGen()
        self.mock_analytics_collector = mock.Mock(spec=WorkflowAnalyticsCollector)
        self.generator = SQLModelCodeGenerator(
            llm_client=self.mock_llm_client,
            analytics_collector=self.mock_analytics_collector,
        )
        SQLModelCodeGenerator._sqlmodel_description_schema_cache = None

    @pytest.fixture(scope="function")
    def comprehensive_code_ast_and_desc(self):
        comprehensive_description = {
            "sql_models": [
                {
                    "table_name": "comprehensive_items",
                    "model_name": "ComprehensiveItem",
                    "description": 'A comprehensive model with "quotes" and\nnewlines.',
                    "fields": [
                        {
                            "name": "id",
                            "type": "Optional[int]",
                            "primary_key": True,
                            "nullable": True,
                        },
                        {
                            "name": "name",
                            "type": "str",
                            "description": "Name of the item.",
                        },
                        {
                            "name": "entity_uuid",
                            "type": "uuid.UUID",
                            "primary_key": False,
                            "default_factory": "uuid.uuid4",
                            "index": True,
                            "nullable": False,
                        },
                        {
                            "name": "unique_name",
                            "type": "str",
                            "unique": True,
                            "index": True,
                        },
                        {
                            "name": "description_field",
                            "type": "Optional[str]",
                            "nullable": True,
                            "description": 'A field with "special" chars & new\nline.',
                        },
                        {"name": "count_val", "type": "int", "default": 0},
                        {"name": "amount_val", "type": "float", "default": 0.0},
                        {"name": "is_active_flag", "type": "bool", "default": True},
                        {
                            "name": "created_timestamp",
                            "type": "datetime.datetime",
                            "default_factory": "datetime.utcnow",
                            "sa_column_kwargs": {"server_default": "FUNC.now()"},
                        },
                        {
                            "name": "updated_timestamp",
                            "type": "Optional[datetime.datetime]",
                            "nullable": True,
                            "sa_column_kwargs": {"onupdate": "FUNC.now()"},
                        },
                        {"name": "tags_list", "type": "List[str]"},
                        {"name": "config_dict", "type": "Dict[str, Any]"},
                        {
                            "name": "related_id",
                            "type": "Optional[int]",
                            "foreign_key": "other_table.id",
                            "nullable": True,
                        },
                        {"name": "optional_only_field", "type": "Optional[float]"},
                        {
                            "name": "class",
                            "type": "str",
                            "description": "A field named 'class'",
                        },
                        {
                            "name": "json_payload",
                            "type": "Dict",
                            "sa_column_kwargs": {"sa_type": "JSON"},
                        },
                        {
                            "name": "sqlalchemy_json_payload",
                            "type": "List",
                            "sa_column_kwargs": {"sa_type": "sqlalchemy.JSON"},
                        },
                    ],
                }
            ]
        }
        code = self.generator._generate_code_from_description(comprehensive_description)
        assert is_valid_python_code(code), (
            f"Generated code is not valid Python:\n{code}"
        )
        tree = ast.parse(code)
        return tree, comprehensive_description

    def test_comprehensive_model_generation_and_ast_validation(
        self, comprehensive_code_ast_and_desc
    ):
        tree, _ = comprehensive_code_ast_and_desc

        # 1. Test Imports
        expected_imports_from_typing = {"Optional", "List", "Dict", "Any"}
        actual_imports_from_typing = set()
        expected_imports_from_sqlmodel = {"SQLModel", "Field", "JSON"}
        actual_imports_from_sqlmodel = set()
        imported_modules = set()

        for node in tree.body:
            if isinstance(node, ast.ImportFrom):
                if node.module == "typing":
                    actual_imports_from_typing.update(
                        alias.name for alias in node.names
                    )
                elif node.module == "sqlmodel":
                    actual_imports_from_sqlmodel.update(
                        alias.name for alias in node.names
                    )
            elif isinstance(node, ast.Import):
                imported_modules.update(alias.name for alias in node.names)

        assert actual_imports_from_typing.issuperset(expected_imports_from_typing)
        assert actual_imports_from_sqlmodel.issuperset(expected_imports_from_sqlmodel)
        assert "uuid" in imported_modules
        assert "datetime" in imported_modules
        assert "sqlalchemy" in imported_modules

        # 2. Test Class Definition
        class_def_node = get_class_def_node(tree, "ComprehensiveItem")
        assert class_def_node is not None, "Class 'ComprehensiveItem' not found."
        assert class_def_node.name == "ComprehensiveItem"
        assert class_def_node.bases[0].id == "SQLModel"
        assert any(
            kw.arg == "table" and kw.value.value is True
            for kw in class_def_node.keywords
        )

        # 3. Test Fields
        fields_ast = {
            node.target.id: node
            for node in class_def_node.body
            if isinstance(node, ast.AnnAssign)
        }

        assert get_annotation_str(fields_ast["id"].annotation) == "Optional[int]"
        assert get_field_call_kwargs(fields_ast["id"].value) == {
            "primary_key": True,
            "nullable": True,
        }

        assert get_annotation_str(fields_ast["name"].annotation) == "str"
        assert get_field_call_kwargs(fields_ast["name"].value) == {
            "description": "Name of the item."
        }

        assert get_annotation_str(fields_ast["entity_uuid"].annotation) == "uuid.UUID"
        assert get_field_call_kwargs(fields_ast["entity_uuid"].value) == {
            "default_factory": "uuid.uuid4",
            "index": True,
            "nullable": False,
        }

        assert get_annotation_str(fields_ast["class_"].annotation) == "str"
        assert get_field_call_kwargs(fields_ast["class_"].value) == {
            "description": "A field named 'class'",
            "alias": "class",
        }

        assert get_annotation_str(fields_ast["json_payload"].annotation) == "Dict"
        assert get_field_call_kwargs(fields_ast["json_payload"].value) == {
            "sa_type": "JSON"
        }

        assert (
            get_annotation_str(fields_ast["sqlalchemy_json_payload"].annotation)
            == "List"
        )
        assert get_field_call_kwargs(fields_ast["sqlalchemy_json_payload"].value) == {
            "sa_type": "sqlalchemy.JSON"
        }

    @pytest.mark.parametrize(
        "test_id, model_desc, expected_exception, match_message",
        [
            (
                "instantiation_validation_error",
                {
                    "model_name": "ValidationErrorModel",
                    "fields": [
                        {"name": "id", "type": "Optional[int]", "primary_key": True},
                        {"name": "field_a", "type": "int"},
                    ],
                },
                SQLModelInstantiationValidationError,
                r"Default instantiation of 'ValidationErrorModel' failed with ValidationError.*",
            ),
            (
                "instantiation_unexpected_error",
                {
                    "model_name": "UnexpectedErrorModel",
                    "fields": [
                        {"name": "id", "type": "Optional[int]", "primary_key": True},
                        {
                            "name": "bad_field",
                            "type": "int",
                            "default_factory": "list.append",
                        },
                    ],
                },
                SQLModelCodeGeneratorError,
                r"failed instantiation with an unexpected error.*unbound method list\.append.*",
            ),
        ],
    )
    def test_generate_and_load_with_natural_errors(
        self, test_id, model_desc, expected_exception, match_message
    ):
        """Tests errors that occur naturally from the code generation and validation process."""
        with pytest.raises(expected_exception, match=match_message):
            self.generator._generate_and_load_class_from_description(
                {"sql_models": [model_desc]}
            )

    @pytest.mark.parametrize(
        "test_id, model_desc, mock_setup, expected_exception, match_message",
        [
            (
                "spec_creation_fails",
                {"model_name": "SpecFailModel", "fields": []},
                lambda mocks: (
                    setattr(
                        mocks["generate_code"],
                        "return_value",
                        "class SpecFailModel: pass",
                    ),
                    setattr(mocks["spec_from_file"], "return_value", None),
                ),
                SQLModelCodeGeneratorError,
                "Failed to create import spec",
            ),
            (
                "attr_not_a_class",
                {"model_name": "NotAClassModel", "fields": []},
                lambda mocks: (
                    setattr(
                        mocks["generate_code"],
                        "return_value",
                        "class NotAClassModel: pass",
                    ),
                    setattr(
                        mocks["module_from_spec"].return_value, "NotAClassModel", 123
                    ),
                ),
                SQLModelCodeGeneratorError,
                "Loaded attribute 'NotAClassModel' is not a class or not a subclass of SQLModel.",
            ),
            (
                "class_not_found_in_module",
                {"model_name": "MissingModel", "fields": []},
                lambda mocks: (
                    setattr(
                        mocks["generate_code"],
                        "return_value",
                        "class FoundModel(SQLModel): pass",
                    ),
                    setattr(
                        mocks["module_from_spec"].return_value, "MissingModel", None
                    ),
                ),
                SQLModelCodeGeneratorError,
                r"Class 'MissingModel' not found in dynamically loaded module",
            ),
        ],
    )
    @mock.patch(
        "extrai.core.sqlmodel_generator.SQLModelCodeGenerator._generate_code_from_description"
    )
    @mock.patch("importlib.util.module_from_spec")
    @mock.patch("importlib.util.spec_from_file_location")
    def test_load_process_with_mocked_errors(
        self,
        mock_spec_from_file,
        mock_module_from_spec,
        mock_generate_code,
        test_id,
        model_desc,
        mock_setup,
        expected_exception,
        match_message,
    ):
        """Tests errors in the loading process that require mocking importlib."""
        mock_spec = mock.Mock()
        mock_spec.loader = mock.Mock()
        mock_spec_from_file.return_value = mock_spec

        if mock_setup:
            mocks = {
                "spec_from_file": mock_spec_from_file,
                "module_from_spec": mock_module_from_spec,
                "generate_code": mock_generate_code,
            }
            mock_setup(mocks)

        with pytest.raises(expected_exception, match=match_message):
            self.generator._generate_and_load_class_from_description(
                {"sql_models": [model_desc]}
            )

    @pytest.mark.parametrize(
        "test_id, description, expected_substrings",
        [
            (
                "keyword_field_name_with_options",
                {
                    "model_name": "KeywordFieldWithOptions",
                    "fields": [
                        {
                            "name": "class",
                            "type": "str",
                            "field_options_str": 'Field(alias="class")',
                        }
                    ],
                },
                ['class_: str = Field(alias="class")'],
            ),
            (
                "custom_complex_import",
                {
                    "model_name": "ComplexImportModel",
                    "fields": [],
                    "imports": ["from a.b import c as d"],
                },
                ["from a.b import c as d"],
            ),
            (
                "no_fields",
                {
                    "model_name": "NoFieldsModel",
                    "table_name": "no_fields",
                    "fields": [],
                },
                ["class NoFieldsModel(SQLModel, table=True):", "    pass"],
            ),
            (
                "no_model_description",
                {"model_name": "NoDescItem", "fields": [{"name": "id", "type": "int"}]},
                [],
            ),  # second assertion checks for absence
            (
                "non_table_model_with_description",
                {
                    "model_name": "NonTableModel",
                    "description": "A test model.",
                    "is_table_model": False,
                    "fields": [],
                },
                [
                    "class NonTableModel(SQLModel):",
                    '    """A test model."""',
                    "    pass",
                ],
            ),
            (
                "multiple_base_classes_with_sqlmodel",
                {
                    "model_name": "MultiBase",
                    "is_table_model": True,
                    "base_classes_str": ["CustomBase", "SQLModel"],
                    "fields": [{"name": "id", "type": "int", "primary_key": True}],
                },
                ["class MultiBase(CustomBase, SQLModel, table=True):"],
            ),
        ],
    )
    def test_special_code_generation_cases(
        self, test_id, description, expected_substrings
    ):
        code = self.generator._generate_code_from_description(
            {"sql_models": [description]}
        )
        assert is_valid_python_code(code)
        for substring in expected_substrings:
            assert substring in code

        if test_id == "no_model_description":
            class_def_line_index = code.find(f"class {description['model_name']}")
            assert '"""' not in code[class_def_line_index:]

    def test_init_with_none_analytics_collector(self):
        generator_no_collector = SQLModelCodeGenerator(
            llm_client=self.mock_llm_client, analytics_collector=None
        )
        assert isinstance(
            generator_no_collector.analytics_collector, WorkflowAnalyticsCollector
        )

    @mock.patch("os.rmdir")
    @mock.patch("os.remove")
    def test_generate_load_and_validate_sqlmodel_class_e2e(
        self, mock_os_remove, mock_os_rmdir
    ):
        model_desc = {
            "sql_models": [
                {
                    "model_name": "E2EProduct",
                    "table_name": "e2e_products",
                    "fields": [
                        {
                            "name": "id",
                            "type": "Optional[int]",
                            "primary_key": True,
                            "default": None,
                            "nullable": True,
                        },
                        {"name": "name", "type": "str", "default": "Default Product"},
                    ],
                }
            ]
        }

        (
            loaded_models,
            _,
        ) = self.generator._generate_and_load_class_from_description(model_desc)
        loaded_product_model = loaded_models.get("E2EProduct")

        assert loaded_product_model is not None
        assert issubclass(loaded_product_model, SQLModel)

        engine = sqlmodel_create_engine("sqlite:///:memory:")
        SQLModel.metadata.create_all(engine)

        inspector = sqlalchemy_inspect(engine)
        assert "e2e_products" in inspector.get_table_names()
        engine.dispose()

        # Verify cleanup was called
        assert mock_os_remove.call_count > 0
        assert mock_os_rmdir.call_count > 0

    @pytest.mark.parametrize(
        "test_id, model_description, expected_code_snippets",
        [
            (
                "union_import",
                {
                    "model_name": "M1",
                    "fields": [{"name": "f", "type": "Union[int, str]"}],
                },
                ["from typing import Union"],
            ),
            (
                "complex_import",
                {"model_name": "M3", "imports": ["import my_library"], "fields": []},
                ["import my_library"],
            ),
            (
                "custom_sqlmodel_import_merge",
                {
                    "model_name": "M4",
                    "imports": ["from sqlmodel import Field, Session"],
                    "fields": [{"name": "id", "type": "int", "primary_key": True}],
                },
                [r"from sqlmodel import Field, SQLModel, Session"],
            ),
        ],
    )
    def test_code_generation_import_logic(
        self, test_id, model_description, expected_code_snippets
    ):
        """Covers multiple import logic paths in _ImportManager and _FieldGenerator."""
        code = self.generator._generate_code_from_description(
            {"sql_models": [model_description]}
        )
        assert is_valid_python_code(code)
        for snippet in expected_code_snippets:
            assert re.search(snippet, code)

    @mock.patch("os.remove")
    @mock.patch("os.rmdir")
    def test_generate_and_load_class_cleanup_os_error(self, mock_rmdir, mock_remove):
        mock_remove.side_effect = OSError("Cannot remove file")
        mock_rmdir.side_effect = OSError("Cannot remove dir")
        model_desc = {
            "sql_models": [
                {
                    "model_name": "CleanupFailModel",
                    "fields": [
                        {"name": "id", "type": "int", "primary_key": True, "default": 0}
                    ],
                }
            ]
        }

        with mock.patch.object(self.generator.logger, "warning") as mock_logger_warning:
            self.generator._generate_and_load_class_from_description(model_desc)

            # Check that the logger's warning method was called with the expected messages
            call_args_list = mock_logger_warning.call_args_list
            warnings = [call[0][0] for call in call_args_list]

            assert any(
                "Could not remove temporary file" in warning for warning in warnings
            )
            assert any(
                "Could not remove temporary directory" in warning
                for warning in warnings
            )

    def test_generate_and_load_from_multiple_model_description(self):
        """Covers the 'models' list path in _generate_and_load_class_from_description."""
        multi_model_desc = {
            "sql_models": [
                {
                    "model_name": "MultiModel1",
                    "fields": [{"name": "id", "type": "int", "primary_key": True}],
                },
                {
                    "model_name": "MultiModel2",
                    "fields": [{"name": "name", "type": "str"}],
                },
            ]
        }

        # This test primarily ensures the loading logic correctly identifies the models to load.
        # We can mock the code generation part to simplify.
        generated_code = """
from sqlmodel import SQLModel
from typing import Optional
class MultiModel1(SQLModel):
    id: Optional[int] = None
class MultiModel2(SQLModel):
    name: Optional[str] = None
"""
        with mock.patch.object(
            self.generator,
            "_generate_code_from_description",
            return_value=generated_code,
        ):
            (
                loaded_classes,
                _,
            ) = self.generator._generate_and_load_class_from_description(
                multi_model_desc
            )
            assert "MultiModel1" in loaded_classes
            assert "MultiModel2" in loaded_classes
            assert issubclass(loaded_classes["MultiModel1"], SQLModel)
            assert issubclass(loaded_classes["MultiModel2"], SQLModel)

    def test_code_generation_with_json_in_field_options_str(self):
        """Covers the JSON import logic when 'JSON' is in field_options_str."""
        model_description = {
            "sql_models": [
                {
                    "model_name": "JsonFieldModel",
                    "fields": [
                        {
                            "name": "data",
                            "type": "Dict",
                            "field_options_str": "Field(sa_column=Column(JSON))",
                        }
                    ],
                }
            ]
        }
        code = self.generator._generate_code_from_description(model_description)
        assert is_valid_python_code(code)

        tree = ast.parse(code)

        # Verify import
        json_imported = False
        for node in tree.body:
            if isinstance(node, ast.ImportFrom) and node.module == "sqlmodel":
                if any(alias.name == "JSON" for alias in node.names):
                    json_imported = True
                    break
        assert json_imported, (
            "from sqlmodel import JSON was not found in generated code."
        )

    @mock.patch(
        "extrai.core.sqlmodel_generator.SQLModelCodeGenerator._import_module_from_path"
    )
    def test_generate_and_load_catches_and_wraps_generic_exception(
        self, mock_import_module
    ):
        """Covers the generic exception handling block in _generate_and_load_class_from_description."""
        mock_import_module.side_effect = Exception("A mocked generic error occurred")
        model_desc = {
            "sql_models": [{"model_name": "GenericExceptionTestModel", "fields": []}]
        }

        with pytest.raises(SQLModelCodeGeneratorError) as exc_info:
            self.generator._generate_and_load_class_from_description(model_desc)

        # Check that the outer exception message is correct
        assert (
            "Failed to dynamically generate and load SQLModel class(es): A mocked generic error occurred"
            in str(exc_info.value)
        )

        # Check that the generated code is included in the error message
        assert "class GenericExceptionTestModel(SQLModel, table=True):" in str(
            exc_info.value
        )

        # Check that the original exception is preserved in the cause chain
        assert isinstance(exc_info.value.__cause__, Exception)
        assert str(exc_info.value.__cause__) == "A mocked generic error occurred"

    def test_code_generation_with_relationship_import(self):
        """Covers the Relationship import logic when 'Relationship' is in field_options_str."""
        model_description = {
            "sql_models": [
                {
                    "model_name": "Invoice",
                    "fields": [
                        {"name": "id", "type": "Optional[int]", "primary_key": True},
                        {
                            "name": "line_items",
                            "type": "List['LineItem']",
                            "field_options_str": 'Relationship(back_populates="invoice")',
                        },
                    ],
                },
                {
                    "model_name": "LineItem",
                    "fields": [
                        {"name": "id", "type": "Optional[int]", "primary_key": True},
                        {
                            "name": "invoice_id",
                            "type": "Optional[int]",
                            "foreign_key": "invoice.id",
                        },
                        {
                            "name": "invoice",
                            "type": "Optional['Invoice']",
                            "field_options_str": 'Relationship(back_populates="line_items")',
                        },
                    ],
                },
            ]
        }
        code = self.generator._generate_code_from_description(model_description)
        assert is_valid_python_code(code)

        assert re.search(r"from sqlmodel import .*Relationship", code)

    def test_load_fails_when_no_models_in_description(self):
        """Covers the error path when the 'sql_models' list is empty."""
        model_desc = {"sql_models": []}
        with pytest.raises(
            SQLModelCodeGeneratorError,
            match="No models found in the 'sql_models' list from the LLM description.",
        ):
            self.generator._generate_and_load_class_from_description(model_desc)
