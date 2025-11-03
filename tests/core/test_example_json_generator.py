import unittest
from unittest.mock import patch, AsyncMock, MagicMock
import json
from typing import Optional

from sqlmodel import SQLModel, Field  # Added

from extrai.core.example_json_generator import (
    ExampleJSONGenerator,
    ExampleGenerationError,
)
from extrai.core.errors import (
    LLMAPICallError,
    LLMOutputValidationError,
    ConfigurationError,
)  # Import for MockLLMError & new test


# Mock SQLModel for testing
class MockSQLModel(SQLModel):
    name: str
    age: int
    description: Optional[str] = None


class MockInvalidSQLModel(SQLModel):  # For testing validation failure
    id: int = Field(gt=0)  # Add a constraint


# Mock BaseLLMClient and its methods for testing purposes
class MockLLMClient:
    def __init__(self, api_key="test_key", model_name="test_model"):
        self.api_key = api_key
        self.model_name = model_name
        # Mock the method that will be called by ExampleJSONGenerator
        self.generate_json_revisions = AsyncMock()

    async def _execute_llm_call(self, system_prompt: str, user_prompt: str) -> str:
        # Not directly used by ExampleJSONGenerator's generate_example, but part of BaseLLMClient
        return "{}"


class TestExampleJSONGenerator(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.mock_llm_client = MockLLMClient()
        self.mock_output_model = MockSQLModel
        self.mock_analytics_collector = MagicMock()
        self.max_retries = 2

        self.generator = ExampleJSONGenerator(
            llm_client=self.mock_llm_client,
            output_model=self.mock_output_model,
            analytics_collector=self.mock_analytics_collector,
            max_validation_retries_per_revision=self.max_retries,
        )
        # Expected derived values
        # In the new implementation, schema generation is more complex.
        # We will mock the functions responsible for it to isolate the test.
        self.patcher_discover = patch(
            "extrai.core.example_json_generator.discover_sqlmodels_from_root"
        )
        self.patcher_generate_schema = patch(
            "extrai.core.example_json_generator.generate_llm_schema_from_models"
        )

        self.mock_discover = self.patcher_discover.start()
        self.mock_generate_schema = self.patcher_generate_schema.start()

        # Define what the mocked functions will return
        self.mock_discover.return_value = {self.mock_output_model}
        self.expected_schema_str = json.dumps(
            self.mock_output_model.model_json_schema()
        )
        self.mock_generate_schema.return_value = self.expected_schema_str

        self.generator = ExampleJSONGenerator(
            llm_client=self.mock_llm_client,
            output_model=self.mock_output_model,
            analytics_collector=self.mock_analytics_collector,
            max_validation_retries_per_revision=self.max_retries,
        )
        self.expected_root_model_name = self.mock_output_model.__name__
        # The target schema for the LLM client is now fixed
        self.expected_llm_client_schema = {
            "type": "object",
            "properties": {"entities": {"type": "array", "items": {"type": "object"}}},
            "required": ["entities"],
        }

    def tearDown(self):
        self.patcher_discover.stop()
        self.patcher_generate_schema.stop()

    def test_initialization_success(self):
        """Test that the ExampleJSONGenerator initializes correctly with the new schema logic."""
        self.mock_discover.assert_called_once_with(self.mock_output_model)
        self.mock_generate_schema.assert_called_once_with(
            initial_model_classes={self.mock_output_model}
        )

        self.assertEqual(self.generator.llm_client, self.mock_llm_client)
        self.assertEqual(self.generator.output_model, self.mock_output_model)
        self.assertEqual(
            self.generator.target_json_schema_for_llm_str, self.expected_schema_str
        )
        self.assertEqual(self.generator.root_model_name, self.expected_root_model_name)
        self.assertEqual(
            self.generator.target_json_schema_dict, self.expected_llm_client_schema
        )

    def test_initialization_failure_invalid_model(self):
        """Test initialization failure if output_model is not a SQLModel subclass."""

        class NotSQLModel:
            pass

        with self.assertRaisesRegex(
            ConfigurationError, "must be a subclass of SQLModel"
        ):
            ExampleJSONGenerator(
                llm_client=self.mock_llm_client,
                output_model=NotSQLModel,  # type: ignore
                max_validation_retries_per_revision=1,
            )

    def test_initialization_failure_no_llm_client(self):
        """Test initialization failure if llm_client is not provided."""
        with self.assertRaisesRegex(ConfigurationError, "llm_client must be provided."):
            ExampleJSONGenerator(
                llm_client=None,  # type: ignore
                output_model=self.mock_output_model,
                max_validation_retries_per_revision=1,
            )

    def test_initialization_failure_no_output_model(self):
        """Test initialization failure if output_model is not provided."""
        with self.assertRaisesRegex(
            ConfigurationError, "output_model must be provided."
        ):
            ExampleJSONGenerator(
                llm_client=self.mock_llm_client,
                output_model=None,  # type: ignore
                max_validation_retries_per_revision=1,
            )

    def test_initialization_failure_invalid_max_retries(self):
        """Test initialization failure if max_validation_retries_per_revision is less than 1."""
        with self.assertRaisesRegex(
            ConfigurationError,
            "max_validation_retries_per_revision must be at least 1.",
        ):
            ExampleJSONGenerator(
                llm_client=self.mock_llm_client,
                output_model=self.mock_output_model,
                max_validation_retries_per_revision=0,
            )

    def test_initialization_failure_schema_derivation_error(self):
        """Test initialization failure if model_json_schema raises an error."""
        mock_model_with_bad_schema = MagicMock(spec=SQLModel)
        mock_model_with_bad_schema.__name__ = "BadSchemaModel"
        # Ensure it's recognized as a subclass of SQLModel to pass the earlier check
        mock_model_with_bad_schema.__mro__ = (
            mock_model_with_bad_schema,
            SQLModel,
            object,
        )

        # Mock issubclass to return True for this specific mock
        def mock_issubclass(obj, classinfo):
            if obj is mock_model_with_bad_schema and classinfo is SQLModel:
                return True
            return orig_issubclass(obj, classinfo)

        orig_issubclass = issubclass  # Store original issubclass

        with patch(
            "extrai.core.example_json_generator.discover_sqlmodels_from_root",
            side_effect=Exception("Discovery failed"),
        ):
            with self.assertRaisesRegex(
                ConfigurationError,
                "Failed to derive JSON schema from output_model MockSQLModel: Discovery failed",
            ):
                ExampleJSONGenerator(
                    llm_client=self.mock_llm_client,
                    output_model=self.mock_output_model,
                    max_validation_retries_per_revision=1,
                )
        # Restore original issubclass if necessary, though patch should handle it
        # For safety, or if not using patch.object for issubclass on the module itself.

    @patch(
        "extrai.core.example_json_generator.generate_prompt_for_example_json_generation"
    )
    async def test_generate_example_success(self, mock_generate_prompt):
        """Test successful generation of a nested example JSON string."""
        expected_prompt_text = "Test system prompt"
        mock_generate_prompt.return_value = expected_prompt_text

        root_object = {"_type": "MockSQLModel", "name": "Test", "age": 25}
        # The LLM is now expected to return a list of validated objects.
        generated_llm_output = [root_object]
        expected_final_json_str = json.dumps(generated_llm_output)

        self.mock_llm_client.generate_json_revisions.return_value = generated_llm_output

        result_json_str = await self.generator.generate_example()

        mock_generate_prompt.assert_called_once_with(
            target_model_schema_str=self.expected_schema_str,
            root_model_name=self.expected_root_model_name,
        )
        # The model_schema_map is generated inside the method, so we replicate it for the check
        expected_model_schema_map = {
            self.mock_output_model.__name__: self.mock_output_model
        }
        self.mock_llm_client.generate_json_revisions.assert_called_once_with(
            system_prompt=expected_prompt_text,
            user_prompt="Please generate a sample JSON object based on the schema and instructions provided in the system prompt.",
            num_revisions=1,
            model_schema_map=expected_model_schema_map,
            max_validation_retries_per_revision=self.max_retries,
            analytics_collector=self.mock_analytics_collector,
        )
        self.assertEqual(result_json_str, expected_final_json_str)

    @patch(
        "extrai.core.example_json_generator.generate_prompt_for_example_json_generation"
    )
    async def test_generate_example_llm_failure(self, mock_generate_prompt):
        """Test failure during LLM call in example generation."""
        expected_prompt_text = "Test system prompt for example generation"
        mock_generate_prompt.return_value = expected_prompt_text

        self.mock_llm_client.generate_json_revisions.side_effect = LLMAPICallError(
            "LLM API failed"
        )

        with self.assertRaises(ExampleGenerationError) as context:
            await self.generator.generate_example()

        self.assertIn(
            "Failed to generate example JSON due to LLM client error",
            str(context.exception),
        )

        self.assertIsInstance(context.exception.original_exception, LLMAPICallError)

        mock_generate_prompt.assert_called_once_with(
            target_model_schema_str=self.expected_schema_str,
            root_model_name=self.expected_root_model_name,
        )
        self.mock_llm_client.generate_json_revisions.assert_called_once()

    @patch(
        "extrai.core.example_json_generator.generate_prompt_for_example_json_generation"
    )
    async def test_generate_example_llm_returns_empty_list(self, mock_generate_prompt):
        """Test scenario where LLM returns an empty list (no valid revisions)."""
        expected_prompt_text = "Test system prompt for example generation"
        mock_generate_prompt.return_value = expected_prompt_text

        self.mock_llm_client.generate_json_revisions.return_value = []

        with self.assertRaisesRegex(
            ExampleGenerationError,
            "LLM client returned no valid example JSON revisions.",
        ):
            await self.generator.generate_example()

        mock_generate_prompt.assert_called_once_with(
            target_model_schema_str=self.expected_schema_str,
            root_model_name=self.expected_root_model_name,
        )
        self.mock_llm_client.generate_json_revisions.assert_called_once()

    @patch(
        "extrai.core.example_json_generator.generate_prompt_for_example_json_generation"
    )
    async def test_generate_example_validation_failure(self, mock_generate_prompt):
        """Test failure during the SQLModel validation step."""
        expected_prompt_text = "Test system prompt for example generation"
        mock_generate_prompt.return_value = expected_prompt_text

        validation_error_msg = "Pydantic validation failed"
        self.mock_llm_client.generate_json_revisions.side_effect = (
            LLMOutputValidationError(validation_error_msg)
        )

        with self.assertRaises(ExampleGenerationError) as context:
            await self.generator.generate_example()

        self.assertIn(
            "Failed to generate example JSON due to LLM client error",
            str(context.exception),
        )
        self.assertIsInstance(
            context.exception.original_exception, LLMOutputValidationError
        )

        mock_generate_prompt.assert_called_once_with(
            target_model_schema_str=self.expected_schema_str,
            root_model_name=self.expected_root_model_name,
        )

    @patch(
        "extrai.core.example_json_generator.generate_prompt_for_example_json_generation"
    )
    async def test_generate_example_json_dumps_failure(self, mock_generate_prompt):
        """Test failure during json.dumps if LLM output is not serializable."""
        expected_prompt_text = "Test system prompt for example generation"
        mock_generate_prompt.return_value = expected_prompt_text

        # Create a structure that is valid up to the point of serialization
        non_serializable_entity = {"_type": "MockSQLModel", "data": object()}

        # Configure the mock to return the non-serializable object.
        # This bypasses the need to mock the validation function, as we are controlling the output directly.
        self.mock_llm_client.generate_json_revisions.return_value = [
            non_serializable_entity
        ]

        with self.assertRaises(ExampleGenerationError) as context:
            await self.generator.generate_example()

        self.assertIn(
            "Failed to serialize the generated example JSON", str(context.exception)
        )

        self.assertIsInstance(context.exception.original_exception, TypeError)
        self.assertIn(
            "not JSON serializable", str(context.exception.original_exception)
        )

        mock_generate_prompt.assert_called_once_with(
            target_model_schema_str=self.expected_schema_str,
            root_model_name=self.expected_root_model_name,
        )

    @patch(
        "extrai.core.example_json_generator.generate_prompt_for_example_json_generation"
    )
    async def test_generate_example_unexpected_error(self, mock_generate_prompt):
        """Test a generic unexpected error during example generation."""
        expected_prompt_text = "Test system prompt for example generation"
        mock_generate_prompt.return_value = expected_prompt_text

        # Make the LLM call raise an unexpected error
        self.mock_llm_client.generate_json_revisions.side_effect = Exception(
            "A very unexpected LLM error!"
        )

        with self.assertRaises(ExampleGenerationError) as context:
            await self.generator.generate_example()

        self.assertIn(
            "An unexpected error occurred during example JSON generation",
            str(context.exception),
        )

        self.assertIsInstance(context.exception.original_exception, Exception)

        mock_generate_prompt.assert_called_once_with(
            target_model_schema_str=self.expected_schema_str,
            root_model_name=self.expected_root_model_name,
        )
        # Check that the mocked LLM method was called
        expected_model_schema_map = {
            self.mock_output_model.__name__: self.mock_output_model
        }
        self.mock_llm_client.generate_json_revisions.assert_called_once_with(
            system_prompt=expected_prompt_text,
            user_prompt="Please generate a sample JSON object based on the schema and instructions provided in the system prompt.",
            num_revisions=1,
            model_schema_map=expected_model_schema_map,
            max_validation_retries_per_revision=self.max_retries,
            analytics_collector=self.mock_analytics_collector,
        )


if __name__ == "__main__":
    unittest.main()
