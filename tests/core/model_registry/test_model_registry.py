import unittest
from unittest.mock import MagicMock, patch
from typing import Optional
from sqlmodel import SQLModel, Field

from extrai.core.model_registry import ModelRegistry, ConfigurationError


class MockRootModel(SQLModel):
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str


class TestModelRegistry(unittest.TestCase):
    def setUp(self):
        self.mock_logger = MagicMock()

    def test_init_success(self):
        with patch("extrai.core.model_registry.SchemaInspector") as MockInspector:
            mock_inspector = MockInspector.return_value
            mock_inspector.discover_sqlmodels_from_root.return_value = [MockRootModel]
            mock_inspector.generate_llm_schema_from_models.return_value = (
                '{"type": "object"}'
            )

            registry = ModelRegistry(MockRootModel, self.mock_logger)

            self.assertEqual(len(registry.models), 1)
            self.assertEqual(registry.models[0], MockRootModel)
            self.assertEqual(registry.llm_schema_json, '{"type": "object"}')
            self.assertEqual(registry.get_model_by_name("MockRootModel"), MockRootModel)

    def test_init_invalid_root_model(self):
        with self.assertRaises(ConfigurationError):
            ModelRegistry("NotAModel", self.mock_logger)

    def test_init_discovery_failure(self):
        with patch("extrai.core.model_registry.SchemaInspector") as MockInspector:
            mock_inspector = MockInspector.return_value
            mock_inspector.discover_sqlmodels_from_root.side_effect = Exception(
                "Discovery failed"
            )

            with self.assertRaises(ConfigurationError):
                ModelRegistry(MockRootModel, self.mock_logger)

    def test_init_empty_discovery(self):
        with patch("extrai.core.model_registry.SchemaInspector") as MockInspector:
            mock_inspector = MockInspector.return_value
            mock_inspector.discover_sqlmodels_from_root.return_value = []

            with self.assertRaises(ConfigurationError):
                ModelRegistry(MockRootModel, self.mock_logger)

    def test_init_schema_generation_failure(self):
        with patch("extrai.core.model_registry.SchemaInspector") as MockInspector:
            mock_inspector = MockInspector.return_value
            mock_inspector.discover_sqlmodels_from_root.return_value = [MockRootModel]
            mock_inspector.generate_llm_schema_from_models.side_effect = Exception(
                "Schema gen failed"
            )

            with self.assertRaises(ConfigurationError):
                ModelRegistry(MockRootModel, self.mock_logger)

    def test_init_invalid_json_schema(self):
        with patch("extrai.core.model_registry.SchemaInspector") as MockInspector:
            mock_inspector = MockInspector.return_value
            mock_inspector.discover_sqlmodels_from_root.return_value = [MockRootModel]
            mock_inspector.generate_llm_schema_from_models.return_value = "invalid json"

            with self.assertRaises(ConfigurationError):
                ModelRegistry(MockRootModel, self.mock_logger)

    def test_get_schema_for_models(self):
        with patch("extrai.core.model_registry.SchemaInspector") as MockInspector:
            mock_inspector = MockInspector.return_value
            mock_inspector.discover_sqlmodels_from_root.return_value = [MockRootModel]
            mock_inspector.generate_llm_schema_from_models.return_value = (
                '{"full": "schema"}'
            )

            registry = ModelRegistry(MockRootModel, self.mock_logger)

            # Reset mock to verify specific call
            mock_inspector.generate_llm_schema_from_models.return_value = (
                '{"partial": "schema"}'
            )

            schema = registry.get_schema_for_models(["MockRootModel"])

            self.assertEqual(schema, '{"partial": "schema"}')
            mock_inspector.generate_llm_schema_from_models.assert_called_with(
                [MockRootModel]
            )

    def test_get_schema_for_models_fallback(self):
        with patch("extrai.core.model_registry.SchemaInspector") as MockInspector:
            mock_inspector = MockInspector.return_value
            mock_inspector.discover_sqlmodels_from_root.return_value = [MockRootModel]
            mock_inspector.generate_llm_schema_from_models.return_value = (
                '{"full": "schema"}'
            )

            registry = ModelRegistry(MockRootModel, self.mock_logger)

            # Reset mock
            mock_inspector.generate_llm_schema_from_models.reset_mock()

            schema = registry.get_schema_for_models(["NonExistentModel"])

            self.assertEqual(schema, '{"full": "schema"}')
            mock_inspector.generate_llm_schema_from_models.assert_not_called()

    def test_init_root_model_class_not_subclass(self):
        class NotAModel:
            pass

        with self.assertRaisesRegex(
            ConfigurationError, "root_model must be a valid SQLModel class"
        ):
            ModelRegistry(NotAModel, self.mock_logger)

    def test_init_empty_generated_schema(self):
        with patch("extrai.core.model_registry.SchemaInspector") as MockInspector:
            mock_inspector = MockInspector.return_value
            mock_inspector.discover_sqlmodels_from_root.return_value = [MockRootModel]
            # Return empty schema
            mock_inspector.generate_llm_schema_from_models.return_value = ""

            with self.assertRaisesRegex(
                ConfigurationError, "Generated LLM schema is empty"
            ):
                ModelRegistry(MockRootModel, self.mock_logger)

    def test_get_schema_for_models_exception(self):
        with patch("extrai.core.model_registry.SchemaInspector") as MockInspector:
            mock_inspector = MockInspector.return_value
            mock_inspector.discover_sqlmodels_from_root.return_value = [MockRootModel]
            mock_inspector.generate_llm_schema_from_models.return_value = (
                '{"full": "schema"}'
            )

            registry = ModelRegistry(MockRootModel, self.mock_logger)

            # Make generation fail for specific call
            mock_inspector.generate_llm_schema_from_models.side_effect = Exception(
                "Boom"
            )

            schema = registry.get_schema_for_models(["MockRootModel"])

            self.assertEqual(schema, '{"full": "schema"}')
            # Verify error log
            self.mock_logger.error.assert_called()

    def test_get_all_model_names(self):
        with patch("extrai.core.model_registry.SchemaInspector") as MockInspector:
            mock_inspector = MockInspector.return_value
            mock_inspector.discover_sqlmodels_from_root.return_value = [MockRootModel]
            mock_inspector.generate_llm_schema_from_models.return_value = "{}"

            registry = ModelRegistry(MockRootModel, self.mock_logger)

            names = registry.get_all_model_names()
            self.assertEqual(names, ["MockRootModel"])

    def test_has_model(self):
        with patch("extrai.core.model_registry.SchemaInspector") as MockInspector:
            mock_inspector = MockInspector.return_value
            mock_inspector.discover_sqlmodels_from_root.return_value = [MockRootModel]
            mock_inspector.generate_llm_schema_from_models.return_value = "{}"

            registry = ModelRegistry(MockRootModel, self.mock_logger)

            self.assertTrue(registry.has_model("MockRootModel"))
            self.assertFalse(registry.has_model("Unknown"))

    def test_repr(self):
        with patch("extrai.core.model_registry.SchemaInspector") as MockInspector:
            mock_inspector = MockInspector.return_value
            mock_inspector.discover_sqlmodels_from_root.return_value = [MockRootModel]
            mock_inspector.generate_llm_schema_from_models.return_value = "{}"

            registry = ModelRegistry(MockRootModel, self.mock_logger)

            repr_str = repr(registry)
            self.assertIn("ModelRegistry", repr_str)
            self.assertIn("MockRootModel", repr_str)
            self.assertIn("models=1", repr_str)


if __name__ == "__main__":
    unittest.main()
