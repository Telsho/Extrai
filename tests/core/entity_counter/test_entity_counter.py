import unittest
from unittest.mock import MagicMock, patch, AsyncMock
from extrai.core.entity_counter import EntityCounter
from extrai.core.model_registry import ModelRegistry
from extrai.core.extraction_config import ExtractionConfig
from extrai.core.base_llm_client import BaseLLMClient

class TestEntityCounter(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.mock_model_registry = MagicMock(spec=ModelRegistry)
        self.mock_client = MagicMock(spec=BaseLLMClient)
        self.mock_config = MagicMock(spec=ExtractionConfig)
        self.mock_config.max_validation_retries_per_revision = 1
        self.mock_analytics = MagicMock()
        self.mock_logger = MagicMock()
        
        self.counter = EntityCounter(
            self.mock_model_registry,
            self.mock_client,
            self.mock_config,
            self.mock_analytics,
            self.mock_logger
        )

    @patch('extrai.core.entity_counter.generate_entity_counting_system_prompt')
    @patch('extrai.core.entity_counter.generate_entity_counting_user_prompt')
    @patch('extrai.core.entity_counter.create_model')
    async def test_count_entities_success(self, mock_create_model, mock_user_prompt, mock_system_prompt):
        # Setup mocks
        self.mock_model_registry.get_schema_for_models.return_value = '{"type": "object"}'
        self.mock_client.generate_and_validate_raw_json_output = AsyncMock(return_value={"ModelA": 5})
        
        mock_model_instance = MagicMock()
        mock_model_instance.model_dump.return_value = {"ModelA": 5}
        
        # Mock the dynamically created Pydantic model
        MockPydanticModel = MagicMock()
        MockPydanticModel.return_value = mock_model_instance
        mock_create_model.return_value = MockPydanticModel
        
        counts = await self.counter.count_entities(["doc"], ["ModelA"])
        
        self.assertEqual(counts, {"ModelA": 5})
        self.mock_model_registry.get_schema_for_models.assert_called_with(["ModelA"])
        self.mock_client.generate_and_validate_raw_json_output.assert_called_once()
        mock_create_model.assert_called_once()

    async def test_count_entities_llm_failure(self):
        self.mock_model_registry.get_schema_for_models.return_value = "{}"
        self.mock_client.generate_and_validate_raw_json_output = AsyncMock(side_effect=Exception("LLM Fail"))
        
        with patch('extrai.core.entity_counter.create_model'):
            counts = await self.counter.count_entities(["doc"], ["ModelA"])
        
        self.assertEqual(counts, {})
        self.mock_logger.error.assert_called_once()

    async def test_count_entities_invalid_output(self):
        self.mock_model_registry.get_schema_for_models.return_value = "{}"
        self.mock_client.generate_and_validate_raw_json_output = AsyncMock(return_value="Not a dict")
        
        with patch('extrai.core.entity_counter.create_model'):
             counts = await self.counter.count_entities(["doc"], ["ModelA"])
             
        self.assertEqual(counts, {})
        self.mock_logger.warning.assert_called_once()

if __name__ == "__main__":
    unittest.main()
