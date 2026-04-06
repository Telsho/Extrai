import unittest
from unittest.mock import MagicMock, patch, AsyncMock
from extrai.core.entity_counter import EntityCounter, EntityCountResult
from extrai.core.model_registry import ModelRegistry
from extrai.core.extraction_config import ExtractionConfig
from extrai.core.base_llm_client import BaseLLMClient


class TestEntityCounter(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.mock_model_registry = MagicMock(spec=ModelRegistry)
        self.mock_client = MagicMock(spec=BaseLLMClient)
        self.mock_config = MagicMock(spec=ExtractionConfig)
        self.mock_config.max_validation_retries_per_revision = 1
        self.mock_config.num_counting_revisions = 1
        self.mock_config.use_structured_output = False
        self.mock_analytics = MagicMock()
        self.mock_logger = MagicMock()

        self.counter = EntityCounter(
            self.mock_model_registry,
            self.mock_client,
            self.mock_config,
            self.mock_analytics,
            self.mock_logger,
        )
        self.counter.counting_consensus.achieve_consensus = AsyncMock()

    @patch("extrai.core.entity_counter.generate_entity_counting_system_prompt")
    @patch("extrai.core.entity_counter.generate_entity_counting_user_prompt")
    async def test_count_entities_success(self, mock_user_prompt, mock_system_prompt):
        # Setup mocks
        self.mock_model_registry.get_schema_for_models.return_value = (
            '{"type": "object"}'
        )
        mock_result = [
            {
                "counted_entities": [
                    {"model": "ModelA", "temp_id": "1", "description": "desc"}
                ]
            }
        ]
        self.mock_client.generate_and_validate_raw_json_output = AsyncMock(
            return_value=mock_result
        )

        expected_consensus = [
            {"model": "ModelA", "temp_id": "1", "description": "desc"}
        ]
        self.counter.counting_consensus.achieve_consensus.return_value = (
            expected_consensus
        )

        counts = await self.counter.count_entities(["doc"], ["ModelA"])

        self.assertEqual(counts, expected_consensus)
        self.mock_model_registry.get_schema_for_models.assert_called_with(["ModelA"])
        self.mock_client.generate_and_validate_raw_json_output.assert_called_once()
        self.counter.counting_consensus.achieve_consensus.assert_called_once()

    async def test_count_entities_llm_failure(self):
        self.mock_model_registry.get_schema_for_models.return_value = "{}"
        self.mock_client.generate_and_validate_raw_json_output = AsyncMock(
            side_effect=Exception("LLM Fail")
        )

        counts = await self.counter.count_entities(["doc"], ["ModelA"])

        self.assertEqual(counts, [])
        self.mock_logger.error.assert_called_once()

    async def test_count_entities_invalid_output(self):
        self.mock_model_registry.get_schema_for_models.return_value = "{}"
        self.mock_client.generate_and_validate_raw_json_output = AsyncMock(
            return_value="Not a dict or list"
        )

        counts = await self.counter.count_entities(["doc"], ["ModelA"])

        self.assertEqual(counts, [])
        self.mock_logger.warning.assert_called_once()


if __name__ == "__main__":
    unittest.main()
