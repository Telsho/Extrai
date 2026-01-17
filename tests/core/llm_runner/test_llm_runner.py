import unittest
from unittest.mock import MagicMock, patch, AsyncMock
from extrai.core.llm_runner import LLMRunner, LLMInteractionError
from extrai.core.base_llm_client import BaseLLMClient
from extrai.core.extraction_config import ExtractionConfig

class TestLLMRunner(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.mock_model_registry = MagicMock()
        self.mock_config = MagicMock(spec=ExtractionConfig)
        self.mock_config.num_llm_revisions = 2
        self.mock_config.consensus_threshold = 0.6
        self.mock_config.conflict_resolver = None
        self.mock_config.max_validation_retries_per_revision = 1
        self.mock_analytics_collector = MagicMock()
        self.mock_logger = MagicMock()
        
        self.mock_client1 = MagicMock(spec=BaseLLMClient)
        self.mock_client2 = MagicMock(spec=BaseLLMClient)
        
        self.runner = LLMRunner(
            self.mock_model_registry,
            [self.mock_client1, self.mock_client2],
            self.mock_config,
            self.mock_analytics_collector,
            self.mock_logger
        )

    def test_init_validation(self):
        with self.assertRaises(ValueError):
            LLMRunner(
                self.mock_model_registry,
                [],
                self.mock_config,
                self.mock_analytics_collector,
                self.mock_logger
            )

    def test_client_rotation(self):
        c1 = self.runner.get_next_client()
        c2 = self.runner.get_next_client()
        c3 = self.runner.get_next_client()
        
        self.assertEqual(c1, self.mock_client1)
        self.assertEqual(c2, self.mock_client2)
        self.assertEqual(c3, self.mock_client1)

    @patch('extrai.core.llm_runner.normalize_json_revisions')
    async def test_run_extraction_cycle_success(self, mock_normalize):
        # Setup mocks
        self.mock_client1.generate_json_revisions = AsyncMock(return_value=[{"id": 1}])
        self.mock_client2.generate_json_revisions = AsyncMock(return_value=[{"id": 1}])
        
        mock_normalize.return_value = [{"id": 1}, {"id": 1}]
        
        # Mock consensus
        with patch.object(self.runner, 'consensus') as mock_consensus:
            mock_consensus.get_consensus.return_value = ([{"id": 1}], {})
            
            results = await self.runner.run_extraction_cycle("sys", "user")
            
            self.assertEqual(len(results), 1)
            self.assertEqual(results[0], {"id": 1})
            
            # Verify calls
            self.assertEqual(self.mock_client1.generate_json_revisions.call_count, 1)
            self.assertEqual(self.mock_client2.generate_json_revisions.call_count, 1)
            mock_normalize.assert_called_once()
            mock_consensus.get_consensus.assert_called_once()

    async def test_run_extraction_cycle_llm_failure(self):
        self.mock_client1.generate_json_revisions = AsyncMock(side_effect=Exception("API Error"))
        self.mock_client2.generate_json_revisions = AsyncMock(return_value=[])
        
        # The runner uses asyncio.gather without return_exceptions=True (default is False), 
        # so it propagates the exception.
        
        with self.assertRaises(LLMInteractionError):
            await self.runner.run_extraction_cycle("sys", "user")

    def test_process_consensus_output(self):
        # List
        res = self.runner._process_consensus_output([{"a": 1}])
        self.assertEqual(res, [{"a": 1}])
        
        # None
        res = self.runner._process_consensus_output(None)
        self.assertEqual(res, [])
        
        # Dict
        res = self.runner._process_consensus_output({"a": 1})
        self.assertEqual(res, [{"a": 1}])
        
        # Dict with results
        res = self.runner._process_consensus_output({"results": [{"a": 1}]})
        self.assertEqual(res, [{"a": 1}])

    def test_get_client_count(self):
        self.assertEqual(self.runner.get_client_count(), 2)

    def test_reset_client_rotation(self):
        # Advance client index
        self.runner.get_next_client()
        self.assertEqual(self.runner.client_index, 1)
        
        self.runner.reset_client_rotation()
        self.assertEqual(self.runner.client_index, 0)
        
        self.mock_logger.debug.assert_called_with("Client rotation reset to index 0")

    def test_repr(self):
        repr_str = repr(self.runner)
        self.assertIn("LLMRunner", repr_str)
        self.assertIn("clients=2", repr_str)
        self.assertIn("revisions=2", repr_str)

if __name__ == "__main__":
    unittest.main()
