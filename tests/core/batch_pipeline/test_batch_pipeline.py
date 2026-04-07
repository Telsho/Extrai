import unittest
from unittest.mock import MagicMock, patch, AsyncMock
from extrai.core.batch.batch_pipeline import BatchPipeline
from extrai.core.model_registry import ModelRegistry
from extrai.core.extraction_config import ExtractionConfig
from extrai.core.base_llm_client import BaseLLMClient
from extrai.core.batch_models import BatchJobContext, BatchJobStatus


class TestBatchPipeline(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.mock_model_registry = MagicMock(spec=ModelRegistry)
        self.mock_model_registry.root_model = MagicMock()
        self.mock_model_registry.root_model.__name__ = "RootModel"
        self.mock_model_registry.models = [self.mock_model_registry.root_model]
        self.mock_model_registry.llm_schema_json = "{}"
        self.mock_model_registry.model_map = {
            "RootModel": self.mock_model_registry.root_model
        }

        self.mock_client = MagicMock(spec=BaseLLMClient)
        self.mock_client.temperature = 0.0
        self.mock_config = MagicMock(spec=ExtractionConfig)
        self.mock_config.consensus_threshold = 0.6
        self.mock_config.conflict_resolver = None
        self.mock_config.num_llm_revisions = 1
        self.mock_config.max_validation_retries_per_revision = 1
        self.mock_config.use_structured_output = False
        self.mock_config.use_hierarchical_extraction = False
        self.mock_analytics = MagicMock()
        self.mock_session = MagicMock()
        self.mock_logger = MagicMock()

        with (
            patch(
                "extrai.core.batch.batch_pipeline.ClientRotator"
            ) as MockClientRotator,
            patch(
                "extrai.core.batch.batch_pipeline.ExtractionContextPreparer"
            ) as MockContextPreparer,
            patch("extrai.core.batch.batch_pipeline.PromptBuilder") as MockBuilder,
            patch("extrai.core.batch.batch_pipeline.EntityCounter") as MockCounter,
            patch("extrai.core.batch.batch_pipeline.ConsensusRunner") as MockConsensus,
            patch("extrai.core.batch.batch_pipeline.ModelWrapperBuilder"),
            patch("extrai.core.batch.batch_pipeline.BatchStatusChecker"),
            patch("extrai.core.batch.batch_pipeline.BatchResultRetriever"),
            patch("extrai.core.batch.batch_pipeline.BatchProcessor"),
        ):
            self.pipeline = BatchPipeline(
                self.mock_model_registry,
                self.mock_client,
                self.mock_config,
                self.mock_analytics,
                self.mock_logger,
            )
            self.pipeline.client_rotator = MockClientRotator.return_value
            self.pipeline.context_preparer = MockContextPreparer.return_value
            self.pipeline.prompt_builder = MockBuilder.return_value
            self.pipeline.entity_counter = MockCounter.return_value
            self.pipeline.consensus_runner = MockConsensus.return_value

    async def test_submit_batch_success(self):
        # We instantiate a real submitter to test the logic
        from extrai.core.batch.batch_submitter import BatchSubmitter

        self.pipeline.submitter = BatchSubmitter(
            self.mock_model_registry,
            self.pipeline.client_rotator,
            self.mock_config,
            self.pipeline.entity_counter,
            self.pipeline.context_preparer,
            self.pipeline.request_factory,
            self.mock_logger,
        )

        self.pipeline.prompt_builder.build_prompts.return_value = ("sys", "user")

        mock_batch_job = MagicMock()
        mock_batch_job.id = "provider_id_123"

        mock_client_instance = self.pipeline.client_rotator.get_next_client.return_value
        mock_client_instance.create_batch_job = AsyncMock(return_value=mock_batch_job)

        self.pipeline.context_preparer.prepare_example = AsyncMock(return_value="")

        root_id = await self.pipeline.submit_batch(self.mock_session, ["doc"])

        self.assertIsInstance(root_id, str)
        self.assertEqual(self.mock_session.add.call_count, 2)
        self.assertEqual(self.mock_session.commit.call_count, 2)

        added_context = self.mock_session.add.call_args[0][0]
        self.assertIsInstance(added_context, BatchJobContext)
        self.assertEqual(added_context.current_batch_id, "provider_id_123")
        self.assertEqual(added_context.status, BatchJobStatus.SUBMITTED)

    async def test_retrieve_and_validate_results(self):
        # Use real retriever
        from extrai.core.batch.batch_result_retriever import BatchResultRetriever

        self.pipeline.retriever = BatchResultRetriever(
            self.mock_model_registry, self.mock_logger
        )

        mock_context = BatchJobContext(current_batch_id="prov_1")

        mock_client = self.pipeline.client_rotator.get_next_client.return_value
        mock_client.retrieve_batch_results = AsyncMock(return_value=["res1", "res2"])

        with patch(
            "extrai.core.batch.batch_result_retriever.process_and_validate_llm_output"
        ) as mock_validate:
            mock_validate.side_effect = [[{"id": 1}], [{"id": 2}]]
            results, _ = await self.pipeline.retriever.retrieve_and_validate_results(
                mock_context, mock_client
            )
            self.assertEqual(len(results), 2)

    async def test_monitor_batch_job_success(self):
        root_batch_id = "root_123"
        db_session = MagicMock()
        from extrai.core.batch_models import BatchProcessResult

        # 1. READY_TO_PROCESS
        self.pipeline.status_checker.get_status = AsyncMock(
            return_value=BatchJobStatus.READY_TO_PROCESS
        )

        process_result = BatchProcessResult(
            status=BatchJobStatus.COMPLETED, hydrated_objects=["obj1"]
        )

        # Mock processor.process_batch
        self.pipeline.processor.process_batch = AsyncMock(return_value=process_result)

        # Ensure processor has result_processor
        self.pipeline.processor.result_processor = MagicMock()

        result = await self.pipeline.monitor_batch_job(
            root_batch_id, db_session, poll_interval=0.001
        )

        self.assertEqual(result.status, BatchJobStatus.COMPLETED)
        self.pipeline.processor.result_processor.persist.assert_called_once_with(
            ["obj1"], db_session
        )


if __name__ == "__main__":
    unittest.main()
