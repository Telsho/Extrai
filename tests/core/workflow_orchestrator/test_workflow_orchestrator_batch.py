import unittest
from unittest import mock
import logging
from sqlmodel import Session
from extrai.core.workflow_orchestrator import WorkflowOrchestrator
from extrai.core.batch_models import BatchJobStatus, BatchProcessResult
from tests.core.helpers.orchestrator_test_models import DepartmentModel
from tests.core.helpers.mock_llm_clients import (
    MockLLMClientForWorkflow as MockLLMClient,
)


class TestWorkflowOrchestratorBatch(unittest.IsolatedAsyncioTestCase):
    @mock.patch("extrai.core.workflow_orchestrator.ModelRegistry")
    def setUp(self, MockModelRegistry):
        self.mock_llm_client = MockLLMClient()
        self.root_sqlmodel_class = DepartmentModel

        # Create orchestrator with mocks
        self.orchestrator = WorkflowOrchestrator(
            root_sqlmodel_class=self.root_sqlmodel_class,
            llm_client=self.mock_llm_client,
        )

        # Mock the internal components that we want to verify delegation to
        self.orchestrator.batch_pipeline = mock.AsyncMock()
        self.orchestrator.result_processor = mock.Mock()
        self.orchestrator.logger = mock.Mock(spec=logging.Logger)

    async def test_synthesize_batch(self):
        input_strings = ["input1", "input2"]
        db_session = mock.Mock(spec=Session)
        expected_batch_id = "batch_123"
        self.orchestrator.batch_pipeline.submit_batch.return_value = expected_batch_id

        result = await self.orchestrator.synthesize_batch(input_strings, db_session)

        # Verify call with default args
        self.orchestrator.batch_pipeline.submit_batch.assert_called_once()
        call_kwargs = self.orchestrator.batch_pipeline.submit_batch.call_args[1]
        self.assertEqual(call_kwargs["input_strings"], input_strings)
        self.assertEqual(call_kwargs["db_session"], db_session)
        self.assertEqual(result, expected_batch_id)

    async def test_get_batch_status(self):
        batch_id = "batch_123"
        db_session = mock.Mock(spec=Session)
        expected_status = BatchJobStatus.COMPLETED
        self.orchestrator.batch_pipeline.get_status.return_value = expected_status

        result = await self.orchestrator.get_batch_status(batch_id, db_session)

        self.orchestrator.batch_pipeline.get_status.assert_called_once_with(
            batch_id, db_session
        )
        self.assertEqual(result, expected_status)

    async def test_process_batch_success(self):
        batch_id = "batch_123"
        db_session = mock.Mock(spec=Session)
        hydrated_objects = ["obj1"]
        expected_result = BatchProcessResult(
            status=BatchJobStatus.COMPLETED, hydrated_objects=hydrated_objects
        )
        self.orchestrator.batch_pipeline.process_batch.return_value = expected_result

        result = await self.orchestrator.process_batch(batch_id, db_session)

        self.orchestrator.batch_pipeline.process_batch.assert_called_once_with(
            batch_id, db_session
        )
        self.orchestrator.result_processor.persist.assert_called_once_with(
            hydrated_objects, db_session
        )
        self.assertEqual(result, expected_result)

    async def test_process_batch_not_completed(self):
        batch_id = "batch_123"
        db_session = mock.Mock(spec=Session)
        expected_result = BatchProcessResult(status=BatchJobStatus.PROCESSING)
        self.orchestrator.batch_pipeline.process_batch.return_value = expected_result

        result = await self.orchestrator.process_batch(batch_id, db_session)

        self.orchestrator.batch_pipeline.process_batch.assert_called_once_with(
            batch_id, db_session
        )
        self.orchestrator.result_processor.persist.assert_not_called()
        self.assertEqual(result, expected_result)

    async def test_process_batch_persistence_failure(self):
        batch_id = "batch_123"
        db_session = mock.Mock(spec=Session)
        hydrated_objects = ["obj1"]
        process_result = BatchProcessResult(
            status=BatchJobStatus.COMPLETED, hydrated_objects=hydrated_objects
        )

        self.orchestrator.batch_pipeline.process_batch.return_value = process_result
        self.orchestrator.result_processor.persist.side_effect = Exception(
            "Persistence Error"
        )

        with self.assertRaisesRegex(Exception, "Persistence Error"):
            await self.orchestrator.process_batch(batch_id, db_session)

        self.orchestrator.logger.error.assert_called()
        self.assertIn(
            "Extraction successful but persistence failed", process_result.message
        )

    async def test_monitor_batch_job_delegation(self):
        batch_id = "batch_123"
        db_session = mock.Mock(spec=Session)
        poll_interval = 5

        expected_result = BatchProcessResult(
            status=BatchJobStatus.COMPLETED, hydrated_objects=["obj1"]
        )
        self.orchestrator.batch_pipeline.monitor_batch_job.return_value = expected_result

        result = await self.orchestrator.monitor_batch_job(
            batch_id, db_session, poll_interval
        )

        self.orchestrator.batch_pipeline.monitor_batch_job.assert_called_once_with(
            batch_id, db_session, poll_interval
        )
        self.assertEqual(result, expected_result)


if __name__ == "__main__":
    unittest.main()
