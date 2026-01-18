import unittest
import json
from unittest import mock
from unittest.mock import AsyncMock

from sqlmodel import SQLModel, create_engine, Session as SQLModelSession

from extrai.core.workflow_orchestrator import WorkflowOrchestrator



from tests.core.helpers.orchestrator_test_models import DepartmentModel, EmployeeModel
from tests.core.helpers.mock_llm_clients import (
    MockLLMClientForWorkflow as MockLLMClient,
)


class TestWorkflowOrchestratorExecution(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.mock_llm_client1 = MockLLMClient()
        self.mock_llm_client2 = MockLLMClient()

        self.discovered_sqlmodels_for_execution = [DepartmentModel, EmployeeModel]
        self.prompt_llm_schema_for_execution = json.dumps(
            {"schema_for_prompt": "mock_llm_prompt_schema"}
        )

        self.patcher_inspector = mock.patch(
            "extrai.core.model_registry.SchemaInspector"
        )
        self.MockSchemaInspector = self.patcher_inspector.start()
        mock_inspector_instance = self.MockSchemaInspector.return_value
        mock_inspector_instance.discover_sqlmodels_from_root.return_value = (
            self.discovered_sqlmodels_for_execution
        )
        mock_inspector_instance.generate_llm_schema_from_models.return_value = (
            self.prompt_llm_schema_for_execution
        )

        self.orchestrator = WorkflowOrchestrator(
            root_sqlmodel_class=DepartmentModel,
            llm_client=[self.mock_llm_client1, self.mock_llm_client2],
            num_llm_revisions=2,
            max_validation_retries_per_revision=1,
        )

        self.engine = create_engine("sqlite:///:memory:")
        SQLModel.metadata.create_all(self.engine)
        self.db_session: SQLModelSession = SQLModelSession(self.engine)

    def tearDown(self):
        self.patcher_inspector.stop()
        self.db_session.close()
        SQLModel.metadata.drop_all(self.engine)

    async def test_successful_synthesis_clear_consensus(self):
        dept_rev_content = {
            "_type": "DepartmentModel",
            "_temp_id": "dept1",
            "name": "Engineering",
        }
        emp_rev_content = {
            "_type": "EmployeeModel",
            "_temp_id": "emp1",
            "name": "Jane Doe",
            "department_ref_id": "dept1",
        }

        revision_content = [dept_rev_content, emp_rev_content]

        llm_output_to_unwrap = {"results": revision_content}
        self.mock_llm_client1.set_revisions_to_return([llm_output_to_unwrap])
        self.mock_llm_client2.set_revisions_to_return([llm_output_to_unwrap])

        from extrai.utils.flattening_utils import flatten_json

        example_flat_revision = flatten_json(revision_content)
        num_unique_paths = len(example_flat_revision)

        mock_consensus_output = revision_content
        mock_analytics_for_clear_consensus = {
            "revisions_processed": 2,
            "unique_paths_considered": num_unique_paths,
            "paths_agreed_by_threshold": num_unique_paths,
            "paths_resolved_by_conflict_resolver": 0,
            "paths_omitted_due_to_no_consensus_or_resolver_omission": 0,
        }
        expected_consensus_input = [revision_content] * 2

        with mock.patch.object(
            self.orchestrator.pipeline.llm_runner.consensus,
            "get_consensus",
            return_value=(mock_consensus_output, mock_analytics_for_clear_consensus),
        ) as mock_get_consensus_call:
            mock_get_consensus_call.assert_called_once_with(expected_consensus_input)

        self.assertEqual(self.mock_llm_client1.call_count, 1)
        self.assertEqual(self.mock_llm_client2.call_count, 1)

    async def test_synthesize_with_user_provided_example_json(self):
        user_example_json = json.dumps(
            {"_type": "DepartmentModel", "name": "HR", "_temp_id": "hr_dept_example"}
        )
        input_strings = ["Some HR related text"]

        llm_main_extraction_response = [
            {
                "results": [
                    {
                        "_type": "DepartmentModel",
                        "_temp_id": "dept_hr_actual",
                        "name": "Human Resources",
                    }
                ]
            }
        ]
        self.mock_llm_client1.set_revisions_to_return(llm_main_extraction_response)
        self.mock_llm_client2.set_revisions_to_return(llm_main_extraction_response)

        with mock.patch.object(
            self.orchestrator.pipeline.context_preparer,
            "prepare_example",
            new_callable=AsyncMock,
        ) as mock_prepare:
            mock_prepare.return_value = user_example_json
            await self.orchestrator.synthesize(
                input_strings,
                self.db_session,
                extraction_example_json=user_example_json,
            )
            mock_prepare.assert_called_once()


if __name__ == "__main__":
    unittest.main()
