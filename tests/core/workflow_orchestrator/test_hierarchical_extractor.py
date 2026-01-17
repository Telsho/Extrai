import unittest
import json
from unittest import mock
from unittest.mock import AsyncMock

from sqlmodel import SQLModel, create_engine, Session as SQLModelSession

from extrai.core.workflow_orchestrator import WorkflowOrchestrator
from extrai.core.errors import LLMInteractionError
from tests.core.helpers.orchestrator_test_models import (
    SimpleModel,
    ParentModel,
    ChildModel,
    GrandChildModel,
)
from tests.core.helpers.mock_llm_clients import (
    MockLLMClientForWorkflow as MockLLMClient,
)


class TestHierarchicalExtractor(unittest.IsolatedAsyncioTestCase):
    @mock.patch("extrai.core.model_registry.SchemaInspector")
    def setUp(self, MockSchemaInspector):
        self.mock_llm_client = MockLLMClient()
        self.engine = create_engine("sqlite:///:memory:")
        SQLModel.metadata.create_all(self.engine)
        self.db_session: SQLModelSession = SQLModelSession(self.engine)

        self.mock_discovered_sqlmodels = [
            ParentModel,
            ChildModel,
            GrandChildModel,
            SimpleModel,
        ]
        self.mock_prompt_llm_schema_str = json.dumps(
            {"schema_for_prompt": "mock_schema"}
        )

        mock_inspector = MockSchemaInspector.return_value
        mock_inspector.discover_sqlmodels_from_root.return_value = self.mock_discovered_sqlmodels
        mock_inspector.generate_llm_schema_from_models.return_value = self.mock_prompt_llm_schema_str

    def tearDown(self):
        self.db_session.close()
        SQLModel.metadata.drop_all(self.engine)

    async def test_synthesize_and_hierarchical_extraction_full_traversal(self):
        orchestrator = WorkflowOrchestrator(
            root_sqlmodel_class=ParentModel,
            llm_client=self.mock_llm_client,
            use_hierarchical_extraction=True,
        )
        orchestrator.model_registry.models = [ParentModel, ChildModel]

        parent_entity = {"_type": "ParentModel", "_temp_id": "p1", "name": "Parent 1"}
        child_entity = {
            "_type": "ChildModel",
            "_temp_id": "c1",
            "name": "Child 1",
            "parent_ref_id": "p1",
        }
        final_entities = [parent_entity, child_entity]

        def mock_extraction_cycle(*args, **kwargs):
            system_prompt = kwargs.get("system_prompt") or args[0]
            if "extract **only** entities of type 'ParentModel'" in system_prompt:
                return [parent_entity]
            if "extract **only** entities of type 'ChildModel'" in system_prompt:
                return [child_entity]
            return []


        orchestrator.model_registry.inspector.discover_sqlmodels_from_root = mock.Mock(
            return_value=[ParentModel, ChildModel]
        )

        with (
            mock.patch.object(
                orchestrator.pipeline.llm_runner,
                "run_extraction_cycle",
                side_effect=mock_extraction_cycle,
            ) as mock_run_cycle,
            mock.patch.object(
                orchestrator.pipeline.context_preparer, "prepare_example", new_callable=AsyncMock
            ) as mock_prepare,
            mock.patch.object(orchestrator.result_processor, "hydrate") as mock_hydrate,
        ):
            mock_prepare.return_value = "mock_example_json"
            mock_hydrate.return_value = [
                ParentModel(name="Test"),
                ChildModel(name="Test"),
            ]

            await orchestrator.synthesize(["Some text"], self.db_session)

            self.assertEqual(mock_run_cycle.call_count, 2)

            call_args = mock_run_cycle.call_args_list[1]
            second_call_system_prompt = call_args.kwargs.get("system_prompt") or call_args.args[0]
            self.assertIn("'ChildModel'", second_call_system_prompt)
            self.assertIn(
                "Parent 1", second_call_system_prompt
            )

            mock_hydrate.assert_called_once()
            hydrator_arg = mock_hydrate.call_args[0][0]
            self.assertCountEqual(hydrator_arg, final_entities)

    async def test_run_single_extraction_cycle_llm_failure(self):
        orchestrator = WorkflowOrchestrator(
            root_sqlmodel_class=SimpleModel,
            llm_client=self.mock_llm_client,
            num_llm_revisions=1,
            use_hierarchical_extraction=True,
        )

        self.mock_llm_client.set_should_raise_exception(ValueError("LLM client failed"))

        with self.assertRaisesRegex(
            LLMInteractionError,
            "An unexpected error occurred during LLM interaction: LLM client failed",
        ):
            await orchestrator.pipeline.llm_runner.run_extraction_cycle("system", "user")

        self.assertEqual(self.mock_llm_client.call_count, 1)


if __name__ == "__main__":
    unittest.main()
