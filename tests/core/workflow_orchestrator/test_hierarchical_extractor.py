# tests/core/workflow_orchestrator/test_hierarchical_extractor.py

import unittest
import json
from unittest import mock
from unittest.mock import AsyncMock

from sqlmodel import SQLModel, create_engine, Session as SQLModelSession

from extrai.core.workflow_orchestrator import WorkflowOrchestrator
from extrai.core.errors import WorkflowError
from extrai.core.example_json_generator import ExampleGenerationError
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
    @mock.patch("extrai.core.workflow_orchestrator.discover_sqlmodels_from_root")
    @mock.patch("extrai.core.workflow_orchestrator.generate_llm_schema_from_models")
    def setUp(self, mock_generate_llm_schema, mock_discover_sqlmodels):
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

        mock_discover_sqlmodels.return_value = self.mock_discovered_sqlmodels
        mock_generate_llm_schema.return_value = self.mock_prompt_llm_schema_str

    def tearDown(self):
        self.db_session.close()
        SQLModel.metadata.drop_all(self.engine)

    @mock.patch("extrai.core.workflow_orchestrator.ExampleJSONGenerator")
    async def test_prepare_extraction_example_auto_generation_failure(
        self, mocked_example_generator
    ):
        # This test covers the error handling in `_prepare_extraction_example` (line 282)
        mock_generator_instance = mocked_example_generator.return_value
        mock_generator_instance.generate_example = AsyncMock(
            side_effect=ExampleGenerationError("Generation failed")
        )

        orchestrator = WorkflowOrchestrator(
            root_sqlmodel_class=SimpleModel,
            llm_client=self.mock_llm_client,
            use_hierarchical_extraction=True,
        )

        with self.assertRaisesRegex(
            WorkflowError,
            "Failed to auto-generate extraction example: Generation failed",
        ):
            await orchestrator.synthesize(["Some input text"], self.db_session)

        mocked_example_generator.assert_called_once()
        mock_generator_instance.generate_example.assert_called_once()

    @mock.patch("builtins.print")
    async def test_hierarchical_extraction_handles_missing_temp_id_and_duplicates(
        self, mock_print
    ):
        # This test covers lines 389-395 (missing temp_id and duplicate entities)
        orchestrator = WorkflowOrchestrator(
            root_sqlmodel_class=ParentModel,
            llm_client=self.mock_llm_client,
            use_hierarchical_extraction=True,
        )

        # Entity with a temp_id, one without, and a duplicate
        entity1 = {"_type": "ParentModel", "_temp_id": "p1", "name": "Parent 1"}
        entity_no_id = {"_type": "ParentModel", "name": "Parent No ID"}
        entity_duplicate = {
            "_type": "ParentModel",
            "_temp_id": "p1",
            "name": "Duplicate",
        }

        def mock_extraction_cycle(*args, **kwargs):
            return [entity1, entity_no_id, entity_duplicate]

        with (
            mock.patch(
                "extrai.core.workflow_orchestrator.discover_sqlmodels_from_root",
                return_value=[ParentModel],
            ),
            mock.patch.object(
                orchestrator,
                "_run_single_extraction_cycle",
                side_effect=mock_extraction_cycle,
            ),
        ):
            final_list = await orchestrator._execute_hierarchical_extraction(
                input_strings=["Some text"],
                current_extraction_example_json="",
                custom_extraction_process="",
                custom_extraction_guidelines="",
                custom_final_checklist="",
            )

            # Only the first valid entity should be in the final list
            self.assertEqual(len(final_list), 1)
            self.assertIn(entity1, final_list)
            self.assertNotIn(entity_no_id, final_list)
            # Ensure the duplicate did not overwrite the original
            self.assertEqual(final_list[0]["name"], "Parent 1")

    async def test_synthesize_and_hierarchical_extraction_full_traversal(self):
        # This test covers the main logic of `_execute_hierarchical_extraction` (lines 342-395)
        # and ensures it's called correctly by `synthesize`.
        orchestrator = WorkflowOrchestrator(
            root_sqlmodel_class=ParentModel,
            llm_client=self.mock_llm_client,
            use_hierarchical_extraction=True,
        )

        parent_entity = {"_type": "ParentModel", "_temp_id": "p1", "name": "Parent 1"}
        child_entity = {
            "_type": "ChildModel",
            "_temp_id": "c1",
            "name": "Child 1",
            "parent_ref_id": "p1",
        }
        final_entities = [parent_entity, child_entity]

        # Mock the extraction cycle to simulate LLM responses for each model type
        def mock_extraction_cycle(*args, **kwargs):
            system_prompt = args[0]
            if "extract **only** entities of type 'ParentModel'" in system_prompt:
                return [parent_entity]
            if "extract **only** entities of type 'ChildModel'" in system_prompt:
                return [child_entity]
            return []

        # We need to control the order of models processed in the loop
        with (
            mock.patch(
                "extrai.core.workflow_orchestrator.discover_sqlmodels_from_root",
                return_value=[ParentModel, ChildModel],
            ),
            mock.patch.object(
                orchestrator,
                "_run_single_extraction_cycle",
                side_effect=mock_extraction_cycle,
            ) as mock_run_cycle,
            mock.patch.object(
                orchestrator, "_prepare_extraction_example", new_callable=AsyncMock
            ) as mock_prepare,
            mock.patch.object(orchestrator, "_hydrate_results") as mock_hydrate,
        ):
            mock_prepare.return_value = "mock_example_json"
            mock_hydrate.return_value = [
                ParentModel(name="Test"),
                ChildModel(name="Test"),
            ]

            await orchestrator.synthesize(["Some text"], self.db_session)

            # Verify that the extraction cycle was called for each model
            self.assertEqual(mock_run_cycle.call_count, 2)

            # Check that the context for the second call (ChildModel) contains the first entity
            second_call_system_prompt = mock_run_cycle.call_args_list[1].args[0]
            self.assertIn("'ChildModel'", second_call_system_prompt)
            self.assertIn(
                json.dumps([parent_entity], indent=2), second_call_system_prompt
            )

            # Verify that the final list was passed to the hydrator
            mock_hydrate.assert_called_once()
            # The order might not be guaranteed, so we check for content equivalence
            hydrator_arg = mock_hydrate.call_args[0][0]
            self.assertCountEqual(hydrator_arg, final_entities)

    async def test_run_single_extraction_cycle_success(self):
        # This test covers a successful run of `_run_single_extraction_cycle` (lines 408-447)
        orchestrator = WorkflowOrchestrator(
            root_sqlmodel_class=SimpleModel,
            llm_client=self.mock_llm_client,
            num_llm_revisions=2,
            use_hierarchical_extraction=True,
        )

        llm_revision = {"results": [{"_type": "SimpleModel", "name": "Test"}]}
        self.mock_llm_client.set_revisions_to_return([llm_revision] * 2)

        consensus_output = [{"_type": "SimpleModel", "name": "Consensus Result"}]
        mock_consensus_details = {"revisions_processed": 2}

        with mock.patch.object(
            orchestrator.json_consensus,
            "get_consensus",
            return_value=(consensus_output, mock_consensus_details),
        ) as mock_get_consensus:
            result = await orchestrator._run_single_extraction_cycle("system", "user")

            self.assertEqual(self.mock_llm_client.call_count, 2)
            mock_get_consensus.assert_called_once()
            self.assertEqual(result, consensus_output)

    async def test_run_single_extraction_cycle_llm_failure(self):
        # This test covers error handling in `_run_single_extraction_cycle`
        orchestrator = WorkflowOrchestrator(
            root_sqlmodel_class=SimpleModel,
            llm_client=self.mock_llm_client,
            num_llm_revisions=1,
            use_hierarchical_extraction=True,
        )

        self.mock_llm_client.set_should_raise_exception(ValueError("LLM client failed"))

        with self.assertRaisesRegex(
            Exception,
            "An unexpected error occurred during LLM interaction: LLM client failed",
        ):
            await orchestrator._run_single_extraction_cycle("system", "user")

        self.assertEqual(self.mock_llm_client.call_count, 1)


if __name__ == "__main__":
    unittest.main()
