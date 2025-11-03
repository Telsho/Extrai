# tests/core/test_workflow_orchestrator_execution.py

import unittest
import json
from unittest import mock
from unittest.mock import AsyncMock

from sqlmodel import SQLModel, create_engine, Session as SQLModelSession

from extrai.core.errors import (
    LLMAPICallError,
    LLMOutputParseError,
    LLMOutputValidationError,
    WorkflowError,
    ConsensusProcessError,
    HydrationError,
)
from extrai.core.workflow_orchestrator import (
    WorkflowOrchestrator,
    LLMInteractionError,
)
from extrai.core.analytics_collector import (
    WorkflowAnalyticsCollector,
)
from extrai.core.db_writer import DatabaseWriterError

from extrai.core.example_json_generator import (
    ExampleGenerationError,
)


from tests.core.helpers.orchestrator_test_models import DepartmentModel, EmployeeModel
from tests.core.helpers.mock_llm_clients import (
    MockLLMClientForWorkflow as MockLLMClient,
)


class TestWorkflowOrchestratorExecution(unittest.IsolatedAsyncioTestCase):
    @mock.patch("extrai.core.workflow_orchestrator.discover_sqlmodels_from_root")
    @mock.patch("extrai.core.workflow_orchestrator.generate_llm_schema_from_models")
    def setUp(self, mock_generate_llm_schema, mock_discover_sqlmodels):
        self.mock_llm_client1 = MockLLMClient()
        self.mock_llm_client2 = MockLLMClient()

        self.discovered_sqlmodels_for_execution = [DepartmentModel, EmployeeModel]
        self.prompt_llm_schema_for_execution = json.dumps(
            {"schema_for_prompt": "mock_llm_prompt_schema"}
        )

        mock_discover_sqlmodels.return_value = self.discovered_sqlmodels_for_execution
        mock_generate_llm_schema.return_value = self.prompt_llm_schema_for_execution

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

        # The content of a single, successful revision is a list of entities.
        revision_content = [dept_rev_content, emp_rev_content]

        # The raw output from the LLM is a dict that needs to be unwrapped by the client.
        # The orchestrator will call each client `num_llm_revisions` times in total.
        # With 2 clients and num_llm_revisions=2, each client is called once.
        llm_output_to_unwrap = {"results": revision_content}
        self.mock_llm_client1.set_revisions_to_return([llm_output_to_unwrap])
        self.mock_llm_client2.set_revisions_to_return([llm_output_to_unwrap])

        input_strings = ["Some text about Jane Doe in Engineering."]

        from extrai.utils.flattening_utils import flatten_json

        example_flat_revision = flatten_json(revision_content)
        num_unique_paths = len(example_flat_revision)

        # The output of the consensus process is now a flat list of entities.
        mock_consensus_output = revision_content
        mock_analytics_for_clear_consensus = {
            "revisions_processed": 2,  # 2 clients * 1 revision each = 2
            "unique_paths_considered": num_unique_paths,
            "paths_agreed_by_threshold": num_unique_paths,
            "paths_resolved_by_conflict_resolver": 0,
            "paths_omitted_due_to_no_consensus_or_resolver_omission": 0,
        }

        # The input to the consensus function is a list of revisions.
        # With 2 clients and num_llm_revisions=2, we expect 2 revisions in total.
        expected_consensus_input = [revision_content] * 2

        with mock.patch.object(
            self.orchestrator.json_consensus,
            "get_consensus",
            return_value=(mock_consensus_output, mock_analytics_for_clear_consensus),
        ) as mock_get_consensus_call:
            hydrated_objects = await self.orchestrator.synthesize(
                input_strings, self.db_session
            )
            mock_get_consensus_call.assert_called_once_with(expected_consensus_input)

        self.assertEqual(self.mock_llm_client1.call_count, 1)
        self.assertEqual(self.mock_llm_client2.call_count, 1)
        self.assertFalse(hasattr(self.mock_llm_client1, "last_formal_json_schema_str"))
        self.assertEqual(self.mock_llm_client1.last_max_validation_retries, 1)
        self.assertIn(
            "mock_llm_prompt_schema", self.mock_llm_client1.last_system_prompt
        )
        self.assertIn(
            "mock_llm_prompt_schema", self.mock_llm_client2.last_system_prompt
        )

        self.assertIsInstance(hydrated_objects, list)
        self.assertEqual(len(hydrated_objects), 2)
        self.assertIsInstance(hydrated_objects[0], SQLModel)
        self.assertIsInstance(hydrated_objects[1], SQLModel)

        self.assertIs(
            self.mock_llm_client1.last_analytics_collector_passed,
            self.orchestrator.analytics_collector,
        )
        self.assertIs(
            self.mock_llm_client2.last_analytics_collector_passed,
            self.orchestrator.analytics_collector,
        )
        report = self.orchestrator.get_analytics_report()
        self.assertEqual(report["llm_api_call_failures"], 0)
        self.assertEqual(report["total_invalid_parsing_errors"], 0)
        self.assertEqual(report["number_of_consensus_runs"], 1)

        self.assertIn("all_consensus_run_details", report)
        self.assertEqual(len(report["all_consensus_run_details"]), 1)
        run_details = report["all_consensus_run_details"][0]
        self.assertEqual(run_details["revisions_processed"], 2)
        self.assertGreater(run_details["unique_paths_considered"], 0)
        self.assertGreaterEqual(run_details["paths_agreed_by_threshold"], 0)
        if run_details["unique_paths_considered"] > 0:
            self.assertAlmostEqual(
                report["average_path_agreement_ratio"],
                run_details["paths_agreed_by_threshold"]
                / run_details["unique_paths_considered"],
            )

    async def test_successful_synthesis_without_db_session(self):
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

        input_strings = ["Some text about Jane Doe in Engineering."]

        mock_consensus_output = revision_content

        with mock.patch.object(
            self.orchestrator.json_consensus,
            "get_consensus",
            return_value=(mock_consensus_output, {}),
        ) as mock_get_consensus_call:
            hydrated_objects = await self.orchestrator.synthesize(
                input_strings, db_session_for_hydration=None
            )
            expected_consensus_input = [revision_content] * 2
            mock_get_consensus_call.assert_called_once_with(expected_consensus_input)

        self.assertEqual(len(hydrated_objects), 2)
        self.assertIsInstance(hydrated_objects[0], DepartmentModel)
        self.assertIsInstance(hydrated_objects[1], EmployeeModel)

        # Verify that the objects are not attached to any session
        from sqlalchemy.orm import object_session

        self.assertIsNone(object_session(hydrated_objects[0]))
        self.assertIsNone(object_session(hydrated_objects[1]))

    async def test_synthesis_llm_client_raises_parse_error(self):
        self.mock_llm_client1.set_should_raise_exception(
            LLMOutputParseError("Parsing failed", raw_content="bad json")
        )
        self.mock_llm_client1.set_should_raise_exception_for_example_gen(None)

        with self.assertRaisesRegex(
            LLMInteractionError, "LLM client operation failed: Parsing failed"
        ):
            await self.orchestrator.synthesize(["Some text"], self.db_session)

        self.assertEqual(self.mock_llm_client1.example_gen_call_count, 1)
        self.assertEqual(self.mock_llm_client1.call_count, 1)

        report = self.orchestrator.get_analytics_report()
        self.assertEqual(report["llm_output_parse_errors"], 1)
        self.assertEqual(report["total_invalid_parsing_errors"], 1)
        self.assertEqual(report["number_of_consensus_runs"], 0)

    async def test_synthesis_llm_client_raises_validation_error(self):
        self.mock_llm_client1.set_should_raise_exception(
            LLMOutputValidationError(
                "Validation failed", parsed_json={"key": "wrong_type"}
            )
        )
        self.mock_llm_client1.set_should_raise_exception_for_example_gen(None)

        with self.assertRaisesRegex(
            LLMInteractionError, "LLM client operation failed: Validation failed"
        ):
            await self.orchestrator.synthesize(["Some text"], self.db_session)

        self.assertEqual(self.mock_llm_client1.example_gen_call_count, 1)
        self.assertEqual(self.mock_llm_client1.call_count, 1)

        report = self.orchestrator.get_analytics_report()
        self.assertEqual(report["llm_output_validation_errors"], 1)
        self.assertEqual(report["total_invalid_parsing_errors"], 1)
        self.assertEqual(report["number_of_consensus_runs"], 0)

    async def test_synthesis_llm_client_raises_api_call_error(self):
        self.mock_llm_client1.set_should_raise_exception(
            LLMAPICallError("API unavailable")
        )
        self.mock_llm_client1.set_should_raise_exception_for_example_gen(None)

        with self.assertRaisesRegex(
            LLMInteractionError, "LLM client operation failed: API unavailable"
        ):
            await self.orchestrator.synthesize(["Some text"], self.db_session)

        self.assertEqual(self.mock_llm_client1.example_gen_call_count, 1)
        self.assertEqual(self.mock_llm_client1.call_count, 1)

        report = self.orchestrator.get_analytics_report()
        self.assertEqual(report["llm_api_call_failures"], 1)
        self.assertEqual(report["total_invalid_parsing_errors"], 0)
        self.assertEqual(report["number_of_consensus_runs"], 0)

    async def test_synthesis_no_consensus_reached_analytics(self):
        llm_revisions = [
            {
                "results": [
                    {
                        "_type": "DepartmentModel",
                        "name": "Engineering",
                        "_temp_id": "d1",
                    }
                ]
            },
            {
                "results": [
                    {"_type": "DepartmentModel", "name": "Marketing", "_temp_id": "d2"}
                ]
            },
        ]
        # With 2 clients and num_llm_revisions=2, each client will be called once.
        # We set each to return one of the different revisions.
        self.mock_llm_client1.set_revisions_to_return([llm_revisions[0]])
        self.mock_llm_client2.set_revisions_to_return([llm_revisions[1]])

        mock_consensus_details = {
            "revisions_processed": 2,
            "unique_paths_considered": 2,
            "paths_agreed_by_threshold": 0,
            "paths_resolved_by_conflict_resolver": 0,
            "paths_omitted_due_to_no_consensus_or_resolver_omission": 2,
        }

        with mock.patch.object(
            self.orchestrator.json_consensus,
            "get_consensus",
            return_value=([], mock_consensus_details),
        ) as mock_get_consensus:
            hydrated_objects = await self.orchestrator.synthesize(
                ["Some text"], self.db_session
            )
            # The orchestrator will gather one revision from each of the two clients.
            expected_consensus_input = [
                llm_revisions[0]["results"],
                llm_revisions[1]["results"],
            ]
            mock_get_consensus.assert_called_once()
            actual_call_args = mock_get_consensus.call_args[0][0]
            self.assertCountEqual(actual_call_args, expected_consensus_input)

        self.assertEqual(len(hydrated_objects), 0)

        report = self.orchestrator.get_analytics_report()
        self.assertEqual(report["llm_api_call_failures"], 0)
        self.assertEqual(report["total_invalid_parsing_errors"], 0)
        self.assertEqual(report["number_of_consensus_runs"], 1)
        self.assertEqual(report["average_path_agreement_ratio"], 0.0)
        self.assertEqual(report["all_consensus_run_details"][0], mock_consensus_details)

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

        with mock.patch(
            "extrai.core.workflow_orchestrator.ExampleJSONGenerator"
        ) as mocked_example_generator:
            await self.orchestrator.synthesize(
                input_strings,
                self.db_session,
                extraction_example_json=user_example_json,
            )
            mocked_example_generator.assert_not_called()

        self.assertIn(user_example_json, self.mock_llm_client1.last_system_prompt)
        if user_example_json:
            self.assertIn(
                "# EXAMPLE OF EXTRACTION:", self.mock_llm_client1.last_system_prompt
            )

    @mock.patch("extrai.core.workflow_orchestrator.ExampleJSONGenerator")
    async def test_synthesize_auto_generate_example_json_success(
        self, mocked_example_generator
    ):
        input_strings = ["Some text for engineering department"]
        auto_generated_example_str = json.dumps(
            {"_type": "DepartmentModel", "name": "AutoGeneratedExample"}
        )

        mock_generator_instance = mocked_example_generator.return_value
        mock_generator_instance.generate_example = AsyncMock(
            return_value=auto_generated_example_str
        )

        llm_main_extraction_response = [
            {
                "results": [
                    {
                        "_type": "DepartmentModel",
                        "_temp_id": "dept_eng_actual",
                        "name": "Engineering",
                    }
                ]
            }
        ]
        self.mock_llm_client1.set_revisions_to_return(llm_main_extraction_response)
        self.mock_llm_client2.set_revisions_to_return(llm_main_extraction_response)

        await self.orchestrator.synthesize(
            input_strings, self.db_session, extraction_example_json=""
        )

        mocked_example_generator.assert_called_once_with(
            llm_client=self.mock_llm_client1,
            output_model=self.orchestrator.root_sqlmodel_class,
            analytics_collector=self.orchestrator.analytics_collector,
            max_validation_retries_per_revision=self.orchestrator.max_validation_retries_per_revision,
            logger=self.orchestrator.logger,
        )
        mock_generator_instance.generate_example.assert_called_once()

        self.assertIn(
            auto_generated_example_str, self.mock_llm_client2.last_system_prompt
        )
        self.assertIn(
            "# EXAMPLE OF EXTRACTION:", self.mock_llm_client2.last_system_prompt
        )

    @mock.patch("extrai.core.workflow_orchestrator.ExampleJSONGenerator")
    async def test_synthesize_auto_generate_example_json_failure(
        self, mocked_example_generator
    ):
        input_strings = ["Some text"]
        mock_generator_instance = mocked_example_generator.return_value
        example_gen_error = ExampleGenerationError("Failed to generate example")
        mock_generator_instance.generate_example = AsyncMock(
            side_effect=example_gen_error
        )

        with self.assertRaisesRegex(
            WorkflowError,
            r"Failed to auto-generate extraction example:.*Failed to generate example",
        ):
            await self.orchestrator.synthesize(
                input_strings, self.db_session, extraction_example_json=""
            )

        mocked_example_generator.assert_called_once_with(
            llm_client=self.mock_llm_client1,
            output_model=self.orchestrator.root_sqlmodel_class,
            analytics_collector=self.orchestrator.analytics_collector,
            max_validation_retries_per_revision=self.orchestrator.max_validation_retries_per_revision,
            logger=self.orchestrator.logger,
        )
        mock_generator_instance.generate_example.assert_called_once()
        self.assertEqual(self.mock_llm_client1.call_count, 0)
        self.assertEqual(self.mock_llm_client2.call_count, 0)

    def test_get_analytics_collector_method(self):
        collector_instance = self.orchestrator.get_analytics_collector()
        self.assertIsInstance(collector_instance, WorkflowAnalyticsCollector)
        self.assertIs(collector_instance, self.orchestrator.analytics_collector)

    async def test_synthesis_llm_returns_malformed_revisions_now_caught_by_client(self):
        self.mock_llm_client1.set_should_raise_exception(
            LLMOutputParseError("Client failed to parse", raw_content="not a dict")
        )
        self.mock_llm_client1.set_should_raise_exception_for_example_gen(None)

        with self.assertRaisesRegex(
            LLMInteractionError, "LLM client operation failed: Client failed to parse"
        ):
            await self.orchestrator.synthesize(["Some text"], self.db_session)

        self.assertEqual(self.mock_llm_client1.example_gen_call_count, 1)
        self.assertEqual(self.mock_llm_client1.call_count, 1)

        report = self.orchestrator.get_analytics_report()
        self.assertEqual(report["llm_output_parse_errors"], 1)
        self.assertEqual(report["total_invalid_parsing_errors"], 1)
        self.assertEqual(report["number_of_consensus_runs"], 0)

    async def test_synthesize_empty_input_strings(self):
        with self.assertRaisesRegex(ValueError, "Input strings list cannot be empty."):
            await self.orchestrator.synthesize([], self.db_session)

    @mock.patch("extrai.core.workflow_orchestrator.ExampleJSONGenerator")
    async def test_synthesize_auto_generate_example_json_raises_generic_exception(
        self, mocked_example_generator
    ):
        mock_generator_instance = mocked_example_generator.return_value
        mock_generator_instance.generate_example = AsyncMock(
            side_effect=Exception("Unexpected boom during example gen")
        )

        current_orchestrator = WorkflowOrchestrator(
            root_sqlmodel_class=DepartmentModel,
            llm_client=self.mock_llm_client1,
            num_llm_revisions=self.orchestrator.num_llm_revisions,
            max_validation_retries_per_revision=self.orchestrator.max_validation_retries_per_revision,
        )
        current_orchestrator.analytics_collector.record_custom_event = mock.Mock()

        with self.assertRaisesRegex(
            WorkflowError,
            r"An unexpected error occurred during auto-generation of extraction example:.*Unexpected boom during example gen",
        ):
            await current_orchestrator.synthesize(
                ["Some text"], self.db_session, extraction_example_json=""
            )

        mocked_example_generator.assert_called_once()
        mock_generator_instance.generate_example.assert_called_once()
        current_orchestrator.analytics_collector.record_custom_event.assert_any_call(
            "example_json_auto_generation_unexpected_failure"
        )
        self.assertEqual(self.mock_llm_client1.call_count, 0)

    async def test_synthesis_llm_client_returns_no_revisions(self):
        # Each call to the mock should return an empty list, simulating no content.
        self.mock_llm_client1.set_revisions_to_return([[]] * 2)
        self.mock_llm_client2.set_revisions_to_return([[]] * 2)
        self.mock_llm_client1.set_should_raise_exception_for_example_gen(None)

        # The orchestrator should handle cases where all revisions are empty without raising an error.
        hydrated_objects = await self.orchestrator.synthesize(
            ["Some text"], self.db_session
        )

        self.assertEqual(len(hydrated_objects), 0)
        self.assertEqual(self.mock_llm_client1.example_gen_call_count, 1)
        self.assertEqual(self.mock_llm_client1.call_count, 1)
        self.assertEqual(self.mock_llm_client2.call_count, 1)

    async def test_synthesis_llm_client_returns_no_revisions_at_all(self):
        self.mock_llm_client1.set_revisions_to_return([[]])
        self.mock_llm_client2.set_revisions_to_return([[]])
        self.mock_llm_client1.set_should_raise_exception_for_example_gen(None)

        with mock.patch("asyncio.gather", new_callable=AsyncMock) as mock_gather:
            mock_gather.return_value = []
            with self.assertRaisesRegex(
                LLMInteractionError,
                "LLM client returned no revisions despite being requested.",
            ):
                await self.orchestrator.synthesize(["Some text"], self.db_session)

    async def test_synthesis_llm_client_returns_malformed_revision_item(self):
        # The mock now expects a list of revisions to return.
        # We simulate one client returning a valid-looking list, and the other returning a malformed one.
        self.mock_llm_client1.set_revisions_to_return([[{"key": "value"}]])
        self.mock_llm_client2.set_revisions_to_return([["not a dict"]])

        # Malformed items are now filtered out before consensus, so no error should be raised.
        # The process should complete and return no hydrated objects.
        hydrated_objects = await self.orchestrator.synthesize(
            ["Some text"], self.db_session
        )
        self.assertEqual(len(hydrated_objects), 0)

    async def test_synthesis_llm_client_raises_generic_exception(self):
        self.mock_llm_client1.set_should_raise_exception(Exception("LLM Generic Boom!"))
        self.mock_llm_client1.set_should_raise_exception_for_example_gen(None)

        with self.assertRaisesRegex(
            LLMInteractionError,
            "An unexpected error occurred during LLM interaction: LLM Generic Boom!",
        ):
            await self.orchestrator.synthesize(["Some text"], self.db_session)

        self.assertEqual(self.mock_llm_client1.example_gen_call_count, 1)
        self.assertEqual(self.mock_llm_client1.call_count, 1)

    async def test_synthesis_consensus_returns_none(self):
        self.mock_llm_client1.set_revisions_to_return([{"key": "value"}])
        self.mock_llm_client2.set_revisions_to_return([{"key": "value"}])

        mock_consensus_details = {"revisions_processed": 1}

        with (
            mock.patch.object(
                self.orchestrator.analytics_collector, "record_consensus_run_details"
            ) as mock_record_details,
            mock.patch.object(
                self.orchestrator.json_consensus,
                "get_consensus",
                return_value=(None, mock_consensus_details),
            ) as mock_get_consensus,
        ):
            hydrated_objects = await self.orchestrator.synthesize(
                ["Some text"], self.db_session
            )

            mock_get_consensus.assert_called_once()
            mock_record_details.assert_called_once_with(mock_consensus_details)

        self.assertEqual(len(hydrated_objects), 0)

    async def test_synthesis_consensus_returns_empty_dict(self):
        self.mock_llm_client1.set_revisions_to_return([{"key": "value"}])
        mock_consensus_details = {"revisions_processed": 1}

        with mock.patch.object(
            self.orchestrator.json_consensus,
            "get_consensus",
            return_value=([], mock_consensus_details),
        ) as mock_get_consensus:
            hydrated_objects = await self.orchestrator.synthesize(
                ["Some text"], self.db_session
            )
            mock_get_consensus.assert_called_once()

        self.assertEqual(len(hydrated_objects), 0)

    async def test_synthesis_consensus_results_not_all_dicts(self):
        self.mock_llm_client1.set_revisions_to_return([{"key": "value"}])
        mock_consensus_output = {"results": ["not_a_dict", {"item": 2}]}
        mock_consensus_details = {"revisions_processed": 1}

        with mock.patch.object(
            self.orchestrator.json_consensus,
            "get_consensus",
            return_value=(mock_consensus_output["results"], mock_consensus_details),
        ):
            # This scenario now raises a HydrationError because the list contains non-dict items.
            with self.assertRaises(HydrationError):
                await self.orchestrator.synthesize(["Some text"], self.db_session)

    async def test_synthesis_consensus_returns_dict(self):
        llm_revisions = [[{"_type": "DepartmentModel", "name": "ConsensusDept"}]]
        self.mock_llm_client1.set_revisions_to_return(llm_revisions)
        self.mock_llm_client2.set_revisions_to_return(llm_revisions)

        # Case 1: Consensus returns a single dictionary
        consensus_single_dict = {
            "_type": "DepartmentModel",
            "_temp_id": "cd1",
            "name": "Single Dict Dept",
        }
        # Case 2: Consensus returns a dictionary with a 'results' key
        consensus_dict_with_results = {"results": [consensus_single_dict]}

        test_cases = [
            ("single_dict", consensus_single_dict),
            ("dict_with_results", consensus_dict_with_results),
        ]

        for name, consensus_output in test_cases:
            with self.subTest(name=name):
                with mock.patch.object(
                    self.orchestrator.json_consensus,
                    "get_consensus",
                    return_value=(consensus_output, {}),
                ):
                    hydrated_objects = await self.orchestrator.synthesize(
                        ["Some text"], self.db_session
                    )

                self.assertEqual(len(hydrated_objects), 1)
                self.assertIsInstance(hydrated_objects[0], DepartmentModel)
                self.assertEqual(hydrated_objects[0].name, "Single Dict Dept")

    async def test_synthesis_consensus_returns_unexpected_type(self):
        self.mock_llm_client1.set_revisions_to_return([{"key": "value"}])
        mock_consensus_details = {"revisions_processed": 1}

        with mock.patch.object(
            self.orchestrator.json_consensus,
            "get_consensus",
            return_value=("unexpected_string_type", mock_consensus_details),
        ):
            with self.assertRaisesRegex(
                ConsensusProcessError,
                "Unexpected type from json_consensus.get_consensus: <class 'str'>.",
            ):
                await self.orchestrator.synthesize(["Some text"], self.db_session)

    async def test_synthesis_consensus_get_consensus_raises_exception(self):
        self.mock_llm_client1.set_revisions_to_return([{"key": "value"}])

        with mock.patch.object(
            self.orchestrator.json_consensus,
            "get_consensus",
            side_effect=Exception("Extrai boom!"),
        ):
            with self.assertRaisesRegex(
                ConsensusProcessError,
                "Failed during JSON consensus processing: Extrai boom!",
            ):
                await self.orchestrator.synthesize(["Some text"], self.db_session)

    async def test_synthesis_hydration_fails(self):
        # This test covers hydration failure with and without a db session.
        llm_return = [{"results": [{"_type": "DepartmentModel", "name": "Valid Dept"}]}]
        self.mock_llm_client1.set_revisions_to_return(llm_return)
        self.mock_llm_client2.set_revisions_to_return(llm_return)

        mock_consensus_output = [
            {"_type": "DepartmentModel", "_temp_id": "d1", "name": "Dept For Hydration"}
        ]

        for session in [self.db_session, None]:
            with self.subTest(session=session):
                with (
                    mock.patch.object(
                        self.orchestrator.json_consensus,
                        "get_consensus",
                        return_value=(mock_consensus_output, {}),
                    ),
                    mock.patch(
                        "extrai.core.sqlalchemy_hydrator.SQLAlchemyHydrator.hydrate",
                        side_effect=Exception("Hydration boom!"),
                    ) as mock_hydrate,
                ):
                    with self.assertRaisesRegex(
                        HydrationError,
                        "Failed during SQLAlchemy object hydration: Hydration boom!",
                    ):
                        await self.orchestrator.synthesize(
                            ["Some text"], db_session_for_hydration=session
                        )
                    mock_hydrate.assert_called_once()

    @mock.patch("extrai.core.workflow_orchestrator.persist_objects")
    async def test_synthesize_and_save_no_hydrated_objects(self, mock_persist_objects):
        with (
            mock.patch.object(
                self.orchestrator, "synthesize", AsyncMock(return_value=[])
            ),
            mock.patch.object(self.db_session, "rollback") as mock_rollback,
            mock.patch.object(self.orchestrator.logger, "info") as mock_logger_info,
        ):
            await self.orchestrator.synthesize_and_save(["some input"], self.db_session)

        mock_logger_info.assert_called_with(
            "WorkflowOrchestrator: No objects were hydrated, thus nothing to persist."
        )
        mock_persist_objects.assert_not_called()
        mock_rollback.assert_not_called()

    @mock.patch("extrai.core.workflow_orchestrator.persist_objects")
    async def test_synthesize_and_save_persist_raises_db_writer_error(
        self, mock_persist_objects
    ):
        mock_persist_objects.side_effect = DatabaseWriterError("DB write failed")

        mock_hydrated_object = DepartmentModel(name="Test Dept")
        with (
            mock.patch.object(
                self.orchestrator,
                "synthesize",
                AsyncMock(return_value=[mock_hydrated_object]),
            ),
            mock.patch.object(self.db_session, "rollback") as mock_rollback,
        ):
            with self.assertRaises(DatabaseWriterError):
                await self.orchestrator.synthesize_and_save(
                    ["some input"], self.db_session
                )

        mock_persist_objects.assert_called_once_with(
            db_session=self.db_session,
            objects_to_persist=[mock_hydrated_object],
            logger=self.orchestrator.logger,
        )
        mock_rollback.assert_called_once()

    @mock.patch("extrai.core.workflow_orchestrator.persist_objects")
    async def test_synthesize_and_save_persist_raises_generic_exception(
        self, mock_persist_objects
    ):
        mock_persist_objects.side_effect = Exception("Generic DB boom")

        mock_hydrated_object = DepartmentModel(name="Test Dept")
        with (
            mock.patch.object(
                self.orchestrator,
                "synthesize",
                AsyncMock(return_value=[mock_hydrated_object]),
            ),
            mock.patch.object(self.db_session, "rollback") as mock_rollback,
        ):
            with self.assertRaisesRegex(
                WorkflowError,
                "An unexpected error occurred during database persistence phase: Generic DB boom",
            ):
                await self.orchestrator.synthesize_and_save(
                    ["some input"], self.db_session
                )

        mock_persist_objects.assert_called_once_with(
            db_session=self.db_session,
            objects_to_persist=[mock_hydrated_object],
            logger=self.orchestrator.logger,
        )
        mock_rollback.assert_called_once()


if __name__ == "__main__":
    unittest.main()
