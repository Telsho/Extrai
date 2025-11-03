# tests/core/test_workflow_orchestrator_init.py

import unittest
import json
from unittest import mock

from extrai.core.workflow_orchestrator import (
    WorkflowOrchestrator,
    ConfigurationError,
)
from extrai.core.analytics_collector import (
    WorkflowAnalyticsCollector,
)
from tests.core.helpers.orchestrator_test_models import DepartmentModel, EmployeeModel
from tests.core.helpers.mock_llm_clients import (
    MockLLMClientForWorkflow as MockLLMClient,
)


class TestWorkflowOrchestratorInitialization(unittest.TestCase):
    def setUp(self):
        self.mock_llm_client = MockLLMClient()
        self.root_sqlmodel_class = DepartmentModel
        self.mock_discovered_sqlmodel_classes = [DepartmentModel, EmployeeModel]
        self.mock_prompt_llm_schema_str = json.dumps(
            {"schema_for_prompt": "mock_llm_prompt_schema"}
        )

    @mock.patch("extrai.core.workflow_orchestrator.discover_sqlmodels_from_root")
    @mock.patch("extrai.core.workflow_orchestrator.generate_llm_schema_from_models")
    def test_successful_initialization(
        self, mock_generate_llm_schema, mock_discover_sqlmodels
    ):
        mock_discover_sqlmodels.return_value = self.mock_discovered_sqlmodel_classes
        mock_generate_llm_schema.return_value = self.mock_prompt_llm_schema_str

        orchestrator = WorkflowOrchestrator(
            root_sqlmodel_class=self.root_sqlmodel_class,
            llm_client=self.mock_llm_client,
            max_validation_retries_per_revision=2,
        )
        mock_discover_sqlmodels.assert_called_once_with(self.root_sqlmodel_class)

        expected_sqla_models_set = {DepartmentModel, EmployeeModel}

        mock_generate_llm_schema.assert_called_once()
        called_initial_models_list = mock_generate_llm_schema.call_args.kwargs[
            "initial_model_classes"
        ]
        self.assertEqual(set(called_initial_models_list), expected_sqla_models_set)

        expected_model_map = {
            "DepartmentModel": DepartmentModel,
            "EmployeeModel": EmployeeModel,
        }
        self.assertEqual(
            orchestrator.model_schema_map_for_hydration, expected_model_map
        )
        self.assertEqual(
            orchestrator.target_json_schema_for_llm, self.mock_prompt_llm_schema_str
        )
        self.assertFalse(hasattr(orchestrator, "formal_json_schema_for_validation"))
        self.assertEqual(orchestrator.max_validation_retries_per_revision, 2)
        self.assertIsNotNone(orchestrator.analytics_collector)
        self.assertIsInstance(
            orchestrator.analytics_collector, WorkflowAnalyticsCollector
        )
        self.assertFalse(orchestrator.use_hierarchical_extraction)  # Default

    @mock.patch("extrai.core.workflow_orchestrator.discover_sqlmodels_from_root")
    @mock.patch("extrai.core.workflow_orchestrator.generate_llm_schema_from_models")
    def test_initialization_with_hierarchical_extraction_enabled(
        self, mock_generate_llm_schema, mock_discover_sqlmodels
    ):
        mock_discover_sqlmodels.return_value = self.mock_discovered_sqlmodel_classes
        mock_generate_llm_schema.return_value = self.mock_prompt_llm_schema_str

        with mock.patch("logging.Logger.warning") as mock_logger_warning:
            orchestrator = WorkflowOrchestrator(
                root_sqlmodel_class=self.root_sqlmodel_class,
                llm_client=self.mock_llm_client,
                use_hierarchical_extraction=True,
            )
            self.assertTrue(orchestrator.use_hierarchical_extraction)
            mock_logger_warning.assert_called_once_with(
                "Hierarchical extraction is enabled. "
                "This may significantly increase LLM API calls and processing time "
                "based on model complexity and the number of entities."
            )

    def test_init_with_provided_analytics_collector(self):
        custom_collector = WorkflowAnalyticsCollector()
        orchestrator = WorkflowOrchestrator(
            root_sqlmodel_class=self.root_sqlmodel_class,
            llm_client=self.mock_llm_client,
            analytics_collector=custom_collector,
        )
        self.assertIs(orchestrator.analytics_collector, custom_collector)

    def test_init_invalid_max_validation_retries(self):
        with self.assertRaisesRegex(
            ConfigurationError, "Max validation retries per revision must be at least 1"
        ):
            WorkflowOrchestrator(
                self.root_sqlmodel_class,
                self.mock_llm_client,
                max_validation_retries_per_revision=0,
            )

    def test_init_invalid_root_sqlmodel_class(self):
        with self.assertRaisesRegex(
            ConfigurationError, "root_sqlmodel_class must be a valid SQLModel class."
        ):
            WorkflowOrchestrator(None, self.mock_llm_client)  # type: ignore

        class NotASQLModel:
            pass

        with self.assertRaisesRegex(
            ConfigurationError, "root_sqlmodel_class must be a valid SQLModel class."
        ):
            WorkflowOrchestrator(NotASQLModel, self.mock_llm_client)  # type: ignore

    def test_init_invalid_num_llm_revisions(self):
        with self.assertRaisesRegex(
            ConfigurationError, "Number of LLM revisions must be at least 1."
        ):
            WorkflowOrchestrator(
                self.root_sqlmodel_class, self.mock_llm_client, num_llm_revisions=0
            )

    def test_init_invalid_consensus_threshold(self):
        with self.assertRaisesRegex(
            ConfigurationError,
            "Extrai threshold must be between 0.0 and 1.0 inclusive.",
        ):
            WorkflowOrchestrator(
                self.root_sqlmodel_class, self.mock_llm_client, consensus_threshold=-0.1
            )
        with self.assertRaisesRegex(
            ConfigurationError,
            "Extrai threshold must be between 0.0 and 1.0 inclusive.",
        ):
            WorkflowOrchestrator(
                self.root_sqlmodel_class, self.mock_llm_client, consensus_threshold=1.1
            )

    @mock.patch("extrai.core.workflow_orchestrator.discover_sqlmodels_from_root")
    def test_init_discover_sqlmodels_fails_generic_exception(
        self, mock_discover_sqlmodels
    ):
        mock_discover_sqlmodels.side_effect = Exception("Discovery boom!")
        with self.assertRaisesRegex(
            ConfigurationError, "Failed to discover SQLModel classes: Discovery boom!"
        ):
            WorkflowOrchestrator(self.root_sqlmodel_class, self.mock_llm_client)

    @mock.patch("extrai.core.workflow_orchestrator.discover_sqlmodels_from_root")
    def test_init_discover_sqlmodels_returns_empty(self, mock_discover_sqlmodels):
        mock_discover_sqlmodels.return_value = []
        with self.assertRaisesRegex(
            ConfigurationError,
            "No SQLModel classes were discovered from the root model.",
        ):
            WorkflowOrchestrator(self.root_sqlmodel_class, self.mock_llm_client)

    @mock.patch("extrai.core.workflow_orchestrator.discover_sqlmodels_from_root")
    @mock.patch("extrai.core.workflow_orchestrator.generate_llm_schema_from_models")
    def test_init_generate_llm_schema_returns_empty_string(
        self, mock_generate_llm_schema, mock_discover_sqlmodels
    ):
        mock_discover_sqlmodels.return_value = self.mock_discovered_sqlmodel_classes
        mock_generate_llm_schema.return_value = ""
        with self.assertRaisesRegex(
            ConfigurationError,
            r"Generated target_json_schema_for_llm \(prompt schema\) is empty.",
        ):
            WorkflowOrchestrator(self.root_sqlmodel_class, self.mock_llm_client)

    @mock.patch("extrai.core.workflow_orchestrator.discover_sqlmodels_from_root")
    @mock.patch("extrai.core.workflow_orchestrator.generate_llm_schema_from_models")
    def test_init_generate_llm_schema_returns_invalid_json(
        self, mock_generate_llm_schema, mock_discover_sqlmodels
    ):
        mock_discover_sqlmodels.return_value = self.mock_discovered_sqlmodel_classes
        mock_generate_llm_schema.return_value = "not a valid json"
        with self.assertRaisesRegex(
            ConfigurationError,
            "The internally generated LLM prompt JSON schema is not valid:",
        ):
            WorkflowOrchestrator(self.root_sqlmodel_class, self.mock_llm_client)

    @mock.patch("extrai.core.workflow_orchestrator.discover_sqlmodels_from_root")
    @mock.patch("extrai.core.workflow_orchestrator.generate_llm_schema_from_models")
    def test_init_generate_llm_schema_fails_generic_exception(
        self, mock_generate_llm_schema, mock_discover_sqlmodels
    ):
        mock_discover_sqlmodels.return_value = self.mock_discovered_sqlmodel_classes
        mock_generate_llm_schema.side_effect = Exception("Schema gen boom!")
        with self.assertRaisesRegex(
            ConfigurationError,
            "Failed to generate the LLM prompt JSON schema: Schema gen boom!",
        ):
            WorkflowOrchestrator(self.root_sqlmodel_class, self.mock_llm_client)

    @mock.patch("extrai.core.workflow_orchestrator.discover_sqlmodels_from_root")
    @mock.patch("extrai.core.workflow_orchestrator.generate_llm_schema_from_models")
    def test_init_with_invalid_llm_client_in_list(
        self, mock_generate_llm_schema, mock_discover_sqlmodels
    ):
        mock_discover_sqlmodels.return_value = self.mock_discovered_sqlmodel_classes
        mock_generate_llm_schema.return_value = self.mock_prompt_llm_schema_str
        with self.assertRaisesRegex(
            ConfigurationError,
            "All items in llm_client list must be instances of BaseLLMClient.",
        ):
            WorkflowOrchestrator(
                root_sqlmodel_class=self.root_sqlmodel_class,
                llm_client=[self.mock_llm_client, "not a client"],
            )

    @mock.patch("extrai.core.workflow_orchestrator.discover_sqlmodels_from_root")
    @mock.patch("extrai.core.workflow_orchestrator.generate_llm_schema_from_models")
    def test_init_with_empty_llm_client_list(
        self, mock_generate_llm_schema, mock_discover_sqlmodels
    ):
        mock_discover_sqlmodels.return_value = self.mock_discovered_sqlmodel_classes
        mock_generate_llm_schema.return_value = self.mock_prompt_llm_schema_str
        with self.assertRaisesRegex(
            ConfigurationError,
            "llm_client list cannot be empty.",
        ):
            WorkflowOrchestrator(
                root_sqlmodel_class=self.root_sqlmodel_class,
                llm_client=[],
            )

    @mock.patch("extrai.core.workflow_orchestrator.discover_sqlmodels_from_root")
    @mock.patch("extrai.core.workflow_orchestrator.generate_llm_schema_from_models")
    def test_init_with_invalid_llm_client_type(
        self, mock_generate_llm_schema, mock_discover_sqlmodels
    ):
        mock_discover_sqlmodels.return_value = self.mock_discovered_sqlmodel_classes
        mock_generate_llm_schema.return_value = self.mock_prompt_llm_schema_str
        with self.assertRaisesRegex(
            ConfigurationError,
            "llm_client must be an instance of BaseLLMClient or a list of them.",
        ):
            WorkflowOrchestrator(
                root_sqlmodel_class=self.root_sqlmodel_class,
                llm_client="not a valid client type",  # type: ignore
            )


if __name__ == "__main__":
    unittest.main()
