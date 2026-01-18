# tests/core/test_workflow_orchestrator_init.py

import unittest
import json
from unittest import mock

from extrai.core.errors import ConfigurationError
from extrai.core.workflow_orchestrator import WorkflowOrchestrator

from extrai.core.analytics_collector import WorkflowAnalyticsCollector
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

    @mock.patch("extrai.core.workflow_orchestrator.ModelRegistry")
    def test_successful_initialization(self, MockModelRegistry):
        orchestrator = WorkflowOrchestrator(
            root_sqlmodel_class=self.root_sqlmodel_class,
            llm_client=self.mock_llm_client,
            max_validation_retries_per_revision=2,
        )

        MockModelRegistry.assert_called_once_with(self.root_sqlmodel_class, mock.ANY)

        self.assertEqual(orchestrator.config.max_validation_retries_per_revision, 2)
        self.assertIsNotNone(orchestrator.analytics_collector)
        self.assertIsInstance(
            orchestrator.analytics_collector, WorkflowAnalyticsCollector
        )
        self.assertFalse(orchestrator.config.use_hierarchical_extraction)

    @mock.patch("extrai.core.workflow_orchestrator.ModelRegistry")
    def test_initialization_with_hierarchical_extraction_enabled(
        self, MockModelRegistry
    ):
        # We don't check for log warning here as it's likely handled inside ExtractionConfig or not logged anymore
        # If it is logged, it would be by config init.

        orchestrator = WorkflowOrchestrator(
            root_sqlmodel_class=self.root_sqlmodel_class,
            llm_client=self.mock_llm_client,
            use_hierarchical_extraction=True,
        )
        self.assertTrue(orchestrator.config.use_hierarchical_extraction)

    def test_init_with_provided_analytics_collector(self):
        custom_collector = WorkflowAnalyticsCollector()
        orchestrator = WorkflowOrchestrator(
            root_sqlmodel_class=self.root_sqlmodel_class,
            llm_client=self.mock_llm_client,
            analytics_collector=custom_collector,
        )
        self.assertIs(orchestrator.analytics_collector, custom_collector)

    def test_init_invalid_max_validation_retries(self):
        # Validation happens in ExtractionConfig
        with self.assertRaisesRegex(
            ValueError, "max_validation_retries_per_revision must be at least 1"
        ):
            WorkflowOrchestrator(
                self.root_sqlmodel_class,
                self.mock_llm_client,
                max_validation_retries_per_revision=0,
            )

    @mock.patch("extrai.core.workflow_orchestrator.ModelRegistry")
    def test_init_invalid_root_sqlmodel_class(self, MockModelRegistry):
        MockModelRegistry.side_effect = ConfigurationError(
            "root_sqlmodel_class must be a valid SQLModel class."
        )

        with self.assertRaisesRegex(
            ConfigurationError, "root_sqlmodel_class must be a valid SQLModel class."
        ):
            WorkflowOrchestrator(None, self.mock_llm_client)  # type: ignore

    def test_init_invalid_num_llm_revisions(self):
        with self.assertRaisesRegex(ValueError, "num_llm_revisions must be at least 1"):
            WorkflowOrchestrator(
                self.root_sqlmodel_class, self.mock_llm_client, num_llm_revisions=0
            )

    def test_init_invalid_consensus_threshold(self):
        with self.assertRaisesRegex(
            ValueError,
            "consensus_threshold must be between 0.0 and 1.0",
        ):
            WorkflowOrchestrator(
                self.root_sqlmodel_class, self.mock_llm_client, consensus_threshold=-0.1
            )
        with self.assertRaisesRegex(
            ValueError,
            "consensus_threshold must be between 0.0 and 1.0",
        ):
            WorkflowOrchestrator(
                self.root_sqlmodel_class, self.mock_llm_client, consensus_threshold=1.1
            )

    @mock.patch("extrai.core.workflow_orchestrator.ModelRegistry")
    def test_init_discover_sqlmodels_fails_generic_exception(self, MockModelRegistry):
        MockModelRegistry.side_effect = ConfigurationError(
            "Failed to discover SQLModel classes: Discovery boom!"
        )

        with self.assertRaisesRegex(
            ConfigurationError, "Failed to discover SQLModel classes: Discovery boom!"
        ):
            WorkflowOrchestrator(self.root_sqlmodel_class, self.mock_llm_client)

    @mock.patch("extrai.core.workflow_orchestrator.ModelRegistry")
    def test_init_discover_sqlmodels_returns_empty(self, MockModelRegistry):
        MockModelRegistry.side_effect = ConfigurationError(
            "No SQLModel classes were discovered from the root model."
        )

        with self.assertRaisesRegex(
            ConfigurationError,
            "No SQLModel classes were discovered from the root model.",
        ):
            WorkflowOrchestrator(self.root_sqlmodel_class, self.mock_llm_client)

    # These tests about schema generation failure are now part of ModelRegistry tests
    # But we can verify WorkflowOrchestrator bubbles up the error if ModelRegistry raises it.
    @mock.patch("extrai.core.workflow_orchestrator.ModelRegistry")
    def test_init_generate_llm_schema_fails(self, MockModelRegistry):
        MockModelRegistry.side_effect = ConfigurationError(
            "Failed to generate the LLM prompt JSON schema"
        )

        with self.assertRaisesRegex(
            ConfigurationError,
            "Failed to generate the LLM prompt JSON schema",
        ):
            WorkflowOrchestrator(self.root_sqlmodel_class, self.mock_llm_client)

    @mock.patch("extrai.core.workflow_orchestrator.ModelRegistry")
    def test_init_with_invalid_llm_client_in_list(self, MockModelRegistry):
        # Validation happens in ExtractionPipeline which is initialized after ModelRegistry
        with self.assertRaisesRegex(
            ValueError,
            "All items in llm_client list must be instances of BaseLLMClient",
        ):
            WorkflowOrchestrator(
                root_sqlmodel_class=self.root_sqlmodel_class,
                llm_client=[self.mock_llm_client, "not a client"],
            )

    @mock.patch("extrai.core.workflow_orchestrator.ModelRegistry")
    def test_init_with_empty_llm_client_list(self, MockModelRegistry):
        with self.assertRaisesRegex(
            ValueError,
            "At least one client must be provided",
        ):
            WorkflowOrchestrator(
                root_sqlmodel_class=self.root_sqlmodel_class,
                llm_client=[],
            )

    @mock.patch("extrai.core.workflow_orchestrator.ModelRegistry")
    def test_init_with_invalid_llm_client_type(self, MockModelRegistry):
        with self.assertRaisesRegex(
            ValueError,
            "llm_client must be an instance of BaseLLMClient or a list of them",
        ):
            WorkflowOrchestrator(
                root_sqlmodel_class=self.root_sqlmodel_class,
                llm_client="not a valid client type",  # type: ignore
            )


if __name__ == "__main__":
    unittest.main()
