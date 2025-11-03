import pytest
from extrai.core.analytics_collector import (
    WorkflowAnalyticsCollector,
)
import math


class TestWorkflowAnalyticsCollector:
    def test_initial_state(self):
        collector = WorkflowAnalyticsCollector()
        assert collector.llm_api_call_failures == 0
        assert collector.llm_output_parse_errors == 0
        assert collector.llm_output_validation_errors == 0
        assert collector.total_invalid_parsing_errors == 0
        assert collector.number_of_consensus_runs == 0
        assert math.isclose(collector.average_path_agreement_ratio, 0.0)
        assert len(collector._consensus_run_details_list) == 0
        report = collector.get_report()
        assert report["llm_api_call_failures"] == 0
        assert report["llm_output_parse_errors"] == 0
        assert report["llm_output_validation_errors"] == 0
        assert report["total_invalid_parsing_errors"] == 0
        assert report["number_of_consensus_runs"] == 0
        assert math.isclose(report["average_path_agreement_ratio"], 0.0)
        assert "custom_events" not in report  # Or assert it's empty if always present
        assert "workflow_errors" not in report  # Or assert it's empty if always present

    def test_record_custom_event(self):
        collector = WorkflowAnalyticsCollector()
        collector.record_custom_event("test_event_1")
        assert len(collector._custom_events) == 1
        assert collector._custom_events[0] == {"event_name": "test_event_1"}

        details = {"key": "value", "count": 5}
        collector.record_custom_event("test_event_2", details=details)
        assert len(collector._custom_events) == 2
        expected_event_2 = {"event_name": "test_event_2"}
        expected_event_2.update(details)
        assert collector._custom_events[1] == expected_event_2

        report = collector.get_report()
        assert "custom_events" in report
        assert len(report["custom_events"]) == 2
        assert report["custom_events"][0] == {"event_name": "test_event_1"}
        assert report["custom_events"][1] == expected_event_2

    def test_record_workflow_error(self):
        collector = WorkflowAnalyticsCollector()
        collector.record_workflow_error(error_type="TestErrorType1")
        assert len(collector._workflow_errors) == 1
        assert collector._workflow_errors[0] == {"error_type": "TestErrorType1"}

        error_details = {"file": "test.py", "line": 10}
        collector.record_workflow_error(
            error_type="TestErrorType2",
            context="testing_context",
            message="This is a test error message.",
            details=error_details,
        )
        assert len(collector._workflow_errors) == 2
        expected_error_2 = {
            "error_type": "TestErrorType2",
            "context": "testing_context",
            "message": "This is a test error message.",
        }
        expected_error_2.update(error_details)
        assert collector._workflow_errors[1] == expected_error_2

        report = collector.get_report()
        assert "workflow_errors" in report
        assert len(report["workflow_errors"]) == 2
        assert report["workflow_errors"][0] == {"error_type": "TestErrorType1"}
        assert report["workflow_errors"][1] == expected_error_2

    def test_get_report_empty_custom_events_and_errors(self):
        collector = WorkflowAnalyticsCollector()
        report = collector.get_report()
        # Depending on implementation, these keys might be absent or present with empty lists
        # Based on current analytics_collector.py, they are absent if empty.
        assert "custom_events" not in report
        assert "workflow_errors" not in report
        assert "all_consensus_run_details" in report
        assert len(report["all_consensus_run_details"]) == 0

    def test_record_llm_api_call_failure(self):
        collector = WorkflowAnalyticsCollector()
        collector.record_llm_api_call_failure()
        assert collector.llm_api_call_failures == 1
        collector.record_llm_api_call_failure()
        assert collector.llm_api_call_failures == 2
        assert collector.get_report()["llm_api_call_failures"] == 2

    def test_record_llm_output_parse_error(self):
        collector = WorkflowAnalyticsCollector()
        collector.record_llm_output_parse_error()
        assert collector.llm_output_parse_errors == 1
        assert collector.total_invalid_parsing_errors == 1
        collector.record_llm_output_parse_error()
        assert collector.llm_output_parse_errors == 2
        assert collector.total_invalid_parsing_errors == 2
        assert collector.get_report()["llm_output_parse_errors"] == 2
        assert collector.get_report()["total_invalid_parsing_errors"] == 2

    def test_record_llm_output_validation_error(self):
        collector = WorkflowAnalyticsCollector()
        collector.record_llm_output_validation_error()
        assert collector.llm_output_validation_errors == 1
        assert collector.total_invalid_parsing_errors == 1
        collector.record_llm_output_validation_error()
        assert collector.llm_output_validation_errors == 2
        assert collector.total_invalid_parsing_errors == 2
        assert collector.get_report()["llm_output_validation_errors"] == 2
        assert collector.get_report()["total_invalid_parsing_errors"] == 2

    def test_total_invalid_parsing_errors_combined(self):
        collector = WorkflowAnalyticsCollector()
        collector.record_llm_output_parse_error()
        collector.record_llm_output_validation_error()
        collector.record_llm_output_parse_error()
        assert collector.llm_output_parse_errors == 2
        assert collector.llm_output_validation_errors == 1
        assert collector.total_invalid_parsing_errors == 3
        assert collector.get_report()["total_invalid_parsing_errors"] == 3

    def test_record_consensus_run_details_single(self):
        collector = WorkflowAnalyticsCollector()
        details = {
            "revisions_processed": 3,
            "unique_paths_considered": 10,
            "paths_agreed_by_threshold": 7,
            "paths_resolved_by_conflict_resolver": 2,
            "paths_omitted_due_to_no_consensus_or_resolver_omission": 1,
        }
        collector.record_consensus_run_details(details)
        assert collector.number_of_consensus_runs == 1
        assert len(collector._consensus_run_details_list) == 1
        assert collector._consensus_run_details_list[0] == details
        assert collector.average_path_agreement_ratio == pytest.approx(7 / 10)
        assert (
            collector.average_paths_resolved_by_conflict_resolver_ratio
            == pytest.approx(2 / 10)
        )
        assert collector.average_paths_omitted_ratio == pytest.approx(1 / 10)
        report = collector.get_report()
        assert report["number_of_consensus_runs"] == 1
        assert report["average_path_agreement_ratio"] == pytest.approx(7 / 10)

    def test_record_consensus_run_details_multiple(self):
        collector = WorkflowAnalyticsCollector()
        details1 = {
            "revisions_processed": 3,
            "unique_paths_considered": 10,
            "paths_agreed_by_threshold": 7,
            "paths_resolved_by_conflict_resolver": 2,
            "paths_omitted_due_to_no_consensus_or_resolver_omission": 1,
        }  # 7/10 = 0.7
        details2 = {
            "revisions_processed": 2,
            "unique_paths_considered": 8,
            "paths_agreed_by_threshold": 4,
            "paths_resolved_by_conflict_resolver": 3,
            "paths_omitted_due_to_no_consensus_or_resolver_omission": 1,
        }  # 4/8 = 0.5
        details3 = {
            "revisions_processed": 3,
            "unique_paths_considered": 12,
            "paths_agreed_by_threshold": 9,
            "paths_resolved_by_conflict_resolver": 1,
            "paths_omitted_due_to_no_consensus_or_resolver_omission": 2,
        }  # 9/12 = 0.75

        collector.record_consensus_run_details(details1)
        collector.record_consensus_run_details(details2)
        collector.record_consensus_run_details(details3)

        assert collector.number_of_consensus_runs == 3
        expected_avg_agreement = (0.7 + 0.5 + 0.75) / 3
        assert collector.average_path_agreement_ratio == pytest.approx(
            expected_avg_agreement
        )

        expected_avg_resolved = ((2 / 10) + (3 / 8) + (1 / 12)) / 3
        assert (
            collector.average_paths_resolved_by_conflict_resolver_ratio
            == pytest.approx(expected_avg_resolved)
        )

        expected_avg_omitted = ((1 / 10) + (1 / 8) + (2 / 12)) / 3
        assert collector.average_paths_omitted_ratio == pytest.approx(
            expected_avg_omitted
        )

    def test_record_consensus_run_details_malformed(self):
        collector = WorkflowAnalyticsCollector()
        malformed_details = {"revisions_processed": 1}  # Missing other keys
        collector.record_consensus_run_details(malformed_details)
        assert collector.number_of_consensus_runs == 0  # Should not record
        assert math.isclose(collector.average_path_agreement_ratio, 0.0)

    def test_average_ratios_no_runs(self):
        collector = WorkflowAnalyticsCollector()
        assert math.isclose(collector.average_path_agreement_ratio, 0.0)
        assert math.isclose(
            collector.average_paths_resolved_by_conflict_resolver_ratio, 0.0
        )
        assert math.isclose(collector.average_paths_omitted_ratio, 0.0)

    def test_average_ratios_zero_unique_paths(self):
        collector = WorkflowAnalyticsCollector()
        details = {
            "revisions_processed": 1,
            "unique_paths_considered": 0,
            "paths_agreed_by_threshold": 0,
            "paths_resolved_by_conflict_resolver": 0,
            "paths_omitted_due_to_no_consensus_or_resolver_omission": 0,
        }
        collector.record_consensus_run_details(details)
        assert math.isclose(collector.average_path_agreement_ratio, 0.0)
        assert math.isclose(
            collector.average_paths_resolved_by_conflict_resolver_ratio, 0.0
        )
        assert math.isclose(collector.average_paths_omitted_ratio, 0.0)

    def test_get_report_comprehensive_with_new_consensus(self):
        collector = WorkflowAnalyticsCollector()
        collector.record_llm_api_call_success()
        collector.record_llm_api_call_failure()
        collector.record_llm_output_parse_error()
        collector.record_llm_output_validation_error()

        details1 = {
            "revisions_processed": 2,
            "unique_paths_considered": 5,
            "paths_agreed_by_threshold": 4,
            "paths_resolved_by_conflict_resolver": 0,
            "paths_omitted_due_to_no_consensus_or_resolver_omission": 1,
        }  # 4/5 = 0.8
        details2 = {
            "revisions_processed": 3,
            "unique_paths_considered": 10,
            "paths_agreed_by_threshold": 6,
            "paths_resolved_by_conflict_resolver": 2,
            "paths_omitted_due_to_no_consensus_or_resolver_omission": 2,
        }  # 6/10 = 0.6
        collector.record_consensus_run_details(details1)
        collector.record_consensus_run_details(details2)

        report = collector.get_report()
        assert report["llm_api_call_successes"] == 1
        assert report["llm_api_call_failures"] == 1
        assert report["llm_api_call_success_rate"] == 0.5
        assert report["llm_output_parse_errors"] == 1
        assert report["llm_output_validation_errors"] == 1
        assert report["total_invalid_parsing_errors"] == 2
        assert report["number_of_consensus_runs"] == 2
        assert report["average_path_agreement_ratio"] == pytest.approx((0.8 + 0.6) / 2)
        assert report[
            "average_paths_resolved_by_conflict_resolver_ratio"
        ] == pytest.approx(((0 / 5) + (2 / 10)) / 2)
        assert report["average_paths_omitted_ratio"] == pytest.approx(
            ((1 / 5) + (2 / 10)) / 2
        )
        assert len(report["all_consensus_run_details"]) == 2
        assert report["all_consensus_run_details"][0] == details1

    def test_reset_with_new_consensus(self):
        collector = WorkflowAnalyticsCollector()
        collector.record_llm_api_call_failure()
        collector.record_llm_output_parse_error()
        details = {
            "revisions_processed": 1,
            "unique_paths_considered": 1,
            "paths_agreed_by_threshold": 1,
            "paths_resolved_by_conflict_resolver": 0,
            "paths_omitted_due_to_no_consensus_or_resolver_omission": 0,
        }
        collector.record_consensus_run_details(details)
        collector.record_custom_event("before_reset_event", {"data": 1})
        collector.record_workflow_error(
            error_type="BeforeResetError", context="reset_test"
        )

        collector.reset()

        assert collector.llm_api_call_failures == 0
        assert collector.llm_output_parse_errors == 0
        assert collector.llm_output_validation_errors == 0
        assert collector.total_invalid_parsing_errors == 0
        assert collector.number_of_consensus_runs == 0
        assert math.isclose(collector.average_path_agreement_ratio, 0.0)
        assert len(collector._consensus_run_details_list) == 0
        assert len(collector._custom_events) == 0
        assert len(collector._workflow_errors) == 0

        report = collector.get_report()
        assert report["llm_api_call_failures"] == 0
        assert report["total_invalid_parsing_errors"] == 0
        assert report["number_of_consensus_runs"] == 0
        assert math.isclose(report["average_path_agreement_ratio"], 0.0)
        assert "custom_events" not in report  # Or assert it's empty
        assert "workflow_errors" not in report  # Or assert it's empty
