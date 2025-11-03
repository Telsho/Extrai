import logging
from typing import List, Optional, Dict, Any


class WorkflowAnalyticsCollector:
    """
    Collects analytics data throughout the LLM workflow.
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        if not logger:
            self.logger.setLevel(logging.WARNING)

        self._llm_api_call_successes: int = 0
        self._llm_api_call_failures: int = 0
        self._llm_output_parse_errors: int = 0
        self._llm_output_validation_errors: int = 0
        self._hydrated_objects_successes: int = 0
        self._hydration_failures: int = 0
        # Stores a list of dictionaries, each dictionary being the analytics_details from a consensus run
        self._consensus_run_details_list: List[Dict[str, Any]] = []
        self._custom_events: List[Dict[str, Any]] = []
        self._workflow_errors: List[Dict[str, Any]] = []

    def record_llm_api_call_success(self):
        """Increments the count of successful LLM API calls."""
        self._llm_api_call_successes += 1

    def record_llm_api_call_failure(self):
        """Increments the count of LLM API call failures."""
        self._llm_api_call_failures += 1

    def record_llm_output_parse_error(self):
        """Increments the count of LLM output parsing errors."""
        self._llm_output_parse_errors += 1

    def record_llm_output_validation_error(self):
        """Increments the count of LLM output validation errors."""
        self._llm_output_validation_errors += 1

    def record_hydration_success(self, count: int):
        """Records the number of successfully hydrated objects."""
        self._hydrated_objects_successes += count

    def record_hydration_failure(self):
        """Increments the count of hydration failures."""
        self._hydration_failures += 1

    def record_consensus_run_details(self, consensus_analytics_details: Dict[str, Any]):
        """
        Records detailed analytics from a single consensus calculation.

        Args:
            consensus_analytics_details: A dictionary containing statistics from
                                         JSONConsensus.get_consensus. Expected keys include:
                                         'revisions_processed', 'unique_paths_considered',
                                         'paths_agreed_by_threshold',
                                         'paths_resolved_by_conflict_resolver',
                                         'paths_omitted_due_to_no_consensus_or_resolver_omission'.
        """
        # Basic validation of expected keys
        expected_keys = [
            "revisions_processed",
            "unique_paths_considered",
            "paths_agreed_by_threshold",
            "paths_resolved_by_conflict_resolver",
            "paths_omitted_due_to_no_consensus_or_resolver_omission",
        ]
        if not all(key in consensus_analytics_details for key in expected_keys):
            # Handle error or log warning, for now, we'll skip if malformed
            # In a real scenario, might raise an error or log.
            self.logger.warning(
                f"Malformed consensus_analytics_details: {consensus_analytics_details}"
            )
            return

        self._consensus_run_details_list.append(consensus_analytics_details)

    @property
    def llm_api_call_failures(self) -> int:
        """Returns the total count of LLM API call failures."""
        return self._llm_api_call_failures

    @property
    def total_invalid_parsing_errors(self) -> int:
        """
        Returns the total count of invalid parsing errors, which includes
        both output parsing errors and output validation errors.
        """
        return self._llm_output_parse_errors + self._llm_output_validation_errors

    @property
    def llm_output_parse_errors(self) -> int:
        """Returns the total count of LLM output parsing errors."""
        return self._llm_output_parse_errors

    @property
    def llm_output_validation_errors(self) -> int:
        """Returns the total count of LLM output validation errors."""
        return self._llm_output_validation_errors

    @property
    def number_of_consensus_runs(self) -> int:
        """Returns the total number of consensus runs recorded."""
        return len(self._consensus_run_details_list)

    @property
    def average_path_agreement_ratio(self) -> float:
        """
        Calculates the average ratio of paths agreed by threshold to unique paths considered,
        across all consensus runs. Returns 0.0 if no runs or no paths considered.
        This represents the "average consensus strength" at the path level.
        """
        if not self._consensus_run_details_list:
            return 0.0

        total_ratio_sum = 0.0
        valid_runs_for_this_metric = 0
        for details in self._consensus_run_details_list:
            unique_paths = details.get("unique_paths_considered", 0)
            agreed_paths = details.get("paths_agreed_by_threshold", 0)
            if unique_paths > 0:
                total_ratio_sum += agreed_paths / unique_paths
                valid_runs_for_this_metric += 1

        return (
            (total_ratio_sum / valid_runs_for_this_metric)
            if valid_runs_for_this_metric > 0
            else 0.0
        )

    @property
    def average_paths_resolved_by_conflict_resolver_ratio(self) -> float:
        """
        Calculates the average ratio of paths resolved by the conflict resolver
        to unique paths considered, across all consensus runs.
        """
        if not self._consensus_run_details_list:
            return 0.0

        total_ratio_sum = 0.0
        valid_runs_for_this_metric = 0
        for details in self._consensus_run_details_list:
            unique_paths = details.get("unique_paths_considered", 0)
            resolved_paths = details.get("paths_resolved_by_conflict_resolver", 0)
            if unique_paths > 0:
                total_ratio_sum += resolved_paths / unique_paths
                valid_runs_for_this_metric += 1

        return (
            (total_ratio_sum / valid_runs_for_this_metric)
            if valid_runs_for_this_metric > 0
            else 0.0
        )

    @property
    def average_paths_omitted_ratio(self) -> float:
        """
        Calculates the average ratio of paths omitted (due to no consensus or resolver omission)
        to unique paths considered, across all consensus runs.
        """
        if not self._consensus_run_details_list:
            return 0.0

        total_ratio_sum = 0.0
        valid_runs_for_this_metric = 0
        for details in self._consensus_run_details_list:
            unique_paths = details.get("unique_paths_considered", 0)
            omitted_paths = details.get(
                "paths_omitted_due_to_no_consensus_or_resolver_omission", 0
            )
            if unique_paths > 0:
                total_ratio_sum += omitted_paths / unique_paths
                valid_runs_for_this_metric += 1

        return (
            (total_ratio_sum / valid_runs_for_this_metric)
            if valid_runs_for_this_metric > 0
            else 0.0
        )

    def get_report(self) -> Dict[str, Any]:
        """
        Returns a dictionary summarizing all collected analytics.
        """
        total_llm_calls = self._llm_api_call_successes + self._llm_api_call_failures
        llm_api_call_success_rate = (
            (self._llm_api_call_successes / total_llm_calls)
            if total_llm_calls > 0
            else 0
        )

        report = {
            "llm_api_call_successes": self._llm_api_call_successes,
            "llm_api_call_failures": self.llm_api_call_failures,
            "llm_api_call_success_rate": llm_api_call_success_rate,
            "llm_output_parse_errors": self.llm_output_parse_errors,
            "llm_output_validation_errors": self.llm_output_validation_errors,
            "total_invalid_parsing_errors": self.total_invalid_parsing_errors,
            "number_of_consensus_runs": self.number_of_consensus_runs,
            "hydrated_objects_successes": self._hydrated_objects_successes,
            "hydration_failures": self._hydration_failures,
        }
        if self._consensus_run_details_list:
            report.update(
                {
                    "average_path_agreement_ratio": self.average_path_agreement_ratio,
                    "average_paths_resolved_by_conflict_resolver_ratio": self.average_paths_resolved_by_conflict_resolver_ratio,
                    "average_paths_omitted_ratio": self.average_paths_omitted_ratio,
                    "all_consensus_run_details": self._consensus_run_details_list,  # For detailed inspection
                }
            )
        else:  # Default values if no consensus runs
            report.update(
                {
                    "average_path_agreement_ratio": 0.0,
                    "average_paths_resolved_by_conflict_resolver_ratio": 0.0,
                    "average_paths_omitted_ratio": 0.0,
                    "all_consensus_run_details": [],
                }
            )

        if self._custom_events:
            report["custom_events"] = self._custom_events
        if self._workflow_errors:
            report["workflow_errors"] = self._workflow_errors

        return report

    def record_custom_event(
        self, event_name: str, details: Optional[Dict[str, Any]] = None
    ):
        """Records a generic custom event."""
        event_record = {"event_name": event_name}
        if details:
            event_record.update(details)
        self._custom_events.append(event_record)

    def record_workflow_error(
        self,
        error_type: str,
        context: Optional[str] = None,
        message: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        """Records a generic workflow error."""
        error_record = {"error_type": error_type}
        if context:
            error_record["context"] = context
        if message:
            error_record["message"] = message
        if details:
            error_record.update(details)
        self._workflow_errors.append(error_record)

    def reset(self):
        """Resets all collected analytics to their initial states."""
        self._llm_api_call_successes = 0
        self._llm_api_call_failures = 0
        self._llm_output_parse_errors = 0
        self._llm_output_validation_errors = 0
        self._hydrated_objects_successes = 0
        self._hydration_failures = 0
        self._consensus_run_details_list = []
        self._custom_events = []
        self._workflow_errors = []
