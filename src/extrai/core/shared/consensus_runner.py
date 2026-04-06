import logging
from typing import Any

from extrai.core.analytics_collector import WorkflowAnalyticsCollector
from extrai.core.errors import ConsensusProcessError
from extrai.core.extraction_config import ExtractionConfig
from extrai.core.json_consensus import JSONConsensus, default_conflict_resolver
from extrai.utils.alignment_utils import normalize_json_revisions


class ConsensusRunner:
    def __init__(
        self,
        config: ExtractionConfig,
        analytics_collector: WorkflowAnalyticsCollector,
        logger: logging.Logger,
    ):
        self.config = config
        self.analytics_collector = analytics_collector
        self.logger = logger
        self.consensus = JSONConsensus(
            consensus_threshold=config.consensus_threshold,
            conflict_resolver=config.conflict_resolver or default_conflict_resolver,
            logger=logger,
        )

    def run(self, revisions: list[list[dict]]) -> list[dict]:
        try:
            self.logger.debug(f"Running consensus on {len(revisions)} revisions")

            normalized_revisions = normalize_json_revisions(revisions)

            consensus_output, details = self.consensus.get_consensus(
                normalized_revisions
            )

            if details:
                self.analytics_collector.record_consensus_run_details(details)
                self.logger.debug(f"Consensus details: {details}")

            processed = self._process_output(consensus_output)

            self.logger.debug(f"Consensus produced {len(processed)} entities")

            return processed

        except ConsensusProcessError:
            raise

        except Exception as e:
            self.logger.error(f"Consensus processing failed: {e}")
            raise ConsensusProcessError(
                f"Failed during JSON consensus processing: {e}"
            ) from e

    def _process_output(self, consensus_output: Any) -> list[dict[str, Any]]:
        if consensus_output is None:
            self.logger.warning("Consensus returned None, returning empty list")
            return []

        if isinstance(consensus_output, list):
            return consensus_output

        if isinstance(consensus_output, dict):
            if "results" in consensus_output and isinstance(
                consensus_output["results"], list
            ):
                return consensus_output["results"]

            return [consensus_output]

        raise ConsensusProcessError(
            f"Unexpected consensus output type: {type(consensus_output)}. "
            f"Expected None, list, or dict."
        )
