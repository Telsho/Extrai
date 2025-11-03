import pytest
from unittest.mock import MagicMock
from typing import List, Dict, Any

from extrai.core.json_consensus import JSONConsensus
from extrai.core.analytics_collector import (
    WorkflowAnalyticsCollector,
)


@pytest.fixture
def mock_json_consensus() -> MagicMock:
    consensus_mock = MagicMock(spec=JSONConsensus)

    def mock_get_consensus(revisions: List[Dict[str, Any]]):
        # Ensure revisions is a list and not empty before accessing revisions[0]
        if (
            revisions
            and isinstance(revisions, list)
            and revisions[0].get("results")
            and isinstance(revisions[0]["results"], list)
        ):
            return revisions[0], {"consensus_details": "mocked_success"}
        return {"results": []}, {"consensus_details": "mocked_empty"}

    consensus_mock.get_consensus = MagicMock(side_effect=mock_get_consensus)
    return consensus_mock


@pytest.fixture
def analytics_collector_mock() -> MagicMock:
    return MagicMock(spec=WorkflowAnalyticsCollector)
