import pytest
from unittest.mock import MagicMock, AsyncMock
from extrai.core.analytics_collector import WorkflowAnalyticsCollector
from extrai.core.base_llm_client import ResponseMode
from extrai.llm_providers.generic_openai_client import GenericOpenAIClient


@pytest.mark.asyncio
async def test_cost_tracking_generic_openai():
    # Setup
    collector = WorkflowAnalyticsCollector()
    client = GenericOpenAIClient(
        api_key="test", model_name="gpt-4o", base_url="http://test"
    )
    client.client = MagicMock()

    # Mock chat completion response with usage
    mock_completion = MagicMock()
    mock_completion.choices = [MagicMock(message=MagicMock(content="test response"))]
    mock_completion.usage = MagicMock(prompt_tokens=10, completion_tokens=20)

    client.client.chat.completions.create = AsyncMock(return_value=mock_completion)

    # Execute
    await client._execute_llm_call("system", "user", analytics_collector=collector)

    # Verify
    assert collector.total_input_tokens == 10
    assert collector.total_output_tokens == 20
    assert len(collector._llm_cost_details) == 1
    assert collector._llm_cost_details[0]["model"] == "gpt-4o"
    assert collector._llm_cost_details[0]["input_tokens"] == 10


@pytest.mark.asyncio
async def test_structured_cost_tracking():
    # Setup
    collector = WorkflowAnalyticsCollector()
    client = GenericOpenAIClient(
        api_key="test", model_name="gpt-4o", base_url="http://test"
    )
    client.client = MagicMock()

    # Mock parse response with usage
    mock_completion = MagicMock()
    mock_completion.choices = [
        MagicMock(message=MagicMock(parsed={"foo": "bar"}, refusal=None))
    ]
    mock_completion.usage = MagicMock(prompt_tokens=5, completion_tokens=15)

    client.client.beta.chat.completions.parse = AsyncMock(return_value=mock_completion)

    # Execute
    await client._execute_llm_call(
        "system",
        "user",
        response_mode=ResponseMode.STRUCTURED,
        response_model=MagicMock(),
        analytics_collector=collector,
    )

    # Verify
    assert collector.total_input_tokens == 5
    assert collector.total_output_tokens == 15
