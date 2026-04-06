import unittest
from unittest.mock import MagicMock, patch, AsyncMock
from extrai.core.extraction_pipeline import ExtractionPipeline
from extrai.core.model_registry import ModelRegistry
from extrai.core.extraction_config import ExtractionConfig
from extrai.core.base_llm_client import BaseLLMClient
from extrai.core.extraction_request_factory import ExtractionRequest


class TestExtractionPipeline(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.mock_model_registry = MagicMock(spec=ModelRegistry)
        self.mock_model_registry.root_model = MagicMock()
        self.mock_model_registry.root_model.__name__ = "RootModel"
        self.mock_model_registry.llm_schema_json = "{}"

        self.mock_client = MagicMock(spec=BaseLLMClient)
        self.mock_config = MagicMock(spec=ExtractionConfig)
        self.mock_config.use_hierarchical_extraction = False
        self.mock_config.use_structured_output = False
        self.mock_config.max_validation_retries_per_revision = 1
        self.mock_analytics = MagicMock()
        self.mock_logger = MagicMock()

        with (
            patch("extrai.core.extraction_pipeline.ClientRotator"),
            patch("extrai.core.extraction_pipeline.ExtractionContextPreparer"),
            patch("extrai.core.extraction_pipeline.PromptBuilder"),
            patch("extrai.core.extraction_pipeline.EntityCounter"),
            patch("extrai.core.extraction_pipeline.LLMRunner"),
            patch("extrai.core.extraction_pipeline.ModelWrapperBuilder"),
            patch(
                "extrai.core.extraction_pipeline.ExtractionRequestFactory"
            ) as MockFactory,
        ):
            # Setup default request factory behavior
            self.mock_request_factory = MockFactory.return_value
            self.mock_request = ExtractionRequest(
                system_prompt="sys",
                user_prompt="user",
                json_schema=None,
                model_name=None,
                response_model=None,
            )
            self.mock_request_factory.prepare_request.return_value = self.mock_request

            self.pipeline = ExtractionPipeline(
                self.mock_model_registry,
                self.mock_client,
                self.mock_config,
                self.mock_analytics,
                self.mock_logger,
            )

    async def test_extract_standard_flow(self):
        self.pipeline.llm_runner.run_extraction_cycle = AsyncMock(
            return_value=[{"id": 1}]
        )
        self.pipeline.entity_counter.count_entities = AsyncMock(return_value={})

        self.pipeline.context_preparer.prepare_example = AsyncMock(return_value="{}")

        results = await self.pipeline.extract(["doc"])

        self.assertEqual(results, [{"id": 1}])
        self.pipeline.context_preparer.prepare_example.assert_called_once()
        self.pipeline.request_factory.prepare_request.assert_called_once()
        self.pipeline.llm_runner.run_extraction_cycle.assert_called_once_with(
            system_prompt="sys", user_prompt="user"
        )

    async def test_extract_structured_flow(self):
        self.pipeline.config.use_structured_output = True

        # Setup structured request
        structured_request = ExtractionRequest(
            system_prompt="sys_struct",
            user_prompt="user_struct",
            json_schema={"type": "object"},
            model_name=None,
            response_model=MagicMock(),
        )
        self.mock_request_factory.prepare_request.return_value = structured_request
        self.pipeline.llm_runner.run_structured_extraction_cycle = AsyncMock(
            return_value=[{"id": 1}]
        )
        self.pipeline.context_preparer.prepare_example = AsyncMock(return_value="{}")

        results = await self.pipeline.extract(["doc"])

        self.assertEqual(results, [{"id": 1}])
        self.pipeline.request_factory.prepare_request.assert_called_once()
        self.pipeline.llm_runner.run_structured_extraction_cycle.assert_called_once_with(
            system_prompt="sys_struct",
            user_prompt="user_struct",
            response_model=structured_request.response_model,
        )

    async def test_extract_hierarchical_flow(self):
        self.pipeline.config.use_hierarchical_extraction = True
        self.pipeline.hierarchical_extractor = MagicMock()
        self.pipeline.hierarchical_extractor.extract = AsyncMock(
            return_value=[{"id": 1}]
        )

        self.pipeline.context_preparer.prepare_example = AsyncMock(return_value="{}")

        results = await self.pipeline.extract(["doc"])

        self.assertEqual(results, [{"id": 1}])
        self.pipeline.context_preparer.prepare_example.assert_called_once()
        self.pipeline.hierarchical_extractor.extract.assert_called_once()

    async def test_count_entities(self):
        self.pipeline.entity_counter.count_entities = AsyncMock(
            return_value={"RootModel": 10}
        )
        self.pipeline.context_preparer.prepare_example = AsyncMock(return_value="{}")
        self.pipeline.llm_runner.run_extraction_cycle = AsyncMock(return_value=[])

        await self.pipeline.extract(["doc"], count_entities=True)

        self.pipeline.entity_counter.count_entities.assert_called_once()

    async def test_count_entities_failure(self):
        self.pipeline.entity_counter.count_entities = AsyncMock(
            side_effect=Exception("Count failed")
        )
        self.pipeline.context_preparer.prepare_example = AsyncMock(return_value="{}")
        self.pipeline.llm_runner.run_extraction_cycle = AsyncMock(return_value=[])

        await self.pipeline.extract(["doc"], count_entities=True)

        self.mock_logger.warning.assert_called_with(
            "Entity counting failed or returned None, proceeding with extraction without descriptions"
        )

    def test_repr(self):
        self.pipeline.config.use_hierarchical_extraction = False
        repr_str = repr(self.pipeline)
        self.assertEqual(repr_str, "ExtractionPipeline(mode=standard, root=RootModel)")

        self.pipeline.config.use_hierarchical_extraction = True
        repr_str = repr(self.pipeline)
        self.assertEqual(
            repr_str, "ExtractionPipeline(mode=hierarchical, root=RootModel)"
        )


if __name__ == "__main__":
    unittest.main()
