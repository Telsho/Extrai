import unittest
from unittest.mock import MagicMock, patch, AsyncMock
from sqlmodel import SQLModel, Session, select
from datetime import datetime, timezone
import json

from extrai.core.batch.batch_pipeline import BatchPipeline
from extrai.core.batch_models import BatchJobContext, BatchJobStatus, BatchJobStep
from extrai.core.model_registry import ModelRegistry
from extrai.core.extraction_config import ExtractionConfig
from extrai.core.base_llm_client import BaseLLMClient
from extrai.core.config.batch_job_config import BatchJobConfig

class TestBatchPipelineContext(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        # Mocks
        self.mock_registry = MagicMock(spec=ModelRegistry)
        models = [MagicMock(), MagicMock()]
        models[0].__name__ = "ModelA"
        models[1].__name__ = "ModelB"
        self.mock_registry.models = models
        self.mock_registry.get_all_model_names.return_value = ["ModelA", "ModelB"]
        self.mock_registry.llm_schema_json = "{}"
        
        self.mock_client = MagicMock(spec=BaseLLMClient)
        self.mock_client.create_batch_job = AsyncMock(return_value=MagicMock(id="job_id"))
        self.mock_client.temperature = 0.0  # Required for _create_batch_requests
        self.mock_client.model_name = "gpt-4o" # Optional but good for completeness

        self.mock_config = MagicMock(spec=ExtractionConfig)
        self.mock_config.use_hierarchical_extraction = True
        self.mock_config.num_llm_revisions = 1
        
        self.mock_analytics = MagicMock()
        self.mock_logger = MagicMock()
        
        # Patch dependencies
        self.patchers = {
            "ClientRotator": patch("extrai.core.batch.batch_pipeline.ClientRotator"),
            "ExtractionContextPreparer": patch("extrai.core.batch.batch_pipeline.ExtractionContextPreparer"),
            "PromptBuilder": patch("extrai.core.batch.batch_pipeline.PromptBuilder"),
            "EntityCounter": patch("extrai.core.batch.batch_pipeline.EntityCounter"),
            "ConsensusRunner": patch("extrai.core.batch.batch_pipeline.ConsensusRunner"),
            "ExtractionRequestFactory": patch("extrai.core.batch.batch_pipeline.ExtractionRequestFactory"),
        }
        
        self.mock_deps = {}
        for name, p in self.patchers.items():
            self.mock_deps[name] = p.start()
            
        self.pipeline = BatchPipeline(
            self.mock_registry,
            self.mock_client,
            self.mock_config,
            self.mock_analytics,
            self.mock_logger
        )
        
        # Setup ClientRotator mock to return our mock client
        self.pipeline.client_rotator.get_next_client.return_value = self.mock_client
        self.pipeline.client_rotator.current_client = self.mock_client

        # Access mock instances
        self.pipeline.entity_counter = MagicMock()
        self.pipeline.entity_counter.llm_client = self.mock_client
        self.pipeline.entity_counter.prepare_counting_prompts.return_value = ("sys", "user")
        
        self.pipeline.request_factory = MagicMock()
        self.pipeline.request_factory.prepare_request.return_value = MagicMock(
            system_prompt="sys", user_prompt="user", json_schema={}, response_model=None
        )

        # Update submitter references
        self.pipeline.submitter.entity_counter = self.pipeline.entity_counter
        self.pipeline.submitter.request_factory = self.pipeline.request_factory
        self.pipeline.submitter.client_rotator = self.pipeline.client_rotator

    def tearDown(self):
        for p in self.patchers.values():
            p.stop()

    async def test_submit_counting_phase_passes_context(self):
        # Setup context and DB session
        context = BatchJobContext(
            root_batch_id="batch_123",
            input_strings=["doc"],
            config=BatchJobConfig(
                hierarchical=True,
                current_model_index=1,
                custom_counting_context="ctx",
            )
        )
        
        # Mock Session
        mock_session = MagicMock(spec=Session)
        
        # Mock Previous Step Result
        prev_result = [{"id": 1, "_type": "ModelA"}]
        mock_step = BatchJobStep(
            batch_id="batch_123", step_index=0, status=BatchJobStatus.COMPLETED, result=prev_result
        )
        
        # Mock DB query execution
        # The code uses db_session.exec(select(...)).all()
        mock_exec = MagicMock()
        mock_exec.all.return_value = [mock_step]
        mock_session.exec.return_value = mock_exec
        
        # Call method under test - use submitter
        await self.pipeline.submitter._submit_counting_phase(context, mock_session, step_index=1)
        
        # Verification
        # 1. Verify DB query was made to fetch previous steps
        self.assertTrue(mock_session.exec.called)
        
        # 2. Verify prepare_counting_prompts called with previous_entities
        self.pipeline.entity_counter.prepare_counting_prompts.assert_called_once()
        call_args = self.pipeline.entity_counter.prepare_counting_prompts.call_args
        
        # Args: input_strings, model_names, custom_context, previous_entities
        self.assertEqual(call_args[0][0], ["doc"]) # input_strings
        self.assertEqual(call_args[0][1], ["ModelB"]) # model_names (index 1)
        self.assertEqual(call_args[0][2], "ctx") # custom_context
        
        # Keyword arg 'previous_entities' check
        kwargs = call_args[1]
        self.assertIn("previous_entities", kwargs)
        self.assertEqual(kwargs["previous_entities"], prev_result)

    async def test_submit_extraction_phase_passes_context(self):
        # Setup context
        context = BatchJobContext(
            root_batch_id="batch_123",
            input_strings=["doc"],
            config=BatchJobConfig(
                hierarchical=True,
                current_model_index=1,
            )
        )
        
        mock_session = MagicMock(spec=Session)
        
        prev_result = [{"id": 1, "_type": "ModelA"}]
        mock_step = BatchJobStep(
            batch_id="batch_123", step_index=0, status=BatchJobStatus.COMPLETED, result=prev_result
        )
        
        mock_exec = MagicMock()
        mock_exec.all.return_value = [mock_step]
        mock_session.exec.return_value = mock_exec
        
        # Call method - use submitter
        await self.pipeline.submitter._submit_extraction_phase(context, mock_session, step_index=1)
        
        # Verify request factory call
        self.pipeline.request_factory.prepare_request.assert_called_once()
