# tests/core/helpers/mock_llm_clients.py

import json
from typing import List, Dict, Any, Type, Optional
from unittest import mock
from unittest.mock import AsyncMock

from sqlmodel import SQLModel

from extrai.core.errors import (
    LLMAPICallError,
    LLMOutputParseError,
    LLMOutputValidationError,
)
from extrai.core.workflow_orchestrator import BaseLLMClient
from extrai.core.analytics_collector import (
    WorkflowAnalyticsCollector,
)


class MockLLMClientForWorkflow(BaseLLMClient):  # Renamed
    def __init__(
        self,
        api_key: str = "mock_api_key",
        model_name: str = "mock_model_name",
        base_url: Optional[str] = None,
        temperature: Optional[float] = None,
    ):
        super().__init__(
            api_key=api_key,
            model_name=model_name,
            base_url=base_url,
            temperature=temperature,
        )
        self.revisions_to_return: List[Dict[str, Any]] = []
        self.revision_index = 0
        self.should_raise_exception: Optional[Exception] = (
            None  # For generate_json_revisions
        )
        self.should_raise_exception_for_example_gen: Optional[Exception] = (
            None  # For generate_and_validate_raw_json_output
        )
        self.call_count = 0  # Tracks calls to generate_json_revisions
        self.example_gen_call_count = (
            0  # Tracks calls to generate_and_validate_raw_json_output
        )
        self.last_system_prompt = ""
        self.last_user_prompt = ""
        self.last_num_revisions = 0
        self.last_max_validation_retries = 0
        self.last_analytics_collector_passed: Optional[WorkflowAnalyticsCollector] = (
            None
        )
        self.last_target_json_schema_for_validation: Optional[Dict[str, Any]] = (
            None  # For generate_and_validate_raw_json_output
        )

        self._execute_llm_call_mock = AsyncMock(return_value="{}")

    async def _execute_llm_call(self, system_prompt: str, user_prompt: str) -> str:
        return await self._execute_llm_call_mock(system_prompt, user_prompt)

    async def generate_and_validate_raw_json_output(
        self,
        system_prompt: str,
        user_prompt: str,
        num_revisions: int,  # Though ExampleJSONGenerator calls it with 1
        target_json_schema: Dict[str, Any],  # This is the schema of the output_model
        max_validation_retries_per_revision: int,
        analytics_collector: Optional[WorkflowAnalyticsCollector] = None,
    ) -> List[Dict[str, Any]]:
        """Needed by ExampleJSONGenerator."""
        self.example_gen_call_count += 1
        self.last_system_prompt = system_prompt
        self.last_user_prompt = user_prompt
        self.last_num_revisions = num_revisions
        self.last_target_json_schema_for_validation = target_json_schema
        self.last_max_validation_retries = max_validation_retries_per_revision
        self.last_analytics_collector_passed = analytics_collector

        if self.should_raise_exception_for_example_gen:
            if analytics_collector:
                if isinstance(
                    self.should_raise_exception_for_example_gen, LLMAPICallError
                ):
                    analytics_collector.record_llm_api_call_failure()
                elif isinstance(
                    self.should_raise_exception_for_example_gen, LLMOutputParseError
                ):
                    analytics_collector.record_llm_output_parse_error()
                elif isinstance(
                    self.should_raise_exception_for_example_gen,
                    LLMOutputValidationError,
                ):
                    analytics_collector.record_llm_output_validation_error()
            raise self.should_raise_exception_for_example_gen

        # The schema passed for validation now is the wrapped `{"entities": [...]}` schema.
        # We need to simulate the LLM's behavior of generating an example for the *actual* underlying model.
        # The system prompt contains the full schema that guides the LLM.
        # We can extract the model name from the system prompt for this mock.
        model_name_for_example = "Unknown Model"
        example_dict = {}
        # Correctly check for the model name in the system prompt's schema definition
        if '"DepartmentModel":' in system_prompt:
            model_name_for_example = "DepartmentModel"
            example_dict = {
                "_type": model_name_for_example,
                "name": "Mock Generated Department",
            }
        elif '"EmployeeModel":' in system_prompt:
            model_name_for_example = "EmployeeModel"
            example_dict = {
                "_type": model_name_for_example,
                "name": "Mock Generated Employee",
            }
        elif '"ProductModel":' in system_prompt:
            model_name_for_example = "ProductModel"
            example_dict = {
                "_type": model_name_for_example,
                "name": "Mock Generated Product",
                "description": "A mock product description",
            }
        elif '"OrderModel":' in system_prompt:
            model_name_for_example = "OrderModel"
            example_dict = {
                "_type": model_name_for_example,
                "order_ref": "MOCK-ORD-GEN",
                "customer_name": "Mock Gen Customer",
            }
        elif '"OrderItemModel":' in system_prompt:
            model_name_for_example = "OrderItemModel"
            example_dict = {
                "_type": model_name_for_example,
                "product_sku": "MOCK-SKU-GEN",
                "quantity": 1,
            }
        else:
            example_dict = {
                "_type": model_name_for_example,
                "fallback_mock_field": f"Data for {model_name_for_example}",
            }

        # The mock should now return the wrapped structure, as the real LLM is expected to.
        return [{"entities": [example_dict]}]

    async def generate_json_revisions(
        self,
        system_prompt: str,
        user_prompt: str,
        num_revisions: int,
        model_schema_map: Dict[str, Type[SQLModel]],
        max_validation_retries_per_revision: int,
        analytics_collector: Optional[WorkflowAnalyticsCollector] = None,
    ) -> List[Dict[str, Any]]:
        # This is the new logic to differentiate between example generation and main extraction.
        if (
            "You are an AI assistant tasked with generating a sample JSON object."
            in system_prompt
        ):
            # This is an example generation call.
            # We can simulate the behavior of the old `generate_and_validate_raw_json_output` method here.
            self.example_gen_call_count += 1
            if self.should_raise_exception_for_example_gen:
                raise self.should_raise_exception_for_example_gen

            # Simplified example generation logic for the mock
            example_dict = {
                "_type": "DepartmentModel",
                "name": "Mock Generated Department",
            }
            return [{"entities": [example_dict]}]

        # This is a main extraction call.
        self.call_count += 1
        self.last_system_prompt = system_prompt
        self.last_user_prompt = user_prompt
        self.last_num_revisions = num_revisions
        self.last_max_validation_retries = max_validation_retries_per_revision
        self.last_analytics_collector_passed = analytics_collector

        if self.should_raise_exception:
            if analytics_collector:
                if isinstance(self.should_raise_exception, LLMAPICallError):
                    analytics_collector.record_llm_api_call_failure()
                elif isinstance(self.should_raise_exception, LLMOutputParseError):
                    analytics_collector.record_llm_output_parse_error()
                elif isinstance(self.should_raise_exception, LLMOutputValidationError):
                    analytics_collector.record_llm_output_validation_error()
            raise self.should_raise_exception

        if self.revision_index < len(self.revisions_to_return):
            # The orchestrator expects a list of entities from each call.
            # The mock data is often wrapped (e.g., in {"results": [...]}), so we unwrap it here.
            raw_revision = self.revisions_to_return[self.revision_index]
            self.revision_index += 1

            # The process_and_validate_llm_output function is the source of truth for unwrapping.
            # We can simulate its core unwrapping logic here for the mock.
            if (
                isinstance(raw_revision, dict)
                and "results" in raw_revision
                and isinstance(raw_revision["results"], list)
            ):
                return raw_revision["results"]
            elif isinstance(raw_revision, list):
                return raw_revision
            return [raw_revision]  # Fallback for simpler structures
        return []

    def set_revisions_to_return(self, revisions: List[Dict[str, Any]]):
        self.revisions_to_return = revisions

    def set_should_raise_exception(self, exception: Optional[Exception]):
        self.should_raise_exception = exception

    def set_should_raise_exception_for_example_gen(
        self, exception: Optional[Exception]
    ):
        self.should_raise_exception_for_example_gen = exception


class MockE2ELLMClient(BaseLLMClient):
    def __init__(
        self,
        api_key: str = "mock_e2e_api_key",
        model_name: str = "mock_e2e_model_name",
        base_url: Optional[str] = None,
        temperature: Optional[float] = None,
        mock_responses: Optional[List[Dict[str, Any]]] = None,
    ):
        super().__init__(
            api_key=api_key,
            model_name=model_name,
            base_url=base_url,
            temperature=temperature,
        )
        self.mock_responses = mock_responses or []
        self.revisions_to_return: List[Dict[str, Any]] = []
        self.should_raise_exception: Optional[Exception] = None
        self.return_products = True
        self.last_max_validation_retries = 0
        self.last_target_json_schema_for_validation_e2e: Optional[Dict[str, Any]] = (
            None  # For generate_and_validate_raw_json_output
        )
        self._execute_llm_call_mock = AsyncMock(return_value="{}")

    async def _execute_llm_call(self, system_prompt: str, user_prompt: str) -> str:
        return await self._execute_llm_call_mock(system_prompt, user_prompt)

    def set_return_orders(self, return_orders: bool):
        self.return_products = not return_orders

    async def generate_and_validate_raw_json_output(
        self,
        system_prompt: str,
        user_prompt: str,
        num_revisions: int,
        target_json_schema: Dict[str, Any],
        max_validation_retries_per_revision: int,
        analytics_collector: Optional[WorkflowAnalyticsCollector] = None,
    ) -> List[Dict[str, Any]]:
        """Needed by ExampleJSONGenerator for E2E tests."""
        self.last_target_json_schema_for_validation_e2e = target_json_schema
        self.last_max_validation_retries = max_validation_retries_per_revision

        if self.should_raise_exception:
            raise self.should_raise_exception

        # Similar to the other mock, we determine the model from the system prompt
        # to generate the correct example structure.
        model_name_for_example = "Unknown E2E Model"
        example_dict = {}
        if '"ProductModel":' in system_prompt:
            model_name_for_example = "ProductModel"
            example_dict = {
                "_type": model_name_for_example,
                "name": "MockE2E Product",
                "description": "A mock E2E product",
            }
        elif '"OrderModel":' in system_prompt:
            model_name_for_example = "OrderModel"
            example_dict = {
                "_type": model_name_for_example,
                "order_ref": "MOCKE2E-ORD",
                "customer_name": "MockE2E Customer",
            }
        else:
            example_dict = {
                "_type": model_name_for_example,
                "fallback_e2e_mock_field": f"Data for {model_name_for_example}",
            }

        # Return the wrapped structure consistent with the new expected output format.
        return [{"entities": [example_dict]}]

    async def generate_json_revisions(
        self,
        system_prompt: str,
        user_prompt: str,
        model_schema_map: Dict[str, Type[SQLModel]],
        num_revisions: int,
        max_validation_retries_per_revision: int,
        analytics_collector: Optional[WorkflowAnalyticsCollector] = None,
    ) -> List[Dict[str, Any]]:
        self.last_max_validation_retries = max_validation_retries_per_revision

        if self.should_raise_exception:
            raise self.should_raise_exception

        if self.mock_responses:
            # If mock_responses is provided, return its content directly.
            # The new expected format is a list of entities, not a dict with 'results'.
            return self.mock_responses

        if self.return_products:
            revisions_data = [
                [
                    {
                        "_type": "ProductModel",
                        "_temp_id": "prod1",
                        "name": "Super Widget",
                        "description": "A great product.",
                        "version": 1,
                    },
                    {
                        "_type": "ProductModel",
                        "_temp_id": "prod2",
                        "name": "Mega Gadget",
                        "description": "Another fine item.",
                        "version": 1,
                    },
                ],
                [
                    {
                        "_type": "ProductModel",
                        "_temp_id": "prod1",
                        "name": "Super Widget",
                        "description": "An excellent product.",
                        "version": 1,
                    },
                    {
                        "_type": "ProductModel",
                        "_temp_id": "prod2",
                        "name": "Mega Gadget",
                        "description": "Another fine item.",
                        "version": 2,
                    },
                ],
                [
                    {
                        "_type": "ProductModel",
                        "_temp_id": "prod1",
                        "name": "Super Widget",
                        "description": "A great product.",
                        "version": 1,
                    },
                    {
                        "_type": "ProductModel",
                        "_temp_id": "prod2",
                        "name": "Mega Gadget",
                        "description": "Another fine item.",
                        "version": 1,
                    },
                ],
            ]
        else:
            revisions_data = [
                [
                    {
                        "_type": "OrderModel",
                        "_temp_id": "order1",
                        "order_ref": "ORD123",
                        "customer_name": "John Doe",
                        "order_entries_ref_ids": ["item1a", "item1b"],
                    },
                    {
                        "_type": "OrderModel",
                        "_temp_id": "order2",
                        "order_ref": "ORD456",
                        "customer_name": "Jane Smith",
                        "order_entries_ref_ids": ["item2a"],
                    },
                    {
                        "_type": "OrderItemModel",
                        "_temp_id": "item1a",
                        "product_sku": "SW001",
                        "quantity": 1,
                        "order_ref_id": "order1",
                    },
                    {
                        "_type": "OrderItemModel",
                        "_temp_id": "item1b",
                        "product_sku": "MG002",
                        "quantity": 1,
                        "order_ref_id": "order1",
                    },
                    {
                        "_type": "OrderItemModel",
                        "_temp_id": "item2a",
                        "product_sku": "HG003",
                        "quantity": 1,
                        "order_ref_id": "order2",
                    },
                ],
                [
                    {
                        "_type": "OrderModel",
                        "_temp_id": "order1",
                        "order_ref": "ORD123",
                        "customer_name": "John Doe",
                        "order_entries_ref_ids": ["item1a", "item1b"],
                    },
                    {
                        "_type": "OrderModel",
                        "_temp_id": "order2",
                        "order_ref": "ORD456",
                        "customer_name": "J. Smith",
                        "order_entries_ref_ids": ["item2a"],
                    },
                    {
                        "_type": "OrderItemModel",
                        "_temp_id": "item1a",
                        "product_sku": "SW001",
                        "quantity": 2,
                        "order_ref_id": "order1",
                    },
                    {
                        "_type": "OrderItemModel",
                        "_temp_id": "item1b",
                        "product_sku": "MG002",
                        "quantity": 1,
                        "order_ref_id": "order1",
                    },
                    {
                        "_type": "OrderItemModel",
                        "_temp_id": "item2a",
                        "product_sku": "HG003",
                        "quantity": 1,
                        "order_ref_id": "order2",
                    },
                ],
                [
                    {
                        "_type": "OrderModel",
                        "_temp_id": "order1",
                        "order_ref": "ORD123",
                        "customer_name": "John Doe",
                        "order_entries_ref_ids": ["item1a", "item1b"],
                    },
                    {
                        "_type": "OrderModel",
                        "_temp_id": "order2",
                        "order_ref": "ORD456",
                        "customer_name": "Jane Smith",
                        "order_entries_ref_ids": ["item2a"],
                    },
                    {
                        "_type": "OrderItemModel",
                        "_temp_id": "item1a",
                        "product_sku": "SW001",
                        "quantity": 1,
                        "order_ref_id": "order1",
                    },
                    {
                        "_type": "OrderItemModel",
                        "_temp_id": "item1b",
                        "product_sku": "MG002",
                        "quantity": 1,
                        "order_ref_id": "order1",
                    },
                    {
                        "_type": "OrderItemModel",
                        "_temp_id": "item2a",
                        "product_sku": "HG003",
                        "quantity": 1,
                        "order_ref_id": "order2",
                    },
                ],
            ]
        # The orchestrator now calls generate_json_revisions for each revision,
        # so we return one revision at a time.
        # This mock will return the same list for each call, which is sufficient for testing.
        return revisions_data[0]


# --- Mock LLM Client for SQLModelGenerator Tests (moved here) ---
class MockLLMClientSqlGen(BaseLLMClient):
    def __init__(
        self,
        api_key: str = "mock_sqlgen_api_key",
        model_name: str = "mock_sqlgen_model",
        base_url: Optional[str] = None,
        temperature: Optional[float] = None,
    ):
        # Using BaseLLMClient from extrai.core.workflow_orchestrator for consistency in this file
        super().__init__(
            api_key=api_key,
            model_name=model_name,
            base_url=base_url,
            temperature=temperature,
        )
        self.raw_json_outputs_to_return: List[Dict[str, Any]] = []
        self.should_raise_exception: Optional[Exception] = None
        self.call_count = 0
        self.last_system_prompt = ""
        self.last_user_prompt = ""
        self.last_num_revisions = 0
        self.last_target_json_schema: Optional[Dict[str, Any]] = None
        self.last_max_validation_retries = 0
        self.last_analytics_collector_passed: Optional[WorkflowAnalyticsCollector] = (
            None
        )
        self._execute_llm_call_mock = mock.AsyncMock(
            return_value="{}"
        )  # For BaseLLMClient

    async def _execute_llm_call(self, system_prompt: str, user_prompt: str) -> str:
        return await self._execute_llm_call_mock(system_prompt, user_prompt)

    async def generate_and_validate_raw_json_output(
        self,
        system_prompt: str,
        user_prompt: str,
        num_revisions: int,
        max_validation_retries_per_revision: int,
        target_json_schema: Optional[Dict[str, Any]] = None,
        analytics_collector: Optional[WorkflowAnalyticsCollector] = None,
    ) -> List[Dict[str, Any]]:
        self.call_count += 1
        self.last_system_prompt = system_prompt
        self.last_user_prompt = user_prompt
        self.last_num_revisions = num_revisions
        self.last_target_json_schema = target_json_schema
        self.last_max_validation_retries = max_validation_retries_per_revision
        self.last_analytics_collector_passed = analytics_collector

        if self.should_raise_exception:
            if analytics_collector:  # Simulate analytics recording on error
                if isinstance(self.should_raise_exception, LLMAPICallError):
                    analytics_collector.record_llm_api_call_failure()
                elif isinstance(self.should_raise_exception, LLMOutputParseError):
                    analytics_collector.record_llm_output_parse_error(
                        raw_content="mock_error_content"
                    )
                elif isinstance(self.should_raise_exception, LLMOutputValidationError):
                    analytics_collector.record_llm_output_validation_error(
                        parsed_json={}, errors=[]
                    )
            raise self.should_raise_exception

        return [json.loads(json.dumps(rev)) for rev in self.raw_json_outputs_to_return]

    def set_raw_json_outputs_to_return(self, outputs: List[Dict[str, Any]]):
        self.raw_json_outputs_to_return = outputs

    def set_should_raise_exception(self, exception: Optional[Exception]):
        self.should_raise_exception = exception
