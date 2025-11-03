# tests/core/test_workflow_orchestrator_e2e.py

import unittest
import json
from unittest import mock
import aiofiles

from sqlmodel import SQLModel, create_engine, Session as SQLModelSession, select

from extrai.core.workflow_orchestrator import WorkflowOrchestrator
from tests.core.helpers.orchestrator_test_models import (
    ProductModel,
    OrderModel,
    OrderItemModel,
)
from tests.core.helpers.mock_llm_clients import MockE2ELLMClient


class TestWorkflowOrchestratorE2E(unittest.IsolatedAsyncioTestCase):
    @mock.patch("extrai.core.workflow_orchestrator.discover_sqlmodels_from_root")
    @mock.patch("extrai.core.workflow_orchestrator.generate_llm_schema_from_models")
    def setUp(self, mock_generate_llm_schema, mock_discover_sqlmodels):
        self.mock_llm_client = MockE2ELLMClient()
        self.engine = create_engine("sqlite:///:memory:")
        SQLModel.metadata.create_all(self.engine)
        self.db_session: SQLModelSession = SQLModelSession(self.engine)

        # Define paths to E2E test data files
        # Assuming these files exist in the specified location relative to the test execution directory
        self.doc1_path = "tests/core/e2e_test_data/doc1.txt"
        self.doc2_path = "tests/core/e2e_test_data/doc2.txt"
        self.doc3_path = "tests/core/e2e_test_data/doc3.txt"
        self.doc4_path = "tests/core/e2e_test_data/doc4.txt"

        # Mock discovery and schema generation results
        self.mock_discovered_sqlmodels_e2e = [ProductModel, OrderModel, OrderItemModel]
        self.mock_prompt_llm_schema_str_e2e = json.dumps(
            {"schema_for_prompt_e2e": "mock_e2e_prompt_schema"}
        )

        mock_discover_sqlmodels.return_value = self.mock_discovered_sqlmodels_e2e
        mock_generate_llm_schema.return_value = self.mock_prompt_llm_schema_str_e2e

    def tearDown(self):
        self.db_session.close()
        SQLModel.metadata.drop_all(self.engine)

    async def test_e2e_workflow_with_mock_llm_and_consensus(self):
        orchestrator = WorkflowOrchestrator(
            root_sqlmodel_class=ProductModel,
            llm_client=self.mock_llm_client,
            num_llm_revisions=3,
            max_validation_retries_per_revision=1,
            consensus_threshold=0.51,
        )
        self.mock_llm_client.set_return_orders(
            False
        )  # Configure mock to return product data

        input_doc_paths = [self.doc1_path, self.doc2_path]
        input_strings = []
        for doc_path in input_doc_paths:
            # This assumes the test files are accessible from the CWD where tests are run.
            # If not, adjust path or ensure files are copied/created.
            try:
                async with aiofiles.open(doc_path, "r", encoding="utf-8") as f:
                    input_strings.append(await f.read())
            except FileNotFoundError:
                self.fail(
                    f"E2E test data file not found: {doc_path}. Ensure it's in the correct path relative to test execution."
                )

        hydrated_objects = await orchestrator.synthesize_and_save(
            input_strings, self.db_session
        )
        self.assertEqual(
            len(hydrated_objects), 2
        )  # Expecting 2 products based on mock LLM data

        statement = select(ProductModel).order_by(ProductModel.name)  # type: ignore[attr-defined]
        retrieved_products = self.db_session.exec(statement).all()

        self.assertEqual(len(retrieved_products), 2)
        # Assertions based on the consensus of MockE2ELLMClient's product data
        self.assertEqual(retrieved_products[0].name, "Mega Gadget")
        self.assertEqual(
            retrieved_products[0].version, 1
        )  # Version 1 agreed by 2/3 revisions
        self.assertEqual(retrieved_products[1].name, "Super Widget")
        self.assertEqual(
            retrieved_products[1].description, "A great product."
        )  # Agreed by 2/3

    async def test_e2e_workflow_with_nested_objects(self):
        orchestrator = WorkflowOrchestrator(
            root_sqlmodel_class=OrderModel,  # Root is OrderModel
            llm_client=self.mock_llm_client,
            num_llm_revisions=3,
            max_validation_retries_per_revision=1,
            consensus_threshold=0.51,
        )
        self.mock_llm_client.set_return_orders(
            True
        )  # Configure mock to return order data

        input_doc_paths = [self.doc3_path, self.doc4_path]
        input_strings = []
        for doc_path in input_doc_paths:
            try:
                async with aiofiles.open(doc_path, "r", encoding="utf-8") as f:
                    input_strings.append(await f.read())
            except FileNotFoundError:
                self.fail(
                    f"E2E test data file not found: {doc_path}. Ensure it's in the correct path relative to test execution."
                )

        # In the mock data for orders:
        # Order1: ORD123, John Doe. Item1a quantity is 1 (2/3 revisions), Item1b quantity is 1 (3/3)
        # Order2: ORD456, Jane Smith (2/3 revisions for name). Item2a quantity is 1 (3/3)
        # Total objects: 2 Orders + 3 OrderItems = 5
        hydrated_objects = await orchestrator.synthesize_and_save(
            input_strings, self.db_session
        )

        # Check total number of objects (Orders + OrderItems)
        # Based on MockE2ELLMClient, consensus should yield 2 orders and 3 items.
        self.assertEqual(len(hydrated_objects), 5)

        statement = select(OrderModel).order_by(OrderModel.order_ref)  # type: ignore[attr-defined]
        retrieved_orders = self.db_session.exec(statement).all()

        self.assertEqual(len(retrieved_orders), 2)

        order1 = next((o for o in retrieved_orders if o.order_ref == "ORD123"), None)
        self.assertIsNotNone(order1)
        if order1:
            self.assertEqual(order1.customer_name, "John Doe")  # Agreed by 3/3
            self.assertIsNotNone(order1.order_entries)
            self.assertEqual(len(order1.order_entries), 2)
            item1a = next(
                (item for item in order1.order_entries if item.product_sku == "SW001"),
                None,
            )
            self.assertIsNotNone(item1a)
            if item1a:
                self.assertEqual(item1a.quantity, 1)  # Agreed by 2/3 revisions

        order2 = next((o for o in retrieved_orders if o.order_ref == "ORD456"), None)
        self.assertIsNotNone(order2)
        if order2:
            self.assertEqual(
                order2.customer_name, "Jane Smith"
            )  # Agreed by 2/3 revisions
            self.assertIsNotNone(order2.order_entries)
            self.assertEqual(len(order2.order_entries), 1)
            item2a = next(
                (item for item in order2.order_entries if item.product_sku == "HG003"),
                None,
            )
            self.assertIsNotNone(item2a)
            if item2a:
                self.assertEqual(item2a.quantity, 1)  # Agreed by 3/3

    async def test_nested_extraction_e2e(self):
        # 1. Setup - This test now simulates extracting an Order with its OrderItems
        mock_llm_output = [
            {
                "_type": "OrderModel",
                "_temp_id": "order_1",
                "order_ref": "E2E-ORD-001",
                "customer_name": "Test Customer",
                "order_entries_ref_ids": ["item_1", "item_2"],
            },
            {
                "_type": "OrderItemModel",
                "_temp_id": "item_1",
                "product_sku": "LAPTOP-01",
                "quantity": 1,
                "order_ref_id": "order_1",
            },
            {
                "_type": "OrderItemModel",
                "_temp_id": "item_2",
                "product_sku": "MOUSE-01",
                "quantity": 1,
                "order_ref_id": "order_1",
            },
        ]

        # The mock client will return this same list for each of the 3 revisions.
        mock_llm_client = MockE2ELLMClient(mock_responses=mock_llm_output)

        orchestrator = WorkflowOrchestrator(
            root_sqlmodel_class=OrderModel,  # The root model is now OrderModel
            llm_client=mock_llm_client,
            num_llm_revisions=1,  # Set to 1 as the mock response is static
        )

        # 2. Execution
        input_docs = ["Some document about a customer order"]
        hydrated_objects = await orchestrator.synthesize(
            input_strings=input_docs, db_session_for_hydration=self.db_session
        )

        # 3. Assertions
        self.assertEqual(len(hydrated_objects), 3)

        order_instances = [
            obj for obj in hydrated_objects if isinstance(obj, OrderModel)
        ]
        item_instances = [
            obj for obj in hydrated_objects if isinstance(obj, OrderItemModel)
        ]

        self.assertEqual(len(order_instances), 1)
        self.assertEqual(len(item_instances), 2)

        order = order_instances[0]
        self.assertEqual(order.customer_name, "Test Customer")
        self.assertEqual(order.order_ref, "E2E-ORD-001")
        self.assertEqual(len(order.order_entries), 2)

        item_skus = {item.product_sku for item in order.order_entries}
        self.assertIn("LAPTOP-01", item_skus)
        self.assertIn("MOUSE-01", item_skus)

        for item in order.order_entries:
            self.assertEqual(item.order_ref_on_item, order)


if __name__ == "__main__":
    unittest.main()
