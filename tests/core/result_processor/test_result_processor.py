import unittest
from unittest.mock import MagicMock, patch
import logging
from typing import List, Dict, Any

from extrai.core.result_processor import ResultProcessor, HydrationError, WorkflowError
from sqlmodel import Session

class TestResultProcessor(unittest.TestCase):
    def setUp(self):
        self.mock_model_registry = MagicMock()
        self.mock_model_registry.model_map = {}
        self.mock_analytics_collector = MagicMock()
        self.mock_logger = MagicMock()
        self.result_processor = ResultProcessor(
            self.mock_model_registry,
            self.mock_analytics_collector,
            self.mock_logger
        )

    def test_hydrate_success(self):
        results = [{"_type": "test", "_temp_id": "1"}]
        self.mock_model_registry.model_map = {"test": MagicMock()}
        
        with patch('extrai.core.result_processor.SQLAlchemyHydrator') as MockHydrator:
            mock_hydrator_instance = MockHydrator.return_value
            mock_hydrator_instance.hydrate.return_value = [MagicMock()]
            
            hydrated = self.result_processor.hydrate(results)
            
            self.assertEqual(len(hydrated), 1)
            self.mock_analytics_collector.record_hydration_success.assert_called_with(1)
            mock_hydrator_instance.hydrate.assert_called_once()

    def test_hydrate_empty(self):
        results = []
        hydrated = self.result_processor.hydrate(results)
        self.assertEqual(hydrated, [])
        self.mock_analytics_collector.record_hydration_success.assert_not_called()

    def test_hydrate_failure(self):
        results = [{"_type": "test", "_temp_id": "1"}]
        
        with patch('extrai.core.result_processor.SQLAlchemyHydrator') as MockHydrator:
            mock_hydrator_instance = MockHydrator.return_value
            mock_hydrator_instance.hydrate.side_effect = Exception("Hydration failed")
            
            with self.assertRaises(HydrationError):
                self.result_processor.hydrate(results)
            
            self.mock_analytics_collector.record_hydration_failure.assert_called_once()

    def test_persist_success(self):
        objects = [MagicMock()]
        mock_session = MagicMock(spec=Session)
        
        with patch('extrai.core.result_processor.persist_objects') as mock_persist_objects:
            self.result_processor.persist(objects, mock_session)
            mock_persist_objects.assert_called_once_with(
                db_session=mock_session,
                objects_to_persist=objects,
                logger=self.mock_logger
            )

    def test_persist_empty(self):
        objects = []
        mock_session = MagicMock(spec=Session)
        
        with patch('extrai.core.result_processor.persist_objects') as mock_persist_objects:
            self.result_processor.persist(objects, mock_session)
            mock_persist_objects.assert_not_called()

    def test_persist_failure(self):
        objects = [MagicMock()]
        mock_session = MagicMock(spec=Session)
        
        with patch('extrai.core.result_processor.persist_objects') as mock_persist_objects:
            mock_persist_objects.side_effect = Exception("Persistence failed")
            
            with self.assertRaises(WorkflowError):
                self.result_processor.persist(objects, mock_session)
            
            mock_session.rollback.assert_called_once()

if __name__ == "__main__":
    unittest.main()
