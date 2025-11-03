# tests/core/test_db_writer.py
import unittest
from unittest.mock import MagicMock
import logging

from sqlalchemy.exc import SQLAlchemyError

# Adjust the import path based on your project structure
from extrai.core.db_writer import (
    persist_objects,
    DatabaseWriterError,
)

# Suppress logging output during tests for cleaner test runs
logging.disable(logging.CRITICAL)


class TestDBWriter(unittest.TestCase):
    def setUp(self):
        self.mock_session = MagicMock()
        self.mock_objects = [MagicMock(), MagicMock()]
        self.logger = logging.getLogger(self.__class__.__name__)

    def tearDown(self):
        # Re-enable logging if it was disabled for tests
        logging.disable(logging.NOTSET)

    def test_persist_objects_success(self):
        """Test successful persistence of objects."""
        persist_objects(self.mock_session, self.mock_objects, self.logger)
        self.mock_session.add_all.assert_called_once_with(self.mock_objects)
        self.mock_session.commit.assert_called_once()
        self.mock_session.rollback.assert_not_called()

    def test_persist_objects_empty_list(self):
        """Test persisting an empty list of objects."""
        persist_objects(self.mock_session, [], self.logger)
        self.mock_session.add_all.assert_not_called()
        self.mock_session.commit.assert_not_called()
        self.mock_session.rollback.assert_not_called()

    def test_persist_objects_commit_failure_sqlalchemy_error(self):
        """Test rollback on SQLAlchemyError during commit."""
        self.mock_session.commit.side_effect = SQLAlchemyError("Commit failed")

        with self.assertRaises(DatabaseWriterError) as context:
            persist_objects(self.mock_session, self.mock_objects, self.logger)

        self.mock_session.add_all.assert_called_once_with(self.mock_objects)
        self.mock_session.commit.assert_called_once()
        self.mock_session.rollback.assert_called_once()
        self.assertIn("Commit failed", str(context.exception))

    def test_persist_objects_commit_failure_generic_exception(self):
        """Test rollback on a generic Exception during commit."""
        self.mock_session.commit.side_effect = Exception("Unexpected error")
        self.mock_session.is_active = (
            True  # Assume session is active before rollback attempt
        )

        with self.assertRaises(DatabaseWriterError) as context:
            persist_objects(self.mock_session, self.mock_objects, self.logger)

        self.mock_session.add_all.assert_called_once_with(self.mock_objects)
        self.mock_session.commit.assert_called_once()
        self.mock_session.rollback.assert_called_once()
        self.assertIn("Unexpected error", str(context.exception))

    def test_persist_objects_rollback_failure(self):
        """Test scenario where rollback itself fails after a commit error."""
        self.mock_session.commit.side_effect = SQLAlchemyError("Commit failed")
        self.mock_session.rollback.side_effect = SQLAlchemyError("Rollback failed")

        with self.assertRaises(DatabaseWriterError) as context:
            persist_objects(self.mock_session, self.mock_objects, self.logger)

        self.mock_session.add_all.assert_called_once_with(self.mock_objects)
        self.mock_session.commit.assert_called_once()
        self.mock_session.rollback.assert_called_once()
        self.assertIn("Commit failed", str(context.exception))


if __name__ == "__main__":
    unittest.main()
