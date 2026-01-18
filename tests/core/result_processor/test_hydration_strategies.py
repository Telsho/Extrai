import unittest
import unittest.mock
import logging
from typing import List, Optional, Dict
from sqlmodel import Relationship, SQLModel, Field, create_engine, Session
from extrai.core.result_processor import (
    ResultProcessor,
    DirectHydrator,
    SQLAlchemyHydrator,
)
from extrai.core.model_registry import ModelRegistry


# Define models
class StrategyAuthor(SQLModel, table=True):
    __tablename__ = "strategy_author"
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str
    nested_books: List["StrategyNestedBook"] = Relationship(back_populates="author")


class StrategyBook(SQLModel, table=True):
    __tablename__ = "strategy_book"
    id: Optional[int] = Field(default=None, primary_key=True)
    title: str
    author_id: Optional[int] = Field(default=None, foreign_key="strategy_author.id")


class StrategyNestedBook(SQLModel, table=True):
    __tablename__ = "strategy_nested_book"
    id: Optional[int] = Field(default=None, primary_key=True)
    title: str
    author_id: Optional[int] = Field(default=None, foreign_key="strategy_author.id")
    author: Optional[StrategyAuthor] = Relationship(back_populates="nested_books")


class StrategyLibrary(SQLModel):
    books: List[StrategyNestedBook]


# Update forward refs
StrategyAuthor.model_rebuild()
StrategyNestedBook.model_rebuild()


class TestResultProcessorStrategies(unittest.TestCase):
    def setUp(self):
        self.engine = create_engine("sqlite:///:memory:")
        SQLModel.metadata.create_all(self.engine)
        self.session = Session(self.engine)
        self.logger = logging.getLogger("test")

        # Mock ModelRegistry
        self.model_registry = unittest.mock.Mock(spec=ModelRegistry)
        self.model_registry.model_map = {
            "author": StrategyAuthor,
            "book": StrategyBook,
            "library": StrategyLibrary,
            "nested_book": StrategyNestedBook,
        }
        self.model_registry.root_model = StrategyAuthor  # Default

        self.processor = ResultProcessor(
            self.model_registry, unittest.mock.Mock(), self.logger
        )

    def tearDown(self):
        self.session.close()

    def test_auto_detect_sqlalchemy_hydrator(self):
        data = [{"_type": "author", "_temp_id": "1", "name": "Test"}]

        # Mock hydrators to verify which one is called
        with unittest.mock.patch(
            "extrai.core.result_processor.SQLAlchemyHydrator"
        ) as MockSQLHydrator:
            mock_instance = MockSQLHydrator.return_value
            mock_instance.hydrate.return_value = []

            self.processor.hydrate(data, self.session)

            MockSQLHydrator.assert_called()

    def test_auto_detect_direct_hydrator_no_temp_id(self):
        data = [{"name": "Test"}]  # No _temp_id

        with unittest.mock.patch(
            "extrai.core.result_processor.DirectHydrator"
        ) as MockDirectHydrator:
            mock_instance = MockDirectHydrator.return_value
            mock_instance.hydrate.return_value = []

            self.processor.hydrate(data, self.session)

            MockDirectHydrator.assert_called()

    def test_auto_detect_direct_hydrator_explicit_default(self):
        data = [{"name": "Test"}]

        with unittest.mock.patch(
            "extrai.core.result_processor.DirectHydrator"
        ) as MockDirectHydrator:
            mock_instance = MockDirectHydrator.return_value
            mock_instance.hydrate.return_value = []

            self.processor.hydrate(data, self.session, default_model_type="author")

            MockDirectHydrator.assert_called()

    def test_direct_hydrator_with_type(self):
        hydrator = DirectHydrator(self.session, self.logger)
        data = [{"_type": "author", "name": "Direct Author"}]

        results = hydrator.hydrate(data, self.model_registry.model_map)

        self.assertEqual(len(results), 1)
        self.assertIsInstance(results[0], StrategyAuthor)
        self.assertEqual(results[0].name, "Direct Author")

    def test_direct_hydrator_default_model(self):
        hydrator = DirectHydrator(self.session, self.logger)
        data = [{"name": "Default Author"}]  # No _type

        results = hydrator.hydrate(
            data, self.model_registry.model_map, default_model_class=StrategyAuthor
        )

        self.assertEqual(len(results), 1)
        self.assertIsInstance(results[0], StrategyAuthor)
        self.assertEqual(results[0].name, "Default Author")

    def test_direct_hydrator_nested(self):
        hydrator = DirectHydrator(self.session, self.logger)
        data = [
            {
                "_type": "nested_book",
                "title": "Nested Title",
                "author": {"name": "Nested Author"},
            }
        ]

        results = hydrator.hydrate(data, self.model_registry.model_map)

        self.assertEqual(len(results), 1)
        self.assertIsInstance(results[0], StrategyNestedBook)
        self.assertEqual(results[0].title, "Nested Title")
        self.assertIsInstance(results[0].author, StrategyAuthor)
        self.assertEqual(results[0].author.name, "Nested Author")
