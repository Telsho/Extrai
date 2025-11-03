import unittest
import uuid
import sys
import io
from contextlib import contextmanager
from typing import List, Optional, Dict, Type
from sqlmodel import (
    Field,
    Relationship,
    SQLModel,
    create_engine,
    Session as SQLModelSession,
    select,
)

from extrai.core.sqlalchemy_hydrator import SQLAlchemyHydrator


# 1. Setup SQLModel Models
class Book(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    title: str
    author_id: Optional[int] = Field(
        default=None, foreign_key="author.id"
    )  # SQLModel uses table name 'author'
    author: Optional["Author"] = Relationship(back_populates="books")

    def __repr__(self):
        return f"<Book(id={self.id}, title='{self.title}', author_id={self.author_id})>"


class Author(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str
    books: List["Book"] = Relationship(back_populates="author")

    def __repr__(self):
        return f"<Author(id={self.id}, name='{self.name}')>"


# Update forward references
Book.model_rebuild()
Author.model_rebuild()


# New SQLModel Definitions for Testing
class ModelWithUUIDPK(SQLModel, table=True):
    uid: Optional[uuid.UUID] = Field(default=None, primary_key=True)
    name: str


class ModelWithStrPK(SQLModel, table=True):  # For string-based UUIDs
    sid: Optional[str] = Field(default=None, primary_key=True)
    name: str


class ModelWithFactoryUUIDPK(SQLModel, table=True):
    fid: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    name: str


ModelWithUUIDPK.model_rebuild()
ModelWithStrPK.model_rebuild()
ModelWithFactoryUUIDPK.model_rebuild()


class ModelWithoutPK(SQLModel):  # Not a table model
    name: str


# Removed ModelWithStrangeOptionalPK as it caused SQLModel collection errors
# class ModelWithStrangeOptionalPK(SQLModel, table=True):
#     # This type hint is unusual for a PK but should trigger the desired path in hydrator
#     strange_id: Optional[NoneType] = Field(default=None, primary_key=True)
#     name: str
# ModelWithStrangeOptionalPK.model_rebuild()


class TestSQLAlchemyHydrator(unittest.TestCase):
    def setUp(self):
        self.engine = create_engine("sqlite:///:memory:")
        SQLModel.metadata.create_all(self.engine)
        self.session = SQLModelSession(self.engine)
        self.hydrator = SQLAlchemyHydrator(self.session)
        # Correctly define the map with SQLModel classes
        self.model_sqlmodel_map: Dict[str, Type[SQLModel]] = {
            "author": Author,
            "book": Book,
            "uuid_model": ModelWithUUIDPK,
            "str_model": ModelWithStrPK,
            "factory_uuid_model": ModelWithFactoryUUIDPK,
            # "no_pk_model": ModelWithoutPK, # Not used in hydrate tests
            # "strange_pk_model": ModelWithStrangeOptionalPK, # Removed
        }

    def tearDown(self):
        SQLModel.metadata.drop_all(self.engine)
        self.session.close()

    @contextmanager
    def captured_stdout(self):
        """Context manager to capture stdout."""
        new_out = io.StringIO()
        old_out = sys.stdout
        try:
            sys.stdout = new_out
            yield new_out
        finally:
            sys.stdout = old_out

    def _hydrate_and_query(self, entities_list: List[Dict]) -> List[SQLModel]:
        """Helper to hydrate, commit, and query entities."""
        instances = self.hydrator.hydrate(entities_list, self.model_sqlmodel_map)
        self.session.commit()
        # Re-fetch instances from the database to ensure they were committed correctly
        refetched_instances = []
        for instance in instances:
            model_class = type(instance)
            pk_name = self.hydrator._get_primary_key_info(model_class).name
            if pk_name:
                pk_value = getattr(instance, pk_name)
                refetched = self.session.get(model_class, pk_value)
                if refetched:
                    refetched_instances.append(refetched)
        return refetched_instances

    def test_hydrate_simple_entities_no_relations(self):
        entities_list = [
            {"_type": "author", "_temp_id": "a1", "name": "Jane Doe"},
            {"_type": "book", "_temp_id": "b1", "title": "The Great Book"},
        ]
        instances = self._hydrate_and_query(entities_list)

        self.assertEqual(len(instances), 2)
        author_instance = next((i for i in instances if isinstance(i, Author)), None)
        book_instance = next((i for i in instances if isinstance(i, Book)), None)

        self.assertIsNotNone(author_instance)
        self.assertEqual(author_instance.name, "Jane Doe")
        self.assertIsNotNone(book_instance)
        self.assertEqual(book_instance.title, "The Great Book")
        self.assertIsNone(book_instance.author)

    def test_hydrate_one_to_many_and_many_to_one_relation(self):
        entities_list = [
            {"_type": "author", "_temp_id": "auth1", "name": "John Smith"},
            {
                "_type": "book",
                "_temp_id": "book1",
                "title": "Adventure Tales",
                "author_ref_id": "auth1",
            },
        ]
        instances = self._hydrate_and_query(entities_list)
        self.assertEqual(len(instances), 2)

        author = next((i for i in instances if isinstance(i, Author)), None)
        book = next((i for i in instances if isinstance(i, Book)), None)

        self.assertIsNotNone(author)
        self.assertIsNotNone(book)
        self.assertEqual(book.author, author)
        self.assertIn(book, author.books)
        self.assertEqual(book.author_id, author.id)

    def test_hydrate_one_to_many_from_one_side_ref_ids(self):
        entities_list = [
            {
                "_type": "author",
                "_temp_id": "alice",
                "name": "Alice Wonderland",
                "books_ref_ids": ["book_curiouser", "book_looking_glass"],
            },
            {
                "_type": "book",
                "_temp_id": "book_curiouser",
                "title": "Curiouser and Curiouser",
            },
            {
                "_type": "book",
                "_temp_id": "book_looking_glass",
                "title": "Through the Looking Glass",
            },
        ]
        instances = self._hydrate_and_query(entities_list)
        self.assertEqual(len(instances), 3)

        author = next((i for i in instances if isinstance(i, Author)), None)
        books = [i for i in instances if isinstance(i, Book)]
        book_curiouser = next(
            (b for b in books if b.title == "Curiouser and Curiouser"), None
        )
        book_looking_glass = next(
            (b for b in books if b.title == "Through the Looking Glass"), None
        )

        self.assertIsNotNone(author)
        self.assertEqual(len(author.books), 2)
        self.assertIn(book_curiouser, author.books)
        self.assertIn(book_looking_glass, author.books)
        self.assertEqual(book_curiouser.author, author)
        self.assertEqual(book_looking_glass.author, author)

    def test_hydrate_bidirectional_linking_multiple_books(self):
        entities_list = [
            {
                "_type": "author",
                "_temp_id": "multi_author",
                "name": "Multi Book Author",
            },
            {
                "_type": "book",
                "_temp_id": "book_first",
                "title": "First of Many",
                "author_ref_id": "multi_author",
            },
            {
                "_type": "book",
                "_temp_id": "book_second",
                "title": "Second of Many",
                "author_ref_id": "multi_author",
            },
        ]
        instances = self._hydrate_and_query(entities_list)
        self.assertEqual(len(instances), 3)

        author = next((i for i in instances if isinstance(i, Author)), None)
        books = [i for i in instances if isinstance(i, Book)]
        book1 = next((b for b in books if b.title == "First of Many"), None)
        book2 = next((b for b in books if b.title == "Second of Many"), None)

        self.assertIsNotNone(author)
        self.assertIsNotNone(book1)
        self.assertIsNotNone(book2)
        self.assertCountEqual(author.books, [book1, book2])
        self.assertEqual(book1.author, author)
        self.assertEqual(book2.author, author)

    def test_hydration_raises_on_invalid_input(self):
        test_cases = [
            (
                "missing_temp_id",
                [{"_type": "author", "name": "No ID"}],
                ValueError,
                "missing '_temp_id' or '_type'",
            ),
            (
                "missing_type",
                [{"_temp_id": "a1", "name": "No Type"}],
                ValueError,
                "missing '_temp_id' or '_type'",
            ),
            (
                "unknown_type",
                [{"_type": "unknown_type", "_temp_id": "u1"}],
                ValueError,
                "No SQLModel class found .* for type: 'unknown_type'",
            ),
            (
                "duplicate_temp_id",
                [
                    {"_type": "author", "_temp_id": "dup1", "name": "First"},
                    {"_type": "author", "_temp_id": "dup1", "name": "Second"},
                ],
                ValueError,
                "Duplicate _temp_id 'dup1' found",
            ),
            (
                "input_not_a_list",
                {"some_key": "some_value"},
                TypeError,
                "Input 'entities_list' must be a list",
            ),
            (
                "list_with_non_dict_item",
                [
                    {"_type": "author", "_temp_id": "a1", "name": "Valid Author"},
                    "not_a_dict",
                ],
                ValueError,
                "All items in 'entities_list' must be dictionaries",
            ),
            (
                "model_validation_error",
                [{"_type": "author", "_temp_id": "a_invalid_data", "name": 12345}],
                ValueError,
                "Failed to instantiate/validate SQLModel 'author' for _temp_id 'a_invalid_data'",
            ),
        ]

        for name, entities_list, error, regex in test_cases:
            with self.subTest(name=name):
                with self.assertRaisesRegex(error, regex):
                    self.hydrator.hydrate(entities_list, self.model_sqlmodel_map)

    def test_hydrate_broken_ref_id(self):
        entities_list = [
            {
                "_type": "book",
                "_temp_id": "b_broken",
                "title": "Book with Broken Link",
                "author_ref_id": "non_existent_author",
            }
        ]
        with self.captured_stdout() as captured:
            instances = self._hydrate_and_query(entities_list)

        self.assertEqual(len(instances), 1)
        book = instances[0]
        self.assertEqual(book.title, "Book with Broken Link")
        self.assertIsNone(book.author)
        self.assertIn(
            "Warning: Referenced _temp_id 'non_existent_author' for relation 'author'",
            captured.getvalue(),
        )

    def test_hydrate_null_ref_id(self):
        entities_list = [
            {
                "_type": "book",
                "_temp_id": "b_null_author",
                "title": "Book with Null Author",
                "author_ref_id": None,
            }
        ]
        instances = self._hydrate_and_query(entities_list)
        self.assertEqual(len(instances), 1)
        book = instances[0]
        self.assertEqual(book.title, "Book with Null Author")
        self.assertIsNone(book.author)

    def test_hydrate_empty_entities_list(self):
        entities_list = []  # Directly an empty list
        instances = self.hydrator.hydrate(entities_list, self.model_sqlmodel_map)
        self.assertEqual(len(instances), 0)

    def test_primary_key_generation_strategies(self):
        with self.subTest("int_pk_auto_increment_and_ignore_input_id"):
            entities_list = [
                {"_type": "author", "_temp_id": "a1", "name": "Author AutoInc"},
                {
                    "_type": "author",
                    "_temp_id": "a2",
                    "id": 999,
                    "name": "Author IgnoreInputID",
                },
            ]
            self.hydrator.hydrate(entities_list, self.model_sqlmodel_map)
            self.session.commit()
            author1 = self.session.exec(
                select(Author).where(Author.name == "Author AutoInc")
            ).first()
            author2 = self.session.exec(
                select(Author).where(Author.name == "Author IgnoreInputID")
            ).first()
            self.assertIsNotNone(author1)
            self.assertIsInstance(author1.id, int)
            self.assertIsNotNone(author2)
            self.assertIsInstance(author2.id, int)
            self.assertNotEqual(author2.id, 999)

        with self.subTest("uuid_pk_generated_by_hydrator"):
            entities_list = [
                {"_type": "uuid_model", "_temp_id": "u1", "name": "UUID Test 1"},
                {
                    "_type": "uuid_model",
                    "_temp_id": "u2",
                    "uid": str(uuid.uuid4()),
                    "name": "UUID Test 2 IgnoreInput",
                },
            ]
            original_uid = entities_list[1]["uid"]
            self.hydrator.hydrate(entities_list, self.model_sqlmodel_map)
            instance1 = self.hydrator.temp_id_to_instance_map["u1"]
            instance2 = self.hydrator.temp_id_to_instance_map["u2"]
            self.assertIsInstance(instance1.uid, uuid.UUID)
            self.assertIsInstance(instance2.uid, uuid.UUID)
            self.assertNotEqual(instance2.uid, uuid.UUID(original_uid))

        with self.subTest("str_pk_generated_by_hydrator_as_uuid_string"):
            entities_list = [
                {"_type": "str_model", "_temp_id": "s1", "name": "Str UUID Test 1"},
                {
                    "_type": "str_model",
                    "_temp_id": "s2",
                    "sid": "manual-id",
                    "name": "Str UUID Test 2 IgnoreInput",
                },
            ]
            self.hydrator.hydrate(entities_list, self.model_sqlmodel_map)
            instance1 = self.hydrator.temp_id_to_instance_map["s1"]
            instance2 = self.hydrator.temp_id_to_instance_map["s2"]
            self.assertIsInstance(instance1.sid, str)
            self.assertIsInstance(instance2.sid, str)
            self.assertNotEqual(instance2.sid, "manual-id")
            try:
                uuid.UUID(instance1.sid, version=4)
                uuid.UUID(instance2.sid, version=4)
            except ValueError:
                self.fail("Generated string ID is not a valid UUIDv4 string.")

        with self.subTest("uuid_pk_generated_by_model_factory"):
            entities_list = [
                {
                    "_type": "factory_uuid_model",
                    "_temp_id": "f1",
                    "name": "Factory UUID Test 1",
                },
                {
                    "_type": "factory_uuid_model",
                    "_temp_id": "f2",
                    "fid": str(uuid.uuid4()),
                    "name": "Factory UUID Test 2 IgnoreInput",
                },
            ]
            original_fid = entities_list[1]["fid"]
            self.hydrator.hydrate(entities_list, self.model_sqlmodel_map)
            instance1 = self.hydrator.temp_id_to_instance_map["f1"]
            instance2 = self.hydrator.temp_id_to_instance_map["f2"]
            self.assertIsInstance(instance1.fid, uuid.UUID)
            self.assertIsInstance(instance2.fid, uuid.UUID)
            self.assertNotEqual(instance2.fid, uuid.UUID(original_fid))

    def test_hydrate_two_entities_same_input_int_pk(self):
        entities_list = [
            {
                "_type": "author",
                "_temp_id": "auth_dup_id1",
                "id": 1,
                "name": "Author Same ID 1",
            },
            {
                "_type": "author",
                "_temp_id": "auth_dup_id2",
                "id": 1,
                "name": "Author Same ID 2",
            },
        ]
        instances = self.hydrator.hydrate(entities_list, self.model_sqlmodel_map)
        self.session.commit()

        self.assertEqual(len(instances), 2)

        # Retrieve instances from the hydrator's map or query them
        # Using names for retrieval as temp_ids are internal to hydrator processing for this check
        author1_db = self.session.exec(
            select(Author).where(Author.name == "Author Same ID 1")
        ).first()
        author2_db = self.session.exec(
            select(Author).where(Author.name == "Author Same ID 2")
        ).first()

        self.assertIsNotNone(
            author1_db, "First author with same input ID should be created"
        )
        self.assertIsNotNone(
            author2_db, "Second author with same input ID should be created"
        )

        self.assertEqual(
            author1_db.id,
            1,
            "The first author instance should receive ID 1 from auto-increment.",
        )

        self.assertEqual(
            author2_db.id,
            2,
            "The second author instance should receive ID 2 from auto-increment.",
        )

        # Verify they are integers (already implied by assertEqual with int, but good for clarity)
        self.assertIsInstance(author1_db.id, int)
        self.assertIsInstance(author2_db.id, int)

    # Removed test_hydrate_model_with_strange_optional_pk_type as the model was problematic for SQLModel itself

    def test_hydrate_ref_ids_with_invalid_value_type(self):
        entities_list = [
            {
                "_type": "author",
                "_temp_id": "auth_invalid_ref_ids",
                "name": "Author Invalid Ref IDs",
                "books_ref_ids": "not_a_list",
            },
        ]
        with self.captured_stdout() as captured:
            instances = self._hydrate_and_query(entities_list)

        author = next((i for i in instances if isinstance(i, Author)), None)
        self.assertIsNotNone(author)
        self.assertEqual(len(author.books), 0)
        self.assertIn(
            "Warning: Value for 'books_ref_ids' on instance 'auth_invalid_ref_ids' is not a list",
            captured.getvalue(),
        )

    def test_hydrate_ref_ids_list_with_invalid_item_type_or_missing_ref(self):
        with self.subTest("invalid_item_type"):
            entities_list = [
                {
                    "_type": "author",
                    "_temp_id": "auth1",
                    "name": "Author A",
                    "books_ref_ids": ["book1", 123],
                },
                {"_type": "book", "_temp_id": "book1", "title": "Book 1"},
            ]
            with self.captured_stdout() as captured:
                self._hydrate_and_query(entities_list)
            self.assertIn(
                "Warning: Referenced _temp_id '123' in list for relation 'books' on instance 'auth1' (type: author) not found or invalid type.",
                captured.getvalue(),
            )

        with self.subTest("missing_ref"):
            entities_list = [
                {
                    "_type": "author",
                    "_temp_id": "auth2",
                    "name": "Author B",
                    "books_ref_ids": ["book2", "non_existent"],
                },
                {"_type": "book", "_temp_id": "book2", "title": "Book 2"},
            ]
            with self.captured_stdout() as captured:
                self._hydrate_and_query(entities_list)
            self.assertIn(
                "Warning: Referenced _temp_id 'non_existent' in list for relation 'books' on instance 'auth2' (type: author) not found or invalid type.",
                captured.getvalue(),
            )

    def test_no_pk_coverage(self):
        """
        Directly tests the code paths for models without a primary key.
        """
        # This covers the return path in _get_primary_key_info
        pk_info = self.hydrator._get_primary_key_info(ModelWithoutPK)
        self.assertIsNone(pk_info.name)
        self.assertIsNone(pk_info.type)
        self.assertFalse(pk_info.has_uuid_factory)

        # This covers the return path in _generate_pk_if_needed
        instance = ModelWithoutPK(name="test")
        # This should run without error and do nothing.
        self.hydrator._generate_pk_if_needed(instance, ModelWithoutPK)


if __name__ == "__main__":
    unittest.main(argv=["first-arg-is-ignored"], exit=False)
