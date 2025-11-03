import unittest
import enum
import json
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Set,
    Union,
    ForwardRef,
    get_origin as typing_get_origin,
    get_args as typing_get_args,
)
import datetime
from unittest.mock import patch, MagicMock

from sqlalchemy import create_engine, Column, Integer, String, ForeignKey
from sqlalchemy.exc import NoInspectionAvailable
from sqlalchemy.orm import (
    relationship as sa_relationship,
    declarative_base,
    Mapped,
    mapped_column,
    RelationshipProperty,
)
from sqlalchemy.orm.attributes import InstrumentedAttribute
from sqlmodel import SQLModel, Field, Relationship

from extrai.core.schema_inspector import (
    inspect_sqlalchemy_model,
    generate_llm_schema_from_models,
    _get_python_type_str_from_pydantic_annotation,
    _map_sql_type_to_llm_type,
    _collect_all_sqla_models_recursively,
    discover_sqlmodels_from_root,
    _get_involved_foreign_keys,
    _process_relationship_for_llm_schema,
)

from tests.core.helpers.orchestrator_test_models import (
    Base,
    Department,
    Employee,
    Project,
    Member,
    ArticleScenarioModel,
    TableModel,
    ModelWithColumnProperty,
    ModelWithNonSerializableColInfo,
    ModelRelatedToNonSerializable,
    ModelWithNonSerializableRelInfo,
    ModelWithNonSerializableTableInfo,
    PlainSQLAlchemyModelWithPydanticHints,
    ModelWithCustomColType,
    FKParent,
    FKParentDirect,
    ViewOnlyParent,
)


class SQLAlchemyBaseTestCase(unittest.TestCase):
    engine = None

    @classmethod
    def setUpClass(cls):
        cls.engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(cls.engine, checkfirst=True)  # Uses imported Base
        SQLModel.metadata.create_all(
            cls.engine, checkfirst=True
        )  # Uses imported SQLModel


class TestInspectSqlalchemyModel(SQLAlchemyBaseTestCase):
    def test_employee_model_inspection_with_metadata(self):
        schema = inspect_sqlalchemy_model(Employee)
        self.assertEqual(
            schema["comment"], "Stores detailed information about company employees."
        )
        self.assertEqual(schema["info_dict"], {"confidentiality": "high"})
        self.assertEqual(schema["columns"]["id"]["comment"], "Unique Employee ID (PK)")
        self.assertEqual(
            schema["columns"]["email"]["info_dict"],
            {"validation_rule": "standard_email_format"},
        )
        self.assertIsNone(schema["columns"]["email"]["comment"])
        self.assertEqual(
            schema["relationships"]["department"]["info_dict"],
            {"description": "The department this employee is assigned to."},
        )

    def test_department_model_inspection_with_metadata(self):
        schema = inspect_sqlalchemy_model(Department)
        self.assertEqual(schema["comment"], "Stores all company departments.")
        self.assertEqual(schema["info_dict"], {"schema_version": "1.2"})
        self.assertEqual(
            schema["columns"]["id"]["comment"], "Unique Department ID (PK)"
        )
        self.assertEqual(schema["columns"]["id"]["info_dict"], {"pk_strategy": "auto"})
        self.assertEqual(
            schema["columns"]["name"]["comment"], "Official name of the department."
        )
        self.assertEqual(
            schema["relationships"]["employees"]["info_dict"],
            {"relationship_detail": "All employees belonging to this department."},
        )

    def test_employee_with_nested_department_and_loops(self):
        schema = inspect_sqlalchemy_model(Employee)
        self.assertEqual(schema["table_name"], "employees")
        dept_rel_info = schema["relationships"]["department"]
        nested_dept_schema = dept_rel_info["nested_schema"]
        self.assertEqual(nested_dept_schema["table_name"], "departments")
        dept_employees_rel = nested_dept_schema["relationships"]["employees"]
        employee_recursion_schema = dept_employees_rel["nested_schema"]
        self.assertEqual(
            employee_recursion_schema["recursion_detected_for_type"], "Employee"
        )
        manager_rel_info = schema["relationships"]["manager"]
        manager_recursion_schema = manager_rel_info["nested_schema"]
        self.assertEqual(
            manager_recursion_schema["recursion_detected_for_type"], "Employee"
        )

    def test_department_with_nested_employees(self):
        schema = inspect_sqlalchemy_model(Department)
        emp_rel_info = schema["relationships"]["employees"]
        nested_emp_schema = emp_rel_info["nested_schema"]
        self.assertEqual(nested_emp_schema["table_name"], "employees")
        emp_dept_rel = nested_emp_schema["relationships"]["department"]
        department_recursion_schema = emp_dept_rel["nested_schema"]
        self.assertEqual(
            department_recursion_schema["recursion_detected_for_type"], "Department"
        )

    def test_project_with_nested_members_and_loops_m2m(self):
        schema = inspect_sqlalchemy_model(Project)
        members_rel_info = schema["relationships"]["members"]
        self.assertEqual(members_rel_info["related_model_name"], "Member")
        nested_member_schema = members_rel_info["nested_schema"]
        member_projects_rel = nested_member_schema["relationships"]["projects"]
        project_recursion_schema = member_projects_rel["nested_schema"]
        self.assertEqual(
            project_recursion_schema["recursion_detected_for_type"], "Project"
        )
        self.assertEqual(
            members_rel_info["info_dict"],
            {"description": "Members participating in this project."},
        )
        self.assertEqual(members_rel_info["secondary_table_name"], "project_member")

    def test_foreign_key_details_still_correct(self):
        emp_schema = inspect_sqlalchemy_model(Employee)
        dept_rel = emp_schema["relationships"]["department"]
        self.assertIn(
            "employees.department_id", dept_rel["foreign_key_constraints_involved"]
        )
        proj_schema = inspect_sqlalchemy_model(Project)
        members_rel = proj_schema["relationships"]["members"]
        self.assertIn(
            "project_member.project_id", members_rel["foreign_key_constraints_involved"]
        )
        self.assertIn(
            "project_member.member_id", members_rel["foreign_key_constraints_involved"]
        )

    def test_table_support_screws_deep_inspection(self):
        schema = inspect_sqlalchemy_model(
            TableModel
        )  # Updated from Table to TableModel
        self.assertEqual(schema["table_name"], "tables")
        self.assertEqual(schema["comment"], "Stores information about tables.")

        supports_rel_info = schema["relationships"]["supports"]
        self.assertEqual(supports_rel_info["related_model_name"], "Support")
        self.assertEqual(
            supports_rel_info["info_dict"]["description"],
            "List of supports for this table",
        )
        nested_support_schema = supports_rel_info["nested_schema"]
        self.assertEqual(nested_support_schema["table_name"], "supports")
        self.assertEqual(
            nested_support_schema["comment"], "Stores information about supports."
        )
        self.assertEqual(
            nested_support_schema["columns"]["name"]["comment"], "Name of the support"
        )

        screws_rel_info = nested_support_schema["relationships"]["screws_list"]
        self.assertEqual(screws_rel_info["related_model_name"], "Screws")
        self.assertEqual(
            screws_rel_info["info_dict"]["description"],
            "List of screws for this support",
        )
        nested_screws_schema = screws_rel_info["nested_schema"]
        self.assertEqual(nested_screws_schema["table_name"], "screws")
        self.assertEqual(
            nested_screws_schema["comment"], "Stores information about screws."
        )
        self.assertEqual(
            nested_screws_schema["columns"]["size"]["comment"],
            "Size of the screw (e.g., M5x20)",
        )

        screw_support_rel_info = nested_screws_schema["relationships"]["support"]
        screw_support_recursion_schema = screw_support_rel_info["nested_schema"]
        self.assertEqual(
            screw_support_recursion_schema["recursion_detected_for_type"], "Support"
        )

        support_table_rel_info = nested_support_schema["relationships"]["table"]
        support_table_recursion_schema = support_table_rel_info["nested_schema"]
        self.assertEqual(
            support_table_recursion_schema["recursion_detected_for_type"], "TableModel"
        )  # Updated from "Table"

    def test_skip_column_property_in_columns_list(self):
        schema = inspect_sqlalchemy_model(ModelWithColumnProperty)
        self.assertNotIn(
            "error", schema, f"Schema inspection failed: {schema.get('error')}"
        )
        self.assertIn("id", schema["columns"])
        self.assertIn("data", schema["columns"])
        self.assertNotIn(
            "data_length",
            schema["columns"],
            "Column properties should not be in the 'columns' dict",
        )

    @patch("extrai.core.schema_inspector.inspect")
    def test_relationship_not_relationship_property(self, mock_inspect_outer):
        local_base_for_rel_test: Any = declarative_base()

        class ModelForNonRelPropTest(local_base_for_rel_test):
            __tablename__ = "model_non_rel_prop_test_local"
            id: Mapped[int] = mapped_column(primary_key=True)

        ModelForNonRelPropTest.metadata.create_all(self.engine, checkfirst=True)

        class MockNonRelProperty:
            key = "fake_non_rel"

        mock_inspector_instance = MagicMock()
        mock_id_attr = MagicMock(spec=InstrumentedAttribute)
        mock_id_attr.expression = Column(Integer)
        mock_id_attr.key = "id"
        mock_inspector_instance.column_attrs = [mock_id_attr]

        mock_inspector_instance.relationships = {"fake_non_rel": MockNonRelProperty()}
        mock_inspector_instance.selectable = ModelForNonRelPropTest.__table__
        mock_inspector_instance.mapper = MagicMock()
        mock_inspector_instance.mapper.class_ = ModelForNonRelPropTest

        mock_inspect_outer.return_value = mock_inspector_instance

        schema = inspect_sqlalchemy_model(ModelForNonRelPropTest)

        self.assertNotIn(
            "error", schema, f"Schema inspection failed: {schema.get('error')}"
        )
        self.assertNotIn(
            "fake_non_rel",
            schema.get("relationships", {}),
            "Items not of type RelationshipProperty should be skipped.",
        )

    def test_inspector_is_none_error(self):
        class SomeModelInternal(Base):
            __tablename__ = "some_model_for_inspector_none_internal"
            id: Mapped[int] = mapped_column(primary_key=True)

        SomeModelInternal.metadata.create_all(self.engine, checkfirst=True)

        with patch(
            "extrai.core.schema_inspector.inspect",
            return_value=None,
        ) as mock_inspect:
            schema = inspect_sqlalchemy_model(SomeModelInternal)
            mock_inspect.assert_called_once_with(SomeModelInternal)
            self.assertIn("error", schema)
            self.assertIn(f"Inspector is None for {SomeModelInternal}", schema["error"])

    def test_column_python_type_access_error(self):
        schema = inspect_sqlalchemy_model(ModelWithCustomColType)
        self.assertNotIn(
            "error", schema, f"Schema inspection failed: {schema.get('error')}"
        )
        self.assertIn("custom_field", schema["columns"])
        self.assertEqual(
            schema["columns"]["custom_field"]["python_type"],
            "unknown_error_accessing_type",
        )

    def test_relationship_fk_path_synchronize_pairs(self):
        schema = inspect_sqlalchemy_model(FKParent)
        self.assertNotIn(
            "error", schema, f"Schema inspection failed: {schema.get('error')}"
        )

        children_sync_rel = schema["relationships"].get("children_sync")
        self.assertIsNotNone(children_sync_rel, "children_sync relationship not found")
        self.assertIn(
            "fk_child_sync.parent_id_col",
            children_sync_rel["foreign_key_constraints_involved"],
        )

    def test_relationship_fk_path_direct_foreign_keys_attr(self):
        schema = inspect_sqlalchemy_model(FKParentDirect)
        self.assertNotIn(
            "error", schema, f"Schema inspection failed: {schema.get('error')}"
        )

        children_direct_rel = schema["relationships"].get("children_direct")
        self.assertIsNotNone(
            children_direct_rel, "children_direct relationship not found"
        )
        self.assertIn(
            "fk_child_direct.parent_fk_col_name",
            children_direct_rel["foreign_key_constraints_involved"],
        )

    def test_relationship_fk_path_viewonly_rel_foreign_keys_attr(self):
        schema = inspect_sqlalchemy_model(ViewOnlyParent)
        self.assertNotIn(
            "error", schema, f"Schema inspection failed: {schema.get('error')}"
        )

        children_rel = schema["relationships"].get("children")
        self.assertIsNotNone(
            children_rel, "children relationship not found in ViewOnlyParent schema"
        )
        self.assertIn(
            "viewonly_child.parent_id",
            children_rel["foreign_key_constraints_involved"],
            "Expected 'viewonly_child.parent_id' to be in foreign_key_constraints_involved for viewonly relationship.",
        )

    def test_column_python_type_attribute_error(self):
        # Covers line 265: python_type_name = 'unknown_no_python_type_attr'
        local_base_for_attr_error: Any = declarative_base()

        class ModelForAttrError(local_base_for_attr_error):
            __tablename__ = "model_for_attr_error"
            id: Mapped[int] = mapped_column(primary_key=True)
            problematic_col: Mapped[str] = mapped_column(String)

        ModelForAttrError.metadata.create_all(self.engine, checkfirst=True)

        mock_inspector = MagicMock()
        mock_inspector.selectable = ModelForAttrError.__table__
        mock_inspector.mapper = MagicMock()
        mock_inspector.mapper.class_ = ModelForAttrError

        # Mock column attribute and its expression (the Column object)
        mock_col_attr = MagicMock(spec=InstrumentedAttribute)
        mock_col_attr.key = "problematic_col"

        mock_column_obj = MagicMock(spec=Column)
        mock_column_obj.name = "problematic_col"
        mock_column_obj.unique = False
        mock_column_obj.table = ModelForAttrError.__table__
        mock_column_obj.primary_key = False
        mock_column_obj.nullable = True
        mock_column_obj.comment = None
        mock_column_obj.info = {}
        mock_column_obj.foreign_keys = set()

        # Mock the 'type' attribute of the column object
        mock_type_obj = MagicMock()
        # Make accessing 'python_type' on mock_type_obj raise AttributeError
        del mock_type_obj.python_type  # Ensure it doesn't exist
        type_prop = MagicMock(
            side_effect=AttributeError("Simulated AttributeError for python_type")
        )
        type(mock_type_obj).python_type = type_prop  # Mocking the property itself

        mock_column_obj.type = mock_type_obj
        mock_col_attr.expression = mock_column_obj

        mock_inspector.column_attrs = [mock_col_attr]
        mock_inspector.relationships = {}  # No relationships for this test

        with patch(
            "extrai.core.schema_inspector.inspect",
            return_value=mock_inspector,
        ):
            schema = inspect_sqlalchemy_model(ModelForAttrError)

        self.assertNotIn(
            "error", schema, f"Schema inspection failed: {schema.get('error')}"
        )
        self.assertIn("problematic_col", schema["columns"])
        self.assertEqual(
            schema["columns"]["problematic_col"]["python_type"],
            "unknown_no_python_type_attr",
        )

    def test_relationship_with_no_foreign_keys(self):
        """
        Tests that a relationship with no discernible foreign keys results in an empty set
        for 'foreign_key_constraints_involved' when testing the public API.
        """
        # Define models locally to avoid polluting the global Base metadata
        local_base: Any = declarative_base()

        class LocalSomeOtherModel(local_base):
            __tablename__ = "local_some_other_model"
            id: Mapped[int] = mapped_column(primary_key=True)

        class LocalModelWithNoFKs(local_base):
            __tablename__ = "local_model_no_fks"
            id: Mapped[int] = mapped_column(primary_key=True)
            related = sa_relationship(
                "LocalSomeOtherModel",
                viewonly=True,
                primaryjoin="foreign(LocalModelWithNoFKs.id) == remote(LocalSomeOtherModel.id)",
            )

        # Create tables locally for this test
        local_base.metadata.create_all(self.engine)

        schema = inspect_sqlalchemy_model(LocalModelWithNoFKs)
        self.assertNotIn(
            "error", schema, f"Schema inspection failed: {schema.get('error')}"
        )
        self.assertIn("related", schema["relationships"])
        self.assertEqual(
            schema["relationships"]["related"]["foreign_key_constraints_involved"], []
        )

    def test_direct_foreign_keys_and_sync_pairs_coverage(self):
        """
        Covers the remaining lines using mocks to ensure the exact conditions are met.
        """
        with self.subTest("Test _get_fks_from_direct_foreign_keys"):
            mock_rel_prop = MagicMock(spec=RelationshipProperty)
            mock_fk_col = MagicMock(spec=Column)
            mock_fk_col.__str__.return_value = "mock_table.fk_col"
            # Configure all attributes that will be accessed on the mock
            mock_rel_prop.secondary = None
            mock_rel_prop.synchronize_pairs = None
            mock_rel_prop.foreign_keys = {mock_fk_col}

            result = _get_involved_foreign_keys(mock_rel_prop)
            self.assertEqual(result, {"mock_table.fk_col"})

        with self.subTest("Test _get_fks_from_synchronize_pairs with local FK"):
            mock_rel_prop_sync = MagicMock(spec=RelationshipProperty)
            mock_local_col = MagicMock(spec=Column)
            mock_local_col.foreign_keys = {"a_foreign_key"}
            mock_local_col.__str__.return_value = "mock_table.local_col"
            mock_remote_col = MagicMock(spec=Column)
            mock_remote_col.foreign_keys = set()
            mock_remote_col.__str__.return_value = "mock_table.remote_col"

            # Configure all attributes that will be accessed on the mock
            mock_rel_prop_sync.secondary = None
            mock_rel_prop_sync.synchronize_pairs = [(mock_local_col, mock_remote_col)]
            # To ensure we don't short-circuit, make foreign_keys None
            # This is because hasattr check is now more robust.
            type(mock_rel_prop_sync).foreign_keys = None

            result_sync = _get_involved_foreign_keys(mock_rel_prop_sync)
            self.assertEqual(result_sync, {"mock_table.local_col"})

        with self.subTest("Test fallback to empty set"):
            mock_rel_prop_fallback = MagicMock(spec=RelationshipProperty)
            mock_rel_prop_fallback.secondary = None
            mock_rel_prop_fallback.synchronize_pairs = None
            type(mock_rel_prop_fallback).foreign_keys = None

            result_fallback = _get_involved_foreign_keys(mock_rel_prop_fallback)
            self.assertEqual(result_fallback, set())


class TestGenerateLLMSchemaFromModels(SQLAlchemyBaseTestCase):
    def test_process_relationship_for_llm_schema_uselist_flag(self):
        """
        Tests that the `uselist` flag correctly determines whether to generate
        a singular or plural reference ID field.
        """
        # Test case for uselist=False (e.g., ManyToOne, OneToOne)
        rel_data_single = {
            "related_model_name": "SomeModel",
            "type": "MANYTOONE",
            "uselist": False,
        }
        custom_descs = {}
        result_single = _process_relationship_for_llm_schema(
            "some_rel", rel_data_single, custom_descs
        )
        self.assertIsNotNone(result_single)
        self.assertEqual(result_single[0], "some_rel_ref_id")
        self.assertIn("The _temp_id of the related SomeModel", result_single[1])

        # Test case for uselist=True (e.g., OneToMany, ManyToMany)
        rel_data_list = {
            "related_model_name": "SomeModel",
            "type": "ONETOMANY",
            "uselist": True,
        }
        result_list = _process_relationship_for_llm_schema(
            "some_rel", rel_data_list, custom_descs
        )
        self.assertIsNotNone(result_list)
        self.assertEqual(result_list[0], "some_rel_ref_ids")
        self.assertIn("A list of _temp_ids for related SomeModel", result_list[1])

    def test_process_relationship_for_llm_schema_no_uselist_flag(self):
        """
        Tests that if the `uselist` flag is missing, the function returns None.
        """
        rel_data_no_uselist = {
            "related_model_name": "SomeModel",
            "type": "ONETOMANY",
            # 'uselist' key is intentionally omitted
        }
        custom_descs = {}
        result = _process_relationship_for_llm_schema(
            "some_rel", rel_data_no_uselist, custom_descs
        )
        self.assertIsNone(result)

    def test_single_model_basic_schema(self):
        llm_schema_str = generate_llm_schema_from_models([Department])
        self.assertIsInstance(llm_schema_str, str)
        schema = json.loads(llm_schema_str)
        self.assertIn("Department", schema)
        dept_schema = schema["Department"]
        self.assertIn("description", dept_schema)
        self.assertIn("fields", dept_schema)
        self.assertIn("notes_for_llm", dept_schema)
        self.assertIn("Unique Department ID (PK)", dept_schema["fields"]["id"])
        self.assertIn("Official name of the department.", dept_schema["fields"]["name"])
        self.assertIn("employees_ref_ids", dept_schema["fields"])
        self.assertIn(
            "All employees belonging to this department.",
            dept_schema["fields"]["employees_ref_ids"],
        )

    def test_related_models_schema_and_descriptions(self):
        llm_schema_str = generate_llm_schema_from_models([Employee, Department])
        schema = json.loads(llm_schema_str)
        self.assertIn("Employee", schema)
        self.assertIn("Department", schema)
        emp_schema = schema["Employee"]
        self.assertIn("Full legal name of the employee.", emp_schema["fields"]["name"])
        self.assertIn("standard_email_format", emp_schema["fields"]["email"])
        self.assertIn(
            "The department this employee is assigned to.",
            emp_schema["fields"]["department_ref_id"],
        )
        dept_schema = schema["Department"]
        self.assertIn("Official name of the department.", dept_schema["fields"]["name"])
        self.assertIn("pk_strategy", dept_schema["fields"]["id"])

    def test_many_to_many_relationship_schema(self):
        llm_schema_str = generate_llm_schema_from_models([Project, Member])
        schema = json.loads(llm_schema_str)
        self.assertIn("Project", schema)
        proj_schema = schema["Project"]
        self.assertIn("members_ref_ids", proj_schema["fields"])
        self.assertIn(
            "Members participating in this project.",
            proj_schema["fields"]["members_ref_ids"],
        )
        self.assertIn("Member", schema)
        mem_schema = schema["Member"]
        self.assertIn("projects_ref_ids", mem_schema["fields"])
        self.assertIn(
            "Projects this member is associated with.",
            mem_schema["fields"]["projects_ref_ids"],
        )

    def test_custom_field_descriptions_override(self):
        custom_descriptions = {
            "Employee": {
                "name": "Employee's complete name as per official records.",
                "_model_description": "Custom model description for Employee.",
                "department_ref_id": "Custom description for department link.",
            }
        }
        llm_schema_str = generate_llm_schema_from_models(
            [Employee], custom_field_descriptions=custom_descriptions
        )
        schema = json.loads(llm_schema_str)
        emp_schema = schema["Employee"]
        self.assertIn(
            "Employee's complete name as per official records.",
            emp_schema["fields"]["name"],
        )
        self.assertIn(
            "Custom model description for Employee.", emp_schema["description"]
        )
        self.assertIn(
            "Custom description for department link.",
            emp_schema["fields"]["department_ref_id"],
        )

    def test_empty_model_list(self):
        llm_schema_str = generate_llm_schema_from_models([])
        schema = json.loads(llm_schema_str)
        self.assertEqual(schema, {})

    def test_model_inspection_error_handling(self):
        class NotAModel:
            __name__ = "NotAModel"

        llm_schema_str = generate_llm_schema_from_models([Department, NotAModel])  # type: ignore
        schema = json.loads(llm_schema_str)
        self.assertIn("Department", schema)
        self.assertNotIn("NotAModel", schema)

    def test_article_scenario_list_str_as_json_is_array_string(self):
        llm_schema_str = generate_llm_schema_from_models([ArticleScenarioModel])
        self.assertIsInstance(llm_schema_str, str)
        try:
            schema = json.loads(llm_schema_str)
        except json.JSONDecodeError as e:
            self.fail(
                f"LLM Schema is not valid JSON: {e}\nSchema content:\n{llm_schema_str}"
            )

        self.assertIn("ArticleScenarioModel", schema)
        model_schema = schema["ArticleScenarioModel"]
        self.assertIn("fields", model_schema)
        fields = model_schema["fields"]

        self.assertIn("key_topics", fields)
        field_desc_key_topics = fields["key_topics"].lower()
        self.assertIn("list of key topics.", field_desc_key_topics)
        self.assertNotIn("object //", field_desc_key_topics)
        self.assertNotIn("dict //", field_desc_key_topics)
        self.assertTrue(
            "array[string]" in field_desc_key_topics
            or "list[string]" in field_desc_key_topics,
            f"Expected 'array[string]' or 'list[string]' for key_topics, got: {fields['key_topics']}",
        )

        self.assertIn("categories", fields)
        field_desc_categories = fields["categories"].lower()
        self.assertIn("list of categories.", field_desc_categories)
        self.assertNotIn("object //", field_desc_categories)
        self.assertNotIn("dict //", field_desc_categories)
        self.assertTrue(
            "array[string]" in field_desc_categories
            or "list[string]" in field_desc_categories,
            f"Expected 'array[string]' or 'list[string]' for categories, got: {fields['categories']}",
        )

        self.assertIn("meta_data", fields)
        field_desc_meta_data = fields["meta_data"].lower()
        self.assertIn("meta data dictionary.", field_desc_meta_data)
        self.assertTrue(
            "object[string,any]" in field_desc_meta_data
            or "dict[string,any]" in field_desc_meta_data,
            f"Expected 'object[string,any]' or 'dict[string,any]' for meta_data, got: {fields['meta_data']}",
        )

    def test_generate_schema_with_non_serializable_info_dicts(self):
        llm_schema_col_str = generate_llm_schema_from_models(
            [ModelWithNonSerializableColInfo]
        )
        schema_col = json.loads(llm_schema_col_str)
        self.assertIn("ModelWithNonSerializableColInfo", schema_col)
        field_info = schema_col["ModelWithNonSerializableColInfo"]["fields"]["data"]
        self.assertIn("non_serializable", field_info)
        self.assertIn("Info: {", field_info)
        self.assertTrue(
            "set()" in field_info
            or "{1, 2, 3}" in field_info
            or "{2, 1, 3}" in field_info
        )

        llm_schema_rel_str = generate_llm_schema_from_models(
            [ModelWithNonSerializableRelInfo, ModelRelatedToNonSerializable]
        )
        schema_rel = json.loads(llm_schema_rel_str)
        self.assertIn("ModelWithNonSerializableRelInfo", schema_rel)
        rel_field_name = "related_ref_id"
        self.assertIn(
            rel_field_name, schema_rel["ModelWithNonSerializableRelInfo"]["fields"]
        )
        rel_field_description = schema_rel["ModelWithNonSerializableRelInfo"]["fields"][
            rel_field_name
        ]
        self.assertIn("non_serializable", rel_field_description)
        self.assertIn("(Info: {", rel_field_description)
        self.assertIn("<built-in method now of type object at", rel_field_description)

        llm_schema_table_str = generate_llm_schema_from_models(
            [ModelWithNonSerializableTableInfo]
        )
        schema_table = json.loads(llm_schema_table_str)
        self.assertIn("ModelWithNonSerializableTableInfo", schema_table)
        model_desc = schema_table["ModelWithNonSerializableTableInfo"]["description"]

        self.assertTrue(
            model_desc.startswith("table (Info: {"),
            f"Model description should start with 'table (Info: {{': {model_desc}",
        )
        self.assertIn("(Info: {'non_serializable': ", model_desc)
        self.assertIn("<object object at ", model_desc)
        self.assertIn(
            "}) When processing a ModelWithNonSerializableTableInfo", model_desc
        )

    def test_generate_schema_for_plain_sqlalchemy_model(self):
        llm_schema_str = generate_llm_schema_from_models(
            [PlainSQLAlchemyModelWithPydanticHints]
        )
        schema = json.loads(llm_schema_str)

        self.assertIn("PlainSQLAlchemyModelWithPydanticHints", schema)
        model_fields = schema["PlainSQLAlchemyModelWithPydanticHints"]["fields"]

        self.assertIn("name", model_fields)
        self.assertTrue(
            model_fields["name"].startswith("string //"),
            f"Expected 'string' for name, got {model_fields['name']}",
        )

        self.assertIn("complex_field", model_fields)
        self.assertTrue(
            model_fields["complex_field"].startswith("object //"),
            f"Expected 'object' for complex_field, got {model_fields['complex_field']}",
        )

        self.assertIn("simple_sqla_type_field", model_fields)
        self.assertTrue(
            model_fields["simple_sqla_type_field"].startswith("integer //"),
            f"Expected 'integer' for simple_sqla_type_field, got {model_fields['simple_sqla_type_field']}",
        )

    def test_default_desc_for_bare_many_to_one_relationship(self):
        # Define minimal models for this specific test case

        # Define BareChild first or use ForwardRef for BareParent's type hint for children
        class BareParentForDefaultDesc(SQLModel, table=True):
            __tablename__ = "bareparent_for_default_desc_rel_fix"  # Unique table name
            id: Optional[int] = Field(default=None, primary_key=True)
            name: str = "ParentName"  # Add a dummy field to make it a valid table
            # Relationship uses string forward reference if BareChildForDefaultDesc is defined later
            children: List["BareChildForDefaultDescRelFix"] = Relationship(
                back_populates="parent"
            )

        class BareChildForDefaultDescRelFix(SQLModel, table=True):
            __tablename__ = "barechild_for_default_desc_rel_fix"  # Unique table name
            id: Optional[int] = Field(default=None, primary_key=True)
            data: str = "ChildData"  # Add a dummy field
            parent_id: Optional[int] = Field(
                default=None, foreign_key="bareparent_for_default_desc_rel_fix.id"
            )
            # Relationship uses string forward reference if BareParentForDefaultDesc is defined earlier or in same scope
            parent: Optional["BareParentForDefaultDesc"] = Relationship(
                back_populates="children"
            )

        # Resolve forward references
        BareParentForDefaultDesc.model_rebuild()
        BareChildForDefaultDescRelFix.model_rebuild()

        # Ensure tables are created for these local models
        SQLModel.metadata.create_all(
            self.engine,
            tables=[
                BareParentForDefaultDesc.__table__,
                BareChildForDefaultDescRelFix.__table__,
            ],
            checkfirst=True,
        )

        # Generate schema for the child model, which contains the target relationship
        llm_schema_str = generate_llm_schema_from_models(
            [BareChildForDefaultDescRelFix]
        )
        schema = json.loads(llm_schema_str)

        self.assertIn("BareChildForDefaultDescRelFix", schema)
        child_schema = schema["BareChildForDefaultDescRelFix"]
        self.assertIn("parent_ref_id", child_schema["fields"])
        # Assert that the default description for a MANYTOONE relationship is used
        self.assertIn(
            "The _temp_id of the related BareParentForDefaultDesc for 'parent'.",
            child_schema["fields"]["parent_ref_id"],
        )
        self.assertIn(
            "BareParentForDefaultDesc", schema
        )  # Check parent model is also processed

    def test_get_python_type_str_from_pydantic_annotation(self):
        class MyTestEnum(enum.Enum):
            VAL_A = 1

        class MyCustomNamedType:
            __name__ = "MyCustomNamedType"

        class ActualSecretStrMimic:
            pass

        ActualSecretStrMimic.__name__ = "SecretStr"

        class MimicTildeForwardRefStr:
            def __str__(self):
                return "~MyActualModel"

        class MimicTildeTypingForwardRefStr:
            def __str__(self):
                return "typing.~AnotherActualModel"

        class MimicTypingSyntax:
            def __str__(self):
                return "typing.FooBarType"

        class MimicTypingListSyntax:
            def __str__(self):
                return "typing.List[custom.SubType]"

        obj_instance = object()

        cases = [
            # From test_list_types
            (List[int], "list[int]"),
            (List, "list"),
            # From test_dict_types
            (Dict[str, int], "dict[str,int]"),
            (Dict, "dict"),
            # From test_union_types
            (Union[int, str], "union[int,str]"),
            (Union[str, int], "union[int,str]"),
            (Union[int, None, str], "union[int,str]"),
            (Union[None, int], "int"),
            (Union[int, int], "int"),
            (Union[None, type(None)], "none"),
            (Union, "union"),
            # From test_base_types
            (int, "int"),
            (str, "str"),
            (bool, "bool"),
            (float, "float"),
            (datetime.date, "date"),
            (datetime.datetime, "datetime"),
            (bytes, "bytes"),
            (Any, "any"),
            (type(None), "none"),
            # From test_enum_type
            (MyTestEnum, "enum"),
            # From test_custom_named_types_and_fallbacks
            (MyCustomNamedType, "mycustomnamedtype"),
            (ActualSecretStrMimic, "str"),
            (ForwardRef("MyModelNameRef"), "forwardref('mymodelnameref')"),
            (ForwardRef("~AnotherModelRef"), "forwardref('~anothermodelref')"),
            (MimicTildeForwardRefStr(), "myactualmodel"),
            (MimicTildeTypingForwardRefStr(), "anotheractualmodel"),
            (MimicTypingSyntax(), "foobartype"),
            (MimicTypingListSyntax(), "list[custom.subtype]"),
            (obj_instance, str(obj_instance).lower()),
        ]
        for annotation, expected in cases:
            with self.subTest(annotation=str(annotation)):
                self.assertEqual(
                    _get_python_type_str_from_pydantic_annotation(annotation), expected
                )

    @patch("extrai.core.schema_inspector.get_args")
    @patch("extrai.core.schema_inspector.get_origin")
    def test_optional_origin_direct_paths_with_mock(
        self, mock_get_origin, mock_get_args
    ):
        mock_get_origin.side_effect = (
            lambda x: Optional if x is Optional[int] else typing_get_origin(x)
        )
        mock_get_args.side_effect = (
            lambda x: (int, type(None)) if x is Optional[int] else typing_get_args(x)
        )
        with self.subTest(scenario="Optional[int] -> int (args[1] is NoneType)"):
            self.assertEqual(
                _get_python_type_str_from_pydantic_annotation(Optional[int]), "int"
            )

        mock_get_origin.side_effect = (
            lambda x: Optional if x is Optional[bool] else typing_get_origin(x)
        )
        mock_get_args.side_effect = (
            lambda x: (bool,) if x is Optional[bool] else typing_get_args(x)
        )
        with self.subTest(scenario="Optional[bool] -> bool (single arg)"):
            self.assertEqual(
                _get_python_type_str_from_pydantic_annotation(Optional[bool]), "bool"
            )

        mock_get_origin.side_effect = (
            lambda x: Optional if x is Optional[type(None)] else typing_get_origin(x)
        )
        mock_get_args.side_effect = (
            lambda x: (type(None),) if x is Optional[type(None)] else typing_get_args(x)
        )
        with self.subTest(
            scenario="Optional[type(None)] -> none (args[0] is NoneType)"
        ):
            self.assertEqual(
                _get_python_type_str_from_pydantic_annotation(Optional[type(None)]),
                "none",
            )

        mock_get_origin.side_effect = (
            lambda x: Optional if x is Optional[Any] else typing_get_origin(x)
        )
        mock_get_args.side_effect = (
            lambda x: () if x is Optional[Any] else typing_get_args(x)
        )
        with self.subTest(scenario="Optional[Any] -> none (empty args)"):
            self.assertEqual(
                _get_python_type_str_from_pydantic_annotation(Optional[Any]), "none"
            )

    def test_union_no_args_fallback(self):
        with patch(
            "extrai.core.schema_inspector.get_args",
            return_value=(),
        ) as mock_get_args_union:
            with patch(
                "extrai.core.schema_inspector.get_origin",
                side_effect=lambda x: Union if x is Union else typing_get_origin(x),
            ) as mock_get_origin_union:
                result = _get_python_type_str_from_pydantic_annotation(Union)
                self.assertEqual(result, "union")
                mock_get_args_union.assert_called_with(Union)
                mock_get_origin_union.assert_called_with(Union)

    def test_fallback_typing_prefix_in_cleaned_str(self):
        class WeirdTypingList:
            def __str__(self):
                return "typing.typing.List[custom.SubType]"

        class WeirdTypingDict:
            def __str__(self):
                return "typing.typing.Dict[str, int]"

        cases = [
            (WeirdTypingList(), "list[custom.subtype]"),
            (WeirdTypingDict(), "dict[str, int]"),
        ]
        for annotation, expected in cases:
            with self.subTest(annotation=str(annotation)):
                with (
                    patch(
                        "extrai.core.schema_inspector.get_origin",
                        return_value=None,
                    ),
                    patch(
                        "extrai.core.schema_inspector.get_args",
                        return_value=(),
                    ),
                ):
                    self.assertEqual(
                        _get_python_type_str_from_pydantic_annotation(annotation),
                        expected,
                    )


class TestMapSqlTypeToLlmType(unittest.TestCase):
    def test_map_sql_type_to_llm_type(self):
        cases = [
            # From test_basic_python_type_mappings
            (("INT", "int"), "integer"),
            (("VARCHAR", "str"), "string"),
            (("BOOLEAN", "bool"), "boolean"),
            (("FLOAT", "float"), "number (float/decimal)"),
            (("DATE", "date"), "string (date format)"),
            (("DATETIME", "datetime"), "string (datetime format)"),
            (("BLOB", "bytes"), "string (base64 encoded)"),
            (("ENUM", "enum"), "string (enum)"),
            (("ANYSQL", "any"), "any"),
            (("ANYSQL", "none"), "null"),
            # From test_list_python_types
            (("JSON", "list[str]"), "array[string]"),
            (("ARRAY", "list[int]"), "array[integer]"),
            (("JSON", "list[list[str]]"), "array[array[string]]"),
            (("ANYSQL", "list[float]"), "array[number (float/decimal)]"),
            # From test_dict_python_types
            (("JSON", "dict[str,int]"), "object[string,integer]"),
            (("JSON", "dict[str,list[int]]"), "object[string,array[integer]]"),
            (("JSON", "dict[invalidformat]"), "object"),
            (("JSON", "dict[str]"), "object"),
            # From test_union_python_types
            (("ANYSQL", "union[int,str]"), "union[integer,string]"),
            (("ANYSQL", "union[str,int]"), "union[integer,string]"),
            (("ANYSQL", "union[int,none,str]"), "union[integer,null,string]"),
            (("ANYSQL", "union[int,null,str]"), "union[integer,string]"),
            (("ANYSQL", "union[]"), "any"),
            (("ANYSQL", "union[ ]"), "any"),
            (("ANYSQL", "union[int]"), "integer"),
            (("ANYSQL", "union[str,str]"), "string"),
            # From test_sql_type_fallbacks_and_priority
            (("INT", "str"), "string"),
            (("VARCHAR", "int"), "integer"),
            (("INTEGER", "some_generic_type"), "integer"),
            (("TEXT", "some_generic_type"), "string"),
            (("CHAR(50)", "some_generic_type"), "string"),
            (("CLOB", "some_generic_type"), "string"),
            (("BOOLEAN", "some_generic_type"), "boolean"),
            (("TIMESTAMP", "some_generic_type"), "string (date/datetime format)"),
            (("TIME", "some_other_type"), "string (date/datetime format)"),
            (("TIMESTAMP", "date"), "string (date format)"),
            (("DATE", "datetime"), "string (datetime format)"),
            (("NUMERIC(10,2)", "some_generic_type"), "number (float/decimal)"),
            (("DECIMAL", "some_generic_type"), "number (float/decimal)"),
            (("FLOAT", "some_generic_type"), "number (float/decimal)"),
            (("DOUBLE", "some_generic_type"), "number (float/decimal)"),
            # From test_list_dict_fallbacks_with_sql_types
            (("JSON", "list"), "array"),
            (("TEXT[]", "list"), "string"),
            (("CUSTOM_ARRAY_TYPE", "list"), "array"),
            (("SOME_OTHER_SQL_TYPE", "list"), "array"),
            (("JSON", "dict"), "object"),
            (("CUSTOM_OBJECT_TYPE", "dict"), "object"),
            # From test_generic_json_sql_type
            (("JSONB", "some_other_py_type"), "object"),
            (("JSON", "unspecific_python"), "object"),
            # From test_unknown_python_types
            (("JSON", "unknown_py_type"), "object"),
            (("TEXT[]", "unknown_py_type_array"), "string"),
            (("SOME_ARRAY_TYPE", "unknown_py_type_array_fallback"), "array"),
            (("VARCHAR", "unknown_py_type_str"), "string"),
            (("SOME_OTHER_SQL", "unknown_py_type_str_fallback"), "string"),
            # From test_final_fallback
            (("RARE_SQL_TYPE", "rare_python_type"), "string"),
            # From test_map_dict_with_json_in_sql_type_returns_object
            (("JSON", "dict"), "object"),
            (("jsonb", "dict"), "object"),
            (("some_json_type", "dict"), "object"),
            (("APPLICATION/JSON", "dict"), "object"),
            # From test_dict_fallback_with_non_json_non_empty_sql_type
            (("ARBITRARYSQLTYPE", "dict"), "object"),
            (("SOMEOTHEROPAQUETYPE", "dict"), "object"),
            # From test_unknown_python_type_with_json_sql_type
            (("JSON_VARIANT", "unknown_specific_case"), "object"),
            (("this_is_json_too", "unknown_anything"), "object"),
        ]
        for (sql_type, py_type), expected in cases:
            with self.subTest(sql_type=sql_type, py_type=py_type):
                self.assertEqual(_map_sql_type_to_llm_type(sql_type, py_type), expected)


class TestDiscoveryAndCollectionFunctions(SQLAlchemyBaseTestCase):
    def test_collect_all_sqla_models_no_inspection(self):
        class NonInspectableModelForCollect:
            __name__ = "NonInspectableModelForCollect"

        discovered_models: List[Any] = []
        with patch(
            "extrai.core.schema_inspector.inspect",
            side_effect=NoInspectionAvailable,
        ) as mock_inspect:
            _collect_all_sqla_models_recursively(
                NonInspectableModelForCollect, discovered_models, set()
            )
            mock_inspect.assert_called_once_with(NonInspectableModelForCollect)
        self.assertIn(NonInspectableModelForCollect, discovered_models)

    def test_collect_all_sqla_models_inspector_none(self):
        class InspectReturnsNoneModelForCollect(Base):
            __tablename__ = "inspect_none_model_collect"
            id: Mapped[int] = mapped_column(primary_key=True)

        discovered_models: List[Any] = []
        with patch(
            "extrai.core.schema_inspector.inspect",
            return_value=None,
        ) as mock_inspect:
            _collect_all_sqla_models_recursively(
                InspectReturnsNoneModelForCollect, discovered_models, set()
            )
            mock_inspect.assert_called_once_with(InspectReturnsNoneModelForCollect)
        self.assertIn(InspectReturnsNoneModelForCollect, discovered_models)

    def test_discover_sqlmodels_from_root_with_invalid_input(self):
        class NotASQLModelForDiscover:
            __name__ = "NotASQLModelForDiscover"

        invalid_inputs = [None, NotASQLModelForDiscover]
        for invalid_input in invalid_inputs:
            with self.subTest(invalid_input=invalid_input):
                with patch("builtins.print") as mock_print:
                    result = discover_sqlmodels_from_root(invalid_input)
                    self.assertEqual(result, [])
                    mock_print.assert_called_once()
                    self.assertIn(
                        "is not a valid SQLModel class", mock_print.call_args[0][0]
                    )

    def test_discover_sqlmodels_from_root_collection_exception(self):
        class RootSQLModelForDiscoverException(SQLModel, table=True):
            __tablename__ = "root_sql_model_discover_exc"
            id: Optional[int] = Field(default=None, primary_key=True)

        with (
            patch(
                "extrai.core.schema_inspector._collect_all_sqla_models_recursively",
                side_effect=Exception("Test collection error"),
            ) as mock_collect,
            patch("builtins.print") as mock_print,
        ):
            result = discover_sqlmodels_from_root(RootSQLModelForDiscoverException)
            self.assertEqual(result, [])
            mock_collect.assert_called_once()
            mock_print.assert_called_once()
            self.assertIn(
                "Error during SQLModel discovery starting from RootSQLModelForDiscoverException: Test collection error",
                mock_print.call_args[0][0],
            )

    def test_collect_all_sqla_models_recursion_guard(self):
        local_base_for_rec_guard_hit: Any = declarative_base()

        class ModelToGuard(local_base_for_rec_guard_hit):
            __tablename__ = "model_to_guard"
            id: Mapped[int] = mapped_column(primary_key=True)
            related_dummy_id: Mapped[Optional[int]] = mapped_column(
                ForeignKey("model_to_guard.id")
            )  # Self-referential to simplify
            dummy_rel = sa_relationship("ModelToGuard")

        local_base_for_rec_guard_hit.metadata.create_all(self.engine, checkfirst=True)

        discovered_models: Set[Any] = set()
        recursion_guard: Set[Any] = {ModelToGuard}

        with patch("extrai.core.schema_inspector.inspect") as mock_inspect_call:
            _collect_all_sqla_models_recursively(
                ModelToGuard, discovered_models, recursion_guard
            )
            mock_inspect_call.assert_not_called()

        self.assertEqual(
            len(discovered_models),
            0,
            "Discovered models should be empty if recursion guard is hit immediately.",
        )
        self.assertIn(
            ModelToGuard,
            recursion_guard,
            "ModelToGuard should remain in recursion_guard as it was pre-populated and hit.",
        )

    def test_discover_sqlmodels_from_root_successful_collection(self):
        # Test the successful path of discover_sqlmodels_from_root (line 706)
        # where _collect_all_sqla_models_recursively completes without error.

        # Define models locally for clarity and to avoid altering shared test models
        class DiscoverRoot(SQLModel, table=True):
            __tablename__ = (
                "discover_root_table_for_line_706"  # Ensure unique table name
            )
            id: Optional[int] = Field(default=None, primary_key=True)
            name: str = "Root"

            related_items: List["DiscoverRelatedForLine706"] = Relationship(
                back_populates="root_model"
            )

        class DiscoverRelatedForLine706(SQLModel, table=True):
            __tablename__ = (
                "discover_related_table_for_line_706"  # Ensure unique table name
            )
            id: Optional[int] = Field(default=None, primary_key=True)
            data: str = "Related Data"
            root_model_id: Optional[int] = Field(
                default=None, foreign_key="discover_root_table_for_line_706.id"
            )

            root_model: Optional[DiscoverRoot] = Relationship(
                back_populates="related_items"
            )

        DiscoverRoot.model_rebuild()
        DiscoverRelatedForLine706.model_rebuild()

        SQLModel.metadata.create_all(
            self.engine,
            tables=[DiscoverRoot.__table__, DiscoverRelatedForLine706.__table__],
            checkfirst=True,
        )

        discovered_models = discover_sqlmodels_from_root(DiscoverRoot)
        self.assertIsInstance(discovered_models, list)
        self.assertIn(DiscoverRoot, discovered_models)
        self.assertIn(DiscoverRelatedForLine706, discovered_models)
        self.assertEqual(len(discovered_models), 2)

        # Test with a standalone model to ensure it also hits the line
        class DiscoverStandaloneForLine706(SQLModel, table=True):
            __tablename__ = "discover_standalone_for_line_706"
            id: Optional[int] = Field(default=None, primary_key=True)

        SQLModel.metadata.create_all(
            self.engine,
            tables=[DiscoverStandaloneForLine706.__table__],
            checkfirst=True,
        )

        discovered_standalone = discover_sqlmodels_from_root(
            DiscoverStandaloneForLine706
        )
        self.assertIsInstance(discovered_standalone, list)
        self.assertIn(DiscoverStandaloneForLine706, discovered_standalone)
        self.assertEqual(len(discovered_standalone), 1)


if __name__ == "__main__":
    unittest.main(argv=["first-arg-is-ignored"], verbosity=2, exit=False)
