import json
import pytest
from typing import List, Optional
from unittest.mock import patch, MagicMock

from sqlalchemy import (
    create_engine,
    Column,
    String,
    Enum,
    UniqueConstraint,
)
from sqlalchemy.orm import (
    declarative_base,
    Mapped,
    mapped_column,
    relationship,
    RelationshipProperty,
)
from sqlalchemy.exc import NoInspectionAvailable
from sqlalchemy.orm.attributes import InstrumentedAttribute
from sqlmodel import SQLModel, Field, Relationship

from extrai.core.schema_inspector import SchemaInspector
from tests.core.helpers.orchestrator_test_models import (
    Base,
    Department,
    Employee,
    StatusEnum,
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


@pytest.fixture(scope="module")
def engine():
    e = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(e)
    SQLModel.metadata.create_all(e)
    return e


@pytest.fixture
def inspector(engine):
    return SchemaInspector()


@pytest.mark.parametrize(
    "model_cls, checks",
    [
        (
            Employee,
            [
                lambda s: s["comment"]
                == "Stores detailed information about company employees.",
                lambda s: s["info_dict"] == {"confidentiality": "high"},
                lambda s: s["columns"]["id"]["comment"] == "Unique Employee ID (PK)",
                lambda s: s["columns"]["email"]["info_dict"]["validation_rule"]
                == "standard_email_format",
                lambda s: s["relationships"]["department"]["info_dict"]["description"]
                == "The department this employee is assigned to.",
                lambda s: s["relationships"]["department"]["nested_schema"][
                    "table_name"
                ]
                == "departments",
                lambda s: s["relationships"]["department"]["nested_schema"][
                    "relationships"
                ]["employees"]["nested_schema"]["recursion_detected_for_type"]
                == "Employee",
                lambda s: "employees.department_id"
                in s["relationships"]["department"]["foreign_key_constraints_involved"],
            ],
        ),
        (
            Department,
            [
                lambda s: s["comment"] == "Stores all company departments.",
                lambda s: s["columns"]["id"]["comment"] == "Unique Department ID (PK)",
                lambda s: s["relationships"]["employees"]["info_dict"][
                    "relationship_detail"
                ]
                == "All employees belonging to this department.",
                lambda s: s["relationships"]["employees"]["nested_schema"]["table_name"]
                == "employees",
                lambda s: s["relationships"]["employees"]["nested_schema"][
                    "relationships"
                ]["department"]["nested_schema"]["recursion_detected_for_type"]
                == "Department",
            ],
        ),
        (
            Project,
            [
                lambda s: s["relationships"]["members"]["related_model_name"]
                == "Member",
                lambda s: s["relationships"]["members"]["secondary_table_name"]
                == "project_member",
                lambda s: "project_member.project_id"
                in s["relationships"]["members"]["foreign_key_constraints_involved"],
            ],
        ),
        (
            TableModel,
            [
                lambda s: s["table_name"] == "tables",
                lambda s: s["relationships"]["supports"]["nested_schema"][
                    "relationships"
                ]["screws_list"]["nested_schema"]["relationships"]["support"][
                    "nested_schema"
                ]["recursion_detected_for_type"]
                == "Support",
            ],
        ),
        (ModelWithColumnProperty, [lambda s: "data_length" not in s["columns"]]),
        (
            ModelWithCustomColType,
            [
                lambda s: s["columns"]["custom_field"]["python_type"]
                == "unknown_error_accessing_type"
            ],
        ),
        (
            FKParent,
            [
                lambda s: "fk_child_sync.parent_id_col"
                in s["relationships"]["children_sync"][
                    "foreign_key_constraints_involved"
                ]
            ],
        ),
        (
            FKParentDirect,
            [
                lambda s: "fk_child_direct.parent_fk_col_name"
                in s["relationships"]["children_direct"][
                    "foreign_key_constraints_involved"
                ]
            ],
        ),
        (
            ViewOnlyParent,
            [
                lambda s: "viewonly_child.parent_id"
                in s["relationships"]["children"]["foreign_key_constraints_involved"]
            ],
        ),
    ],
)
def test_inspect_sqlalchemy_model_basics(inspector, model_cls, checks):
    schema = inspector.inspect_sqlalchemy_model(model_cls)
    assert "error" not in schema, f"Inspection failed: {schema.get('error')}"
    for check in checks:
        assert check(schema)


def test_relationship_not_relationship_property(inspector):
    local_base = declarative_base()

    class ModelNonRel(local_base):
        __tablename__ = "model_non_rel"
        id: Mapped[int] = mapped_column(primary_key=True)

    mock_inspector = MagicMock()
    mock_inspector.relationships = {"fake": MagicMock()}  # Not RelationshipProperty
    mock_inspector.column_attrs = []
    mock_inspector.selectable = ModelNonRel.__table__
    mock_inspector.mapper.class_ = ModelNonRel

    with patch("extrai.core.schema_inspector.inspect", return_value=mock_inspector):
        schema = inspector.inspect_sqlalchemy_model(ModelNonRel)
        assert "fake" not in schema.get("relationships", {})


def test_inspector_none_error(inspector, engine):
    class SomeModel(Base):
        __tablename__ = "some_model_none"
        id: Mapped[int] = mapped_column(primary_key=True)

    SomeModel.metadata.create_all(engine)

    with patch("extrai.core.schema_inspector.inspect", return_value=None):
        schema = inspector.inspect_sqlalchemy_model(SomeModel)
        assert "error" in schema


def test_column_python_type_attribute_error(inspector, engine):
    local_base = declarative_base()

    class ModelAttrError(local_base):
        __tablename__ = "model_attr_error"
        id: Mapped[int] = mapped_column(primary_key=True)
        col: Mapped[str] = mapped_column(String)

    mock_insp = MagicMock()
    mock_insp.selectable = ModelAttrError.__table__
    mock_insp.mapper.class_ = ModelAttrError
    mock_col = MagicMock(spec=Column)
    mock_col.name = "col"
    mock_col.unique = False
    mock_col.table = None
    mock_col.primary_key = False
    mock_col.nullable = True
    mock_col.comment = None
    mock_col.info = {}
    mock_col.foreign_keys = set()

    mock_col.type = MagicMock()
    type(mock_col.type).python_type = MagicMock(side_effect=AttributeError)

    mock_attr = MagicMock(spec=InstrumentedAttribute)
    mock_attr.key = "col"
    mock_attr.expression = mock_col
    mock_insp.column_attrs = [mock_attr]
    mock_insp.relationships = {}

    with patch("extrai.core.schema_inspector.inspect", return_value=mock_insp):
        schema = inspector.inspect_sqlalchemy_model(ModelAttrError)
        assert schema["columns"]["col"]["python_type"] == "unknown_no_python_type_attr"


def test_relationship_no_fks(inspector, engine):
    local_base = declarative_base()

    class Dest(local_base):
        __tablename__ = "dest"
        id: Mapped[int] = mapped_column(primary_key=True)

    class Src(local_base):
        __tablename__ = "src"
        id: Mapped[int] = mapped_column(primary_key=True)
        rel = relationship(
            "Dest", viewonly=True, primaryjoin="foreign(Src.id)==remote(Dest.id)"
        )

    local_base.metadata.create_all(engine)

    schema = inspector.inspect_sqlalchemy_model(Src)
    assert schema["relationships"]["rel"]["foreign_key_constraints_involved"] == []


def test_string_based_enum_in_column(inspector, engine):
    local_base = declarative_base()

    class ModelEnum(local_base):
        __tablename__ = "model_enum"
        id: Mapped[int] = mapped_column(primary_key=True)
        status: Mapped[str] = mapped_column(Enum("A", "B", "C", name="status_enum"))

    local_base.metadata.create_all(engine)
    schema = inspector.inspect_sqlalchemy_model(ModelEnum)
    assert schema["columns"]["status"]["enum_values"] == ["A", "B", "C"]


def test_helper_coverage(inspector):
    # _get_fks_from_direct_foreign_keys
    m_fk = MagicMock()
    m_fk.__str__.return_value = "t.c"
    mock_rel = MagicMock(spec=RelationshipProperty)
    mock_rel.secondary = None
    mock_rel.synchronize_pairs = None
    mock_rel.foreign_keys = {m_fk}
    assert inspector._get_involved_foreign_keys(mock_rel) == {"t.c"}

    # _get_fks_from_synchronize_pairs
    m_local = MagicMock()
    m_local.__str__.return_value = "t.l"
    m_local.foreign_keys = {"fk"}
    m_remote = MagicMock()
    m_remote.__str__.return_value = "t.r"
    m_remote.foreign_keys = set()

    mock_rel_sync = MagicMock(spec=RelationshipProperty)
    mock_rel_sync.secondary = None
    mock_rel_sync.synchronize_pairs = [(m_local, m_remote)]
    type(mock_rel_sync).foreign_keys = None
    assert inspector._get_involved_foreign_keys(mock_rel_sync) == {"t.l"}

    # fallback
    mock_rel_fall = MagicMock(spec=RelationshipProperty)
    mock_rel_fall.secondary = None
    mock_rel_fall.synchronize_pairs = None
    type(mock_rel_fall).foreign_keys = None
    assert inspector._get_involved_foreign_keys(mock_rel_fall) == set()


def test_default_model_description(inspector, engine):
    class NoDesc(SQLModel, table=True):
        __tablename__ = "no_desc"
        id: Optional[int] = Field(default=None, primary_key=True)

    SQLModel.metadata.create_all(engine)

    schema = json.loads(inspector.generate_llm_schema_from_models([NoDesc]))
    assert "Represents a NoDesc entity" in schema["NoDesc"]["description"]


@pytest.mark.parametrize(
    "rel_data, expected_id, expected_desc_part",
    [
        (
            {"related_model_name": "M", "type": "MANYTOONE", "uselist": False},
            "some_rel_ref_id",
            "_temp_id of the related M",
        ),
        (
            {"related_model_name": "M", "type": "ONETOMANY", "uselist": True},
            "some_rel_ref_ids",
            "list of _temp_ids for related M",
        ),
        ({"related_model_name": "M", "type": "ONETOMANY"}, None, None),  # No uselist
    ],
)
def test_process_relationship_for_llm_schema(
    inspector, rel_data, expected_id, expected_desc_part
):
    res = inspector._process_relationship_for_llm_schema("some_rel", rel_data, {})
    if expected_id is None:
        assert res is None
    else:
        assert res[0] == expected_id
        assert expected_desc_part in res[1]


@pytest.mark.parametrize(
    "models, check_fn",
    [
        ([Department], lambda s: "employees_ref_ids" in s["Department"]["fields"]),
        (
            [Employee, Department],
            lambda s: "department_ref_id" in s["Employee"]["fields"],
        ),
        (
            [Project, Member],
            lambda s: "members_ref_ids" in s["Project"]["fields"]
            and "projects_ref_ids" in s["Member"]["fields"],
        ),
        ([], lambda s: s == {}),
        (
            [Employee],
            lambda s: all(
                m.value in s["Employee"]["fields"]["status"] for m in StatusEnum
            ),
        ),
        (
            [ArticleScenarioModel],
            lambda s: "array[string]"
            in s["ArticleScenarioModel"]["fields"]["key_topics"].lower(),
        ),
        (
            [PlainSQLAlchemyModelWithPydanticHints],
            lambda s: s["PlainSQLAlchemyModelWithPydanticHints"]["fields"][
                "complex_field"
            ].startswith("object //"),
        ),
    ],
)
def test_generate_llm_schema(inspector, models, check_fn):
    schema_str = inspector.generate_llm_schema_from_models(models)
    schema = json.loads(schema_str)
    assert check_fn(schema)


def test_custom_descriptions_override(inspector):
    custom = {"Employee": {"name": "Custom Name", "_model_description": "Custom Model"}}
    schema = json.loads(
        inspector.generate_llm_schema_from_models(
            [Employee], custom_field_descriptions=custom
        )
    )
    assert "Custom Name" in schema["Employee"]["fields"]["name"]
    assert "Custom Model" in schema["Employee"]["description"]


def test_not_a_model_skip(inspector):
    class NotModel:
        pass

    schema = json.loads(
        inspector.generate_llm_schema_from_models([Department, NotModel])
    )
    assert "Department" in schema
    assert "NotModel" not in schema


def test_non_serializable_info(inspector):
    # Column info
    schema_col = json.loads(
        inspector.generate_llm_schema_from_models([ModelWithNonSerializableColInfo])
    )
    assert "Info: {" in schema_col["ModelWithNonSerializableColInfo"]["fields"]["data"]

    # Relationship info
    schema_rel = json.loads(
        inspector.generate_llm_schema_from_models(
            [ModelWithNonSerializableRelInfo, ModelRelatedToNonSerializable]
        )
    )
    assert (
        "Info: {"
        in schema_rel["ModelWithNonSerializableRelInfo"]["fields"]["related_ref_id"]
    )

    # Table info
    schema_table = json.loads(
        inspector.generate_llm_schema_from_models([ModelWithNonSerializableTableInfo])
    )
    assert "Info: {" in schema_table["ModelWithNonSerializableTableInfo"]["description"]


def test_default_desc_bare_rel(inspector, engine):
    class P(SQLModel, table=True):
        __tablename__ = "p_bare"
        id: Optional[int] = Field(default=None, primary_key=True)
        children: List["C"] = Relationship(back_populates="parent")

    class C(SQLModel, table=True):
        __tablename__ = "c_bare"
        id: Optional[int] = Field(default=None, primary_key=True)
        parent_id: Optional[int] = Field(default=None, foreign_key="p_bare.id")
        parent: Optional[P] = Relationship(back_populates="children")

    P.model_rebuild()
    C.model_rebuild()
    SQLModel.metadata.create_all(engine)

    schema = json.loads(inspector.generate_llm_schema_from_models([C]))
    assert "_temp_id of the related P" in schema["C"]["fields"]["parent_ref_id"]


def test_discovery_functions(inspector):
    class NoInspect:
        pass

    with patch(
        "extrai.core.schema_inspector.inspect", side_effect=NoInspectionAvailable
    ):
        res = []
        inspector._collect_all_sqla_models_recursively(NoInspect, res, set())
        assert NoInspect in res

    class NoneInspect(Base):
        __tablename__ = "none_insp"
        id: Mapped[int] = mapped_column(primary_key=True)

    with patch("extrai.core.schema_inspector.inspect", return_value=None):
        res = []
        inspector._collect_all_sqla_models_recursively(NoneInspect, res, set())
        assert NoneInspect in res

    with patch.object(inspector.logger, "warning") as mock_log:
        assert inspector.discover_sqlmodels_from_root(None) == []
        mock_log.assert_called()

    with patch.object(
        inspector, "_collect_all_sqla_models_recursively", side_effect=Exception
    ):

        class M(SQLModel, table=True):
            id: int = Field(primary_key=True)

        assert inspector.discover_sqlmodels_from_root(M) == []

    # Recursion guard
    base = declarative_base()

    class GuardM(base):
        __tablename__ = "guard_m"
        id: Mapped[int] = mapped_column(primary_key=True)

    with patch("extrai.core.schema_inspector.inspect") as mock_insp:
        inspector._collect_all_sqla_models_recursively(GuardM, set(), {GuardM})
        mock_insp.assert_not_called()

    # Success discovery
    class R(SQLModel, table=True):
        __tablename__ = "root_disc"
        id: Optional[int] = Field(default=None, primary_key=True)

    SQLModel.metadata.create_all(create_engine("sqlite:///:memory:"))
    assert len(inspector.discover_sqlmodels_from_root(R)) == 1


def test_unique_constraint_on_table(inspector, engine):
    local_base = declarative_base()

    class ModelUnique(local_base):
        __tablename__ = "model_unique"
        id: Mapped[int] = mapped_column(primary_key=True)
        col: Mapped[str] = mapped_column(String)
        __table_args__ = (UniqueConstraint("col"),)

    local_base.metadata.create_all(engine)
    schema = inspector.inspect_sqlalchemy_model(ModelUnique)
    assert schema["columns"]["col"]["unique"] is True
