# tests/core/helpers/orchestrator_test_models.py

import enum
import datetime
from typing import List, Optional, Dict, Any
from sqlmodel import SQLModel, Field, Relationship

from sqlalchemy import Column, Integer, String, Boolean, ForeignKey, Table, func
from sqlalchemy import Enum as SAEnum, JSON as SAJson
from sqlalchemy.orm import (
    relationship as sa_relationship,
    declarative_base,
    Mapped,
    mapped_column,
    column_property,
)
from sqlalchemy.types import TypeDecorator

# SQLAlchemy Base for models defined in test_schema_inspector
Base: Any = declarative_base()


# --- Model Definitions moved from test_schema_inspector.py ---
class StatusEnum(enum.Enum):
    PENDING = "pending"
    COMPLETED = "completed"
    FAILED = "failed"


project_member_association_table_args = {
    "comment": "Association table for many-to-many between projects and members.",
    "info": {"association_type": "membership"},
}
project_member_association = Table(
    "project_member",
    Base.metadata,
    Column("project_id", ForeignKey("projects.id"), primary_key=True),
    Column("member_id", ForeignKey("members.id"), primary_key=True),
    **project_member_association_table_args,
)


class Department(Base):
    __tablename__ = "departments"
    __table_args__ = {
        "comment": "Stores all company departments.",
        "info": {"schema_version": "1.2"},
    }

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        comment="Unique Department ID (PK)",
        info={"pk_strategy": "auto"},
    )
    name: Mapped[str] = mapped_column(
        String(100),
        unique=True,
        nullable=False,
        comment="Official name of the department.",
    )
    employees: Mapped[List["Employee"]] = sa_relationship(
        back_populates="department",
        info={"relationship_detail": "All employees belonging to this department."},
    )


class Employee(Base):
    __tablename__ = "employees"
    __table_args__ = {
        "comment": "Stores detailed information about company employees.",
        "info": {"confidentiality": "high"},
    }

    id: Mapped[int] = mapped_column(
        Integer, primary_key=True, comment="Unique Employee ID (PK)"
    )
    name: Mapped[str] = mapped_column(
        String(100), nullable=False, comment="Full legal name of the employee."
    )
    email: Mapped[str] = mapped_column(
        String(100),
        unique=True,
        nullable=False,
        info={"validation_rule": "standard_email_format"},
    )
    age: Mapped[Optional[int]] = mapped_column(
        Integer, nullable=True, comment="Employee's age in years, if provided."
    )
    is_active: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
        nullable=False,
        comment="Indicates if the employee account is active.",
    )
    status: Mapped[Optional[StatusEnum]] = mapped_column(
        SAEnum(StatusEnum),
        nullable=True,
        comment="Current employment status of the employee.",
    )
    profile_data: Mapped[Optional[Dict[str, Any]]] = mapped_column(
        SAJson, nullable=True, comment="Additional profile information in JSON format."
    )

    department_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("departments.id"),
        nullable=False,
        comment="Foreign key linking to the employee's department.",
    )
    department: Mapped["Department"] = sa_relationship(
        back_populates="employees",
        info={"description": "The department this employee is assigned to."},
    )

    manager_id: Mapped[Optional[int]] = mapped_column(
        Integer,
        ForeignKey("employees.id"),
        nullable=True,
        comment="Foreign key linking to the employee's manager (self-referential).",
    )
    manager: Mapped[Optional["Employee"]] = sa_relationship(
        remote_side=[id],
        back_populates="managed_employees",
        info={"description": "The direct manager of this employee."},
    )
    managed_employees: Mapped[List["Employee"]] = sa_relationship(
        back_populates="manager",
        info={"description": "List of employees directly managed by this employee."},
    )


class Project(Base):
    __tablename__ = "projects"
    __table_args__ = {"comment": "Stores information about company projects."}

    id: Mapped[int] = mapped_column(
        Integer, primary_key=True, comment="Unique Project ID (PK)"
    )
    name: Mapped[str] = mapped_column(
        String(200), nullable=False, comment="Official name of the project."
    )
    members: Mapped[List["Member"]] = sa_relationship(
        secondary=project_member_association,
        back_populates="projects",
        info={"description": "Members participating in this project."},
    )


class Member(Base):
    __tablename__ = "members"
    __table_args__ = {"comment": "Stores information about project members."}
    id: Mapped[int] = mapped_column(
        Integer, primary_key=True, comment="Unique Member ID (PK)"
    )
    name: Mapped[str] = mapped_column(
        String(100), nullable=False, comment="Full name of the member."
    )
    projects: Mapped[List["Project"]] = sa_relationship(
        secondary=project_member_association,
        back_populates="members",
        info={"description": "Projects this member is associated with."},
    )


class StandaloneModel(Base):
    __tablename__ = "standalone_model_for_test"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(50))


class Screws(Base):
    __tablename__ = "screws"
    __table_args__ = {"comment": "Stores information about screws."}
    id: Mapped[int] = mapped_column(Integer, primary_key=True, comment="Screw ID")
    size: Mapped[str] = mapped_column(
        String(50), comment="Size of the screw (e.g., M5x20)"
    )
    support_id: Mapped[int] = mapped_column(
        ForeignKey("supports.id"), comment="FK to Support"
    )
    support: Mapped["Support"] = sa_relationship(back_populates="screws_list")


class Support(Base):
    __tablename__ = "supports"
    __table_args__ = {"comment": "Stores information about supports."}
    id: Mapped[int] = mapped_column(Integer, primary_key=True, comment="Support ID")
    name: Mapped[str] = mapped_column(String(100), comment="Name of the support")
    table_id: Mapped[int] = mapped_column(
        ForeignKey("tables.id"), comment="FK to TableModel"
    )  # Updated comment
    table: Mapped["TableModel"] = sa_relationship(
        back_populates="supports"
    )  # Renamed Mapped type
    screws_list: Mapped[List["Screws"]] = sa_relationship(
        back_populates="support",
        info={"description": "List of screws for this support"},
    )


class TableModel(Base):
    __tablename__ = "tables"
    __table_args__ = {"comment": "Stores information about tables."}
    id: Mapped[int] = mapped_column(Integer, primary_key=True, comment="Table ID")
    material: Mapped[str] = mapped_column(String(100), comment="Material of the table")
    supports: Mapped[List["Support"]] = sa_relationship(
        back_populates="table", info={"description": "List of supports for this table"}
    )


class ModelWithColumnProperty(Base):
    __tablename__ = "model_with_col_prop_test"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    data: Mapped[str] = mapped_column(String)
    data_length: Mapped[int] = column_property(func.length(data))


class ModelWithNonSerializableColInfo(Base):
    __tablename__ = "model_non_serial_col"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    data: Mapped[str] = mapped_column(
        String, info={"description": "data field", "non_serializable": set([1, 2, 3])}
    )


class ModelRelatedToNonSerializable(Base):
    __tablename__ = "model_related_to_non_serial"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String)


class ModelWithNonSerializableRelInfo(Base):
    __tablename__ = "model_non_serial_rel"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    related_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("model_related_to_non_serial.id")
    )
    related: Mapped[Optional[ModelRelatedToNonSerializable]] = sa_relationship(
        info={"description": "relation", "non_serializable": datetime.datetime.now}  # type: ignore
    )


class ModelWithNonSerializableTableInfo(Base):
    __tablename__ = "model_non_serial_table"
    __table_args__ = {"info": {"description": "table", "non_serializable": object()}}
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String)


class PlainSQLAlchemyModelWithPydanticHints(Base):
    __tablename__ = "plain_sa_model_pydantic_hints"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String)
    complex_field: Mapped[Optional[List[Dict[str, Any]]]] = mapped_column(
        SAJson, nullable=True
    )
    simple_sqla_type_field: Mapped[Optional[int]] = mapped_column(
        Integer, nullable=True
    )


class CustomExceptionAccessType(TypeDecorator):
    impl = String
    cache_ok = True

    @property
    def python_type(self):
        raise Exception("Simulated error accessing python_type")


class ModelWithCustomColType(Base):
    __tablename__ = "model_with_custom_col_type"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    custom_field: Mapped[str] = mapped_column(CustomExceptionAccessType)


class FKParent(Base):
    __tablename__ = "fk_parent"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String)
    children_sync: Mapped[List["FKChildSync"]] = sa_relationship(
        back_populates="parent"
    )


class FKChildSync(Base):
    __tablename__ = "fk_child_sync"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    parent_id_col: Mapped[int] = mapped_column(ForeignKey("fk_parent.id"))
    parent: Mapped["FKParent"] = sa_relationship(back_populates="children_sync")


class FKParentDirect(Base):
    __tablename__ = "fk_parent_direct"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    children_direct: Mapped[List["FKChildDirect"]] = sa_relationship(
        "FKChildDirect",
        primaryjoin="FKParentDirect.id == FKChildDirect.parent_fk_col_name",
        foreign_keys="[FKChildDirect.parent_fk_col_name]",
    )


class FKChildDirect(Base):
    __tablename__ = "fk_child_direct"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    parent_fk_col_name: Mapped[int] = mapped_column(ForeignKey("fk_parent_direct.id"))


class ViewOnlyParent(Base):
    __tablename__ = "viewonly_parent"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String)
    children: Mapped[List["ViewOnlyChild"]] = sa_relationship(
        "ViewOnlyChild",
        primaryjoin="ViewOnlyParent.id == ViewOnlyChild.parent_id",
        foreign_keys="[ViewOnlyChild.parent_id]",
        viewonly=True,
    )


class ViewOnlyChild(Base):
    __tablename__ = "viewonly_child"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    parent_id: Mapped[int] = mapped_column(ForeignKey("viewonly_parent.id"))
    data: Mapped[str] = mapped_column(String)


# --- SQLModel Definition for ArticleInfo-like Scenario (moved from test_schema_inspector) ---
class ArticleScenarioModel(SQLModel, table=True):
    __tablename__ = "article_scenario_model"  # Keep original name if it's already in DB
    id: Optional[int] = Field(default=None, primary_key=True)
    title: Optional[str] = Field(default=None, description="Article title")
    key_topics: Optional[List[str]] = Field(
        default=None, sa_column=Column(SAJson), description="List of key topics."
    )  # Use SAJson
    categories: Optional[List[str]] = Field(
        default=None, sa_column=Column(SAJson), description="List of categories."
    )  # Use SAJson
    meta_data: Optional[Dict[str, Any]] = Field(
        default=None, sa_column=Column(SAJson), description="Meta data dictionary."
    )  # Use SAJson


# --- Existing SQLModel definitions from orchestrator_test_models.py ---
class EmployeeModel(SQLModel, table=True):
    __tablename__ = "test_employees"
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str
    department_id: Optional[int] = Field(
        default=None, foreign_key="test_departments.id"
    )
    department: Optional["DepartmentModel"] = Relationship(back_populates="employees")
    manager_id: Optional[int] = Field(default=None, foreign_key="test_employees.id")
    manager: Optional["EmployeeModel"] = Relationship(
        back_populates="managed_employees",
        sa_relationship_kwargs={"remote_side": "EmployeeModel.id"},
    )
    managed_employees: List["EmployeeModel"] = Relationship(back_populates="manager")


class DepartmentModel(SQLModel, table=True):
    __tablename__ = "test_departments"
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str
    employees: List["EmployeeModel"] = Relationship(back_populates="department")


# E2E Models
class OrderItemModel(SQLModel, table=True):
    __tablename__ = "test_order_items"
    id: Optional[int] = Field(default=None, primary_key=True)
    product_sku: str
    quantity: int
    order_id: Optional[int] = Field(default=None, foreign_key="test_orders.id")
    order_ref_on_item: Optional["OrderModel"] = Relationship(
        back_populates="order_entries"
    )


class ProductModel(SQLModel, table=True):
    __tablename__ = "test_products"
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str
    description: str
    version: Optional[int] = Field(default=None)


class OrderModel(SQLModel, table=True):
    __tablename__ = "test_orders"
    id: Optional[int] = Field(default=None, primary_key=True)
    order_ref: str = Field(unique=True)
    customer_name: str
    order_entries: List["OrderItemModel"] = Relationship(
        back_populates="order_ref_on_item",
        sa_relationship_kwargs={"collection_class": list},
    )


# Update forward references for SQLModel classes
EmployeeModel.model_rebuild()
DepartmentModel.model_rebuild()
OrderItemModel.model_rebuild()
ProductModel.model_rebuild()
OrderModel.model_rebuild()
ArticleScenarioModel.model_rebuild()  # Added for the moved SQLModel


# --- Models from test_hierarchical_extractor.py ---
class SimpleModel(SQLModel, table=True):
    __tablename__ = "simplemodel"
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str


class GrandChildModel(SQLModel, table=True):
    __tablename__ = "grandchildmodel"  # Explicit table name for clarity
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str
    child_id: Optional[int] = Field(default=None, foreign_key="childmodel.id")
    child: Optional["ChildModel"] = Relationship(back_populates="grand_children")


class ChildModel(SQLModel, table=True):
    __tablename__ = "childmodel"  # Explicit table name for clarity
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str
    parent_id: Optional[int] = Field(default=None, foreign_key="parentmodel.id")
    parent: Optional["ParentModel"] = Relationship(back_populates="children")
    grand_children: List[GrandChildModel] = Relationship(back_populates="child")


class ParentModel(SQLModel, table=True):
    __tablename__ = "parentmodel"  # Explicit table name for clarity
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str
    children: List[ChildModel] = Relationship(back_populates="parent")


class EmployeeCycleTest(SQLModel, table=True):  # Renamed from Employee
    __tablename__ = "employee_table_for_cycle_test"
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str
    manager_id: Optional[int] = Field(
        default=None, foreign_key="employee_table_for_cycle_test.id"
    )

    manager: Optional["EmployeeCycleTest"] = Relationship(
        back_populates="reports",
        sa_relationship_kwargs={"remote_side": "EmployeeCycleTest.id"},
    )
    reports: List["EmployeeCycleTest"] = Relationship(back_populates="manager")


# --- Test-specific models for processed_parent_child_relationship_instances test ---
class GrandChildProcTest(SQLModel, table=True):
    __tablename__ = "grandchild_proctest_table"
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str
    shared_child_id: Optional[int] = Field(
        default=None, foreign_key="sharedchildm_proctest_table.id"
    )
    shared_child: Optional["SharedChildMProcTest"] = Relationship(
        back_populates="grand_children"
    )


class SharedChildMProcTest(SQLModel, table=True):
    __tablename__ = "sharedchildm_proctest_table"
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str

    parent_x_id: Optional[int] = Field(
        default=None, foreign_key="parentx_proctest_table.id"
    )
    parent_y_id: Optional[int] = Field(
        default=None, foreign_key="parenty_proctest_table.id"
    )

    parent_x: Optional["ParentXProcTest"] = Relationship(
        back_populates="shared_children_m_from_x"
    )
    parent_y: Optional["ParentYProcTest"] = Relationship(
        back_populates="shared_children_m_from_y"
    )
    grand_children: List[GrandChildProcTest] = Relationship(
        back_populates="shared_child"
    )


class ParentXProcTest(SQLModel, table=True):
    __tablename__ = "parentx_proctest_table"
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str
    root_id: Optional[int] = Field(default=None, foreign_key="root_proctest_table.id")
    root: Optional["RootProcTest"] = Relationship(back_populates="parents_x")
    shared_children_m_from_x: List[SharedChildMProcTest] = Relationship(
        back_populates="parent_x"
    )


class ParentYProcTest(SQLModel, table=True):
    __tablename__ = "parenty_proctest_table"
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str
    root_id: Optional[int] = Field(default=None, foreign_key="root_proctest_table.id")
    root: Optional["RootProcTest"] = Relationship(back_populates="parents_y")
    shared_children_m_from_y: List[SharedChildMProcTest] = Relationship(
        back_populates="parent_y"
    )


class RootProcTest(SQLModel, table=True):
    __tablename__ = "root_proctest_table"
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str
    parents_x: List[ParentXProcTest] = Relationship(back_populates="root")
    parents_y: List[ParentYProcTest] = Relationship(back_populates="root")


# --- Test-specific models for visited_items_in_queue test ---
class CommonChildVisitedTest(SQLModel, table=True):
    __tablename__ = "commonchild_visited_table"
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str
    parent_pwtp_id: Optional[int] = Field(
        default=None, foreign_key="parentwithtwopaths_visited_table.id"
    )
    parent_pwtp: Optional["ParentWithTwoPathsVisitedTest"] = Relationship(
        back_populates="children_via_a"
    )


class ParentWithTwoPathsVisitedTest(SQLModel, table=True):
    __tablename__ = "parentwithtwopaths_visited_table"
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str
    root_id: Optional[int] = Field(default=None, foreign_key="root_visited_table.id")
    root: Optional["RootVisitedTest"] = Relationship(back_populates="parents_pwtp")

    children_via_a: List[CommonChildVisitedTest] = Relationship(
        back_populates="parent_pwtp",
        sa_relationship_kwargs={
            "foreign_keys": "[CommonChildVisitedTest.parent_pwtp_id]"
        },
    )
    children_via_b: List[CommonChildVisitedTest] = Relationship(
        back_populates="parent_pwtp",
        sa_relationship_kwargs={
            "foreign_keys": "[CommonChildVisitedTest.parent_pwtp_id]",
            "overlaps": "children_via_a,parent_pwtp",
        },
    )


class RootVisitedTest(SQLModel, table=True):
    __tablename__ = "root_visited_table"
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str
    parents_pwtp: List[ParentWithTwoPathsVisitedTest] = Relationship(
        back_populates="root"
    )


# Update forward references for newly added SQLModel classes
SimpleModel.model_rebuild()
GrandChildModel.model_rebuild()
ChildModel.model_rebuild()
ParentModel.model_rebuild()
EmployeeCycleTest.model_rebuild()
GrandChildProcTest.model_rebuild()
SharedChildMProcTest.model_rebuild()
ParentXProcTest.model_rebuild()
ParentYProcTest.model_rebuild()
RootProcTest.model_rebuild()
CommonChildVisitedTest.model_rebuild()
ParentWithTwoPathsVisitedTest.model_rebuild()
RootVisitedTest.model_rebuild()
