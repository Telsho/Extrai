import logging
from typing import List, Optional
from sqlmodel import Field, SQLModel, Relationship, create_engine, Session, select
from extrai.core.result_processor import ResultProcessor
from extrai.core.model_registry import ModelRegistry

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Define Models
class Parent(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str
    children: List["Child"] = Relationship(back_populates="parent")


class Child(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str
    parent_id: Optional[int] = Field(default=None, foreign_key="parent.id")
    parent: Optional[Parent] = Relationship(back_populates="children")


# Mock Analytics
class MockAnalytics:
    def record_hydration_success(self, count):
        logger.info(f"Analytics: Hydration Success {count}")

    def record_hydration_failure(self):
        logger.info("Analytics: Hydration Failure")


def test_pk_collision_prevention():
    # 1. Setup DB
    engine = create_engine("sqlite:///:memory:")
    SQLModel.metadata.create_all(engine)

    with Session(engine) as session:
        # 2. Pre-populate DB with ID=1 for Parent and Child
        existing_parent = Parent(id=1, name="Existing Parent")
        existing_child = Child(id=1, name="Existing Child", parent=existing_parent)
        session.add(existing_parent)
        session.add(existing_child)
        session.commit()

        # Verify they exist
        p1 = session.get(Parent, 1)
        c1 = session.get(Child, 1)
        assert p1 is not None and p1.name == "Existing Parent"
        assert c1 is not None and c1.name == "Existing Child"
        logger.info("Pre-existing Parent ID=1 and Child ID=1 confirmed.")

    # 3. Prepare ResultProcessor
    registry = ModelRegistry(root_model=Parent, logger=logger)
    processor = ResultProcessor(
        model_registry=registry, analytics_collector=MockAnalytics(), logger=logger
    )

    # 4. Create Input Data with ID Collision (ID=1) for both Parent and Child
    # The LLM outputs ID=1, but we want it to be ignored and a new ID assigned.
    input_data = [
        {
            "_type": "Parent",
            "id": 1,  # CONFLICTING ID!
            "name": "New LLM Parent",
            "children": [
                {
                    "_type": "Child",
                    "id": 1,  # CONFLICTING ID!
                    "name": "New LLM Child",
                }
            ],
        }
    ]

    # 5. Run Hydration (DirectHydrator)
    # Passing default_model_type forces DirectHydrator if _temp_id is missing,
    # but here we rely on _temp_id check or explicit default.
    # The input_data does NOT have _temp_id, so ResultProcessor selects DirectHydrator.

    logger.info("Starting Hydration...")
    # Use a new session for the operation
    with Session(engine) as db_session:
        # Since we're using a real DB session (not in-memory inside hydrate), we pass it
        hydrated_objects = processor.hydrate(input_data, db_session=db_session)
        processor.persist(hydrated_objects, db_session)

        # 6. Verify Results
        # We expect:
        # - The original Parent (ID=1) is UNTOUCHED.
        # - A NEW Parent is created (ID!=1, likely 2).
        # - The Child is linked to the NEW Parent.

        parents = db_session.exec(select(Parent)).all()
        children = db_session.exec(select(Child)).all()
        logger.info(f"Total Parents in DB: {len(parents)}")
        logger.info(f"Total Children in DB: {len(children)}")

        for p in parents:
            logger.info(f"Parent ID: {p.id}, Name: {p.name}")
        for c in children:
            logger.info(f"Child ID: {c.id}, Name: {c.name}, ParentID: {c.parent_id}")

        assert len(parents) == 2, "Should have 2 parents"
        assert len(children) == 2, "Should have 2 children"

        # Verify New Parent
        new_parent = next(p for p in parents if p.name == "New LLM Parent")
        assert new_parent.id != 1, (
            f"New parent should NOT have ID 1, got {new_parent.id}"
        )

        # Verify New Child
        new_child = next(c for c in children if c.name == "New LLM Child")
        assert new_child.id != 1, f"New child should NOT have ID 1, got {new_child.id}"
        assert new_child.parent_id == new_parent.id, (
            f"New child should be linked to new parent {new_parent.id}, got {new_child.parent_id}"
        )

        # Verify Integrity of Existing Data
        existing_p = db_session.get(Parent, 1)
        existing_c = db_session.get(Child, 1)
        assert existing_p.name == "Existing Parent"
        assert existing_c.name == "Existing Child"
        assert existing_c.parent_id == 1


if __name__ == "__main__":
    try:
        test_pk_collision_prevention()
        logger.info("✅ TEST PASSED: ID Collision successfully prevented!")
    except Exception as e:
        logger.error(f"❌ TEST FAILED: {e}", exc_info=True)
        exit(1)
