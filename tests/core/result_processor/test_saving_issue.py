import logging
import pytest
from typing import Optional
from sqlmodel import Field, SQLModel, Session, create_engine, select
from extrai.core.result_processor import ResultProcessor
from extrai.core.model_registry import ModelRegistry
from extrai.core.analytics_collector import WorkflowAnalyticsCollector


# 1. Define a simple SQLModel for testing
class PetConfig(SQLModel, table=True):
    config_id: Optional[int] = Field(default=None, primary_key=True)
    airline_id: Optional[int] = Field(default=1)
    transport_type: str
    max_weight_lbs: Optional[float] = None
    advance_booking_days: Optional[int] = None
    notes: Optional[str] = None


# Mock data mimicking consensus output for structured extraction
mock_consensus_data = [
    {
        "transport_type": "cabin",
        "max_weight_lbs": 17.64,
        "notes": "For cats and dogs weighing less than 8 kg / 17.64 lb.",
    },
    {
        "transport_type": "cargo",
        "max_weight_lbs": 165.35,
        "notes": "For cats and dogs weighing more than 8 kg/17.64 lb. and up to 75 kg/165.35 lb.",
    },
    {
        "transport_type": "assistance",
        "advance_booking_days": 2,
        "notes": "Trained guide or service dogs.",
    },
]


@pytest.fixture
def session():
    engine = create_engine("sqlite:///:memory:")
    SQLModel.metadata.create_all(engine)
    with Session(engine) as session:
        yield session


def test_hydrate_and_persist_structured_output(session: Session):
    # Arrange
    logger = logging.getLogger("test_logger")
    model_registry = ModelRegistry(PetConfig, logger)
    analytics_collector = WorkflowAnalyticsCollector(logger)

    result_processor = ResultProcessor(
        model_registry=model_registry,
        analytics_collector=analytics_collector,
        logger=logger,
    )

    # Act
    # Hydrate objects, simulating direct hydration for structured output
    hydrated_objects = result_processor.hydrate(
        results=mock_consensus_data, db_session=session, default_model_type="PetConfig"
    )

    # Assert
    assert len(hydrated_objects) == 3

    # Verify that objects are in the session before commit (they should be)
    assert len(session.new) > 0

    # Persist the hydrated objects
    result_processor.persist(hydrated_objects, session)

    # Query the database to confirm persistence
    all_configs = session.exec(select(PetConfig)).all()
    assert len(all_configs) == 3

    # Check some data points
    cabin_config = session.exec(
        select(PetConfig).where(PetConfig.transport_type == "cabin")
    ).one()
    assert cabin_config.max_weight_lbs == 17.64

    cargo_config = session.exec(
        select(PetConfig).where(PetConfig.transport_type == "cargo")
    ).one()
    assert cargo_config.max_weight_lbs == 165.35

    assistance_config = session.exec(
        select(PetConfig).where(PetConfig.transport_type == "assistance")
    ).one()
    assert assistance_config.advance_booking_days == 2
