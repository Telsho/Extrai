
import logging
import pytest
from typing import List, Optional
from decimal import Decimal
from sqlmodel import Field, SQLModel, Session, create_engine, Relationship, select
from extrai.core.result_processor import ResultProcessor, ModelRegistry
from extrai.core.analytics_collector import WorkflowAnalyticsCollector

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Simplified Models for Reproducing the Issue ---

class Zone(SQLModel, table=True):
    """A named geographic zone used in pricing rules."""
    __tablename__ = "zones"
    
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str = Field(max_length=100)
    items_text: str = Field(description="Comma-separated list of locations.")
    
    origin_rules: List["PricingRule"] = Relationship(
        back_populates="origin_zone",
        sa_relationship_kwargs={"primaryjoin": "PricingRule.origin_zone_id==Zone.id", "foreign_keys": "[PricingRule.origin_zone_id]"}
    )
    destination_rules: List["PricingRule"] = Relationship(
        back_populates="destination_zone",
        sa_relationship_kwargs={"primaryjoin": "PricingRule.destination_zone_id==Zone.id", "foreign_keys": "[PricingRule.destination_zone_id]"}
    )

class PricingRule(SQLModel, table=True):
    """Simplified pricing rules for pet transport using Zone-based logic."""
    __tablename__ = "pricing_rules"
    
    rule_id: Optional[int] = Field(default=None, primary_key=True)
    
    origin_zone_id: int = Field(foreign_key="zones.id")
    destination_zone_id: int = Field(foreign_key="zones.id")
    
    base_fee: Decimal = Field(max_digits=10, decimal_places=2)
    currency: str = Field(default="USD", max_length=7)
    
    origin_zone: Optional[Zone] = Relationship(
        back_populates="origin_rules",
        sa_relationship_kwargs={"foreign_keys": "[PricingRule.origin_zone_id]"}
    )
    destination_zone: Optional[Zone] = Relationship(
        back_populates="destination_rules",
        sa_relationship_kwargs={"foreign_keys": "[PricingRule.destination_zone_id]"}
    )

@pytest.fixture
def db_session():
    engine = create_engine("sqlite:///:memory:")
    SQLModel.metadata.create_all(engine)
    with Session(engine) as session:
        # Pre-populate with EXISTING objects to create the ID offset
        z1 = Zone(name="Existing Zone 1", items_text="country:US")
        z2 = Zone(name="Existing Zone 2", items_text="country:CA")
        session.add(z1)
        session.add(z2)
        
        session.commit()
        
        # Verify IDs are 1 and 2
        session.refresh(z1)
        session.refresh(z2)
        logger.info(f"Pre-populated DB. Existing Zone IDs: {z1.id}, {z2.id}")
        
        yield session

@pytest.fixture
def result_processor():
    # Register only the necessary models
    model_registry = ModelRegistry(Zone, logger)
    model_registry.model_map["Zone"] = Zone
    model_registry.model_map["PricingRule"] = PricingRule
        
    analytics = WorkflowAnalyticsCollector(logger)
    return ResultProcessor(model_registry, analytics, logger)

def test_pricing_rule_zone_linkage_offset_repro(db_session, result_processor):
    """
    Reproduce the issue where PricingRule links to the WRONG Zone ID because
    existing objects in the DB cause an ID offset.
    """
    
    input_data = [
        # New Zones (Input IDs 10, 11)
        {
            "_type": "Zone",
            "id": 10,
            "name": "New Zone A",
            "items_text": "country:FR"
        },
        {
            "_type": "Zone",
            "id": 11,
            "name": "New Zone B",
            "items_text": "country:DE"
        },
        # PricingRule linking to Zones 10, 11
        {
            "_type": "PricingRule",
            "rule_id": 999,
            "origin_zone_id": 10,      # Should link to "New Zone A" (which will get ID 3)
            "destination_zone_id": 11, # Should link to "New Zone B" (which will get ID 4)
            "base_fee": 150.00,
            "currency": "USD"
        }
    ]
    
    # Process
    objects = result_processor.hydrate(input_data, db_session=db_session)
    result_processor.persist(objects, db_session)
    
    # Verify persistence
    zone_a = db_session.exec(select(Zone).where(Zone.name == "New Zone A")).first()
    zone_b = db_session.exec(select(Zone).where(Zone.name == "New Zone B")).first()
    
    assert zone_a is not None
    assert zone_b is not None
    
    # Expected IDs for new zones should be 3 and 4 (since 1 and 2 exist)
    logger.info(f"Zone A ID: {zone_a.id} (Input: 10)")
    logger.info(f"Zone B ID: {zone_b.id} (Input: 11)")
    
    assert zone_a.id == 3
    assert zone_b.id == 4

    # Verify PricingRule
    rule = db_session.exec(select(PricingRule).where(PricingRule.base_fee == 150.00)).first()
    assert rule is not None
    
    logger.info(f"PricingRule OriginFK: {rule.origin_zone_id}")
    logger.info(f"PricingRule DestFK: {rule.destination_zone_id}")
    
    # ASSERTIONS
    assert rule.origin_zone_id == zone_a.id, (
        f"PricingRule Origin FK ({rule.origin_zone_id}) does not match Zone A ID ({zone_a.id}). "
        f"It might be pointing to the input ID (10) or an offset ID."
    )
    assert rule.destination_zone_id == zone_b.id, (
        f"PricingRule Dest FK ({rule.destination_zone_id}) does not match Zone B ID ({zone_b.id})."
    )
