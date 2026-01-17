import pytest
from decimal import Decimal
from typing import List, Optional
from sqlmodel import SQLModel, Field, Relationship
from extrai.utils.serialization_utils import serialize_sqlmodel_with_relationships

# Define test models
class Item(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str
    price: Decimal
    order_id: Optional[int] = Field(default=None, foreign_key="order.id")
    order: Optional["Order"] = Relationship(back_populates="items")

class Order(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    description: str
    items: List[Item] = Relationship(back_populates="order")

def test_serialize_basic():
    item = Item(name="Test Item", price=Decimal("10.50"), id=1)
    serialized = serialize_sqlmodel_with_relationships(item)
    assert serialized["name"] == "Test Item"
    # Pydantic v2 default for Decimal serialization in mode='json' is string
    assert serialized["price"] == "10.50" 
    assert serialized["id"] == 1

def test_serialize_with_relationships():
    order = Order(id=1, description="Test Order")
    item1 = Item(name="Item 1", price=Decimal("10.50"), id=1)
    item2 = Item(name="Item 2", price=Decimal("20.00"), id=2)
    order.items = [item1, item2]
    
    serialized = serialize_sqlmodel_with_relationships(order)
    
    assert serialized["id"] == 1
    assert "items" in serialized
    assert len(serialized["items"]) == 2
    assert serialized["items"][0]["name"] == "Item 1"
    assert serialized["items"][0]["price"] == "10.50"
    assert serialized["items"][1]["name"] == "Item 2"

def test_circular_reference():
    order = Order(id=1, description="Circular Order")
    item = Item(name="Circular Item", price=Decimal("5.00"), id=1)
    
    # Create cycle
    order.items = [item]
    item.order = order
    
    serialized = serialize_sqlmodel_with_relationships(order)
    
    assert serialized["id"] == 1
    assert serialized["items"][0]["name"] == "Circular Item"
    # The 'order' inside the item should be empty dict to avoid recursion
    assert serialized["items"][0]["order"] == {}

def test_transient_relationship_access():
    """Test that we can access relationships on transient objects correctly."""
    order = Order(id=1, description="Transient")
    # items not set
    serialized = serialize_sqlmodel_with_relationships(order)
    # Should not crash.
    # SQLModel relationships might return [] or not be present depending on version/config.
    # But if present, it should be empty list.
    assert serialized["id"] == 1
    if "items" in serialized:
        assert serialized["items"] == []
