from typing import List, Optional
from sqlmodel import SQLModel, Field, Relationship
from pydantic import BaseModel
from extrai.core.model_wrapper_builder import ModelWrapperBuilder

# Define test models
class TestChild(SQLModel, table=True):
    __tablename__ = "test_child"
    __table_args__ = {'extend_existing': True}
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str
    parent_id: Optional[int] = Field(default=None, foreign_key="test_parent.id")
    parent: Optional["TestParent"] = Relationship(back_populates="children")

class TestParent(SQLModel, table=True):
    __tablename__ = "test_parent"
    __table_args__ = {'extend_existing': True}
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str
    children: List[TestChild] = Relationship(back_populates="parent")

class TestModelWrapperBuilder:
    def test_generate_wrapper_model(self):
        builder = ModelWrapperBuilder()
        wrapper_model = builder.generate_wrapper_model(TestParent)
        
        # Check wrapper structure
        assert issubclass(wrapper_model, BaseModel)
        assert "entities" in wrapper_model.model_fields
        assert wrapper_model.__name__ == "TestParentExtractionResult"
        
        # Check nested structure
        entities_field = wrapper_model.model_fields["entities"]
        
        # We can try to instantiate it to verify structure
        parent_structure_cls = entities_field.annotation.__args__[0]
        assert parent_structure_cls.__name__ == "TestParentStructure"
        
        assert "name" in parent_structure_cls.model_fields
        assert "children" in parent_structure_cls.model_fields
        
        children_field = parent_structure_cls.model_fields["children"]
        # Should be List[TestChildStructure]
        child_structure_cls = children_field.annotation.__args__[0]
        assert child_structure_cls.__name__ == "TestChildStructure"
        
        assert "name" in child_structure_cls.model_fields
        # Child should NOT have parent field to avoid recursion if we implemented that logic
        assert "parent" not in child_structure_cls.model_fields

    def test_circular_reference_handling(self):
        # Already covered by the check above (Child should not have 'parent' field)
        # because the builder skips MANYTOONE relationships.
        builder = ModelWrapperBuilder()
        wrapper_model = builder.generate_wrapper_model(TestParent)
        
        parent_structure_cls = wrapper_model.model_fields["entities"].annotation.__args__[0]
        children_field = parent_structure_cls.model_fields["children"]
        child_structure_cls = children_field.annotation.__args__[0]
        
        assert "parent" not in child_structure_cls.model_fields
