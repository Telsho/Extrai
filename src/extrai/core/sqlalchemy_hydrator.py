from typing import (
    Dict,
    List,
    Any,
    Type,
    Optional,
    get_origin,
    get_args,
    Union,
    NamedTuple,
)
import uuid
from sqlalchemy.orm import Session
from sqlmodel import SQLModel

SQLModelInstance = SQLModel


class PrimaryKeyInfo(NamedTuple):
    name: Optional[str]
    type: Optional[Type[Any]]
    has_uuid_factory: bool


class SQLAlchemyHydrator:
    """
    Hydrates SQLModel objects from consensus JSON data.
    It uses a two-pass strategy: first, create all object instances,
    then link their relationships using temporary IDs.
    """

    def __init__(self, session: Session):
        """
        Initializes the Hydrator.

        Args:
            session: The SQLAlchemy session to use for database operations
                     and instance management (e.g., adding instances).
        """
        self.session: Session = session
        self.temp_id_to_instance_map: Dict[
            str, SQLModelInstance
        ] = {}  # Stores _temp_id -> SQLModel instance

    def _filter_special_fields(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Removes _temp_id, _type, and relationship reference fields before Pydantic validation."""
        return {
            k: v
            for k, v in data.items()
            if k not in ["_temp_id", "_type"]
            and not k.endswith("_ref_id")
            and not k.endswith("_ref_ids")
        }

    def _validate_entities_list(self, entities_list: List[Dict[str, Any]]) -> None:
        """Performs initial validation on the input entities list."""
        if not isinstance(entities_list, list):
            raise TypeError(
                f"Input 'entities_list' must be a list. Got: {type(entities_list)}"
            )
        if not all(isinstance(item, dict) for item in entities_list):
            first_non_dict = next(
                (item for item in entities_list if not isinstance(item, dict)), None
            )
            raise ValueError(
                "All items in 'entities_list' must be dictionaries. "
                f"Found an item of type: {type(first_non_dict)}."
            )

    def _get_primary_key_info(self, model_class: Type[SQLModel]) -> PrimaryKeyInfo:
        """Introspects the model to find primary key details."""
        for field_name, model_field in model_class.model_fields.items():
            if getattr(model_field, "primary_key", False):
                pk_type = model_field.annotation
                origin_type = get_origin(pk_type)
                if origin_type is Union:
                    args = get_args(pk_type)
                    pk_type = next(
                        (
                            arg
                            for arg in args
                            if arg is not type(None) and arg is not None
                        ),
                        None,
                    )

                has_uuid_factory = False
                if model_field.default_factory:
                    factory_func = model_field.default_factory
                    if factory_func is uuid.uuid4 or (
                        callable(factory_func)
                        and getattr(factory_func, "__name__", "").lower() == "uuid4"
                    ):
                        has_uuid_factory = True

                return PrimaryKeyInfo(
                    name=field_name, type=pk_type, has_uuid_factory=has_uuid_factory
                )

        return PrimaryKeyInfo(name=None, type=None, has_uuid_factory=False)

    def _generate_pk_if_needed(
        self, instance: SQLModelInstance, model_class: Type[SQLModel]
    ) -> None:
        """Generates a primary key for the instance if it's needed."""
        pk_info = self._get_primary_key_info(model_class)

        if not pk_info.name:
            return

        current_pk_value = getattr(instance, pk_info.name, None)

        if current_pk_value is not None or pk_info.has_uuid_factory:
            return

        if pk_info.type is uuid.UUID:
            setattr(instance, pk_info.name, uuid.uuid4())
        elif pk_info.type is str:
            setattr(instance, pk_info.name, str(uuid.uuid4()))

    def _create_single_instance(
        self,
        entity_data: Dict[str, Any],
        model_schema_map: Dict[str, Type[SQLModel]],
    ) -> None:
        """Creates a single SQLModel instance from its dictionary representation."""
        _temp_id = entity_data.get("_temp_id")
        _type = entity_data.get("_type")

        if not _temp_id or not _type:
            raise ValueError(
                "Entity data in 'entities' list is missing '_temp_id' or '_type'."
            )
        if _type not in model_schema_map:
            raise ValueError(
                f"No SQLModel class found in model_schema_map for type: '{_type}'."
            )
        if _temp_id in self.temp_id_to_instance_map:
            raise ValueError(
                f"Duplicate _temp_id '{_temp_id}' found in 'entities' list."
            )

        model_class = model_schema_map[_type]

        filtered_data = self._filter_special_fields(entity_data.copy())

        pk_field_name: Optional[str] = None
        for field_name, model_field in model_class.model_fields.items():
            if getattr(model_field, "primary_key", False):
                pk_field_name = field_name
                break

        if pk_field_name and pk_field_name in filtered_data:
            del filtered_data[pk_field_name]

        try:
            instance = model_class.model_validate(filtered_data)
        except Exception as e:
            raise ValueError(
                f"Failed to instantiate/validate SQLModel '{_type}' for _temp_id '{_temp_id}': {e}"
            ) from e

        self._generate_pk_if_needed(instance, model_class)
        self.temp_id_to_instance_map[_temp_id] = instance

    def _create_and_map_instances(
        self,
        entities_list: List[Dict[str, Any]],
        model_schema_map: Dict[str, Type[SQLModel]],
    ) -> None:
        """Pass 1: Creates and maps all SQLModel instances."""
        for entity_data in entities_list:
            self._create_single_instance(entity_data, model_schema_map)

    def _link_to_one_relation(
        self,
        instance: SQLModelInstance,
        relation_name: str,
        ref_id: Any,
        entity_data: Dict[str, Any],
    ) -> None:
        """Handles the logic for a single to-one relationship."""
        if ref_id is None:
            setattr(instance, relation_name, None)
            return

        if isinstance(ref_id, str) and ref_id in self.temp_id_to_instance_map:
            related_instance = self.temp_id_to_instance_map[ref_id]
            setattr(instance, relation_name, related_instance)
        else:
            _temp_id = entity_data.get("_temp_id", "N/A")
            _type = entity_data.get("_type", "N/A")
            print(
                f"Warning: Referenced _temp_id '{ref_id}' for relation "
                f"'{relation_name}' on instance '{_temp_id}' (type: {_type}) not found or invalid type."
            )

    def _link_to_many_relation(
        self,
        instance: SQLModelInstance,
        relation_name: str,
        ref_ids: Any,
        entity_data: Dict[str, Any],
    ) -> None:
        """Handles the logic for a single to-many relationship."""
        _temp_id = entity_data.get("_temp_id", "N/A")
        _type = entity_data.get("_type", "N/A")

        if not isinstance(ref_ids, list):
            if ref_ids is not None:
                print(
                    f"Warning: Value for '{relation_name}_ref_ids' on instance '{_temp_id}' is not a list as expected for '_ref_ids'. Value: {ref_ids}"
                )
            setattr(instance, relation_name, [])
            return

        related_instances = []
        for ref_id in ref_ids:
            if isinstance(ref_id, str) and ref_id in self.temp_id_to_instance_map:
                related_instances.append(self.temp_id_to_instance_map[ref_id])
            else:
                print(
                    f"Warning: Referenced _temp_id '{ref_id}' in list for relation "
                    f"'{relation_name}' on instance '{_temp_id}' (type: {_type}) not found or invalid type."
                )
        setattr(instance, relation_name, related_instances)

    def _link_relations_for_instance(self, entity_data: Dict[str, Any]) -> None:
        """Links relationships for a single instance by dispatching to specialized helpers."""
        _temp_id = entity_data["_temp_id"]
        instance = self.temp_id_to_instance_map[_temp_id]

        for key, value in entity_data.items():
            if key.endswith("_ref_id"):
                relation_name = key[:-7]
                if hasattr(instance, relation_name):
                    self._link_to_one_relation(
                        instance, relation_name, value, entity_data
                    )
            elif key.endswith("_ref_ids"):
                relation_name = key[:-8]
                if hasattr(instance, relation_name):
                    self._link_to_many_relation(
                        instance, relation_name, value, entity_data
                    )

    def _link_relationships(self, entities_list: List[Dict[str, Any]]) -> None:
        """Pass 2: Links all created instances together."""
        for entity_data in entities_list:
            self._link_relations_for_instance(entity_data)

    def _add_instances_to_session(self) -> None:
        """Adds all created instances to the SQLAlchemy session."""
        for instance in self.temp_id_to_instance_map.values():
            self.session.add(instance)

    def hydrate(
        self,
        entities_list: List[Dict[str, Any]],
        model_schema_map: Dict[str, Type[SQLModel]],
    ) -> List[SQLModelInstance]:
        """
        Hydrates SQLModel objects from a list of entity data dictionaries.
        """
        self._validate_entities_list(entities_list)

        self.temp_id_to_instance_map.clear()

        # Pass 1: Create all object instances without relationships.
        self._create_and_map_instances(entities_list, model_schema_map)

        # Pass 2: Link the created instances together.
        self._link_relationships(entities_list)

        # Add the completed object graph to the session.
        self._add_instances_to_session()

        return list(self.temp_id_to_instance_map.values())
