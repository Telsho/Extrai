import logging
import uuid
from typing import (
    List,
    Dict,
    Any,
    Optional,
    Type,
    get_origin,
    get_args,
    Union,
    NamedTuple,
)
from sqlalchemy.orm import Session, RelationshipProperty
from sqlalchemy import create_engine, inspect
from sqlalchemy.exc import SQLAlchemyError
from sqlmodel import SQLModel

from .model_registry import ModelRegistry
from .errors import HydrationError, WorkflowError

SQLModelInstance = SQLModel


class DatabaseWriterError(Exception):
    """Custom exception for database writer errors."""

    pass


class PrimaryKeyInfo(NamedTuple):
    name: Optional[str]
    type: Optional[Type[Any]]
    has_uuid_factory: bool


class DirectHydrator:
    """
    Hydrates SQLModel objects directly from structured nested dictionaries.
    Used when the LLM output is guaranteed to match the model structure (e.g. Structured Output).
    Does not require _temp_id or _type fields.
    Supports recursive hydration of nested relationships.
    """

    def __init__(
        self,
        session: Session,
        logger: Optional[logging.Logger] = None,
        original_pk_map: Dict[tuple[str, Any], SQLModelInstance] = None,
        all_instances: List[SQLModelInstance] = None,
    ):
        self.session = session
        self.logger = logger or logging.getLogger(__name__)
        self.original_pk_map = original_pk_map if original_pk_map is not None else {}
        self.all_instances = all_instances if all_instances is not None else []

    def hydrate(
        self,
        data: List[Dict[str, Any]],
        model_map: Dict[str, Type[SQLModel]],
        default_model_class: Optional[Type[SQLModel]] = None,
    ) -> List[SQLModelInstance]:
        instances = []
        for item in data:
            try:
                # Determine model class
                _type = item.get("_type")
                model_class = None

                if _type and _type in model_map:
                    model_class = model_map[_type]
                elif default_model_class:
                    model_class = default_model_class

                if not model_class:
                    raise ValueError(
                        f"Could not determine model class for item (missing _type and no default): {item}"
                    )

                instance = self._hydrate_recursive(item, model_class, model_map)
                self.session.add(instance)
                instances.append(instance)
            except Exception as e:
                self.logger.error(
                    f"Failed to hydrate item directly: {e}", exc_info=True
                )
                raise ValueError(f"Direct hydration failed: {e}") from e
        return instances

    def _hydrate_recursive(
        self,
        data: Dict[str, Any],
        model_class: Type[SQLModel],
        model_map: Dict[str, Type[SQLModel]],
    ) -> SQLModelInstance:
        """
        Recursively hydrates an instance and its relationships.
        """
        # 1. Identify relationship fields
        mapper = inspect(model_class)
        relationships = {r.key: r for r in mapper.relationships}

        # 2. Separate scalar data from relationship data
        scalar_data = {}
        relation_data = {}

        for k, v in data.items():
            if k in relationships:
                relation_data[k] = v
            else:
                scalar_data[k] = v

        pk_field_name = None
        for field_name, model_field in model_class.model_fields.items():
            if getattr(model_field, "primary_key", False):
                pk_field_name = field_name
                break

        # Capture Original PK
        if pk_field_name and pk_field_name in scalar_data:
            original_pk = scalar_data[pk_field_name]
            if original_pk is not None:
                # We store it temporarily, will map to instance after creation
                # Note: We need the type name. Assuming _type is in data or model_class.__name__
                type_name = data.get("_type", model_class.__name__)
                self.original_pk_map[(type_name, original_pk)] = None

            del scalar_data[pk_field_name]

        if "_type" in scalar_data:
            del scalar_data["_type"]

        instance = model_class.model_validate(scalar_data)

        # Map original PK to instance
        if pk_field_name and pk_field_name in data:  # check original data
            original_pk = data[pk_field_name]
            if original_pk is not None:
                type_name = data.get("_type", model_class.__name__)
                self.original_pk_map[(type_name, original_pk)] = instance

        self.all_instances.append(instance)

        # 4. Populate relationships
        for rel_key, rel_value in relation_data.items():
            if rel_value is None:
                setattr(instance, rel_key, None)
                continue

            rel_prop = relationships[rel_key]
            target_class = rel_prop.mapper.class_

            if isinstance(rel_value, list):
                # One-to-Many / Many-to-Many
                related_instances = []
                for child_data in rel_value:
                    if isinstance(child_data, dict):
                        # Handle polymorphism in child
                        child_class = target_class
                        if "_type" in child_data and child_data["_type"] in model_map:
                            child_class = model_map[child_data["_type"]]

                        child_instance = self._hydrate_recursive(
                            child_data, child_class, model_map
                        )
                        related_instances.append(child_instance)
                setattr(instance, rel_key, related_instances)

            elif isinstance(rel_value, dict):
                # Many-to-One / One-to-One
                child_class = target_class
                if "_type" in rel_value and rel_value["_type"] in model_map:
                    child_class = model_map[rel_value["_type"]]

                child_instance = self._hydrate_recursive(
                    rel_value, child_class, model_map
                )
                setattr(instance, rel_key, child_instance)

        return instance


class SQLAlchemyHydrator:
    """
    Hydrates SQLModel objects from consensus JSON data.
    It uses a two-pass strategy: first, create all object instances,
    then link their relationships using temporary IDs.
    """

    def __init__(
        self,
        session: Session,
        logger: Optional[logging.Logger] = None,
        original_pk_map: Dict[tuple[str, Any], SQLModelInstance] = None,
        all_instances: List[SQLModelInstance] = None,
    ):
        """
        Initializes the Hydrator.

        Args:
            session: The SQLAlchemy session to use for database operations
                     and instance management (e.g., adding instances).
            logger: Optional logger instance.
        """
        self.session: Session = session
        self.temp_id_to_instance_map: Dict[
            str, SQLModelInstance
        ] = {}  # Stores _temp_id -> SQLModel instance
        self.original_pk_map = original_pk_map if original_pk_map is not None else {}
        self.all_instances = all_instances if all_instances is not None else []
        self.logger = logger or logging.getLogger(__name__)

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
            # Store the original PK value for later foreign key resolution
            original_pk = filtered_data[pk_field_name]
            if original_pk is not None:
                self.original_pk_map[(_type, original_pk)] = (
                    None  # Will be set to instance later
                )
            del filtered_data[pk_field_name]

        try:
            instance = model_class.model_validate(filtered_data)
        except Exception as e:
            raise ValueError(
                f"Failed to instantiate/validate SQLModel '{_type}' for _temp_id '{_temp_id}': {e}"
            ) from e

        # Update the original_pk_map with the actual instance
        if pk_field_name and pk_field_name in entity_data:
            original_pk = entity_data[pk_field_name]
            if original_pk is not None:
                self.original_pk_map[(_type, original_pk)] = instance

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
            self.logger.warning(
                f"Referenced _temp_id '{ref_id}' for relation "
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
                self.logger.warning(
                    f"Value for '{relation_name}_ref_ids' on instance '{_temp_id}' is not a list as expected for '_ref_ids'. Value: {ref_ids}"
                )
            setattr(instance, relation_name, [])
            return

        related_instances = []
        for ref_id in ref_ids:
            if isinstance(ref_id, str) and ref_id in self.temp_id_to_instance_map:
                related_instances.append(self.temp_id_to_instance_map[ref_id])
            else:
                self.logger.warning(
                    f"Referenced _temp_id '{ref_id}' in list for relation "
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

        self._add_instances_to_session()

        return list(self.temp_id_to_instance_map.values())


def persist_objects(
    db_session: Session, objects_to_persist: List[Any], logger: logging.Logger
) -> None:
    """
    Persists a list of SQLAlchemy objects to the database using the provided session.

    Args:
        db_session: The SQLAlchemy session to use for database operations.
        objects_to_persist: A list of SQLAlchemy model instances to be saved.

    Raises:
        DatabaseWriterError: If an error occurs during the database commit.
    """
    if not objects_to_persist:
        logger.info("No objects provided to persist.")
        return

    try:
        # All objects should already be associated with the session
        # from the hydration phase
        db_session.add_all(objects_to_persist)
        db_session.commit()
        logger.info(
            f"Successfully persisted {len(objects_to_persist)} objects to the database."
        )
    except SQLAlchemyError as e:
        logger.error(f"Database commit failed: {e}", exc_info=True)
        try:
            db_session.rollback()
            logger.info("Database session rolled back successfully.")
        except SQLAlchemyError as rollback_e:
            logger.error(
                f"Failed to rollback database session: {rollback_e}", exc_info=True
            )
            # Potentially raise a more critical error or handle nested failure
        raise DatabaseWriterError(f"Failed to persist objects due to: {e}")
    except Exception as e:
        logger.error(
            f"An unexpected error occurred during object persistence: {e}",
            exc_info=True,
        )

        if db_session.is_active:
            db_session.rollback()
            logger.info("Database session rolled back due to unexpected error.")

        raise DatabaseWriterError(f"An unexpected error occurred: {e}")


class ResultProcessor:
    """Handles hydration and persistence of extraction results."""

    def __init__(
        self,
        model_registry: ModelRegistry,
        analytics_collector,
        logger: logging.Logger,
    ):
        self.model_registry = model_registry
        self.analytics_collector = analytics_collector
        self.logger = logger
        self.original_pk_map: Dict[tuple[str, Any], SQLModelInstance] = {}
        self.all_hydrated_instances: List[SQLModelInstance] = []

    def hydrate(
        self,
        results: List[Dict[str, Any]],
        db_session: Optional[Session] = None,
        default_model_type: Optional[str] = None,
    ) -> List[Any]:
        """
        Hydrates dictionaries into SQLModel objects.

        Args:
            results: List of dictionaries to hydrate.
            db_session: Optional SQLAlchemy session.
            default_model_type: Optional override for the default model type.
                              If provided, it guides the DirectHydrator fallback.
        """
        if not results:
            return []

        session = self._get_or_create_session(db_session)

        try:
            self.logger.info(f"Hydrating {len(results)} objects...")

            # Determine Strategy based on data content
            first_item = results[0]
            use_direct_hydration = False

            # If _temp_id is missing, we must use DirectHydrator (Graph Reconstruction requires _temp_id)
            if "_temp_id" not in first_item:
                use_direct_hydration = True

            # If default_model_type is explicitly provided, we assume DirectHydrator
            if default_model_type:
                use_direct_hydration = True

            if use_direct_hydration:
                self.logger.info(
                    f"Using DirectHydrator (default_model_type={default_model_type or 'Auto-detect'})"
                )

                default_model_class = None
                if default_model_type:
                    default_model_class = self.model_registry.model_map.get(
                        default_model_type
                    )
                if not default_model_class:
                    default_model_class = self.model_registry.root_model

                hydrator = DirectHydrator(
                    session,
                    self.logger,
                    self.original_pk_map,
                    self.all_hydrated_instances,
                )
                hydrated = hydrator.hydrate(
                    results,
                    model_map=self.model_registry.model_map,
                    default_model_class=default_model_class,
                )
            else:
                self.logger.info("Using SQLAlchemyHydrator for graph reconstruction")
                hydrator = SQLAlchemyHydrator(
                    session=session,
                    logger=self.logger,
                    original_pk_map=self.original_pk_map,
                    all_instances=self.all_hydrated_instances,
                )
                hydrated = hydrator.hydrate(results, self.model_registry.model_map)
                self.all_hydrated_instances.extend(hydrated)

            self.analytics_collector.record_hydration_success(len(hydrated))
            self.logger.info(f"Successfully hydrated {len(hydrated)} objects")

            return hydrated

        except Exception as e:
            self.analytics_collector.record_hydration_failure()
            raise HydrationError(f"Hydration failed: {e}") from e

        finally:
            if db_session is None and session:
                session.close()

    def persist(self, objects: List[Any], db_session: Session):
        """Persists objects to database."""
        if not objects:
            self.logger.info("No objects to persist")
            return

        self._link_foreign_keys(objects)

        try:
            persist_objects(
                db_session=db_session,
                objects_to_persist=objects,
                logger=self.logger,
            )
        except DatabaseWriterError:
            db_session.rollback()
            raise
        except Exception as e:
            db_session.rollback()
            raise WorkflowError(f"Persistence failed: {e}") from e

    def _link_foreign_keys(
        self, instances: Optional[List[SQLModelInstance]] = None
    ) -> None:
        """
        Links foreign keys for all hydrated instances before persisting.
        """
        target_instances = (
            instances if instances is not None else self.all_hydrated_instances
        )
        if self.original_pk_map:
            self._perform_fk_recovery(target_instances, self.original_pk_map)

    def _perform_fk_recovery(
        self,
        instances: List[SQLModelInstance],
        original_pk_map: Dict[tuple[str, Any], SQLModelInstance],
    ) -> None:
        """
        Scans all hydrated instances for Foreign Key fields that are set (not None)
        but might refer to an original ID that was stripped.
        Attempts to link these to the correct instance using original_pk_map.
        """
        count_recovered = 0
        for instance in instances:
            model_class = type(instance)
            mapper = inspect(model_class)

            for rel in mapper.relationships:
                # We only care about Many-to-One (FK holder)
                if rel.direction.name != "MANYTOONE":
                    continue

                if not rel.local_remote_pairs:
                    continue

                local_col, remote_col = rel.local_remote_pairs[0]

                # Check if FK field has a value on the instance
                fk_value = getattr(instance, local_col.name, None)
                if fk_value is None:
                    continue

                # Check if relationship is already set
                current_rel_value = getattr(instance, rel.key, None)
                if current_rel_value is not None:
                    continue

                # Try to find target instance in map
                target_class = rel.mapper.class_
                target_type = target_class.__name__

                key = (target_type, fk_value)
                if key in original_pk_map:
                    target_instance = original_pk_map[key]
                    setattr(instance, rel.key, target_instance)
                    count_recovered += 1
                    self.logger.debug(
                        f"Recovered relationship {model_class.__name__}.{rel.key} "
                        f"using FK {fk_value} -> {target_type}"
                    )

        if count_recovered > 0:
            self.logger.info(
                f"Universal FK Recovery: Restored {count_recovered} relationships."
            )

    def _get_or_create_session(self, db_session: Optional[Session]) -> Session:
        """Creates temporary in-memory session if none provided."""
        if db_session:
            return db_session

        engine = create_engine("sqlite:///:memory:")
        SQLModel.metadata.create_all(engine)
        return Session(engine)
