# extrai/core/model_registry.py

import json
import logging

from sqlmodel import SQLModel

from .errors import ConfigurationError
from .schema_inspector import SchemaInspector


class ModelRegistry:
    """
    Manages SQLModel schemas and their JSON representations.

    Responsibilities:
    - Discover all models from root
    - Generate JSON schemas for LLM
    - Provide model lookup by name
    - Cache schemas for performance
    """

    def __init__(
        self, root_model: type[SQLModel], logger: logging.Logger | None = None
    ):
        """
        Initialize the model registry.

        Args:
            root_model: The root SQLModel class to discover from
            logger: Optional logger instance

        Raises:
            ConfigurationError: If model discovery or schema generation fails
        """
        self.logger = logger or logging.getLogger(__name__)
        self.root_model = root_model
        self.inspector = SchemaInspector(self.logger)

        # Validate root model
        try:
            if not root_model or not issubclass(root_model, SQLModel):
                raise ConfigurationError("root_model must be a valid SQLModel class")
        except TypeError:
            raise ConfigurationError("root_model must be a valid SQLModel class")

        # Discover and validate models
        self.models = self._discover_models(root_model)
        self.model_map = {m.__name__: m for m in self.models}

        # Generate schemas
        self.llm_schema_json = self._generate_llm_schema()

        self.logger.info(
            f"ModelRegistry initialized with {len(self.models)} models: "
            f"{', '.join(self.model_map.keys())}"
        )

    def _discover_models(self, root_model: type[SQLModel]) -> list[type[SQLModel]]:
        """
        Discovers all SQLModel classes from root.

        Args:
            root_model: The root model to start discovery from

        Returns:
            List of discovered SQLModel classes

        Raises:
            ConfigurationError: If discovery fails or no models found
        """
        try:
            models = self.inspector.discover_sqlmodels_from_root(root_model)
            if not models:
                raise ConfigurationError(
                    f"No SQLModel classes discovered from root model {root_model.__name__}"
                )
            return models
        except Exception as e:
            raise ConfigurationError(
                f"Failed to discover SQLModel classes from {root_model.__name__}: {e}"
            ) from e

    def _generate_llm_schema(self) -> str:
        """
        Generates JSON schema for LLM prompts.

        Returns:
            JSON string representation of the schema

        Raises:
            ConfigurationError: If schema generation fails or produces invalid JSON
        """
        try:
            schema = self.inspector.generate_llm_schema_from_models(self.models)
            if not schema:
                raise ConfigurationError("Generated LLM schema is empty")

            # Validate it's valid JSON
            json.loads(schema)
            return schema
        except json.JSONDecodeError as e:
            raise ConfigurationError(
                f"Generated LLM schema is invalid JSON: {e}"
            ) from e
        except Exception as e:
            raise ConfigurationError(f"Failed to generate LLM schema: {e}") from e

    def get_schema_for_models(self, model_names: list[str]) -> str:
        """
        Generates schema JSON for specific models.

        Args:
            model_names: List of model class names to include in schema

        Returns:
            JSON string representation of the schema for specified models

        Note:
            If no valid models are found in model_names, returns the full schema
        """
        models = [
            self.model_map[name] for name in model_names if name in self.model_map
        ]

        if not models:
            self.logger.warning(
                f"No valid models found in {model_names}, using full schema"
            )
            return self.llm_schema_json

        try:
            return self.inspector.generate_llm_schema_from_models(models)
        except Exception as e:
            self.logger.error(f"Failed to generate schema for {model_names}: {e}")
            return self.llm_schema_json

    def get_model_by_name(self, name: str) -> type[SQLModel] | None:
        """
        Retrieves a model class by name.

        Args:
            name: The name of the model class

        Returns:
            The SQLModel class if found, None otherwise
        """
        return self.model_map.get(name)

    def get_all_model_names(self) -> list[str]:
        """
        Returns list of all discovered model names.

        Returns:
            List of model class names
        """
        return list(self.model_map.keys())

    def has_model(self, name: str) -> bool:
        """
        Checks if a model with the given name exists.

        Args:
            name: The name of the model class

        Returns:
            True if model exists, False otherwise
        """
        return name in self.model_map

    def __repr__(self) -> str:
        """String representation of the registry."""
        return (
            f"ModelRegistry(root={self.root_model.__name__}, models={len(self.models)})"
        )
