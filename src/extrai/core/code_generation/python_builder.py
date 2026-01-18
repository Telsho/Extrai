import keyword
from typing import Any, Dict, Set, List


class ImportManager:
    """Manages imports for the generated code, handling consolidation."""

    def __init__(self):
        self.typing_imports: Set[str] = set()
        self.sqlmodel_imports: Set[str] = {"SQLModel"}
        self.module_imports: Set[str] = set()
        self.custom_imports: Set[str] = set()

    def add_import_for_type(self, type_str: str):
        if "datetime." in type_str:
            self.module_imports.add("datetime")
        if "uuid." in type_str:
            self.module_imports.add("uuid")
        if "Optional[" in type_str:
            self.typing_imports.add("Optional")
        if "List[" in type_str:
            self.typing_imports.add("List")
        if "Dict[" in type_str:
            self.typing_imports.add("Dict")
        if "Union[" in type_str:
            self.typing_imports.add("Union")
        if "Any" in type_str:
            self.typing_imports.add("Any")

    def add_custom_imports(self, imports: List[str]):
        for imp in imports:
            self.custom_imports.add(imp.strip())

    def render(self) -> str:
        import_lines = []

        # Consolidate custom imports with auto-detected ones
        for custom_imp in self.custom_imports:
            if custom_imp.startswith("from sqlmodel"):
                try:
                    items = {
                        item.strip()
                        for item in custom_imp.split(" import ")[1].split(",")
                    }
                    self.sqlmodel_imports.update(items)
                except IndexError:
                    import_lines.append(custom_imp)
            elif custom_imp.startswith("from typing"):
                try:
                    items = {
                        item.strip()
                        for item in custom_imp.split(" import ")[1].split(",")
                    }
                    self.typing_imports.update(items)
                except IndexError:
                    import_lines.append(custom_imp)
            elif custom_imp.startswith("import "):
                modules = {
                    mod.strip()
                    for mod in custom_imp.replace("import ", "").split(",")
                    if mod.strip()
                }
                if modules:
                    self.module_imports.update(modules)
                else:
                    import_lines.append(custom_imp)
            else:
                import_lines.append(custom_imp)  # Add other complex imports as is

        if self.sqlmodel_imports:
            import_lines.append(
                f"from sqlmodel import {', '.join(sorted(list(self.sqlmodel_imports)))}"
            )
        if self.typing_imports:
            import_lines.append(
                f"from typing import {', '.join(sorted(list(self.typing_imports)))}"
            )
        if self.module_imports:
            for mod in sorted(self.module_imports):
                import_lines.append(f"import {mod}")

        return "\n".join(sorted(set(import_lines)))


class FieldGenerator:
    """Generates the code for a single field in a SQLModel."""

    def __init__(self, field_info: Dict[str, Any], import_manager: ImportManager):
        self.field_info = field_info
        self.imports = import_manager
        self.field_name_original = self.field_info["name"]
        self.field_name_python = self.field_name_original
        self.args_map: Dict[str, str] = {}

    def _handle_keyword_name(self):
        if keyword.iskeyword(self.field_name_original):
            self.field_name_python = self.field_name_original + "_"
            self.args_map["alias"] = f'"{self.field_name_original}"'

    def _get_default_value_arg(self):
        if "default_factory" in self.field_info:
            factory_str = self.field_info["default_factory"]
            self.args_map["default_factory"] = factory_str
            if "." in factory_str:
                potential_module = factory_str.split(".")[0]
                if potential_module.isidentifier() and potential_module not in [
                    "list",
                    "dict",
                    "set",
                    "tuple",
                ]:
                    self.imports.module_imports.add(potential_module)
        elif "default" in self.field_info:
            default_val = self.field_info["default"]
            if isinstance(default_val, str):
                self.args_map["default"] = f'"{default_val}"'
            elif isinstance(default_val, bool):
                self.args_map["default"] = str(default_val)
            elif default_val is None:
                self.args_map["default"] = "None"
            else:
                self.args_map["default"] = str(default_val)

    def _get_nullable_arg(self):
        field_type_str = self.field_info["type"]
        is_optional_type = field_type_str.startswith("Optional[")
        is_pk = self.field_info.get("primary_key", False)
        explicit_nullable = self.field_info.get("nullable")

        if explicit_nullable is True:
            self.args_map["nullable"] = "True"
        elif explicit_nullable is False:
            if not (is_pk and not is_optional_type) or is_optional_type:
                self.args_map["nullable"] = "False"
        elif is_optional_type:
            self.args_map["nullable"] = "True"

        if is_pk and is_optional_type and self.args_map.get("nullable") != "False":
            self.args_map["nullable"] = "True"

    def _get_sa_column_args(self):
        sa_column_kwargs = self.field_info.get("sa_column_kwargs", {})
        for k, v in sa_column_kwargs.items():
            if k in ["server_default", "onupdate"]:
                self.args_map[k] = f'"{v}"' if isinstance(v, str) else str(v)
            elif k == "sa_type":
                self.args_map["sa_type"] = str(v)
                if str(v) == "JSON":
                    self.imports.sqlmodel_imports.add("JSON")
                if "sqlalchemy." in str(v):
                    self.imports.module_imports.add("sqlalchemy")

    def _get_common_args(self):
        if "description" in self.field_info:
            desc = self.field_info["description"]
            escaped_desc = (
                desc.replace("\\", "\\\\")
                .replace('"', '\\"')
                .replace("\n", "\\n")
                .replace("\r", "\\r")
            )
            self.args_map["description"] = f'"{escaped_desc}"'
        if "foreign_key" in self.field_info:
            self.args_map["foreign_key"] = f'"{self.field_info["foreign_key"]}"'
        if self.field_info.get("index"):
            self.args_map["index"] = "True"
        if self.field_info.get("primary_key"):
            self.args_map["primary_key"] = "True"
        if self.field_info.get("unique"):
            self.args_map["unique"] = "True"

    def _determine_field_arguments(self):
        self._handle_keyword_name()
        self._get_default_value_arg()
        self._get_nullable_arg()
        self._get_sa_column_args()
        self._get_common_args()

    def generate_code(self) -> str:
        field_type_str = self.field_info["type"]
        self.imports.add_import_for_type(field_type_str)

        if "field_options_str" in self.field_info:
            field_options = self.field_info["field_options_str"]
            if "JSON" in field_options:
                self.imports.sqlmodel_imports.add("JSON")
            if "Relationship" in field_options:
                self.imports.sqlmodel_imports.add("Relationship")

            if keyword.iskeyword(self.field_name_original):
                self.field_name_python = self.field_name_original + "_"
            return f"    {self.field_name_python}: {field_type_str} = {field_options}"

        self._determine_field_arguments()

        if not self.args_map:
            return f"    {self.field_name_python}: {field_type_str}"

        self.imports.sqlmodel_imports.add("Field")
        ordered_keys = [
            "primary_key",
            "alias",
            "default",
            "default_factory",
            "unique",
            "index",
            "foreign_key",
            "nullable",
            "sa_type",
            "description",
            "server_default",
            "onupdate",
        ]
        final_args_list = [
            f"{key}={self.args_map[key]}"
            for key in ordered_keys
            if key in self.args_map
        ]
        field_args_str = ", ".join(final_args_list)
        return (
            f"    {self.field_name_python}: {field_type_str} = Field({field_args_str})"
        )


class ClassCodeBuilder:
    """Builds the final Python code string from its components."""

    def __init__(
        self,
        model_name: str,
        import_manager: ImportManager,
        description: str,
        table_name: str,
        base_classes: List[str],
        is_table_model: bool,
    ):
        self.model_name = model_name
        self.import_manager = import_manager
        self.description = description
        self.table_name = table_name
        self.base_classes = base_classes
        self.is_table_model = is_table_model
        self.fields_code: List[str] = []

    def add_field(self, field_code: str):
        self.fields_code.append(field_code)

    def render_class_definition(self) -> str:
        fields_str = "\n".join(self.fields_code) if self.fields_code else "    pass"

        base_classes_str = ", ".join(self.base_classes)
        class_decorator_args = []
        if self.is_table_model:
            class_decorator_args.append("table=True")

        class_header = f"class {self.model_name}({base_classes_str}"
        if "SQLModel" in base_classes_str and class_decorator_args:
            class_header += f", {', '.join(class_decorator_args)}"
        class_header += "):"

        docstring_section = ""
        if self.description:
            docstring_section = f'    """{self.description}"""\n'

        table_name_section = ""
        if self.is_table_model:
            table_name_section = f'    __tablename__ = "{self.table_name}"\n\n'
        elif self.description:
            table_name_section = "\n"

        return f"{class_header}\n{docstring_section}{table_name_section}{fields_str}\n"


class PythonModelBuilder:
    """Facade for generating Python code for SQLModels from description dictionaries."""

    def generate_model_code(self, model_descriptions: List[Dict[str, Any]]) -> str:
        """
        Generates Python code for the provided model descriptions.

        Args:
            model_descriptions: A list of dictionaries, where each dictionary describes a SQLModel.

        Returns:
            A string containing the complete Python code with imports and class definitions.
        """
        import_manager = ImportManager()
        class_definitions = []

        # First pass to gather all imports from all model definitions
        for model_desc in model_descriptions:
            import_manager.add_custom_imports(model_desc.get("imports", []))
            base_classes = model_desc.get("base_classes_str", ["SQLModel"])
            if "SQLModel" in base_classes:
                import_manager.sqlmodel_imports.add("SQLModel")
            if "fields" in model_desc and model_desc["fields"]:
                for f_info in model_desc["fields"]:
                    # This populates the import manager with types from fields
                    _ = FieldGenerator(f_info, import_manager).generate_code()

        # Now, build each class definition
        for model_desc in model_descriptions:
            model_name = model_desc["model_name"]
            base_classes = model_desc.get("base_classes_str", ["SQLModel"])

            builder = ClassCodeBuilder(
                model_name=model_name,
                import_manager=import_manager,
                description=model_desc.get("description", ""),
                table_name=model_desc.get("table_name", f"{model_name.lower()}s"),
                base_classes=base_classes,
                is_table_model=model_desc.get("is_table_model", True),
            )

            if "fields" in model_desc and model_desc["fields"]:
                for f_info in model_desc["fields"]:
                    field_generator = FieldGenerator(f_info, import_manager)
                    builder.add_field(field_generator.generate_code())

            class_definitions.append(builder.render_class_definition())

        imports_str = import_manager.render()
        full_code = f"{imports_str}\n\n\n" + "\n\n".join(class_definitions)
        return full_code
