from dataclasses import dataclass, field, replace


@dataclass
class BatchJobConfig:
    extraction_example_json: str = ""
    custom_extraction_process: str | list[str] = ""
    custom_extraction_guidelines: str | list[str] = ""
    custom_final_checklist: str | list[str] = ""
    custom_context: str | list[str] = ""
    count_entities: bool = False
    custom_counting_context: str | list[str] = ""
    schema_json: str = ""
    # batch-specific runtime state
    hierarchical: bool = False
    current_model_index: int = 0
    expected_entity_descriptions: list[dict] | None = None
    partial_results: list[dict] = field(default_factory=list)

    def evolve(self, **changes) -> "BatchJobConfig":
        return replace(self, **changes)
