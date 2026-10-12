"""JSON-serializable configuration models for user-facing inference tasks."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class InputFieldConfig:
    """Describe one field that a task asks the user to provide."""

    key: str
    label: str
    kind: str
    required: bool = True
    placeholder: str = ""
    help_text: str = ""
    accept: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class OutputFieldConfig:
    """Describe one task-owned result that the interface can render."""

    key: str
    label: str
    kind: str
    downloadable: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TaskConfig:
    """Stable task metadata shared by the router and the web interface."""

    task_id: str
    title: str
    technical_name: str
    description: str
    detail: str
    icon: str
    accent_color: str
    action_label: str
    result_title: str
    checkpoint: str
    adapter_path: str
    inputs: tuple[InputFieldConfig, ...]
    outputs: tuple[OutputFieldConfig, ...]
    runtime_defaults: dict[str, Any] = field(default_factory=dict)
    interface_status: str = "pending"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
