"""Task catalog and model-agnostic inference routing for the HFSA web shell."""

from .config import InputFieldConfig, OutputFieldConfig, TaskConfig
from .registry import get_task_config, list_task_configs
from .router import (
    TaskInputError,
    TaskInterfaceUnavailableError,
    TaskRouter,
    UnknownTaskError,
)

__all__ = (
    "InputFieldConfig",
    "OutputFieldConfig",
    "TaskConfig",
    "TaskInputError",
    "TaskInterfaceUnavailableError",
    "TaskRouter",
    "UnknownTaskError",
    "get_task_config",
    "list_task_configs",
)
