"""Model-agnostic router with lazy task-adapter registration."""

from __future__ import annotations

from collections.abc import Callable, Mapping
import threading
from typing import Any, Protocol

from .config import TaskConfig
from .registry import get_task_config, list_task_configs


class UnknownTaskError(KeyError):
    """Raised when a request names a task that is not registered."""


class TaskInputError(ValueError):
    """Raised when a task request does not satisfy its declared input schema."""


class TaskInterfaceUnavailableError(RuntimeError):
    """Raised while a configured task still has no runtime adapter."""


class TaskAdapter(Protocol):
    def predict(self, **inputs: Any) -> Any:
        """Run one task-specific prediction."""


AdapterFactory = Callable[[TaskConfig], TaskAdapter]


class TaskRouter:
    """Expose task metadata now and accept concrete adapters incrementally."""

    def __init__(self) -> None:
        self._factories: dict[str, AdapterFactory] = {}
        self._instances: dict[str, TaskAdapter] = {}
        self._instance_lock = threading.Lock()

    def list_tasks(self) -> list[dict[str, Any]]:
        tasks = []
        for config in list_task_configs():
            payload = config.to_dict()
            payload["interface_status"] = (
                "loaded"
                if config.task_id in self._instances
                else "ready"
                if config.task_id in self._factories
                else config.interface_status
            )
            tasks.append(payload)
        return tasks

    @staticmethod
    def describe(task_id: str) -> TaskConfig:
        try:
            return get_task_config(task_id)
        except KeyError as exc:
            raise UnknownTaskError(str(exc)) from exc

    def register_adapter(self, task_id: str, factory: AdapterFactory) -> None:
        config = self.describe(task_id)
        if not callable(factory):
            raise TypeError("Task adapter factory must be callable.")
        self.unload(config.task_id)
        self._factories[config.task_id] = factory

    @staticmethod
    def validate_inputs(config: TaskConfig, inputs: Mapping[str, Any]) -> dict[str, Any]:
        if not isinstance(inputs, Mapping):
            raise TaskInputError("Task inputs must be a mapping.")
        allowed = {field.key for field in config.inputs}
        unknown = sorted(set(inputs) - allowed)
        if unknown:
            raise TaskInputError(f"Unexpected inputs for {config.task_id}: {', '.join(unknown)}")

        normalized = dict(inputs)
        missing = []
        for field in config.inputs:
            value = normalized.get(field.key)
            if field.required and (value is None or (isinstance(value, str) and not value.strip())):
                missing.append(field.label)
        if missing:
            raise TaskInputError(f"Missing required inputs: {', '.join(missing)}")
        return normalized

    def predict(self, task_id: str, inputs: Mapping[str, Any]) -> Any:
        config = self.describe(task_id)
        normalized = self.validate_inputs(config, inputs)
        factory = self._factories.get(config.task_id)
        if factory is None:
            raise TaskInterfaceUnavailableError(
                f"{config.title}界面与路由已就绪，模型推理接口将在后续接入。"
            )
        if config.task_id not in self._instances:
            with self._instance_lock:
                if config.task_id not in self._instances:
                    self._instances[config.task_id] = factory(config)
        return self._instances[config.task_id].predict(**normalized)

    def unload(self, task_id: str) -> None:
        instance = self._instances.pop(str(task_id).strip().lower(), None)
        close = getattr(instance, "close", None)
        if callable(close):
            close()

    def close(self) -> None:
        for task_id in tuple(self._instances):
            self.unload(task_id)
