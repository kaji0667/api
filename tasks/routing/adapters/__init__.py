"""Dependency-light lazy exports for task-specific Web adapters."""

from importlib import import_module


_EXPORT_MODULES = {
    "SceneClassificationAdapter": ".classification",
    "DetectionAdapter": ".detection",
    "CountingAdapter": ".counting",
    "RefSegAdapter": ".refseg",
}

__all__ = tuple(_EXPORT_MODULES)


def __getattr__(name):
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value
