"""Scene-classification task package with dependency-light lazy exports."""

from importlib import import_module


_EXPORT_MODULES = {
    "SceneClassificationConfig": ".config",
    "SceneDataModule": ".data",
    "SceneClassificationEvaluator": ".engine",
    "SceneClassificationPredictor": ".inference",
    "SceneClassificationTrainer": ".engine",
    "SceneClassificationMetrics": ".metrics",
    "SceneClassificationNetwork": ".model",
    "VRSBenchSceneDatasetBuilder": ".prepare",
}

__all__ = tuple(_EXPORT_MODULES)


def __getattr__(name):
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value
