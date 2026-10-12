"""Strict single-checkpoint persistence for referring segmentation."""

from __future__ import annotations

import argparse
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch

from ultralytics.nn import SemanticSegmentationModel


class RefSegCheckpointManager:
    """Own atomic persistence of the sole RefSeg checkpoint, ``best_raw.pt``."""

    @staticmethod
    def improvement_flags(
        fitness: float,
        best_fitness: float,
        best_raw_fitness: float,
        min_delta: float,
    ) -> Tuple[bool, bool]:
        """Return min-delta and raw checkpoint improvement decisions."""
        return fitness > best_fitness + min_delta, fitness > best_raw_fitness

    @staticmethod
    def save_deployment(
        path: Path,
        model: SemanticSegmentationModel,
        epoch: int,
        args: argparse.Namespace,
        data: Dict[str, Any],
        metrics: Dict[str, float],
        retries: int = 5,
        retry_delay: float = 1.0,
    ) -> None:
        """Save the sole inference/test checkpoint without restart-only state."""
        payload = {
            "format": "hfsa_refseg_deployment_v1",
            "epoch": int(epoch),
            "model": model.state_dict(),
            "args": vars(args),
            "data": data,
            "metrics": metrics,
        }
        RefSegCheckpointManager._atomic_save_payload(
            path,
            payload,
            retries=retries,
            retry_delay=retry_delay,
        )

    @staticmethod
    def _atomic_save_payload(
        path: Path,
        payload: Dict[str, Any],
        retries: int,
        retry_delay: float,
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        attempts = max(int(retries), 1)
        last_error: Optional[BaseException] = None
        for attempt in range(1, attempts + 1):
            fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
            os.close(fd)
            temp_path = Path(temp_name)
            try:
                torch.save(payload, temp_path)
                os.replace(temp_path, path)
                return
            except (OSError, RuntimeError) as exc:
                last_error = exc
                temp_path.unlink(missing_ok=True)
                if attempt >= attempts:
                    break
                print(f"checkpoint save retry {attempt}/{attempts - 1} for {path}: {exc}")
                time.sleep(max(float(retry_delay), 0.0))
        raise RuntimeError(f"Failed to save checkpoint after {attempts} attempt(s): {path}") from last_error


checkpoint_improvement_flags = RefSegCheckpointManager.improvement_flags
save_deployment_checkpoint = RefSegCheckpointManager.save_deployment


__all__ = (
    "RefSegCheckpointManager",
    "checkpoint_improvement_flags",
    "save_deployment_checkpoint",
)
