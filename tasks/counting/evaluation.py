"""Counting metrics, reports, and visualization helpers."""

from collections import defaultdict
from pathlib import Path
from typing import Any, DefaultDict, Dict, Iterable, List
import json

import cv2
import numpy as np


class CountingEvaluator:
    """Accumulate exact-match, MAE, and RMSE counting metrics."""

    def __init__(self) -> None:
        self.records: List[Dict[str, Any]] = []

    def update(
        self,
        pred_count: int,
        gt_count: int,
        class_name: str = "",
        image_id: str = "",
    ) -> None:
        self.records.append(
            {
                "image_id": str(image_id),
                "class": str(class_name),
                "gt": int(gt_count),
                "pred": int(pred_count),
                "diff": abs(int(pred_count) - int(gt_count)),
            }
        )

    @staticmethod
    def _metrics(predictions: Iterable[int], ground_truths: Iterable[int]) -> Dict[str, float | int]:
        preds = np.asarray(list(predictions), dtype=np.float64)
        gts = np.asarray(list(ground_truths), dtype=np.float64)
        if gts.size == 0:
            return {"total_samples": 0, "exact_match": 0.0, "mae": 0.0, "rmse": 0.0}
        return {
            "total_samples": int(gts.size),
            "exact_match": float(np.mean(preds == gts) * 100.0),
            "mae": float(np.mean(np.abs(preds - gts))),
            "rmse": float(np.sqrt(np.mean((preds - gts) ** 2))),
        }

    def compute_metrics(self) -> Dict[str, float | int]:
        return self._metrics(
            (record["pred"] for record in self.records),
            (record["gt"] for record in self.records),
        )

    def compute_per_class(self) -> Dict[str, Dict[str, float | int]]:
        grouped: DefaultDict[str, List[Dict[str, Any]]] = defaultdict(list)
        for record in self.records:
            grouped[str(record["class"])].append(record)
        return {
            class_name: self._metrics(
                (record["pred"] for record in records),
                (record["gt"] for record in records),
            )
            for class_name, records in sorted(grouped.items())
        }

    def build_report(self, dataset: str, split: str, weights: str, extra: Dict[str, Any] | None = None) -> Dict[str, Any]:
        report = {
            "task": "text_guided_object_counting",
            "protocol": "positive_voc_class_queries",
            "dataset": str(dataset),
            "split": str(split),
            "weights": str(weights),
            "overall": self.compute_metrics(),
            "per_class": self.compute_per_class(),
            "samples": list(self.records),
        }
        if extra:
            report.update(extra)
        return report

    def save_report(
        self,
        save_dir: str | Path,
        dataset: str,
        split: str,
        weights: str,
        extra: Dict[str, Any] | None = None,
    ) -> Dict[str, Path]:
        output_dir = Path(save_dir).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        report = self.build_report(dataset=dataset, split=split, weights=weights, extra=extra)
        stem = f"{dataset}_{Path(split).stem}_counting_report"
        json_path = output_dir / f"{stem}.json"
        text_path = output_dir / f"{stem}.txt"
        with open(json_path, "w", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2)

        overall = report["overall"]
        with open(text_path, "w", encoding="utf-8") as handle:
            handle.write(f"=== {dataset} Object Counting Evaluation Report ===\n")
            handle.write(f"Weights: {weights}\n")
            handle.write(f"Split: {split}\n")
            handle.write(f"Total Evaluated Pairs: {overall['total_samples']}\n\n")
            handle.write(f"Exact Match (EM %): {overall['exact_match']:.2f}%\n")
            handle.write(f"MAE: {overall['mae']:.4f}\n")
            handle.write(f"RMSE: {overall['rmse']:.4f}\n\n")
            handle.write("[Per-Class Breakdown]\n")
            for class_name, metrics in report["per_class"].items():
                handle.write(
                    f"{class_name}: samples={metrics['total_samples']}, "
                    f"EM={metrics['exact_match']:.2f}%, MAE={metrics['mae']:.4f}, "
                    f"RMSE={metrics['rmse']:.4f}\n"
                )
        return {"json": json_path, "text": text_path}


class CountingVisualizer:
    """Render the original box-count comparison visualization."""

    @staticmethod
    def render(
        image: np.ndarray,
        boxes: np.ndarray,
        image_id: str,
        class_name: str,
        gt_count: int,
        pred_count: int,
    ) -> np.ndarray:
        canvas = image.copy()
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 45), (20, 20, 20), -1)
        status = f"ID: {image_id} | Class: {class_name} | GT: {gt_count} | Pred: {pred_count}"
        cv2.putText(canvas, status, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        difference = abs(int(pred_count) - int(gt_count))
        label = "EXACT MATCH" if difference == 0 else f"DIFF: {difference}"
        color = (0, 255, 0) if difference == 0 else (0, 0, 255)
        cv2.putText(canvas, label, (max(15, canvas.shape[1] - 220), 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        return canvas

    @staticmethod
    def save(image: np.ndarray, output_path: str | Path) -> Path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(path), image):
            raise OSError(f"Failed to write visualization: {path}")
        return path
