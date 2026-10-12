"""Metrics, visualization, and validation for referring segmentation."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from ultralytics.nn import SemanticSegmentationModel

from .runtime import move_batch_to_device, predict_with_optional_text, validation_selection_score


SAMPLE_IOU_THRESHOLDS = (0.5, 0.6, 0.7, 0.8, 0.9)


def make_palette(nc: int) -> np.ndarray:
    base = np.array(
        [
            [0, 0, 0],
            [220, 40, 40],
            [235, 170, 30],
            [40, 130, 220],
            [160, 110, 70],
            [45, 170, 90],
            [150, 205, 55],
            [180, 80, 200],
            [80, 200, 200],
            [240, 120, 160],
        ],
        dtype=np.uint8,
    )
    if nc <= len(base):
        return base[:nc]
    extra = np.random.default_rng(0).integers(0, 255, size=(nc - len(base), 3), dtype=np.uint8)
    return np.concatenate([base, extra], axis=0)


def mask_to_rgb(mask: torch.Tensor, palette: np.ndarray, ignore_index: int = 255) -> np.ndarray:
    mask_np = mask.detach().cpu().numpy().astype(np.int64)
    rgb = np.zeros((*mask_np.shape, 3), dtype=np.uint8)
    valid = (mask_np >= 0) & (mask_np < len(palette))
    rgb[valid] = palette[mask_np[valid]]
    rgb[mask_np == ignore_index] = np.array([45, 45, 45], dtype=np.uint8)
    return rgb


def image_to_uint8(image: torch.Tensor) -> np.ndarray:
    img = image.detach().cpu().float().clamp(0, 1)
    return (img.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)


def save_preview(
    path: Path,
    batch: Dict[str, Any],
    preds: Optional[torch.Tensor],
    palette: np.ndarray,
    max_images: int = 4,
    threshold: float = 0.5,
) -> None:
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    images = batch["img"].detach().cpu()
    masks = batch["mask"].detach().cpu()
    pred_masks = None
    if preds is not None:
        if preds.shape[1] == 1:
            pred_masks = (preds.detach().cpu().sigmoid().squeeze(1) > float(threshold)).long()
        else:
            pred_masks = preds.detach().cpu().argmax(1)

    rows = []
    for index in range(min(max_images, images.shape[0])):
        panels = [image_to_uint8(images[index]), mask_to_rgb(masks[index], palette)]
        if pred_masks is not None:
            panels.append(mask_to_rgb(pred_masks[index], palette))
        rows.append(np.concatenate(panels, axis=1))

    grid = np.concatenate(rows, axis=0)
    cv2.imwrite(str(path), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))


def update_confusion_matrix(
    confusion: torch.Tensor,
    preds: torch.Tensor,
    targets: torch.Tensor,
    nc: int,
    ignore_index: int,
    threshold: float = 0.5,
) -> None:
    if preds.shape[1] == 1:
        pred_labels = (preds.squeeze(1).sigmoid() > float(threshold)).long().detach()
    else:
        pred_labels = preds.argmax(1).detach()
    target_labels = targets.detach()
    valid = (target_labels != ignore_index) & (target_labels >= 0) & (target_labels < nc)
    if valid.any():
        indices = target_labels[valid] * nc + pred_labels[valid]
        confusion += torch.bincount(indices, minlength=nc * nc).reshape(nc, nc).cpu()


def update_text_class_confusion(
    class_confusion: torch.Tensor,
    preds: torch.Tensor,
    targets: torch.Tensor,
    class_idx: torch.Tensor,
    ignore_index: int,
    threshold: float = 0.5,
) -> None:
    pred_labels = (preds.squeeze(1).sigmoid() > float(threshold)).long().detach()
    target_labels = targets.detach()
    valid = (target_labels != ignore_index) & (target_labels >= 0) & (target_labels <= 1)
    for index, class_index in enumerate(class_idx.detach().cpu().view(-1).tolist()):
        sample_valid = valid[index]
        if sample_valid.any():
            indices = target_labels[index][sample_valid] * 2 + pred_labels[index][sample_valid]
            class_confusion[int(class_index)] += torch.bincount(indices, minlength=4).reshape(2, 2).cpu()


def metrics_from_confusion(confusion: torch.Tensor, fbeta_beta: float = 1.0) -> Dict[str, Any]:
    matrix = confusion.float()
    diag = matrix.diag()
    total = matrix.sum().clamp_min(1)
    pixel_acc = float(diag.sum() / total)
    denom = matrix.sum(1) + matrix.sum(0) - diag
    iou = torch.where(denom > 0, diag / denom.clamp_min(1), torch.full_like(diag, float("nan")))
    valid_iou = iou[~torch.isnan(iou)]
    miou = float(valid_iou.mean()) if valid_iou.numel() else 0.0
    per_class_iou = [float(value) if not torch.isnan(value) else float("nan") for value in iou]
    target_iou = per_class_iou[1] if len(per_class_iou) > 1 else miou
    pred_pos_rate = float(matrix[:, 1].sum() / total) if matrix.shape[0] > 1 else float("nan")
    target_pos_rate = float(matrix[1, :].sum() / total) if matrix.shape[0] > 1 else float("nan")
    if matrix.shape[0] > 1:
        tp = matrix[1, 1]
        fp = matrix[0, 1]
        fn = matrix[1, 0]
        precision = float(tp / (tp + fp).clamp_min(1.0))
        recall = float(tp / (tp + fn).clamp_min(1.0))
        precision_recall_sum = precision + recall
        f1 = 2.0 * precision * recall / precision_recall_sum if precision_recall_sum > 0 else 0.0
        beta2 = float(fbeta_beta) ** 2
        fbeta_denom = beta2 * precision + recall
        fbeta = (1.0 + beta2) * precision * recall / fbeta_denom if fbeta_denom > 0 else 0.0
    else:
        precision = recall = f1 = fbeta = float("nan")
    return {
        "pixel_acc": pixel_acc,
        "miou": miou,
        "per_class_iou": per_class_iou,
        "target_iou": target_iou,
        "pred_pos_rate": pred_pos_rate,
        "target_pos_rate": target_pos_rate,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fbeta": fbeta,
    }


def sample_iou_values(
    preds: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int,
    threshold: float,
) -> torch.Tensor:
    pred_labels = (preds.squeeze(1).sigmoid() > float(threshold)).bool().detach()
    values = []
    for pred, target in zip(pred_labels, targets.detach()):
        valid = target != ignore_index
        if not valid.any():
            values.append(torch.full((), float("nan"), device=preds.device))
            continue
        pred = pred[valid]
        target = target[valid].clamp(0, 1).bool()
        intersection = (pred & target).sum().float()
        union = (pred | target).sum().float()
        values.append(torch.ones((), device=preds.device) if union <= 0 else intersection / union)
    return torch.stack(values) if values else torch.empty(0, device=preds.device)


def text_class_target_ious(class_confusion: torch.Tensor, names: List[str]) -> Dict[str, float]:
    result: Dict[str, float] = {}
    for index, matrix in enumerate(class_confusion.float()):
        if matrix.sum() <= 0:
            continue
        tp = matrix[1, 1]
        fp = matrix[0, 1]
        fn = matrix[1, 0]
        denom = tp + fp + fn
        result[names[index] if index < len(names) else str(index)] = (
            float(tp / denom.clamp_min(1.0)) if denom > 0 else float("nan")
        )
    return result


def text_class_mean_ious(
    iou_sums: torch.Tensor,
    iou_counts: torch.Tensor,
    names: List[str],
) -> Dict[str, float]:
    """Return official per-category mIoU by averaging sample IoUs within each semantic category."""
    result: Dict[str, float] = {}
    for index, (iou_sum, count) in enumerate(zip(iou_sums.double(), iou_counts.long())):
        if int(count) <= 0:
            continue
        result[names[index] if index < len(names) else str(index)] = float(iou_sum / count)
    return result


def append_results(path: Path, row: Dict[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "epoch", "train_loss", "val_loss", "pixel_acc", "oiou", "official_miou", "miou",
        "target_iou", "binary_miou", "best_threshold", "precision", "recall", "f1", "fbeta",
        "selection_score", "sample_miou", "pr_0_5", "pr_0_6", "pr_0_7", "pr_0_8", "pr_0_9",
        "pred_pos_rate", "target_pos_rate", "class_macro_miou", "class_miou", "class_oiou",
        "class_iou", "threshold_metrics", "lr", "epoch_time",
    ]
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in fieldnames})


def plot_results(results_csv: Path, save_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with results_csv.open("r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return

    epochs = [int(row["epoch"]) for row in rows]
    train_loss = [float(row["train_loss"]) for row in rows]
    val_loss = [float(row["val_loss"]) if row["val_loss"] else np.nan for row in rows]
    official_miou = [
        float(row["official_miou"])
        if row.get("official_miou")
        else float(row["sample_miou"])
        if row.get("sample_miou")
        else np.nan
        for row in rows
    ]
    oiou = [
        float(row["oiou"])
        if row.get("oiou")
        else float(row["target_iou"])
        if row.get("target_iou")
        else np.nan
        for row in rows
    ]
    pixel_acc = [float(row["pixel_acc"]) if row["pixel_acc"] else np.nan for row in rows]

    figure, axes = plt.subplots(2, 2, figsize=(10, 7), tight_layout=True)
    axes[0, 0].plot(epochs, train_loss, marker="o")
    axes[0, 0].set_title("train loss")
    axes[0, 1].plot(epochs, val_loss, marker="o")
    axes[0, 1].set_title("val loss")
    axes[1, 0].plot(epochs, oiou, marker="o", label="oIoU")
    axes[1, 0].plot(epochs, official_miou, marker="o", label="mIoU")
    axes[1, 0].set_title("official IoU metrics")
    axes[1, 0].legend()
    axes[1, 1].plot(epochs, pixel_acc, marker="o")
    axes[1, 1].set_title("pixel accuracy")
    for axis in axes.ravel():
        axis.set_xlabel("epoch")
        axis.grid(True, alpha=0.3)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(save_path, dpi=160)
    plt.close(figure)


def plot_confusion_matrix(confusion: torch.Tensor, names: List[str], save_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matrix = confusion.float().numpy()
    row_sum = matrix.sum(1, keepdims=True)
    normalized = np.divide(matrix, np.maximum(row_sum, 1), where=row_sum > 0)

    figure, axis = plt.subplots(figsize=(8, 7), tight_layout=True)
    image = axis.imshow(normalized, cmap="Blues", vmin=0, vmax=1)
    axis.set_title("normalized confusion matrix")
    axis.set_xlabel("predicted")
    axis.set_ylabel("target")
    axis.set_xticks(range(len(names)), names, rotation=45, ha="right")
    axis.set_yticks(range(len(names)), names)
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(save_path, dpi=160)
    plt.close(figure)


@torch.no_grad()
def validate(
    model: SemanticSegmentationModel,
    loader: DataLoader,
    device: torch.device,
    nc: int,
    ignore_index: int,
    max_batches: int,
    preview_path: Optional[Path],
    palette: np.ndarray,
    text_queries: bool = False,
    prompt_embeddings: Optional[torch.Tensor] = None,
    class_names_for_text: Optional[List[str]] = None,
    val_thresholds: Optional[List[float]] = None,
    val_select_metric: str = "iou",
    val_fbeta: float = 0.7,
    preview_paths: Optional[List[Path]] = None,
) -> Dict[str, Any]:
    model.eval()
    thresholds = val_thresholds or [0.5]
    confusions = {float(value): torch.zeros((nc, nc), dtype=torch.int64) for value in thresholds}
    sample_iou_sums = {float(value): 0.0 for value in thresholds}
    sample_iou_counts = {float(value): 0 for value in thresholds}
    sample_iou_hits = {
        float(value): {iou_threshold: 0 for iou_threshold in SAMPLE_IOU_THRESHOLDS}
        for value in thresholds
    }
    class_count = len(class_names_for_text or [])
    class_confusions = (
        {float(value): torch.zeros((class_count, 2, 2), dtype=torch.int64) for value in thresholds}
        if text_queries
        else None
    )
    class_sample_iou_sums = (
        {float(value): torch.zeros(class_count, dtype=torch.float64) for value in thresholds}
        if text_queries
        else None
    )
    class_sample_iou_counts = (
        {float(value): torch.zeros(class_count, dtype=torch.int64) for value in thresholds}
        if text_queries
        else None
    )
    running = 0.0
    seen = 0
    requested_preview_paths = list(preview_paths) if preview_paths is not None else ([preview_path] if preview_path else [])
    preview_items: List[Tuple[Dict[str, Any], torch.Tensor]] = []

    for batch_index, batch in enumerate(loader, start=1):
        batch = move_batch_to_device(batch, device)
        preds = predict_with_optional_text(model, batch, text_queries, prompt_embeddings)
        loss, _ = model.loss(batch, preds)
        logits = preds[0] if isinstance(preds, (list, tuple)) else preds
        for threshold, confusion in confusions.items():
            update_confusion_matrix(confusion, logits, batch["mask"], nc, ignore_index, threshold=threshold)
            sample_ious = sample_iou_values(logits, batch["mask"], ignore_index, threshold)
            if sample_ious.numel():
                sample_ious_cpu = sample_ious.detach().cpu()
                finite = torch.isfinite(sample_ious_cpu)
                sample_iou_sums[threshold] += float(sample_ious_cpu[finite].sum())
                sample_iou_counts[threshold] += int(finite.sum())
                for iou_threshold in SAMPLE_IOU_THRESHOLDS:
                    sample_iou_hits[threshold][iou_threshold] += int(
                        (sample_ious_cpu[finite] >= iou_threshold).sum()
                    )
                if (
                    class_sample_iou_sums is not None
                    and class_sample_iou_counts is not None
                    and "class_idx" in batch
                    and sample_ious_cpu.numel() == batch["class_idx"].numel()
                ):
                    class_indices = batch["class_idx"].detach().cpu().view(-1).long()
                    for class_index, sample_iou, is_finite in zip(class_indices, sample_ious_cpu, finite):
                        index = int(class_index)
                        if bool(is_finite) and 0 <= index < len(class_sample_iou_sums[threshold]):
                            class_sample_iou_sums[threshold][index] += float(sample_iou)
                            class_sample_iou_counts[threshold][index] += 1
            if class_confusions is not None and "class_idx" in batch:
                update_text_class_confusion(
                    class_confusions[threshold],
                    logits,
                    batch["mask"],
                    batch["class_idx"],
                    ignore_index,
                    threshold=threshold,
                )
        running += float(loss.detach())
        seen = batch_index

        if len(preview_items) < len(requested_preview_paths):
            preview_items.append(
                (
                    {"img": batch["img"].detach().cpu(), "mask": batch["mask"].detach().cpu()},
                    logits.detach().cpu(),
                )
            )
        if max_batches and batch_index >= max_batches:
            break

    threshold_metrics = []
    for threshold, confusion in confusions.items():
        metrics = metrics_from_confusion(confusion, fbeta_beta=val_fbeta)
        metrics["threshold"] = threshold
        metrics["binary_miou"] = metrics["miou"]
        if text_queries:
            metrics["miou"] = metrics["target_iou"]
        sample_count = max(sample_iou_counts[threshold], 1)
        metrics["sample_miou"] = sample_iou_sums[threshold] / sample_count
        metrics["oiou"] = metrics["target_iou"]
        metrics["official_miou"] = metrics["sample_miou"]
        for iou_threshold in SAMPLE_IOU_THRESHOLDS:
            key = f"pr_{str(iou_threshold).replace('.', '_')}"
            metrics[key] = sample_iou_hits[threshold][iou_threshold] / sample_count
        metrics["selection_score"] = validation_selection_score(metrics, val_select_metric)
        threshold_metrics.append(metrics)

    best_metrics = max(
        threshold_metrics,
        key=lambda item: (float(item["selection_score"]), float(item["miou"]), float(item["f1"])),
    )
    best_threshold = float(best_metrics["threshold"])
    saved_preview_paths: List[str] = []
    for target_path, (preview_batch, preview_logits) in zip(requested_preview_paths, preview_items):
        save_preview(target_path, preview_batch, preview_logits, palette, threshold=best_threshold)
        saved_preview_paths.append(str(target_path))

    text_class_iou = (
        text_class_target_ious(class_confusions[best_threshold], class_names_for_text or [])
        if class_confusions is not None
        else {}
    )
    text_class_miou = (
        text_class_mean_ious(
            class_sample_iou_sums[best_threshold],
            class_sample_iou_counts[best_threshold],
            class_names_for_text or [],
        )
        if class_sample_iou_sums is not None and class_sample_iou_counts is not None
        else {}
    )
    class_macro_miou = float(np.mean(list(text_class_miou.values()))) if text_class_miou else float("nan")
    model.train()
    return {
        "loss": running / max(seen, 1),
        "pixel_acc": best_metrics["pixel_acc"],
        "miou": best_metrics["miou"],
        "oiou": best_metrics["oiou"],
        "official_miou": best_metrics["official_miou"],
        "target_iou": best_metrics["target_iou"],
        "binary_miou": best_metrics["binary_miou"],
        "best_threshold": best_threshold,
        "precision": best_metrics["precision"],
        "recall": best_metrics["recall"],
        "f1": best_metrics["f1"],
        "fbeta": best_metrics["fbeta"],
        "selection_score": best_metrics["selection_score"],
        "sample_miou": best_metrics["sample_miou"],
        **{
            f"pr_{str(iou_threshold).replace('.', '_')}": best_metrics[
                f"pr_{str(iou_threshold).replace('.', '_')}"
            ]
            for iou_threshold in SAMPLE_IOU_THRESHOLDS
        },
        "pred_pos_rate": best_metrics["pred_pos_rate"],
        "target_pos_rate": best_metrics["target_pos_rate"],
        "text_class_iou": text_class_iou,
        "text_class_miou": text_class_miou,
        "class_macro_miou": class_macro_miou,
        "per_class_iou": best_metrics["per_class_iou"],
        "threshold_metrics": threshold_metrics,
        "confusion": confusions[best_threshold],
        "evaluated_samples": int(sample_iou_counts[best_threshold]),
        "preview_files": saved_preview_paths,
    }


__all__ = (
    "SAMPLE_IOU_THRESHOLDS",
    "append_results",
    "image_to_uint8",
    "make_palette",
    "mask_to_rgb",
    "metrics_from_confusion",
    "plot_confusion_matrix",
    "plot_results",
    "sample_iou_values",
    "save_preview",
    "text_class_mean_ious",
    "text_class_target_ious",
    "update_confusion_matrix",
    "update_text_class_confusion",
    "validate",
)
