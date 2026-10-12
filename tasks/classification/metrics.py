"""Metrics for single-label scene classification."""

from typing import Dict, List

import torch


class SceneClassificationMetrics:
    def __init__(self, num_classes, class_names=None):
        self.num_classes = int(num_classes)
        self.class_names = list(class_names or [str(i) for i in range(self.num_classes)])
        self.confusion = torch.zeros((self.num_classes, self.num_classes), dtype=torch.long)
        self.total = 0
        self.top1_correct = 0
        self.top5_correct = 0

    def update(self, logits, labels):
        logits = logits.detach().cpu()
        labels = labels.detach().cpu().long().view(-1)
        predictions = logits.argmax(dim=1)
        topk = logits.topk(min(5, self.num_classes), dim=1).indices
        self.total += labels.numel()
        self.top1_correct += int((predictions == labels).sum())
        self.top5_correct += int((topk == labels.unsqueeze(1)).any(dim=1).sum())
        flat = labels * self.num_classes + predictions
        self.confusion += torch.bincount(flat, minlength=self.num_classes**2).reshape(self.num_classes, self.num_classes)

    def compute(self) -> Dict[str, object]:
        matrix = self.confusion.float()
        tp = matrix.diag()
        precision = tp / matrix.sum(dim=0).clamp_min(1.0)
        recall = tp / matrix.sum(dim=1).clamp_min(1.0)
        f1 = 2 * precision * recall / (precision + recall).clamp_min(1e-8)
        per_class: List[Dict[str, object]] = []
        for index, name in enumerate(self.class_names):
            per_class.append(
                {
                    "class_id": index,
                    "class_name": name,
                    "support": int(matrix[index].sum().item()),
                    "precision": float(precision[index]),
                    "recall": float(recall[index]),
                    "f1": float(f1[index]),
                }
            )
        denominator = max(self.total, 1)
        return {
            "evaluated_samples": self.total,
            "accuracy": self.top1_correct / denominator,
            "top1_accuracy": self.top1_correct / denominator,
            "top5_accuracy": self.top5_correct / denominator,
            "correct_predictions": self.top1_correct,
            "incorrect_predictions": self.total - self.top1_correct,
            "macro_precision": float(precision.mean()),
            "macro_recall": float(recall.mean()),
            "macro_f1": float(f1.mean()),
            "confusion_matrix": self.confusion.tolist(),
            "per_class": per_class,
        }
