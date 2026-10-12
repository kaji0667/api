"""Train the integrated YOLOv12m multi-scale scene-classification head."""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from tasks.classification import SceneClassificationConfig, SceneClassificationNetwork, SceneClassificationTrainer, SceneDataModule


class SceneClassificationTrainingApplication:
    def __init__(self, args):
        self.args = args

    @staticmethod
    def build_parser():
        parser = argparse.ArgumentParser(
            description="Train single-label remote-sensing scene classification.",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
        parser.add_argument("--data-dir", default="data/VRSBench_scene")
        parser.add_argument("--model", default="ultralytics/cfg/models/v12/yolov12m-classification.yaml")
        parser.add_argument("--weights", default="pretrain_model/yolov12m.pt")
        parser.add_argument("--save-dir", default="runs/classification/vrsbench_scene")
        parser.add_argument("--batch", type=int, default=32)
        parser.add_argument("--epochs", type=int, default=20)
        parser.add_argument("--lr", type=float, default=1e-3)
        parser.add_argument("--weight-decay", type=float, default=1e-4)
        parser.add_argument("--warmup-epochs", type=int, default=2)
        parser.add_argument("--grad-clip-norm", type=float, default=1.0)
        parser.add_argument("--imgsz", type=int, default=640)
        parser.add_argument("--val-ratio", type=float, default=0.2)
        parser.add_argument("--seed", type=int, default=42)
        parser.add_argument("--workers", type=int, default=2)
        parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
        parser.add_argument("--sampling", choices=("balanced", "none"), default="balanced")
        parser.add_argument("--max-train-batches", type=int, default=0)
        parser.add_argument("--max-val-batches", type=int, default=0)
        return parser

    @classmethod
    def from_cli(cls, argv=None):
        return cls(cls.build_parser().parse_args(argv))

    def run(self):
        config = SceneClassificationConfig(**vars(self.args))
        config.validate()
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.seed)

        data = SceneDataModule(
            config.data_dir,
            imgsz=config.imgsz,
            batch=config.batch,
            workers=config.workers,
            val_ratio=config.val_ratio,
            seed=config.seed,
            sampling=config.sampling,
        )
        train_loader, val_loader, classes = data.build_train_val()
        network = SceneClassificationNetwork(config.model, len(classes), config.weights, config.device)
        output_dir = Path(config.save_dir).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "args.json").write_text(
            json.dumps({**config.to_dict(), "classes": classes}, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"scene classes: {len(classes)} | train={len(train_loader.dataset)} | val={len(val_loader.dataset)}")
        print(
            f"pretrained matched/skipped tensors: {network.pretrained_report['matched_tensors']}/"
            f"{network.pretrained_report['skipped_tensors']}"
        )
        print(
            f"trainable/frozen parameters: "
            f"{sum(p.numel() for p in network.parameters() if p.requires_grad):,}/"
            f"{sum(p.numel() for p in network.parameters() if not p.requires_grad):,}"
        )
        best = SceneClassificationTrainer(network, config, classes).fit(train_loader, val_loader)
        print(f"best checkpoint: {best}")


def main():
    SceneClassificationTrainingApplication.from_cli().run()


if __name__ == "__main__":
    main()
