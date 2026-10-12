"""Train the teammate's text-guided detection-based object counter."""

import argparse
from pathlib import Path
from typing import Any, Dict, Sequence

import torch
from ultralytics import YOLO

from tasks.counting.config import CountingTextConfig
from dataset.utils import prepare_dataset
from lib.general import (
    _parse_phrase_types,
    _parse_phrase_weight_string,
    print_metrics,
    resolve_device,
    resolve_local_weights,
)
from text_encoder import TextGuidedDetectionTrainer, TextGuidedDetectionValidator, configure_text_guidance
from text_encoder.train_set import add_text_guidance_train_args, build_text_guidance_config


class CountingTrainingApplication:
    """Class-based counting training entry that preserves the original pipeline."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = self._normalize_args(args)

    @staticmethod
    def build_parser() -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(
            description="Train the YOLOv12 text-guided object-counting task.",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
        parser.add_argument("--dataset", type=str, default="VRSBench", help="Dataset name used for paths and run naming.")
        parser.add_argument("--model", type=str, default="ultralytics/cfg/models/v12/yolov12m-counting.yaml", help="Counting model YAML.")
        parser.add_argument("--weights", type=str, default="pretrain_model/yolov12m.pt", help="Local YOLOv12m pretrained weights.")
        parser.add_argument("--text-model-name", type=str, default="ViT-L-14", help="OpenCLIP model name.")
        parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs.")
        parser.add_argument("--batch", type=int, default=4, help="Batch size.")
        parser.add_argument("--imgsz", type=int, default=800, help="Training image size.")
        parser.add_argument("--device", type=str, default="0", help="Device such as 0, cuda:0, cpu, or auto.")
        parser.add_argument("--data", type=str, default="", help="Existing data.yaml; when set, skip dataset preparation.")
        parser.add_argument("--voc-root", type=str, default="", help="VOC root containing images, annotations, and split files.")
        parser.add_argument("--prepared-dir", type=str, default="", help="Prepared dataset and embedding output directory.")
        parser.add_argument("--images-dir", type=str, default="JPEGImages", help="Image directory under the VOC root.")
        parser.add_argument("--annotations-dir", type=str, default="Annotations", help="Annotation directory under the VOC root.")
        parser.add_argument("--train-list", type=str, default="train.txt", help="Training split filename.")
        parser.add_argument("--val-list", type=str, default="val.txt", help="Validation split filename.")
        parser.add_argument("--test-list", type=str, default="test.txt", help="Test split filename.")
        parser.add_argument("--workers", type=int, default=2, help="Dataloader worker count.")
        parser.add_argument("--project", type=str, default="runs/counting", help="Training output project directory.")
        parser.add_argument("--name", type=str, default="", help="Experiment name.")
        parser.add_argument("--exist-ok", action="store_true", help="Reuse an existing experiment directory.")
        parser.add_argument("--patience", type=int, default=15, help="Early stopping patience.")
        parser.add_argument("--seed", type=int, default=0, help="Random seed.")
        parser.add_argument("--resume", action="store_true", help="Resume from the latest checkpoint.")
        parser.add_argument("--cache", action="store_true", help="Cache images during training.")
        parser.add_argument("--run-val", action="store_true", help="Run an extra validation after training.")
        parser.add_argument("--save-json", action="store_true", help="Save COCO JSON during extra validation.")
        add_text_guidance_train_args(parser)
        return parser

    @classmethod
    def from_cli(cls, argv: Sequence[str] | None = None) -> "CountingTrainingApplication":
        return cls(cls.build_parser().parse_args(argv))

    @staticmethod
    def _normalize_args(args: argparse.Namespace) -> argparse.Namespace:
        dataset = str(args.dataset or "VRSBench").strip()
        args.dataset = dataset
        args.voc_root = str(args.voc_root or f"data/{dataset}")
        args.prepared_dir = str(args.prepared_dir or f"pre_datasets/{dataset}")
        args.name = str(args.name or f"counting-{dataset}-{args.text_model_name}")
        return args

    @staticmethod
    def _safe_tag(value: str) -> str:
        return str(value or "").lower().replace("/", "-").replace(" ", "")

    def _resolve_prepared_root(self, data_path: str) -> Path:
        args = self.args
        base_root = (
            Path(args.prepared_dir).expanduser().resolve()
            if str(args.prepared_dir).strip()
            else Path(data_path).expanduser().resolve().parent
        )
        model_tag = f"openclip_{self._safe_tag(args.text_model_name)}_{self._safe_tag(args.text_pretrained)}"
        suffix = f"_{model_tag}"
        return base_root if base_root.name.endswith(suffix) else base_root.with_name(f"{base_root.name}{suffix}")

    def _resolve_embedding_root(self, data_path: str) -> Path:
        explicit = str(self.args.text_embedding_dir or "").strip()
        return Path(explicit).expanduser().resolve() if explicit else self._resolve_prepared_root(data_path)

    @staticmethod
    def _infer_embedding_dim(embedding_root: Path, fallback: int = 768) -> int:
        for embedding_file in sorted(embedding_root.glob("*_text_embeddings.pt")):
            try:
                payload = torch.load(embedding_file, map_location="cpu")
            except Exception:
                continue
            metadata = payload.get("model_meta", {})
            if isinstance(metadata, dict):
                try:
                    dimension = int(metadata.get("embedding_dim", 0))
                except (TypeError, ValueError):
                    dimension = 0
                if dimension > 0:
                    return dimension
            embeddings = payload.get("embeddings")
            if isinstance(embeddings, torch.Tensor) and embeddings.ndim >= 2 and int(embeddings.shape[-1]) > 0:
                return int(embeddings.shape[-1])
        return int(fallback)

    def _build_guidance_config(self, embedding_root: Path) -> Dict[str, Any]:
        args = self.args
        phrase_types = _parse_phrase_types(args.text_phrase_types)
        phrase_type_weights = _parse_phrase_weight_string(args.text_phrase_type_weights)
        base_config = build_text_guidance_config(args, embedding_root, phrase_types, phrase_type_weights)
        return CountingTextConfig.build(base_config)

    def run(self) -> None:
        args = self.args
        device = resolve_device(args.device)
        data_path = str(Path(args.data).expanduser().resolve()) if str(args.data).strip() else prepare_dataset(args, device)
        weights_path = resolve_local_weights(args.weights)
        model_path = Path(args.model).expanduser().resolve()

        embedding_root = self._resolve_embedding_root(data_path)
        args.text_embedding_dim = (
            int(args.text_embedding_dim)
            if int(args.text_embedding_dim) > 0
            else self._infer_embedding_dim(embedding_root)
        )
        configure_text_guidance(self._build_guidance_config(embedding_root))

        print("=" * 72)
        print(f"Task: text-guided object counting")
        print(f"Dataset: {args.dataset} ({args.voc_root})")
        print(f"Model: {model_path}")
        print(f"Pretrained weights: {weights_path}")
        print(f"Text embeddings: {embedding_root} (dim={args.text_embedding_dim})")
        print(f"Device: {device} | Batch: {args.batch} | Image size: {args.imgsz}")
        print("=" * 72)

        model = YOLO(str(model_path)).load(weights_path)
        train_kwargs: Dict[str, Any] = {
            "data": data_path,
            "epochs": args.epochs,
            "batch": args.batch,
            "imgsz": args.imgsz,
            "workers": args.workers,
            "project": args.project,
            "name": args.name,
            "exist_ok": args.exist_ok,
            "patience": args.patience,
            "seed": args.seed,
            "resume": args.resume,
            "cache": args.cache,
            "device": device,
        }
        train_metrics = model.train(trainer=TextGuidedDetectionTrainer, **train_kwargs)
        print_metrics("Train Metrics", train_metrics)

        trainer = getattr(model, "trainer", None)
        if trainer is not None:
            for label in ("best", "last"):
                checkpoint = getattr(trainer, label, None)
                if isinstance(checkpoint, Path):
                    print(f"{label} checkpoint: {checkpoint}")

        if args.run_val:
            val_kwargs: Dict[str, Any] = {
                "data": data_path,
                "imgsz": args.imgsz,
                "batch": args.batch,
                "workers": args.workers,
                "project": args.project,
                "name": f"{args.name}-val",
                "exist_ok": args.exist_ok,
                "save_json": args.save_json,
                "device": device,
            }
            val_metrics = model.val(validator=TextGuidedDetectionValidator, **val_kwargs)
            print_metrics("Validation Metrics", val_metrics)


def main() -> None:
    CountingTrainingApplication.from_cli().run()


if __name__ == "__main__":
    main()
