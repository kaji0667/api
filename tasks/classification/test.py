"""Evaluate or run image inference with an integrated scene-classification checkpoint."""

import argparse
from pathlib import Path

import torch

from tasks.evaluation_resources import environment_summary, file_size_mb, model_parameters

from tasks.classification import (
    SceneClassificationEvaluator,
    SceneClassificationNetwork,
    SceneClassificationPredictor,
    SceneDataModule,
)


class SceneClassificationEvaluationApplication:
    def __init__(self, args):
        self.args = args

    @staticmethod
    def build_parser():
        parser = argparse.ArgumentParser(
            description="Evaluate single-label remote-sensing scene classification.",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
        parser.add_argument("--checkpoint", default="runs/classification/vrsbench_scene/weights/best.pt")
        parser.add_argument("--data-dir", default="data/VRSBench_scene")
        parser.add_argument("--model", default="")
        parser.add_argument("--weights", default="pretrain_model/yolov12m.pt")
        parser.add_argument("--split", choices=("train", "val", "test", "all"), default="val")
        parser.add_argument("--save-dir", default="runs/classification/vrsbench_scene_eval")
        parser.add_argument("--batch", type=int, default=32)
        parser.add_argument("--imgsz", type=int, default=640)
        parser.add_argument("--val-ratio", type=float, default=0.2)
        parser.add_argument("--seed", type=int, default=42)
        parser.add_argument("--workers", type=int, default=2)
        parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
        parser.add_argument("--max-batches", type=int, default=0)
        parser.add_argument("--image", default="", help="Optional single image for Top-K inference instead of split evaluation.")
        parser.add_argument("--topk", type=int, default=5)
        parser.add_argument("--benchmark-samples", type=int, default=50)
        parser.add_argument("--benchmark-warmup", type=int, default=5)
        return parser

    @classmethod
    def from_cli(cls, argv=None):
        return cls(cls.build_parser().parse_args(argv))

    def run(self):
        checkpoint_path = Path(self.args.checkpoint).expanduser().resolve()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Scene classification checkpoint not found: {checkpoint_path}")
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        classes = list(payload.get("classes", []))
        if not classes:
            raise ValueError("Checkpoint does not contain the scene class list.")
        model_yaml = self.args.model or payload.get(
            "model_yaml", "ultralytics/cfg/models/v12/yolov12m-classification.yaml"
        )
        if self.args.image:
            predictor = SceneClassificationPredictor(
                checkpoint=checkpoint_path,
                pretrained_weights=self.args.weights,
                device=self.args.device,
                imgsz=self.args.imgsz,
                model_yaml=model_yaml,
            )
            result = predictor.predict(self.args.image, topk=self.args.topk)
            for item in result["predictions"]:
                print(f"{item['class_name']}: {item['probability']:.6f}")
            return

        network = SceneClassificationNetwork(model_yaml, len(classes), self.args.weights, self.args.device)
        network.load_head_checkpoint(checkpoint_path, strict=True)

        data = SceneDataModule(
            self.args.data_dir,
            imgsz=self.args.imgsz,
            batch=self.args.batch,
            workers=self.args.workers,
            val_ratio=self.args.val_ratio,
            seed=self.args.seed,
            sampling="none",
        )
        loader, dataset_classes = data.build_eval(self.args.split)
        if dataset_classes != classes:
            raise ValueError("Checkpoint class order does not match the evaluation ImageFolder class order.")
        evaluator = SceneClassificationEvaluator(network, self.args.device, classes)
        report = evaluator.evaluate(loader, max_batches=self.args.max_batches)
        benchmark_data = SceneDataModule(
            self.args.data_dir,
            imgsz=self.args.imgsz,
            batch=1,
            workers=0,
            val_ratio=self.args.val_ratio,
            seed=self.args.seed,
            sampling="none",
        )
        benchmark_loader, benchmark_classes = benchmark_data.build_eval(self.args.split)
        if benchmark_classes != classes:
            raise ValueError("Benchmark class order does not match the checkpoint class order.")
        report.update(
            {
                "resource": {
                    "parameters": model_parameters(network),
                    "artifacts": {
                        "classification_checkpoint_mb": file_size_mb(checkpoint_path),
                        "backbone_pretrained_mb": file_size_mb(self.args.weights),
                    },
                    "batch1_end_to_end": evaluator.benchmark(
                        benchmark_loader,
                        max_samples=self.args.benchmark_samples,
                        warmup=self.args.benchmark_warmup,
                    ),
                },
                "environment": environment_summary(self.args.device),
            }
        )
        path = evaluator.save_report(report, self.args.save_dir, checkpoint_path, self.args.split)
        print(
            f"scene classification: top1={report['top1_accuracy']:.6f}, top5={report['top5_accuracy']:.6f}, "
            f"macro_f1={report['macro_f1']:.6f}, samples={report['evaluated_samples']}"
        )
        print(f"saved report: {path}")


def main():
    SceneClassificationEvaluationApplication.from_cli().run()


if __name__ == "__main__":
    main()
