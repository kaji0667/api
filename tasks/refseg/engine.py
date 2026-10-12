import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from ultralytics.nn import SemanticSegmentationModel
from ultralytics.utils import yaml_load, yaml_save

from .checkpoint import (
    checkpoint_improvement_flags,
    save_deployment_checkpoint,
)

from .data import (
    build_prompt_embeddings,
    build_rrsisd_text_embedding_cache,
    build_semseg_dataset,
    build_text_query_sampler,
)
from .evaluation import (
    SAMPLE_IOU_THRESHOLDS,
    append_results,
    make_palette,
    plot_confusion_matrix,
    plot_results,
    save_preview,
    validate,
)
from .runtime import (
    PROJECT_ROOT,
    class_names,
    configure_trainable_layers,
    dataset_type,
    load_pretrained_backbone,
    move_batch_to_device,
    parse_val_thresholds,
    predict_with_optional_text,
    resolve_checkpoint_path,
    resolve_split_path,
    seed_worker,
    set_random_seed,
)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the YOLOv12 text-guided semantic segmentation head on RRSIS-D.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data", type=str, default="pre_datasets/RRSIS-D_refseg/data.yaml", help="Semantic dataset yaml.")
    parser.add_argument("--model", type=str, default="ultralytics/cfg/models/v12/yolov12m-semseg.yaml", help="Model yaml.")
    parser.add_argument("--weights", type=str, default="pretrain_model/yolov12m.pt", help="Optional YOLO pretrained weights for backbone/neck initialization; empty disables loading.")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs.")
    parser.add_argument("--batch", type=int, default=2, help="Batch size.")
    parser.add_argument("--imgsz", type=int, default=128, help="Square training image size.")
    parser.add_argument("--workers", type=int, default=0, help="Dataloader workers.")
    parser.add_argument("--device", type=str, default="cpu", help="Training device, e.g. cpu, cuda, cuda:0.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for Python, NumPy, PyTorch, samplers, and dataloader workers.")
    parser.add_argument("--deterministic", action="store_true", help="Enable stricter deterministic PyTorch/CUDA behavior when supported.")
    parser.add_argument("--lr", type=float, default=1e-4, help="AdamW learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="AdamW weight decay.")
    parser.add_argument("--scheduler", type=str, default="none", choices=("none", "cosine"), help="Learning-rate scheduler.")
    parser.add_argument("--min-lr", type=float, default=1e-6, help="Minimum learning rate for cosine scheduler.")
    parser.add_argument("--patience", type=int, default=0, help="Early-stop after N epochs without validation selection-score improvement; 0 disables.")
    parser.add_argument("--min-delta", type=float, default=0.0, help="Minimum validation selection-score improvement required to reset early-stop patience.")
    parser.add_argument("--train-head-only", action="store_true", help="Freeze all layers except the final text segmentation head.")
    parser.add_argument("--freeze-backbone", action="store_true", help="Freeze YOLO backbone layers.")
    parser.add_argument("--freeze-neck", action="store_true", help="Freeze YOLO neck layers before the final segmentation head.")
    parser.add_argument("--pos-weight-max", type=float, default=20.0, help="Maximum positive BCE weight for binary mask loss.")
    parser.add_argument("--loss-small-target-weight", type=float, default=1.0, help="Per-sample loss multiplier for masks with foreground area <= --loss-small-target-area; 1 disables.")
    parser.add_argument("--loss-small-target-area", type=float, default=0.0025, help="Foreground mask area ratio threshold for area-aware loss weighting.")
    parser.add_argument("--loss-tversky-fp-weight", type=float, default=0.5, help="Tversky false-positive weight for binary mask loss; >0.5 penalizes over-segmentation more.")
    parser.add_argument("--loss-fp-weight", type=float, default=0.0, help="Extra mean foreground-probability penalty on GT background pixels; 0 disables.")
    parser.add_argument("--small-target-boost", type=float, default=2.0, help="Sampler multiplier for small-mask/bbox samples; 1 disables.")
    parser.add_argument("--small-target-area", type=float, default=0.0025, help="Area ratio threshold for small-target sampler boosting.")
    parser.add_argument("--augment", action=argparse.BooleanOptionalAction, default=True, help="Enable lightweight train-time RRSIS-D image/mask augmentation.")
    parser.add_argument("--augment-hflip", type=float, default=0.5, help="Train-time horizontal flip probability for samples without directional text.")
    parser.add_argument("--augment-vflip", type=float, default=0.5, help="Train-time vertical flip probability for samples without directional text.")
    parser.add_argument("--augment-color-jitter", type=float, default=0.15, help="Train-time brightness/contrast jitter strength.")
    parser.add_argument(
        "--augment-direction-policy",
        type=str,
        default="axis-aware",
        choices=("legacy", "axis-aware"),
        help="legacy blocks both flips for any position word; axis-aware blocks only the constrained axis.",
    )
    parser.add_argument(
        "--empty-mask-policy",
        type=str,
        default="drop",
        choices=("drop", "error", "keep"),
        help="How to handle decoded RRSIS-D annotations with zero foreground; drop is the cleaned default.",
    )
    parser.add_argument("--val-thresholds", type=str, default="0.5", help="Comma-separated sigmoid thresholds to scan during binary validation.")
    parser.add_argument(
        "--val-select-metric",
        type=str,
        default="iou",
        choices=("iou", "oiou", "miou", "f1", "fbeta", "precision"),
        help="Metric used to choose validation threshold and best checkpoint; iou is the legacy alias for oIoU.",
    )
    parser.add_argument("--val-fbeta", type=float, default=0.7, help="Beta for precision-biased F-beta validation score when --val-select-metric=fbeta; beta < 1 favors precision.")
    parser.add_argument("--max-batches", type=int, default=2, help="Stop each epoch after this many train batches; 0 means full epoch.")
    parser.add_argument("--max-val-batches", type=int, default=2, help="Stop validation after this many batches; 0 means full validation.")
    parser.add_argument("--test-after-train", action="store_true", help="Evaluate the best checkpoint on the test split using the validation-selected threshold.")
    parser.add_argument("--eval-only", action="store_true", help="Evaluate an existing full checkpoint without building the train/val datasets.")
    parser.add_argument("--checkpoint", type=str, default="", help="Full semantic-segmentation checkpoint used by --eval-only.")
    parser.add_argument("--max-test-batches", type=int, default=0, help="Stop test evaluation after this many batches; 0 means full test split.")
    parser.add_argument("--test-preview-batches", type=int, default=5, help="Save previews for the first N test batches; 0 disables test previews.")
    parser.add_argument("--print-interval", type=int, default=10, help="Print train progress every N batches.")
    parser.add_argument("--save-dir", type=str, default="runs/semseg/train", help="Directory for run artifacts.")
    parser.add_argument("--nosave", action="store_true", help="Disable checkpoint and artifact saving.")
    parser.add_argument("--no-val", action="store_true", help="Disable validation.")
    parser.add_argument("--no-plots", action="store_true", help="Disable results and confusion-matrix plots.")
    parser.add_argument("--no-preview", action="store_true", help="Disable train/val/test image preview saving.")
    parser.add_argument("--no-tensorboard", action="store_true", help="Disable TensorBoard event logging.")
    parser.add_argument("--text-queries", action="store_true", help="Use image + text prompt -> binary mask samples.")
    parser.add_argument("--text-encoder", type=str, default="openclip", choices=("learned", "openclip"), help="Prompt embedding source for text-query training.")
    parser.add_argument("--text-model-name", type=str, default="ViT-L-14", help="OpenCLIP text model name.")
    parser.add_argument("--text-pretrained", type=str, default="openai", help="OpenCLIP pretrained weights tag.")
    parser.add_argument("--text-precision", type=str, default="auto", choices=("auto", "fp32", "fp16", "bf16"), help="OpenCLIP encoding precision.")
    parser.add_argument("--text-embed-batch-size", type=int, default=256, help="Batch size for OpenCLIP split text embedding cache generation.")
    parser.add_argument("--text-overwrite", action="store_true", help="Overwrite cached split text embeddings.")
    return parser.parse_args(argv)


def build_summary_writer(save_dir: Path, disabled: bool):
    if disabled:
        return None
    try:
        from torch.utils.tensorboard import SummaryWriter

        return SummaryWriter(str(save_dir))
    except Exception as exc:
        print(f"TensorBoard unavailable: {exc}")
        return None


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    if int(args.test_preview_batches) < 0:
        raise ValueError("--test-preview-batches must be >= 0")
    if args.eval_only:
        if not args.checkpoint:
            raise ValueError("--eval-only requires --checkpoint.")
        args.epochs = 0
        args.test_after_train = True
        args.no_val = True
        args.no_tensorboard = True
    set_random_seed(args.seed, args.deterministic)
    val_thresholds = parse_val_thresholds(args.val_thresholds)
    data = yaml_load(args.data)
    data["_yaml_dir"] = str(Path(args.data).expanduser().resolve().parent)
    dtype = dataset_type(data)
    if dtype == "rrsisd_refseg":
        if not args.text_queries:
            print("RRSIS-D refseg data detected; enabling --text-queries.")
            args.text_queries = True
        if args.text_encoder != "openclip":
            raise ValueError("RRSIS-D refseg training requires --text-encoder openclip to use each free-text expression.")
    nc = int(data["nc"])
    ignore_index = int(data.get("ignore_index", 255))
    names = class_names(data, nc)
    metric_nc = 2 if args.text_queries else nc
    metric_names = ["other", "target"] if args.text_queries else names
    palette = make_palette(metric_nc)
    device = torch.device(args.device)
    save_dir = Path(args.save_dir)
    weights_dir = save_dir / "weights"
    results_csv = save_dir / "results.csv"

    if args.test_after_train and args.nosave:
        raise ValueError("--test-after-train requires checkpoint saving; remove --nosave.")

    train_jsonl = None if args.eval_only else resolve_split_path(data, "train")
    val_jsonl = resolve_split_path(data, "val") if "val" in data and not args.no_val and not args.eval_only else None
    test_jsonl = resolve_split_path(data, "test") if args.test_after_train and "test" in data else None
    split_paths = {}
    if train_jsonl:
        split_paths["train"] = train_jsonl
    if val_jsonl:
        split_paths["val"] = val_jsonl
    if test_jsonl:
        split_paths["test"] = test_jsonl
    text_embedding_paths = build_rrsisd_text_embedding_cache(args, data, split_paths, device)
    train_set = (
        build_semseg_dataset(
            data,
            train_jsonl,
            args,
            "train",
            text_embedding_file=text_embedding_paths.get("train"),
        )
        if train_jsonl
        else None
    )
    val_set = (
        build_semseg_dataset(
            data,
            val_jsonl,
            args,
            "val",
            text_embedding_file=text_embedding_paths.get("val"),
        )
        if val_jsonl
        else None
    )
    test_set = (
        build_semseg_dataset(
            data,
            test_jsonl,
            args,
            "test",
            text_embedding_file=text_embedding_paths.get("test"),
        )
        if test_jsonl
        else None
    )
    data_generator = torch.Generator()
    data_generator.manual_seed(int(args.seed))
    train_sampler = (
        build_text_query_sampler(train_set, args.small_target_boost, args.small_target_area, generator=data_generator)
        if args.text_queries and train_set is not None
        else None
    )
    train_loader = (
        DataLoader(
            train_set,
            batch_size=args.batch,
            shuffle=train_sampler is None,
            sampler=train_sampler,
            num_workers=args.workers,
            pin_memory=device.type == "cuda",
            worker_init_fn=seed_worker,
            generator=data_generator,
        )
        if train_set is not None
        else None
    )
    val_loader = (
        DataLoader(
            val_set,
            batch_size=args.batch,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=device.type == "cuda",
            worker_init_fn=seed_worker,
            generator=data_generator,
        )
        if val_set is not None
        else None
    )
    test_loader = (
        DataLoader(
            test_set,
            batch_size=args.batch,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=device.type == "cuda",
            worker_init_fn=seed_worker,
            generator=data_generator,
        )
        if test_set is not None
        else None
    )

    if not args.nosave:
        save_dir.mkdir(parents=True, exist_ok=True)
        weights_dir.mkdir(parents=True, exist_ok=True)
        yaml_save(save_dir / "args.yaml", vars(args))

    writer = build_summary_writer(save_dir, args.no_tensorboard or args.nosave)
    model = SemanticSegmentationModel(args.model, ch=3, nc=nc, verbose=False).to(device)
    if not args.eval_only:
        load_pretrained_backbone(model, args.weights, device)
    model.loss_pos_weight_max = float(args.pos_weight_max)
    model.loss_small_target_weight = float(args.loss_small_target_weight)
    model.loss_small_target_area = float(args.loss_small_target_area)
    model.loss_tversky_fp_weight = float(args.loss_tversky_fp_weight)
    model.loss_fp_weight = float(args.loss_fp_weight)
    trainable_params = configure_trainable_layers(model, args)
    if trainable_params <= 0 and not args.eval_only:
        raise RuntimeError("No trainable parameters remain after applying freeze options.")
    if args.eval_only:
        model.eval()
    else:
        model.train()
    optimizer = (
        None
        if args.eval_only
        else torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr, weight_decay=args.weight_decay)
    )
    scheduler = None
    if args.scheduler == "cosine" and optimizer is not None:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(int(args.epochs), 1),
            eta_min=float(args.min_lr),
        )
    prompt_embeddings = build_prompt_embeddings(args, data, names, device)

    print(f"data: {args.data}")
    print(f"dataset_type: {dtype}")
    print(f"mode: {'evaluation only' if args.eval_only else 'training'}")
    print(f"train samples: {len(train_set) if train_set is not None else 0}")
    print(f"val samples: {len(val_set) if val_set is not None else 0}")
    print(f"test samples: {len(test_set) if test_set is not None else 0}")
    print(f"classes: {nc}")
    print(f"weights: {'<not loaded in eval-only mode>' if args.eval_only else (args.weights or '<none>')}")
    if args.eval_only:
        print(f"checkpoint: {args.checkpoint}")
    print(f"text queries: {args.text_queries}")
    print(f"text encoder: {args.text_encoder}")
    print(f"device: {device}")
    print(
        f"imgsz: {args.imgsz}, batch: {args.batch}, "
        f"max_batches: {args.max_batches}, max_val_batches: {args.max_val_batches}"
    )
    print(f"scheduler: {args.scheduler}, min_lr: {args.min_lr}")
    print(f"seed: {args.seed}, deterministic: {args.deterministic}")
    print(f"early stopping: patience={args.patience}, min_delta={args.min_delta}")
    print(f"loss pos_weight max: {args.pos_weight_max}")
    print(
        "loss small target: "
        f"weight={args.loss_small_target_weight}, area<={args.loss_small_target_area}"
    )
    print(
        "loss over-segmentation: "
        f"tversky_fp_weight={args.loss_tversky_fp_weight}, fp_weight={args.loss_fp_weight}"
    )
    print(
        "train augmentation: "
        f"enabled={args.augment}, hflip={args.augment_hflip}, "
        f"vflip={args.augment_vflip}, color_jitter={args.augment_color_jitter}, "
        f"direction_policy={args.augment_direction_policy}"
    )
    print(f"validation thresholds: {val_thresholds}")
    print(
        "validation selection: "
        f"metric={args.val_select_metric}, fbeta={args.val_fbeta}"
    )
    if not args.nosave:
        print(f"save dir: {save_dir}")

    best_fitness = -float("inf")
    best_raw_fitness = -float("inf")
    epochs_without_improvement = 0
    last_confusion = None

    for epoch in range(args.epochs):
        epoch_start = time.time()
        running = 0.0
        seen = 0
        for batch_i, batch in enumerate(train_loader, start=1):
            batch = move_batch_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            preds = predict_with_optional_text(model, batch, args.text_queries, prompt_embeddings)
            loss, loss_items = model.loss(batch, preds)
            loss.backward()
            optimizer.step()

            running += float(loss.detach())
            seen = batch_i
            should_print = batch_i == 1 or batch_i % max(args.print_interval, 1) == 0
            if should_print:
                print(
                    f"epoch {epoch + 1}/{args.epochs} "
                    f"batch {batch_i}/{len(train_loader)} "
                    f"loss {float(loss.detach()):.4f} "
                    f"items {[round(float(x), 4) for x in loss_items.detach().cpu().view(-1)]}"
                )

            if not args.nosave and not args.no_preview and epoch == 0 and batch_i == 1:
                logits = preds[0] if isinstance(preds, (list, tuple)) else preds
                save_preview(save_dir / "train_batch0.jpg", batch, logits, palette)

            if args.max_batches and batch_i >= args.max_batches:
                break

        train_loss = running / max(seen, 1)
        val_metrics = None
        if val_loader is not None:
            preview_path = None
            if not args.nosave and not args.no_preview:
                preview_path = save_dir / f"val_batch0_pred_epoch{epoch + 1}.jpg"
            val_metrics = validate(
                model,
                val_loader,
                device,
                metric_nc,
                ignore_index,
                args.max_val_batches,
                preview_path,
                palette,
                args.text_queries,
                prompt_embeddings,
                names,
                val_thresholds,
                args.val_select_metric,
                args.val_fbeta,
            )
            last_confusion = val_metrics["confusion"]

        epoch_time = time.time() - epoch_start
        metrics = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"] if val_metrics else "",
            "pixel_acc": val_metrics["pixel_acc"] if val_metrics else "",
            "oiou": val_metrics["oiou"] if val_metrics else "",
            "official_miou": val_metrics["official_miou"] if val_metrics else "",
            "miou": val_metrics["miou"] if val_metrics else "",
            "target_iou": val_metrics["target_iou"] if val_metrics else "",
            "binary_miou": val_metrics["binary_miou"] if val_metrics else "",
            "best_threshold": val_metrics["best_threshold"] if val_metrics else "",
            "precision": val_metrics["precision"] if val_metrics else "",
            "recall": val_metrics["recall"] if val_metrics else "",
            "f1": val_metrics["f1"] if val_metrics else "",
            "fbeta": val_metrics["fbeta"] if val_metrics else "",
            "selection_score": val_metrics["selection_score"] if val_metrics else "",
            "sample_miou": val_metrics["sample_miou"] if val_metrics else "",
            "pr_0_5": val_metrics["pr_0_5"] if val_metrics else "",
            "pr_0_6": val_metrics["pr_0_6"] if val_metrics else "",
            "pr_0_7": val_metrics["pr_0_7"] if val_metrics else "",
            "pr_0_8": val_metrics["pr_0_8"] if val_metrics else "",
            "pr_0_9": val_metrics["pr_0_9"] if val_metrics else "",
            "pred_pos_rate": val_metrics["pred_pos_rate"] if val_metrics else "",
            "target_pos_rate": val_metrics["target_pos_rate"] if val_metrics else "",
            "class_macro_miou": val_metrics["class_macro_miou"] if val_metrics else "",
            "class_miou": json.dumps(val_metrics["text_class_miou"], sort_keys=True) if val_metrics else "",
            "class_oiou": json.dumps(val_metrics["text_class_iou"], sort_keys=True) if val_metrics else "",
            "class_iou": json.dumps(val_metrics["text_class_iou"], sort_keys=True) if val_metrics else "",
            "threshold_metrics": json.dumps(val_metrics["threshold_metrics"], sort_keys=True) if val_metrics else "",
            "lr": optimizer.param_groups[0]["lr"],
            "epoch_time": epoch_time,
        }
        print(
            f"epoch {epoch + 1} train_loss {train_loss:.4f} "
            + (
                f"val_loss {val_metrics['loss']:.4f} "
                f"pixel_acc {val_metrics['pixel_acc']:.4f} "
                f"oIoU {val_metrics['oiou']:.4f} "
                f"mIoU {val_metrics['official_miou']:.4f} "
                f"thr {val_metrics['best_threshold']:.2f} "
                f"P/R/F1/Fb {val_metrics['precision']:.4f}/{val_metrics['recall']:.4f}/{val_metrics['f1']:.4f}/{val_metrics['fbeta']:.4f} "
                f"select {val_metrics['selection_score']:.4f} "
                f"Pr@0.5/0.7/0.9 {val_metrics['pr_0_5']:.4f}/{val_metrics['pr_0_7']:.4f}/{val_metrics['pr_0_9']:.4f} "
                f"pred_pos {val_metrics['pred_pos_rate']:.4f} "
                if val_metrics
                else ""
            )
            + f"time {epoch_time:.1f}s"
        )
        if val_metrics and val_metrics.get("text_class_miou"):
            class_miou_text = ", ".join(
                f"{name}:{value:.3f}"
                for name, value in val_metrics["text_class_miou"].items()
                if name != "background"
            )
            print(f"per-class mIoU: {class_miou_text}")

        if writer is not None:
            writer.add_scalar("loss/train", train_loss, epoch + 1)
            writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch + 1)
            if val_metrics:
                writer.add_scalar("loss/val", val_metrics["loss"], epoch + 1)
                writer.add_scalar("metrics/pixel_acc", val_metrics["pixel_acc"], epoch + 1)
                writer.add_scalar("metrics/oIoU", val_metrics["oiou"], epoch + 1)
                writer.add_scalar("metrics/mIoU", val_metrics["official_miou"], epoch + 1)
                writer.add_scalar("metrics/best_threshold", val_metrics["best_threshold"], epoch + 1)
                writer.add_scalar("metrics/precision", val_metrics["precision"], epoch + 1)
                writer.add_scalar("metrics/recall", val_metrics["recall"], epoch + 1)
                writer.add_scalar("metrics/f1", val_metrics["f1"], epoch + 1)
                writer.add_scalar("metrics/fbeta", val_metrics["fbeta"], epoch + 1)
                writer.add_scalar("metrics/selection_score", val_metrics["selection_score"], epoch + 1)
                for iou_threshold in SAMPLE_IOU_THRESHOLDS:
                    key = f"pr_{str(iou_threshold).replace('.', '_')}"
                    writer.add_scalar(f"metrics/Pr@{iou_threshold}", val_metrics[key], epoch + 1)
                writer.add_scalar("metrics/pred_pos_rate", val_metrics["pred_pos_rate"], epoch + 1)
                writer.add_scalar("metrics/target_pos_rate", val_metrics["target_pos_rate"], epoch + 1)
                writer.add_scalar("metrics/class_macro_mIoU", val_metrics["class_macro_miou"], epoch + 1)
                for name, value in val_metrics.get("text_class_miou", {}).items():
                    writer.add_scalar(f"metrics/class_mIoU/{name}", value, epoch + 1)
                for name, value in val_metrics.get("text_class_iou", {}).items():
                    writer.add_scalar(f"metrics/class_oIoU/{name}", value, epoch + 1)

        fitness = float(val_metrics["selection_score"]) if val_metrics else -train_loss
        improved, raw_improved = checkpoint_improvement_flags(
            fitness,
            best_fitness,
            best_raw_fitness,
            float(args.min_delta),
        )
        next_best_fitness = fitness if improved else best_fitness
        next_best_raw_fitness = fitness if raw_improved else best_raw_fitness
        next_epochs_without_improvement = (
            0 if improved else epochs_without_improvement + (1 if val_metrics else 0)
        )

        if scheduler is not None:
            scheduler.step()

        if not args.nosave:
            checkpoint_metrics = {
                "train_loss": float(train_loss),
                "val_loss": float(val_metrics["loss"]) if val_metrics else float("nan"),
                "pixel_acc": float(val_metrics["pixel_acc"]) if val_metrics else float("nan"),
                "oiou": float(val_metrics["oiou"]) if val_metrics else float("nan"),
                "official_miou": float(val_metrics["official_miou"]) if val_metrics else float("nan"),
                "miou": float(val_metrics["miou"]) if val_metrics else float("nan"),
                "target_iou": float(val_metrics["target_iou"]) if val_metrics else float("nan"),
                "binary_miou": float(val_metrics["binary_miou"]) if val_metrics else float("nan"),
                "best_threshold": float(val_metrics["best_threshold"]) if val_metrics else float("nan"),
                "precision": float(val_metrics["precision"]) if val_metrics else float("nan"),
                "recall": float(val_metrics["recall"]) if val_metrics else float("nan"),
                "f1": float(val_metrics["f1"]) if val_metrics else float("nan"),
                "fbeta": float(val_metrics["fbeta"]) if val_metrics else float("nan"),
                "selection_score": float(val_metrics["selection_score"]) if val_metrics else float("nan"),
                "sample_miou": float(val_metrics["sample_miou"]) if val_metrics else float("nan"),
                "pr_0_5": float(val_metrics["pr_0_5"]) if val_metrics else float("nan"),
                "pr_0_6": float(val_metrics["pr_0_6"]) if val_metrics else float("nan"),
                "pr_0_7": float(val_metrics["pr_0_7"]) if val_metrics else float("nan"),
                "pr_0_8": float(val_metrics["pr_0_8"]) if val_metrics else float("nan"),
                "pr_0_9": float(val_metrics["pr_0_9"]) if val_metrics else float("nan"),
                "pred_pos_rate": float(val_metrics["pred_pos_rate"]) if val_metrics else float("nan"),
                "target_pos_rate": float(val_metrics["target_pos_rate"]) if val_metrics else float("nan"),
                "class_macro_miou": float(val_metrics["class_macro_miou"]) if val_metrics else float("nan"),
            }
            if raw_improved:
                save_deployment_checkpoint(
                    weights_dir / "best_raw.pt",
                    model,
                    epoch + 1,
                    args,
                    data,
                    checkpoint_metrics,
                )
                print(f"saved raw-best checkpoint: {weights_dir / 'best_raw.pt'}")
            append_results(results_csv, metrics)

            if not args.no_plots:
                plot_results(results_csv, save_dir / "results.png")
                if last_confusion is not None:
                    plot_confusion_matrix(last_confusion, metric_names, save_dir / "confusion_matrix.png")

        best_fitness = next_best_fitness
        best_raw_fitness = next_best_raw_fitness
        epochs_without_improvement = next_epochs_without_improvement

        if val_metrics and args.patience > 0 and epochs_without_improvement >= args.patience:
            print(
                f"early stopping: no {args.val_select_metric} improvement for {epochs_without_improvement} "
                f"epoch(s); best score {best_fitness:.4f}"
            )
            break

    if test_loader is not None:
        test_checkpoint_path = (
            resolve_checkpoint_path(args.checkpoint) if args.eval_only else weights_dir / "best_raw.pt"
        )
        if not test_checkpoint_path.exists():
            raise FileNotFoundError(f"Cannot run test evaluation without best checkpoint: {test_checkpoint_path}")
        checkpoint = torch.load(test_checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"], strict=True)
        checkpoint_metrics = checkpoint.get("metrics", {})
        checkpoint_args = checkpoint.get("args", {})
        checkpoint_selection_metric = str(checkpoint_args.get("val_select_metric", args.val_select_metric))
        test_threshold = float(checkpoint_metrics.get("best_threshold", val_thresholds[0]))
        if not np.isfinite(test_threshold):
            test_threshold = float(val_thresholds[0])
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
        test_start = time.perf_counter()
        test_preview_paths = (
            []
            if args.no_preview
            else [save_dir / f"test_batch{batch_index}_pred.jpg" for batch_index in range(args.test_preview_batches)]
        )
        test_metrics = validate(
            model,
            test_loader,
            device,
            metric_nc,
            ignore_index,
            args.max_test_batches,
            None,
            palette,
            args.text_queries,
            prompt_embeddings,
            names,
            [test_threshold],
            "oiou",
            args.val_fbeta,
            preview_paths=test_preview_paths,
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        test_seconds = time.perf_counter() - test_start
        evaluated_samples = max(int(test_metrics["evaluated_samples"]), 1)
        test_report = {
            "split": "test",
            "checkpoint": str(test_checkpoint_path),
            "checkpoint_epoch": int(checkpoint.get("epoch", 0)),
            "checkpoint_selection_metric": checkpoint_selection_metric,
            "threshold_source": (
                "explicit checkpoint validation threshold"
                if args.eval_only
                else "raw-best validation checkpoint"
            ),
            "threshold": test_threshold,
            "evaluated_samples": int(test_metrics["evaluated_samples"]),
            "oIoU": test_metrics["oiou"],
            "mIoU": test_metrics["official_miou"],
            "Pr@0.5": test_metrics["pr_0_5"],
            "Pr@0.6": test_metrics["pr_0_6"],
            "Pr@0.7": test_metrics["pr_0_7"],
            "Pr@0.8": test_metrics["pr_0_8"],
            "Pr@0.9": test_metrics["pr_0_9"],
            "precision": test_metrics["precision"],
            "recall": test_metrics["recall"],
            "f1": test_metrics["f1"],
            "pixel_acc": test_metrics["pixel_acc"],
            "pred_pos_rate": test_metrics["pred_pos_rate"],
            "target_pos_rate": test_metrics["target_pos_rate"],
            "class_macro_mIoU": test_metrics["class_macro_miou"],
            "class_mIoU": test_metrics["text_class_miou"],
            "class_oIoU": test_metrics["text_class_iou"],
            "total_parameters": sum(parameter.numel() for parameter in model.parameters()),
            "trainable_parameters": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
            "checkpoint_size_mb": test_checkpoint_path.stat().st_size / (1024**2),
            "evaluation_seconds": test_seconds,
            "mean_ms_per_sample": test_seconds * 1000.0 / evaluated_samples,
            "peak_gpu_memory_mb": (
                torch.cuda.max_memory_allocated(device) / (1024**2) if device.type == "cuda" else None
            ),
            "device": str(device),
            "imgsz": int(args.imgsz),
            "batch": int(args.batch),
            "text_encoder": args.text_encoder,
            "text_model_name": args.text_model_name,
            "text_pretrained": args.text_pretrained,
            "test_preview_batches_requested": int(args.test_preview_batches),
            "test_preview_files": test_metrics["preview_files"],
        }
        with (save_dir / "test_results.json").open("w", encoding="utf-8") as f:
            json.dump(test_report, f, ensure_ascii=False, indent=2, sort_keys=True)
        if not args.no_plots:
            plot_confusion_matrix(test_metrics["confusion"], metric_names, save_dir / "test_confusion_matrix.png")
        print(
            "test metrics: "
            f"oIoU={test_metrics['oiou']:.4f}, mIoU={test_metrics['official_miou']:.4f}, "
            f"Pr@0.5/0.7/0.9={test_metrics['pr_0_5']:.4f}/{test_metrics['pr_0_7']:.4f}/{test_metrics['pr_0_9']:.4f}, "
            f"threshold={test_threshold:.2f}"
        )
        print(f"saved test report: {save_dir / 'test_results.json'}")

    if writer is not None:
        writer.close()

    print("referring segmentation evaluation ok" if args.eval_only else "referring segmentation train ok")


if __name__ == "__main__":
    main()
