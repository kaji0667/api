import argparse
from pathlib import Path
from typing import Any, Dict
import torch
from ultralytics import YOLO

from dataset.utils import prepare_dataset
from lib.general import print_metrics, resolve_device, resolve_local_weights, _parse_phrase_types, _parse_phrase_weight_string
from text_encoder import TextGuidedDetectionTrainer, TextGuidedDetectionValidator, configure_text_guidance
from text_encoder.train_set import add_text_guidance_train_args, build_text_guidance_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a YOLOv12 model with common settings.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset", type=str, default="DIOR-RSVG", help="Dataset name used to derive voc-root, prepared-dir, and run name.") #
    parser.add_argument("--text-model-name", type=str, default="ViT-L-14", help="OpenCLIP model name used when (re)building text embeddings.")
    parser.add_argument("--epochs", type=int, default=150, help="Number of training epochs.")
    parser.add_argument("--batch", type=int, default=16, help="Batch size.")
    parser.add_argument("--imgsz", type=int, default=800, help="Train image size.")
    parser.add_argument("--device", type=str, default="0", help="Device, e.g. '0', '0,1', 'cpu' or 'auto'.")

    parser.add_argument("--data", type=str, default="", help="Optional existing data.yaml path. If set, skip dataset preparation.")
    parser.add_argument("--images-dir", type=str, default="JPEGImages", help="Image folder name under VOC root.")
    parser.add_argument("--annotations-dir", type=str, default="Annotations", help="XML annotation folder name.")
    parser.add_argument("--train-list", type=str, default="train.txt", help="Train split txt file name.")
    parser.add_argument("--val-list", type=str, default="val.txt", help="Val split txt file name.")
    parser.add_argument("--test-list", type=str, default="test.txt", help="Test split txt file name.")
    parser.add_argument("--voc-root", type=str, help="VOC root with JPGEImages/JPEGImages, Annotations and split txt files.")
    parser.add_argument("--prepared-dir", type=str, help="Output dir for generated data.yaml and image-xml mapping files.")


    parser.add_argument("--workers", type=int, default=8, help="Dataloader worker count.")
    parser.add_argument("--project", type=str, default="runs/detect", help="Project directory.")
    parser.add_argument("--name", type=str, help="Experiment name.")
    parser.add_argument("--exist-ok", action="store_true", help="Reuse existing experiment directory.")
    parser.add_argument("--patience", type=int, default=100, help="Early stopping patience.")
    parser.add_argument("--weights","--model",dest="weights",type=str,default="./pretrain_model/yolov12m.pt",help="Local pretrained weights path, e.g. ./weights/yolov12n.pt",)
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint.")
    parser.add_argument("--cache", action="store_true", help="Cache images for faster training.")
    parser.add_argument("--run-val", action="store_true", help="Run an extra validation after training.")
    parser.add_argument("--save-json", action="store_true", help="Save COCO JSON during extra validation.")

    add_text_guidance_train_args(parser)
    args = parser.parse_args()

    dataset = str(getattr(args, "dataset", "") or "").strip() or "OPT_RSVG"
    args.dataset = dataset
    args.voc_root = str(args.voc_root or f"data/{dataset}")
    args.prepared_dir = str(args.prepared_dir or f"pre_datasets/{dataset}")
    args.name = str(args.name or f"{dataset}-{args.text_model_name}")
    return args


def _safe_tag(value: str) -> str:
    return str(value or "").lower().replace("/", "-").replace(" ", "")


def _resolve_prepared_root(args: argparse.Namespace, data_path: str) -> Path:
    base_root = Path(args.prepared_dir).expanduser().resolve() if str(args.prepared_dir or "").strip() else Path(data_path).expanduser().resolve().parent
    model_tag = f"openclip_{_safe_tag(args.text_model_name)}_{_safe_tag(args.text_pretrained)}"
    suffix = f"_{model_tag}"
    if base_root.name.endswith(suffix):
        return base_root
    return base_root.with_name(f"{base_root.name}{suffix}")


def _resolve_embedding_root(args: argparse.Namespace, data_path: str) -> Path:
    explicit = str(args.text_embedding_dir or "").strip()
    if explicit:
        return Path(explicit).expanduser().resolve()
    return _resolve_prepared_root(args, data_path)


def _infer_embedding_dim(embedding_root: Path, fallback: int = 768) -> int:
    files = sorted(embedding_root.glob("*_text_embeddings.pt"))

    for emb_file in files:
        try:
            payload = torch.load(emb_file, map_location="cpu")
        except Exception:
            continue

        model_meta = payload.get("model_meta", {})
        try:
            meta_dim = int(model_meta.get("embedding_dim", 0)) if isinstance(model_meta, dict) else 0
        except Exception:
            meta_dim = 0
        if meta_dim > 0:
            return meta_dim

        embeddings = payload.get("embeddings")
        if isinstance(embeddings, torch.Tensor) and embeddings.ndim >= 2 and int(embeddings.shape[-1]) > 0:
            return int(embeddings.shape[-1])

    return int(fallback)


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)

    data_path = str(Path(args.data).expanduser().resolve()) if str(args.data).strip() else prepare_dataset(args, device)
    weights_path = resolve_local_weights(args.weights)
    print(f"Using local pretrained weights: {weights_path}")

    model = YOLO(weights_path)
    print(f"Using device: {device}")

    embedding_root = _resolve_embedding_root(args, data_path)
    inferred_embedding_dim = int(args.text_embedding_dim) if int(args.text_embedding_dim) > 0 else _infer_embedding_dim(embedding_root)
    args.text_embedding_dim = inferred_embedding_dim
    phrase_types = _parse_phrase_types(args.text_phrase_types)
    phrase_type_weights = _parse_phrase_weight_string(args.text_phrase_type_weights)
    configure_text_guidance(build_text_guidance_config(args, embedding_root, phrase_types, phrase_type_weights))
    print(f"Text guidance enabled, embedding root: {embedding_root}")
    print(f"Text embedding dim: {inferred_embedding_dim}")
    print(
        "Text phrase config: "
        f"types={phrase_types}, aggregation={args.text_aggregation_mode}, "
        f"type_weights={phrase_type_weights if phrase_type_weights else '<default>'}"
    )
    print(
        "Text cls fusion config: "
        f"mode={args.text_cls_fusion_mode}, strength={args.text_cls_gate_strength}, "
        f"nonnegative={args.text_cls_gate_nonnegative}, "
        f"temperature={args.text_cls_gate_temperature}, cap={args.text_cls_gate_bias_cap}"
    )
    print(
        "Text/visual enhancement config: "
        f"text_seq={args.text_seq_enhance}({args.text_seq_conv_layers}x{args.text_seq_kernel_size}, "
        f"pool={args.text_seq_pooling_mode}), "
        f"visual_attr={args.visual_attr_enabled}(scale={args.visual_attr_scale}), "
        f"multi_proj={args.text_multi_proj_enabled}(score_scale={args.text_multi_proj_score_scale}, "
        f"orth_w={args.text_orth_loss_weight})"
    )
    print(
        "Text v2 config: "
        f"lora={args.text_lora_enabled}(r={args.text_lora_rank},a={args.text_lora_alpha}), "
        f"film={args.text_film_enabled}(s={args.text_film_strength}), "
        f"cross_attn={args.text_cross_attn_enabled}(h={args.text_cross_attn_heads},d={args.text_cross_attn_dim}), "
        f"contrastive={args.text_contrastive_loss_type}(temp={args.text_infonce_temperature},k={args.text_hard_neg_k}), "
        f"dependency={args.text_dependency_enabled}(strength={args.text_dependency_strength},"
        f"lambda={args.text_lambda_dependency}), "
        f"lambda_diou={args.text_lambda_diou}"
    )
    print(
        "Text embedding build config: "
        f"model={args.text_model_name}, pretrained={args.text_pretrained}, "
        f"precision={args.text_precision}, overwrite={args.text_overwrite}"
    )

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
    }
    train_kwargs["device"] = device


    train_metrics = model.train(trainer=TextGuidedDetectionTrainer, **train_kwargs)
    print_metrics("Train Metrics", train_metrics)

    trainer = getattr(model, "trainer", None)
    if trainer is not None:
        best = getattr(trainer, "best", None)
        last = getattr(trainer, "last", None)
        if isinstance(best, Path):
            print(f"best checkpoint: {best}")
        if isinstance(last, Path):
            print(f"last checkpoint: {last}")

    if args.run_val:
        print("Running extra validation...")
        val_kwargs: Dict[str, Any] = {
            "data": data_path,
            "imgsz": args.imgsz,
            "batch": args.batch,
            "workers": args.workers,
            "project": args.project,
            "name": f"{args.name}-val",
            "exist_ok": args.exist_ok,
            "save_json": args.save_json,
        }
        val_kwargs["device"] = device

        val_metrics = model.val(validator=TextGuidedDetectionValidator, **val_kwargs)
        print_metrics("Validation Metrics", val_metrics)


if __name__ == "__main__":
    main()
