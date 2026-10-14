"""Shared versioned configuration, paths and integrity helpers."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

FORMAT_VERSION = "1.0"
ARCHITECTURE_ID = "hfsa_frozen_qformer_vqa_v2"
QFORMER_ARCHITECTURE_ID = ARCHITECTURE_ID
ARCHITECTURE_IDS = (ARCHITECTURE_ID,)
QFORMER_DEFAULTS = dict(qformer_queries=32, qformer_layers=2, qformer_heads=4,
                        qformer_ffn_multiplier=4, qformer_cross_attention_freq=2,
                        qformer_dropout=0.1)
SUPPORTED = ("category", "color", "shape", "existence", "scene",
             "quantity", "position", "size", "direction", "comparison", "relation", "other")
TYPES = SUPPORTED
DEFAULTS = dict(imgsz=640, grids=[16, 8, 4], channels=[256, 512, 512], text_dim=768,
                hidden_dim=192, dropout=0.2, seed=42, batch=128, epochs=15, lr=0.001,
                weight_decay=0.0001, patience=4, min_answer_count=1, workers=0,
                type_loss_weight=0.5, augmentation="none", sampling="type_balanced",
                precision="fp32_head_fp16_cache", text_model="ViT-L-14", text_chunk_tokens=75,
                model_yaml="ultralytics/cfg/models/v12/yolov12m-classification.yaml",
                fusion_backend="blip2_qformer", **QFORMER_DEFAULTS)

def validate_fusion_config(config):
    backend = config.get("fusion_backend", "blip2_qformer")
    if backend != "blip2_qformer":
        raise ValueError("VQA requires blip2_qformer fusion")
    for key, default in QFORMER_DEFAULTS.items():
        value = config.get(key, default)
        if key == "qformer_dropout":
            if not isinstance(value, (int, float)) or not 0 <= value < 1:
                raise ValueError("Invalid Q-Former dropout")
        elif type(value) is not int or value <= 0:
            raise ValueError(f"Invalid Q-Former setting: {key}")
    width = config["hidden_dim"]
    heads = config.get("qformer_heads", QFORMER_DEFAULTS["qformer_heads"])
    if type(width) is not int or width <= 0 or width % heads:
        raise ValueError("Q-Former width must be a positive multiple of attention heads")
    return backend

def architecture_id(config):
    validate_fusion_config(config)
    return ARCHITECTURE_ID

def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))

def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")

def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def resolve_path(value, base=None):
    value = str(value)
    import os
    if os.name != "nt" and len(value) > 2 and value[1] == ":":
        value = "/mnt/" + value[0].lower() + value[2:].replace("\\", "/")
    p = Path(value).expanduser()
    return (p if p.is_absolute() else Path(base or Path.cwd()) / p).resolve()

def resolve_device(value):
    import torch
    value = "cuda:0" if value == "auto" and torch.cuda.is_available() else "cpu" if value == "auto" else value
    d = torch.device(value)
    if d.type not in {"cpu", "cuda"} or (d.type == "cuda" and not torch.cuda.is_available()):
        raise ValueError("Unsupported or unavailable device")
    return d

def config_from_file(path):
    c = {**DEFAULTS, **read_json(path)}
    if c["channels"] != [256, 512, 512] or c["text_dim"] != 768 or c["text_model"] != "ViT-L-14" or c["text_chunk_tokens"] != 75:
        raise ValueError("Incompatible frozen-feature configuration")
    if c["augmentation"] != "none" or c["imgsz"] <= 0 or c["batch"] <= 0 or c["epochs"] <= 0:
        raise ValueError("Invalid training configuration")
    validate_fusion_config(c)
    return c
