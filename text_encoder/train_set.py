from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Sequence


def add_text_guidance_train_args(parser: argparse.ArgumentParser) -> None:
    """Register text-guided training arguments on a parser."""
    parser.add_argument("--text-pretrained", type=str, default="openai", help="OpenCLIP pretrained tag used when (re)building text embeddings.")
    parser.add_argument("--text-precision", type=str, default="auto", choices=("auto", "fp32", "fp16", "bf16"), help="Text encoding precision for embedding generation.")
    parser.add_argument("--text-overwrite", action="store_true", help="Overwrite existing split embedding files when preparing dataset.")
    parser.add_argument("--text-embedding-dir", type=str, default="", help="Directory that contains *_text_embeddings.pt files. Empty means --prepared-dir.")
    parser.add_argument("--text-embedding-dim", type=int, default=0, help="Input text embedding dimension. 0 means infer from embedding files.")
    parser.add_argument("--text-phrase-types", type=str, default="NP,PP,ADJP", help="Phrase types used in preprocessing, comma-separated.")
    parser.add_argument("--text-phrase-type-weights",type=str,default="NP:1.0,PP:1.2,ADJP:0.8,FALLBACK:1.0",help="Phrase type priors for weighted phrase aggregation.")
    parser.add_argument("--text-aggregation-mode",type=str,choices=("lse", "weighted_sum", "mean", "dependency_lse"),default="lse",help="Phrase-to-heatmap aggregation strategy.")
    parser.add_argument("--text-guidance-strength", type=float, default=0.25, help="Feature modulation strength for text guidance.")
    parser.add_argument("--text-cls-gate-strength", type=float, default=0.8, help="Logit gating strength applied to the classification branch.")
    parser.add_argument("--text-cls-fusion-mode",type=str,choices=("additive", "multiplicative"),default="additive",help="How text guidance is fused into classification logits.")
    parser.add_argument("--text-cls-gate-temperature",type=float,default=1.0,help="Temperature for transforming text gate logits before cls fusion.")
    parser.add_argument("--text-cls-gate-bias-cap",type=float,default=1.0,help="Clamp magnitude for cls gate bias term (0 disables clamping).")
    parser.add_argument("--text-allow-negative-cls-bias",action="store_false",dest="text_cls_gate_nonnegative",help="Allow text cls gate bias to be negative.")
    parser.add_argument("--text-alignment-temperature", type=float, default=0.07, help="Temperature used in phrase-visual similarity computation.")
    parser.add_argument("--text-fuse-temperature", type=float, default=0.5, help="Temperature used in log-sum-exp phrase fusion.")
    parser.add_argument("--text-lambda-heatmap", type=float, default=0.3, help="Phrase contrastive supervision weight.")
    parser.add_argument("--text-lambda-phrase", type=float, default=0.2, help="Phrase-level supervision weight.")
    parser.add_argument("--text-lambda-set", type=float, default=0.6, help="Set matching supervision weight.")
    parser.add_argument("--text-matching-temperature", type=float, default=0.7, help="Temperature for set matching logits.")
    parser.add_argument("--text-seq-enhance", action="store_true", help="Enable lightweight text sequence enhancement before similarity.")
    parser.add_argument("--text-seq-conv-layers", type=int, default=1, help="Number of 1D conv residual layers for text sequence enhancement.")
    parser.add_argument("--text-seq-kernel-size", type=int, default=3, help="Kernel size for text sequence 1D conv.")
    parser.add_argument("--text-seq-dropout", type=float, default=0.0, help="Dropout ratio inside text sequence enhancement block.")
    parser.add_argument("--text-seq-pooling-mode",type=str,choices=("none", "learnable_weight"),default="learnable_weight",help="Adaptive token weighting mode used before phrase aggregation.")
    parser.add_argument("--text-seq-pool-temperature", type=float, default=1.0, help="Temperature for adaptive token weighting.")
    parser.add_argument("--visual-attr-enabled", type=bool, default=True, help="Enable normalized geometry/stat attributes on visual features before similarity.")
    parser.add_argument("--visual-attr-scale", type=float, default=1.0, help="Scale factor for visual attribute channels.")
    parser.add_argument("--visual-attr-eps", type=float, default=1e-6, help="Numerical epsilon used in visual attribute normalization.")
    parser.add_argument("--text-multi-proj-enabled", type=bool, default=True, help="Enable three parallel text-visual projections (geo/attr/sem) before similarity.")
    parser.add_argument("--text-multi-proj-score-scale", type=float, default=1.0, help="Scale factor applied after summing three projection scores.")
    parser.add_argument("--text-orth-loss-weight", type=float, default=0.05, help="Weight for orthogonality regularizer across three visual projections.")
    parser.add_argument("--text-lora-enabled", action="store_true", help="Enable LoRA adapters on text/visual projection layers.")
    parser.add_argument("--text-lora-rank", type=int, default=8, help="LoRA rank for text-guidance projection adapters.")
    parser.add_argument("--text-lora-alpha", type=float, default=16.0, help="LoRA scaling alpha for projection adapters.")
    parser.add_argument("--text-lora-dropout", type=float, default=0.0, help="Dropout ratio used before LoRA low-rank branches.")
    parser.add_argument("--text-film-enabled", action="store_true", help="Enable FiLM modulation for feature guidance.")
    parser.add_argument("--text-film-strength", type=float, default=0.25, help="Residual strength used by FiLM feature modulation.")
    parser.add_argument("--text-cross-attn-enabled", action="store_true", default=True, help="Enable cross-attention from visual tokens to text tokens.")
    parser.add_argument("--no-text-cross-attn-enabled", action="store_false", dest="text_cross_attn_enabled", help="Disable cross-attention from visual tokens to text tokens.")
    parser.add_argument("--text-cross-attn-heads", type=int, default=4, help="Number of heads in text-visual cross-attention.")
    parser.add_argument("--text-cross-attn-dim", type=int, default=128, help="Projection dim used in text-visual cross-attention.")
    parser.add_argument("--text-cross-attn-dropout", type=float, default=0.0, help="Dropout ratio in text-visual cross-attention.")
    parser.add_argument("--text-dependency-enabled", action="store_true", help="Enable dependency-aware token weighting in phrase aggregation.")
    parser.add_argument("--text-dependency-strength", type=float, default=0.25, help="Residual strength for dependency-aware token weighting.")
    parser.add_argument("--text-lambda-dependency", type=float, default=0.05, help="Weight for dependency consistency loss.")
    parser.add_argument("--text-contrastive-loss-type", type=str, choices=("logsigmoid_margin", "infonce"), default="infonce", help="Phrase contrastive loss type.")
    parser.add_argument("--text-infonce-temperature", type=float, default=0.25, help="Temperature for InfoNCE phrase contrastive loss.")
    parser.add_argument("--text-hard-neg-k", type=int, default=32, help="Top-k hard negatives used in phrase contrastive loss.")
    parser.add_argument("--text-use-in-batch-negatives", action="store_true", help="Use in-batch negatives in phrase InfoNCE loss.")
    parser.add_argument("--text-lambda-diou", type=float, default=0.15, help="Weight for DIoU grounding loss.")
    parser.add_argument("--text-diou-temperature", type=float, default=1.0, help="Temperature for token-to-box soft localization used by DIoU grounding.")
    parser.add_argument("--strict-match-iou", type=float, default=0.5, help="IoU threshold used by strict 1:1 grounding metrics.")
    parser.add_argument("--strict-nms-conf", type=float, default=0.25, help="Confidence threshold for strict grounding candidate boxes.")
    parser.add_argument("--strict-max-candidates-factor",type=int,default=4,help="Limit strict grounding candidates to max(descriptions, factor * gt_count).")
    parser.add_argument("--no-strict-grounding", action="store_false", dest="strict_grounding", help="Disable strict 1:1 grounding metrics.")
    parser.set_defaults(strict_grounding=True)
    parser.add_argument("--text-keep-augmentations", action="store_false", dest="text_disable_augmentations", help="Keep original augmentations in text-guided training.")


def build_text_guidance_config(
    args: argparse.Namespace,
    embedding_root: Path,
    phrase_types: Sequence[str],
    phrase_type_weights: Dict[str, float],
) -> Dict[str, Any]:
    """Build configure_text_guidance payload from command args."""
    return {
        "enabled": True,
        "embedding_dir": str(Path(embedding_root).expanduser().resolve()),
        "embedding_dim": int(args.text_embedding_dim),
        "phrase_types": list(phrase_types),
        "phrase_type_weights": dict(phrase_type_weights),
        "aggregation_mode": str(args.text_aggregation_mode),
        "guidance_strength": float(args.text_guidance_strength),
        "cls_gate_strength": float(args.text_cls_gate_strength),
        "cls_fusion_mode": str(args.text_cls_fusion_mode),
        "cls_gate_nonnegative": bool(args.text_cls_gate_nonnegative),
        "cls_gate_temperature": float(args.text_cls_gate_temperature),
        "cls_gate_bias_cap": float(args.text_cls_gate_bias_cap),
        "alignment_temperature": float(args.text_alignment_temperature),
        "fuse_temperature": float(args.text_fuse_temperature),
        "lambda_heatmap": float(args.text_lambda_heatmap),
        "lambda_phrase": float(args.text_lambda_phrase),
        "lambda_set": float(args.text_lambda_set),
        "matching_temperature": float(args.text_matching_temperature),
        "strict_grounding": bool(args.strict_grounding),
        "strict_match_iou": float(args.strict_match_iou),
        "strict_nms_conf": float(args.strict_nms_conf),
        "strict_max_candidates_factor": int(args.strict_max_candidates_factor),
        "disable_augmentations": bool(args.text_disable_augmentations),
        "text_seq_enhance": bool(args.text_seq_enhance),
        "text_seq_conv_layers": int(args.text_seq_conv_layers),
        "text_seq_kernel_size": int(args.text_seq_kernel_size),
        "text_seq_dropout": float(args.text_seq_dropout),
        "text_seq_pooling_mode": str(args.text_seq_pooling_mode),
        "text_seq_pool_temperature": float(args.text_seq_pool_temperature),
        "visual_attr_enabled": bool(args.visual_attr_enabled),
        "visual_attr_scale": float(args.visual_attr_scale),
        "visual_attr_eps": float(args.visual_attr_eps),
        "multi_proj_enabled": bool(args.text_multi_proj_enabled),
        "multi_proj_score_scale": float(args.text_multi_proj_score_scale),
        "orth_loss_weight": float(args.text_orth_loss_weight),
        "lora_enabled": bool(args.text_lora_enabled),
        "lora_rank": int(args.text_lora_rank),
        "lora_alpha": float(args.text_lora_alpha),
        "lora_dropout": float(args.text_lora_dropout),
        "film_enabled": bool(args.text_film_enabled),
        "film_strength": float(args.text_film_strength),
        "cross_attn_enabled": bool(args.text_cross_attn_enabled),
        "cross_attn_heads": int(args.text_cross_attn_heads),
        "cross_attn_dim": int(args.text_cross_attn_dim),
        "cross_attn_dropout": float(args.text_cross_attn_dropout),
        "dependency_enabled": bool(args.text_dependency_enabled),
        "dependency_strength": float(args.text_dependency_strength),
        "lambda_dependency": float(args.text_lambda_dependency),
        "contrastive_loss_type": str(args.text_contrastive_loss_type),
        "infonce_temperature": float(args.text_infonce_temperature),
        "hard_neg_k": int(args.text_hard_neg_k),
        "use_in_batch_negatives": bool(args.text_use_in_batch_negatives),
        "lambda_diou": float(args.text_lambda_diou),
        "diou_temperature": float(args.text_diou_temperature),
    }
