"""Counting-specific text-guidance configuration."""

from typing import Any, Dict, Mapping, Optional

from text_encoder.settings import DEFAULT_TEXT_GUIDANCE_CONFIG


class CountingTextConfig:
    """Apply the teammate's counting preset without changing detection logic."""

    OVERRIDES: Dict[str, Any] = {
        "enabled": True,
        # Count every prompted object regardless of its position in the image.
        "spatial_encoding_enabled": False,
        "visual_attr_include_geom": False,
        "lambda_spatial_quadrant": 0.0,
        "lambda_relation": 0.0,
        # Preserve the original appearance- and semantics-focused enhancements.
        "visual_attr_include_stats": True,
        "multi_proj_enabled": True,
        "film_enabled": True,
        "cross_attn_enabled": True,
        "cls_gate_strength": 0.8,
        "cls_fusion_mode": "additive",
    }

    @classmethod
    def build(cls, base: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
        """Return a complete text-guidance payload for counting."""
        config = dict(DEFAULT_TEXT_GUIDANCE_CONFIG if base is None else base)
        config.update(cls.OVERRIDES)
        return config
