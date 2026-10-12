from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple

import torch


def sanitize_text(text: Any, empty_placeholder: str = "[NO_DESCRIPTION]") -> str:
    """Normalize one text sample and return a non-empty fallback string."""
    if text is None:
        value = ""
    elif isinstance(text, str):
        value = text
    else:
        value = str(text)

    value = value.replace("\r\n", "\n").replace("\r", "\n")
    value = " ".join(part.strip() for part in value.split("\n") if part.strip())
    value = " ".join(value.split())
    return value if value else empty_placeholder


def resolve_torch_device(device_arg: str = "auto") -> torch.device:
    """Parse device arg into torch.device with CUDA-first behavior."""
    if not device_arg or device_arg == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    raw = str(device_arg).strip().lower()
    if raw == "cpu":
        return torch.device("cpu")
    if raw in {"cuda", "gpu"}:
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if raw.isdigit():
        return torch.device(f"cuda:{raw}" if torch.cuda.is_available() else "cpu")
    try:
        return torch.device(device_arg)
    except (TypeError, ValueError, RuntimeError) as e:
        raise ValueError(f"Invalid device argument: {device_arg}") from e


def _resolve_precision(precision: str, device: torch.device) -> torch.dtype:
    value = (precision or "auto").lower()
    if value == "auto":
        return torch.float16 if device.type == "cuda" else torch.float32
    if value == "fp16":
        return torch.float16
    if value == "bf16":
        return torch.bfloat16
    if value == "fp32":
        return torch.float32
    raise ValueError(f"Unsupported precision: {precision}")


class OpenCLIPTextEncoder:
    """OpenCLIP text encoder wrapper for object-description vectorization."""

    def __init__(
        self,
        model_name: str = "ViT-L-14",
        pretrained: str = "openai",
        device: str = "auto",
        precision: str = "auto",
        normalize: bool = True,
    ) -> None:
        try:
            import open_clip
        except ImportError as e:
            raise ImportError(
                "open-clip-torch is required for text backbone. Install with: pip install open-clip-torch"
            ) from e

        self.model_name = model_name
        self.pretrained = pretrained
        self.device = resolve_torch_device(device)
        self.normalize = normalize
        self.precision = _resolve_precision(precision, self.device)
        if self.device.type != "cuda" and self.precision in {torch.float16, torch.bfloat16}:
            self.precision = torch.float32

        model, _, _ = open_clip.create_model_and_transforms(
            model_name=model_name,
            pretrained=pretrained,
            device=self.device,
        )
        self.model = model.eval()
        for param in self.model.parameters():
            param.requires_grad_(False)

        if self.device.type == "cuda" and self.precision in {torch.float16, torch.bfloat16}:
            self.model = self.model.to(dtype=self.precision)

        self.tokenizer = open_clip.get_tokenizer(model_name)
        with torch.inference_mode():
            probe_tokens = self.tokenizer(["probe"]).to(self.device)
            probe_embeddings = self.model.encode_text(probe_tokens)
            self.embedding_dim = int(probe_embeddings.shape[-1])

    def _encode_token_features(self, tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return contextualized token embeddings and non-special token mask."""
        if not hasattr(self.model, "token_embedding") or not hasattr(self.model, "transformer"):
            raise RuntimeError("Current OpenCLIP model does not expose token-level text modules.")

        x = self.model.token_embedding(tokens)
        pos = getattr(self.model, "positional_embedding", None)
        if isinstance(pos, torch.Tensor):
            x = x + pos[: x.shape[1]]

        x = x.permute(1, 0, 2)
        attn_mask = getattr(self.model, "attn_mask", None)
        try:
            x = self.model.transformer(x, attn_mask=attn_mask)
        except TypeError:
            x = self.model.transformer(x)
        x = x.permute(1, 0, 2)

        ln_final = getattr(self.model, "ln_final", None)
        if ln_final is not None:
            x = ln_final(x)

        text_projection = getattr(self.model, "text_projection", None)
        if isinstance(text_projection, torch.Tensor):
            x = x @ text_projection

        # For OpenCLIP tokenized inputs, argmax gives EOT index (largest token id).
        eot_idx = tokens.argmax(dim=-1)
        seq_len = int(tokens.shape[1])
        positions = torch.arange(seq_len, device=tokens.device).view(1, seq_len)
        token_mask = (positions > 0) & (positions < eot_idx.view(-1, 1)) & (tokens != 0)
        return x, token_mask

    def metadata(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "pretrained": self.pretrained,
            "device": str(self.device),
            "precision": str(self.precision),
            "normalize": self.normalize,
            "embedding_dim": self.embedding_dim,
        }

    def encode(
        self,
        texts: Sequence[str],
        batch_size: int = 256,
        normalize: Optional[bool] = None,
        return_tokens: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")

        clean_texts = [sanitize_text(t, empty_placeholder="") for t in texts]
        if not clean_texts:
            return torch.empty((0, self.embedding_dim), dtype=torch.float32)

        do_normalize = self.normalize if normalize is None else normalize
        encoded_batches = []
        token_mask_batches = []

        use_autocast = self.device.type == "cuda" and self.precision in {torch.float16, torch.bfloat16}
        for start in range(0, len(clean_texts), batch_size):
            chunk = clean_texts[start : start + batch_size]
            tokens = self.tokenizer(chunk)
            tokens = tokens.to(self.device, non_blocking=self.device.type == "cuda")

            with torch.inference_mode():
                if return_tokens:
                    if use_autocast:
                        with torch.autocast(device_type="cuda", dtype=self.precision):
                            feats, token_mask = self._encode_token_features(tokens)
                    else:
                        feats, token_mask = self._encode_token_features(tokens)
                    feats = feats.float()
                    if do_normalize:
                        feats = feats / feats.norm(p=2, dim=-1, keepdim=True).clamp_min(1e-12)
                    encoded_batches.append(feats.cpu())
                    token_mask_batches.append(token_mask.cpu())
                else:
                    if use_autocast:
                        with torch.autocast(device_type="cuda", dtype=self.precision):
                            feats = self.model.encode_text(tokens)
                    else:
                        feats = self.model.encode_text(tokens)
                    feats = feats.float()
                    if do_normalize:
                        feats = feats / feats.norm(p=2, dim=-1, keepdim=True).clamp_min(1e-12)
                    encoded_batches.append(feats.cpu())

        features = torch.cat(encoded_batches, dim=0)
        if return_tokens:
            token_mask = torch.cat(token_mask_batches, dim=0).to(dtype=torch.bool)
            return features, token_mask
        return features
