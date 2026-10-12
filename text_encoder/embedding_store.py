from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
import warnings

import torch


class ObjectTextEmbeddingStore:
    """In-memory lookup for sample_id -> phrase matrix [num_phrases, dim]."""

    def __init__(self, embedding_root: str, expected_dim: int = 0) -> None:
        self.root = Path(embedding_root).expanduser().resolve()
        self.expected_dim = int(expected_dim) if int(expected_dim) > 0 else 0
        self._vectors: Dict[str, torch.Tensor] = {}
        self._weights: Dict[str, torch.Tensor] = {}
        self._targets: Dict[str, torch.Tensor] = {}
        self._dependency_edges: Dict[str, torch.Tensor] = {}
        self._phrase_types: Dict[str, List[str]] = {}  # NEW: store phrase types
        self._dim = self.expected_dim
        self.loaded_files: List[str] = []
        self._warned_legacy_dependency = False

        if not self.root.exists():
            raise FileNotFoundError(f"Embedding root not found: {self.root}")

        files = sorted(self.root.glob("*_text_embeddings.pt"))
        if not files:
            files = sorted((self.root / "embeddings").rglob("*_text_embeddings.pt")) if (self.root / "embeddings").exists() else []

        for emb_file in files:
            payload = torch.load(emb_file, map_location="cpu")
            ids = payload.get("ids", [])
            embeddings = payload.get("embeddings")
            if embeddings is None or not isinstance(embeddings, torch.Tensor):
                continue

            embeddings = embeddings.float().cpu()
            if self._dim <= 0 and embeddings.ndim >= 2:
                self._dim = int(embeddings.shape[-1])

            loaded_any = False

            # New phrase-level payload.
            if embeddings.ndim == 2:
                offsets = payload.get("sample_offsets")
                phrase_mask = payload.get("phrase_mask")
                phrase_weights = payload.get("phrase_weights")
                token_target_indices = payload.get("token_target_indices")
                dependency_edges = payload.get("dependency_edges")
                dependency_sample_offsets = payload.get("dependency_sample_offsets")
                phrase_types_raw = payload.get("phrase_types", [])
                if not isinstance(phrase_mask, torch.Tensor) or int(phrase_mask.numel()) != int(embeddings.shape[0]):
                    phrase_mask = torch.ones((int(embeddings.shape[0]),), dtype=torch.bool)
                else:
                    phrase_mask = phrase_mask.reshape(-1).to(dtype=torch.bool).cpu()

                if not isinstance(phrase_weights, torch.Tensor) or int(phrase_weights.numel()) != int(embeddings.shape[0]):
                    phrase_weights = torch.ones((int(embeddings.shape[0]),), dtype=torch.float32)
                else:
                    phrase_weights = phrase_weights.reshape(-1).float().cpu()

                if (
                    not self._warned_legacy_dependency
                    and (dependency_edges is None or dependency_sample_offsets is None)
                ):
                    warnings.warn(
                        f"Legacy embedding file without dependency fields detected: {emb_file}. "
                        "Falling back to empty dependency graph.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    self._warned_legacy_dependency = True

                if isinstance(offsets, (list, tuple)) and len(offsets) == len(ids) + 1:
                    for idx, sample_id in enumerate(ids):
                        try:
                            start = int(offsets[idx])
                            end = int(offsets[idx + 1])
                        except Exception:
                            continue
                        if start < 0 or end <= start or end > int(embeddings.shape[0]):
                            continue

                        local_mask = phrase_mask[start:end]
                        if not local_mask.any():
                            continue

                        sid = str(sample_id)
                        seq = embeddings[start:end][local_mask]
                        w = phrase_weights[start:end][local_mask]
                        self._vectors[sid] = self._fit_dim(seq)
                        self._weights[sid] = self._fit_weight(w, int(seq.shape[0]))
                        if isinstance(token_target_indices, torch.Tensor) and token_target_indices.ndim == 1:
                            local_targets = token_target_indices[start:end]
                            if int(local_targets.numel()) == int(local_mask.numel()):
                                local_targets = local_targets[local_mask].to(dtype=torch.long).cpu().reshape(-1)
                                if int(local_targets.numel()) == int(seq.shape[0]):
                                    self._targets[sid] = local_targets
                        if (
                            isinstance(dependency_edges, torch.Tensor)
                            and dependency_edges.ndim == 2
                            and int(dependency_edges.shape[1]) == 2
                            and isinstance(dependency_sample_offsets, (list, tuple))
                            and len(dependency_sample_offsets) == len(ids) + 1
                        ):
                            try:
                                dep_start = int(dependency_sample_offsets[idx])
                                dep_end = int(dependency_sample_offsets[idx + 1])
                            except Exception:
                                dep_start, dep_end = 0, 0
                            if dep_end > dep_start >= 0 and dep_end <= int(dependency_edges.shape[0]):
                                local_edges = dependency_edges[dep_start:dep_end].to(dtype=torch.long).cpu()
                                if int(local_edges.numel()) > 0:
                                    self._dependency_edges[sid] = local_edges
                        # NEW: Store phrase types
                        if phrase_types_raw and len(phrase_types_raw) > start:
                            local_types = phrase_types_raw[start:end]
                            local_types = [t for i, t in enumerate(local_types) if i < len(local_mask) and local_mask[i]]
                            if local_types:
                                self._phrase_types[sid] = local_types
                        loaded_any = True

            # Legacy token-level payload fallback.
            token_masks = payload.get("token_masks")
            if (not loaded_any) and embeddings.ndim == 3 and isinstance(token_masks, torch.Tensor):
                token_masks = token_masks.to(dtype=torch.bool).cpu()
                if (
                    token_masks.ndim == 2
                    and int(embeddings.shape[0]) == int(token_masks.shape[0])
                    and int(embeddings.shape[1]) == int(token_masks.shape[1])
                ):
                    offsets = payload.get("sample_offsets")
                    if isinstance(offsets, (list, tuple)) and len(offsets) == len(ids) + 1:
                        for idx, sample_id in enumerate(ids):
                            try:
                                start = int(offsets[idx])
                                end = int(offsets[idx + 1])
                            except Exception:
                                continue
                            if start < 0 or end <= start or end > int(embeddings.shape[0]):
                                continue
                            sid = str(sample_id)
                            seq = self._collapse_tokens(embeddings[start:end], token_masks[start:end])
                            if int(seq.shape[0]) <= 0:
                                continue
                            self._vectors[sid] = self._fit_dim(seq)
                            self._weights[sid] = torch.ones((int(seq.shape[0]),), dtype=torch.float32)
                            self._targets[sid] = torch.arange(int(seq.shape[0]), dtype=torch.long)
                            loaded_any = True
                    elif len(ids) == int(embeddings.shape[0]):
                        for idx, sample_id in enumerate(ids):
                            sid = str(sample_id)
                            seq = self._collapse_tokens(embeddings[idx : idx + 1], token_masks[idx : idx + 1])
                            if int(seq.shape[0]) <= 0:
                                continue
                            self._vectors[sid] = self._fit_dim(seq)
                            self._weights[sid] = torch.ones((int(seq.shape[0]),), dtype=torch.float32)
                            self._targets[sid] = torch.arange(int(seq.shape[0]), dtype=torch.long)
                            loaded_any = True

            if loaded_any:
                self.loaded_files.append(str(emb_file))

        if self._dim <= 0:
            raise RuntimeError(
                "No valid embedding vectors loaded. "
                f"Checked root: {self.root}"
            )

    @property
    def dim(self) -> int:
        return int(self._dim)

    @staticmethod
    def _candidate_ids(sample_id: str) -> List[str]:
        """Generate fallback keys for robust sample_id -> embedding lookup."""
        raw = str(sample_id or "").strip()
        if not raw:
            return []

        candidates: List[str] = []

        def _push(value: str) -> None:
            value = str(value).strip()
            if value and value not in candidates:
                candidates.append(value)

        _push(raw)
        _push(raw.replace("\\", "/"))

        raw_path = Path(raw)
        _push(raw_path.name)
        _push(raw_path.stem)

        normalized_path = Path(raw.replace("\\", "/"))
        _push(normalized_path.name)
        _push(normalized_path.stem)

        for key in list(candidates):
            if "__" in key:
                _push(key.split("__", 1)[1])

        return candidates

    def _get_vectors_for_id(self, sample_id: str) -> Optional[torch.Tensor]:
        for key in self._candidate_ids(sample_id):
            vecs = self._vectors.get(key)
            if vecs is not None:
                return vecs
        return None

    def _get_weights_for_id(self, sample_id: str) -> Optional[torch.Tensor]:
        for key in self._candidate_ids(sample_id):
            weights = self._weights.get(key)
            if weights is not None:
                return weights
        return None

    def _get_targets_for_id(self, sample_id: str) -> Optional[torch.Tensor]:
        for key in self._candidate_ids(sample_id):
            targets = self._targets.get(key)
            if targets is not None:
                return targets
        return None

    def _get_phrase_types_for_id(self, sample_id: str) -> Optional[List[str]]:
        """Get phrase types for a sample ID."""
        for key in self._candidate_ids(sample_id):
            types = self._phrase_types.get(key)
            if types is not None:
                return types
        return None

    def _get_dependency_edges_for_id(self, sample_id: str) -> Optional[torch.Tensor]:
        """Get dependency edges [E, 2] for a sample ID."""
        for key in self._candidate_ids(sample_id):
            edges = self._dependency_edges.get(key)
            if edges is not None:
                return edges
        return None

    def _fit_dim(self, vecs: torch.Tensor) -> torch.Tensor:
        if vecs.ndim == 1:
            vecs = vecs.view(1, -1)
        if vecs.ndim != 2:
            raise ValueError(f"Expected phrase tensor [N, D], got shape={tuple(vecs.shape)}")

        if self._dim <= 0:
            return vecs.float().cpu()

        in_dim = int(vecs.shape[-1])
        if in_dim == self._dim:
            return vecs.float().cpu()
        if in_dim > self._dim:
            return vecs[:, : self._dim].float().cpu()

        pad = torch.zeros((int(vecs.shape[0]), self._dim - in_dim), dtype=torch.float32)
        return torch.cat((vecs.float().cpu(), pad), dim=-1)

    @staticmethod
    def _fit_weight(weights: torch.Tensor, expected_len: int) -> torch.Tensor:
        if weights.ndim != 1:
            weights = weights.reshape(-1)
        weights = weights.float().cpu()
        if int(weights.shape[0]) > expected_len:
            weights = weights[:expected_len]
        elif int(weights.shape[0]) < expected_len:
            pad = torch.ones((expected_len - int(weights.shape[0]),), dtype=torch.float32)
            weights = torch.cat((weights, pad), dim=0)

        weights = torch.clamp(weights, min=0.0)
        if not bool((weights > 0).any()):
            return torch.ones((expected_len,), dtype=torch.float32)
        return weights

    @staticmethod
    def _collapse_tokens(embeddings: torch.Tensor, token_masks: torch.Tensor) -> torch.Tensor:
        """Flatten [K, L, D] texts into one token sequence [T, D] using token mask."""
        if embeddings.ndim != 3 or token_masks.ndim != 2:
            return torch.zeros((0, int(embeddings.shape[-1]) if embeddings.ndim >= 1 else 0), dtype=torch.float32)

        flat_vecs = embeddings.view(-1, int(embeddings.shape[-1])).float()
        flat_mask = token_masks.reshape(-1).to(dtype=torch.bool)
        if not flat_mask.any():
            return torch.zeros((0, int(embeddings.shape[-1])), dtype=torch.float32)
        return flat_vecs[flat_mask]

    def get_batch(
        self,
        sample_ids: Iterable[str],
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        vectors, valid, token_mask, phrase_weight, _, _ = self.get_batch_with_targets(
            sample_ids=sample_ids,
            device=device,
            dtype=dtype,
        )
        return vectors, valid, token_mask, phrase_weight

    def get_batch_with_targets(
        self,
        sample_ids: Iterable[str],
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        ids = [str(x) for x in sample_ids]
        bsz = len(ids)
        max_tokens = 1
        for sid in ids:
            vecs = self._get_vectors_for_id(sid)
            if vecs is not None:
                max_tokens = max(max_tokens, int(vecs.shape[0]))

        vectors = torch.zeros((bsz, max_tokens, self._dim), dtype=torch.float32)
        token_mask = torch.zeros((bsz, max_tokens), dtype=torch.bool)
        phrase_weight = torch.zeros((bsz, max_tokens), dtype=torch.float32)
        token_target_idx = torch.full((bsz, max_tokens), -1, dtype=torch.long)
        dependency_adj = torch.zeros((bsz, max_tokens, max_tokens), dtype=torch.float32)
        valid = torch.zeros((bsz,), dtype=torch.bool)

        for i, sid in enumerate(ids):
            vecs = self._get_vectors_for_id(sid)
            if vecs is None:
                continue
            k = int(vecs.shape[0])
            if k <= 0:
                continue
            vectors[i, :k] = vecs
            token_mask[i, :k] = True
            weights = self._get_weights_for_id(sid)
            if weights is None:
                phrase_weight[i, :k] = 1.0
            else:
                phrase_weight[i, :k] = self._fit_weight(weights, k)

            targets = self._get_targets_for_id(sid)
            if targets is None:
                token_target_idx[i, :k] = torch.arange(k, dtype=torch.long)
            else:
                if targets.ndim != 1:
                    targets = targets.reshape(-1)
                targets = targets.to(dtype=torch.long).cpu()
                if int(targets.shape[0]) > k:
                    targets = targets[:k]
                elif int(targets.shape[0]) < k:
                    pad = torch.full((k - int(targets.shape[0]),), -1, dtype=torch.long)
                    targets = torch.cat((targets, pad), dim=0)
                token_target_idx[i, :k] = targets

            edges = self._get_dependency_edges_for_id(sid)
            if isinstance(edges, torch.Tensor) and edges.ndim == 2 and int(edges.shape[1]) == 2:
                edges = edges.to(dtype=torch.long)
                if int(edges.numel()) > 0:
                    child = edges[:, 0].clamp(min=0, max=max(0, k - 1))
                    head = edges[:, 1].clamp(min=0, max=max(0, k - 1))
                    dependency_adj[i, child, head] = 1.0
                    dependency_adj[i, head, child] = 1.0

            if k > 0:
                dependency_adj[i, :k, :k] += torch.eye(k, dtype=torch.float32)
            valid[i] = True

        return (
            vectors.to(device=device, dtype=dtype),
            valid.to(device=device),
            token_mask.to(device=device),
            phrase_weight.to(device=device, dtype=dtype),
            token_target_idx.to(device=device),
            dependency_adj.to(device=device, dtype=dtype),
        )

    def get_phrase_types_batch(self, sample_ids: Iterable[str], max_tokens: int = 0) -> List[List[str]]:
        """Get phrase types for a batch of samples.
        
        Args:
            sample_ids: Iterable of sample IDs
            max_tokens: Maximum tokens per sample (for padding)
            
        Returns:
            List of lists of phrase types
        """
        ids = [str(x) for x in sample_ids]
        batch_phrase_types: List[List[str]] = []
        
        if max_tokens <= 0:
            # Determine max tokens
            for sid in ids:
                types = self._get_phrase_types_for_id(sid)
                if types is not None:
                    max_tokens = max(max_tokens, len(types))
        
        for sid in ids:
            types = self._get_phrase_types_for_id(sid)
            if types is None:
                # Default to "object" if not available
                types = ["object"]
            
            # Pad or trim to max_tokens
            if len(types) < max_tokens:
                types = types + ["object"] * (max_tokens - len(types))
            elif len(types) > max_tokens:
                types = types[:max_tokens]
            
            batch_phrase_types.append(types)
        
        return batch_phrase_types
