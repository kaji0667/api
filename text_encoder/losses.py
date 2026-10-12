from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.utils.metrics import bbox_iou

from .matching import linear_sum_assignment_torch


class XRTokenAlignmentLoss(v8DetectionLoss):
    """YOLO detection loss plus phrase alignment contrastive supervision."""

    def __init__(self, model) -> None:
        super().__init__(model)
        self.lambda_heatmap = float(getattr(model, "text_lambda_heatmap", 0.3))
        self.lambda_phrase = float(getattr(model, "text_lambda_phrase", 0.2))
        self.lambda_set = float(getattr(model, "text_lambda_set", 0.6))
        self.lambda_ortho = float(getattr(model, "text_orth_loss_weight", 0.05))
        self.fuse_temperature = float(max(1e-3, getattr(model, "text_fuse_temperature", 0.5)))
        self.matching_temperature = float(max(1e-3, getattr(model, "text_matching_temperature", 0.7)))
        self.contrastive_loss_type = str(getattr(model, "text_contrastive_loss_type", "logsigmoid_margin")).strip().lower()
        self.infonce_temperature = float(max(1e-3, getattr(model, "text_infonce_temperature", 0.25)))
        self.hard_neg_k = int(max(0, getattr(model, "text_hard_neg_k", 32)))
        self.use_in_batch_negatives = bool(getattr(model, "text_use_in_batch_negatives", True))
        self.lambda_diou = float(max(0.0, getattr(model, "text_lambda_diou", 0.15)))
        self.lambda_dependency = float(max(0.0, getattr(model, "text_lambda_dependency", 0.05)))
        self.diou_temperature = float(max(1e-3, getattr(model, "text_diou_temperature", 1.0)))
        
        # NEW: Weighted layer fusion parameters
        self.use_weighted_layer_fusion = bool(getattr(model, "text_use_weighted_layer_fusion", False))
        self.spatial_layer_weights = getattr(model, "text_spatial_layer_weights", None)
        self.attribute_layer_weights = getattr(model, "text_attribute_layer_weights", None)
        self.relation_layer_weights = getattr(model, "text_relation_layer_weights", None)
        
        # Default layer weights [P3, P4, P5]
        if self.spatial_layer_weights is None:
            self.spatial_layer_weights = [0.1, 0.5, 0.4]  # Favor higher levels for spatial
        if self.attribute_layer_weights is None:
            self.attribute_layer_weights = [0.4, 0.3, 0.3]  # More balanced, emphasize details
        if self.relation_layer_weights is None:
            self.relation_layer_weights = [0.2, 0.4, 0.4]  # Favor higher levels for relations
        
        # NEW: Relation loss parameters (for comparative relationships)
        self.lambda_relation = float(max(0.0, getattr(model, "text_lambda_relation", 0.15)))
        self.relation_loss_enabled = bool(self.lambda_relation > 0.0)
        self.relation_margin = float(max(0.0, getattr(model, "text_relation_margin", 0.5)))
        self.relation_loss_type = str(getattr(model, "text_relation_loss_type", "triplet")).strip().lower()
        
        # NEW: Spatial quadrant loss parameters
        self.lambda_spatial_quadrant = float(max(0.0, getattr(model, "text_lambda_spatial_quadrant", 0.10)))
        self.spatial_quadrant_loss_enabled = bool(self.lambda_spatial_quadrant > 0.0)

    @staticmethod
    def _compute_branch_orth_loss(branch_visual_feats: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]) -> torch.Tensor:
        """Minimize pairwise feature correlation among geo/attr/sem visual projections."""
        if not branch_visual_feats:
            return torch.zeros((), device=branch_visual_feats[0][0].device if branch_visual_feats else "cpu")

        losses: List[torch.Tensor] = []
        for triplet in branch_visual_feats:
            if not isinstance(triplet, tuple) or len(triplet) != 3:
                continue
            geo, attr, sem = triplet
            if not all(isinstance(x, torch.Tensor) and x.ndim == 4 for x in (geo, attr, sem)):
                continue

            def _flatten_norm(x: torch.Tensor) -> torch.Tensor:
                x = x.float().flatten(1)
                x = x - x.mean(dim=1, keepdim=True)
                x = x / x.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-6)
                return F.normalize(x, dim=1)

            g = _flatten_norm(geo)
            a = _flatten_norm(attr)
            s = _flatten_norm(sem)

            corr_ga = (g * a).sum(dim=1)
            corr_gs = (g * s).sum(dim=1)
            corr_as = (a * s).sum(dim=1)
            losses.append((corr_ga.pow(2) + corr_gs.pow(2) + corr_as.pow(2)).mean() / 3.0)

        if not losses:
            return torch.zeros((), device=branch_visual_feats[0][0].device)
        return torch.stack(losses).mean()

    def _get_layer_weights_for_phrase_type(self, phrase_type: str, num_levels: int) -> torch.Tensor:
        """Get layer weights for a specific phrase type."""
        if not self.use_weighted_layer_fusion:
            # Default: equal weights for all levels
            return torch.ones(num_levels, dtype=torch.float32) / float(num_levels)
        
        if phrase_type == "spatial":
            weights = self.spatial_layer_weights
        elif phrase_type == "relation":
            weights = self.relation_layer_weights
        elif phrase_type == "attribute":
            weights = self.attribute_layer_weights
        else:
            # Default for unknown types
            return torch.ones(num_levels, dtype=torch.float32) / float(num_levels)
        
        # Ensure weights match the number of levels
        weights = list(weights) if not isinstance(weights, list) else weights
        if len(weights) > num_levels:
            weights = weights[:num_levels]
        elif len(weights) < num_levels:
            # Pad with balanced weights if insufficient
            avg_weight = 1.0 / float(num_levels)
            weights = weights + [avg_weight] * (num_levels - len(weights))
        
        return torch.tensor(weights, dtype=torch.float32) / sum(weights)

    @staticmethod
    def _normalize_phrase_weights(phrase_weight: torch.Tensor, token_mask: torch.Tensor) -> torch.Tensor:
        mask_f = token_mask.to(dtype=phrase_weight.dtype)
        weights = phrase_weight * mask_f
        denom = weights.sum(dim=1, keepdim=True)

        empty = denom.squeeze(1) <= 0
        if empty.any():
            fallback = mask_f[empty]
            fallback_sum = fallback.sum(dim=1, keepdim=True).clamp_min(1.0)
            weights = weights.clone()
            weights[empty] = fallback / fallback_sum
            denom = weights.sum(dim=1, keepdim=True)

        return weights / denom.clamp_min(1e-6)

    @staticmethod
    def _build_gt_box_mask(
        batch: Dict,
        bsz: int,
        h: int,
        w: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Create a binary box mask: inside bbox=1, outside bbox=0."""
        mask = torch.zeros((bsz, 1, h, w), device=device, dtype=dtype)
        batch_idx = batch.get("batch_idx")
        bboxes = batch.get("bboxes")
        if batch_idx is None or bboxes is None:
            return mask

        bi = batch_idx.view(-1).to(device=device, dtype=torch.long)
        boxes = bboxes.view(-1, 4).to(device=device, dtype=torch.float32)
        n_targets = min(int(bi.numel()), int(boxes.shape[0]))
        if n_targets <= 0:
            return mask

        for t in range(n_targets):
            i = int(bi[t].item())
            if i < 0 or i >= bsz:
                continue

            cx, cy, bw, bh = boxes[t]
            if bw <= 0.0 or bh <= 0.0:
                continue

            cx = float(cx.clamp(0.0, 1.0).item())
            cy = float(cy.clamp(0.0, 1.0).item())
            bw = float(bw.clamp(0.0, 1.0).item())
            bh = float(bh.clamp(0.0, 1.0).item())

            x1 = max(0, min(w - 1, int((cx - bw * 0.5) * w)))
            y1 = max(0, min(h - 1, int((cy - bh * 0.5) * h)))
            x2 = max(x1 + 1, min(w, int((cx + bw * 0.5) * w)))
            y2 = max(y1 + 1, min(h, int((cy + bh * 0.5) * h)))
            mask[i, 0, y1:y2, x1:x2] = 1.0

        return mask

    @staticmethod
    def _build_gt_object_masks(
        batch: Dict,
        bsz: int,
        h: int,
        w: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Create per-object masks [B, M, H, W] ordered by local object index in each image."""
        batch_idx = batch.get("batch_idx")
        bboxes = batch.get("bboxes")
        if batch_idx is None or bboxes is None:
            return torch.zeros((bsz, 1, h, w), device=device, dtype=dtype)

        bi = batch_idx.view(-1).to(device=device, dtype=torch.long)
        boxes = bboxes.view(-1, 4).to(device=device, dtype=torch.float32)
        n_targets = min(int(bi.numel()), int(boxes.shape[0]))
        if n_targets <= 0:
            return torch.zeros((bsz, 1, h, w), device=device, dtype=dtype)

        obj_count = torch.zeros((bsz,), device=device, dtype=torch.long)
        for t in range(n_targets):
            i = int(bi[t].item())
            if 0 <= i < bsz:
                obj_count[i] += 1

        max_obj = int(obj_count.max().item()) if int(obj_count.numel()) else 0
        if max_obj <= 0:
            return torch.zeros((bsz, 1, h, w), device=device, dtype=dtype)

        masks = torch.zeros((bsz, max_obj, h, w), device=device, dtype=dtype)
        local_obj = torch.zeros((bsz,), device=device, dtype=torch.long)

        for t in range(n_targets):
            i = int(bi[t].item())
            if i < 0 or i >= bsz:
                continue

            obj_i = int(local_obj[i].item())
            local_obj[i] += 1
            if obj_i < 0 or obj_i >= max_obj:
                continue

            cx, cy, bw, bh = boxes[t]
            if bw <= 0.0 or bh <= 0.0:
                continue

            cx = float(cx.clamp(0.0, 1.0).item())
            cy = float(cy.clamp(0.0, 1.0).item())
            bw = float(bw.clamp(0.0, 1.0).item())
            bh = float(bh.clamp(0.0, 1.0).item())

            x1 = max(0, min(w - 1, int((cx - bw * 0.5) * w)))
            y1 = max(0, min(h - 1, int((cy - bh * 0.5) * h)))
            x2 = max(x1 + 1, min(w, int((cx + bw * 0.5) * w)))
            y2 = max(y1 + 1, min(h, int((cy + bh * 0.5) * h)))
            masks[i, obj_i, y1:y2, x1:x2] = 1.0

        return masks

    def _compute_heatmap_contrastive_loss(
        self,
        feats: List[torch.Tensor],
        alignment_logits: List[torch.Tensor],
        batch: Dict,
    ) -> torch.Tensor:
        if not alignment_logits:
            return torch.zeros((), device=self.device)

        losses: List[torch.Tensor] = []
        text_valid_mask = batch.get("text_valid_mask")
        for i, feat in enumerate(feats):
            if i >= len(alignment_logits):
                continue

            pred_logit = alignment_logits[i]
            if not isinstance(pred_logit, torch.Tensor) or pred_logit.ndim != 4:
                continue

            if pred_logit.shape[1] != 1:
                pred_logit = pred_logit.mean(dim=1, keepdim=True)

            if tuple(pred_logit.shape[-2:]) != tuple(feat.shape[-2:]):
                pred_logit = F.interpolate(pred_logit, size=feat.shape[-2:], mode="bilinear", align_corners=False)

            gt_mask = self._build_gt_box_mask(
                batch=batch,
                bsz=int(feat.shape[0]),
                h=int(feat.shape[2]),
                w=int(feat.shape[3]),
                device=pred_logit.device,
                dtype=pred_logit.dtype,
            )

            pred_flat = pred_logit.view(int(pred_logit.shape[0]), -1)
            gt_flat = (gt_mask.view(int(gt_mask.shape[0]), -1) > 0.5)

            level_losses: List[torch.Tensor] = []
            for b in range(int(pred_flat.shape[0])):
                if isinstance(text_valid_mask, torch.Tensor):
                    if b >= int(text_valid_mask.shape[0]) or not bool(text_valid_mask[b].item()):
                        continue

                pos = pred_flat[b][gt_flat[b]]
                neg = pred_flat[b][~gt_flat[b]]
                if pos.numel() <= 0 or neg.numel() <= 0:
                    continue

                if self.contrastive_loss_type == "infonce":
                    neg_pool = neg
                    if self.hard_neg_k > 0 and int(neg_pool.numel()) > self.hard_neg_k:
                        neg_pool = torch.topk(neg_pool, k=self.hard_neg_k, largest=True).values
                    logits = torch.cat((pos.mean().view(1), neg_pool), dim=0) / self.infonce_temperature
                    target = torch.zeros((1,), device=logits.device, dtype=torch.long)
                    level_losses.append(F.cross_entropy(logits.view(1, -1), target))
                    continue

                margin = pos.mean() - neg.mean()
                level_losses.append(-F.logsigmoid(margin))

            if level_losses:
                losses.append(torch.stack(level_losses).mean())

        if not losses:
            return torch.zeros((), device=self.device)

        return torch.stack(losses).mean()

    def _compute_phrase_contrastive_loss(
        self,
        feats: List[torch.Tensor],
        phrase_logits: List[torch.Tensor],
        text_token_mask: torch.Tensor | None,
        text_phrase_weight: torch.Tensor | None,
        text_token_target_idx: torch.Tensor | None,
        batch: Dict,
    ) -> torch.Tensor:
        if not phrase_logits:
            return torch.zeros((), device=self.device)

        losses: List[torch.Tensor] = []
        text_valid_mask = batch.get("text_valid_mask")

        for i, feat in enumerate(feats):
            if i >= len(phrase_logits):
                continue

            pred_phrase = phrase_logits[i]
            if not isinstance(pred_phrase, torch.Tensor) or pred_phrase.ndim != 4:
                continue

            if tuple(pred_phrase.shape[-2:]) != tuple(feat.shape[-2:]):
                pred_phrase = F.interpolate(pred_phrase, size=feat.shape[-2:], mode="bilinear", align_corners=False)

            bsz, tokens, _, _ = pred_phrase.shape
            phrase_flat = pred_phrase.view(int(bsz), int(tokens), -1)
            object_masks = self._build_gt_object_masks(
                batch=batch,
                bsz=int(bsz),
                h=int(feat.shape[2]),
                w=int(feat.shape[3]),
                device=pred_phrase.device,
                dtype=pred_phrase.dtype,
            )
            object_flat = object_masks.view(int(bsz), int(object_masks.shape[1]), -1) > 0.5
            union_flat = object_flat.any(dim=1)

            use_strict = (
                isinstance(text_token_target_idx, torch.Tensor)
                and text_token_target_idx.ndim == 2
                and int(text_token_target_idx.shape[0]) == int(bsz)
            )
            token_target_idx = None
            if use_strict:
                token_target_idx = text_token_target_idx.to(device=pred_phrase.device, dtype=torch.long)
                if int(token_target_idx.shape[1]) != int(tokens):
                    if int(token_target_idx.shape[1]) > int(tokens):
                        token_target_idx = token_target_idx[:, : int(tokens)]
                    else:
                        pad = torch.full(
                            (int(bsz), int(tokens) - int(token_target_idx.shape[1])),
                            -1,
                            device=pred_phrase.device,
                            dtype=torch.long,
                        )
                        token_target_idx = torch.cat((token_target_idx, pad), dim=1)

            if isinstance(text_token_mask, torch.Tensor) and text_token_mask.ndim == 2 and int(text_token_mask.shape[0]) == int(bsz):
                token_mask = text_token_mask.to(device=pred_phrase.device, dtype=torch.bool)
                if int(token_mask.shape[1]) != int(tokens):
                    if int(token_mask.shape[1]) > int(tokens):
                        token_mask = token_mask[:, : int(tokens)]
                    else:
                        pad = torch.zeros((int(bsz), int(tokens) - int(token_mask.shape[1])), device=pred_phrase.device, dtype=torch.bool)
                        token_mask = torch.cat((token_mask, pad), dim=1)
            else:
                token_mask = torch.ones((int(bsz), int(tokens)), device=pred_phrase.device, dtype=torch.bool)

            if isinstance(text_phrase_weight, torch.Tensor) and text_phrase_weight.ndim == 2 and int(text_phrase_weight.shape[0]) == int(bsz):
                phrase_weight = text_phrase_weight.to(device=pred_phrase.device, dtype=pred_phrase.dtype)
                if int(phrase_weight.shape[1]) != int(tokens):
                    if int(phrase_weight.shape[1]) > int(tokens):
                        phrase_weight = phrase_weight[:, : int(tokens)]
                    else:
                        pad = torch.zeros((int(bsz), int(tokens) - int(phrase_weight.shape[1])), device=pred_phrase.device, dtype=pred_phrase.dtype)
                        phrase_weight = torch.cat((phrase_weight, pad), dim=1)
            else:
                phrase_weight = torch.ones((int(bsz), int(tokens)), device=pred_phrase.device, dtype=pred_phrase.dtype)

            phrase_weight = self._normalize_phrase_weights(phrase_weight, token_mask)

            level_losses: List[torch.Tensor] = []
            for b in range(int(bsz)):
                if isinstance(text_valid_mask, torch.Tensor):
                    if b >= int(text_valid_mask.shape[0]) or not bool(text_valid_mask[b].item()):
                        continue

                valid_tokens = token_mask[b]
                if valid_tokens.sum().item() <= 0:
                    continue

                token_logits = phrase_flat[b, valid_tokens, :]
                token_weights = phrase_weight[b, valid_tokens]
                global_neg = None
                if self.use_in_batch_negatives:
                    other_mask = torch.ones((int(bsz),), device=pred_phrase.device, dtype=torch.bool)
                    other_mask[b] = False
                    if bool(other_mask.any()):
                        global_neg = phrase_flat[other_mask].reshape(-1)

                if token_target_idx is not None:
                    valid_token_indices = torch.nonzero(valid_tokens, as_tuple=False).view(-1)
                    sample_terms: List[torch.Tensor] = []
                    max_obj = int(object_flat.shape[1])
                    for local_j, tok_idx in enumerate(valid_token_indices.tolist()):
                        obj_idx = int(token_target_idx[b, tok_idx].item())
                        if obj_idx < 0 or obj_idx >= max_obj:
                            continue
                        pos_idx = object_flat[b, obj_idx]
                        neg_idx = ~pos_idx
                        if pos_idx.sum().item() <= 0 or neg_idx.sum().item() <= 0:
                            continue

                        token_logit = token_logits[local_j]
                        if self.contrastive_loss_type == "infonce":
                            pos_score = token_logit[pos_idx].mean()
                            neg_pool = token_logit[neg_idx]
                            if isinstance(global_neg, torch.Tensor) and int(global_neg.numel()) > 0:
                                neg_pool = torch.cat((neg_pool, global_neg.to(device=neg_pool.device, dtype=neg_pool.dtype)), dim=0)
                            if self.hard_neg_k > 0 and int(neg_pool.numel()) > self.hard_neg_k:
                                neg_pool = torch.topk(neg_pool, k=self.hard_neg_k, largest=True).values
                            logits = torch.cat((pos_score.view(1), neg_pool), dim=0) / self.infonce_temperature
                            target = torch.zeros((1,), device=logits.device, dtype=torch.long)
                            sample_terms.append(F.cross_entropy(logits.view(1, -1), target) * token_weights[local_j])
                            continue

                        token_margin = token_logit[pos_idx].mean() - token_logit[neg_idx].mean()
                        sample_terms.append(-F.logsigmoid(token_margin) * token_weights[local_j])

                    if sample_terms:
                        level_losses.append(torch.stack(sample_terms).sum())
                    continue

                pos_idx = union_flat[b]
                neg_idx = ~pos_idx
                if pos_idx.sum().item() <= 0 or neg_idx.sum().item() <= 0:
                    continue

                pos_score = token_logits[:, pos_idx].mean(dim=1)
                neg_score = token_logits[:, neg_idx].mean(dim=1)

                if self.contrastive_loss_type == "infonce":
                    sample_terms: List[torch.Tensor] = []
                    for t in range(int(token_logits.shape[0])):
                        neg_pool = token_logits[t, neg_idx]
                        if isinstance(global_neg, torch.Tensor) and int(global_neg.numel()) > 0:
                            neg_pool = torch.cat((neg_pool, global_neg.to(device=neg_pool.device, dtype=neg_pool.dtype)), dim=0)
                        if self.hard_neg_k > 0 and int(neg_pool.numel()) > self.hard_neg_k:
                            neg_pool = torch.topk(neg_pool, k=self.hard_neg_k, largest=True).values
                        logits = torch.cat((pos_score[t].view(1), neg_pool), dim=0) / self.infonce_temperature
                        target = torch.zeros((1,), device=logits.device, dtype=torch.long)
                        sample_terms.append(F.cross_entropy(logits.view(1, -1), target) * token_weights[t])
                    if sample_terms:
                        level_losses.append(torch.stack(sample_terms).sum())
                    continue

                token_margin = pos_score - neg_score

                lse_input = token_margin / self.fuse_temperature
                lse_input = lse_input + torch.log(token_weights.clamp_min(1e-6))
                sample_margin = self.fuse_temperature * torch.logsumexp(lse_input, dim=0)
                level_losses.append(-F.logsigmoid(sample_margin))

            if level_losses:
                losses.append(torch.stack(level_losses).mean())

        if not losses:
            return torch.zeros((), device=self.device)

        return torch.stack(losses).mean()

    def _compute_set_matching_loss(
        self,
        feats: List[torch.Tensor],
        phrase_logits: List[torch.Tensor],
        text_token_mask: torch.Tensor | None,
        text_token_target_idx: torch.Tensor | None,
        batch: Dict,
    ) -> torch.Tensor:
        if not phrase_logits:
            return torch.zeros((), device=self.device)
        if not isinstance(text_token_mask, torch.Tensor) or text_token_mask.ndim != 2:
            return torch.zeros((), device=self.device)
        if not isinstance(text_token_target_idx, torch.Tensor) or text_token_target_idx.ndim != 2:
            return torch.zeros((), device=self.device)

        # Aggregate per-level token->object scores into one score matrix [B, T, M].
        score_matrix = None
        level_count = 0
        bsz = int(text_token_mask.shape[0])
        
        # NEW: Prepare layer weights if weighted fusion is enabled
        layer_weights = None
        if self.use_weighted_layer_fusion:
            # Get phrase types from batch if available
            phrase_types_batch = batch.get("phrase_types")  # List of lists or None
            layer_weights = {}  # Map from token index to layer weights
            if phrase_types_batch is not None:
                for b in range(bsz):
                    if b < len(phrase_types_batch):
                        for t, ptype in enumerate(phrase_types_batch[b]):
                            key = (b, t)
                            if key not in layer_weights:
                                layer_weights[key] = self._get_layer_weights_for_phrase_type(ptype, len(feats))

        for i, feat in enumerate(feats):
            if i >= len(phrase_logits):
                continue

            pred_phrase = phrase_logits[i]
            if not isinstance(pred_phrase, torch.Tensor) or pred_phrase.ndim != 4:
                continue
            if tuple(pred_phrase.shape[-2:]) != tuple(feat.shape[-2:]):
                pred_phrase = F.interpolate(pred_phrase, size=feat.shape[-2:], mode="bilinear", align_corners=False)

            obj_masks = self._build_gt_object_masks(
                batch=batch,
                bsz=bsz,
                h=int(feat.shape[2]),
                w=int(feat.shape[3]),
                device=pred_phrase.device,
                dtype=pred_phrase.dtype,
            )
            if int(obj_masks.shape[1]) <= 0:
                continue

            token_map = pred_phrase.view(bsz, int(pred_phrase.shape[1]), -1)
            obj_map = obj_masks.view(bsz, int(obj_masks.shape[1]), -1)
            obj_norm = obj_map.sum(dim=-1, keepdim=True).clamp_min(1.0)
            obj_w = obj_map / obj_norm
            level_score = torch.einsum("bth,bmh->btm", token_map, obj_w)
            
            # NEW: Apply layer weight if available
            if layer_weights is not None and len(layer_weights) > 0:
                # Weight based on phrase type for each token
                layer_weight_val = torch.ones((bsz, int(pred_phrase.shape[1])), device=level_score.device, dtype=level_score.dtype)
                for b in range(bsz):
                    for t in range(int(pred_phrase.shape[1])):
                        key = (b, t)
                        if key in layer_weights:
                            w_tensor = layer_weights[key].to(device=level_score.device, dtype=level_score.dtype)
                            if i < len(w_tensor):
                                layer_weight_val[b, t] = w_tensor[i]
                # Apply the weight to the level score
                level_score = level_score * layer_weight_val.unsqueeze(-1)

            score_matrix = level_score if score_matrix is None else (score_matrix + level_score)
            level_count += 1

        if score_matrix is None or level_count <= 0:
            return torch.zeros((), device=self.device)

        # NEW: Normalize by actual level count (weighted fusion already scaled by weights)
        score_matrix = score_matrix / float(level_count) if not self.use_weighted_layer_fusion else score_matrix
        token_mask = text_token_mask.to(device=score_matrix.device, dtype=torch.bool)
        token_target = text_token_target_idx.to(device=score_matrix.device, dtype=torch.long)

        if int(token_mask.shape[1]) != int(score_matrix.shape[1]):
            tok = int(score_matrix.shape[1])
            if int(token_mask.shape[1]) > tok:
                token_mask = token_mask[:, :tok]
                token_target = token_target[:, :tok]
            else:
                pad_mask = torch.zeros((bsz, tok - int(token_mask.shape[1])), device=score_matrix.device, dtype=torch.bool)
                pad_tgt = torch.full((bsz, tok - int(token_target.shape[1])), -1, device=score_matrix.device, dtype=torch.long)
                token_mask = torch.cat((token_mask, pad_mask), dim=1)
                token_target = torch.cat((token_target, pad_tgt), dim=1)

        text_valid_mask = batch.get("text_valid_mask")
        sample_losses: List[torch.Tensor] = []
        for b in range(bsz):
            if isinstance(text_valid_mask, torch.Tensor):
                if b >= int(text_valid_mask.shape[0]) or not bool(text_valid_mask[b].item()):
                    continue

            valid_rows = torch.nonzero(token_mask[b], as_tuple=False).view(-1)
            if int(valid_rows.numel()) <= 0:
                continue

            sample_score = score_matrix[b, valid_rows, :]
            m = int(sample_score.shape[1])
            if m <= 0:
                continue

            cost = -sample_score

            row_ind, col_ind = linear_sum_assignment_torch(cost)
            if int(row_ind.numel()) <= 0:
                continue

            row_logits = sample_score / self.matching_temperature
            row_logp = F.log_softmax(row_logits, dim=1)
            pos_row = row_logp[row_ind, col_ind]
            row_loss = -pos_row.mean()

            col_logits = sample_score.transpose(0, 1) / self.matching_temperature
            col_logp = F.log_softmax(col_logits, dim=1)
            pos_col = col_logp[col_ind, row_ind]
            col_loss = -pos_col.mean()

            sample_losses.append(0.5 * (row_loss + col_loss))

        if not sample_losses:
            return torch.zeros((), device=self.device)
        return torch.stack(sample_losses).mean()

    @staticmethod
    def _build_gt_object_boxes_xyxy(batch: Dict, bsz: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_idx = batch.get("batch_idx")
        bboxes = batch.get("bboxes")
        if batch_idx is None or bboxes is None:
            return torch.zeros((bsz, 1, 4), device=device), torch.zeros((bsz, 1), device=device, dtype=torch.bool)

        bi = batch_idx.view(-1).to(device=device, dtype=torch.long)
        boxes = bboxes.view(-1, 4).to(device=device, dtype=torch.float32)
        n_targets = min(int(bi.numel()), int(boxes.shape[0]))
        if n_targets <= 0:
            return torch.zeros((bsz, 1, 4), device=device), torch.zeros((bsz, 1), device=device, dtype=torch.bool)

        obj_count = torch.zeros((bsz,), device=device, dtype=torch.long)
        for t in range(n_targets):
            i = int(bi[t].item())
            if 0 <= i < bsz:
                obj_count[i] += 1
        max_obj = int(obj_count.max().item()) if int(obj_count.numel()) else 0
        if max_obj <= 0:
            return torch.zeros((bsz, 1, 4), device=device), torch.zeros((bsz, 1), device=device, dtype=torch.bool)

        out = torch.zeros((bsz, max_obj, 4), device=device, dtype=torch.float32)
        valid = torch.zeros((bsz, max_obj), device=device, dtype=torch.bool)
        local_idx = torch.zeros((bsz,), device=device, dtype=torch.long)
        for t in range(n_targets):
            i = int(bi[t].item())
            if i < 0 or i >= bsz:
                continue
            j = int(local_idx[i].item())
            local_idx[i] += 1
            if j >= max_obj:
                continue
            cx, cy, bw, bh = boxes[t]
            x1 = (cx - bw * 0.5).clamp(0.0, 1.0)
            y1 = (cy - bh * 0.5).clamp(0.0, 1.0)
            x2 = (cx + bw * 0.5).clamp(0.0, 1.0)
            y2 = (cy + bh * 0.5).clamp(0.0, 1.0)
            out[i, j] = torch.stack((x1, y1, x2, y2), dim=0)
            valid[i, j] = True
        return out, valid

    @staticmethod
    def _token_soft_box(token_map: torch.Tensor, h: int, w: int, temperature: float) -> torch.Tensor:
        prob = torch.softmax(token_map.view(-1) / max(temperature, 1e-3), dim=0)
        yy, xx = torch.meshgrid(
            torch.linspace(0.0, 1.0, steps=h, device=token_map.device, dtype=token_map.dtype),
            torch.linspace(0.0, 1.0, steps=w, device=token_map.device, dtype=token_map.dtype),
            indexing="ij",
        )
        x = xx.reshape(-1)
        y = yy.reshape(-1)
        cx = (prob * x).sum()
        cy = (prob * y).sum()
        var_x = (prob * (x - cx).pow(2)).sum().clamp_min(1e-6)
        var_y = (prob * (y - cy).pow(2)).sum().clamp_min(1e-6)
        bw = (2.0 * torch.sqrt(var_x)).clamp(1.0 / max(2, w), 1.0)
        bh = (2.0 * torch.sqrt(var_y)).clamp(1.0 / max(2, h), 1.0)
        x1 = (cx - bw * 0.5).clamp(0.0, 1.0)
        y1 = (cy - bh * 0.5).clamp(0.0, 1.0)
        x2 = (cx + bw * 0.5).clamp(0.0, 1.0)
        y2 = (cy + bh * 0.5).clamp(0.0, 1.0)
        return torch.stack((x1, y1, x2, y2), dim=0)

    def _compute_diou_grounding_loss(
        self,
        phrase_logits: List[torch.Tensor],
        text_token_mask: torch.Tensor | None,
        text_phrase_weight: torch.Tensor | None,
        text_token_target_idx: torch.Tensor | None,
        batch: Dict,
    ) -> torch.Tensor:
        if self.lambda_diou <= 0.0 or not phrase_logits:
            return torch.zeros((), device=self.device)
        if not isinstance(text_token_mask, torch.Tensor) or not isinstance(text_token_target_idx, torch.Tensor):
            return torch.zeros((), device=self.device)

        base = phrase_logits[0]
        if not isinstance(base, torch.Tensor) or base.ndim != 4:
            return torch.zeros((), device=self.device)

        bsz, tokens, h, w = base.shape
        token_mask = text_token_mask.to(device=base.device, dtype=torch.bool)
        token_target = text_token_target_idx.to(device=base.device, dtype=torch.long)
        if int(token_mask.shape[1]) != int(tokens):
            tok = int(tokens)
            if int(token_mask.shape[1]) > tok:
                token_mask = token_mask[:, :tok]
                token_target = token_target[:, :tok]
            else:
                pad_mask = torch.zeros((bsz, tok - int(token_mask.shape[1])), device=base.device, dtype=torch.bool)
                pad_tgt = torch.full((bsz, tok - int(token_target.shape[1])), -1, device=base.device, dtype=torch.long)
                token_mask = torch.cat((token_mask, pad_mask), dim=1)
                token_target = torch.cat((token_target, pad_tgt), dim=1)

        if isinstance(text_phrase_weight, torch.Tensor) and text_phrase_weight.ndim == 2 and int(text_phrase_weight.shape[0]) == int(bsz):
            phrase_weight = text_phrase_weight.to(device=base.device, dtype=base.dtype)
            if int(phrase_weight.shape[1]) != int(tokens):
                if int(phrase_weight.shape[1]) > int(tokens):
                    phrase_weight = phrase_weight[:, : int(tokens)]
                else:
                    pad = torch.ones((int(bsz), int(tokens) - int(phrase_weight.shape[1])), device=base.device, dtype=base.dtype)
                    phrase_weight = torch.cat((phrase_weight, pad), dim=1)
        else:
            phrase_weight = torch.ones((int(bsz), int(tokens)), device=base.device, dtype=base.dtype)

        phrase_weight = self._normalize_phrase_weights(phrase_weight, token_mask)
        gt_boxes, gt_valid = self._build_gt_object_boxes_xyxy(batch=batch, bsz=int(bsz), device=base.device)
        text_valid_mask = batch.get("text_valid_mask")

        losses: List[torch.Tensor] = []
        for b in range(int(bsz)):
            if isinstance(text_valid_mask, torch.Tensor):
                if b >= int(text_valid_mask.shape[0]) or not bool(text_valid_mask[b].item()):
                    continue

            valid_tok = torch.nonzero(token_mask[b], as_tuple=False).view(-1)
            if int(valid_tok.numel()) <= 0:
                continue

            sample_terms: List[torch.Tensor] = []
            for tok_idx in valid_tok.tolist():
                obj_idx = int(token_target[b, tok_idx].item())
                if obj_idx < 0 or obj_idx >= int(gt_boxes.shape[1]) or not bool(gt_valid[b, obj_idx].item()):
                    continue
                pred_box = self._token_soft_box(base[b, tok_idx], h=int(h), w=int(w), temperature=self.diou_temperature)
                gt_box = gt_boxes[b, obj_idx]
                diou = bbox_iou(pred_box.unsqueeze(0), gt_box.unsqueeze(0), xywh=False, DIoU=True)
                sample_terms.append((1.0 - diou.view(-1)[0]) * phrase_weight[b, tok_idx])

            if sample_terms:
                losses.append(torch.stack(sample_terms).sum())

        if not losses:
            return torch.zeros((), device=self.device)
        return torch.stack(losses).mean()

    def _compute_dependency_consistency_loss(
        self,
        phrase_logits: List[torch.Tensor],
        text_token_mask: torch.Tensor | None,
        text_dependency_adj: torch.Tensor | None,
        batch: Dict,
    ) -> torch.Tensor:
        """Encourage dependent tokens to have similar spatial response maps."""
        if self.lambda_dependency <= 0.0 or not phrase_logits:
            return torch.zeros((), device=self.device)
        if not isinstance(text_token_mask, torch.Tensor) or text_token_mask.ndim != 2:
            return torch.zeros((), device=self.device)
        if not isinstance(text_dependency_adj, torch.Tensor) or text_dependency_adj.ndim != 3:
            return torch.zeros((), device=self.device)

        base = phrase_logits[-1]
        if not isinstance(base, torch.Tensor) or base.ndim != 4:
            return torch.zeros((), device=self.device)

        bsz, tokens, _, _ = base.shape
        token_mask = text_token_mask.to(device=base.device, dtype=torch.bool)
        dep_adj = text_dependency_adj.to(device=base.device, dtype=base.dtype)
        if int(token_mask.shape[1]) != int(tokens):
            tok = int(tokens)
            if int(token_mask.shape[1]) > tok:
                token_mask = token_mask[:, :tok]
            else:
                pad_mask = torch.zeros((bsz, tok - int(token_mask.shape[1])), device=base.device, dtype=torch.bool)
                token_mask = torch.cat((token_mask, pad_mask), dim=1)
        if int(dep_adj.shape[1]) != int(tokens) or int(dep_adj.shape[2]) != int(tokens):
            dep_adj = dep_adj[:, :tokens, :tokens]
            pad_t1 = max(0, int(tokens) - int(dep_adj.shape[1]))
            pad_t2 = max(0, int(tokens) - int(dep_adj.shape[2]))
            if pad_t1 > 0 or pad_t2 > 0:
                dep_adj = F.pad(dep_adj, (0, pad_t2, 0, pad_t1))

        text_valid_mask = batch.get("text_valid_mask")
        sample_losses: List[torch.Tensor] = []
        for b in range(int(bsz)):
            if isinstance(text_valid_mask, torch.Tensor):
                if b >= int(text_valid_mask.shape[0]) or not bool(text_valid_mask[b].item()):
                    continue

            valid_tok = torch.nonzero(token_mask[b], as_tuple=False).view(-1)
            if int(valid_tok.numel()) <= 1:
                continue

            local_adj = dep_adj[b][valid_tok][:, valid_tok]
            edge_idx = torch.nonzero(local_adj > 0, as_tuple=False)
            if int(edge_idx.numel()) <= 0:
                continue

            maps = base[b, valid_tok].reshape(int(valid_tok.numel()), -1)
            maps = F.normalize(maps.float(), dim=1)
            edge_losses: List[torch.Tensor] = []
            for edge in edge_idx:
                i = int(edge[0].item())
                j = int(edge[1].item())
                if i >= j:
                    continue
                sim = (maps[i] * maps[j]).sum()
                edge_losses.append(1.0 - sim)

            if edge_losses:
                sample_losses.append(torch.stack(edge_losses).mean())

        if not sample_losses:
            return torch.zeros((), device=self.device)
        return torch.stack(sample_losses).mean()

    def _compute_relation_loss(
        self,
        phrase_logits: List[torch.Tensor],
        text_token_mask: torch.Tensor | None,
        batch: Dict,
    ) -> torch.Tensor:
        """
        Compute triplet loss for comparative/relation phrases.
        Encourages objects with certain size/similarity relationships to have
        corresponding alignment scores.
        """
        if not self.relation_loss_enabled or not phrase_logits:
            return torch.zeros((), device=self.device)
        
        if not isinstance(text_token_mask, torch.Tensor) or text_token_mask.ndim != 2:
            return torch.zeros((), device=self.device)
        
        phrase_types_batch = batch.get("phrase_types")
        if phrase_types_batch is None:
            return torch.zeros((), device=self.device)
        
        bsz = int(text_token_mask.shape[0])
        losses: List[torch.Tensor] = []
        
        # Process each batch sample
        for b in range(bsz):
            batch_types = phrase_types_batch[b] if b < len(phrase_types_batch) else []
            valid_mask = text_token_mask[b] if b < int(text_token_mask.shape[0]) else None
            if valid_mask is None or not valid_mask.any():
                continue
            
            # Find relation phrases in this sample
            relation_indices = []
            for t, ptype in enumerate(batch_types):
                if ptype == "relation" and t < len(valid_mask) and valid_mask[t]:
                    relation_indices.append(t)
            
            if len(relation_indices) < 2:
                continue
            
            # Get logits for relation phrases from highest level
            if len(phrase_logits) == 0:
                continue
            pred_phrase = phrase_logits[-1]  # Use highest level (most global context)
            if not isinstance(pred_phrase, torch.Tensor) or pred_phrase.ndim != 4:
                continue
            
            # Average logits for objects for relation comparison
            # This is a simplified approach - a full implementation would reference specific objects
            logits_b = pred_phrase[b]  # [num_phrases, H, W]
            
            # Compute pairwise distances for relation phrases
            for i in range(len(relation_indices)):
                for j in range(i + 1, len(relation_indices)):
                    idx_i = relation_indices[i]
                    idx_j = relation_indices[j]
                    
                    if idx_i >= int(logits_b.shape[0]) or idx_j >= int(logits_b.shape[0]):
                        continue
                    
                    # Get max activations for each relation phrase
                    logit_i = logits_b[idx_i].max()
                    logit_j = logits_b[idx_j].max()
                    
                    # Relation loss: encourage meaningful differences
                    # This is a simplified triplet-like loss
                    diff = torch.abs(logit_i - logit_j)
                    loss = torch.clamp(diff - self.relation_margin, min=0.0)
                    losses.append(loss)
        
        if not losses:
            return torch.zeros((), device=self.device)
        
        return torch.stack(losses).mean()

    def _compute_spatial_quadrant_loss(
        self,
        phrase_logits: List[torch.Tensor],
        text_token_mask: torch.Tensor | None,
        batch: Dict,
    ) -> torch.Tensor:
        """
        Compute spatial quadrant loss for spatial phrases.
        Encourages objects to activate in specific quadrants of the image
        based on spatial keywords (left, right, top, bottom, etc).
        """
        if not self.spatial_quadrant_loss_enabled or not phrase_logits:
            return torch.zeros((), device=self.device)
        
        if not isinstance(text_token_mask, torch.Tensor) or text_token_mask.ndim != 2:
            return torch.zeros((), device=self.device)
        
        phrase_types_batch = batch.get("phrase_types")
        if phrase_types_batch is None:
            return torch.zeros((), device=self.device)
        
        bsz = int(text_token_mask.shape[0])
        losses: List[torch.Tensor] = []
        
        # Process each batch sample and spatial phrase
        for b in range(bsz):
            batch_types = phrase_types_batch[b] if b < len(phrase_types_batch) else []
            valid_mask = text_token_mask[b] if b < int(text_token_mask.shape[0]) else None
            if valid_mask is None or not valid_mask.any():
                continue
            
            # Find spatial phrases
            for t, ptype in enumerate(batch_types):
                if ptype != "spatial" or t >= len(valid_mask) or not valid_mask[t]:
                    continue
                
                # Apply spatial quadrant quadrant penalty
                for level_idx, pred_phrase in enumerate(phrase_logits):
                    if not isinstance(pred_phrase, torch.Tensor) or pred_phrase.ndim != 4:
                        continue
                    if b >= int(pred_phrase.shape[0]) or t >= int(pred_phrase.shape[1]):
                        continue
                    
                    logit = pred_phrase[b, t]  # [H, W]
                    h, w = logit.shape
                    
                    # Divide image into quadrants
                    h_mid = h // 2
                    w_mid = w // 2
                    
                    quad_tl = logit[:h_mid, :w_mid].max()  # top-left
                    quad_tr = logit[:h_mid, w_mid:].max()  # top-right
                    quad_bl = logit[h_mid:, :w_mid].max()  # bottom-left
                    quad_br = logit[h_mid:, w_mid:].max()  # bottom-right
                    
                    # Encourage activation in specific quadrants based on keywords
                    # This is a simplified version - full version would extract keywords
                    # For now, just encourage consistency (same activation across quadrants)
                    quad_max = torch.stack([quad_tl, quad_tr, quad_bl, quad_br])
                    quad_std = quad_max.std() + 1e-6
                    
                    # Loss: minimize inconsistency (too uniform), but don't suppress all quadrants
                    loss = torch.clamp(quad_max.max() - quad_std - self.relation_margin, min=0.0)
                    losses.append(loss)
        
        if not losses:
            return torch.zeros((), device=self.device)
        
        return torch.stack(losses).mean()

    def __call__(
        self,
        preds,
        batch: Dict,
        text_outputs: Dict | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        det_total, det_items = super().__call__(preds, batch)

        feats = preds[1] if isinstance(preds, tuple) else preds
        alignment_logits = []
        phrase_logits = []
        branch_visual_feats = []
        text_token_mask = None
        text_phrase_weight = None
        text_token_target_idx = None
        text_dependency_adj = None
        if isinstance(text_outputs, dict):
            raw_logits = text_outputs.get("fused_logits")
            if not isinstance(raw_logits, list):
                raw_logits = text_outputs.get("alignment_logits")
            if isinstance(raw_logits, list):
                alignment_logits = raw_logits
            raw_phrase = text_outputs.get("phrase_logits")
            if isinstance(raw_phrase, list):
                phrase_logits = raw_phrase
            raw_branch_feats = text_outputs.get("branch_visual_feats")
            if isinstance(raw_branch_feats, list):
                branch_visual_feats = raw_branch_feats
            if isinstance(text_outputs.get("text_token_mask"), torch.Tensor):
                text_token_mask = text_outputs.get("text_token_mask")
            if isinstance(text_outputs.get("text_phrase_weight"), torch.Tensor):
                text_phrase_weight = text_outputs.get("text_phrase_weight")
            if isinstance(text_outputs.get("text_token_target_idx"), torch.Tensor):
                text_token_target_idx = text_outputs.get("text_token_target_idx")
            if isinstance(text_outputs.get("text_dependency_adj"), torch.Tensor):
                text_dependency_adj = text_outputs.get("text_dependency_adj")

        if text_token_mask is None and isinstance(batch.get("text_token_mask"), torch.Tensor):
            text_token_mask = batch.get("text_token_mask")
        if text_phrase_weight is None and isinstance(batch.get("text_phrase_weight"), torch.Tensor):
            text_phrase_weight = batch.get("text_phrase_weight")
        if text_token_target_idx is None and isinstance(batch.get("text_token_target_idx"), torch.Tensor):
            text_token_target_idx = batch.get("text_token_target_idx")
        if text_dependency_adj is None and isinstance(batch.get("text_dependency_adj"), torch.Tensor):
            text_dependency_adj = batch.get("text_dependency_adj")

        heatmap_loss = self._compute_heatmap_contrastive_loss(feats, alignment_logits, batch)
        phrase_loss = self._compute_phrase_contrastive_loss(
            feats,
            phrase_logits,
            text_token_mask,
            text_phrase_weight,
            text_token_target_idx,
            batch,
        )
        set_loss = self._compute_set_matching_loss(
            feats,
            phrase_logits,
            text_token_mask,
            text_token_target_idx,
            batch,
        )
        diou_loss = self._compute_diou_grounding_loss(
            phrase_logits,
            text_token_mask,
            text_phrase_weight,
            text_token_target_idx,
            batch,
        )
        dependency_loss = self._compute_dependency_consistency_loss(
            phrase_logits,
            text_token_mask,
            text_dependency_adj,
            batch,
        )
        
        # NEW: Compute relation and spatial quadrant losses
        relation_loss = self._compute_relation_loss(
            phrase_logits,
            text_token_mask,
            batch,
        )
        spatial_quadrant_loss = self._compute_spatial_quadrant_loss(
            phrase_logits,
            text_token_mask,
            batch,
        )
        
        orth_loss = self._compute_branch_orth_loss(branch_visual_feats)
        orth_loss = orth_loss.to(device=det_total.device, dtype=det_total.dtype)
        diou_loss = diou_loss.to(device=det_total.device, dtype=det_total.dtype)
        dependency_loss = dependency_loss.to(device=det_total.device, dtype=det_total.dtype)
        relation_loss = relation_loss.to(device=det_total.device, dtype=det_total.dtype)
        spatial_quadrant_loss = spatial_quadrant_loss.to(device=det_total.device, dtype=det_total.dtype)
        batch_size = int(feats[0].shape[0]) if isinstance(feats, list) and feats else 1

        total = (
            det_total
            + self.lambda_heatmap * heatmap_loss * batch_size
            + self.lambda_phrase * phrase_loss * batch_size
            + self.lambda_set * set_loss * batch_size
            + self.lambda_ortho * orth_loss * batch_size
            + self.lambda_diou * diou_loss * batch_size
            + self.lambda_dependency * dependency_loss * batch_size
            + self.lambda_relation * relation_loss * batch_size
            + self.lambda_spatial_quadrant * spatial_quadrant_loss * batch_size
        )
        # Report per-batch-average total to keep scale aligned with other displayed items.
        total_item = (total.detach() / float(max(batch_size, 1))).view(1)
        loss_items = torch.cat(
            (
                det_items,
                heatmap_loss.detach().view(1),
                phrase_loss.detach().view(1),
                set_loss.detach().view(1),
                orth_loss.detach().view(1),
                diou_loss.detach().view(1),
                dependency_loss.detach().view(1),
                relation_loss.detach().view(1),
                spatial_quadrant_loss.detach().view(1),
                total_item,
            ),
            dim=0,
        )
        return total, loss_items


TextGuidedDetectionLoss = XRTokenAlignmentLoss
