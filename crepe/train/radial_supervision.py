"""Pseudo radial-distance supervision for the shared radial head.

The radial head of one CRePE block (layer 15) is supervised with UniK3D
camera-centre distances that were cached offline at latent resolution (see
``scripts/cache_radial_distance.py``):

1. The cached distances of a clip are divided by a per-clip scale so that they
   are in the units of the scene-normalised camera translations:
   ``target_scale="near_depth"`` (released checkpoint) uses the PanFlow
   near-plane depth that also normalises the trajectory;
   ``target_scale="radial_near"`` uses a UniK3D *teacher near scale*
   (``scripts/build_radial_near.py``).
2. Pixels outside ``[1e-3, exp(MAX_LOG_DEPTH)]`` are dropped and the rest is
   average-pooled onto the patch grid (a patch is valid if > 50 % of its pixels are).
3. The head's prediction ``(mu, sigma)`` is decoded into a mean distance
   ``exp(mu)`` and the variance of a uniform distribution over
   ``[exp(mu - |sigma|), exp(mu + |sigma|)]`` (log-distances clamped to
   ``[-MAX_LOG_DEPTH, MAX_LOG_DEPTH]``, or only capped above with
   ``log_clamp="upper_only"``), and scored with a confidence-weighted L1 loss
   ``|d - d_gt| / s + alpha * log(s)`` with ``s = sqrt(clamp(var))``.

With ``head_only=False`` (released checkpoint) the loss also trains the adapter
layers that produce the head input. With ``head_only=True`` the head is re-run
on a *detached* copy of the adapter feature, so this loss only trains the radial
head; the rest of the adapter is then trained by the diffusion loss alone (which
also reaches the head through the CRePE phases). The supervision is skipped at
very low noise levels (``timestep / 1000 < min_timestep``).
"""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F

from ..crepe_attention import MAX_DEPTH, MAX_LOG_DEPTH


@dataclass
class RadialSupervisionConfig:
    enabled: bool = True
    target_layer: int = 15
    loss_weight: float = 1e-3
    min_timestep: float = 0.03
    head_only: bool = True
    confidence_alpha: float = 1.0
    var_floor: float = 1e-3
    var_ceil: float = 100.0
    target_scale: str = "radial_near"   # "radial_near" (UniK3D teacher near) or "near_depth" (PanFlow near plane)
    log_clamp: str = "upper_only"       # "upper_only" or "symmetric" log-distance clamp when decoding

    def __post_init__(self):
        if self.target_scale not in ("radial_near", "near_depth"):
            raise ValueError(f"unknown target_scale {self.target_scale!r}")
        if self.log_clamp not in ("upper_only", "symmetric"):
            raise ValueError(f"unknown log_clamp {self.log_clamp!r}")


def valid_distance_mask(distance: torch.Tensor, valid_min: float = 1e-3, valid_max: float = MAX_DEPTH) -> torch.Tensor:
    return (torch.isfinite(distance) & (distance >= valid_min) & (distance <= valid_max)).float()


def match_frame_count(frames: torch.Tensor, target: int) -> torch.Tensor:
    """Resample ``(B, T, H, W)`` along T to ``target`` frames."""
    if frames.shape[1] == target:
        return frames
    if frames.shape[1] == 1:
        return frames.expand(-1, target, -1, -1)
    idx = torch.linspace(0, frames.shape[1] - 1, target, device=frames.device).round().long()
    return frames.index_select(1, idx)


def pool_to_patches(distance: torch.Tensor, valid: torch.Tensor, patches_y: int, patches_x: int):
    """Average valid pixels per patch. Returns ``(B, N, P, 1)`` distances and validity."""
    batch, num_cameras, height, width = distance.shape
    valid_flat = valid.reshape(batch * num_cameras, 1, height, width).float()
    distance = torch.where(valid > 0.5, torch.nan_to_num(distance, nan=0.0, posinf=0.0, neginf=0.0),
                           torch.zeros_like(distance))
    distance_flat = distance.reshape(batch * num_cameras, 1, height, width)
    num = F.adaptive_avg_pool2d(distance_flat * valid_flat, (patches_y, patches_x))
    frac = F.adaptive_avg_pool2d(valid_flat, (patches_y, patches_x))
    tokens = torch.where(frac > 0.0, num / frac.clamp(min=1e-6), torch.zeros_like(num))
    tokens = tokens.reshape(batch, num_cameras, patches_y * patches_x, 1)
    valid_tokens = (frac > 0.5).float().reshape(batch, num_cameras, patches_y * patches_x, 1)
    return tokens, valid_tokens


def decode_radial_mean_var(radial_params: torch.Tensor, batch: int, num_cameras: int, num_patches: int,
                           num_rays: int = 3, log_clamp: str = "upper_only"):
    """``(B, N*P, 2)`` head output -> mean / variance of the radial distance, ``(B, N, P, 1)``.

    The prediction is shared by the ``num_rays`` rays of a token; it is decoded per
    ray and averaged, exactly as in the original training code.
    """
    def clamp(log_d):
        if log_clamp == "upper_only":
            return torch.clamp(log_d, max=MAX_LOG_DEPTH)
        return torch.clamp(log_d, min=-MAX_LOG_DEPTH, max=MAX_LOG_DEPTH)

    pred = torch.nan_to_num(radial_params.float(), nan=0.0, posinf=MAX_LOG_DEPTH, neginf=-MAX_LOG_DEPTH)
    pred = pred.reshape(batch, num_cameras, num_patches, 1, pred.shape[-1]).expand(-1, -1, -1, num_rays, -1)
    mu, sigma = pred[..., 0:1], pred[..., 1:2].abs()
    mean = torch.exp(clamp(mu))
    d1 = torch.exp(clamp(mu - sigma))
    d2 = torch.exp(clamp(mu + sigma))
    var = (d2 - d1).square() / 12.0
    mean = torch.clamp(mean, min=1e-3, max=MAX_DEPTH)
    var = torch.clamp(var, min=0.0, max=MAX_DEPTH ** 2)
    return mean.mean(dim=-2), var.mean(dim=-2)


def _masked_mean(values: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    return (values * valid).sum() / valid.sum().clamp(min=1.0)


def confidence_weighted_l1(mean, var, gt, valid, alpha: float, var_floor: float, var_ceil: float):
    pred = mean.clamp(min=1e-3, max=MAX_DEPTH)
    pred_var = var.clamp(min=0.0, max=MAX_DEPTH ** 2)
    target = gt.clamp(min=1e-3, max=MAX_DEPTH)
    scale = torch.sqrt(pred_var.clamp(min=var_floor, max=var_ceil))
    depth_term = _masked_mean((pred - target).abs() / scale, valid)
    confidence_term = _masked_mean(alpha * torch.log(scale), valid)
    return depth_term + confidence_term, depth_term, confidence_term


def radial_supervision_loss(
    radial_head: torch.nn.Module,
    features: torch.Tensor,           # (B, N*P, dim) adapter feature at the target layer (detached if head_only)
    distance: torch.Tensor,           # (B, T, H, W) cached UniK3D distance
    distance_valid: Optional[torch.Tensor],
    scale: torch.Tensor,              # (B,) per-clip scale, see RadialSupervisionConfig.target_scale
    num_cameras: int,
    patches_y: int,
    patches_x: int,
    cfg: RadialSupervisionConfig,
):
    distance = distance.float()
    if distance.ndim == 5 and distance.shape[2] == 1:
        distance = distance.squeeze(2)
    if distance_valid is None:
        valid = valid_distance_mask(distance)
    else:
        valid = distance_valid.float()
        if valid.ndim == 5 and valid.shape[2] == 1:
            valid = valid.squeeze(2)
    distance = match_frame_count(distance, num_cameras)
    valid = match_frame_count(valid, num_cameras)
    valid = valid * valid_distance_mask(distance)

    scale = torch.as_tensor(scale, device=distance.device, dtype=distance.dtype)
    if scale.ndim == 0:
        scale = scale.unsqueeze(0)
    scale = scale.reshape(distance.shape[0], *([1] * (distance.ndim - 1)))
    if not bool((torch.isfinite(scale) & (scale > 1e-6)).all()):
        raise RuntimeError(f"invalid radial target scale ({cfg.target_scale})")
    distance = distance / scale
    valid = valid * valid_distance_mask(distance, valid_max=float(torch.exp(distance.new_tensor(MAX_LOG_DEPTH)).item()))

    gt, valid_tokens = pool_to_patches(distance, valid, patches_y, patches_x)
    radial = radial_head(features)
    mean, var = decode_radial_mean_var(radial, distance.shape[0], num_cameras, patches_y * patches_x,
                                       log_clamp=cfg.log_clamp)
    loss, depth_term, conf_term = confidence_weighted_l1(mean, var, gt, valid_tokens, cfg.confidence_alpha,
                                                         cfg.var_floor, cfg.var_ceil)
    metrics = {
        "radial/loss": loss.detach(),
        "radial/l1": depth_term.detach(),
        "radial/confidence": conf_term.detach(),
        "radial/valid_fraction": valid_tokens.mean().detach(),
    }
    return loss, metrics
