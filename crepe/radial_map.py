"""External radial-map conditioning (Radial MixForcing at inference, "CRePE-Mix").

A model trained with Radial MixForcing accepts an externally supplied radial
map (camera-centre distance per pixel, e.g. UniK3D run on a source video)
through the same positional pathway that camera-only generation uses: in every
CRePE block the predicted radial distribution ``(mu, sigma)`` of a token is
replaced by a narrow interval around the supplied distance wherever the map is
valid.

Per token (after average-pooling the map to the latent patch grid):

* map valid and within ``exp(MAX_LOG_DEPTH)`` (after dividing by ``scale``):
  ``mu = log(distance)`` (optionally re-centred / re-scaled to the predicted
  log-distance statistics, see ``mu_calibration``) and ``sigma`` from
  ``sigma_mode``;
* map valid but farther than ``exp(MAX_LOG_DEPTH)``: the model's own prediction;
* map invalid (no UniK3D estimate): a neutral wide interval
  ``(invalid_mu, invalid_sigma)`` if ``invalid_override`` else the prediction.

The defaults (``sigma_mode="constant"`` with the training teacher width 0.1, no
calibration) are the setting of the paper's DAVIS evaluation and of the
project-page examples.
"""

import math
from dataclasses import dataclass
from typing import Optional

import torch

from .crepe_attention import MAX_DEPTH, MAX_LOG_DEPTH
from .train.radial_supervision import match_frame_count, pool_to_patches, valid_distance_mask


@dataclass
class RadialMapConfig:
    # Defaults = the paper's external-control setting (DAVIS Table 5 / Fig. 5): the map's log-distance is used as is
    # with the training teacher width. "predicted" sigma with mean/std calibration rescales the map to the model's own
    # prediction and largely removes its effect on the generated geometry.
    sigma_mode: str = "constant"                    # "constant" (teacher_sigma) or "predicted" (model's sigma)
    mu_calibration: str = "none"                    # "none", "match_predicted_mean", "match_predicted_mean_std"
    teacher_sigma: float = 0.1                      # used when sigma_mode == "constant" (training value)
    invalid_override: bool = True
    invalid_mu: float = 0.0
    invalid_sigma: float = 3.0
    start_step: int = 0                             # apply the map on denoising steps [start_step, end_step)
    end_step: Optional[int] = None


def teacher_log_distance(distance_tokens: torch.Tensor, max_log_depth: float = MAX_LOG_DEPTH, eps: float = 1e-4):
    upper = math.exp(float(max_log_depth))
    mu = distance_tokens.clamp(min=float(eps), max=upper).log()
    return mu.clamp(min=-float(max_log_depth), max=float(max_log_depth))


def build_radial_condition(
    distance: torch.Tensor,
    valid: Optional[torch.Tensor],
    scale,
    num_cameras: int,
    patches_y: int,
    patches_x: int,
    device,
    input_dtype: Optional[torch.dtype] = torch.bfloat16,
) -> dict:
    """Pool a ``(B, T, H, W)`` radial map to per-token teacher values.

    ``scale`` divides the distances (the clip's near-plane depth for PanShot clips,
    1.0 for external videos). As for camera inputs, the map, its validity mask and
    the scale are first cast to ``input_dtype`` (bf16), matching the released runs.
    """
    distance = torch.as_tensor(distance)
    if distance.ndim == 3:
        distance = distance.unsqueeze(0)
    scale = torch.as_tensor(scale, dtype=torch.float64).reshape(-1)
    if valid is not None:
        valid = torch.as_tensor(valid)
        if valid.ndim == 3:
            valid = valid.unsqueeze(0)
    if input_dtype is not None:
        distance = distance.to(input_dtype)
        scale = scale.to(input_dtype)
        valid = None if valid is None else valid.float().to(input_dtype)

    distance = distance.float().to(device)
    if distance.ndim == 5 and distance.shape[2] == 1:
        distance = distance.squeeze(2)
    valid = valid_distance_mask(distance) if valid is None else valid.float().to(device)
    if valid.ndim == 5 and valid.shape[2] == 1:
        valid = valid.squeeze(2)

    distance = match_frame_count(distance, num_cameras)
    valid = match_frame_count(valid, num_cameras)
    valid_source = valid * valid_distance_mask(distance)

    scale = torch.as_tensor(scale, device=distance.device, dtype=distance.dtype)
    if scale.shape[0] != distance.shape[0]:
        raise ValueError(f"scale batch size {scale.shape[0]} != map batch size {distance.shape[0]}")
    distance = distance / scale.reshape(-1, 1, 1, 1).clamp(min=1e-6)
    max_distance = float(torch.exp(distance.new_tensor(MAX_LOG_DEPTH)).item())
    valid_teacher = valid_source * valid_distance_mask(distance, valid_max=max_distance)

    tokens, valid_tokens = pool_to_patches(distance, valid_teacher, patches_y, patches_x)
    _, source_valid_tokens = pool_to_patches(distance, valid_source, patches_y, patches_x)
    batch = tokens.shape[0]
    seq = num_cameras * patches_y * patches_x
    return {
        "mu": teacher_log_distance(tokens).reshape(batch, seq, 1),
        "use_teacher": valid_tokens.bool().reshape(batch, seq, 1),
        "source_valid": source_valid_tokens.bool().reshape(batch, seq, 1),
    }


def _masked_mean_std(values, mask, eps):
    dims = tuple(range(1, values.ndim))
    mask_f = mask.to(device=values.device, dtype=values.dtype)
    count = mask_f.sum(dim=dims, keepdim=True)
    denom = count.clamp_min(1.0)
    mean = (values * mask_f).sum(dim=dims, keepdim=True) / denom
    var = ((values - mean).square() * mask_f).sum(dim=dims, keepdim=True) / denom
    return mean, var.sqrt().clamp_min(float(eps)), count


def _calibrate(mu_teacher, mu_pred, use_teacher, mode, eps=1e-6, max_log_depth=MAX_LOG_DEPTH):
    if mode == "none":
        return mu_teacher
    if mode not in {"match_predicted_mean", "match_predicted_mean_std"}:
        raise ValueError(f"unknown mu_calibration {mode!r}")
    t_mean, t_std, count = _masked_mean_std(mu_teacher, use_teacher, eps)
    p_mean, p_std, _ = _masked_mean_std(mu_pred.detach(), use_teacher, eps)
    calibrated = mu_teacher + (p_mean - t_mean)
    if mode == "match_predicted_mean_std":
        calibrated = (mu_teacher - t_mean) / t_std * p_std + p_mean
    calibrated = torch.where(count > 1.0, calibrated, mu_teacher)
    return calibrated.clamp(min=-float(max_log_depth), max=float(max_log_depth))


def apply_radial_condition(radial_params: torch.Tensor, condition: dict, cfg: RadialMapConfig) -> torch.Tensor:
    """Replace predicted ``(mu, sigma)`` ``(B, L, 2)`` according to a radial condition."""
    dtype = radial_params.dtype
    mu_pred, sigma_pred = radial_params[..., 0:1], radial_params[..., 1:2]
    use_teacher = condition["use_teacher"].to(device=radial_params.device).bool()
    use_invalid = (~condition["source_valid"].to(device=radial_params.device).bool()
                   if cfg.invalid_override else torch.zeros_like(use_teacher))
    mu_teacher = condition["mu"].to(device=radial_params.device, dtype=dtype).detach()
    mu_teacher = _calibrate(mu_teacher, mu_pred, use_teacher, cfg.mu_calibration)

    mu_invalid = torch.full_like(mu_pred, float(cfg.invalid_mu))
    sigma_invalid = torch.full_like(sigma_pred, float(cfg.invalid_sigma))
    if cfg.sigma_mode == "constant":
        sigma_teacher = torch.full_like(sigma_pred, float(cfg.teacher_sigma))
    elif cfg.sigma_mode == "predicted":
        sigma_teacher = sigma_pred.detach()
    else:
        raise ValueError(f"unknown sigma_mode {cfg.sigma_mode!r}")
    mu = torch.where(use_teacher, mu_teacher, torch.where(use_invalid, mu_invalid, mu_pred))
    sigma = torch.where(use_teacher, sigma_teacher, torch.where(use_invalid, sigma_invalid, sigma_pred))
    out = torch.cat([mu, sigma], dim=-1)
    return torch.cat([out, radial_params[..., 2:]], dim=-1) if radial_params.shape[-1] > 2 else out


def condition_active(cfg: RadialMapConfig, step: Optional[int]) -> bool:
    if step is None:
        return True
    return step >= cfg.start_step and (cfg.end_step is None or step < cfg.end_step)
