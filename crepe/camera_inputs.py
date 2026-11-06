"""Build the per-sample camera context consumed by the adapters.

Given a camera trajectory and UCM lens parameters for every *video* frame, the
context is expressed on the latent grid: Wan2.1's VAE keeps every 4th frame
(81 frames -> 21 latent frames) and each latent frame is split into
``(H / 16) x (W / 16)`` patches.

Numerics
--------
The released checkpoints were trained and evaluated with PyTorch Lightning's
``bf16-true`` precision, which casts every floating-point tensor of a batch to
bfloat16 *before* the camera geometry is computed. :func:`build_camera_context`
therefore casts ``pose``, ``x_fov`` and ``xi`` to ``camera_dtype``
(``torch.bfloat16`` by default) first; keep that default to reproduce the
released results.
"""

from typing import Optional, Sequence, Tuple, Union

import numpy as np
import torch
from einops import rearrange, repeat

from .crepe_attention import DEFAULT_RAY_OFFSETS
from .geometry import absolute_orientation_map, invert_se3, ucm_ray_grid, ucm_rays_at_patch_offsets, world_to_ray_matrices

TEMPORAL_STRIDE = 4
PATCH_FACTOR = 16  # VAE (8x) * DiT patch (2x)


def _select_frames(values: torch.Tensor, frame_indices: torch.Tensor, target_len: int, device, dtype) -> torch.Tensor:
    """Pick per-latent-frame intrinsics from scalar / ``[B]`` / ``[B, T]`` values."""
    values = values.to(device=device, dtype=dtype) if torch.is_tensor(values) else torch.tensor(values, device=device, dtype=dtype)
    if values.ndim == 0:
        return values.view(1)
    if values.ndim == 1:
        return values
    if values.ndim != 2:
        raise ValueError(f"Expected scalar, [B] or [B, T] intrinsics, got {tuple(values.shape)}.")
    if values.shape[1] == target_len:
        return values
    if values.shape[1] == 1:
        return values[:, 0]
    if int(frame_indices[-1].item()) < values.shape[1]:
        return values.index_select(1, frame_indices.to(device=values.device))
    idx = torch.linspace(0, values.shape[1] - 1, target_len, device=values.device).round().long()
    return values.index_select(1, idx)


def _per_camera(values: torch.Tensor, num_cams: int) -> torch.Tensor:
    if values.ndim == 1:
        return values.unsqueeze(-1).expand(-1, num_cams).contiguous()
    if values.ndim == 2:
        if values.shape[1] != num_cams:
            raise ValueError(f"Per-frame intrinsics T={values.shape[1]} do not match camera count {num_cams}.")
        return values.contiguous()
    raise ValueError(f"Unexpected intrinsic shape: {tuple(values.shape)}")


def as_batched_tensors(pose, x_fov, xi):
    """Convert one sample's numpy/float inputs to batched tensors (``B = 1``).

    Mirrors PyTorch's default collation: Python floats become float64, numpy
    arrays keep their dtype.
    """
    def to_tensor(v):
        if torch.is_tensor(v):
            return v.unsqueeze(0)
        if isinstance(v, np.ndarray):
            return torch.from_numpy(v).unsqueeze(0)
        return torch.tensor([float(v)], dtype=torch.float64)
    return to_tensor(pose), to_tensor(x_fov), to_tensor(xi)


def build_camera_context(
    pose: torch.Tensor,
    x_fov: torch.Tensor,
    xi: torch.Tensor,
    height: int = 480,
    width: int = 832,
    device: Union[str, torch.device] = "cuda",
    camera_dtype: Optional[torch.dtype] = torch.bfloat16,
    ray_offsets: Sequence[Tuple[float, float]] = DEFAULT_RAY_OFFSETS,
) -> dict:
    """Camera context for a batch of trajectories.

    Args:
        pose: ``[B, T, 3, 4]`` (or ``[B, T, 4, 4]``) camera-to-world matrices for all
            ``T`` video frames (``T = 81`` for the released model), OpenCV axes,
            world ``-y`` up, scene-normalised translation.
        x_fov: horizontal field of view in degrees, ``[B]`` or ``[B, T]``.
        xi: UCM mirror parameter, ``[B]`` or ``[B, T]``.
        camera_dtype: dtype the camera inputs are cast to before any geometry
            (see module docstring). ``None`` keeps the input dtype.

    Returns:
        dict with ``w2cs [B, N, 4, 4]``, ``ray_dirs [B, 1|N, P, R, 3]``,
        ``xi / x_fov [B, N]``, ``raymats [B, N*P, 4, 4]`` and
        ``orientation [B, N*P, 3]``, where ``N`` is the number of latent frames.
    """
    device = torch.device(device)
    pose = pose.to(device)
    x_fov = x_fov.to(device) if torch.is_tensor(x_fov) else torch.tensor([float(x_fov)], dtype=torch.float64, device=device)
    xi = xi.to(device) if torch.is_tensor(xi) else torch.tensor([float(xi)], dtype=torch.float64, device=device)
    if camera_dtype is not None:
        pose, x_fov, xi = pose.to(camera_dtype), x_fov.to(camera_dtype), xi.to(camera_dtype)
    dtype = pose.dtype

    patches_x, patches_y = width // PATCH_FACTOR, height // PATCH_FACTOR
    frame_indices = torch.arange(0, pose.shape[1], TEMPORAL_STRIDE, device=device)
    target_len = int(frame_indices.numel())
    x_fov = _select_frames(x_fov, frame_indices, target_len, device, dtype)
    xi = _select_frames(xi, frame_indices, target_len, device, dtype)

    pose = pose[:, ::TEMPORAL_STRIDE]
    c2w = repeat(torch.eye(4, device=device, dtype=dtype), "... -> B T ...", B=pose.shape[0], T=pose.shape[1]).clone()
    c2w[..., :3, :4] = pose[..., :3, :4]
    num_cams = c2w.shape[1]

    # CRePE window: query/key cameras, UCM rays at 3 offsets per patch, lens per camera.
    w2cs = invert_se3(c2w)
    ray_dirs = ucm_rays_at_patch_offsets(x_fov=x_fov, xi=xi, image_h=height, image_w=width,
                                         patches_y=patches_y, patches_x=patches_x, offsets=ray_offsets,
                                         device=device, dtype=dtype)

    # Relative ray encoding: one world->ray matrix per token.
    d_cam = ucm_ray_grid(x_fov=x_fov, xi=xi, height=patches_y, width=patches_x, device=device, dtype=dtype)
    raymats = rearrange(world_to_ray_matrices(d_cam, c2w), "B T H W ... -> B (T H W) ...")

    # Absolute orientation encoding: per-token up direction + latitude.
    up_map, lat_map = absolute_orientation_map(R=pose[..., :3, :3], x_fov=x_fov, xi=xi,
                                               height=patches_y, width=patches_x, device=device)
    orientation = rearrange(torch.cat([up_map, lat_map], dim=-1), "B T H W C -> B (T H W) C")

    return {
        "w2cs": w2cs,
        "ray_dirs": ray_dirs,
        "xi": _per_camera(xi, num_cams),
        "x_fov": _per_camera(x_fov, num_cams),
        "raymats": raymats,
        "orientation": orientation,
    }
