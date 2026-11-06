"""Unified Camera Model (UCM) geometry used by CRePE.

Conventions
-----------
* Camera frame: x right, y down, z forward (OpenCV).
* Poses are camera-to-world ``c2w`` matrices.
* A UCM camera is parameterised by its horizontal field of view ``x_fov`` (in
  degrees) and the mirror parameter ``xi`` (``xi = 0`` is a pinhole camera).
  The focal length follows from requiring the image border to map to the
  half-FoV angle: ``fx = (W / 2) (cos(fov / 2) + xi) / sin(fov / 2)``, and
  ``fy = fx`` with the principal point at the image centre.

The functions below are ported from UCPE (https://github.com/chengzhag/UCPE,
MIT License, Copyright (c) 2026 Cheng Zhang; pixel grid / UCM unprojection
originally from equilib, Apache-2.0) and keep the exact floating-point operation order of the code the released
checkpoints were trained with.
"""

from typing import Sequence, Tuple, Union

import torch
import torch.nn.functional as F
from einops import einsum, rearrange, repeat

Number = Union[float, int]


def _as_1d(x, device, dtype) -> torch.Tensor:
    if torch.is_tensor(x):
        return x.to(device=device, dtype=dtype)
    return torch.tensor([x], dtype=dtype, device=device)


def focal_from_fov(x_fov, xi, width: int, device="cpu", dtype=torch.float32) -> torch.Tensor:
    """UCM focal length (pixels) from horizontal FoV (degrees) and ``xi``. Returns ``[B]``."""
    x_fov = _as_1d(x_fov, device, dtype)
    xi = _as_1d(xi, device, dtype)
    batch = max(x_fov.shape[0], xi.shape[0])
    x_fov = x_fov.view(-1).expand(batch)
    xi = xi.view(-1).expand(batch)
    theta = torch.deg2rad(0.5 * x_fov)
    eps = torch.finfo(dtype).eps
    denom = torch.sin(theta).clamp_min(eps)
    return (width * 0.5) * (torch.cos(theta) + xi) / denom


def fov_from_focal(fx, xi, width: int, device="cpu", dtype=torch.float32) -> torch.Tensor:
    """Inverse of :func:`focal_from_fov` (degrees)."""
    fx = _as_1d(fx, device, dtype).view(-1)
    xi = _as_1d(xi, device, dtype).view(-1)
    batch = max(fx.shape[0], xi.shape[0])
    fx = fx.expand(batch)
    xi = xi.expand(batch)
    a = 2.0 * fx / width
    phi = torch.atan(1.0 / a)
    ratio = (xi / torch.sqrt(a * a + 1.0)).clamp(-1.0, 1.0)
    return torch.rad2deg(2.0 * (torch.asin(ratio) + phi))


def pixel_grid(height: int, width: int, batch: int, dtype=torch.float32, device="cpu") -> torch.Tensor:
    """``[B, H, W, 3]`` grid of homogeneous pixel coordinates ``(u, v, 1)``."""
    xs = torch.linspace(0, width - 1, width, dtype=dtype, device=device)
    ys = torch.linspace(0, height - 1, height, dtype=dtype, device=device)
    ys, xs = torch.meshgrid([ys, xs], indexing="ij")
    zs = torch.ones_like(xs, dtype=dtype, device=device)
    grid = torch.stack((xs, ys, zs), dim=2)
    return repeat(grid, "... -> b ...", b=batch)


def ucm_unproject_pixels(height, width, fx, fy, cx, cy, xi, dtype=torch.float32, device="cpu") -> torch.Tensor:
    """Back-project every pixel centre of an ``H x W`` image to a UCM ray (``[B, H, W, 3]``).

    The returned directions are *not* unit-normalised (they lie on the UCM
    unit sphere for valid pixels) and are NaN outside the UCM image domain.
    """
    fx, fy, cx, cy, xi = (_as_1d(v, device, dtype) for v in (fx, fy, cx, cy, xi))
    grid = pixel_grid(height, width, batch=fx.shape[0], dtype=dtype, device=device)
    u, v = grid[..., 0], grid[..., 1]
    fx, fy, cx, cy, xi = (t[:, None, None] for t in (fx, fy, cx, cy, xi))
    x = (u - cx) / fx
    y = (v - cy) / fy
    r2 = x * x + y * y
    alpha = xi + torch.sqrt(1 + (1 - xi * xi) * r2)
    gamma = alpha / (1 + r2)
    return torch.stack([gamma * x, gamma * y, gamma - xi], dim=-1)


def ucm_ray_grid(x_fov, xi, height: int, width: int, device="cpu", dtype=torch.float32) -> torch.Tensor:
    """UCM ray directions on an ``H x W`` grid.

    ``x_fov`` / ``xi`` may be scalars, ``[B]`` or ``[B, T]``; the output is
    ``[H, W, 3]``, ``[B, H, W, 3]`` or ``[B, T, H, W, 3]`` respectively.
    """
    device = torch.device(device)
    x_fov = x_fov.to(device=device, dtype=dtype) if torch.is_tensor(x_fov) else torch.tensor(x_fov, dtype=dtype, device=device)
    xi = xi.to(device=device, dtype=dtype) if torch.is_tensor(xi) else torch.tensor(xi, dtype=dtype, device=device)
    scalar_input = x_fov.ndim == 0 and xi.ndim == 0
    if x_fov.ndim == 0:
        x_fov = x_fov.view(1)
    if xi.ndim == 0:
        xi = xi.view(1)
    x_fov, xi = torch.broadcast_tensors(x_fov, xi)
    batch_shape = x_fov.shape
    x_fov_flat, xi_flat = x_fov.reshape(-1), xi.reshape(-1)
    fx = focal_from_fov(x_fov_flat, xi_flat, width, device, dtype)
    d_cam = ucm_unproject_pixels(height, width, fx, fx, width / 2, height / 2, xi_flat, dtype=dtype, device=device)
    d_cam = d_cam.reshape(*batch_shape, height, width, 3)
    return d_cam[0] if scalar_input else d_cam


def ucm_rays_at_patch_offsets(
    x_fov: torch.Tensor,
    xi: torch.Tensor,
    image_h: int,
    image_w: int,
    patches_y: int,
    patches_x: int,
    offsets: Sequence[Tuple[float, float]],
    device="cpu",
    dtype=torch.float32,
) -> torch.Tensor:
    """Unit UCM rays at sub-patch offsets, ``[B, N_or_1, P, R, 3]``.

    For every latent patch ``(i, j)`` and offset ``(ox, oy)`` (in patch units)
    the pixel ``((j + ox) * W / Px, (i + oy) * H / Py)`` is back-projected
    analytically. Pixels outside the UCM domain get a neutral forward ray.
    """
    if not torch.is_tensor(x_fov):
        x_fov = torch.tensor(x_fov, device=device, dtype=dtype)
    if not torch.is_tensor(xi):
        xi = torch.tensor(xi, device=device, dtype=dtype)
    x_fov = x_fov.to(device=device, dtype=dtype)
    xi = xi.to(device=device, dtype=dtype)
    if x_fov.ndim == 0:
        x_fov = x_fov.view(1, 1)
    elif x_fov.ndim == 1:
        x_fov = x_fov.unsqueeze(-1)
    if xi.ndim == 0:
        xi = xi.view(1, 1)
    elif xi.ndim == 1:
        xi = xi.unsqueeze(-1)
    x_fov, xi = torch.broadcast_tensors(x_fov, xi)
    B, N = x_fov.shape

    eps = torch.finfo(dtype).eps
    theta = torch.deg2rad(0.5 * x_fov)
    fx = (image_w * 0.5) * (torch.cos(theta) + xi) / torch.sin(theta).clamp_min(eps)
    intr_valid = torch.isfinite(fx) & (fx > eps)

    cx, cy = image_w * 0.5, image_h * 0.5
    pw, ph = image_w / patches_x, image_h / patches_y
    gx, gy = torch.meshgrid(
        torch.arange(patches_x, device=device, dtype=dtype),
        torch.arange(patches_y, device=device, dtype=dtype),
        indexing="xy",
    )
    fx_b, xi_b, intr_valid_b = fx[..., None, None], xi[..., None, None], intr_valid[..., None, None]

    dirs_per_offset = []
    for ox, oy in offsets:
        u = (gx + float(ox)) * pw
        v = (gy + float(oy)) * ph
        x = (u[None, None] - cx) / fx_b
        y = (v[None, None] - cy) / fx_b
        r2 = x * x + y * y
        inside_raw = 1.0 + (1.0 - xi_b * xi_b) * r2
        valid = intr_valid_b & torch.isfinite(inside_raw) & (inside_raw > eps)
        inside = inside_raw.clamp_min(eps)
        alpha = xi_b + torch.sqrt(inside)
        gamma = alpha / (1.0 + r2).clamp_min(eps)
        d = torch.stack([gamma * x, gamma * y, gamma - xi_b], dim=-1)
        d_norm = torch.linalg.norm(d, dim=-1, keepdim=True)
        valid = valid & torch.isfinite(d).all(dim=-1) & (d_norm[..., 0] > eps)
        d = d / d_norm.clamp_min(eps)
        neutral = torch.zeros_like(d)
        neutral[..., 2] = 1.0
        d = torch.where(valid[..., None], d, neutral)
        dirs_per_offset.append(d.reshape(B, N, patches_y * patches_x, 3))

    out = torch.stack(dirs_per_offset, dim=3)
    out = torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    return out / out.norm(dim=-1, keepdim=True).clamp_min(eps)


def ucm_project(X, Y, Z, x_fov, xi, height: int, width: int):
    """Project camera-frame points to UCM pixel coordinates ``(u, v)``."""
    x_fov = x_fov.to(device=X.device, dtype=X.dtype) if torch.is_tensor(x_fov) else torch.tensor(x_fov, device=X.device, dtype=X.dtype)
    xi = xi.to(device=X.device, dtype=X.dtype) if torch.is_tensor(xi) else torch.tensor(xi, device=X.device, dtype=X.dtype)
    if x_fov.ndim == 0:
        x_fov = x_fov.view(1)
    if xi.ndim == 0:
        xi = xi.view(1)
    x_fov, xi = torch.broadcast_tensors(x_fov, xi)
    eps = torch.finfo(X.dtype).eps
    theta = torch.deg2rad(0.5 * x_fov)
    fx = (width * 0.5) * (torch.cos(theta) + xi) / torch.sin(theta).clamp_min(eps)
    fy = fx
    while fx.ndim < X.ndim:
        fx, fy, xi = fx.unsqueeze(-1), fy.unsqueeze(-1), xi.unsqueeze(-1)
    r = torch.sqrt(X * X + Y * Y + Z * Z)
    alpha = Z + xi * r
    return fx * (X / alpha) + width / 2, fy * (Y / alpha) + height / 2


def invert_se3(transforms: torch.Tensor) -> torch.Tensor:
    """Invert ``[..., 4, 4]`` rigid transforms."""
    assert transforms.shape[-2:] == (4, 4)
    R_inv = transforms[..., :3, :3].transpose(-1, -2)
    out = torch.zeros_like(transforms)
    out[..., :3, :3] = R_inv
    out[..., :3, 3] = -torch.einsum("...ij,...j->...i", R_inv, transforms[..., :3, 3])
    out[..., 3, 3] = 1.0
    return out


def world_to_ray_matrices(d_cam: torch.Tensor, c2w: torch.Tensor) -> torch.Tensor:
    """Per-ray world-to-ray SE(3) matrices used by relative ray encoding (UCPE).

    Each ray gets a local frame with ``z`` along the ray, ``x = cam_y x z`` and
    ``y = z x x``, centred at the camera.

    Args:
        d_cam: ``[B, H, W, 3]`` or ``[B, T, H, W, 3]`` camera-frame ray directions.
        c2w: ``[B, T, 4, 4]`` camera-to-world poses.

    Returns:
        ``[B, T, H, W, 4, 4]``.
    """
    T = c2w.shape[1]
    if d_cam.ndim == 4:
        B, H, W, _ = d_cam.shape
        d_cam = repeat(d_cam, "b h w c -> b t h w c", t=T)
    elif d_cam.ndim == 5:
        B, d_cam_T, H, W, _ = d_cam.shape
        if d_cam_T != T:
            raise ValueError(f"d_cam T={d_cam_T} does not match c2w T={T}.")
    else:
        raise ValueError(f"Expected d_cam rank 4 or 5, got shape {tuple(d_cam.shape)}.")
    device, dtype = d_cam.device, d_cam.dtype

    R_cam = c2w[..., :3, :3]
    t_cam = c2w[..., :3, 3]
    d_world = einsum(R_cam, d_cam, "b t i j, b t h w j -> b t h w i")
    cam_y = repeat(R_cam[..., :, 1], "b t c -> b t h w c", h=H, w=W)

    z_ray = F.normalize(d_world, dim=-1, eps=1e-6)
    x_ray = F.normalize(torch.cross(cam_y, z_ray, dim=-1), dim=-1, eps=1e-6)
    y_ray = F.normalize(torch.cross(z_ray, x_ray, dim=-1), dim=-1, eps=1e-6)
    R_l2w = torch.stack([x_ray, y_ray, z_ray], dim=-1)
    R_w2l = rearrange(R_l2w, "b t h w i j -> b t h w j i")
    t_world = repeat(t_cam, "b t c -> b t h w c", h=H, w=W)
    t_w2l = -einsum(R_w2l, t_world, "b t h w i j, b t h w j -> b t h w i")

    raymats = torch.zeros(B, T, H, W, 4, 4, device=device, dtype=dtype)
    raymats[..., :3, :3] = R_w2l
    raymats[..., :3, 3] = t_w2l
    raymats[..., 3, 3] = 1.0
    mask = torch.isnan(d_world).any(-1)
    raymats[mask] = torch.eye(4, device=device, dtype=dtype)
    return raymats


def absolute_orientation_map(
    R: torch.Tensor,
    x_fov: torch.Tensor,
    xi: torch.Tensor,
    height: int,
    width: int,
    device=torch.device("cpu"),
    delta: float = 0.1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-pixel gravity "up" direction and latitude (UCPE's absolute orientation map).

    Args:
        R: ``[B, T, 3, 3]`` camera-to-world rotations (world ``-y`` is up).
        x_fov, xi: ``[B]`` or ``[B, T]`` lens parameters.
        height, width: map resolution (the latent patch grid).

    Returns:
        ``up_map [B, T, H, W, 2]`` (unit image-plane vectors) and
        ``lat_map [B, T, H, W, 1]`` (radians), both in ``R.dtype``.
    """
    B, T = R.shape[:2]
    dtype = R.dtype
    R = R.float()

    d_cam = ucm_ray_grid(x_fov=x_fov, xi=xi, height=height, width=width, device=device, dtype=torch.float32)
    if d_cam.ndim == 3:
        d_cam = d_cam.unsqueeze(0)
    mask = d_cam.isnan().any(dim=-1, keepdim=True)
    if d_cam.ndim == 4:
        d_cam_exp = repeat(d_cam, "B H W C -> B T H W C", T=T)
        mask = repeat(mask, "B H W C -> B T H W C", T=T)
    elif d_cam.ndim == 5:
        if d_cam.shape[1] != T:
            raise ValueError(f"UCM ray T={d_cam.shape[1]} does not match R T={T}.")
        d_cam_exp = d_cam
    else:
        raise ValueError(f"Unexpected UCM ray shape: {tuple(d_cam.shape)}")

    d_world = torch.einsum("btij,bthwj->bthwi", R, d_cam_exp)
    d_world = d_world / torch.clamp_min(d_world.norm(dim=-1, keepdim=True), 1e-8)

    Xw, Yw, Zw = d_world[..., 0], d_world[..., 1], d_world[..., 2]
    lat_map = torch.atan2(-Yw, torch.sqrt(Xw ** 2 + Zw ** 2)).unsqueeze(-1)

    # Rotate each ray slightly towards world-up and re-project: the image-space
    # displacement is the local "up" direction.
    v = d_world
    up_world = torch.tensor([0, -1, 0], device=device, dtype=torch.float32)
    k = torch.cross(v, up_world.unsqueeze(0).unsqueeze(0).unsqueeze(0).expand_as(v), dim=-1)
    k = k / torch.clamp_min(k.norm(dim=-1, keepdim=True), 1e-8)
    delta = torch.tensor(delta, device=device, dtype=torch.float32)
    cos_eps, sin_eps = torch.cos(delta), torch.sin(delta)
    v_rot = v * cos_eps + torch.cross(k, v, dim=-1) * sin_eps + k * (k * (v * 1).sum(dim=-1, keepdim=True)) * (1 - cos_eps)
    dirs_cam = torch.einsum("btij,bthwj->bthwi", R.transpose(-1, -2), v_rot)
    du, dv = ucm_project(dirs_cam[..., 0], dirs_cam[..., 1], dirs_cam[..., 2],
                         x_fov=x_fov.float(), xi=xi.float(), height=height, width=width)

    grid = pixel_grid(height=height, width=width, batch=B, dtype=torch.float32, device=device)
    grid_x = grid[..., 0].unsqueeze(1)
    grid_y = grid[..., 1].unsqueeze(1)
    up_map = torch.stack((du - grid_x, dv - grid_y), dim=-1)
    up_map = up_map / torch.clamp_min(up_map.norm(dim=-1, keepdim=True), 1e-8)

    up_map, lat_map = up_map.to(dtype=dtype), lat_map.to(dtype=dtype)
    mask_exp = mask.unsqueeze(1).expand(B, T, height, width, 1) if mask.ndim == 4 else mask
    return up_map.masked_fill(mask_exp, 0.0), lat_map.masked_fill(mask_exp, 0.0)
