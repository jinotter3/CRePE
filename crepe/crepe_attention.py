"""Curved Ray Expectation Positional Encoding (CRePE) attention.

Every image token ``j`` predicts a radial distribution along its UCM viewing
ray, parameterised by a log-distance centre ``mu_j`` and a log half-width
``sigma_j``. For a query camera ``n`` the token is lifted to ``K`` points on its
ray (log-uniform breakpoints in ``[mu - sigma, mu + sigma]``), transported to
camera ``n`` and projected with ``n``'s UCM lens. Under a wide-angle or fisheye
lens these projections trace a *curved* path; CRePE integrates the rotary
phasor ``exp(i * omega * c(t))`` along that path analytically, segment by
segment, and rotates keys/values with the expected phasor while queries use
their own camera's coordinates.

Per ray, the encoded coordinate ``c`` is 3-dimensional:
    * the bounded image-plane position ``normalize([u, v, 1])[:2]`` where
      ``(u, v)`` are the focal-normalised UCM coordinates in camera ``n``;
    * the radial range ``||X_n||`` of the point in camera ``n``.
Each token uses 3 rays (patch corner offsets), and additionally encodes the
source camera centre in camera ``n``'s frame (3 coordinates), i.e. 12
coordinates per token. With ``8`` frequencies per coordinate this fills a
192-dimensional attention head exactly.

This module started from a port of RayRoPE (https://github.com/Lucas-707/RayRoPE,
MIT License); only the code path used by the released
CRePE checkpoints is kept here, with the same floating-point operation order.
"""

from typing import Sequence, Tuple


import torch
import torch.nn as nn
import torch.nn.functional as F

from .geometry import invert_se3

MAX_DEPTH = 100.0
MAX_LOG_DEPTH = 3.0
MAX_RANGE_FEATURE = 10.0

# Max RoPE period per encoded coordinate group (the min period is
# ``max_period / freq_base ** (num_freqs - 1)``).
_MAX_PERIOD = {
    "cam_center": 1.0 * 4,             # source camera centre in the query frame
    "ray_dir": 2.0 * 4,                # bounded image-plane coordinates in [-1, 1]
    "ray_range": MAX_RANGE_FEATURE * 2 * 4,  # radial range, clamped to [-10, 10]
}

DEFAULT_RAY_OFFSETS = ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0))


# ---------------------------------------------------------------------------
# Radial distribution -> breakpoints -> 3D points
# ---------------------------------------------------------------------------

def radial_breakpoints(
    radial_params: torch.Tensor,  # (B, N*P, 2): (mu, sigma) in log-distance
    batch: int,
    num_cameras: int,
    num_patches: int,
    num_rays: int,
    num_breakpoints: int,
    point_sample_mode: str = "upper_only",
) -> torch.Tensor:
    """Distances of the ``K`` breakpoints, shape ``(K, B, N, P, R, 1)``.

    The interval ``[mu - sigma, mu + sigma]`` is clamped in log-distance and split
    uniformly. ``"upper_only"`` (camera-only checkpoints) caps it at
    ``MAX_LOG_DEPTH``; ``"clamped"`` (CRePE-Mix) clamps it to
    ``[-MAX_LOG_DEPTH, MAX_LOG_DEPTH]``.
    """
    radial_params = torch.nan_to_num(radial_params.float(), nan=0.0, posinf=MAX_LOG_DEPTH, neginf=-MAX_LOG_DEPTH)
    radial_params = radial_params.reshape(batch, num_cameras, num_patches, 1, 2)
    mu = radial_params[..., 0:1]
    sigma = radial_params[..., 1:2]
    if point_sample_mode == "upper_only":
        lo = torch.clamp(mu - sigma, max=MAX_LOG_DEPTH)
        hi = torch.clamp(mu + sigma, max=MAX_LOG_DEPTH)
    elif point_sample_mode == "clamped":
        lo = torch.clamp(mu - sigma, min=-MAX_LOG_DEPTH, max=MAX_LOG_DEPTH)
        hi = torch.clamp(mu + sigma, min=-MAX_LOG_DEPTH, max=MAX_LOG_DEPTH)
    else:
        raise ValueError(f"point_sample_mode must be 'upper_only' or 'clamped', got {point_sample_mode!r}")
    alphas = torch.linspace(0.0, 1.0, steps=num_breakpoints, device=mu.device, dtype=mu.dtype)
    alphas = alphas.view(num_breakpoints, 1, 1, 1, 1, 1)
    log_dist = (1.0 - alphas) * lo.unsqueeze(0) + alphas * hi.unsqueeze(0)
    return torch.exp(log_dist).expand(-1, -1, -1, -1, num_rays, -1)


def points_on_rays(ray_dirs: torch.Tensor, c2ws: torch.Tensor, distances: torch.Tensor) -> torch.Tensor:
    """World-space homogeneous points ``(K, B, N, P, R, 4)`` at ``distances`` along each ray.

    ``ray_dirs`` are camera-frame unit rays ``(B, 1|N, P, R, 3)``.
    """
    distances = torch.clamp(distances, min=1e-2, max=MAX_DEPTH)
    R = c2ws[..., :3, :3]
    t = c2ws[..., :3, 3]
    points_cam = ray_dirs.unsqueeze(0) * distances
    points_world = torch.einsum("bcij,ebcprj->ebcpri", R, points_cam) + t[None, :, :, None, None, :]
    return torch.cat([points_world, torch.ones_like(points_world[..., :1])], dim=-1)


def project_to_query_ucm(
    points_world: torch.Tensor,  # (K, B, N, P, R, 4)
    w2c: torch.Tensor,           # (B, 4, 4) query camera
    xi: torch.Tensor,            # (B,)
    x_fov: torch.Tensor,         # (B,) degrees
    image_width: int,
    image_height: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """UCM-project world points into a query camera.

    Returns the bounded image-plane coordinates ``(..., 2)`` and the radial
    range ``(..., 1)``. Points that cannot be projected get ``0`` for both, so
    they contribute an identity rotation.
    """
    points_cam = torch.einsum("bij,...bcprj->...bcpri", w2c, points_world)
    w_raw = points_cam[..., 3:4]
    w_safe = torch.where(w_raw.abs() > 1e-6, w_raw, torch.ones_like(w_raw))
    xyz = points_cam[..., :3] / w_safe
    X, Y, Z = xyz[..., 0:1], xyz[..., 1:2], xyz[..., 2:3]

    r = torch.linalg.norm(xyz, dim=-1, keepdim=True).clamp(min=1e-6)
    xi_b = xi.to(device=xyz.device, dtype=xyz.dtype).view(1, -1, 1, 1, 1, 1)
    denom_raw = Z + xi_b * r
    valid = torch.isfinite(xyz).all(dim=-1, keepdim=True) & torch.isfinite(denom_raw) & (denom_raw > 1e-6)
    denom = denom_raw.clamp(min=1e-6)
    x_ucm = X / denom
    y_ucm = Y / denom

    # Focal-normalised coordinates (fx / W, fy / H), centred principal point.
    x_fov_b = x_fov.to(device=xyz.device, dtype=xyz.dtype).view(1, -1, 1, 1, 1, 1)
    theta = torch.deg2rad(0.5 * x_fov_b)
    sin_theta = torch.sin(theta).clamp(min=1e-6)
    fx_over_w = 0.5 * (torch.cos(theta) + xi_b) / sin_theta
    fy_over_h = fx_over_w * (float(image_width) / float(image_height))
    valid = valid & (fx_over_w > 0.0) & torch.isfinite(fx_over_w) & torch.isfinite(fy_over_h)
    u = fx_over_w * x_ucm
    v = fy_over_h * y_ucm
    valid = valid & torch.isfinite(u) & torch.isfinite(v)
    u = torch.where(valid, u, torch.zeros_like(u))
    v = torch.where(valid, v, torch.zeros_like(v))

    uv1 = torch.cat([u, v, torch.ones_like(u)], dim=-1)
    ray_dir = (uv1 / torch.linalg.norm(uv1, dim=-1, keepdim=True).clamp(min=1e-6))[..., :2]
    ray_range = torch.where(valid, r.clamp(min=1e-4), torch.zeros_like(r))
    return ray_dir, ray_range


def camera_centers_in_query(c2ws: torch.Tensor, w2c: torch.Tensor, num_patches: int) -> torch.Tensor:
    """Source camera centres expressed in the query frame, broadcast to ``(B, N, P, 3)``."""
    batch, num_cameras = c2ws.shape[:2]
    centers = c2ws[:, :, :, 3].view(batch, num_cameras, 1, 4).expand(batch, num_cameras, num_patches, 4)
    centers = centers.unsqueeze(-2)  # (B, N, P, 1, 4)
    points_cam = torch.einsum("bij,...bcprj->...bcpri", w2c, centers)
    points_cam = points_cam / torch.clamp(points_cam[..., -1:], min=1e-4)
    return points_cam[..., :3].flatten(-2, -1)


# ---------------------------------------------------------------------------
# Expected rotary phasors
# ---------------------------------------------------------------------------

def rope_frequencies(num_freqs: int, max_period: float, freq_base: float) -> torch.Tensor:
    min_period = max_period / (freq_base ** (num_freqs - 1))
    log_min = torch.log(torch.tensor(2 * torch.pi / max_period))
    log_max = torch.log(torch.tensor(2 * torch.pi / min_period))
    return torch.exp(torch.linspace(log_min, log_max, num_freqs))


def _phase(pos: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """``(B, N, P, D) x (F,) -> (B, N, P, F, D)``."""
    return torch.einsum("f,bcpd->bcpfd", freqs, pos)


def expected_phasor_on_segment(theta_a: torch.Tensor, theta_b: torch.Tensor, atol: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """``E[cos]``, ``E[sin]`` of a phase that moves linearly from ``theta_a`` to ``theta_b``."""
    same = torch.isclose(theta_a, theta_b, atol=atol, rtol=0)
    delta = theta_b - theta_a
    delta_safe = torch.where(same, torch.ones_like(delta), delta)
    e_cos = (torch.sin(theta_b) - torch.sin(theta_a)) / delta_safe
    e_sin = (torch.cos(theta_a) - torch.cos(theta_b)) / delta_safe
    e_cos = torch.where(same, torch.cos(theta_a), e_cos)
    e_sin = torch.where(same, torch.sin(theta_a), e_sin)
    e_cos = torch.nan_to_num(e_cos, nan=1.0, posinf=0.0, neginf=0.0)
    e_sin = torch.nan_to_num(e_sin, nan=0.0, posinf=0.0, neginf=0.0)
    return e_cos, e_sin


def expected_phasor_on_path(positions: torch.Tensor, freqs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Expected phasor along a piecewise-linear path through ``K`` breakpoints.

    ``positions`` has shape ``(K, B, N, P, D)``; every one of the ``K - 1``
    equal-mass segments is integrated in closed form and the results are averaged.
    """
    num_breakpoints = positions.shape[0]
    if num_breakpoints < 2:
        raise ValueError("at least two breakpoints are required")
    atol = 1e-2 if num_breakpoints == 2 else 1e-4
    weight = 1.0 / float(num_breakpoints - 1)
    theta_prev = _phase(positions[0], freqs)
    cos_acc = sin_acc = None
    for i in range(1, num_breakpoints):
        theta_cur = _phase(positions[i], freqs)
        c, s = expected_phasor_on_segment(theta_prev, theta_cur, atol=atol)
        if num_breakpoints > 2:
            c, s = c * weight, s * weight
        cos_acc = c if cos_acc is None else cos_acc + c
        sin_acc = s if sin_acc is None else sin_acc + s
        theta_prev = theta_cur
    return cos_acc, sin_acc


def rope_coefficients(positions: dict, num_freqs: int, freq_base: float, batch: int, num_tokens: int):
    """Turn a dict of encoded coordinates into ``(cos, sin)`` of shape ``(B, num_tokens, F * D)``.

    Values are ``(B, N, P, D)`` for point positions (camera centres) or
    ``(K * B, N, P, D)`` for positions sampled along a radial distribution.
    """
    cos_list, sin_list = [], []
    for name, pos in positions.items():
        max_period = _MAX_PERIOD[name]
        if name == "ray_range":
            pos = torch.clamp(pos, min=-MAX_RANGE_FEATURE, max=MAX_RANGE_FEATURE)
        freqs = rope_frequencies(num_freqs, max_period, freq_base).to(device=pos.device, dtype=pos.dtype)
        if pos.shape[0] == batch:
            angle = _phase(pos, freqs)
            cos, sin = torch.cos(angle), torch.sin(angle)
        else:
            samples = pos.reshape(pos.shape[0] // batch, batch, *pos.shape[1:])
            cos, sin = expected_phasor_on_path(samples, freqs)
        cos_list.append(cos)
        sin_list.append(sin)
    coord_dim = sum(v.shape[-1] for v in positions.values())
    cos = torch.cat(cos_list, dim=-1).reshape(batch, num_tokens, num_freqs * coord_dim).contiguous()
    sin = torch.cat(sin_list, dim=-1).reshape(batch, num_tokens, num_freqs * coord_dim).contiguous()
    return cos, sin


def apply_rope(feats: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, inverse: bool = False) -> torch.Tensor:
    """Rotate ``(B, heads, seqlen, D)`` features by ``(B, seqlen, D // 2)`` coefficients."""
    cos = cos.to(device=feats.device, dtype=feats.dtype).unsqueeze(1)
    sin = sin.to(device=feats.device, dtype=feats.dtype).unsqueeze(1)
    rotary_dim = cos.shape[-1] * 2
    if rotary_dim > feats.shape[-1]:
        raise ValueError(f"RoPE expects rotary_dim={rotary_dim} <= head_dim={feats.shape[-1]}")
    feats_rot, feats_tail = feats[..., :rotary_dim], feats[..., rotary_dim:]
    x1, x2 = torch.chunk(feats_rot, 2, dim=-1)
    if inverse:
        rotated = torch.cat([x1 * cos + x2 * sin, -x1 * sin + x2 * cos], dim=-1)
    else:
        rotated = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    if feats_tail.shape[-1] == 0:
        return rotated
    return torch.cat([rotated, feats_tail], dim=-1)


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------

class CRePEAttention(nn.Module):
    """Parameter-free CRePE positional encoding + attention.

    Args:
        head_dim: attention head width (192 for the released adapter).
        patches_x, patches_y: latent patch grid of one frame (52 x 30 at 832 x 480).
        image_width, image_height: pixel resolution, used for focal normalisation.
        num_breakpoints: number of breakpoints ``K`` along each radial interval.
        freq_base: ratio between the largest and smallest RoPE period per step.
        ray_offsets: sub-patch offsets of the rays encoded for every token.
        point_sample_mode: clamping of the radial interval (see :func:`radial_breakpoints`).
    """

    def __init__(
        self,
        head_dim: int,
        patches_x: int,
        patches_y: int,
        image_width: int,
        image_height: int,
        num_breakpoints: int = 5,
        freq_base: float = 3.0,
        ray_offsets: Sequence[Tuple[float, float]] = DEFAULT_RAY_OFFSETS,
        point_sample_mode: str = "upper_only",
    ):
        super().__init__()
        self.point_sample_mode = point_sample_mode
        self.head_dim = head_dim
        self.patches_x = patches_x
        self.patches_y = patches_y
        self.num_patches = patches_x * patches_y
        self.image_width = image_width
        self.image_height = image_height
        self.num_breakpoints = int(num_breakpoints)
        self.freq_base = float(freq_base)
        self.ray_offsets = [list(o) for o in ray_offsets]
        self.num_rays = len(self.ray_offsets)
        # 3 camera-centre coordinates + (2 image-plane + 1 range) per ray.
        self.coord_dim = 3 + self.num_rays * 3
        self.num_freqs = head_dim // (2 * self.coord_dim)
        if self.num_freqs == 0:
            raise ValueError(f"head_dim={head_dim} is too small for {self.coord_dim} coordinates")

    def rope_coefficients(self, radial_params: torch.Tensor, geometry: dict, storage_dtype: torch.dtype):
        """Query coefficients ``(B, L, D/2)`` and per-query-camera key coefficients ``(N, B, L, D/2)``."""
        w2cs = geometry["w2cs"]
        batch, num_cameras = w2cs.shape[:2]
        num_tokens = num_cameras * self.num_patches
        # The camera inverse is taken in the camera-input dtype, then promoted.
        c2ws = invert_se3(w2cs).float()
        w2cs = w2cs.float()
        ray_dirs = geometry["ray_dirs"].float()
        xi = geometry["xi"].float()
        x_fov = geometry["x_fov"].float()

        distances = radial_breakpoints(radial_params, batch, num_cameras, self.num_patches,
                                       self.num_rays, self.num_breakpoints, self.point_sample_mode)
        points_world = points_on_rays(ray_dirs, c2ws, distances)

        q_positions = {"cam_center": [], "ray_dir": [], "ray_range": []}
        cos_kv, sin_kv = [], []
        for n in range(num_cameras):
            w2c_n = w2cs[:, n]
            kv_positions = {}
            kv_positions["cam_center"] = camera_centers_in_query(c2ws, w2c_n, self.num_patches)
            ray_dir, ray_range = project_to_query_ucm(points_world, w2c_n, xi[:, n], x_fov[:, n],
                                                      self.image_width, self.image_height)
            kv_positions["ray_dir"] = ray_dir.flatten(0, 1).flatten(-2, -1)
            kv_positions["ray_range"] = ray_range.flatten(0, 1).flatten(-2, -1)
            for key, value in kv_positions.items():
                q_positions[key].append(value[:, n])
            c, s = rope_coefficients(kv_positions, self.num_freqs, self.freq_base, batch, num_tokens)
            cos_kv.append(c.to(dtype=storage_dtype))
            sin_kv.append(s.to(dtype=storage_dtype))

        q_positions = {key: torch.stack(value, dim=1) for key, value in q_positions.items()}
        cos_q, sin_q = rope_coefficients(q_positions, self.num_freqs, self.freq_base, batch, num_tokens)
        return (cos_q.to(dtype=storage_dtype), sin_q.to(dtype=storage_dtype),
                torch.stack(cos_kv, dim=0), torch.stack(sin_kv, dim=0))

    def forward(self, q, k, v, radial_params: torch.Tensor, geometry: dict) -> torch.Tensor:
        """
        Args:
            q, k, v: ``(B, heads, L, head_dim)`` with ``L = N * P``.
            radial_params: ``(B, L, 2)`` predicted ``(mu, sigma)``.
            geometry: dict with ``w2cs (B, N, 4, 4)``, ``ray_dirs (B, 1|N, P, R, 3)``,
                ``xi (B, N)`` and ``x_fov (B, N)``.
        """
        cos_q, sin_q, cos_kv, sin_kv = self.rope_coefficients(radial_params, geometry, q.dtype)
        batch, num_heads, seqlen, head_dim = q.shape
        N = cos_kv.shape[0]
        P = seqlen // N

        q = apply_rope(q, cos_q, sin_q, inverse=True)
        cos_kv = cos_kv.reshape(N * batch, seqlen, -1)
        sin_kv = sin_kv.reshape(N * batch, seqlen, -1)
        k_exp = k.unsqueeze(0).expand(N, -1, -1, -1, -1).reshape(N * batch, num_heads, seqlen, head_dim)
        v_exp = v.unsqueeze(0).expand(N, -1, -1, -1, -1).reshape(N * batch, num_heads, seqlen, head_dim)
        k_enc = apply_rope(k_exp, cos_kv, sin_kv, inverse=True)
        v_enc = apply_rope(v_exp, cos_kv, sin_kv, inverse=True)

        # Queries of camera n attend to all keys encoded in camera n's frame.
        q_batched = torch.cat([q[:, :, i * P:(i + 1) * P, :] for i in range(N)], dim=0)
        out = F.scaled_dot_product_attention(query=q_batched.contiguous(), key=k_enc.contiguous(),
                                             value=v_enc.contiguous())
        out = out.reshape(N, batch, num_heads, P, head_dim).permute(1, 2, 0, 3, 4).reshape(batch, num_heads, seqlen, head_dim)
        out = apply_rope(out, cos_q, sin_q, inverse=False)
        return out.contiguous()
