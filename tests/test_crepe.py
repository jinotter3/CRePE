"""CPU unit tests: UCM geometry, CRePE encoding properties and checkpoint handling.

    python -m pytest tests -q
"""

import math

import numpy as np
import pytest
import torch

from crepe.adapter import CRePEConfig, GeometricAttentionAdapter, RadialHead
from crepe.camera_inputs import as_batched_tensors, build_camera_context
from crepe.checkpoint import convert_legacy_state_dict
from crepe.crepe_attention import (CRePEAttention, expected_phasor_on_path, expected_phasor_on_segment,
                                   rope_frequencies)
from crepe.geometry import focal_from_fov, fov_from_focal, ucm_project, ucm_ray_grid
from crepe.trajectory import load_trajectory, resolve_lens_value


def _orbit(num_frames=9, radius=1.0):
    poses = []
    for i in range(num_frames):
        a = 0.3 * i / max(num_frames - 1, 1)
        R = torch.tensor([[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]], dtype=torch.float64)
        t = torch.tensor([radius * math.sin(a), 0.05 * i, radius * (1 - math.cos(a))], dtype=torch.float64)
        poses.append(torch.cat([R, t[:, None]], dim=1))
    return torch.stack(poses)


@pytest.mark.parametrize("fov,xi", [(90.0, 0.0), (160.0, 1.5), (200.0, 2.3)])
def test_focal_fov_roundtrip(fov, xi):
    fx = focal_from_fov(fov, xi, 832, dtype=torch.float64)
    assert torch.allclose(fov_from_focal(fx, xi, 832, dtype=torch.float64), torch.tensor([fov], dtype=torch.float64))


def test_pinhole_focal():
    fx = focal_from_fov(90.0, 0.0, 832, dtype=torch.float64)
    assert torch.allclose(fx, torch.tensor([416.0], dtype=torch.float64))


@pytest.mark.parametrize("fov,xi", [(100.0, 0.0), (160.0, 1.5)])
def test_ucm_unproject_project_roundtrip(fov, xi):
    H, W = 30, 52
    d = ucm_ray_grid(torch.tensor([fov], dtype=torch.float64), torch.tensor([xi], dtype=torch.float64), H, W,
                     dtype=torch.float64)[0]
    valid = torch.isfinite(d).all(-1)
    u, v = ucm_project(d[..., 0] * 2.0, d[..., 1] * 2.0, d[..., 2] * 2.0,
                       torch.tensor([fov], dtype=torch.float64), torch.tensor([xi], dtype=torch.float64), H, W)
    grid_v, grid_u = torch.meshgrid(torch.arange(H, dtype=torch.float64), torch.arange(W, dtype=torch.float64), indexing="ij")
    assert torch.allclose(u[valid], grid_u[valid], atol=1e-6) and torch.allclose(v[valid], grid_v[valid], atol=1e-6)


def test_segment_phasor_matches_numerical_integral():
    a, b = torch.tensor([0.3, -1.0, 2.0]), torch.tensor([1.7, 0.5, 2.00001])
    c, s = expected_phasor_on_segment(a, b, atol=1e-4)
    t = torch.linspace(0, 1, 20001)[:, None]
    th = a + t * (b - a)
    assert torch.allclose(c, torch.cos(th).mean(0), atol=1e-4) and torch.allclose(s, torch.sin(th).mean(0), atol=1e-4)


def test_partition_invariance_for_linear_phase():
    """For a linear phase trajectory the number of uniform breakpoints does not matter.

    (K = 2 uses a coarser equal-phase fallback threshold than K >= 3, as in the
    released implementation, so K = 3 and K = 5 are compared here.)
    """
    g = torch.Generator().manual_seed(0)
    freqs = rope_frequencies(8, 8.0, 3.0).double()
    start = torch.rand(1, 1, 4, 6, dtype=torch.float64, generator=g)
    end = torch.rand(1, 1, 4, 6, dtype=torch.float64, generator=g)

    def path(k):
        alphas = torch.linspace(0, 1, k, dtype=torch.float64).view(k, 1, 1, 1, 1)
        return (1 - alphas) * start + alphas * end

    c3, s3 = expected_phasor_on_path(path(3), freqs)
    c5, s5 = expected_phasor_on_path(path(5), freqs)
    assert torch.allclose(c3, c5, atol=1e-4) and torch.allclose(s3, s5, atol=1e-4)
    # ... and K = 2 agrees up to its fallback threshold.
    c2, s2 = expected_phasor_on_path(path(2), freqs)
    assert torch.allclose(c2, c5, atol=1e-2) and torch.allclose(s2, s5, atol=1e-2)


def _context(pose, fov, xi, H=96, W=160):
    p, f, x = as_batched_tensors(pose.numpy() if torch.is_tensor(pose) else pose, fov, xi)
    return build_camera_context(p, f, x, height=H, width=W, device="cpu", camera_dtype=torch.float64)


def test_camera_context_shapes():
    pose = _orbit(9)
    ctx = _context(pose, 160.0, 1.5)
    N, P = 3, (96 // 16) * (160 // 16)
    assert ctx["w2cs"].shape == (1, N, 4, 4)
    assert ctx["ray_dirs"].shape == (1, 1, P, 3, 3)
    assert ctx["raymats"].shape == (1, N * P, 4, 4)
    assert ctx["orientation"].shape == (1, N * P, 3)
    assert ctx["xi"].shape == ctx["x_fov"].shape == (1, N)


def test_crepe_coefficients_invariant_to_global_rigid_transform():
    torch.manual_seed(0)
    pose = _orbit(9)
    H, W = 96, 160
    attn = CRePEAttention(head_dim=192, patches_x=W // 16, patches_y=H // 16, image_width=W, image_height=H)
    radial = torch.randn(1, 3 * attn.num_patches, 2, dtype=torch.float64) * 0.3
    ctx = _context(pose, 160.0, 1.5, H, W)
    # Apply the same world transform to every camera.
    a = 0.7
    G = torch.eye(4, dtype=torch.float64)
    G[:3, :3] = torch.tensor([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]], dtype=torch.float64)
    G[:3, 3] = torch.tensor([0.4, -0.2, 1.3], dtype=torch.float64)
    c2w = torch.cat([pose, torch.tensor([[[0, 0, 0, 1]]], dtype=torch.float64).expand(9, 1, 4)], dim=1)
    ctx_g = _context((G @ c2w)[:, :3], 160.0, 1.5, H, W)
    out = attn.rope_coefficients(radial, ctx, torch.float64)
    out_g = attn.rope_coefficients(radial, ctx_g, torch.float64)
    for x, y in zip(out, out_g):
        # CRePE evaluates the geometry in float32, so allow float32-level phase noise.
        assert (x - y).abs().max() < 5e-3


def test_adapter_is_identity_at_init():
    cfg = CRePEConfig(height=96, width=160)
    head = RadialHead(64, 16)
    ctx = _context(_orbit(9), 160.0, 1.5)
    ctx = {k: v.float() for k, v in ctx.items()}
    x = torch.randn(1, 3 * 60, 64)
    for use_crepe in (True, False):
        ad = GeometricAttentionAdapter(64, 48 if not use_crepe else 24 * 8, 1, 10, 6, cfg, 0, use_crepe, head)
        assert torch.count_nonzero(ad(x, ctx)) == 0


def test_legacy_key_conversion():
    legacy = {
        "pipe.dit.blocks.3.cam_self_attn.cam_encoder.weight": torch.zeros(1),
        "pipe.dit.blocks.3.cam_self_attn.q_proj.bias": torch.zeros(1),
        "pipe.dit.shared_rayrope_depth_mlp.0.weight": torch.zeros(1),
        "pipe.dit.shared_rayrope_depth_mlp.1.weight": torch.zeros(1),
        "pipe.dit.shared_rayrope_depth_mlp.3.bias": torch.zeros(1),
    }
    assert set(convert_legacy_state_dict({"state_dict": legacy})) == {
        "blocks.3.camera_adapter.orientation_proj.weight", "blocks.3.camera_adapter.q_proj.bias",
        "radial_head.norm.weight", "radial_head.fc1.weight", "radial_head.fc2.bias"}
    with pytest.raises(KeyError):
        convert_legacy_state_dict({"pipe.dit.blocks.0.self_attn.q.weight": torch.zeros(1)})


def test_lens_schedules():
    assert resolve_lens_value(160.0, None, 81) == 160.0
    lin = resolve_lens_value(110.0, {"type": "linear", "start": 110.0, "end": 180.0}, 81)
    assert lin.dtype == np.float32 and lin[0] == 110.0 and lin[-1] == 180.0
    cos = resolve_lens_value(0.0, {"type": "cosine", "start": 0.0, "end": 2.0}, 81)
    assert abs(cos[40] - 1.0) < 1e-6


def test_demo_trajectories_load():
    import glob
    files = sorted(glob.glob("demo/trajectories/*.npy"))
    assert files
    for f in files:
        pose = load_trajectory(f)
        assert pose.shape == (81, 3, 4)
        R = pose[:, :, :3].astype(np.float64)
        assert np.allclose(R @ R.transpose(0, 2, 1), np.eye(3), atol=1e-4)
