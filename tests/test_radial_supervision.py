"""CPU unit tests of the radial-distance supervision used for training.

    python -m pytest tests -q
"""

import math

import torch

from crepe.adapter import CRePEConfig, GeometricAttentionAdapter, RadialHead
from crepe.crepe_attention import MAX_LOG_DEPTH
from crepe.train.radial_supervision import RadialSupervisionConfig, decode_radial_mean_var, radial_supervision_loss

from test_crepe import _context, _orbit


def test_decode_log_clamp_modes():
    radial = torch.tensor([[[-5.0, 0.0], [1.0, 0.5]]])                 # (B=1, N*P=2, 2)
    up, _ = decode_radial_mean_var(radial, 1, 1, 2, log_clamp="upper_only")
    sym, _ = decode_radial_mean_var(radial, 1, 1, 2, log_clamp="symmetric")
    assert math.isclose(up[0, 0, 0, 0].item(), math.exp(-5.0), rel_tol=1e-5)
    assert math.isclose(sym[0, 0, 0, 0].item(), math.exp(-MAX_LOG_DEPTH), rel_tol=1e-5)
    assert torch.allclose(up[0, 0, 1], sym[0, 0, 1])                  # inside the range both agree


def test_loss_divides_targets_by_scale():
    torch.manual_seed(0)
    head = RadialHead(16, 8)
    features = torch.randn(2, 4 * 6, 16)
    distance = torch.rand(2, 4, 24, 36) * 5 + 0.5
    scale = torch.tensor([2.0, 0.5])
    cfg = RadialSupervisionConfig(target_scale="near_depth", log_clamp="symmetric", confidence_alpha=0.2)
    a, _ = radial_supervision_loss(head, features, distance, None, scale, 4, 2, 3, cfg)
    b, _ = radial_supervision_loss(head, features, distance / scale[:, None, None, None], None, torch.ones(2), 4, 2, 3,
                                   cfg)
    assert torch.allclose(a, b)


def test_capture_keeps_graph_unless_head_only():
    cfg = CRePEConfig(height=96, width=160)
    head = RadialHead(64, 16)
    ctx = {k: v.float() for k, v in _context(_orbit(9), 160.0, 1.5).items()}
    x = torch.randn(1, 3 * 60, 64, requires_grad=True)
    ad = GeometricAttentionAdapter(64, 24 * 8, 1, 10, 6, cfg, 0, True, head)
    for head_only in (True, False):
        ctx["radial_capture"] = {"layer_idx": 0, "head_only": head_only}
        ad(x, ctx)
        assert ctx["radial_capture"]["features"].requires_grad is (not head_only)
