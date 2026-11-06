"""CPU tests for external radial-map conditioning (CRePE-Mix)."""

import math

import torch

from crepe.radial_map import RadialMapConfig, apply_radial_condition, build_radial_condition, condition_active


def _map(T=21, H=60, W=104):
    d = torch.full((T, H, W), 4.0)
    d[:, :12, :] = float("nan")          # invalid sky band (top 3 token rows of 30x52)
    d[:, 12:24, :] = 50.0                # valid but beyond exp(3) after scaling
    return d


def test_condition_token_classes():
    cond = build_radial_condition(_map(), None, 1.0, num_cameras=21, patches_y=30, patches_x=52, device="cpu")
    use = cond["use_teacher"].view(21, 30, 52)
    src = cond["source_valid"].view(21, 30, 52)
    assert not src[:, :6].any()                      # NaN band -> no source estimate
    assert src[:, 6:12].all() and not use[:, 6:12].any()   # far rays: valid source, no teacher
    assert use[:, 12:].all()
    assert torch.allclose(cond["mu"].view(21, 30, 52)[:, 12:], torch.tensor(math.log(4.0)))


def test_scale_divides_distances():
    c1 = build_radial_condition(torch.full((21, 60, 104), 4.0), None, 2.0, 21, 30, 52, "cpu")
    assert torch.allclose(c1["mu"], torch.tensor(math.log(2.0)), atol=1e-2)   # bf16 input rounding


def test_apply_overrides_and_calibration():
    cond = build_radial_condition(_map(), None, 1.0, 21, 30, 52, "cpu")
    L = 21 * 30 * 52
    pred = torch.stack([torch.linspace(-1, 1, L), torch.full((L,), 0.5)], dim=-1).unsqueeze(0)
    cfg = RadialMapConfig(sigma_mode="constant", mu_calibration="none", teacher_sigma=0.1)
    out = apply_radial_condition(pred, cond, cfg).view(21, 30, 52, 2)
    p = pred.view(21, 30, 52, 2)
    assert torch.all(out[:, :6, :, 0] == 0.0) and torch.all(out[:, :6, :, 1] == 3.0)      # invalid -> neutral prior
    assert torch.equal(out[:, 6:12], p[:, 6:12])                                          # far -> prediction
    assert torch.allclose(out[:, 12:, :, 0], torch.tensor(math.log(4.0))) and torch.all(out[:, 12:, :, 1] == 0.1)
    # mean/std calibration maps the teacher onto the prediction's statistics over teacher tokens
    d = torch.rand(21, 60, 104) * 10 + 0.5
    cond = build_radial_condition(d, None, 1.0, 21, 30, 52, "cpu")
    calib = RadialMapConfig(sigma_mode="predicted", mu_calibration="match_predicted_mean_std")
    out = apply_radial_condition(pred, cond, calib).view(-1, 2)
    m = cond["use_teacher"].view(-1)
    assert torch.allclose(out[m, 0].mean(), pred.view(-1, 2)[m, 0].mean(), atol=1e-4)
    assert torch.allclose(out[m, 0].std(unbiased=False), pred.view(-1, 2)[m, 0].std(unbiased=False), atol=1e-3)
    assert torch.equal(out[:, 1], pred.view(-1, 2)[:, 1])                                 # predicted sigma kept


def test_step_gating():
    cfg = RadialMapConfig(start_step=5, end_step=10)
    assert not condition_active(cfg, 4) and condition_active(cfg, 5) and not condition_active(cfg, 10)
    assert condition_active(cfg, None)
