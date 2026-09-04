"""Regression tests for the quantile head and the composite loss."""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from rtcqr.config import RTCQRConfig
from rtcqr.losses import composite_quantile_loss, pinball_loss, quantile_crossing_penalty
from rtcqr.model import TCNQuantileNet


@pytest.fixture
def cfg():
    return RTCQRConfig()


def build(cfg, **kw):
    torch.manual_seed(0)
    return TCNQuantileNet(cfg.in_channels, cfg.quantile_levels, cfg.num_blocks,
                          cfg.channels, cfg.kernel_size, cfg.dropout, **kw)


# --------------------------------------------------------------------------
# Structural non-crossing
# --------------------------------------------------------------------------

def test_quantiles_never_cross_even_at_extreme_parameters(cfg):
    """The head emits q[k] = q[k-1] + softplus(.), so this is structural, not
    a tendency the penalty has to enforce."""
    for seed in range(10):
        torch.manual_seed(seed)
        m = build(cfg)
        with torch.no_grad():
            for p in m.parameters():
                p.mul_(50.0)
            q = m(torch.randn(128, cfg.in_channels, cfg.window_size) * 20)
        assert (torch.diff(q, dim=1) >= 0).all()
        assert torch.isfinite(q).all()


def test_crossing_penalty_is_identically_zero_against_this_head(cfg):
    m = build(cfg)
    with torch.no_grad():
        q = m(torch.randn(64, cfg.in_channels, cfg.window_size))
    assert float(quantile_crossing_penalty(q).max()) == 0.0


def test_crossing_penalty_matches_the_pairwise_definition():
    """Vectorised form must equal sum_{j<k} relu(q_j - q_k)."""
    torch.manual_seed(0)
    q = torch.randn(32, 8)
    expected = torch.zeros(32)
    for j in range(8):
        for k in range(j + 1, 8):
            expected = expected + torch.relu(q[:, j] - q[:, k])
    assert torch.allclose(expected, quantile_crossing_penalty(q))


def test_crossing_penalty_is_positive_on_deliberately_crossed_input():
    q = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    got = quantile_crossing_penalty(q)
    assert float(got[0]) == pytest.approx(1.0) and float(got[1]) == 0.0


# --------------------------------------------------------------------------
# Head initialisation
# --------------------------------------------------------------------------

def test_head_starts_at_a_sane_interval_width(cfg):
    """Default softplus(0) increments stack to a ~4.85-wide initial fan --
    almost 5x the entire [0,1] SoC range -- and softplus's small gradient in
    the negative tail makes recovering from that slow."""
    m = build(cfg)
    m.eval()
    with torch.no_grad():
        q = m(torch.randn(256, cfg.in_channels, cfg.window_size))
    width = float((q[:, -1] - q[:, 0]).mean())
    assert width < 0.5, f"initial 95% PI width {width:.3f} is far wider than the label range"


def test_initial_gap_is_configurable(cfg):
    m = build(cfg, initial_gap=0.05)
    expected = math.log(math.expm1(0.05))
    assert torch.allclose(m.head.bias[1:], torch.full_like(m.head.bias[1:], expected))


def test_single_output_head_is_a_plain_point_model(cfg):
    """The point baseline reuses this class with one level, where the head is
    just `base` with no increments."""
    torch.manual_seed(0)
    m = TCNQuantileNet(cfg.in_channels, [0.5], cfg.num_blocks, cfg.channels,
                       cfg.kernel_size, cfg.dropout)
    with torch.no_grad():
        out = m(torch.randn(8, cfg.in_channels, cfg.window_size))
    assert out.shape == (8, 1) and torch.isfinite(out).all()


# --------------------------------------------------------------------------
# Loss semantics
# --------------------------------------------------------------------------

def test_pinball_is_minimised_at_the_true_quantile():
    torch.manual_seed(0)
    y = torch.rand(20000)
    for tau in (0.05, 0.5, 0.95):
        truth = torch.quantile(y, tau)
        at_truth = pinball_loss(y, torch.full_like(y, float(truth)), tau).mean()
        for offset in (-0.05, 0.05):
            worse = pinball_loss(y, torch.full_like(y, float(truth) + offset), tau).mean()
            assert worse >= at_truth


def test_lower_tail_regularizer_pushes_the_lower_quantile_up(cfg):
    """eq. (17)/(18): [SoC_min - q_tau_l]_+ is a surrogate for the lower-bound
    violation event {q_tau_l < SoC_min}, so minimising it raises q_tau_l."""
    q = torch.zeros(4, len(cfg.quantile_levels), requires_grad=True)
    loss = composite_quantile_loss(torch.full((4,), 0.5), q, cfg.quantile_levels,
                                   soc_min=0.10, lambda_nc=0.0, lambda_l=1.0,
                                   tau_l_index=cfg.tau_l_index)
    loss.backward()
    assert q.grad[:, cfg.tau_l_index].sum() < 0, "gradient must push q_tau_l upward"


def test_lower_tail_regularizer_reaches_every_quantile_through_base(cfg):
    """With the monotone head the LTR's only path into the network is the
    `base` unit, which is added to every quantile -- so a step on it
    translates the whole fan rather than moving q_tau_l alone. This did not
    hold for the original unconstrained head; pinning it so the behaviour
    change is not silently reintroduced or silently lost."""
    m = build(cfg)
    x = torch.randn(8, cfg.in_channels, cfg.window_size)
    ltr = torch.relu(torch.tensor(cfg.soc_min) - m(x)[:, cfg.tau_l_index]).mean()
    grad = torch.autograd.grad(ltr, m.head.bias)[0]
    assert grad[0] != 0
    assert torch.allclose(grad[1:], torch.zeros_like(grad[1:]))

    m.eval()
    with torch.no_grad():
        before = m(x).clone()
        m.head.bias[0] += 0.1
        shift = m(x) - before
    assert torch.allclose(shift, torch.full_like(shift, 0.1), atol=1e-5)


def test_lambda_nc_zero_skips_the_penalty_without_changing_the_value(cfg):
    torch.manual_seed(0)
    q = torch.sort(torch.randn(16, len(cfg.quantile_levels)), dim=1).values
    y = torch.rand(16)
    kw = dict(quantile_levels=cfg.quantile_levels, soc_min=cfg.soc_min,
              lambda_l=cfg.lambda_l, tau_l_index=cfg.tau_l_index)
    assert composite_quantile_loss(y, q, lambda_nc=0.0, **kw) == pytest.approx(
        float(composite_quantile_loss(y, q, lambda_nc=1.0, **kw)))


def test_no_ltr_ablation_removes_the_term(cfg):
    torch.manual_seed(0)
    q = torch.sort(torch.randn(16, len(cfg.quantile_levels)) - 2.0, dim=1).values
    y = torch.rand(16)
    kw = dict(quantile_levels=cfg.quantile_levels, soc_min=cfg.soc_min,
              lambda_nc=0.0, tau_l_index=cfg.tau_l_index)
    with_ltr = composite_quantile_loss(y, q, lambda_l=0.1, **kw)
    without = composite_quantile_loss(y, q, lambda_l=0.0, **kw)
    assert float(with_ltr) > float(without)


# --------------------------------------------------------------------------
# Mixed precision
# --------------------------------------------------------------------------

def test_loss_stays_fp32_under_autocast(cfg):
    """q_pred is reduced precision under autocast, but soc_true is fp32 and
    the subtraction promotes, so the loss keeps full precision -- which the
    early-stopping comparison against a 1e-6 threshold depends on."""
    m = build(cfg)
    x = torch.randn(8, cfg.in_channels, cfg.window_size)
    y = torch.rand(8)
    with torch.amp.autocast(device_type="cpu", enabled=True):
        q = m(x)
        loss = composite_quantile_loss(y, q, cfg.quantile_levels, cfg.soc_min,
                                       cfg.lambda_nc, cfg.lambda_l, cfg.tau_l_index)
    assert q.dtype in (torch.bfloat16, torch.float16)
    assert loss.dtype == torch.float32


def test_forward_backward_runs_under_autocast_and_gradscaler(cfg):
    m = build(cfg)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler("cpu", enabled=False)
    x = torch.randn(16, cfg.in_channels, cfg.window_size)
    y = torch.rand(16)
    opt.zero_grad(set_to_none=True)
    with torch.amp.autocast(device_type="cpu", enabled=True):
        loss = composite_quantile_loss(y, m(x), cfg.quantile_levels, cfg.soc_min,
                                       cfg.lambda_nc, cfg.lambda_l, cfg.tau_l_index)
    scaler.scale(loss).backward()
    scaler.step(opt)
    scaler.update()
    assert all(torch.isfinite(p).all() for p in m.parameters())


def test_state_dict_round_trips(cfg):
    """train.py saves best_state and diagnose_calibration.py reloads it."""
    a = build(cfg)
    b = build(cfg)
    with torch.no_grad():
        for p in b.parameters():
            p.add_(1.0)
    b.load_state_dict({k: v.detach().cpu().clone() for k, v in a.state_dict().items()})
    a.eval(); b.eval()
    x = torch.randn(4, cfg.in_channels, cfg.window_size)
    with torch.no_grad():
        assert torch.allclose(a(x), b(x))


def test_only_the_receptive_field_influences_the_output(cfg):
    """The backbone is causal and finite: exactly the last `receptive_field`
    steps reach the output, and everything older is dead input. With the
    paper's 4 blocks / kernel 3 that is 61, so the default window_size=100
    carries 39 steps the network cannot see."""
    m = build(cfg)
    m.eval()
    rf = m.receptive_field
    assert rf == 1 + 2 * sum((cfg.kernel_size - 1) * 2 ** b for b in range(cfg.num_blocks))

    n = rf + 20
    x = torch.randn(1, cfg.in_channels, n)
    with torch.no_grad():
        base = m(x)

    def moved(step):
        x2 = x.clone()
        x2[:, :, step] += 10.0
        with torch.no_grad():
            return not torch.allclose(base, m(x2), atol=1e-7)

    assert moved(n - 1), "the most recent step must influence the output"
    assert moved(n - rf), "the oldest in-receptive-field step must influence the output"
    assert not moved(n - rf - 1), "a step older than the receptive field must not"
    assert not moved(0)


def test_window_longer_than_receptive_field_is_reported(cfg, capsys):
    m = build(cfg)
    m.warn_if_window_exceeds_receptive_field(cfg.window_size)
    out = capsys.readouterr().out
    assert "receptive field" in out and str(m.receptive_field) in out


def test_no_warning_when_window_fits(cfg, capsys):
    m = build(cfg)
    m.warn_if_window_exceeds_receptive_field(m.receptive_field)
    assert capsys.readouterr().out == ""
