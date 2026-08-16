from dataclasses import replace

import torch
import torch.nn.functional as F

from yolox.tracker.gsvlt.config import GSVLTConfig
from yolox.tracker.gsvlt.gmvi import GMVI
from yolox.tracker.gsvlt.gva import GVA, sinusoidal_encoding


def small_config():
    return replace(
        GSVLTConfig(),
        history_size=5,
        hidden_dim=32,
        positional_dim=32,
        attention_heads=4,
        attention_layers=1,
        max_group_size=4,
        dropout=0.0,
    )


def test_gva_shapes_masks_and_probability_range():
    torch.manual_seed(3)
    config = small_config()
    model = GVA(config).eval()
    batch, group, candidates = 2, 4, 3
    past = torch.randn(batch, group, config.history_size, config.state_dim)
    current = torch.randn_like(past)
    initial_mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]], dtype=torch.bool)
    survivor_mask = torch.tensor([[1, 1, 0, 0], [1, 1, 0, 0]], dtype=torch.bool)
    detection_mask = torch.tensor([[1, 1, 1], [1, 1, 0]], dtype=torch.bool)
    output = model.affinity(
        past,
        current,
        initial_mask,
        survivor_mask,
        torch.tensor([30.0, 60.0]),
        torch.rand(batch, candidates, 4),
        detection_mask,
    )
    assert output.logits.shape == (batch, candidates)
    assert output.pgms.shape == (batch, config.hidden_dim)
    assert output.attention.shape == (batch, group, group)
    assert output.core_mask[:, 0].all()
    assert torch.all((output.logits[detection_mask] >= 0) & (output.logits[detection_mask] <= 1))


def test_sinusoidal_encoding_and_gmvi_residual():
    config = small_config()
    encoding = sinusoidal_encoding(torch.tensor([0.0, 10.0]), config.hidden_dim)
    assert encoding.shape == (2, config.hidden_dim)
    assert torch.allclose(encoding[0, 0::2], torch.zeros(config.hidden_dim // 2))
    model = GMVI(config).eval()
    residual = model(torch.randn(2, config.hidden_dim), torch.rand(2, 4))
    assert residual.shape == (2, 4)
    pre_velocity = torch.ones(2, 4)
    assert torch.allclose(model.infer_velocity(pre_velocity, residual), pre_velocity + residual)


def test_default_parameter_count_matches_reported_scale():
    config = GSVLTConfig()
    total = sum(parameter.numel() for parameter in GVA(config).parameters())
    total += sum(parameter.numel() for parameter in GMVI(config).parameters())
    assert 2_970_000 <= total <= 3_000_000


def test_staged_losses_backpropagate():
    torch.manual_seed(11)
    config = small_config()
    gva = GVA(config).train()
    group_mask = torch.tensor([[1, 1, 0, 0]], dtype=torch.bool)
    candidate_mask = torch.ones(1, 2, dtype=torch.bool)
    output = gva(
        torch.randn(1, 4, config.history_size, config.state_dim),
        torch.randn(1, 4, config.history_size, config.state_dim),
        group_mask,
        group_mask,
        torch.tensor([30.0]),
        torch.rand(1, 2, 4),
        candidate_mask,
    )
    gva_loss = F.binary_cross_entropy_with_logits(output.logits, torch.tensor([[1.0, 0.0]]))
    gva_loss.backward()
    assert gva.motion_encoder.lstm.weight_ih_l0.grad is not None

    gmvi = GMVI(config).train()
    velocity_residual = gmvi(output.pgms.detach(), torch.rand(1, 4))
    gmvi_loss = F.mse_loss(velocity_residual, torch.zeros_like(velocity_residual))
    gmvi_loss.backward()
    assert gmvi.velocity_head[-1].weight.grad is not None
