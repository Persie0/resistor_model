import torch

from resistor_model.models.bandnet import ResistorBandNet
from resistor_model.losses import LossWeights, compute_loss, monotonic_order_loss


def test_bandnet_outputs_expected_shapes():
    model = ResistorBandNet(num_colors=12, max_bands=6, sequence_bins=64, base_channels=16, d_model=64, transformer_layers=2, transformer_heads=4, slot_decoder_layers=1, use_chromatic_branch=True)
    out = model(torch.rand(2, 3, 64, 256))
    assert out["dense_logits"].shape == (2, 13, 64)
    assert out["slot_exist_logits"].shape == (2, 6)
    assert out["slot_color_logits"].shape == (2, 6, 12)
    assert out["slot_center"].shape == (2, 6)
    assert out["slot_width"].shape == (2, 6)
    assert out["count_logits"].shape == (2, 7)
    assert out["embedding"].shape == (2, 64)
    assert torch.all((out["slot_center"] >= 0) & (out["slot_center"] <= 1))


def test_monotonic_loss_penalizes_reversed_adjacent_slots():
    good = torch.tensor([[0.1, 0.3, 0.6]]); bad = torch.tensor([[0.3, 0.2, 0.6]]); exists = torch.tensor([[1.0, 1.0, 1.0]])
    assert monotonic_order_loss(good, exists, margin=0.01).item() == 0.0
    assert monotonic_order_loss(bad, exists, margin=0.01).item() > 0.0


def test_compute_loss_is_finite_and_backpropagates():
    model = ResistorBandNet(num_colors=12, max_bands=6, sequence_bins=32, base_channels=8, d_model=32, transformer_layers=1, transformer_heads=4, slot_decoder_layers=1)
    out = model(torch.rand(2, 3, 32, 128))
    targets = {
        "dense_target": torch.full((2, 32), 12, dtype=torch.long),
        "slot_exists": torch.tensor([[1,1,1,1,0,0],[1,1,1,1,0,0]], dtype=torch.float32),
        "slot_colors": torch.tensor([[1,0,2,10,-100,-100],[2,2,3,10,-100,-100]], dtype=torch.long),
        "slot_centers": torch.tensor([[.15,.35,.55,.8,0,0],[.15,.35,.55,.8,0,0]], dtype=torch.float32),
        "slot_widths": torch.tensor([[.05,.05,.05,.05,0,0],[.05,.05,.05,.05,0,0]], dtype=torch.float32),
        "count": torch.tensor([4,4], dtype=torch.long),
    }
    result = compute_loss(out, targets, LossWeights())
    assert torch.isfinite(result.total)
    assert {"dense", "color", "exist", "center", "width", "order", "count"}.issubset(result.parts)
    result.total.backward()
    assert any(p.grad is not None for p in model.parameters() if p.requires_grad)


def test_consistency_loss_accepts_second_view_embedding():
    model = ResistorBandNet(num_colors=12, max_bands=6, sequence_bins=16, base_channels=8, d_model=32, transformer_layers=1, transformer_heads=4, slot_decoder_layers=1)
    out1 = model(torch.rand(1,3,32,64)); out2 = model(torch.rand(1,3,32,64))
    targets = {"dense_target": torch.full((1,16),12,dtype=torch.long), "slot_exists": torch.zeros((1,6)), "slot_colors": torch.full((1,6),-100,dtype=torch.long), "slot_centers": torch.zeros((1,6)), "slot_widths": torch.zeros((1,6)), "count": torch.zeros((1,),dtype=torch.long)}
    result = compute_loss(out1, targets, LossWeights(consistency=0.5), second_view=out2)
    assert "consistency" in result.parts
    assert torch.isfinite(result.total)
