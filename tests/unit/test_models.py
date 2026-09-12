import torch

from sync2act.evaluation import TemporalEnsembler
from sync2act.policies import BCMLP, ACTLite, QualityAwareACT, quality_weighted_action_loss
from sync2act.training import inspect_checkpoint, load_checkpoint, save_checkpoint


def test_model_forward_shapes():
    state, image = torch.randn(3, 6), torch.randn(3, 3, 32, 32)
    multi_camera_image = torch.randn(3, 4, 3, 32, 32)
    assert BCMLP(6, 3)(state).shape == (3, 1, 3)
    assert BCMLP(6, 3, "image_state")(state, image).shape == (3, 1, 3)
    assert ACTLite(6, 3, horizon=5, hidden_dim=32, num_layers=1)(state, image).shape == (3, 5, 3)
    assert QualityAwareACT(6, 3, horizon=5, hidden_dim=32, num_layers=1)(
        state=state,
        image=image,
        image_quality=torch.ones(3, 1),
        state_quality=torch.ones(3, 1),
        missing=torch.zeros(3, 1),
        time_offset=torch.zeros(3, 1),
    ).shape == (3, 5, 3)
    assert BCMLP(6, 3, "image_state")(state, multi_camera_image).shape == (3, 1, 3)
    assert ACTLite(6, 3, horizon=5, hidden_dim=32, num_layers=1)(
        state, multi_camera_image
    ).shape == (3, 5, 3)


def test_act_lite_retains_one_observation_token_per_camera():
    model = ACTLite(6, 3, horizon=2, hidden_dim=32, num_layers=1)
    tokens = model.observation_tokens(torch.randn(2, 6), torch.randn(2, 3, 3, 16, 16))
    assert tokens.shape == (2, 4, 32)  # three camera tokens plus one state token


def test_quality_act_uses_per_camera_and_state_quality():
    model = QualityAwareACT(6, 3, horizon=2, hidden_dim=32, num_layers=1)
    tokens = model.observation_tokens(
        state=torch.randn(2, 6),
        image=torch.randn(2, 3, 3, 16, 16),
        image_quality=torch.tensor([[1.0, 0.5, 0.0], [0.25, 0.75, 1.0]]),
        state_quality=torch.tensor([[0.5], [1.0]]),
        missing=torch.zeros(2, 1),
        time_offset=torch.zeros(2, 1),
    )
    assert tokens.shape == (2, 4, 32)  # three camera tokens plus one state token


def test_weighted_loss_padding_and_zero_boundary():
    prediction, target = torch.randn(2, 4, 3), torch.randn(2, 4, 3)
    ones = torch.ones(2, 4)
    expected = torch.nn.functional.mse_loss(prediction, target)
    assert torch.allclose(quality_weighted_action_loss(prediction, target, ones), expected)
    padding = torch.tensor([[False, False, True, True], [False, True, True, True]])
    loss = quality_weighted_action_loss(prediction, target, ones, padding)
    assert torch.isfinite(loss)
    zero = quality_weighted_action_loss(prediction, target, torch.zeros_like(ones))
    assert zero.item() == 0 and torch.isfinite(zero)


def test_checkpoint_roundtrip_outputs_match(tmp_path):
    torch.manual_seed(2)
    model = BCMLP(6, 3)
    state = torch.randn(2, 6)
    expected = model(state).detach()
    optimizer = torch.optim.Adam(model.parameters())
    path = save_checkpoint(tmp_path / "model.pt", model, optimizer, None, 4, 1, {"name": "bc"})
    restored = BCMLP(6, 3)
    payload = load_checkpoint(path, restored)
    assert payload["step"] == 4
    assert payload["metadata"]["checkpoint_schema_version"] == 2
    compatible, reason, _ = inspect_checkpoint(path, expected_model=restored)
    assert compatible, reason
    assert torch.equal(expected, restored(state).detach())


def test_temporal_ensemble_uses_overlapping_predictions():
    ensemble = TemporalEnsembler(decay=0.5)
    ensemble.add(0, torch.tensor([[0.0], [2.0]]))
    ensemble.add(1, torch.tensor([[4.0], [6.0]]))
    assert torch.allclose(ensemble.action(1), torch.tensor([10 / 3]))
