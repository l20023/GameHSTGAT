import pytest
import torch

from src.models import RecurrentGATAgent, SharedSequentialLoss


def _make_complete_edge_index(num_nodes: int) -> torch.Tensor:
    adjacency = torch.ones((num_nodes, num_nodes), dtype=torch.long) - torch.eye(
        num_nodes, dtype=torch.long
    )
    return adjacency.nonzero(as_tuple=False).t().contiguous()


def test_forward_returns_expected_shape() -> None:
    num_nodes = 10
    max_horizon = 6
    edge_index = _make_complete_edge_index(num_nodes)
    x_sequences = torch.rand((num_nodes, max_horizon, 1), dtype=torch.float32)

    model = RecurrentGATAgent(num_features_signal=1, hidden_dim=32, num_heads=2)
    logits = model(x_sequences, edge_index, max_horizon=max_horizon)
    assert logits.shape == (max_horizon, num_nodes, 2)


def test_shared_sequential_loss_is_scalar_and_backward_works() -> None:
    num_nodes = 8
    max_horizon = 5
    edge_index = _make_complete_edge_index(num_nodes)
    x_sequences = torch.rand((num_nodes, max_horizon, 1), dtype=torch.float32)
    targets = torch.randint(low=0, high=2, size=(num_nodes,), dtype=torch.long)

    model = RecurrentGATAgent(num_features_signal=1, hidden_dim=32, num_heads=2)
    criterion = SharedSequentialLoss()
    logits = model(x_sequences, edge_index, max_horizon=max_horizon)
    loss = criterion(logits, targets)
    assert loss.ndim == 0

    loss.backward()
    grad_norm = sum(
        p.grad.norm().item()
        for p in model.parameters()
        if p.grad is not None and p.requires_grad
    )
    assert grad_norm > 0.0


def test_loss_accepts_scalar_target_and_expands_to_num_nodes() -> None:
    num_nodes = 7
    max_horizon = 4
    edge_index = _make_complete_edge_index(num_nodes)
    x_sequences = torch.rand((num_nodes, max_horizon, 1), dtype=torch.float32)

    model = RecurrentGATAgent(num_features_signal=1, hidden_dim=32, num_heads=2)
    criterion = SharedSequentialLoss()
    logits = model(x_sequences, edge_index, max_horizon=max_horizon)

    scalar_target = torch.tensor(1, dtype=torch.long)
    loss = criterion(logits, scalar_target)
    assert torch.isfinite(loss)
    assert loss.item() >= 0.0


@pytest.mark.parametrize(
    ("hidden_dim", "num_heads"),
    [
        (31, 2),
        (32, 0),
    ],
)
def test_model_invalid_constructor_params_raise(hidden_dim: int, num_heads: int) -> None:
    with pytest.raises(ValueError):
        RecurrentGATAgent(num_features_signal=1, hidden_dim=hidden_dim, num_heads=num_heads)


def test_model_invalid_communication_params_raise() -> None:
    with pytest.raises(ValueError):
        RecurrentGATAgent(
            num_features_signal=1,
            hidden_dim=32,
            num_heads=2,
            communication_mode="invalid",
        )
    with pytest.raises(ValueError):
        RecurrentGATAgent(
            num_features_signal=1,
            hidden_dim=32,
            num_heads=2,
            communication_mode="fair_1bit",
            communication_dim=2,
        )
    with pytest.raises(ValueError):
        RecurrentGATAgent(
            num_features_signal=1,
            hidden_dim=32,
            num_heads=2,
            communication_mode="vector",
            communication_dim=0,
        )


def test_forward_rejects_invalid_horizon() -> None:
    num_nodes = 5
    edge_index = _make_complete_edge_index(num_nodes)
    x_sequences = torch.rand((num_nodes, 3, 1), dtype=torch.float32)
    model = RecurrentGATAgent(num_features_signal=1, hidden_dim=32, num_heads=2)

    with pytest.raises(ValueError):
        model(x_sequences, edge_index, max_horizon=4)


def test_loss_rejects_incompatible_targets_shape() -> None:
    criterion = SharedSequentialLoss()
    logits = torch.rand((3, 5, 2), dtype=torch.float32)
    bad_targets = torch.ones((5, 1), dtype=torch.long)

    with pytest.raises(ValueError):
        criterion(logits, bad_targets)


def test_forward_does_not_use_future_signals_for_earlier_predictions() -> None:
    torch.manual_seed(7)
    num_nodes = 6
    horizon = 5
    split_t = 2  # rounds 0..2 must be invariant
    edge_index = _make_complete_edge_index(num_nodes)
    x_sequences = torch.rand((num_nodes, horizon, 1), dtype=torch.float32)

    model = RecurrentGATAgent(num_features_signal=1, hidden_dim=32, num_heads=2)
    model.eval()
    with torch.no_grad():
        logits_reference = model(x_sequences, edge_index, max_horizon=horizon)

        x_modified = x_sequences.clone()
        x_modified[:, split_t + 1 :, :] = 1.0 - x_modified[:, split_t + 1 :, :]
        logits_modified = model(x_modified, edge_index, max_horizon=horizon)

    assert torch.allclose(
        logits_reference[: split_t + 1],
        logits_modified[: split_t + 1],
        atol=1e-7,
        rtol=0.0,
    )


def test_forward_vector_mode_returns_expected_shape() -> None:
    num_nodes = 9
    horizon = 4
    edge_index = _make_complete_edge_index(num_nodes)
    x_sequences = torch.rand((num_nodes, horizon, 1), dtype=torch.float32)
    model = RecurrentGATAgent(
        num_features_signal=1,
        hidden_dim=32,
        num_heads=2,
        communication_mode="vector",
        communication_dim=5,
    )
    logits = model(x_sequences, edge_index, max_horizon=horizon)
    assert logits.shape == (horizon, num_nodes, 2)
