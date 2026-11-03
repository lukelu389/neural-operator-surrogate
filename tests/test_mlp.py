from pathlib import Path

import pytest
import torch
import yaml
from torch import nn

from nop.models.mlp import MLP
from nop.reproducibility import seed_everything


def _moments(dtype: torch.dtype = torch.float32) -> tuple[torch.Tensor, torch.Tensor]:
    mean = torch.linspace(0.15, 0.55, 6, dtype=dtype).reshape(2, 3)
    std = torch.linspace(0.04, 0.12, 6, dtype=dtype).reshape(2, 3)
    return mean, std


def _model(dtype: torch.dtype = torch.float32) -> MLP:
    return MLP((7, 6), *_moments(dtype))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_initial_predictions_equal_training_means(dtype: torch.dtype) -> None:
    model = _model(dtype)
    prediction = model(torch.randn(4, 5, dtype=dtype))
    iv = prediction * model.target_std + model.target_mean
    assert prediction.shape == (4, 2, 3)
    assert prediction.dtype == dtype
    assert (iv > 0).all()
    torch.testing.assert_close(iv, model.target_mean.expand(4, -1, -1))
    torch.testing.assert_close(prediction, torch.zeros_like(prediction), atol=1e-6, rtol=0.0)
    assert torch.count_nonzero(model.output.weight) == 0


def test_output_is_softplus_decimal_iv_then_standardized_without_clipping() -> None:
    model = _model(torch.float64)
    scores = torch.tensor([-8.0, -3.0, -1.0, 0.0, 1.0, 25.0], dtype=torch.float64)
    with torch.no_grad():
        model.output.bias.copy_(scores)
    prediction = model(torch.zeros(2, 5, dtype=torch.float64))
    iv = prediction * model.target_std + model.target_mean
    expected = torch.nn.functional.softplus(scores).reshape(2, 3).expand(2, -1, -1)
    torch.testing.assert_close(iv, expected)
    assert (iv > 0).all()
    assert prediction.min() < 0
    assert iv.max() > 20  # No arbitrary ceiling on decimal IV.


def test_softplus_underflow_is_not_silently_floored() -> None:
    model = MLP((3,), torch.ones(1, 1), torch.ones(1, 1))
    with torch.no_grad():
        model.output.bias.fill_(-1000.0)
    prediction = model(torch.zeros(1, 5))
    assert (prediction * model.target_std + model.target_mean).item() == 0.0


def test_declared_architecture_shape_and_parameter_budget() -> None:
    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load((root / "config/m0.yaml").read_text())
    protocol = yaml.safe_load((root / "config/training.yaml").read_text())
    shape = tuple(config["native_grid_shape"])
    model = MLP(tuple(config["hidden_widths"]), torch.full(shape, 0.3), torch.full(shape, 0.1))
    count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    nominal = protocol["nominal_parameters"]
    assert config["model_id"] == "M0"
    assert config["activation"] == "SiLU"
    assert config["input_features"] == 5
    assert config["hidden_widths"] == [560, 560, 560, 560]
    assert config["native_grid_shape"] == [8, 15]
    assert config["nominal_parameters"] == nominal
    assert count == config["trainable_parameters"] == 1_013_160
    assert abs(count / nominal - 1.0) <= protocol["max_relative_parameter_difference_from_M0"]
    assert sum(isinstance(layer, nn.SiLU) for layer in model.hidden) == 4
    assert model(torch.zeros(2, 5)).shape == (2, 8, 15)


def test_fixed_seed_reproduces_initialization() -> None:
    seed_everything(11)
    first = _model()
    seed_everything(11)
    repeated = _model()
    seed_everything(29)
    different = _model()
    for name, value in first.state_dict().items():
        torch.testing.assert_close(value, repeated.state_dict()[name], rtol=0.0, atol=0.0)
    assert not torch.equal(first.hidden[0].weight, different.hidden[0].weight)


def test_input_first_and_second_derivatives_match_finite_differences() -> None:
    seed_everything(11)
    model = _model(torch.float64)
    # At mean initialization input derivatives are zero. Nonzero head weights
    # exercise the learned function's derivatives without training on labels.
    with torch.no_grad():
        model.output.weight.normal_(std=0.2)
    parameters = torch.randn(2, 5, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(model, (parameters,))
    assert torch.autograd.gradgradcheck(model, (parameters,))
    first = torch.autograd.grad(model(parameters).sum(), parameters, create_graph=True)[0]
    second = torch.autograd.grad(first.sum(), parameters)[0]
    assert torch.isfinite(first).all() and torch.isfinite(second).all()
    assert first.abs().max() > 0 and second.abs().max() > 0


def test_trainable_parameters_receive_gradients_after_head_update() -> None:
    seed_everything(11)
    model = _model()
    parameters = torch.randn(3, 5)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    for _ in range(2):
        optimizer.zero_grad()
        (model(parameters) - 1).square().mean().backward()
        optimizer.step()
    for parameter in model.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().max() > 0


def test_moments_are_owned_buffers_preserved_by_serialization(tmp_path: Path) -> None:
    mean, std = _moments()
    model = MLP((7, 6), mean, std)
    expected_mean, expected_std = mean.clone(), std.clone()
    mean.zero_()
    std.zero_()
    assert set(dict(model.named_buffers())) == {"target_mean", "target_std"}
    assert not model.target_mean.requires_grad and not model.target_std.requires_grad
    torch.testing.assert_close(model.target_mean, expected_mean)
    torch.testing.assert_close(model.target_std, expected_std)
    path = tmp_path / "state.pt"
    torch.save(model.state_dict(), path)
    restored = MLP((7, 6), expected_mean * 2, expected_std * 2)
    restored.load_state_dict(torch.load(path, weights_only=True))
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, restored.state_dict()[name], rtol=0.0, atol=0.0)
    inputs = torch.randn(3, 5)
    torch.testing.assert_close(model(inputs), restored(inputs), rtol=0.0, atol=0.0)
    restored.double()
    assert restored.target_mean.dtype == restored.target_std.dtype == torch.float64
    assert restored(inputs.double()).dtype == torch.float64


@pytest.mark.parametrize("widths", [(), (0,), (-1,), (True,), (1.5,)])
def test_invalid_widths_rejected(widths: tuple[int, ...]) -> None:
    with pytest.raises(ValueError, match="positive integers"):
        MLP(widths, *_moments())


@pytest.mark.parametrize("which", [0, 1])
@pytest.mark.parametrize("value", [0.0, -1.0, float("nan"), float("inf")])
def test_invalid_target_moments_rejected(which: int, value: float) -> None:
    moments = list(_moments())
    moments[which][0, 0] = value
    with pytest.raises(ValueError, match="finite and positive"):
        MLP((7,), *moments)


def test_moment_shape_dtype_and_gradient_mismatches_rejected() -> None:
    mean, std = _moments()
    with pytest.raises(ValueError, match=r"shape \[T,Z\]"):
        MLP((7,), mean.flatten(), std.flatten())
    with pytest.raises(ValueError, match=r"shape \[T,Z\]"):
        MLP((7,), mean[:0], std[:0])
    with pytest.raises(ValueError, match="same shape"):
        MLP((7,), mean, std[:1])
    with pytest.raises(TypeError, match="float32 or float64"):
        MLP((7,), mean.half(), std.half())
    with pytest.raises(ValueError, match="share dtype and device"):
        MLP((7,), mean, std.double())
    with pytest.raises(ValueError, match="must not require gradients"):
        MLP((7,), mean.requires_grad_(), std)


@pytest.mark.parametrize("shape", [(5,), (2, 4), (2, 5, 1)])
def test_wrong_input_shape_rejected(shape: tuple[int, ...]) -> None:
    with pytest.raises(ValueError, match=r"shape \[N,5\]"):
        _model()(torch.zeros(shape))


def test_nonfinite_or_wrong_dtype_input_rejected() -> None:
    with pytest.raises(ValueError, match="must be finite"):
        _model()(torch.full((2, 5), float("nan")))
    with pytest.raises(ValueError, match="share the model's dtype and device"):
        _model()(torch.zeros(2, 5, dtype=torch.float64))
