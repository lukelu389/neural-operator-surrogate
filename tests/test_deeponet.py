from pathlib import Path

import pytest
import torch
import yaml

from nop.models.deeponet import DeepONet
from nop.train.coordinates import CoordinateNormalization


def _model(dtype: torch.dtype = torch.float32) -> DeepONet:
    z = torch.tensor([-1.0, 0.0, 1.0], dtype=dtype)
    T_axis = torch.tensor([0.5, 1.0], dtype=dtype)
    mean = torch.tensor([[0.2, 0.25, 0.3], [0.22, 0.27, 0.32]], dtype=dtype)
    std = torch.full_like(mean, 0.05)
    normalization = CoordinateNormalization(z, T_axis, mean, std, k_scale=0.2)
    native_T = T_axis[:, None].expand_as(mean)
    native_k = z[None, :] * 0.2 * native_T.sqrt()
    return DeepONet((8, 8), 4, (0.3, 1.0), normalization, native_k, native_T)


def test_native_shape_and_positive_decimal_iv() -> None:
    model = _model()
    state = model.state_dict()
    assert {"normalization_z_axis", "normalization_T_axis", "target_mean", "target_std"} <= set(
        state
    )
    prediction = model(torch.randn(5, 5))
    assert prediction.shape == (5, 2, 3)
    iv = prediction * model.coordinate_normalization.std + model.coordinate_normalization.mean
    assert torch.isfinite(iv).all()
    assert (iv > 0).all()


def test_arbitrary_query_shape_and_permutation_equivariance() -> None:
    model = _model()
    parameters = torch.randn(3, 5)
    k = torch.tensor([-0.03, 0.02, 0.06])
    T = torch.tensor([0.6, 0.8, 0.9])
    prediction = model.query(parameters, k, T)
    order = torch.tensor([2, 0, 1])
    reordered = model.query(parameters, k[order], T[order])
    assert prediction.shape == (3, 3)
    torch.testing.assert_close(reordered, prediction[:, order])


def test_coordinate_gradients_are_finite_and_nonzero() -> None:
    model = _model(torch.float64)
    parameters = torch.randn(2, 5, dtype=torch.float64, requires_grad=True)
    k = torch.tensor([-0.037, 0.041], dtype=torch.float64, requires_grad=True)
    T = torch.tensor([0.67, 0.83], dtype=torch.float64, requires_grad=True)
    output = model.query(parameters, k, T)
    grad = torch.autograd.grad(output.sum(), (parameters, k, T))
    assert all(torch.isfinite(value).all() and value.abs().max() > 0 for value in grad)


def test_parameter_count_within_shared_m0_budget() -> None:
    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load((root / "config/m1.yaml").read_text())
    protocol = yaml.safe_load((root / "config/training.yaml").read_text())
    z = torch.linspace(-2.5, 2.5, 15)
    T_axis = torch.tensor([1 / 12, 2 / 12, 3 / 12, 6 / 12, 9 / 12, 1.0, 1.5, 2.0])
    grid_T = T_axis[:, None].expand(8, 15)
    grid_z = z[None, :].expand(8, 15)
    grid_k = grid_z * config["output"]["normalization_k_scale"] * grid_T.sqrt()
    target_mean = torch.full((8, 15), 0.2)
    target_std = torch.full((8, 15), 0.1)
    normalization = CoordinateNormalization(
        z,
        T_axis,
        target_mean,
        target_std,
        k_scale=config["output"]["normalization_k_scale"],
    )
    model = DeepONet(
        tuple(config["hidden_widths"]),
        config["rank"],
        (
            config["trunk_coordinate_scales"]["k"],
            config["trunk_coordinate_scales"]["maturity_years"],
        ),
        normalization,
        grid_k,
        grid_T,
    )
    count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    m0_count = 1_013_160
    tolerance = protocol["max_relative_parameter_difference_from_M0"]
    assert count == config["parameter_count"] == 1_050_977
    assert abs(count / m0_count - 1.0) <= tolerance


@pytest.mark.parametrize(
    ("parameters", "k", "T", "message"),
    [
        (torch.zeros(2, 4), torch.zeros(3), torch.ones(3), r"shape \[N,5\]"),
        (torch.zeros(2, 5), torch.zeros(3), torch.zeros(3), "positive maturity"),
        (torch.zeros(2, 5), torch.zeros(3), torch.full((3,), float("nan")), "finite"),
    ],
)
def test_invalid_inputs_fail_explicitly(
    parameters: torch.Tensor, k: torch.Tensor, T: torch.Tensor, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _model().query(parameters, k, T)
