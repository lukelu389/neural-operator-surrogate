"""Small synthetic stream slices; no research dataset is materialized."""

import hashlib
import math

import pytest
import torch

from nop.data.sampling import sample_id_parameters

BOUNDS = ((0.01, 0.25), (0.5, 5.0), (0.01, 0.25), (0.1, 1.0), (-0.95, -0.1))


def _digest(value: torch.Tensor) -> str:
    return hashlib.sha256(value.numpy().tobytes()).hexdigest()


def test_reproducible_hashes_and_distinct_seeds() -> None:
    first = sample_id_parameters(32, 0, 7, BOUNDS)
    second = sample_id_parameters(32, 0, 7, BOUNDS)
    assert _digest(first) == _digest(second)
    assert _digest(first) != _digest(sample_id_parameters(32, 0, 8, BOUNDS))
    assert first.dtype == torch.float64
    assert first.device.type == "cpu"
    assert first.shape == (32, 5)
    lower, upper = torch.tensor(BOUNDS, dtype=torch.float64).T
    assert ((first >= lower) & (first <= upper)).all()


def test_split_slices_are_disjoint_and_chunk_invariant() -> None:
    whole = sample_id_parameters(53, 0, 13, BOUNDS)
    train = sample_id_parameters(37, 0, 13, BOUNDS)
    validation = sample_id_parameters(16, 37, 13, BOUNDS)
    chunks = [
        sample_id_parameters(count, offset, 13, BOUNDS) for offset, count in ((0, 7), (7, 30))
    ]
    assert _digest(torch.cat(chunks)) == _digest(train)
    assert _digest(torch.cat((train, validation))) == _digest(whole)
    assert torch.equal(sample_id_parameters(9, 0, 13, BOUNDS), train[:9])
    assert torch.unique(torch.cat((train, validation)), dim=0).shape[0] == 53


def test_sample_is_the_native_scrambled_sobol_stream_without_feller_filtering() -> None:
    engine = torch.quasirandom.SobolEngine(5, scramble=True, seed=23)
    unit = engine.draw(32, dtype=torch.float64)
    lower, upper = torch.tensor(BOUNDS, dtype=torch.float64).T
    actual = sample_id_parameters(32, 0, 23, BOUNDS)
    assert torch.equal(actual, lower + (upper - lower) * unit)
    assert (2 * actual[:, 1] * actual[:, 2] < actual[:, 3].square()).any()


def test_sampling_does_not_consume_global_torch_rng() -> None:
    before = torch.random.get_rng_state().clone()
    sample_id_parameters(8, 19, 0, BOUNDS)
    assert torch.equal(before, torch.random.get_rng_state())


def test_empty_slice_has_full_schema() -> None:
    result = sample_id_parameters(0, 11, 0, BOUNDS)
    assert result.shape == (0, 5)
    assert result.dtype == torch.float64


@pytest.mark.parametrize("name", ["count", "offset", "seed"])
@pytest.mark.parametrize("value", [True, 1.5])
def test_integer_arguments_reject_coercion(name: str, value: object) -> None:
    arguments = {"count": 4, "offset": 0, "seed": 1}
    arguments[name] = value
    with pytest.raises(TypeError, match="integer"):
        sample_id_parameters(**arguments, bounds=BOUNDS)


@pytest.mark.parametrize("count,offset,seed", [(-1, 0, 0), (1, -1, 0), (1, 0, -1), (1, 0, 2**32)])
def test_invalid_ranges(count: int, offset: int, seed: int) -> None:
    with pytest.raises(ValueError):
        sample_id_parameters(count, offset, seed, BOUNDS)


def test_stream_capacity_is_checked_before_drawing() -> None:
    with pytest.raises(ValueError, match="capacity"):
        sample_id_parameters(2, 2**torch.quasirandom.SobolEngine.MAXBIT - 1, 0, BOUNDS)


@pytest.mark.parametrize(
    "bounds",
    [
        BOUNDS[:-1],
        ((0.0, 1.0, 2.0),) + BOUNDS[1:],
        ((0.0, 0.0),) + BOUNDS[1:],
        ((1.0, 0.0),) + BOUNDS[1:],
        ((math.nan, 1.0),) + BOUNDS[1:],
        ((0.0, math.inf),) + BOUNDS[1:],
        ((-1e308, 1e308),) + BOUNDS[1:],
    ],
)
def test_invalid_bounds(bounds: tuple[tuple[float, ...], ...]) -> None:
    with pytest.raises(ValueError, match="bound"):
        sample_id_parameters(4, 0, 0, bounds)
