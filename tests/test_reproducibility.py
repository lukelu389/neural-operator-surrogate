"""Checks for the experiment and DataLoader-worker RNG contract."""

import os
import random
from collections.abc import Iterator

import numpy as np
import pytest
import torch

from nop.reproducibility import seed_everything, seed_worker


@pytest.fixture(autouse=True)
def restore_rng_state() -> Iterator[None]:
    """Keep process-global RNG and deterministic settings local to each test."""
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    cudnn_benchmark = torch.backends.cudnn.benchmark
    cudnn_deterministic = torch.backends.cudnn.deterministic
    yield
    random.setstate(python_state)
    np.random.set_state(numpy_state)
    torch.set_rng_state(torch_state)
    if cuda_states is not None:
        torch.cuda.set_rng_state_all(cuda_states)
    torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)
    torch.backends.cudnn.benchmark = cudnn_benchmark
    torch.backends.cudnn.deterministic = cudnn_deterministic


def draw_random_values() -> tuple[list[float], list[float], list[float]]:
    return (
        [random.random() for _ in range(8)],
        np.random.random(8).tolist(),
        torch.rand(8).tolist(),
    )


@pytest.mark.parametrize("seed", [0, 42, 2**32 - 1])
def test_seed_everything_repeats_all_rngs(seed: int) -> None:
    seed_everything(seed)
    first = draw_random_values()
    seed_everything(seed)
    assert draw_random_values() == first


def test_different_seeds_change_each_rng_stream() -> None:
    seed_everything(42)
    first = draw_random_values()
    seed_everything(43)
    second = draw_random_values()
    assert all(left != right for left, right in zip(first, second, strict=True))


def test_seed_everything_enforces_strict_torch_determinism() -> None:
    torch.use_deterministic_algorithms(False)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    seed_everything(42)
    assert torch.are_deterministic_algorithms_enabled()
    assert not torch.is_deterministic_algorithms_warn_only_enabled()
    assert not torch.backends.cudnn.benchmark
    assert torch.backends.cudnn.deterministic


@pytest.mark.parametrize("seed", [-1, 2**32])
def test_seed_everything_rejects_out_of_range_seed(seed: int) -> None:
    with pytest.raises(ValueError, match="seed must be in"):
        seed_everything(seed)


@pytest.mark.parametrize("seed", [True, 1.5, "42", None])
def test_seed_everything_rejects_non_integer_seed(seed: object) -> None:
    with pytest.raises(TypeError, match="seed must be an integer"):
        seed_everything(seed)  # type: ignore[arg-type]


def test_seed_everything_preserves_hash_seed_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PYTHONHASHSEED", "13")
    seed_everything(42)
    assert os.environ["PYTHONHASHSEED"] == "13"
    monkeypatch.delenv("PYTHONHASHSEED")
    seed_everything(42)
    assert "PYTHONHASHSEED" not in os.environ


def test_worker_seed_uses_assigned_torch_seed_and_preserves_torch_stream() -> None:
    # A 64-bit DataLoader seed must fit NumPy's range without reducing torch's seed.
    assigned_seed = 2**32 + 47
    torch.manual_seed(assigned_seed)
    torch_state = torch.get_rng_state().clone()
    seed_worker(3)

    assert torch.initial_seed() == assigned_seed
    assert torch.equal(torch.get_rng_state(), torch_state)
    assert random.random() == random.Random(47).random()
    assert np.random.random() == np.random.RandomState(47).random_sample()


def test_worker_streams_are_distinct_and_replayable() -> None:
    streams = []
    for worker_id in range(2):
        torch.manual_seed(2**40 + 19 + worker_id)
        seed_worker(worker_id)
        streams.append(draw_random_values())

    assert all(left != right for left, right in zip(*streams, strict=True))
    for worker_id, expected in enumerate(streams):
        torch.manual_seed(2**40 + 19 + worker_id)
        seed_worker(worker_id)
        assert draw_random_values() == expected


def test_worker_seed_rejects_negative_worker_id() -> None:
    with pytest.raises(ValueError, match="worker_id must be nonnegative"):
        seed_worker(-1)


@pytest.mark.parametrize("worker_id", [True, 1.5, "0", None])
def test_worker_seed_rejects_non_integer_worker_id(worker_id: object) -> None:
    with pytest.raises(TypeError, match="worker_id must be an integer"):
        seed_worker(worker_id)  # type: ignore[arg-type]
