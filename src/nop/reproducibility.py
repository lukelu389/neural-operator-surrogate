"""Seed the random generators used by experiments and DataLoader workers.

Call ``seed_everything`` before constructing models or data loaders. Pass
``seed_worker`` as the DataLoader's ``worker_init_fn`` and use a dedicated
``torch.Generator().manual_seed(seed)`` for reproducible sampling and worker seeds.
Recreate that generator for each replay of an experiment.

Determinism is limited to the same software versions, hardware, and execution
path. Unsupported deterministic torch operations raise instead of silently
falling back. CUDA operations may also require a compatible
``CUBLAS_WORKSPACE_CONFIG`` at process launch. Python's ``PYTHONHASHSEED`` must
likewise be set before interpreter startup; these helpers do not change it.
Independent ``numpy.random.Generator`` objects must be seeded by their callers.
"""

import random

import numpy as np
import torch

_SEED_MODULUS = 2**32


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy's global RNG, and torch on CPU and available devices.

    ``seed`` must be a Python integer in ``[0, 2**32 - 1]``, the common range
    accepted by these RNGs. Booleans are rejected. This changes process-global
    RNG state and enables strict deterministic torch algorithms.
    """
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    if not 0 <= seed < _SEED_MODULUS:
        raise ValueError("seed must be in [0, 2**32 - 1]")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def seed_worker(worker_id: int) -> None:
    """Initialize Python and NumPy from a DataLoader worker's torch seed.

    DataLoader has already assigned a distinct torch seed to each worker; its
    full-width seed is preserved. The worker ID is supplied by DataLoader and
    does not replace the assigned seed. Restart loaders/generators to replay a
    run; persistent workers naturally continue their RNG streams across epochs.
    """
    if isinstance(worker_id, bool) or not isinstance(worker_id, int):
        raise TypeError("worker_id must be an integer")
    if worker_id < 0:
        raise ValueError("worker_id must be nonnegative")

    worker_seed = torch.initial_seed() % _SEED_MODULUS
    random.seed(worker_seed)
    np.random.seed(worker_seed)
