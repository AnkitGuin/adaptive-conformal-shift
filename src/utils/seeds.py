"""Explicit, hierarchical seed management.

No module in this repository touches NumPy's global random state.  Instead,
every stochastic component receives its own ``numpy.random.Generator`` (or an
integer seed for scikit-learn) derived deterministically from the experiment
seed and a string key describing the component, e.g.::

    rng = make_rng(seed, "shift", "noise")
    model_seed = derive_seed(seed, "model", "gbr")

Deriving streams this way means that adding a new random component does not
change the random numbers drawn by existing ones.
"""
from __future__ import annotations

import zlib

import numpy as np


def _key_to_int(key: str) -> int:
    return zlib.crc32(key.encode("utf-8")) & 0xFFFFFFFF


def derive_seed(seed: int, *keys: str) -> int:
    """Deterministically derive a 32-bit integer seed from ``seed`` and string keys."""
    entropy = [int(seed) & 0xFFFFFFFF] + [_key_to_int(k) for k in keys]
    ss = np.random.SeedSequence(entropy)
    return int(ss.generate_state(1, dtype=np.uint32)[0])


def make_rng(seed: int, *keys: str) -> np.random.Generator:
    """Return an independent ``numpy.random.Generator`` for (seed, *keys)."""
    return np.random.default_rng(derive_seed(seed, *keys))
