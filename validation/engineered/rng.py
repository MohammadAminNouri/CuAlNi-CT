from __future__ import annotations

import hashlib
import numpy as np


def seed64(namespace: str, index: int) -> int:
    payload = f"CuAlNi-CT-engineered-v2|{namespace}|{index}".encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "little", signed=False)


def rng(namespace: str, index: int) -> np.random.Generator:
    return np.random.default_rng(seed64(namespace, index))
