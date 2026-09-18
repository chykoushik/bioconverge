import os

for name in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"]:
    os.environ[name] = "1"

import numpy as np
import pandas as pd
import pytest
import requests
from threadpoolctl import threadpool_limits


@pytest.fixture(autouse=True)
def isolate_runtime(monkeypatch):
    def blocked(*args, **kwargs):
        raise AssertionError("unmocked network request")
    monkeypatch.setattr(requests.sessions.Session, "request", blocked)
    with threadpool_limits(limits=1):
        yield


@pytest.fixture
def scores():
    rng = np.random.default_rng(18)
    return pd.DataFrame({"patient_id": [f"P{i}" for i in range(24)],
                         "a": rng.normal(size=24), "b": rng.normal(size=24), "c": rng.normal(size=24)})
