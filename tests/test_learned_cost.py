"""Test the linear-vs-learned cost-model comparison (R3-O4/O5)."""

import importlib.util
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


def test_gbdt_and_linear_cost_models_fit():
    import numpy as np
    lc = _load("benchmark_learned_cost")
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 4))
    y = 2.0 + X[:, 0] * 1.5 - X[:, 1] * 0.5 + rng.normal(0, 0.05, size=200)
    lin = lc._fit_linear(X, y, interact=False)
    gb = lc._fit_gbdt(X, y)
    r2_lin = lc._r2(y, lc._pred_linear(lin, X))
    r2_gb = lc._r2(y, lc._pred_gbdt(gb, X))
    # both should fit a near-linear target well
    assert r2_lin > 0.9
    assert r2_gb > 0.8
