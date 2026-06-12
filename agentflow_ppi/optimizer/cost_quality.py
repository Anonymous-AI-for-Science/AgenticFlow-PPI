"""Learned cost and quality models for the optimizer.

From the (query, plan) -> (cost_ms, quality) traces we fit two ridge regressors:
  * a COST model    predicting plan execution latency, and
  * a QUALITY model predicting plan F1@k.

The optimizer uses these predictors to choose a plan per query WITHOUT executing
every plan -- i.e. it performs plan selection from predicted cost/quality under a
latency budget, which is exactly the optimization Eq. (1) describes and which the
prior 1-D threshold did not implement.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from .trace import Trace, plan_query_features


def _expand_features(X: np.ndarray) -> np.ndarray:
    """Add pairwise interaction terms so a linear model can capture query x plan
    effects (e.g. 'reranking helps only when modality mismatch is high'). This is
    what lets the quality predictor move beyond a near-constant fit."""
    n, d = X.shape
    cols = [X]
    # squared terms + pairwise products (bounded d keeps this cheap)
    for i in range(d):
        cols.append((X[:, i:i+1]) ** 2)
        for j in range(i + 1, d):
            cols.append(X[:, i:i+1] * X[:, j:j+1])
    return np.hstack(cols)


@dataclass
class Model:
    w: np.ndarray
    mu: np.ndarray
    sd: np.ndarray
    expand: bool = False

    def predict(self, x: np.ndarray) -> float:
        xx = _expand_features(x.reshape(1, -1))[0] if self.expand else x
        xs = np.append((xx - self.mu) / self.sd, 1.0)
        return float(xs @ self.w)


def _fit_ridge(X: np.ndarray, y: np.ndarray, lam: float = 1.0, expand: bool = False) -> Model:
    Xe = _expand_features(X) if expand else X
    mu, sd = Xe.mean(0), Xe.std(0) + 1e-9
    Xs = np.hstack([(Xe - mu) / sd, np.ones((len(Xe), 1))])
    w = np.linalg.solve(Xs.T @ Xs + lam * np.eye(Xs.shape[1]), Xs.T @ y)
    return Model(w=w, mu=mu, sd=sd, expand=expand)


def train_cost_quality(h, traces: List[Trace]):
    """Fit (cost_model, quality_model) from traces. Cost is near-linear in the plan,
    so it uses the plain features; quality has query x plan interactions, so it uses
    the expanded feature map (design rationale)."""
    X = np.array([plan_query_features(h, t.qid, t.plan) for t in traces])
    cost_model = _fit_ridge(X, np.array([t.cost_ms for t in traces]), expand=False)
    quality_model = _fit_ridge(X, np.array([t.quality for t in traces]), lam=2.0, expand=True)
    return cost_model, quality_model


def model_diagnostics(h, traces: List[Trace], cost_model: Model, quality_model: Model):
    """Held-out-style fit diagnostics (R^2 on the provided traces)."""
    X = np.array([plan_query_features(h, t.qid, t.plan) for t in traces])
    cy = np.array([t.cost_ms for t in traces]); qy = np.array([t.quality for t in traces])
    cp = np.array([cost_model.predict(x) for x in X])
    qp = np.array([quality_model.predict(x) for x in X])
    def r2(y, p):
        ss_res = float(np.sum((y - p) ** 2)); ss_tot = float(np.sum((y - y.mean()) ** 2))
        return 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
    return {"cost_r2": round(r2(cy, cp), 3), "quality_r2": round(r2(qy, qp), 3)}
