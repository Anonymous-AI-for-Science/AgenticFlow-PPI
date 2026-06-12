"""Dispatch ablation, plan-selection, and gain-predictor calibration.

All experiments use the shared pathway-grounded harness (design rationale). With
independent labels the learned reranker does NOT beat symbolic ranking on this
workload, so the systems value of cost-aware dispatch is precisely that it
SUPPRESSES an operator that would otherwise cost quality and latency. This script
shows that, isolates the dispatch policy from the reranker (design rationale), and
calibrates the gain predictor against realized F1 lift (design rationale).
"""

from __future__ import annotations

import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np

from agentflow_ppi.eval.harness import (
    build_harness, train_reranker, rerank, symbolic_order, f1_at_k,
    expected_gain, train_gain_predictor, predict_gain,
    SEED_MANIFEST, MODALITY_FEATURE_IDX,
)

SYMBOLIC_MS = 0.05
RERANK_MS = 0.9
FRONTIER_BUDGET = 50
# Uncalibrated heuristic floor (modality-ambiguity proxy); kept for the ablation.
GAIN_FLOOR = 0.30


def main():
    root = Path(__file__).resolve().parents[1]
    out = root / "results"; out.mkdir(parents=True, exist_ok=True)
    h = build_harness()

    policies = ["never-on", "always-on", "frontier-only",
                "gain-only (uncalibrated)", "cost-aware (calibrated)"]
    pol = {p: {"f1": [], "ms": [], "calls": []} for p in policies}
    flips = {"total": 0, "flips": 0, "saved_calls": 0, "f1_delta_sum": 0.0,
             "ms_saved_sum": 0.0, "beneficial_flips": 0}
    calib_pred, calib_real = [], []

    for seed in SEED_MANIFEST:
        rng = np.random.default_rng(seed)
        idx = list(range(len(h.pools))); rng.shuffle(idx)
        nt = max(1, int(round(0.2 * len(idx))))
        test = set(idx[:nt]); train = [i for i in idx if i not in test]
        model = train_reranker(h, train, seed)
        predictor = train_gain_predictor(h, train, seed)
        if model is None or predictor is None:
            continue
        for i in test:
            p = h.pools[i]; fr = len(p.cands)
            g_unc = expected_gain(h, p.s, p.t, p.modality, p.cands)
            g_cal = predict_gain(h, predictor, p)
            r_rank = rerank(h, model, p.s, p.t, p.modality, p.cands)
            s_rank = symbolic_order(h, p.s, p.t, p.cands)
            r_f1, s_f1 = f1_at_k(r_rank, p.positives), f1_at_k(s_rank, p.positives)
            r_ms, s_ms = SYMBOLIC_MS * fr + RERANK_MS, SYMBOLIC_MS * fr
            calib_pred.append(g_cal); calib_real.append(r_f1 - s_f1)

            def rec(name, admit):
                pol[name]["f1"].append(r_f1 if admit else s_f1)
                pol[name]["ms"].append(r_ms if admit else s_ms)
                pol[name]["calls"].append(1 if admit else 0)

            rec("never-on", False)
            rec("always-on", True)
            rec("frontier-only", fr <= FRONTIER_BUDGET)
            rec("gain-only (uncalibrated)", g_unc >= GAIN_FLOOR)
            ca = (fr <= FRONTIER_BUDGET) and (g_cal > 0.0)  # calibrated: admit iff predicted lift positive
            rec("cost-aware (calibrated)", ca)

            flips["total"] += 1
            if not ca:
                flips["flips"] += 1; flips["saved_calls"] += 1
                flips["f1_delta_sum"] += (s_f1 - r_f1)
                flips["ms_saved_sum"] += (r_ms - s_ms)
                if s_f1 >= r_f1:
                    flips["beneficial_flips"] += 1

    rows = []
    for p in policies:
        d = pol[p]; ms = sorted(d["ms"])
        rows.append({"policy": p, "macro_f1_at_2": round(statistics.mean(d["f1"]), 4),
                     "mean_latency_ms": round(statistics.mean(d["ms"]), 4),
                     "p95_latency_ms": round(ms[int(0.95 * len(ms)) - 1], 4),
                     "reranker_call_rate": round(statistics.mean(d["calls"]), 4)})
    with (out / "dispatch_policy_ablation.csv").open("w", newline="") as f:
        wtr = csv.DictWriter(f, fieldnames=list(rows[0].keys())); wtr.writeheader(); wtr.writerows(rows)

    ps = {
        "label_source": "independent canonical-pathway membership",
        "total_test_queries": flips["total"],
        "route_flips_vs_always_on": flips["flips"],
        "flip_rate": round(flips["flips"] / max(flips["total"], 1), 4),
        "flips_that_helped_or_held_quality": flips["beneficial_flips"],
        "reranker_calls_saved": flips["saved_calls"],
        "mean_f1_delta_per_flip": round(flips["f1_delta_sum"] / max(flips["flips"], 1), 4),
        "total_latency_saved_ms": round(flips["ms_saved_sum"], 4),
        "interpretation": "With independent labels the reranker does not beat symbolic ranking, "
                          "so suppression generally PRESERVES quality while saving cost; a positive "
                          "mean_f1_delta_per_flip means cost-aware dispatch improves quality over always-on.",
    }
    (out / "plan_selection_disagreement.json").write_text(json.dumps(ps, indent=2))

    # Gain-predictor calibration (design rationale).
    cp = np.array(calib_pred); cr = np.array(calib_real)
    if len(cp) >= 3 and np.std(cp) > 1e-9 and np.std(cr) > 1e-9:
        pear = float(np.corrcoef(cp, cr)[0, 1])
    else:
        pear = float("nan")
    calib = {
        "n": int(len(cp)),
        "pearson_pred_gain_vs_realized_lift": round(pear, 4) if pear == pear else None,
        "mean_predicted_gain": round(float(np.mean(cp)), 4),
        "mean_realized_lift": round(float(np.mean(cr)), 4),
        "note": "Calibration of the gain proxy against realized rerank-minus-symbolic F1. "
                "A weak/negative correlation explains why the reranker should usually be suppressed.",
    }
    (out / "gain_predictor_calibration.json").write_text(json.dumps(calib, indent=2))

    for r in rows:
        print(r)
    print(json.dumps(ps, indent=2))
    print(json.dumps(calib, indent=2))


if __name__ == "__main__":
    main()
