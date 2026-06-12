#!/usr/bin/env python3
"""Run every AgentFlow-PPI v1 experiment and regenerate all result CSVs/JSONs.

Order matters: scripts that own a given CSV are run last so no stale generator
overwrites a measured result. Running this script reproduces every number in the
paper from scratch on commodity hardware.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

SCRIPTS = [
    "collect_dataset_metrics.py",
    "benchmark_reachability.py",
    "benchmark_aorm_bulk.py",
    "benchmark_cost_model.py",
    "generate_shrc_component_ablation.py",   # measured SHRC component ablation
    "benchmark_string_scale.py",               # 5k/10k/20k scale
    "benchmark_biological_queries.py",         # measured reranking F1
    "benchmark_multiagent.py",                 # measured multi-agent flow
    "benchmark_endtoend_baselines.py",         # production / agentic / learned-cost baselines
    "benchmark_dispatch_ablation.py",          # dispatch isolation + plan-selection
    "benchmark_robustness.py",                 # negative-pool / core-growth / sensitivity
    "benchmark_large_dispatch.py",             # 320-node mixed-sign dispatch (admit + decline)
    "build_external_manifest.py",              # external biological manifest (download or fixture)
    "benchmark_external_reranking.py",         # external reranking, protein/pathway-disjoint splits
    "benchmark_engine_baselines.py",           # production graph-engine baselines + oracle equivalence
    "benchmark_strong_baselines.py",           # GRAIL/PReaCH/PLL/LCR vs SHRC + ablation + failure modes
    "benchmark_optimizer_dispatch.py",         # plan-space optimizer: regret, budget violation, Pareto
    "benchmark_codesign_dispatch.py",          # O(1) label-only dispatch vs recompute (co-design)
    "benchmark_modality_labeling.py",          # fallback-free label-constrained reachability (MPL)
    "benchmark_llm_multiagent.py",             # Ollama LLM multi-agent collaboration (design rationale)
    "benchmark_pipeline_baselines.py",         # system-level fixed-order/agentic/agentflow at scale (design rationale)
    "benchmark_published_index.py",            # published GRAIL/PLL C++ vs BFS oracle (design rationale)
    "benchmark_plan_flips.py",                 # cost objective selects plans a fixed order would not (design rationale)
    "benchmark_generality.py",                 # SHRC+dispatch on non-PPI graphs + DBMS integration (design rationale)
    "benchmark_learned_cost.py",               # linear vs Bao/Lero-style learned cost model (design rationale)
]


def main():
    root = Path(__file__).resolve().parent
    for s in SCRIPTS:
        print(f"\n===== {s} =====")
        # The external-manifest step uses --offline here so run_all is deterministic
        # and host-independent; run download_external_data.py + build_external_manifest.py
        # directly (without --offline) to use freshly downloaded data.
        extra = ["--offline"] if s == "build_external_manifest.py" else []
        r = subprocess.run([sys.executable, str(root / s), *extra], cwd=str(root.parent))
        if r.returncode != 0:
            print(f"FAILED: {s}")
            sys.exit(1)
    print("\nAll experiments completed; results/ regenerated.")


if __name__ == "__main__":
    main()
