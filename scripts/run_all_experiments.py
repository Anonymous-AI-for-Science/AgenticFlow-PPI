#!/usr/bin/env python3
"""Run every AgentFlow-PPI v1 experiment and regenerate all result CSVs/JSONs.

Order matters: scripts that own a given CSV are run last so no stale generator
overwrites a measured result. Running this script reproduces every number in the
paper from scratch on commodity hardware.
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

SCRIPTS = [
    "collect_dataset_metrics.py",
    "benchmark_reachability.py",
    "benchmark_aorm_bulk.py",
    "benchmark_cost_model.py",
    "generate_revision_response_results.py",   # measured SHRC component ablation
    "benchmark_string_scale.py",               # 5k/10k/20k scale
    "benchmark_largescale_shrc.py",            # 50k-1M scale: SHRC tracks the core, not |V| (R3-O3)
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
    "benchmark_llm_multiagent.py",             # Ollama LLM multi-agent collaboration (R1-O1.A)
    "benchmark_pipeline_baselines.py",         # system-level fixed-order/agentic/agentflow at scale (R1-O2.A/B)
    "benchmark_published_index.py",            # published GRAIL/PLL C++ vs BFS oracle (R1-O2.A)
    "benchmark_plan_flips.py",                 # cost objective selects plans a fixed order would not (R1-O3.A)
    "benchmark_generality.py",                 # SHRC+dispatch on non-PPI graphs + DBMS integration (R3-O2)
    "benchmark_learned_cost.py",               # linear vs Bao/Lero-style learned cost model (R3-O4/O5)
]


# --------------------------------------------------------------------------- #
# Lightweight, dependency-free progress reporting for the orchestrator.
# Each child script streams its own output; here we add a master stage tracker
# (banner + overall bar + per-stage timing) so the long full run is legible.
# --------------------------------------------------------------------------- #
_BOLD = "\033[1m"; _DIM = "\033[2m"; _GRN = "\033[32m"; _RED = "\033[31m"; _RST = "\033[0m"


def _fmt_hms(seconds: float) -> str:
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:d}:{s:02d}"


def _stage_banner(idx: int, total: int, name: str, started: float, width: int = 30) -> None:
    """Print a master progress bar + which stage is starting, to stderr."""
    done = idx - 1
    frac = done / total
    filled = int(width * frac)
    bar = "█" * filled + "░" * (width - filled)
    elapsed = time.time() - started
    eta = (elapsed / done * (total - done)) if done else 0.0
    print(f"\n{_BOLD}[{idx:2d}/{total}]{_RST} |{bar}| "
          f"{done}/{total} done  "
          f"{_DIM}[elapsed {_fmt_hms(elapsed)}"
          + (f", ETA {_fmt_hms(eta)}" if done else "") + f"]{_RST}",
          file=sys.stderr, flush=True)
    print(f"{_BOLD}▶ running {name}{_RST}", file=sys.stderr, flush=True)


def main():
    root = Path(__file__).resolve().parent
    total = len(SCRIPTS)
    started = time.time()
    timings = []  # (name, seconds, ok)
    print(f"{_BOLD}Running {total} experiment stages — child output streams below; "
          f"a master progress bar precedes each stage.{_RST}", file=sys.stderr, flush=True)

    for i, s in enumerate(SCRIPTS, 1):
        _stage_banner(i, total, s, started)
        t0 = time.time()
        # The external-manifest step uses --offline here so run_all is deterministic
        # and host-independent; run download_external_data.py + build_external_manifest.py
        # directly (without --offline) to use freshly downloaded data.
        extra = ["--offline"] if s == "build_external_manifest.py" else []
        # The large-scale sweep defaults to a 1M-node grid; keep run_all tractable
        # by running its quick (50k/100k) tier here. Run it directly for the full grid.
        if s == "benchmark_largescale_shrc.py":
            extra = ["--quick"]
        r = subprocess.run([sys.executable, str(root / s), *extra], cwd=str(root.parent))
        dt = time.time() - t0
        ok = r.returncode == 0
        timings.append((s, dt, ok))
        if ok:
            print(f"{_GRN}✓ {s} done in {_fmt_hms(dt)}{_RST}", file=sys.stderr, flush=True)
        else:
            print(f"{_RED}✗ FAILED: {s} (after {_fmt_hms(dt)}){_RST}", file=sys.stderr, flush=True)
            _print_summary(timings, total, started)
            sys.exit(1)

    _print_summary(timings, total, started)


def _print_summary(timings, total: int, started: float) -> None:
    """Per-stage timing table + grand total, to stderr."""
    print(f"\n{_BOLD}===== run summary ({len(timings)}/{total} stages) ====="
          f"{_RST}", file=sys.stderr, flush=True)
    slowest = sorted(timings, key=lambda x: -x[1])[:5]
    for name, dt, ok in timings:
        mark = f"{_GRN}✓{_RST}" if ok else f"{_RED}✗{_RST}"
        print(f"  {mark} {name:<40} {_fmt_hms(dt):>8}", file=sys.stderr, flush=True)
    print(f"{_BOLD}  total wall time: {_fmt_hms(time.time() - started)}{_RST}",
          file=sys.stderr, flush=True)
    if slowest and slowest[0][1] > 0:
        print(f"{_DIM}  slowest: "
              + ", ".join(f"{n} ({_fmt_hms(d)})" for n, d, _ in slowest if d > 0)
              + f"{_RST}", file=sys.stderr, flush=True)
    if all(ok for _, _, ok in timings):
        print(f"\n{_GRN}All experiments completed; results/ regenerated.{_RST}",
              file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
