# AgentFlow-PPI code artifact

## Environment

- Python 3.13.5
- PyTorch 2.10.0+cpu
- PyTorch Geometric
- Ollama running locally for the planner
- Apple Silicon optional: M1/M2/M3 with the PyTorch `mps` backend enabled

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
pip install -e .
```

## Graph query pipeline

### Train on synthetic data

```bash
python scripts/train_model.py --num-graphs 128 --epochs 10
```

### Execute a query

```bash
python scripts/run_query.py \
  --query "Find proteins that bridge TP53 and BRCA1 through a regulatory path with confidence above 0.8" \
  --data-root /path/to/string/files
```

### Build the SHRC reachability index

```bash
python scripts/build_shrc_demo.py
```

The SHRC implementation lives in `agentflow_ppi/reachability/shrc.py` and provides:

- tree-core decomposition with directed refinement,
- interval labels for the sparse tree region,
- exact 2-hop labels for the augmented core, and
- greedy redundancy elimination for tree-to-core exit anchors under existential reachability semantics, and
- an auditable approximation fallback branch for oversized cores.

The released fallback threshold is `fallback_core_threshold=5000`. This is a
practical guardrail rather than a theoretical constant: it is chosen so that
all public workloads in the paper remain in exact mode while larger reruns can
explicitly opt into approximation. When the guardrail is crossed, the current
artifact records `approximate_core_used=1` and a conservative
`delta_bound <= epsilon * |C|`; the paper does not claim a validated recall/runtime
trade-off for that branch yet.

## Dispatch and reranker gating

The dispatcher implementation now exposes the paper's explicit reranker
admission thresholds in `agentflow_ppi/execution/cost_dispatcher.py`:

- `reranker_frontier_budget = 50`
- `reranker_gain_threshold = 0.05`

The helper `should_execute_neural(frontier_size, expected_gain)` mirrors the
pseudocode in the paper and keeps suppression logic auditable. In the released
artifact, the frontier budget is treated as a capacity guardrail, while the
gain threshold is the data-informed selectivity parameter chosen from the
validation replay recorded in `results/reranker_gate_sensitivity.csv`.

The released dispatcher also exposes the paper-level cost/benefit trade-off
through `mu_gain = 0.6`, which is the executable counterpart of the lambda
weight in Equation (1) after feature normalization.

## Apple-Silicon asynchronous R&D flow

The optional `agentflow_ppi/rdflow/` package implements a laptop-scale learned-operator
collaboration runtime for research automation.

Key modules:

- `rdflow/device.py`: MPS-aware device selection and memory telemetry
- `rdflow/messages.py`: typed work items and agent envelopes
- `rdflow/bus.py`: asynchronous queue-isolated message transport
- `rdflow/router.py`: batched neural priority router for agent selection
- `rdflow/coordinator.py`: orchestration across planner, theory, engineering, and evaluation agents

Run the demo:

```bash
python scripts/run_rdflow_demo.py \
  --request "Design proofs, code, and experiments for an MPS-accelerated learned-operator graph system."
```

The runtime automatically falls back to CPU if MPS is unavailable, which keeps
artifact evaluation reproducible on non-Mac machines.

## STRING loader notes

The loader supports version-aware parsing for current STRING v12.x file families:

- `protein.links*.txt.gz`
- `protein.physical.links*.txt.gz`
- `protein.info*.txt.gz`
- `protein.aliases*.txt.gz`
- `protein.sequences*.fa.gz`
- `protein.network.embeddings*.h5`
- `protein.sequence.embeddings*.h5`

Species-prefixed exports such as `9606.protein.links.detailed.v12.0.txt.gz` are also supported. The released paper evaluates a 5,024-node subset for reproducibility, but the ingestion path itself is designed for the full STRING v12 family (roughly 20k proteins before task-specific filtering).

## Benchmark and data-collection scripts

The revised artifact includes the following reproducibility scripts:

- `scripts/benchmark_reachability.py`: compares SHRC against online BFS, GRAIL-style, PLL-style, PReaCH-style, and TF-label-style reference baselines on sparse and adversarial DAGs.
- `scripts/collect_dataset_metrics.py`: reports `|V|`, `|E|`, residual core ratio `sigma`, exit-anchor width statistics, SHRC build time, and index size.
- `scripts/benchmark_cost_model.py`: records predicted versus realized intermediate sizes on a synthetic operator-state replay (separate from the biological test split) and emits correlation metrics.
- `scripts/benchmark_biological_queries.py`: runs the released biological query families over the reduced STRING-like workload and reports candidate counts, macro-F1, and reranker wall-clock time.
- `scripts/benchmark_core_scaling.py`: emits the synthetic residual-core sweep used to justify the fallback guardrail.
- `scripts/check_baseline_invariants.py`: verifies that the exact-label reference baselines (GRAIL-style, PLL-style, TF-label-style) agree with online BFS on the released DAG workloads.
- `scripts/generate_paper_artifacts.py`: derives aggregate CSV files used by the paper figures.

Generated result files now include:

- `results/reachability_summary_by_index.csv`
- `results/biological_training_protocol.json`
- `results/paper_artifact_manifest.json`
- `results/baseline_invariant_checks.json`

## Supplementary material support

The artifact now includes `scripts/generate_supplementary_results.py`, which derives the supplementary replay tables and mapping CSVs from the released JSON/CSV traces. The resulting files are written to `results/supplementary_*.csv` and are consumed by `paper/supplementary.tex`.

- Supplementary dispatch diagnostics now include route-flip and suppression statistics in `code/results/coefficient_sensitivity.csv` and KL-gap bands in `code/results/supplementary_kl_gap_diagnostics.csv`.


## Reviewer-driven reframing note

The revised paper deliberately narrows the original learned-operator wording. The implemented contribution is cost-aware learned-operator dispatch over typed graph operators: SHRC provides exact reachability filtering on acyclic snapshots, and the dispatcher admits neural reranking only when its predicted utility exceeds its measured cost. The package includes dispatch-ablation and SHRC component-ablation CSV files to support this revised claim.
