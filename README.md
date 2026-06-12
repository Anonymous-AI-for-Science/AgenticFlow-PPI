# AgentFlow-PPI

Cost-based execution for multimodal protein-interaction graph queries. AgentFlow-PPI
answers exact typed reachability over a multi-modal directed graph while deciding, under
a resource budget, whether a learned reranking operator should be admitted to reorder
the answer. It combines:

- **SHRC** — an exact, sparsity-aware hybrid reachability index (interval labeling on a
  sparse periphery + a coverage-pruned 2-hop cover on the residual core);
- a **cost-aware dispatch** layer that prices a learned reranker against exact graph
  operators and admits it only when a calibrated predictor estimates the quality gain
  exceeds the measured cost;
- a small set of **specialized agents** (planner, reachability, executor, aggregator)
  that exchange messages over a shared bus, optionally instantiated as prompted LLMs
  over a local Ollama runtime;
- a **plan-space optimizer** (physical operators, enumerated plans, learned cost/quality
  models) with a per-query regret bound.

All quality experiments use an independent canonical-pathway ground truth that
references none of the reranker's features, so the evaluation is leakage-controlled.

---

## Repository layout

```
agentflow_ppi/        # the library
  reachability/       # SHRC index
  engines/            # in-process SHRC + SQLite/PostgreSQL/Neo4j/TigerGraph adapters
  optimizer/          # plan-space optimizer, cost/quality models, regret bound
  agents/             # planner/reachability/executor/aggregator + LLM (Ollama) layer
  benchmarks/         # reachability baselines, strong baselines, modality labeling
  data/               # synthetic + external (STRING/BioGRID/Reactome) loaders
  eval/               # shared evaluation harness (independent pathway labels)
scripts/              # one runnable experiment per file; run_all_experiments.py runs them
tests/                # deterministic test suite
examples/             # minimal usage examples
```

---

## Installation (pyenv + virtualenv)

The package is layered so you install only what you need. Most experiments need only
NumPy.

```bash
# Python 3.11 (>=3.10 supported)
pyenv install 3.11.9
pyenv virtualenv 3.11.9 agentflow-ppi
pyenv local agentflow-ppi          # or: python -m virtualenv .venv && source .venv/bin/activate
pip install --upgrade pip setuptools wheel
```

### Tier 0 — core (NumPy only): the fast path

Reproduces the full test suite and every deterministic experiment with no GPU and no
servers.

```bash
pip install numpy
pip install -e . --no-deps          # install the package without pulling torch etc.
python -m pytest tests/ -q          # expect 40 passed
```

### Optional tiers

| Add | Enables |
|-----|---------|
| `pip install duckdb` | DuckDB recursive-SQL end-to-end baseline |
| `pip install scikit-learn` | gradient-boosted learned cost model (else a NumPy fallback is used) |
| `pip install torch torch-geometric pandas h5py pyyaml` | training the GNN reranker (`scripts/train_model.py`) |
| `pip install requests` | downloading the real STRING/BioGRID/Reactome/OmniPath data |
| local Ollama runtime | running the agents as prompted LLMs |
| Docker (`postgres:16`, `neo4j:5`) | PostgreSQL / Neo4j production-engine comparison |

A real SQLite recursive-CTE engine ships in the Python standard library, so a
production-grade SQL baseline runs on every host with no server.

---

## Quick start

```bash
# trace one query through the agent flow (offline; no Ollama needed)
python scripts/run_llm_query.py --source EGFR --target STAT3 --offline

# representative deterministic experiments (NumPy only)
python scripts/benchmark_reachability.py        # SHRC vs faithful GRAIL/PReaCH/PLL
python scripts/benchmark_modality_labeling.py   # fallback-free label-constrained index
python scripts/benchmark_optimizer_dispatch.py  # plan-space optimizer: regret, budget
python scripts/benchmark_dispatch_ablation.py   # reranker fixed, dispatch rule varied
python scripts/benchmark_pipeline_baselines.py  # fixed-order vs agentic vs cost-aware, 5k/10k/20k
python scripts/benchmark_plan_flips.py          # the objective selects plans a fixed order would not
python scripts/benchmark_generality.py          # SHRC + dispatch on non-PPI graphs, SQLite integration
python scripts/benchmark_learned_cost.py        # linear vs gradient-boosted cost model
python scripts/benchmark_engine_baselines.py    # SHRC vs SQLite (and Postgres/Neo4j if provisioned)
```

### Run everything

```bash
python scripts/run_all_experiments.py           # regenerates results/ for all stages
```

Stages whose optional servers, data, or models are absent are skipped or fall back to a
labeled offline/fixture mode, so this completes on a plain Tier 0 install and fills in
more as you add tiers. Every output records its provenance (`backend`,
`using_fixture`, `offline-deterministic`) so measured and simulated numbers are never
confused.

---

## LLM agents (Ollama)

The four agents can run as prompted LLMs. Their system prompts are editable files under
`agentflow_ppi/agents/llm/prompts/`.

```bash
ollama serve &
ollama pull llama3.1:8b qwen2.5:7b phi3:medium mistral-nemo
python scripts/benchmark_llm_multiagent.py       # sweeps the models; writes results/
```

If no Ollama server is reachable, the same scripts fall back to a deterministic schema
oracle (labeled `offline-deterministic`), so the protocol and tests are verifiable
without a GPU. An authority constraint guarantees the exact reachable set and ranking
are always computed symbolically and never altered by a model.

---

## External (real) data

```bash
pip install requests
python scripts/download_external_data.py         # STRING/BioGRID/Reactome/OmniPath -> data/external/
python scripts/build_external_manifest.py
python scripts/benchmark_external_reranking.py   # protein-/pathway-disjoint splits
```

Identifiers are reconciled to gene symbols; gold mediators are defined by Reactome
pathway position (independent of every reranker feature). Without network access the
same scripts run on a bundled fixture, flagged `using_fixture`.

---

## Reproducibility notes

- Quality, index, regret, answer-equivalence, generality, and cost-R-squared numbers
  are deterministic given the seed manifests and match across runs.
- Latency is machine-dependent and reported per host; no cross-hardware speedup claims
  are made.
- Cross-platform setup details for macOS (Intel and Apple Silicon) and Ubuntu are in
  `SETUP_CROSS_PLATFORM.md`.

## License

Released for research use. See `LICENSE`.
