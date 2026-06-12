# AgentFlow-PPI — Cross-Platform Setup & Reproduction Guide

> 한국어 version: **`SETUP_CROSS_PLATFORM.ko.md`**

This guide explains how readers reproduce the AgentFlow-PPI experiments on three
environments — **macOS (Intel CPU)**, **macOS (Apple Silicon, M3)**, and
**Ubuntu Linux** — using `pyenv` + `virtualenv` for an isolated Python environment.

The package is layered so you install only what an experiment needs:

| Tier | What it runs | Required packages |
|---|---|---|
| **Tier 0 — core** | reachability indexes, SHRC, optimizer, dispatch, modality labeling, multi-agent structure, **all 40 tests** | `numpy` only |
| **Tier 1 — end-to-end** | DuckDB recursive-SQL baseline (Q5) | `+ duckdb` |
| **Tier 2 — learned ops** | GNN reranker training (`train_model.py`) | `+ torch torch-geometric pandas h5py pyyaml` |
| **Tier 3 — real data** | STRING/BioGRID/Reactome download + external benchmark | `+ requests` (network) |
| **Tier 4 — LLM agents** | Ollama multi-agent collaboration | `+ Ollama runtime` (no extra pip) |
| **Tier 5 — production engines** | PostgreSQL / Neo4j / TigerGraph harness (§4.11) | engine servers via Docker |

> **Key point:** Tier 0 reproduces every *deterministic* number in the
> paper (index sizes, regret, answer-equivalence, Sperner bound, dispatch accuracy,
> multi-agent message counts) on a laptop with **no GPU and no servers**. Latency
> numbers are machine-dependent and reported per-host; we make no cross-hardware
> speedup claims. Higher tiers add the engine/LLM/real-data measurements.

---

## 0. One-time: install `pyenv` and a Python interpreter

We pin **Python 3.11** (the package requires `>=3.10`; 3.11 is the tested default).

### macOS (Intel **or** Apple Silicon)
```bash
# Homebrew (if not present): https://brew.sh
brew update
brew install pyenv pyenv-virtualenv
# build deps for compiling CPython
brew install openssl readline sqlite3 xz zlib tcl-tk

# add pyenv to your shell (zsh default on modern macOS)
echo 'export PYENV_ROOT="$HOME/.pyenv"'                   >> ~/.zshrc
echo 'export PATH="$PYENV_ROOT/bin:$PATH"'                >> ~/.zshrc
echo 'eval "$(pyenv init -)"'                             >> ~/.zshrc
echo 'eval "$(pyenv virtualenv-init -)"'                  >> ~/.zshrc
exec "$SHELL"
```
- **Intel:** the above is sufficient.
- **Apple Silicon (M3):** Homebrew lives at `/opt/homebrew`. Ensure your terminal is
  running natively (arm64), not under Rosetta: `arch` should print `arm64`. A native
  interpreter lets `numpy`/`torch` use Apple-Silicon wheels.

### Ubuntu (20.04 / 22.04 / 24.04)
```bash
sudo apt-get update
sudo apt-get install -y make build-essential libssl-dev zlib1g-dev \
  libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm libncursesw5-dev \
  xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev git

curl https://pyenv.run | bash    # installs pyenv + pyenv-virtualenv
echo 'export PYENV_ROOT="$HOME/.pyenv"'                   >> ~/.bashrc
echo 'command -v pyenv >/dev/null && export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
echo 'eval "$(pyenv init -)"'                             >> ~/.bashrc
echo 'eval "$(pyenv virtualenv-init -)"'                  >> ~/.bashrc
exec "$SHELL"
```

### Install the interpreter and create the virtualenv (all platforms)
```bash
pyenv install 3.11.9
cd /path/to/AgenticFlow-PPI          # the unzipped package root
pyenv virtualenv 3.11.9 agentflow-ppi    # named virtualenv
pyenv local agentflow-ppi                # auto-activates in this directory
python --version                         # -> Python 3.11.9
pip install --upgrade pip setuptools wheel
```

> If you prefer plain `virtualenv` instead of `pyenv-virtualenv`:
> ```bash
> pyenv install 3.11.9 && pyenv shell 3.11.9
> pip install virtualenv && python -m virtualenv .venv
> source .venv/bin/activate               # macOS/Linux
> ```

---

## 1. Tier 0 — core experiments (numpy only): the fast path

This is the recommended first run on **any** of the three OSes. It needs only numpy
and reproduces all 40 tests plus the deterministic experiments.

```bash
cd AgenticFlow-PPI/code
pip install numpy
pip install -e . --no-deps          # install the package without pulling torch etc.

# sanity: the full test suite (expect 40 passed)
python -m pytest tests/ -q

# representative deterministic experiments (no GPU, no servers):
python scripts/benchmark_reachability.py        # SHRC vs faithful GRAIL/PReaCH/PLL
python scripts/benchmark_modality_labeling.py   # fallback-free LCR + Sperner bound
python scripts/benchmark_optimizer_dispatch.py  # plan-space optimizer, regret/budget
python scripts/benchmark_dispatch_ablation.py   # Q4: reranker fixed, rule varied
python scripts/benchmark_multiagent.py          # measured 4-agent message flow
python scripts/benchmark_pipeline_baselines.py   # system-level fixed-order/agentic/agentflow at 5k/10k/20k
python scripts/benchmark_published_index.py      # published GRAIL/PLL C++ vs BFS oracle
python scripts/benchmark_plan_flips.py            # objective selects plans a fixed order would not
python scripts/benchmark_generality.py           # SHRC+dispatch on non-PPI graphs + DBMS integration
python scripts/benchmark_learned_cost.py         # linear vs Bao/Lero-style learned cost model 
#   (learned_cost uses scikit-learn's GBDT if installed, else a bundled numpy fallback;
#    `pip install scikit-learn` is optional and only sharpens the learned-model R^2)
python scripts/run_llm_query.py --source EGFR --target STAT3 --offline   # one query, full trace
```

`--no-deps` is what keeps Tier 0 light: the package's `pyproject.toml` lists torch and
friends, but the core modules import only numpy, so this installs the code and lets the
deterministic experiments run immediately. (`run_query.py` runs the full pipeline on a
prepared data root and takes `--query`/`--data-root`; the simplest single-query demo is
`run_llm_query.py` above, which needs no prepared data.)

**Platform notes**
- *Intel macOS / Ubuntu:* numpy installs a prebuilt wheel; nothing to compile.
- *M3 macOS:* `pip install numpy` fetches the arm64 wheel automatically. Confirm with
  `python -c "import platform; print(platform.machine())"` → `arm64`.

---

## 2. Tier 4 — Ollama LLM multi-agent collaboration

The four agents (planner, reachability, executor, aggregator) run as **prompted LLMs**
over a local Ollama runtime. No extra pip packages are needed — the client uses the
standard library — but you must install Ollama and pull the models.

### Install Ollama
- **macOS (Intel or M3):** download the app from <https://ollama.com/download> (or
  `brew install ollama`), then start the server:
  ```bash
  ollama serve &                  # leave running in a terminal/background
  ```
- **Ubuntu:**
  ```bash
  curl -fsSL https://ollama.com/install.sh | sh
  sudo systemctl enable --now ollama     # or: ollama serve &
  ```

### Pull the four models used in the paper
```bash
ollama pull llama3.1:8b
ollama pull qwen2.5:7b
ollama pull phi3:medium
ollama pull mistral-nemo
```

**Memory guidance.** These are 7–8B-class models (~5–8 GB each, run one at a time).
- **M3 with 128 GB unified memory:** all four run comfortably and fast (GPU-backed via
  Metal). This is the reference host for the paper's LLM table.
- **M3 with 16–24 GB:** run them one model at a time (the benchmark already loads
  sequentially); prefer `llama3.1:8b` / `qwen2.5:7b`.
- **Intel macOS (no discrete GPU):** runs on CPU — correctness is identical but latency
  is higher; use the smaller models and expect seconds-per-call.
- **Ubuntu with an NVIDIA GPU:** Ollama uses CUDA automatically; fastest after M3.
- **Ubuntu CPU-only:** works, slower; fine for verifying schema/dispatch/answer rates.

### Run the LLM experiments
```bash
cd AgenticFlow-PPI/code        # Tier 0 install is enough (stdlib client)

# sweep all four models, writing results/llm_multiagent.{json,csv}
python scripts/benchmark_llm_multiagent.py

# trace a single query through the four agents (full inter-agent trace)
python scripts/run_llm_query.py --source EGFR --target STAT3 --model llama3.1:8b

# choose a subset of models or force the offline oracle:
AGENTFLOW_OLLAMA_MODELS="llama3.1:8b,qwen2.5:7b" python scripts/benchmark_llm_multiagent.py
AGENTFLOW_FORCE_OFFLINE=1 python scripts/benchmark_llm_multiagent.py   # no server needed
```

If no Ollama server is reachable, the same scripts fall back to a **deterministic schema
oracle** and label every row `offline-deterministic`, so the protocol, message counts,
and authority constraints are verifiable even without Ollama. The per-model
schema/dispatch/answer/latency numbers in the paper's LLM table come from a host running
`ollama serve`. The agents' system prompts are plain editable files under
`agentflow_ppi/agents/llm/prompts/`.

---

## 3. Tier 2 — training the GNN reranker (optional, needs PyTorch)

Only required if you want to retrain the learned reranker; the released experiments do
not need it.

```bash
cd AgenticFlow-PPI/code
pip install -e .                  # WITHOUT --no-deps: pulls torch, torch-geometric, etc.
```
- **M3 macOS:** PyTorch ships arm64 wheels with the **MPS** (Metal) backend; the package
  auto-selects MPS when available (`agentflow_ppi/rdflow/device.py`). Verify:
  ```bash
  python -c "import torch; print(torch.backends.mps.is_available())"   # -> True on M3
  ```
- **Intel macOS:** CPU wheels only (no CUDA/MPS); training runs on CPU.
- **Ubuntu + NVIDIA GPU:** install the CUDA build of PyTorch first, matching your
  driver, e.g.:
  ```bash
  pip install torch --index-url https://download.pytorch.org/whl/cu121
  pip install -e .
  python -c "import torch; print(torch.cuda.is_available())"           # -> True
  ```
- **Ubuntu CPU-only:** the default `pip install torch` CPU wheel is fine.

```bash
python scripts/train_model.py            # trains the reranker; writes a checkpoint
```

---

## 4. Tier 1 / Tier 3 / Tier 5 — engines, real data, production DBs

```bash
# Tier 1: DuckDB end-to-end baseline (Q5)
pip install duckdb
python scripts/benchmark_endtoend_baselines.py

# Tier 3: real STRING/BioGRID/Reactome download + external benchmark (needs network)
pip install requests
python scripts/download_external_data.py          # downloads to data/external/
python scripts/build_external_manifest.py
python scripts/benchmark_external_reranking.py    # protein-/pathway-disjoint splits
#   (no network: if data/external/ is absent, the script uses the bundled fixture,
#    flagged using_fixture and never reported as a headline number)

# Tier 5: production engine harness (§4.11) — start servers via Docker, then run
#   PostgreSQL
docker run -d --name afppi-pg  -e POSTGRES_PASSWORD=pw -p 5432:5432 postgres:16
#   Neo4j
docker run -d --name afppi-neo -e NEO4J_AUTH=neo4j/password -p 7687:7687 -p 7474:7474 neo4j:5
#   (TigerGraph optional; see its own install docs)
python scripts/benchmark_engine_baselines.py      # checks each engine vs a BFS oracle
#   NOTE: a real SQLite recursive-CTE engine runs with NO server (stdlib sqlite3),
#   so this script always produces at least one measured production-grade SQL baseline
#   on Ubuntu / macOS Intel / M3; Postgres/Neo4j are added if you start their Docker servers.
```
- **Docker** is the simplest way to get identical engines on all three OSes
  (Docker Desktop on macOS, `docker.io` on Ubuntu). The harness runs whichever engines
  it can reach and **skips the rest gracefully**, always running the in-process SHRC
  engine as the correctness reference.

---

## 5. Reproduce everything in one command

After installing the tiers you care about:
```bash
cd AgenticFlow-PPI/code
python scripts/run_all_experiments.py     # regenerates results/ for all 20 stages
```
Stages whose servers/data/models are absent are skipped or fall back to labeled
offline/fixture mode, so this completes on a plain Tier 0 install and fills in more as
you add tiers. Every output records its provenance (`using_fixture`, `backend`,
`offline-deterministic`) so measured and simulated numbers are never confused.

---

## 6. Building the paper PDFs (optional)

Requires a TeX distribution (MacTeX on macOS, TeX Live on Ubuntu).
```bash
# macOS:  brew install --cask mactex-no-gui
# Ubuntu: sudo apt-get install -y texlive-full
cd AgenticFlow-PPI/paper
pdflatex main && bibtex main && pdflatex main && pdflatex main          # -> main.pdf
pdflatex supplementary && bibtex supplementary && pdflatex supplementary && pdflatex supplementary
```

---

## 7. Troubleshooting

- **`pip install -e .` tries to build torch and fails on a CPU-only box** → use the
  Tier 0 path `pip install numpy && pip install -e . --no-deps`.
- **`externally-managed-environment` error (Ubuntu 23.04+/Homebrew Python)** → you are
  not inside the virtualenv. Re-activate (`pyenv local agentflow-ppi` or
  `source .venv/bin/activate`); the venv avoids needing `--break-system-packages`.
- **Ollama connection refused** → `ollama serve` is not running, or it is on a nonstandard
  port. Set the URL in code or use `AGENTFLOW_FORCE_OFFLINE=1` to verify structure.
- **M3 imports x86 numpy / slow** → your shell is under Rosetta. Open a native arm64
  terminal (`arch` → `arm64`) and recreate the venv.
- **`torch.backends.mps` missing on M3** → upgrade to a recent PyTorch (`pip install -U
  torch`); MPS needs PyTorch ≥ 2.0 and macOS ≥ 12.3.
