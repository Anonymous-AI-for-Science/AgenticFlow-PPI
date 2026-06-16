#!/usr/bin/env python3
"""Cross-platform environment check for AgentFlow-PPI.

Verifies that the current host (Ubuntu, macOS Intel, or Apple Silicon M-series)
can run the AgentFlow-PPI experiments, and prints exactly which tiers are
available. Pure standard library except for the optional probes, so it runs
before any heavy dependency is installed.

Tiers
  Tier 0 (core, required): Python >= 3.10 and numpy -> all deterministic
          experiments (reachability, dispatch ablation, scale, generality, ...).
  Tier 1 (large-scale)   : enough RAM for the 50k--1M SHRC sweep.
  Tier 2 (accelerator)   : torch with CUDA (Ubuntu/GPU) or MPS (Apple Silicon);
          CPU torch on macOS Intel. Optional; only the learned-stage timing uses it.
  Tier 3 (engines)       : psycopg2 + neo4j drivers and a reachable Docker, for the
          live cross-engine table. Optional.
  Tier 4 (LLM)           : a reachable Ollama server, for the LLM multi-agent table.
          Optional; the run falls back to a deterministic oracle without it.

Exit code is 0 if Tier 0 is satisfied (everything else is optional), else 1.
"""
from __future__ import annotations

import importlib.util
import os
import platform
import shutil
import sys


def _ok(b: bool) -> str:
    return "\033[32mOK\033[0m" if b else "\033[33m--\033[0m"


def _has(mod: str) -> bool:
    return importlib.util.find_spec(mod) is not None


def main() -> int:
    print("=== AgentFlow-PPI environment check ===")
    sysname = platform.system()        # 'Linux', 'Darwin'
    machine = platform.machine()       # 'x86_64', 'arm64', 'aarch64'
    if sysname == "Darwin" and machine == "arm64":
        plat = "macOS Apple Silicon (M-series)"
    elif sysname == "Darwin":
        plat = "macOS Intel (x86_64)"
    elif sysname == "Linux":
        plat = "Ubuntu/Linux"
    else:
        plat = f"{sysname}/{machine}"
    print(f"platform        : {plat}  [{sysname} {machine}]")
    print(f"python          : {platform.python_version()}  ({sys.executable})")

    # ---- Tier 0: core ----
    py_ok = sys.version_info >= (3, 10)
    numpy_ok = _has("numpy")
    print(f"[{_ok(py_ok)}] Tier 0  Python >= 3.10")
    print(f"[{_ok(numpy_ok)}] Tier 0  numpy importable"
          + ("" if numpy_ok else "  -> pip install numpy && pip install -e . --no-deps"))

    # ---- Tier 1: memory for large-scale sweep ----
    total_gb = _total_ram_gb()
    cap = (("1M-node grid" if total_gb >= 64 else
            "up to 250k (use --max-nodes 250000)" if total_gb >= 16 else
            "50k/100k only (use --quick)"))
    print(f"[{_ok(total_gb is not None)}] Tier 1  RAM ~{total_gb:.0f} GB -> large-scale: {cap}"
          if total_gb else "[--] Tier 1  RAM unknown")

    # ---- Tier 2: accelerator (optional) ----
    torch_ok = _has("torch")
    dev = "none"
    if torch_ok:
        try:
            import torch
            if torch.cuda.is_available():
                dev = f"CUDA ({torch.cuda.get_device_name(0)})"
            elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                dev = "MPS (Apple Silicon)"
            else:
                dev = "CPU"
        except Exception as e:  # noqa: BLE001
            dev = f"torch present but probe failed: {e}"
    print(f"[{_ok(torch_ok)}] Tier 2  torch {'-> ' + dev if torch_ok else '(optional; CPU paths work without it)'}")

    # ---- Tier 3: engines (optional) ----
    pg_ok = _has("psycopg2")
    neo_ok = _has("neo4j")
    docker_ok = shutil.which("docker") is not None
    print(f"[{_ok(pg_ok)}] Tier 3  psycopg2 (PostgreSQL driver)"
          + ("" if pg_ok else "  -> pip install psycopg2-binary"))
    print(f"[{_ok(neo_ok)}] Tier 3  neo4j (Bolt driver)"
          + ("" if neo_ok else "  -> pip install neo4j"))
    print(f"[{_ok(docker_ok)}] Tier 3  docker on PATH"
          + ("" if docker_ok else "  -> install Docker Desktop / docker engine"))
    for var in ["AGENTFLOW_PG_DSN", "AGENTFLOW_NEO4J_URI"]:
        if os.environ.get(var):
            print(f"        env {var} is set")

    # ---- Tier 4: LLM (optional) ----
    ollama_ok = shutil.which("ollama") is not None
    print(f"[{_ok(ollama_ok)}] Tier 4  ollama on PATH"
          + ("" if ollama_ok else "  -> optional; offline-deterministic oracle is used without it"))

    # ---- TeX (for building the PDF) ----
    tex_ok = shutil.which("pdflatex") is not None
    print(f"[{_ok(tex_ok)}] Paper   pdflatex on PATH"
          + ("" if tex_ok else "  -> install TeX Live (Ubuntu) or MacTeX (macOS) to build the PDF"))

    print()
    if py_ok and numpy_ok:
        print("\033[32mTier 0 satisfied: you can run all deterministic experiments now.\033[0m")
        print("Next: python scripts/run_all_experiments.py   (or individual benchmark_*.py)")
        return 0
    print("\033[31mTier 0 NOT satisfied.\033[0m Install Python >= 3.10 and numpy, then re-run.")
    return 1


def _total_ram_gb():
    try:
        if hasattr(os, "sysconf") and "SC_PAGE_SIZE" in os.sysconf_names and "SC_PHYS_PAGES" in os.sysconf_names:
            return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / (1024 ** 3)
    except (ValueError, OSError):
        pass
    # macOS fallback
    try:
        import subprocess
        out = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True)
        if out.returncode == 0:
            return int(out.stdout.strip()) / (1024 ** 3)
    except Exception:  # noqa: BLE001
        pass
    return None


if __name__ == "__main__":
    raise SystemExit(main())
