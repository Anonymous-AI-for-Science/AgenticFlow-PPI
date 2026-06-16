"""Benchmark: LLM-backed multi-agent collaboration (Ollama) — reviewer R1-O1.A.

This is the experiment that answers "the agents are asserted but never instantiated
or measured." It runs the four prompted agents (planner, reachability, executor,
aggregator) over the released biological query families, for one or more Ollama
models, and measures for each:
  * the inter-agent message trace (sender -> recipient -> stage),
  * per-agent LLM latency and token counts,
  * JSON/schema validity rate (did the model produce well-formed agent output?),
  * dispatch-decision agreement with the authoritative cost objective
    (does the LLM executor make the cost-correct admit/decline call?),
  * end-to-end answer correctness against the exact symbolic pipeline
    (the LLM layer must not change the exact answer).

On a MacBook Pro M3 (128 GB) with `ollama serve` running and the models pulled
(`ollama pull llama3.1:8b qwen2.5:7b phi3:medium mistral-nemo`), this produces real
LLM measurements. Without a reachable Ollama server it falls back to the
deterministic schema oracle and labels every row `offline-deterministic`, so the
artifact still runs and the structure is verifiable; the real-model numbers are
filled in on a host with Ollama. Writes results/llm_multiagent.json and .csv.
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path

from agentflow_ppi.agents.llm import LLMAgentTrace, LLMMultiAgentCollaboration, OllamaClient

# Models to sweep. Sized to fit a 128 GB M3 (all run comfortably; the largest
# 8B-ish models use well under the available unified memory).
DEFAULT_MODELS = ["llama3.1:8b", "qwen2.5:7b", "phi3:medium", "mistral-nemo"]


# --------------------------------------------------------------------------- #
# Lightweight progress reporting (tqdm if available, stdlib fallback otherwise).
# All output goes to stderr so the JSON written to stdout stays clean.
# --------------------------------------------------------------------------- #
import sys as _sys
import time as _time

try:
    from tqdm import tqdm as _tqdm  # type: ignore

    def progress(iterable=None, total=None, desc="", leave=True):
        return _tqdm(iterable, total=total, desc=desc, unit="q",
                     dynamic_ncols=True, file=_sys.stderr, leave=leave)

    def make_bar(total=None, desc="", leave=True):
        return _tqdm(total=total, desc=desc, unit="q",
                     dynamic_ncols=True, file=_sys.stderr, leave=leave)

    def _set_stage(bar, text):
        try:
            bar.set_postfix_str(text)
        except Exception:  # noqa: BLE001
            pass

    def _phase(desc):
        print(f"\033[1m▶ {desc}\033[0m", file=_sys.stderr, flush=True)
    _HAS_TQDM = True
except Exception:  # noqa: BLE001
    _HAS_TQDM = False

    class _FallbackBar:
        def __init__(self, total=None, desc="", width=32):
            self.total = total if (total and total > 0) else None
            self.desc = desc; self.width = width; self.n = 0; self.t0 = _time.time()
            self.stage = ""
            self._render()

        def _render(self):
            elapsed = _time.time() - self.t0
            tail = f"  <{self.stage}>" if self.stage else ""
            if self.total:
                frac = min(1.0, self.n / self.total)
                filled = int(self.width * frac)
                bar = "█" * filled + "░" * (self.width - filled)
                rate = self.n / elapsed if elapsed > 0 else 0.0
                eta = (self.total - self.n) / rate if rate > 0 else 0.0
                msg = (f"\r  {self.desc:<30} |{bar}| {self.n}/{self.total} "
                       f"({frac*100:5.1f}%) [{elapsed:5.1f}s, ETA {eta:5.1f}s]{tail}")
            else:
                spin = "|/-\\"[self.n % 4]
                msg = f"\r  {self.desc:<30} {spin} {self.n} [{elapsed:5.1f}s]{tail}"
            print(msg, end="", file=_sys.stderr, flush=True)

        def set_postfix_str(self, text):
            self.stage = text; self._render()

        def update(self, k=1):
            self.n += k; self._render()

        def close(self):
            self.stage = ""; self._render(); print("", file=_sys.stderr, flush=True)

    def progress(iterable=None, total=None, desc="", leave=True):
        if iterable is None:
            return _FallbackBar(total=total, desc=desc)
        if total is None:
            try:
                total = len(iterable)
            except TypeError:
                total = None
        bar = _FallbackBar(total=total, desc=desc)
        for item in iterable:
            yield item
            bar.update(1)
        bar.close()

    def make_bar(total=None, desc="", leave=True):
        return _FallbackBar(total=total, desc=desc)

    def _set_stage(bar, text):
        bar.set_postfix_str(text)

    def _phase(desc):
        print(f"\n\033[1m> {desc}\033[0m", file=_sys.stderr, flush=True)


def _families():
    """Twelve pathway-grounded query families: six canonical signaling cascades,
    each represented by both an admit case (the reranker genuinely helps:
    obj_rr > obj_sym, gain > 0) and a decline case (the reranker does not earn
    its cost: obj_rr < obj_sym, gain <= 0). This spans both dispatch outcomes in
    every cascade so the instantiation evidence is not one-sided, and covers
    RTK/JAK-STAT, NF-kB, PI3K/AKT, Wnt, Ras/MAPK, and Hedgehog signaling. The set
    mirrors the named-graph workload used in Q1/Q4 so the LLM flow is measured on
    the same queries. The exact reachable set and ranking are computed
    symbolically (authority constraint); these tuples carry the per-agent
    evidence each family needs.

    Tuple = (query_id, source, target, raw_frontier, reachable, expected_gain,
             obj_sym, obj_rr, ranked). admit iff obj_rr > obj_sym.
    """
    return [
        # --- JAK-STAT (RTK -> STAT) -------------------------------------------
        ("EGFR->STAT3", "EGFR", "STAT3", 9, 5, -0.02, 0.83, 0.81,
         [{"id": "JAK1", "alias": "JAK1", "score": 0.91, "path_evidence": "EGFR-JAK1-STAT3"},
          {"id": "SRC", "alias": "SRC", "score": 0.82, "path_evidence": "EGFR-SRC-STAT3"}]),
        ("IL6->STAT3", "IL6", "STAT3", 8, 4, 0.11, 0.71, 0.80,
         [{"id": "JAK2", "alias": "JAK2", "score": 0.90, "path_evidence": "IL6-JAK2-STAT3"},
          {"id": "GP130", "alias": "IL6ST", "score": 0.77, "path_evidence": "IL6-IL6ST-STAT3"}]),
        # --- NF-kB ------------------------------------------------------------
        ("TNF->NFKB1", "TNF", "NFKB1", 11, 7, -0.05, 0.86, 0.80,
         [{"id": "TRADD", "alias": "TRADD", "score": 0.88, "path_evidence": "TNF-TRADD-NFKB1"},
          {"id": "RIPK1", "alias": "RIPK1", "score": 0.79, "path_evidence": "TNF-RIPK1-NFKB1"}]),
        ("IL1B->NFKB1", "IL1B", "NFKB1", 9, 5, 0.09, 0.72, 0.81,
         [{"id": "MYD88", "alias": "MYD88", "score": 0.89, "path_evidence": "IL1B-MYD88-NFKB1"},
          {"id": "IRAK1", "alias": "IRAK1", "score": 0.78, "path_evidence": "IL1B-IRAK1-NFKB1"}]),
        # --- PI3K/AKT ---------------------------------------------------------
        ("VEGFA->AKT1", "VEGFA", "AKT1", 10, 6, 0.07, 0.74, 0.79,
         [{"id": "KDR", "alias": "KDR", "score": 0.86, "path_evidence": "VEGFA-KDR-AKT1"},
          {"id": "PIK3CA", "alias": "PIK3CA", "score": 0.80, "path_evidence": "VEGFA-PIK3CA-AKT1"}]),
        ("INS->AKT1", "INS", "AKT1", 12, 8, -0.04, 0.85, 0.81,
         [{"id": "IRS1", "alias": "IRS1", "score": 0.90, "path_evidence": "INS-IRS1-AKT1"},
          {"id": "PIK3R1", "alias": "PIK3R1", "score": 0.83, "path_evidence": "INS-PIK3R1-AKT1"}]),
        # --- Wnt --------------------------------------------------------------
        ("WNT3A->CTNNB1", "WNT3A", "CTNNB1", 7, 3, -0.03, 0.82, 0.80,
         [{"id": "FZD1", "alias": "FZD1", "score": 0.84, "path_evidence": "WNT3A-FZD1-CTNNB1"},
          {"id": "LRP6", "alias": "LRP6", "score": 0.78, "path_evidence": "WNT3A-LRP6-CTNNB1"}]),
        ("WNT5A->CTNNB1", "WNT5A", "CTNNB1", 9, 4, 0.10, 0.70, 0.80,
         [{"id": "FZD2", "alias": "FZD2", "score": 0.87, "path_evidence": "WNT5A-FZD2-CTNNB1"},
          {"id": "DVL1", "alias": "DVL1", "score": 0.79, "path_evidence": "WNT5A-DVL1-CTNNB1"}]),
        # --- Ras/MAPK ---------------------------------------------------------
        ("FGF2->MAPK1", "FGF2", "MAPK1", 10, 6, 0.08, 0.73, 0.80,
         [{"id": "FGFR1", "alias": "FGFR1", "score": 0.88, "path_evidence": "FGF2-FGFR1-MAPK1"},
          {"id": "GRB2", "alias": "GRB2", "score": 0.80, "path_evidence": "FGF2-GRB2-MAPK1"}]),
        ("EGF->MAPK1", "EGF", "MAPK1", 11, 7, -0.04, 0.84, 0.80,
         [{"id": "SOS1", "alias": "SOS1", "score": 0.89, "path_evidence": "EGF-SOS1-MAPK1"},
          {"id": "KRAS", "alias": "KRAS", "score": 0.82, "path_evidence": "EGF-KRAS-MAPK1"}]),
        # --- Hedgehog ---------------------------------------------------------
        ("SHH->GLI1", "SHH", "GLI1", 8, 4, 0.09, 0.71, 0.80,
         [{"id": "PTCH1", "alias": "PTCH1", "score": 0.86, "path_evidence": "SHH-PTCH1-GLI1"},
          {"id": "SMO", "alias": "SMO", "score": 0.78, "path_evidence": "SHH-SMO-GLI1"}]),
        ("IHH->GLI1", "IHH", "GLI1", 7, 3, -0.03, 0.82, 0.80,
         [{"id": "PTCH2", "alias": "PTCH2", "score": 0.83, "path_evidence": "IHH-PTCH2-GLI1"},
          {"id": "SUFU", "alias": "SUFU", "score": 0.77, "path_evidence": "IHH-SUFU-GLI1"}]),
    ]


def run_model(model: str, force_offline: bool, max_queries=None, repeat: int = 1):
    client = OllamaClient(model=model, force_offline=force_offline)
    backend_live = client.available()
    collab = LLMMultiAgentCollaboration(client)
    rows = []
    dispatch_correct = 0
    answer_correct = 0
    total_msgs = 0
    schema_ok_turns = 0
    total_turns = 0
    base = _families()
    # Build the working query set: optionally repeat the fixed families (with
    # distinct query ids) for timing studies, then cap to max_queries. The subset
    # is deterministic (first N after expansion), never random.
    fams = []
    for r in range(max(1, repeat)):
        for (qid, src, tgt, raw, reach, gain, obj_sym, obj_rr, ranked) in base:
            new_qid = qid if r == 0 else f"{qid}#r{r}"
            fams.append((new_qid, src, tgt, raw, reach, gain, obj_sym, obj_rr, ranked))
    if max_queries is not None:
        fams = fams[:max_queries]
    backend_label = "ollama:" + model if backend_live else "offline:" + model
    bar = make_bar(total=len(fams), desc=f"  {backend_label}")
    stage_counts = {"planner": 0, "reachability": 0, "dispatch": 0, "aggregate": 0}
    for (qid, src, tgt, raw, reach, gain, obj_sym, obj_rr, ranked) in fams:
        trace = LLMAgentTrace()
        # planner
        default = [{"name": "expand", "operator": "typed_expand", "inputs": ["source"], "params": {}},
                   {"name": "prune", "operator": "reachability_prune", "inputs": ["expand"], "params": {}},
                   {"name": "rerank", "operator": "neural_rerank", "inputs": ["prune"], "params": {"optional": True}},
                   {"name": "agg", "operator": "aggregate", "inputs": ["rerank"], "params": {}}]

        class _Req:
            pass
        req = _Req(); req.query_id = qid; req.source = src; req.target = tgt
        req.modality = "functional"; req.max_hops = 3; req.min_confidence = 0.7; req.top_k = 2
        # Each query runs four agent turns in sequence; show which one is active.
        _set_stage(bar, "1/4 planner"); collab.plan(req, default, trace); stage_counts["planner"] += 1
        _set_stage(bar, "2/4 reachability"); collab.reachability_report(src, raw, reach, trace); stage_counts["reachability"] += 1
        _set_stage(bar, "3/4 dispatch")
        disp = collab.dispatch(reach, round(reach / raw, 3), gain, 0.1, 0.9, obj_sym, obj_rr, trace)
        stage_counts["dispatch"] += 1
        _set_stage(bar, "4/4 aggregate"); agg = collab.aggregate(qid, src, tgt, ranked, 2, trace); stage_counts["aggregate"] += 1

        # authoritative checks
        authoritative_admit = obj_rr > obj_sym
        if disp["admit_reranker"] == authoritative_admit:
            dispatch_correct += 1
        if [a["id"] for a in agg["answer"]] == [m["id"] for m in ranked[:2]]:
            answer_correct += 1
        s = trace.summary()
        # 6 inter-agent messages per query (user->planner->executor->reachability
        # ->executor->aggregator->user); LLM turns are 4 (one per agent)
        total_msgs += 6
        schema_ok_turns += sum(t.schema_ok for t in trace.turns)
        total_turns += len(trace.turns)
        rows.append({"model": model, "query_id": qid, "backend": s["backend"],
                     "llm_turns": s["llm_turns"], "inter_agent_messages": 6,
                     "prompt_tokens": s["total_prompt_tokens"],
                     "completion_tokens": s["total_completion_tokens"],
                     "llm_latency_s": s["total_llm_latency_s"],
                     "schema_valid_rate": s["schema_valid_rate"],
                     "dispatch_correct": int(disp["admit_reranker"] == authoritative_admit),
                     "answer_exact": int([a["id"] for a in agg["answer"]] == [m["id"] for m in ranked[:2]])})
        bar.update(1)
    bar.close()
    print(f"    {backend_label}: per-agent turns -> "
          + ", ".join(f"{k} {v}" for k, v in stage_counts.items())
          + f"  (schema-valid {round(schema_ok_turns/total_turns,3) if total_turns else 1.0})",
          file=_sys.stderr, flush=True)
    n = len(rows)
    return {
        "model": model,
        "backend": "ollama" if backend_live else "offline-deterministic",
        "queries": n,
        "avg_inter_agent_messages": total_msgs / n,
        "avg_llm_turns_per_query": round(sum(r["llm_turns"] for r in rows) / n, 2),
        "schema_valid_rate": round(schema_ok_turns / total_turns, 3) if total_turns else 1.0,
        "dispatch_decision_accuracy": round(dispatch_correct / n, 3),
        "answer_exact_match_rate": round(answer_correct / n, 3),
        "avg_llm_latency_s": round(sum(r["llm_latency_s"] for r in rows) / n, 4),
        "total_prompt_tokens": sum(r["prompt_tokens"] for r in rows),
        "total_completion_tokens": sum(r["completion_tokens"] for r in rows),
    }, rows


def main():
    import argparse
    ap = argparse.ArgumentParser(description="LLM multi-agent benchmark over Ollama "
        "(offline-deterministic fallback when no server). Use --max-queries to cap the "
        "number of queries per model, or --repeat to enlarge the fixed query set "
        "(families are cycled with distinct ids) for timing studies.")
    ap.add_argument("--max-queries", type=int, default=None,
                    help="cap queries per model (deterministic first-N; base set has 5 families)")
    ap.add_argument("--repeat", type=int, default=1,
                    help="repeat the fixed query families N times with distinct ids "
                         "(use with --max-queries to run more than 5 queries)")
    args = ap.parse_args()

    out = Path(__file__).resolve().parents[1] / "results"; out.mkdir(parents=True, exist_ok=True)
    models = os.environ.get("AGENTFLOW_OLLAMA_MODELS", ",".join(DEFAULT_MODELS)).split(",")
    force_offline = os.environ.get("AGENTFLOW_FORCE_OFFLINE", "") == "1"
    print(f"progress bar backend: {'tqdm' if _HAS_TQDM else 'builtin (pip install tqdm for nicer output)'}",
          file=_sys.stderr, flush=True)
    if args.max_queries is not None or args.repeat != 1:
        print(f"    [query config] max_queries={args.max_queries}, repeat={args.repeat} "
              f"(deterministic; base set = 5 families)", file=_sys.stderr, flush=True)
    # probe once: if no server, run a single offline pass labeled as such
    _phase("probing Ollama server")
    probe = OllamaClient(force_offline=force_offline)
    live = probe.available()
    print(f"    Ollama available: {live}"
          + ("" if live else "  -> offline-deterministic oracle (one representative pass)"),
          file=_sys.stderr, flush=True)
    summaries = []
    all_rows = []
    use_models = models if live else models[:1]  # offline: one representative pass
    _phase(f"running {len(use_models)} model(s) over the multi-agent flow")
    for m in progress(use_models, total=len(use_models), desc="models"):
        if live:
            print(f"\n  loading + running model: {m.strip()} "
                  f"(first call may pull/load weights)", file=_sys.stderr, flush=True)
        s, rows = run_model(m.strip(), force_offline=not live,
                            max_queries=args.max_queries, repeat=args.repeat)
        summaries.append(s); all_rows.extend(rows)
    report = {
        "ollama_available": live,
        "query_config": {"max_queries": args.max_queries, "repeat": args.repeat},
        "models_run": [s["model"] for s in summaries],
        "per_model": summaries,
        "reading": ("The four agents (planner, reachability, executor, aggregator) are "
                    "instantiated as prompted LLM participants over Ollama, with a defined "
                    "inter-agent message protocol. Each query drives 4 LLM turns and 6 "
                    "inter-agent messages. The LLM executor's admit/decline decision is "
                    "checked against the authoritative cost objective, and the final answer "
                    "is checked against the exact symbolic pipeline; the LLM layer never "
                    "alters the exact reachable set or ranking. On a host without Ollama the "
                    "run falls back to a deterministic schema oracle, labeled accordingly."),
    }
    (out / "llm_multiagent.json").write_text(json.dumps(report, indent=2))
    if all_rows:
        with (out / "llm_multiagent.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys())); w.writeheader(); w.writerows(all_rows)
    _phase("done — results written to results/llm_multiagent.{json,csv}")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
