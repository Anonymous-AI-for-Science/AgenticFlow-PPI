#!/usr/bin/env python3
"""Replace the TODO-REAL placeholder macros in paper/placeholders_realdata.tex with
measured values from the result files produced by the real-data experiments.

This operationalizes TODO_REAL_DATA.md groups 1-3. It reads:
  * results/external_reranking_results.json   (group 1: real STRING/Reactome benchmark)
  * results/engine_baselines.json             (group 2: live cross-engine table)
  * results/strong_baselines.csv              (group 3: real-graph index sizes)
  * results/shrc_failure_modes.json           (group 3: filter-inconclusive fractions)
  * results/shrc_scaling*.csv  / *.json       (group 3: measured sigma at scale)  [optional]

For each macro it (a) pulls the measured value from the JSON/CSV, (b) rewrites the
\\newcommand line in placeholders_realdata.tex, and (c) strips the "TODO-REAL" marker
from that line's trailing comment so a later `grep -rn "TODO-REAL" paper/` shows what
is still outstanding.

SAFETY:
  * --dry-run prints every intended change without writing.
  * A timestamped .bak of the .tex file is written before any edit.
  * Group 1 refuses to apply if the run used fixtures ("using_fixture": true) unless
    --allow-fixture is passed (NEVER pass it for a real submission).
  * Macros whose source value is missing are reported and left untouched (still TODO).

USAGE (from the package root, after the experiments have been run):
  python scripts/replace_real_macros.py --dry-run        # preview
  python scripts/replace_real_macros.py                  # apply groups whose files exist
  python scripts/replace_real_macros.py --groups 1 2     # apply only specific groups
  python scripts/replace_real_macros.py --results results --tex paper/placeholders_realdata.tex
"""
from __future__ import annotations

import argparse
import csv
import datetime as _dt
import json
import re
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple


# --------------------------------------------------------------------------- #
# Formatting helpers
# --------------------------------------------------------------------------- #
def fmt_int(x: Any) -> str:
    """Integer with LaTeX thousands separators: 18412 -> '18{,}412'."""
    n = int(round(float(x)))
    s = f"{n:,}"
    return s.replace(",", "{,}")


def fmt_num(x: Any, places: int = 3) -> str:
    """Plain decimal, no trailing-zero trimming (keeps the paper's style)."""
    return f"{float(x):.{places}f}"


def fmt_num2(x: Any) -> str:
    return fmt_num(x, 2)


def fmt_ci(ci: Any) -> str:
    """[lo, hi] list/tuple -> '[0.706, 0.749]' (3 dp)."""
    lo, hi = float(ci[0]), float(ci[1])
    return f"[{lo:.3f}, {hi:.3f}]"


def fmt_rate_decline(call_rate: Any) -> str:
    """decline = 1 - call_rate, 2 dp."""
    return fmt_num2(1.0 - float(call_rate))


# A spec maps a macro name -> (how to pull the raw value, how to format it).
# `getter` receives the loaded source object(s) and returns the raw value, or
# raises KeyError/IndexError/TypeError when the field is absent.
Spec = Tuple[str, Callable[[Any], Any], Callable[[Any], str]]


# --------------------------------------------------------------------------- #
# Group 1 — external_reranking_results.json
# --------------------------------------------------------------------------- #
def group1_specs(j: Dict) -> List[Tuple[str, str]]:
    pd = j["protein_disjoint"]
    pw = j["pathway_disjoint"]
    out: List[Tuple[str, str]] = []

    def add(macro: str, value: str):
        out.append((macro, value))

    add("RealStringProteins", fmt_int(j["graph_proteins"]))
    add("RealStringEdges", fmt_int(j["graph_edges"]))
    add("RealStringPathways", fmt_int(j["num_pathways"]))
    add("RealQueriesProteinDisjoint", str(int(pd["test_decisions"])))
    add("RealQueriesPathwayDisjoint", str(int(pw["test_decisions"])))

    add("RealSymFPd", fmt_num(pd["symbolic_f1"]))
    add("RealSymCIpd", fmt_ci(pd["symbolic_ci"]))
    add("RealDispFPd", fmt_num(pd["dispatch_f1"]))
    add("RealDispCIpd", fmt_ci(pd["dispatch_ci"]))
    add("RealAlwaysFPd", fmt_num(pd["always_on_f1"]))
    add("RealAlwaysCIpd", fmt_ci(pd["always_on_ci"]))

    add("RealSymFPw", fmt_num(pw["symbolic_f1"]))
    add("RealDispFPw", fmt_num(pw["dispatch_f1"]))
    add("RealAlwaysFPw", fmt_num(pw["always_on_f1"]))

    add("RealDispCallRate", fmt_num2(pd["reranker_call_rate"]))
    add("RealDispDeclineRate", fmt_rate_decline(pd["reranker_call_rate"]))
    return out


# --------------------------------------------------------------------------- #
# Group 2 — engine_baselines.json
# --------------------------------------------------------------------------- #
ENGINE_NAME = {
    "shrc": "shrc-inproc",
    "pg": "postgres-recursive-cte",
    "neo": "neo4j-cypher",
}


def _engine_row(j: Dict, name: str) -> Dict:
    for e in j.get("engines", []):
        if e.get("engine") == name:
            return e
    raise KeyError(f"engine '{name}' not in engine_baselines.json")


def group2_specs(j: Dict) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    plan = [
        ("shrc", "EngShrc"),
        ("pg", "EngPg"),
        ("neo", "EngNeo"),
    ]
    for key, prefix in plan:
        row = _engine_row(j, ENGINE_NAME[key])
        if not row.get("available", False):
            print(f"  [skip] engine '{ENGINE_NAME[key]}' not available "
                  f"(available=false) -- its macros stay TODO-REAL", file=sys.stderr)
            continue
        eq = row.get("answer_equivalence", {})
        if not eq.get("all_match", False):
            print(f"  [warn] engine '{ENGINE_NAME[key]}' answers do NOT all match the "
                  f"BFS oracle -- refusing to apply its numbers", file=sys.stderr)
            continue
        out.append((f"{prefix}Load", fmt_num2(row["load_seconds"])))
        out.append((f"{prefix}Mem", fmt_num(row["peak_mem_mb"], 1)))
        out.append((f"{prefix}Cold", fmt_num2(row["mean_cold_ms"])))
        out.append((f"{prefix}Warm", fmt_num2(row["mean_warm_ms"])))
    return out


# --------------------------------------------------------------------------- #
# Group 3 — strong_baselines.csv + shrc_failure_modes.json + sigma
# --------------------------------------------------------------------------- #
def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def group3_specs(results: Path) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []

    # --- index entries from strong_baselines.csv (dataset == real-string) ---
    sb_path = results / "strong_baselines.csv"
    if sb_path.exists():
        rows = _read_csv(sb_path)
        real = [r for r in rows if r.get("dataset") == "real-string"]
        if not real:
            print("  [skip] strong_baselines.csv has no 'real-string' dataset rows "
                  "(ran on fixture?) -- index-size macros stay TODO-REAL", file=sys.stderr)
        else:
            def entries_for(index_substr: str) -> Optional[str]:
                for r in real:
                    if index_substr.lower() in r.get("index", "").lower():
                        return fmt_int(r["entries"])
                return None
            for macro, sub in [("RealShrcEntries", "shrc"),
                               ("RealPreachEntries", "preach"),
                               ("RealPllEntries", "pll")]:
                v = entries_for(sub)
                if v is None:
                    print(f"  [skip] no index matching '{sub}' in real-string rows "
                          f"-- {macro} stays TODO-REAL", file=sys.stderr)
                else:
                    out.append((macro, v))
    else:
        print("  [skip] results/strong_baselines.csv missing -- run "
              "benchmark_strong_baselines.py", file=sys.stderr)

    # --- inconclusive / fallback from shrc_failure_modes.json ---
    fm_path = results / "shrc_failure_modes.json"
    if fm_path.exists():
        fm = json.loads(fm_path.read_text())
        # the script may emit a list of per-dataset dicts or a single dict; handle both
        rec = None
        if isinstance(fm, list):
            rec = next((d for d in fm if "real" in str(d.get("dataset", "")).lower()), None)
            if rec is None and fm:
                rec = fm[0]
        elif isinstance(fm, dict):
            rec = fm
        if rec:
            if "grail_filter_inconclusive_frac" in rec:
                out.append(("RealGrailInconclusive", fmt_num2(rec["grail_filter_inconclusive_frac"])))
            if "preach_filter_inconclusive_frac" in rec:
                out.append(("RealPreachInconclusive", fmt_num2(rec["preach_filter_inconclusive_frac"])))
            if "shrc_core_inconclusive_frac" in rec:
                out.append(("RealShrcFallback", fmt_num(rec["shrc_core_inconclusive_frac"], 1)))
    else:
        print("  [skip] results/shrc_failure_modes.json missing -- inconclusive/"
              "fallback macros stay TODO-REAL", file=sys.stderr)

    # --- measured sigma at 5k/10k/20k from a scaling result, if present ---
    sigma = _load_sigma(results)
    for macro, key in [("RealSigmaFiveK", "5000"),
                       ("RealSigmaTenK", "10000"),
                       ("RealSigmaTwentyK", "20000")]:
        if key in sigma:
            out.append((macro, fmt_num2(sigma[key])))
        else:
            print(f"  [skip] no measured sigma at {key} proteins -- {macro} stays "
                  f"TODO-REAL", file=sys.stderr)
    return out


def _load_sigma(results: Path) -> Dict[str, float]:
    """Best-effort: look for a CSV/JSON with (num_nodes/proteins, sigma) rows at
    5k/10k/20k. Different scaling scripts name this differently, so we scan a few."""
    sigma: Dict[str, float] = {}
    candidates = list(results.glob("*scal*")) + list(results.glob("*string_scale*"))
    for path in candidates:
        try:
            if path.suffix == ".csv":
                for r in _read_csv(path):
                    n = r.get("num_nodes") or r.get("proteins") or r.get("nodes")
                    s = r.get("sigma") or r.get("residual_core_ratio") or r.get("measured_sigma")
                    if n and s:
                        sigma[str(int(float(n)))] = float(s)
            elif path.suffix == ".json":
                obj = json.loads(path.read_text())
                rows = obj if isinstance(obj, list) else obj.get("rows", [])
                for r in rows:
                    n = r.get("num_nodes") or r.get("proteins") or r.get("nodes")
                    s = r.get("sigma") or r.get("residual_core_ratio")
                    if n and s:
                        sigma[str(int(float(n)))] = float(s)
        except Exception as e:  # noqa: BLE001
            print(f"  [warn] could not parse {path.name} for sigma: {e}", file=sys.stderr)
    return sigma


# --------------------------------------------------------------------------- #
# .tex rewriting
# --------------------------------------------------------------------------- #
# The value may itself contain braces (LaTeX thousands separators, e.g. 18{,}412),
# so we match the whole macro body up to the LAST '}' that precedes the trailing
# comment. Anchoring on the trailing '%' comment (every line in this file has one)
# makes the value capture unambiguous.
LINE_RE_TMPL = (
    r"(?P<head>\\newcommand\{{\\{macro}\}}\{{)"
    r"(?P<val>.*)"
    r"(?P<tail>\}})"
    r"(?P<rest>\s*%.*)$"
)


def rewrite_macros(tex_path: Path, updates: List[Tuple[str, str]],
                   dry_run: bool) -> Tuple[int, List[str]]:
    text = tex_path.read_text()
    lines = text.splitlines(keepends=True)
    applied = 0
    not_found: List[str] = []

    by_macro = dict(updates)
    for macro, new_val in updates:
        pat = re.compile(LINE_RE_TMPL.format(macro=re.escape(macro)))
        hit_idx = None
        for i, ln in enumerate(lines):
            m = pat.match(ln.rstrip("\n"))
            if m:
                hit_idx = (i, m)
                break
        if hit_idx is None:
            not_found.append(macro)
            continue
        i, m = hit_idx
        old_val = m.group("val")
        rest = m.group("rest")
        # strip a TODO-REAL marker from the trailing comment (keep the human note)
        new_rest = re.sub(r"TODO-REAL\s*", "", rest)
        # if the comment becomes an empty "% " leave a tidy marker that it's measured
        if re.fullmatch(r"\s*%\s*", new_rest):
            new_rest = "  % measured"
        newline = f"\\newcommand{{\\{macro}}}{{{new_val}}}{new_rest}\n"
        eol = "\n" if lines[i].endswith("\n") else ""
        if not lines[i].endswith("\n"):
            newline = newline.rstrip("\n")
        print(f"  {macro:<26} {old_val!r:>22}  ->  {new_val!r}")
        if not dry_run:
            lines[i] = newline
        applied += 1

    if not dry_run and applied:
        ts = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        bak = tex_path.with_suffix(tex_path.suffix + f".{ts}.bak")
        bak.write_text(text)
        tex_path.write_text("".join(lines))
        print(f"\n  backup written: {bak}")
    return applied, not_found


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="results",
                    help="directory with the experiment result files (default: results)")
    ap.add_argument("--tex", default="paper/placeholders_realdata.tex",
                    help="path to placeholders_realdata.tex")
    ap.add_argument("--groups", nargs="*", type=int, choices=[1, 2, 3], default=[1, 2, 3],
                    help="which placeholder groups to apply (default: all available)")
    ap.add_argument("--dry-run", action="store_true",
                    help="print intended changes without writing")
    ap.add_argument("--allow-fixture", action="store_true",
                    help="DANGER: apply group-1 numbers even if the run used fixtures "
                         "(never use for a real submission)")
    args = ap.parse_args()

    results = Path(args.results)
    tex = Path(args.tex)
    if not tex.exists():
        print(f"error: tex file not found: {tex}", file=sys.stderr)
        return 2

    all_updates: List[Tuple[str, str]] = []

    # ---- Group 1 ----
    if 1 in args.groups:
        p = results / "external_reranking_results.json"
        if p.exists():
            j = json.loads(p.read_text())
            if j.get("using_fixture", True) and not args.allow_fixture:
                print("== Group 1: SKIPPED -- external_reranking_results.json was produced "
                      "from FIXTURES (using_fixture=true). Run download_external_data.py + "
                      "build_external_manifest.py on a networked host, then re-run. "
                      "(Override with --allow-fixture, never for submission.)\n",
                      file=sys.stderr)
            else:
                print("== Group 1: real STRING/Reactome benchmark ==")
                g1 = group1_specs(j)
                all_updates += g1
                print()
        else:
            print(f"== Group 1: SKIPPED -- {p} not found ==\n", file=sys.stderr)

    # ---- Group 2 ----
    if 2 in args.groups:
        p = results / "engine_baselines.json"
        if p.exists():
            j = json.loads(p.read_text())
            print("== Group 2: live cross-engine table ==")
            all_updates += group2_specs(j)
            print()
        else:
            print(f"== Group 2: SKIPPED -- {p} not found ==\n", file=sys.stderr)

    # ---- Group 3 ----
    if 3 in args.groups:
        print("== Group 3: real-graph index sizes / sigma ==")
        all_updates += group3_specs(results)
        print()

    if not all_updates:
        print("Nothing to apply. Run the experiments first (see TODO_REAL_DATA.md), "
              "then re-run this script.", file=sys.stderr)
        return 1

    print(f"== Rewriting {len(all_updates)} macro(s) in {tex} "
          f"({'dry-run' if args.dry_run else 'APPLY'}) ==")
    applied, not_found = rewrite_macros(tex, all_updates, args.dry_run)

    print(f"\nSummary: {applied} macro(s) "
          f"{'would be ' if args.dry_run else ''}updated.")
    if not_found:
        print(f"  WARNING: {len(not_found)} macro(s) not found in {tex}: "
              f"{', '.join(not_found)}", file=sys.stderr)

    # remind about remaining TODO-REAL cells
    remaining = sum(1 for ln in tex.read_text().splitlines() if "TODO-REAL" in ln) \
        if not args.dry_run else None
    if remaining is not None:
        print(f"  Remaining TODO-REAL lines in the file: {remaining} "
              f"(target for submission: 0)")
        if remaining:
            print("  -> run the missing experiments (groups not yet applied) and re-run, "
                  "or check the [skip]/[warn] notes above.")
    print("\nNext: recompile both PDFs and verify 0 undefined / 0 overfull / "
          "References at p13 top (see LOCAL_EXECUTION_RUNBOOK.ko.md, step 5).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
