"""Reachability Agent: exact SHRC pruning and selectivity evidence.

This agent owns the exact symbolic-reduction stage (paper goal G2). Given a
candidate frontier produced by typed expansion, it uses the SHRC index to keep
only candidates that are exactly reachable from the source and that can exactly
reach the target on the acyclic execution snapshot. It then emits a selectivity
report (pre/post frontier sizes) that the executor uses as cost-model evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

from agentflow_ppi.reachability import SHRCIndex


@dataclass(slots=True)
class ReachabilityReport:
    pre_frontier: int
    post_frontier: int
    kept: List[int]
    selectivity: float  # post / pre


class ReachabilityAgent:
    name = "reachability"

    def __init__(self, index: SHRCIndex) -> None:
        self.index = index

    def prune(self, source: int, target: int, candidates: Sequence[int]) -> ReachabilityReport:
        pre = len(candidates)
        kept: List[int] = []
        for v in candidates:
            if v == source or v == target:
                continue
            if self.index.reachable(source, v) and self.index.reachable(v, target):
                kept.append(v)
        post = len(kept)
        selectivity = (post / pre) if pre else 0.0
        return ReachabilityReport(pre_frontier=pre, post_frontier=post, kept=kept, selectivity=selectivity)
