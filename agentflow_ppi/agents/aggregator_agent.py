"""Aggregator Agent: top-k selection, alias restoration, and provenance.

The aggregator owns the Result Aggregator stage. It selects the top-k ranked
candidates, restores surface identifiers from the canonical id map, and attaches
a provenance record naming every operator that touched the result.
"""

from __future__ import annotations

from typing import Dict, List, Mapping, Sequence


class AggregatorAgent:
    name = "aggregator"

    def __init__(self, id_to_name: Mapping[int, str] | None = None) -> None:
        self.id_to_name = dict(id_to_name or {})

    def aggregate(
        self,
        ranked: Sequence[int],
        top_k: int,
        operators_applied: Sequence[str],
    ) -> Dict[str, object]:
        chosen = list(ranked[:top_k])
        names = [self.id_to_name.get(v, str(v)) for v in chosen]
        provenance = {
            "operators_applied": list(operators_applied),
            "num_returned": len(names),
        }
        return {"ranked_names": names, "ranked_ids": chosen, "provenance": provenance}
