"""Ollama-backed planner for decomposing protein interaction queries.

The planner converts a natural-language or semi-structured query into a canonical
logical plan that matches the operator vocabulary used in the paper. The plan is
returned as a Python dictionary to simplify downstream execution.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import requests


CANONICAL_SCHEMA = {
    "query_type": "path|motif|ranking|hybrid",
    "entities": ["protein identifiers or aliases"],
    "constraints": {
        "edge_types": ["physical", "regulatory", "functional"],
        "max_hops": 3,
        "min_confidence": 0.7,
        "require_sequence_signal": True,
        "require_structure_signal": False,
    },
    "subqueries": [
        {
            "name": "candidate_generation",
            "operator": "typed_expand",
            "inputs": ["entity"],
            "params": {"edge_type": "regulatory", "max_hops": 2},
        }
    ],
    "ranking": {
        "mode": "multimodal_gnn",
        "top_k": 25,
    },
}


@dataclass(slots=True)
class OllamaPlannerConfig:
    model: str = "llama3.1:8b"
    host: str = "http://localhost:11434"
    temperature: float = 0.0
    timeout_sec: int = 60


class OllamaPlanner:
    """Planner agent that emits a canonical JSON plan.

    The planner intentionally constrains the output format to reduce the risk of
    free-form generations that are hard to execute safely. If the Ollama service
    is unavailable, the planner falls back to a deterministic heuristic plan.
    """

    def __init__(self, config: Optional[OllamaPlannerConfig] = None) -> None:
        self.config = config or OllamaPlannerConfig()

    def build_prompt(self, user_query: str) -> str:
        return (
            "You are a database query planner for multimodal protein interaction graphs. "
            "Decompose the input query into a canonical JSON plan. "
            "Use only these operators: typed_expand, motif_match, confidence_filter, "
            "neural_rerank, aggregate. "
            "Return JSON only.\n\n"
            f"Schema example:\n{json.dumps(CANONICAL_SCHEMA, indent=2)}\n\n"
            f"User query:\n{user_query}\n"
        )

    def plan_query(self, user_query: str) -> Dict[str, Any]:
        payload = {
            "model": self.config.model,
            "prompt": self.build_prompt(user_query),
            "stream": False,
            "options": {"temperature": self.config.temperature},
        }
        try:
            response = requests.post(
                f"{self.config.host}/api/generate",
                json=payload,
                timeout=self.config.timeout_sec,
            )
            response.raise_for_status()
            raw_text = response.json().get("response", "").strip()
            return json.loads(raw_text)
        except Exception:
            return self._heuristic_fallback(user_query)

    def _heuristic_fallback(self, user_query: str) -> Dict[str, Any]:
        query_lower = user_query.lower()
        edge_types: List[str] = []
        if "regulatory" in query_lower:
            edge_types.append("regulatory")
        if "physical" in query_lower:
            edge_types.append("physical")
        if not edge_types:
            edge_types = ["physical", "regulatory"]

        max_hops = 3 if "through" in query_lower or "path" in query_lower else 2
        min_conf = 0.8 if "0.8" in query_lower else 0.7

        plan: Dict[str, Any] = {
            "query_type": "hybrid",
            "entities": self._extract_entities(user_query),
            "constraints": {
                "edge_types": edge_types,
                "max_hops": max_hops,
                "min_confidence": min_conf,
                "require_sequence_signal": True,
                "require_structure_signal": True,
            },
            "subqueries": [
                {
                    "name": "candidate_generation",
                    "operator": "typed_expand",
                    "inputs": ["entities"],
                    "params": {"edge_types": edge_types, "max_hops": max_hops},
                },
                {
                    "name": "evidence_pruning",
                    "operator": "confidence_filter",
                    "inputs": ["candidate_generation"],
                    "params": {"min_confidence": min_conf},
                },
                {
                    "name": "semantic_ranking",
                    "operator": "neural_rerank",
                    "inputs": ["evidence_pruning"],
                    "params": {"model": "multimodal_gin"},
                },
                {
                    "name": "final_aggregation",
                    "operator": "aggregate",
                    "inputs": ["semantic_ranking"],
                    "params": {"top_k": 25},
                },
            ],
            "ranking": {"mode": "multimodal_gnn", "top_k": 25},
        }
        return plan

    @staticmethod
    def _extract_entities(user_query: str) -> List[str]:
        tokens = [token.strip(",.()") for token in user_query.split()]
        return [tok for tok in tokens if tok.isupper() and len(tok) >= 3] or ["TP53", "BRCA1"]


