from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass(slots=True)
class BiologicalQuery:
    source: str
    target: str
    modality: str
    min_score: float
    description: str


def load_biological_queries(path: str | Path) -> List[BiologicalQuery]:
    payload = json.loads(Path(path).read_text())
    return [BiologicalQuery(**item) for item in payload]


