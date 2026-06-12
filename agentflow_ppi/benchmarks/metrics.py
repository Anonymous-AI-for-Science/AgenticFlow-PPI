from __future__ import annotations

import math
from typing import Sequence


def pearson(xs: Sequence[float], ys: Sequence[float]) -> float:
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    denx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    deny = math.sqrt(sum((y - my) ** 2 for y in ys))
    return num / max(1e-9, denx * deny)


def spearman(xs: Sequence[float], ys: Sequence[float]) -> float:
    rx = {v: i for i, v in enumerate(sorted(xs))}
    ry = {v: i for i, v in enumerate(sorted(ys))}
    return pearson([rx[x] for x in xs], [ry[y] for y in ys])


def mape(pred: Sequence[float], actual: Sequence[float]) -> float:
    vals = [abs(p - a) / max(1.0, abs(a)) for p, a in zip(pred, actual)]
    return 100.0 * sum(vals) / len(vals)


