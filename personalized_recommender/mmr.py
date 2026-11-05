from __future__ import annotations

import numpy as np


def mmr(items: list[dict], scores: dict[int, float], k: int, lambda_: float, similarity: dict[tuple[int, int], float] | None = None) -> list[dict]:
    if not 0 <= lambda_ <= 1: raise ValueError("lambda must be between 0 and 1")
    remaining = {int(x["item_id"]): x for x in items}; chosen = []
    while remaining and len(chosen) < k:
        def value(item_id):
            redundancy = max((similarity.get((item_id, other), 0.0) if similarity else (1.0 if item_id == other else 0.0) for other in chosen), default=0.0)
            return lambda_ * scores[item_id] - (1 - lambda_) * redundancy
        item_id = max(remaining, key=lambda x: (value(x), -x)); chosen.append(item_id); del remaining[item_id]
    return [{**next(x for x in items if int(x["item_id"]) == item_id), "score": scores[item_id]} for item_id in chosen]
