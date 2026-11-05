from __future__ import annotations

import os
from pathlib import Path
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

from .mmr import mmr
from .model import score
from .candidates import candidate_ids, load_als_candidates

ARTIFACT_DIR = Path(os.getenv("RECOMMENDER_ARTIFACT_DIR", "artifacts")); EVENTS = pd.read_parquet(ARTIFACT_DIR / "train_events.parquet") if (ARTIFACT_DIR / "train_events.parquet").exists() else None
PROFILES = EVENTS.groupby("item_id").session_id.apply(lambda x: set(x)).to_dict() if EVENTS is not None else {}
ALS = load_als_candidates(ARTIFACT_DIR / "candidates")
app = FastAPI(title="Session Personalized Recommender")


class Request(BaseModel):
    session_id: int; k: int = Field(default=10, ge=1, le=50); lambda_: float = Field(default=.5, ge=0, le=1)


@app.post("/recommend")
def recommend(payload: Request):
    if EVENTS is None: return {"error": "artifacts unavailable"}
    history = EVENTS[EVENTS.session_id == payload.session_id].sort_values("timestamp"); now = history.timestamp.max() if not history.empty else EVENTS.timestamp.max(); popularity = EVENTS[EVENTS.event == "click"].item_id.value_counts().head(100).index.astype(int).tolist(); ids, source = candidate_ids(payload.session_id, history, popularity, ALS); pop = EVENTS[EVENTS.event == "click"].item_id.value_counts(); buy = EVENTS[EVENTS.event == "buy"].item_id.value_counts(); counts = history.item_id.value_counts(); candidates = pd.DataFrame([{"item_id": item, "item_pop": int(pop.get(item, 0)), "item_buy": int(buy.get(item, 0)), "session_len": len(history), "recent_count": int(counts.get(item, 0)), "position": len(history), "hour": now.hour, "click": 0, "buy": 0} for item in ids]); candidates["rank_score"] = score(ARTIFACT_DIR, candidates); items = candidates[["item_id", "rank_score"]].to_dict("records"); scores = {int(x["item_id"]): float(x["rank_score"]) for x in items}; similarity = {}; ids = [int(x["item_id"]) for x in items]
    for i, left in enumerate(ids):
        for right in ids[i + 1:]:
            union = PROFILES.get(left, set()) | PROFILES.get(right, set()); similarity[(left, right)] = similarity[(right, left)] = len(PROFILES.get(left, set()) & PROFILES.get(right, set())) / len(union) if union else 0.0
    return {"session_id": payload.session_id, "candidate_source": source, "items": mmr(items, scores, payload.k, payload.lambda_, similarity)}
