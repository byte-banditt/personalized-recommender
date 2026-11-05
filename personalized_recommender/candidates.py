from __future__ import annotations

from pathlib import Path
import pandas as pd


def load_als_candidates(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["session_id", "item_id", "candidate_score", "source"])
    rows = pd.read_parquet(path)
    return rows[rows.source == "als"]


def candidate_ids(session_id: int, history: pd.DataFrame, popularity: list[int], als: pd.DataFrame, limit: int = 50) -> tuple[list[int], str]:
    seen = set(history.item_id.astype(int))
    als_ids = als.loc[als.session_id == session_id, "item_id"].dropna().astype(int).tolist()
    ids = [item for item in als_ids if item not in seen]
    if ids:
        return ids[:limit], "als"
    return [item for item in popularity if item not in seen][:limit], "fallback_popularity"
