from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

from .model import make_examples, fit_ranker
from .spark_candidates import generate


def read_events(data_dir: Path, limit: int | None = None) -> pd.DataFrame:
    c = pd.read_csv(data_dir / "yoochoose-clicks.dat", names=["session_id", "timestamp", "item_id", "category"], parse_dates=["timestamp"], nrows=limit)
    c["session_id"] = -c["session_id"].abs()
    c["event"] = "click"
    b = pd.read_csv(data_dir / "yoochoose-buys.dat", names=["session_id", "timestamp", "item_id", "price", "quantity"], parse_dates=["timestamp"], nrows=limit)
    b = b[["session_id", "timestamp", "item_id"]]; b["event"] = "buy"
    return pd.concat([c[["session_id", "timestamp", "item_id", "event"]], b], ignore_index=True).drop_duplicates()


def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("--data-dir", type=Path, required=True); p.add_argument("--output-dir", type=Path, required=True); p.add_argument("--spark-limit", type=int, default=100000); p.add_argument("--event-limit", type=int, default=200000); a = p.parse_args()
    a.output_dir.mkdir(parents=True, exist_ok=True); events = read_events(a.data_dir, a.event_limit); events.to_parquet(a.output_dir / "events.parquet")
    generate(str(a.data_dir / "yoochoose-clicks.dat"), str(a.data_dir / "yoochoose-buys.dat"), str(a.output_dir), a.spark_limit)
    ordered = events.sort_values(["session_id", "timestamp", "item_id"])
    test_indices = ordered.groupby("session_id").tail(1).index
    session_sizes = ordered.groupby("session_id").size()
    validation_pool = ordered.drop(test_indices)
    validation_pool = validation_pool[validation_pool.session_id.map(session_sizes) >= 3]
    validation_indices = validation_pool.groupby("session_id").tail(1).index
    train = events.drop(test_indices.union(validation_indices)).reset_index(drop=True)
    events.loc[validation_indices].to_parquet(a.output_dir / "validation_events.parquet")
    events.loc[test_indices].to_parquet(a.output_dir / "test_events.parquet")
    train.to_parquet(a.output_dir / "train_events.parquet")
    cutoff = train.timestamp.max(); frame, _ = make_examples(train, cutoff); metrics = fit_ranker(frame, a.output_dir)
    print({"events": len(events), "sessions": events.session_id.nunique(), "items": events.item_id.nunique(), "train_rows": len(frame), **metrics})


if __name__ == "__main__": main()
