from __future__ import annotations

import argparse, itertools, json, subprocess, time
from pathlib import Path
import mlflow
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .mmr import mmr
from .model import make_examples, score
from .candidates import candidate_ids, load_als_candidates


def diversity(items: list[dict]) -> float:
    if len(items) < 2: return 0.0
    return 1.0 - sum(a["item_id"] == b["item_id"] for a, b in itertools.combinations(items, 2)) / (len(items) * (len(items) - 1) / 2)

def cooccurrence_similarity(train: pd.DataFrame) -> dict[tuple[int, int], float]:
    popular = train.item_id.value_counts().head(300).index
    profiles = train[train.item_id.isin(popular)].groupby("item_id").session_id.apply(lambda x: set(x)).to_dict(); result = {}
    for a, b in itertools.combinations(profiles, 2):
        union = profiles[a] | profiles[b]; result[(int(a), int(b))] = result[(int(b), int(a))] = len(profiles[a] & profiles[b]) / len(union) if union else 0.0
    return result


def candidate_features(history: pd.DataFrame, items: list[int], known: pd.DataFrame, target_time: pd.Timestamp) -> pd.DataFrame:
    pop = known[known.event == "click"].item_id.value_counts(); buy = known[known.event == "buy"].item_id.value_counts(); counts = history.item_id.value_counts()
    return pd.DataFrame([{"session_id": int(history.session_id.iloc[0]) if not history.empty else -1, "item_id": int(item), "timestamp": target_time,
                          "session_len": len(history), "recent_count": int(counts.get(item, 0)), "position": len(history), "hour": target_time.hour,
                          "item_pop": int(pop.get(item, 0)), "item_buy": int(buy.get(item, 0)), "click": 0, "buy": 0} for item in items])


def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("--artifact-dir", type=Path, required=True); p.add_argument("--output-dir", type=Path, required=True); p.add_argument("--k", type=int, default=10); p.add_argument("--max-sessions", type=int, default=2000); p.add_argument("--session-offset", type=int, default=0); p.add_argument("--eval-split", choices=["validation", "test"], default="test"); a = p.parse_args(); a.output_dir.mkdir(parents=True, exist_ok=True)
    train = pd.read_parquet(a.artifact_dir / "train_events.parquet").sort_values(["session_id", "timestamp"])
    test = pd.read_parquet(a.artifact_dir / ("validation_events.parquet" if a.eval_split == "validation" else "test_events.parquet")).sort_values(["session_id", "timestamp"])
    # Session IDs are the only identity in Yoochoose; this is deliberately a session holdout.
    similarities = cooccurrence_similarity(train); popularity = train[train.event == "click"].item_id.value_counts().head(200).index.astype(int).tolist(); als = load_als_candidates(a.artifact_dir / "candidates")
    rows = []; baseline_hits = []; baseline_ndcgs = []
    for lambda_ in [0.0, 0.25, 0.5, 0.75, 1.0]:
        hits, ndcgs, lists, selected_items, sources = [], [], [], set(), []
        for sid, truth_group in list(test.groupby("session_id"))[a.session_offset:a.session_offset + a.max_sessions]:
            truth = set(truth_group.item_id.astype(int)); target_time = truth_group.timestamp.min(); history = train[(train.session_id == sid) & (train.timestamp < target_time)]; known = train[train.timestamp < target_time]
            ids, source = candidate_ids(int(sid), history, popularity, als); sources.append(source)
            features = candidate_features(history, ids, known, target_time); features["rank_score"] = score(a.artifact_dir, features); items = features[["item_id", "rank_score"]].to_dict("records"); scores = {int(x["item_id"]): float(x["rank_score"]) for x in items}; result = mmr(items, scores, a.k, lambda_, similarities); returned = [int(x["item_id"]) for x in result]; selected_items.update(returned)
            if lambda_ == 0.0:
                baseline = [item for item in popularity if item not in set(history.item_id.astype(int))][:a.k]
                baseline_hits.append(len(set(baseline) & truth) / a.k)
                baseline_ndcgs.append(sum(1 / np.log2(i + 2) for i, item in enumerate(baseline) if item in truth))
            hits.append(len(set(returned) & truth) / a.k); ndcgs.append(sum(1 / np.log2(i + 2) for i, item in enumerate(returned) if item in truth)); lists.append(1.0 - sum(similarities.get((int(x["item_id"]), int(y["item_id"])), 0.0) for x, y in itertools.combinations(result, 2)) / max(1, len(result) * (len(result) - 1) / 2))
        row = {"lambda": lambda_, "recall_at_k": float(np.mean(hits) if hits else 0), "ndcg_at_k": float(np.mean(ndcgs) if ndcgs else 0), "coverage": len(selected_items) / max(1, train.item_id.nunique()), "intra_list_diversity": float(np.mean(lists) if lists else 0), "sessions": len(hits), "als_candidate_rate": sources.count("als") / max(1, len(sources)), "fallback_rate": sources.count("fallback_popularity") / max(1, len(sources)), "popularity_recall_at_k": float(np.mean(baseline_hits) if baseline_hits else 0), "popularity_ndcg_at_k": float(np.mean(baseline_ndcgs) if baseline_ndcgs else 0)}; rows.append(row)
        with mlflow.start_run(run_name=f"session-recommender-lambda-{lambda_:.2f}"):
            mlflow.log_params({"lambda": lambda_, "k": a.k, "holdout": "final_event_per_session", "eval_split": a.eval_split}); mlflow.log_metrics({k: v for k, v in row.items() if k != "lambda"})
    frame = pd.DataFrame(rows); frame.to_csv(a.output_dir / "lambda_sweep.csv", index=False)
    columns = list(frame.columns); lines = ["| " + " | ".join(columns) + " |", "|" + "---:|" * len(columns)]
    lines += ["| " + " | ".join(str(row[c]) for c in columns) + " |" for row in rows]
    (a.output_dir / "lambda_sweep.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    plt.figure(); plt.plot([x["lambda"] for x in rows], [x["recall_at_k"] for x in rows], marker="o", label="Recall@K"); plt.plot([x["lambda"] for x in rows], [x["intra_list_diversity"] for x in rows], marker="s", label="Diversity"); plt.xlabel("MMR lambda"); plt.legend(); plt.tight_layout(); plt.savefig(a.output_dir / "lambda_sweep.png", dpi=160); plt.close()
    print(json.dumps(rows))


if __name__ == "__main__": main()
