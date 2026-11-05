from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import log_loss, roc_auc_score
from torch import nn

FEATURES = ["item_pop", "item_buy", "session_len", "recent_count", "position", "hour"]


class MultiTaskRanker(nn.Module):
    def __init__(self, n_features: int):
        super().__init__()
        self.shared = nn.Sequential(nn.Linear(n_features, 32), nn.ReLU(), nn.Linear(32, 16), nn.ReLU())
        self.click = nn.Linear(16, 1)
        self.buy = nn.Linear(16, 1)

    def forward(self, x):
        h = self.shared(x)
        return self.click(h).squeeze(-1), self.buy(h).squeeze(-1)


def make_examples(events: pd.DataFrame, train_end: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Make click-positive plus deterministic sampled-unobserved examples.

    Counts are snapshots immediately before each event.  The sampled item is
    a candidate, not a confirmed non-click; this is a sampled classification
    task because Yoochoose has no impression log.
    """
    events = events.sort_values(["timestamp", "session_id", "item_id"]).copy()
    valid_sessions = set(events.groupby("session_id").timestamp.max().loc[lambda x: x <= train_end].index)
    purchases = {(int(s), int(i)): set(g.timestamp) for (s, i), g in events[events.event == "buy"].groupby(["session_id", "item_id"])}
    pop_counts: dict[int, int] = {}; buy_counts: dict[int, int] = {}; histories: dict[int, list[int]] = {}; rows = []
    for row in events.itertuples(index=False):
        sid, item = int(row.session_id), int(row.item_id)
        if sid in valid_sessions and row.event == "click":
            history = histories.get(sid, []); candidates = sorted(pop_counts, key=lambda x: (-pop_counts[x], x))
            negative = next((x for x in candidates if x not in history and x != item), None)
            buy_label = int(any(t >= row.timestamp for t in purchases.get((sid, item), set())))
            base = {"session_id": sid, "timestamp": row.timestamp, "session_len": len(history), "recent_count": history.count(item), "position": len(history), "hour": row.timestamp.hour}
            rows.append({**base, "item_id": item, "click": 1, "buy": buy_label, "item_pop": pop_counts.get(item, 0), "item_buy": buy_counts.get(item, 0)})
            if negative is not None:
                rows.append({**base, "item_id": negative, "click": 0, "buy": 0, "item_pop": pop_counts[negative], "item_buy": buy_counts.get(negative, 0)})
        histories.setdefault(sid, []).append(item)
        pop_counts[item] = pop_counts.get(item, 0) + int(row.event == "click")
        buy_counts[item] = buy_counts.get(item, 0) + int(row.event == "buy")
    frame = pd.DataFrame(rows)
    return frame, events


def fit_ranker(frame: pd.DataFrame, artifact_dir: Path, epochs: int = 8) -> dict[str, float]:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(7)
    x = frame[FEATURES].astype(float)
    mean, std = x.mean(), x.std().replace(0, 1)
    xn = ((x - mean) / std).to_numpy(dtype=np.float32)
    model = MultiTaskRanker(len(FEATURES)); optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    xt = torch.tensor(xn); yc = torch.tensor(frame.click.to_numpy(dtype=np.float32)); yb = torch.tensor(frame.buy.to_numpy(dtype=np.float32))
    for _ in range(epochs):
        optimizer.zero_grad(); pc, pb = model(xt)
        (nn.functional.binary_cross_entropy_with_logits(pc, yc) + nn.functional.binary_cross_entropy_with_logits(pb, yb)).backward(); optimizer.step()
    with torch.no_grad(): pc, pb = model(xt); pc, pb = torch.sigmoid(pc).numpy(), torch.sigmoid(pb).numpy()
    metrics = {"click_auc": float(roc_auc_score(yc, pc)) if len(set(yc.tolist())) > 1 else 0.0, "click_logloss": float(log_loss(yc, pc, labels=[0, 1])),
               "buy_auc": float(roc_auc_score(yb, pb)) if len(set(yb.tolist())) > 1 else 0.0,
               "buy_logloss": float(log_loss(yb, pb, labels=[0, 1]))}
    torch.save(model.state_dict(), artifact_dir / "ranker.pt")
    (artifact_dir / "ranker_meta.json").write_text(json.dumps({"mean": mean.to_dict(), "std": std.to_dict(), "metrics": metrics}, indent=2), encoding="utf-8")
    return metrics


def score(model_dir: Path, frame: pd.DataFrame) -> np.ndarray:
    meta = json.loads((model_dir / "ranker_meta.json").read_text(encoding="utf-8")); model = MultiTaskRanker(len(FEATURES)); model.load_state_dict(torch.load(model_dir / "ranker.pt", map_location="cpu", weights_only=True)); model.eval()
    x = frame[FEATURES].astype(float); x = ((x - pd.Series(meta["mean"])) / pd.Series(meta["std"])).to_numpy(dtype=np.float32)
    with torch.no_grad():
        click, buy = model(torch.tensor(x)); return (0.7 * torch.sigmoid(click) + 0.3 * torch.sigmoid(buy)).numpy()
