from pathlib import Path

import pandas as pd

from personalized_recommender.candidates import candidate_ids
from personalized_recommender.mmr import mmr
from personalized_recommender.model import make_examples


def events(rows):
    return pd.DataFrame(rows, columns=["session_id", "timestamp", "item_id", "event"]).assign(timestamp=lambda x: pd.to_datetime(x.timestamp))


def test_features_use_only_events_before_prediction_and_purchase_matches_session_item():
    frame, _ = make_examples(events([
        (1, "2024-01-01 00:00", 10, "click"),
        (1, "2024-01-01 00:01", 20, "click"),
        (1, "2024-01-01 00:02", 20, "buy"),
        (2, "2024-01-01 00:00", 30, "click"),
        (2, "2024-01-01 00:01", 40, "click"),
    ]), pd.Timestamp("2024-01-01 00:02"))
    positive = frame[(frame.session_id == 1) & (frame.item_id == 20) & (frame.click == 1)].iloc[0]
    assert positive.item_pop == 0
    assert positive.item_buy == 0
    assert positive.buy == 1
    assert set(frame.click) == {0, 1}


def test_unseen_session_uses_explicit_popularity_fallback():
    history = events([(7, "2024-01-01", 10, "click")])
    als = pd.DataFrame({"session_id": [8], "item_id": [11]})
    ids, source = candidate_ids(7, history, [10, 11, 12], als)
    assert source == "fallback_popularity"
    assert ids == [11, 12]


def test_mmr_uses_similarity_and_empty_candidates():
    items = [{"item_id": 1}, {"item_id": 2}, {"item_id": 3}]
    scores = {1: 1.0, 2: .99, 3: .8}
    similarity = {(1, 2): .99, (2, 1): .99, (1, 3): 0.0, (3, 1): 0.0}
    assert [x["item_id"] for x in mmr(items, scores, 2, 0.5, similarity)] == [1, 3]
    assert mmr([], {}, 10, .5) == []
