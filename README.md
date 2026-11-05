# Personalized interaction recommender

End-to-end session-personalized recommender on the public Yoochoose clickstream.
The implementation is separate from `cve_rag` and intentionally reuses its
architecture pattern: Spark candidate generation, learned ranking, MMR,
MLflow evaluation, and FastAPI serving.

## Dataset decision

| dataset | fit | limitation |
|---|---|---|
| MovieLens-25M | user feedback and timestamps; good long-term user split | no native sessions; engagement is mostly one rating task |
| H&M Personalized Fashion Recommendations | customer IDs, timestamps, item metadata | sessionization is an extra assumption; no click signal |
| **Yoochoose** | native sessions, ordered clicks, and purchases; directly supports session context and click/buy tasks | anonymous sessions: no cross-session user identity |

Yoochoose is therefore the narrowest honest fit for the named in-session
requirement. It does not claim persistent-user personalization or online A/B
testing.

## Setup

Install the project and its test tools from the repository root:

```bash
python -m pip install -e '.[test]'
```

## Run

```bash
python -m personalized_recommender.download
python -m personalized_recommender.pipeline --data-dir data/yoochoose --output-dir artifacts
python -m personalized_recommender.evaluate --artifact-dir artifacts --output-dir artifacts/evaluation
uvicorn personalized_recommender.api:app --port 8001
```

The downloader uses public mirrors because the original challenge archive URL
is unavailable. The click mirror has generated session IDs and the buy mirror
has a different ID namespace; therefore this project does not claim a joined
purchase experiment. Every report number is written only after the
corresponding command succeeds on downloaded data. Spark candidate output is
Parquet locally; with
`RECOMMENDER_BIGQUERY_PROJECT` and `RECOMMENDER_BIGQUERY_DATASET` set, the
pipeline also loads the same rows into BigQuery.

`pipeline` trains a shared-hidden-layer, two-head PyTorch model. The current
validated task is sampled click classification; the purchase head is retained
only for a future source with joined session IDs. `evaluate` uses a
train/validation/test final-event session holdout, computes Recall@K, NDCG@K,
coverage, intra-list diversity, ALS/fallback rates, popularity baselines, and
an MLflow lambda sweep. Generated click-session IDs are negative; source buy
IDs remain positive, preventing accidental namespace joins.
`serve_benchmark` measures p50/p99 from actual concurrent HTTP requests.
