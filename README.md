# Session-Based Yoochoose Recommender

A standalone session recommendation project using PySpark ALS for candidate
generation, a shared-trunk PyTorch ranker, popularity fallback, MMR
diversification, and a FastAPI endpoint. It does not depend on CVE-RAG.

## What it does

1. Reads click and purchase files from Yoochoose-compatible public mirrors.
2. Builds implicit-feedback ALS candidates with Spark. ALS fits on up to 20
   rows per session and generates up to 50 candidates per fitted session.
   Spark also writes per-item event counts. Serving and evaluation calculate
   their click-popularity fallback from training events when ALS has no
   eligible candidates for a session.
3. Trains a PyTorch model with a shared trunk and separate click and purchase
   heads. The model rank score blends click and purchase sigmoid outputs at
   `0.7` and `0.3`; the ranker reranks candidate IDs and does not use the ALS
   `candidate_score` as an input feature.
4. Applies MMR to balance ranker relevance against item co-occurrence
   similarity, then evaluates a per-session holdout and lambda sweep. Results
   go to CSV, Markdown, and PNG files and are logged to MLflow.
5. Serves recommendations from the training artifacts through FastAPI.

This is anonymous session-based recommendation. It does not provide a durable
user profile, an online experiment, or a validated purchase predictor.

## Data and interpretation limits

The downloader uses mirrors because the original Yoochoose challenge archive
is unavailable. It fetches a ZIP with the purchase file, then separately
downloads three click period text files:

| data | mirror | transformation |
|---|---|---|
| purchases | [Yoochoose ZIP mirror](https://raw.githubusercontent.com/Arrietti-li/Dataset/main/yoochoose.zip) | extracts `yoochoose-buys.dat`; retains its session IDs and timestamps |
| clicks | [ADER period files](https://github.com/doublemul/ADER/tree/master/data/YOOCHOOSE) | combines `period_0.txt` through `period_2.txt`; assigns synthetic session IDs and one timestamp per day |

The click mirror has no original session IDs or event times. The pipeline maps
click IDs to negative values and keeps purchase IDs positive to prevent
accidental joins between the two mirrors.

These limits affect what the evaluation means:

- Click and purchase rows cannot be joined reliably. The purchase head remains
  in the model architecture, but this dataset has no valid purchase positives;
  do not present purchase AUC or purchase quality results.
- Click timestamps from the mirror tie within a day. The pipeline sorts ties
  by session and item ID, so it cannot recover the original click order. The
  evaluator also builds session history using timestamps strictly earlier
  than the held-out row; for these day-stamped click sessions that history is
  empty. The final-row holdout is an exploratory proxy, not a faithful
  chronological next-click evaluation.
- Click negatives are popularity-ranked items not previously seen in the
  session. Yoochoose has no impression log, so these are sampled-unobserved
  candidates, not confirmed non-clicks.
- The reported `recall_at_k` field divides held-out hits by `k`. With one held
  out item per session, this is hit-per-k (precision-like), not conventional
  Recall@K.
- The `ndcg_at_k` field is discounted gain without ideal-score normalization;
  it is not conventional NDCG despite the output name.

See [REPORT.md](REPORT.md) for the audit, exact run conditions, measured
candidate counts, current limitations, and the local serving benchmark.

## Requirements and installation

- Python 3.11 or newer
- A Java runtime supported by the installed PySpark version
- `curl` for downloading the click mirror

Create an environment and install the project and test tools from the
repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
java -version
```

The runtime dependencies, including PySpark, are declared in `pyproject.toml`.
Local data, Spark output, model artifacts, and MLflow files are ignored by Git.

## Download and build

Download the mirrors:

```bash
python -m personalized_recommender.download
```

The default data directory is `data/yoochoose`. For the bounded run documented
in [REPORT.md](REPORT.md), build model and candidate artifacts with:

```bash
python -m personalized_recommender.pipeline \
  --data-dir data/yoochoose \
  --output-dir artifacts \
  --event-limit 50000 \
  --spark-limit 20000
```

`--event-limit` caps rows read from each click and purchase file for model and
evaluation data. `--spark-limit` independently caps rows read from each file
for candidate generation. Defaults are `200000` and `100000` respectively;
the measured report run used `50000` and `20000`.

The pipeline writes `events.parquet`, train/validation/test event files,
`ranker.pt`, `ranker_meta.json`, and a Parquet `candidates/` directory beneath
the output directory. ALS rows have a session ID and ALS `candidate_score`;
global count rows have `event_count` and `buy_count` with a null session ID and
candidate score. Serving and evaluation read ALS rows and build fallback lists
from training click counts.

## Evaluate

Run both validation and test sweeps:

```bash
python -m personalized_recommender.evaluate \
  --artifact-dir artifacts \
  --output-dir artifacts/evaluation_validation \
  --k 10 --max-sessions 500 --eval-split validation

python -m personalized_recommender.evaluate \
  --artifact-dir artifacts \
  --output-dir artifacts/evaluation_test \
  --k 10 --max-sessions 500 --eval-split test
```

The evaluator holds out the final row per derived session for test and the
penultimate row for validation when a session has at least three rows. It
compares the ranker/MMR list with a popularity baseline and records candidate
source rates, item coverage, discounted gain, and diversity for lambda values
`0`, `0.25`, `0.5`, `0.75`, and `1`. Outputs include `lambda_sweep.csv`,
`lambda_sweep.md`, and `lambda_sweep.png`; MLflow records one run per lambda.
Interpret these measurements under the mirror and metric limits above.

MMR selects each next item using:

```text
lambda_ * relevance - (1 - lambda_) * maximum_similarity_to_selected_items
```

Relevance is the blended ranker score; similarity comes from item/session
co-occurrence in the training data.

## Serve the API

Start from the repository root after building artifacts:

```bash
uvicorn personalized_recommender.api:app --host 127.0.0.1 --port 8001
```

The API reads `artifacts/` by default. Set `RECOMMENDER_ARTIFACT_DIR` before
starting Uvicorn to use a different artifact directory. Request
`POST /recommend` with `session_id` and optional `k` and `lambda_` fields:

| field | type | default/constraint |
|---|---|---|
| `session_id` | integer | required |
| `k` | integer | `10`; from 1 to 50 |
| `lambda_` | number | `0.5`; from 0 to 1 |

```bash
curl -sS -X POST http://127.0.0.1:8001/recommend \
  -H 'Content-Type: application/json' \
  -d '{"session_id":999999,"k":10,"lambda_":0.5}'
```

The response contains `session_id`, `candidate_source` (`als` or
`fallback_popularity`), and `items`. Each item includes `item_id`,
`rank_score`, and `score`; results are ordered by the MMR selection. Invalid
request values receive FastAPI validation errors. The fallback excludes items
in the known session history. An unseen session draws from the top 100 global
click-popular items, then returns up to `k` items.

Run the HTTP benchmark in a second terminal. Its defaults are 100 requests and
concurrency 10; use these options to match the 20-request report run:

```bash
python -m personalized_recommender.serve_benchmark \
  --url http://127.0.0.1:8001/recommend \
  --session-id 999999 --requests 20 --concurrency 2
```

The benchmark reports p50 and a value labeled p99. With 20 requests its p99
calculation selects the maximum observation, so treat it as a small-sample
upper bound, not a stable percentile or production capacity measurement.

## Tests

```bash
python -m pytest -q
python -m compileall -q personalized_recommender
```

The tests cover prefix-only features and session/item purchase-label matching,
unseen-session popularity fallback, MMR similarity selection, and empty
candidates.

## Optional BigQuery output

The Spark pipeline has an environment-variable hook for writing candidate
rows to BigQuery when both `RECOMMENDER_BIGQUERY_PROJECT` and
`RECOMMENDER_BIGQUERY_DATASET` are set. Local Parquet is the supported default.
BigQuery was not validated for this project; using the hook also requires a
Spark BigQuery connector compatible with the installed Spark version, which
is not bundled here.
