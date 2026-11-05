# Audit and validation report

## Executive summary

The original result claims were not all valid. `make_examples` used statistics
from future events, the evaluator ranked only items already seen in a session,
Spark ALS did not preserve its score, and the downloaded click and purchase
mirrors had disjoint session-ID namespaces. Therefore the former purchase
metrics and former report are withdrawn.

Fixes now use pre-event cumulative features, explicit sampled-unobserved click
negatives, a session/item/timestamp purchase-label rule, ALS candidates with a
popularity fallback, and API artifacts containing training events only.
Generated click-session IDs now use a negative namespace; buy IDs remain
positive, preventing accidental joins between the two mirrors.

## Restored artifact audit

The stash contained two different candidate runs. `artifacts_run9` and
`artifacts_run10` match the old counts of `1002164` rows and `993550`
non-null ALS scores. Those runs also contain `49899` distinct merged session
IDs, but that count is invalid as a session count: click and buy IDs overlap
for `1297` values. The corrected namespace-separated input has `51196`
session IDs (`25000` click IDs plus `26196` buy IDs). Its `artifacts_run13`
output contains `1028864` candidate rows, of which `1020250` have an ALS
score. Do not mix the old candidate totals with the corrected session count.

The old artifacts record ALS candidate rate `1.0` for 500 fitted-session
evaluation cases and popularity fallback rate `1.0` for 500 unseen-session
cases. The corrected validation sample has ALS candidate rate `0.516`; its
500-session test sample is cold-start and has fallback rate `1.0`. The
corrected run's ALS output covers `20405` sessions. These are distinct
checks, not one combined `500/500` metric.

## Findings and evidence

| finding | class | evidence/fix |
|---|---|---|
| Spark ALS treated session IDs as users | CODE-VERIFIED | `spark_candidates.py`; valid only for sessions seen by fitted ALS |
| unseen sessions had no explicit final-path fallback | CODE-VERIFIED | evaluator/API now return `fallback_popularity` |
| future global item statistics entered features | CODE-VERIFIED | `model.py` now updates counters after each event |
| evaluator omitted unseen next-item candidates | CODE-VERIFIED | candidates now come from ALS or popularity, excluding history |
| ALS score was discarded in output | CODE-VERIFIED | `candidate_score` is retained |
| click labels conflated click and buy events | CODE-VERIFIED | click head now uses click positives plus deterministic sampled-unobserved items |
| click/purchase mirrors share source session IDs | RESULT-VERIFIED | generated click IDs are now negative and buy IDs positive; no accidental namespace join |
| old AUC/logloss values | DOCUMENTATION-ONLY, withdrawn | not reproduced after correction |

## Current task semantics

The available click mirror contains sequence rows without original session
IDs; generated negative IDs are used for click sessions. The buy mirror retains
positive original IDs. Since those namespaces do not join, no valid clicked-item
purchase experiment is claimed. The model keeps a purchase head for fixtures
and future data with a valid join, but its current run has no positive purchase
examples and must not be used as a purchase result.

Click training is sampled classification: each observed click is positive and
one popularity-ranked item not previously seen in that session is sampled as a
candidate negative. The negative is not a confirmed non-click because no
impression log exists. Features are computed from events strictly before the
example. The evaluator holds out the final row after sorting by timestamp,
session ID, and item ID. The click mirror timestamps events only at day
granularity, so item ID breaks ties; the holdout is not the original
chronological next click. Evaluation history uses timestamps strictly earlier
than the target; for day-stamped click sessions that history is empty.

## Reproduced run

```bash
python -m personalized_recommender.pipeline --data-dir data/yoochoose \
  --output-dir artifacts_run13 --spark-limit 20000 --event-limit 50000
python -m personalized_recommender.evaluate --artifact-dir artifacts_run13 \
  --output-dir artifacts_run13/evaluation_test --k 10 --max-sessions 500 \
  --eval-split test
python -m personalized_recommender.evaluate --artifact-dir artifacts_run13 \
  --output-dir artifacts_run13/evaluation_validation --k 10 --max-sessions 500 \
  --eval-split validation
```

Measured input: 99985 events (50000 clicks and 49985 purchases), 51196
namespace-separated session IDs, and 16363 items. Spark candidate generation
was limited to 20000 input rows per mirror. It wrote 1028864 candidate rows,
of which 1020250 have a non-null ALS score. The corrected model produced
49999 training examples;
its click AUC was `0.9391278123124925` and click logloss was
`0.6534747741152953`. These are training-set diagnostics, not held-out model
quality, and are not resume claims. `buy_auc=0.0` because no joined purchase
positive exists.

Validation, 500 sessions, lambda `0.75`: Recall@10 `0.005`, NDCG@10
`0.021861206671551746`, coverage `0.01292233140395749`, popularity baseline
Recall@10 `0.0`, popularity baseline NDCG@10 `0.0`. ALS candidate rate was
`0.516`; fallback rate was `0.484`.

Untouched test, 500 sessions: Recall@10 `0.0`, NDCG@10 `0.0`, popularity
baseline Recall@10 `0.0`, popularity baseline NDCG@10 `0.0`. ALS candidate rate
was `0.0`; fallback rate was `1.0` in this slice. This is a valid cold-start
result, not hidden.

The corrected evaluator writes an MLflow run per lambda and reports
`recall_at_k`, `ndcg_at_k`, coverage, intra-list diversity, ALS rate, and
fallback rate. `recall_at_k` is hits divided by `k`; `ndcg_at_k` is discounted
gain without ideal-score normalization. No corrected aggregate quality table
is promoted as a resume result:
the available click mirror and generated session IDs do not support a
credible joined click/purchase comparison.

## Serving

Against `artifacts_run13`, loopback `POST /recommend`, 20 requests,
concurrency 2, unseen session `999999`:

| p50 | p99 |
|---:|---:|
| 44.80153800250264 ms | 53.94843099929858 ms |

Known and unseen sessions both returned HTTP 200 with 10 items. This is a
small local benchmark; it excludes model startup and is not production
capacity.

A fresh rerun on the restored checkout, after one excluded readiness request,
measured p50 `42.63 ms` and p99 `59.38 ms` for the same 20 requests and
concurrency. The benchmark's p99 calculation selects the maximum observation
for 20 requests. The resume claim's `33.6 ms / 44.7 ms` is not present in the
restored benchmark outputs and was not reproduced in this rerun.

## Tests

```bash
python -m pytest -q tests/test_personalized_recommender.py
# 3 passed
python -m compileall -q personalized_recommender
# passed
```

Fixtures cover prefix-only statistics, session/item purchase matching,
unseen-session fallback, MMR similarity selection, and empty candidates.

## Remaining limitations

- A valid click/purchase joined source is required before purchase metrics can
  be claimed; current mirrors cannot provide it.
- The pre-correction `49899` merged-ID count and `1002164 / 993550`
  candidate counts belong to `artifacts_run9`/`artifacts_run10`; the corrected
  run has `51196` namespace-separated IDs and `1028864 / 1020250` candidate
  counts.
- The original host repository had 92 passing and 104 skipped tests at
  extraction; the standalone project suite contains the 3 focused tests above.
- The training AUC is in-sample and must not be presented as generalization.
- The split holds the final sorted row per session and the penultimate row for
  validation, but day-level click timestamps and item-ID tie breaking mean it
  is not a faithful within-session chronology. The strict earlier-than history
  filter also leaves day-stamped click sessions with empty history. This is
  not a global timestamp-period split; cross-session long-term evolution
  remains unverified.
- Spark local execution used a 20000-row input limit per mirror (up to 40000
  click/buy rows); this proves Spark code execution, not full-input or
  billion-scale performance.
- No BigQuery credentials were available, so BigQuery load was not executed.
- No persistent user identity, online A/B test, retention/dwell label, or
  causal long-term evaluation exists.

## Resume audit

| claim | status |
|---|---|
| Spark ALS recommendation | verified under the measured 20000-row run; coverage is conditional |
| Popularity baseline | verified as cold-start fallback |
| multi-task click/purchase prediction | architecture verified; purchase result unsupported with current data join |
| temporal holdout | verified as final-event-per-session holdout |
| MMR trade-off | implementation verified; corrected aggregate result not promoted |
| former AUC/logloss values | unsupported; withdrawn |
| p50/p99 latency | verified only for the stated 20-request local unseen-session benchmark |

## Truthful resume bullets

- Implemented a PySpark `ALS` plus popularity candidate pipeline for Yoochoose
  sessions, preserving ALS scores and adding explicit unseen-session fallback.
- Built a shared-trunk, two-head PyTorch scorer with prefix-only features and
  sampled-unobserved click labels; purchase evaluation remains gated on a
  valid joined source.
- Added a per-session final-row holdout, MMR co-occurrence diversity, MLflow
  lambda runs, and regression tests for leakage and cold-start behavior. The
  click mirror does not preserve original within-session chronology.
- Served ALS and fallback recommendations through FastAPI; measured
  `42.63 ms` p50 and `59.38 ms` p99 in a fresh local 20-request benchmark
  after one excluded readiness request.
