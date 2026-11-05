from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from pyspark.sql import SparkSession, functions as F, Window


def generate(clicks_path: str, buys_path: str, output_dir: str, limit: int | None = None) -> None:
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    spark = SparkSession.builder.appName("yoochoose-candidate-generation").getOrCreate()
    clicks = spark.read.csv(clicks_path, schema="session_id long, timestamp timestamp, item_id long, category string")
    buys = spark.read.csv(buys_path, schema="session_id long, timestamp timestamp, item_id long, price double, quantity int")
    if limit:
        clicks, buys = clicks.limit(limit), buys.limit(limit)
    clicks = clicks.select((-F.abs(F.col("session_id"))).alias("session_id"), "timestamp", "item_id").withColumn("event", F.lit("click"))
    buys = buys.select("session_id", "timestamp", "item_id").withColumn("event", F.lit("buy"))
    events = clicks.unionByName(buys).dropDuplicates(["session_id", "timestamp", "item_id", "event"])
    popularity = events.groupBy("item_id").agg(F.count("*").alias("event_count"), F.sum(F.expr("event = 'buy'").cast("int")).alias("buy_count"))
    pop = popularity.withColumn("source", F.lit("popularity"))
    w = Window.partitionBy("session_id").orderBy(F.col("timestamp").asc())
    indexed = events.withColumn("row", F.row_number().over(w))
    train = indexed.filter(F.col("row") <= 20)
    als_input = train.select("session_id", "item_id").dropDuplicates()
    from pyspark.ml.recommendation import ALS
    als = ALS(userCol="session_id", itemCol="item_id", ratingCol="rating", implicitPrefs=True,
              rank=16, maxIter=5, regParam=0.1, coldStartStrategy="drop", seed=7)
    model = als.fit(als_input.withColumn("rating", F.lit(1.0)))
    cf = model.recommendForAllUsers(50).select("session_id", F.explode("recommendations").alias("r")) \
        .select("session_id", F.col("r.item_id").alias("item_id"), F.col("r.rating").alias("candidate_score")) \
        .withColumn("source", F.lit("als"))
    result = pop.select(F.lit(None).cast("long").alias("session_id"), "item_id", "event_count", "buy_count", F.lit(None).cast("double").alias("candidate_score"), "source") \
        .unionByName(cf.select("session_id", "item_id", F.lit(None).cast("long").alias("event_count"), F.lit(None).cast("long").alias("buy_count"), "candidate_score", "source"))
    result.write.mode("overwrite").parquet(str(Path(output_dir) / "candidates"))
    bq_project, bq_dataset = os.getenv("RECOMMENDER_BIGQUERY_PROJECT"), os.getenv("RECOMMENDER_BIGQUERY_DATASET")
    if bq_project and bq_dataset:
        result.write.format("bigquery").option("table", f"{bq_project}.{bq_dataset}.recommendation_candidates").mode("overwrite").save()
    spark.stop()


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--clicks", required=True); p.add_argument("--buys", required=True); p.add_argument("--output-dir", required=True); p.add_argument("--limit", type=int); a = p.parse_args(); generate(a.clicks, a.buys, a.output_dir, a.limit)
