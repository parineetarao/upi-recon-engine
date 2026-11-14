"""DEV ONLY: compare engine output with the generator's answer key. Never part of the pipeline."""
from pyspark.sql import functions as F

from clean_sources import get_spark

spark = get_spark()
spark.sparkContext.setLogLevel("ERROR")

labels = spark.read.csv("data/raw/truth_labels.csv", header=True)
expected = (labels.filter(F.col("gw_txn_id").isNotNull())
            .select("gw_txn_id", F.col("expected_state").alias("expected")))
engine = (spark.read.parquet("data/processed/clean_transactions")
          .select("gw_txn_id", "recon_state")
          .withColumn("engine", F.when(F.col("recon_state") == "MATCHED", "NORMAL")
                                 .otherwise(F.col("recon_state"))))

(expected.join(engine, "gw_txn_id", "left")
 .groupBy("expected", "engine").count()
 .orderBy("expected", F.desc("count")).show(60, truncate=False))

exp_orph = labels.filter(F.col("expected_state") == "ORPHAN_SETTLEMENT").count()
got_orph = (spark.read.parquet("data/processed/reconciliation_breaks")
            .filter("recon_state = 'ORPHAN_SETTLEMENT'").count())
print(f"orphans expected {exp_orph}, engine found {got_orph}")
spark.stop()