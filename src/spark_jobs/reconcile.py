"""Step 3b: 3-way reconciliation -> clean_transactions + reconciliation_breaks."""
from pyspark.sql import Window
from pyspark.sql import functions as F

from clean_sources import get_spark

STAGE = "data/processed/stage"
OUT = "data/processed"
AS_OF = "2026-11-09"        # recon run date; bank files exist up to this date
SETTLE_SLA_DAYS = 3         # a success is overdue once txn_date + 3 days <= AS_OF


def main():
    spark = get_spark()
    spark.sparkContext.setLogLevel("ERROR")
    as_of = F.lit(AS_OF).cast("date")

    gw = (spark.read.parquet(f"{STAGE}/gateway")
          .select("gw_txn_id", "merchant_id", "merchant_order_id",
                  F.col("payer_bank").alias("gw_bank"), "status",
                  F.col("utr").alias("gw_utr"), "initiated_ts",
                  F.col("amount_paise").alias("gw_amount_paise"),
                  F.lit(True).alias("in_gw")))
    npci = (spark.read.parquet(f"{STAGE}/npci")
            .select("gw_txn_id", F.col("utr").alias("npci_utr"),
                    F.col("payer_bank").alias("npci_bank"),
                    F.col("amount_paise").alias("npci_amount_paise"),
                    "response_code", "is_success", "response_ts",
                    F.lit(True).alias("in_npci")))
    merchants = spark.read.parquet(f"{STAGE}/merchants")

    # One row per UTR, so the join below cannot multiply transactions
    bank_agg = (spark.read.parquet(f"{STAGE}/bank").groupBy("utr").agg(
        F.count("*").alias("bank_rows"),
        F.min("settlement_date").alias("first_settle_date"),
        F.min_by("gross_paise", "settlement_date").alias("bank_gross_paise"),
        F.min_by("settled_amount_paise", "settlement_date").alias("bank_settled_paise"),
        F.min("merchant_id").alias("bank_merchant_id")))

    # 1) gateway <-> NPCI on gw_txn_id
    txn = (gw.join(npci, "gw_txn_id", "full_outer")
           .withColumn("in_gw", F.coalesce("in_gw", F.lit(False)))
           .withColumn("in_npci", F.coalesce("in_npci", F.lit(False)))
           .withColumn("gw_success", F.coalesce(F.col("status") == "SUCCESS", F.lit(False)))
           .withColumn("npci_success", F.coalesce("is_success", F.lit(False)))
           .withColumn("utr", F.coalesce("npci_utr", "gw_utr"))
           .withColumn("payer_bank", F.coalesce("npci_bank", "gw_bank"))
           .withColumn("amount_paise", F.coalesce("npci_amount_paise", "gw_amount_paise"))
           .withColumn("txn_date", F.to_date(F.coalesce("initiated_ts", "response_ts"))))

    # Duplicate order: 2nd and later successful charge on the same merchant_order_id
    w = (Window.partitionBy("merchant_order_id").orderBy("initiated_ts")
         .rowsBetween(Window.unboundedPreceding, Window.currentRow))
    txn = txn.withColumn("success_seq",
                         F.sum(F.when(F.col("gw_success"), 1).otherwise(0)).over(w))

    # 2) NPCI <-> bank on utr (bank already one row per utr); 3) merchant lookup (broadcast)
    txn = (txn.join(bank_agg, "utr", "left")
              .join(F.broadcast(merchants), "merchant_id", "left"))

    overdue = F.date_add("txn_date", SETTLE_SLA_DAYS) <= as_of
    state = (F.when(~F.col("in_npci"), "MISSING_IN_NPCI")
             .when(~F.col("in_gw"), "MISSING_IN_GATEWAY")
             .when(F.col("gw_success") & ~F.col("npci_success"), "PHANTOM_SUCCESS")
             .when(~F.col("gw_success") & F.col("npci_success"), "GW_FAILED_NPCI_SUCCESS")
             .when(F.col("gw_amount_paise") != F.col("npci_amount_paise"), "GW_NPCI_AMOUNT_MISMATCH")
             .when(F.col("gw_success") & (F.col("success_seq") > 1), "DUPLICATE_ORDER")
             .when(F.col("npci_success") & F.col("bank_rows").isNull(),
                   F.when(overdue, "MISSING_SETTLEMENT").otherwise("PENDING"))
             .when(F.col("npci_success") & (F.col("bank_rows") > 1), "DOUBLE_SETTLEMENT")
             .when(F.col("npci_success") & (F.col("bank_gross_paise") != F.col("npci_amount_paise")),
                   "AMOUNT_MISMATCH")
             .otherwise("MATCHED"))
    at_risk = (F.when(F.col("recon_state") == "MATCHED", 0)
               .when(F.col("recon_state") == "DOUBLE_SETTLEMENT",
                     F.col("bank_settled_paise") * (F.col("bank_rows") - 1))
               .when(F.col("recon_state") == "AMOUNT_MISMATCH",
                     F.abs(F.col("npci_amount_paise") - F.col("bank_gross_paise")))
               .otherwise(F.col("amount_paise")))

    txn = (txn.withColumn("recon_state", state)
              .withColumn("amount_at_risk_paise", at_risk)
              .withColumn("age_days", F.datediff(as_of, F.col("txn_date")))
              .cache())

    n = txn.count()
    print(f"transactions after joins: {n:,}   (must equal 1,001,985: no row fan-out)")
    txn.groupBy("recon_state").count().orderBy(F.desc("count")).show(truncate=False)

    # Orphans: bank UTRs that NPCI never saw (left anti join)
    npci_utrs = npci.select(F.col("npci_utr").alias("utr")).where(F.col("utr").isNotNull())
    orphans = (bank_agg.join(npci_utrs, "utr", "left_anti")
               .select(F.lit(None).cast("string").alias("gw_txn_id"), "utr",
                       F.col("bank_merchant_id").alias("merchant_id"),
                       F.col("first_settle_date").alias("txn_date"),
                       F.lit("ORPHAN_SETTLEMENT").alias("recon_state"),
                       F.col("bank_settled_paise").alias("amount_paise"),
                       F.col("bank_settled_paise").alias("amount_at_risk_paise"),
                       F.datediff(as_of, F.col("first_settle_date")).alias("age_days")))
    print(f"orphan settlements: {orphans.count():,}")

    breaks = (txn.filter("recon_state != 'MATCHED'")
              .select("gw_txn_id", "utr", "merchant_id", "payer_bank", "txn_date",
                      "recon_state", "amount_paise", "amount_at_risk_paise", "age_days")
              .unionByName(orphans, allowMissingColumns=True))

    clean = txn.select("gw_txn_id", "utr", "merchant_id", "merchant_name", "category",
                       "merchant_order_id", "payer_bank", "txn_date", "initiated_ts",
                       "status", "response_code", "amount_paise", "recon_state",
                       "bank_rows", "first_settle_date", "amount_at_risk_paise", "age_days")

    clean.write.mode("overwrite").partitionBy("txn_date").parquet(f"{OUT}/clean_transactions")
    breaks.write.mode("overwrite").partitionBy("txn_date").parquet(f"{OUT}/reconciliation_breaks")
    spark.stop()


if __name__ == "__main__":
    main()