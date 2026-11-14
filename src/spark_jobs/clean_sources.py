"""Step 3a: read raw CSVs with explicit schemas, clean, dedupe, write staging Parquet."""
from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F

from schemas import BANK_SCHEMA, GATEWAY_SCHEMA, MERCHANT_SCHEMA, NPCI_SCHEMA

RAW = "data/raw"
STAGE = "data/processed/stage"


def get_spark():
    return (SparkSession.builder.master("local[*]").appName("upi-recon-clean")
            .config("spark.driver.memory", "2g")
            .config("spark.sql.shuffle.partitions", "8")        # default 200 is overkill for 1M rows
            .config("spark.sql.session.timeZone", "Asia/Kolkata")
            .getOrCreate())


def read_csv(spark, name, schema):
    return spark.read.csv(f"{RAW}/{name}", header=True, schema=schema,
                          mode="FAILFAST", dateFormat="yyyy-MM-dd")


def clean_gateway(df):
    ts = F.coalesce(
        F.try_to_timestamp("initiated_at", F.lit("yyyy-MM-dd HH:mm:ss")),
        F.try_to_timestamp("initiated_at", F.lit("dd/MM/yyyy HH:mm:ss")),
    )
    return (df
            .withColumn("initiated_ts", ts)
            .withColumn("status", F.upper(F.trim("status")))
            .withColumn("amount_paise",
                        (F.regexp_replace("amount_inr", ",", "").cast("decimal(18,2)") * 100).cast("long"))
            .drop("amount_inr", "initiated_at"))


def dedupe_gateway(df):
    w = Window.partitionBy("gw_txn_id").orderBy(
        F.col("initiated_ts").desc(), F.col("latency_ms").desc())
    return (df.withColumn("rn", F.row_number().over(w))
              .filter("rn = 1").drop("rn"))


def clean_npci(df):
    return (df
            .withColumn("response_ts",
                        F.try_to_timestamp("response_ts", F.lit("yyyy-MM-dd HH:mm:ss.SSS")))
            .withColumn("is_success", F.col("response_code") == "00"))


def clean_bank(df):
    # Deliberately NOT deduplicated: repeated UTRs are the double-settlement signal.
    return df.withColumn("gross_paise", F.col("settled_amount_paise") + F.col("fee_paise"))


def main():
    spark = get_spark()
    spark.sparkContext.setLogLevel("ERROR")

    gw_raw = read_csv(spark, "gateway_events.csv", GATEWAY_SCHEMA)
    gw = clean_gateway(gw_raw)
    gw_dedup = dedupe_gateway(gw)
    npci = clean_npci(read_csv(spark, "npci_switch_logs.csv", NPCI_SCHEMA))
    bank = clean_bank(read_csv(spark, "bank_settlements.csv", BANK_SCHEMA))
    merchants = read_csv(spark, "merchants.csv", MERCHANT_SCHEMA)

    n_raw, n_dedup, n_npci = gw_raw.count(), gw_dedup.count(), npci.count()
    print(f"gateway raw rows:        {n_raw:,}")
    print(f"gateway after dedupe:    {n_dedup:,}  (removed {n_raw - n_dedup:,})")
    print(f"npci rows:               {n_npci:,}")
    print(f"gateway unparsed times:  {gw_dedup.filter(F.col('initiated_ts').isNull()).count()}")
    print(f"gateway unparsed amount: {gw_dedup.filter(F.col('amount_paise').isNull()).count()}")
    print(f"npci unparsed times:     {npci.filter(F.col('response_ts').isNull()).count()}")
    print(f"bank rows:               {bank.count():,}")
    print(f"bank distinct UTRs:      {bank.select('utr').distinct().count():,}")

    gw_dedup.printSchema()
    gw_dedup.write.mode("overwrite").parquet(f"{STAGE}/gateway")
    npci.write.mode("overwrite").parquet(f"{STAGE}/npci")
    bank.write.mode("overwrite").parquet(f"{STAGE}/bank")
    merchants.write.mode("overwrite").parquet(f"{STAGE}/merchants")
    spark.stop()


if __name__ == "__main__":
    main()