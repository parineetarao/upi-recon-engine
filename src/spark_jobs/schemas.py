"""Explicit schemas for the raw CSVs. Column order must match the files."""
from pyspark.sql.types import (DateType, IntegerType, LongType, StringType,
                               StructField, StructType)

GATEWAY_SCHEMA = StructType([
    StructField("gw_txn_id", StringType(), False),
    StructField("merchant_id", StringType(), False),
    StructField("merchant_order_id", StringType(), False),
    StructField("payer_bank", StringType(), False),
    StructField("amount_inr", StringType(), True),      # messy text like "1,250.00"
    StructField("status", StringType(), True),          # messy casing
    StructField("utr", StringType(), True),             # null when the payment failed
    StructField("initiated_at", StringType(), True),    # two timestamp formats
    StructField("latency_ms", IntegerType(), True),
])

NPCI_SCHEMA = StructType([
    StructField("npci_txn_id", StringType(), False),
    StructField("gw_txn_id", StringType(), False),
    StructField("utr", StringType(), True),
    StructField("payer_bank", StringType(), False),
    StructField("amount_paise", LongType(), False),
    StructField("response_code", StringType(), False),  # string: "00", "F01"...
    StructField("response_ts", StringType(), True),     # parsed in cleaning
    StructField("switch_latency_ms", IntegerType(), True),
])

BANK_SCHEMA = StructType([
    StructField("settlement_id", StringType(), False),
    StructField("utr", StringType(), False),
    StructField("merchant_id", StringType(), False),
    StructField("settled_amount_paise", LongType(), False),
    StructField("fee_paise", LongType(), False),
    StructField("settlement_date", DateType(), False),
    StructField("batch_file", StringType(), True),
])

MERCHANT_SCHEMA = StructType([
    StructField("merchant_id", StringType(), False),
    StructField("merchant_name", StringType(), True),
    StructField("category", StringType(), True),
])