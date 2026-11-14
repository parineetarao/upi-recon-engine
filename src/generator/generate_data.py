"""Synthetic UPI payment data: build ground truth, then derive 3 messy sources."""
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
N_TXNS = 1_000_000
N_MERCHANTS = 500
START_DATE = pd.Timestamp("2026-11-01")
N_DAYS = 7
AS_OF = pd.Timestamp("2026-11-09")      # settlement files exist up to this date
OUT = Path("data/raw")

BANKS = [f"BNK_{c}" for c in "ABCDEFGHIJ"]
BANK_SHARE = np.array([.22, .18, .15, .12, .10, .08, .06, .04, .03, .02])
BASE_FAIL = np.array([.020, .025, .030, .020, .035, .025, .030, .040, .030, .045])
SPIKE_BANK, SPIKE_DAY = "BNK_D", 3          # day index 3 = 2026-11-04
SPIKE_START_H, SPIKE_END_H, SPIKE_FAIL = 14, 16, 0.22

HOURS = np.arange(24)
HOUR_W = 0.3 + np.exp(-((HOURS - 20) ** 2) / 18) + 0.6 * np.exp(-((HOURS - 12) ** 2) / 10)
HOUR_W = HOUR_W / HOUR_W.sum()
MERCHANT_W = 1 / np.arange(1, N_MERCHANTS + 1) ** 1.1
MERCHANT_W = MERCHANT_W / MERCHANT_W.sum()


def mid(i):
    return f"M{i + 1:04d}"


def build_truth(rng):
    n = N_TXNS
    day = rng.integers(0, N_DAYS, n)
    hour = rng.choice(24, n, p=HOUR_W)
    ts = START_DATE + pd.to_timedelta(day * 86400 + hour * 3600 + rng.integers(0, 3600, n), unit="s")
    bank_idx = rng.choice(len(BANKS), n, p=BANK_SHARE)

    fail_p = BASE_FAIL[bank_idx].copy()
    in_spike = ((bank_idx == BANKS.index(SPIKE_BANK)) & (day == SPIKE_DAY)
                & (hour >= SPIKE_START_H) & (hour < SPIKE_END_H))
    fail_p[in_spike] = SPIKE_FAIL
    latency = rng.lognormal(5.5, 0.5, n) * np.where(in_spike, 3, 1)
    rupees = np.clip(rng.lognormal(6.0, 1.1, n), 1, 50_000)

    df = pd.DataFrame({
        "ts": ts,
        "payer_bank": np.array(BANKS)[bank_idx],
        "merchant_id": [mid(i) for i in rng.choice(N_MERCHANTS, n, p=MERCHANT_W)],
        "merchant_order_id": [f"ORD{i:010d}" for i in range(n)],
        "amount_paise": (rupees * 100).round().astype("int64"),
        "latency_ms": latency.round().astype("int64"),
        "success": rng.random(n) >= fail_p,
        "in_spike": in_spike,
        "scenario": "NORMAL",
    })

    # Retry/idempotency bug: same order charged twice under a new gateway id
    dup_idx = np.flatnonzero(df.success.to_numpy() & (rng.random(n) < 0.002))
    clones = df.iloc[dup_idx].copy()
    clones["ts"] = clones["ts"] + pd.to_timedelta(rng.integers(30, 300, len(clones)), unit="s")
    clones["scenario"] = "DUPLICATE_ORDER"
    df = pd.concat([df, clones], ignore_index=True)

    m = len(df)
    df["gw_txn_id"] = [f"GW{i:010d}" for i in range(m)]
    df["utr"] = (500_000_000_000 + rng.permutation(m)).astype("int64")   # unique 12-digit
    codes = rng.choice(["F01", "F02", "F03", "F04"], m, p=[.4, .3, .2, .1])  # simulated
    df["response_code"] = np.where(df.success, "00", codes)

    # Mutually exclusive break scenarios, drawn from one uniform number
    u = rng.random(m)
    ok = df.success & (df.scenario == "NORMAL")
    df.loc[ok & (u < 0.003), "scenario"] = "MISSING_SETTLEMENT"
    df.loc[ok & (u >= 0.003) & (u < 0.008), "scenario"] = "AMOUNT_MISMATCH"
    df.loc[ok & (u >= 0.008) & (u < 0.009), "scenario"] = "DOUBLE_SETTLEMENT"
    df.loc[ok & (u >= 0.009) & (u < 0.011), "scenario"] = "GW_FAILED_NPCI_SUCCESS"
    df.loc[~df.success & (u < 0.001), "scenario"] = "PHANTOM_SUCCESS"

    lag = rng.choice([1, 2, 3], m, p=[.90, .08, .02])        # T+1 / T+2 / T+3
    df["settle_date"] = df.ts.dt.normalize() + pd.to_timedelta(lag, unit="D")
    df["is_pending"] = df.success & (df.settle_date > AS_OF)
    return df


def build_bank_file(df, rng):
    m = len(df)
    fee = (df.amount_paise * 0.0015).round().astype("int64")      # simulated PG fee
    settled = df.amount_paise - fee
    squeezed = (settled * rng.uniform(0.5, 0.99, m)).round().astype("int64")
    settled = settled.where(df.scenario != "AMOUNT_MISMATCH", squeezed)

    in_c = df.success & ~df.is_pending & (df.scenario != "MISSING_SETTLEMENT")
    C = df.loc[in_c, ["utr", "merchant_id", "settle_date"]].copy()
    C["settled_amount_paise"] = settled[in_c]
    C["fee_paise"] = fee[in_c]

    dbl = C[df.loc[C.index, "scenario"] == "DOUBLE_SETTLEMENT"].copy()
    dbl["settle_date"] = dbl["settle_date"] + pd.Timedelta(days=1)
    dbl = dbl[dbl.settle_date <= AS_OF]

    n_orph = int(0.0005 * len(df))
    orph = pd.DataFrame({
        "utr": 700_000_000_000 + rng.permutation(n_orph),
        "merchant_id": [mid(i) for i in rng.integers(0, N_MERCHANTS, n_orph)],
        "settle_date": START_DATE + pd.to_timedelta(
            rng.integers(1, (AS_OF - START_DATE).days + 1, n_orph), unit="D"),
        "settled_amount_paise": (np.clip(rng.lognormal(6, 1.1, n_orph), 1, 50_000) * 100)
        .round().astype("int64"),
        "fee_paise": 0,
    })

    C = pd.concat([C, dbl, orph], ignore_index=True).sample(frac=1, random_state=SEED)
    C = C.reset_index(drop=True)
    C["settlement_id"] = [f"STL{i:010d}" for i in range(len(C))]
    C["settlement_date"] = C.settle_date.dt.strftime("%Y-%m-%d")
    C["batch_file"] = "SETTLE_" + C.settle_date.dt.strftime("%Y%m%d") + ".csv"
    cols = ["settlement_id", "utr", "merchant_id", "settled_amount_paise",
            "fee_paise", "settlement_date", "batch_file"]
    return C[cols], dbl, orph


def build_npci_file(df):
    return pd.DataFrame({
        "npci_txn_id": [f"NP{i:012d}" for i in range(len(df))],
        "gw_txn_id": df.gw_txn_id,
        "utr": df.utr.where(df.success).astype("Int64"),
        "payer_bank": df.payer_bank,
        "amount_paise": df.amount_paise,
        "response_code": df.response_code,
        "response_ts": df.ts + pd.to_timedelta(df.latency_ms, unit="ms"),
        "switch_latency_ms": df.latency_ms,
    })


def build_gateway_file(df, rng):
    m = len(df)
    st = np.where(df.success, "SUCCESS", "FAILED")
    st = np.where(df.scenario == "GW_FAILED_NPCI_SUCCESS", rng.choice(["FAILED", "PENDING"], m), st)
    st = np.where(df.scenario == "PHANTOM_SUCCESS", "SUCCESS", st)
    r = rng.random(m)                                    # messy casing
    st = np.where(r < .80, st, np.where(r < .90, pd.Series(st).str.title(), pd.Series(st).str.lower()))

    iso = df.ts.dt.strftime("%Y-%m-%d %H:%M:%S")         # messy timestamp formats
    alt = df.ts.dt.strftime("%d/%m/%Y %H:%M:%S")
    has_utr = df.success & (df.scenario != "GW_FAILED_NPCI_SUCCESS")

    A = pd.DataFrame({
        "gw_txn_id": df.gw_txn_id,
        "merchant_id": df.merchant_id,
        "merchant_order_id": df.merchant_order_id,
        "payer_bank": df.payer_bank,
        "amount_inr": [f"{x:,.2f}" for x in df.amount_paise / 100],   # "1,250.00"
        "status": st,
        "utr": df.utr.where(has_utr).astype("Int64"),
        "initiated_at": np.where(rng.random(m) < 0.7, iso, alt),
        "latency_ms": df.latency_ms + rng.integers(20, 120, m),
    })
    dup_lines = A.sample(frac=0.01, random_state=SEED)   # benign duplicate log lines
    return pd.concat([A, dup_lines]).sample(frac=1, random_state=SEED).reset_index(drop=True)


def main():
    rng = np.random.default_rng(SEED)
    OUT.mkdir(parents=True, exist_ok=True)
    t = build_truth(rng)
    C, dbl, orph = build_bank_file(t, rng)

    labels = t[["gw_txn_id", "utr", "merchant_id", "payer_bank", "amount_paise", "scenario"]].copy()
    labels["expected_state"] = np.where(t.is_pending, "PENDING", t.scenario)
    stale = (t.scenario == "DOUBLE_SETTLEMENT") & ~t.utr.isin(set(dbl.utr)) & ~t.is_pending
    labels.loc[stale, "expected_state"] = "NORMAL"       # 2nd copy would land after AS_OF
    orphan_labels = pd.DataFrame({"utr": orph.utr, "scenario": "ORPHAN_SETTLEMENT",
                                  "expected_state": "ORPHAN_SETTLEMENT"})
    labels = pd.concat([labels, orphan_labels], ignore_index=True)

    merchants = pd.DataFrame({
        "merchant_id": [mid(i) for i in range(N_MERCHANTS)],
        "merchant_name": [f"Merchant {i + 1}" for i in range(N_MERCHANTS)],
        "category": rng.choice(["Grocery", "Food", "Travel", "Utilities", "Retail", "Gaming"], N_MERCHANTS),
    })

    build_gateway_file(t, rng).to_csv(OUT / "gateway_events.csv", index=False)
    build_npci_file(t).to_csv(OUT / "npci_switch_logs.csv", index=False)
    C.to_csv(OUT / "bank_settlements.csv", index=False)
    merchants.to_csv(OUT / "merchants.csv", index=False)
    labels.to_csv(OUT / "truth_labels.csv", index=False)   # validation only, never read by the pipeline

    spike = t[t.in_spike]
    print(f"transactions: {len(t):,}")
    print(f"spike-window failure rate ({SPIKE_BANK}): {1 - spike.success.mean():.1%}")
    print(labels.expected_state.value_counts())


if __name__ == "__main__":
    main()