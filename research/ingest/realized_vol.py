"""
Family: realized_vol  (HC #563 R2 — multi-horizon realised volatility)

WHAT: Per-(ticker, date) realised volatility at 5d / 20d / 60d / 252d horizons,
plus the Yang-Zhang OHLC estimator (more efficient than close-to-close at the
weekly cadence). Computed from daily OHLCV.

SOURCE: wheel_strategy_v1/data/cache/prices_v2.parquet

OUTPUT: data/feature_store/realized_vol/daily.parquet
Schema:
  ticker, date,
  ret_d                     daily log-return
  rv_cc_5d / 20d / 60d / 252d   close-to-close annualised vol (sqrt(252)*std(ret))
  rv_yz_20d / 60d           Yang-Zhang annualised vol (uses O/H/L/C)
  rv_pk_20d                 Parkinson high-low estimator (annualised)
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, "/home/jupiter/Lvl3Quant/research/ingest")
from _common import write_parquet, smoke_log  # type: ignore

FAMILY = "realized_vol"
SRC = Path("/home/jupiter/Lvl3Quant/wheel_strategy_v1/data/cache/prices_v2.parquet")
ANN = float(np.sqrt(252.0))


def _rv_cc(s: pd.Series, w: int) -> pd.Series:
    return s.rolling(w, min_periods=max(2, w // 2)).std() * ANN


def _parkinson(high, low, w):
    """Parkinson estimator: sigma^2 = (1/(4 ln 2)) * mean(ln(H/L)^2). Annualise sqrt(252)."""
    lr = np.log(high / low)
    var = (lr ** 2) / (4.0 * np.log(2.0))
    return np.sqrt(var.rolling(w, min_periods=max(2, w // 2)).mean()) * ANN


def _yang_zhang(open_, high, low, close, w):
    """Yang-Zhang estimator (close-to-open + open-to-close Rogers-Satchell)."""
    ln_ho = np.log(high / open_)
    ln_lo = np.log(low  / open_)
    ln_co = np.log(close / open_)
    ln_oc = np.log(open_ / close.shift(1))
    rs = ln_ho * (ln_ho - ln_co) + ln_lo * (ln_lo - ln_co)  # Rogers-Satchell
    sigma_o = ln_oc.rolling(w, min_periods=max(2, w // 2)).var()
    sigma_c = ln_co.rolling(w, min_periods=max(2, w // 2)).var()
    sigma_rs = rs.rolling(w, min_periods=max(2, w // 2)).mean()
    k = 0.34 / (1.34 + (w + 1) / (w - 1))
    return np.sqrt(sigma_o + k * sigma_c + (1 - k) * sigma_rs) * ANN


def main():
    df = pd.read_parquet(SRC)
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    print(f"[{FAMILY}] input shape={df.shape}")

    df["log_ret_close"] = np.log(df.groupby("ticker")["close"].pct_change().add(1.0))

    out_parts = []
    for tk, sub in df.groupby("ticker", sort=False):
        sub = sub.copy()
        s = sub["log_ret_close"]
        sub["rv_cc_5d"]   = _rv_cc(s, 5)
        sub["rv_cc_20d"]  = _rv_cc(s, 20)
        sub["rv_cc_60d"]  = _rv_cc(s, 60)
        sub["rv_cc_252d"] = _rv_cc(s, 252)
        sub["rv_pk_20d"]  = _parkinson(sub["high"], sub["low"], 20)
        sub["rv_yz_20d"]  = _yang_zhang(sub["open"], sub["high"], sub["low"], sub["close"], 20)
        sub["rv_yz_60d"]  = _yang_zhang(sub["open"], sub["high"], sub["low"], sub["close"], 60)
        sub["ret_d"]      = s
        out_parts.append(sub[["ticker","date","ret_d",
                              "rv_cc_5d","rv_cc_20d","rv_cc_60d","rv_cc_252d",
                              "rv_pk_20d","rv_yz_20d","rv_yz_60d"]])
    out = pd.concat(out_parts, ignore_index=True)
    out = out.dropna(subset=["rv_cc_20d"]).reset_index(drop=True)

    p = write_parquet(out, FAMILY, "daily.parquet")
    smoke_log(FAMILY, True, f"{len(out)} rows, {out['ticker'].nunique()} tickers -> {p}")
    print(f"OK {FAMILY}: {len(out)} rows, {out['ticker'].nunique()} tickers -> {p}")
    print(out.tail(3).to_string())
    return p


def run_full():
    return main()


if __name__ == "__main__":
    main()
