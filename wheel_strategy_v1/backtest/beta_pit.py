"""
beta_pit.py — Point-in-time rolling-beta computation for the wheel.

Per HC #555 R2 (prerequisite for HC #552 high-β sleeve).
Source: extracted/adapted from the MacroStrategy engine.py beta formula
        (beta = cov(asset, SPY) / var(SPY) over a rolling window).

Key property: BETA AT DATE D USES ONLY RETURNS THROUGH D-1.
No lookahead. Result column is `.shift(1)` before being consumed by
the wheel selector — but the function itself only ever sees the
historical returns slice you pass in.

Usage:
    from backtest.beta_pit import compute_pit_beta
    beta_df = compute_pit_beta(returns_wide, spy_returns, window=504, min_periods=126)
    # beta_df: (date x ticker) — already PIT-safe (shift applied)
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def compute_pit_beta(
    returns: pd.DataFrame,
    spy_returns: pd.Series,
    window: int = 504,          # 2y default per HC #552
    min_periods: int | None = None,
    eps: float = 1e-9,
) -> pd.DataFrame:
    """
    Rolling-window beta of each column in `returns` against `spy_returns`.

    Parameters
    ----------
    returns : DataFrame (date index, ticker columns) of daily returns.
    spy_returns : Series (date index) of daily SPY returns. Indexes must align.
    window : rolling window in trading days. 504 ~= 2y.
    min_periods : minimum non-NaN observations required (default = window // 4).
    eps : small constant to keep the denominator positive.

    Returns
    -------
    beta : DataFrame same shape as `returns`. Already PIT-shifted by 1 day
           (so beta_df.loc[D, T] is computable using only returns through D-1).
    """
    if min_periods is None:
        min_periods = max(21, window // 4)

    # Align indexes
    idx = returns.index.intersection(spy_returns.index)
    r = returns.loc[idx]
    spy = spy_returns.loc[idx]

    var_spy = spy.rolling(window, min_periods=min_periods).var()

    # cov(asset, spy) for each ticker; rolling().cov(Series) is supported
    cov_df = r.rolling(window, min_periods=min_periods).cov(spy)

    # Broadcast division
    beta = cov_df.div(var_spy + eps, axis=0)

    # PIT shift — beta_df.loc[D] computed from returns through D, so shift to D+1
    beta = beta.shift(1)

    return beta


def beta_bucket(beta_series: pd.Series, low: float = 0.8, high: float = 1.2) -> pd.Series:
    """Bucket beta into {'low', 'mid', 'high'} for sleeve assignment.

    Defaults (per HC #552 R1): high-β sleeve = β ≥ 1.2.
    """
    out = pd.Series("mid", index=beta_series.index, dtype=object)
    out[beta_series < low] = "low"
    out[beta_series >= high] = "high"
    out[beta_series.isna()] = "unknown"
    return out


def high_beta_universe(
    beta_panel: pd.DataFrame,
    threshold: float = 1.2,
    on_date: pd.Timestamp | None = None,
) -> list[str]:
    """Return list of tickers with β ≥ threshold as of `on_date` (PIT).

    If `on_date` is None, uses the last row.
    """
    if on_date is None:
        row = beta_panel.iloc[-1]
    else:
        # asof — use last available beta on/before on_date
        row = beta_panel.loc[:on_date].iloc[-1]
    return [t for t, b in row.items() if pd.notna(b) and b >= threshold]


if __name__ == "__main__":
    # Smoke test on the wheel's prices.parquet
    from pathlib import Path
    ROOT = Path(__file__).resolve().parents[1]
    prices = pd.read_parquet(ROOT / "data" / "cache" / "prices.parquet")
    # Expect long format: date, ticker, close (or adj_close)
    print("prices cols:", list(prices.columns)[:10], "shape:", prices.shape)

    # Pivot to wide
    close_col = "adj_close" if "adj_close" in prices.columns else "close"
    px_wide = prices.pivot(index="date", columns="ticker", values=close_col).sort_index()
    rets = px_wide.pct_change()
    if "SPY" not in rets.columns:
        raise SystemExit("SPY not in prices.parquet — required for beta computation")
    spy = rets["SPY"]

    beta = compute_pit_beta(rets, spy, window=504, min_periods=126)
    print("beta shape:", beta.shape, "non-nan%:", beta.notna().mean().mean())
    latest = beta.iloc[-1].dropna().sort_values(ascending=False)
    print("top-10 β:", latest.head(10).to_dict())
    print("low-10 β:", latest.tail(10).to_dict())
    print("high-β universe (β≥1.2):", len(high_beta_universe(beta)))
