#!/usr/bin/env python3
"""
backtest_template.py — Reusable Backtest Quality Gates
======================================================

Every new income research backtest MUST run these quality gates before
reporting results. This eliminates the recurring pattern of discovering
bugs post-hoc (zero-cost closes, outlier-driven returns, regime bias, etc.).

Usage:
    from backtest_template import run_all_quality_gates, load_spy_prices, save_results

    spy = load_spy_prices('2019-01-01', '2026-07-15')
    gates = run_all_quality_gates(trades_df, spy)
    save_results('/path/to/output/', gates)

Expected trades_df columns (minimum):
    - 'date' or 'entry_date': datetime-like, when trade was opened
    - 'pnl' or 'trade_pnl' or 'pnl_per_contract': float, realized P&L
    - 'ticker': str, instrument symbol (optional but needed for concentration test)
    - 'close_price' or 'close_cost': float, price at which position was closed
    - 'dte_at_close': int, days to expiry remaining at close (optional)

If your DataFrame uses different column names, either rename before calling
or pass the column mapping via the `col_map` parameter.

HC #705 — Adversarial backtest validation foundation.
HC #428 R1 — Regime-agnostic validation (regime gap < 0.50).
HC #694 — Commission-free for RH options.

Cost Constants (canonical):
    ES_TICK_VALUE       = $12.50
    ES_RT_COMMISSION    = $4.70  (AMP/Rithmic round-trip)
    OPTION_COMMISSION   = $0.00  (Robinhood — HC #694)
    OPTION_SLIPPAGE     = $0.02  per contract (conservative bid/ask crossing estimate)
"""
from __future__ import annotations

import json
import os
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════════════════════
# COST CONSTANTS (canonical values — DO NOT change without HC update)
# ═══════════════════════════════════════════════════════════════════════════════

ES_TICK_VALUE = 12.50
ES_RT_COMMISSION = 4.70
ES_RT_COMMISSION_TICKS = ES_RT_COMMISSION / ES_TICK_VALUE  # 0.376

OPTION_COMMISSION_PER_CONTRACT = 0.00   # Robinhood (HC #694)
OPTION_SLIPPAGE_PER_CONTRACT = 0.02     # conservative bid/ask estimate
OPTION_COST_PER_CONTRACT = OPTION_COMMISSION_PER_CONTRACT + OPTION_SLIPPAGE_PER_CONTRACT

# ═══════════════════════════════════════════════════════════════════════════════
# COLUMN RESOLUTION HELPER
# ═══════════════════════════════════════════════════════════════════════════════

# Default column name candidates — first match wins
_COL_CANDIDATES = {
    "pnl": ["pnl", "trade_pnl", "pnl_per_contract", "return", "profit"],
    "date": ["date", "entry_date", "trade_date", "open_date"],
    "ticker": ["ticker", "symbol", "underlying"],
    "close_price": ["close_price", "close_cost", "exit_price"],
    "dte_at_close": ["dte_at_close", "dte_close", "close_dte"],
}


def _resolve_col(df: pd.DataFrame, key: str, col_map: Optional[dict] = None) -> Optional[str]:
    """Find the actual column name in df for a logical key."""
    if col_map and key in col_map:
        candidate = col_map[key]
        if candidate in df.columns:
            return candidate
    for c in _COL_CANDIDATES.get(key, []):
        if c in df.columns:
            return c
    return None


def _get_pnl(df: pd.DataFrame, col_map: Optional[dict] = None) -> pd.Series:
    """Extract P&L series from df, raising if not found."""
    col = _resolve_col(df, "pnl", col_map)
    if col is None:
        raise ValueError(
            f"Cannot find P&L column. Available: {list(df.columns)}. "
            f"Pass col_map={{'pnl': 'your_column_name'}}"
        )
    return df[col].astype(float)


def _get_dates(df: pd.DataFrame, col_map: Optional[dict] = None) -> pd.Series:
    """Extract date series from df."""
    col = _resolve_col(df, "date", col_map)
    if col is None:
        raise ValueError(
            f"Cannot find date column. Available: {list(df.columns)}. "
            f"Pass col_map={{'date': 'your_column_name'}}"
        )
    return pd.to_datetime(df[col])


# ═══════════════════════════════════════════════════════════════════════════════
# SPY DATA LOADER (with local cache)
# ═══════════════════════════════════════════════════════════════════════════════

_SPY_CACHE_DIR = Path("/home/jupiter/Lvl3Quant/data/cache")


def load_spy_prices(
    start: str = "2015-01-01",
    end: str = "2026-07-15",
    cache_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """Download or load cached SPY daily prices.

    Returns DataFrame with columns: Date (index), Open, High, Low, Close, Volume.
    Also adds 'daily_return' (close-to-close pct change).

    Args:
        start: Start date string (YYYY-MM-DD).
        end: End date string (YYYY-MM-DD).
        cache_dir: Where to cache the parquet file. Defaults to Lvl3Quant/data/cache/.

    Returns:
        pd.DataFrame with SPY daily OHLCV + daily_return.
    """
    cache_dir = cache_dir or _SPY_CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"spy_daily_{start}_{end}.parquet"

    if cache_path.exists():
        spy = pd.read_parquet(cache_path)
        if len(spy) > 0:
            return spy

    try:
        import yfinance as yf
    except ImportError:
        raise ImportError("yfinance is required: pip install yfinance")

    print(f"  Downloading SPY {start} to {end}...")
    spy = yf.download("SPY", start=start, end=end, progress=False)
    if isinstance(spy.columns, pd.MultiIndex):
        spy.columns = spy.columns.droplevel(1)
    spy.index.name = "Date"
    spy["daily_return"] = spy["Close"].pct_change()

    spy.to_parquet(cache_path)
    print(f"  Cached SPY data: {len(spy)} rows -> {cache_path}")
    return spy


# ═══════════════════════════════════════════════════════════════════════════════
# GATE 1: PRICING VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════

def validate_pricing(
    trades_df: pd.DataFrame,
    col_map: Optional[dict] = None,
) -> Dict[str, Any]:
    """Check for zero-cost closes and intrinsic-only pricing.

    Flags:
        - Any option close = $0 when DTE > 0 (should have time value)
        - >10% of closes are $0 or intrinsic-only
        - Mean close price suspiciously low

    Args:
        trades_df: DataFrame with trade records.
        col_map: Optional column name mapping.

    Returns:
        dict with 'passed', 'zero_close_count', 'zero_close_pct', 'warnings'.
    """
    result = {
        "gate": "pricing_validation",
        "passed": True,
        "zero_close_count": 0,
        "zero_close_pct": 0.0,
        "zero_with_dte_count": 0,
        "warnings": [],
    }

    close_col = _resolve_col(trades_df, "close_price", col_map)
    if close_col is None:
        result["warnings"].append("No close_price column found — skipping pricing validation")
        result["passed"] = None  # indeterminate
        return result

    closes = trades_df[close_col].astype(float)
    n_total = len(closes)
    if n_total == 0:
        result["warnings"].append("No trades to validate")
        result["passed"] = None
        return result

    # Zero-cost closes
    n_zero = (closes == 0).sum()
    zero_pct = n_zero / n_total * 100
    result["zero_close_count"] = int(n_zero)
    result["zero_close_pct"] = round(zero_pct, 2)

    if n_zero > 0:
        result["warnings"].append(
            f"{n_zero} closes at $0.00 ({zero_pct:.1f}% of {n_total} trades)"
        )

    # Zero closes with DTE > 0 (should have time value)
    dte_col = _resolve_col(trades_df, "dte_at_close", col_map)
    if dte_col is not None:
        mask = (closes == 0) & (trades_df[dte_col].astype(float) > 0)
        n_zero_dte = mask.sum()
        result["zero_with_dte_count"] = int(n_zero_dte)
        if n_zero_dte > 0:
            result["warnings"].append(
                f"WARNING: {n_zero_dte} closes at $0 with DTE > 0 — likely missing price data"
            )

    # Threshold: >10% zero closes = FAIL
    if zero_pct > 10:
        result["passed"] = False
        result["warnings"].append(
            f"FAIL: {zero_pct:.1f}% zero-cost closes exceeds 10% threshold"
        )

    # Suspiciously low mean close
    mean_close = closes.mean()
    if mean_close < 0.01 and n_total > 10:
        result["warnings"].append(
            f"WARNING: Mean close price = ${mean_close:.4f} — suspiciously low"
        )

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# GATE 2: CLOSE DIVERGENCE (real vs synthetic)
# ═══════════════════════════════════════════════════════════════════════════════

def validate_close_divergence(
    trades_df: pd.DataFrame,
    col_map: Optional[dict] = None,
) -> Dict[str, Any]:
    """Compare real vs synthetic close pricing if both columns exist.

    If trades have both 'real_close' and 'synthetic_close' columns, compute
    WR and PF for each pricing method. Flag if the gap exceeds 20%.

    Args:
        trades_df: DataFrame with trade records.
        col_map: Optional column name mapping.

    Returns:
        dict with 'passed', 'real_wr', 'synth_wr', 'gap_pct', 'warnings'.
    """
    result = {
        "gate": "close_divergence",
        "passed": True,
        "real_wr": None,
        "synth_wr": None,
        "real_pf": None,
        "synth_pf": None,
        "wr_gap_pct": None,
        "pf_gap_pct": None,
        "warnings": [],
    }

    # Try to find real/synthetic close columns
    real_col = None
    synth_col = None
    for c in trades_df.columns:
        cl = c.lower()
        if "real" in cl and "close" in cl:
            real_col = c
        elif "synth" in cl and "close" in cl:
            synth_col = c

    if real_col is None or synth_col is None:
        result["warnings"].append("No real_close/synthetic_close columns — skipping divergence test")
        result["passed"] = None  # indeterminate
        return result

    pnl_col = _resolve_col(trades_df, "pnl", col_map)
    if pnl_col is None:
        result["warnings"].append("No P&L column — cannot compute divergence metrics")
        result["passed"] = None
        return result

    # Compute P&L under each pricing
    pnl = trades_df[pnl_col].astype(float)
    real_prices = trades_df[real_col].astype(float)
    synth_prices = trades_df[synth_col].astype(float)

    # Approximate: adjust P&L by the difference in close pricing
    price_diff = real_prices - synth_prices
    pnl_real = pnl  # assume base P&L uses real prices
    pnl_synth = pnl - price_diff * 100  # adjust for 100-share contracts

    def _wr(s):
        return (s > 0).mean() * 100 if len(s) > 0 else 0

    def _pf(s):
        wins = s[s > 0].sum()
        losses = abs(s[s < 0].sum())
        return wins / losses if losses > 0 else float("inf")

    result["real_wr"] = round(_wr(pnl_real), 2)
    result["synth_wr"] = round(_wr(pnl_synth), 2)
    result["real_pf"] = round(_pf(pnl_real), 3)
    result["synth_pf"] = round(_pf(pnl_synth), 3)

    # WR gap
    max_wr = max(result["real_wr"], result["synth_wr"])
    wr_gap = abs(result["real_wr"] - result["synth_wr"])
    wr_gap_pct = (wr_gap / max_wr * 100) if max_wr > 0 else 0
    result["wr_gap_pct"] = round(wr_gap_pct, 2)

    # PF gap
    real_pf = min(result["real_pf"], 100)  # cap inf
    synth_pf = min(result["synth_pf"], 100)
    max_pf = max(real_pf, synth_pf)
    pf_gap = abs(real_pf - synth_pf)
    pf_gap_pct = (pf_gap / max_pf * 100) if max_pf > 0 else 0
    result["pf_gap_pct"] = round(pf_gap_pct, 2)

    if wr_gap_pct > 20:
        result["passed"] = False
        result["warnings"].append(
            f"FAIL: WR gap {wr_gap_pct:.1f}% between real ({result['real_wr']:.1f}%) "
            f"and synthetic ({result['synth_wr']:.1f}%) exceeds 20% threshold"
        )
    if pf_gap_pct > 20:
        result["passed"] = False
        result["warnings"].append(
            f"FAIL: PF gap {pf_gap_pct:.1f}% between real ({result['real_pf']:.2f}) "
            f"and synthetic ({result['synth_pf']:.2f}) exceeds 20% threshold"
        )

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# GATE 3: PERMUTATION TEST
# ═══════════════════════════════════════════════════════════════════════════════

def run_permutation_test(
    returns: np.ndarray | pd.Series,
    n_perms: int = 200,
    seed: int = 42,
) -> Dict[str, Any]:
    """Shuffle returns and compare real mean to null distribution.

    Tests whether the strategy's mean return is significantly different from
    what you'd get by randomly shuffling trade order/assignment.

    Args:
        returns: Array of per-trade returns (or P&L values).
        n_perms: Number of permutations (200 default, use 1000 for publication).
        seed: Random seed for reproducibility.

    Returns:
        dict with 'passed' (True if p < 0.05), 'p_value', 'real_mean',
        'null_mean', 'null_std'.
    """
    returns = np.asarray(returns, dtype=float)
    returns = returns[~np.isnan(returns)]

    result = {
        "gate": "permutation_test",
        "passed": False,
        "p_value": 1.0,
        "real_mean": float(np.mean(returns)),
        "null_mean": 0.0,
        "null_std": 0.0,
        "n_perms": n_perms,
        "warnings": [],
    }

    if len(returns) < 10:
        result["warnings"].append("Too few trades (<10) for meaningful permutation test")
        result["passed"] = None
        return result

    rng = np.random.default_rng(seed)
    real_mean = np.mean(returns)
    null_means = np.empty(n_perms)

    for i in range(n_perms):
        shuffled = rng.permutation(returns)
        null_means[i] = np.mean(shuffled)

    # Note: shuffling preserves the same set of values, so null_mean ~ real_mean.
    # The real test is whether the SEQUENCE matters (for time-dependent strategies).
    # For simple P&L shuffle, we instead test if real_mean > 0 is significant
    # by comparing to a bootstrap of the mean under random sign flips.
    sign_flips = np.empty(n_perms)
    for i in range(n_perms):
        signs = rng.choice([-1, 1], size=len(returns))
        sign_flips[i] = np.mean(returns * signs)

    p_value = np.mean(sign_flips >= real_mean)
    result["p_value"] = round(float(p_value), 4)
    result["null_mean"] = round(float(np.mean(sign_flips)), 6)
    result["null_std"] = round(float(np.std(sign_flips)), 6)
    result["passed"] = bool(p_value < 0.05)

    if not result["passed"]:
        result["warnings"].append(
            f"FAIL: p-value={p_value:.4f} >= 0.05 — returns not significantly different from random"
        )

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# GATE 4: REGIME TEST (HC #428 R1)
# ═══════════════════════════════════════════════════════════════════════════════

def _classify_regime(spy_return: float) -> str:
    """Classify a day as green/red/flat based on SPY close-to-close return.

    Thresholds: green > +0.25%, red < -0.25%, else flat.
    """
    if spy_return > 0.0025:   # >0.25%
        return "green"
    elif spy_return < -0.0025:  # <-0.25%
        return "red"
    else:
        return "flat"


def _sharpe(returns: np.ndarray) -> float:
    """Annualized Sharpe ratio from per-trade returns."""
    if len(returns) < 2:
        return 0.0
    mean = np.mean(returns)
    std = np.std(returns, ddof=1)
    if std == 0:
        return 0.0 if mean == 0 else np.sign(mean) * 99.0
    # Annualize assuming ~252 trading days
    return float(mean / std * np.sqrt(min(252, len(returns))))


def run_regime_test(
    trades_df: pd.DataFrame,
    spy_prices: pd.DataFrame,
    col_map: Optional[dict] = None,
    max_gap: float = 0.50,
) -> Dict[str, Any]:
    """Classify trades by PRIOR-DAY SPY regime, compute per-regime Sharpe.

    CRITICAL: Uses the PRIOR trading day's SPY close-to-close return, NOT
    same-day. Using same-day would be look-ahead bias / leakage because
    you wouldn't know the day's return at trade entry time.

    REJECT if |max_sharpe - min_sharpe| / max(|max_sharpe|, |min_sharpe|) > max_gap.
    This catches strategies that only work in one regime (e.g., all-short in red markets).

    Per HC #428 R1: regime-agnostic OOT validation is mandatory.

    Args:
        trades_df: DataFrame with trade records.
        spy_prices: SPY daily DataFrame (from load_spy_prices).
        col_map: Optional column name mapping.
        max_gap: Maximum allowed normalized Sharpe gap (default 0.50).

    Returns:
        dict with 'passed', per-regime stats, 'regime_gap'.

    Example:
        >>> spy = load_spy_prices()
        >>> result = run_regime_test(trades, spy)
        >>> print(result['regimes'])  # green/red/flat Sharpe breakdown
    """
    result = {
        "gate": "regime_test",
        "passed": True,
        "regimes": {},
        "regime_gap": None,
        "warnings": [],
    }

    pnl = _get_pnl(trades_df, col_map)
    dates = _get_dates(trades_df, col_map)

    # Ensure SPY has daily_return
    spy = spy_prices.copy()
    if "daily_return" not in spy.columns:
        spy["daily_return"] = spy["Close"].pct_change()

    # PRIOR-DAY return: shift(1) so today's regime = yesterday's SPY return.
    # This avoids look-ahead bias -- at trade entry time, you only know
    # yesterday's close, not today's.
    spy["prior_day_return"] = spy["daily_return"].shift(1)

    # Build prior-day regime map
    spy_ret_map = spy["prior_day_return"].to_dict()
    # Normalize index to date
    if hasattr(spy.index, "date"):
        spy_ret_map = {d.date() if hasattr(d, "date") else d: v for d, v in spy_ret_map.items()}

    regimes = []
    for d in dates:
        d_date = d.date() if hasattr(d, "date") else d
        ret = spy_ret_map.get(d_date, spy_ret_map.get(d, None))
        if ret is not None and not np.isnan(ret):
            regimes.append(_classify_regime(ret))
        else:
            regimes.append("unknown")

    trades_df = trades_df.copy()
    trades_df["_regime"] = regimes
    trades_df["_pnl"] = pnl.values

    sharpes = {}
    for regime in ["green", "red", "flat"]:
        sub = trades_df[trades_df["_regime"] == regime]
        n = len(sub)
        if n < 5:
            result["warnings"].append(f"Only {n} trades in {regime} regime — limited confidence")
        s = _sharpe(sub["_pnl"].values) if n >= 2 else 0.0
        wr = (sub["_pnl"] > 0).mean() * 100 if n > 0 else 0.0
        mean_pnl = sub["_pnl"].mean() if n > 0 else 0.0
        sharpes[regime] = s
        result["regimes"][regime] = {
            "n_trades": int(n),
            "sharpe": round(s, 3),
            "win_rate": round(wr, 1),
            "mean_pnl": round(float(mean_pnl), 2),
        }

    # Compute regime gap (HC #428 R1)
    valid_sharpes = [v for k, v in sharpes.items() if result["regimes"][k]["n_trades"] >= 5]
    if len(valid_sharpes) >= 2:
        max_s = max(abs(s) for s in valid_sharpes)
        gap = max(valid_sharpes) - min(valid_sharpes)
        regime_gap = gap / max_s if max_s > 0 else 0.0
        result["regime_gap"] = round(regime_gap, 3)

        if regime_gap > max_gap:
            result["passed"] = False
            result["warnings"].append(
                f"FAIL: Regime gap {regime_gap:.3f} > {max_gap} threshold. "
                f"Sharpes: green={sharpes.get('green', 0):.2f}, "
                f"red={sharpes.get('red', 0):.2f}, flat={sharpes.get('flat', 0):.2f}"
            )
    else:
        result["warnings"].append("Insufficient regime data — need >=5 trades in at least 2 regimes")
        result["passed"] = None

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# GATE 5: SUB-PERIOD TEST
# ═══════════════════════════════════════════════════════════════════════════════

def run_sub_period_test(
    trades_df: pd.DataFrame,
    n_periods: int = 3,
    col_map: Optional[dict] = None,
) -> Dict[str, Any]:
    """Split trades into N chronological non-overlapping periods.

    ALL periods must show positive average return. If any period is negative,
    the edge may be concentrated in one era and unreliable going forward.

    Args:
        trades_df: DataFrame with trade records (must be sorted by date).
        n_periods: Number of sub-periods to test (default 3).
        col_map: Optional column name mapping.

    Returns:
        dict with 'passed', per-period stats.
    """
    result = {
        "gate": "sub_period_test",
        "passed": True,
        "n_periods": n_periods,
        "periods": [],
        "warnings": [],
    }

    pnl = _get_pnl(trades_df, col_map)
    dates = _get_dates(trades_df, col_map)

    # Sort by date
    sort_idx = dates.argsort()
    pnl_sorted = pnl.iloc[sort_idx].values
    dates_sorted = dates.iloc[sort_idx].values

    n = len(pnl_sorted)
    if n < n_periods * 5:
        result["warnings"].append(
            f"Only {n} trades — need at least {n_periods * 5} for meaningful sub-period test"
        )
        result["passed"] = None
        return result

    chunk_size = n // n_periods
    negative_periods = 0

    for i in range(n_periods):
        start_idx = i * chunk_size
        end_idx = (i + 1) * chunk_size if i < n_periods - 1 else n
        chunk_pnl = pnl_sorted[start_idx:end_idx]
        chunk_dates = dates_sorted[start_idx:end_idx]

        mean_ret = float(np.mean(chunk_pnl))
        wr = float((chunk_pnl > 0).mean() * 100)
        sharpe = _sharpe(chunk_pnl)

        period_info = {
            "period": i + 1,
            "n_trades": int(len(chunk_pnl)),
            "date_range": f"{pd.Timestamp(chunk_dates[0]).strftime('%Y-%m-%d')} to "
                          f"{pd.Timestamp(chunk_dates[-1]).strftime('%Y-%m-%d')}",
            "mean_pnl": round(mean_ret, 2),
            "win_rate": round(wr, 1),
            "sharpe": round(sharpe, 3),
        }
        result["periods"].append(period_info)

        if mean_ret <= 0:
            negative_periods += 1

    if negative_periods > 0:
        result["passed"] = False
        result["warnings"].append(
            f"FAIL: {negative_periods}/{n_periods} sub-periods have negative mean return"
        )

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# GATE 6: OUTLIER TEST
# ═══════════════════════════════════════════════════════════════════════════════

def run_outlier_test(
    trades_df: pd.DataFrame,
    n_remove: int = 5,
    col_map: Optional[dict] = None,
) -> Dict[str, Any]:
    """Remove top N trades by P&L, check if Sharpe drops >50%.

    If removing a handful of outlier wins collapses the strategy, it's not
    a robust edge — it's a few lucky trades masking noise.

    Args:
        trades_df: DataFrame with trade records.
        n_remove: Number of top trades to remove (default 5).
        col_map: Optional column name mapping.

    Returns:
        dict with 'passed', 'full_sharpe', 'trimmed_sharpe', 'sharpe_drop_pct'.
    """
    result = {
        "gate": "outlier_test",
        "passed": True,
        "full_sharpe": 0.0,
        "trimmed_sharpe": 0.0,
        "sharpe_drop_pct": 0.0,
        "n_removed": n_remove,
        "warnings": [],
    }

    pnl = _get_pnl(trades_df, col_map).values
    n = len(pnl)

    if n < n_remove + 10:
        result["warnings"].append(f"Only {n} trades — too few for outlier removal test")
        result["passed"] = None
        return result

    full_sharpe = _sharpe(pnl)
    result["full_sharpe"] = round(full_sharpe, 3)

    # Remove top N by absolute P&L (best trades)
    sorted_idx = np.argsort(pnl)[::-1]  # descending
    keep_idx = sorted_idx[n_remove:]
    trimmed_pnl = pnl[keep_idx]

    trimmed_sharpe = _sharpe(trimmed_pnl)
    result["trimmed_sharpe"] = round(trimmed_sharpe, 3)

    if abs(full_sharpe) > 0.01:
        drop_pct = (full_sharpe - trimmed_sharpe) / abs(full_sharpe) * 100
        result["sharpe_drop_pct"] = round(drop_pct, 1)

        if drop_pct > 50:
            result["passed"] = False
            result["warnings"].append(
                f"FAIL: Removing top {n_remove} trades drops Sharpe by {drop_pct:.1f}% "
                f"({full_sharpe:.2f} -> {trimmed_sharpe:.2f}) — outlier-driven"
            )
    else:
        result["warnings"].append("Full Sharpe near zero — outlier test not meaningful")
        result["passed"] = None

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# GATE 7: TICKER CONCENTRATION
# ═══════════════════════════════════════════════════════════════════════════════

def run_ticker_concentration(
    trades_df: pd.DataFrame,
    max_concentration: float = 0.40,
    col_map: Optional[dict] = None,
) -> Dict[str, Any]:
    """Check if any single ticker accounts for >40% of total P&L.

    A strategy shouldn't depend on one name. If NVDA drives 60% of returns,
    you have an NVDA strategy, not a generalizable edge.

    Args:
        trades_df: DataFrame with trade records.
        max_concentration: Maximum allowed fraction from one ticker (default 0.40).
        col_map: Optional column name mapping.

    Returns:
        dict with 'passed', 'top_ticker', 'top_concentration', ticker breakdown.
    """
    result = {
        "gate": "ticker_concentration",
        "passed": True,
        "top_ticker": None,
        "top_concentration": 0.0,
        "ticker_pnl": {},
        "warnings": [],
    }

    ticker_col = _resolve_col(trades_df, "ticker", col_map)
    if ticker_col is None:
        result["warnings"].append("No ticker column — skipping concentration test")
        result["passed"] = None
        return result

    pnl = _get_pnl(trades_df, col_map)
    total_abs_pnl = abs(pnl).sum()
    if total_abs_pnl == 0:
        result["warnings"].append("Total absolute P&L is zero — cannot assess concentration")
        result["passed"] = None
        return result

    # Group by ticker
    grouped = pd.DataFrame({"pnl": pnl.values, "ticker": trades_df[ticker_col].values})
    ticker_pnl = grouped.groupby("ticker")["pnl"].sum()
    ticker_abs = ticker_pnl.abs()
    ticker_frac = ticker_abs / total_abs_pnl

    top_ticker = ticker_frac.idxmax()
    top_conc = float(ticker_frac.max())

    result["top_ticker"] = str(top_ticker)
    result["top_concentration"] = round(top_conc, 3)
    result["ticker_pnl"] = {
        str(k): round(float(v), 2) for k, v in ticker_pnl.nlargest(10).items()
    }

    if top_conc > max_concentration:
        result["passed"] = False
        result["warnings"].append(
            f"FAIL: {top_ticker} accounts for {top_conc*100:.1f}% of total P&L "
            f"(threshold: {max_concentration*100:.0f}%)"
        )

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# SUSPICIOUS PATTERN SCANNER
# ═══════════════════════════════════════════════════════════════════════════════

def print_suspicious_patterns(
    trades_df: pd.DataFrame,
    col_map: Optional[dict] = None,
) -> Dict[str, Any]:
    """Scan for known backtest failure modes.

    Checks:
        1. All closes at $0
        2. All trades profitable (unrealistic)
        3. Win rate > 90% (suspicious for options selling — real WR ~70-80%)
        4. Returns grow geometrically (compounding bug)
        5. Single day accounts for >20% of P&L

    Args:
        trades_df: DataFrame with trade records.
        col_map: Optional column name mapping.

    Returns:
        dict with 'n_flags', 'flags' list.
    """
    result = {
        "gate": "suspicious_patterns",
        "n_flags": 0,
        "flags": [],
        "warnings": [],
    }

    pnl = _get_pnl(trades_df, col_map)
    n = len(pnl)
    if n == 0:
        result["warnings"].append("No trades to scan")
        return result

    # 1. All closes at $0
    close_col = _resolve_col(trades_df, "close_price", col_map)
    if close_col is not None:
        closes = trades_df[close_col].astype(float)
        if (closes == 0).all() and n > 5:
            result["flags"].append("ALL closes at $0 — data is missing or expiration-only")
            result["n_flags"] += 1

    # 2. All trades profitable
    if (pnl > 0).all() and n > 10:
        result["flags"].append(
            f"ALL {n} trades are profitable — unrealistic, likely missing costs or using wrong pricing"
        )
        result["n_flags"] += 1

    # 3. Win rate > 90%
    wr = (pnl > 0).mean() * 100
    if wr > 90 and n > 20:
        result["flags"].append(
            f"Win rate = {wr:.1f}% on {n} trades — suspiciously high even for options selling"
        )
        result["n_flags"] += 1

    # 4. Geometric growth pattern (compounding bug)
    if n >= 20:
        cumulative = pnl.cumsum().values
        # Check if later trades are systematically larger
        first_half_mean = abs(pnl.iloc[:n//2]).mean()
        second_half_mean = abs(pnl.iloc[n//2:]).mean()
        if first_half_mean > 0 and second_half_mean / first_half_mean > 3.0:
            result["flags"].append(
                f"Trade sizes grow {second_half_mean/first_half_mean:.1f}x from first to second half "
                f"— possible compounding bug (position sizing on equity curve?)"
            )
            result["n_flags"] += 1

    # 5. Single day > 20% of P&L
    try:
        dates = _get_dates(trades_df, col_map)
        date_groups = pd.DataFrame({"pnl": pnl.values, "date": dates.dt.date.values})
        daily_pnl = date_groups.groupby("date")["pnl"].sum()
        total_pnl = abs(pnl).sum()
        if total_pnl > 0:
            max_day_pnl = daily_pnl.abs().max()
            max_day = daily_pnl.abs().idxmax()
            day_frac = max_day_pnl / total_pnl
            if day_frac > 0.20:
                result["flags"].append(
                    f"Single day ({max_day}) accounts for {day_frac*100:.1f}% of total P&L"
                )
                result["n_flags"] += 1
    except (ValueError, KeyError):
        pass  # skip if date parsing fails

    return result


# Alias for HC #705 compatibility
check_suspicious_patterns = print_suspicious_patterns


# ═══════════════════════════════════════════════════════════════════════════════
# MASTER GATE RUNNER
# ═══════════════════════════════════════════════════════════════════════════════

def run_all_quality_gates(
    trades_df: pd.DataFrame,
    spy_prices: pd.DataFrame,
    col_map: Optional[dict] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Run ALL quality gates and print a summary table.

    This is the single entry point for backtest validation. Every new strategy
    script should call this after computing trades.

    Args:
        trades_df: DataFrame with trade records.
        spy_prices: SPY daily DataFrame (from load_spy_prices).
        col_map: Optional column name mapping override.
        verbose: Print summary table to stdout (default True).

    Returns:
        dict with gate name -> result dict, plus overall 'all_passed' bool.
    """
    results = {}

    # Run all gates
    results["pricing"] = validate_pricing(trades_df, col_map)
    results["close_divergence"] = validate_close_divergence(trades_df, col_map)

    pnl = _get_pnl(trades_df, col_map)
    results["permutation"] = run_permutation_test(pnl)
    results["regime"] = run_regime_test(trades_df, spy_prices, col_map)
    results["sub_period"] = run_sub_period_test(trades_df, col_map=col_map)
    results["outlier"] = run_outlier_test(trades_df, col_map=col_map)
    results["ticker_concentration"] = run_ticker_concentration(trades_df, col_map=col_map)
    results["suspicious_patterns"] = print_suspicious_patterns(trades_df, col_map)

    # Overall pass/fail (None = indeterminate, not counted as fail)
    hard_fails = []
    soft_warns = []
    for name, r in results.items():
        passed = r.get("passed", None)
        if passed == False and passed is not None:
            hard_fails.append(name)
        if r.get("warnings"):
            soft_warns.extend(r["warnings"])
        if r.get("flags"):
            soft_warns.extend(r["flags"])

    n_suspicious = results["suspicious_patterns"].get("n_flags", 0)
    all_passed = len(hard_fails) == 0 and n_suspicious == 0

    results["_summary"] = {
        "all_passed": all_passed,
        "hard_fails": hard_fails,
        "n_suspicious_flags": n_suspicious,
        "total_trades": len(trades_df),
        "overall_sharpe": round(_sharpe(pnl.values), 3),
        "overall_wr": round((pnl > 0).mean() * 100, 1),
        "overall_pf": round(
            pnl[pnl > 0].sum() / abs(pnl[pnl < 0].sum())
            if abs(pnl[pnl < 0].sum()) > 0 else float("inf"),
            3,
        ),
    }

    if verbose:
        _print_summary(results)

    return results


def _print_summary(results: Dict[str, Any]) -> None:
    """Print a formatted summary table of all quality gates."""
    summary = results["_summary"]

    print("\n" + "=" * 72)
    print("  BACKTEST QUALITY GATE RESULTS")
    print("=" * 72)
    print(f"  Total trades: {summary['total_trades']}")
    print(f"  Overall Sharpe: {summary['overall_sharpe']:.3f}  |  "
          f"WR: {summary['overall_wr']:.1f}%  |  PF: {summary['overall_pf']:.2f}")
    print("-" * 72)

    gate_order = [
        ("pricing", "Pricing Validation"),
        ("close_divergence", "Close Divergence"),
        ("permutation", "Permutation Test"),
        ("regime", "Regime Test (HC#428)"),
        ("sub_period", "Sub-Period Stability"),
        ("outlier", "Outlier Robustness"),
        ("ticker_concentration", "Ticker Concentration"),
        ("suspicious_patterns", "Suspicious Patterns"),
    ]

    for key, label in gate_order:
        r = results.get(key, {})
        passed = r.get("passed", None)
        n_flags = r.get("n_flags", None)

        if key == "suspicious_patterns":
            if n_flags == 0:
                status = "PASS"
                icon = "[OK]"
            else:
                status = f"{n_flags} FLAGS"
                icon = "[!!]"
        elif passed is None:
            status = "SKIP"
            icon = "[--]"
        elif passed:
            status = "PASS"
            icon = "[OK]"
        else:
            status = "FAIL"
            icon = "[XX]"

        # Add key detail
        detail = ""
        if key == "permutation" and r.get("p_value") is not None:
            detail = f"  p={r['p_value']:.4f}"
        elif key == "regime" and r.get("regime_gap") is not None:
            detail = f"  gap={r['regime_gap']:.3f}"
        elif key == "outlier" and r.get("sharpe_drop_pct") is not None:
            detail = f"  drop={r['sharpe_drop_pct']:.1f}%"
        elif key == "ticker_concentration" and r.get("top_ticker"):
            detail = f"  top={r['top_ticker']} ({r['top_concentration']*100:.0f}%)"
        elif key == "pricing":
            detail = f"  {r.get('zero_close_pct', 0):.1f}% zero"

        print(f"  {icon} {label:<28s} {status:<8s}{detail}")

    print("-" * 72)

    if summary["all_passed"]:
        print("  VERDICT: ALL GATES PASSED")
    else:
        fails = summary["hard_fails"]
        flags = summary["n_suspicious_flags"]
        parts = []
        if fails:
            parts.append(f"{len(fails)} FAILED ({', '.join(fails)})")
        if flags > 0:
            parts.append(f"{flags} suspicious flags")
        print(f"  VERDICT: {' + '.join(parts)}")

    # Print all warnings
    all_warns = []
    for key, _ in gate_order:
        r = results.get(key, {})
        all_warns.extend(r.get("warnings", []))
        all_warns.extend(r.get("flags", []))

    if all_warns:
        print("\n  DETAILS:")
        for w in all_warns:
            print(f"    - {w}")

    print("=" * 72 + "\n")


# ═══════════════════════════════════════════════════════════════════════════════
# RESULTS SAVER
# ═══════════════════════════════════════════════════════════════════════════════

def save_results(
    output_dir: str | Path,
    results_dict: Dict[str, Any],
    trades_df: Optional[pd.DataFrame] = None,
    filename: str = "quality_gates.json",
) -> Path:
    """Save quality gate results to JSON and optionally trades to CSV.

    Args:
        output_dir: Directory to save to (created if needed).
        results_dict: The dict returned by run_all_quality_gates.
        trades_df: Optional trades DataFrame to save as CSV alongside results.
        filename: Output filename (default 'quality_gates.json').

    Returns:
        Path to the saved JSON file.

    Example:
        >>> save_results('/tmp/output', results, trades_df=trades)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / filename

    # Make JSON-serializable
    def _serialize(obj):
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (pd.Timestamp, datetime)):
            return obj.isoformat()
        if isinstance(obj, float) and (np.isinf(obj) or np.isnan(obj)):
            return str(obj)
        raise TypeError(f"Not serializable: {type(obj)}")

    with open(out_path, "w") as f:
        json.dump(results_dict, f, indent=2, default=_serialize)

    print(f"  Quality gate results saved to {out_path}")

    if trades_df is not None:
        trades_path = output_dir / "trades.csv"
        trades_df.to_csv(trades_path, index=False)
        print(f"  Trades saved to {trades_path} ({len(trades_df)} rows)")

    return out_path


# ═══════════════════════════════════════════════════════════════════════════════
# STANDALONE DEMO / SELF-TEST
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("backtest_template.py — self-test with synthetic data\n")

    # Generate synthetic trades for self-test
    np.random.seed(42)
    n_trades = 200
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "JPM"]

    dates = pd.date_range("2023-01-01", periods=n_trades, freq="B")
    synthetic = pd.DataFrame({
        "date": dates,
        "ticker": np.random.choice(tickers, n_trades),
        "pnl": np.random.normal(50, 200, n_trades),  # slight positive edge
        "close_price": np.random.uniform(0.5, 5.0, n_trades),
    })

    # Inject a few zero closes for testing
    synthetic.loc[synthetic.index[:5], "close_price"] = 0.0

    print("Loading SPY prices...")
    spy = load_spy_prices("2023-01-01", "2023-12-31")

    print("\nRunning all quality gates on synthetic data...")
    gates = run_all_quality_gates(synthetic, spy)

    # Save
    test_out = Path("/tmp/backtest_template_test")
    save_results(test_out, gates)

    print(f"Self-test complete. Passed: {gates['_summary']['all_passed']}")
