#!/usr/bin/env python3
"""
Relative Value Intra-Sector Backtest
=====================================
Trade individual stocks relative to their sector ETF.
When a quality stock underperforms its sector by a large margin (z-score < -2),
it tends to mean-revert back to sector-normal performance.

Walk-forward OOT: Jan 2022 – Jul 2026.
5-gate validation: Sharpe>0.5, perm p<0.05, regime gap<0.5, MaxDD>-50%, >=20 trades.
Regime: Bull = SPY > 200-SMA, Bear = SPY < 200-SMA.

6 Variants:
A) Single Best Signal: most extreme z < -2 stock, hold 10d or until z > 0
B) Multi-Position (up to 3): one per sector, $215 each
C) Quality Filter: mega-cap only (AAPL, MSFT, GOOGL, JPM, JNJ, PG)
D) Sector Rotation + Relative Value: RS rotation picks sector, then buy lowest z stock
E) Momentum Confirmation: z < -2 AND sector ETF has positive 20d momentum
F) Symmetric: also buy sector ETF when z > +2 (stock leads, sector catches up)
"""

import json
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

# ── Configuration ──────────────────────────────────────────────────────────────
ACCOUNT_SIZE = 645.0
SLIPPAGE = 0.0002  # 0.02% each way
OOT_START = "2022-01-01"
OOT_END = "2026-07-30"
DATA_START = "2020-06-01"  # need lookback for z-scores

SECTOR_MAP = {
    "Tech": {"etf": "XLK", "stocks": ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA"]},
    "Financials": {"etf": "XLF", "stocks": ["JPM", "GS", "BAC", "MS"]},
    "Healthcare": {"etf": "XLV", "stocks": ["JNJ", "UNH", "PFE", "ABBV"]},
    "Consumer": {"etf": "XLP", "stocks": ["PG", "KO", "PEP", "WMT"]},
}

QUALITY_STOCKS = {"AAPL", "MSFT", "GOOGL", "JPM", "JNJ", "PG"}

Z_ENTRY = -2.0
Z_EXIT = 0.0
REL_RET_WINDOW = 20
Z_LOOKBACK = 60
HOLD_MAX = 10

PERM_ITERATIONS = 1000

OUTPUT_PATH = Path("/home/jupiter/Lvl3Quant/data/relative_value_intra_sector_results.json")


# ── Data Download ──────────────────────────────────────────────────────────────
def download_data():
    all_tickers = set(["SPY"])
    for sec in SECTOR_MAP.values():
        all_tickers.add(sec["etf"])
        all_tickers.update(sec["stocks"])
    tickers = sorted(all_tickers)

    print(f"Downloading {len(tickers)} tickers...")
    data = yf.download(tickers, start=DATA_START, end=OOT_END, auto_adjust=True, progress=False)

    if isinstance(data.columns, pd.MultiIndex):
        close = data["Close"]
    else:
        close = data

    close = close.dropna(axis=1, thresh=int(len(close) * 0.8))
    close = close.ffill().bfill()

    print(f"Downloaded {len(close.columns)} tickers, {len(close)} trading days")
    print(f"Date range: {close.index[0].date()} to {close.index[-1].date()}")

    # Check for missing tickers
    missing = set(tickers) - set(close.columns)
    if missing:
        print(f"WARNING: Missing tickers: {missing}")

    return close


# ── Compute Relative Z-Scores ─────────────────────────────────────────────────
def compute_relative_zscores(close):
    """For each stock, compute z-score of 20-day relative return vs sector ETF."""
    zscores = {}
    for sector_name, sector_info in SECTOR_MAP.items():
        etf = sector_info["etf"]
        if etf not in close.columns:
            continue
        etf_ret_20d = close[etf].pct_change(REL_RET_WINDOW)
        for stock in sector_info["stocks"]:
            if stock not in close.columns:
                continue
            stock_ret_20d = close[stock].pct_change(REL_RET_WINDOW)
            rel_ret = stock_ret_20d - etf_ret_20d

            # Z-score using 60-day lookback of relative returns
            rel_mean = rel_ret.rolling(Z_LOOKBACK).mean()
            rel_std = rel_ret.rolling(Z_LOOKBACK).std()
            z = (rel_ret - rel_mean) / rel_std

            zscores[stock] = {
                "z": z,
                "sector": sector_name,
                "etf": etf,
                "rel_ret": rel_ret,
            }
    return zscores


def get_stock_sector(stock):
    """Return sector name for a stock."""
    for name, info in SECTOR_MAP.items():
        if stock in info["stocks"]:
            return name
    return None


# ── Helper Functions ───────────────────────────────────────────────────────────
def calc_metrics(returns, trades_count):
    """Calculate Sharpe, Sortino, MaxDD, PF, WR from daily returns series."""
    if len(returns) == 0 or returns.std() == 0:
        return {"sharpe": 0, "sortino": 0, "maxdd": -1.0, "pf": 0, "wr": 0,
                "total_ret": 0, "annual_ret": 0, "n_trades": trades_count, "valid": False}

    annual_factor = np.sqrt(252)
    sharpe = returns.mean() / returns.std() * annual_factor

    downside = returns[returns < 0]
    sortino = returns.mean() / downside.std() * annual_factor if len(downside) > 0 and downside.std() > 0 else sharpe * 1.5

    cum = (1 + returns).cumprod()
    rolling_max = cum.cummax()
    drawdowns = cum / rolling_max - 1
    maxdd = drawdowns.min()

    pos = returns[returns > 0].sum()
    neg = abs(returns[returns < 0].sum())
    pf = pos / neg if neg > 0 else 99.0

    wr = (returns > 0).sum() / len(returns) if len(returns) > 0 else 0

    total_ret = cum.iloc[-1] - 1 if len(cum) > 0 else 0
    years = len(returns) / 252
    annual_ret = (1 + total_ret) ** (1 / years) - 1 if years > 0 else 0

    return {
        "sharpe": round(float(sharpe), 3),
        "sortino": round(float(sortino), 3),
        "maxdd": round(float(maxdd), 4),
        "pf": round(float(pf), 3),
        "wr": round(float(wr), 4),
        "total_ret": round(float(total_ret), 4),
        "annual_ret": round(float(annual_ret), 4),
        "n_trades": trades_count,
        "valid": True,
    }


def regime_split(returns, spy_close):
    """Split returns into bull/bear based on SPY vs 200-SMA."""
    spy_sma200 = spy_close.rolling(200).mean()
    bull_mask = spy_close > spy_sma200
    bear_mask = spy_close <= spy_sma200

    bull_mask = bull_mask.reindex(returns.index).fillna(False)
    bear_mask = bear_mask.reindex(returns.index).fillna(False)

    bull_ret = returns[bull_mask]
    bear_ret = returns[bear_mask]
    return bull_ret, bear_ret


def regime_gap(returns, spy_close):
    """Calculate regime gap: |Sharpe_bull - Sharpe_bear| / max(|Sharpe_bull|, |Sharpe_bear|)."""
    bull_ret, bear_ret = regime_split(returns, spy_close)

    af = np.sqrt(252)
    sharpe_bull = bull_ret.mean() / bull_ret.std() * af if len(bull_ret) > 10 and bull_ret.std() > 0 else 0
    sharpe_bear = bear_ret.mean() / bear_ret.std() * af if len(bear_ret) > 10 and bear_ret.std() > 0 else 0

    max_abs = max(abs(sharpe_bull), abs(sharpe_bear))
    gap = abs(sharpe_bull - sharpe_bear) / max_abs if max_abs > 0 else 0

    return round(float(gap), 4), round(float(sharpe_bull), 3), round(float(sharpe_bear), 3)


def permutation_test(returns, n_iter=PERM_ITERATIONS):
    """Permutation test: shuffle which days are active to test signal timing."""
    if len(returns) == 0:
        return 1.0

    ret_vals = returns.values
    n = len(ret_vals)

    active_mask = ret_vals != 0
    n_active = int(active_mask.sum())

    if n_active < 5 or n_active >= n:
        return 1.0

    actual_std = ret_vals[active_mask].std()
    actual_sharpe = ret_vals[active_mask].mean() / actual_std * np.sqrt(252) if actual_std > 0 else 0

    count_better = 0
    for _ in range(n_iter):
        perm_idx = np.random.choice(n, size=n_active, replace=False)
        perm_active = ret_vals[perm_idx]
        perm_std = perm_active.std()
        perm_sharpe = perm_active.mean() / perm_std * np.sqrt(252) if perm_std > 0 else 0
        if perm_sharpe >= actual_sharpe:
            count_better += 1

    return round(count_better / n_iter, 4)


# ── Strategy Implementations ──────────────────────────────────────────────────

def variant_a_single_best(close, zscores, capital):
    """Single Best Signal: each day pick most extreme z < -2 stock. Hold 10d or until z > 0."""
    oot_mask = pd.Series(close.index >= OOT_START, index=close.index)
    daily_ret = pd.Series(0.0, index=close.index)
    trades = 0
    current_stock = None
    hold_days = 0
    trade_list = []

    for i in range(1, len(close)):
        if not oot_mask.iloc[i]:
            continue

        date = close.index[i]

        # If in a position, check exit
        if current_stock is not None:
            stock_ret = close[current_stock].iloc[i] / close[current_stock].iloc[i - 1] - 1
            daily_ret.iloc[i] = stock_ret
            hold_days += 1

            z_now = zscores[current_stock]["z"].iloc[i]
            if (not np.isnan(z_now) and z_now > Z_EXIT) or hold_days >= HOLD_MAX:
                daily_ret.iloc[i] -= SLIPPAGE * 2  # exit slippage
                trade_list.append({"stock": current_stock, "exit_date": str(date.date()), "hold_days": hold_days})
                current_stock = None
                hold_days = 0
            continue

        # Find stock with most extreme negative z-score
        best_stock = None
        best_z = 0
        for stock, data in zscores.items():
            z = data["z"].iloc[i]
            if not np.isnan(z) and z < Z_ENTRY and z < best_z:
                best_z = z
                best_stock = stock

        if best_stock is not None:
            current_stock = best_stock
            hold_days = 0
            trades += 1
            daily_ret.iloc[i] -= SLIPPAGE * 2  # entry slippage

    daily_ret = daily_ret[close.index >= OOT_START]
    return daily_ret, trades


def variant_b_multi_position(close, zscores, capital):
    """Multi-Position: up to 3 concurrent positions, one per sector. $215 each."""
    oot_mask = pd.Series(close.index >= OOT_START, index=close.index)
    daily_ret = pd.Series(0.0, index=close.index)
    trades = 0

    # Track positions: {sector: {"stock": str, "hold_days": int}}
    positions = {}
    max_positions = 3
    pos_capital = capital / max_positions

    for i in range(1, len(close)):
        if not oot_mask.iloc[i]:
            continue

        date = close.index[i]
        day_ret = 0.0

        # Update existing positions
        sectors_to_close = []
        for sector, pos in positions.items():
            stock = pos["stock"]
            stock_ret = close[stock].iloc[i] / close[stock].iloc[i - 1] - 1
            # Weight by position fraction
            day_ret += stock_ret * (pos_capital / capital)
            pos["hold_days"] += 1

            z_now = zscores[stock]["z"].iloc[i]
            if (not np.isnan(z_now) and z_now > Z_EXIT) or pos["hold_days"] >= HOLD_MAX:
                day_ret -= SLIPPAGE * 2 * (pos_capital / capital)
                sectors_to_close.append(sector)

        for sector in sectors_to_close:
            del positions[sector]

        # Look for new entries if we have room
        if len(positions) < max_positions:
            # Collect all eligible signals
            candidates = []
            for stock, data in zscores.items():
                z = data["z"].iloc[i]
                sector = data["sector"]
                if not np.isnan(z) and z < Z_ENTRY and sector not in positions:
                    candidates.append((z, stock, sector))

            # Sort by most extreme z (most negative first)
            candidates.sort()

            for z_val, stock, sector in candidates:
                if len(positions) >= max_positions:
                    break
                if sector in positions:
                    continue
                positions[sector] = {"stock": stock, "hold_days": 0}
                trades += 1
                day_ret -= SLIPPAGE * 2 * (pos_capital / capital)

        daily_ret.iloc[i] = day_ret

    daily_ret = daily_ret[close.index >= OOT_START]
    return daily_ret, trades


def variant_c_quality_filter(close, zscores, capital):
    """Quality Filter: same as A but only trade mega-cap quality stocks."""
    oot_mask = pd.Series(close.index >= OOT_START, index=close.index)
    daily_ret = pd.Series(0.0, index=close.index)
    trades = 0
    current_stock = None
    hold_days = 0

    quality_zscores = {k: v for k, v in zscores.items() if k in QUALITY_STOCKS}

    for i in range(1, len(close)):
        if not oot_mask.iloc[i]:
            continue

        if current_stock is not None:
            stock_ret = close[current_stock].iloc[i] / close[current_stock].iloc[i - 1] - 1
            daily_ret.iloc[i] = stock_ret
            hold_days += 1

            z_now = quality_zscores[current_stock]["z"].iloc[i]
            if (not np.isnan(z_now) and z_now > Z_EXIT) or hold_days >= HOLD_MAX:
                daily_ret.iloc[i] -= SLIPPAGE * 2
                current_stock = None
                hold_days = 0
            continue

        best_stock = None
        best_z = 0
        for stock, data in quality_zscores.items():
            z = data["z"].iloc[i]
            if not np.isnan(z) and z < Z_ENTRY and z < best_z:
                best_z = z
                best_stock = stock

        if best_stock is not None:
            current_stock = best_stock
            hold_days = 0
            trades += 1
            daily_ret.iloc[i] -= SLIPPAGE * 2

    daily_ret = daily_ret[close.index >= OOT_START]
    return daily_ret, trades


def variant_d_sector_rotation_rv(close, zscores, capital):
    """Sector Rotation + Relative Value: use RS rotation for sector, then buy lowest z stock."""
    oot_mask = pd.Series(close.index >= OOT_START, index=close.index)
    daily_ret = pd.Series(0.0, index=close.index)
    trades = 0
    current_stock = None
    hold_days = 0

    # Pre-compute sector momentum (3-month relative strength vs SPY)
    sector_rs = {}
    spy_ret_63 = close["SPY"].pct_change(63)  # ~3 months
    for sector_name, sector_info in SECTOR_MAP.items():
        etf = sector_info["etf"]
        if etf in close.columns:
            etf_ret_63 = close[etf].pct_change(63)
            sector_rs[sector_name] = etf_ret_63 - spy_ret_63  # relative strength

    for i in range(1, len(close)):
        if not oot_mask.iloc[i]:
            continue

        if current_stock is not None:
            stock_ret = close[current_stock].iloc[i] / close[current_stock].iloc[i - 1] - 1
            daily_ret.iloc[i] = stock_ret
            hold_days += 1

            z_now = zscores[current_stock]["z"].iloc[i]
            if (not np.isnan(z_now) and z_now > Z_EXIT) or hold_days >= HOLD_MAX:
                daily_ret.iloc[i] -= SLIPPAGE * 2
                current_stock = None
                hold_days = 0
            continue

        # Find strongest sector by RS
        best_sector = None
        best_rs = -999
        for sector_name, rs_series in sector_rs.items():
            rs_val = rs_series.iloc[i]
            if not np.isnan(rs_val) and rs_val > best_rs:
                best_rs = rs_val
                best_sector = sector_name

        if best_sector is None:
            continue

        # Within best sector, find stock with lowest z (most undervalued)
        best_stock = None
        best_z = 0
        for stock, data in zscores.items():
            if data["sector"] != best_sector:
                continue
            z = data["z"].iloc[i]
            if not np.isnan(z) and z < Z_ENTRY and z < best_z:
                best_z = z
                best_stock = stock

        if best_stock is not None:
            current_stock = best_stock
            hold_days = 0
            trades += 1
            daily_ret.iloc[i] -= SLIPPAGE * 2

    daily_ret = daily_ret[close.index >= OOT_START]
    return daily_ret, trades


def variant_e_momentum_confirm(close, zscores, capital):
    """Mean Reversion with Momentum Confirmation: z < -2 AND sector ETF has positive 20d momentum."""
    oot_mask = pd.Series(close.index >= OOT_START, index=close.index)
    daily_ret = pd.Series(0.0, index=close.index)
    trades = 0
    current_stock = None
    hold_days = 0

    # Pre-compute sector ETF 20-day momentum
    sector_mom = {}
    for sector_name, sector_info in SECTOR_MAP.items():
        etf = sector_info["etf"]
        if etf in close.columns:
            sector_mom[sector_name] = close[etf].pct_change(20)

    for i in range(1, len(close)):
        if not oot_mask.iloc[i]:
            continue

        if current_stock is not None:
            stock_ret = close[current_stock].iloc[i] / close[current_stock].iloc[i - 1] - 1
            daily_ret.iloc[i] = stock_ret
            hold_days += 1

            z_now = zscores[current_stock]["z"].iloc[i]
            if (not np.isnan(z_now) and z_now > Z_EXIT) or hold_days >= HOLD_MAX:
                daily_ret.iloc[i] -= SLIPPAGE * 2
                current_stock = None
                hold_days = 0
            continue

        best_stock = None
        best_z = 0
        for stock, data in zscores.items():
            z = data["z"].iloc[i]
            sector = data["sector"]
            if np.isnan(z) or z >= Z_ENTRY:
                continue
            # Check sector momentum is positive
            if sector in sector_mom:
                mom = sector_mom[sector].iloc[i]
                if np.isnan(mom) or mom <= 0:
                    continue
            else:
                continue
            if z < best_z:
                best_z = z
                best_stock = stock

        if best_stock is not None:
            current_stock = best_stock
            hold_days = 0
            trades += 1
            daily_ret.iloc[i] -= SLIPPAGE * 2

    daily_ret = daily_ret[close.index >= OOT_START]
    return daily_ret, trades


def variant_f_symmetric(close, zscores, capital):
    """Symmetric: buy stock when z < -2, buy sector ETF when z > +2."""
    oot_mask = pd.Series(close.index >= OOT_START, index=close.index)
    daily_ret = pd.Series(0.0, index=close.index)
    trades = 0
    current_asset = None  # ticker of what we're holding
    hold_days = 0
    current_z_stock = None  # which stock's z-score to track for exit

    for i in range(1, len(close)):
        if not oot_mask.iloc[i]:
            continue

        if current_asset is not None:
            asset_ret = close[current_asset].iloc[i] / close[current_asset].iloc[i - 1] - 1
            daily_ret.iloc[i] = asset_ret
            hold_days += 1

            z_now = zscores[current_z_stock]["z"].iloc[i]
            # For long stock (z was < -2): exit when z > 0
            # For long ETF (z was > +2): exit when z < 0
            exit_cond = False
            if current_asset == current_z_stock:
                # Long the stock (it was underperforming)
                if not np.isnan(z_now) and z_now > Z_EXIT:
                    exit_cond = True
            else:
                # Long the ETF (stock was overperforming, sector catching up)
                if not np.isnan(z_now) and z_now < -Z_EXIT:
                    exit_cond = True

            if exit_cond or hold_days >= HOLD_MAX:
                daily_ret.iloc[i] -= SLIPPAGE * 2
                current_asset = None
                current_z_stock = None
                hold_days = 0
            continue

        # Find most extreme signal in either direction
        best_stock = None
        best_abs_z = 0
        best_direction = 0  # -1 = stock lags (buy stock), +1 = stock leads (buy ETF)

        for stock, data in zscores.items():
            z = data["z"].iloc[i]
            if np.isnan(z):
                continue
            if z < Z_ENTRY and abs(z) > best_abs_z:
                best_abs_z = abs(z)
                best_stock = stock
                best_direction = -1
            elif z > -Z_ENTRY and abs(z) > best_abs_z:  # z > 2
                best_abs_z = abs(z)
                best_stock = stock
                best_direction = 1

        if best_stock is not None:
            if best_direction == -1:
                # Stock underperforming -> buy stock
                current_asset = best_stock
            else:
                # Stock overperforming -> buy sector ETF
                current_asset = zscores[best_stock]["etf"]
            current_z_stock = best_stock
            hold_days = 0
            trades += 1
            daily_ret.iloc[i] -= SLIPPAGE * 2

    daily_ret = daily_ret[close.index >= OOT_START]
    return daily_ret, trades


# ── 5-Gate Validation ──────────────────────────────────────────────────────────
def validate_5gate(metrics, perm_p, r_gap):
    """Apply 5-gate validation."""
    gates = {
        "sharpe_gt_0.5": metrics["sharpe"] > 0.5,
        "perm_p_lt_0.05": perm_p < 0.05,
        "regime_gap_lt_0.5": r_gap < 0.5,
        "maxdd_gt_neg50pct": metrics["maxdd"] > -0.50,
        "trades_gte_20": metrics["n_trades"] >= 20,
    }
    gates["all_passed"] = all(gates.values())
    return gates


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("RELATIVE VALUE INTRA-SECTOR BACKTEST")
    print("=" * 70)

    np.random.seed(42)

    # Download data
    close = download_data()

    # Compute z-scores
    print("\nComputing relative z-scores...")
    zscores = compute_relative_zscores(close)
    print(f"Computed z-scores for {len(zscores)} stocks across {len(SECTOR_MAP)} sectors")

    # Quick stats on signal frequency
    oot_dates = close.index[close.index >= OOT_START]
    signal_counts = {}
    for stock, data in zscores.items():
        z_oot = data["z"].reindex(oot_dates)
        n_signals = (z_oot < Z_ENTRY).sum()
        signal_counts[stock] = int(n_signals)
    print(f"\nSignal counts (z < -2 days) per stock during OOT:")
    for stock, cnt in sorted(signal_counts.items(), key=lambda x: -x[1]):
        print(f"  {stock}: {cnt} days")

    spy_close = close["SPY"]

    # Run all variants
    variants = {
        "A_single_best": ("Single Best Signal", variant_a_single_best),
        "B_multi_position": ("Multi-Position (up to 3)", variant_b_multi_position),
        "C_quality_filter": ("Quality Filter (mega-cap only)", variant_c_quality_filter),
        "D_sector_rotation_rv": ("Sector Rotation + Relative Value", variant_d_sector_rotation_rv),
        "E_momentum_confirm": ("Momentum Confirmation", variant_e_momentum_confirm),
        "F_symmetric": ("Symmetric (both directions)", variant_f_symmetric),
    }

    results = {
        "strategy": "Relative Value Intra-Sector",
        "thesis": "Quality stocks underperforming their sector ETF by z < -2 tend to mean-revert",
        "account_size": ACCOUNT_SIZE,
        "oot_period": f"{OOT_START} to {OOT_END}",
        "parameters": {
            "z_entry": Z_ENTRY,
            "z_exit": Z_EXIT,
            "rel_ret_window": REL_RET_WINDOW,
            "z_lookback": Z_LOOKBACK,
            "hold_max": HOLD_MAX,
            "slippage": SLIPPAGE,
        },
        "variants": {},
        "run_timestamp": datetime.now().isoformat(),
    }

    for var_key, (var_name, var_func) in variants.items():
        print(f"\n{'─' * 60}")
        print(f"Running Variant {var_key}: {var_name}")
        print(f"{'─' * 60}")

        daily_ret, n_trades = var_func(close, zscores, ACCOUNT_SIZE)

        # Overall metrics
        metrics = calc_metrics(daily_ret, n_trades)

        # Regime analysis
        r_gap, sharpe_bull, sharpe_bear = regime_gap(daily_ret, spy_close)

        # Bull/bear separate metrics
        bull_ret, bear_ret = regime_split(daily_ret, spy_close)
        bull_active = int((bull_ret != 0).sum())
        bear_active = int((bear_ret != 0).sum())
        bull_metrics = calc_metrics(bull_ret, bull_active) if bull_active > 0 else {"sharpe": 0, "sortino": 0, "n_trades": 0}
        bear_metrics = calc_metrics(bear_ret, bear_active) if bear_active > 0 else {"sharpe": 0, "sortino": 0, "n_trades": 0}

        # Permutation test
        print(f"  Running permutation test ({PERM_ITERATIONS} iterations)...")
        perm_p = permutation_test(daily_ret, n_iter=PERM_ITERATIONS)

        # 5-gate validation
        gates = validate_5gate(metrics, perm_p, r_gap)

        # Equity curve stats
        cum = (1 + daily_ret).cumprod()
        final_equity = round(float(ACCOUNT_SIZE * cum.iloc[-1]), 2)

        var_result = {
            "name": var_name,
            "metrics": metrics,
            "regime": {
                "gap": r_gap,
                "sharpe_bull": sharpe_bull,
                "sharpe_bear": sharpe_bear,
                "bull_trades": bull_active,
                "bear_trades": bear_active,
                "bull_sharpe_detail": round(float(bull_metrics["sharpe"]), 3),
                "bear_sharpe_detail": round(float(bear_metrics["sharpe"]), 3),
            },
            "permutation": {
                "p_value": perm_p,
                "iterations": PERM_ITERATIONS,
            },
            "gates": gates,
            "equity": {
                "start": ACCOUNT_SIZE,
                "end": final_equity,
            },
        }

        results["variants"][var_key] = var_result

        # Print summary
        status = "PASS" if gates["all_passed"] else "FAIL"
        print(f"\n  [{status}] {var_name}")
        print(f"  Sharpe: {metrics['sharpe']:.3f} | Sortino: {metrics['sortino']:.3f} | "
              f"PF: {metrics['pf']:.2f} | WR: {metrics['wr']:.1%}")
        print(f"  Trades: {n_trades} | MaxDD: {metrics['maxdd']:.2%} | "
              f"Total Return: {metrics['total_ret']:.2%}")
        print(f"  Regime Gap: {r_gap:.3f} (Bull: {sharpe_bull:.3f}, Bear: {sharpe_bear:.3f})")
        print(f"  Perm p-value: {perm_p:.4f}")
        print(f"  Equity: ${ACCOUNT_SIZE:.0f} → ${final_equity:.0f}")
        print(f"  Gates: {gates}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    passing = []
    for var_key, var_data in results["variants"].items():
        status = "PASS" if var_data["gates"]["all_passed"] else "FAIL"
        m = var_data["metrics"]
        print(f"  [{status}] {var_key}: Sharpe={m['sharpe']:.3f}, Trades={m['n_trades']}, "
              f"MaxDD={m['maxdd']:.2%}, Perm-p={var_data['permutation']['p_value']:.4f}, "
              f"RegGap={var_data['regime']['gap']:.3f}")
        if var_data["gates"]["all_passed"]:
            passing.append(var_key)

    if passing:
        print(f"\n  PASSING VARIANTS: {', '.join(passing)}")
    else:
        print(f"\n  NO VARIANTS PASSED all 5 gates.")

    # Save results
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {OUTPUT_PATH}")

    return results


if __name__ == "__main__":
    main()
