#!/usr/bin/env python3
"""
Utility / Staples / Low-Volatility Sector Backtest
===================================================
6 variants designed to be UNCORRELATED to QQQ.
OOT: 2022-01-01 to 2026-07-29, capital $645.

Variants:
  A: Utility Rate Signal (XLU + TLT)
  B: Defensive Sector Rotation (XLU/XLP/QQQ + VIX)
  C: Low Vol Factor (SPLV/SPY + VIX)
  D: Utility-Tech Divergence (XLU/QQQ ratio)
  E: Staples Momentum + VIX (XLP)
  F: Multi-Defensive Score (composite)

Validation: 5-gate framework + QQQ correlation + permutation test + regime split.
"""

import json
import sys
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

# ─── Config ───────────────────────────────────────────────────────────────────
CAPITAL = 645.0
SLIPPAGE_PCT = 0.0002  # 0.02%
COMMISSION = 0.0
OOT_START = "2022-01-01"
OOT_END = "2026-07-29"
DATA_START = "2021-01-01"  # extra history for indicators
PERM_ITERS = 1000
RESULT_PATH = "/home/jupiter/Lvl3Quant/data/utility_lowvol_results.json"

TICKERS = ["XLU", "XLP", "SPLV", "USMV", "VPU", "SPY", "QQQ", "TLT", "^VIX"]


def fetch_data():
    """Download all needed price data via yfinance."""
    print("Fetching data...")
    raw = yf.download(TICKERS, start=DATA_START, end=OOT_END, auto_adjust=True, progress=False)
    close = raw["Close"].copy()
    # VIX comes in as ^VIX
    if "^VIX" in close.columns:
        close.rename(columns={"^VIX": "VIX"}, inplace=True)
    close = close.ffill().dropna(how="all")
    return close


def apply_slippage(price, direction="buy"):
    """Apply slippage to price."""
    if direction == "buy":
        return price * (1 + SLIPPAGE_PCT)
    return price * (1 - SLIPPAGE_PCT)


def run_backtest(close, variant_fn, variant_name):
    """
    Generic backtest engine.
    variant_fn(close, date_idx) -> (ticker_to_buy_or_None, hold_days)
    Returns daily equity series aligned to OOT period.
    """
    oot_mask = close.index >= pd.Timestamp(OOT_START)
    oot_dates = close.index[oot_mask]

    equity = CAPITAL
    position = None  # (ticker, shares, entry_price, exit_date)
    equity_series = []

    for i, dt in enumerate(close.index):
        if dt < pd.Timestamp(OOT_START):
            continue

        # Check if position should be closed
        if position is not None:
            ticker, shares, entry_price, exit_date = position
            current_price = close.loc[dt, ticker] if ticker in close.columns and not pd.isna(close.loc[dt, ticker]) else entry_price
            if dt >= exit_date or pd.isna(close.loc[dt, ticker]):
                sell_price = apply_slippage(current_price, "sell")
                equity = shares * sell_price - COMMISSION
                position = None
            else:
                equity = shares * current_price

        # If no position, check for signal
        if position is None:
            # Get the index position within the full dataframe
            full_idx = close.index.get_loc(dt)
            signal = variant_fn(close, full_idx)
            if signal is not None:
                ticker, hold_days = signal
                if ticker in close.columns and not pd.isna(close.loc[dt, ticker]):
                    buy_price = apply_slippage(close.loc[dt, ticker], "buy")
                    shares = (equity - COMMISSION) / buy_price
                    exit_idx = min(full_idx + hold_days, len(close.index) - 1)
                    exit_date = close.index[exit_idx]
                    position = (ticker, shares, buy_price, exit_date)
                    equity = shares * close.loc[dt, ticker]

        equity_series.append({"date": dt, "equity": equity})

    eq_df = pd.DataFrame(equity_series)
    eq_df.set_index("date", inplace=True)
    return eq_df


# ─── Variant Definitions ─────────────────────────────────────────────────────

def variant_a(close, idx):
    """Utility Rate Signal: Long XLU when TLT 20d return > 0."""
    if idx < 20:
        return None
    tlt_now = close.iloc[idx]["TLT"]
    tlt_20ago = close.iloc[idx - 20]["TLT"]
    if pd.isna(tlt_now) or pd.isna(tlt_20ago):
        return None
    if (tlt_now - tlt_20ago) / tlt_20ago > 0:
        return ("XLU", 15)
    return None


def variant_b(close, idx):
    """Defensive Sector Rotation: monthly rebal, rank XLU/XLP by 1m mom when VIX>18, else QQQ."""
    if idx < 21:
        return None
    # Only rebalance ~monthly (every 21 trading days)
    oot_start_idx = close.index.get_loc(close.index[close.index >= pd.Timestamp(OOT_START)][0])
    days_since_oot = idx - oot_start_idx
    if days_since_oot < 0:
        return None
    if days_since_oot % 21 != 0:
        return None

    vix = close.iloc[idx].get("VIX", np.nan)
    if pd.isna(vix):
        return None

    if vix > 18:
        xlu_mom = (close.iloc[idx]["XLU"] / close.iloc[idx - 21]["XLU"]) - 1
        xlp_mom = (close.iloc[idx]["XLP"] / close.iloc[idx - 21]["XLP"]) - 1
        if pd.isna(xlu_mom) or pd.isna(xlp_mom):
            return None
        ticker = "XLU" if xlu_mom > xlp_mom else "XLP"
        return (ticker, 21)
    else:
        return ("QQQ", 21)


def variant_c(close, idx):
    """Low Vol Factor: Long SPLV when VIX>22, SPY when VIX<18, cash 18-22."""
    if idx < 1:
        return None
    vix = close.iloc[idx].get("VIX", np.nan)
    if pd.isna(vix):
        return None
    if vix > 22:
        return ("SPLV", 10)
    elif vix < 18:
        return ("SPY", 10)
    return None


def variant_d(close, idx):
    """Utility-Tech Divergence: Long XLU when XLU/QQQ ratio >3% above 20d MA."""
    if idx < 25:
        return None
    xlu = close.iloc[idx]["XLU"]
    qqq = close.iloc[idx]["QQQ"]
    if pd.isna(xlu) or pd.isna(qqq) or qqq == 0:
        return None
    ratio = xlu / qqq
    # 20d MA of ratio
    ratios = []
    for j in range(idx - 19, idx + 1):
        x = close.iloc[j]["XLU"]
        q = close.iloc[j]["QQQ"]
        if pd.isna(x) or pd.isna(q) or q == 0:
            return None
        ratios.append(x / q)
    ma20 = np.mean(ratios)
    if ratio > ma20 * 1.03:
        return ("XLU", 15)
    return None


def variant_e(close, idx):
    """Staples Momentum + VIX: Long XLP when 20d mom > 0 AND VIX > 18."""
    if idx < 20:
        return None
    xlp_now = close.iloc[idx]["XLP"]
    xlp_20ago = close.iloc[idx - 20]["XLP"]
    vix = close.iloc[idx].get("VIX", np.nan)
    if pd.isna(xlp_now) or pd.isna(xlp_20ago) or pd.isna(vix):
        return None
    mom = (xlp_now - xlp_20ago) / xlp_20ago
    if mom > 0 and vix > 18:
        return ("XLP", 10)
    return None


def variant_f(close, idx):
    """Multi-Defensive Score: composite of 4 signals."""
    if idx < 25:
        return None

    score = 0
    # XLU 1m momentum > 0
    xlu_now = close.iloc[idx]["XLU"]
    xlu_21ago = close.iloc[idx - 21]["XLU"] if idx >= 21 else np.nan
    if not pd.isna(xlu_now) and not pd.isna(xlu_21ago) and xlu_21ago > 0:
        if (xlu_now / xlu_21ago - 1) > 0:
            score += 1

    # XLP 1m momentum > 0
    xlp_now = close.iloc[idx]["XLP"]
    xlp_21ago = close.iloc[idx - 21]["XLP"] if idx >= 21 else np.nan
    if not pd.isna(xlp_now) and not pd.isna(xlp_21ago) and xlp_21ago > 0:
        if (xlp_now / xlp_21ago - 1) > 0:
            score += 1

    # VIX > 20
    vix = close.iloc[idx].get("VIX", np.nan)
    if not pd.isna(vix) and vix > 20:
        score += 1

    # TLT uptrend (price > 20d SMA)
    tlt_now = close.iloc[idx]["TLT"]
    if not pd.isna(tlt_now) and idx >= 20:
        tlt_sma = np.mean([close.iloc[j]["TLT"] for j in range(idx - 19, idx + 1)
                           if not pd.isna(close.iloc[j]["TLT"])])
        if tlt_now > tlt_sma:
            score += 1

    # Only rebalance every 10 days
    oot_start_idx = close.index.get_loc(close.index[close.index >= pd.Timestamp(OOT_START)][0])
    days_since_oot = idx - oot_start_idx
    if days_since_oot < 0:
        return None
    if days_since_oot % 10 != 0:
        return None

    if score >= 3:
        return ("XLU", 10)
    elif score <= 1:
        return None
    # score == 2 → hold previous (return None = stay in cash if no position)
    return None


# ─── Analytics ────────────────────────────────────────────────────────────────

def compute_metrics(eq_df, qqq_returns, spy_close):
    """Compute Sharpe, Sortino, PF, WR, MDD, QQQ correlation, regime analysis."""
    if eq_df.empty or len(eq_df) < 5:
        return None

    returns = eq_df["equity"].pct_change().dropna()
    if len(returns) < 5:
        return None

    # Align with QQQ returns
    common_idx = returns.index.intersection(qqq_returns.index)
    if len(common_idx) < 5:
        return None

    strat_ret = returns.loc[common_idx]
    qqq_ret = qqq_returns.loc[common_idx]

    # Basic metrics
    ann_factor = np.sqrt(252)
    mean_ret = strat_ret.mean()
    std_ret = strat_ret.std()
    sharpe = (mean_ret / std_ret * ann_factor) if std_ret > 0 else 0

    downside = strat_ret[strat_ret < 0].std()
    sortino = (mean_ret / downside * ann_factor) if downside > 0 else 0

    positive = strat_ret[strat_ret > 0].sum()
    negative = abs(strat_ret[strat_ret < 0].sum())
    pf = (positive / negative) if negative > 0 else float("inf")

    wr = (strat_ret > 0).sum() / len(strat_ret) if len(strat_ret) > 0 else 0

    # MDD
    cummax = eq_df["equity"].cummax()
    drawdown = (eq_df["equity"] - cummax) / cummax
    mdd = drawdown.min()

    # QQQ correlation
    qqq_corr = strat_ret.corr(qqq_ret)

    # Total return
    total_return = (eq_df["equity"].iloc[-1] / eq_df["equity"].iloc[0]) - 1

    # CAGR
    n_years = len(eq_df) / 252
    cagr = (eq_df["equity"].iloc[-1] / eq_df["equity"].iloc[0]) ** (1 / n_years) - 1 if n_years > 0 else 0

    # Trade count (approximate: count transitions from flat to invested)
    equity_changes = eq_df["equity"].pct_change().fillna(0)
    # Count days where we entered a new position (equity changed from ~0 change to non-zero)
    trades = 0
    in_trade = False
    for r in strat_ret:
        if abs(r) > 1e-8 and not in_trade:
            trades += 1
            in_trade = True
        elif abs(r) < 1e-8:
            in_trade = False

    # Regime split: SPY above/below 200-SMA
    spy_sma200 = spy_close.rolling(200).mean()
    common_spy = common_idx.intersection(spy_sma200.dropna().index)
    if len(common_spy) > 10:
        bull_mask = spy_close.loc[common_spy] > spy_sma200.loc[common_spy]
        bear_mask = ~bull_mask

        bull_ret = strat_ret.loc[common_spy][bull_mask]
        bear_ret = strat_ret.loc[common_spy][bear_mask]

        bull_sharpe = (bull_ret.mean() / bull_ret.std() * ann_factor) if len(bull_ret) > 5 and bull_ret.std() > 0 else 0
        bear_sharpe = (bear_ret.mean() / bear_ret.std() * ann_factor) if len(bear_ret) > 5 and bear_ret.std() > 0 else 0

        regime_gap = abs(bull_sharpe - bear_sharpe) / max(abs(bull_sharpe), abs(bear_sharpe), 1e-8)
    else:
        bull_sharpe = bear_sharpe = regime_gap = 0

    return {
        "sharpe": round(sharpe, 4),
        "sortino": round(sortino, 4),
        "profit_factor": round(pf, 4),
        "win_rate": round(wr, 4),
        "mdd": round(mdd, 4),
        "total_return": round(total_return, 4),
        "cagr": round(cagr, 4),
        "qqq_correlation": round(qqq_corr, 4),
        "trades": trades,
        "bull_sharpe": round(bull_sharpe, 4),
        "bear_sharpe": round(bear_sharpe, 4),
        "regime_gap": round(regime_gap, 4),
        "final_equity": round(eq_df["equity"].iloc[-1], 2),
        "n_days": len(eq_df),
    }


def permutation_test(eq_df, qqq_returns, underlying_returns=None, n_iters=PERM_ITERS):
    """
    Permutation test: compare observed Sharpe vs random entry timing.
    We identify invested vs cash days, then randomly reassign which days
    are invested (preserving the fraction of time in market).
    """
    returns = eq_df["equity"].pct_change().dropna()
    if len(returns) < 10:
        return 1.0

    # Identify invested days (non-trivial returns)
    invested_mask = returns.abs() > 1e-10
    n_invested = invested_mask.sum()
    n_total = len(returns)

    if n_invested < 5 or n_invested == n_total:
        return 1.0

    # For the null, we need the underlying asset returns on all days
    # Use the actual non-zero returns as the pool of "what happens when invested"
    # The key insight: under null hypothesis, signal timing is random
    # So we randomly pick which days we're "invested" from the available asset returns

    # Get the underlying returns for all days in the period
    common_idx = returns.index
    if underlying_returns is not None:
        asset_ret = underlying_returns.reindex(common_idx).fillna(0).values
    else:
        asset_ret = returns.values.copy()

    # Observed: mean return only on invested days, annualized
    obs_invested_ret = returns[invested_mask].values
    obs_mean = obs_invested_ret.mean()
    # Strategy Sharpe: use all returns (including cash=0 days)
    observed_sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0

    count_above = 0
    rng = np.random.default_rng(42)

    for _ in range(n_iters):
        # Randomly pick n_invested days to be "invested"
        perm_mask = np.zeros(n_total, dtype=bool)
        perm_idx = rng.choice(n_total, size=n_invested, replace=False)
        perm_mask[perm_idx] = True

        # Construct synthetic returns: asset return on invested days, 0 on cash days
        synth_ret = np.where(perm_mask, asset_ret, 0.0)
        std = synth_ret.std()
        s = synth_ret.mean() / std * np.sqrt(252) if std > 0 else 0
        if s >= observed_sharpe:
            count_above += 1

    return round(count_above / n_iters, 4)


def validate_5gate(metrics, perm_p):
    """5-gate validation: Sharpe>0.5, perm p<0.05, regime_gap<0.5, MDD>-50%, trades>=20."""
    if metrics is None:
        return {"pass": False, "reason": "no_data"}

    gates = {
        "sharpe_gt_0.5": metrics["sharpe"] > 0.5,
        "perm_p_lt_0.05": perm_p < 0.05,
        "regime_gap_lt_0.5": metrics["regime_gap"] < 0.5,
        "mdd_gt_neg50pct": metrics["mdd"] > -0.50,
        "trades_gte_20": metrics["trades"] >= 20,
    }
    passed = all(gates.values())
    return {"pass": passed, "gates": gates, "perm_p": perm_p}


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    close = fetch_data()
    print(f"Data shape: {close.shape}, columns: {list(close.columns)}")
    print(f"Date range: {close.index[0].date()} to {close.index[-1].date()}")

    # QQQ returns for correlation
    qqq_returns = close["QQQ"].pct_change().dropna()
    spy_close = close["SPY"]

    variants = {
        "A_utility_rate_signal": (variant_a, "XLU"),
        "B_defensive_sector_rotation": (variant_b, "XLU"),
        "C_low_vol_factor": (variant_c, "SPY"),
        "D_utility_tech_divergence": (variant_d, "XLU"),
        "E_staples_momentum_vix": (variant_e, "XLP"),
        "F_multi_defensive_score": (variant_f, "XLU"),
    }

    results = {
        "metadata": {
            "run_timestamp": datetime.now().isoformat(),
            "oot_start": OOT_START,
            "oot_end": OOT_END,
            "capital": CAPITAL,
            "slippage_pct": SLIPPAGE_PCT,
            "commission": COMMISSION,
            "perm_iterations": PERM_ITERS,
            "purpose": "Find strategies UNCORRELATED to QQQ (Sharpe 2.38)",
        },
        "variants": {},
    }

    # QQQ benchmark
    qqq_oot = close["QQQ"][close.index >= pd.Timestamp(OOT_START)]
    qqq_ret_oot = qqq_oot.pct_change().dropna()
    qqq_sharpe = qqq_ret_oot.mean() / qqq_ret_oot.std() * np.sqrt(252) if qqq_ret_oot.std() > 0 else 0
    qqq_total = (qqq_oot.iloc[-1] / qqq_oot.iloc[0]) - 1
    results["qqq_benchmark"] = {
        "sharpe": round(qqq_sharpe, 4),
        "total_return": round(qqq_total, 4),
    }
    print(f"\nQQQ benchmark: Sharpe={qqq_sharpe:.4f}, Return={qqq_total:.2%}")

    for name, (fn, primary_etf) in variants.items():
        print(f"\n{'='*60}")
        print(f"Running variant: {name}")
        eq_df = run_backtest(close, fn, name)
        metrics = compute_metrics(eq_df, qqq_returns, spy_close)

        if metrics is None:
            print(f"  SKIP: insufficient data")
            results["variants"][name] = {"status": "insufficient_data"}
            continue

        print(f"  Sharpe={metrics['sharpe']:.4f}, Sortino={metrics['sortino']:.4f}, "
              f"PF={metrics['profit_factor']:.4f}, WR={metrics['win_rate']:.2%}")
        print(f"  MDD={metrics['mdd']:.2%}, Return={metrics['total_return']:.2%}, "
              f"QQQ_corr={metrics['qqq_correlation']:.4f}")
        print(f"  Trades={metrics['trades']}, Bull_Sharpe={metrics['bull_sharpe']:.4f}, "
              f"Bear_Sharpe={metrics['bear_sharpe']:.4f}, Regime_Gap={metrics['regime_gap']:.4f}")

        # Permutation test - use primary ETF returns as underlying
        underlying_ret = close[primary_etf].pct_change().dropna()
        print(f"  Running permutation test ({PERM_ITERS} iterations)...")
        perm_p = permutation_test(eq_df, qqq_returns, underlying_returns=underlying_ret)
        print(f"  Perm p-value={perm_p:.4f}")

        # 5-gate validation
        validation = validate_5gate(metrics, perm_p)
        print(f"  5-Gate PASS: {validation['pass']}")
        if not validation["pass"] and "gates" in validation:
            failed = [g for g, v in validation["gates"].items() if not v]
            print(f"  Failed gates: {failed}")

        results["variants"][name] = {
            "metrics": metrics,
            "permutation_p_value": perm_p,
            "validation": validation,
        }

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    any_pass = False
    for name, data in results["variants"].items():
        if "validation" in data and data["validation"].get("pass"):
            any_pass = True
            m = data["metrics"]
            print(f"  PASS: {name} — Sharpe={m['sharpe']}, QQQ_corr={m['qqq_correlation']}")
        elif "metrics" in data:
            m = data["metrics"]
            v = data.get("validation", {})
            failed = [g for g, ok in v.get("gates", {}).items() if not ok]
            print(f"  FAIL: {name} — Sharpe={m['sharpe']}, QQQ_corr={m['qqq_correlation']}, failed={failed}")
        else:
            print(f"  SKIP: {name}")

    if not any_pass:
        print("\n  No variants passed all 5 gates.")

    results["summary"] = {
        "any_passed": any_pass,
        "total_variants": len(variants),
        "passed_count": sum(1 for d in results["variants"].values()
                           if "validation" in d and d["validation"].get("pass")),
    }

    # Save results
    Path(RESULT_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(RESULT_PATH, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {RESULT_PATH}")

    return results


if __name__ == "__main__":
    main()
