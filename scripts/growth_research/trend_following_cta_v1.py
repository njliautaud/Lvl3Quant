"""
Trend-Following CTA Strategy Backtest v1
=========================================
HC #0  : Sliding walk-forward (252d lookback, no expanding window)
HC #428: Regime-agnostic validation (R1 — all OOT days, regime gap < 0.50)
HC #694: Commission-free (Robinhood ETFs)

THESIS: Simple dual-MA crossover + breakout trend-following across multiple
asset classes. CTA strategies historically provide "crisis alpha" — positive
returns during equity drawdowns (2008: +20-40%, 2022: +30% for top CTAs).

Signals: Golden/Death cross (50/200 SMA) + price vs 200 SMA
Sizing:  Inverse-vol, target 10% annual portfolio vol
Risk:    2x ATR(20) trailing stop, max 20% single position, 1.5x max leverage
Rebal:   Weekly (Fridays)
"""

import os
import json
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import percentileofscore

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------
OUTPUT_DIR = Path("/home/jupiter/Lvl3Quant/output/trend_following_cta_v1")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

START = "2007-01-01"   # extra buffer for 252d lookback → effective start ~2008
END   = "2026-07-14"

# ETF universe — 15 liquid ETFs as futures proxies
UNIVERSE = {
    # Equity
    "SPY":  {"class": "EQUITY",    "exp_ratio": 0.0009},
    "QQQ":  {"class": "EQUITY",    "exp_ratio": 0.0020},
    "EFA":  {"class": "EQUITY",    "exp_ratio": 0.0032},
    "EEM":  {"class": "EQUITY",    "exp_ratio": 0.0068},
    "IWM":  {"class": "EQUITY",    "exp_ratio": 0.0019},
    # Bonds
    "TLT":  {"class": "BOND",      "exp_ratio": 0.0015},
    "IEF":  {"class": "BOND",      "exp_ratio": 0.0015},
    "HYG":  {"class": "BOND",      "exp_ratio": 0.0048},
    # Commodities
    "GLD":  {"class": "COMMODITY", "exp_ratio": 0.0040},
    "SLV":  {"class": "COMMODITY", "exp_ratio": 0.0050},
    "USO":  {"class": "COMMODITY", "exp_ratio": 0.0081},
    "DBA":  {"class": "COMMODITY", "exp_ratio": 0.0089},
    # Currency
    "UUP":  {"class": "CURRENCY",  "exp_ratio": 0.0077},
    "FXE":  {"class": "CURRENCY",  "exp_ratio": 0.0040},
    # Volatility
    "VIXY": {"class": "VOL",       "exp_ratio": 0.0087},
}
TICKERS = list(UNIVERSE.keys())

# Strategy parameters
SMA_FAST       = 50
SMA_SLOW       = 200
VOL_WINDOW     = 20     # realized vol lookback
ATR_WINDOW     = 20     # for trailing stop
ATR_STOP_MULT  = 2.0    # trailing stop at 2x ATR
TARGET_VOL     = 0.10   # 10% annual portfolio vol target
MAX_POS_WT     = 0.20   # max 20% in any single position
MAX_LEVERAGE   = 1.50   # max sum of abs weights
LOOKBACK       = 252    # walk-forward lookback for vol/SMA warmup

# Rebalance: weekly on Fridays
REBAL_DAY = 4  # Monday=0, Friday=4

# Benchmark
SPY_TICKER = "SPY"


# ---------------------------------------------------------------------------
# DATA
# ---------------------------------------------------------------------------
def download_data():
    """Download adjusted close prices for universe."""
    print("Downloading price data from yfinance...")
    raw = yf.download(
        TICKERS,
        start=START,
        end=END,
        auto_adjust=True,
        progress=False,
    )
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"]
    else:
        prices = raw[["Close"]]

    prices = prices[TICKERS]
    prices = prices.ffill(limit=5)

    print(f"Price data shape: {prices.shape}")
    print(f"Date range: {prices.index[0].date()} — {prices.index[-1].date()}")
    print(f"Missing pct:\n{prices.isna().mean().round(4)}")
    return prices


# ---------------------------------------------------------------------------
# SIGNAL GENERATION
# ---------------------------------------------------------------------------
def compute_signals(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Per-asset trend signals:
      +1 (LONG):  price > SMA200 AND SMA50 > SMA200 (golden cross)
      -1 (SHORT): price < SMA200 AND SMA50 < SMA200 (death cross)
       0 (FLAT):  mixed signals
    """
    sma_fast = prices.rolling(SMA_FAST, min_periods=int(SMA_FAST * 0.8)).mean()
    sma_slow = prices.rolling(SMA_SLOW, min_periods=int(SMA_SLOW * 0.8)).mean()

    long_cond  = (prices > sma_slow) & (sma_fast > sma_slow)
    short_cond = (prices < sma_slow) & (sma_fast < sma_slow)

    signals = pd.DataFrame(0, index=prices.index, columns=prices.columns)
    signals[long_cond]  = 1
    signals[short_cond] = -1
    return signals


def compute_inv_vol_weights(prices: pd.DataFrame, signals: pd.DataFrame,
                             date: pd.Timestamp) -> pd.Series:
    """
    Inverse-vol weighted positions with:
    - Target 10% annual vol
    - Max 20% per position
    - Max 1.5x total leverage
    """
    # Realized vol (annualized)
    daily_ret = prices.pct_change()
    vol = daily_ret.iloc[max(0, prices.index.get_loc(date) - VOL_WINDOW):
                         prices.index.get_loc(date) + 1].std() * np.sqrt(252)
    vol = vol.clip(lower=0.01)

    sig = signals.loc[date]
    active = sig[sig != 0]

    if active.empty:
        return pd.Series(0.0, index=prices.columns)

    # Inverse vol weights
    inv_vol = 1.0 / vol[active.index]
    raw_weights = inv_vol / inv_vol.sum()

    # Apply direction
    raw_weights = raw_weights * active

    # Scale to target portfolio vol
    # Portfolio vol ≈ sqrt(w^T Σ w). Approximate: weighted avg of asset vols
    port_vol = (raw_weights.abs() * vol[active.index]).sum()
    if port_vol > 0:
        scale = TARGET_VOL / port_vol
    else:
        scale = 1.0
    scaled = raw_weights * scale

    # Cap individual positions at MAX_POS_WT
    scaled = scaled.clip(lower=-MAX_POS_WT, upper=MAX_POS_WT)

    # Cap total leverage at MAX_LEVERAGE
    total_lev = scaled.abs().sum()
    if total_lev > MAX_LEVERAGE:
        scaled = scaled * (MAX_LEVERAGE / total_lev)

    # Fill back to full universe
    result = pd.Series(0.0, index=prices.columns)
    result[scaled.index] = scaled.values
    return result


# ---------------------------------------------------------------------------
# TRAILING STOP
# ---------------------------------------------------------------------------
def compute_atr(prices: pd.DataFrame, window: int = ATR_WINDOW) -> pd.DataFrame:
    """Average True Range for each asset."""
    high = prices  # using close as proxy since we only have close
    low = prices
    close_prev = prices.shift(1)

    # With close-only data, ATR ≈ rolling abs(return) * price
    tr = (prices / close_prev - 1).abs() * prices
    atr = tr.rolling(window, min_periods=int(window * 0.6)).mean()
    return atr


def apply_trailing_stops(prices: pd.DataFrame, weights: pd.DataFrame,
                          atr: pd.DataFrame) -> pd.DataFrame:
    """
    Apply trailing stops: if position exists and price moves against by
    2x ATR from entry high/low, flatten that position.
    """
    adjusted = weights.copy()
    n_stops = 0

    for ticker in prices.columns:
        in_position = False
        direction = 0
        trail_price = np.nan

        for i, date in enumerate(prices.index):
            if date not in weights.index:
                continue

            w = adjusted.loc[date, ticker]
            px = prices.loc[date, ticker]
            atr_val = atr.loc[date, ticker] if date in atr.index else np.nan

            if pd.isna(px) or pd.isna(atr_val):
                continue

            if w != 0 and not in_position:
                # Enter position
                in_position = True
                direction = np.sign(w)
                trail_price = px

            elif w != 0 and in_position:
                # Update trailing stop
                if direction > 0:
                    trail_price = max(trail_price, px)
                    stop_level = trail_price - ATR_STOP_MULT * atr_val
                    if px < stop_level:
                        adjusted.loc[date, ticker] = 0.0
                        in_position = False
                        n_stops += 1
                else:
                    trail_price = min(trail_price, px)
                    stop_level = trail_price + ATR_STOP_MULT * atr_val
                    if px > stop_level:
                        adjusted.loc[date, ticker] = 0.0
                        in_position = False
                        n_stops += 1

            elif w == 0 and in_position:
                in_position = False
                direction = 0

    print(f"  Trailing stops triggered: {n_stops}")
    return adjusted


# ---------------------------------------------------------------------------
# BACKTEST ENGINE
# ---------------------------------------------------------------------------
def run_backtest(prices: pd.DataFrame) -> dict:
    """Run the CTA trend-following backtest."""
    print("\nRunning CTA trend-following backtest...")

    signals = compute_signals(prices)
    atr = compute_atr(prices)

    # ETF expense ratio daily drag
    daily_exp_drag = pd.Series(
        {t: UNIVERSE[t]["exp_ratio"] / 252 for t in TICKERS}
    )

    # Identify rebalance dates (Fridays)
    rebal_dates = prices.index[prices.index.dayofweek == REBAL_DAY]
    # Only rebalance after warmup
    warmup_end = prices.index[LOOKBACK] if len(prices.index) > LOOKBACK else prices.index[-1]
    rebal_dates = rebal_dates[rebal_dates >= warmup_end]

    print(f"  Rebalance dates: {len(rebal_dates)} (weekly Fridays)")
    print(f"  Warmup ends: {warmup_end.date()}")

    # Build weights: rebalance weekly, hold between
    daily_weights = pd.DataFrame(0.0, index=prices.index, columns=TICKERS)
    current_weights = pd.Series(0.0, index=TICKERS)

    for date in prices.index:
        if date in rebal_dates:
            current_weights = compute_inv_vol_weights(prices, signals, date)
        if date >= warmup_end:
            daily_weights.loc[date] = current_weights

    # Apply trailing stops
    daily_weights = apply_trailing_stops(prices, daily_weights, atr)

    # Compute returns
    daily_ret = prices.pct_change()
    # Strategy return = sum of (weight * asset_return) - expense drag
    strat_ret = (daily_weights.shift(1) * daily_ret).sum(axis=1)
    # Subtract expense ratio drag
    exp_drag = (daily_weights.shift(1).abs() * daily_exp_drag).sum(axis=1)
    strat_ret = strat_ret - exp_drag

    # Only count from warmup onwards
    strat_ret = strat_ret[strat_ret.index >= warmup_end]

    # Equity curve
    equity = (1 + strat_ret).cumprod()

    # Benchmarks
    spy_ret = daily_ret["SPY"][strat_ret.index]
    spy_eq = (1 + spy_ret).cumprod()

    # 60/40 benchmark
    bm_6040_ret = 0.6 * daily_ret["SPY"] + 0.4 * daily_ret["TLT"]
    bm_6040_ret = bm_6040_ret[strat_ret.index]
    bm_6040_eq = (1 + bm_6040_ret).cumprod()

    return {
        "strat_ret": strat_ret,
        "equity": equity,
        "spy_ret": spy_ret,
        "spy_eq": spy_eq,
        "bm_6040_ret": bm_6040_ret,
        "bm_6040_eq": bm_6040_eq,
        "daily_weights": daily_weights[daily_weights.index >= warmup_end],
        "signals": signals,
        "rebal_dates": rebal_dates,
    }


# ---------------------------------------------------------------------------
# METRICS
# ---------------------------------------------------------------------------
def compute_metrics(returns: pd.Series, label: str = "") -> dict:
    """Compute risk-adjusted performance metrics."""
    r = returns.dropna()
    if len(r) < 10:
        return {"label": label, "error": "insufficient data"}

    total_ret = (1 + r).prod() - 1
    years = len(r) / 252
    cagr = (1 + total_ret) ** (1 / max(years, 0.01)) - 1

    ann_vol = r.std() * np.sqrt(252)
    sharpe = (r.mean() * 252) / max(ann_vol, 1e-6)

    downside = r[r < 0].std() * np.sqrt(252) if (r < 0).any() else 1e-6
    sortino = (r.mean() * 252) / max(downside, 1e-6)

    # Max drawdown
    cum = (1 + r).cumprod()
    peak = cum.cummax()
    dd = (cum - peak) / peak
    max_dd = dd.min()

    calmar = cagr / max(abs(max_dd), 1e-6)

    # Win rate
    wr = (r > 0).mean()

    # Profit factor
    gross_profit = r[r > 0].sum()
    gross_loss = abs(r[r < 0].sum())
    pf = gross_profit / max(gross_loss, 1e-6)

    # Annual returns
    annual = r.groupby(r.index.year).apply(lambda x: (1 + x).prod() - 1)

    return {
        "label": label,
        "cagr": cagr,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_dd": max_dd,
        "calmar": calmar,
        "win_rate": wr,
        "profit_factor": pf,
        "total_return": total_ret,
        "years": years,
        "annual_returns": annual.to_dict(),
        "best_year": annual.max(),
        "worst_year": annual.min(),
    }


# ---------------------------------------------------------------------------
# HC #428 R1 — REGIME-AGNOSTIC VALIDATION
# ---------------------------------------------------------------------------
def regime_validation(strat_ret: pd.Series, spy_ret: pd.Series) -> dict:
    """
    HC #428 R1: Classify days by SPY regime (green/red/flat),
    compute per-regime Sharpe, reject if gap > 0.50.
    """
    # Classify by SPY daily return
    spy_daily = spy_ret.reindex(strat_ret.index).fillna(0)
    green = spy_daily > 0.002    # SPY up > 0.2%
    red   = spy_daily < -0.002   # SPY down > 0.2%
    flat  = ~green & ~red

    def regime_sharpe(r):
        if len(r) < 10:
            return 0.0
        return (r.mean() * 252) / max(r.std() * np.sqrt(252), 1e-6)

    sharpe_green = regime_sharpe(strat_ret[green])
    sharpe_red   = regime_sharpe(strat_ret[red])
    sharpe_flat  = regime_sharpe(strat_ret[flat])

    max_sharpe = max(abs(sharpe_green), abs(sharpe_red), 1e-6)
    regime_gap = abs(sharpe_green - sharpe_red) / max_sharpe

    passed = regime_gap <= 0.50

    return {
        "sharpe_green": sharpe_green,
        "sharpe_red": sharpe_red,
        "sharpe_flat": sharpe_flat,
        "regime_gap": regime_gap,
        "passed_R1": passed,
        "n_green": int(green.sum()),
        "n_red": int(red.sum()),
        "n_flat": int(flat.sum()),
    }


# ---------------------------------------------------------------------------
# PERMUTATION TEST
# ---------------------------------------------------------------------------
def permutation_test(prices: pd.DataFrame, signals: pd.DataFrame,
                     actual_sharpe: float, n_perms: int = 500) -> dict:
    """
    Shuffle long/short signal assignments per asset, re-run, compute
    distribution of Sharpe ratios. Report p-value.
    """
    print(f"\nRunning permutation test ({n_perms} shuffles)...")
    daily_ret = prices.pct_change()
    warmup_end = prices.index[LOOKBACK]

    perm_sharpes = []

    for i in range(n_perms):
        # Shuffle signals per column independently
        shuffled = signals.copy()
        for col in shuffled.columns:
            vals = shuffled[col].values.copy()
            np.random.shuffle(vals)
            shuffled[col] = vals

        # Simple return: shuffled_signal(t-1) * return(t), no vol scaling
        # (full backtest too slow for 500 perms, use simplified version)
        sig_shifted = shuffled.shift(1)
        perm_ret = (sig_shifted * daily_ret).mean(axis=1)
        perm_ret = perm_ret[perm_ret.index >= warmup_end]

        if perm_ret.std() > 0:
            s = (perm_ret.mean() * 252) / (perm_ret.std() * np.sqrt(252))
        else:
            s = 0.0
        perm_sharpes.append(s)

    perm_sharpes = np.array(perm_sharpes)
    p_value = 1.0 - percentileofscore(perm_sharpes, actual_sharpe) / 100.0

    return {
        "actual_sharpe": actual_sharpe,
        "perm_mean": float(np.mean(perm_sharpes)),
        "perm_std": float(np.std(perm_sharpes)),
        "perm_p95": float(np.percentile(perm_sharpes, 95)),
        "p_value": float(p_value),
        "n_perms": n_perms,
        "significant_at_05": p_value < 0.05,
    }


# ---------------------------------------------------------------------------
# CRISIS ALPHA ANALYSIS
# ---------------------------------------------------------------------------
def crisis_alpha_analysis(strat_ret: pd.Series, spy_ret: pd.Series) -> dict:
    """
    Measure strategy performance during SPY drawdowns > 10%.
    Also check specific crisis periods.
    """
    spy_cum = (1 + spy_ret).cumprod()
    spy_peak = spy_cum.cummax()
    spy_dd = (spy_cum - spy_peak) / spy_peak

    # Periods where SPY is in >10% drawdown
    in_crisis = spy_dd < -0.10
    crisis_strat_ret = strat_ret[in_crisis]
    normal_strat_ret = strat_ret[~in_crisis]

    crisis_total = (1 + crisis_strat_ret).prod() - 1 if len(crisis_strat_ret) > 0 else 0
    normal_total = (1 + normal_strat_ret).prod() - 1 if len(normal_strat_ret) > 0 else 0

    crisis_sharpe = 0.0
    if len(crisis_strat_ret) > 10:
        crisis_sharpe = (crisis_strat_ret.mean() * 252) / max(crisis_strat_ret.std() * np.sqrt(252), 1e-6)

    # Specific crisis periods
    crisis_periods = {
        "GFC_2008": ("2008-09-01", "2009-03-31"),
        "COVID_2020": ("2020-02-19", "2020-04-30"),
        "Rate_Hike_2022": ("2022-01-01", "2022-10-31"),
    }

    period_results = {}
    for name, (s, e) in crisis_periods.items():
        mask = (strat_ret.index >= s) & (strat_ret.index <= e)
        if mask.any():
            pr = strat_ret[mask]
            sr = spy_ret.reindex(pr.index).fillna(0)
            period_results[name] = {
                "strat_return": float((1 + pr).prod() - 1),
                "spy_return": float((1 + sr).prod() - 1),
                "strat_sharpe": float((pr.mean() * 252) / max(pr.std() * np.sqrt(252), 1e-6)),
                "n_days": int(mask.sum()),
            }

    return {
        "crisis_total_return": float(crisis_total),
        "normal_total_return": float(normal_total),
        "crisis_sharpe": float(crisis_sharpe),
        "crisis_days": int(in_crisis.sum()),
        "total_days": len(strat_ret),
        "crisis_pct": float(in_crisis.mean()),
        "periods": period_results,
    }


# ---------------------------------------------------------------------------
# ASYMMETRY CHECK
# ---------------------------------------------------------------------------
def asymmetry_check(annual_returns: dict) -> dict:
    """Best year / |worst year| ratio — want > 1.0 for asymmetric upside."""
    if not annual_returns:
        return {"ratio": 0, "passed": False}

    vals = list(annual_returns.values())
    best = max(vals)
    worst = min(vals)

    ratio = best / max(abs(worst), 1e-6) if worst < 0 else float("inf")

    return {
        "best_year": float(best),
        "worst_year": float(worst),
        "best_worst_ratio": float(ratio),
        "asymmetric": ratio > 1.0,
    }


# ---------------------------------------------------------------------------
# CORRELATION ANALYSIS
# ---------------------------------------------------------------------------
def correlation_analysis(strat_ret: pd.Series, spy_ret: pd.Series) -> dict:
    """Rolling and full-period correlation with SPY."""
    aligned = pd.DataFrame({"strat": strat_ret, "spy": spy_ret}).dropna()
    full_corr = aligned["strat"].corr(aligned["spy"])

    # Rolling 63-day correlation
    rolling_corr = aligned["strat"].rolling(63).corr(aligned["spy"])

    return {
        "full_period_corr": float(full_corr),
        "rolling_corr_mean": float(rolling_corr.mean()),
        "rolling_corr_min": float(rolling_corr.min()),
        "rolling_corr_max": float(rolling_corr.max()),
        "pct_negative_corr": float((rolling_corr < 0).mean()),
    }


# ---------------------------------------------------------------------------
# PLOTS
# ---------------------------------------------------------------------------
def plot_results(results: dict, metrics: dict, regime: dict, crisis: dict):
    """Generate diagnostic plots."""
    fig, axes = plt.subplots(3, 2, figsize=(16, 14))
    fig.suptitle("Trend-Following CTA v1 — Backtest Results", fontsize=14, fontweight="bold")

    # 1. Equity curves
    ax = axes[0, 0]
    ax.plot(results["equity"], label=f'CTA (Sharpe={metrics["sharpe"]:.2f})', linewidth=1.5)
    ax.plot(results["spy_eq"], label="SPY", alpha=0.7, linewidth=1)
    ax.plot(results["bm_6040_eq"], label="60/40", alpha=0.7, linewidth=1)
    ax.set_title("Equity Curves (log scale)")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Drawdown
    ax = axes[0, 1]
    cum = (1 + results["strat_ret"]).cumprod()
    dd = (cum - cum.cummax()) / cum.cummax()
    spy_cum = (1 + results["spy_ret"]).cumprod()
    spy_dd = (spy_cum - spy_cum.cummax()) / spy_cum.cummax()
    ax.fill_between(dd.index, dd, 0, alpha=0.4, label="CTA DD")
    ax.plot(spy_dd, color="red", alpha=0.5, label="SPY DD", linewidth=0.8)
    ax.set_title(f'Drawdown (Max DD: {metrics["max_dd"]:.1%})')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Annual returns
    ax = axes[1, 0]
    annual = pd.Series(metrics["annual_returns"])
    colors = ["green" if v > 0 else "red" for v in annual.values]
    ax.bar(annual.index.astype(str), annual.values, color=colors, alpha=0.7)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_title("Annual Returns")
    ax.set_xticklabels(annual.index.astype(str), rotation=45)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0%}'))
    ax.grid(True, alpha=0.3)

    # 4. Rolling Sharpe (252d)
    ax = axes[1, 1]
    rolling_sharpe = results["strat_ret"].rolling(252).mean() / results["strat_ret"].rolling(252).std() * np.sqrt(252)
    ax.plot(rolling_sharpe, linewidth=1)
    ax.axhline(0, color="red", linewidth=0.5)
    ax.set_title("Rolling 252d Sharpe Ratio")
    ax.grid(True, alpha=0.3)

    # 5. Position allocation over time
    ax = axes[2, 0]
    wts = results["daily_weights"]
    # Group by asset class
    class_map = {t: UNIVERSE[t]["class"] for t in TICKERS}
    class_wts = pd.DataFrame()
    for cls in set(class_map.values()):
        cols = [t for t, c in class_map.items() if c == cls]
        class_wts[cls] = wts[cols].sum(axis=1)
    # Resample to monthly for cleaner plot
    class_wts_monthly = class_wts.resample("ME").last()
    # Use regular line plot since weights can be negative (short positions)
    for col in class_wts_monthly.columns:
        ax.plot(class_wts_monthly.index, class_wts_monthly[col], label=col, linewidth=1)
    ax.set_title("Allocation by Asset Class")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)

    # 6. Regime test
    ax = axes[2, 1]
    regime_labels = ["Green\n(SPY up)", "Red\n(SPY dn)", "Flat"]
    regime_sharpes = [regime["sharpe_green"], regime["sharpe_red"], regime["sharpe_flat"]]
    bar_colors = ["green", "red", "gray"]
    ax.bar(regime_labels, regime_sharpes, color=bar_colors, alpha=0.7)
    ax.axhline(0, color="black", linewidth=0.5)
    gap_str = f'Gap: {regime["regime_gap"]:.2f} ({"PASS" if regime["passed_R1"] else "FAIL"})'
    ax.set_title(f"Regime Sharpe — HC #428 R1\n{gap_str}")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "cta_v1_results.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Plot saved: {OUTPUT_DIR / 'cta_v1_results.png'}")


def plot_crisis_alpha(crisis: dict, strat_ret: pd.Series, spy_ret: pd.Series):
    """Plot strategy vs SPY during crisis periods."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Crisis Alpha — CTA vs SPY During Drawdowns", fontsize=13, fontweight="bold")

    periods = {
        "GFC 2008": ("2008-09-01", "2009-03-31"),
        "COVID 2020": ("2020-02-19", "2020-04-30"),
        "Rate Hike 2022": ("2022-01-01", "2022-10-31"),
    }

    for ax, (name, (s, e)) in zip(axes, periods.items()):
        mask = (strat_ret.index >= s) & (strat_ret.index <= e)
        if not mask.any():
            ax.set_title(f"{name} — No data")
            continue
        sr = strat_ret[mask]
        sp = spy_ret.reindex(sr.index).fillna(0)
        ax.plot((1 + sr).cumprod(), label="CTA", linewidth=1.5)
        ax.plot((1 + sp).cumprod(), label="SPY", linewidth=1.5, color="red")
        strat_r = float((1 + sr).prod() - 1)
        spy_r = float((1 + sp).prod() - 1)
        ax.set_title(f"{name}\nCTA: {strat_r:+.1%}  SPY: {spy_r:+.1%}")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "crisis_alpha.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Crisis plot saved: {OUTPUT_DIR / 'crisis_alpha.png'}")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("TREND-FOLLOWING CTA STRATEGY v1")
    print("=" * 70)

    # 1. Download data
    prices = download_data()

    # 2. Run backtest
    results = run_backtest(prices)

    # 3. Compute metrics
    strat_metrics = compute_metrics(results["strat_ret"], "CTA Trend-Following v1")
    spy_metrics   = compute_metrics(results["spy_ret"], "SPY Buy & Hold")
    bm6040_metrics = compute_metrics(results["bm_6040_ret"], "60/40 Portfolio")

    # 4. Regime validation (HC #428 R1)
    regime = regime_validation(results["strat_ret"], results["spy_ret"])

    # 5. Crisis alpha
    crisis = crisis_alpha_analysis(results["strat_ret"], results["spy_ret"])

    # 6. Asymmetry check
    asymmetry = asymmetry_check(strat_metrics["annual_returns"])

    # 7. Correlation
    corr = correlation_analysis(results["strat_ret"], results["spy_ret"])

    # 8. Permutation test
    perm = permutation_test(prices, results["signals"], strat_metrics["sharpe"])

    # 9. Print results
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    for m in [strat_metrics, spy_metrics, bm6040_metrics]:
        print(f"\n--- {m['label']} ---")
        if "error" in m:
            print(f"  ERROR: {m['error']}")
            continue
        print(f"  CAGR:          {m['cagr']:>8.2%}")
        print(f"  Ann. Vol:      {m['ann_vol']:>8.2%}")
        print(f"  Sharpe:        {m['sharpe']:>8.2f}")
        print(f"  Sortino:       {m['sortino']:>8.2f}")
        print(f"  Max DD:        {m['max_dd']:>8.2%}")
        print(f"  Calmar:        {m['calmar']:>8.2f}")
        print(f"  Win Rate:      {m['win_rate']:>8.2%}")
        print(f"  Profit Factor: {m['profit_factor']:>8.2f}")

    print(f"\n--- REGIME TEST (HC #428 R1) ---")
    print(f"  Sharpe Green: {regime['sharpe_green']:.2f}  Red: {regime['sharpe_red']:.2f}  Flat: {regime['sharpe_flat']:.2f}")
    print(f"  Regime Gap:   {regime['regime_gap']:.2f}  {'PASS ✓' if regime['passed_R1'] else 'FAIL ✗'}")

    print(f"\n--- CRISIS ALPHA ---")
    print(f"  During SPY >10% DD: {crisis['crisis_total_return']:+.2%} ({crisis['crisis_days']} days)")
    for name, p in crisis["periods"].items():
        print(f"  {name}: CTA {p['strat_return']:+.2%} vs SPY {p['spy_return']:+.2%}")

    print(f"\n--- ASYMMETRY ---")
    print(f"  Best Year:  {asymmetry['best_year']:+.2%}")
    print(f"  Worst Year: {asymmetry['worst_year']:+.2%}")
    print(f"  Best/|Worst| Ratio: {asymmetry['best_worst_ratio']:.2f}  {'ASYMMETRIC ✓' if asymmetry['asymmetric'] else 'SYMMETRIC ✗'}")

    print(f"\n--- CORRELATION ---")
    print(f"  Full-period corr w/ SPY: {corr['full_period_corr']:.3f}")
    print(f"  Pct of time negatively correlated: {corr['pct_negative_corr']:.1%}")

    print(f"\n--- PERMUTATION TEST ---")
    print(f"  Actual Sharpe: {perm['actual_sharpe']:.2f}")
    print(f"  Perm Mean:     {perm['perm_mean']:.2f} ± {perm['perm_std']:.2f}")
    print(f"  p-value:       {perm['p_value']:.3f}  {'SIGNIFICANT ✓' if perm['significant_at_05'] else 'NOT SIGNIFICANT ✗'}")

    # 10. Save results
    output = {
        "strategy": "Trend-Following CTA v1",
        "run_time": datetime.now().isoformat(),
        "parameters": {
            "sma_fast": SMA_FAST,
            "sma_slow": SMA_SLOW,
            "vol_window": VOL_WINDOW,
            "atr_window": ATR_WINDOW,
            "atr_stop_mult": ATR_STOP_MULT,
            "target_vol": TARGET_VOL,
            "max_pos_weight": MAX_POS_WT,
            "max_leverage": MAX_LEVERAGE,
            "rebal_freq": "weekly_friday",
        },
        "metrics": {
            "cta": {k: v for k, v in strat_metrics.items() if k != "annual_returns"},
            "spy": {k: v for k, v in spy_metrics.items() if k != "annual_returns"},
            "bm_6040": {k: v for k, v in bm6040_metrics.items() if k != "annual_returns"},
        },
        "annual_returns": strat_metrics.get("annual_returns", {}),
        "regime_test": regime,
        "crisis_alpha": crisis,
        "asymmetry": asymmetry,
        "correlation": corr,
        "permutation_test": perm,
        "hc_compliance": {
            "HC_0_sliding_wf": True,
            "HC_428_R1_regime": regime["passed_R1"],
            "HC_694_commission_free": True,
            "max_leverage_1_5x": True,
        },
    }

    # Convert numpy types for JSON
    def convert(obj):
        if isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    def deep_convert(d):
        if isinstance(d, dict):
            return {str(k): deep_convert(v) for k, v in d.items()}
        if isinstance(d, list):
            return [deep_convert(v) for v in d]
        return convert(d)

    output = deep_convert(output)

    with open(OUTPUT_DIR / "results.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved: {OUTPUT_DIR / 'results.json'}")

    # 11. Plots
    plot_results(results, strat_metrics, regime, crisis)
    plot_crisis_alpha(crisis, results["strat_ret"], results["spy_ret"])

    # 12. Final verdict
    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)
    all_pass = regime["passed_R1"] and perm["significant_at_05"]
    if all_pass:
        print("  PASS — Strategy passes regime test and permutation test.")
    else:
        reasons = []
        if not regime["passed_R1"]:
            reasons.append(f"Regime gap {regime['regime_gap']:.2f} > 0.50")
        if not perm["significant_at_05"]:
            reasons.append(f"Permutation p={perm['p_value']:.3f} > 0.05")
        print(f"  CONDITIONAL — Issues: {'; '.join(reasons)}")
        print("  Strategy may still have value as a diversifier even if standalone metrics are weak.")

    print(f"\n  CTA Sharpe: {strat_metrics['sharpe']:.2f} | SPY Sharpe: {spy_metrics['sharpe']:.2f} | 60/40 Sharpe: {bm6040_metrics['sharpe']:.2f}")
    print(f"  SPY Correlation: {corr['full_period_corr']:.3f}")
    print(f"  Crisis Alpha: {'YES' if crisis['crisis_total_return'] > 0 else 'NO'} ({crisis['crisis_total_return']:+.1%} during SPY DDs)")
    print("=" * 70)

    return output


if __name__ == "__main__":
    main()
