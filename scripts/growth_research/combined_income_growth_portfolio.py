#!/usr/bin/env python3
"""
Combined Income + Growth Portfolio Backtest
============================================
Three-book portfolio:
  CORE  (70%): v4.4 Macro-Scaled timing (UPRO/SPY/SHY/GLD)
  INCOME(15%): VRP harvesting — short vol via SVXY when contango>7% & VIX pctile<50th
  SPIKE (15%): Cash (SHY) normally, deploy UPRO when VIX>30

HC compliance:
  HC #0  : Sliding window (trailing lookback for signals)
  HC #69 : Risk-adjusted metrics primary (Sharpe, Sortino, PF, WR)
  HC #694: Commission-free (Robinhood ETFs)
  HC #705: Adversarial validation (permutation + sub-period)
  HC #713: Fixed $100K capital, NO DCA
  HC #428: Regime-agnostic validation (R1)
"""

import json
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

sys.stdout.reconfigure(line_buffering=True)
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
OUTPUT_DIR = Path("/home/jupiter/Lvl3Quant/output/combined_income_growth_portfolio")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

START = "2010-01-01"
END = "2026-07-17"
INITIAL_CAPITAL = 100_000
WARMUP = 260  # ~1 year for signal computation

# Allocation weights
W_CORE = 0.70
W_INCOME = 0.15
W_SPIKE = 0.15

# VRP harvesting params
CONTANGO_ENTRY = 7.0   # Enter when VIX3M/VIX contango > 7%
VIX_PCTILE_MAX_VRP = 50  # Only harvest VRP when VIX pctile < 50th

# Spike buying params
VIX_SPIKE_THRESHOLD = 30  # Buy UPRO when VIX > 30

# ---------------------------------------------------------------------------
# DATA
# ---------------------------------------------------------------------------
def download_data():
    """Download all required data."""
    tickers = ["SPY", "UPRO", "SHY", "GLD", "SVXY", "^VIX"]
    print("Downloading price data...")
    raw = yf.download(tickers, start=START, end=END, auto_adjust=True, progress=False)

    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"]
    else:
        prices = raw

    # Flatten MultiIndex columns if needed
    if hasattr(prices.columns, "droplevel"):
        try:
            prices.columns = prices.columns.droplevel(1)
        except Exception:
            pass

    prices = prices.rename(columns={"^VIX": "VIX"})

    # Download VIX3M for contango calculation
    print("Downloading VIX3M...")
    try:
        vix3m_raw = yf.download("^VIX3M", start=START, end=END, auto_adjust=True, progress=False)
        if isinstance(vix3m_raw.columns, pd.MultiIndex):
            vix3m = vix3m_raw["Close"].squeeze()
        else:
            vix3m = vix3m_raw["Close"] if "Close" in vix3m_raw.columns else vix3m_raw.squeeze()
        # Handle multi-level column from yfinance
        if hasattr(vix3m, 'columns'):
            vix3m = vix3m.iloc[:, 0]
        prices["VIX3M"] = vix3m
    except Exception as e:
        print(f"VIX3M download failed ({e}), computing proxy from VIX...")
        # Use VIX 63d mean as rough VIX3M proxy (futures tend toward mean)
        prices["VIX3M"] = prices["VIX"].rolling(63).mean()

    prices = prices.ffill()
    print(f"Data: {prices.index[0].date()} to {prices.index[-1].date()}, {len(prices)} days")
    print(f"SVXY available from: {prices['SVXY'].first_valid_index()}")
    return prices


# ---------------------------------------------------------------------------
# SIGNALS
# ---------------------------------------------------------------------------
def compute_signals(prices):
    """Compute all signals for the three books."""
    spy = prices["SPY"]
    vix = prices["VIX"]
    spy_ret = spy.pct_change()
    sig = {}

    # -- CORE book signals (v4.4) --
    sig["mom_5d"] = spy.pct_change(5)
    delta = spy_ret.copy()
    gain = delta.where(delta > 0, 0).rolling(10).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(10).mean()
    rs = gain / loss.replace(0, np.nan)
    sig["rsi_10"] = 100 - (100 / (1 + rs))
    sig["sma_20"] = spy.rolling(20).mean()
    sig["sma_50"] = spy.rolling(50).mean()
    sig["sma_200"] = spy.rolling(200).mean()
    sig["sma_200_slope"] = sig["sma_200"].pct_change(20)
    sig["vol_21d"] = spy_ret.rolling(21).std() * np.sqrt(252) * 100
    sig["vol_63d"] = spy_ret.rolling(63).std() * np.sqrt(252) * 100
    sig["vol_63d_trend"] = sig["vol_63d"] - sig["vol_63d"].rolling(21).mean()
    sig["vix_pctile_63"] = vix.rolling(63).apply(
        lambda x: (x.iloc[-1] > x.iloc[:-1]).sum() / (len(x) - 1) * 100, raw=False
    )

    # -- INCOME book signals --
    sig["vix"] = vix
    sig["vix3m"] = prices["VIX3M"]
    # Contango = (VIX3M - VIX) / VIX * 100
    sig["contango_pct"] = (prices["VIX3M"] - vix) / vix * 100

    return sig


def confluence_score(sig, i):
    """6-factor confluence score (0-3) for v4.4."""
    s = 0.0
    m = sig["mom_5d"].iloc[i]
    r = sig["rsi_10"].iloc[i]
    s20 = sig["sma_20"].iloc[i]
    s50 = sig["sma_50"].iloc[i]
    v21 = sig["vol_21d"].iloc[i]
    slope = sig["sma_200_slope"].iloc[i]
    vt = sig["vol_63d_trend"].iloc[i]
    if not np.isnan(m) and m > 0: s += 0.5
    if not np.isnan(r) and r > 50: s += 0.5
    if not np.isnan(s20) and not np.isnan(s50) and s20 > s50: s += 0.5
    if not np.isnan(v21) and v21 < 15: s += 0.5
    if not np.isnan(slope) and slope > 0: s += 0.5
    if not np.isnan(vt) and vt < 0: s += 0.5
    return s


# ---------------------------------------------------------------------------
# BOOK DECISION FUNCTIONS
# ---------------------------------------------------------------------------
def core_book_decision(sig, i, date, in_upro):
    """
    v4.4 Macro-Scaled timing.
    Returns (ticker, fractional_weights_dict, new_in_upro).
    When growth signal is ON: 67% UPRO / 33% SPY.
    VIX pctile > 80 -> SHY (defensive).
    VIX pctile 20-80 with weak confluence -> GLD.
    """
    s20 = sig["sma_20"].iloc[i]
    s200 = sig["sma_200"].iloc[i]

    # September effect
    if date.month == 9:
        return {"SPY": 1.0}, False

    # Death cross guard
    if not np.isnan(s20) and not np.isnan(s200) and s20 < s200:
        return {"SPY": 1.0}, False

    pctile = sig["vix_pctile_63"].iloc[i]
    if np.isnan(pctile):
        return {"SPY": 1.0}, False

    # High fear -> defensive (SHY per user spec)
    if pctile > 80:
        return {"SHY": 1.0}, False

    # Adaptive confluence thresholds
    if pctile > 60:
        entry, exit_t = 3.0, 2.5
    elif pctile < 30:
        entry, exit_t = 2.0, 1.5
    else:
        entry, exit_t = 2.5, 2.0

    if pctile > 20:
        entry = max(entry, 2.5)

    score = confluence_score(sig, i)

    if in_upro:
        if score < exit_t:
            return {"SPY": 1.0}, False
        # Low VIX pctile: fractional UPRO/SPY
        if pctile < 20:
            return {"UPRO": 0.67, "SPY": 0.33}, True
        else:
            return {"UPRO": 0.67, "SPY": 0.33}, True
    else:
        if score >= entry:
            if pctile < 20:
                return {"UPRO": 0.67, "SPY": 0.33}, True
            else:
                return {"UPRO": 0.67, "SPY": 0.33}, True
        # Mid-range fear -> GLD
        if pctile > 50:
            return {"GLD": 1.0}, False
        return {"SPY": 1.0}, False


def income_book_decision(sig, i, svxy_available):
    """
    VRP harvesting: short vol via SVXY when contango > 7% AND VIX pctile < 50th.
    Otherwise SHY.
    """
    contango = sig["contango_pct"].iloc[i]
    pctile = sig["vix_pctile_63"].iloc[i]

    if np.isnan(contango) or np.isnan(pctile):
        return {"SHY": 1.0}

    if svxy_available and contango > CONTANGO_ENTRY and pctile < VIX_PCTILE_MAX_VRP:
        return {"SVXY": 1.0}

    return {"SHY": 1.0}


def spike_book_decision(sig, i):
    """
    Spike reserve: deploy UPRO when VIX > 30 (crisis buying).
    Otherwise SHY (cash reserve).
    """
    vix = sig["vix"].iloc[i]
    if np.isnan(vix):
        return {"SHY": 1.0}

    if vix > VIX_SPIKE_THRESHOLD:
        return {"UPRO": 1.0}

    return {"SHY": 1.0}


# ---------------------------------------------------------------------------
# SIMULATION ENGINE
# ---------------------------------------------------------------------------
def simulate_combined(prices, sig, warmup=WARMUP):
    """
    Simulate the combined 3-book portfolio.
    Signal at close T -> trade T+1 (next-day execution).
    Returns daily equity series.
    """
    dates = prices.index[warmup:]
    rets = prices.pct_change()
    svxy_start = prices["SVXY"].first_valid_index()

    equity = INITIAL_CAPITAL
    equity_series = []
    holdings_log = []

    in_upro_core = False

    for idx in range(warmup, len(prices) - 1):
        date = prices.index[idx]
        next_date = prices.index[idx + 1]
        svxy_avail = svxy_start is not None and date >= svxy_start

        # --- Signal at close T ---
        core_alloc, in_upro_core = core_book_decision(sig, idx, date, in_upro_core)
        income_alloc = income_book_decision(sig, idx, svxy_avail)
        spike_alloc = spike_book_decision(sig, idx)

        # --- Combine into portfolio weights ---
        combined = {}
        for ticker, w in core_alloc.items():
            combined[ticker] = combined.get(ticker, 0) + W_CORE * w
        for ticker, w in income_alloc.items():
            combined[ticker] = combined.get(ticker, 0) + W_INCOME * w
        for ticker, w in spike_alloc.items():
            combined[ticker] = combined.get(ticker, 0) + W_SPIKE * w

        # --- Apply T+1 returns ---
        port_ret = 0.0
        for ticker, w in combined.items():
            if ticker in rets.columns:
                r = rets[ticker].iloc[idx + 1]
                if not np.isnan(r):
                    port_ret += w * r
            # If ticker not available (SVXY before inception), treat as SHY
            elif ticker == "SVXY" and not svxy_avail:
                r = rets["SHY"].iloc[idx + 1]
                if not np.isnan(r):
                    port_ret += w * r

        equity *= (1 + port_ret)
        equity_series.append({"date": next_date, "equity": equity, "return": port_ret})
        holdings_log.append({
            "date": next_date,
            "core": core_alloc,
            "income": income_alloc,
            "spike": spike_alloc,
            "in_upro": in_upro_core,
        })

    eq_df = pd.DataFrame(equity_series).set_index("date")
    return eq_df, holdings_log


def simulate_benchmark(prices, ticker, warmup=WARMUP):
    """Buy-and-hold benchmark."""
    rets = prices[ticker].pct_change()
    equity = INITIAL_CAPITAL
    series = []
    for idx in range(warmup, len(prices) - 1):
        next_date = prices.index[idx + 1]
        r = rets.iloc[idx + 1]
        if not np.isnan(r):
            equity *= (1 + r)
        series.append({"date": next_date, "equity": equity, "return": r if not np.isnan(r) else 0})
    return pd.DataFrame(series).set_index("date")


def simulate_v44_only(prices, sig, warmup=WARMUP):
    """v4.4-only benchmark (100% allocated to core book logic)."""
    rets = prices.pct_change()
    equity = INITIAL_CAPITAL
    series = []
    in_upro = False

    for idx in range(warmup, len(prices) - 1):
        date = prices.index[idx]
        next_date = prices.index[idx + 1]
        alloc, in_upro = core_book_decision(sig, idx, date, in_upro)

        port_ret = 0.0
        for ticker, w in alloc.items():
            if ticker in rets.columns:
                r = rets[ticker].iloc[idx + 1]
                if not np.isnan(r):
                    port_ret += w * r

        equity *= (1 + port_ret)
        series.append({"date": next_date, "equity": equity, "return": port_ret})

    return pd.DataFrame(series).set_index("date")


# ---------------------------------------------------------------------------
# METRICS
# ---------------------------------------------------------------------------
def compute_metrics(eq_df, name="Strategy"):
    """Compute risk-adjusted performance metrics."""
    rets = eq_df["return"]
    n_years = len(rets) / 252

    total_ret = eq_df["equity"].iloc[-1] / INITIAL_CAPITAL - 1
    cagr = (eq_df["equity"].iloc[-1] / INITIAL_CAPITAL) ** (1 / n_years) - 1

    ann_ret = rets.mean() * 252
    ann_vol = rets.std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0

    downside = rets[rets < 0].std() * np.sqrt(252)
    sortino = ann_ret / downside if downside > 0 else 0

    # Max drawdown
    cum = (1 + rets).cumprod()
    peak = cum.cummax()
    dd = (cum - peak) / peak
    max_dd = dd.min()

    calmar = cagr / abs(max_dd) if max_dd != 0 else 0

    # Win rate
    win_rate = (rets > 0).sum() / (rets != 0).sum() * 100 if (rets != 0).sum() > 0 else 0

    # Profit factor
    gross_profit = rets[rets > 0].sum()
    gross_loss = abs(rets[rets < 0].sum())
    pf = gross_profit / gross_loss if gross_loss > 0 else np.inf

    return {
        "Name": name,
        "CAGR": f"{cagr:.1%}",
        "Total Return": f"{total_ret:.1%}",
        "Sharpe": f"{sharpe:.2f}",
        "Sortino": f"{sortino:.2f}",
        "MaxDD": f"{max_dd:.1%}",
        "Calmar": f"{calmar:.2f}",
        "Ann Vol": f"{ann_vol:.1%}",
        "Win Rate": f"{win_rate:.1f}%",
        "Profit Factor": f"{pf:.2f}",
        "Final Equity": f"${eq_df['equity'].iloc[-1]:,.0f}",
    }


def worst_drawdowns(eq_df, n=5):
    """Find the N worst drawdown periods."""
    cum = (1 + eq_df["return"]).cumprod()
    peak = cum.cummax()
    dd = (cum - peak) / peak

    drawdowns = []
    in_dd = False
    dd_start = None

    for i in range(len(dd)):
        if dd.iloc[i] < -0.01 and not in_dd:
            in_dd = True
            dd_start = dd.index[i]
        elif dd.iloc[i] >= 0 and in_dd:
            in_dd = False
            dd_period = dd.loc[dd_start:dd.index[i]]
            worst = dd_period.min()
            worst_date = dd_period.idxmin()
            drawdowns.append({
                "Start": dd_start.strftime("%Y-%m-%d"),
                "Trough": worst_date.strftime("%Y-%m-%d"),
                "End": dd.index[i].strftime("%Y-%m-%d"),
                "Depth": f"{worst:.1%}",
                "Duration (days)": (dd.index[i] - dd_start).days,
            })

    # Handle ongoing drawdown
    if in_dd and dd_start is not None:
        dd_period = dd.loc[dd_start:]
        worst = dd_period.min()
        worst_date = dd_period.idxmin()
        drawdowns.append({
            "Start": dd_start.strftime("%Y-%m-%d"),
            "Trough": worst_date.strftime("%Y-%m-%d"),
            "End": "ongoing",
            "Depth": f"{worst:.1%}",
            "Duration (days)": (dd.index[-1] - dd_start).days,
        })

    drawdowns.sort(key=lambda x: float(x["Depth"].replace("%", "")) / 100)
    return drawdowns[:n]


def yearly_returns(eq_df):
    """Calculate yearly returns."""
    eq_df_copy = eq_df.copy()
    eq_df_copy["year"] = eq_df_copy.index.year
    yearly = {}
    for year in sorted(eq_df_copy["year"].unique()):
        mask = eq_df_copy["year"] == year
        yr_rets = eq_df_copy.loc[mask, "return"]
        yearly[year] = (1 + yr_rets).prod() - 1
    return yearly


def income_contribution_analysis(holdings_log, eq_df, prices, sig):
    """Analyze how much the income and spike books contribute."""
    total_days = len(holdings_log)

    # Count active days for each book
    vrp_active = sum(1 for h in holdings_log if "SVXY" in h["income"])
    spike_active = sum(1 for h in holdings_log if "UPRO" in h["spike"])
    upro_core = sum(1 for h in holdings_log if "UPRO" in h["core"])

    # Compute attribution by simulating each book in isolation
    rets = prices.pct_change()
    svxy_start = prices["SVXY"].first_valid_index()

    income_only_rets = []
    spike_only_rets = []

    for j, h in enumerate(holdings_log):
        date = h["date"]
        idx = prices.index.get_loc(date)

        # Income book return
        inc_ret = 0.0
        for ticker, w in h["income"].items():
            if ticker in rets.columns:
                r = rets[ticker].iloc[idx]
                if not np.isnan(r):
                    inc_ret += w * r
        income_only_rets.append(inc_ret)

        # Spike book return
        sp_ret = 0.0
        for ticker, w in h["spike"].items():
            if ticker in rets.columns:
                r = rets[ticker].iloc[idx]
                if not np.isnan(r):
                    sp_ret += w * r
        spike_only_rets.append(sp_ret)

    income_ann = np.mean(income_only_rets) * 252
    spike_ann = np.mean(spike_only_rets) * 252

    return {
        "VRP Active Days": f"{vrp_active} ({vrp_active/total_days*100:.1f}%)",
        "Spike Active Days": f"{spike_active} ({spike_active/total_days*100:.1f}%)",
        "Core UPRO Days": f"{upro_core} ({upro_core/total_days*100:.1f}%)",
        "Income Book Ann Return (isolated)": f"{income_ann:.2%}",
        "Spike Book Ann Return (isolated)": f"{spike_ann:.2%}",
        "Income Weight": f"{W_INCOME:.0%}",
        "Spike Weight": f"{W_SPIKE:.0%}",
        "Weighted Income Contribution": f"{income_ann * W_INCOME:.2%}",
        "Weighted Spike Contribution": f"{spike_ann * W_SPIKE:.2%}",
    }


# ---------------------------------------------------------------------------
# ADVERSARIAL VALIDATION (HC #705)
# ---------------------------------------------------------------------------
def permutation_test(prices, sig, eq_df, n_perms=100):
    """
    Permutation test: randomize VRP and spike-buying decisions.
    Keep core book intact. Shuffle income/spike signals.
    """
    print(f"\nRunning permutation test ({n_perms} shuffles)...")
    real_sharpe = eq_df["return"].mean() / eq_df["return"].std() * np.sqrt(252)

    rets = prices.pct_change()
    svxy_start = prices["SVXY"].first_valid_index()
    rng = np.random.default_rng(42)
    perm_sharpes = []

    # Pre-compute core decisions (these stay fixed)
    core_decisions = []
    in_upro = False
    for idx in range(WARMUP, len(prices) - 1):
        date = prices.index[idx]
        alloc, in_upro = core_book_decision(sig, idx, date, in_upro)
        core_decisions.append(alloc)

    # Pre-compute real income and spike signals
    n_days = len(core_decisions)
    income_signals = []
    spike_signals = []
    for idx in range(WARMUP, len(prices) - 1):
        date = prices.index[idx]
        svxy_avail = svxy_start is not None and date >= svxy_start
        income_signals.append(income_book_decision(sig, idx, svxy_avail))
        spike_signals.append(spike_book_decision(sig, idx))

    for p in range(n_perms):
        # Shuffle income and spike decisions independently
        shuf_income = [income_signals[j] for j in rng.permutation(n_days)]
        shuf_spike = [spike_signals[j] for j in rng.permutation(n_days)]

        equity = INITIAL_CAPITAL
        perm_rets = []
        for j in range(n_days):
            idx = WARMUP + j + 1
            if idx >= len(prices):
                break

            combined = {}
            for ticker, w in core_decisions[j].items():
                combined[ticker] = combined.get(ticker, 0) + W_CORE * w
            for ticker, w in shuf_income[j].items():
                combined[ticker] = combined.get(ticker, 0) + W_INCOME * w
            for ticker, w in shuf_spike[j].items():
                combined[ticker] = combined.get(ticker, 0) + W_SPIKE * w

            port_ret = 0.0
            for ticker, w in combined.items():
                if ticker in rets.columns:
                    r = rets[ticker].iloc[idx]
                    if not np.isnan(r):
                        port_ret += w * r
            perm_rets.append(port_ret)

        perm_rets = np.array(perm_rets)
        perm_sharpe = perm_rets.mean() / perm_rets.std() * np.sqrt(252) if perm_rets.std() > 0 else 0
        perm_sharpes.append(perm_sharpe)

    perm_sharpes = np.array(perm_sharpes)
    p_value = (perm_sharpes >= real_sharpe).sum() / n_perms

    return {
        "Real Sharpe": f"{real_sharpe:.3f}",
        "Mean Permuted Sharpe": f"{perm_sharpes.mean():.3f}",
        "Std Permuted Sharpe": f"{perm_sharpes.std():.3f}",
        "p-value": f"{p_value:.3f}",
        "PASS": p_value < 0.05,
        "Note": "Permutes income+spike decisions only (core intact)",
    }


def sub_period_consistency(eq_df, n_blocks=3):
    """Split into n_blocks equal time blocks, compute Sharpe for each, check CV < 0.6."""
    rets = eq_df["return"]
    block_size = len(rets) // n_blocks
    sharpes = []

    for b in range(n_blocks):
        start = b * block_size
        end = (b + 1) * block_size if b < n_blocks - 1 else len(rets)
        block_rets = rets.iloc[start:end]
        s = block_rets.mean() / block_rets.std() * np.sqrt(252) if block_rets.std() > 0 else 0
        sharpes.append(s)

    mean_s = np.mean(sharpes)
    std_s = np.std(sharpes)
    cv = std_s / abs(mean_s) if mean_s != 0 else np.inf

    block_dates = []
    for b in range(n_blocks):
        start = b * block_size
        end = (b + 1) * block_size if b < n_blocks - 1 else len(rets)
        block_dates.append(f"{rets.index[start].strftime('%Y-%m')} to {rets.index[end-1].strftime('%Y-%m')}")

    return {
        "Block Sharpes": {block_dates[i]: f"{sharpes[i]:.3f}" for i in range(n_blocks)},
        "Mean Sharpe": f"{mean_s:.3f}",
        "Std Sharpe": f"{std_s:.3f}",
        "CV": f"{cv:.3f}",
        "PASS (CV < 0.6)": cv < 0.6,
    }


# ---------------------------------------------------------------------------
# REGIME ANALYSIS (HC #428 R1)
# ---------------------------------------------------------------------------
def regime_analysis(eq_df, prices):
    """Stratified analysis by market regime (green/red/flat days based on SPY)."""
    spy_ret = prices["SPY"].pct_change()
    aligned = eq_df.join(spy_ret.rename("spy_ret"), how="inner")

    green = aligned[aligned["spy_ret"] > 0.002]
    red = aligned[aligned["spy_ret"] < -0.002]
    flat = aligned[(aligned["spy_ret"] >= -0.002) & (aligned["spy_ret"] <= 0.002)]

    def block_sharpe(r):
        return r.mean() / r.std() * np.sqrt(252) if len(r) > 10 and r.std() > 0 else 0

    s_green = block_sharpe(green["return"])
    s_red = block_sharpe(red["return"])
    s_flat = block_sharpe(flat["return"])

    imbalance = abs(s_green - s_red) / max(abs(s_green), abs(s_red), 0.001)

    return {
        "Green Days Sharpe": f"{s_green:.3f} (n={len(green)})",
        "Red Days Sharpe": f"{s_red:.3f} (n={len(red)})",
        "Flat Days Sharpe": f"{s_flat:.3f} (n={len(flat)})",
        "Regime Imbalance": f"{imbalance:.3f}",
        "PASS (imbalance < 0.50)": imbalance < 0.50,
    }


# ---------------------------------------------------------------------------
# PLOTTING
# ---------------------------------------------------------------------------
def plot_equity_curves(results_dict, output_path):
    """Plot equity curves for all strategies."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={"height_ratios": [3, 1]})

    # Equity curves
    ax = axes[0]
    colors = {"Combined Portfolio": "#2196F3", "v4.4 Only": "#FF9800",
              "SPY B&H": "#9E9E9E", "UPRO B&H": "#E91E63"}
    for name, eq_df in results_dict.items():
        c = colors.get(name, "#000000")
        ax.plot(eq_df.index, eq_df["equity"], label=name, color=c, linewidth=1.5)

    ax.set_ylabel("Portfolio Value ($)")
    ax.set_title("Combined Income + Growth Portfolio vs Benchmarks")
    ax.legend(loc="upper left")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)

    # Drawdown
    ax2 = axes[1]
    for name, eq_df in results_dict.items():
        cum = (1 + eq_df["return"]).cumprod()
        dd = (cum - cum.cummax()) / cum.cummax()
        c = colors.get(name, "#000000")
        ax2.fill_between(dd.index, dd.values, 0, alpha=0.3, color=c, label=name)

    ax2.set_ylabel("Drawdown")
    ax2.set_xlabel("Date")
    ax2.legend(loc="lower left", fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved equity curve plot to {output_path}")


def plot_allocation_timeline(holdings_log, output_path):
    """Plot allocation over time showing which books are active."""
    dates = [h["date"] for h in holdings_log]
    core_upro = [1 if "UPRO" in h["core"] else 0 for h in holdings_log]
    income_svxy = [1 if "SVXY" in h["income"] else 0 for h in holdings_log]
    spike_upro = [1 if "UPRO" in h["spike"] else 0 for h in holdings_log]

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.fill_between(dates, 0, core_upro, alpha=0.5, label="Core: UPRO", color="#2196F3")
    ax.fill_between(dates, [1.1]*len(dates), [1.1 + s for s in income_svxy],
                    alpha=0.5, label="Income: SVXY (VRP)", color="#4CAF50")
    ax.fill_between(dates, [2.2]*len(dates), [2.2 + s for s in spike_upro],
                    alpha=0.5, label="Spike: UPRO Buy", color="#F44336")

    ax.set_yticks([0.5, 1.6, 2.7])
    ax.set_yticklabels(["Core Book", "Income Book", "Spike Book"])
    ax.set_title("Book Allocation Timeline (shaded = active/risk-on)")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3, axis="x")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved allocation timeline to {output_path}")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("COMBINED INCOME + GROWTH PORTFOLIO BACKTEST")
    print(f"Run: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 70)

    # 1. Download data
    prices = download_data()
    sig = compute_signals(prices)

    # 2. Run simulations
    print("\nSimulating combined portfolio...")
    combined_eq, holdings_log = simulate_combined(prices, sig)

    print("Simulating v4.4 only...")
    v44_eq = simulate_v44_only(prices, sig)

    print("Simulating benchmarks...")
    spy_eq = simulate_benchmark(prices, "SPY")
    upro_eq = simulate_benchmark(prices, "UPRO")

    # Align all to same date range
    common_start = max(combined_eq.index[0], v44_eq.index[0], spy_eq.index[0], upro_eq.index[0])
    common_end = min(combined_eq.index[-1], v44_eq.index[-1], spy_eq.index[-1], upro_eq.index[-1])

    combined_eq = combined_eq.loc[common_start:common_end]
    v44_eq = v44_eq.loc[common_start:common_end]
    spy_eq = spy_eq.loc[common_start:common_end]
    upro_eq = upro_eq.loc[common_start:common_end]

    # Rescale to common start
    for df in [combined_eq, v44_eq, spy_eq, upro_eq]:
        scale = INITIAL_CAPITAL / df["equity"].iloc[0]
        df["equity"] *= scale

    # 3. Compute metrics
    print("\n" + "=" * 70)
    print("PERFORMANCE COMPARISON")
    print("=" * 70)

    metrics_list = [
        compute_metrics(combined_eq, "Combined Portfolio"),
        compute_metrics(v44_eq, "v4.4 Only"),
        compute_metrics(spy_eq, "SPY Buy & Hold"),
        compute_metrics(upro_eq, "UPRO Buy & Hold"),
    ]

    metrics_df = pd.DataFrame(metrics_list).set_index("Name")
    print(metrics_df.to_string())

    # 4. Yearly returns
    print("\n" + "=" * 70)
    print("YEARLY RETURNS")
    print("=" * 70)

    yr_combined = yearly_returns(combined_eq)
    yr_v44 = yearly_returns(v44_eq)
    yr_spy = yearly_returns(spy_eq)
    yr_upro = yearly_returns(upro_eq)

    yr_df = pd.DataFrame({
        "Combined": {y: f"{r:.1%}" for y, r in yr_combined.items()},
        "v4.4 Only": {y: f"{r:.1%}" for y, r in yr_v44.items()},
        "SPY": {y: f"{r:.1%}" for y, r in yr_spy.items()},
        "UPRO": {y: f"{r:.1%}" for y, r in yr_upro.items()},
    })
    print(yr_df.to_string())

    # 5. Worst drawdowns
    print("\n" + "=" * 70)
    print("WORST DRAWDOWNS (Combined Portfolio)")
    print("=" * 70)
    wdd = worst_drawdowns(combined_eq, n=5)
    for i, dd in enumerate(wdd, 1):
        print(f"  #{i}: {dd['Depth']} from {dd['Start']} to {dd['End']} "
              f"(trough {dd['Trough']}, {dd['Duration (days)']}d)")

    # 6. Income contribution
    print("\n" + "=" * 70)
    print("INCOME CONTRIBUTION ANALYSIS")
    print("=" * 70)
    income_contrib = income_contribution_analysis(holdings_log, combined_eq, prices, sig)
    for k, v in income_contrib.items():
        print(f"  {k}: {v}")

    # 7. Adversarial validation
    print("\n" + "=" * 70)
    print("ADVERSARIAL VALIDATION (HC #705)")
    print("=" * 70)

    perm = permutation_test(prices, sig, combined_eq, n_perms=100)
    print("\nPermutation Test (income+spike signals shuffled):")
    for k, v in perm.items():
        print(f"  {k}: {v}")

    sub = sub_period_consistency(combined_eq, n_blocks=3)
    print("\nSub-Period Consistency (3 blocks):")
    for k, v in sub.items():
        if isinstance(v, dict):
            print(f"  {k}:")
            for kk, vv in v.items():
                print(f"    {kk}: {vv}")
        else:
            print(f"  {k}: {v}")

    regime = regime_analysis(combined_eq, prices)
    print("\nRegime Analysis (HC #428 R1):")
    for k, v in regime.items():
        print(f"  {k}: {v}")

    # 8. Plots
    print("\nGenerating plots...")
    results_dict = {
        "Combined Portfolio": combined_eq,
        "v4.4 Only": v44_eq,
        "SPY B&H": spy_eq,
        "UPRO B&H": upro_eq,
    }
    plot_equity_curves(results_dict, OUTPUT_DIR / "equity_curves.png")
    plot_allocation_timeline(holdings_log, OUTPUT_DIR / "allocation_timeline.png")

    # 9. Save results
    results = {
        "run_date": datetime.now().isoformat(),
        "config": {
            "initial_capital": INITIAL_CAPITAL,
            "core_weight": W_CORE,
            "income_weight": W_INCOME,
            "spike_weight": W_SPIKE,
            "contango_entry": CONTANGO_ENTRY,
            "vix_pctile_max_vrp": VIX_PCTILE_MAX_VRP,
            "vix_spike_threshold": VIX_SPIKE_THRESHOLD,
            "warmup": WARMUP,
        },
        "metrics": {m["Name"]: m for m in metrics_list},
        "yearly_returns": {
            "Combined": {str(y): round(r, 4) for y, r in yr_combined.items()},
            "v4.4 Only": {str(y): round(r, 4) for y, r in yr_v44.items()},
            "SPY": {str(y): round(r, 4) for y, r in yr_spy.items()},
            "UPRO": {str(y): round(r, 4) for y, r in yr_upro.items()},
        },
        "worst_drawdowns": wdd,
        "income_contribution": income_contrib,
        "adversarial": {
            "permutation_test": perm,
            "sub_period_consistency": sub,
            "regime_analysis": regime,
        },
    }

    with open(OUTPUT_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {OUTPUT_DIR / 'results.json'}")

    # Save equity curves as CSV
    combined_eq.to_csv(OUTPUT_DIR / "equity_combined.csv")
    v44_eq.to_csv(OUTPUT_DIR / "equity_v44_only.csv")

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)

    return results


if __name__ == "__main__":
    main()
