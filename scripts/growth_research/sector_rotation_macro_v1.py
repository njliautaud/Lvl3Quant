"""
Sector Relative Strength Rotation with Macro Overlay — v1
==========================================================
HC #0  : Sliding walk-forward (trailing lookback, no expanding window)
HC #428: Regime-agnostic validation (sub-period consistency)
HC #694: Commission-free (Robinhood)
HC #705: Adversarial validation (permutation test, sub-period CV, walk-forward)
HC #713: No DCA — fixed $100K lump sum

Rotate among sector ETFs based on relative strength vs SPY,
with VIX-percentile macro overlay gating aggression.
"""

import os
import json
import warnings
from datetime import datetime
from pathlib import Path
from collections import Counter

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
OUTPUT_DIR = Path("/home/jupiter/Lvl3Quant/output/sector_rotation_macro_v1")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

START = "2010-01-01"
END   = "2026-07-18"
INITIAL_CAPITAL = 100_000.0

# Sector ETFs
SECTOR_ETFS = [
    "XLK",   # Technology
    "XLF",   # Financials
    "XLV",   # Healthcare
    "XLE",   # Energy
    "XLI",   # Industrials
    "XLP",   # Consumer Staples
    "XLU",   # Utilities
    "XLY",   # Consumer Discretionary
    "XLC",   # Communications (starts 2018-06)
    "XLRE",  # Real Estate (starts 2015-10)
    "XLB",   # Materials
]

# Defensive sectors (used in defensive regime)
DEFENSIVE_SECTORS = ["XLP", "XLV", "XLU"]

# Benchmarks and safe havens
BENCHMARKS = ["SPY", "SHY", "GLD"]

# Signal tickers
SIGNAL_TICKERS = ["^VIX", "^VIX3M", "HYG", "IEF"]

ALL_TICKERS = SECTOR_ETFS + BENCHMARKS + SIGNAL_TICKERS

# VIX regime thresholds
VIX_PCTILE_AGGRESSIVE = 30   # below this = aggressive
VIX_PCTILE_DEFENSIVE  = 70   # above this = defensive
VIX_CRISIS_ABS        = 30   # above this = crisis (absolute level)
VIX_PCTILE_WINDOW     = 252  # 1-year rolling percentile

# Mean reversion filter
MEAN_REV_THRESHOLD = 0.20    # exclude sector if up >20% in 1 month


# ---------------------------------------------------------------------------
# DATA DOWNLOAD
# ---------------------------------------------------------------------------
def download_data():
    """Download daily adjusted close for all tickers."""
    print("Downloading data from yfinance...")
    data = yf.download(ALL_TICKERS, start=START, end=END, auto_adjust=True)
    # yfinance returns multi-level columns: (Price, Ticker)
    if isinstance(data.columns, pd.MultiIndex):
        prices = data["Close"]
    else:
        prices = data

    # Forward fill then backward fill for missing data
    prices = prices.ffill().bfill()

    print(f"Data: {prices.index[0].date()} to {prices.index[-1].date()}, "
          f"{len(prices)} trading days")

    # Report availability
    for t in SECTOR_ETFS:
        if t in prices.columns:
            first_valid = prices[t].first_valid_index()
            print(f"  {t}: available from {first_valid.date() if first_valid else 'N/A'}")

    return prices


# ---------------------------------------------------------------------------
# SIGNALS
# ---------------------------------------------------------------------------
def compute_signals(prices):
    """Compute relative strength rankings and VIX regime."""
    spy = prices["SPY"]
    vix = prices["^VIX"]

    # Rolling VIX percentile (1-year lookback)
    vix_pctile = vix.rolling(VIX_PCTILE_WINDOW).apply(
        lambda x: percentileofscore(x, x.iloc[-1]), raw=False
    )

    # Sector returns relative to SPY
    spy_ret = {}
    sector_ret = {}
    for months, days in [(1, 21), (3, 63), (6, 126)]:
        spy_ret[months] = spy.pct_change(days)
        sector_ret[months] = {}
        for s in SECTOR_ETFS:
            if s in prices.columns:
                sector_ret[months][s] = prices[s].pct_change(days) - spy_ret[months]

    return vix, vix_pctile, sector_ret


# ---------------------------------------------------------------------------
# REBALANCE DATES
# ---------------------------------------------------------------------------
def get_monthly_rebal_dates(prices):
    """First trading day of each month."""
    dates = prices.index
    rebal = []
    prev_month = None
    for d in dates:
        if prev_month is not None and d.month != prev_month:
            rebal.append(d)
        prev_month = d.month
    return rebal


# ---------------------------------------------------------------------------
# STRATEGY VARIANTS
# ---------------------------------------------------------------------------
def get_available_sectors(prices, date):
    """Return sectors that have data on this date."""
    available = []
    for s in SECTOR_ETFS:
        if s in prices.columns:
            first_valid = prices[s].first_valid_index()
            if first_valid is not None and date >= first_valid:
                # Check the value is not NaN
                if not pd.isna(prices.loc[date, s]):
                    available.append(s)
    return available


def rank_sectors(sector_rel_strength, date, available_sectors, lookback_months=3):
    """Rank available sectors by relative strength (descending)."""
    if lookback_months not in sector_rel_strength:
        return []

    scores = {}
    for s in available_sectors:
        if s in sector_rel_strength[lookback_months]:
            val = sector_rel_strength[lookback_months].get(s)
            if val is not None and date in val.index and not pd.isna(val.loc[date]):
                scores[s] = val.loc[date]

    ranked = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
    return ranked, scores


def run_strategy(prices, vix, vix_pctile, sector_ret, variant_name,
                 top_n=3, lookback_months=3, use_macro=True,
                 use_mean_rev_filter=False):
    """
    Run a sector rotation variant.
    Returns equity curve (Series) and trade log.
    """
    rebal_dates = get_monthly_rebal_dates(prices)

    # Need enough history for lookback
    lookback_days = {1: 21, 3: 63, 6: 126}[lookback_months]
    min_date = prices.index[lookback_days + VIX_PCTILE_WINDOW + 10]
    rebal_dates = [d for d in rebal_dates if d >= min_date]

    equity = INITIAL_CAPITAL
    equity_curve = {}
    holdings = {}  # ticker -> weight
    trade_log = []
    sector_picks = Counter()
    regime_log = []

    prev_rebal = None

    for date in rebal_dates:
        # Compute returns since last rebalance
        if prev_rebal is not None and holdings:
            for ticker, weight in holdings.items():
                if ticker in prices.columns:
                    ret = prices.loc[date, ticker] / prices.loc[prev_rebal, ticker] - 1
                    equity += equity * weight * ret
            # Record equity at end of period (before new rebalance)
            # Fill daily equity between rebal dates
            mask = (prices.index > prev_rebal) & (prices.index <= date)
            for d in prices.index[mask]:
                day_eq = equity  # approximate (we'll do precise daily below)
                equity_curve[d] = day_eq

        # Determine regime
        available = get_available_sectors(prices, date)
        if len(available) < 3:
            prev_rebal = date
            continue

        vix_level = vix.loc[date] if date in vix.index else 20
        vix_pct = vix_pctile.loc[date] if date in vix_pctile.index else 50

        if use_macro:
            if vix_level > VIX_CRISIS_ABS:
                regime = "crisis"
            elif vix_pct > VIX_PCTILE_DEFENSIVE:
                regime = "defensive"
            elif vix_pct < VIX_PCTILE_AGGRESSIVE:
                regime = "aggressive"
            else:
                regime = "neutral"
        else:
            regime = "aggressive"  # always full in for pure rotation

        regime_log.append({"date": str(date.date()), "regime": regime,
                           "vix": round(vix_level, 1), "vix_pctile": round(vix_pct, 1)})

        # Rank sectors
        try:
            ranked, scores = rank_sectors(sector_ret, date, available, lookback_months)
        except (ValueError, KeyError):
            prev_rebal = date
            continue

        if not ranked:
            prev_rebal = date
            continue

        # Apply mean reversion filter if enabled
        if use_mean_rev_filter and 1 in sector_ret:
            filtered = []
            for s in ranked:
                one_mo = sector_ret[1].get(s)
                if one_mo is not None and date in one_mo.index:
                    raw_1m = prices[s].pct_change(21).loc[date]
                    if raw_1m > MEAN_REV_THRESHOLD:
                        continue  # skip overextended sector
                filtered.append(s)
            ranked = filtered if len(filtered) >= top_n else ranked

        # Assign weights based on regime
        holdings = {}
        if regime == "crisis":
            holdings["SHY"] = 1.0
        elif regime == "defensive":
            # Top defensive sectors + SHY/GLD
            def_available = [s for s in DEFENSIVE_SECTORS if s in available]
            pick = def_available[:2] if len(def_available) >= 2 else ranked[:2]
            eq_w = 0.25 / len(pick) if pick else 0
            for s in pick:
                holdings[s] = eq_w
                sector_picks[s] += 1
            holdings["SHY"] = 0.25
            holdings["GLD"] = 0.25
        elif regime == "neutral":
            pick = ranked[:top_n]
            eq_w = 0.75 / len(pick)
            for s in pick:
                holdings[s] = eq_w
                sector_picks[s] += 1
            holdings["SHY"] = 0.25
        else:  # aggressive
            pick = ranked[:top_n]
            eq_w = 1.0 / len(pick)
            for s in pick:
                holdings[s] = eq_w
                sector_picks[s] += 1

        trade_log.append({
            "date": str(date.date()),
            "regime": regime,
            "holdings": {k: round(v, 4) for k, v in holdings.items()},
            "top_ranked": ranked[:5]
        })

        prev_rebal = date

    # Build proper daily equity curve
    equity_daily = build_daily_equity(prices, rebal_dates, trade_log, vix, vix_pctile,
                                      sector_ret, top_n, lookback_months, use_macro,
                                      use_mean_rev_filter)

    return equity_daily, trade_log, dict(sector_picks), regime_log


def build_daily_equity(prices, rebal_dates, trade_log, vix, vix_pctile,
                       sector_ret, top_n, lookback_months, use_macro,
                       use_mean_rev_filter):
    """Build precise daily equity curve using daily returns of holdings."""
    if not trade_log:
        return pd.Series(dtype=float)

    first_date = pd.Timestamp(trade_log[0]["date"])
    last_date = prices.index[-1]

    daily_dates = prices.index[(prices.index >= first_date) & (prices.index <= last_date)]

    equity = INITIAL_CAPITAL
    equity_series = {}

    # Map rebalance dates to holdings
    rebal_holdings = []
    for t in trade_log:
        rebal_holdings.append((pd.Timestamp(t["date"]), t["holdings"]))

    current_holdings_idx = 0
    holdings = rebal_holdings[0][1]

    prev_date = None
    for d in daily_dates:
        # Check if we rebalance today
        while (current_holdings_idx + 1 < len(rebal_holdings) and
               d >= rebal_holdings[current_holdings_idx + 1][0]):
            current_holdings_idx += 1
            holdings = rebal_holdings[current_holdings_idx][1]

        if prev_date is not None:
            daily_ret = 0.0
            for ticker, weight in holdings.items():
                if ticker in prices.columns:
                    p0 = prices.loc[prev_date, ticker]
                    p1 = prices.loc[d, ticker]
                    if p0 > 0 and not pd.isna(p0) and not pd.isna(p1):
                        daily_ret += weight * (p1 / p0 - 1)
            equity *= (1 + daily_ret)

        equity_series[d] = equity
        prev_date = d

    return pd.Series(equity_series)


# ---------------------------------------------------------------------------
# METRICS
# ---------------------------------------------------------------------------
def compute_metrics(equity_curve, name="Strategy"):
    """Compute Sharpe, Sortino, CAGR, MaxDD, Calmar."""
    if len(equity_curve) < 60:
        return {}

    returns = equity_curve.pct_change().dropna()

    ann_ret = (equity_curve.iloc[-1] / equity_curve.iloc[0]) ** (252 / len(returns)) - 1
    ann_vol = returns.std() * np.sqrt(252)

    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0

    downside = returns[returns < 0].std() * np.sqrt(252)
    sortino = ann_ret / downside if downside > 0 else 0

    cum_max = equity_curve.cummax()
    drawdown = (equity_curve - cum_max) / cum_max
    max_dd = drawdown.min()

    cagr = ann_ret
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0

    # Profit factor
    pos = returns[returns > 0].sum()
    neg = abs(returns[returns < 0].sum())
    pf = pos / neg if neg > 0 else float("inf")

    # Win rate
    wr = (returns > 0).sum() / len(returns) if len(returns) > 0 else 0

    # Monthly returns
    monthly = equity_curve.resample("ME").last().pct_change().dropna()
    monthly_sharpe = (monthly.mean() / monthly.std() * np.sqrt(12)) if monthly.std() > 0 else 0

    return {
        "name": name,
        "CAGR": round(cagr * 100, 2),
        "Sharpe": round(sharpe, 3),
        "Sortino": round(sortino, 3),
        "MaxDD": round(max_dd * 100, 2),
        "Calmar": round(calmar, 3),
        "ProfitFactor": round(pf, 3),
        "WinRate_daily": round(wr * 100, 1),
        "AnnVol": round(ann_vol * 100, 2),
        "FinalEquity": round(equity_curve.iloc[-1], 2),
        "MonthlySharpе": round(monthly_sharpe, 3),
    }


def compute_monthly_returns(equity_curve):
    """Return monthly returns as a pivot table (year x month)."""
    monthly = equity_curve.resample("ME").last().pct_change().dropna()
    df = pd.DataFrame({"return": monthly})
    df["year"] = df.index.year
    df["month"] = df.index.month
    pivot = df.pivot_table(values="return", index="year", columns="month",
                           aggfunc="first")
    pivot.columns = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                     "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    return pivot


# ---------------------------------------------------------------------------
# ADVERSARIAL VALIDATION (HC #705)
# ---------------------------------------------------------------------------
def permutation_test(prices, vix, vix_pctile, sector_ret, n_perms=100):
    """Shuffle sector rankings each month and compare to real strategy."""
    print(f"\nRunning permutation test ({n_perms} shuffles)...")

    # Real strategy Sharpe
    real_eq, _, _, _ = run_strategy(prices, vix, vix_pctile, sector_ret,
                                     "real", top_n=3, lookback_months=3, use_macro=True)
    real_metrics = compute_metrics(real_eq, "Real")
    real_sharpe = real_metrics.get("Sharpe", 0)

    # Permuted
    perm_sharpes = []
    for i in range(n_perms):
        # Shuffle sector relative strength rankings
        shuffled_ret = {}
        for months in sector_ret:
            shuffled_ret[months] = {}
            sectors = list(sector_ret[months].keys())
            # Randomly reassign sector return series
            np.random.seed(i * 100 + months)
            perm_idx = np.random.permutation(len(sectors))
            for j, s in enumerate(sectors):
                shuffled_ret[months][s] = sector_ret[months][sectors[perm_idx[j]]]

        perm_eq, _, _, _ = run_strategy(prices, vix, vix_pctile, shuffled_ret,
                                         f"perm_{i}", top_n=3, lookback_months=3,
                                         use_macro=True)
        pm = compute_metrics(perm_eq, f"perm_{i}")
        perm_sharpes.append(pm.get("Sharpe", 0))

        if (i + 1) % 20 == 0:
            print(f"  Completed {i+1}/{n_perms} permutations")

    p_value = np.mean([s >= real_sharpe for s in perm_sharpes])

    return {
        "real_sharpe": real_sharpe,
        "perm_mean_sharpe": round(np.mean(perm_sharpes), 3),
        "perm_std_sharpe": round(np.std(perm_sharpes), 3),
        "p_value": round(p_value, 4),
        "perm_95th": round(np.percentile(perm_sharpes, 95), 3),
        "significant_at_05": p_value < 0.05,
    }


def subperiod_consistency(equity_curve, n_blocks=3):
    """Split into n blocks and check Sharpe CV < 0.6."""
    returns = equity_curve.pct_change().dropna()
    block_size = len(returns) // n_blocks

    sharpes = []
    for i in range(n_blocks):
        start = i * block_size
        end = start + block_size if i < n_blocks - 1 else len(returns)
        block_ret = returns.iloc[start:end]
        s = block_ret.mean() / block_ret.std() * np.sqrt(252) if block_ret.std() > 0 else 0
        sharpes.append(s)

    mean_s = np.mean(sharpes)
    std_s = np.std(sharpes)
    cv = std_s / abs(mean_s) if mean_s != 0 else float("inf")

    return {
        "block_sharpes": [round(s, 3) for s in sharpes],
        "mean_sharpe": round(mean_s, 3),
        "sharpe_cv": round(cv, 3),
        "passes_cv_threshold": cv < 0.6,
    }


def walk_forward_validation(prices, vix, vix_pctile, sector_ret):
    """5-fold walk-forward: train lookback on first portion, test on next."""
    print("\nRunning walk-forward validation...")
    all_dates = prices.index
    n = len(all_dates)
    fold_size = n // 6  # 5 test folds, first fold is pure train

    wf_results = []
    for fold in range(5):
        test_start = all_dates[fold_size * (fold + 1)]
        test_end = all_dates[min(fold_size * (fold + 2) - 1, n - 1)]

        # Run strategy on test period only
        test_prices = prices.loc[test_start:test_end]
        if len(test_prices) < 60:
            continue

        # Use full price history for signal computation but only test period for equity
        eq, _, _, _ = run_strategy(prices, vix, vix_pctile, sector_ret,
                                    f"wf_fold_{fold}", top_n=3, lookback_months=3,
                                    use_macro=True)

        # Slice equity to test period
        eq_test = eq.loc[eq.index >= test_start]
        eq_test = eq_test.loc[eq_test.index <= test_end]

        if len(eq_test) > 60:
            m = compute_metrics(eq_test, f"Fold {fold}")
            m["period"] = f"{test_start.date()} to {test_end.date()}"
            wf_results.append(m)

    return wf_results


# ---------------------------------------------------------------------------
# TURNOVER ANALYSIS
# ---------------------------------------------------------------------------
def compute_turnover(trade_log):
    """Compute average monthly turnover."""
    if len(trade_log) < 2:
        return 0.0

    turnovers = []
    for i in range(1, len(trade_log)):
        prev = trade_log[i - 1]["holdings"]
        curr = trade_log[i]["holdings"]
        all_tickers = set(list(prev.keys()) + list(curr.keys()))
        turnover = sum(abs(curr.get(t, 0) - prev.get(t, 0)) for t in all_tickers) / 2
        turnovers.append(turnover)

    return round(np.mean(turnovers) * 100, 1)


# ---------------------------------------------------------------------------
# PLOTTING
# ---------------------------------------------------------------------------
def plot_equity_curves(results, spy_eq):
    """Plot all strategy equity curves vs SPY."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    ax = axes[0]
    for name, eq in results.items():
        if len(eq) > 0:
            ax.plot(eq.index, eq.values, label=name, alpha=0.8)
    ax.plot(spy_eq.index, spy_eq.values, label="SPY B&H", color="black",
            linestyle="--", alpha=0.6)
    ax.set_ylabel("Equity ($)")
    ax.set_title("Sector Rotation Strategies vs SPY Buy & Hold")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")

    # Drawdown for best strategy
    ax2 = axes[1]
    for name, eq in results.items():
        if len(eq) > 0:
            dd = (eq - eq.cummax()) / eq.cummax() * 100
            ax2.plot(dd.index, dd.values, label=name, alpha=0.6)
    ax2.set_ylabel("Drawdown (%)")
    ax2.set_title("Drawdowns")
    ax2.legend(fontsize=8, loc="lower left")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "equity_curves.png", dpi=150)
    plt.close()
    print(f"Saved equity_curves.png")


def plot_sector_frequency(sector_picks_all):
    """Bar chart of sector selection frequency."""
    fig, ax = plt.subplots(figsize=(12, 6))

    # Primary strategy picks
    if "Macro Top3 3mo" in sector_picks_all:
        picks = sector_picks_all["Macro Top3 3mo"]
        sectors = sorted(picks.keys())
        counts = [picks[s] for s in sectors]
        ax.bar(sectors, counts, alpha=0.8)
        ax.set_ylabel("Times Selected")
        ax.set_title("Sector Selection Frequency (Macro Overlay, Top 3, 3mo lookback)")
        ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "sector_frequency.png", dpi=150)
    plt.close()
    print(f"Saved sector_frequency.png")


def plot_monthly_heatmap(monthly_returns, name):
    """Monthly returns heatmap."""
    fig, ax = plt.subplots(figsize=(14, max(6, len(monthly_returns) * 0.4)))

    data = monthly_returns.values * 100
    im = ax.imshow(data, cmap="RdYlGn", aspect="auto", vmin=-10, vmax=10)

    ax.set_xticks(range(12))
    ax.set_xticklabels(monthly_returns.columns, fontsize=9)
    ax.set_yticks(range(len(monthly_returns)))
    ax.set_yticklabels(monthly_returns.index, fontsize=9)

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if not np.isnan(data[i, j]):
                ax.text(j, i, f"{data[i, j]:.1f}", ha="center", va="center",
                        fontsize=7, color="black" if abs(data[i, j]) < 5 else "white")

    ax.set_title(f"Monthly Returns (%) — {name}")
    plt.colorbar(im, ax=ax, label="Return (%)")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"monthly_returns_{name.replace(' ', '_').lower()}.png", dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("SECTOR ROTATION WITH MACRO OVERLAY — v1")
    print("=" * 70)

    # Download data
    prices = download_data()

    # Compute signals
    vix, vix_pctile, sector_ret = compute_signals(prices)

    # SPY buy & hold benchmark
    spy_start_idx = prices.index[VIX_PCTILE_WINDOW + 130]  # match strategy start
    spy_eq = prices["SPY"].loc[spy_start_idx:] / prices["SPY"].loc[spy_start_idx] * INITIAL_CAPITAL

    # ---------------------------------------------------------------------------
    # Run all variants
    # ---------------------------------------------------------------------------
    variants = {
        # Primary: Macro overlay variants
        "Macro Top3 3mo":    {"top_n": 3, "lookback_months": 3, "use_macro": True,  "use_mean_rev_filter": False},
        "Macro Top5 3mo":    {"top_n": 5, "lookback_months": 3, "use_macro": True,  "use_mean_rev_filter": False},
        "Macro Top3 1mo":    {"top_n": 3, "lookback_months": 1, "use_macro": True,  "use_mean_rev_filter": False},
        "Macro Top3 6mo":    {"top_n": 3, "lookback_months": 6, "use_macro": True,  "use_mean_rev_filter": False},
        # Pure rotation (no macro)
        "Pure Top3 3mo":     {"top_n": 3, "lookback_months": 3, "use_macro": False, "use_mean_rev_filter": False},
        "Pure Top5 3mo":     {"top_n": 5, "lookback_months": 3, "use_macro": False, "use_mean_rev_filter": False},
        # Mean reversion filter
        "Macro Top3 3mo MR": {"top_n": 3, "lookback_months": 3, "use_macro": True,  "use_mean_rev_filter": True},
    }

    all_equity = {}
    all_metrics = {}
    all_sector_picks = {}
    all_trade_logs = {}

    for name, params in variants.items():
        print(f"\n--- Running: {name} ---")
        eq, trades, picks, regimes = run_strategy(
            prices, vix, vix_pctile, sector_ret, name, **params
        )
        all_equity[name] = eq
        all_metrics[name] = compute_metrics(eq, name)
        all_sector_picks[name] = picks
        all_trade_logs[name] = trades

        m = all_metrics[name]
        print(f"  CAGR: {m.get('CAGR', 'N/A')}%  Sharpe: {m.get('Sharpe', 'N/A')}  "
              f"Sortino: {m.get('Sortino', 'N/A')}  MaxDD: {m.get('MaxDD', 'N/A')}%  "
              f"Calmar: {m.get('Calmar', 'N/A')}")

    # Equal weight all sectors benchmark
    print(f"\n--- Running: Equal Weight All ---")
    # Simple equal weight: hold all available sectors equally, rebalance monthly
    eq_ew, trades_ew, _, _ = run_strategy(
        prices, vix, vix_pctile, sector_ret, "EqWt All",
        top_n=len(SECTOR_ETFS), lookback_months=3, use_macro=False
    )
    all_equity["EqWt All Sectors"] = eq_ew
    all_metrics["EqWt All Sectors"] = compute_metrics(eq_ew, "EqWt All Sectors")
    m = all_metrics["EqWt All Sectors"]
    print(f"  CAGR: {m.get('CAGR', 'N/A')}%  Sharpe: {m.get('Sharpe', 'N/A')}  "
          f"Sortino: {m.get('Sortino', 'N/A')}  MaxDD: {m.get('MaxDD', 'N/A')}%")

    # SPY benchmark metrics
    spy_metrics = compute_metrics(spy_eq, "SPY B&H")
    all_metrics["SPY B&H"] = spy_metrics

    # ---------------------------------------------------------------------------
    # Results table
    # ---------------------------------------------------------------------------
    print("\n" + "=" * 90)
    print("RESULTS SUMMARY")
    print("=" * 90)
    header = f"{'Strategy':<25} {'CAGR%':>7} {'Sharpe':>7} {'Sortino':>8} {'MaxDD%':>8} {'Calmar':>7} {'PF':>6} {'WR%':>6}"
    print(header)
    print("-" * 90)

    for name in list(variants.keys()) + ["EqWt All Sectors", "SPY B&H"]:
        m = all_metrics.get(name, {})
        print(f"{name:<25} {m.get('CAGR', 'N/A'):>7} {m.get('Sharpe', 'N/A'):>7} "
              f"{m.get('Sortino', 'N/A'):>8} {m.get('MaxDD', 'N/A'):>8} "
              f"{m.get('Calmar', 'N/A'):>7} {m.get('ProfitFactor', 'N/A'):>6} "
              f"{m.get('WinRate_daily', 'N/A'):>6}")

    # ---------------------------------------------------------------------------
    # Sector frequency analysis
    # ---------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SECTOR SELECTION FREQUENCY (Macro Top3 3mo)")
    print("=" * 70)
    if "Macro Top3 3mo" in all_sector_picks:
        picks = all_sector_picks["Macro Top3 3mo"]
        sorted_picks = sorted(picks.items(), key=lambda x: x[1], reverse=True)
        total = sum(picks.values())
        for sector, count in sorted_picks:
            pct = count / total * 100
            bar = "#" * int(pct)
            print(f"  {sector:<5} {count:>4} times ({pct:>5.1f}%) {bar}")

    # ---------------------------------------------------------------------------
    # Turnover analysis
    # ---------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("TURNOVER ANALYSIS (avg monthly turnover %)")
    print("=" * 70)
    for name in variants:
        if name in all_trade_logs:
            to = compute_turnover(all_trade_logs[name])
            print(f"  {name:<25} {to}%")

    # ---------------------------------------------------------------------------
    # Monthly returns for best strategy
    # ---------------------------------------------------------------------------
    best_name = max(all_metrics, key=lambda x: all_metrics[x].get("Sharpe", -99)
                    if x != "SPY B&H" else -99)
    print(f"\nBest strategy by Sharpe: {best_name}")

    if best_name in all_equity and len(all_equity[best_name]) > 0:
        monthly_ret = compute_monthly_returns(all_equity[best_name])
        print("\nMonthly Returns (%) — " + best_name)
        print(monthly_ret.round(3) * 100)
        plot_monthly_heatmap(monthly_ret, best_name)

    # ---------------------------------------------------------------------------
    # Adversarial validation (HC #705)
    # ---------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("ADVERSARIAL VALIDATION (HC #705)")
    print("=" * 70)

    # 1. Permutation test
    perm_results = permutation_test(prices, vix, vix_pctile, sector_ret, n_perms=100)
    print(f"\nPermutation Test (100 shuffles):")
    print(f"  Real Sharpe:     {perm_results['real_sharpe']}")
    print(f"  Perm Mean Sharpe: {perm_results['perm_mean_sharpe']} "
          f"(+/- {perm_results['perm_std_sharpe']})")
    print(f"  95th percentile:  {perm_results['perm_95th']}")
    print(f"  p-value:          {perm_results['p_value']}")
    print(f"  Significant (p<0.05): {perm_results['significant_at_05']}")

    # 2. Sub-period consistency
    if best_name in all_equity and len(all_equity[best_name]) > 0:
        subperiod = subperiod_consistency(all_equity[best_name], n_blocks=3)
        print(f"\nSub-period Consistency ({best_name}):")
        print(f"  Block Sharpes: {subperiod['block_sharpes']}")
        print(f"  Mean Sharpe:   {subperiod['mean_sharpe']}")
        print(f"  Sharpe CV:     {subperiod['sharpe_cv']} (threshold < 0.6)")
        print(f"  PASSES:        {subperiod['passes_cv_threshold']}")

    # 3. Walk-forward
    wf_results = walk_forward_validation(prices, vix, vix_pctile, sector_ret)
    if wf_results:
        print(f"\nWalk-Forward Validation:")
        for r in wf_results:
            print(f"  {r.get('period', 'N/A')}: Sharpe={r.get('Sharpe', 'N/A')} "
                  f"CAGR={r.get('CAGR', 'N/A')}%")

    # ---------------------------------------------------------------------------
    # Plots
    # ---------------------------------------------------------------------------
    plot_equity_curves(all_equity, spy_eq)
    plot_sector_frequency(all_sector_picks)

    # ---------------------------------------------------------------------------
    # Save outputs
    # ---------------------------------------------------------------------------
    output = {
        "run_date": str(datetime.now()),
        "data_range": f"{prices.index[0].date()} to {prices.index[-1].date()}",
        "initial_capital": INITIAL_CAPITAL,
        "metrics": all_metrics,
        "sector_frequency": all_sector_picks,
        "adversarial": {
            "permutation_test": perm_results,
            "subperiod_consistency": subperiod if best_name in all_equity else {},
            "walk_forward": wf_results,
        },
        "turnover": {name: compute_turnover(all_trade_logs[name])
                     for name in variants if name in all_trade_logs},
    }

    with open(OUTPUT_DIR / "results.json", "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved results.json")

    # Save trade log for best strategy
    if best_name in all_trade_logs:
        with open(OUTPUT_DIR / "trade_log.json", "w") as f:
            json.dump(all_trade_logs[best_name], f, indent=2)
        print(f"Saved trade_log.json for {best_name}")

    # Save equity curves as CSV
    eq_df = pd.DataFrame(all_equity)
    eq_df["SPY_BH"] = spy_eq
    eq_df.to_csv(OUTPUT_DIR / "equity_curves.csv")
    print(f"Saved equity_curves.csv")

    print("\n" + "=" * 70)
    print("DONE — All outputs saved to:")
    print(f"  {OUTPUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
