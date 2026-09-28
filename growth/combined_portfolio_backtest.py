#!/usr/bin/env python3
"""
Combined Growth Portfolio Backtest
===================================
Combines two uncorrelated strategies into portfolio blends:
  1. Momentum Top 10 (growth engine, CAGR ~23%, regime-dependent)
  2. Trend-Following 200MA (crisis hedge, CAGR ~8%, cuts DD in half)

Tests allocations: 50/50, 70/30, 30/70, Risk Parity
Benchmarks: SPY buy-and-hold, each strategy standalone

Rebuilds daily equity curves from scratch (JSON summaries lack daily data),
then aligns and combines.

Usage:
  python growth/combined_portfolio_backtest.py
"""

import json
import os
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent))
from universe import build_universe

# ── Momentum Constants ────────────────────────────────────────────────────────

LOOKBACK_DAYS = 252
SKIP_RECENT = 21
HOLD_PERIOD = 21
ATR_PERIOD = 20
STOP_LOSS_PCT = 0.08
TRAILING_STOP_ATR_MULT = 2.0
MOM_BREAKDOWN_DAYS = 10
COST_PER_TRADE = 0.001
MOMENTUM_PORTFOLIO_SIZE = 10

MOM_DATA_START = "2018-01-01"
MOM_BACKTEST_START = "2019-01-01"

# ── Trend Constants ───────────────────────────────────────────────────────────

TREND_UNIVERSE = {
    "US Equities": ["SPY", "QQQ", "IWM"],
    "International": ["EFA", "EEM"],
    "Bonds": ["TLT", "IEF", "AGG"],
    "Commodities": ["GLD", "DBA"],
    "Real Estate": ["VNQ"],
}
TREND_TICKERS = [t for ts in TREND_UNIVERSE.values() for t in ts]
TREND_DATA_START = "2018-01-01"  # Align with momentum start
TREND_SMA_SLOW = 200
TREND_SMA_FAST = 50
TREND_COST_BPS = 5
TREND_REBALANCE_FREQ = "ME"

# ── Risk Parity ──────────────────────────────────────────────────────────────

RISK_PARITY_LOOKBACK = 63  # 3-month trailing vol for risk parity weights

# ── Output ───────────────────────────────────────────────────────────────────

OUTPUT_DIR = Path(__file__).parent / "output"


# ═══════════════════════════════════════════════════════════════════════════════
#  MOMENTUM STRATEGY — rebuild daily equity curve
# ═══════════════════════════════════════════════════════════════════════════════

class Position:
    __slots__ = ["ticker", "entry_price", "entry_date", "entry_idx",
                 "trailing_high", "shares", "cost_basis"]

    def __init__(self, ticker, entry_price, entry_date, entry_idx, shares):
        self.ticker = ticker
        self.entry_price = entry_price
        self.entry_date = entry_date
        self.entry_idx = entry_idx
        self.trailing_high = entry_price
        self.shares = shares
        self.cost_basis = entry_price * shares * (1 + COST_PER_TRADE)

    def check_exit(self, current_price, current_idx, atr_value, mom_10d):
        if current_price > self.trailing_high:
            self.trailing_high = current_price
        if current_price <= self.entry_price * (1 - STOP_LOSS_PCT):
            return True, "stop_loss"
        if not np.isnan(atr_value) and atr_value > 0:
            if current_price <= self.trailing_high - TRAILING_STOP_ATR_MULT * atr_value:
                return True, "trailing_stop"
        if not np.isnan(mom_10d) and mom_10d < 0:
            return True, "momentum_breakdown"
        return False, None

    def pnl(self, exit_price):
        return exit_price * self.shares * (1 - COST_PER_TRADE) - self.cost_basis


def download_batch(tickers, start, end=None):
    """Download OHLCV data for tickers."""
    if end is None:
        end = datetime.now().strftime("%Y-%m-%d")
    batch_size = 50
    all_close, all_high, all_low = {}, {}, {}
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i + batch_size]
        try:
            data = yf.download(" ".join(batch), start=start, end=end,
                               progress=False, threads=True, group_by="ticker")
            if data.empty:
                continue
            if isinstance(data.columns, pd.MultiIndex):
                for t in batch:
                    try:
                        if t in data.columns.get_level_values(0):
                            td = data[t]
                            cc = "Adj Close" if "Adj Close" in td.columns else "Close"
                            c = td[cc].dropna()
                            if len(c) > 200:
                                all_close[t] = c
                                all_high[t] = td["High"].dropna()
                                all_low[t] = td["Low"].dropna()
                    except Exception:
                        pass
            elif len(batch) == 1:
                cc = "Adj Close" if "Adj Close" in data.columns else "Close"
                c = data[cc].dropna()
                if len(c) > 200:
                    all_close[batch[0]] = c
                    all_high[batch[0]] = data["High"].dropna()
                    all_low[batch[0]] = data["Low"].dropna()
        except Exception as e:
            print(f"  [WARN] Batch failed: {e}")
        if i + batch_size < len(tickers):
            time.sleep(0.5)

    close_df = pd.DataFrame(all_close)
    high_df = pd.DataFrame(all_high).reindex(close_df.index)
    low_df = pd.DataFrame(all_low).reindex(close_df.index)
    return close_df, high_df, low_df


def run_momentum_backtest(close, high, low):
    """Run momentum top-10 backtest, return daily equity Series."""
    dates = close.index
    n_dates = len(dates)
    backtest_start = pd.Timestamp(MOM_BACKTEST_START)
    first_valid_idx = 0
    for i in range(n_dates):
        if dates[i] >= backtest_start and i >= LOOKBACK_DAYS:
            first_valid_idx = i
            break

    cash = 1_000_000.0
    positions = {}
    equity_dates, equity_vals = [], []
    rebalance_indices = list(range(first_valid_idx, n_dates, HOLD_PERIOD))
    current_rebalance_ptr = 0
    next_rebalance_idx = rebalance_indices[0] if rebalance_indices else n_dates

    for day_idx in range(first_valid_idx, n_dates):
        current_date = dates[day_idx]

        # Daily exit checks
        tickers_to_exit = []
        for ticker, pos in positions.items():
            try:
                cp = close[ticker].iloc[day_idx]
                if np.isnan(cp):
                    tickers_to_exit.append(ticker)
                    continue
                atr_val = np.nan
                if day_idx >= ATR_PERIOD:
                    try:
                        h = high[ticker].iloc[day_idx - ATR_PERIOD:day_idx + 1].values
                        l = low[ticker].iloc[day_idx - ATR_PERIOD:day_idx + 1].values
                        c = close[ticker].iloc[day_idx - ATR_PERIOD:day_idx + 1].values
                        pc = np.roll(c, 1); pc[0] = c[0]
                        tr = np.maximum(h - l, np.maximum(np.abs(h - pc), np.abs(l - pc)))
                        atr_val = np.nanmean(tr[1:])
                    except Exception:
                        pass
                mom_10d = np.nan
                if day_idx >= MOM_BREAKDOWN_DAYS:
                    try:
                        p_now = close[ticker].iloc[day_idx]
                        p_10 = close[ticker].iloc[day_idx - MOM_BREAKDOWN_DAYS]
                        if not np.isnan(p_10) and p_10 > 0:
                            mom_10d = (p_now / p_10) - 1.0
                    except Exception:
                        pass
                should_exit, _ = pos.check_exit(cp, day_idx, atr_val, mom_10d)
                if should_exit:
                    tickers_to_exit.append(ticker)
            except Exception:
                tickers_to_exit.append(ticker)

        for ticker in tickers_to_exit:
            pos = positions[ticker]
            try:
                ep = close[ticker].iloc[day_idx]
                if np.isnan(ep):
                    ep = pos.entry_price
                cash += ep * pos.shares * (1 - COST_PER_TRADE)
            except Exception:
                pass
            del positions[ticker]

        # Rebalance
        if day_idx == next_rebalance_idx:
            for ticker in list(positions.keys()):
                pos = positions[ticker]
                try:
                    ep = close[ticker].iloc[day_idx]
                    if np.isnan(ep):
                        ep = pos.entry_price
                    cash += ep * pos.shares * (1 - COST_PER_TRADE)
                except Exception:
                    pass
            positions.clear()

            # Momentum ranking
            start_idx = max(0, day_idx - LOOKBACK_DAYS)
            end_idx = day_idx - SKIP_RECENT
            if end_idx > start_idx:
                p_start = close.iloc[start_idx]
                p_end = close.iloc[end_idx]
                momentum = ((p_end / p_start) - 1.0).dropna()
                valid = [t for t in momentum.index
                         if not np.isnan(close[t].iloc[day_idx])
                         and close[t].iloc[day_idx] > 5.0]
                momentum = momentum[valid].sort_values(ascending=False)
                selected = momentum.head(MOMENTUM_PORTFOLIO_SIZE).index.tolist()

                if selected:
                    per_stock = cash / len(selected)
                    for t in selected:
                        try:
                            ep = close[t].iloc[day_idx]
                            if np.isnan(ep) or ep <= 0:
                                continue
                            shares = int(per_stock / (ep * (1 + COST_PER_TRADE)))
                            if shares > 0:
                                cash -= ep * shares * (1 + COST_PER_TRADE)
                                positions[t] = Position(t, ep, current_date, day_idx, shares)
                        except Exception:
                            pass

            current_rebalance_ptr += 1
            if current_rebalance_ptr < len(rebalance_indices):
                next_rebalance_idx = rebalance_indices[current_rebalance_ptr]
            else:
                next_rebalance_idx = n_dates

        # Record equity
        portfolio_value = cash
        for ticker, pos in positions.items():
            try:
                p = close[ticker].iloc[day_idx]
                portfolio_value += (p if not np.isnan(p) else pos.entry_price) * pos.shares
            except Exception:
                portfolio_value += pos.entry_price * pos.shares

        equity_dates.append(current_date)
        equity_vals.append(portfolio_value)

    return pd.Series(equity_vals, index=equity_dates, name="Momentum_Top10")


# ═══════════════════════════════════════════════════════════════════════════════
#  TREND-FOLLOWING STRATEGY — rebuild daily returns
# ═══════════════════════════════════════════════════════════════════════════════

def run_trend_backtest(prices):
    """Run trend-following 200MA backtest. Returns daily returns Series."""
    available = [t for t in TREND_TICKERS if t in prices.columns]
    p = prices[available]

    sma_slow = p.rolling(TREND_SMA_SLOW, min_periods=TREND_SMA_SLOW).mean()
    signals = (p > sma_slow).astype(int)

    daily_returns = p.pct_change()
    monthly_signals = signals.resample(TREND_REBALANCE_FREQ).last()
    daily_signals = monthly_signals.reindex(signals.index).ffill()
    n_on = daily_signals.sum(axis=1).replace(0, np.nan)
    weights = daily_signals.div(n_on, axis=0).fillna(0)
    weight_changes = weights.diff().abs().sum(axis=1)
    costs = weight_changes * (TREND_COST_BPS / 10000)
    port_returns = (weights.shift(1) * daily_returns).sum(axis=1) - costs
    port_returns = port_returns.loc[MOM_BACKTEST_START:]
    return port_returns


# ═══════════════════════════════════════════════════════════════════════════════
#  PORTFOLIO COMBINATION ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

def compute_metrics(returns, name=""):
    """Full risk-adjusted metrics from daily returns."""
    returns = returns.dropna()
    if len(returns) < 30:
        return {"name": name, "error": "insufficient data"}

    total_return = (1 + returns).prod()
    n_years = len(returns) / 252
    cagr = total_return ** (1 / max(n_years, 0.01)) - 1
    vol = returns.std() * np.sqrt(252)
    sharpe = (returns.mean() * 252) / vol if vol > 0 else 0
    downside_std = returns[returns < 0].std() * np.sqrt(252)
    sortino = (returns.mean() * 252) / downside_std if downside_std > 0 else 0

    cum = (1 + returns).cumprod()
    running_max = cum.cummax()
    dd = (cum - running_max) / running_max
    max_dd = dd.min()
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0

    # Worst drawdown periods
    worst_dds = _top_drawdowns(returns, n=5)

    return {
        "name": name,
        "cagr_pct": round(cagr * 100, 2),
        "total_return_pct": round((total_return - 1) * 100, 2),
        "annual_vol_pct": round(vol * 100, 2),
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "max_drawdown_pct": round(max_dd * 100, 2),
        "calmar": round(calmar, 3),
        "n_years": round(n_years, 2),
        "n_days": len(returns),
        "worst_drawdowns": worst_dds,
    }


def _top_drawdowns(returns, n=5):
    cum = (1 + returns).cumprod()
    running_max = cum.cummax()
    dd = (cum - running_max) / running_max
    dd_copy = dd.copy()
    results = []
    for _ in range(n):
        if dd_copy.min() >= -0.005:
            break
        trough_idx = dd_copy.idxmin()
        trough_val = dd_copy.loc[trough_idx]
        pre = cum.loc[:trough_idx]
        peak_idx = pre.idxmax()
        post = cum.loc[trough_idx:]
        peak_val = cum.loc[peak_idx]
        recovered = post[post >= peak_val]
        rec_idx = recovered.index[0] if len(recovered) > 0 else cum.index[-1]
        results.append({
            "start": str(peak_idx.date()),
            "trough": str(trough_idx.date()),
            "recovery": str(rec_idx.date()) if len(recovered) > 0 else "ongoing",
            "depth_pct": round(trough_val * 100, 2),
            "duration_days": (rec_idx - peak_idx).days,
        })
        dd_copy.loc[peak_idx:rec_idx] = 0
    return results


def regime_analysis(returns, spy_returns, name=""):
    """Monthly regime analysis: green/red by SPY."""
    monthly_strat = returns.resample("ME").sum()
    monthly_spy = spy_returns.resample("ME").sum()
    common = monthly_strat.index.intersection(monthly_spy.index)
    if len(common) < 10:
        return {"name": name, "error": "insufficient months"}

    s = monthly_strat.loc[common]
    m = monthly_spy.loc[common]
    green = s[m > 0]
    red = s[m <= 0]

    def _sh(x):
        if len(x) < 2 or x.std() == 0:
            return 0
        return (x.mean() / x.std()) * np.sqrt(12)

    return {
        "name": name,
        "green_months": int((m > 0).sum()),
        "green_avg_ret_pct": round(green.mean() * 100, 3) if len(green) > 0 else 0,
        "green_sharpe": round(_sh(green), 3),
        "red_months": int((m <= 0).sum()),
        "red_avg_ret_pct": round(red.mean() * 100, 3) if len(red) > 0 else 0,
        "red_sharpe": round(_sh(red), 3),
    }


def yearly_returns_table(returns_dict):
    """Year-by-year return table."""
    yearly = {}
    for name, ret in returns_dict.items():
        annual = ret.resample("YE").apply(lambda x: (1 + x).prod() - 1)
        yearly[name] = annual * 100
    df = pd.DataFrame(yearly)
    df.index = df.index.year
    df.index.name = "Year"
    return df.round(2)


def risk_parity_weights(mom_returns, trend_returns, lookback=RISK_PARITY_LOOKBACK):
    """
    Daily risk parity weights: inversely proportional to trailing vol.
    Returns two Series: w_mom, w_trend (summing to 1.0 each day).
    """
    vol_mom = mom_returns.rolling(lookback, min_periods=20).std()
    vol_trend = trend_returns.rolling(lookback, min_periods=20).std()

    # Inverse vol
    inv_vol_mom = 1.0 / vol_mom.replace(0, np.nan)
    inv_vol_trend = 1.0 / vol_trend.replace(0, np.nan)

    total_inv = inv_vol_mom + inv_vol_trend
    w_mom = (inv_vol_mom / total_inv).fillna(0.5)
    w_trend = (inv_vol_trend / total_inv).fillna(0.5)

    return w_mom, w_trend


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print(f"\n{'#' * 80}")
    print(f"  COMBINED GROWTH PORTFOLIO BACKTEST")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#' * 80}\n")

    # ── 1. Download all data needed ──────────────────────────────────────────
    print("[1/5] Building universe and downloading data...")

    mom_tickers = build_universe()
    all_tickers = sorted(set(mom_tickers + TREND_TICKERS + ["SPY", "AGG"]))

    print(f"  Downloading {len(all_tickers)} tickers...")
    close, high, low = download_batch(all_tickers, start=MOM_DATA_START)
    print(f"  Got data for {len(close.columns)} tickers, {len(close)} trading days")

    # SPY for benchmarks
    spy_close = close["SPY"] if "SPY" in close.columns else None
    if spy_close is None:
        spy_data = yf.download("SPY", start=MOM_DATA_START, progress=False)
        cc = "Adj Close" if "Adj Close" in spy_data.columns else "Close"
        spy_close = spy_data[cc]
        if isinstance(spy_close, pd.DataFrame):
            spy_close = spy_close.iloc[:, 0]

    # ── 2. Run momentum backtest ─────────────────────────────────────────────
    print(f"\n[2/5] Running Momentum Top-10 backtest...")
    mom_equity = run_momentum_backtest(close, high, low)
    mom_returns = mom_equity.pct_change().dropna()
    print(f"  Momentum: {len(mom_returns)} trading days, "
          f"equity ${mom_equity.iloc[0]:,.0f} -> ${mom_equity.iloc[-1]:,.0f}")

    # ── 3. Run trend backtest ────────────────────────────────────────────────
    print(f"\n[3/5] Running Trend-Following 200MA backtest...")
    trend_returns = run_trend_backtest(close)
    trend_equity = (1 + trend_returns).cumprod() * 1_000_000
    print(f"  Trend: {len(trend_returns)} trading days, "
          f"equity $1,000,000 -> ${trend_equity.iloc[-1]:,.0f}")

    # ── 4. Align series ──────────────────────────────────────────────────────
    print(f"\n[4/5] Aligning equity curves and building combined portfolios...")

    # Align to common date range
    common_idx = mom_returns.index.intersection(trend_returns.index)
    mom_r = mom_returns.loc[common_idx]
    trend_r = trend_returns.loc[common_idx]

    spy_aligned = spy_close.reindex(common_idx).dropna()
    common_idx = mom_r.index.intersection(spy_aligned.index)
    mom_r = mom_r.loc[common_idx]
    trend_r = trend_r.loc[common_idx]
    spy_r = spy_aligned.pct_change().dropna()
    common_idx = mom_r.index.intersection(spy_r.index)
    mom_r = mom_r.loc[common_idx]
    trend_r = trend_r.loc[common_idx]
    spy_r = spy_r.loc[common_idx]

    print(f"  Common period: {common_idx[0].date()} to {common_idx[-1].date()} "
          f"({len(common_idx)} days)")

    # ── Strategy correlation ──
    corr = mom_r.corr(trend_r)
    print(f"  Momentum-Trend daily return correlation: {corr:.4f}")

    # ── Build combined portfolios ──
    # Fixed-weight combos
    combos = {
        "50/50 Mom+Trend": (0.50, 0.50),
        "70/30 Mom+Trend": (0.70, 0.30),
        "30/70 Mom+Trend": (0.30, 0.70),
    }

    all_returns = {
        "Momentum Top10": mom_r,
        "Trend 200MA": trend_r,
        "SPY Buy&Hold": spy_r,
    }

    for name, (w_m, w_t) in combos.items():
        all_returns[name] = w_m * mom_r + w_t * trend_r

    # Risk parity
    w_mom_rp, w_trend_rp = risk_parity_weights(mom_r, trend_r)
    rp_returns = w_mom_rp * mom_r + w_trend_rp * trend_r
    all_returns["Risk Parity"] = rp_returns

    # ── 5. Compute metrics ───────────────────────────────────────────────────
    print(f"\n[5/5] Computing metrics...")

    all_metrics = {}
    for name, ret in all_returns.items():
        all_metrics[name] = compute_metrics(ret, name)

    # Regime analysis
    all_regime = {}
    for name, ret in all_returns.items():
        all_regime[name] = regime_analysis(ret, spy_r, name)

    # Year-by-year
    yr_table = yearly_returns_table(all_returns)

    # Risk parity avg weights
    avg_w_mom = w_mom_rp.mean()
    avg_w_trend = w_trend_rp.mean()

    # ═══════════════════════════════════════════════════════════════════════
    #  DISPLAY RESULTS
    # ═══════════════════════════════════════════════════════════════════════

    print(f"\n{'=' * 90}")
    print(f"  COMBINED PORTFOLIO BACKTEST RESULTS")
    print(f"  Period: {common_idx[0].date()} to {common_idx[-1].date()} "
          f"({len(common_idx)} days, {len(common_idx)/252:.1f} years)")
    print(f"  Momentum-Trend Correlation: {corr:.4f}")
    print(f"{'=' * 90}")

    # Summary table
    order = ["SPY Buy&Hold", "Momentum Top10", "Trend 200MA",
             "50/50 Mom+Trend", "70/30 Mom+Trend", "30/70 Mom+Trend", "Risk Parity"]

    print(f"\n  {'Strategy':<20} {'CAGR%':>7} {'Sharpe':>7} {'Sortino':>8} "
          f"{'MaxDD%':>8} {'Calmar':>7} {'Vol%':>7}")
    print(f"  {'-' * 20}-+-{'-' * 7}-+-{'-' * 7}-+-{'-' * 8}-+-"
          f"{'-' * 8}-+-{'-' * 7}-+-{'-' * 7}")

    for name in order:
        m = all_metrics[name]
        if "error" in m:
            print(f"  {name:<20} ERROR")
            continue
        print(f"  {name:<20} {m['cagr_pct']:>7.2f} {m['sharpe']:>7.3f} "
              f"{m['sortino']:>8.3f} {m['max_drawdown_pct']:>8.2f} "
              f"{m['calmar']:>7.3f} {m['annual_vol_pct']:>7.2f}")

    # Risk parity weights
    print(f"\n  Risk Parity average weights: "
          f"Momentum {avg_w_mom:.1%} / Trend {avg_w_trend:.1%}")

    # Regime analysis
    print(f"\n  {'Strategy':<20} {'GrnMo':>6} {'GrnAvg%':>8} {'GrnSh':>7} "
          f"{'RedMo':>6} {'RedAvg%':>8} {'RedSh':>7}")
    print(f"  {'-' * 20}-+-{'-' * 6}-+-{'-' * 8}-+-{'-' * 7}-+-"
          f"{'-' * 6}-+-{'-' * 8}-+-{'-' * 7}")

    for name in order:
        r = all_regime[name]
        if "error" in r:
            continue
        print(f"  {name:<20} {r['green_months']:>6} {r['green_avg_ret_pct']:>8.3f} "
              f"{r['green_sharpe']:>7.3f} {r['red_months']:>6} "
              f"{r['red_avg_ret_pct']:>8.3f} {r['red_sharpe']:>7.3f}")

    # Year-by-year
    print(f"\n{'=' * 90}")
    print(f"  YEAR-BY-YEAR RETURNS (%)")
    print(f"{'=' * 90}")
    # Reorder columns
    cols_order = [c for c in order if c in yr_table.columns]
    print(yr_table[cols_order].to_string())

    # Worst drawdowns for combined strategies
    print(f"\n{'=' * 90}")
    print(f"  WORST DRAWDOWNS (Top 3)")
    print(f"{'=' * 90}")

    for name in ["SPY Buy&Hold", "Momentum Top10", "50/50 Mom+Trend", "Risk Parity"]:
        m = all_metrics[name]
        dds = m.get("worst_drawdowns", [])
        print(f"\n  {name}:")
        for i, dd in enumerate(dds[:3], 1):
            print(f"    #{i}: {dd['depth_pct']:>7.2f}%  "
                  f"{dd['start']} -> {dd['trough']}  "
                  f"Recovery: {dd['recovery']}  ({dd['duration_days']}d)")

    # Key insights
    print(f"\n{'=' * 90}")
    print(f"  KEY INSIGHTS")
    print(f"{'=' * 90}")

    m_spy = all_metrics["SPY Buy&Hold"]
    m_mom = all_metrics["Momentum Top10"]
    m_trend = all_metrics["Trend 200MA"]
    m_5050 = all_metrics["50/50 Mom+Trend"]
    m_rp = all_metrics["Risk Parity"]

    best = max(
        [(m_5050, "50/50"), (all_metrics["70/30 Mom+Trend"], "70/30"),
         (all_metrics["30/70 Mom+Trend"], "30/70"), (m_rp, "Risk Parity")],
        key=lambda x: x[0].get("sharpe", 0)
    )

    print(f"\n  Correlation between strategies: {corr:.4f} "
          f"({'LOW - good diversification' if abs(corr) < 0.3 else 'MODERATE' if abs(corr) < 0.6 else 'HIGH - limited diversification'})")

    print(f"\n  Best risk-adjusted combined portfolio: {best[1]}")
    print(f"    Sharpe: {best[0]['sharpe']:.3f} vs Momentum {m_mom['sharpe']:.3f} "
          f"vs Trend {m_trend['sharpe']:.3f} vs SPY {m_spy['sharpe']:.3f}")
    print(f"    MaxDD:  {best[0]['max_drawdown_pct']:.2f}% vs Momentum {m_mom['max_drawdown_pct']:.2f}% "
          f"vs SPY {m_spy['max_drawdown_pct']:.2f}%")
    print(f"    CAGR:   {best[0]['cagr_pct']:.2f}% vs SPY {m_spy['cagr_pct']:.2f}%")

    dd_reduction = (1 - best[0]["max_drawdown_pct"] / m_mom["max_drawdown_pct"]) * 100
    sharpe_improvement = best[0]["sharpe"] - m_mom["sharpe"]

    print(f"\n  Combining strategies:")
    print(f"    Max DD reduction vs Momentum-only: {dd_reduction:+.1f}%")
    print(f"    Sharpe change vs Momentum-only:    {sharpe_improvement:+.3f}")
    print(f"    Diversification {'WORKS' if dd_reduction > 10 else 'LIMITED'} "
          f"— correlation {corr:.3f}")

    # ── Save results ─────────────────────────────────────────────────────────
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    datestamp = datetime.now().strftime("%Y%m%d")
    outfile = OUTPUT_DIR / f"combined_portfolio_{datestamp}.json"

    # Serialize
    output = {
        "run_date": datetime.now().isoformat(),
        "backtest_period": {
            "start": str(common_idx[0].date()),
            "end": str(common_idx[-1].date()),
            "trading_days": len(common_idx),
            "years": round(len(common_idx) / 252, 2),
        },
        "strategy_correlation": round(corr, 4),
        "risk_parity_avg_weights": {
            "momentum": round(avg_w_mom, 4),
            "trend": round(avg_w_trend, 4),
        },
        "metrics": {name: all_metrics[name] for name in order},
        "regime_analysis": {name: all_regime[name] for name in order},
        "yearly_returns": yr_table[[c for c in order if c in yr_table.columns]].to_dict(),
    }

    with open(outfile, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Results saved to {outfile}")

    print(f"\n{'#' * 80}")
    print(f"  BACKTEST COMPLETE — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#' * 80}\n")

    return output


if __name__ == "__main__":
    main()
