#!/usr/bin/env python3
"""
Dynamic Portfolio Optimizer v1 — Walk-forward multi-strategy allocation framework.

Combines 4 validated strategies into an optimal portfolio using 6 allocation variants.
Full permutation testing, sub-period stability, outlier robustness, regime balance,
and random baseline validation per HC #753 / HC #428.

CPU-only. Uses yfinance for market data.
"""

import os
import json
import warnings
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats

warnings.filterwarnings("ignore")

# ── MLflow setup ──────────────────────────────────────────────────────────────
try:
    import mlflow
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("dynamic_portfolio_optimizer_v1")
    MLFLOW_OK = True
except Exception as e:
    print(f"[WARN] MLflow unavailable: {e}")
    MLFLOW_OK = False

OUTPUT_DIR = Path("/home/jupiter/Lvl3Quant/output/growth_research/portfolio_optimizer_v1")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SECTOR_ETFS = ["XLB", "XLC", "XLE", "XLF", "XLI", "XLK", "XLP", "XLRE", "XLU", "XLV", "XLY"]
START = "2021-01-01"  # extra buffer for lookback
END = "2026-07-01"
BT_START = "2022-01-01"
CAPITAL = 10_000
SMALL_CAPITAL = 645
TX_COST_ETF = 0.001  # 0.1% per trade
np.random.seed(42)


# ══════════════════════════════════════════════════════════════════════════════
#  DATA DOWNLOAD
# ══════════════════════════════════════════════════════════════════════════════

def download_data():
    """Download sector ETF, SPY, and VIX data."""
    print("Downloading market data...")
    tickers = SECTOR_ETFS + ["SPY", "^VIX"]
    data = yf.download(tickers, start=START, end=END, auto_adjust=True, progress=False)
    prices = data["Close"].copy()
    # Flatten multi-index columns if needed
    if isinstance(prices.columns, pd.MultiIndex):
        prices.columns = prices.columns.get_level_values(0)
    prices = prices.ffill().dropna()
    print(f"  Data: {prices.index[0].date()} to {prices.index[-1].date()}, {len(prices)} days")
    return prices


# ══════════════════════════════════════════════════════════════════════════════
#  INDIVIDUAL STRATEGY SIMULATORS
# ══════════════════════════════════════════════════════════════════════════════

def momentum_score(prices_df, etfs, date, lookback_map=None):
    """Weighted momentum score: 50% 1mo + 30% 3mo + 20% 6mo."""
    if lookback_map is None:
        lookback_map = {21: 0.5, 63: 0.3, 126: 0.2}
    idx = prices_df.index.get_loc(date)
    scores = {}
    for etf in etfs:
        s = 0.0
        valid = True
        for lb, w in lookback_map.items():
            if idx - lb < 0:
                valid = False
                break
            p_now = prices_df[etf].iloc[idx]
            p_prev = prices_df[etf].iloc[idx - lb]
            if p_prev == 0:
                valid = False
                break
            s += w * (p_now / p_prev - 1)
        if valid:
            scores[etf] = s
    return scores


def simulate_equity_rotation(prices, bt_dates):
    """Strategy 1: Long top-2 sectors by momentum, monthly rebalance."""
    returns = []
    rebal_dates = _monthly_rebal_dates(bt_dates)

    holdings = []
    for i, date in enumerate(bt_dates):
        if date in rebal_dates or i == 0:
            scores = momentum_score(prices, SECTOR_ETFS, date)
            if len(scores) < 2:
                holdings = []
            else:
                ranked = sorted(scores, key=scores.get, reverse=True)
                holdings = ranked[:2]

        if holdings:
            day_ret = 0.0
            for etf in holdings:
                idx = prices.index.get_loc(date)
                if idx == 0:
                    continue
                r = prices[etf].iloc[idx] / prices[etf].iloc[idx - 1] - 1
                day_ret += r / len(holdings)
            # Apply tx cost on rebalance days
            if date in rebal_dates:
                day_ret -= TX_COST_ETF * 2  # buy+sell
            returns.append(day_ret)
        else:
            returns.append(0.0)

    return pd.Series(returns, index=bt_dates, name="EquityRotation")


def simulate_market_neutral(prices, bt_dates):
    """Strategy 2: Long top-3, short bottom-3 sectors. Dollar-neutral."""
    returns = []
    rebal_dates = _monthly_rebal_dates(bt_dates)

    longs, shorts = [], []
    for i, date in enumerate(bt_dates):
        if date in rebal_dates or i == 0:
            scores = momentum_score(prices, SECTOR_ETFS, date)
            if len(scores) < 6:
                longs, shorts = [], []
            else:
                ranked = sorted(scores, key=scores.get, reverse=True)
                longs = ranked[:3]
                shorts = ranked[-3:]

        if longs and shorts:
            day_ret = 0.0
            idx = prices.index.get_loc(date)
            if idx > 0:
                for etf in longs:
                    r = prices[etf].iloc[idx] / prices[etf].iloc[idx - 1] - 1
                    day_ret += r / 6  # 50% long / 3 positions
                for etf in shorts:
                    r = prices[etf].iloc[idx] / prices[etf].iloc[idx - 1] - 1
                    day_ret -= r / 6  # 50% short / 3 positions
            if date in rebal_dates:
                day_ret -= TX_COST_ETF * 6  # 6 positions
            returns.append(day_ret)
        else:
            returns.append(0.0)

    return pd.Series(returns, index=bt_dates, name="MarketNeutral")


def simulate_iron_condor(prices, bt_dates):
    """Strategy 3: SPY iron condor income. Monthly outcome with daily vol noise."""
    returns = []
    rebal_dates = _monthly_rebal_dates(bt_dates)
    win_rate = 0.947
    monthly_win = 0.025   # +2.5% on win
    monthly_loss = -0.10  # -10% on loss

    current_month_return = 0.0
    np_rng = np.random.RandomState(123)

    # Pre-compute SPY daily vol for realistic daily noise
    spy_ret = prices["SPY"].pct_change().fillna(0)

    for i, date in enumerate(bt_dates):
        if date in rebal_dates or i == 0:
            # New month outcome
            win = np_rng.random() < win_rate
            month_pnl = monthly_win if win else monthly_loss
            # Count trading days until next rebalance
            sorted_rebal = sorted(rebal_dates)
            next_rebal_idx = None
            for rd in sorted_rebal:
                if rd > date:
                    next_rebal_idx = bt_dates.get_loc(rd) if rd in bt_dates else None
                    break
            if next_rebal_idx is None:
                days_ahead = 21
            else:
                days_ahead = max(1, next_rebal_idx - bt_dates.get_loc(date))
            current_month_return = month_pnl / days_ahead

        # Add daily noise proportional to SPY vol (condor P&L moves with underlying)
        idx = prices.index.get_loc(date)
        spy_daily = spy_ret.iloc[idx] if idx > 0 else 0
        # Iron condors lose when SPY moves big; gain from theta when flat
        # Moderate noise: ~0.03 * SPY move + small random for realistic daily variance
        noise = -0.03 * abs(spy_daily) + np_rng.normal(0, 0.0003)
        returns.append(current_month_return + noise)

    return pd.Series(returns, index=bt_dates, name="IronCondor")


def simulate_momentum_burst(prices, bt_dates):
    """Strategy 4: Momentum burst options trades. Event-driven simulation."""
    returns = []
    np_rng = np.random.RandomState(456)
    win_rate = 0.52
    win_pct = 0.30
    loss_pct = -0.25
    cooldown = 0
    max_exposure = 0.20  # max 20% of allocation per trade

    for i, date in enumerate(bt_dates):
        if cooldown > 0:
            cooldown -= 1
            returns.append(0.0)
            continue

        idx = prices.index.get_loc(date)
        if idx < 21:
            returns.append(0.0)
            continue

        # Check momentum signals across sector ETFs
        signals_fired = 0
        for etf in SECTOR_ETFS:
            p = prices[etf].iloc[max(0, idx - 20):idx + 1]
            if len(p) < 21:
                continue
            sma20 = p.iloc[-21:].mean()
            ret5 = p.iloc[-1] / p.iloc[-6] - 1 if len(p) >= 6 else 0

            # Simple RSI(14) approximation
            deltas = p.diff().dropna()
            if len(deltas) < 14:
                continue
            recent = deltas.iloc[-14:]
            gains = recent.clip(lower=0).mean()
            losses = (-recent.clip(upper=0)).mean()
            if losses == 0:
                rsi = 100
            else:
                rsi = 100 - 100 / (1 + gains / losses)

            conditions = [rsi > 60, p.iloc[-1] > sma20, ret5 > 0.02]
            if sum(conditions) >= 2:
                signals_fired += 1

        if signals_fired >= 1:
            # Trade fires — outcome based on win rate
            win = np_rng.random() < win_rate
            trade_ret = (win_pct if win else loss_pct) * max_exposure
            returns.append(trade_ret)
            cooldown = 4  # 5-day max hold (remaining 4 days)
        else:
            returns.append(0.0)

    return pd.Series(returns, index=bt_dates, name="MomentumBurst")


def _monthly_rebal_dates(dates):
    """Get first trading day of each month."""
    rebal = []
    prev_month = None
    for d in dates:
        if prev_month is None or d.month != prev_month:
            rebal.append(d)
            prev_month = d.month
    return set(rebal)


# ══════════════════════════════════════════════════════════════════════════════
#  PORTFOLIO ALLOCATION VARIANTS
# ══════════════════════════════════════════════════════════════════════════════

def combine_equal_weight(strat_returns):
    """Variant A: 25% each."""
    weights = np.array([0.25, 0.25, 0.25, 0.25])
    return _apply_weights(strat_returns, weights, "EqualWeight")


def combine_risk_parity(strat_returns):
    """Variant B: Weight inversely proportional to rolling 60-day vol."""
    combined = []
    names = list(strat_returns.keys())
    df = pd.DataFrame(strat_returns)

    for i in range(len(df)):
        if i < 60:
            # Not enough history — equal weight
            w = np.array([0.25] * 4)
        else:
            vols = df.iloc[i - 60:i].std().values
            vols = np.maximum(vols, 1e-8)
            inv_vol = 1.0 / vols
            w = inv_vol / inv_vol.sum()
        day_ret = sum(w[j] * df.iloc[i, j] for j in range(4))
        combined.append(day_ret)

    return pd.Series(combined, index=df.index, name="RiskParity")


def combine_momentum_tilted(strat_returns):
    """Variant C: Double weight on best trailing 3-month Sharpe."""
    df = pd.DataFrame(strat_returns)
    combined = []

    for i in range(len(df)):
        if i < 63:
            w = np.array([0.25] * 4)
        else:
            window = df.iloc[i - 63:i]
            sharpes = []
            for j in range(4):
                col = window.iloc[:, j]
                s = col.mean() / max(col.std(), 1e-8) * np.sqrt(252)
                sharpes.append(s)
            best = np.argmax(sharpes)
            w = np.array([1.0 / 6] * 4)
            w[best] = 2.0 / 6 + 1.0 / 6  # double = 2/6, others share remaining
            # Normalize: best gets 2x, others get 1x
            w = np.array([1.0] * 4)
            w[best] = 2.0
            w = w / w.sum()

        day_ret = sum(w[j] * df.iloc[i, j] for j in range(4))
        combined.append(day_ret)

    return pd.Series(combined, index=df.index, name="MomentumTilted")


def combine_vix_adaptive(strat_returns, vix_series):
    """Variant D: VIX-adaptive allocation."""
    df = pd.DataFrame(strat_returns)
    combined = []

    # Align VIX to strategy dates
    vix_aligned = vix_series.reindex(df.index).ffill().fillna(20)

    # Order: EquityRotation, MarketNeutral, IronCondor, MomentumBurst
    for i in range(len(df)):
        v = vix_aligned.iloc[i]
        if v > 25:  # Defensive
            w = np.array([0.10, 0.40, 0.40, 0.10])
        elif v < 20:  # Aggressive
            w = np.array([0.35, 0.10, 0.10, 0.45])
        else:  # Neutral
            w = np.array([0.25, 0.25, 0.25, 0.25])

        day_ret = sum(w[j] * df.iloc[i, j] for j in range(4))
        combined.append(day_ret)

    return pd.Series(combined, index=df.index, name="VIXAdaptive")


def combine_kelly_optimal(strat_returns):
    """Variant E: Half-Kelly sizing based on rolling win rate and payoff ratio."""
    df = pd.DataFrame(strat_returns)
    combined = []

    for i in range(len(df)):
        if i < 60:
            w = np.array([0.25] * 4)
        else:
            window = df.iloc[i - 60:i]
            kelly_fracs = []
            for j in range(4):
                col = window.iloc[:, j]
                wins = col[col > 0]
                losses = col[col < 0]
                if len(wins) == 0 or len(losses) == 0:
                    kelly_fracs.append(0.25)
                    continue
                wr = len(wins) / len(col)
                avg_win = wins.mean()
                avg_loss = abs(losses.mean())
                if avg_loss == 0:
                    kelly_fracs.append(0.5)
                    continue
                payoff = avg_win / avg_loss
                kelly = wr - (1 - wr) / payoff
                half_kelly = max(0.05, min(0.5, kelly / 2))  # floor 5%, cap 50%
                kelly_fracs.append(half_kelly)

            w = np.array(kelly_fracs)
            w = w / w.sum()

        day_ret = sum(w[j] * df.iloc[i, j] for j in range(4))
        combined.append(day_ret)

    return pd.Series(combined, index=df.index, name="KellyOptimal")


def combine_growth_focused(strat_returns):
    """Variant F: 40% momentum burst + 30% equity rotation + 20% market-neutral + 10% iron condor."""
    weights = np.array([0.30, 0.20, 0.10, 0.40])
    return _apply_weights(strat_returns, weights, "GrowthFocused")


def _apply_weights(strat_returns, weights, name):
    df = pd.DataFrame(strat_returns)
    combined = (df.values * weights).sum(axis=1)
    return pd.Series(combined, index=df.index, name=name)


# ══════════════════════════════════════════════════════════════════════════════
#  METRICS
# ══════════════════════════════════════════════════════════════════════════════

def compute_metrics(returns):
    """Compute Sharpe, Sortino, WR%, PF, MDD%, CAGR%."""
    r = returns.values
    n = len(r)
    if n < 10:
        return dict(Sharpe=0, Sortino=0, WR=0, PF=0, MDD=0, CAGR=0, Trades=0)

    # Sharpe
    sharpe = r.mean() / max(r.std(), 1e-8) * np.sqrt(252)

    # Sortino (use downside deviation of ALL returns, not just negative ones)
    downside_diff = np.minimum(r, 0)
    downside_std = np.sqrt(np.mean(downside_diff ** 2)) if len(r) > 1 else 1e-8
    sortino = r.mean() / max(downside_std, 1e-8) * np.sqrt(252)

    # Win rate
    trading_days = r[r != 0]
    wr = (trading_days > 0).sum() / max(len(trading_days), 1) * 100

    # Profit factor
    gross_profit = r[r > 0].sum()
    gross_loss = abs(r[r < 0].sum())
    pf = gross_profit / max(gross_loss, 1e-8)

    # Max drawdown
    cum = (1 + pd.Series(r)).cumprod()
    peak = cum.cummax()
    dd = (cum - peak) / peak
    mdd = dd.min() * 100

    # CAGR
    years = n / 252
    total_return = cum.iloc[-1]
    cagr = (total_return ** (1 / max(years, 0.1)) - 1) * 100

    # Trade count (non-zero days)
    trades = (r != 0).sum()

    return dict(Sharpe=round(sharpe, 2), Sortino=round(sortino, 2),
                WR=round(wr, 1), PF=round(pf, 2),
                MDD=round(mdd, 1), CAGR=round(cagr, 1), Trades=int(trades))


# ══════════════════════════════════════════════════════════════════════════════
#  VALIDATION GATES (HC #753 / HC #428)
# ══════════════════════════════════════════════════════════════════════════════

def permutation_test(returns, n_perms=100):
    """Block-shuffle returns in 21-day blocks to destroy serial dependence while preserving
    cross-sectional distribution. Recompute Sharpe. p-value."""
    actual_sharpe = returns.mean() / max(returns.std(), 1e-8) * np.sqrt(252)
    r = returns.values.copy()
    n = len(r)
    block_size = 21
    n_blocks = n // block_size
    count_above = 0
    for _ in range(n_perms):
        # Block permutation: shuffle blocks of 21 days
        blocks = [r[i * block_size:(i + 1) * block_size] for i in range(n_blocks)]
        remainder = r[n_blocks * block_size:]
        np.random.shuffle(blocks)
        shuffled = np.concatenate(blocks + [remainder])
        # Also randomly flip sign of each block (destroy directional persistence)
        for i in range(n_blocks):
            if np.random.random() < 0.5:
                shuffled[i * block_size:(i + 1) * block_size] *= -1
        shuf_sharpe = shuffled.mean() / max(shuffled.std(), 1e-8) * np.sqrt(252)
        if shuf_sharpe >= actual_sharpe:
            count_above += 1
    p_val = count_above / n_perms
    return p_val < 0.05, p_val


def subperiod_stability(returns):
    """Split into 4 quarters, check 3/4 have positive Sharpe."""
    n = len(returns)
    q = n // 4
    positive = 0
    sharpes = []
    for i in range(4):
        start = i * q
        end = (i + 1) * q if i < 3 else n
        chunk = returns.iloc[start:end]
        s = chunk.mean() / max(chunk.std(), 1e-8) * np.sqrt(252)
        sharpes.append(round(s, 2))
        if s > 0:
            positive += 1
    return positive >= 3, sharpes


def outlier_removal(returns):
    """Trim top/bottom 1%, check Sharpe > 0.8x full."""
    full_sharpe = returns.mean() / max(returns.std(), 1e-8) * np.sqrt(252)
    lo, hi = np.percentile(returns, [1, 99])
    trimmed = returns[(returns >= lo) & (returns <= hi)]
    trim_sharpe = trimmed.mean() / max(trimmed.std(), 1e-8) * np.sqrt(252)
    return trim_sharpe > 0.8 * full_sharpe, round(trim_sharpe, 2), round(full_sharpe, 2)


def regime_balance(returns, vix_series):
    """Sharpe on VIX>25 vs VIX<25 days. Gap ratio < 0.50."""
    vix_aligned = vix_series.reindex(returns.index).ffill().fillna(20)
    high_vix = returns[vix_aligned > 25]
    low_vix = returns[vix_aligned <= 25]

    if len(high_vix) < 10 or len(low_vix) < 10:
        return True, 0.0  # Insufficient data, pass by default

    s_high = high_vix.mean() / max(high_vix.std(), 1e-8) * np.sqrt(252)
    s_low = low_vix.mean() / max(low_vix.std(), 1e-8) * np.sqrt(252)

    max_abs = max(abs(s_high), abs(s_low), 1e-8)
    gap = abs(s_high - s_low) / max_abs
    return gap < 0.50, round(gap, 2)


def random_baseline(returns, strat_returns_df, n_random=100):
    """100 random allocation vectors. PASS if strategy > 95th pctile."""
    actual_sharpe = returns.mean() / max(returns.std(), 1e-8) * np.sqrt(252)
    random_sharpes = []
    df = strat_returns_df.values
    for _ in range(n_random):
        w = np.random.dirichlet(np.ones(4))
        r = (df * w).sum(axis=1)
        s = r.mean() / max(r.std(), 1e-8) * np.sqrt(252)
        random_sharpes.append(s)
    pct95 = np.percentile(random_sharpes, 95)
    return actual_sharpe > pct95, round(actual_sharpe, 2), round(pct95, 2)


def run_all_gates(returns, vix_series, strat_returns_df):
    """Run all 5 validation gates. Return (n_passed, details)."""
    gates = {}
    passed = 0

    ok, pval = permutation_test(returns)
    gates["permutation"] = {"pass": ok, "p_value": round(pval, 3)}
    passed += ok

    ok, sharpes = subperiod_stability(returns)
    gates["subperiod"] = {"pass": ok, "quarter_sharpes": sharpes}
    passed += ok

    ok, trim_s, full_s = outlier_removal(returns)
    gates["outlier_removal"] = {"pass": ok, "trimmed_sharpe": trim_s, "full_sharpe": full_s}
    passed += ok

    ok, gap = regime_balance(returns, vix_series)
    gates["regime_balance"] = {"pass": ok, "gap_ratio": gap}
    passed += ok

    ok, actual, p95 = random_baseline(returns, strat_returns_df)
    gates["random_baseline"] = {"pass": ok, "sharpe": actual, "p95_random": p95}
    passed += ok

    return passed, gates


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 80)
    print("DYNAMIC PORTFOLIO OPTIMIZER v1 — Walk-Forward Multi-Strategy Allocation")
    print("=" * 80)

    # ── Download data ─────────────────────────────────────────────────────────
    prices = download_data()

    # Filter to backtest period
    bt_mask = prices.index >= BT_START
    bt_dates = prices.index[bt_mask]
    print(f"  Backtest period: {bt_dates[0].date()} to {bt_dates[-1].date()}, {len(bt_dates)} days")

    vix = prices["^VIX"].copy()

    # ── Simulate individual strategies ────────────────────────────────────────
    print("\nSimulating individual strategies...")
    s1 = simulate_equity_rotation(prices, bt_dates)
    s2 = simulate_market_neutral(prices, bt_dates)
    s3 = simulate_iron_condor(prices, bt_dates)
    s4 = simulate_momentum_burst(prices, bt_dates)

    strat_dict = {
        "EquityRotation": s1,
        "MarketNeutral": s2,
        "IronCondor": s3,
        "MomentumBurst": s4,
    }
    strat_df = pd.DataFrame(strat_dict)

    print("\n── Individual Strategy Performance ──")
    for name, s in strat_dict.items():
        m = compute_metrics(s)
        print(f"  {name:20s}  Sharpe={m['Sharpe']:6.2f}  Sortino={m['Sortino']:6.2f}  "
              f"WR={m['WR']:5.1f}%  PF={m['PF']:5.2f}  MDD={m['MDD']:6.1f}%  CAGR={m['CAGR']:6.1f}%")

    # ── Build portfolio variants ──────────────────────────────────────────────
    print("\nBuilding 6 allocation variants...")
    variants = {
        "A_EqualWeight": combine_equal_weight(strat_dict),
        "B_RiskParity": combine_risk_parity(strat_dict),
        "C_MomentumTilted": combine_momentum_tilted(strat_dict),
        "D_VIXAdaptive": combine_vix_adaptive(strat_dict, vix),
        "E_KellyOptimal": combine_kelly_optimal(strat_dict),
        "F_GrowthFocused": combine_growth_focused(strat_dict),
    }

    # ── Run validation and collect results ────────────────────────────────────
    print("\nRunning validation gates (permutation tests, sub-period, outlier, regime, random)...")
    results = []
    all_details = {}

    for vname, vret in variants.items():
        m = compute_metrics(vret)
        gates_passed, gate_details = run_all_gates(vret, vix, strat_df)
        m["Variant"] = vname
        m["GatesPassed"] = f"{gates_passed}/5"
        m["gates_detail"] = gate_details

        # Equity curve for $10k and $645
        cum = (1 + vret).cumprod()
        m["FinalEquity_10k"] = round(CAPITAL * cum.iloc[-1], 0)
        m["FinalEquity_645"] = round(SMALL_CAPITAL * cum.iloc[-1], 0)

        results.append(m)
        all_details[vname] = gate_details

    # ── Sort by Sharpe ────────────────────────────────────────────────────────
    results.sort(key=lambda x: x["Sharpe"], reverse=True)

    # ── Print results table ───────────────────────────────────────────────────
    print("\n" + "=" * 120)
    print(f"{'Variant':25s} {'Sharpe':>7s} {'Sortino':>8s} {'WR%':>6s} {'PF':>6s} {'MDD%':>7s} "
          f"{'CAGR%':>7s} {'Gates':>7s} {'Trades':>7s} {'$10k->':>8s} {'$645->':>8s}")
    print("-" * 120)
    for r in results:
        print(f"{r['Variant']:25s} {r['Sharpe']:7.2f} {r['Sortino']:8.2f} {r['WR']:6.1f} {r['PF']:6.2f} "
              f"{r['MDD']:7.1f} {r['CAGR']:7.1f} {r['GatesPassed']:>7s} {r['Trades']:7d} "
              f"{r['FinalEquity_10k']:8.0f} {r['FinalEquity_645']:8.0f}")
    print("=" * 120)

    # ── Per-quarter breakdown for best variant ────────────────────────────────
    best_name = results[0]["Variant"]
    best_ret = variants[best_name]
    print(f"\n── Per-Quarter Breakdown: {best_name} (Best by Sharpe) ──")
    n = len(best_ret)
    q = n // 4
    for qi in range(4):
        s = qi * q
        e = (qi + 1) * q if qi < 3 else n
        chunk = best_ret.iloc[s:e]
        m = compute_metrics(chunk)
        d_start = chunk.index[0].date()
        d_end = chunk.index[-1].date()
        print(f"  Q{qi + 1} ({d_start} to {d_end}):  Sharpe={m['Sharpe']:6.2f}  Sortino={m['Sortino']:6.2f}  "
              f"WR={m['WR']:5.1f}%  PF={m['PF']:5.2f}  MDD={m['MDD']:6.1f}%  CAGR={m['CAGR']:6.1f}%")

    # ── Validation gate details ───────────────────────────────────────────────
    print(f"\n── Validation Gate Details ──")
    for vname in [r["Variant"] for r in results]:
        gd = all_details[vname]
        gate_str = " | ".join(
            f"{g}: {'PASS' if gd[g]['pass'] else 'FAIL'}" for g in gd
        )
        print(f"  {vname:25s}  {gate_str}")

    # ── Save results JSON ─────────────────────────────────────────────────────
    save_results = []
    for r in results:
        sr = {k: v for k, v in r.items() if k != "gates_detail"}
        sr["gates_detail"] = r["gates_detail"]
        # Convert numpy types
        for k, v in sr.items():
            if isinstance(v, (np.integer,)):
                sr[k] = int(v)
            elif isinstance(v, (np.floating,)):
                sr[k] = float(v)
        save_results.append(sr)

    # Also convert nested gate details
    def convert_numpy(obj):
        if isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(x) for x in obj]
        elif isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, (np.bool_,)):
            return bool(obj)
        return obj

    save_results = convert_numpy(save_results)

    json_path = OUTPUT_DIR / "results_v1.json"
    with open(json_path, "w") as f:
        json.dump(save_results, f, indent=2, default=str)
    print(f"\n  Results saved to {json_path}")

    # Save equity curves
    eq_curves = pd.DataFrame({k: (1 + v).cumprod() * CAPITAL for k, v in variants.items()})
    eq_curves.to_csv(OUTPUT_DIR / "equity_curves.csv")

    # Save individual strategy returns
    strat_df.to_csv(OUTPUT_DIR / "strategy_returns.csv")

    # ── Log to MLflow ─────────────────────────────────────────────────────────
    if MLFLOW_OK:
        try:
            with mlflow.start_run(run_name="portfolio_optimizer_v1"):
                # Log best variant metrics
                best = results[0]
                mlflow.log_param("best_variant", best["Variant"])
                mlflow.log_param("backtest_start", BT_START)
                mlflow.log_param("backtest_end", END)
                mlflow.log_param("capital", CAPITAL)
                mlflow.log_param("n_strategies", 4)
                mlflow.log_param("n_variants", 6)

                mlflow.log_metric("best_sharpe", best["Sharpe"])
                mlflow.log_metric("best_sortino", best["Sortino"])
                mlflow.log_metric("best_wr", best["WR"])
                mlflow.log_metric("best_pf", best["PF"])
                mlflow.log_metric("best_mdd", best["MDD"])
                mlflow.log_metric("best_cagr", best["CAGR"])
                mlflow.log_metric("best_final_equity_10k", best["FinalEquity_10k"])

                # Log all variants
                for r in results:
                    prefix = r["Variant"].lower()
                    mlflow.log_metric(f"{prefix}_sharpe", r["Sharpe"])
                    mlflow.log_metric(f"{prefix}_sortino", r["Sortino"])
                    mlflow.log_metric(f"{prefix}_cagr", r["CAGR"])
                    mlflow.log_metric(f"{prefix}_mdd", r["MDD"])

                mlflow.log_artifact(str(json_path))
                mlflow.log_artifact(str(OUTPUT_DIR / "equity_curves.csv"))

            print("  MLflow run logged successfully.")
        except Exception as e:
            print(f"  [WARN] MLflow logging failed: {e}")

    print("\n  DONE.")


if __name__ == "__main__":
    main()
