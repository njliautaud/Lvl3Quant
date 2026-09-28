#!/usr/bin/env python3
"""
REIT Mean Reversion Backtest
============================
Test Dual Signal D logic on REITs — high-dividend real estate trusts
with different return drivers than tech/consumer stocks.

Variants:
  A: Large-cap REITs — Dual Signal D. Hold 10 days.
  B: REIT ETFs — Dual Signal D. Hold 10 days.
  C: REITs + quality stocks combined — Dual Signal D. Hold 10 days.
  D: REIT-specific: drop >5% AND TLT up >2% in 5d (rates falling). Hold 10 days.
  E: Rate-filtered: Dual Signal D on REITs ONLY when TLT > 50-SMA. Hold 10 days.
  F: High-yield REIT dip: O/SPG/VNQ, 3% dip + RSI<40 + green after 2 red. Hold 15 days.

5-Gate Validation + Permutation Test (1000 iterations)
Period: 2022-01-01 to 2026-07-31 | Capital: $645 | Max $200/trade | Max 3 concurrent
"""

import json
import warnings
import sys
from datetime import datetime
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
np.random.seed(42)

try:
    import yfinance as yf
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "yfinance", "-q"])
    import yfinance as yf

# ── Configuration ─────────────────────────────────────────────────────────
CAPITAL = 645.0
MAX_PER_TRADE = 200.0
MAX_CONCURRENT = 3
SLIPPAGE_PCT = 0.0002  # 2bps each way
DATA_START = "2020-01-01"  # extra lookback for indicators
OOT_START = "2022-01-01"
END = "2026-07-31"
N_PERM = 1000

# Universe definitions
LARGE_CAP_REITS = ["O", "AMT", "PLD", "EQIX", "SPG", "PSA", "CCI", "DLR"]
REIT_ETFS = ["VNQ", "XLRE", "IYR", "RWR"]
QUALITY_STOCKS = ["AAPL", "MSFT", "JPM", "JNJ", "PG", "KO", "PEP", "HD", "COST", "UNH"]
HIGH_YIELD_REITS = ["O", "SPG", "VNQ"]

ALL_SYMBOLS = sorted(set(
    LARGE_CAP_REITS + REIT_ETFS + QUALITY_STOCKS + HIGH_YIELD_REITS +
    ["SPY", "TLT"]
))

GATES = {
    "sharpe_min": 0.5,
    "perm_p_max": 0.05,
    "regime_gap_max": 0.5,
    "mdd_min": -50.0,
    "min_trades": 20,
}

# ── Data Download ─────────────────────────────────────────────────────────
print("Downloading price data ...")
raw = yf.download(ALL_SYMBOLS, start=DATA_START, end=END,
                  group_by="ticker", auto_adjust=True, progress=False)


def get_close(ticker):
    try:
        if len(ALL_SYMBOLS) == 1:
            s = raw["Close"].dropna()
        else:
            s = raw[ticker]["Close"].dropna()
        if isinstance(s, pd.DataFrame):
            s = s.iloc[:, 0]
        return s
    except Exception:
        return pd.Series(dtype=float)


closes = {}
for t in set(LARGE_CAP_REITS + REIT_ETFS + QUALITY_STOCKS + HIGH_YIELD_REITS):
    closes[t] = get_close(t)

spy_close = get_close("SPY")
tlt_close = get_close("TLT")

loaded = sum(1 for t in closes if len(closes[t]) > 100)
print(f"  Tickers with data: {loaded}/{len(closes)}")
print(f"  SPY rows: {len(spy_close)}, TLT rows: {len(tlt_close)}")

# ── Indicator Helpers ─────────────────────────────────────────────────────
def calc_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def rolling_high(series, period=20):
    return series.rolling(period).max()


def consecutive_red_days(series):
    """Count consecutive red (close < prior close) days ending at each date."""
    is_red = (series.diff() < 0).astype(int)
    result = pd.Series(0, index=series.index, dtype=int)
    count = 0
    for i in range(len(is_red)):
        if is_red.iloc[i] == 1:
            count += 1
        else:
            count = 0
        result.iloc[i] = count
    return result


def is_green_day(series):
    return series.diff() > 0


# ── Pre-compute indicators ───────────────────────────────────────────────
print("Computing indicators ...")
indicators = {}
for t in closes:
    c = closes[t]
    if len(c) < 50:
        continue
    indicators[t] = {
        "rsi14": calc_rsi(c, 14),
        "high20": rolling_high(c, 20),
        "consec_red": consecutive_red_days(c),
        "green": is_green_day(c),
    }

spy_sma200 = spy_close.rolling(200).mean()
tlt_sma50 = tlt_close.rolling(50).mean()
tlt_ret5d = tlt_close.pct_change(5)  # 5-day return of TLT

oot_start_ts = pd.Timestamp(OOT_START)

# ── Signal Generators ─────────────────────────────────────────────────────

def generate_dual_signal_d(universe):
    """Dual Signal D: QMR-A AND Recovery-E must both fire on same date+ticker.
    QMR-A: drop >5% from 20d high AND RSI(14) < 35
    Recovery-E: first green day after 3+ consecutive red days, post 5% drawdown
    """
    qmr_a = set()
    recovery_e = set()

    for t in universe:
        if t not in indicators:
            continue
        c = closes[t]
        rsi = indicators[t]["rsi14"]
        h20 = indicators[t]["high20"]
        consec = indicators[t]["consec_red"]
        green = indicators[t]["green"]

        for i in range(1, len(c)):
            date = c.index[i]
            if date < oot_start_ts:
                continue
            try:
                price = float(c.iloc[i])
                high = float(h20.iloc[i])
                r = float(rsi.iloc[i])
                if pd.isna(price) or pd.isna(high) or pd.isna(r) or high == 0:
                    continue

                drop_pct = (price - high) / high

                # QMR-A check
                if drop_pct < -0.05 and r < 35:
                    qmr_a.add((date, t))

                # Recovery-E check
                if green.iloc[i] and consec.iloc[i-1] >= 3 and drop_pct < -0.05:
                    recovery_e.add((date, t))
            except (KeyError, IndexError):
                continue

    # Intersection = Dual Signal D
    intersection = sorted(qmr_a & recovery_e)
    return intersection, len(qmr_a), len(recovery_e)


def generate_reit_rate_signal(universe):
    """Variant D: REIT drops >5% AND TLT up >2% in last 5 days (rates falling)."""
    signals = []
    for t in universe:
        if t not in indicators:
            continue
        c = closes[t]
        h20 = indicators[t]["high20"]

        for i in range(1, len(c)):
            date = c.index[i]
            if date < oot_start_ts:
                continue
            try:
                price = float(c.iloc[i])
                high = float(h20.iloc[i])
                if pd.isna(price) or pd.isna(high) or high == 0:
                    continue
                drop_pct = (price - high) / high
                if drop_pct >= -0.05:
                    continue

                # TLT up >2% in last 5 days
                tlt_r = tlt_ret5d.asof(date)
                if pd.isna(tlt_r) or tlt_r <= 0.02:
                    continue

                signals.append((date, t))
            except (KeyError, IndexError):
                continue
    return sorted(signals)


def generate_rate_filtered_dual_d(universe):
    """Variant E: Dual Signal D ONLY when TLT > 50-SMA (falling rate env)."""
    base_signals, _, _ = generate_dual_signal_d(universe)
    filtered = []
    for date, t in base_signals:
        try:
            tlt_val = float(tlt_close.asof(date))
            sma_val = float(tlt_sma50.asof(date))
            if pd.isna(tlt_val) or pd.isna(sma_val):
                continue
            if tlt_val > sma_val:
                filtered.append((date, t))
        except Exception:
            continue
    return filtered


def generate_high_yield_dip(universe):
    """Variant F: 3% dip from 20d high + RSI<40 + green after 2+ red. Hold 15."""
    signals = []
    for t in universe:
        if t not in indicators:
            continue
        c = closes[t]
        rsi = indicators[t]["rsi14"]
        h20 = indicators[t]["high20"]
        consec = indicators[t]["consec_red"]
        green = indicators[t]["green"]

        for i in range(1, len(c)):
            date = c.index[i]
            if date < oot_start_ts:
                continue
            try:
                price = float(c.iloc[i])
                high = float(h20.iloc[i])
                r = float(rsi.iloc[i])
                if pd.isna(price) or pd.isna(high) or pd.isna(r) or high == 0:
                    continue
                drop_pct = (price - high) / high
                if drop_pct >= -0.03:
                    continue
                if r >= 40:
                    continue
                if not green.iloc[i]:
                    continue
                if consec.iloc[i-1] < 2:
                    continue
                signals.append((date, t))
            except (KeyError, IndexError):
                continue
    return sorted(signals)


# ── Generate all variant signals ──────────────────────────────────────────
print("Generating signals ...")

# Variant A: Large-cap REITs, Dual Signal D, hold 10
sig_a, qmr_a_count_a, rec_e_count_a = generate_dual_signal_d(LARGE_CAP_REITS)
signals_a = [(d, t, 10) for d, t in sig_a]

# Variant B: REIT ETFs, Dual Signal D, hold 10
sig_b, qmr_a_count_b, rec_e_count_b = generate_dual_signal_d(REIT_ETFS)
signals_b = [(d, t, 10) for d, t in sig_b]

# Variant C: REITs + quality stocks, Dual Signal D, hold 10
combined_universe = list(set(LARGE_CAP_REITS + QUALITY_STOCKS))
sig_c, qmr_a_count_c, rec_e_count_c = generate_dual_signal_d(combined_universe)
signals_c = [(d, t, 10) for d, t in sig_c]

# Variant D: REIT-specific rate signal, hold 10
sig_d = generate_reit_rate_signal(LARGE_CAP_REITS)
signals_d = [(d, t, 10) for d, t in sig_d]

# Variant E: Rate-filtered Dual D on REITs, hold 10
sig_e = generate_rate_filtered_dual_d(LARGE_CAP_REITS)
signals_e = [(d, t, 10) for d, t in sig_e]

# Variant F: High-yield REIT dip, hold 15
sig_f = generate_high_yield_dip(HIGH_YIELD_REITS)
signals_f = [(d, t, 15) for d, t in sig_f]

print(f"  A (Large-cap REITs Dual D): {len(signals_a)} signals (QMR-A: {qmr_a_count_a}, Rec-E: {rec_e_count_a})")
print(f"  B (REIT ETFs Dual D): {len(signals_b)} signals (QMR-A: {qmr_a_count_b}, Rec-E: {rec_e_count_b})")
print(f"  C (REITs+Quality Dual D): {len(signals_c)} signals (QMR-A: {qmr_a_count_c}, Rec-E: {rec_e_count_c})")
print(f"  D (REIT rate signal): {len(signals_d)} signals")
print(f"  E (Rate-filtered Dual D): {len(signals_e)} signals")
print(f"  F (High-yield dip): {len(signals_f)} signals")


# ── Trade Simulator ──────────────────────────────────────────────────────
def simulate_trades(signals, capital=CAPITAL, max_per_trade=MAX_PER_TRADE,
                    max_concurrent=MAX_CONCURRENT):
    """Simulate trades with position limits and concurrent position tracking."""
    if not signals:
        return []

    signals = sorted(signals, key=lambda x: x[0])

    # Deduplicate: no same-ticker entry within hold period
    deduped = []
    last_entry = {}
    for date, ticker, hold_days in signals:
        if ticker in last_entry:
            delta = (date - last_entry[ticker]).days
            if delta < hold_days:
                continue
        deduped.append((date, ticker, hold_days))
        last_entry[ticker] = date

    trades = []
    open_positions = []

    for date, ticker, hold_days in deduped:
        # Close expired positions
        open_positions = [(ed, tk) for ed, tk in open_positions if ed > date]

        if len(open_positions) >= max_concurrent:
            continue

        c = closes.get(ticker, pd.Series(dtype=float))
        if len(c) == 0:
            continue
        try:
            loc = c.index.get_loc(date)
        except KeyError:
            mask = c.index >= date
            if mask.sum() == 0:
                continue
            loc = c.index.get_loc(c.index[mask][0])

        exit_loc = min(loc + hold_days, len(c) - 1)
        entry_price = float(c.iloc[loc]) * (1 + SLIPPAGE_PCT)
        exit_price = float(c.iloc[exit_loc]) * (1 - SLIPPAGE_PCT)
        exit_date = c.index[exit_loc]

        if entry_price <= 0:
            continue

        shares = max_per_trade / entry_price
        pnl = shares * (exit_price - entry_price)
        ret = (exit_price - entry_price) / entry_price

        trades.append({
            "ticker": ticker,
            "entry_date": str(c.index[loc].date()),
            "exit_date": str(exit_date.date()),
            "entry_price": round(entry_price, 2),
            "exit_price": round(exit_price, 2),
            "pnl": round(float(pnl), 2),
            "return": round(float(ret), 6),
            "hold_days": hold_days,
            "shares": round(float(shares), 4),
        })

        open_positions.append((exit_date, ticker))

    return trades


# ── Metrics Calculation ──────────────────────────────────────────────────
def calc_metrics(trades, capital=CAPITAL):
    if not trades:
        return {
            "total_return_pct": 0, "sharpe": 0, "sortino": 0,
            "max_drawdown_pct": 0, "win_rate": 0, "profit_factor": 0,
            "n_trades": 0, "total_pnl": 0, "avg_return_pct": 0,
        }

    returns = np.array([t["return"] for t in trades])
    pnls = np.array([t["pnl"] for t in trades])
    total_pnl = pnls.sum()

    wins = (pnls > 0).sum()
    win_rate = wins / len(pnls)

    gross_profit = pnls[pnls > 0].sum() if (pnls > 0).any() else 0
    gross_loss = abs(pnls[pnls < 0].sum()) if (pnls < 0).any() else 1e-9
    profit_factor = gross_profit / gross_loss

    oot_years = 4.5  # Jan 2022 - Jul 2026
    if len(returns) > 1 and returns.std() > 0:
        trades_per_year = max(len(returns) / oot_years, 1)
        sharpe = (returns.mean() / returns.std()) * np.sqrt(trades_per_year)
    else:
        sharpe = 0.0

    downside = returns[returns < 0]
    if len(downside) > 1 and downside.std() > 0:
        trades_per_year = max(len(returns) / oot_years, 1)
        sortino = (returns.mean() / downside.std()) * np.sqrt(trades_per_year)
    else:
        sortino = sharpe * 1.5 if sharpe > 0 else 0.0

    cum_pnl = np.cumsum(pnls)
    peak = np.maximum.accumulate(cum_pnl + capital)
    drawdowns = (cum_pnl + capital - peak) / peak
    max_dd = drawdowns.min() * 100 if len(drawdowns) > 0 else 0

    return {
        "total_return_pct": round(total_pnl / capital * 100, 2),
        "sharpe": round(float(sharpe), 3),
        "sortino": round(float(sortino), 3),
        "max_drawdown_pct": round(float(max_dd), 2),
        "win_rate": round(float(win_rate), 4),
        "profit_factor": round(float(profit_factor), 3),
        "n_trades": len(trades),
        "total_pnl": round(float(total_pnl), 2),
        "avg_return_pct": round(float(returns.mean() * 100), 3),
    }


def regime_stratified_sharpe(trades):
    """Split trades into bull/bear based on SPY vs 200-SMA."""
    if not trades:
        return {"bull_sharpe": 0, "bear_sharpe": 0, "regime_gap": 0,
                "bull_trades": 0, "bear_trades": 0}

    bull_rets, bear_rets = [], []
    for t in trades:
        entry = pd.Timestamp(t["entry_date"])
        try:
            spy_val = spy_close.asof(entry)
            sma_val = spy_sma200.asof(entry)
            if pd.isna(spy_val) or pd.isna(sma_val):
                continue
            if spy_val > sma_val:
                bull_rets.append(t["return"])
            else:
                bear_rets.append(t["return"])
        except Exception:
            continue

    def _sharpe(rets):
        if len(rets) < 2:
            return 0.0
        arr = np.array(rets)
        if arr.std() == 0:
            return 0.0
        tpy = max(len(arr) / 4.5, 1)
        return float((arr.mean() / arr.std()) * np.sqrt(tpy))

    bull_s = _sharpe(bull_rets)
    bear_s = _sharpe(bear_rets)
    max_abs = max(abs(bull_s), abs(bear_s), 1e-9)
    gap = abs(bull_s - bear_s) / max_abs

    return {
        "bull_sharpe": round(bull_s, 3),
        "bear_sharpe": round(bear_s, 3),
        "regime_gap": round(gap, 3),
        "bull_trades": len(bull_rets),
        "bear_trades": len(bear_rets),
    }


def permutation_test(trades, n_perm=N_PERM):
    """Shuffle signal timing to test if returns are due to signal or luck."""
    if len(trades) < 5:
        return 1.0

    real_returns = np.array([t["return"] for t in trades])
    real_mean = real_returns.mean()

    ticker_dates = {}
    for t in trades:
        tk = t["ticker"]
        if tk not in ticker_dates:
            c = closes.get(tk, pd.Series(dtype=float))
            valid = c.index[c.index >= oot_start_ts]
            if len(valid) > 20:
                ticker_dates[tk] = valid

    count_better = 0
    for _ in range(n_perm):
        perm_returns = []
        for t in trades:
            tk = t["ticker"]
            hold = t["hold_days"]
            if tk not in ticker_dates:
                perm_returns.append(0.0)
                continue
            dates = ticker_dates[tk]
            max_idx = max(0, len(dates) - hold - 1)
            if max_idx < 1:
                perm_returns.append(0.0)
                continue
            rand_loc = np.random.randint(0, max_idx)
            c = closes[tk]
            try:
                entry_p = float(c.iloc[c.index.get_loc(dates[rand_loc])]) * (1 + SLIPPAGE_PCT)
                exit_loc = min(c.index.get_loc(dates[rand_loc]) + hold, len(c) - 1)
                exit_p = float(c.iloc[exit_loc]) * (1 - SLIPPAGE_PCT)
                if entry_p > 0:
                    perm_returns.append((exit_p - entry_p) / entry_p)
                else:
                    perm_returns.append(0.0)
            except Exception:
                perm_returns.append(0.0)

        if np.mean(perm_returns) >= real_mean:
            count_better += 1

    return count_better / n_perm


def five_gate(metrics, regime, perm_p):
    """Apply 5-gate validation."""
    gates = {
        "sharpe_gt_0.5": metrics["sharpe"] > GATES["sharpe_min"],
        "perm_p_lt_0.05": perm_p < GATES["perm_p_max"],
        "regime_gap_lt_0.5": regime["regime_gap"] < GATES["regime_gap_max"],
        "maxdd_gt_neg50": metrics["max_drawdown_pct"] > GATES["mdd_min"],
        "trades_gte_20": metrics["n_trades"] >= GATES["min_trades"],
    }
    gates["pass_all"] = all(v for k, v in gates.items() if k != "pass_all")
    return gates


# ── Run All Variants ─────────────────────────────────────────────────────
print("\n" + "="*70)
print("RUNNING REIT MEAN REVERSION BACKTEST — 6 VARIANTS")
print("="*70)

variant_configs = {
    "A": ("Large-cap REITs Dual D", signals_a),
    "B": ("REIT ETFs Dual D", signals_b),
    "C": ("REITs+Quality Dual D", signals_c),
    "D": ("REIT rate signal", signals_d),
    "E": ("Rate-filtered Dual D", signals_e),
    "F": ("High-yield dip RSI<40", signals_f),
}

results = {}

for var_name, (description, sigs) in variant_configs.items():
    print(f"\n--- Variant {var_name}: {description} ---")

    trades = simulate_trades(sigs)
    metrics = calc_metrics(trades)
    regime = regime_stratified_sharpe(trades)

    print(f"  Trades: {metrics['n_trades']}, Return: {metrics['total_return_pct']:.1f}%, "
          f"Sharpe: {metrics['sharpe']:.3f}, WR: {metrics['win_rate']:.1%}")

    # Permutation test
    if metrics["n_trades"] >= 5:
        print(f"  Running permutation test ({N_PERM} iterations) ...")
        perm_p = permutation_test(trades, N_PERM)
    else:
        perm_p = 1.0
    print(f"  Perm p-value: {perm_p:.4f}")

    gates = five_gate(metrics, regime, perm_p)
    passed = sum(1 for k, v in gates.items() if k != "pass_all" and v)
    print(f"  Gates: {passed}/5 {'PASS' if gates['pass_all'] else 'FAIL'}")

    # Top tickers
    top_tickers = {}
    if trades:
        ticker_counts = defaultdict(int)
        ticker_pnl = defaultdict(float)
        for t in trades:
            ticker_counts[t["ticker"]] += 1
            ticker_pnl[t["ticker"]] += t["pnl"]
        top5 = sorted(ticker_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        top_tickers = {tk: {"trades": cnt, "pnl": round(ticker_pnl[tk], 2)} for tk, cnt in top5}

    results[var_name] = {
        "description": description,
        "metrics": metrics,
        "regime": regime,
        "perm_p": round(perm_p, 4),
        "gates": gates,
        "top_tickers": top_tickers,
        "sample_trades": trades[:5] if trades else [],
    }


# ── Print Comparison Table ──────────────────────────────────────────────
print("\n" + "="*110)
print("COMPARISON TABLE — REIT MEAN REVERSION BACKTEST")
print("="*110)
print(f"{'Var':<5} {'Description':<28} {'Trades':>6} {'Return%':>8} {'Sharpe':>7} "
      f"{'Sortino':>8} {'WR':>6} {'PF':>6} {'MDD%':>7} {'PermP':>6} {'Gates':>6}")
print("-"*110)

for var_name in ["A", "B", "C", "D", "E", "F"]:
    r = results[var_name]
    m = r["metrics"]
    g = r["gates"]
    passed = sum(1 for k, v in g.items() if k != "pass_all" and v)
    tag = "PASS" if g["pass_all"] else "FAIL"
    print(f"{var_name:<5} {r['description']:<28} {m['n_trades']:>6} {m['total_return_pct']:>7.1f}% "
          f"{m['sharpe']:>7.3f} {m['sortino']:>8.3f} {m['win_rate']:>5.1%} {m['profit_factor']:>6.2f} "
          f"{m['max_drawdown_pct']:>6.1f}% {r['perm_p']:>6.4f} {passed}/5 {tag}")

# ── Regime Detail ────────────────────────────────────────────────────────
print("\n" + "="*90)
print("REGIME STRATIFICATION (SPY vs 200-SMA)")
print("="*90)
print(f"{'Var':<5} {'Bull Sharpe':>11} {'Bear Sharpe':>11} {'Gap':>6} {'Bull#':>6} {'Bear#':>6}")
print("-"*90)
for var_name in ["A", "B", "C", "D", "E", "F"]:
    reg = results[var_name]["regime"]
    print(f"{var_name:<5} {reg['bull_sharpe']:>11.3f} {reg['bear_sharpe']:>11.3f} "
          f"{reg['regime_gap']:>6.3f} {reg['bull_trades']:>6} {reg['bear_trades']:>6}")

# ── Best Variant Analysis ────────────────────────────────────────────────
print("\n" + "="*70)
print("BEST VARIANTS (by Sharpe, passing all 5 gates)")
print("="*70)

passing = {k: v for k, v in results.items() if v["gates"]["pass_all"]}
if passing:
    sorted_pass = sorted(passing.items(), key=lambda x: x[1]["metrics"]["sharpe"], reverse=True)
    for rank, (var, data) in enumerate(sorted_pass, 1):
        m = data["metrics"]
        print(f"  #{rank}: Variant {var} — Sharpe {m['sharpe']:.3f}, "
              f"Sortino {m['sortino']:.3f}, WR {m['win_rate']:.1%}, "
              f"PF {m['profit_factor']:.2f}, {m['n_trades']} trades, "
              f"Return {m['total_return_pct']:.1f}%")
else:
    print("  No variants passed all 5 gates.")
    sorted_all = sorted(results.items(), key=lambda x: x[1]["metrics"]["sharpe"], reverse=True)
    for rank, (var, data) in enumerate(sorted_all[:3], 1):
        m = data["metrics"]
        g = data["gates"]
        passed = sum(1 for k, v in g.items() if k != "pass_all" and v)
        print(f"  #{rank} (best effort, {passed}/5 gates): Variant {var} — "
              f"Sharpe {m['sharpe']:.3f}, {m['n_trades']} trades, Return {m['total_return_pct']:.1f}%")

# ── REIT vs Rate Environment Analysis ────────────────────────────────────
print("\n" + "="*70)
print("REIT-SPECIFIC ANALYSIS: Rate Environment Impact")
print("="*70)

# Compare Variant A (pure Dual D) vs E (rate-filtered Dual D)
if results["A"]["metrics"]["n_trades"] > 0 and results["E"]["metrics"]["n_trades"] > 0:
    a_sharpe = results["A"]["metrics"]["sharpe"]
    e_sharpe = results["E"]["metrics"]["sharpe"]
    print(f"  Dual D without rate filter:  Sharpe {a_sharpe:.3f} ({results['A']['metrics']['n_trades']} trades)")
    print(f"  Dual D WITH rate filter:     Sharpe {e_sharpe:.3f} ({results['E']['metrics']['n_trades']} trades)")
    delta = e_sharpe - a_sharpe
    print(f"  Rate filter impact: {'+'if delta>0 else ''}{delta:.3f} Sharpe")
else:
    print("  Insufficient trades for rate filter comparison.")

# Compare Variant D (pure rate signal) standalone
if results["D"]["metrics"]["n_trades"] > 0:
    print(f"\n  Pure rate-driven REIT signal: Sharpe {results['D']['metrics']['sharpe']:.3f}, "
          f"WR {results['D']['metrics']['win_rate']:.1%}, {results['D']['metrics']['n_trades']} trades")
    print(f"  (Buy REIT dip when rates are falling — TLT up >2% in 5d)")
else:
    print("\n  Pure rate-driven signal: 0 trades (TLT+REIT dip combination too rare)")

# REIT vs Quality stocks contribution in Variant C
if results["C"]["metrics"]["n_trades"] > 0:
    reit_tickers = set(LARGE_CAP_REITS)
    quality_tickers = set(QUALITY_STOCKS)
    c_trades = results["C"].get("sample_trades", [])
    # Use top_tickers for analysis
    top = results["C"]["top_tickers"]
    reit_in_top = {k: v for k, v in top.items() if k in reit_tickers}
    quality_in_top = {k: v for k, v in top.items() if k in quality_tickers}
    print(f"\n  Variant C universe breakdown (top tickers):")
    for tk, info in sorted(top.items(), key=lambda x: x[1]["trades"], reverse=True):
        label = "REIT" if tk in reit_tickers else "QUALITY"
        print(f"    {tk} ({label}): {info['trades']} trades, PnL ${info['pnl']:.2f}")


# ── Save Results ─────────────────────────────────────────────────────────
output = {
    "strategy": "REIT Mean Reversion",
    "run_date": datetime.now().isoformat(),
    "config": {
        "capital": CAPITAL,
        "max_per_trade": MAX_PER_TRADE,
        "max_concurrent": MAX_CONCURRENT,
        "slippage_bps": SLIPPAGE_PCT * 10000,
        "oot_period": f"{OOT_START} to {END}",
        "n_permutations": N_PERM,
    },
    "universes": {
        "A": LARGE_CAP_REITS,
        "B": REIT_ETFS,
        "C": list(set(LARGE_CAP_REITS + QUALITY_STOCKS)),
        "D": LARGE_CAP_REITS,
        "E": LARGE_CAP_REITS,
        "F": HIGH_YIELD_REITS,
    },
    "variants": {},
}

for var_name in ["A", "B", "C", "D", "E", "F"]:
    r = results[var_name]
    output["variants"][var_name] = {
        "description": r["description"],
        "metrics": r["metrics"],
        "regime": r["regime"],
        "perm_p": r["perm_p"],
        "gates": r["gates"],
        "top_tickers": r["top_tickers"],
    }

out_path = Path("/home/jupiter/Lvl3Quant/data/reit_mr_results.json")
with open(out_path, "w") as f:
    json.dump(output, f, indent=2, default=str)
print(f"\nResults saved to {out_path}")

# Summary
passing_count = sum(1 for v in results.values() if v["gates"]["pass_all"])
print(f"\n{'='*70}")
print(f"SUMMARY: {passing_count}/6 variants passed all 5 gates")
print(f"{'='*70}")
