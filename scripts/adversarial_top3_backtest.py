#!/usr/bin/env python3
"""
Adversarial Validation — Top 3 Strategy Variants
=================================================
Tests 5 adversarial checks on each of the 3 strategies that passed 5-gate validation:
  1. Signal Aggregator A (Threshold 3) — QQQ when >=3/5 signals
  2. Signal Aggregator E (Sector Selection) — best sector ETF by 20d RS when >=3/5 signals
  3. Strategy Rotation v2 F (Dynamic Contrarian) — VIX fade / bull QQQ / contrarian SPY 8d

Tests:
  T1 — Inverse Direction (short when original goes long)
  T2 — Random Timing (500 iterations, compare to 75th pctl)
  T3 — Sub-period Consistency (3 equal sub-periods, each Sharpe > 0)
  T4 — Top-Trade Removal (remove best 5, Sharpe must stay > 0.5)
  T5 — Parameter Sensitivity (variant-specific)

Scoring: 0-5 per variant. Need 4/5+ to be validated.
"""

import json
import warnings
import sys
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")
np.random.seed(42)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
OOT_START = "2022-01-01"
OOT_END = "2026-07-28"
CAPITAL = 645.0
SLIPPAGE_PCT = 0.0002  # 0.02%
RANDOM_ITERATIONS = 500
OUTPUT_PATH = Path("/home/jupiter/Lvl3Quant/data/adversarial_top3_results.json")

TICKERS = ["SPY", "QQQ", "XLK", "XLC", "XLY", "XLE", "XLF", "XLV", "RSP", "TLT", "^VIX"]

# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------
def load_data():
    """Download all required data from yfinance."""
    print("Downloading market data...")
    data = {}
    for t in TICKERS:
        key = t.replace("^", "")
        df = yf.download(t, start="2021-01-01", end=OOT_END, progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        data[key] = df
    print(f"  Loaded {len(data)} tickers, date range: {data['SPY'].index[0].date()} to {data['SPY'].index[-1].date()}")
    return data


def align_data(data):
    """Align all dataframes to common trading dates within OOT period."""
    idx = data["SPY"].loc[OOT_START:].index
    for key in data:
        if key == "VIX":
            continue
        idx = idx.intersection(data[key].loc[OOT_START:].index)
    return idx


# ---------------------------------------------------------------------------
# Signal Computation (Composite 5-signal system)
# ---------------------------------------------------------------------------
def compute_signals(data, idx):
    """
    Composite scoring:
      S1: REGIME — SPY > 200-SMA
      S2: VIX CALM — VIX < 20 OR VIX declining from >25
      S3: MOMENTUM — QQQ 20d return > 0
      S4: VOLUME — any sector ETF has 5+ days above-avg volume (20d)
      S5: BREADTH — SPY outperforming RSP (proxy >60% above 50-SMA)
    Returns Series of signal counts (0-5) aligned to idx.
    """
    spy = data["SPY"]["Close"].reindex(idx).ffill()
    qqq = data["QQQ"]["Close"].reindex(idx).ffill()
    rsp = data["RSP"]["Close"].reindex(idx).ffill()
    vix = data["VIX"]["Close"].reindex(idx).ffill()

    # Need full history for SMAs
    spy_full = data["SPY"]["Close"]
    spy_sma200 = spy_full.rolling(200).mean().reindex(idx).ffill()

    # S1: Regime
    s1 = (spy > spy_sma200).astype(int)

    # S2: VIX calm
    vix_declining = vix.diff(5) < 0
    s2 = ((vix < 20) | ((vix > 25) & vix_declining)).astype(int)

    # S3: Momentum
    qqq_ret20 = qqq.pct_change(20)
    s3 = (qqq_ret20 > 0).astype(int)

    # S4: Volume — check sector ETFs
    sectors = ["XLK", "XLC", "XLY", "XLE", "XLF", "XLV"]
    s4 = pd.Series(0, index=idx)
    for sec in sectors:
        vol = data[sec]["Volume"].reindex(idx).ffill()
        vol_avg = vol.rolling(20).mean()
        above_avg = (vol > vol_avg).rolling(5).sum()
        s4 = s4 | (above_avg >= 5).astype(int)
    s4 = s4.astype(int)

    # S5: Breadth proxy — SPY outperforming RSP over 20d
    spy_ret20 = spy.pct_change(20)
    rsp_ret20 = rsp.pct_change(20)
    s5 = (spy_ret20 > rsp_ret20).astype(int)

    total = s1 + s2 + s3 + s4 + s5
    return total, {"s1": s1, "s2": s2, "s3": s3, "s4": s4, "s5": s5}


# ---------------------------------------------------------------------------
# Strategy Implementations
# ---------------------------------------------------------------------------
def apply_slippage(returns, invested):
    """Apply slippage on entry/exit days."""
    transitions = invested.astype(int).diff().abs().fillna(0)
    cost = transitions * SLIPPAGE_PCT
    return returns - cost


def strategy_agg_a(data, idx, threshold=3):
    """Signal Aggregator A: Buy QQQ when signal_count >= threshold, else cash."""
    signals, _ = compute_signals(data, idx)
    qqq_ret = data["QQQ"]["Close"].reindex(idx).ffill().pct_change()
    invested = (signals >= threshold)
    strat_ret = qqq_ret * invested.shift(1).fillna(False)
    strat_ret = apply_slippage(strat_ret, invested)
    return strat_ret.fillna(0), invested


def strategy_agg_e(data, idx, use_random_sector=False):
    """Signal Aggregator E: Buy best sector ETF by 20d relative strength when >=3/5 signals."""
    signals, _ = compute_signals(data, idx)
    sectors = ["XLK", "XLC", "XLY", "XLE", "XLF", "XLV"]

    # Compute 20d returns for each sector
    sector_rets = {}
    sector_daily = {}
    for sec in sectors:
        close = data[sec]["Close"].reindex(idx).ffill()
        sector_rets[sec] = close.pct_change(20)
        sector_daily[sec] = close.pct_change()

    # Pick best sector each day by 20d relative strength (or random)
    strat_ret = pd.Series(0.0, index=idx)
    invested = pd.Series(False, index=idx)

    for i in range(1, len(idx)):
        dt = idx[i]
        dt_prev = idx[i - 1]
        if signals.loc[dt_prev] >= 3:
            invested.iloc[i] = True
            if use_random_sector:
                best = np.random.choice(sectors)
            else:
                best_ret = -np.inf
                best = sectors[0]
                for sec in sectors:
                    r = sector_rets[sec].get(dt_prev, np.nan)
                    if not np.isnan(r) and r > best_ret:
                        best_ret = r
                        best = sec
            strat_ret.iloc[i] = sector_daily[best].iloc[i]

    strat_ret = apply_slippage(strat_ret, invested)
    return strat_ret.fillna(0), invested


def strategy_rot_f(data, idx, contrarian_mult=1.5, hold_days=8):
    """
    Strategy Rotation v2 F (Dynamic Contrarian):
      - VIX > 25 and declining 5d → VIX fade: buy SPY
      - SPY > 200-SMA → bull: buy QQQ
      - Bear + weekly return < -contrarian_mult * VIX/100 → contrarian buy SPY, hold 8d
      - Otherwise cash
    """
    spy_close = data["SPY"]["Close"].reindex(idx).ffill()
    qqq_close = data["QQQ"]["Close"].reindex(idx).ffill()
    vix_close = data["VIX"]["Close"].reindex(idx).ffill()
    spy_full = data["SPY"]["Close"]
    spy_sma200 = spy_full.rolling(200).mean().reindex(idx).ffill()

    spy_ret = spy_close.pct_change()
    qqq_ret = qqq_close.pct_change()
    spy_weekly_ret = spy_close.pct_change(5)
    vix_declining = vix_close.diff(5) < 0

    strat_ret = pd.Series(0.0, index=idx)
    invested = pd.Series(False, index=idx)
    contrarian_hold_until = None
    asset_held = None

    for i in range(1, len(idx)):
        dt = idx[i]
        dt_prev = idx[i - 1]

        # Check if in contrarian hold period
        if contrarian_hold_until is not None and dt <= contrarian_hold_until:
            invested.iloc[i] = True
            strat_ret.iloc[i] = spy_ret.iloc[i]
            continue

        contrarian_hold_until = None

        v = vix_close.get(dt_prev, 20)
        v_dec = vix_declining.get(dt_prev, False)
        spy_above_200 = spy_close.get(dt_prev, 0) > spy_sma200.get(dt_prev, 0)
        w_ret = spy_weekly_ret.get(dt_prev, 0)

        if v > 25 and v_dec:
            # VIX fade → buy SPY
            invested.iloc[i] = True
            strat_ret.iloc[i] = spy_ret.iloc[i]
        elif spy_above_200:
            # Bull → buy QQQ
            invested.iloc[i] = True
            strat_ret.iloc[i] = qqq_ret.iloc[i]
        elif w_ret < -contrarian_mult * v / 100:
            # Contrarian buy SPY with hold
            invested.iloc[i] = True
            strat_ret.iloc[i] = spy_ret.iloc[i]
            contrarian_hold_until = idx[min(i + hold_days - 1, len(idx) - 1)]
        # else: cash

    strat_ret = apply_slippage(strat_ret, invested)
    return strat_ret.fillna(0), invested


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def compute_sharpe(returns, annualize=True):
    """Annualized Sharpe ratio (assuming 252 trading days)."""
    if len(returns) < 2 or returns.std() == 0:
        return 0.0
    sr = returns.mean() / returns.std()
    if annualize:
        sr *= np.sqrt(252)
    return sr


def compute_total_return(returns):
    """Total return as multiplier."""
    return (1 + returns).prod() - 1


# ---------------------------------------------------------------------------
# Adversarial Tests
# ---------------------------------------------------------------------------
def test_inverse(returns, invested, data, idx, strategy_fn, **kwargs):
    """T1: Flip direction — short when original goes long. If also profitable, edge is beta."""
    # Inverse: negative return when invested
    inv_ret = returns.copy()
    inv_ret[invested.shift(1).fillna(False)] *= -1
    inv_sharpe = compute_sharpe(inv_ret)
    orig_sharpe = compute_sharpe(returns)

    passed = inv_sharpe <= 0
    print(f"    Inverse Sharpe: {inv_sharpe:.3f} (original: {orig_sharpe:.3f})")
    print(f"    {'PASS' if passed else 'FAIL'} — inverse {'unprofitable' if passed else 'also profitable (beta exposure)'}")
    return passed, {"inverse_sharpe": round(inv_sharpe, 4), "original_sharpe": round(orig_sharpe, 4)}


def test_random_timing(returns, invested, data, idx, n_iter=RANDOM_ITERATIONS):
    """T2: Random entry dates, same number of trades. 500 iterations."""
    n_invested = invested.sum()
    if n_invested == 0:
        return False, {"reason": "no trades"}

    # Get the underlying asset return (use SPY as default proxy)
    spy_ret = data["SPY"]["Close"].reindex(idx).ffill().pct_change().fillna(0)
    orig_sharpe = compute_sharpe(returns)
    random_sharpes = []

    for _ in range(n_iter):
        rand_mask = pd.Series(False, index=idx)
        rand_days = np.random.choice(len(idx), size=int(n_invested), replace=False)
        rand_mask.iloc[rand_days] = True
        rand_ret = spy_ret * rand_mask.shift(1).fillna(False)
        rand_ret = apply_slippage(rand_ret, rand_mask)
        random_sharpes.append(compute_sharpe(rand_ret))

    p75 = np.percentile(random_sharpes, 75)
    passed = orig_sharpe > p75
    print(f"    Original Sharpe: {orig_sharpe:.3f}, Random 75th pctl: {p75:.3f}")
    print(f"    {'PASS' if passed else 'FAIL'} — signal {'has' if passed else 'lacks'} timing edge vs random")
    return passed, {
        "original_sharpe": round(orig_sharpe, 4),
        "random_p75": round(p75, 4),
        "random_median": round(np.median(random_sharpes), 4),
        "random_p95": round(np.percentile(random_sharpes, 95), 4),
    }


def test_subperiod(returns, idx):
    """T3: Split into 3 equal sub-periods. Each must have Sharpe > 0."""
    n = len(returns)
    split1 = n // 3
    split2 = 2 * n // 3
    periods = [
        ("P1", returns.iloc[:split1]),
        ("P2", returns.iloc[split1:split2]),
        ("P3", returns.iloc[split2:]),
    ]
    results = {}
    all_positive = True
    for name, r in periods:
        sr = compute_sharpe(r)
        results[name] = round(sr, 4)
        date_range = f"{r.index[0].date()} to {r.index[-1].date()}"
        status = "+" if sr > 0 else "-"
        print(f"    {name} ({date_range}): Sharpe {sr:.3f} [{status}]")
        if sr <= 0:
            all_positive = False

    print(f"    {'PASS' if all_positive else 'FAIL'} — {'all' if all_positive else 'NOT all'} sub-periods positive")
    return all_positive, results


def test_top_trade_removal(returns, n_remove=5, threshold=0.5):
    """T4: Remove best 5 trades. Sharpe must stay > threshold."""
    # Find top N return days
    sorted_idx = returns.nlargest(n_remove).index
    modified = returns.copy()
    modified.loc[sorted_idx] = 0.0

    orig_sharpe = compute_sharpe(returns)
    mod_sharpe = compute_sharpe(modified)

    passed = mod_sharpe > threshold
    print(f"    Original Sharpe: {orig_sharpe:.3f}, After removing top {n_remove}: {mod_sharpe:.3f}")
    print(f"    {'PASS' if passed else 'FAIL'} — Sharpe {'above' if passed else 'below'} {threshold} threshold")
    return passed, {
        "original_sharpe": round(orig_sharpe, 4),
        "modified_sharpe": round(mod_sharpe, 4),
        "removed_trades": [str(d.date()) for d in sorted_idx],
    }


def test_param_sensitivity_agg_a(data, idx):
    """T5 for Signal Aggregator A: Test threshold 2 and 4."""
    results = {}
    all_pass = True
    for thresh in [2, 4]:
        ret, inv = strategy_agg_a(data, idx, threshold=thresh)
        sr = compute_sharpe(ret)
        results[f"threshold_{thresh}"] = round(sr, 4)
        status = "PASS" if sr > 0.3 else "FAIL"
        print(f"    Threshold {thresh}: Sharpe {sr:.3f} [{status}]")
        if sr <= 0.3:
            all_pass = False

    print(f"    {'PASS' if all_pass else 'FAIL'} — parameter sensitivity")
    return all_pass, results


def test_param_sensitivity_agg_e(data, idx):
    """T5 for Signal Aggregator E: Test random sector selection."""
    # Run 50 iterations of random sector picking
    random_sharpes = []
    for _ in range(50):
        ret, inv = strategy_agg_e(data, idx, use_random_sector=True)
        random_sharpes.append(compute_sharpe(ret))

    orig_ret, _ = strategy_agg_e(data, idx, use_random_sector=False)
    orig_sharpe = compute_sharpe(orig_ret)
    rand_median = np.median(random_sharpes)
    rand_p75 = np.percentile(random_sharpes, 75)

    # Sector selection adds value if original >> random median
    edge_ratio = (orig_sharpe - rand_median) / max(abs(orig_sharpe), 0.01)
    passed = edge_ratio > 0.20  # sector selection must add at least 20% of total Sharpe

    print(f"    Original Sharpe: {orig_sharpe:.3f}, Random sector median: {rand_median:.3f}, p75: {rand_p75:.3f}")
    print(f"    Edge from sector selection: {edge_ratio:.1%}")
    print(f"    {'PASS' if passed else 'FAIL'} — sector selection {'adds' if passed else 'does NOT add'} meaningful value")
    return passed, {
        "original_sharpe": round(orig_sharpe, 4),
        "random_sector_median": round(rand_median, 4),
        "random_sector_p75": round(rand_p75, 4),
        "edge_ratio": round(edge_ratio, 4),
    }


def test_param_sensitivity_rot_f(data, idx):
    """T5 for Strategy Rotation F: Test contrarian mult and hold days."""
    results = {}
    all_pass = True

    for mult in [1.0, 2.0]:
        ret, inv = strategy_rot_f(data, idx, contrarian_mult=mult, hold_days=8)
        sr = compute_sharpe(ret)
        results[f"mult_{mult}"] = round(sr, 4)
        status = "PASS" if sr > 0.3 else "FAIL"
        print(f"    Contrarian mult {mult}: Sharpe {sr:.3f} [{status}]")
        if sr <= 0.3:
            all_pass = False

    for hd in [5, 12]:
        ret, inv = strategy_rot_f(data, idx, contrarian_mult=1.5, hold_days=hd)
        sr = compute_sharpe(ret)
        results[f"hold_{hd}d"] = round(sr, 4)
        status = "PASS" if sr > 0.3 else "FAIL"
        print(f"    Hold {hd}d: Sharpe {sr:.3f} [{status}]")
        if sr <= 0.3:
            all_pass = False

    print(f"    {'PASS' if all_pass else 'FAIL'} — parameter sensitivity")
    return all_pass, results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run_all():
    data = load_data()
    idx = align_data(data)
    print(f"OOT period: {idx[0].date()} to {idx[-1].date()}, {len(idx)} trading days\n")

    # Define strategies
    strategies = {
        "Signal_Aggregator_A": {
            "fn": lambda: strategy_agg_a(data, idx),
            "param_test": lambda: test_param_sensitivity_agg_a(data, idx),
            "claimed_sharpe": 2.702,
        },
        "Signal_Aggregator_E": {
            "fn": lambda: strategy_agg_e(data, idx),
            "param_test": lambda: test_param_sensitivity_agg_e(data, idx),
            "claimed_sharpe": 4.974,
        },
        "Strategy_Rotation_F": {
            "fn": lambda: strategy_rot_f(data, idx),
            "param_test": lambda: test_param_sensitivity_rot_f(data, idx),
            "claimed_sharpe": 1.886,
        },
    }

    all_results = {}

    for name, spec in strategies.items():
        print("=" * 70)
        print(f"  STRATEGY: {name}")
        print(f"  Claimed Sharpe: {spec['claimed_sharpe']}")
        print("=" * 70)

        returns, invested = spec["fn"]()
        actual_sharpe = compute_sharpe(returns)
        total_ret = compute_total_return(returns)
        n_invested = invested.sum()
        pct_invested = n_invested / len(idx) * 100

        print(f"  Reproduced Sharpe: {actual_sharpe:.3f}")
        print(f"  Total Return: {total_ret:.1%} on ${CAPITAL}")
        print(f"  Days invested: {int(n_invested)}/{len(idx)} ({pct_invested:.0f}%)\n")

        strat_results = {
            "reproduced_sharpe": round(actual_sharpe, 4),
            "total_return_pct": round(total_ret * 100, 2),
            "days_invested": int(n_invested),
            "total_days": len(idx),
            "tests": {},
            "score": 0,
        }

        # T1 — Inverse
        print("  [T1] Inverse Direction Test")
        p1, r1 = test_inverse(returns, invested, data, idx, spec["fn"])
        strat_results["tests"]["T1_inverse"] = {"passed": p1, "details": r1}
        print()

        # T2 — Random Timing
        print("  [T2] Random Timing Test (500 iterations)")
        p2, r2 = test_random_timing(returns, invested, data, idx)
        strat_results["tests"]["T2_random_timing"] = {"passed": p2, "details": r2}
        print()

        # T3 — Sub-period Consistency
        print("  [T3] Sub-period Consistency Test")
        p3, r3 = test_subperiod(returns, idx)
        strat_results["tests"]["T3_subperiod"] = {"passed": p3, "details": r3}
        print()

        # T4 — Top-Trade Removal
        print("  [T4] Top-Trade Removal Test")
        p4, r4 = test_top_trade_removal(returns)
        strat_results["tests"]["T4_top_trade_removal"] = {"passed": p4, "details": r4}
        print()

        # T5 — Parameter Sensitivity
        print("  [T5] Parameter Sensitivity Test")
        p5, r5 = spec["param_test"]()
        strat_results["tests"]["T5_param_sensitivity"] = {"passed": p5, "details": r5}
        print()

        score = sum([p1, p2, p3, p4, p5])
        strat_results["score"] = score
        verdict = "VALIDATED" if score >= 4 else "REJECTED"
        strat_results["verdict"] = verdict

        print(f"  >>> ADVERSARIAL SCORE: {score}/5 — {verdict}")
        print()

        all_results[name] = strat_results

    # Save results
    output = {
        "run_timestamp": datetime.now().isoformat(),
        "oot_period": f"{OOT_START} to {OOT_END}",
        "capital": CAPITAL,
        "slippage_pct": SLIPPAGE_PCT,
        "random_iterations": RANDOM_ITERATIONS,
        "strategies": all_results,
    }

    # Convert numpy types for JSON serialization
    def make_serializable(obj):
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_serializable(v) for v in obj]
        elif isinstance(obj, (np.bool_, np.integer)):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, (bool, int, float, str, type(None))):
            return obj
        return str(obj)

    output = make_serializable(output)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Results saved to {OUTPUT_PATH}")

    # Final summary
    print("\n" + "=" * 70)
    print("  FINAL ADVERSARIAL SUMMARY")
    print("=" * 70)
    for name, res in all_results.items():
        tests = res["tests"]
        test_str = " | ".join(
            f"T{i+1}:{'P' if list(tests.values())[i]['passed'] else 'F'}"
            for i in range(5)
        )
        print(f"  {name:30s}  Score: {res['score']}/5  [{test_str}]  {res['verdict']}")
    print("=" * 70)


if __name__ == "__main__":
    run_all()
