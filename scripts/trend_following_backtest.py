#!/usr/bin/env python3
"""
Trend Following with Multi-Timeframe Filters — Backtest
========================================================
Tests whether requiring MULTIPLE timeframe trend confirmations
creates genuine alpha beyond simple buy-and-hold.

Signals (all binary 0/1):
  1. QQQ > 200-SMA  (long-term trend)
  2. QQQ > 50-SMA   (medium-term trend)
  3. QQQ > 20-SMA   (short-term trend)
  4. QQQ 5d return > 0  (immediate momentum)
  5. SPY > 200-SMA   (market confirmation)
  6. VIX < 25        (no panic)

Variants:
  A: Long QQQ when ALL 6 true (strictest). Cash otherwise.
  B: Long QQQ when 5+ of 6 true. Cash otherwise.
  C: Long QQQ when 4+ of 6 true. Cash otherwise.
  D: LEVERAGED — Long QQQ when all 6, long SPY when 4-5, cash <4.
  E: TRAILING STOP — Long QQQ when 5+ true, exit if QQQ drops 5%
     from peak while invested. Re-enter when 5+ true again.
  F: INVERSE REGIME FILTER — Long QQQ when conditions 1-4 true
     AND VIX < 20 (ignoring SPY filter).

OOT: Jan 2022 – Jul 2026. Capital: $645. Slippage: 0.02%.
Permutation test: 100 iterations.
"""

import json
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

# ── Config ──────────────────────────────────────────────────────────────
CAPITAL = 645.0
SLIPPAGE_PCT = 0.0002  # 0.02%
OOT_START = "2022-01-01"
OOT_END = "2026-07-29"
N_PERMS = 100
SEED = 42

TICKERS = ["QQQ", "SPY"]
VIX_TICKER = "^VIX"

# Validation gates
SHARPE_MIN = 0.5
PERM_P_MAX = 0.05
REGIME_GAP_MAX = 0.5
MDD_FLOOR = -0.50  # MDD must be > -50%
MIN_TRADES = 20


def download_data():
    """Download QQQ, SPY, VIX with buffer for 200-SMA."""
    all_tickers = TICKERS + [VIX_TICKER]
    start = (pd.Timestamp(OOT_START) - pd.DateOffset(days=300)).strftime("%Y-%m-%d")

    print(f"Downloading data for {all_tickers} from {start}...")
    data = yf.download(all_tickers, start=start, end=OOT_END, auto_adjust=True, progress=False)

    close = data["Close"].copy()
    close = close.dropna(subset=["SPY"])
    print(f"Data: {close.index[0].date()} to {close.index[-1].date()}, {len(close)} trading days")
    return close


def compute_signals(close):
    """Compute all 6 trend signals daily. Returns DataFrame."""
    idx = close.index
    signals = pd.DataFrame(index=idx, dtype=float)

    # Signal 1: QQQ > 200-SMA
    signals["qqq_200sma"] = (close["QQQ"] > close["QQQ"].rolling(200).mean()).astype(float)

    # Signal 2: QQQ > 50-SMA
    signals["qqq_50sma"] = (close["QQQ"] > close["QQQ"].rolling(50).mean()).astype(float)

    # Signal 3: QQQ > 20-SMA
    signals["qqq_20sma"] = (close["QQQ"] > close["QQQ"].rolling(20).mean()).astype(float)

    # Signal 4: QQQ 5d return > 0
    signals["qqq_5d_mom"] = (close["QQQ"].pct_change(5) > 0).astype(float)

    # Signal 5: SPY > 200-SMA
    signals["spy_200sma"] = (close["SPY"] > close["SPY"].rolling(200).mean()).astype(float)

    # Signal 6: VIX < 25
    vix = close[VIX_TICKER] if VIX_TICKER in close.columns else close.get("^VIX")
    if vix is not None:
        signals["vix_calm"] = (vix < 25).astype(float)
    else:
        signals["vix_calm"] = 1.0  # assume calm if no VIX data

    signal_cols = ["qqq_200sma", "qqq_50sma", "qqq_20sma", "qqq_5d_mom", "spy_200sma", "vix_calm"]
    signals["score"] = signals[signal_cols].sum(axis=1)

    return signals, signal_cols


def apply_slippage(returns, trades_mask):
    """Apply slippage cost on trade entry days."""
    adj = returns.copy()
    adj[trades_mask] -= SLIPPAGE_PCT
    return adj


def identify_regime(close):
    """Classify each day as bull/bear based on SPY 200-SMA."""
    sma200 = close["SPY"].rolling(200).mean()
    regime = pd.Series("bull", index=close.index)
    regime[close["SPY"] < sma200] = "bear"
    return regime


def calc_metrics(equity_curve, daily_returns, regime, trades_count):
    """Calculate performance metrics."""
    dr = daily_returns.dropna()
    if len(dr) < 10:
        return None

    ann_ret = dr.mean() * 252
    ann_vol = dr.std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0

    downside = dr[dr < 0].std() * np.sqrt(252)
    sortino = ann_ret / downside if downside > 0 else 0

    peak = equity_curve.cummax()
    dd = (equity_curve - peak) / peak
    mdd = dd.min()

    gains = dr[dr > 0].sum()
    losses = abs(dr[dr < 0].sum())
    pf = gains / losses if losses > 0 else float("inf")

    wr = (dr[dr != 0] > 0).mean() if (dr != 0).any() else 0

    bull_days = regime == "bull"
    bear_days = regime == "bear"
    bull_ret = dr[bull_days]
    bear_ret = dr[bear_days]

    bull_sharpe = (bull_ret.mean() * 252) / (bull_ret.std() * np.sqrt(252)) if len(bull_ret) > 10 and bull_ret.std() > 0 else 0
    bear_sharpe = (bear_ret.mean() * 252) / (bear_ret.std() * np.sqrt(252)) if len(bear_ret) > 10 and bear_ret.std() > 0 else 0

    max_sharpe = max(abs(bull_sharpe), abs(bear_sharpe))
    regime_gap = abs(bull_sharpe - bear_sharpe) / max_sharpe if max_sharpe > 0 else 0

    total_return = (equity_curve.iloc[-1] / CAPITAL - 1) * 100

    return {
        "total_return_pct": round(total_return, 2),
        "ann_return_pct": round(ann_ret * 100, 2),
        "ann_vol_pct": round(ann_vol * 100, 2),
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "max_drawdown_pct": round(mdd * 100, 2),
        "profit_factor": round(pf, 3),
        "win_rate": round(wr, 4),
        "trades": trades_count,
        "bull_sharpe": round(bull_sharpe, 3),
        "bear_sharpe": round(bear_sharpe, 3),
        "regime_gap": round(regime_gap, 3),
    }


# ── Variant Implementations ────────────────────────────────────────────

def run_variant_a(signals, close, regime):
    """A: Long QQQ when ALL 6 conditions true. Cash otherwise."""
    oot = signals.loc[OOT_START:]
    qqq_ret = close["QQQ"].pct_change().loc[oot.index]

    invested = oot["score"] == 6
    trades_mask = invested != invested.shift(1)
    trades_count = trades_mask.sum()

    daily_ret = pd.Series(0.0, index=oot.index)
    daily_ret[invested] = qqq_ret[invested]
    daily_ret = apply_slippage(daily_ret, trades_mask & invested)

    equity = (1 + daily_ret).cumprod() * CAPITAL
    metrics = calc_metrics(equity, daily_ret, regime.loc[oot.index], trades_count)
    return metrics, daily_ret, equity, oot["score"]


def run_variant_b(signals, close, regime):
    """B: Long QQQ when 5+ of 6 true. Cash otherwise."""
    oot = signals.loc[OOT_START:]
    qqq_ret = close["QQQ"].pct_change().loc[oot.index]

    invested = oot["score"] >= 5
    trades_mask = invested != invested.shift(1)
    trades_count = trades_mask.sum()

    daily_ret = pd.Series(0.0, index=oot.index)
    daily_ret[invested] = qqq_ret[invested]
    daily_ret = apply_slippage(daily_ret, trades_mask & invested)

    equity = (1 + daily_ret).cumprod() * CAPITAL
    metrics = calc_metrics(equity, daily_ret, regime.loc[oot.index], trades_count)
    return metrics, daily_ret, equity, oot["score"]


def run_variant_c(signals, close, regime):
    """C: Long QQQ when 4+ of 6 true. Cash otherwise."""
    oot = signals.loc[OOT_START:]
    qqq_ret = close["QQQ"].pct_change().loc[oot.index]

    invested = oot["score"] >= 4
    trades_mask = invested != invested.shift(1)
    trades_count = trades_mask.sum()

    daily_ret = pd.Series(0.0, index=oot.index)
    daily_ret[invested] = qqq_ret[invested]
    daily_ret = apply_slippage(daily_ret, trades_mask & invested)

    equity = (1 + daily_ret).cumprod() * CAPITAL
    metrics = calc_metrics(equity, daily_ret, regime.loc[oot.index], trades_count)
    return metrics, daily_ret, equity, oot["score"]


def run_variant_d(signals, close, regime):
    """D: LEVERAGED — Long QQQ when all 6, long SPY when 4-5, cash <4."""
    oot = signals.loc[OOT_START:]
    qqq_ret = close["QQQ"].pct_change().loc[oot.index]
    spy_ret = close["SPY"].pct_change().loc[oot.index]

    score = oot["score"]
    all_6 = score == 6
    mid = (score >= 4) & (score < 6)

    daily_ret = pd.Series(0.0, index=oot.index)
    daily_ret[all_6] = qqq_ret[all_6]
    daily_ret[mid] = spy_ret[mid]

    # Track state changes for trade counting
    state = pd.Series("cash", index=oot.index)
    state[all_6] = "qqq"
    state[mid] = "spy"
    trades_mask = state != state.shift(1)
    trades_count = trades_mask.sum()

    invested = all_6 | mid
    daily_ret = apply_slippage(daily_ret, trades_mask & invested)

    equity = (1 + daily_ret).cumprod() * CAPITAL
    metrics = calc_metrics(equity, daily_ret, regime.loc[oot.index], trades_count)
    return metrics, daily_ret, equity, score


def run_variant_e(signals, close, regime):
    """E: TRAILING STOP — Long QQQ when 5+ true, exit if QQQ drops 5%
    from peak while invested. Re-enter when 5+ true again."""
    oot = signals.loc[OOT_START:]
    qqq_ret = close["QQQ"].pct_change().loc[oot.index]
    qqq_price = close["QQQ"].loc[oot.index]

    score = oot["score"]
    TRAIL_STOP_PCT = 0.05

    # Need to iterate for trailing stop logic
    invested_arr = np.zeros(len(oot), dtype=bool)
    stopped_out = False
    peak_price = 0.0

    for i in range(len(oot)):
        sc = score.iloc[i]
        px = qqq_price.iloc[i]

        if not invested_arr[i - 1] if i > 0 else True:
            # Not invested — enter if score >= 5 and not stopped out on this bar
            if sc >= 5 and not stopped_out:
                invested_arr[i] = True
                peak_price = px
            else:
                invested_arr[i] = False
                # Reset stopped_out if score drops below 5 (allow re-entry next time >= 5)
                if sc < 5:
                    stopped_out = False
        else:
            # Currently invested
            peak_price = max(peak_price, px)
            drawdown = (px - peak_price) / peak_price

            if drawdown <= -TRAIL_STOP_PCT:
                # Stop out
                invested_arr[i] = False
                stopped_out = True
            elif sc < 5:
                # Signal gone — exit
                invested_arr[i] = False
                stopped_out = False
            else:
                invested_arr[i] = True

    invested = pd.Series(invested_arr, index=oot.index)
    trades_mask = invested != invested.shift(1)
    trades_count = trades_mask.sum()

    daily_ret = pd.Series(0.0, index=oot.index)
    daily_ret[invested] = qqq_ret[invested]
    daily_ret = apply_slippage(daily_ret, trades_mask & invested)

    equity = (1 + daily_ret).cumprod() * CAPITAL
    metrics = calc_metrics(equity, daily_ret, regime.loc[oot.index], trades_count)
    return metrics, daily_ret, equity, score


def run_variant_f(signals, close, regime):
    """F: INVERSE REGIME FILTER — Long QQQ when conditions 1-4 true
    AND VIX < 20 (ignoring SPY filter)."""
    oot = signals.loc[OOT_START:]
    qqq_ret = close["QQQ"].pct_change().loc[oot.index]

    vix = close[VIX_TICKER].loc[oot.index] if VIX_TICKER in close.columns else close.get("^VIX").loc[oot.index]

    # Conditions 1-4 (QQQ trend filters only) + VIX < 20 (stricter than < 25)
    cond_1_4 = (
        (oot["qqq_200sma"] == 1) &
        (oot["qqq_50sma"] == 1) &
        (oot["qqq_20sma"] == 1) &
        (oot["qqq_5d_mom"] == 1)
    )
    vix_strict = vix < 20
    invested = cond_1_4 & vix_strict

    trades_mask = invested != invested.shift(1)
    trades_count = trades_mask.sum()

    daily_ret = pd.Series(0.0, index=oot.index)
    daily_ret[invested] = qqq_ret[invested]
    daily_ret = apply_slippage(daily_ret, trades_mask & invested)

    equity = (1 + daily_ret).cumprod() * CAPITAL
    metrics = calc_metrics(equity, daily_ret, regime.loc[oot.index], trades_count)

    # For permutation: use sum of the 4 QQQ signals + vix_strict as pseudo-score
    pseudo_score = (
        oot["qqq_200sma"] + oot["qqq_50sma"] +
        oot["qqq_20sma"] + oot["qqq_5d_mom"] +
        vix_strict.astype(float)
    )
    return metrics, daily_ret, equity, pseudo_score


# ── Permutation Test ────────────────────────────────────────────────────

def permutation_test(daily_returns, scores, run_fn_from_scores, n_perms=N_PERMS):
    """Shuffle signal dates, recompute Sharpe. Returns p-value."""
    rng = np.random.RandomState(SEED)
    dr_std = daily_returns.std()
    actual_sharpe = daily_returns.mean() / dr_std * np.sqrt(252) if dr_std > 0 else 0

    score_vals = scores.values.copy()
    count_better = 0
    for _ in range(n_perms):
        rng.shuffle(score_vals)
        shuffled = pd.Series(score_vals.copy(), index=scores.index)
        perm_ret = run_fn_from_scores(shuffled)
        ps = perm_ret.std()
        perm_sharpe = perm_ret.mean() / ps * np.sqrt(252) if ps > 0 else 0
        if perm_sharpe >= actual_sharpe:
            count_better += 1

    return count_better / n_perms


def validate_5gate(metrics, perm_p):
    """Apply 5-gate validation framework."""
    if metrics is None:
        return {"pass": False, "gates": {}, "reason": "No metrics"}

    gates = {
        "sharpe_gt_0.5": metrics["sharpe"] > SHARPE_MIN,
        "perm_p_lt_0.05": perm_p < PERM_P_MAX,
        "regime_gap_lt_0.5": metrics["regime_gap"] < REGIME_GAP_MAX,
        "mdd_gt_neg50": metrics["max_drawdown_pct"] > MDD_FLOOR * 100,
        "trades_gte_20": metrics["trades"] >= MIN_TRADES,
    }
    passed = sum(gates.values())
    return {
        "pass": passed >= 5,
        "gates_passed": passed,
        "gates": gates,
    }


# ── Main ────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("TREND FOLLOWING WITH MULTI-TIMEFRAME FILTERS — BACKTEST")
    print("=" * 70)

    close = download_data()
    print("\nComputing signals...")
    signals, signal_cols = compute_signals(close)
    regime = identify_regime(close)

    # OOT stats
    oot_signals = signals.loc[OOT_START:]
    print(f"\nOOT period: {oot_signals.index[0].date()} to {oot_signals.index[-1].date()} ({len(oot_signals)} days)")
    print(f"Score distribution (0-6):")
    for s in range(7):
        pct = (oot_signals["score"] == s).mean() * 100
        print(f"  Score {s}: {pct:.1f}%")
    print(f"Mean score: {oot_signals['score'].mean():.2f}")

    oot_regime = regime.loc[oot_signals.index]
    bull_pct = (oot_regime == "bull").mean() * 100
    print(f"Bull days: {bull_pct:.1f}%, Bear days: {100 - bull_pct:.1f}%")

    # Signal correlation
    print(f"\nSignal co-occurrence (% of OOT days each signal is ON):")
    for col in signal_cols:
        pct = oot_signals[col].mean() * 100
        print(f"  {col}: {pct:.1f}%")

    # Benchmarks
    oot_idx = oot_signals.index
    spy_bh_ret = close["SPY"].pct_change().loc[oot_idx]
    spy_bh_eq = (1 + spy_bh_ret).cumprod() * CAPITAL
    spy_metrics = calc_metrics(spy_bh_eq, spy_bh_ret, regime.loc[oot_idx], 1)

    qqq_bh_ret = close["QQQ"].pct_change().loc[oot_idx]
    qqq_bh_eq = (1 + qqq_bh_ret).cumprod() * CAPITAL
    qqq_metrics = calc_metrics(qqq_bh_eq, qqq_bh_ret, regime.loc[oot_idx], 1)

    print(f"\n{'='*70}")
    print("BENCHMARKS (OOT)")
    print(f"{'='*70}")
    print(f"SPY B&H: Return={spy_metrics['total_return_pct']:.1f}%, Sharpe={spy_metrics['sharpe']:.3f}, MDD={spy_metrics['max_drawdown_pct']:.1f}%, Sortino={spy_metrics['sortino']:.3f}")
    print(f"QQQ B&H: Return={qqq_metrics['total_return_pct']:.1f}%, Sharpe={qqq_metrics['sharpe']:.3f}, MDD={qqq_metrics['max_drawdown_pct']:.1f}%, Sortino={qqq_metrics['sortino']:.3f}")

    results = {
        "metadata": {
            "script": "trend_following_backtest.py",
            "strategy": "Trend Following with Multi-Timeframe Filters",
            "run_date": datetime.now().isoformat(),
            "oot_start": OOT_START,
            "oot_end": OOT_END,
            "capital": CAPITAL,
            "slippage_pct": SLIPPAGE_PCT,
            "n_permutations": N_PERMS,
            "oot_days": len(oot_signals),
        },
        "signal_stats": {
            "mean_score": round(oot_signals["score"].mean(), 2),
            "score_distribution": {
                str(s): round((oot_signals["score"] == s).mean(), 4) for s in range(7)
            },
            "signal_on_pct": {
                col: round(oot_signals[col].mean() * 100, 1) for col in signal_cols
            },
            "bull_pct": round(bull_pct, 1),
            "bear_pct": round(100 - bull_pct, 1),
        },
        "benchmarks": {
            "SPY_buy_hold": spy_metrics,
            "QQQ_buy_hold": qqq_metrics,
        },
        "variants": {},
    }

    # ── Permutation helper lambdas ──────────────────────────────────────
    qqq_ret_oot = close["QQQ"].pct_change().loc[oot_idx]
    spy_ret_oot = close["SPY"].pct_change().loc[oot_idx]

    def perm_a(shuffled_scores):
        inv = shuffled_scores == 6
        r = pd.Series(0.0, index=oot_idx)
        r[inv] = qqq_ret_oot[inv]
        return r

    def perm_b(shuffled_scores):
        inv = shuffled_scores >= 5
        r = pd.Series(0.0, index=oot_idx)
        r[inv] = qqq_ret_oot[inv]
        return r

    def perm_c(shuffled_scores):
        inv = shuffled_scores >= 4
        r = pd.Series(0.0, index=oot_idx)
        r[inv] = qqq_ret_oot[inv]
        return r

    def perm_d(shuffled_scores):
        all6 = shuffled_scores == 6
        mid = (shuffled_scores >= 4) & (shuffled_scores < 6)
        r = pd.Series(0.0, index=oot_idx)
        r[all6] = qqq_ret_oot[all6]
        r[mid] = spy_ret_oot[mid]
        return r

    def perm_e(shuffled_scores):
        # Simplified permutation for trailing stop — ignore stop logic,
        # just test signal timing (score >= 5 → invested)
        inv = shuffled_scores >= 5
        r = pd.Series(0.0, index=oot_idx)
        r[inv] = qqq_ret_oot[inv]
        return r

    def perm_f(shuffled_scores):
        # For F, pseudo-score is 5 signals (4 QQQ + VIX<20), invest when all 5 true
        inv = shuffled_scores == 5
        r = pd.Series(0.0, index=oot_idx)
        r[inv] = qqq_ret_oot[inv]
        return r

    # ── Run Variants ────────────────────────────────────────────────────
    variants = {
        "A_all6_strict": (run_variant_a, perm_a),
        "B_5plus": (run_variant_b, perm_b),
        "C_4plus": (run_variant_c, perm_c),
        "D_leveraged": (run_variant_d, perm_d),
        "E_trailing_stop": (run_variant_e, perm_e),
        "F_inverse_regime": (run_variant_f, perm_f),
    }

    print(f"\n{'='*70}")
    print("VARIANT RESULTS")
    print(f"{'='*70}")

    for name, (run_fn, perm_fn) in variants.items():
        print(f"\n--- Variant {name} ---")
        metrics, daily_ret, equity, scores = run_fn(signals, close, regime)

        if metrics is None:
            print("  SKIPPED: insufficient data")
            results["variants"][name] = {"status": "skipped"}
            continue

        # Time invested
        invested_pct = (daily_ret != 0).mean() * 100
        print(f"  Time invested: {invested_pct:.1f}%")

        # Permutation test
        print(f"  Running {N_PERMS} permutations...")
        perm_p = permutation_test(daily_ret, scores, perm_fn, N_PERMS)

        validation = validate_5gate(metrics, perm_p)

        print(f"  Return: {metrics['total_return_pct']:+.1f}%")
        print(f"  Sharpe: {metrics['sharpe']:.3f}  Sortino: {metrics['sortino']:.3f}")
        print(f"  MDD: {metrics['max_drawdown_pct']:.1f}%  PF: {metrics['profit_factor']:.2f}  WR: {metrics['win_rate']:.1%}")
        print(f"  Trades: {metrics['trades']}")
        print(f"  Bull Sharpe: {metrics['bull_sharpe']:.3f}  Bear Sharpe: {metrics['bear_sharpe']:.3f}  Gap: {metrics['regime_gap']:.3f}")
        print(f"  Perm p-value: {perm_p:.4f}")
        print(f"  Validation: {validation['gates_passed']}/5 gates {'PASS' if validation['pass'] else 'FAIL'}")
        for gate, passed in validation["gates"].items():
            print(f"    {gate}: {'PASS' if passed else 'FAIL'}")

        results["variants"][name] = {
            "metrics": metrics,
            "time_invested_pct": round(invested_pct, 1),
            "perm_p_value": round(perm_p, 4),
            "validation": validation,
        }

    # ── Summary Table ───────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"{'Variant':<22} {'Sharpe':>8} {'Sortino':>8} {'Return%':>9} {'MDD%':>8} {'PermP':>8} {'Invested':>9} {'Gates':>6}")
    print("-" * 80)

    for name, v in results["variants"].items():
        if v.get("status") == "skipped":
            continue
        m = v["metrics"]
        vd = v["validation"]
        print(
            f"{name:<22} {m['sharpe']:>8.3f} {m['sortino']:>8.3f} "
            f"{m['total_return_pct']:>+8.1f}% {m['max_drawdown_pct']:>7.1f}% "
            f"{v['perm_p_value']:>8.4f} {v['time_invested_pct']:>8.1f}% "
            f"{vd['gates_passed']:>2}/5 {'OK' if vd['pass'] else 'XX'}"
        )

    print(f"\n{'SPY B&H':<22} {spy_metrics['sharpe']:>8.3f} {spy_metrics['sortino']:>8.3f} {spy_metrics['total_return_pct']:>+8.1f}% {spy_metrics['max_drawdown_pct']:>7.1f}%")
    print(f"{'QQQ B&H':<22} {qqq_metrics['sharpe']:>8.3f} {qqq_metrics['sortino']:>8.3f} {qqq_metrics['total_return_pct']:>+8.1f}% {qqq_metrics['max_drawdown_pct']:>7.1f}%")

    # Key insight
    print(f"\n{'='*70}")
    print("KEY INSIGHT")
    print(f"{'='*70}")
    passed_variants = [n for n, v in results["variants"].items()
                       if v.get("validation", {}).get("pass", False)]
    if passed_variants:
        print(f"PASSED 5-gate: {', '.join(passed_variants)}")
        best = max(passed_variants,
                   key=lambda n: results["variants"][n]["metrics"]["sharpe"])
        bm = results["variants"][best]["metrics"]
        print(f"Best: {best} — Sharpe {bm['sharpe']:.3f}, Return {bm['total_return_pct']:+.1f}%, MDD {bm['max_drawdown_pct']:.1f}%")
    else:
        print("NO variant passed all 5 gates.")
        print("Multi-timeframe trend filters likely do NOT add genuine alpha")
        print("beyond what's explained by market-timing luck or regime exposure.")

    # Save results
    output_path = Path("/home/jupiter/Lvl3Quant/data/trend_following_results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")

    return results


if __name__ == "__main__":
    results = main()
