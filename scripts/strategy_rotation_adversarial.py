#!/usr/bin/env python3
"""
Adversarial Validation for Strategy Rotation Variant A (Simple Regime Switch).
Uses the SAME daily-return simulation engine as strategy_rotation_backtest.py.

Variant A: 75% earnings momentum + 15% VIX fade + 10% contrarian.
Bull -> more QQQ/earnings momentum, Bear -> contrarian dip-buys, VIX spike -> VIX fade.
Passed all 5 gates with Sharpe 2.13, 69 trades, $645 -> $2,721.

Tests:
1. Inverse Signal — short instead of long. Should have negative Sharpe.
2. Random Instruments — same rotation logic on non-overlapping instruments. Sharpe < 0.3.
3. Sub-Period Stability — 3 sub-periods, all Sharpe >= 0.
4. Top Trade Removal — remove top 3/5, Sharpe > 0.5 after removing top 5.
5. Parameter Sensitivity — sweep allocation weights/hold period. >=3/4 combos Sharpe > 0.5.
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
np.random.seed(42)

# ── CONFIG ──────────────────────────────────────────────────────────────────
OOT_START = "2022-01-01"
OOT_END   = "2026-07-29"
STARTING_CAPITAL = 645.0
RESULTS_PATH = Path("/home/jupiter/Lvl3Quant/data/strategy_rotation_adversarial_results.json")

# Random instruments for test 2
RANDOM_INSTRUMENTS = ["TLT", "GLD", "SLV", "USO", "EEM", "FXI", "EWJ",
                      "VNQ", "IYR", "HYG", "LQD", "DBA", "UNG"]

# Sub-periods for test 3
SUB_PERIODS = [
    ("2022H1", "2022-01-01", "2022-07-01"),
    ("2022H2-2023", "2022-07-01", "2024-01-01"),
    ("2024-2026", "2024-01-01", "2026-08-01"),
]


# ── DATA HELPERS ────────────────────────────────────────────────────────────
def fetch_data(tickers):
    """Download daily close data for a list of tickers."""
    raw = yf.download(tickers, start="2021-01-01", end=OOT_END,
                      auto_adjust=True, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Close"]
    else:
        close = raw[["Close"]].copy()
        close.columns = tickers if isinstance(tickers, list) else [tickers]
    if isinstance(close.columns, pd.MultiIndex):
        close.columns = close.columns.get_level_values(-1)
    return close.dropna(how="all")


def _rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss
    return 100 - 100 / (1 + rs)


def _daily_returns(prices):
    return prices.pct_change().fillna(0)


def compute_regime_signals(df, spy_col="SPY", vix_col="^VIX"):
    """Add regime indicator columns using SPY and VIX."""
    df = df.copy()
    df["SMA200"]    = df[spy_col].rolling(200).mean()
    df["RSI14"]     = _rsi(df[spy_col], 14)
    df["SPY_ret5"]  = df[spy_col].pct_change(5)
    df["SPY_ret20"] = df[spy_col].pct_change(20)
    df["SPY_ret60"] = df[spy_col].pct_change(60)
    df["VIX_chg5"]  = df[vix_col].pct_change(5)
    df["bull"]      = (df[spy_col] > df["SMA200"]).astype(int)
    df["VIX"]       = df[vix_col]
    return df.dropna()


# ── VARIANT A ROTATION LOGIC ───────────────────────────────────────────────
def variant_A_labels(df):
    """Simple Regime Switch: VIX>25 -> vix_fade, bull -> earnings_momentum, else -> contrarian."""
    labels = []
    for i in range(len(df)):
        if df["VIX"].iloc[i] > 25:
            labels.append("vix_fade")
        elif df["bull"].iloc[i]:
            labels.append("earnings_momentum")
        else:
            labels.append("contrarian")
    return pd.Series(labels, index=df.index)


# ── EXECUTE ROTATION (matching original backtest engine) ───────────────────
def execute_rotation(df, rotation_labels, bull_ticker="QQQ", bear_ticker="SPY",
                     vix_ticker="SPY", inverse=False, dip_threshold=-0.03,
                     bear_hold_days=5):
    """
    Given strategy labels per day, produce equity curve using daily returns.
    If inverse=True, negate all returns (short instead of long).
    """
    spy_ret = _daily_returns(df[bear_ticker]) if bear_ticker in df.columns else pd.Series(0, index=df.index)
    bull_ret = _daily_returns(df[bull_ticker]) if bull_ticker in df.columns else pd.Series(0, index=df.index)
    vix_ret = _daily_returns(df[vix_ticker]) if vix_ticker in df.columns else pd.Series(0, index=df.index)

    daily_ret = pd.Series(0.0, index=df.index)
    n_trades = 0
    prev_label = None
    hold_remaining = 0
    trade_rets = []  # list of (start_idx, end_idx, cumulative_ret) for each trade
    current_trade_start = None
    current_trade_rets = []

    for i in range(len(df)):
        label = rotation_labels.iloc[i]
        if label != prev_label:
            # Close previous trade
            if current_trade_start is not None and len(current_trade_rets) > 0:
                cum = np.prod([1 + r for r in current_trade_rets]) - 1
                trade_rets.append(cum)
            n_trades += 1
            hold_remaining = 0
            current_trade_start = i
            current_trade_rets = []
        prev_label = label

        ret = 0.0
        if label == "earnings_momentum":
            ret = bull_ret.iloc[i]
        elif label == "contrarian":
            if hold_remaining > 0:
                ret = spy_ret.iloc[i]
                hold_remaining -= 1
            elif "SPY_ret5" in df.columns and df["SPY_ret5"].iloc[i] <= dip_threshold:
                ret = spy_ret.iloc[i]
                hold_remaining = bear_hold_days - 1
            # else: cash (0)
        elif label == "vix_fade":
            if df["VIX"].iloc[i] > 25 and df["VIX_chg5"].iloc[i] < 0:
                ret = vix_ret.iloc[i]
        elif label == "spy_hold":
            ret = spy_ret.iloc[i]
        # cash -> 0

        if inverse:
            ret = -ret

        daily_ret.iloc[i] = ret
        current_trade_rets.append(ret)

    # Close last trade
    if current_trade_rets:
        cum = np.prod([1 + r for r in current_trade_rets]) - 1
        trade_rets.append(cum)

    equity = STARTING_CAPITAL * (1 + daily_ret).cumprod()
    return daily_ret, equity, n_trades, trade_rets


# ── METRICS ──────────────────────────────────────────────────────────────
def compute_metrics(daily_ret, equity, n_trades, df=None):
    trading_days = daily_ret[daily_ret != 0]
    total_ret = (equity.iloc[-1] / equity.iloc[0]) - 1
    ann_ret = (1 + total_ret) ** (252 / max(len(daily_ret), 1)) - 1
    vol = daily_ret.std() * np.sqrt(252) if daily_ret.std() > 0 else 1e-9
    sharpe = ann_ret / vol if vol > 0 else 0

    downside = daily_ret[daily_ret < 0].std() * np.sqrt(252) if (daily_ret < 0).sum() > 0 else 1e-9
    sortino = ann_ret / downside

    wins = (trading_days > 0).sum()
    losses = (trading_days < 0).sum()
    wr = wins / (wins + losses) if (wins + losses) > 0 else 0
    avg_win = trading_days[trading_days > 0].mean() if wins > 0 else 0
    avg_loss = abs(trading_days[trading_days < 0].mean()) if losses > 0 else 1e-9
    pf = (avg_win * wins) / (avg_loss * losses) if (avg_loss * losses) > 0 else 999

    peak = equity.cummax()
    dd = (equity - peak) / peak
    mdd = dd.min()

    result = {
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "profit_factor": round(pf, 3),
        "win_rate": round(wr, 3),
        "max_drawdown": round(mdd, 4),
        "n_trades": n_trades,
        "total_return_pct": round(total_ret * 100, 2),
        "final_equity": round(equity.iloc[-1], 2),
    }

    # Regime-stratified Sharpe if df available
    if df is not None and "bull" in df.columns:
        bull_mask = df["bull"] == 1
        bear_mask = df["bull"] == 0

        def _sharpe_subset(rets):
            if len(rets) < 5 or rets.std() == 0:
                return 0.0
            return (rets.mean() / rets.std()) * np.sqrt(252)

        result["sharpe_bull"] = round(_sharpe_subset(daily_ret[bull_mask]), 3)
        result["sharpe_bear"] = round(_sharpe_subset(daily_ret[bear_mask]), 3)
        result["regime_gap"] = round(
            abs(result["sharpe_bull"] - result["sharpe_bear"]) /
            max(abs(result["sharpe_bull"]), abs(result["sharpe_bear"]), 0.01), 3)

    return result


# ── TEST 1: INVERSE SIGNAL ─────────────────────────────────────────────────
def test_inverse(df_oot):
    """Short instead of long on all signals. Should have negative Sharpe."""
    print("\n=== TEST 1: INVERSE SIGNAL (short instead of long) ===")
    labels = variant_A_labels(df_oot)
    daily_ret, equity, n_trades, _ = execute_rotation(df_oot, labels, inverse=True)
    metrics = compute_metrics(daily_ret, equity, n_trades, df_oot)
    passed = metrics["sharpe"] < 0
    print(f"  Inverse Sharpe: {metrics['sharpe']} (should be < 0)")
    print(f"  Final equity: ${metrics['final_equity']} (from ${STARTING_CAPITAL})")
    print(f"  Result: {'PASS' if passed else 'FAIL'}")
    return {
        "pass": passed,
        "inverse_sharpe": metrics["sharpe"],
        "threshold": "Sharpe < 0 (negative)",
        "detail": metrics,
    }


# ── TEST 2: RANDOM INSTRUMENTS ────────────────────────────────────────────
def test_random_instruments(df_oot, original_sharpe):
    """Apply same rotation logic to non-overlapping instruments. Sharpe < 0.3."""
    print("\n=== TEST 2: RANDOM INSTRUMENTS ===")
    print(f"  Downloading random instrument data: {RANDOM_INSTRUMENTS}")

    # Download random instruments + SPY + VIX for regime signals
    all_tickers = RANDOM_INSTRUMENTS + ["SPY", "^VIX"]
    close = fetch_data(all_tickers)

    # Build regime signals from SPY/VIX
    regime_df = close[["SPY", "^VIX"]].dropna()
    regime_df = compute_regime_signals(regime_df)
    regime_df = regime_df.loc[OOT_START:OOT_END]

    # Find available random tickers
    available = [t for t in RANDOM_INSTRUMENTS if t in close.columns]
    print(f"  Available random instruments: {len(available)}/{len(RANDOM_INSTRUMENTS)}")

    # Pick 2 as bull_ticker and bear_ticker from random, use a 3rd for VIX fade
    # Use the first few available as substitutes for QQQ/SPY
    if len(available) < 3:
        print("  Not enough random instruments available!")
        return {"pass": False, "error": "insufficient data"}

    # Run multiple combos and average
    combos = [
        (available[0], available[1], available[2]),  # TLT, GLD, SLV
        (available[3], available[4], available[5]) if len(available) > 5 else (available[0], available[2], available[1]),
        (available[6], available[7], available[8]) if len(available) > 8 else (available[1], available[0], available[2]),
    ]

    sharpes = []
    combo_details = {}
    for bull_t, bear_t, vix_t in combos:
        combo_label = f"{bull_t}/{bear_t}/{vix_t}"
        # Build df with needed columns
        needed_dates = regime_df.index
        combo_df = regime_df.copy()

        for t in [bull_t, bear_t, vix_t]:
            if t in close.columns:
                combo_df[t] = close[t].reindex(needed_dates)

        combo_df = combo_df.dropna()
        if len(combo_df) < 50:
            print(f"    {combo_label}: insufficient data ({len(combo_df)} days)")
            continue

        labels = variant_A_labels(combo_df)
        daily_ret, equity, n_trades, _ = execute_rotation(
            combo_df, labels, bull_ticker=bull_t, bear_ticker=bear_t, vix_ticker=vix_t)
        m = compute_metrics(daily_ret, equity, n_trades, combo_df)
        sharpes.append(m["sharpe"])
        combo_details[combo_label] = m
        print(f"    {combo_label}: Sharpe={m['sharpe']}, final=${m['final_equity']}")

    avg_sharpe = np.mean(sharpes) if sharpes else 0
    passed = avg_sharpe < 0.3
    print(f"  Average random Sharpe: {round(avg_sharpe, 3)} (threshold: < 0.3)")
    print(f"  Original Sharpe: {original_sharpe}")
    print(f"  Result: {'PASS' if passed else 'FAIL'}")
    return {
        "pass": passed,
        "avg_random_sharpe": round(avg_sharpe, 3),
        "original_sharpe": original_sharpe,
        "threshold": "avg Sharpe < 0.3",
        "instruments_used": RANDOM_INSTRUMENTS[:len(available)],
        "combo_details": combo_details,
    }


# ── TEST 3: SUB-PERIOD STABILITY ──────────────────────────────────────────
def test_subperiod_stability(df_oot):
    """Split OOT into 3 sub-periods. All should have Sharpe >= 0."""
    print("\n=== TEST 3: SUB-PERIOD STABILITY ===")
    sub_results = {}
    all_positive = True

    for label, start, end in SUB_PERIODS:
        sub_df = df_oot.loc[start:end].copy()
        if len(sub_df) < 20:
            print(f"  {label}: too few days ({len(sub_df)})")
            sub_results[label] = {"sharpe": 0, "n_days": len(sub_df), "note": "too few days"}
            continue

        labels = variant_A_labels(sub_df)
        daily_ret, equity, n_trades, _ = execute_rotation(sub_df, labels)
        m = compute_metrics(daily_ret, equity, n_trades, sub_df)
        sub_results[label] = m
        print(f"  {label}: Sharpe={m['sharpe']}, PF={m['profit_factor']}, "
              f"WR={m['win_rate']:.1%}, final=${m['final_equity']}, trades={m['n_trades']}")
        if m["sharpe"] < 0:
            all_positive = False

    passed = all_positive
    print(f"  Result: {'PASS' if passed else 'FAIL'}")
    return {
        "pass": passed,
        "sub_periods": sub_results,
        "criterion": "all sub-period Sharpe >= 0",
    }


# ── TEST 4: TOP TRADE REMOVAL ─────────────────────────────────────────────
def test_top_trade_removal(df_oot):
    """Remove top 3 and top 5 trades by P&L. Sharpe should remain > 0.5 after removing top 5."""
    print("\n=== TEST 4: TOP TRADE REMOVAL ===")
    labels = variant_A_labels(df_oot)
    daily_ret, equity, n_trades, trade_rets = execute_rotation(df_oot, labels)
    original_metrics = compute_metrics(daily_ret, equity, n_trades, df_oot)
    original_sharpe = original_metrics["sharpe"]

    # Identify trade boundaries
    trade_boundaries = []
    prev_label = None
    trade_start = 0
    for i in range(len(labels)):
        if labels.iloc[i] != prev_label:
            if prev_label is not None:
                trade_boundaries.append((trade_start, i - 1))
            trade_start = i
        prev_label = labels.iloc[i]
    trade_boundaries.append((trade_start, len(labels) - 1))

    # Compute P&L per trade segment
    trade_pnls = []
    for start, end in trade_boundaries:
        segment_ret = daily_ret.iloc[start:end + 1]
        # P&L in return terms
        cum_ret = np.prod(1 + segment_ret.values) - 1
        trade_pnls.append({
            "start": start,
            "end": end,
            "cum_ret": cum_ret,
            "n_days": end - start + 1,
        })

    # Sort by absolute P&L descending
    sorted_trades = sorted(trade_pnls, key=lambda t: t["cum_ret"], reverse=True)

    removal_results = {}
    for n_remove in [3, 5]:
        if len(sorted_trades) <= n_remove:
            removal_results[f"remove_top_{n_remove}"] = {"sharpe": 0, "note": "too few trades"}
            continue

        # Zero out the top N trades' daily returns
        adj_ret = daily_ret.copy()
        removed_rets = []
        for t in sorted_trades[:n_remove]:
            removed_rets.append(round(t["cum_ret"] * 100, 2))
            adj_ret.iloc[t["start"]:t["end"] + 1] = 0.0

        adj_equity = STARTING_CAPITAL * (1 + adj_ret).cumprod()
        adj_metrics = compute_metrics(adj_ret, adj_equity, n_trades - n_remove, df_oot)
        removal_results[f"remove_top_{n_remove}"] = {
            "sharpe": adj_metrics["sharpe"],
            "final_equity": adj_metrics["final_equity"],
            "removed_trade_returns_pct": removed_rets,
        }
        print(f"  Remove top {n_remove}: Sharpe={adj_metrics['sharpe']}, "
              f"final=${adj_metrics['final_equity']}")
        print(f"    Removed trade returns: {removed_rets}")

    sharpe_after_5 = removal_results.get("remove_top_5", {}).get("sharpe", 0)
    passed = sharpe_after_5 > 0.5
    print(f"  Sharpe after removing top 5: {sharpe_after_5} (threshold: > 0.5)")
    print(f"  Original Sharpe: {original_sharpe}")
    print(f"  Result: {'PASS' if passed else 'FAIL'}")
    return {
        "pass": passed,
        "original_sharpe": original_sharpe,
        "removal_results": removal_results,
        "criterion": "Sharpe > 0.5 after removing top 5 trades",
    }


# ── TEST 5: PARAMETER SENSITIVITY ─────────────────────────────────────────
def test_parameter_sensitivity(df_oot):
    """
    Sweep allocation weights (earnings momentum 50-90%) and hold period +/-50%.
    At least 3/4 parameter combos should maintain Sharpe > 0.5.

    We vary the VIX threshold (controls how much time in VIX fade vs momentum)
    and the bear hold period.
    """
    print("\n=== TEST 5: PARAMETER SENSITIVITY ===")

    # The original uses VIX > 25 threshold and bear hold 5 days.
    # Varying VIX threshold changes how much is allocated to earnings_momentum.
    # Varying hold period changes contrarian trade duration.
    variations = [
        {"label": "VIX_thresh_20_hold_3d", "vix_thresh": 20, "hold": 3,
         "desc": "More VIX fade, shorter hold"},
        {"label": "VIX_thresh_30_hold_5d", "vix_thresh": 30, "hold": 5,
         "desc": "Less VIX fade (more momentum), default hold"},
        {"label": "VIX_thresh_25_hold_8d", "vix_thresh": 25, "hold": 8,
         "desc": "Default VIX, longer hold (+60%)"},
        {"label": "VIX_thresh_35_hold_3d", "vix_thresh": 35, "hold": 3,
         "desc": "Much more momentum, shorter hold"},
    ]

    var_results = {}
    sharpes_above = 0

    for v in variations:
        # Modify labels based on VIX threshold
        vix_thresh = v["vix_thresh"]
        hold = v["hold"]
        labels = []
        for i in range(len(df_oot)):
            if df_oot["VIX"].iloc[i] > vix_thresh:
                labels.append("vix_fade")
            elif df_oot["bull"].iloc[i]:
                labels.append("earnings_momentum")
            else:
                labels.append("contrarian")
        labels = pd.Series(labels, index=df_oot.index)

        # Compute allocation percentages
        alloc = labels.value_counts(normalize=True).to_dict()
        em_pct = alloc.get("earnings_momentum", 0)

        daily_ret, equity, n_trades, _ = execute_rotation(
            df_oot, labels, dip_threshold=-0.03, bear_hold_days=hold)
        m = compute_metrics(daily_ret, equity, n_trades, df_oot)
        m["earnings_momentum_pct"] = round(em_pct * 100, 1)
        m["allocation"] = {k: round(vv, 3) for k, vv in alloc.items()}

        var_results[v["label"]] = m
        if m["sharpe"] > 0.5:
            sharpes_above += 1
        print(f"  {v['label']}: Sharpe={m['sharpe']}, EM%={m['earnings_momentum_pct']}, "
              f"trades={m['n_trades']}, final=${m['final_equity']}")

    passed = sharpes_above >= 3
    print(f"  Combos with Sharpe > 0.5: {sharpes_above}/{len(variations)} (need >= 3)")
    print(f"  Result: {'PASS' if passed else 'FAIL'}")
    return {
        "pass": passed,
        "above_threshold": sharpes_above,
        "total_variations": len(variations),
        "variations": var_results,
        "criterion": ">=3/4 parameter combos maintain Sharpe > 0.5",
    }


# ── MAIN ────────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("ADVERSARIAL VALIDATION — Strategy Rotation Variant A")
    print(f"OOT: {OOT_START} -> {OOT_END}  |  Capital: ${STARTING_CAPITAL}")
    print("=" * 70)

    # Fetch core data
    print("\n[1/6] Fetching core market data (SPY, QQQ, ^VIX)...")
    tickers = ["SPY", "QQQ", "^VIX"]
    close = fetch_data(tickers)
    df = pd.DataFrame(index=close.index)
    df["SPY"] = close["SPY"]
    df["QQQ"] = close["QQQ"]
    df["^VIX"] = close["^VIX"] if "^VIX" in close.columns else close.get("GSPC", np.nan)
    df = df.dropna()

    df = compute_regime_signals(df)
    df_oot = df.loc[OOT_START:OOT_END].copy()
    print(f"  OOT period: {len(df_oot)} trading days "
          f"({df_oot.index[0].date()} -> {df_oot.index[-1].date()})")

    # Baseline
    print("\n[2/6] Computing baseline (original Variant A)...")
    labels = variant_A_labels(df_oot)
    daily_ret, equity, n_trades, _ = execute_rotation(df_oot, labels)
    baseline = compute_metrics(daily_ret, equity, n_trades, df_oot)

    alloc = labels.value_counts(normalize=True).to_dict()
    baseline["allocation_pct"] = {k: round(v, 3) for k, v in alloc.items()}

    print(f"  Sharpe: {baseline['sharpe']}")
    print(f"  Sortino: {baseline['sortino']}")
    print(f"  PF: {baseline['profit_factor']}, WR: {baseline['win_rate']:.1%}")
    print(f"  MDD: {baseline['max_drawdown']:.1%}")
    print(f"  Trades: {baseline['n_trades']}")
    print(f"  Final equity: ${baseline['final_equity']} (from ${STARTING_CAPITAL})")
    print(f"  Allocation: {baseline['allocation_pct']}")

    original_sharpe = baseline["sharpe"]

    # Run all 5 adversarial tests
    print("\n[3/6] Running adversarial tests...")
    t1 = test_inverse(df_oot)
    t2 = test_random_instruments(df_oot, original_sharpe)
    t3 = test_subperiod_stability(df_oot)
    t4 = test_top_trade_removal(df_oot)
    t5 = test_parameter_sensitivity(df_oot)

    tests = {
        "test1_inverse": t1,
        "test2_random_instruments": t2,
        "test3_subperiod": t3,
        "test4_top_trade_removal": t4,
        "test5_param_sensitivity": t5,
    }

    passed_count = sum(1 for t in tests.values() if t["pass"])
    total = len(tests)

    # Summary
    print("\n" + "=" * 70)
    print(f"ADVERSARIAL VALIDATION SCORE: {passed_count}/{total}")
    print("=" * 70)
    for name, t in tests.items():
        status = "PASS" if t["pass"] else "FAIL"
        print(f"  [{status}] {name}")

    if passed_count == total:
        verdict = "ALL PASSED - Strategy shows genuine edge"
    elif passed_count >= 4:
        verdict = "MOSTLY ROBUST - Minor concerns"
    elif passed_count >= 3:
        verdict = "MIXED - Significant concerns, review failed tests"
    else:
        verdict = "FRAGILE - Strategy likely overfitted or regime-dependent"
    print(f"\nVERDICT: {verdict}")

    # Save results (matching volume_anomaly_adversarial_results.json structure)
    output = {
        "test": "strategy_rotation_adversarial_validation",
        "run_timestamp": datetime.now().isoformat(),
        "oot_period": f"{OOT_START} to {OOT_END}",
        "starting_capital": STARTING_CAPITAL,
        "strategy": "Rotation Variant A (Simple Regime Switch)",
        "strategy_description": ("75% earnings momentum + 15% VIX fade + 10% contrarian, "
                                 "rotating weights by regime. Bull -> QQQ/earnings momentum, "
                                 "Bear -> SPY contrarian dip-buys, VIX spike -> VIX fade."),
        "baseline": baseline,
        "adversarial_tests": tests,
        "score": f"{passed_count}/{total}",
        "passed": passed_count,
        "total": total,
        "verdict": verdict,
    }

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {RESULTS_PATH}")

    return passed_count, total


if __name__ == "__main__":
    passed, total = main()
    sys.exit(0 if passed >= 3 else 1)
