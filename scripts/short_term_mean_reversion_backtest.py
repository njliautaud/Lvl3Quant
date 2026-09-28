#!/usr/bin/env python3
"""
Short-Term Mean Reversion Backtest — 6 Variants
Walk-forward OOT: Jan 2022 – Jul 2026
5-gate validation + adversarial tests on passing variants.

Cost model: $0 commission (Robinhood), 0.02% slippage per trade (RT).
"""

import json
import datetime as dt
import warnings
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")
np.random.seed(42)

# ── Data download ────────────────────────────────────────────────────────────

def get_data():
    """Download SPY daily data with margin for indicators."""
    # Need 200-day SMA so fetch extra history
    start = "2020-06-01"
    end = "2026-07-30"
    df = yf.download("SPY", start=start, end=end, auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.sort_index()
    df.index = df.index.tz_localize(None) if df.index.tz else df.index
    return df


# ── Indicators ───────────────────────────────────────────────────────────────

def rsi(series, period=2):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - 100 / (1 + rs)


def ibs(df):
    """Internal Bar Strength = (Close - Low) / (High - Low)"""
    return (df["Close"] - df["Low"]) / (df["High"] - df["Low"])


def sma(series, period):
    return series.rolling(period).mean()


# ── Strategy signal generators ───────────────────────────────────────────────
# Each returns a DataFrame with columns: entry (bool), exit (bool)
# plus any intermediate cols needed.

SLIPPAGE_RT = 0.0002  # 0.02% round-trip (entry + exit)


def variant_a(df):
    """Classic RSI(2) < 10. Exit RSI(2) > 70. Max hold 10 days."""
    r = rsi(df["Close"], 2)
    entries = r < 10
    exits = r > 70
    return entries, exits, 10


def variant_b(df):
    """RSI(2) < 5 + Above 200-SMA. Exit RSI(2) > 60."""
    r = rsi(df["Close"], 2)
    s200 = sma(df["Close"], 200)
    entries = (r < 5) & (df["Close"] > s200)
    exits = r > 60
    return entries, exits, 10


def variant_c(df):
    """3 consecutive down days (each close < prior close). Exit first up day or 5d max."""
    down = df["Close"] < df["Close"].shift(1)
    streak3 = down & down.shift(1) & down.shift(2)
    up = df["Close"] > df["Close"].shift(1)
    return streak3, up, 5


def variant_d(df):
    """2-day drop > 3%. Hold 5 days (fixed)."""
    ret2 = df["Close"].pct_change(2)
    entries = ret2 < -0.03
    # Exit: fixed 5-day hold (no signal-based exit)
    exits = pd.Series(False, index=df.index)
    return entries, exits, 5


def variant_e(df):
    """IBS < 0.2 for 2 consecutive days. Exit IBS > 0.8 or 5 days."""
    ib = ibs(df)
    low_ibs = ib < 0.2
    entries = low_ibs & low_ibs.shift(1)
    exits = ib > 0.8
    return entries, exits, 5


def variant_f(df):
    """RSI(2) < 15 AND 2-day return < -1.5% AND above 200-SMA. Exit RSI(2) > 50 or 7d."""
    r = rsi(df["Close"], 2)
    s200 = sma(df["Close"], 200)
    ret2 = df["Close"].pct_change(2)
    entries = (r < 15) & (ret2 < -0.015) & (df["Close"] > s200)
    exits = r > 50
    return entries, exits, 7


VARIANTS = {
    "A_RSI2_lt10": variant_a,
    "B_RSI2_lt5_above200SMA": variant_b,
    "C_3day_streak": variant_c,
    "D_2day_drop_3pct": variant_d,
    "E_IBS_lt02": variant_e,
    "F_combined_filter": variant_f,
}


# ── Backtest engine ──────────────────────────────────────────────────────────

def run_backtest(df, entries, exits, max_hold, oot_start="2022-01-01"):
    """
    Simple long-only backtest. One position at a time.
    Returns trade list with entry/exit dates, returns, regime info.
    """
    oot = df.loc[oot_start:]
    closes = oot["Close"].values
    dates = oot.index
    entry_sig = entries.reindex(oot.index).fillna(False).values
    exit_sig = exits.reindex(oot.index).fillna(False).values

    # 20-SMA for regime classification
    sma20_full = sma(df["Close"], 20)
    sma20 = sma20_full.reindex(oot.index).values

    trades = []
    in_trade = False
    entry_price = 0
    entry_date = None
    hold_days = 0

    for i in range(len(closes)):
        if in_trade:
            hold_days += 1
            should_exit = exit_sig[i] or hold_days >= max_hold
            if should_exit:
                exit_price = closes[i]
                raw_ret = (exit_price / entry_price) - 1
                net_ret = raw_ret - SLIPPAGE_RT
                regime = "green" if entry_price > sma20[trades[-1]["entry_idx"]] else "red"
                trades[-1].update({
                    "exit_date": str(dates[i].date()),
                    "exit_price": float(exit_price),
                    "raw_return": float(raw_ret),
                    "net_return": float(net_ret),
                    "hold_days": hold_days,
                    "regime": regime,
                })
                in_trade = False
        else:
            if entry_sig[i]:
                entry_price = closes[i]
                entry_date = dates[i]
                hold_days = 0
                in_trade = True
                trades.append({
                    "entry_date": str(entry_date.date()),
                    "entry_price": float(entry_price),
                    "entry_idx": i,
                })

    # Drop incomplete trades
    trades = [t for t in trades if "exit_date" in t]
    return trades


# ── Metrics ──────────────────────────────────────────────────────────────────

def calc_metrics(trades, df, oot_start="2022-01-01"):
    """Calculate all required metrics from trade list."""
    if len(trades) < 2:
        return None

    returns = np.array([t["net_return"] for t in trades])
    n_trades = len(trades)
    win_rate = np.mean(returns > 0)

    # Build daily return series for Sharpe/Sortino
    oot = df.loc[oot_start:]
    daily_strat = pd.Series(0.0, index=oot.index)
    for t in trades:
        entry = pd.Timestamp(t["entry_date"])
        exit_ = pd.Timestamp(t["exit_date"])
        mask = (oot.index > entry) & (oot.index <= exit_)
        daily_rets = oot.loc[mask, "Close"].pct_change().fillna(0)
        daily_strat.loc[mask] = daily_rets.values[:mask.sum()]
    # Apply slippage on entry/exit days
    for t in trades:
        entry = pd.Timestamp(t["entry_date"])
        exit_ = pd.Timestamp(t["exit_date"])
        # spread slippage cost across entry day
        if entry in daily_strat.index:
            daily_strat.loc[entry] -= SLIPPAGE_RT / 2
        if exit_ in daily_strat.index:
            daily_strat.loc[exit_] -= SLIPPAGE_RT / 2

    ann_factor = np.sqrt(252)
    mean_d = daily_strat.mean()
    std_d = daily_strat.std()
    sharpe = (mean_d / std_d * ann_factor) if std_d > 0 else 0

    downside = daily_strat[daily_strat < 0].std()
    sortino = (mean_d / downside * ann_factor) if downside > 0 else 0

    # Profit factor
    gross_profit = returns[returns > 0].sum() if (returns > 0).any() else 0
    gross_loss = abs(returns[returns < 0].sum()) if (returns < 0).any() else 1e-9
    pf = gross_profit / gross_loss

    # Max drawdown from equity curve
    equity = (1 + daily_strat).cumprod()
    peak = equity.cummax()
    dd = (equity - peak) / peak
    max_dd = dd.min()

    # Regime analysis
    green_rets = [t["net_return"] for t in trades if t.get("regime") == "green"]
    red_rets = [t["net_return"] for t in trades if t.get("regime") == "red"]

    def trade_sharpe(rets):
        if len(rets) < 2:
            return 0
        r = np.array(rets)
        return (r.mean() / r.std() * np.sqrt(len(r))) if r.std() > 0 else 0

    sharpe_green = trade_sharpe(green_rets)
    sharpe_red = trade_sharpe(red_rets)
    denom = max(abs(sharpe_green), abs(sharpe_red), 1e-9)
    regime_gap = abs(sharpe_green - sharpe_red) / denom

    # Permutation test (500 iterations)
    actual_mean = returns.mean()
    n_perm = 500
    perm_count = 0
    for _ in range(n_perm):
        signs = np.random.choice([-1, 1], size=n_trades)
        perm_mean = (returns * signs).mean()
        if perm_mean >= actual_mean:
            perm_count += 1
    perm_p = perm_count / n_perm

    return {
        "n_trades": n_trades,
        "sharpe": round(float(sharpe), 4),
        "sortino": round(float(sortino), 4),
        "profit_factor": round(float(pf), 4),
        "win_rate": round(float(win_rate), 4),
        "max_drawdown": round(float(max_dd), 4),
        "mean_return_per_trade": round(float(returns.mean()), 6),
        "total_return": round(float((1 + returns).prod() - 1), 4),
        "sharpe_green": round(float(sharpe_green), 4),
        "sharpe_red": round(float(sharpe_red), 4),
        "regime_gap": round(float(regime_gap), 4),
        "n_green": len(green_rets),
        "n_red": len(red_rets),
        "perm_p": round(float(perm_p), 4),
        "daily_strat": daily_strat,  # keep for adversarial tests
        "returns": returns,
        "trades": trades,
    }


# ── 5-Gate Check ─────────────────────────────────────────────────────────────

def five_gate_check(m):
    gates = {
        "sharpe_gt_0.5": m["sharpe"] > 0.5,
        "perm_p_lt_0.05": m["perm_p"] < 0.05,
        "regime_gap_lt_0.5": m["regime_gap"] < 0.5,
        "maxdd_gt_neg50pct": m["max_drawdown"] > -0.50,
        "trades_gte_20": m["n_trades"] >= 20,
    }
    return gates, all(gates.values())


# ── Adversarial Validation ───────────────────────────────────────────────────

def adversarial_validation(trades, metrics, df, entry_func, oot_start="2022-01-01"):
    """Run 5 adversarial tests. Returns dict of test results."""
    results = {}
    returns = metrics["returns"]
    n_trades = len(returns)
    actual_sharpe = metrics["sharpe"]

    # 1. Inverse direction: when strategy says BUY -> CASH, vice versa
    # Approximate: daily returns when NOT in trade (SPY buy-and-hold minus strategy)
    oot = df.loc[oot_start:]
    spy_daily = oot["Close"].pct_change().fillna(0)
    strat_daily = metrics["daily_strat"]
    inverse_daily = spy_daily - strat_daily  # in market when strategy is out, out when in
    inv_mean = inverse_daily.mean()
    inv_std = inverse_daily.std()
    inv_sharpe = (inv_mean / inv_std * np.sqrt(252)) if inv_std > 0 else 0
    results["inverse_direction"] = {
        "inverse_sharpe": round(float(inv_sharpe), 4),
        "pass": inv_sharpe <= 0,
        "note": "FAIL if inverse Sharpe > 0 (strategy is just long bias)"
    }

    # 2. Random timing: 500 random entry sets, same n_trades and avg hold
    avg_hold = int(np.mean([t["hold_days"] for t in trades]))
    avg_hold = max(avg_hold, 1)
    oot_closes = oot["Close"].values
    n_days = len(oot_closes)
    random_sharpes = []
    for _ in range(500):
        # Pick random entry indices
        possible = list(range(n_days - avg_hold - 1))
        if len(possible) < n_trades:
            break
        idxs = sorted(np.random.choice(possible, size=min(n_trades, len(possible)), replace=False))
        rand_rets = []
        for idx in idxs:
            exit_idx = min(idx + avg_hold, n_days - 1)
            r = (oot_closes[exit_idx] / oot_closes[idx]) - 1 - SLIPPAGE_RT
            rand_rets.append(r)
        rand_rets = np.array(rand_rets)
        if rand_rets.std() > 0:
            rs = rand_rets.mean() / rand_rets.std() * np.sqrt(len(rand_rets))
        else:
            rs = 0
        random_sharpes.append(rs)

    if random_sharpes:
        pctile = np.mean(np.array(random_sharpes) >= actual_sharpe) * 100
        # Strategy Sharpe must be > 95th percentile of random
        # i.e., less than 5% of random achieve this Sharpe
        results["random_timing"] = {
            "percentile_of_random_beating_strategy": round(float(pctile), 2),
            "pass": pctile < 5.0,
            "note": "Strategy must beat >95% of random timing"
        }
    else:
        results["random_timing"] = {"pass": False, "note": "Could not run"}

    # 3. Top-trade removal: remove best 3 trades
    sorted_rets = np.sort(returns)[::-1]
    trimmed = sorted_rets[3:]  # remove top 3
    if len(trimmed) > 1 and trimmed.std() > 0:
        trimmed_sharpe = trimmed.mean() / trimmed.std() * np.sqrt(len(trimmed))
    else:
        trimmed_sharpe = 0
    results["top_trade_removal"] = {
        "trimmed_sharpe": round(float(trimmed_sharpe), 4),
        "pass": trimmed_sharpe >= 0.3,
        "note": "FAIL if Sharpe < 0.3 after removing best 3 trades"
    }

    # 4. Sub-period stability: split into 3 equal periods
    n = len(returns)
    chunk = n // 3
    sub_sharpes = []
    for i in range(3):
        start_i = i * chunk
        end_i = (i + 1) * chunk if i < 2 else n
        sub = returns[start_i:end_i]
        if len(sub) > 1 and sub.std() > 0:
            ss = sub.mean() / sub.std() * np.sqrt(len(sub))
        else:
            ss = 0
        sub_sharpes.append(round(float(ss), 4))
    results["sub_period_stability"] = {
        "sub_sharpes": sub_sharpes,
        "pass": all(s > 0 for s in sub_sharpes),
        "note": "ALL sub-periods must have Sharpe > 0"
    }

    # 5. Parameter sensitivity: test nearby parameters
    # This is variant-specific. We'll do a generic approach:
    # re-run with slightly different thresholds and check Sharpe doesn't collapse
    # For simplicity, we check if the trade-level Sharpe is within 50% across
    # a narrow parameter band (handled per-variant outside this function)
    results["parameter_sensitivity"] = {
        "note": "Tested via variant-specific parameter sweeps below",
        "pass": None  # filled in by caller
    }

    return results


def param_sensitivity_test(df, variant_name, oot_start="2022-01-01"):
    """
    Test parameter sensitivity for each variant.
    Returns True if original is not a sharp peak (nearby params within 50% of Sharpe).
    """
    test_configs = {
        "A_RSI2_lt10": [
            # (rsi_thresh, exit_thresh, max_hold)
            (8, 70, 10), (10, 70, 10), (12, 70, 10),
            (10, 60, 10), (10, 80, 10),
            (10, 70, 8), (10, 70, 12),
        ],
        "B_RSI2_lt5_above200SMA": [
            (3, 60, 10), (5, 60, 10), (7, 60, 10),
            (5, 50, 10), (5, 70, 10),
        ],
        "C_3day_streak": [
            # streak length: 2, 3, 4
            (2,), (3,), (4,),
        ],
        "D_2day_drop_3pct": [
            # (drop_pct, hold)
            (0.02, 5), (0.03, 5), (0.04, 5),
            (0.03, 3), (0.03, 7),
        ],
        "E_IBS_lt02": [
            # (ibs_entry, ibs_exit, max_hold)
            (0.15, 0.8, 5), (0.2, 0.8, 5), (0.25, 0.8, 5),
            (0.2, 0.7, 5), (0.2, 0.9, 5),
        ],
        "F_combined_filter": [
            # (rsi_thresh, ret2_thresh, exit_rsi, max_hold)
            (10, -0.015, 50, 7), (15, -0.015, 50, 7), (20, -0.015, 50, 7),
            (15, -0.01, 50, 7), (15, -0.02, 50, 7),
            (15, -0.015, 40, 7), (15, -0.015, 60, 7),
        ],
    }

    configs = test_configs.get(variant_name, [])
    if not configs:
        return True, []

    r = rsi(df["Close"], 2)
    s200 = sma(df["Close"], 200)
    ib = ibs(df)
    ret2 = df["Close"].pct_change(2)
    down = df["Close"] < df["Close"].shift(1)

    sharpes = []
    for cfg in configs:
        if variant_name == "A_RSI2_lt10":
            thresh, exit_t, mh = cfg
            entries = r < thresh
            exits = r > exit_t
        elif variant_name == "B_RSI2_lt5_above200SMA":
            thresh, exit_t, mh = cfg
            entries = (r < thresh) & (df["Close"] > s200)
            exits = r > exit_t
        elif variant_name == "C_3day_streak":
            streak_len = cfg[0]
            mh = 5
            cond = down.copy()
            for s in range(1, streak_len):
                cond = cond & down.shift(s)
            entries = cond
            exits = df["Close"] > df["Close"].shift(1)
        elif variant_name == "D_2day_drop_3pct":
            drop, mh = cfg
            entries = ret2 < -drop
            exits = pd.Series(False, index=df.index)
        elif variant_name == "E_IBS_lt02":
            ib_entry, ib_exit, mh = cfg
            low_ib = ib < ib_entry
            entries = low_ib & low_ib.shift(1)
            exits = ib > ib_exit
        elif variant_name == "F_combined_filter":
            rsi_t, ret_t, exit_rsi, mh = cfg
            entries = (r < rsi_t) & (ret2 < ret_t) & (df["Close"] > s200)
            exits = r > exit_rsi
        else:
            continue

        trades = run_backtest(df, entries, exits, mh, oot_start)
        if len(trades) < 5:
            sharpes.append(0)
            continue
        rets = np.array([t["net_return"] for t in trades])
        if rets.std() > 0:
            sh = rets.mean() / rets.std() * np.sqrt(len(rets))
        else:
            sh = 0
        sharpes.append(round(float(sh), 4))

    # Check: original should not be a sharp peak
    # Find which config is the "original" (variant default)
    # The original is always in the list. Check if max Sharpe is within 2x of neighbors
    if not sharpes or max(sharpes) == 0:
        return True, sharpes

    max_sh = max(sharpes)
    median_sh = float(np.median(sharpes))
    # Pass if median of nearby params is at least 40% of max
    passed = median_sh >= 0.4 * max_sh or max_sh < 0.3
    return passed, sharpes


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("Downloading SPY data...")
    df = get_data()
    print(f"Data: {df.index[0].date()} to {df.index[-1].date()}, {len(df)} bars")

    oot_start = "2022-01-01"
    results = {}

    for name, func in VARIANTS.items():
        print(f"\n{'='*60}")
        print(f"Variant: {name}")
        print(f"{'='*60}")

        entries, exits, max_hold = func(df)
        trades = run_backtest(df, entries, exits, max_hold, oot_start)

        if len(trades) < 2:
            print(f"  Only {len(trades)} trades — SKIP")
            results[name] = {"status": "INSUFFICIENT_TRADES", "n_trades": len(trades)}
            continue

        m = calc_metrics(trades, df, oot_start)
        if m is None:
            results[name] = {"status": "CALC_FAILED"}
            continue

        gates, passed_all = five_gate_check(m)

        print(f"  Trades: {m['n_trades']}")
        print(f"  Sharpe: {m['sharpe']}")
        print(f"  Sortino: {m['sortino']}")
        print(f"  PF: {m['profit_factor']}")
        print(f"  WR: {m['win_rate']:.1%}")
        print(f"  MaxDD: {m['max_drawdown']:.2%}")
        print(f"  Total Return: {m['total_return']:.2%}")
        print(f"  Mean Return/Trade: {m['mean_return_per_trade']:.4%}")
        print(f"  Regime: green={m['n_green']}, red={m['n_red']}")
        print(f"  Sharpe green={m['sharpe_green']}, red={m['sharpe_red']}, gap={m['regime_gap']}")
        print(f"  Perm p-value: {m['perm_p']}")
        print(f"  5-Gate: {gates}")
        print(f"  PASS ALL 5 GATES: {passed_all}")

        variant_result = {
            "n_trades": m["n_trades"],
            "sharpe": m["sharpe"],
            "sortino": m["sortino"],
            "profit_factor": m["profit_factor"],
            "win_rate": m["win_rate"],
            "max_drawdown": m["max_drawdown"],
            "total_return": m["total_return"],
            "mean_return_per_trade": m["mean_return_per_trade"],
            "sharpe_green": m["sharpe_green"],
            "sharpe_red": m["sharpe_red"],
            "regime_gap": m["regime_gap"],
            "n_green": m["n_green"],
            "n_red": m["n_red"],
            "perm_p": m["perm_p"],
            "five_gates": gates,
            "passed_5_gates": passed_all,
        }

        # Adversarial validation if passed 5/5
        if passed_all:
            print(f"\n  >>> PASSED 5/5 — Running adversarial validation...")
            adv = adversarial_validation(trades, m, df, func, oot_start)

            # Parameter sensitivity
            ps_pass, ps_sharpes = param_sensitivity_test(df, name, oot_start)
            adv["parameter_sensitivity"]["pass"] = ps_pass
            adv["parameter_sensitivity"]["nearby_sharpes"] = ps_sharpes

            adv_pass_count = sum(1 for k, v in adv.items() if v.get("pass") is True)
            adv_total = len(adv)

            # Clean for JSON
            adv_clean = {}
            for k, v in adv.items():
                adv_clean[k] = {kk: vv for kk, vv in v.items()}

            variant_result["adversarial"] = adv_clean
            variant_result["adversarial_passed"] = f"{adv_pass_count}/{adv_total}"

            print(f"  Adversarial results:")
            for k, v in adv_clean.items():
                status = "PASS" if v.get("pass") else "FAIL" if v.get("pass") is False else "N/A"
                print(f"    {k}: {status} — {v}")
        else:
            variant_result["adversarial"] = "NOT_RUN (failed 5-gate)"

        results[name] = variant_result

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    any_passed = False
    for name, r in results.items():
        if isinstance(r.get("passed_5_gates"), bool):
            status = "PASS 5/5" if r["passed_5_gates"] else "FAIL"
            adv_status = r.get("adversarial_passed", "")
            if adv_status:
                adv_status = f" | Adversarial: {adv_status}"
            print(f"  {name}: {status} | Sharpe={r.get('sharpe','?')} | Trades={r.get('n_trades','?')}{adv_status}")
            if r["passed_5_gates"]:
                any_passed = True
        else:
            print(f"  {name}: {r.get('status', 'UNKNOWN')}")

    if not any_passed:
        print("\n  NO VARIANT PASSED ALL 5 GATES.")
        print("  Mean reversion on SPY 2022-2026 does not show robust, regime-agnostic edge.")

    # Save results
    out_path = Path("/home/jupiter/Lvl3Quant/data/short_term_mean_reversion_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")

    return results


if __name__ == "__main__":
    main()
