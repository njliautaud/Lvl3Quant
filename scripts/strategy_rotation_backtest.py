#!/usr/bin/env python3
"""
Strategy Rotation Backtest — META-STRATEGY
Instead of rotating assets, rotate between validated trading strategies
based on regime conditions.

Strategies (simulated via ETF proxies):
  1. Earnings Momentum (bull-dominant): QQQ long
  2. Contrarian Reversion (bear-tolerant): Buy SPY on weekly dips
  3. VIX Fade (crisis recovery): Buy SPY when VIX spikes then declines
  4. Cash: 0% return

Variants A–F with different rotation logic.
"""

import json, sys, warnings, datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")
np.random.seed(42)

# ── CONFIG ──────────────────────────────────────────────────────────────
OOT_START = "2022-01-01"
OOT_END   = "2026-07-25"
STARTING_CAPITAL = 645.0
PERM_ITERS = 1000
RESULTS_PATH = Path("/home/jupiter/Lvl3Quant/data/strategy_rotation_results.json")

# 5-gate thresholds
GATE_SHARPE   = 0.5
GATE_PERM_P   = 0.05
GATE_REGIME   = 0.50
GATE_MDD      = -0.50
GATE_TRADES   = 20


# ── HELPERS ─────────────────────────────────────────────────────────────
def fetch_data():
    """Download SPY, QQQ, ^VIX with progress."""
    print("[1/6] Downloading market data via yfinance …")
    tickers = ["SPY", "QQQ", "^VIX"]
    # Fetch with enough history for 200-SMA lookback
    raw = yf.download(tickers, start="2021-01-01", end=OOT_END,
                      auto_adjust=True, progress=False)

    # Handle multi-level columns from yfinance
    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Close"]
    else:
        close = raw[["Close"]].copy()
        close.columns = tickers

    # Flatten any remaining multi-level
    if isinstance(close.columns, pd.MultiIndex):
        close.columns = close.columns.get_level_values(-1)

    df = pd.DataFrame(index=close.index)
    df["SPY"] = close["SPY"]
    df["QQQ"] = close["QQQ"]
    df["VIX"] = close["^VIX"] if "^VIX" in close.columns else close.get("^GSPC", np.nan)
    df = df.dropna()
    print(f"    Got {len(df)} trading days ({df.index[0].date()} → {df.index[-1].date()})")
    return df


def compute_regime_signals(df):
    """Add regime indicator columns."""
    print("[2/6] Computing regime signals …")
    df = df.copy()
    df["SMA200"]  = df["SPY"].rolling(200).mean()
    df["RSI14"]   = _rsi(df["SPY"], 14)
    df["SPY_ret20"] = df["SPY"].pct_change(20)
    df["SPY_ret60"] = df["SPY"].pct_change(60)
    df["SPY_ret5"]  = df["SPY"].pct_change(5)
    df["VIX_chg5"]  = df["VIX"].pct_change(5)
    df["bull"] = (df["SPY"] > df["SMA200"]).astype(int)
    df = df.dropna()
    return df


def _rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss
    return 100 - 100 / (1 + rs)


# ── STRATEGY SIMULATORS ────────────────────────────────────────────────
# Each returns daily pnl series for dates where strategy is "on".
# Max 1 position at a time, $0 commissions, invest full capital.

def _daily_returns(prices):
    return prices.pct_change().fillna(0)


def sim_earnings_momentum(df, active_mask):
    """QQQ long on active days."""
    qqq_ret = _daily_returns(df["QQQ"])
    return qqq_ret * active_mask


def sim_contrarian(df, active_mask):
    """Buy SPY on active days. Enter only when SPY had ≥-3% weekly drop
    (SPY_ret5 <= -0.03). Hold 5 days then exit. If no trigger, stay cash."""
    spy_ret = _daily_returns(df["SPY"])
    positions = pd.Series(0.0, index=df.index)
    hold_remaining = 0
    for i in range(len(df)):
        if hold_remaining > 0:
            positions.iloc[i] = 1.0
            hold_remaining -= 1
        elif active_mask.iloc[i] and df["SPY_ret5"].iloc[i] <= -0.03:
            positions.iloc[i] = 1.0
            hold_remaining = 4  # today + 4 more = 5 total
    return spy_ret * positions


def sim_vix_fade(df, active_mask):
    """Buy SPY when VIX > 25 AND VIX declining (5d change < 0) on active days."""
    spy_ret = _daily_returns(df["SPY"])
    trigger = ((df["VIX"] > 25) & (df["VIX_chg5"] < 0)).astype(float)
    return spy_ret * trigger * active_mask


def sim_cash(df, active_mask):
    """Returns 0."""
    return pd.Series(0.0, index=df.index)


# ── VARIANT ROTATION LOGIC ─────────────────────────────────────────────

def variant_A(df):
    """Simple Regime Switch: bull→QQQ, bear→Contrarian, VIX>25→VIX Fade."""
    print("    Variant A (Simple Regime Switch)")
    n = len(df)
    strat_labels = []
    for i in range(n):
        if df["VIX"].iloc[i] > 25:
            strat_labels.append("vix_fade")
        elif df["bull"].iloc[i]:
            strat_labels.append("earnings_momentum")
        else:
            strat_labels.append("contrarian")
    return pd.Series(strat_labels, index=df.index)


def variant_B(df):
    """Momentum + Regime: 60d ret > 0 → QQQ, else SPY dips. VIX gate."""
    print("    Variant B (Momentum + Regime)")
    strat_labels = []
    for i in range(len(df)):
        if df["VIX"].iloc[i] > 25:
            strat_labels.append("vix_fade")
        elif df["SPY_ret60"].iloc[i] > 0:
            strat_labels.append("earnings_momentum")
        else:
            strat_labels.append("contrarian")
    return pd.Series(strat_labels, index=df.index)


def variant_C(df):
    """RSI Rotation: RSI>70 → cash, 30-70 → QQQ, <30 → buy SPY aggressively."""
    print("    Variant C (RSI Rotation)")
    strat_labels = []
    for i in range(len(df)):
        rsi = df["RSI14"].iloc[i]
        if rsi > 70:
            strat_labels.append("cash")
        elif rsi < 30:
            strat_labels.append("contrarian")
        else:
            strat_labels.append("earnings_momentum")
    return pd.Series(strat_labels, index=df.index)


def variant_D(df):
    """Composite Score 0-100. >70→QQQ, 30-70→SPY hold, <30→Contrarian."""
    print("    Variant D (Composite Score)")
    strat_labels = []
    for i in range(len(df)):
        score = 50.0
        # Bull/bear: +20 if bull
        score += 20 if df["bull"].iloc[i] else -20
        # VIX: low VIX good
        vix = df["VIX"].iloc[i]
        if vix < 15:
            score += 15
        elif vix > 25:
            score -= 15
        # Momentum
        ret20 = df["SPY_ret20"].iloc[i]
        score += np.clip(ret20 * 200, -15, 15)
        score = np.clip(score, 0, 100)

        if score > 70:
            strat_labels.append("earnings_momentum")
        elif score < 30:
            strat_labels.append("contrarian")
        else:
            strat_labels.append("spy_hold")
    return pd.Series(strat_labels, index=df.index)


def variant_E(df):
    """Adaptive Weights: all strategies run, weight by trailing 60d Sharpe.
    Monthly rebalance. Best one gets 100% (simplified from continuous weights
    since max 1 position)."""
    print("    Variant E (Adaptive Weights)")
    # Pre-compute each strategy's daily returns (always on)
    ones = pd.Series(1.0, index=df.index)
    strat_rets = {
        "earnings_momentum": sim_earnings_momentum(df, ones),
        "contrarian": sim_contrarian(df, ones),
        "vix_fade": sim_vix_fade(df, ones),
    }
    strat_labels = []
    current_best = "earnings_momentum"
    for i in range(len(df)):
        # Rebalance on first trading day of each month
        if i == 0 or df.index[i].month != df.index[i-1].month:
            lookback = max(0, i - 60)
            best_sharpe = -999
            for name, rets in strat_rets.items():
                window = rets.iloc[lookback:i]
                if len(window) > 10:
                    sr = window.mean() / (window.std() + 1e-9) * np.sqrt(252)
                else:
                    sr = 0
                if sr > best_sharpe:
                    best_sharpe = sr
                    current_best = name
        strat_labels.append(current_best)
    return pd.Series(strat_labels, index=df.index)


def variant_F(df):
    """Adversarial Random Rotation: randomly pick strategy each week."""
    print("    Variant F (Adversarial Random)")
    rng = np.random.RandomState(123)
    choices = ["earnings_momentum", "contrarian", "vix_fade", "cash"]
    strat_labels = []
    current = rng.choice(choices)
    for i in range(len(df)):
        # New random pick each Monday (weekday 0)
        if df.index[i].weekday() == 0:
            current = rng.choice(choices)
        strat_labels.append(current)
    return pd.Series(strat_labels, index=df.index)


# ── EXECUTE VARIANT ─────────────────────────────────────────────────────

def execute_rotation(df, rotation_labels):
    """Given strategy labels per day, produce equity curve."""
    spy_ret = _daily_returns(df["SPY"])
    qqq_ret = _daily_returns(df["QQQ"])

    daily_ret = pd.Series(0.0, index=df.index)
    n_trades = 0
    prev_label = None

    # For contrarian: track hold_remaining
    hold_remaining = 0

    for i in range(len(df)):
        label = rotation_labels.iloc[i]
        if label != prev_label:
            n_trades += 1
            hold_remaining = 0
        prev_label = label

        if label == "earnings_momentum":
            daily_ret.iloc[i] = qqq_ret.iloc[i]
        elif label == "contrarian":
            # Buy SPY on dips (<= -3% weekly), hold 5d
            if hold_remaining > 0:
                daily_ret.iloc[i] = spy_ret.iloc[i]
                hold_remaining -= 1
            elif df["SPY_ret5"].iloc[i] <= -0.03:
                daily_ret.iloc[i] = spy_ret.iloc[i]
                hold_remaining = 4
            # else: cash (0)
        elif label == "vix_fade":
            if df["VIX"].iloc[i] > 25 and df["VIX_chg5"].iloc[i] < 0:
                daily_ret.iloc[i] = spy_ret.iloc[i]
        elif label == "spy_hold":
            daily_ret.iloc[i] = spy_ret.iloc[i]
        elif label == "cash":
            pass  # 0

    equity = STARTING_CAPITAL * (1 + daily_ret).cumprod()
    return daily_ret, equity, n_trades


# ── METRICS ─────────────────────────────────────────────────────────────

def compute_metrics(daily_ret, equity, n_trades, df, rotation_labels):
    trading_days = daily_ret[daily_ret != 0]

    total_ret = (equity.iloc[-1] / equity.iloc[0]) - 1
    ann_ret = (1 + total_ret) ** (252 / len(daily_ret)) - 1
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

    # Max drawdown
    peak = equity.cummax()
    dd = (equity - peak) / peak
    mdd = dd.min()

    # Regime-stratified Sharpe
    bull_mask = df["bull"] == 1
    bear_mask = df["bull"] == 0

    def _sharpe_subset(rets):
        if len(rets) < 5 or rets.std() == 0:
            return 0.0
        return (rets.mean() / rets.std()) * np.sqrt(252)

    sharpe_bull = _sharpe_subset(daily_ret[bull_mask])
    sharpe_bear = _sharpe_subset(daily_ret[bear_mask])
    regime_gap = abs(sharpe_bull - sharpe_bear) / max(abs(sharpe_bull), abs(sharpe_bear), 0.01)

    return {
        "total_return": round(total_ret, 4),
        "annual_return": round(ann_ret, 4),
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "profit_factor": round(pf, 3),
        "win_rate": round(wr, 3),
        "max_drawdown": round(mdd, 4),
        "n_trades": n_trades,
        "n_active_days": int((trading_days != 0).sum()) if len(trading_days) > 0 else 0,
        "sharpe_bull": round(sharpe_bull, 3),
        "sharpe_bear": round(sharpe_bear, 3),
        "regime_gap": round(regime_gap, 3),
        "final_equity": round(equity.iloc[-1], 2),
    }


# ── PERMUTATION TEST ───────────────────────────────────────────────────

def _fast_execute_sharpe(spy_ret_arr, qqq_ret_arr, vix_fade_mask_arr,
                         contrarian_mask_arr, label_arr, label_map, n_days):
    """Vectorized Sharpe computation for permutation test.
    Simplified: earnings_momentum→QQQ ret, contrarian→SPY ret if dip triggered,
    vix_fade→SPY ret if VIX trigger, spy_hold→SPY ret, cash→0."""
    daily_ret = np.zeros(n_days)
    for i in range(n_days):
        lab = label_arr[i]
        if lab == label_map["earnings_momentum"]:
            daily_ret[i] = qqq_ret_arr[i]
        elif lab == label_map["contrarian"]:
            if contrarian_mask_arr[i]:
                daily_ret[i] = spy_ret_arr[i]
        elif lab == label_map["vix_fade"]:
            if vix_fade_mask_arr[i]:
                daily_ret[i] = spy_ret_arr[i]
        elif lab == label_map["spy_hold"]:
            daily_ret[i] = spy_ret_arr[i]
        # cash → 0
    mu = daily_ret.mean()
    std = daily_ret.std()
    if std < 1e-12:
        return 0.0
    return (mu / std) * np.sqrt(252)


def permutation_test(df, rotation_labels, observed_sharpe, n_iter=PERM_ITERS):
    """Shuffle rotation decisions (block-shuffle by week), recompute Sharpe.
    Uses numpy arrays for speed."""
    spy_ret_arr = _daily_returns(df["SPY"]).values
    qqq_ret_arr = _daily_returns(df["QQQ"]).values
    n_days = len(df)

    # Pre-compute trigger masks
    vix_fade_mask = ((df["VIX"] > 25) & (df["VIX_chg5"] < 0)).values
    contrarian_mask = (df["SPY_ret5"] <= -0.03).values

    # Map labels to ints for speed
    all_labels = list(set(rotation_labels.unique()) | {"earnings_momentum", "contrarian",
                                                        "vix_fade", "cash", "spy_hold"})
    label_to_int = {l: i for i, l in enumerate(all_labels)}
    label_map = {k: label_to_int[k] for k in
                 ["earnings_momentum", "contrarian", "vix_fade", "cash", "spy_hold"]}

    orig_label_arr = np.array([label_to_int[l] for l in rotation_labels])
    unique_ints = np.array([label_to_int[l] for l in rotation_labels.unique()])

    # Build week boundaries
    dates = df.index
    week_starts = [0]
    for i in range(1, n_days):
        if dates[i].isocalendar()[1] != dates[i-1].isocalendar()[1] or \
           dates[i].year != dates[i-1].year:
            week_starts.append(i)
    week_starts.append(n_days)
    n_weeks = len(week_starts) - 1

    rng = np.random.RandomState(7)
    count_ge = 0

    for _ in range(n_iter):
        # Block-shuffle: assign random label per week
        shuffled = orig_label_arr.copy()
        rand_labels = rng.choice(unique_ints, size=n_weeks)
        for w in range(n_weeks):
            shuffled[week_starts[w]:week_starts[w+1]] = rand_labels[w]

        perm_sharpe = _fast_execute_sharpe(spy_ret_arr, qqq_ret_arr,
                                           vix_fade_mask, contrarian_mask,
                                           shuffled, label_map, n_days)
        if perm_sharpe >= observed_sharpe:
            count_ge += 1

    return round(count_ge / n_iter, 4)


# ── 5-GATE CHECK ───────────────────────────────────────────────────────

def five_gate_check(m, perm_p):
    gates = {
        "sharpe_gt_0.5": m["sharpe"] > GATE_SHARPE,
        "perm_p_lt_0.05": perm_p < GATE_PERM_P,
        "regime_gap_lt_0.5": m["regime_gap"] < GATE_REGIME,
        "mdd_gt_neg50pct": m["max_drawdown"] > GATE_MDD,
        "trades_gte_20": m["n_trades"] >= GATE_TRADES,
    }
    return gates, all(gates.values())


# ── BENCHMARKS ──────────────────────────────────────────────────────────

def compute_benchmarks(df_oot):
    """SPY and QQQ buy-and-hold benchmarks."""
    benchmarks = {}
    for ticker in ["SPY", "QQQ"]:
        rets = _daily_returns(df_oot[ticker])
        eq = STARTING_CAPITAL * (1 + rets).cumprod()
        total = eq.iloc[-1] / eq.iloc[0] - 1
        ann = (1 + total) ** (252 / len(rets)) - 1
        vol = rets.std() * np.sqrt(252)
        sharpe = ann / vol if vol > 0 else 0
        peak = eq.cummax()
        mdd = ((eq - peak) / peak).min()
        benchmarks[f"{ticker}_buyhold"] = {
            "sharpe": round(sharpe, 3),
            "total_return": round(total, 4),
            "max_drawdown": round(mdd, 4),
            "final_equity": round(eq.iloc[-1], 2),
        }
    return benchmarks


# ── MAIN ────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("STRATEGY ROTATION BACKTEST — META-STRATEGY")
    print(f"OOT: {OOT_START} → {OOT_END}  |  Capital: ${STARTING_CAPITAL}")
    print("=" * 70)

    df = fetch_data()
    df = compute_regime_signals(df)

    # Trim to OOT period
    df_oot = df.loc[OOT_START:OOT_END].copy()
    print(f"    OOT period: {len(df_oot)} days ({df_oot.index[0].date()} → {df_oot.index[-1].date()})")

    # Benchmarks
    print("[3/6] Computing benchmarks …")
    benchmarks = compute_benchmarks(df_oot)
    for name, bm in benchmarks.items():
        print(f"    {name}: Sharpe={bm['sharpe']}, Return={bm['total_return']:.1%}, "
              f"MDD={bm['max_drawdown']:.1%}, Final=${bm['final_equity']}")

    # Run all variants
    print("[4/6] Running strategy rotation variants …")
    variant_funcs = {
        "A_simple_regime": variant_A,
        "B_momentum_regime": variant_B,
        "C_rsi_rotation": variant_C,
        "D_composite_score": variant_D,
        "E_adaptive_weights": variant_E,
        "F_random_adversarial": variant_F,
    }

    results = {"benchmarks": benchmarks, "variants": {}, "meta": {
        "oot_start": OOT_START, "oot_end": OOT_END,
        "starting_capital": STARTING_CAPITAL,
        "run_timestamp": dt.datetime.now().isoformat(),
    }}

    for vname, vfunc in variant_funcs.items():
        rotation_labels = vfunc(df_oot)
        daily_ret, equity, n_trades = execute_rotation(df_oot, rotation_labels)
        metrics = compute_metrics(daily_ret, equity, n_trades, df_oot, rotation_labels)

        # Strategy allocation breakdown
        alloc = rotation_labels.value_counts(normalize=True).to_dict()
        metrics["allocation_pct"] = {k: round(v, 3) for k, v in alloc.items()}

        print(f"    {vname}: Sharpe={metrics['sharpe']}, Sortino={metrics['sortino']}, "
              f"PF={metrics['profit_factor']}, WR={metrics['win_rate']:.1%}, "
              f"MDD={metrics['max_drawdown']:.1%}, trades={metrics['n_trades']}, "
              f"final=${metrics['final_equity']}")
        print(f"      Bull Sharpe={metrics['sharpe_bull']}, Bear Sharpe={metrics['sharpe_bear']}, "
              f"Regime Gap={metrics['regime_gap']}")

        results["variants"][vname] = metrics

    # Permutation tests
    print(f"[5/6] Permutation tests ({PERM_ITERS} iterations each) …")
    for vname, vfunc in variant_funcs.items():
        rotation_labels = vfunc(df_oot)
        daily_ret, equity, n_trades = execute_rotation(df_oot, rotation_labels)
        observed_sharpe = results["variants"][vname]["sharpe"]
        perm_p = permutation_test(df_oot, rotation_labels, observed_sharpe)
        results["variants"][vname]["perm_p"] = perm_p

        gates, passed = five_gate_check(results["variants"][vname], perm_p)
        results["variants"][vname]["five_gate"] = gates
        results["variants"][vname]["five_gate_pass"] = passed

        status = "PASS ✓" if passed else "FAIL ✗"
        print(f"    {vname}: perm_p={perm_p}, 5-gate={status}")
        failed = [k for k, v in gates.items() if not v]
        if failed:
            print(f"      Failed: {', '.join(failed)}")

    # Summary
    print("\n[6/6] Summary")
    print("=" * 70)
    passers = [v for v, d in results["variants"].items() if d["five_gate_pass"]]
    if passers:
        print(f"PASSED 5-gate: {', '.join(passers)}")
        best = max(passers, key=lambda v: results["variants"][v]["sharpe"])
        bm = results["variants"][best]
        print(f"Best variant: {best}")
        print(f"  Sharpe={bm['sharpe']}, Sortino={bm['sortino']}, PF={bm['profit_factor']}, "
              f"WR={bm['win_rate']:.1%}, MDD={bm['max_drawdown']:.1%}")
    else:
        print("NO VARIANT passed all 5 gates.")
        best_sharpe = max(results["variants"].items(), key=lambda x: x[1]["sharpe"])
        print(f"Best by Sharpe: {best_sharpe[0]} (Sharpe={best_sharpe[1]['sharpe']})")

    # Compare to benchmarks
    print("\nBenchmark comparison:")
    for bname, bm in benchmarks.items():
        print(f"  {bname}: Sharpe={bm['sharpe']}, Return={bm['total_return']:.1%}")

    # Save
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
