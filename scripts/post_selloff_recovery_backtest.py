#!/usr/bin/env python3
"""
Post-Selloff Recovery Timing Backtest
=====================================
Tests 6 variants of buying growth stocks after significant index-level selloffs.
Walk-forward OOT: Jan 2022 - Jul 2026.
5-gate validation: Sharpe>0.5, perm p<0.05, regime gap<0.5, MaxDD>-50%, >=20 trades.
"""

import json
import warnings
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

# ── CONFIG ──────────────────────────────────────────────────────────────────
CAPITAL = 645.0
SLIPPAGE_PCT = 0.0002  # 0.02%
OOT_START = "2022-01-01"
OOT_END = "2026-07-30"
DATA_START = "2021-01-01"  # extra lookback for SMA etc.
PERM_ITERS = 1000
RF_ANNUAL = 0.04  # risk-free for Sharpe/Sortino

GROWTH = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "AMD", "NFLX", "CRM", "PLTR", "SOFI", "UBER", "COIN"]
SAFE = ["GLD", "TLT", "SHY"]
INDEX = ["SPY", "QQQ"]
ALL_TICKERS = list(set(GROWTH + SAFE + INDEX + ["^VIX"]))

OUTPUT_PATH = Path("/home/jupiter/Lvl3Quant/data/post_selloff_recovery_results.json")


# ── DATA ────────────────────────────────────────────────────────────────────
def fetch_data():
    print("Fetching price data...")
    data = {}
    for t in ALL_TICKERS:
        try:
            df = yf.download(t, start=DATA_START, end=OOT_END, progress=False, auto_adjust=True)
            if len(df) > 100:
                data[t] = df["Close"].squeeze()
                print(f"  {t}: {len(df)} bars")
            else:
                print(f"  {t}: insufficient data ({len(df)} bars), skipping")
        except Exception as e:
            print(f"  {t}: error {e}")
    return data


# ── HELPERS ─────────────────────────────────────────────────────────────────
def get_regime(spy_close, date):
    """Bull if SPY > 200-SMA, else Bear."""
    loc = spy_close.index.get_indexer([date], method="ffill")[0]
    if loc < 200:
        return "bull"
    sma200 = spy_close.iloc[max(0, loc - 199):loc + 1].mean()
    return "bull" if spy_close.iloc[loc] > sma200 else "bear"


def apply_slippage(price, direction="buy"):
    if direction == "buy":
        return price * (1 + SLIPPAGE_PCT)
    return price * (1 - SLIPPAGE_PCT)


def compute_metrics(trades, spy_close):
    """Compute all metrics from a list of trade dicts."""
    if not trades:
        return empty_result()

    pnls = [t["pnl_pct"] for t in trades]
    equities = [CAPITAL]
    for p in pnls:
        equities.append(equities[-1] * (1 + p))
    equities = np.array(equities)
    final_eq = equities[-1]
    total_ret = (final_eq / CAPITAL - 1) * 100

    # Drawdown
    peak = np.maximum.accumulate(equities)
    dd = (equities - peak) / peak
    max_dd = dd.min() * 100

    # Sharpe & Sortino (annualized, assume ~20 trades/yr avg)
    pnl_arr = np.array(pnls)
    avg_hold = np.mean([t["hold_days"] for t in trades])
    trades_per_year = 252 / max(avg_hold, 1)
    excess = pnl_arr - RF_ANNUAL / trades_per_year
    sharpe = (excess.mean() / excess.std() * np.sqrt(trades_per_year)) if excess.std() > 0 else 0.0

    downside = excess[excess < 0]
    down_std = np.sqrt(np.mean(downside ** 2)) if len(downside) > 0 else 1e-9
    sortino = excess.mean() / down_std * np.sqrt(trades_per_year)

    # Profit factor
    gross_profit = pnl_arr[pnl_arr > 0].sum() if (pnl_arr > 0).any() else 0
    gross_loss = abs(pnl_arr[pnl_arr < 0].sum()) if (pnl_arr < 0).any() else 1e-9
    pf = gross_profit / gross_loss

    wr = (pnl_arr > 0).mean() * 100

    # Regime split
    bull_trades = [t for t in trades if t["regime"] == "bull"]
    bear_trades = [t for t in trades if t["regime"] == "bear"]

    def regime_sharpe(rtrades):
        if len(rtrades) < 2:
            return 0.0
        rp = np.array([t["pnl_pct"] for t in rtrades])
        ravg_hold = np.mean([t["hold_days"] for t in rtrades])
        rtpy = 252 / max(ravg_hold, 1)
        rexcess = rp - RF_ANNUAL / rtpy
        return (rexcess.mean() / rexcess.std() * np.sqrt(rtpy)) if rexcess.std() > 0 else 0.0

    bull_sh = regime_sharpe(bull_trades)
    bear_sh = regime_sharpe(bear_trades)
    max_sh = max(abs(bull_sh), abs(bear_sh), 1e-9)
    regime_gap = abs(bull_sh - bear_sh) / max_sh

    metrics = {
        "total_return_pct": round(total_ret, 2),
        "final_equity": round(final_eq, 2),
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "profit_factor": round(pf, 3),
        "win_rate": round(wr, 1),
        "num_trades": len(trades),
        "max_drawdown_pct": round(max_dd, 2),
        "avg_hold_days": round(avg_hold, 1),
    }
    regime = {
        "bull_trades": len(bull_trades),
        "bear_trades": len(bear_trades),
        "bull_sharpe": round(bull_sh, 3),
        "bear_sharpe": round(bear_sh, 3),
        "regime_gap": round(regime_gap, 3),
    }
    return metrics, regime, pnl_arr


def empty_result():
    metrics = {
        "total_return_pct": 0, "final_equity": CAPITAL, "sharpe": 0, "sortino": 0,
        "profit_factor": 0, "win_rate": 0, "num_trades": 0, "max_drawdown_pct": 0, "avg_hold_days": 0,
    }
    regime = {"bull_trades": 0, "bear_trades": 0, "bull_sharpe": 0, "bear_sharpe": 0, "regime_gap": 0}
    return metrics, regime, np.array([])


def permutation_test(trades, spy_close, n_iter=PERM_ITERS):
    """Shuffle signal dates by random offset 1-30 days."""
    if len(trades) < 5:
        return {"perm_p_value": 1.0, "actual_mean_pnl": 0.0, "perm_mean_pnl": 0.0}

    actual_mean = np.mean([t["pnl_pct"] for t in trades])
    rng = np.random.default_rng(42)
    count_ge = 0
    perm_means = []

    for _ in range(n_iter):
        shuffled_pnls = []
        for t in trades:
            offset = rng.integers(1, 31)
            # Shift entry date by random offset
            new_entry = t["entry_date"] + timedelta(days=int(offset))
            ticker = t["ticker"]
            hold = t["hold_days"]
            if ticker in data_cache and new_entry in data_cache[ticker].index:
                idx = data_cache[ticker].index.get_loc(new_entry)
                exit_idx = min(idx + hold, len(data_cache[ticker]) - 1)
                entry_p = apply_slippage(data_cache[ticker].iloc[idx], "buy")
                exit_p = apply_slippage(data_cache[ticker].iloc[exit_idx], "sell")
                shuffled_pnls.append((exit_p - entry_p) / entry_p)
            else:
                shuffled_pnls.append(0.0)

        pm = np.mean(shuffled_pnls) if shuffled_pnls else 0.0
        perm_means.append(pm)
        if pm >= actual_mean:
            count_ge += 1

    return {
        "perm_p_value": round(count_ge / n_iter, 4),
        "actual_mean_pnl": round(actual_mean * 100, 4),
        "perm_mean_pnl": round(np.mean(perm_means) * 100, 4),
    }


def check_gates(metrics, regime, perm):
    g = {
        "sharpe_gt_0.5": metrics["sharpe"] > 0.5,
        "perm_p_lt_0.05": perm["perm_p_value"] < 0.05,
        "regime_gap_lt_0.5": regime["regime_gap"] < 0.5,
        "max_dd_gt_neg50": metrics["max_drawdown_pct"] > -50,
        "num_trades_gte_20": metrics["num_trades"] >= 20,
    }
    g["all_passed"] = all(g.values())
    return g


# ── VARIANT STRATEGIES ──────────────────────────────────────────────────────
def variant_a_simple_dip_buy(data, spy, oot_dates):
    """SPY drops >=3% over 5 days -> buy QQQ, hold 10 days."""
    trades = []
    qqq = data.get("QQQ")
    if qqq is None:
        return trades

    i = 0
    while i < len(oot_dates):
        dt = oot_dates[i]
        loc = spy.index.get_indexer([dt], method="ffill")[0]
        if loc < 5:
            i += 1
            continue
        ret_5d = (spy.iloc[loc] - spy.iloc[loc - 5]) / spy.iloc[loc - 5]
        if ret_5d <= -0.03:
            if dt in qqq.index:
                entry_idx = qqq.index.get_loc(dt)
                exit_idx = min(entry_idx + 10, len(qqq) - 1)
                entry_p = apply_slippage(qqq.iloc[entry_idx], "buy")
                exit_p = apply_slippage(qqq.iloc[exit_idx], "sell")
                pnl = (exit_p - entry_p) / entry_p
                trades.append({
                    "entry_date": dt, "exit_date": qqq.index[exit_idx],
                    "ticker": "QQQ", "entry_price": entry_p, "exit_price": exit_p,
                    "pnl_pct": pnl, "hold_days": 10, "regime": get_regime(spy, dt),
                })
                i += 11  # skip hold period
                continue
        i += 1
    return trades


def variant_b_resilient_picker(data, spy, oot_dates):
    """SPY drops >=3% over 5 days -> buy 3 least-dropped growth stocks, hold 10 days."""
    trades = []
    i = 0
    while i < len(oot_dates):
        dt = oot_dates[i]
        loc = spy.index.get_indexer([dt], method="ffill")[0]
        if loc < 5:
            i += 1
            continue
        ret_5d = (spy.iloc[loc] - spy.iloc[loc - 5]) / spy.iloc[loc - 5]
        if ret_5d <= -0.03:
            # Rank growth stocks by 5-day return (least negative = most resilient)
            stock_rets = {}
            for ticker in GROWTH:
                if ticker in data and dt in data[ticker].index:
                    t_loc = data[ticker].index.get_loc(dt)
                    if t_loc >= 5:
                        t_ret = (data[ticker].iloc[t_loc] - data[ticker].iloc[t_loc - 5]) / data[ticker].iloc[t_loc - 5]
                        stock_rets[ticker] = t_ret

            if len(stock_rets) >= 3:
                # Top 3 most resilient (least negative return)
                top3 = sorted(stock_rets.items(), key=lambda x: x[1], reverse=True)[:3]
                for ticker, _ in top3:
                    ts = data[ticker]
                    entry_idx = ts.index.get_loc(dt)
                    exit_idx = min(entry_idx + 10, len(ts) - 1)
                    entry_p = apply_slippage(ts.iloc[entry_idx], "buy")
                    exit_p = apply_slippage(ts.iloc[exit_idx], "sell")
                    pnl = (exit_p - entry_p) / entry_p
                    trades.append({
                        "entry_date": dt, "exit_date": ts.index[exit_idx],
                        "ticker": ticker, "entry_price": entry_p, "exit_price": exit_p,
                        "pnl_pct": pnl, "hold_days": 10, "regime": get_regime(spy, dt),
                    })
                i += 11
                continue
        i += 1
    return trades


def variant_c_deep_dip(data, spy, oot_dates):
    """SPY drops >=5% over 10 days -> buy QQQ, hold 20 days."""
    trades = []
    qqq = data.get("QQQ")
    if qqq is None:
        return trades

    i = 0
    while i < len(oot_dates):
        dt = oot_dates[i]
        loc = spy.index.get_indexer([dt], method="ffill")[0]
        if loc < 10:
            i += 1
            continue
        ret_10d = (spy.iloc[loc] - spy.iloc[loc - 10]) / spy.iloc[loc - 10]
        if ret_10d <= -0.05:
            if dt in qqq.index:
                entry_idx = qqq.index.get_loc(dt)
                exit_idx = min(entry_idx + 20, len(qqq) - 1)
                entry_p = apply_slippage(qqq.iloc[entry_idx], "buy")
                exit_p = apply_slippage(qqq.iloc[exit_idx], "sell")
                pnl = (exit_p - entry_p) / entry_p
                trades.append({
                    "entry_date": dt, "exit_date": qqq.index[exit_idx],
                    "ticker": "QQQ", "entry_price": entry_p, "exit_price": exit_p,
                    "pnl_pct": pnl, "hold_days": 20, "regime": get_regime(spy, dt),
                })
                i += 21
                continue
        i += 1
    return trades


def variant_d_two_stage(data, spy, oot_dates):
    """Stage 1: SPY -3% in 5d -> buy 50% QQQ. Stage 2: another -2% in 5d -> buy other 50%. Hold 15d from last entry."""
    trades = []
    qqq = data.get("QQQ")
    if qqq is None:
        return trades

    i = 0
    while i < len(oot_dates):
        dt = oot_dates[i]
        loc = spy.index.get_indexer([dt], method="ffill")[0]
        if loc < 5:
            i += 1
            continue
        ret_5d = (spy.iloc[loc] - spy.iloc[loc - 5]) / spy.iloc[loc - 5]
        if ret_5d <= -0.03 and dt in qqq.index:
            entry1_idx = qqq.index.get_loc(dt)
            entry1_p = apply_slippage(qqq.iloc[entry1_idx], "buy")

            # Check for stage 2 within next 5 trading days
            stage2 = False
            entry2_p = None
            last_entry_idx = entry1_idx
            for j in range(1, 6):
                if entry1_idx + j >= len(qqq):
                    break
                check_dt = qqq.index[entry1_idx + j]
                check_loc = spy.index.get_indexer([check_dt], method="ffill")[0]
                if check_loc >= 5:
                    ret2 = (spy.iloc[check_loc] - spy.iloc[check_loc - 5]) / spy.iloc[check_loc - 5]
                    if ret2 <= -0.05:  # cumulative: original -3% + another -2% = -5% threshold
                        stage2 = True
                        last_entry_idx = entry1_idx + j
                        entry2_p = apply_slippage(qqq.iloc[last_entry_idx], "buy")
                        break

            exit_idx = min(last_entry_idx + 15, len(qqq) - 1)
            exit_p = apply_slippage(qqq.iloc[exit_idx], "sell")

            if stage2 and entry2_p is not None:
                avg_entry = (entry1_p + entry2_p) / 2
                pnl = (exit_p - avg_entry) / avg_entry
                hold = exit_idx - entry1_idx
            else:
                pnl = (exit_p - entry1_p) / entry1_p
                hold = exit_idx - entry1_idx

            trades.append({
                "entry_date": dt, "exit_date": qqq.index[exit_idx],
                "ticker": "QQQ", "entry_price": entry1_p, "exit_price": exit_p,
                "pnl_pct": pnl, "hold_days": hold, "regime": get_regime(spy, dt),
                "staged": stage2,
            })
            i += hold + 1
            continue
        i += 1
    return trades


def variant_e_vix_confirm(data, spy, oot_dates):
    """SPY -3% in 5d AND VIX>25 -> buy QQQ when VIX first declines. Hold 10d."""
    trades = []
    qqq = data.get("QQQ")
    vix = data.get("^VIX")
    if qqq is None or vix is None:
        return trades

    i = 0
    while i < len(oot_dates):
        dt = oot_dates[i]
        loc = spy.index.get_indexer([dt], method="ffill")[0]
        if loc < 5:
            i += 1
            continue
        ret_5d = (spy.iloc[loc] - spy.iloc[loc - 5]) / spy.iloc[loc - 5]

        # Check VIX level
        vix_loc = vix.index.get_indexer([dt], method="ffill")[0]
        if ret_5d <= -0.03 and vix_loc >= 0 and vix.iloc[vix_loc] > 25:
            # Wait for VIX to decline (first day VIX drops)
            entry_dt = None
            for j in range(1, 11):  # look ahead up to 10 days for VIX decline
                if vix_loc + j >= len(vix):
                    break
                if vix.iloc[vix_loc + j] < vix.iloc[vix_loc + j - 1]:
                    entry_dt = vix.index[vix_loc + j]
                    break

            if entry_dt is not None and entry_dt in qqq.index:
                entry_idx = qqq.index.get_loc(entry_dt)
                exit_idx = min(entry_idx + 10, len(qqq) - 1)
                entry_p = apply_slippage(qqq.iloc[entry_idx], "buy")
                exit_p = apply_slippage(qqq.iloc[exit_idx], "sell")
                pnl = (exit_p - entry_p) / entry_p
                trades.append({
                    "entry_date": entry_dt, "exit_date": qqq.index[exit_idx],
                    "ticker": "QQQ", "entry_price": entry_p, "exit_price": exit_p,
                    "pnl_pct": pnl, "hold_days": 10, "regime": get_regime(spy, entry_dt),
                })
                i = oot_dates.get_loc(oot_dates[oot_dates >= qqq.index[exit_idx]][0]) if len(oot_dates[oot_dates >= qqq.index[exit_idx]]) > 0 else len(oot_dates)
                continue
        i += 1
    return trades


def variant_f_bear_only(data, spy, oot_dates):
    """Bear market only (SPY<200SMA). Buy QQQ on 3-day >=2% decline. Hold 5 days. Bull -> hold GLD."""
    trades = []
    qqq = data.get("QQQ")
    gld = data.get("GLD")
    if qqq is None:
        return trades

    i = 0
    while i < len(oot_dates):
        dt = oot_dates[i]
        regime = get_regime(spy, dt)
        loc = spy.index.get_indexer([dt], method="ffill")[0]

        if regime == "bear" and loc >= 3:
            ret_3d = (spy.iloc[loc] - spy.iloc[loc - 3]) / spy.iloc[loc - 3]
            if ret_3d <= -0.02 and dt in qqq.index:
                entry_idx = qqq.index.get_loc(dt)
                exit_idx = min(entry_idx + 5, len(qqq) - 1)
                entry_p = apply_slippage(qqq.iloc[entry_idx], "buy")
                exit_p = apply_slippage(qqq.iloc[exit_idx], "sell")
                pnl = (exit_p - entry_p) / entry_p
                trades.append({
                    "entry_date": dt, "exit_date": qqq.index[exit_idx],
                    "ticker": "QQQ", "entry_price": entry_p, "exit_price": exit_p,
                    "pnl_pct": pnl, "hold_days": 5, "regime": regime,
                })
                i += 6
                continue
        i += 1
    return trades


# ── MAIN ────────────────────────────────────────────────────────────────────
def main():
    global data_cache
    data = fetch_data()
    data_cache = data

    spy = data.get("SPY")
    if spy is None:
        print("ERROR: No SPY data")
        sys.exit(1)

    # OOT date range (trading days only, from SPY index)
    oot_mask = (spy.index >= OOT_START) & (spy.index <= OOT_END)
    oot_dates = spy.index[oot_mask]
    print(f"\nOOT period: {oot_dates[0].date()} to {oot_dates[-1].date()} ({len(oot_dates)} trading days)")

    variants = {
        "A_simple_dip_buy": {
            "func": variant_a_simple_dip_buy,
            "description": "Buy QQQ when SPY drops >=3% over 5 days. Hold 10 days.",
        },
        "B_resilient_picker": {
            "func": variant_b_resilient_picker,
            "description": "Buy 3 most resilient growth stocks when SPY drops >=3%/5d. Hold 10 days.",
        },
        "C_deep_dip_buy": {
            "func": variant_c_deep_dip,
            "description": "Buy QQQ on SPY >=5% drop over 10 days. Hold 20 days.",
        },
        "D_two_stage_recovery": {
            "func": variant_d_two_stage,
            "description": "Two-stage averaging: 50% at -3%, another 50% if -5% cumulative. Hold 15d from last entry.",
        },
        "E_vix_confirmed": {
            "func": variant_e_vix_confirm,
            "description": "Buy QQQ after SPY -3%/5d + VIX>25 + first VIX decline day. Hold 10 days.",
        },
        "F_bear_only_recovery": {
            "func": variant_f_bear_only,
            "description": "Bear market only: buy QQQ on 3-day >=2% decline. Hold 5 days.",
        },
    }

    results = {
        "strategy": "Post-Selloff Recovery Timing",
        "run_date": datetime.now().isoformat(),
        "oot_period": f"{oot_dates[0].date()} to {oot_dates[-1].date()}",
        "capital": CAPITAL,
        "variants": {},
    }

    for name, cfg in variants.items():
        print(f"\n{'='*60}")
        print(f"Running {name}: {cfg['description']}")
        print(f"{'='*60}")

        trades = cfg["func"](data, spy, oot_dates)
        print(f"  Trades: {len(trades)}")

        if trades:
            metrics, regime, pnl_arr = compute_metrics(trades, spy)
            perm = permutation_test(trades, spy)
            gates = check_gates(metrics, regime, perm)

            print(f"  Return: {metrics['total_return_pct']:.1f}% | Sharpe: {metrics['sharpe']:.3f} | "
                  f"Sortino: {metrics['sortino']:.3f} | WR: {metrics['win_rate']:.0f}% | "
                  f"MaxDD: {metrics['max_drawdown_pct']:.1f}%")
            print(f"  PF: {metrics['profit_factor']:.2f} | Bull Sharpe: {regime['bull_sharpe']:.3f} | "
                  f"Bear Sharpe: {regime['bear_sharpe']:.3f} | Gap: {regime['regime_gap']:.3f}")
            print(f"  Perm p: {perm['perm_p_value']:.4f} | Gates: {'PASS' if gates['all_passed'] else 'FAIL'}")
            print(f"  Gate details: {gates}")
        else:
            metrics, regime, _ = empty_result()
            perm = {"perm_p_value": 1.0, "actual_mean_pnl": 0.0, "perm_mean_pnl": 0.0}
            gates = check_gates(metrics, regime, perm)
            print("  No trades generated")

        results["variants"][name] = {
            "description": cfg["description"],
            "metrics": metrics,
            "regime": regime,
            "permutation": perm,
            "gates": gates,
        }

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for name, v in results["variants"].items():
        m = v["metrics"]
        g = v["gates"]
        status = "PASS" if g["all_passed"] else "FAIL"
        failed = [k for k, val in g.items() if not val and k != "all_passed"]
        fail_str = f" (failed: {', '.join(failed)})" if failed else ""
        print(f"  {name}: Sharpe={m['sharpe']:.3f} Sortino={m['sortino']:.3f} "
              f"PF={m['profit_factor']:.2f} WR={m['win_rate']:.0f}% "
              f"Ret={m['total_return_pct']:.1f}% DD={m['max_drawdown_pct']:.1f}% "
              f"N={m['num_trades']} [{status}]{fail_str}")

    # Save
    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
