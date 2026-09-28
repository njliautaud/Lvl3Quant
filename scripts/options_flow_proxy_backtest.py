#!/usr/bin/env python3
"""
Options Flow Proxy Backtest — Put/Call Ratio Contrarian via VIX
===============================================================
Since real-time P/C ratio data is hard to get historically, we proxy extreme
fear/greed using VIX percentile + SPY intraday moves.

6 Variants:
  A) Fear contrarian SPY:  VIX>80pct + SPY drop>1% → buy SPY 5d
  B) Fear contrarian QQQ:  same signal → buy QQQ 5d (higher beta bounce)
  C) Extended hold:        same signal → buy SPY 10d
  D) Double fear:          VIX>90pct + SPY drop>2% → buy SPY 5d
  E) Greed contrarian:    VIX<20pct + SPY up>1% → short SPY 5d
  F) Combined:            long on fear, cash on greed, else hold SPY

5-Gate Validation:
  1. Sharpe > 0.5
  2. Permutation p < 0.05 (1000 iters)
  3. Regime gap < 0.5
  4. MaxDD > -50%
  5. >= 20 trades

OOT: Jan 2022 – Jul 2026 | Account: $645 | Slippage: 0.02%
"""

import json
import warnings
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

# ── Constants ──────────────────────────────────────────────────────────────
ACCOUNT = 645.0
SLIPPAGE_PCT = 0.0002  # 0.02%
OOT_START = "2022-01-01"
OOT_END = "2026-07-29"
DOWNLOAD_START = "2019-01-01"  # extra history for rolling calcs
VIX_ROLL = 60
N_PERMS = 1000
RESULTS_PATH = "/home/jupiter/Lvl3Quant/data/options_flow_proxy_results.json"
np.random.seed(42)


def download_data():
    """Download SPY, QQQ, IWM, ^VIX daily data."""
    print("Downloading market data...")
    tickers = {"SPY": "SPY", "QQQ": "QQQ", "IWM": "IWM", "VIX": "^VIX"}
    frames = {}
    for name, sym in tickers.items():
        df = yf.download(sym, start=DOWNLOAD_START, end=OOT_END, progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.index = pd.to_datetime(df.index).tz_localize(None)
        frames[name] = df
        print(f"  {name}: {len(df)} rows ({df.index[0].date()} -> {df.index[-1].date()})")
    return frames


def build_signals(frames):
    """Compute fear/greed signals from VIX + SPY price action."""
    spy = frames["SPY"].copy()
    vix = frames["VIX"][["Close"]].rename(columns={"Close": "VIX_Close"})

    # Merge VIX onto SPY dates
    df = spy.join(vix, how="inner")

    # Intraday return (open->close proxy for "drop" days)
    df["intraday_ret"] = (df["Close"] - df["Open"]) / df["Open"]
    # Daily close-to-close return
    df["daily_ret"] = df["Close"].pct_change()

    # Rolling VIX percentile (where does today's VIX sit in the last 60 days)
    df["vix_pct"] = df["VIX_Close"].rolling(VIX_ROLL).apply(
        lambda x: (x[-1] - x[:-1].min()) / (x[:-1].max() - x[:-1].min() + 1e-9),
        raw=True,
    )

    # Fear signals
    df["fear_80"] = (df["vix_pct"] > 0.80) & (df["intraday_ret"] < -0.01)
    df["fear_90"] = (df["vix_pct"] > 0.90) & (df["intraday_ret"] < -0.02)

    # Greed signal
    df["greed_20"] = (df["vix_pct"] < 0.20) & (df["intraday_ret"] > 0.01)

    # QQQ returns aligned to SPY dates
    qqq = frames["QQQ"][["Close"]].rename(columns={"Close": "QQQ_Close"})
    df = df.join(qqq, how="left")

    # Forward returns for SPY (5d, 10d)
    for h in [5, 10]:
        df[f"fwd_{h}d"] = df["Close"].pct_change(h).shift(-h)

    # QQQ forward returns
    qqq_close = frames["QQQ"]["Close"].reindex(df.index)
    for h in [5, 10]:
        df[f"qqq_fwd_{h}d"] = qqq_close.pct_change(h).shift(-h)

    # Regime: SPY 21-day return > 0 = bull, else bear
    df["regime"] = np.where(df["Close"].pct_change(21) > 0, "bull", "bear")

    return df


def run_variant(df_oot, signal_col, ret_col, hold_days, label):
    """Run a single signal-based variant backtest and return results dict."""
    signals = df_oot[df_oot[signal_col]].copy()
    n_trades = len(signals)

    if n_trades == 0:
        print(f"  [{label}] 0 trades -- SKIP")
        return None

    # Per-trade returns (after slippage on entry+exit)
    trade_rets = signals[ret_col].dropna() - 2 * SLIPPAGE_PCT
    n_trades = len(trade_rets)

    if n_trades == 0:
        print(f"  [{label}] 0 valid trades (NaN forward returns) -- SKIP")
        return None

    # Equity curve (compounding from $645)
    equity = ACCOUNT * (1 + trade_rets).cumprod()
    final_equity = equity.iloc[-1]
    total_ret = (final_equity / ACCOUNT) - 1

    # Sharpe (annualized)
    mean_ret = trade_rets.mean()
    std_ret = trade_rets.std()
    oot_years = max((df_oot.index[-1] - df_oot.index[0]).days / 365.25, 0.5)
    avg_trades_per_year = n_trades / oot_years
    sharpe = (mean_ret / (std_ret + 1e-9)) * np.sqrt(avg_trades_per_year) if std_ret > 0 else 0.0

    # Sortino
    downside = trade_rets[trade_rets < 0].std()
    sortino = (mean_ret / (downside + 1e-9)) * np.sqrt(avg_trades_per_year) if downside > 0 else 0.0

    # Win rate
    win_rate = (trade_rets > 0).mean()

    # Profit factor
    gains = trade_rets[trade_rets > 0].sum()
    losses = abs(trade_rets[trade_rets < 0].sum())
    pf = gains / (losses + 1e-9)

    # Max drawdown on equity curve
    running_max = equity.cummax()
    drawdown = (equity - running_max) / running_max
    max_dd = drawdown.min()

    # Regime-stratified Sharpe
    regime_at_signal = signals.loc[trade_rets.index, "regime"]
    bull_rets = trade_rets[regime_at_signal == "bull"]
    bear_rets = trade_rets[regime_at_signal == "bear"]
    sharpe_bull = (bull_rets.mean() / (bull_rets.std() + 1e-9)) * np.sqrt(max(len(bull_rets), 1)) if len(bull_rets) > 2 else 0.0
    sharpe_bear = (bear_rets.mean() / (bear_rets.std() + 1e-9)) * np.sqrt(max(len(bear_rets), 1)) if len(bear_rets) > 2 else 0.0
    regime_gap = abs(sharpe_bull - sharpe_bear) / max(abs(sharpe_bull), abs(sharpe_bear), 1e-9)

    # Permutation test: compare mean trade return vs randomly sampled periods
    observed_mean = mean_ret
    perm_count = 0
    all_rets = df_oot[ret_col].dropna().values
    if len(all_rets) > n_trades:
        for _ in range(N_PERMS):
            perm_sample = np.random.choice(all_rets, size=n_trades, replace=False)
            if perm_sample.mean() >= observed_mean:
                perm_count += 1
        perm_p = perm_count / N_PERMS
    else:
        perm_p = 1.0

    # 5-gate check
    gates = {
        "sharpe_gt_0.5": bool(sharpe > 0.5),
        "perm_p_lt_0.05": bool(perm_p < 0.05),
        "regime_gap_lt_0.5": bool(regime_gap < 0.5),
        "maxdd_gt_neg50": bool(max_dd > -0.50),
        "trades_gte_20": bool(n_trades >= 20),
    }
    gates_passed = sum(gates.values())

    result = {
        "variant": label,
        "n_trades": int(n_trades),
        "total_return_pct": round(float(total_ret * 100), 2),
        "final_equity": round(float(final_equity), 2),
        "sharpe": round(float(sharpe), 3),
        "sortino": round(float(sortino), 3),
        "win_rate": round(float(win_rate), 4),
        "profit_factor": round(float(pf), 3),
        "max_drawdown_pct": round(float(max_dd * 100), 2),
        "mean_trade_ret_pct": round(float(mean_ret * 100), 4),
        "regime_sharpe_bull": round(float(sharpe_bull), 3),
        "regime_sharpe_bear": round(float(sharpe_bear), 3),
        "regime_gap": round(float(regime_gap), 3),
        "perm_p_value": round(float(perm_p), 4),
        "gates": gates,
        "gates_passed": f"{gates_passed}/5",
        "verdict": "PASS" if gates_passed == 5 else "FAIL",
    }

    status = "PASS" if gates_passed == 5 else f"FAIL ({gates_passed}/5)"
    print(f"  [{label}] {n_trades} trades | Sharpe {sharpe:.3f} | WR {win_rate:.1%} | "
          f"PF {pf:.2f} | MaxDD {max_dd:.1%} | perm-p {perm_p:.3f} | regime_gap {regime_gap:.3f} | {status}")
    for gname, gval in gates.items():
        tag = "OK" if gval else "FAIL"
        print(f"      {tag}: {gname}")

    return result


def run_combined_variant(df_oot, label="F) Combined (fear-long / greed-cash / else-hold SPY)"):
    """
    Combined strategy: fully invested in SPY by default.
    On greed signal day: go to cash for 5 days.
    Fear signals just confirm holding (already long).
    """
    df = df_oot.copy()
    df["position"] = 1.0  # default: fully invested in SPY

    # On greed signal days, go to cash for next 5 days
    greed_dates = df.index[df["greed_20"]]
    for d in greed_dates:
        idx = df.index.get_loc(d)
        end_idx = min(idx + 6, len(df))
        if idx + 1 < len(df):
            df.iloc[idx + 1:end_idx, df.columns.get_loc("position")] = 0.0

    # Daily returns when in position
    df["strat_ret"] = df["daily_ret"] * df["position"]
    # Slippage on transitions (position changes)
    df["strat_ret"] = df["strat_ret"] - abs(df["position"].diff().fillna(0)) * SLIPPAGE_PCT

    equity = ACCOUNT * (1 + df["strat_ret"]).cumprod()
    final_equity = float(equity.iloc[-1])
    total_ret = (final_equity / ACCOUNT) - 1

    # Buy-and-hold comparison
    bnh_equity = ACCOUNT * (1 + df["daily_ret"]).cumprod()
    bnh_final = float(bnh_equity.iloc[-1])
    bnh_ret = (bnh_final / ACCOUNT) - 1

    # Stats
    daily_rets = df["strat_ret"].dropna()
    sharpe = float(daily_rets.mean() / (daily_rets.std() + 1e-9) * np.sqrt(252))
    downside = daily_rets[daily_rets < 0].std()
    sortino = float(daily_rets.mean() / (downside + 1e-9) * np.sqrt(252))
    wr_mask = daily_rets != 0
    win_rate = float((daily_rets[wr_mask] > 0).mean()) if wr_mask.any() else 0.0
    gains = float(daily_rets[daily_rets > 0].sum())
    losses = float(abs(daily_rets[daily_rets < 0].sum()))
    pf = gains / (losses + 1e-9)

    running_max = equity.cummax()
    drawdown = (equity - running_max) / running_max
    max_dd = float(drawdown.min())

    # Count greed exits as "trades"
    n_trades = int(len(greed_dates))

    # Regime
    bull_mask = df["regime"] == "bull"
    bear_mask = df["regime"] == "bear"
    sharpe_bull = float(daily_rets[bull_mask].mean() / (daily_rets[bull_mask].std() + 1e-9) * np.sqrt(252)) if bull_mask.sum() > 10 else 0.0
    sharpe_bear = float(daily_rets[bear_mask].mean() / (daily_rets[bear_mask].std() + 1e-9) * np.sqrt(252)) if bear_mask.sum() > 10 else 0.0
    regime_gap = float(abs(sharpe_bull - sharpe_bear) / max(abs(sharpe_bull), abs(sharpe_bear), 1e-9))

    # Permutation: shuffle which days we go to cash
    observed_ret = total_ret
    perm_count = 0
    daily_ret_vals = df["daily_ret"].values
    n_days = len(df)
    for _ in range(N_PERMS):
        perm_pos = np.ones(n_days)
        perm_greed_idx = np.random.choice(n_days, size=len(greed_dates), replace=False)
        for idx in perm_greed_idx:
            end_idx = min(idx + 6, n_days)
            if idx + 1 < n_days:
                perm_pos[idx + 1:end_idx] = 0.0
        perm_daily = daily_ret_vals * perm_pos
        perm_final = ACCOUNT * np.prod(1 + perm_daily)
        perm_ret_val = (perm_final / ACCOUNT) - 1
        if perm_ret_val >= observed_ret:
            perm_count += 1
    perm_p = float(perm_count / N_PERMS)

    gates = {
        "sharpe_gt_0.5": bool(sharpe > 0.5),
        "perm_p_lt_0.05": bool(perm_p < 0.05),
        "regime_gap_lt_0.5": bool(regime_gap < 0.5),
        "maxdd_gt_neg50": bool(max_dd > -0.50),
        "trades_gte_20": bool(n_trades >= 20),
    }
    gates_passed = sum(gates.values())

    result = {
        "variant": label,
        "n_trades": n_trades,
        "total_return_pct": round(total_ret * 100, 2),
        "final_equity": round(final_equity, 2),
        "bnh_return_pct": round(bnh_ret * 100, 2),
        "bnh_final_equity": round(bnh_final, 2),
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "win_rate": round(win_rate, 4),
        "profit_factor": round(pf, 3),
        "max_drawdown_pct": round(max_dd * 100, 2),
        "regime_sharpe_bull": round(sharpe_bull, 3),
        "regime_sharpe_bear": round(sharpe_bear, 3),
        "regime_gap": round(regime_gap, 3),
        "perm_p_value": round(perm_p, 4),
        "gates": gates,
        "gates_passed": f"{gates_passed}/5",
        "verdict": "PASS" if gates_passed == 5 else "FAIL",
    }

    status = "PASS" if gates_passed == 5 else f"FAIL ({gates_passed}/5)"
    print(f"  [{label[:40]}] {n_trades} greed-exits | Sharpe {sharpe:.3f} | "
          f"PF {pf:.2f} | MaxDD {max_dd:.1%} | Total {total_ret:.1%} vs BnH {bnh_ret:.1%} | {status}")
    for gname, gval in gates.items():
        tag = "OK" if gval else "FAIL"
        print(f"      {tag}: {gname}")

    return result


def main():
    print("=" * 80)
    print("OPTIONS FLOW PROXY BACKTEST -- Put/Call Ratio Contrarian via VIX")
    print(f"OOT: {OOT_START} -> {OOT_END} | Account: ${ACCOUNT} | Slippage: {SLIPPAGE_PCT*100:.2f}%")
    print("=" * 80)

    # Download & build signals
    frames = download_data()
    df = build_signals(frames)

    # Filter to OOT period
    df_oot = df.loc[OOT_START:OOT_END].copy()
    print(f"\nOOT period: {len(df_oot)} trading days ({df_oot.index[0].date()} -> {df_oot.index[-1].date()})")
    print(f"  Fear-80 signals: {df_oot['fear_80'].sum()}")
    print(f"  Fear-90 signals: {df_oot['fear_90'].sum()}")
    print(f"  Greed-20 signals: {df_oot['greed_20'].sum()}")

    print("\n" + "-" * 80)
    print("RUNNING 6 VARIANTS")
    print("-" * 80)

    results = []

    # A) Fear contrarian SPY, 5d hold
    print("\nVariant A: Fear contrarian -> buy SPY, hold 5 days")
    r = run_variant(df_oot, "fear_80", "fwd_5d", 5, "A) Fear->SPY 5d")
    if r:
        results.append(r)

    # B) Fear contrarian QQQ, 5d hold
    print("\nVariant B: Fear contrarian -> buy QQQ, hold 5 days")
    r = run_variant(df_oot, "fear_80", "qqq_fwd_5d", 5, "B) Fear->QQQ 5d")
    if r:
        results.append(r)

    # C) Extended hold SPY, 10d
    print("\nVariant C: Fear contrarian -> buy SPY, hold 10 days")
    r = run_variant(df_oot, "fear_80", "fwd_10d", 10, "C) Fear->SPY 10d")
    if r:
        results.append(r)

    # D) Double fear SPY, 5d
    print("\nVariant D: Double fear (VIX>90pct + drop>2%) -> buy SPY, hold 5 days")
    r = run_variant(df_oot, "fear_90", "fwd_5d", 5, "D) DblFear->SPY 5d")
    if r:
        results.append(r)

    # E) Greed contrarian (short/cash), 5d
    print("\nVariant E: Greed contrarian -> short SPY, hold 5 days")
    # For greed contrarian, we invert: profit from decline
    df_oot_e = df_oot.copy()
    df_oot_e["greed_short_ret"] = -df_oot_e["fwd_5d"]
    r = run_variant(df_oot_e, "greed_20", "greed_short_ret", 5, "E) Greed->short SPY 5d")
    if r:
        results.append(r)

    # F) Combined
    print("\nVariant F: Combined (long on fear, cash on greed, else hold SPY)")
    r = run_combined_variant(df_oot)
    if r:
        results.append(r)

    # ── Summary ──
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    pass_count = 0
    for r in results:
        v = r["verdict"]
        tag = "PASS" if v == "PASS" else "FAIL"
        print(f"  {tag} | {r['variant']:<45s} | Sharpe {r['sharpe']:>6.3f} | "
              f"WR {r['win_rate']:.1%} | PF {r['profit_factor']:>5.2f} | "
              f"MaxDD {r['max_drawdown_pct']:>6.1f}% | {r['gates_passed']} gates | "
              f"Final ${r['final_equity']:,.0f}")
        if v == "PASS":
            pass_count += 1

    print(f"\n{pass_count}/{len(results)} variants passed all 5 gates.")

    # Save results
    output = {
        "strategy": "Options Flow Proxy -- Put/Call Ratio Contrarian via VIX",
        "oot_period": f"{OOT_START} to {OOT_END}",
        "account": ACCOUNT,
        "slippage_pct": SLIPPAGE_PCT * 100,
        "run_date": datetime.now().isoformat(),
        "variants": results,
        "summary": {
            "total_variants": len(results),
            "passed_all_gates": pass_count,
        },
    }

    Path(RESULTS_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
