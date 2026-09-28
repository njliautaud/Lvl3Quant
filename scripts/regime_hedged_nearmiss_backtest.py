#!/usr/bin/env python3
"""
Regime-Hedged Near-Miss Backtest
================================
Tests 3 near-miss strategies × 4 hedge variants = 12 combinations.
Goal: find if regime-aware hedging can reduce regime gap below 0.5.

Strategies: Earnings Surprise Momentum, Composite Signal, Gap Reversal
Hedges: Half-Size Bear, Kill-Switch Skip, Bear Hedge (SH), Adaptive Sizing
"""

import json
import sys
import warnings
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

# ─── Configuration ───────────────────────────────────────────────────────────
OOT_START = "2022-01-01"
OOT_END = "2026-07-28"
STARTING_CAPITAL = 645.0
SLIPPAGE_PCT = 0.0002  # 0.02%
PERM_ITERATIONS = 500
REGIME_GAP_THRESHOLD = 0.5
SHARPE_THRESHOLD = 0.5
PERM_P_THRESHOLD = 0.05
MDD_THRESHOLD = -0.50  # -50%
MIN_TRADES = 20

EARNINGS_UNIVERSE = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "AMD",
    "NFLX", "CRM", "PLTR", "SOFI", "HOOD", "SNAP", "PINS", "UBER",
    "LYFT", "COIN", "RBLX", "DDOG", "TTD", "SHOP", "NET", "ROKU"
]

RESULTS_PATH = Path("/home/jupiter/Lvl3Quant/data/regime_hedged_nearmiss_results.json")


# ─── Data Download ───────────────────────────────────────────────────────────
def download_data():
    """Download all needed price data."""
    print("Downloading price data...")

    # We need extra history for 200-SMA warmup
    fetch_start = "2021-01-01"

    # SPY for regime detection, VIX for kill-switch/adaptive sizing, SH for hedging
    tickers_extra = ["SPY", "^VIX", "SH", "QQQ"]
    all_tickers = list(set(EARNINGS_UNIVERSE + tickers_extra))

    raw = yf.download(all_tickers, start=fetch_start, end=OOT_END, auto_adjust=True, progress=False)

    # Handle multi-level columns from yfinance
    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Close"].copy()
        opn = raw["Open"].copy()
        high = raw["High"].copy()
        low = raw["Low"].copy()
        volume = raw["Volume"].copy()
    else:
        # Single ticker fallback
        close = raw[["Close"]].copy()
        opn = raw[["Open"]].copy()
        high = raw[["High"]].copy()
        low = raw[["Low"]].copy()
        volume = raw[["Volume"]].copy()

    # Rename ^VIX to VIX
    for df in [close, opn, high, low, volume]:
        if "^VIX" in df.columns:
            df.rename(columns={"^VIX": "VIX"}, inplace=True)

    print(f"  Data: {close.index[0].date()} to {close.index[-1].date()}, {len(close)} days, {len(close.columns)} tickers")
    return close, opn, high, low, volume


# ─── Regime Detection ────────────────────────────────────────────────────────
def compute_regime(spy_close):
    """Bull = SPY > 200-SMA, Bear = SPY < 200-SMA."""
    sma200 = spy_close.rolling(200).mean()
    regime = pd.Series("bull", index=spy_close.index)
    regime[spy_close < sma200] = "bear"
    return regime


def compute_vix_regime(vix_close, spy_close):
    """For kill-switch: VIX>20 AND SPY<50-SMA."""
    sma50 = spy_close.rolling(50).mean()
    kill = (vix_close > 20) & (spy_close < sma50)
    return kill


# ─── Hedge Implementations ──────────────────────────────────────────────────
def apply_hedge(hedge_type, regime, vix, kill_switch, base_size=1.0):
    """Return position size multiplier and whether to skip entry."""
    if hedge_type == "A":  # Half-Size Bear
        if regime == "bear":
            return 0.5, False
        return base_size, False

    elif hedge_type == "B":  # Kill-Switch Skip
        if kill_switch:
            return 0.0, True
        return base_size, False

    elif hedge_type == "C":  # Bear Hedge (handled separately in backtest)
        return base_size, False

    elif hedge_type == "D":  # Adaptive Sizing
        if pd.isna(vix) or vix >= 40:
            return 0.0, True
        size = base_size * (1 - vix / 40.0)
        return max(size, 0.0), size <= 0

    return base_size, False


# ─── Strategy 1: Earnings Surprise Momentum ─────────────────────────────────
def strategy_earnings_momentum(close, opn, regime_s, vix_s, kill_sw, hedge_type, sh_close):
    """
    Buy growth stocks after >3% gap up on earnings day proxy, hold 40 days.
    Earnings proxy: gap > 3% with volume spike (we don't have actual earnings dates,
    so we use large gap-ups as proxy for earnings surprise).
    """
    trades = []
    oot_mask = close.index >= pd.Timestamp(OOT_START)
    oot_dates = close.index[oot_mask]

    for ticker in EARNINGS_UNIVERSE:
        if ticker not in close.columns or ticker not in opn.columns:
            continue

        c = close[ticker].dropna()
        o = opn[ticker].dropna()

        # Find earnings surprise proxy: open gap > 3% from prior close
        common_idx = c.index.intersection(o.index)
        prev_close = c.reindex(common_idx).shift(1)
        gap_pct = (o.reindex(common_idx) - prev_close) / prev_close

        # Filter to OOT period
        gap_pct = gap_pct[gap_pct.index >= pd.Timestamp(OOT_START)]
        signal_dates = gap_pct[gap_pct > 0.03].index

        for entry_date in signal_dates:
            idx_pos = c.index.get_loc(entry_date)

            # Get regime/VIX on entry date
            r = regime_s.get(entry_date, "bull")
            v = vix_s.get(entry_date, 15)
            ks = kill_sw.get(entry_date, False)

            size_mult, skip = apply_hedge(hedge_type, r, v, ks)
            if skip:
                continue

            # Entry at open + slippage
            entry_price = o.loc[entry_date] * (1 + SLIPPAGE_PCT)

            # Exit 40 trading days later (or last available)
            exit_idx = min(idx_pos + 40, len(c) - 1)
            exit_date = c.index[exit_idx]
            exit_price = c.iloc[exit_idx] * (1 - SLIPPAGE_PCT)

            ret = (exit_price / entry_price - 1) * size_mult

            # Hedge C: if bear, add SH hedge
            if hedge_type == "C" and r == "bear":
                if entry_date in sh_close.index and exit_date in sh_close.index:
                    sh_entry = sh_close.loc[entry_date] * (1 + SLIPPAGE_PCT)
                    sh_exit = sh_close.loc[exit_date] * (1 - SLIPPAGE_PCT)
                    sh_ret = (sh_exit / sh_entry - 1) * 0.3
                    ret += sh_ret

            trades.append({
                "entry_date": entry_date,
                "exit_date": exit_date,
                "ticker": ticker,
                "ret": ret,
                "regime": r,
                "size_mult": size_mult
            })

    return trades


# ─── Strategy 2: Composite Signal (QQQ) ─────────────────────────────────────
def strategy_composite_signal(close, opn, regime_s, vix_s, kill_sw, hedge_type, sh_close):
    """
    Buy QQQ when composite score > 50.
    Composite: earnings_proxy (20pts) + momentum_5d>0 (30pts) + RSI14<40 (30pts) + vol_spike (20pts).
    Hold 10 days.
    """
    trades = []

    if "QQQ" not in close.columns:
        return trades

    qqq = close["QQQ"].dropna()
    qqq_open = opn["QQQ"].dropna() if "QQQ" in opn.columns else qqq

    # Compute indicators
    mom5 = qqq.pct_change(5)

    # RSI 14
    delta = qqq.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - 100 / (1 + rs)

    # Earnings proxy for the universe: count how many had >3% gaps recently (5d)
    gap_count = pd.Series(0, index=qqq.index)
    for ticker in EARNINGS_UNIVERSE:
        if ticker not in close.columns or ticker not in opn.columns:
            continue
        c = close[ticker].reindex(qqq.index)
        o = opn[ticker].reindex(qqq.index)
        prev_c = c.shift(1)
        gap = (o - prev_c) / prev_c
        # Rolling 5-day count of >3% gaps
        big_gap = (gap > 0.03).astype(float).rolling(5, min_periods=1).sum()
        gap_count = gap_count.add(big_gap, fill_value=0)

    # Volume spike proxy: QQQ momentum reversal
    vol_proxy = qqq.pct_change().rolling(5).std()
    vol_high = vol_proxy > vol_proxy.rolling(60).quantile(0.8)

    oot_dates = qqq.index[qqq.index >= pd.Timestamp(OOT_START)]

    i = 0
    while i < len(oot_dates):
        date = oot_dates[i]

        # Composite score
        score = 0
        if date in gap_count.index and gap_count.loc[date] >= 1:
            score += 20  # earnings proxy
        if date in mom5.index and mom5.loc[date] > 0:
            score += 30  # momentum
        if date in rsi.index and rsi.loc[date] < 40:
            score += 30  # RSI oversold
        if date in vol_high.index and vol_high.loc[date]:
            score += 20  # vol spike

        if score > 50:
            r = regime_s.get(date, "bull")
            v = vix_s.get(date, 15)
            ks = kill_sw.get(date, False)

            size_mult, skip = apply_hedge(hedge_type, r, v, ks)
            if skip:
                i += 1
                continue

            idx_pos = qqq.index.get_loc(date)
            entry_price = qqq_open.loc[date] * (1 + SLIPPAGE_PCT) if date in qqq_open.index else qqq.loc[date] * (1 + SLIPPAGE_PCT)

            exit_idx = min(idx_pos + 10, len(qqq) - 1)
            exit_date = qqq.index[exit_idx]
            exit_price = qqq.iloc[exit_idx] * (1 - SLIPPAGE_PCT)

            ret = (exit_price / entry_price - 1) * size_mult

            if hedge_type == "C" and r == "bear":
                if date in sh_close.index and exit_date in sh_close.index:
                    sh_entry = sh_close.loc[date] * (1 + SLIPPAGE_PCT)
                    sh_exit = sh_close.loc[exit_date] * (1 - SLIPPAGE_PCT)
                    sh_ret = (sh_exit / sh_entry - 1) * 0.3
                    ret += sh_ret

            trades.append({
                "entry_date": date,
                "exit_date": exit_date,
                "ticker": "QQQ",
                "ret": ret,
                "regime": r,
                "size_mult": size_mult
            })

            # Skip forward past hold period to avoid overlapping trades
            i += 10
            continue

        i += 1

    return trades


# ─── Strategy 3: Gap Reversal (Intraday) ────────────────────────────────────
def strategy_gap_reversal(close, opn, regime_s, vix_s, kill_sw, hedge_type, sh_close):
    """
    Buy stocks that gap down >3% on open, sell at close same day.
    """
    trades = []
    oot_mask = close.index >= pd.Timestamp(OOT_START)
    oot_dates = close.index[oot_mask]

    for ticker in EARNINGS_UNIVERSE + ["QQQ", "SPY"]:
        if ticker not in close.columns or ticker not in opn.columns:
            continue

        c = close[ticker].dropna()
        o = opn[ticker].dropna()
        common = c.index.intersection(o.index)

        prev_close = c.reindex(common).shift(1)
        gap_pct = (o.reindex(common) - prev_close) / prev_close

        gap_pct = gap_pct[gap_pct.index >= pd.Timestamp(OOT_START)]
        signal_dates = gap_pct[gap_pct < -0.03].index

        for date in signal_dates:
            r = regime_s.get(date, "bull")
            v = vix_s.get(date, 15)
            ks = kill_sw.get(date, False)

            size_mult, skip = apply_hedge(hedge_type, r, v, ks)
            if skip:
                continue

            entry_price = o.loc[date] * (1 + SLIPPAGE_PCT)
            exit_price = c.loc[date] * (1 - SLIPPAGE_PCT)

            ret = (exit_price / entry_price - 1) * size_mult

            # Hedge C: intraday, use SH proportionally
            if hedge_type == "C" and r == "bear":
                if date in sh_close.index:
                    # For intraday, approximate SH return from open to close
                    if date in opn.get("SH", pd.Series()).index if "SH" in opn.columns else False:
                        sh_o = opn["SH"].loc[date] * (1 + SLIPPAGE_PCT)
                        sh_c = sh_close.loc[date] * (1 - SLIPPAGE_PCT)
                        sh_ret = (sh_c / sh_o - 1) * 0.3
                        ret += sh_ret

            trades.append({
                "entry_date": date,
                "exit_date": date,
                "ticker": ticker,
                "ret": ret,
                "regime": r,
                "size_mult": size_mult
            })

    return trades


# ─── Performance Metrics ─────────────────────────────────────────────────────
def compute_metrics(trades, starting_capital):
    """Compute all required metrics from a list of trades."""
    if not trades or len(trades) == 0:
        return {
            "sharpe": 0, "sortino": 0, "pf": 0, "wr": 0,
            "max_dd": -1.0, "n_trades": 0,
            "sharpe_bull": 0, "sharpe_bear": 0, "regime_gap": 1.0,
            "total_return": 0
        }

    df = pd.DataFrame(trades)
    rets = df["ret"].values
    n = len(rets)

    # Basic metrics
    mean_ret = np.mean(rets)
    std_ret = np.std(rets, ddof=1) if n > 1 else 1e-9

    # Annualize assuming ~252 trades/year scaling
    sharpe = (mean_ret / std_ret) * np.sqrt(252) if std_ret > 1e-9 else 0

    # Sortino
    downside = rets[rets < 0]
    downside_std = np.std(downside, ddof=1) if len(downside) > 1 else 1e-9
    sortino = (mean_ret / downside_std) * np.sqrt(252) if downside_std > 1e-9 else 0

    # Profit Factor
    gross_profit = np.sum(rets[rets > 0])
    gross_loss = abs(np.sum(rets[rets < 0]))
    pf = gross_profit / gross_loss if gross_loss > 1e-9 else 999.0

    # Win Rate
    wr = np.sum(rets > 0) / n

    # Max Drawdown on equity curve
    equity = starting_capital * np.cumprod(1 + rets)
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / peak
    max_dd = np.min(dd)

    # Total return
    total_return = equity[-1] / starting_capital - 1

    # Regime-specific Sharpe
    bull_rets = df[df["regime"] == "bull"]["ret"].values
    bear_rets = df[df["regime"] == "bear"]["ret"].values

    def regime_sharpe(r):
        if len(r) < 2:
            return 0
        m = np.mean(r)
        s = np.std(r, ddof=1)
        return (m / s) * np.sqrt(252) if s > 1e-9 else 0

    sharpe_bull = regime_sharpe(bull_rets)
    sharpe_bear = regime_sharpe(bear_rets)

    max_abs = max(abs(sharpe_bull), abs(sharpe_bear))
    regime_gap = abs(sharpe_bull - sharpe_bear) / max_abs if max_abs > 1e-9 else 0

    return {
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "pf": round(pf, 3),
        "wr": round(wr, 3),
        "max_dd": round(max_dd, 4),
        "n_trades": n,
        "n_bull": len(bull_rets),
        "n_bear": len(bear_rets),
        "sharpe_bull": round(sharpe_bull, 3),
        "sharpe_bear": round(sharpe_bear, 3),
        "regime_gap": round(regime_gap, 3),
        "total_return": round(total_return, 4),
        "mean_ret_per_trade": round(mean_ret, 6)
    }


# ─── Permutation Test ────────────────────────────────────────────────────────
def permutation_test(trades, observed_sharpe, n_iter=PERM_ITERATIONS):
    """Shuffle the sign of each return (randomize entry direction) to test significance.

    Null hypothesis: the strategy's directional calls (long vs short timing) don't matter.
    We randomly flip the sign of each trade's return, recompute Sharpe, and count
    how often the permuted Sharpe exceeds the observed one.
    """
    if len(trades) < 5:
        return 1.0

    rets = np.array([t["ret"] for t in trades])
    n = len(rets)
    count_above = 0

    rng = np.random.default_rng(42)

    for _ in range(n_iter):
        # Randomly flip sign of each return (±1 with equal probability)
        signs = rng.choice([-1, 1], size=n)
        permuted = rets * signs

        m = np.mean(permuted)
        s = np.std(permuted, ddof=1)
        perm_sharpe = (m / s) * np.sqrt(252) if s > 1e-9 else 0

        if perm_sharpe >= observed_sharpe:
            count_above += 1

    return round(count_above / n_iter, 4)


# ─── 5-Gate Check ────────────────────────────────────────────────────────────
def check_gates(metrics, perm_p):
    """Check all 5 gates. Returns dict of gate results."""
    gates = {
        "sharpe_gt_0.5": metrics["sharpe"] > SHARPE_THRESHOLD,
        "perm_p_lt_0.05": perm_p < PERM_P_THRESHOLD,
        "regime_gap_lt_0.5": metrics["regime_gap"] < REGIME_GAP_THRESHOLD,
        "mdd_gt_neg50": metrics["max_dd"] > MDD_THRESHOLD,
        "trades_gte_20": metrics["n_trades"] >= MIN_TRADES,
    }
    gates["all_pass"] = all(gates.values())
    gates["n_pass"] = sum(v for k, v in gates.items() if k not in ("all_pass", "n_pass"))
    return gates


# ─── Main ────────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("REGIME-HEDGED NEAR-MISS BACKTEST")
    print("=" * 70)

    close, opn, high, low, volume = download_data()

    # Regime signals
    spy_close = close["SPY"].dropna()
    regime_s = compute_regime(spy_close).to_dict()

    vix_close = close["VIX"].dropna() if "VIX" in close.columns else pd.Series(15, index=spy_close.index)
    vix_dict = vix_close.to_dict()

    kill_sw = compute_vix_regime(vix_close.reindex(spy_close.index, method="ffill").fillna(15), spy_close).to_dict()

    sh_close = close["SH"].dropna() if "SH" in close.columns else pd.Series(dtype=float)

    strategies = {
        "earnings_momentum": ("Earnings Surprise Momentum", strategy_earnings_momentum),
        "composite_signal": ("Composite Signal (Earnings Req)", strategy_composite_signal),
        "gap_reversal": ("Gap Reversal", strategy_gap_reversal),
    }

    hedge_names = {
        "A": "Half-Size Bear",
        "B": "Kill-Switch Skip",
        "C": "Bear Hedge (SH)",
        "D": "Adaptive Sizing",
    }

    all_results = []
    combo_num = 0
    total_combos = len(strategies) * len(hedge_names)

    for strat_key, (strat_name, strat_func) in strategies.items():
        for hedge_key, hedge_name in hedge_names.items():
            combo_num += 1
            label = f"{strat_name} + {hedge_name}"
            print(f"\n[{combo_num}/{total_combos}] {label}")
            print("-" * 50)

            # Run strategy
            trades = strat_func(close, opn, regime_s, vix_dict, kill_sw, hedge_key, sh_close)

            # Compute metrics
            metrics = compute_metrics(trades, STARTING_CAPITAL)

            # Permutation test
            perm_p = permutation_test(trades, metrics["sharpe"])

            # Gate check
            gates = check_gates(metrics, perm_p)

            result = {
                "strategy": strat_name,
                "hedge": hedge_name,
                "hedge_code": hedge_key,
                "label": label,
                "metrics": metrics,
                "perm_p": perm_p,
                "gates": gates,
            }
            all_results.append(result)

            # Print summary
            status = "PASS ALL 5" if gates["all_pass"] else f"PASS {gates['n_pass']}/5"
            print(f"  Trades: {metrics['n_trades']} (bull={metrics.get('n_bull',0)}, bear={metrics.get('n_bear',0)})")
            print(f"  Sharpe: {metrics['sharpe']:.3f}  Sortino: {metrics['sortino']:.3f}  PF: {metrics['pf']:.3f}  WR: {metrics['wr']:.1%}")
            print(f"  MaxDD: {metrics['max_dd']:.2%}  TotalRet: {metrics['total_return']:.2%}")
            print(f"  Sharpe_bull: {metrics['sharpe_bull']:.3f}  Sharpe_bear: {metrics['sharpe_bear']:.3f}  Regime Gap: {metrics['regime_gap']:.3f}")
            print(f"  Perm p-value: {perm_p:.4f}")
            print(f"  >>> {status} <<<")

            # Print gate details
            for gate_name, passed in gates.items():
                if gate_name in ("all_pass", "n_pass"):
                    continue
                mark = "PASS" if passed else "FAIL"
                print(f"    [{mark}] {gate_name}")

    # ─── Summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    # Sort by gates passed then sharpe
    all_results.sort(key=lambda x: (-x["gates"]["n_pass"], -x["metrics"]["sharpe"]))

    print(f"\n{'Label':<50} {'Sharpe':>7} {'RGap':>6} {'Perm_p':>7} {'Gates':>6}")
    print("-" * 80)
    for r in all_results:
        status = "ALL 5" if r["gates"]["all_pass"] else f"{r['gates']['n_pass']}/5"
        print(f"{r['label']:<50} {r['metrics']['sharpe']:>7.3f} {r['metrics']['regime_gap']:>6.3f} {r['perm_p']:>7.4f} {status:>6}")

    # Any pass all 5?
    winners = [r for r in all_results if r["gates"]["all_pass"]]
    print(f"\n{'='*70}")
    if winners:
        print(f"WINNERS: {len(winners)} combination(s) pass ALL 5 gates!")
        for w in winners:
            print(f"  * {w['label']} — Sharpe={w['metrics']['sharpe']:.3f}, RegimeGap={w['metrics']['regime_gap']:.3f}")
    else:
        print("NO combination passes all 5 gates.")
        best = all_results[0]
        print(f"  Best: {best['label']} — {best['gates']['n_pass']}/5 gates, Sharpe={best['metrics']['sharpe']:.3f}, RegimeGap={best['metrics']['regime_gap']:.3f}")

    # ─── Save Results ────────────────────────────────────────────────────
    # Convert for JSON serialization
    for r in all_results:
        for k, v in list(r["metrics"].items()):
            if isinstance(v, (np.floating, np.integer)):
                r["metrics"][k] = float(v)
        if isinstance(r["perm_p"], (np.floating,)):
            r["perm_p"] = float(r["perm_p"])
        for k, v in list(r["gates"].items()):
            if isinstance(v, (np.bool_,)):
                r["gates"][k] = bool(v)

    output = {
        "run_date": dt.datetime.now().isoformat(),
        "config": {
            "oot_start": OOT_START,
            "oot_end": OOT_END,
            "starting_capital": STARTING_CAPITAL,
            "slippage_pct": SLIPPAGE_PCT,
            "perm_iterations": PERM_ITERATIONS,
        },
        "n_combos": len(all_results),
        "n_all_pass": len(winners),
        "results": all_results
    }

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nResults saved to {RESULTS_PATH}")
    print("Done.")


if __name__ == "__main__":
    main()
