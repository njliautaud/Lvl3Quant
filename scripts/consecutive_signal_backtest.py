#!/usr/bin/env python3
"""
Consecutive/Stacking Signal Strength Backtest
Tests whether multiple independent signals firing simultaneously produces stronger results.

Variants A-F with 5-gate validation.
"""

import json
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from collections import defaultdict

warnings.filterwarnings("ignore")

# ── CONFIG ──────────────────────────────────────────────────────────────────────
UNIVERSE = [
    "AAPL", "MSFT", "AVGO", "JPM", "JNJ", "PG", "KO", "PEP", "HD", "COST",
    "UNH", "LLY", "V", "MA", "ABBV", "MRK", "WMT", "AMZN", "GOOGL", "META",
]
START = "2022-01-01"
END = "2026-07-31"
CAPITAL = 645.0
MAX_PER_TRADE = 200.0
MAX_CONCURRENT = 3
HOLD_DAYS = 10
SLIPPAGE_BPS = 2
N_PERMUTATIONS = 1000
RESULTS_PATH = "/home/jupiter/Lvl3Quant/data/consecutive_signal_results.json"


# ── DATA DOWNLOAD ───────────────────────────────────────────────────────────────
def download_data():
    """Download daily and weekly data for universe."""
    print("Downloading data...")
    # Need extra history for indicators
    fetch_start = (pd.Timestamp(START) - timedelta(days=120)).strftime("%Y-%m-%d")

    daily = {}
    weekly = {}
    for sym in UNIVERSE:
        try:
            tk = yf.Ticker(sym)
            df = tk.history(start=fetch_start, end=END, interval="1d", auto_adjust=True)
            if len(df) < 60:
                print(f"  {sym}: insufficient data ({len(df)} rows), skipping")
                continue
            df.index = df.index.tz_localize(None)
            daily[sym] = df
            # Build weekly from daily
            wk = df.resample("W-FRI").agg({
                "Open": "first", "High": "max", "Low": "min",
                "Close": "last", "Volume": "sum"
            }).dropna()
            weekly[sym] = wk
            print(f"  {sym}: {len(df)} daily bars, {len(wk)} weekly bars")
        except Exception as e:
            print(f"  {sym}: download error: {e}")
    return daily, weekly


# ── SIGNAL DETECTION ────────────────────────────────────────────────────────────
def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def detect_signals(daily, weekly):
    """Return dict: sym -> DataFrame with boolean columns for each signal, aligned to daily dates."""
    all_signals = {}

    for sym in daily:
        df = daily[sym].copy()
        n = len(df)

        # Pre-compute indicators
        close = df["Close"]
        volume = df["Volume"]
        high = df["High"]
        low = df["Low"]

        rsi14 = compute_rsi(close, 14)
        high20 = close.rolling(20).max()
        vol_ma20 = volume.rolling(20).mean()
        sma20 = close.rolling(20).mean()
        std20 = close.rolling(20).std()
        bb_lower = sma20 - 2 * std20

        # Weekly RSI
        wk = weekly[sym]
        wk_rsi = compute_rsi(wk["Close"], 14)

        signals = pd.DataFrame(index=df.index)

        # 1. RSI_DIP: RSI(14) < 35
        signals["RSI_DIP"] = rsi14 < 35

        # 2. PRICE_DIP: Price > 5% below 20-day high
        signals["PRICE_DIP"] = close < (high20 * 0.95)

        # 3. GREEN_AFTER_RED: First green day after 3+ consecutive red days
        is_green = close > df["Open"]
        is_red = close <= df["Open"]

        gar = pd.Series(False, index=df.index)
        red_streak = 0
        for i in range(len(df)):
            if is_red.iloc[i]:
                red_streak += 1
            else:
                if red_streak >= 3 and is_green.iloc[i]:
                    gar.iloc[i] = True
                red_streak = 0
        signals["GREEN_AFTER_RED"] = gar

        # 4. VOLUME_CLIMAX: Volume > 2x 20d avg on a down day, followed by lower-volume recovery
        down_day = close < close.shift(1)
        high_vol_down = (volume > 2 * vol_ma20) & down_day

        vc = pd.Series(False, index=df.index)
        for i in range(1, len(df)):
            if i >= 1 and high_vol_down.iloc[i - 1]:
                # Today is recovery: price up and volume lower than yesterday
                if close.iloc[i] > close.iloc[i - 1] and volume.iloc[i] < volume.iloc[i - 1]:
                    vc.iloc[i] = True
        signals["VOLUME_CLIMAX"] = vc

        # 5. BB_LOWER: Price touches or breaks below lower Bollinger Band
        signals["BB_LOWER"] = low <= bb_lower

        # 6. WEEKLY_OVERSOLD: Weekly RSI declining 2+ consecutive weeks
        wk_rsi_declining = (wk_rsi < wk_rsi.shift(1))
        wk_2consec = wk_rsi_declining & wk_rsi_declining.shift(1)

        # Map weekly signal to daily dates
        wo = pd.Series(False, index=df.index)
        for wk_date in wk_2consec[wk_2consec].index:
            # Signal fires on all daily dates in that week
            week_start = wk_date - timedelta(days=6)
            mask = (df.index >= week_start) & (df.index <= wk_date)
            wo.loc[mask] = True
        signals["WEEKLY_OVERSOLD"] = wo

        # Signal count
        sig_cols = ["RSI_DIP", "PRICE_DIP", "GREEN_AFTER_RED", "VOLUME_CLIMAX", "BB_LOWER", "WEEKLY_OVERSOLD"]
        signals["score"] = signals[sig_cols].sum(axis=1)

        all_signals[sym] = signals

    return all_signals


# ── BACKTEST ENGINE ─────────────────────────────────────────────────────────────
class Position:
    def __init__(self, sym, entry_date, entry_price, shares, score=0):
        self.sym = sym
        self.entry_date = entry_date
        self.entry_price = entry_price
        self.shares = shares
        self.score = score


def run_backtest(daily, signals, variant, start_date, end_date):
    """
    Run backtest for a given variant. Returns list of trade dicts.

    Variants:
    A: any 2 signals, hold 10d
    B: any 3 signals, hold 10d
    C: any 4+ signals, hold 10d
    D: weighted scoring, score>=3, size proportional to score, hold 10d
    E: must have RSI_DIP + PRICE_DIP, any others bonus, hold 10d
    F: sequential - RSI_DIP, then GREEN_AFTER_RED within 3d, then BB/volume within 3d, hold 10d
    """
    trades = []
    positions = []

    # Get sorted trading dates across all symbols (use first symbol's dates as reference)
    all_dates = sorted(set().union(*[daily[s].index for s in daily]))
    all_dates = [d for d in all_dates if start_date <= d <= end_date]

    # For variant F, track pending sequential triggers
    # stage1: RSI_DIP fired, waiting for GREEN_AFTER_RED within 3 days
    # stage2: GREEN_AFTER_RED confirmed, waiting for BB/VOLUME within 3 days
    f_pending = defaultdict(list)  # sym -> [(stage, trigger_date)]

    for date in all_dates:
        # Close expired positions
        new_positions = []
        for pos in positions:
            days_held = len([d for d in all_dates if pos.entry_date < d <= date])
            if days_held >= HOLD_DAYS:
                # Exit
                if pos.sym in daily and date in daily[pos.sym].index:
                    exit_price = daily[pos.sym].loc[date, "Close"]
                    exit_price *= (1 - SLIPPAGE_BPS / 10000)  # slippage on sell
                    pnl = (exit_price - pos.entry_price) * pos.shares
                    ret = (exit_price / pos.entry_price) - 1
                    trades.append({
                        "sym": pos.sym,
                        "entry_date": str(pos.entry_date.date()),
                        "exit_date": str(date.date()),
                        "entry_price": round(pos.entry_price, 4),
                        "exit_price": round(exit_price, 4),
                        "shares": round(pos.shares, 4),
                        "pnl": round(pnl, 4),
                        "return": round(ret, 6),
                        "score": pos.score,
                        "variant": variant,
                    })
                else:
                    new_positions.append(pos)
                    continue
            else:
                new_positions.append(pos)
        positions = new_positions

        if len(positions) >= MAX_CONCURRENT:
            continue

        # Check for new entries
        held_syms = {p.sym for p in positions}

        for sym in daily:
            if sym in held_syms:
                continue
            if len(positions) >= MAX_CONCURRENT:
                break
            if date not in daily[sym].index or sym not in signals:
                continue
            if date not in signals[sym].index:
                continue

            sig = signals[sym].loc[date]
            score = int(sig["score"])

            should_buy = False
            pos_size = MAX_PER_TRADE
            entry_score = score

            if variant == "A":
                should_buy = score >= 2
            elif variant == "B":
                should_buy = score >= 3
            elif variant == "C":
                should_buy = score >= 4
            elif variant == "D":
                if score >= 3:
                    should_buy = True
                    # Size proportional to score: base $100, +$20 per signal above 3, cap $200
                    pos_size = min(MAX_PER_TRADE, 100 + (score - 3) * 33.33)
                    entry_score = score
            elif variant == "E":
                # Must have RSI_DIP and PRICE_DIP
                if sig["RSI_DIP"] and sig["PRICE_DIP"]:
                    should_buy = True
                    entry_score = score
            elif variant == "F":
                # Sequential logic
                # Check stage 2 completions first
                completed = False
                if sym in f_pending:
                    new_pending = []
                    for stage, trigger_date in f_pending[sym]:
                        days_since = len([d for d in all_dates if trigger_date < d <= date])
                        if days_since > 3:
                            continue  # expired
                        if stage == 2:
                            # Waiting for BB or VOLUME confirmation
                            if sig["BB_LOWER"] or sig["VOLUME_CLIMAX"]:
                                should_buy = True
                                entry_score = score
                                completed = True
                                break
                            else:
                                new_pending.append((stage, trigger_date))
                        elif stage == 1:
                            # Waiting for GREEN_AFTER_RED
                            if sig["GREEN_AFTER_RED"]:
                                new_pending.append((2, date))  # advance to stage 2
                            else:
                                new_pending.append((stage, trigger_date))
                    if not completed:
                        f_pending[sym] = new_pending
                    else:
                        f_pending[sym] = []

                # Check new stage 1 triggers
                if sig["RSI_DIP"] and not should_buy:
                    if sym not in f_pending:
                        f_pending[sym] = []
                    f_pending[sym].append((1, date))

            if should_buy:
                entry_price = daily[sym].loc[date, "Close"]
                entry_price *= (1 + SLIPPAGE_BPS / 10000)  # slippage on buy
                shares = pos_size / entry_price
                positions.append(Position(sym, date, entry_price, shares, entry_score))
                held_syms.add(sym)

    # Force-close remaining positions at end
    last_date = all_dates[-1] if all_dates else None
    if last_date:
        for pos in positions:
            if pos.sym in daily and last_date in daily[pos.sym].index:
                exit_price = daily[pos.sym].loc[last_date, "Close"]
                exit_price *= (1 - SLIPPAGE_BPS / 10000)
                pnl = (exit_price - pos.entry_price) * pos.shares
                ret = (exit_price / pos.entry_price) - 1
                trades.append({
                    "sym": pos.sym,
                    "entry_date": str(pos.entry_date.date()),
                    "exit_date": str(last_date.date()),
                    "entry_price": round(pos.entry_price, 4),
                    "exit_price": round(exit_price, 4),
                    "shares": round(pos.shares, 4),
                    "pnl": round(pnl, 4),
                    "return": round(ret, 6),
                    "score": pos.score,
                    "variant": variant,
                })

    return trades


# ── METRICS ─────────────────────────────────────────────────────────────────────
def compute_metrics(trades):
    """Compute performance metrics from trade list."""
    if not trades:
        return {"n_trades": 0, "sharpe": 0, "total_pnl": 0, "win_rate": 0, "max_dd": 0, "profit_factor": 0}

    returns = np.array([t["return"] for t in trades])
    pnls = np.array([t["pnl"] for t in trades])

    n = len(returns)
    win_rate = np.mean(returns > 0)
    total_pnl = np.sum(pnls)
    avg_ret = np.mean(returns)
    std_ret = np.std(returns, ddof=1) if n > 1 else 1e-9

    # Annualize: ~25 trades/year assumption, scale by sqrt
    trades_per_year = max(1, n / 4.5)  # ~4.5 years of data
    sharpe = (avg_ret / std_ret) * np.sqrt(trades_per_year) if std_ret > 1e-9 else 0

    # Sortino
    downside = returns[returns < 0]
    downside_std = np.std(downside, ddof=1) if len(downside) > 1 else 1e-9
    sortino = (avg_ret / downside_std) * np.sqrt(trades_per_year) if downside_std > 1e-9 else 0

    # Profit factor
    gross_profit = np.sum(pnls[pnls > 0])
    gross_loss = abs(np.sum(pnls[pnls < 0]))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    # Max drawdown on equity curve
    equity = CAPITAL + np.cumsum(pnls)
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / peak
    max_dd = np.min(dd) if len(dd) > 0 else 0

    # Avg win / avg loss
    wins = pnls[pnls > 0]
    losses = pnls[pnls < 0]
    avg_win = np.mean(wins) if len(wins) > 0 else 0
    avg_loss = np.mean(losses) if len(losses) > 0 else 0

    return {
        "n_trades": n,
        "total_pnl": round(float(total_pnl), 2),
        "win_rate": round(float(win_rate), 4),
        "sharpe": round(float(sharpe), 4),
        "sortino": round(float(sortino), 4),
        "profit_factor": round(float(profit_factor), 4),
        "max_dd": round(float(max_dd), 4),
        "avg_return": round(float(avg_ret), 6),
        "avg_win": round(float(avg_win), 4),
        "avg_loss": round(float(avg_loss), 4),
        "final_equity": round(float(CAPITAL + total_pnl), 2),
    }


# ── REGIME ANALYSIS ─────────────────────────────────────────────────────────────
def regime_analysis(trades, spy_daily):
    """Split trades by market regime (SPY up/down months) and compute gap."""
    if not trades or spy_daily is None:
        return {"regime_gap": 0, "bull_sharpe": 0, "bear_sharpe": 0}

    # Monthly SPY returns
    spy_monthly = spy_daily["Close"].resample("ME").last().pct_change()

    bull_trades = []
    bear_trades = []

    for t in trades:
        entry = pd.Timestamp(t["entry_date"])
        month_key = entry.to_period("M").to_timestamp()
        # Find nearest month
        nearest = spy_monthly.index[spy_monthly.index <= entry]
        if len(nearest) == 0:
            continue
        m_ret = spy_monthly.get(nearest[-1], 0)
        if pd.isna(m_ret):
            continue
        if m_ret >= 0:
            bull_trades.append(t)
        else:
            bear_trades.append(t)

    bull_m = compute_metrics(bull_trades)
    bear_m = compute_metrics(bear_trades)

    bull_sharpe = bull_m["sharpe"]
    bear_sharpe = bear_m["sharpe"]

    max_abs = max(abs(bull_sharpe), abs(bear_sharpe), 1e-9)
    regime_gap = abs(bull_sharpe - bear_sharpe) / max_abs

    return {
        "regime_gap": round(regime_gap, 4),
        "bull_sharpe": round(bull_sharpe, 4),
        "bear_sharpe": round(bear_sharpe, 4),
        "bull_trades": len(bull_trades),
        "bear_trades": len(bear_trades),
    }


# ── PERMUTATION TEST ────────────────────────────────────────────────────────────
def permutation_test(trades, daily, n_perms=N_PERMUTATIONS):
    """Generate random entry trades and compare to actual performance."""
    if not trades or len(trades) < 5:
        return {"p_value": 1.0, "actual_avg_ret": 0, "random_avg_ret": 0}

    actual_avg = np.mean([t["return"] for t in trades])
    n_trades = len(trades)

    # Pool of all available (sym, date) pairs
    pool = []
    for sym in daily:
        df = daily[sym]
        valid = df.loc[START:END]
        # Only dates where we have 10 days of forward data
        for i in range(len(valid) - HOLD_DAYS):
            pool.append((sym, valid.index[i], valid.index[min(i + HOLD_DAYS, len(valid) - 1)]))

    if not pool:
        return {"p_value": 1.0, "actual_avg_ret": 0, "random_avg_ret": 0}

    rng = np.random.RandomState(42)
    count_better = 0
    random_avgs = []

    for _ in range(n_perms):
        idxs = rng.choice(len(pool), size=min(n_trades, len(pool)), replace=False)
        rets = []
        for idx in idxs:
            sym, entry_date, exit_date = pool[idx]
            ep = daily[sym].loc[entry_date, "Close"] * (1 + SLIPPAGE_BPS / 10000)
            xp = daily[sym].loc[exit_date, "Close"] * (1 - SLIPPAGE_BPS / 10000)
            rets.append(xp / ep - 1)
        rand_avg = np.mean(rets)
        random_avgs.append(rand_avg)
        if rand_avg >= actual_avg:
            count_better += 1

    p_value = count_better / n_perms
    return {
        "p_value": round(p_value, 4),
        "actual_avg_ret": round(float(actual_avg), 6),
        "random_avg_ret": round(float(np.mean(random_avgs)), 6),
    }


# ── 5-GATE VALIDATION ──────────────────────────────────────────────────────────
def validate_5gate(metrics, perm_result, regime_result):
    """Apply 5-gate validation."""
    gates = {}
    gates["G1_sharpe_gt_0.5"] = metrics["sharpe"] > 0.5
    gates["G2_perm_p_lt_0.05"] = perm_result["p_value"] < 0.05
    gates["G3_regime_gap_lt_0.5"] = regime_result["regime_gap"] < 0.5
    gates["G4_max_dd_gt_neg50pct"] = metrics["max_dd"] > -0.50
    gates["G5_min_20_trades"] = metrics["n_trades"] >= 20
    gates["passed_all"] = all(gates.values())
    return gates


# ── MAIN ────────────────────────────────────────────────────────────────────────
def main():
    daily, weekly = download_data()

    # Download SPY for regime analysis
    print("Downloading SPY for regime analysis...")
    spy = yf.Ticker("SPY")
    spy_daily = spy.history(start=START, end=END, interval="1d", auto_adjust=True)
    spy_daily.index = spy_daily.index.tz_localize(None)

    print("\nDetecting signals...")
    signals = detect_signals(daily, weekly)

    # Signal frequency stats
    print("\n── Signal Frequency ──")
    sig_cols = ["RSI_DIP", "PRICE_DIP", "GREEN_AFTER_RED", "VOLUME_CLIMAX", "BB_LOWER", "WEEKLY_OVERSOLD"]
    for col in sig_cols:
        total = sum(signals[s].loc[START:END][col].sum() for s in signals)
        print(f"  {col}: {int(total)} fires")

    # Score distribution
    print("\n── Score Distribution ──")
    for threshold in range(1, 7):
        total = sum((signals[s].loc[START:END]["score"] >= threshold).sum() for s in signals)
        print(f"  Score >= {threshold}: {int(total)} days")

    start_dt = pd.Timestamp(START)
    end_dt = pd.Timestamp(END)

    variants = ["A", "B", "C", "D", "E", "F"]
    results = {}

    for v in variants:
        print(f"\n{'='*60}")
        print(f"  VARIANT {v}")
        print(f"{'='*60}")

        trades = run_backtest(daily, signals, v, start_dt, end_dt)
        metrics = compute_metrics(trades)

        print(f"  Trades: {metrics['n_trades']}")
        print(f"  Total PnL: ${metrics['total_pnl']:.2f}")
        print(f"  Win Rate: {metrics['win_rate']:.1%}")
        print(f"  Sharpe: {metrics['sharpe']:.4f}")
        print(f"  Sortino: {metrics['sortino']:.4f}")
        print(f"  Profit Factor: {metrics['profit_factor']:.4f}")
        print(f"  Max DD: {metrics['max_dd']:.2%}")
        print(f"  Final Equity: ${metrics['final_equity']:.2f}")

        # Permutation test
        print(f"  Running permutation test ({N_PERMUTATIONS} perms)...")
        perm = permutation_test(trades, daily)
        print(f"  Perm p-value: {perm['p_value']:.4f}")

        # Regime analysis
        regime = regime_analysis(trades, spy_daily)
        print(f"  Regime gap: {regime['regime_gap']:.4f} (bull Sharpe: {regime['bull_sharpe']:.4f}, bear: {regime['bear_sharpe']:.4f})")

        # 5-gate
        gates = validate_5gate(metrics, perm, regime)
        print(f"  5-Gate: {'PASS' if gates['passed_all'] else 'FAIL'}")
        for g, v_bool in gates.items():
            if g != "passed_all":
                status = "PASS" if v_bool else "FAIL"
                print(f"    {g}: {status}")

        # Top symbols
        sym_pnl = defaultdict(float)
        sym_count = defaultdict(int)
        for t in trades:
            sym_pnl[t["sym"]] += t["pnl"]
            sym_count[t["sym"]] += 1
        top_syms = sorted(sym_pnl.items(), key=lambda x: x[1], reverse=True)[:5]
        if top_syms:
            print(f"  Top symbols: {', '.join(f'{s}(${p:.1f}/{sym_count[s]}t)' for s, p in top_syms)}")

        results[v] = {
            "variant": v,
            "metrics": metrics,
            "permutation_test": perm,
            "regime_analysis": regime,
            "five_gate": gates,
            "trade_count_by_symbol": dict(sym_count),
            "pnl_by_symbol": {k: round(v2, 2) for k, v2 in sym_pnl.items()},
            "sample_trades": trades[:10] if trades else [],
        }

    # ── SUMMARY ──────────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Var':<4} {'Trades':>6} {'PnL':>10} {'WR':>7} {'Sharpe':>8} {'Sortino':>8} {'PF':>7} {'MaxDD':>8} {'5G':>5}")
    print(f"  {'-'*56}")
    for v in variants:
        m = results[v]["metrics"]
        g = results[v]["five_gate"]["passed_all"]
        print(f"  {v:<4} {m['n_trades']:>6} {m['total_pnl']:>10.2f} {m['win_rate']:>6.1%} {m['sharpe']:>8.4f} {m['sortino']:>8.4f} {m['profit_factor']:>7.2f} {m['max_dd']:>7.2%} {'PASS' if g else 'FAIL':>5}")

    # Best variant
    best = max(results.keys(), key=lambda v: results[v]["metrics"]["sharpe"])
    print(f"\n  Best by Sharpe: Variant {best} (Sharpe={results[best]['metrics']['sharpe']:.4f})")

    passed = [v for v in variants if results[v]["five_gate"]["passed_all"]]
    if passed:
        print(f"  Passed 5-Gate: {', '.join(passed)}")
    else:
        print(f"  No variants passed all 5 gates.")

    # Signal stacking analysis
    print(f"\n── Signal Stacking Effect ──")
    for v in ["A", "B", "C"]:
        m = results[v]["metrics"]
        print(f"  {v} (>={2 if v=='A' else 3 if v=='B' else 4} signals): {m['n_trades']} trades, Sharpe={m['sharpe']:.4f}, WR={m['win_rate']:.1%}")

    # Save results
    output = {
        "metadata": {
            "strategy": "Consecutive Signal Stacking",
            "universe": UNIVERSE,
            "period": f"{START} to {END}",
            "capital": CAPITAL,
            "max_per_trade": MAX_PER_TRADE,
            "max_concurrent": MAX_CONCURRENT,
            "hold_days": HOLD_DAYS,
            "slippage_bps": SLIPPAGE_BPS,
            "run_date": datetime.now().isoformat(),
        },
        "variants": results,
        "summary": {
            "best_by_sharpe": best,
            "five_gate_passers": passed,
        },
    }

    with open(RESULTS_PATH, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
