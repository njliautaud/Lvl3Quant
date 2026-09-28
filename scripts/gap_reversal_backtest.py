#!/usr/bin/env python3
"""
Gap-Down Reversal Patterns on Quality Stocks — Backtest
========================================================
Tests 6 variants of gap-down mean-reversion strategies on a universe of
20 large-cap quality stocks, 2022-01-01 to 2026-07-31.

Output: /home/jupiter/Lvl3Quant/data/gap_reversal_results.json
"""

import json
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

# ── CONFIG ──────────────────────────────────────────────────────────────────
UNIVERSE = [
    "AAPL", "MSFT", "AVGO", "JPM", "JNJ", "PG", "KO", "PEP", "HD", "COST",
    "UNH", "LLY", "V", "MA", "ABBV", "MRK", "WMT", "AMZN", "GOOGL", "META",
]
CAPITAL = 669.0
MAX_PER_TRADE = 200.0
MAX_CONCURRENT = 3
SLIPPAGE_BPS = 2  # 2 bps each way
HOLD_DAYS = 10
BT_START = "2022-01-01"
BT_END = "2026-07-31"
DL_START = "2021-06-01"  # extra lookback for indicators
PERM_SHUFFLES = 1000
OUTPUT_PATH = Path("/home/jupiter/Lvl3Quant/data/gap_reversal_results.json")


# ── DATA DOWNLOAD ───────────────────────────────────────────────────────────
def download_data():
    """Download OHLCV for universe + SPY."""
    tickers = UNIVERSE + ["SPY"]
    print(f"Downloading {len(tickers)} tickers from {DL_START} to {BT_END} ...")
    data = {}
    for t in tickers:
        df = yf.download(t, start=DL_START, end=BT_END, progress=False, auto_adjust=True)
        if df.empty:
            print(f"  WARNING: no data for {t}")
            continue
        # Flatten MultiIndex columns if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
        df.index = pd.to_datetime(df.index).tz_localize(None)
        data[t] = df
    print(f"  Got data for {len(data)} tickers")
    return data


# ── INDICATOR HELPERS ───────────────────────────────────────────────────────
def add_indicators(df):
    """Add RSI-14, 20-day rolling high, 20-day avg volume."""
    # RSI-14
    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["RSI"] = 100 - 100 / (1 + rs)

    df["High20"] = df["Close"].rolling(20).max()
    df["AvgVol20"] = df["Volume"].rolling(20).mean()

    # Gap = (today open - yesterday close) / yesterday close
    df["PrevClose"] = df["Close"].shift(1)
    df["GapPct"] = (df["Open"] - df["PrevClose"]) / df["PrevClose"]

    # Distance below 20-day high at prior close
    df["PctBelowHigh"] = (df["PrevClose"] - df["High20"].shift(1)) / df["High20"].shift(1)

    # Consecutive gap-down count
    df["GapDown"] = (df["GapPct"] < -0.02).astype(int)
    # rolling consecutive: use groupby trick
    groups = (df["GapDown"] != df["GapDown"].shift()).cumsum()
    df["ConsecGapDown"] = df.groupby(groups)["GapDown"].cumsum()

    return df


# ── SIGNAL GENERATORS ──────────────────────────────────────────────────────
def signals_A(df):
    """Large Gap-Down Reversal: gap < -3%, >5% below 20d high."""
    cond = (df["GapPct"] < -0.03) & (df["PctBelowHigh"] < -0.05)
    return cond


def signals_B(df):
    """Gap-Down + Green Close: gap < -2%, close > open, >5% below high."""
    cond = (
        (df["GapPct"] < -0.02)
        & (df["Close"] > df["Open"])
        & (df["PctBelowHigh"] < -0.05)
    )
    return cond


def signals_C(df):
    """Consecutive Gap-Downs: 2+ consec gap-downs, >5% below high, RSI<40."""
    cond = (
        (df["ConsecGapDown"] >= 2)
        & (df["PctBelowHigh"] < -0.05)
        & (df["RSI"] < 40)
    )
    return cond


def signals_D(df):
    """Gap-Down + Volume Spike: gap < -2%, vol > 1.5× avg, RSI<40, >5% below high."""
    cond = (
        (df["GapPct"] < -0.02)
        & (df["Volume"] > 1.5 * df["AvgVol20"])
        & (df["RSI"] < 40)
        & (df["PctBelowHigh"] < -0.05)
    )
    return cond


def signals_E(df):
    """Gap Fill Probability: gap < -3%, close stays below prior close (unfilled)."""
    cond = (
        (df["GapPct"] < -0.03)
        & (df["Close"] < df["PrevClose"])
    )
    return cond


def signals_F(df):
    """Morning Reversal: gap < -2%, low < open, close in upper half of range, >5% below high."""
    rng = df["High"] - df["Low"]
    close_pos = (df["Close"] - df["Low"]) / rng.replace(0, np.nan)
    cond = (
        (df["GapPct"] < -0.02)
        & (df["Low"] < df["Open"])
        & (close_pos > 0.5)
        & (df["PctBelowHigh"] < -0.05)
    )
    return cond


STRATEGIES = {
    "A_large_gap_reversal": signals_A,
    "B_gap_green_close": signals_B,
    "C_consec_gap_downs": signals_C,
    "D_gap_vol_spike": signals_D,
    "E_gap_fill_prob": signals_E,
    "F_morning_reversal": signals_F,
}


# ── BACKTEST ENGINE ─────────────────────────────────────────────────────────
def run_backtest(data, signal_func, entry_price_col="Open"):
    """
    Run a single strategy backtest.

    For most variants, entry is at Open on the signal day.
    For variant B and F, entry is at Close on the signal day (signal uses close info).

    Returns list of trade dicts.
    """
    trades = []
    # Collect all (date, ticker, entry_price) signals
    all_signals = []
    for ticker, df in data.items():
        if ticker == "SPY":
            continue
        mask = signal_func(df)
        sig_dates = df.index[mask & (df.index >= BT_START) & (df.index <= BT_END)]
        for d in sig_dates:
            entry_p = df.loc[d, entry_price_col]
            all_signals.append((d, ticker, entry_p, df))

    all_signals.sort(key=lambda x: x[0])

    # Simulate with position limits
    active_trades = []  # list of (exit_date, ticker)
    equity = CAPITAL
    equity_curve = []

    for sig_date, ticker, entry_price, df in all_signals:
        # Remove expired trades
        active_trades = [(ed, tk) for ed, tk in active_trades if ed > sig_date]

        if len(active_trades) >= MAX_CONCURRENT:
            continue

        # Find exit date (HOLD_DAYS trading days later)
        future_dates = df.index[df.index > sig_date]
        if len(future_dates) < HOLD_DAYS:
            continue
        exit_date = future_dates[HOLD_DAYS - 1]
        exit_price = df.loc[exit_date, "Close"]

        # Position sizing
        shares = int(MAX_PER_TRADE / entry_price)
        if shares < 1:
            continue
        cost = shares * entry_price

        # Apply slippage
        entry_slip = entry_price * (1 + SLIPPAGE_BPS / 10000)
        exit_slip = exit_price * (1 - SLIPPAGE_BPS / 10000)

        pnl = shares * (exit_slip - entry_slip)
        ret = (exit_slip - entry_slip) / entry_slip

        trades.append({
            "ticker": ticker,
            "entry_date": str(sig_date.date()),
            "exit_date": str(exit_date.date()),
            "entry_price": round(float(entry_price), 4),
            "exit_price": round(float(exit_price), 4),
            "shares": shares,
            "pnl": round(float(pnl), 2),
            "return": round(float(ret), 6),
        })

        active_trades.append((exit_date, ticker))

    return trades


# ── METRICS ─────────────────────────────────────────────────────────────────
def compute_metrics(trades):
    """Compute strategy metrics from trade list."""
    if len(trades) == 0:
        return {"n_trades": 0}

    rets = np.array([t["return"] for t in trades])
    pnls = np.array([t["pnl"] for t in trades])

    total_pnl = float(pnls.sum())
    win_rate = float((rets > 0).mean())
    avg_ret = float(rets.mean())
    std_ret = float(rets.std()) if len(rets) > 1 else 0.0001

    # Annualize: assume ~25 trades/year as rough scaling, use per-trade Sharpe × sqrt(N)
    # More precisely, compute daily equity curve
    sharpe = (avg_ret / std_ret) * np.sqrt(252 / HOLD_DAYS) if std_ret > 0 else 0.0

    # Sortino
    downside = rets[rets < 0]
    down_std = float(downside.std()) if len(downside) > 1 else 0.0001
    sortino = (avg_ret / down_std) * np.sqrt(252 / HOLD_DAYS) if down_std > 0 else 0.0

    # Profit factor
    gross_profit = float(pnls[pnls > 0].sum()) if (pnls > 0).any() else 0.0
    gross_loss = float(abs(pnls[pnls < 0].sum())) if (pnls < 0).any() else 0.0001
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    # Max drawdown on cumulative PnL
    cum_pnl = np.cumsum(pnls)
    running_max = np.maximum.accumulate(cum_pnl)
    drawdowns = cum_pnl - running_max
    max_dd = float(drawdowns.min())
    max_dd_pct = float(max_dd / CAPITAL) if CAPITAL > 0 else 0.0

    # Avg/median hold (all are HOLD_DAYS but let's compute from data)
    avg_win = float(rets[rets > 0].mean()) if (rets > 0).any() else 0.0
    avg_loss = float(rets[rets < 0].mean()) if (rets < 0).any() else 0.0

    return {
        "n_trades": len(trades),
        "total_pnl": round(total_pnl, 2),
        "total_return_pct": round(total_pnl / CAPITAL * 100, 2),
        "win_rate": round(win_rate, 4),
        "avg_return": round(avg_ret, 6),
        "sharpe": round(sharpe, 4),
        "sortino": round(sortino, 4),
        "profit_factor": round(profit_factor, 4),
        "max_drawdown_dollars": round(max_dd, 2),
        "max_drawdown_pct": round(max_dd_pct * 100, 2),
        "avg_win": round(avg_win, 6),
        "avg_loss": round(avg_loss, 6),
        "best_trade": round(float(rets.max()), 6),
        "worst_trade": round(float(rets.min()), 6),
    }


# ── 5-GATE VALIDATION ──────────────────────────────────────────────────────
def permutation_test(trades, n_shuffles=PERM_SHUFFLES):
    """Shuffle trade returns, compute fraction of shuffled Sharpes >= actual."""
    if len(trades) < 5:
        return 1.0
    rets = np.array([t["return"] for t in trades])
    actual_sharpe = rets.mean() / (rets.std() + 1e-10)
    count_better = 0
    for _ in range(n_shuffles):
        shuffled = np.random.permutation(rets)
        # Assign random signs (direction shuffle)
        signs = np.random.choice([-1, 1], size=len(rets))
        shuffled_rets = rets * signs
        shuf_sharpe = shuffled_rets.mean() / (shuffled_rets.std() + 1e-10)
        if shuf_sharpe >= actual_sharpe:
            count_better += 1
    return count_better / n_shuffles


def regime_split(trades, spy_df):
    """Split trades into bull/bear regime based on SPY 20-day return at entry."""
    if len(trades) < 5:
        return 1.0, {}, {}

    spy_ret20 = spy_df["Close"].pct_change(20)

    bull_rets, bear_rets = [], []
    for t in trades:
        entry_date = pd.Timestamp(t["entry_date"])
        if entry_date in spy_ret20.index:
            r20 = spy_ret20.loc[entry_date]
        else:
            # Find nearest prior date
            mask = spy_ret20.index <= entry_date
            if mask.any():
                r20 = spy_ret20.loc[spy_ret20.index[mask][-1]]
            else:
                continue
        if r20 >= 0:
            bull_rets.append(t["return"])
        else:
            bear_rets.append(t["return"])

    bull_sharpe = (np.mean(bull_rets) / (np.std(bull_rets) + 1e-10)) if len(bull_rets) > 2 else 0.0
    bear_sharpe = (np.mean(bear_rets) / (np.std(bear_rets) + 1e-10)) if len(bear_rets) > 2 else 0.0

    denom = max(abs(bull_sharpe), abs(bear_sharpe), 1e-10)
    regime_gap = abs(bull_sharpe - bear_sharpe) / denom

    bull_metrics = {
        "n_trades": len(bull_rets),
        "avg_return": round(float(np.mean(bull_rets)), 6) if bull_rets else 0,
        "sharpe_raw": round(float(bull_sharpe), 4),
    }
    bear_metrics = {
        "n_trades": len(bear_rets),
        "avg_return": round(float(np.mean(bear_rets)), 6) if bear_rets else 0,
        "sharpe_raw": round(float(bear_sharpe), 4),
    }

    return round(float(regime_gap), 4), bull_metrics, bear_metrics


def validate_5gate(metrics, trades, spy_df):
    """Run the 5-gate validation. Returns dict of gate results."""
    gates = {}

    # Gate 1: Sharpe > 0.5
    gates["1_sharpe_gt_0.5"] = {
        "pass": metrics["sharpe"] > 0.5,
        "value": metrics["sharpe"],
        "threshold": 0.5,
    }

    # Gate 2: Permutation test p < 0.05
    p_val = permutation_test(trades)
    gates["2_permutation_p_lt_0.05"] = {
        "pass": p_val < 0.05,
        "value": round(p_val, 4),
        "threshold": 0.05,
    }

    # Gate 3: Regime gap < 0.5
    regime_gap, bull_m, bear_m = regime_split(trades, spy_df)
    gates["3_regime_gap_lt_0.5"] = {
        "pass": regime_gap < 0.5,
        "value": regime_gap,
        "threshold": 0.5,
        "bull": bull_m,
        "bear": bear_m,
    }

    # Gate 4: Max drawdown > -50%
    gates["4_max_dd_gt_neg50pct"] = {
        "pass": metrics["max_drawdown_pct"] > -50,
        "value": metrics["max_drawdown_pct"],
        "threshold": -50,
    }

    # Gate 5: At least 20 trades
    gates["5_min_20_trades"] = {
        "pass": metrics["n_trades"] >= 20,
        "value": metrics["n_trades"],
        "threshold": 20,
    }

    gates["all_pass"] = all(g["pass"] for g in gates.values() if isinstance(g, dict) and "pass" in g)
    return gates


# ── MAIN ────────────────────────────────────────────────────────────────────
def main():
    np.random.seed(42)

    data = download_data()
    spy_df = data.get("SPY")
    if spy_df is None:
        raise RuntimeError("Failed to download SPY data")

    # Add indicators to all
    for ticker in data:
        data[ticker] = add_indicators(data[ticker])

    results = {}

    for name, sig_func in STRATEGIES.items():
        print(f"\n{'='*60}")
        print(f"Strategy: {name}")
        print(f"{'='*60}")

        # Determine entry price column
        # B and F use close-of-day info in the signal, so entry = Close
        if name.startswith("B_") or name.startswith("F_"):
            entry_col = "Close"
        else:
            entry_col = "Open"

        trades = run_backtest(data, sig_func, entry_price_col=entry_col)
        metrics = compute_metrics(trades)

        print(f"  Trades: {metrics['n_trades']}")
        if metrics["n_trades"] > 0:
            print(f"  Total PnL: ${metrics['total_pnl']:.2f} ({metrics['total_return_pct']:.1f}%)")
            print(f"  Win Rate: {metrics['win_rate']:.1%}")
            print(f"  Sharpe: {metrics['sharpe']:.2f}")
            print(f"  Sortino: {metrics['sortino']:.2f}")
            print(f"  Profit Factor: {metrics['profit_factor']:.2f}")
            print(f"  Max DD: {metrics['max_drawdown_pct']:.1f}%")

        # 5-gate validation
        if metrics["n_trades"] >= 2:
            gates = validate_5gate(metrics, trades, spy_df)
            passed = sum(1 for k, v in gates.items() if isinstance(v, dict) and v.get("pass"))
            total_gates = sum(1 for k, v in gates.items() if isinstance(v, dict) and "pass" in v)
            print(f"  Gates passed: {passed}/{total_gates}  ALL={'YES' if gates['all_pass'] else 'NO'}")
            for gname, gval in gates.items():
                if isinstance(gval, dict) and "pass" in gval:
                    status = "PASS" if gval["pass"] else "FAIL"
                    print(f"    {gname}: {status} (value={gval['value']}, threshold={gval['threshold']})")
        else:
            gates = {"skipped": "fewer than 2 trades"}

        # Top trades
        top_trades = []
        if trades:
            sorted_trades = sorted(trades, key=lambda x: x["pnl"], reverse=True)
            top_trades = sorted_trades[:5]

        results[name] = {
            "metrics": metrics,
            "gates": gates,
            "entry_type": entry_col,
            "top_5_trades": top_trades,
            "all_trades": trades,
        }

    # ── SUMMARY ─────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'Strategy':<28} {'Trades':>6} {'PnL':>10} {'WR':>6} {'Sharpe':>7} {'Sort':>7} {'PF':>7} {'Gates':>6}")
    print("-" * 85)
    for name, r in results.items():
        m = r["metrics"]
        if m["n_trades"] == 0:
            print(f"{name:<28} {'0':>6}")
            continue
        g = r["gates"]
        if isinstance(g, dict) and "all_pass" in g:
            passed = sum(1 for k, v in g.items() if isinstance(v, dict) and v.get("pass"))
            gate_str = f"{passed}/5"
        else:
            gate_str = "N/A"
        print(
            f"{name:<28} {m['n_trades']:>6} {m['total_pnl']:>10.2f} "
            f"{m['win_rate']:>5.1%} {m['sharpe']:>7.2f} {m['sortino']:>7.2f} "
            f"{m['profit_factor']:>7.2f} {gate_str:>6}"
        )

    # ── SAVE (strip all_trades for compact JSON) ────────────────────────────
    save_results = {}
    for name, r in results.items():
        save_results[name] = {
            "metrics": r["metrics"],
            "gates": r["gates"],
            "entry_type": r["entry_type"],
            "top_5_trades": r["top_5_trades"],
            "n_all_trades": len(r["all_trades"]),
        }

    output = {
        "strategy_family": "gap_down_reversal",
        "universe": UNIVERSE,
        "period": f"{BT_START} to {BT_END}",
        "capital": CAPITAL,
        "max_per_trade": MAX_PER_TRADE,
        "max_concurrent": MAX_CONCURRENT,
        "slippage_bps": SLIPPAGE_BPS,
        "hold_days": HOLD_DAYS,
        "generated": datetime.now().isoformat(),
        "variants": save_results,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
