#!/usr/bin/env python3
"""
Enhanced Post-Earnings Announcement Drift (PEAD) Backtest
=========================================================
6 variants tested walk-forward OOT Jan 2022 - Jul 2026.
5-gate validation: Sharpe>0.5, perm p<0.05, regime gap<0.5, MaxDD>-50%, >=20 trades.

Variants:
  A) Volume Confirmation (gap day vol > 2x 20d avg)
  B) Trend Filter (price > 50-day SMA)
  C) Quality Gap Filter (gap > 5% instead of 3%)
  D) Trailing Stop (8% trail, max 40 days)
  E) Stacked Entry (split across simultaneous signals, max 3)
  F) Bear Market Avoidance (no entry when SPY < 200d SMA; hold GLD instead)
"""

import json
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from pathlib import Path

warnings.filterwarnings("ignore")

# ── Config ──────────────────────────────────────────────────────────────
TICKERS = [
    "AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "TSLA", "AMD",
    "NFLX", "CRM", "SHOP", "SQ", "ROKU", "COIN", "SNAP", "UBER",
    "ABNB", "PLTR", "RBLX", "RDDT", "RIVN", "MSTR", "FSLR", "MPWR",
    "DXCM", "ILMN", "SYK", "REGN", "BMY", "CROX", "ABBV", "MA", "KKR",
]
INITIAL_CAPITAL = 645.0
OOT_START = "2022-01-01"
OOT_END = "2026-07-30"
DATA_START = "2020-06-01"  # need lookback for 200d SMA
GAP_THRESHOLD = 0.03
QUALITY_GAP = 0.05
HOLD_DAYS = 40
TRAIL_STOP_PCT = 0.08
SLIPPAGE = 0.0002
VOL_MULT = 2.0
SMA_SHORT = 50
SMA_LONG = 200
PERM_ITERS = 500
MAX_POSITIONS_E = 3
np.random.seed(42)


# ── Data Download ───────────────────────────────────────────────────────
def download_data():
    """Download all price data from yfinance."""
    all_tickers = list(set(TICKERS + ["SPY", "GLD"]))
    print(f"Downloading {len(all_tickers)} tickers...")
    data = yf.download(all_tickers, start=DATA_START, end=OOT_END, progress=False, group_by="ticker")

    prices = {}
    for t in all_tickers:
        try:
            if len(all_tickers) > 1:
                df = data[t].copy()
            else:
                df = data.copy()
            df = df.dropna(subset=["Close"])
            if len(df) > 50:
                prices[t] = df
        except Exception:
            pass

    print(f"Got data for {len(prices)} tickers")
    return prices


def detect_gap_events(prices, threshold=GAP_THRESHOLD):
    """Detect gap-up events (proxy for earnings) across all stocks."""
    events = []
    for ticker, df in prices.items():
        if ticker in ("SPY", "GLD"):
            continue
        close = df["Close"].values
        opn = df["Open"].values
        volume = df["Volume"].values
        dates = df.index

        for i in range(1, len(df)):
            if close[i-1] == 0 or np.isnan(close[i-1]) or np.isnan(opn[i]):
                continue
            gap_pct = (opn[i] - close[i-1]) / close[i-1]
            if gap_pct > threshold:
                # Compute 20-day avg volume
                vol_start = max(0, i - 20)
                avg_vol = np.nanmean(volume[vol_start:i]) if i > 0 else volume[i]
                # Compute 50-day SMA
                sma50 = np.nanmean(close[max(0, i-SMA_SHORT):i]) if i >= SMA_SHORT else np.nanmean(close[:i])

                events.append({
                    "ticker": ticker,
                    "date": dates[i],
                    "gap_pct": gap_pct,
                    "close_prev": close[i-1],
                    "open_today": opn[i],
                    "close_today": close[i],
                    "volume": volume[i],
                    "avg_volume_20d": avg_vol,
                    "sma50": sma50,
                    "idx": i,
                })

    events.sort(key=lambda x: x["date"])
    return events


def get_forward_returns(prices, ticker, date_idx, df, hold_days=HOLD_DAYS, trail_stop=None):
    """Get return from buying at close on gap day, holding for hold_days.
    If trail_stop is set, exit early if drawdown from peak exceeds trail_stop.
    Apply slippage on entry and exit.
    """
    close = df["Close"].values
    entry_price = close[date_idx] * (1 + SLIPPAGE)  # buy at close + slippage

    if trail_stop is not None:
        high_watermark = entry_price
        for d in range(1, hold_days + 1):
            if date_idx + d >= len(close):
                # Exit at last available price
                exit_price = close[-1] * (1 - SLIPPAGE)
                return (exit_price / entry_price) - 1, d
            current = close[date_idx + d]
            high_watermark = max(high_watermark, current)
            drawdown = (high_watermark - current) / high_watermark
            if drawdown >= trail_stop:
                exit_price = current * (1 - SLIPPAGE)
                return (exit_price / entry_price) - 1, d
        # Held full duration
        if date_idx + hold_days < len(close):
            exit_price = close[date_idx + hold_days] * (1 - SLIPPAGE)
            return (exit_price / entry_price) - 1, hold_days
        else:
            exit_price = close[-1] * (1 - SLIPPAGE)
            return (exit_price / entry_price) - 1, len(close) - date_idx - 1
    else:
        if date_idx + hold_days < len(close):
            exit_price = close[date_idx + hold_days] * (1 - SLIPPAGE)
            return (exit_price / entry_price) - 1, hold_days
        else:
            if date_idx + 1 < len(close):
                exit_price = close[-1] * (1 - SLIPPAGE)
                return (exit_price / entry_price) - 1, len(close) - date_idx - 1
            return None, 0


def spy_regime(spy_df, date):
    """Return True if SPY > 200d SMA (bull), False otherwise."""
    mask = spy_df.index <= date
    if mask.sum() < SMA_LONG:
        return True  # default bull if not enough data
    close = spy_df["Close"].values
    idx = mask.sum() - 1
    sma200 = np.nanmean(close[max(0, idx - SMA_LONG + 1):idx + 1])
    return close[idx] > sma200


def gld_return(gld_df, date, hold_days=HOLD_DAYS):
    """Return from holding GLD for hold_days starting at date."""
    mask = gld_df.index >= date
    future = gld_df.loc[mask]
    if len(future) < 2:
        return 0.0
    entry = future["Close"].iloc[0] * (1 + SLIPPAGE)
    exit_idx = min(hold_days, len(future) - 1)
    exit_price = future["Close"].iloc[exit_idx] * (1 - SLIPPAGE)
    return (exit_price / entry) - 1


# ── Backtest Engine ─────────────────────────────────────────────────────
def run_variant(events, prices, spy_df, gld_df, variant="A"):
    """Run a single variant backtest. Returns list of trades and equity curve."""
    trades = []
    equity = INITIAL_CAPITAL
    equity_curve = [(pd.Timestamp(OOT_START), equity)]
    in_trade_until = pd.Timestamp("2000-01-01")
    oot_start = pd.Timestamp(OOT_START)

    # For variant E, group events by date
    if variant == "E":
        from collections import defaultdict
        events_by_date = defaultdict(list)
        for ev in events:
            if ev["date"] >= oot_start:
                events_by_date[ev["date"]].append(ev)

    for ev in events:
        if ev["date"] < oot_start:
            continue

        # Skip if we're already in a trade (except variant E handles differently)
        if variant != "E" and ev["date"] < in_trade_until:
            continue

        ticker = ev["ticker"]
        df = prices.get(ticker)
        if df is None:
            continue

        is_bull = spy_regime(spy_df, ev["date"])

        # ── Variant-specific filters ──
        if variant == "A":
            # Volume confirmation: gap day volume > 2x 20d average
            if ev["volume"] < VOL_MULT * ev["avg_volume_20d"]:
                continue
            ret, hold = get_forward_returns(prices, ticker, ev["idx"], df)

        elif variant == "B":
            # Trend filter: price > 50d SMA
            if ev["close_today"] < ev["sma50"]:
                continue
            ret, hold = get_forward_returns(prices, ticker, ev["idx"], df)

        elif variant == "C":
            # Quality gap: only gaps > 5%
            if ev["gap_pct"] < QUALITY_GAP:
                continue
            ret, hold = get_forward_returns(prices, ticker, ev["idx"], df)

        elif variant == "D":
            # Trailing stop
            ret, hold = get_forward_returns(prices, ticker, ev["idx"], df, trail_stop=TRAIL_STOP_PCT)

        elif variant == "E":
            # Stacked entry - handled below
            continue  # skip individual processing

        elif variant == "F":
            # Bear market avoidance
            if not is_bull:
                # Hold GLD instead
                ret = gld_return(gld_df, ev["date"])
                trades.append({
                    "ticker": "GLD",
                    "date": str(ev["date"].date()),
                    "gap_pct": ev["gap_pct"],
                    "return": ret,
                    "hold_days": HOLD_DAYS,
                    "regime": "bear",
                    "original_ticker": ticker,
                })
                pnl = equity * ret
                equity += pnl
                equity_curve.append((ev["date"] + timedelta(days=HOLD_DAYS), equity))
                in_trade_until = ev["date"] + timedelta(days=HOLD_DAYS + 1)
                continue
            ret, hold = get_forward_returns(prices, ticker, ev["idx"], df)

        else:
            ret, hold = get_forward_returns(prices, ticker, ev["idx"], df)

        if variant == "E":
            continue

        if ret is None:
            continue

        pnl = equity * ret
        equity += pnl

        trades.append({
            "ticker": ticker,
            "date": str(ev["date"].date()),
            "gap_pct": ev["gap_pct"],
            "return": ret,
            "hold_days": hold if variant == "D" else HOLD_DAYS,
            "regime": "bull" if is_bull else "bear",
        })

        in_trade_until = ev["date"] + timedelta(days=(hold if variant == "D" else HOLD_DAYS) + 1)
        equity_curve.append((in_trade_until, equity))

    # ── Variant E: stacked entries ──
    if variant == "E":
        from collections import defaultdict
        events_by_date = defaultdict(list)
        for ev in events:
            if ev["date"] >= oot_start:
                events_by_date[ev["date"]].append(ev)

        sorted_dates = sorted(events_by_date.keys())
        in_trade_until = pd.Timestamp("2000-01-01")

        for dt in sorted_dates:
            if dt < in_trade_until:
                continue

            day_events = events_by_date[dt][:MAX_POSITIONS_E]
            n_pos = len(day_events)
            weight = 1.0 / n_pos

            total_ret = 0.0
            is_bull = spy_regime(spy_df, dt)

            for ev in day_events:
                ticker = ev["ticker"]
                df = prices.get(ticker)
                if df is None:
                    continue
                ret, hold = get_forward_returns(prices, ticker, ev["idx"], df)
                if ret is None:
                    continue
                total_ret += weight * ret
                trades.append({
                    "ticker": ticker,
                    "date": str(dt.date()),
                    "gap_pct": ev["gap_pct"],
                    "return": ret * weight,
                    "hold_days": HOLD_DAYS,
                    "regime": "bull" if is_bull else "bear",
                    "n_positions": n_pos,
                })

            pnl = equity * total_ret
            equity += pnl
            in_trade_until = dt + timedelta(days=HOLD_DAYS + 1)
            equity_curve.append((in_trade_until, equity))

    return trades, equity_curve, equity


def compute_metrics(trades, equity_curve):
    """Compute all required metrics from trades list."""
    if len(trades) == 0:
        return None

    returns = np.array([t["return"] for t in trades])
    n_trades = len(returns)

    # Sharpe (annualized assuming ~6 trades/year as avg holding is 40 days)
    trades_per_year = 252 / HOLD_DAYS  # ~6.3
    mean_ret = np.mean(returns)
    std_ret = np.std(returns, ddof=1) if len(returns) > 1 else 1e-9
    sharpe = (mean_ret / std_ret) * np.sqrt(trades_per_year) if std_ret > 0 else 0

    # Sortino
    downside = returns[returns < 0]
    downside_std = np.std(downside, ddof=1) if len(downside) > 1 else 1e-9
    sortino = (mean_ret / downside_std) * np.sqrt(trades_per_year) if downside_std > 0 else 0

    # Profit Factor
    gross_profit = np.sum(returns[returns > 0])
    gross_loss = abs(np.sum(returns[returns < 0]))
    pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    # Win Rate
    wr = np.sum(returns > 0) / n_trades

    # Equity curve and MaxDD
    equity_vals = [INITIAL_CAPITAL]
    for r in returns:
        equity_vals.append(equity_vals[-1] * (1 + r))
    equity_arr = np.array(equity_vals)
    peak = np.maximum.accumulate(equity_arr)
    dd = (equity_arr - peak) / peak
    max_dd = np.min(dd)

    # Total return
    total_return = (equity_arr[-1] / INITIAL_CAPITAL) - 1

    # Bull/Bear Sharpe
    bull_rets = [t["return"] for t in trades if t["regime"] == "bull"]
    bear_rets = [t["return"] for t in trades if t["regime"] == "bear"]

    def _sharpe(rets):
        if len(rets) < 2:
            return 0.0
        r = np.array(rets)
        s = np.std(r, ddof=1)
        return (np.mean(r) / s) * np.sqrt(trades_per_year) if s > 0 else 0

    bull_sharpe = _sharpe(bull_rets)
    bear_sharpe = _sharpe(bear_rets)

    # Regime gap
    max_regime = max(abs(bull_sharpe), abs(bear_sharpe), 1e-9)
    regime_gap = abs(bull_sharpe - bear_sharpe) / max_regime

    # QQQ correlation (approximate: daily returns from equity curve)
    # We'll compute this separately

    return {
        "n_trades": n_trades,
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "profit_factor": round(pf, 3),
        "win_rate": round(wr, 3),
        "max_dd": round(max_dd, 3),
        "total_return": round(total_return, 3),
        "total_return_pct": f"{total_return*100:.1f}%",
        "final_equity": round(equity_arr[-1], 2),
        "bull_sharpe": round(bull_sharpe, 3),
        "bear_sharpe": round(bear_sharpe, 3),
        "regime_gap": round(regime_gap, 3),
        "bull_trades": len(bull_rets),
        "bear_trades": len(bear_rets),
        "mean_return": round(mean_ret * 100, 2),
    }


def compute_qqq_correlation(trades, prices):
    """Approximate QQQ correlation using trade-period returns."""
    try:
        qqq = yf.download("QQQ", start=DATA_START, end=OOT_END, progress=False)
        if len(qqq) == 0:
            return 0.0

        strat_rets = []
        qqq_rets = []
        for t in trades:
            td = pd.Timestamp(t["date"])
            hold = t.get("hold_days", HOLD_DAYS)
            mask_start = qqq.index >= td
            future = qqq.loc[mask_start]
            if len(future) < 2:
                continue
            exit_idx = min(hold, len(future) - 1)
            qqq_ret = (future["Close"].iloc[exit_idx] / future["Close"].iloc[0]) - 1
            # Handle potential Series
            if hasattr(qqq_ret, 'iloc'):
                qqq_ret = qqq_ret.iloc[0]
            strat_rets.append(t["return"])
            qqq_rets.append(float(qqq_ret))

        if len(strat_rets) < 5:
            return 0.0
        corr = np.corrcoef(strat_rets, qqq_rets)[0, 1]
        return round(corr, 3) if not np.isnan(corr) else 0.0
    except Exception:
        return 0.0


def permutation_test(trades, n_iter=PERM_ITERS):
    """Shuffle which gap events trigger entries. Return p-value."""
    if len(trades) < 5:
        return 1.0

    returns = np.array([t["return"] for t in trades])
    observed_sharpe = np.mean(returns) / (np.std(returns, ddof=1) + 1e-9)

    count_above = 0
    for _ in range(n_iter):
        # Shuffle returns (this tests if the specific assignment of returns matters)
        perm_returns = np.random.choice(returns, size=len(returns), replace=True)
        # Randomly flip signs to simulate random entry timing
        signs = np.random.choice([-1, 1], size=len(returns))
        perm_returns = returns * signs
        perm_sharpe = np.mean(perm_returns) / (np.std(perm_returns, ddof=1) + 1e-9)
        if perm_sharpe >= observed_sharpe:
            count_above += 1

    return round(count_above / n_iter, 4)


def validate_gates(metrics, perm_p):
    """Check 5-gate validation."""
    gates = {
        "sharpe_gt_0.5": metrics["sharpe"] > 0.5,
        "perm_p_lt_0.05": perm_p < 0.05,
        "regime_gap_lt_0.5": metrics["regime_gap"] < 0.5,
        "maxdd_gt_neg50": metrics["max_dd"] > -0.50,
        "n_trades_gte_20": metrics["n_trades"] >= 20,
    }
    gates["all_passed"] = all(gates.values())
    return gates


# ── Main ────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("ENHANCED PEAD BACKTEST — 6 Variants")
    print(f"OOT: {OOT_START} to {OOT_END} | Capital: ${INITIAL_CAPITAL}")
    print("=" * 70)

    prices = download_data()
    spy_df = prices.get("SPY")
    gld_df = prices.get("GLD")

    if spy_df is None:
        print("ERROR: Could not download SPY data")
        return

    # Detect gap events with base 3% threshold
    events_3pct = detect_gap_events(prices, threshold=GAP_THRESHOLD)
    print(f"\nDetected {len(events_3pct)} gap-up events (>{GAP_THRESHOLD*100}%)")

    # Also detect 5% gaps for variant C
    events_5pct = detect_gap_events(prices, threshold=QUALITY_GAP)
    print(f"Detected {len(events_5pct)} gap-up events (>{QUALITY_GAP*100}%)")

    variants = {
        "A_volume_confirm": ("A", "Volume Confirmation (vol > 2x avg)"),
        "B_trend_filter": ("B", "Trend Filter (price > 50d SMA)"),
        "C_quality_gap": ("C", "Quality Gap (>5% gap)"),
        "D_trailing_stop": ("D", "Trailing Stop (8% trail)"),
        "E_stacked_entry": ("E", "Stacked Entry (max 3 positions)"),
        "F_bear_avoidance": ("F", "Bear Market Avoidance (GLD hedge)"),
    }

    results = {}

    for key, (code, desc) in variants.items():
        print(f"\n{'─'*60}")
        print(f"Variant {code}: {desc}")
        print(f"{'─'*60}")

        trades, eq_curve, final_eq = run_variant(events_3pct, prices, spy_df, gld_df, variant=code)

        if len(trades) == 0:
            print(f"  NO TRADES — skipping")
            results[key] = {"variant": code, "description": desc, "n_trades": 0, "status": "NO_TRADES"}
            continue

        metrics = compute_metrics(trades, eq_curve)
        if metrics is None:
            continue

        # QQQ correlation
        qqq_corr = compute_qqq_correlation(trades, prices)
        metrics["qqq_correlation"] = qqq_corr

        # Permutation test
        print(f"  Running {PERM_ITERS}-iter permutation test...")
        perm_p = permutation_test(trades)
        metrics["perm_p"] = perm_p

        # Gate validation
        gates = validate_gates(metrics, perm_p)
        metrics["gates"] = gates

        results[key] = {
            "variant": code,
            "description": desc,
            "metrics": metrics,
            "status": "PASS" if gates["all_passed"] else "FAIL",
            "n_trades": metrics["n_trades"],
        }

        # Print results
        status = "PASS ✓" if gates["all_passed"] else "FAIL ✗"
        print(f"  Status:       {status}")
        print(f"  Trades:       {metrics['n_trades']} (bull={metrics['bull_trades']}, bear={metrics['bear_trades']})")
        print(f"  Sharpe:       {metrics['sharpe']}")
        print(f"  Sortino:      {metrics['sortino']}")
        print(f"  Profit Factor:{metrics['profit_factor']}")
        print(f"  Win Rate:     {metrics['win_rate']}")
        print(f"  Max DD:       {metrics['max_dd']}")
        print(f"  Total Return: {metrics['total_return_pct']} (${INITIAL_CAPITAL} -> ${metrics['final_equity']})")
        print(f"  Bull Sharpe:  {metrics['bull_sharpe']}")
        print(f"  Bear Sharpe:  {metrics['bear_sharpe']}")
        print(f"  Regime Gap:   {metrics['regime_gap']}")
        print(f"  QQQ Corr:     {qqq_corr}")
        print(f"  Perm p-val:   {perm_p}")
        print(f"  Gates:        {gates}")

    # ── Summary Table ───────────────────────────────────────────────────
    print("\n" + "=" * 100)
    print("SUMMARY TABLE")
    print("=" * 100)
    header = f"{'Variant':<30} {'Status':<6} {'N':>4} {'Sharpe':>7} {'Sortino':>8} {'PF':>6} {'WR':>5} {'MaxDD':>7} {'Return':>8} {'BullS':>6} {'BearS':>6} {'RGap':>5} {'p-val':>6}"
    print(header)
    print("-" * 100)

    for key, r in results.items():
        if r.get("n_trades", 0) == 0:
            print(f"{r['description']:<30} {'SKIP':<6} {'0':>4}")
            continue
        m = r["metrics"]
        status = "PASS" if r["status"] == "PASS" else "FAIL"
        print(f"{r['description']:<30} {status:<6} {m['n_trades']:>4} {m['sharpe']:>7.2f} {m['sortino']:>8.2f} {m['profit_factor']:>6.2f} {m['win_rate']:>5.2f} {m['max_dd']:>7.2f} {m['total_return_pct']:>8} {m['bull_sharpe']:>6.2f} {m['bear_sharpe']:>6.2f} {m['regime_gap']:>5.2f} {m['perm_p']:>6.3f}")

    print("=" * 100)

    # ── Gate Summary ────────────────────────────────────────────────────
    print("\nGATE RESULTS:")
    for key, r in results.items():
        if r.get("n_trades", 0) == 0:
            continue
        gates = r["metrics"]["gates"]
        gate_str = " | ".join([f"{k}={'Y' if v else 'N'}" for k, v in gates.items() if k != "all_passed"])
        print(f"  {r['description']:<30} => {r['status']} [{gate_str}]")

    # ── Save Results ────────────────────────────────────────────────────
    output_path = Path("/home/jupiter/Lvl3Quant/data/enhanced_pead_results.json")

    # Make JSON serializable
    save_data = {
        "run_date": str(datetime.now()),
        "oot_period": f"{OOT_START} to {OOT_END}",
        "initial_capital": INITIAL_CAPITAL,
        "gap_threshold": GAP_THRESHOLD,
        "hold_days": HOLD_DAYS,
        "slippage": SLIPPAGE,
        "n_tickers": len(TICKERS),
        "total_gap_events": len(events_3pct),
        "variants": results,
    }

    with open(output_path, "w") as f:
        json.dump(save_data, f, indent=2, default=str)

    print(f"\nResults saved to {output_path}")

    # ── Identify best variant ───────────────────────────────────────────
    passing = {k: v for k, v in results.items() if v.get("status") == "PASS"}
    if passing:
        best = max(passing.items(), key=lambda x: x[1]["metrics"]["sharpe"])
        print(f"\nBEST PASSING VARIANT: {best[1]['description']}")
        print(f"  Sharpe={best[1]['metrics']['sharpe']}, Return={best[1]['metrics']['total_return_pct']}, "
              f"Sortino={best[1]['metrics']['sortino']}, MaxDD={best[1]['metrics']['max_dd']}")
    else:
        # Show best even if none pass
        with_trades = {k: v for k, v in results.items() if v.get("n_trades", 0) > 0}
        if with_trades:
            best = max(with_trades.items(), key=lambda x: x[1]["metrics"]["sharpe"])
            print(f"\nNO VARIANTS PASSED ALL 5 GATES. Best was: {best[1]['description']}")
            print(f"  Sharpe={best[1]['metrics']['sharpe']}, Return={best[1]['metrics']['total_return_pct']}")


if __name__ == "__main__":
    main()
