#!/usr/bin/env python3
"""
Earnings Beat Chain Backtest
----------------------------
Tests whether stocks with consecutive quarterly earnings beats show persistent drift.

6 Variants:
  A: 2+ consecutive beats, hold 40 days
  B: 3+ consecutive beats, hold 40 days
  C: 2+ consecutive beats, hold 20 days
  D: 2+ beats + gap-up >2% on earnings day, hold 40 days
  E: 2+ beats + within 10% of 52-week high, hold 40 days
  F: 3+ beats + gap-up >2% + RSI(14) > 40, hold 40 days

5-Gate Validation per variant:
  1. Sharpe > 0.5
  2. Permutation p < 0.05 (1000 shuffles)
  3. Regime gap < 0.5
  4. MaxDD > -50%
  5. >= 20 trades

OOT: Jan 2022 - Jul 2026, walk-forward monthly
Starting capital: $645
Slippage: 0.02% each way
"""

import json
import warnings
import sys
import os
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

# ── Configuration ──────────────────────────────────────────────────────────
UNIVERSE = [
    "AAPL", "MSFT", "GOOGL", "META", "AMZN", "NVDA", "AVGO", "CRM",
    "NFLX", "AMD", "TSLA", "ADBE", "COST", "LLY", "UNH", "JPM",
    "V", "MA", "HD", "LOW",
]

START_DATE = "2020-01-01"  # need history before OOT for consecutive beat tracking
OOT_START = "2022-01-01"
OOT_END = "2026-07-31"
STARTING_CAPITAL = 645.0
SLIPPAGE_PCT = 0.0002  # 0.02% each way

PERMUTATION_ITERS = 1000
np.random.seed(42)


# ── Helper Functions ───────────────────────────────────────────────────────

def compute_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Compute RSI."""
    delta = prices.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period, min_periods=period).mean()
    avg_loss = loss.rolling(period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def compute_sma(prices: pd.Series, period: int) -> pd.Series:
    return prices.rolling(period, min_periods=period).mean()


def get_earnings_data(ticker: str) -> pd.DataFrame:
    """
    Get earnings dates and surprise info for a ticker.
    Returns DataFrame with columns: [date, beat] where beat is True/False.
    Uses earnings_dates from yfinance which has 'Surprise(%)' column.
    Falls back to gap-up proxy if surprise data unavailable.
    """
    try:
        tk = yf.Ticker(ticker)

        # Try earnings_dates first (has Surprise %)
        try:
            ed = tk.earnings_dates
            if ed is not None and len(ed) > 0:
                ed = ed.copy()
                ed.index = pd.to_datetime(ed.index).tz_localize(None)
                # Filter to past dates only
                ed = ed[ed.index <= pd.Timestamp.now()]
                ed = ed.sort_index()

                if "Surprise(%)" in ed.columns:
                    records = []
                    for dt, row in ed.iterrows():
                        surprise = row.get("Surprise(%)", None)
                        if pd.notna(surprise):
                            records.append({
                                "date": dt.normalize(),
                                "beat": float(surprise) > 0,
                                "surprise_pct": float(surprise),
                            })
                    if len(records) >= 4:
                        df = pd.DataFrame(records).drop_duplicates(subset="date").sort_values("date").reset_index(drop=True)
                        return df
        except Exception:
            pass

        # Fallback: use gap-up proxy
        # Get earnings calendar dates and check if stock gapped up >1%
        try:
            cal = tk.earnings_dates
            if cal is not None and len(cal) > 0:
                cal.index = pd.to_datetime(cal.index).tz_localize(None)
                dates = sorted(cal.index[cal.index <= pd.Timestamp.now()].normalize().unique())
            else:
                dates = []
        except Exception:
            dates = []

        if len(dates) < 4:
            # Try quarterly earnings from financials
            try:
                qe = tk.quarterly_earnings
                if qe is not None and len(qe) > 0:
                    dates = sorted(pd.to_datetime(qe.index).normalize().unique())
                else:
                    return pd.DataFrame(columns=["date", "beat", "surprise_pct"])
            except Exception:
                return pd.DataFrame(columns=["date", "beat", "surprise_pct"])

        # Get price data for gap analysis
        hist = tk.history(start=START_DATE, end=OOT_END, auto_adjust=True)
        if hist.empty:
            return pd.DataFrame(columns=["date", "beat", "surprise_pct"])
        hist.index = pd.to_datetime(hist.index).tz_localize(None).normalize()

        records = []
        for dt in dates:
            dt = pd.Timestamp(dt).normalize()
            # Find the trading day on or after the earnings date
            mask = hist.index >= dt
            if mask.sum() == 0:
                continue
            trade_day = hist.index[mask][0]
            # Get previous trading day
            prev_mask = hist.index < trade_day
            if prev_mask.sum() == 0:
                continue
            prev_day = hist.index[prev_mask][-1]

            open_price = hist.loc[trade_day, "Open"]
            prev_close = hist.loc[prev_day, "Close"]
            if prev_close > 0:
                gap_pct = (open_price - prev_close) / prev_close
                # >1% gap up = proxy for beat
                records.append({
                    "date": trade_day,
                    "beat": gap_pct > 0.01,
                    "surprise_pct": gap_pct * 100,  # approximate
                })

        if records:
            df = pd.DataFrame(records).drop_duplicates(subset="date").sort_values("date").reset_index(drop=True)
            return df

        return pd.DataFrame(columns=["date", "beat", "surprise_pct"])

    except Exception as e:
        print(f"  Warning: Could not get earnings for {ticker}: {e}")
        return pd.DataFrame(columns=["date", "beat", "surprise_pct"])


def count_consecutive_beats(earnings_df: pd.DataFrame, up_to_date: pd.Timestamp) -> int:
    """Count consecutive beats ending at or before up_to_date."""
    subset = earnings_df[earnings_df["date"] <= up_to_date].sort_values("date")
    if subset.empty:
        return 0
    count = 0
    for _, row in subset.iloc[::-1].iterrows():
        if row["beat"]:
            count += 1
        else:
            break
    return count


def get_gap_pct(price_df: pd.DataFrame, date: pd.Timestamp) -> float:
    """Get gap % on earnings day (open vs prev close)."""
    mask = price_df.index >= date
    if mask.sum() == 0:
        return 0.0
    trade_day = price_df.index[mask][0]
    prev_mask = price_df.index < trade_day
    if prev_mask.sum() == 0:
        return 0.0
    prev_day = price_df.index[prev_mask][-1]
    prev_close = price_df.loc[prev_day, "Close"]
    open_price = price_df.loc[trade_day, "Open"]
    if prev_close > 0:
        return (open_price - prev_close) / prev_close
    return 0.0


def is_near_52w_high(price_df: pd.DataFrame, date: pd.Timestamp, pct: float = 0.10) -> bool:
    """Check if price is within pct of 52-week high on date."""
    mask = price_df.index <= date
    if mask.sum() < 20:
        return False
    lookback = price_df.loc[mask].tail(252)
    high_52w = lookback["High"].max()
    current = lookback["Close"].iloc[-1]
    if high_52w > 0:
        return current >= high_52w * (1 - pct)
    return False


def get_rsi_on_date(price_df: pd.DataFrame, date: pd.Timestamp, period: int = 14) -> float:
    """Get RSI(period) on a given date."""
    mask = price_df.index <= date
    if mask.sum() < period + 5:
        return 50.0  # default
    subset = price_df.loc[mask]["Close"]
    rsi = compute_rsi(subset, period)
    if rsi.empty or pd.isna(rsi.iloc[-1]):
        return 50.0
    return float(rsi.iloc[-1])


# ── Trade Simulation ───────────────────────────────────────────────────────

def simulate_trades(trades: list, starting_capital: float = STARTING_CAPITAL) -> dict:
    """
    Given a list of trades [{entry_date, exit_date, entry_price, exit_price, ticker}],
    simulate sequential execution with slippage.
    Returns metrics dict.
    """
    if not trades:
        return {
            "n_trades": 0, "total_return_pct": 0, "sharpe": 0, "sortino": 0,
            "profit_factor": 0, "win_rate": 0, "max_dd_pct": 0,
            "avg_return_pct": 0, "median_return_pct": 0,
        }

    returns = []
    for t in trades:
        entry_p = t["entry_price"] * (1 + SLIPPAGE_PCT)  # buy slippage
        exit_p = t["exit_price"] * (1 - SLIPPAGE_PCT)    # sell slippage
        ret = (exit_p - entry_p) / entry_p
        returns.append(ret)

    returns = np.array(returns)

    # Equity curve
    equity = starting_capital
    equity_curve = [equity]
    for r in returns:
        equity *= (1 + r)
        equity_curve.append(equity)
    equity_curve = np.array(equity_curve)

    # Max drawdown
    peak = np.maximum.accumulate(equity_curve)
    dd = (equity_curve - peak) / peak
    max_dd = float(dd.min())

    # Sharpe (annualize assuming ~63 trades/year as rough estimate, or use per-trade)
    # Use per-trade Sharpe * sqrt(trades_per_year_estimate)
    avg_ret = np.mean(returns)
    std_ret = np.std(returns, ddof=1) if len(returns) > 1 else 1e-9
    # Annualize: assume average hold ~ 30 days, so ~12 trades/year per position
    trades_per_year = 252 / 30  # ~8.4
    sharpe = (avg_ret / max(std_ret, 1e-9)) * np.sqrt(trades_per_year)

    # Sortino
    downside = returns[returns < 0]
    downside_std = np.std(downside, ddof=1) if len(downside) > 1 else 1e-9
    sortino = (avg_ret / max(downside_std, 1e-9)) * np.sqrt(trades_per_year)

    # Profit factor
    gross_profit = returns[returns > 0].sum() if (returns > 0).any() else 0
    gross_loss = abs(returns[returns < 0].sum()) if (returns < 0).any() else 1e-9
    pf = gross_profit / max(gross_loss, 1e-9)

    # Win rate
    wr = (returns > 0).mean()

    total_ret = (equity_curve[-1] / equity_curve[0] - 1) * 100

    return {
        "n_trades": len(returns),
        "total_return_pct": round(float(total_ret), 2),
        "sharpe": round(float(sharpe), 3),
        "sortino": round(float(sortino), 3),
        "profit_factor": round(float(pf), 3),
        "win_rate": round(float(wr), 4),
        "max_dd_pct": round(float(max_dd * 100), 2),
        "avg_return_pct": round(float(avg_ret * 100), 3),
        "median_return_pct": round(float(np.median(returns) * 100), 3),
        "final_equity": round(float(equity_curve[-1]), 2),
    }


# ── Regime Classification ─────────────────────────────────────────────────

def classify_regime(spy_prices: pd.DataFrame, date: pd.Timestamp) -> str:
    """Bull if SPY > 200-SMA, else Bear."""
    mask = spy_prices.index <= date
    if mask.sum() < 200:
        return "bull"  # default if not enough data
    sma200 = spy_prices.loc[mask]["Close"].rolling(200).mean().iloc[-1]
    price = spy_prices.loc[mask]["Close"].iloc[-1]
    return "bull" if price > sma200 else "bear"


def regime_sharpe(trades: list, spy_prices: pd.DataFrame) -> dict:
    """Compute Sharpe for bull and bear trades separately."""
    bull_rets, bear_rets = [], []
    for t in trades:
        entry_p = t["entry_price"] * (1 + SLIPPAGE_PCT)
        exit_p = t["exit_price"] * (1 - SLIPPAGE_PCT)
        ret = (exit_p - entry_p) / entry_p
        regime = classify_regime(spy_prices, t["entry_date"])
        if regime == "bull":
            bull_rets.append(ret)
        else:
            bear_rets.append(ret)

    trades_per_year = 252 / 30

    def calc_sharpe(rets):
        if len(rets) < 2:
            return 0.0
        rets = np.array(rets)
        return float((np.mean(rets) / max(np.std(rets, ddof=1), 1e-9)) * np.sqrt(trades_per_year))

    sharpe_bull = calc_sharpe(bull_rets)
    sharpe_bear = calc_sharpe(bear_rets)

    denom = max(abs(sharpe_bull), abs(sharpe_bear), 1e-9)
    regime_gap = abs(sharpe_bull - sharpe_bear) / denom

    return {
        "sharpe_bull": round(sharpe_bull, 3),
        "sharpe_bear": round(sharpe_bear, 3),
        "regime_gap": round(regime_gap, 3),
        "n_bull": len(bull_rets),
        "n_bear": len(bear_rets),
    }


# ── Permutation Test ───────────────────────────────────────────────────────

def permutation_test(trades: list, all_price_data: dict, hold_days: int,
                     n_iter: int = PERMUTATION_ITERS) -> float:
    """
    Shuffle entry dates within each ticker's available trading days.
    Returns p-value (fraction of permutations with Sharpe >= observed).
    """
    if len(trades) < 5:
        return 1.0

    observed = simulate_trades(trades)["sharpe"]

    # Build pool of valid entry dates per ticker
    ticker_dates = {}
    for t in trades:
        tk = t["ticker"]
        if tk not in ticker_dates:
            if tk in all_price_data:
                dates = all_price_data[tk].index.tolist()
                # Only dates in OOT range
                oot_start = pd.Timestamp(OOT_START)
                oot_end = pd.Timestamp(OOT_END)
                dates = [d for d in dates if oot_start <= d <= oot_end]
                ticker_dates[tk] = dates
            else:
                ticker_dates[tk] = []

    count_ge = 0
    for _ in range(n_iter):
        shuffled_trades = []
        for t in trades:
            tk = t["ticker"]
            pool = ticker_dates.get(tk, [])
            if len(pool) < hold_days + 5:
                continue
            # Random entry
            max_idx = len(pool) - hold_days - 1
            if max_idx <= 0:
                continue
            idx = np.random.randint(0, max_idx)
            entry_date = pool[idx]
            exit_idx = min(idx + hold_days, len(pool) - 1)
            exit_date = pool[exit_idx]

            price_df = all_price_data[tk]
            if entry_date in price_df.index and exit_date in price_df.index:
                entry_p = float(price_df.loc[entry_date, "Close"])
                exit_p = float(price_df.loc[exit_date, "Close"])
                if entry_p > 0:
                    shuffled_trades.append({
                        "entry_date": entry_date,
                        "exit_date": exit_date,
                        "entry_price": entry_p,
                        "exit_price": exit_p,
                        "ticker": tk,
                    })

        if shuffled_trades:
            perm_sharpe = simulate_trades(shuffled_trades)["sharpe"]
            if perm_sharpe >= observed:
                count_ge += 1

    return count_ge / n_iter


# ── Strategy Variants ──────────────────────────────────────────────────────

def generate_signals(
    ticker: str,
    earnings_df: pd.DataFrame,
    price_df: pd.DataFrame,
    variant: str,
) -> list:
    """Generate trade signals for a given variant."""
    if earnings_df.empty or price_df.empty:
        return []

    oot_start = pd.Timestamp(OOT_START)
    oot_end = pd.Timestamp(OOT_END)

    # Filter earnings within OOT period
    oot_earnings = earnings_df[
        (earnings_df["date"] >= oot_start) & (earnings_df["date"] <= oot_end)
    ]

    trades = []
    for _, row in oot_earnings.iterrows():
        earn_date = row["date"]
        consec = count_consecutive_beats(earnings_df, earn_date)

        # Determine hold days and filters per variant
        if variant == "A":
            if consec < 2:
                continue
            hold_days = 40
        elif variant == "B":
            if consec < 3:
                continue
            hold_days = 40
        elif variant == "C":
            if consec < 2:
                continue
            hold_days = 20
        elif variant == "D":
            if consec < 2:
                continue
            gap = get_gap_pct(price_df, earn_date)
            if gap < 0.02:
                continue
            hold_days = 40
        elif variant == "E":
            if consec < 2:
                continue
            if not is_near_52w_high(price_df, earn_date, pct=0.10):
                continue
            hold_days = 40
        elif variant == "F":
            if consec < 3:
                continue
            gap = get_gap_pct(price_df, earn_date)
            if gap < 0.02:
                continue
            rsi = get_rsi_on_date(price_df, earn_date)
            if rsi <= 40:
                continue
            hold_days = 40
        else:
            continue

        # Find entry: next trading day after earnings
        mask = price_df.index > earn_date
        if mask.sum() == 0:
            continue
        entry_date = price_df.index[mask][0]

        # Find exit: hold_days trading days later
        entry_loc = price_df.index.get_loc(entry_date)
        exit_loc = min(entry_loc + hold_days, len(price_df) - 1)
        exit_date = price_df.index[exit_loc]

        if exit_date > pd.Timestamp(OOT_END):
            continue

        entry_price = float(price_df.loc[entry_date, "Open"])  # buy at open
        exit_price = float(price_df.loc[exit_date, "Close"])   # sell at close

        if entry_price <= 0:
            continue

        trades.append({
            "ticker": ticker,
            "entry_date": entry_date,
            "exit_date": exit_date,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "consecutive_beats": consec,
        })

    return trades


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("EARNINGS BEAT CHAIN BACKTEST")
    print(f"Universe: {len(UNIVERSE)} stocks | OOT: {OOT_START} to {OOT_END}")
    print(f"Starting Capital: ${STARTING_CAPITAL}")
    print("=" * 70)

    # 1. Download all price data
    print("\n[1/4] Downloading price data...")
    all_prices = {}
    for tk in UNIVERSE:
        try:
            df = yf.download(tk, start=START_DATE, end=OOT_END, progress=False, auto_adjust=True)
            if not df.empty:
                # Handle multi-level columns from yfinance
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
                all_prices[tk] = df
                print(f"  {tk}: {len(df)} days")
            else:
                print(f"  {tk}: NO DATA")
        except Exception as e:
            print(f"  {tk}: ERROR - {e}")

    # Download SPY for regime classification
    print("  Downloading SPY for regime...")
    spy = yf.download("SPY", start="2019-01-01", end=OOT_END, progress=False, auto_adjust=True)
    if isinstance(spy.columns, pd.MultiIndex):
        spy.columns = spy.columns.get_level_values(0)
    spy.index = pd.to_datetime(spy.index).tz_localize(None).normalize()

    # 2. Get earnings data
    print("\n[2/4] Fetching earnings data...")
    all_earnings = {}
    for tk in UNIVERSE:
        edf = get_earnings_data(tk)
        all_earnings[tk] = edf
        n_beats = edf["beat"].sum() if not edf.empty else 0
        print(f"  {tk}: {len(edf)} earnings reports, {n_beats} beats")

    # 3. Generate trades for each variant
    print("\n[3/4] Generating trades for each variant...")
    variants = ["A", "B", "C", "D", "E", "F"]
    variant_descriptions = {
        "A": "2+ beats, hold 40d",
        "B": "3+ beats, hold 40d",
        "C": "2+ beats, hold 20d",
        "D": "2+ beats + gap>2%, hold 40d",
        "E": "2+ beats + near 52w high, hold 40d",
        "F": "3+ beats + gap>2% + RSI>40, hold 40d",
    }

    variant_trades = {}
    for v in variants:
        all_trades = []
        for tk in UNIVERSE:
            if tk not in all_prices or all_earnings[tk].empty:
                continue
            trades = generate_signals(tk, all_earnings[tk], all_prices[tk], v)
            all_trades.extend(trades)
        # Sort by entry date
        all_trades.sort(key=lambda x: x["entry_date"])
        variant_trades[v] = all_trades
        print(f"  Variant {v} ({variant_descriptions[v]}): {len(all_trades)} trades")

    # 4. Evaluate each variant
    print("\n[4/4] Evaluating variants...")
    results = {}
    for v in variants:
        trades = variant_trades[v]
        print(f"\n{'─' * 50}")
        print(f"Variant {v}: {variant_descriptions[v]}")
        print(f"{'─' * 50}")

        # Base metrics
        metrics = simulate_trades(trades)
        print(f"  Trades: {metrics['n_trades']}")
        print(f"  Total Return: {metrics['total_return_pct']:.1f}%")
        print(f"  Sharpe: {metrics['sharpe']:.3f}")
        print(f"  Sortino: {metrics['sortino']:.3f}")
        print(f"  Profit Factor: {metrics['profit_factor']:.3f}")
        print(f"  Win Rate: {metrics['win_rate']:.1%}")
        print(f"  Max DD: {metrics['max_dd_pct']:.1f}%")
        print(f"  Final Equity: ${metrics.get('final_equity', 0):.2f}")

        # Regime analysis
        regime = regime_sharpe(trades, spy)
        print(f"  Bull Sharpe: {regime['sharpe_bull']:.3f} ({regime['n_bull']} trades)")
        print(f"  Bear Sharpe: {regime['sharpe_bear']:.3f} ({regime['n_bear']} trades)")
        print(f"  Regime Gap: {regime['regime_gap']:.3f}")

        # Permutation test
        hold_d = 40 if v != "C" else 20
        print(f"  Running permutation test ({PERMUTATION_ITERS} iterations)...", end=" ", flush=True)
        p_val = permutation_test(trades, all_prices, hold_d)
        print(f"p = {p_val:.4f}")

        # 5-Gate Validation
        gate_1 = metrics["sharpe"] > 0.5
        gate_2 = p_val < 0.05
        gate_3 = regime["regime_gap"] < 0.5
        gate_4 = metrics["max_dd_pct"] > -50
        gate_5 = metrics["n_trades"] >= 20

        gates_passed = sum([gate_1, gate_2, gate_3, gate_4, gate_5])

        print(f"\n  5-Gate Validation:")
        print(f"    G1 Sharpe > 0.5:      {'PASS' if gate_1 else 'FAIL'} ({metrics['sharpe']:.3f})")
        print(f"    G2 Perm p < 0.05:     {'PASS' if gate_2 else 'FAIL'} ({p_val:.4f})")
        print(f"    G3 Regime gap < 0.5:  {'PASS' if gate_3 else 'FAIL'} ({regime['regime_gap']:.3f})")
        print(f"    G4 MaxDD > -50%:      {'PASS' if gate_4 else 'FAIL'} ({metrics['max_dd_pct']:.1f}%)")
        print(f"    G5 Trades >= 20:      {'PASS' if gate_5 else 'FAIL'} ({metrics['n_trades']})")
        print(f"    Result: {gates_passed}/5 gates passed {'✓ VIABLE' if gates_passed == 5 else '✗ REJECTED'}")

        # Build ticker breakdown
        ticker_breakdown = {}
        for t in trades:
            tk = t["ticker"]
            if tk not in ticker_breakdown:
                ticker_breakdown[tk] = {"n_trades": 0, "returns": []}
            entry_p = t["entry_price"] * (1 + SLIPPAGE_PCT)
            exit_p = t["exit_price"] * (1 - SLIPPAGE_PCT)
            ret = (exit_p - entry_p) / entry_p
            ticker_breakdown[tk]["n_trades"] += 1
            ticker_breakdown[tk]["returns"].append(ret)
        for tk in ticker_breakdown:
            rets = ticker_breakdown[tk]["returns"]
            ticker_breakdown[tk]["avg_return_pct"] = round(float(np.mean(rets) * 100), 3)
            ticker_breakdown[tk]["win_rate"] = round(float((np.array(rets) > 0).mean()), 3)
            del ticker_breakdown[tk]["returns"]

        results[f"variant_{v}"] = {
            "description": variant_descriptions[v],
            "metrics": metrics,
            "regime": regime,
            "permutation_p_value": round(p_val, 4),
            "gates": {
                "G1_sharpe": gate_1,
                "G2_permutation": gate_2,
                "G3_regime_gap": gate_3,
                "G4_max_dd": gate_4,
                "G5_min_trades": gate_5,
                "total_passed": gates_passed,
                "viable": gates_passed == 5,
            },
            "ticker_breakdown": ticker_breakdown,
            "trade_dates": [
                {
                    "ticker": t["ticker"],
                    "entry": t["entry_date"].strftime("%Y-%m-%d"),
                    "exit": t["exit_date"].strftime("%Y-%m-%d"),
                    "return_pct": round(
                        ((t["exit_price"] * (1 - SLIPPAGE_PCT)) /
                         (t["entry_price"] * (1 + SLIPPAGE_PCT)) - 1) * 100, 3
                    ),
                    "consecutive_beats": t.get("consecutive_beats", 0),
                }
                for t in trades
            ],
        }

    # Save results
    output_path = "/home/jupiter/Lvl3Quant/data/earnings_beat_chain_results.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Convert any remaining timestamps
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")

    # Print summary table
    print("\n" + "=" * 90)
    print("SUMMARY TABLE")
    print("=" * 90)
    header = f"{'Var':<4} {'Description':<35} {'N':>4} {'Sharpe':>7} {'Sortino':>8} {'PF':>6} {'WR':>6} {'MaxDD':>7} {'TotRet':>8} {'Gates':>6} {'Viable':>7}"
    print(header)
    print("-" * 90)
    for v in variants:
        r = results[f"variant_{v}"]
        m = r["metrics"]
        g = r["gates"]
        print(
            f"  {v:<3} {r['description']:<35} {m['n_trades']:>4} "
            f"{m['sharpe']:>7.3f} {m['sortino']:>8.3f} {m['profit_factor']:>6.3f} "
            f"{m['win_rate']:>5.1%} {m['max_dd_pct']:>6.1f}% {m['total_return_pct']:>7.1f}% "
            f"{g['total_passed']:>3}/5  {'YES' if g['viable'] else 'NO':>5}"
        )
    print("=" * 90)


if __name__ == "__main__":
    main()
