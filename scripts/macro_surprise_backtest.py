#!/usr/bin/env python3
"""
Macro Surprise + Quality MR Backtest
=====================================
Uses macro data surprises (rate cuts, dollar weakness, credit spread tightening,
yield curve, gold-to-stocks rotation) to time quality stock mean-reversion entries.

Variants A-F as specified. 5-gate validation.
"""

import json
import warnings
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

# ─── Configuration ───────────────────────────────────────────────────────────

UNIVERSE = [
    "AAPL", "MSFT", "AVGO", "JPM", "JNJ", "PG", "KO", "PEP", "HD", "COST",
    "UNH", "LLY", "V", "MA", "ABBV", "MRK", "WMT", "AMZN", "GOOGL", "META",
]

MACRO_TICKERS = ["^TNX", "^IRX", "SHY", "TLT", "UUP", "GLD", "HYG", "LQD", "SPY"]

START = "2022-01-01"
END = "2026-07-31"

CAPITAL = 669.0
MAX_PER_TRADE = 200.0
MAX_CONCURRENT = 3
SLIPPAGE_BPS = 2  # 2 basis points
HOLD_DAYS = 10
DIP_THRESHOLD = 0.05  # 5% below rolling high
ROLLING_HIGH_WINDOW = 60  # 60-day rolling high for dip detection

OUTPUT_PATH = Path("/home/jupiter/Lvl3Quant/data/macro_surprise_results.json")


# ─── Data Download ───────────────────────────────────────────────────────────

def download_data():
    """Download all required price data."""
    all_tickers = UNIVERSE + MACRO_TICKERS
    print(f"Downloading {len(all_tickers)} tickers...")

    data = {}
    # Download in batches to avoid rate limits
    for ticker in all_tickers:
        try:
            df = yf.download(ticker, start=START, end=END, progress=False, auto_adjust=True)
            if df is not None and len(df) > 50:
                # Handle multi-level columns from yfinance
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                data[ticker] = df
                print(f"  {ticker}: {len(df)} rows")
            else:
                print(f"  {ticker}: SKIPPED (insufficient data: {len(df) if df is not None else 0} rows)")
        except Exception as e:
            print(f"  {ticker}: FAILED ({e})")

    return data


# ─── Signal Generators ───────────────────────────────────────────────────────

def compute_dip_signal(close_series: pd.Series, threshold: float = DIP_THRESHOLD) -> pd.Series:
    """Returns True where stock is >threshold below its rolling high."""
    rolling_high = close_series.rolling(ROLLING_HIGH_WINDOW, min_periods=20).max()
    pct_below = (close_series - rolling_high) / rolling_high
    return pct_below < -threshold


def variant_a_signal(data: dict) -> pd.Series:
    """Rate cut anticipation: TLT-based yield proxy drops >0.15% over 5 days."""
    # Use TLT as inverse proxy for long rates; rising TLT = falling yields
    # Use SHY for short end; if SHY rises faster than usual = short yields dropping
    # We approximate 2Y yield drop via SHY price rise (inverse relationship)
    if "SHY" in data:
        shy_close = data["SHY"]["Close"]
        # SHY rising = short yields falling = rate cut anticipation
        shy_5d_ret = shy_close.pct_change(5)
        # 0.15% rise in SHY ~ meaningful yield drop
        return shy_5d_ret > 0.0015
    elif "^IRX" in data:
        irx_close = data["^IRX"]["Close"]
        irx_5d_chg = irx_close.diff(5)
        # IRX dropping > 0.15 = rate cut anticipation
        return irx_5d_chg < -0.15
    else:
        return pd.Series(False, index=data["SPY"]["Close"].index)


def variant_b_signal(data: dict) -> pd.Series:
    """Dollar weakness: UUP drops >1.5% in 10 days."""
    if "UUP" not in data:
        return pd.Series(False, index=data["SPY"]["Close"].index)
    uup_close = data["UUP"]["Close"]
    uup_10d_ret = uup_close.pct_change(10)
    return uup_10d_ret < -0.015


def variant_c_signal(data: dict) -> pd.Series:
    """Credit spread tightening: HYG outperforms LQD over 5 days."""
    if "HYG" not in data or "LQD" not in data:
        return pd.Series(False, index=data["SPY"]["Close"].index)
    hyg_ret = data["HYG"]["Close"].pct_change(5)
    lqd_ret = data["LQD"]["Close"].pct_change(5)
    # HYG rising faster than LQD = credit spreads tightening
    return (hyg_ret - lqd_ret) > 0.005


def variant_d_signal(data: dict) -> pd.Series:
    """Yield curve steepening: TLT underperforms SHY (long yields rising vs short)
    OR ^TNX - ^IRX spread widening."""
    if "^TNX" in data and "^IRX" in data:
        tnx = data["^TNX"]["Close"]
        irx = data["^IRX"]["Close"]
        spread = tnx - irx
        spread_5d_chg = spread.diff(5)
        return spread_5d_chg > 0.1  # 10bps steepening
    elif "TLT" in data and "SHY" in data:
        # TLT/SHY ratio: if SHY rises faster (short yields drop more) = steepening
        tlt_ret = data["TLT"]["Close"].pct_change(5)
        shy_ret = data["SHY"]["Close"].pct_change(5)
        return (shy_ret - tlt_ret) > 0.005
    else:
        return pd.Series(False, index=data["SPY"]["Close"].index)


def variant_e_signal(data: dict) -> pd.Series:
    """Gold-to-stocks rotation: GLD drops AND SPY rises over 5 days."""
    if "GLD" not in data or "SPY" not in data:
        return pd.Series(False, index=data["SPY"]["Close"].index)
    gld_ret = data["GLD"]["Close"].pct_change(5)
    spy_ret = data["SPY"]["Close"].pct_change(5)
    return (gld_ret < -0.005) & (spy_ret > 0.005)


def variant_f_signal(data: dict) -> pd.Series:
    """Multi-macro composite: 2+ of A, B, C, E must fire."""
    sig_a = variant_a_signal(data).astype(int)
    sig_b = variant_b_signal(data).astype(int)
    sig_c = variant_c_signal(data).astype(int)
    sig_e = variant_e_signal(data).astype(int)

    # Align all on common index
    common_idx = sig_a.index.intersection(sig_b.index).intersection(sig_c.index).intersection(sig_e.index)
    composite = sig_a.reindex(common_idx).fillna(0) + sig_b.reindex(common_idx).fillna(0) + \
                sig_c.reindex(common_idx).fillna(0) + sig_e.reindex(common_idx).fillna(0)
    return composite >= 2


VARIANT_SIGNALS = {
    "A_rate_cut": variant_a_signal,
    "B_dollar_weak": variant_b_signal,
    "C_credit_tight": variant_c_signal,
    "D_curve_steep": variant_d_signal,
    "E_gold_rotation": variant_e_signal,
    "F_multi_composite": variant_f_signal,
}


# ─── Backtest Engine ─────────────────────────────────────────────────────────

def run_backtest(data: dict, macro_signal_func, variant_name: str) -> dict:
    """Run backtest for a single variant."""

    # Get macro signal
    macro_signal = macro_signal_func(data)
    if macro_signal.sum() == 0:
        return {"variant": variant_name, "error": "No macro signals fired", "n_trades": 0}

    # Build common trading day index from SPY
    spy_idx = data["SPY"]["Close"].index

    trades = []
    open_positions = []  # list of dicts: {ticker, entry_date, entry_price, exit_date_target, shares}
    equity_curve = []
    cash = CAPITAL

    for i, date in enumerate(spy_idx):
        # Close positions that have reached hold period
        still_open = []
        for pos in open_positions:
            if date >= pos["exit_target"]:
                # Exit
                ticker = pos["ticker"]
                if ticker in data and date in data[ticker]["Close"].index:
                    exit_price = data[ticker]["Close"].loc[date]
                else:
                    # Find nearest trading day
                    ticker_idx = data[ticker]["Close"].index
                    future_dates = ticker_idx[ticker_idx >= pos["exit_target"]]
                    if len(future_dates) > 0:
                        exit_date = future_dates[0]
                        exit_price = data[ticker]["Close"].loc[exit_date]
                    else:
                        exit_price = pos["entry_price"]  # fallback

                # Apply slippage on exit
                exit_price *= (1 - SLIPPAGE_BPS / 10000)
                pnl = (exit_price - pos["entry_price"]) * pos["shares"]
                cash += exit_price * pos["shares"]

                trades.append({
                    "ticker": pos["ticker"],
                    "entry_date": pos["entry_date"].strftime("%Y-%m-%d"),
                    "exit_date": date.strftime("%Y-%m-%d"),
                    "entry_price": round(pos["entry_price"], 2),
                    "exit_price": round(exit_price, 2),
                    "shares": pos["shares"],
                    "pnl": round(pnl, 2),
                    "return_pct": round((exit_price / pos["entry_price"] - 1) * 100, 2),
                })
            else:
                still_open.append(pos)
        open_positions = still_open

        # Check for new entries
        if date in macro_signal.index and macro_signal.loc[date]:
            for ticker in UNIVERSE:
                if ticker not in data:
                    continue
                if date not in data[ticker]["Close"].index:
                    continue
                if len(open_positions) >= MAX_CONCURRENT:
                    break
                # Check if already holding this ticker
                if any(p["ticker"] == ticker for p in open_positions):
                    continue

                close = data[ticker]["Close"]
                if date not in close.index:
                    continue

                # Check dip condition
                dip_signal = compute_dip_signal(close)
                if date not in dip_signal.index or not dip_signal.loc[date]:
                    continue

                # Size the trade
                price = close.loc[date]
                max_shares = int(min(MAX_PER_TRADE, cash) / price)
                if max_shares < 1:
                    continue

                entry_price = price * (1 + SLIPPAGE_BPS / 10000)  # slippage on entry
                cost = entry_price * max_shares
                if cost > cash:
                    max_shares = int(cash / entry_price)
                    if max_shares < 1:
                        continue
                    cost = entry_price * max_shares

                cash -= cost

                # Target exit date
                future_dates = spy_idx[spy_idx > date]
                if len(future_dates) >= HOLD_DAYS:
                    exit_target = future_dates[HOLD_DAYS - 1]
                elif len(future_dates) > 0:
                    exit_target = future_dates[-1]
                else:
                    continue

                open_positions.append({
                    "ticker": ticker,
                    "entry_date": date,
                    "entry_price": entry_price,
                    "exit_target": exit_target,
                    "shares": max_shares,
                })

        # Mark-to-market
        portfolio_value = cash
        for pos in open_positions:
            ticker = pos["ticker"]
            if date in data[ticker]["Close"].index:
                portfolio_value += data[ticker]["Close"].loc[date] * pos["shares"]
            else:
                portfolio_value += pos["entry_price"] * pos["shares"]

        equity_curve.append({"date": date.strftime("%Y-%m-%d"), "equity": round(portfolio_value, 2)})

    # Force-close remaining positions at last available price
    for pos in open_positions:
        ticker = pos["ticker"]
        last_price = data[ticker]["Close"].iloc[-1] * (1 - SLIPPAGE_BPS / 10000)
        pnl = (last_price - pos["entry_price"]) * pos["shares"]
        cash += last_price * pos["shares"]
        trades.append({
            "ticker": pos["ticker"],
            "entry_date": pos["entry_date"].strftime("%Y-%m-%d"),
            "exit_date": data[ticker]["Close"].index[-1].strftime("%Y-%m-%d"),
            "entry_price": round(pos["entry_price"], 2),
            "exit_price": round(last_price, 2),
            "shares": pos["shares"],
            "pnl": round(pnl, 2),
            "return_pct": round((last_price / pos["entry_price"] - 1) * 100, 2),
        })

    if len(trades) == 0:
        return {"variant": variant_name, "error": "No trades executed", "n_trades": 0}

    return compute_metrics(variant_name, trades, equity_curve, data)


def compute_metrics(variant_name: str, trades: list, equity_curve: list, data: dict) -> dict:
    """Compute performance metrics and run validation gates."""

    trade_returns = [t["return_pct"] / 100 for t in trades]
    n_trades = len(trades)
    total_pnl = sum(t["pnl"] for t in trades)
    win_rate = sum(1 for r in trade_returns if r > 0) / n_trades
    avg_return = np.mean(trade_returns)

    # Sharpe (annualized from per-trade returns, assuming ~25 trades/year)
    if np.std(trade_returns) > 0:
        trades_per_year = max(n_trades / 4.5, 1)  # ~4.5 years of data
        sharpe = (np.mean(trade_returns) / np.std(trade_returns)) * np.sqrt(trades_per_year)
    else:
        sharpe = 0.0

    # Sortino
    downside = [r for r in trade_returns if r < 0]
    if len(downside) > 0 and np.std(downside) > 0:
        trades_per_year = max(n_trades / 4.5, 1)
        sortino = (np.mean(trade_returns) / np.std(downside)) * np.sqrt(trades_per_year)
    else:
        sortino = float("inf") if avg_return > 0 else 0.0

    # Profit Factor
    gross_profit = sum(t["pnl"] for t in trades if t["pnl"] > 0)
    gross_loss = abs(sum(t["pnl"] for t in trades if t["pnl"] < 0))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    # Max drawdown from equity curve
    equities = [e["equity"] for e in equity_curve]
    peak = equities[0]
    max_dd = 0.0
    for eq in equities:
        if eq > peak:
            peak = eq
        dd = (eq - peak) / peak
        if dd < max_dd:
            max_dd = dd

    # Regime analysis: split by SPY returns
    spy_close = data["SPY"]["Close"]
    spy_monthly = spy_close.resample("ME").last().pct_change()

    green_returns = []
    red_returns = []
    for t in trades:
        entry_date = pd.Timestamp(t["entry_date"])
        # Find the month
        month_key = entry_date.to_period("M").to_timestamp("M")
        # Check if that month's SPY return was positive
        nearest_months = spy_monthly.index[spy_monthly.index >= month_key]
        if len(nearest_months) > 0:
            spy_ret = spy_monthly.loc[nearest_months[0]]
            if pd.notna(spy_ret):
                if spy_ret > 0:
                    green_returns.append(t["return_pct"] / 100)
                else:
                    red_returns.append(t["return_pct"] / 100)

    # Regime gap
    sharpe_green = (np.mean(green_returns) / np.std(green_returns) * np.sqrt(len(green_returns))) if len(green_returns) > 1 and np.std(green_returns) > 0 else 0
    sharpe_red = (np.mean(red_returns) / np.std(red_returns) * np.sqrt(len(red_returns))) if len(red_returns) > 1 and np.std(red_returns) > 0 else 0

    max_regime_sharpe = max(abs(sharpe_green), abs(sharpe_red))
    regime_gap = abs(sharpe_green - sharpe_red) / max_regime_sharpe if max_regime_sharpe > 0 else 0

    # Permutation test
    perm_p = permutation_test(trade_returns, n_perms=1000)

    # 5-Gate Validation
    gates = {
        "1_sharpe_gt_0.5": sharpe > 0.5,
        "2_perm_p_lt_0.05": perm_p < 0.05,
        "3_regime_gap_lt_0.5": regime_gap < 0.5,
        "4_max_dd_gt_neg50": max_dd > -0.50,
        "5_min_20_trades": n_trades >= 20,
    }
    gates_passed = sum(gates.values())

    result = {
        "variant": variant_name,
        "n_trades": n_trades,
        "total_pnl": round(total_pnl, 2),
        "total_return_pct": round((equities[-1] / CAPITAL - 1) * 100, 2),
        "win_rate": round(win_rate * 100, 1),
        "avg_return_pct": round(avg_return * 100, 2),
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3) if sortino != float("inf") else "inf",
        "profit_factor": round(profit_factor, 3) if profit_factor != float("inf") else "inf",
        "max_drawdown_pct": round(max_dd * 100, 2),
        "regime_sharpe_green": round(sharpe_green, 3),
        "regime_sharpe_red": round(sharpe_red, 3),
        "regime_gap": round(regime_gap, 3),
        "permutation_p": round(perm_p, 4),
        "gates": gates,
        "gates_passed": f"{gates_passed}/5",
        "PASS": gates_passed == 5,
        "trades_sample": trades[:5],
        "equity_start": equities[0],
        "equity_end": equities[-1],
    }

    return result


def permutation_test(returns: list, n_perms: int = 1000) -> float:
    """Shuffle entry timing 1000 times, compute p-value."""
    actual_mean = np.mean(returns)
    rng = np.random.RandomState(42)
    count_better = 0
    returns_arr = np.array(returns)

    for _ in range(n_perms):
        shuffled = rng.permutation(returns_arr)
        # Simulate: randomly assign returns to sequential "trades"
        # The null hypothesis is that the timing doesn't matter
        if np.mean(shuffled) >= actual_mean:
            count_better += 1

    return count_better / n_perms


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("MACRO SURPRISE + QUALITY MR BACKTEST")
    print(f"Period: {START} to {END}")
    print(f"Capital: ${CAPITAL}, Max/trade: ${MAX_PER_TRADE}, Max concurrent: {MAX_CONCURRENT}")
    print(f"Hold: {HOLD_DAYS} days, Dip threshold: {DIP_THRESHOLD*100}%, Slippage: {SLIPPAGE_BPS}bps")
    print("=" * 70)

    # Download data
    data = download_data()

    if "SPY" not in data:
        print("FATAL: SPY data not available. Cannot run backtest.")
        return

    print(f"\nLoaded {len(data)} tickers successfully.\n")

    # Run all variants
    results = []
    for name, signal_func in VARIANT_SIGNALS.items():
        print(f"\n{'─'*50}")
        print(f"Running variant {name}...")
        result = run_backtest(data, signal_func, name)
        results.append(result)

        if "error" in result:
            print(f"  ERROR: {result['error']}")
        else:
            print(f"  Trades: {result['n_trades']}")
            print(f"  Total PnL: ${result['total_pnl']}")
            print(f"  Total Return: {result['total_return_pct']}%")
            print(f"  Win Rate: {result['win_rate']}%")
            print(f"  Sharpe: {result['sharpe']}")
            print(f"  Sortino: {result['sortino']}")
            print(f"  PF: {result['profit_factor']}")
            print(f"  Max DD: {result['max_drawdown_pct']}%")
            print(f"  Regime Gap: {result['regime_gap']}")
            print(f"  Perm p-value: {result['permutation_p']}")
            print(f"  Gates: {result['gates_passed']} {'PASS' if result.get('PASS') else 'FAIL'}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    passing = [r for r in results if r.get("PASS")]
    print(f"\nVariants passing all 5 gates: {len(passing)}/{len(results)}")

    for r in results:
        status = "PASS" if r.get("PASS") else "FAIL"
        gates = r.get("gates_passed", "0/5")
        n = r.get("n_trades", 0)
        sharpe = r.get("sharpe", "N/A")
        pnl = r.get("total_pnl", "N/A")
        print(f"  {r['variant']:25s} | {status} ({gates}) | {n:3d} trades | Sharpe {sharpe} | PnL ${pnl}")

    # Save results
    output = {
        "metadata": {
            "strategy": "Macro Surprise + Quality MR",
            "period": f"{START} to {END}",
            "capital": CAPITAL,
            "max_per_trade": MAX_PER_TRADE,
            "max_concurrent": MAX_CONCURRENT,
            "slippage_bps": SLIPPAGE_BPS,
            "hold_days": HOLD_DAYS,
            "dip_threshold": DIP_THRESHOLD,
            "universe": UNIVERSE,
            "macro_tickers": MACRO_TICKERS,
            "run_timestamp": dt.datetime.now().isoformat(),
        },
        "variants": results,
        "summary": {
            "total_variants": len(results),
            "passing_variants": len(passing),
            "passing_names": [r["variant"] for r in passing],
        },
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nResults saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
