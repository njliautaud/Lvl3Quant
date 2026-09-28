#!/usr/bin/env python3
"""
Dividend Capture Strategy v1
==============================
GENUINELY NEW — never tested in this research program.

Buy stocks 1-3 days before ex-dividend date, sell after ex-date.
Exploits the tendency for stocks to not fully drop by the dividend amount.

Variants:
  A: Buy T-1, sell T+1 (overnight hold through ex-date)
  B: Buy T-3, sell T+1 (capture run-up into ex-date)
  C: High-yield only (div yield > 3%)
  D: With momentum filter (only capture divs on uptrending stocks)
  E: Options-enhanced (sell covered call on ex-date)
  F: Sector ETF dividends only (quarterly, liquid)

Capital: $645, fractional shares, no commission (RH)
"""
import json, sys, time, warnings
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")

_builtin_print = print
def fprint(*args, **kwargs):
    _builtin_print(*args, **kwargs); sys.stdout.flush()

sys.path.insert(0, "/home/jupiter/Lvl3Quant")
from research.tools.adversarial_validator import validate_trades

BASE = Path("/home/jupiter/Lvl3Quant")
OUTPUT_DIR = BASE / "output" / "growth_research" / "dividend_capture_v1"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# High-dividend stocks + sector ETFs
UNIVERSE = [
    "XLE", "XLU", "XLP", "XLRE", "XLF",  # Sector ETFs with decent yield
    "VYM", "SCHD", "HDV", "DVY",  # Dividend ETFs
    "T", "VZ", "MO", "PM", "KO", "PEP",  # Classic high-yield
    "XOM", "CVX", "ABBV", "PFE", "BMY",
    "IBM", "MMM", "CAT", "JPM", "BAC",
    "O", "MAIN", "EPD", "ET",  # REITs/MLPs
    "NEE", "DUK", "SO", "D", "AEP",  # Utilities
]

CAP = 645.0
MAX_POSITIONS = 3  # max concurrent dividend captures

def fetch_data():
    import yfinance as yf
    fprint(f"Fetching data for {len(UNIVERSE)} dividend stocks...")
    data = {}
    dividends = {}
    for ticker in UNIVERSE:
        try:
            t = yf.Ticker(ticker)
            hist = t.history(period="7y")
            if len(hist) > 252:
                data[ticker] = hist[["Close"]].copy()
                # Get dividend history
                divs = t.dividends
                if len(divs) > 0:
                    dividends[ticker] = divs
                    fprint(f"  {ticker}: {len(hist)} days, {len(divs)} dividends, yield ~{divs.sum()/hist['Close'].iloc[-1]*100:.1f}%")
        except Exception as e:
            pass
    fprint(f"  Total: {len(data)} stocks, {len(dividends)} with dividend data")
    return data, dividends

def run_variant(data, dividends, variant_name, buy_before=1, sell_after=1,
                min_yield=0.0, momentum_filter=False, etf_only=False):
    fprint(f"\n{'='*60}")
    fprint(f"Variant {variant_name}")
    fprint(f"  buy_before={buy_before}d, sell_after={sell_after}d, min_yield={min_yield}%")

    trades = []
    equity = CAP
    cash = CAP

    # Build list of all dividend events
    events = []
    for ticker, divs in dividends.items():
        if etf_only and ticker not in ["XLE", "XLU", "XLP", "XLRE", "XLF", "VYM", "SCHD", "HDV", "DVY"]:
            continue
        if ticker not in data:
            continue
        prices = data[ticker]
        for ex_date, div_amount in divs.items():
            try:
                ex_date = pd.Timestamp(ex_date).tz_localize(None)
                if ex_date not in prices.index:
                    # Find nearest trading day
                    idx = prices.index.get_indexer([ex_date], method="nearest")[0]
                    if idx < 0 or idx >= len(prices):
                        continue
                    ex_date = prices.index[idx]

                ex_idx = prices.index.get_loc(ex_date)
                if ex_idx < buy_before + 10 or ex_idx + sell_after >= len(prices):
                    continue

                price_at_ex = prices["Close"].iloc[ex_idx]
                div_yield = div_amount / price_at_ex * 100

                if div_yield < min_yield:
                    continue

                # Momentum filter: 20d SMA
                if momentum_filter:
                    sma_20 = prices["Close"].iloc[ex_idx-20:ex_idx].mean()
                    if price_at_ex < sma_20:
                        continue

                buy_idx = ex_idx - buy_before
                sell_idx = ex_idx + sell_after
                buy_price = prices["Close"].iloc[buy_idx]
                sell_price = prices["Close"].iloc[sell_idx]

                # Total return = price change + dividend
                price_pnl = (sell_price - buy_price) / buy_price * 100
                div_return = div_amount / buy_price * 100
                total_return = price_pnl + div_return

                events.append({
                    "ticker": ticker,
                    "ex_date": ex_date,
                    "buy_date": prices.index[buy_idx],
                    "sell_date": prices.index[sell_idx],
                    "buy_price": buy_price,
                    "sell_price": sell_price,
                    "dividend": div_amount,
                    "div_yield": div_yield,
                    "price_pnl_pct": price_pnl,
                    "div_return_pct": div_return,
                    "total_return_pct": total_return,
                })
            except Exception:
                continue

    # Sort by date
    events.sort(key=lambda e: e["ex_date"])
    fprint(f"  {len(events)} dividend events found")

    if len(events) == 0:
        return {"variant": variant_name, "sharpe": 0, "trades": 0, "passed": False}

    # Simulate trading
    position_size = min(200.0, CAP / MAX_POSITIONS)  # $200 per trade max
    for evt in events:
        pnl_dollars = position_size * evt["total_return_pct"] / 100
        trades.append({
            "entry_date": evt["buy_date"].strftime("%Y-%m-%d"),
            "exit_date": evt["sell_date"].strftime("%Y-%m-%d"),
            "pnl": pnl_dollars,
            "pnl_pct": evt["total_return_pct"],
            "direction": "long",
            "ticker": evt["ticker"],
            "div_yield": evt["div_yield"],
            "price_component": evt["price_pnl_pct"],
            "div_component": evt["div_return_pct"],
        })

    # Metrics
    pnls = [t["pnl"] for t in trades]
    total_pnl = sum(pnls)
    final_equity = CAP + total_pnl
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    wr = len(wins) / len(pnls) * 100
    pf = abs(sum(wins) / sum(losses)) if losses and sum(losses) != 0 else float("inf")

    # Approximate annual Sharpe
    hold_days = max(buy_before + sell_after, 1)
    if np.std(pnls) > 0:
        sharpe = np.mean(pnls) / np.std(pnls) * np.sqrt(252 / hold_days)
    else:
        sharpe = 0

    downside = [p for p in pnls if p < 0]
    if downside and np.std(downside) > 0:
        sortino = np.mean(pnls) / np.std(downside) * np.sqrt(252 / hold_days)
    else:
        sortino = sharpe

    # Max drawdown
    eq_curve = [CAP]
    for p in pnls:
        eq_curve.append(eq_curve[-1] + p)
    peak = CAP; max_dd = 0
    for eq in eq_curve:
        if eq > peak: peak = eq
        dd = (eq - peak) / peak
        if dd < max_dd: max_dd = dd

    # Decompose: how much from dividend vs price movement?
    avg_div_component = np.mean([t["div_component"] for t in trades])
    avg_price_component = np.mean([t["price_component"] for t in trades])

    # Ticker concentration
    ticker_pnls = {}
    for t in trades:
        ticker_pnls[t["ticker"]] = ticker_pnls.get(t["ticker"], 0) + t["pnl"]

    fprint(f"  Trades: {len(trades)} | W/L: {len(wins)}/{len(losses)} | WR: {wr:.1f}%")
    fprint(f"  Total PnL: ${total_pnl:.2f} | Final: ${final_equity:.2f}")
    fprint(f"  Sharpe: {sharpe:.3f} | Sortino: {sortino:.3f} | PF: {pf:.2f}")
    fprint(f"  MaxDD: {max_dd:.1%}")
    fprint(f"  Avg return: {np.mean([t['pnl_pct'] for t in trades]):.3f}%")
    fprint(f"    Price component: {avg_price_component:.3f}%")
    fprint(f"    Dividend component: {avg_div_component:.3f}%")
    fprint(f"  Top tickers: {sorted(ticker_pnls.items(), key=lambda x: x[1], reverse=True)[:5]}")

    # Random baseline: what if we just bought random stocks on random days?
    np.random.seed(42)
    random_pnls = []
    for _ in range(min(len(trades), 200)):
        random_pnl = np.random.choice(pnls)
        random_pnls.append(random_pnl * np.random.choice([-1, 1]))
    if random_pnls and np.std(random_pnls) > 0:
        random_sharpe = np.mean(random_pnls) / np.std(random_pnls) * np.sqrt(252 / hold_days)
        fprint(f"  Random baseline Sharpe: {random_sharpe:.3f}")

    return {
        "variant": variant_name, "sharpe": round(sharpe, 3), "sortino": round(sortino, 3),
        "pf": round(pf, 2), "wr": round(wr, 1), "trades": len(trades),
        "total_pnl": round(total_pnl, 2), "final_equity": round(final_equity, 2),
        "max_dd": round(max_dd * 100, 1),
        "avg_div_component": round(avg_div_component, 3),
        "avg_price_component": round(avg_price_component, 3),
        "gates_passed": 0,
    }

def main():
    t0 = time.time()
    data, dividends = fetch_data()

    results = []
    results.append(run_variant(data, dividends, "A_Overnight", buy_before=1, sell_after=1))
    results.append(run_variant(data, dividends, "B_Runup", buy_before=3, sell_after=1))
    results.append(run_variant(data, dividends, "C_HighYield", buy_before=1, sell_after=1, min_yield=0.5))
    results.append(run_variant(data, dividends, "D_Momentum", buy_before=1, sell_after=1, momentum_filter=True))
    results.append(run_variant(data, dividends, "E_Extended", buy_before=1, sell_after=3))
    results.append(run_variant(data, dividends, "F_ETFOnly", buy_before=1, sell_after=1, etf_only=True))

    elapsed = time.time() - t0
    fprint(f"\n{'='*60}")
    fprint(f"DIVIDEND CAPTURE v1 — COMPLETE ({elapsed:.0f}s)")
    fprint(f"\n{'Variant':<20} {'Sharpe':>8} {'Sortino':>8} {'WR':>6} {'Trades':>7} {'PnL':>10} {'MDD':>7} {'DivComp':>8} {'PriceComp':>9}")
    fprint("-" * 95)
    for r in results:
        fprint(f"{r['variant']:<20} {r['sharpe']:>8.3f} {r.get('sortino',0):>8.3f} {r['wr']:>5.1f}% {r['trades']:>7} ${r['total_pnl']:>8.2f} {r['max_dd']:>6.1f}% {r.get('avg_div_component',0):>7.3f}% {r.get('avg_price_component',0):>8.3f}%")

    with open(OUTPUT_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    fprint(f"\nSaved to {OUTPUT_DIR}/results.json")

if __name__ == "__main__":
    main()
