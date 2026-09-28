#!/usr/bin/env python3
"""
Relative Strength During Dip on Quality Stocks — Backtest
Tests whether stocks showing relative strength (or weakness) during market dips bounce better.

Variants A-F with 5-gate validation.
"""

import json
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

warnings.filterwarnings("ignore")

# ── Config ──────────────────────────────────────────────────────────────────
UNIVERSE = [
    "AAPL", "MSFT", "AVGO", "JPM", "JNJ", "PG", "KO", "PEP", "HD", "COST",
    "UNH", "LLY", "V", "MA", "ABBV", "MRK", "WMT", "AMZN", "GOOGL", "META"
]

SECTORS = {
    "Tech": ["AAPL", "MSFT", "AVGO", "AMZN", "GOOGL", "META"],
    "Healthcare": ["UNH", "LLY", "ABBV", "MRK", "JNJ"],
    "Finance": ["JPM", "V", "MA"],
    "Consumer": ["PG", "KO", "PEP", "HD", "COST", "WMT"],
}
# Reverse lookup
STOCK_SECTOR = {}
for sector, tickers in SECTORS.items():
    for t in tickers:
        STOCK_SECTOR[t] = sector

CAPITAL = 669.0
MAX_PER_TRADE = 200.0
MAX_CONCURRENT = 3
SLIPPAGE_BPS = 2
HOLD_DAYS = 10
START = "2021-06-01"  # extra lookback
END = "2026-07-31"
BACKTEST_START = "2022-01-01"

# ── Data Download ───────────────────────────────────────────────────────────
print("Downloading data...")
tickers_all = UNIVERSE + ["SPY"]
data = yf.download(tickers_all, start=START, end=END, auto_adjust=True, progress=False)

# Handle multi-level columns from yfinance
if isinstance(data.columns, pd.MultiIndex):
    close = data["Close"]
else:
    close = data

# Make sure we have all tickers
missing = [t for t in tickers_all if t not in close.columns]
if missing:
    print(f"WARNING: Missing tickers: {missing}")
    for t in missing:
        if t in UNIVERSE:
            UNIVERSE.remove(t)

close = close.ffill().dropna(how="all")
print(f"Data: {close.index[0].date()} to {close.index[-1].date()}, {len(close)} days, {len(UNIVERSE)} stocks")

# ── Precompute indicators ──────────────────────────────────────────────────
spy = close["SPY"]

# Returns
ret_10d = close.pct_change(10)
ret_20d = close.pct_change(20)
spy_ret_10d = spy.pct_change(10)
spy_ret_20d = spy.pct_change(20)

# Rolling 20-day high
high_20d = close.rolling(20).max()

# RSI (14-day)
def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

rsi = pd.DataFrame({t: compute_rsi(close[t]) for t in UNIVERSE})

# 60-day beta to SPY
def compute_rolling_beta(stock_prices, spy_prices, window=60):
    stock_ret = stock_prices.pct_change()
    spy_ret = spy_prices.pct_change()
    cov = stock_ret.rolling(window).cov(spy_ret)
    var = spy_ret.rolling(window).var()
    return cov / var

betas = pd.DataFrame({t: compute_rolling_beta(close[t], spy, 60) for t in UNIVERSE})

# 20-day rolling correlation with SPY
def compute_rolling_corr(stock_prices, spy_prices, window=20):
    return stock_prices.pct_change().rolling(window).corr(spy_prices.pct_change())

correlations = pd.DataFrame({t: compute_rolling_corr(close[t], spy, 20) for t in UNIVERSE})

# Sector average 20-day returns
sector_avg_ret_20d = pd.DataFrame(index=close.index)
for sector, tickers in SECTORS.items():
    valid = [t for t in tickers if t in UNIVERSE]
    if valid:
        sector_avg_ret_20d[sector] = ret_20d[valid].mean(axis=1)

print("Indicators computed.")

# ── Signal Generation ──────────────────────────────────────────────────────
def generate_signals(variant):
    """Generate buy signals for a given variant. Returns list of (date, ticker)."""
    signals = []
    dates = close.loc[BACKTEST_START:].index

    for date in dates:
        if date not in spy.index:
            continue

        spy_r10 = spy_ret_10d.get(date, np.nan)
        spy_r20 = spy_ret_20d.get(date, np.nan)

        for ticker in UNIVERSE:
            if date not in close.index:
                continue

            price = close.loc[date, ticker]
            if pd.isna(price) or price <= 0:
                continue

            h20 = high_20d.loc[date, ticker] if date in high_20d.index else np.nan
            if pd.isna(h20) or h20 <= 0:
                continue

            pct_below_high = (price - h20) / h20  # negative when below high
            stock_rsi = rsi.loc[date, ticker] if date in rsi.index else np.nan
            stock_r10 = ret_10d.loc[date, ticker] if date in ret_10d.index else np.nan
            stock_r20 = ret_20d.loc[date, ticker] if date in ret_20d.index else np.nan

            if variant == "A":
                # Outperforming Dip: SPY drops >3% in 10d, stock drops less, >5% below high, RSI<40
                if pd.isna(spy_r10) or pd.isna(stock_r10) or pd.isna(stock_rsi):
                    continue
                if (spy_r10 < -0.03
                    and stock_r10 > spy_r10  # stock dropped less
                    and stock_r10 < 0  # stock did drop
                    and pct_below_high < -0.05
                    and stock_rsi < 40):
                    signals.append((date, ticker))

            elif variant == "B":
                # Underperforming Dip: SPY drops >3%, stock drops MORE, >7% below high, RSI<35
                if pd.isna(spy_r10) or pd.isna(stock_r10) or pd.isna(stock_rsi):
                    continue
                if (spy_r10 < -0.03
                    and stock_r10 < spy_r10  # stock dropped more
                    and pct_below_high < -0.07
                    and stock_rsi < 35):
                    signals.append((date, ticker))

            elif variant == "C":
                # Relative Strength Index vs SPY: RS > 1.0, >5% below high, RSI<40
                if pd.isna(spy_r20) or pd.isna(stock_r20) or pd.isna(stock_rsi):
                    continue
                if spy_r20 == 0:
                    continue
                rs = stock_r20 / spy_r20 if spy_r20 != 0 else np.nan
                if pd.isna(rs):
                    continue
                # RS > 1 means stock outperformed SPY (both negative = stock fell less)
                # Need to handle sign: if both negative, stock_r20/spy_r20 > 1 means stock fell less
                if (rs > 1.0
                    and pct_below_high < -0.05
                    and stock_rsi < 40):
                    signals.append((date, ticker))

            elif variant == "D":
                # Beta-Adjusted Dip: stock drops > beta * SPY_drop, >5% below high
                beta = betas.loc[date, ticker] if date in betas.index else np.nan
                if pd.isna(beta) or pd.isna(spy_r10) or pd.isna(stock_r10):
                    continue
                if spy_r10 >= 0:
                    continue  # only during SPY drops
                expected_drop = beta * spy_r10  # expected stock drop (negative)
                if (stock_r10 < expected_drop  # dropped more than beta predicts
                    and pct_below_high < -0.05):
                    signals.append((date, ticker))

            elif variant == "E":
                # Sector Relative Strength: underperforming sector by >3%, >5% below high, RSI<40
                sector = STOCK_SECTOR.get(ticker)
                if sector is None or pd.isna(stock_r20) or pd.isna(stock_rsi):
                    continue
                sect_avg = sector_avg_ret_20d.loc[date, sector] if date in sector_avg_ret_20d.index else np.nan
                if pd.isna(sect_avg):
                    continue
                underperf = stock_r20 - sect_avg
                if (underperf < -0.03
                    and pct_below_high < -0.05
                    and stock_rsi < 40):
                    signals.append((date, ticker))

            elif variant == "F":
                # Correlation Break: corr with SPY < 0.5, >5% below high, RSI<40
                corr = correlations.loc[date, ticker] if date in correlations.index else np.nan
                if pd.isna(corr) or pd.isna(stock_rsi):
                    continue
                if (corr < 0.5
                    and pct_below_high < -0.05
                    and stock_rsi < 40):
                    signals.append((date, ticker))

    return signals

# ── Backtest Engine ─────────────────────────────────────────────────────────
def run_backtest(signals, variant_name):
    """Run backtest with position sizing, max concurrent, slippage."""
    if not signals:
        return {"variant": variant_name, "total_trades": 0, "error": "No signals"}

    trades = []
    open_positions = []  # list of (exit_date, ticker, entry_price, shares)
    equity = CAPITAL
    equity_curve = [(pd.Timestamp(BACKTEST_START), CAPITAL)]

    # Sort signals by date
    signals_sorted = sorted(signals, key=lambda x: x[0])

    for date, ticker in signals_sorted:
        # Close expired positions
        new_open = []
        for exit_date, t, entry_px, shares in open_positions:
            if date >= exit_date:
                # Position already exited
                pass
            else:
                new_open.append((exit_date, t, entry_px, shares))
        open_positions = new_open

        # Skip if max concurrent reached
        if len(open_positions) >= MAX_CONCURRENT:
            continue

        # Skip if already holding this ticker
        if any(t == ticker for _, t, _, _ in open_positions):
            continue

        # Position sizing
        price = close.loc[date, ticker]
        if pd.isna(price) or price <= 0:
            continue

        position_size = min(MAX_PER_TRADE, equity / MAX_CONCURRENT)
        if position_size < 10:
            continue

        shares = int(position_size / price)
        if shares < 1:
            continue

        # Entry with slippage
        entry_price = price * (1 + SLIPPAGE_BPS / 10000)
        cost = shares * entry_price

        # Find exit date (HOLD_DAYS trading days later)
        future_dates = close.index[close.index > date]
        if len(future_dates) < HOLD_DAYS:
            continue
        exit_date = future_dates[HOLD_DAYS - 1]

        exit_price_raw = close.loc[exit_date, ticker]
        if pd.isna(exit_price_raw):
            continue
        exit_price = exit_price_raw * (1 - SLIPPAGE_BPS / 10000)

        pnl = shares * (exit_price - entry_price)
        pnl_pct = (exit_price - entry_price) / entry_price
        equity += pnl

        trades.append({
            "entry_date": str(date.date()),
            "exit_date": str(exit_date.date()),
            "ticker": ticker,
            "shares": shares,
            "entry_price": round(float(entry_price), 2),
            "exit_price": round(float(exit_price), 2),
            "pnl": round(float(pnl), 2),
            "pnl_pct": round(float(pnl_pct), 4),
        })

        open_positions.append((exit_date, ticker, entry_price, shares))
        equity_curve.append((exit_date, equity))

    if not trades:
        return {"variant": variant_name, "total_trades": 0, "error": "No trades executed"}

    # ── Compute Metrics ─────────────────────────────────────────────────────
    pnl_series = pd.Series([t["pnl"] for t in trades])
    pnl_pct_series = pd.Series([t["pnl_pct"] for t in trades])
    entry_dates = pd.to_datetime([t["entry_date"] for t in trades])

    total_pnl = pnl_series.sum()
    total_trades = len(trades)
    win_rate = (pnl_series > 0).mean()
    avg_win = pnl_pct_series[pnl_pct_series > 0].mean() if (pnl_pct_series > 0).any() else 0
    avg_loss = pnl_pct_series[pnl_pct_series <= 0].mean() if (pnl_pct_series <= 0).any() else 0
    profit_factor = abs(pnl_series[pnl_series > 0].sum() / pnl_series[pnl_series < 0].sum()) if (pnl_series < 0).any() and pnl_series[pnl_series < 0].sum() != 0 else np.inf

    # Sharpe and Sortino (annualized from per-trade returns, ~25 trades/year)
    trades_per_year = max(1, total_trades / 4.5)  # ~4.5 years of data
    mean_ret = pnl_pct_series.mean()
    std_ret = pnl_pct_series.std()
    downside_std = pnl_pct_series[pnl_pct_series < 0].std() if (pnl_pct_series < 0).any() else 1e-9

    sharpe = (mean_ret / std_ret) * np.sqrt(trades_per_year) if std_ret > 0 else 0
    sortino = (mean_ret / downside_std) * np.sqrt(trades_per_year) if downside_std > 0 else 0

    # Max drawdown from equity curve
    eq_df = pd.DataFrame(equity_curve, columns=["date", "equity"]).set_index("date").sort_index()
    eq_df = eq_df[~eq_df.index.duplicated(keep="last")]
    running_max = eq_df["equity"].cummax()
    drawdown = (eq_df["equity"] - running_max) / running_max
    max_dd = drawdown.min()

    # ── Regime Analysis ─────────────────────────────────────────────────────
    # Classify trade entry dates by SPY regime (was SPY up or down over prior 20 days?)
    spy_regime = spy_ret_20d.reindex(entry_dates, method="ffill")
    bull_mask = spy_regime > 0
    bear_mask = spy_regime <= 0

    bull_returns = pnl_pct_series[bull_mask.values] if bull_mask.any() else pd.Series(dtype=float)
    bear_returns = pnl_pct_series[bear_mask.values] if bear_mask.any() else pd.Series(dtype=float)

    sharpe_bull = (bull_returns.mean() / bull_returns.std() * np.sqrt(len(bull_returns))) if len(bull_returns) > 1 and bull_returns.std() > 0 else 0
    sharpe_bear = (bear_returns.mean() / bear_returns.std() * np.sqrt(len(bear_returns))) if len(bear_returns) > 1 and bear_returns.std() > 0 else 0

    max_sharpe = max(abs(sharpe_bull), abs(sharpe_bear), 1e-9)
    regime_gap = abs(sharpe_bull - sharpe_bear) / max_sharpe

    # ── Permutation Test ────────────────────────────────────────────────────
    # Generate random entry returns: pick random (date, ticker) combos, compute 10-day forward return
    observed_sharpe = sharpe
    n_perms = 1000
    valid_dates = close.loc[BACKTEST_START:].index
    # Precompute 10-day forward returns for all stocks
    fwd_ret_10d = close.shift(-HOLD_DAYS) / close - 1  # forward looking
    # Subtract slippage both ways
    fwd_ret_10d_adj = fwd_ret_10d - 2 * SLIPPAGE_BPS / 10000

    rng = np.random.RandomState(42)
    perm_sharpes = []
    n_sample = total_trades
    for _ in range(n_perms):
        # Random entries: pick n_sample random (date, ticker) pairs
        rand_dates = rng.choice(len(valid_dates), size=n_sample, replace=True)
        rand_tickers = rng.choice(UNIVERSE, size=n_sample, replace=True)
        rand_returns = []
        for di, ti in zip(rand_dates, rand_tickers):
            d = valid_dates[di]
            val = fwd_ret_10d_adj.loc[d, ti] if d in fwd_ret_10d_adj.index and ti in fwd_ret_10d_adj.columns else np.nan
            if not pd.isna(val):
                rand_returns.append(val)
        if len(rand_returns) > 2:
            rand_returns = np.array(rand_returns)
            rm = rand_returns.mean()
            rs = rand_returns.std()
            tpy = max(1, len(rand_returns) / 4.5)
            if rs > 0:
                perm_sharpes.append((rm / rs) * np.sqrt(tpy))
            else:
                perm_sharpes.append(0)
        else:
            perm_sharpes.append(0)
    p_value = np.mean([ps >= observed_sharpe for ps in perm_sharpes])

    # ── 5-Gate Validation ───────────────────────────────────────────────────
    gates = {
        "sharpe_gt_0.5": sharpe > 0.5,
        "permutation_p_lt_0.05": p_value < 0.05,
        "regime_gap_lt_0.5": regime_gap < 0.5,
        "max_dd_gt_neg50": max_dd > -0.50,
        "min_20_trades": total_trades >= 20,
    }
    gates_passed = sum(gates.values())

    result = {
        "variant": variant_name,
        "total_trades": total_trades,
        "total_pnl": round(float(total_pnl), 2),
        "final_equity": round(float(equity), 2),
        "total_return_pct": round(float((equity - CAPITAL) / CAPITAL * 100), 2),
        "win_rate": round(float(win_rate), 4),
        "avg_win_pct": round(float(avg_win), 4),
        "avg_loss_pct": round(float(avg_loss), 4),
        "profit_factor": round(float(profit_factor), 4) if profit_factor != np.inf else "inf",
        "sharpe": round(float(sharpe), 4),
        "sortino": round(float(sortino), 4),
        "max_drawdown_pct": round(float(max_dd * 100), 2),
        "regime_sharpe_bull": round(float(sharpe_bull), 4),
        "regime_sharpe_bear": round(float(sharpe_bear), 4),
        "regime_gap": round(float(regime_gap), 4),
        "permutation_p_value": round(float(p_value), 4),
        "gates": gates,
        "gates_passed": f"{gates_passed}/5",
        "passed_all": gates_passed == 5,
        "bull_trades": int(bull_mask.sum()),
        "bear_trades": int(bear_mask.sum()),
        "sample_trades": trades[:5],
    }

    return result


# ── Run All Variants ────────────────────────────────────────────────────────
variants = ["A", "B", "C", "D", "E", "F"]
variant_names = {
    "A": "Outperforming Dip",
    "B": "Underperforming Dip (Contrarian)",
    "C": "Relative Strength Index vs SPY",
    "D": "Beta-Adjusted Dip",
    "E": "Sector Relative Strength",
    "F": "Correlation Break",
}

all_results = {}

for v in variants:
    name = f"{v}: {variant_names[v]}"
    print(f"\n{'='*60}")
    print(f"Running Variant {name}...")
    signals = generate_signals(v)
    print(f"  Signals generated: {len(signals)}")
    result = run_backtest(signals, name)
    all_results[v] = result

    if result.get("total_trades", 0) > 0:
        print(f"  Trades: {result['total_trades']}")
        print(f"  Total P&L: ${result['total_pnl']}")
        print(f"  Win Rate: {result['win_rate']:.1%}")
        print(f"  Sharpe: {result['sharpe']:.3f}")
        print(f"  Sortino: {result['sortino']:.3f}")
        print(f"  Max DD: {result['max_drawdown_pct']:.1f}%")
        print(f"  Regime Gap: {result['regime_gap']:.3f}")
        print(f"  Perm p-value: {result['permutation_p_value']:.3f}")
        print(f"  Gates: {result['gates_passed']} {'✓ PASS' if result['passed_all'] else '✗ FAIL'}")
    else:
        print(f"  No trades: {result.get('error', 'unknown')}")

# ── Summary ─────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("SUMMARY — Relative Strength Dip Backtest")
print(f"{'='*60}")
print(f"{'Variant':<45} {'Trades':>6} {'Sharpe':>7} {'Sortino':>8} {'WR':>6} {'PF':>7} {'MaxDD':>7} {'Gates':>6} {'Pass':>5}")
print("-" * 100)

for v in variants:
    r = all_results[v]
    if r.get("total_trades", 0) > 0:
        pf_str = f"{r['profit_factor']:.2f}" if isinstance(r['profit_factor'], float) else r['profit_factor']
        print(f"{r['variant']:<45} {r['total_trades']:>6} {r['sharpe']:>7.3f} {r['sortino']:>8.3f} {r['win_rate']:>5.1%} {pf_str:>7} {r['max_drawdown_pct']:>6.1f}% {r['gates_passed']:>6} {'YES' if r['passed_all'] else 'NO':>5}")
    else:
        print(f"{r['variant']:<45} {'N/A':>6} {'N/A':>7} {'N/A':>8} {'N/A':>6} {'N/A':>7} {'N/A':>7} {'N/A':>6} {'NO':>5}")

# ── Save Results ────────────────────────────────────────────────────────────
output = {
    "strategy": "Relative Strength During Dip on Quality Stocks",
    "run_date": datetime.now().isoformat(),
    "period": f"{BACKTEST_START} to {END}",
    "universe_size": len(UNIVERSE),
    "capital": CAPITAL,
    "max_per_trade": MAX_PER_TRADE,
    "max_concurrent": MAX_CONCURRENT,
    "slippage_bps": SLIPPAGE_BPS,
    "hold_days": HOLD_DAYS,
    "variants": all_results,
}

output_path = "/home/jupiter/Lvl3Quant/data/relative_strength_dip_results.json"
with open(output_path, "w") as f:
    json.dump(output, f, indent=2, default=str)

print(f"\nResults saved to {output_path}")
