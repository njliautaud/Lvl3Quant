"""
Post-Earnings Drift Standalone Strategy v2
==========================================
Key difference from v1 (REJECTED): Entry is 20-30 days AFTER the crash event,
not immediately. This captures the validated PEAD signal where the drift
continues well after the initial shock.

Signal: Stock drops 12%+ on a single day with volume 2x+ normal
Entry:  20-30 calendar days after event, when MFI < 30
Execute: NEXT DAY OPEN (honest execution)
Hold:   21 calendar days
Exit:   NEXT DAY OPEN after hold expires
"""

import os
import json
import warnings
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats

warnings.filterwarnings("ignore")

# ── Configuration ──────────────────────────────────────────────────────────
CAPITAL = 100_000
MAX_POSITIONS = 10
COST_BPS_RT = 20  # 10 bps each side
DROP_THRESHOLD = -0.12  # 12%+ drop
VOLUME_MULT = 2.0  # 2x normal volume
ENTRY_WINDOW_START = 20  # calendar days after event
ENTRY_WINDOW_END = 30
MFI_THRESHOLD = 30  # Money Flow Index oversold
MFI_PERIOD = 14
HOLD_DAYS = 21  # calendar days
START_DATE = "2010-01-01"  # data fetch start (need lookback)
BACKTEST_START = "2012-01-01"
BACKTEST_END = "2026-07-22"
N_PERMUTATIONS = 100
OUTPUT_DIR = Path("/home/jupiter/Lvl3Quant/output/post_earnings_standalone_v2")

# ── S&P 500 Universe ──────────────────────────────────────────────────────
def get_sp500_tickers():
    """Get current S&P 500 tickers from Wikipedia."""
    import urllib.request
    try:
        req = urllib.request.Request(
            "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
            headers={"User-Agent": "Mozilla/5.0"}
        )
        html = urllib.request.urlopen(req).read().decode()
        tables = pd.read_html(html)
        df = tables[0]
        tickers = df["Symbol"].str.replace(".", "-", regex=False).tolist()
        date_added = df.get("Date added", pd.Series(dtype=str))
        info = {}
        for i, t in enumerate(tickers):
            da = date_added.iloc[i] if i < len(date_added) else None
            info[t] = {"date_added": str(da) if pd.notna(da) else None}
        return tickers, info
    except Exception as e:
        print(f"Wikipedia fetch failed: {e}, using fallback")
        return None, None


def compute_mfi(high, low, close, volume, period=14):
    """Compute Money Flow Index."""
    typical_price = (high + low + close) / 3.0
    money_flow = typical_price * volume

    pos_flow = pd.Series(0.0, index=close.index)
    neg_flow = pd.Series(0.0, index=close.index)

    tp_diff = typical_price.diff()
    pos_flow[tp_diff > 0] = money_flow[tp_diff > 0]
    neg_flow[tp_diff < 0] = money_flow[tp_diff < 0]

    pos_sum = pos_flow.rolling(period).sum()
    neg_sum = neg_flow.rolling(period).sum()

    # Avoid division by zero
    neg_sum = neg_sum.replace(0, 1e-10)
    mfr = pos_sum / neg_sum
    mfi = 100 - (100 / (1 + mfr))
    return mfi


def download_data(tickers):
    """Download price data for all tickers."""
    print(f"Downloading data for {len(tickers)} tickers...")
    # Download in batches to avoid timeouts
    all_data = {}
    batch_size = 50
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i+batch_size]
        print(f"  Batch {i//batch_size + 1}/{(len(tickers)-1)//batch_size + 1}: {batch[0]}..{batch[-1]}")
        try:
            data = yf.download(
                batch, start=START_DATE, end=BACKTEST_END,
                group_by="ticker", auto_adjust=True, threads=True,
                progress=False
            )
            if len(batch) == 1:
                # Single ticker returns differently
                t = batch[0]
                if not data.empty:
                    all_data[t] = data
            else:
                for t in batch:
                    try:
                        df = data[t].dropna(how="all")
                        if len(df) > 100:
                            all_data[t] = df
                    except (KeyError, Exception):
                        pass
        except Exception as e:
            print(f"  Batch failed: {e}")

    print(f"Got data for {len(all_data)} tickers")
    return all_data


def detect_crash_events(price_data):
    """Detect crash events: 12%+ single-day drop with 2x+ volume."""
    events = []
    for ticker, df in price_data.items():
        if len(df) < 50:
            continue

        close = df["Close"]
        volume = df["Volume"]
        high = df["High"]
        low = df["Low"]

        # Daily returns
        ret = close.pct_change()

        # Average volume (20-day rolling)
        avg_vol = volume.rolling(20).mean()

        # Volume ratio
        vol_ratio = volume / avg_vol

        # Find crash days
        crash_mask = (ret <= DROP_THRESHOLD) & (vol_ratio >= VOLUME_MULT)
        crash_dates = df.index[crash_mask]

        for crash_date in crash_dates:
            if crash_date < pd.Timestamp(BACKTEST_START):
                continue
            events.append({
                "ticker": ticker,
                "crash_date": crash_date,
                "return": float(ret.loc[crash_date]),
                "vol_ratio": float(vol_ratio.loc[crash_date]),
            })

    print(f"Found {len(events)} crash events across {len(set(e['ticker'] for e in events))} tickers")
    return events


def run_backtest(events, price_data, spy_data, use_mfi=True, verbose=True):
    """Run the post-earnings drift backtest."""
    trades = []
    positions = []  # Currently open positions
    portfolio_values = []  # Daily portfolio value

    # Sort events by crash date
    events = sorted(events, key=lambda x: x["crash_date"])

    # Build a combined date index from SPY
    all_dates = spy_data.index
    all_dates = all_dates[(all_dates >= pd.Timestamp(BACKTEST_START)) &
                          (all_dates <= pd.Timestamp(BACKTEST_END))]

    # Precompute MFI for all tickers
    mfi_cache = {}
    if use_mfi:
        for ticker, df in price_data.items():
            if len(df) > MFI_PERIOD + 5:
                mfi_cache[ticker] = compute_mfi(
                    df["High"], df["Low"], df["Close"], df["Volume"], MFI_PERIOD
                )

    # Track positions day by day
    active_positions = []  # list of dicts with entry info
    cash = CAPITAL
    daily_equity = []
    entry_queue = []  # events waiting for entry window

    # Process each event to find entry signals
    signal_entries = []
    for ev in events:
        ticker = ev["ticker"]
        crash_date = ev["crash_date"]

        if ticker not in price_data:
            continue
        df = price_data[ticker]

        # Entry window: 20-30 calendar days after crash
        window_start = crash_date + pd.Timedelta(days=ENTRY_WINDOW_START)
        window_end = crash_date + pd.Timedelta(days=ENTRY_WINDOW_END)

        # Get trading days in window
        window_mask = (df.index >= window_start) & (df.index <= window_end)
        window_dates = df.index[window_mask]

        if len(window_dates) == 0:
            continue

        # Check MFI condition
        entry_date = None
        if use_mfi and ticker in mfi_cache:
            mfi = mfi_cache[ticker]
            for d in window_dates:
                if d in mfi.index and mfi.loc[d] < MFI_THRESHOLD:
                    entry_date = d
                    break
        else:
            # Without MFI, take first day in window
            entry_date = window_dates[0]

        if entry_date is None:
            continue

        # NEXT DAY OPEN execution
        later_dates = df.index[df.index > entry_date]
        if len(later_dates) == 0:
            continue
        exec_date = later_dates[0]

        if exec_date not in df.index:
            continue

        entry_price = float(df.loc[exec_date, "Open"])
        if np.isnan(entry_price) or entry_price <= 0:
            continue

        # Exit: hold 21 calendar days, then NEXT DAY OPEN
        target_exit = exec_date + pd.Timedelta(days=HOLD_DAYS)
        exit_dates = df.index[df.index >= target_exit]
        if len(exit_dates) == 0:
            continue
        exit_signal_date = exit_dates[0]

        # NEXT DAY OPEN after hold expires
        exit_later = df.index[df.index > exit_signal_date]
        if len(exit_later) == 0:
            continue
        actual_exit_date = exit_later[0]

        if actual_exit_date not in df.index:
            continue

        exit_price = float(df.loc[actual_exit_date, "Open"])
        if np.isnan(exit_price) or exit_price <= 0:
            continue

        signal_entries.append({
            "ticker": ticker,
            "crash_date": crash_date,
            "signal_date": entry_date,
            "entry_date": exec_date,
            "entry_price": entry_price,
            "exit_date": actual_exit_date,
            "exit_price": exit_price,
        })

    # Sort by entry date
    signal_entries.sort(key=lambda x: x["entry_date"])

    if verbose:
        print(f"Found {len(signal_entries)} potential entries (after MFI filter)")

    # Simulate with position limits
    active = []  # currently open trades
    completed = []

    for entry in signal_entries:
        # Close any positions that should have exited by now
        still_active = []
        for pos in active:
            if entry["entry_date"] >= pos["exit_date"]:
                completed.append(pos)
            else:
                still_active.append(pos)
        active = still_active

        # Check position limit
        if len(active) >= MAX_POSITIONS:
            continue

        # Check we don't already have this ticker
        if any(p["ticker"] == entry["ticker"] for p in active):
            continue

        # Calculate position size
        pos_size = CAPITAL / MAX_POSITIONS  # Equal weight
        shares = int(pos_size / entry["entry_price"])
        if shares <= 0:
            continue

        # Cost
        cost_per_share = entry["entry_price"] * (COST_BPS_RT / 10000)
        gross_pnl = (entry["exit_price"] - entry["entry_price"]) * shares
        net_pnl = gross_pnl - cost_per_share * shares

        trade = {
            **entry,
            "shares": shares,
            "gross_pnl": gross_pnl,
            "net_pnl": net_pnl,
            "return_pct": (entry["exit_price"] / entry["entry_price"] - 1) * 100,
            "net_return_pct": net_pnl / (shares * entry["entry_price"]) * 100,
        }
        active.append(trade)

    # Close remaining
    for pos in active:
        completed.append(pos)

    if verbose:
        print(f"Completed {len(completed)} trades after position limits")

    return completed


def compute_equity_curve(trades, spy_data):
    """Build daily equity curve from trades."""
    if not trades:
        return pd.Series(dtype=float), pd.Series(dtype=float)

    # Build daily P&L attribution
    all_dates = spy_data.index
    all_dates = all_dates[(all_dates >= pd.Timestamp(BACKTEST_START)) &
                          (all_dates <= pd.Timestamp(BACKTEST_END))]

    equity = pd.Series(CAPITAL, index=all_dates)

    # For each trade, distribute P&L linearly across hold period (approximation)
    # More accurate: mark-to-market daily, but we don't need that level for strategy evaluation
    cumulative_pnl = 0.0
    trade_pnl_by_exit = {}
    for t in trades:
        exit_d = t["exit_date"]
        if exit_d not in trade_pnl_by_exit:
            trade_pnl_by_exit[exit_d] = 0
        trade_pnl_by_exit[exit_d] += t["net_pnl"]

    running = CAPITAL
    for d in all_dates:
        if d in trade_pnl_by_exit:
            running += trade_pnl_by_exit[d]
        equity[d] = running

    daily_returns = equity.pct_change().dropna()
    return equity, daily_returns


def compute_metrics(trades, equity_curve, daily_returns):
    """Compute full strategy metrics."""
    if not trades or len(daily_returns) < 10:
        return {"error": "Insufficient data"}

    net_pnls = [t["net_pnl"] for t in trades]
    returns_pct = [t["net_return_pct"] for t in trades]
    winners = [p for p in net_pnls if p > 0]
    losers = [p for p in net_pnls if p <= 0]

    total_pnl = sum(net_pnls)
    win_rate = len(winners) / len(net_pnls) if net_pnls else 0

    gross_profit = sum(winners) if winners else 0
    gross_loss = abs(sum(losers)) if losers else 1
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    # Annualized metrics from equity curve
    years = (equity_curve.index[-1] - equity_curve.index[0]).days / 365.25
    total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1
    cagr = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0

    # Sharpe (annualized from daily returns)
    if daily_returns.std() > 0:
        sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)
    else:
        sharpe = 0

    # Sortino
    downside = daily_returns[daily_returns < 0]
    if len(downside) > 0 and downside.std() > 0:
        sortino = (daily_returns.mean() / downside.std()) * np.sqrt(252)
    else:
        sortino = 0

    # Max drawdown
    running_max = equity_curve.cummax()
    drawdown = (equity_curve - running_max) / running_max
    max_dd = float(drawdown.min())

    calmar = cagr / abs(max_dd) if max_dd != 0 else 0

    # Year-by-year
    yearly = {}
    for t in trades:
        yr = str(t["entry_date"].year)
        if yr not in yearly:
            yearly[yr] = {"trades": 0, "pnl": 0, "winners": 0}
        yearly[yr]["trades"] += 1
        yearly[yr]["pnl"] += t["net_pnl"]
        if t["net_pnl"] > 0:
            yearly[yr]["winners"] += 1

    yearly_summary = {}
    profitable_years = 0
    for yr in sorted(yearly.keys()):
        y = yearly[yr]
        wr = y["winners"] / y["trades"] if y["trades"] > 0 else 0
        yearly_summary[yr] = {
            "trades": y["trades"],
            "pnl": round(y["pnl"], 2),
            "win_rate": round(wr * 100, 1),
            "profitable": y["pnl"] > 0,
        }
        if y["pnl"] > 0:
            profitable_years += 1

    pct_years_profitable = profitable_years / len(yearly) if yearly else 0

    return {
        "total_trades": len(trades),
        "total_pnl": round(total_pnl, 2),
        "cagr": round(cagr * 100, 2),
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "profit_factor": round(profit_factor, 3),
        "win_rate": round(win_rate * 100, 1),
        "max_drawdown": round(max_dd * 100, 2),
        "calmar": round(calmar, 3),
        "avg_pnl_per_trade": round(np.mean(net_pnls), 2),
        "median_pnl_per_trade": round(np.median(net_pnls), 2),
        "avg_return_pct": round(np.mean(returns_pct), 2),
        "total_return_pct": round(total_return * 100, 2),
        "years": round(years, 1),
        "pct_years_profitable": round(pct_years_profitable * 100, 1),
        "profitable_years": profitable_years,
        "total_years": len(yearly),
        "yearly": yearly_summary,
    }


def regime_test(trades, spy_data):
    """Classify trades by regime (green/red day based on SPY close-to-close)."""
    spy_ret = spy_data["Close"].pct_change()

    green_trades = []
    red_trades = []

    for t in trades:
        entry_d = t["entry_date"]
        # Find nearest SPY date
        spy_dates = spy_ret.index[spy_ret.index <= entry_d]
        if len(spy_dates) == 0:
            continue
        nearest = spy_dates[-1]
        if spy_ret.loc[nearest] >= 0:
            green_trades.append(t)
        else:
            red_trades.append(t)

    def _sharpe_from_trades(tlist):
        if len(tlist) < 5:
            return 0
        rets = [t["net_return_pct"] for t in tlist]
        if np.std(rets) == 0:
            return 0
        return float(np.mean(rets) / np.std(rets) * np.sqrt(252 / 21))  # adjust for hold period

    sharpe_green = _sharpe_from_trades(green_trades)
    sharpe_red = _sharpe_from_trades(red_trades)

    max_sharpe = max(abs(sharpe_green), abs(sharpe_red))
    regime_gap = abs(sharpe_green - sharpe_red) / max_sharpe if max_sharpe > 0 else 0

    return {
        "green_trades": len(green_trades),
        "red_trades": len(red_trades),
        "sharpe_green": round(sharpe_green, 3),
        "sharpe_red": round(sharpe_red, 3),
        "regime_gap": round(regime_gap, 3),
        "pass": regime_gap < 0.50,
    }


def permutation_test(events, price_data, spy_data, actual_pnl, n_perms=100):
    """Shuffle which tickers get selected (same dates, random tickers)."""
    print(f"Running {n_perms} permutation tests...")
    all_tickers = list(price_data.keys())
    perm_pnls = []

    for i in range(n_perms):
        if (i + 1) % 20 == 0:
            print(f"  Permutation {i+1}/{n_perms}")

        # Shuffle: for each event, randomly assign a different ticker
        shuffled_events = []
        for ev in events:
            random_ticker = np.random.choice(all_tickers)
            shuffled_events.append({
                **ev,
                "ticker": random_ticker,
            })

        trades = run_backtest(shuffled_events, price_data, spy_data,
                              use_mfi=True, verbose=False)
        total = sum(t["net_pnl"] for t in trades) if trades else 0
        perm_pnls.append(total)

    perm_pnls = np.array(perm_pnls)
    p_value = float(np.mean(perm_pnls >= actual_pnl))

    return {
        "actual_pnl": round(actual_pnl, 2),
        "perm_mean": round(float(np.mean(perm_pnls)), 2),
        "perm_std": round(float(np.std(perm_pnls)), 2),
        "perm_median": round(float(np.median(perm_pnls)), 2),
        "p_value": round(p_value, 3),
        "pass": p_value < 0.05,
    }


def main():
    print("=" * 70)
    print("POST-EARNINGS DRIFT STANDALONE v2")
    print("=" * 70)
    print()

    # 1. Get universe
    print("[1/6] Getting S&P 500 universe...")
    tickers, ticker_info = get_sp500_tickers()
    if tickers is None:
        print("FATAL: Could not get S&P 500 tickers")
        return
    print(f"  Universe: {len(tickers)} tickers")

    # 2. Download data
    print("\n[2/6] Downloading price data...")
    price_data = download_data(tickers)

    # Also get SPY for regime classification
    print("  Downloading SPY for regime test...")
    spy = yf.download("SPY", start=START_DATE, end=BACKTEST_END,
                       auto_adjust=True, progress=False)
    # Flatten multi-level columns if present
    if isinstance(spy.columns, pd.MultiIndex):
        spy.columns = spy.columns.get_level_values(0)

    # 3. Detect crash events
    print("\n[3/6] Detecting crash events...")
    events = detect_crash_events(price_data)

    # Show some stats
    event_years = {}
    for e in events:
        yr = e["crash_date"].year
        event_years[yr] = event_years.get(yr, 0) + 1
    print("  Events by year:")
    for yr in sorted(event_years.keys()):
        print(f"    {yr}: {event_years[yr]} events")

    # 4. Run backtest
    print("\n[4/6] Running backtest...")
    trades = run_backtest(events, price_data, spy, use_mfi=True, verbose=True)

    if not trades:
        print("NO TRADES GENERATED. Strategy did not find any qualifying entries.")
        results = {"error": "No trades", "events_found": len(events)}
        with open(OUTPUT_DIR / "results.json", "w") as f:
            json.dump(results, f, indent=2, default=str)
        return

    # Compute equity curve
    equity, daily_returns = compute_equity_curve(trades, spy)

    # 5. Compute metrics
    print("\n[5/6] Computing metrics...")
    metrics = compute_metrics(trades, equity, daily_returns)

    # 6. Statistical tests
    print("\n[6/6] Running statistical tests...")
    regime = regime_test(trades, spy)
    print(f"  Regime test: gap={regime['regime_gap']:.3f} ({'PASS' if regime['pass'] else 'FAIL'})")

    perm = permutation_test(events, price_data, spy,
                            metrics["total_pnl"], N_PERMUTATIONS)
    print(f"  Permutation test: p={perm['p_value']:.3f} ({'PASS' if perm['pass'] else 'FAIL'})")

    # Gate checks
    gates = {
        "G1_CAGR_gt_10": {"value": metrics["cagr"], "threshold": 10, "pass": metrics["cagr"] > 10},
        "G2_regime_gap_lt_050": {"value": regime["regime_gap"], "threshold": 0.50, "pass": regime["pass"]},
        "G3_perm_p_lt_005": {"value": perm["p_value"], "threshold": 0.05, "pass": perm["pass"]},
        "G4_maxDD_lt_35": {"value": abs(metrics["max_drawdown"]), "threshold": 35, "pass": abs(metrics["max_drawdown"]) < 35},
        "G5_75pct_years_profitable": {"value": metrics["pct_years_profitable"], "threshold": 75, "pass": metrics["pct_years_profitable"] > 75},
    }

    all_pass = all(g["pass"] for g in gates.values())

    # Compile results
    results = {
        "strategy": "Post-Earnings Drift Standalone v2",
        "version": "v2",
        "description": "Enter 20-30 days after 12%+ crash with MFI<30, hold 21 days, next-day execution",
        "run_date": str(dt.datetime.now()),
        "config": {
            "capital": CAPITAL,
            "max_positions": MAX_POSITIONS,
            "cost_bps_rt": COST_BPS_RT,
            "drop_threshold": DROP_THRESHOLD,
            "volume_multiplier": VOLUME_MULT,
            "entry_window_days": f"{ENTRY_WINDOW_START}-{ENTRY_WINDOW_END}",
            "mfi_threshold": MFI_THRESHOLD,
            "hold_days": HOLD_DAYS,
            "period": f"{BACKTEST_START} to {BACKTEST_END}",
            "universe": f"S&P 500 ({len(price_data)} tickers with data)",
        },
        "events_detected": len(events),
        "metrics": metrics,
        "regime_test": regime,
        "permutation_test": perm,
        "gates": gates,
        "all_gates_pass": all_pass,
        "sample_trades": [
            {k: str(v) if isinstance(v, (pd.Timestamp, dt.datetime)) else v
             for k, v in t.items()}
            for t in trades[:20]
        ],
    }

    # Save
    with open(OUTPUT_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Print summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"  Total events:        {len(events)}")
    print(f"  Total trades:        {metrics['total_trades']}")
    print(f"  Total P&L:           ${metrics['total_pnl']:,.2f}")
    print(f"  CAGR:                {metrics['cagr']:.1f}%")
    print(f"  Sharpe:              {metrics['sharpe']:.3f}")
    print(f"  Sortino:             {metrics['sortino']:.3f}")
    print(f"  Profit Factor:       {metrics['profit_factor']:.3f}")
    print(f"  Win Rate:            {metrics['win_rate']:.1f}%")
    print(f"  Max Drawdown:        {metrics['max_drawdown']:.1f}%")
    print(f"  Calmar:              {metrics['calmar']:.3f}")
    print(f"  Avg Return/Trade:    {metrics['avg_return_pct']:.2f}%")
    print(f"  Years Profitable:    {metrics['profitable_years']}/{metrics['total_years']} ({metrics['pct_years_profitable']:.0f}%)")
    print()
    print("  YEAR-BY-YEAR:")
    for yr, ys in sorted(metrics["yearly"].items()):
        status = "+" if ys["profitable"] else "-"
        print(f"    {yr}: {status} ${ys['pnl']:>10,.2f}  ({ys['trades']:>3} trades, {ys['win_rate']:.0f}% WR)")
    print()
    print(f"  Regime: Green Sharpe={regime['sharpe_green']:.3f}, Red Sharpe={regime['sharpe_red']:.3f}, Gap={regime['regime_gap']:.3f}")
    print(f"  Permutation: p={perm['p_value']:.3f} (actual=${perm['actual_pnl']:,.0f} vs perm mean=${perm['perm_mean']:,.0f})")
    print()
    print("  GATES:")
    for name, g in gates.items():
        status = "PASS" if g["pass"] else "FAIL"
        print(f"    {name}: {status} (value={g['value']}, threshold={g['threshold']})")
    print()
    print(f"  OVERALL: {'ALL GATES PASS' if all_pass else 'SOME GATES FAILED'}")
    print(f"  Results saved to: {OUTPUT_DIR / 'results.json'}")


if __name__ == "__main__":
    main()
