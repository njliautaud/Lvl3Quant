#!/usr/bin/env python3
"""
Adversarial Validation for Dividend Capture Strategy - Variant F (Sector-Diversified)
Original: Sharpe 2.98, WR 65.3%, MaxDD -2.82%, 124 trades, perm p=0.0, regime gap 0.343.
Strategy: Buy high-yield stocks 1-2 days before ex-div, hold 5-7 days, always 3 sectors.

6 adversarial tests to determine if the edge is real or spurious.
"""

import json
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from pathlib import Path
from scipy import stats

warnings.filterwarnings('ignore')
np.random.seed(42)

# ── Config ──────────────────────────────────────────────────────────────
TICKERS = ["IBM", "MO", "ABBV", "KO", "PG", "PEP", "VZ", "XOM", "DOW", "INTC", "T", "CVX", "MMM", "PM"]
ACCOUNT = 645.0
HOLD_DAYS = 6  # midpoint of 5-7
SLIPPAGE_RT = 0.0005  # 0.05% round-trip baseline
OOT_START = "2022-01-01"
OOT_END = "2026-07-28"
MAX_POSITIONS = 3
N_PERMUTATIONS = 1000
ORIGINAL_SHARPE = 2.98

SECTOR_MAP = {
    "IBM": "Tech", "INTC": "Tech",
    "MO": "Staples", "KO": "Staples", "PG": "Staples", "PEP": "Staples", "PM": "Staples",
    "ABBV": "Healthcare",
    "VZ": "Telecom", "T": "Telecom",
    "XOM": "Energy", "CVX": "Energy", "DOW": "Materials",
    "MMM": "Industrial",
}

OUTPUT_PATH = Path("/home/jupiter/Lvl3Quant/data/dividend_capture_adversarial_results.json")


def download_data():
    """Download price + dividend data for all tickers."""
    print("Downloading data...")
    data = {}
    dl_start = "2021-06-01"  # buffer before OOT
    dl_end = OOT_END

    for ticker in TICKERS:
        try:
            tk = yf.Ticker(ticker)
            hist = tk.history(start=dl_start, end=dl_end, auto_adjust=False)
            if hist.empty:
                print(f"  WARN: {ticker} returned no data, skipping")
                continue
            divs = tk.dividends
            if divs is None or len(divs) == 0:
                print(f"  WARN: {ticker} has no dividends, skipping")
                continue
            # Make timezone-naive
            hist.index = hist.index.tz_localize(None)
            if divs.index.tz is not None:
                divs.index = divs.index.tz_localize(None)
            data[ticker] = {"prices": hist, "dividends": divs}
            n_divs = len(divs[(divs.index >= OOT_START) & (divs.index <= OOT_END)])
            print(f"  {ticker}: {len(hist)} days, {n_divs} dividends in OOT")
        except Exception as e:
            print(f"  ERROR downloading {ticker}: {e}")

    # Also download SPY for regime classification
    spy = yf.download("SPY", start=dl_start, end=dl_end, auto_adjust=False)
    if spy.index.tz is not None:
        spy.index = spy.index.tz_localize(None)
    data["_SPY"] = spy

    return data


def get_exdiv_dates(data, start, end):
    """Get all ex-dividend dates in OOT period, grouped by date."""
    exdiv_events = []
    for ticker, d in data.items():
        if ticker.startswith("_"):
            continue
        divs = d["dividends"]
        mask = (divs.index >= start) & (divs.index <= end)
        for dt, amount in divs[mask].items():
            exdiv_events.append({"ticker": ticker, "exdate": dt, "amount": amount})
    return sorted(exdiv_events, key=lambda x: x["exdate"])


def compute_sharpe(returns):
    """Annualized Sharpe from daily returns."""
    if len(returns) < 2 or returns.std() == 0:
        return 0.0
    return returns.mean() / returns.std() * np.sqrt(252)


def simulate_strategy(data, events, account, hold_days, slippage_rt, direction=1, include_dividends=True):
    """
    Simulate the dividend capture strategy.
    direction=1: buy before ex-date (normal)
    direction=-1: inverse (sell before, buy after)
    include_dividends: whether to include dividend income in PnL
    """
    oot_start = pd.Timestamp(OOT_START)
    oot_end = pd.Timestamp(OOT_END)

    equity = account
    daily_equity = {}
    trades = []
    positions = {}  # ticker -> {entry_price, entry_date, shares, div_amount}

    # Build a combined trading calendar from all tickers
    all_dates = set()
    for ticker, d in data.items():
        if ticker.startswith("_"):
            continue
        all_dates.update(d["prices"].index)
    trading_days = sorted([d for d in all_dates if oot_start <= d <= oot_end])

    for day in trading_days:
        # Check for exits (held >= hold_days trading days)
        to_close = []
        for ticker, pos in list(positions.items()):
            days_held = len([d for d in trading_days if pos["entry_date"] < d <= day])
            if days_held >= hold_days:
                to_close.append(ticker)

        for ticker in to_close:
            pos = positions.pop(ticker)
            if ticker not in data or data[ticker]["prices"].index[-1] < day:
                continue
            prices = data[ticker]["prices"]
            if day in prices.index:
                exit_price = float(prices.loc[day, "Close"])
            else:
                # Find nearest prior trading day
                prior = prices.index[prices.index <= day]
                if len(prior) == 0:
                    continue
                exit_price = float(prices.loc[prior[-1], "Close"])

            cost = exit_price * pos["shares"] * (slippage_rt / 2)

            if direction == 1:
                pnl = (exit_price - pos["entry_price"]) * pos["shares"] - cost
            else:
                pnl = (pos["entry_price"] - exit_price) * pos["shares"] - cost

            if include_dividends and direction == 1:
                pnl += pos["div_amount"] * pos["shares"]

            equity += pnl
            trades.append({
                "ticker": ticker, "pnl": pnl,
                "entry": str(pos["entry_date"].date()),
                "exit": str(day.date())
            })

        # Check for new entries (1 day before ex-date)
        tomorrow_events = [e for e in events
                          if 0 < (e["exdate"] - day).days <= 2
                          and e["ticker"] not in positions]

        # Sector diversification: ensure max 3 different sectors
        current_sectors = set(SECTOR_MAP.get(t, "?") for t in positions)

        for event in tomorrow_events:
            if len(positions) >= MAX_POSITIONS:
                break
            ticker = event["ticker"]
            sector = SECTOR_MAP.get(ticker, "?")

            # Only add if we can maintain <=3 sectors or it's already in our sectors
            if len(current_sectors) >= 3 and sector not in current_sectors:
                continue

            if ticker not in data:
                continue
            prices = data[ticker]["prices"]
            if day not in prices.index:
                continue

            entry_price = float(prices.loc[day, "Close"])
            if entry_price <= 0:
                continue

            # Position sizing: equal weight
            alloc = equity / MAX_POSITIONS
            shares = int(alloc / entry_price)
            if shares < 1:
                continue

            cost = entry_price * shares * (slippage_rt / 2)
            positions[ticker] = {
                "entry_price": entry_price,
                "entry_date": day,
                "shares": shares,
                "div_amount": event["amount"]
            }
            equity -= cost  # entry cost
            current_sectors.add(sector)

        daily_equity[day] = equity

    # Close remaining positions
    for ticker, pos in positions.items():
        if ticker in data:
            prices = data[ticker]["prices"]
            last = prices.index[-1]
            exit_price = float(prices.loc[last, "Close"])
            cost = exit_price * pos["shares"] * (slippage_rt / 2)
            if direction == 1:
                pnl = (exit_price - pos["entry_price"]) * pos["shares"] - cost
            else:
                pnl = (pos["entry_price"] - exit_price) * pos["shares"] - cost
            if include_dividends and direction == 1:
                pnl += pos["div_amount"] * pos["shares"]
            equity += pnl
            trades.append({"ticker": ticker, "pnl": pnl,
                          "entry": str(pos["entry_date"].date()), "exit": str(last.date())})

    # Compute daily returns
    eq_series = pd.Series(daily_equity).sort_index()
    daily_returns = eq_series.pct_change().dropna()

    sharpe = compute_sharpe(daily_returns)
    total_return = (equity - account) / account
    n_trades = len(trades)
    win_rate = sum(1 for t in trades if t["pnl"] > 0) / max(n_trades, 1)

    # Max drawdown
    cummax = eq_series.cummax()
    drawdown = (eq_series - cummax) / cummax
    max_dd = drawdown.min() if len(drawdown) > 0 else 0

    return {
        "sharpe": round(sharpe, 3),
        "total_return": round(total_return, 4),
        "n_trades": n_trades,
        "win_rate": round(win_rate, 3),
        "max_dd": round(max_dd, 4),
        "equity_final": round(equity, 2),
        "daily_returns": daily_returns,
        "trades": trades,
    }


def test1_inverse_direction(data, events):
    """Test 1: Inverse direction - sell before ex-date, buy after."""
    print("\n" + "="*70)
    print("TEST 1: INVERSE DIRECTION")
    print("  If price movements alone are profitable, dividend timing adds nothing")
    print("="*70)

    result = simulate_strategy(data, events, ACCOUNT, HOLD_DAYS, SLIPPAGE_RT, direction=-1, include_dividends=False)

    passed = result["sharpe"] < 0
    print(f"  Inverse Sharpe: {result['sharpe']}")
    print(f"  Inverse Return: {result['total_return']*100:.1f}%")
    print(f"  Inverse Trades: {result['n_trades']}")
    print(f"  PASS CRITERIA: Inverse Sharpe < 0")
    print(f"  RESULT: {'PASS' if passed else 'FAIL'}")

    return {
        "test": "inverse_direction",
        "inverse_sharpe": result["sharpe"],
        "inverse_return": result["total_return"],
        "inverse_trades": result["n_trades"],
        "passed": passed,
        "criteria": "Inverse Sharpe < 0"
    }


def test2_buy_and_hold(data):
    """Test 2: Buy and hold equal-weight portfolio of same 14 stocks."""
    print("\n" + "="*70)
    print("TEST 2: BUY-AND-HOLD COMPARISON")
    print("  Equal-weight buy & hold of same 14 stocks over OOT period")
    print("="*70)

    oot_start = pd.Timestamp(OOT_START)
    oot_end = pd.Timestamp(OOT_END)

    # Build equal-weight daily returns
    ticker_returns = {}
    for ticker, d in data.items():
        if ticker.startswith("_"):
            continue
        prices = d["prices"]["Close"]
        mask = (prices.index >= oot_start) & (prices.index <= oot_end)
        p = prices[mask]
        if len(p) > 10:
            ticker_returns[ticker] = p.pct_change().dropna()

    if not ticker_returns:
        print("  ERROR: No ticker returns available")
        return {"test": "buy_and_hold", "passed": False, "error": "no data"}

    # Equal-weight portfolio returns
    all_rets = pd.DataFrame(ticker_returns)
    portfolio_returns = all_rets.mean(axis=1)

    bh_sharpe = compute_sharpe(portfolio_returns)
    bh_return = (1 + portfolio_returns).prod() - 1

    # Compare with strategy Sharpe
    passed = ORIGINAL_SHARPE > bh_sharpe

    print(f"  Buy-Hold Sharpe: {bh_sharpe:.3f}")
    print(f"  Buy-Hold Return: {bh_return*100:.1f}%")
    print(f"  Strategy Sharpe: {ORIGINAL_SHARPE}")
    print(f"  PASS CRITERIA: Strategy Sharpe ({ORIGINAL_SHARPE}) > Buy-Hold Sharpe ({bh_sharpe:.3f})")
    print(f"  RESULT: {'PASS' if passed else 'FAIL'}")

    return {
        "test": "buy_and_hold",
        "bh_sharpe": round(bh_sharpe, 3),
        "bh_return": round(bh_return, 4),
        "strategy_sharpe": ORIGINAL_SHARPE,
        "passed": passed,
        "criteria": f"Strategy Sharpe ({ORIGINAL_SHARPE}) > Buy-Hold Sharpe"
    }


def test3_random_timing(data, events):
    """Test 3: Random entry timing - 1000 permutations."""
    print("\n" + "="*70)
    print("TEST 3: RANDOM ENTRY TIMING")
    print(f"  {N_PERMUTATIONS} random entry permutations, same hold period")
    print("="*70)

    # First run actual strategy to get its Sharpe
    actual = simulate_strategy(data, events, ACCOUNT, HOLD_DAYS, SLIPPAGE_RT, direction=1, include_dividends=True)
    actual_sharpe = actual["sharpe"]

    # Now run random timing
    oot_start = pd.Timestamp(OOT_START)
    oot_end = pd.Timestamp(OOT_END)

    # Get all valid trading days across all tickers
    all_trading_days = {}
    for ticker, d in data.items():
        if ticker.startswith("_"):
            continue
        prices = d["prices"]
        mask = (prices.index >= oot_start) & (prices.index <= oot_end)
        valid_days = prices.index[mask].tolist()
        if len(valid_days) > HOLD_DAYS + 5:
            all_trading_days[ticker] = valid_days

    # Count how many trades the strategy took
    n_strategy_trades = actual["n_trades"]

    random_sharpes = []
    for perm in range(N_PERMUTATIONS):
        if (perm + 1) % 100 == 0:
            print(f"  Permutation {perm+1}/{N_PERMUTATIONS}...")

        # Generate random events: same number of trades, random tickers + dates
        fake_events = []
        available_tickers = list(all_trading_days.keys())
        for _ in range(n_strategy_trades):
            ticker = np.random.choice(available_tickers)
            days = all_trading_days[ticker]
            # Pick random entry day (leave room for hold period)
            max_idx = max(0, len(days) - HOLD_DAYS - 2)
            if max_idx == 0:
                continue
            idx = np.random.randint(0, max_idx)
            entry_day = days[idx]

            # Simulate single trade
            prices = data[ticker]["prices"]
            entry_price = float(prices.loc[entry_day, "Close"])
            exit_idx = min(idx + HOLD_DAYS, len(days) - 1)
            exit_day = days[exit_idx]
            exit_price = float(prices.loc[exit_day, "Close"])

            cost = (entry_price + exit_price) * (SLIPPAGE_RT / 2)
            pnl = (exit_price - entry_price) - cost
            fake_events.append(pnl / entry_price)

        if len(fake_events) > 5:
            # Compute pseudo-Sharpe from trade returns
            rets = np.array(fake_events)
            if rets.std() > 0:
                # Annualize assuming ~1 trade per 2 trading days
                trades_per_year = 252 / (HOLD_DAYS + 1)
                sharpe = (rets.mean() / rets.std()) * np.sqrt(trades_per_year)
                random_sharpes.append(sharpe)

    random_sharpes = np.array(random_sharpes)
    percentile = np.mean(random_sharpes < actual_sharpe) * 100

    passed = percentile > 90

    print(f"  Strategy Sharpe (re-simulated): {actual_sharpe:.3f}")
    print(f"  Random Sharpe Mean: {random_sharpes.mean():.3f}")
    print(f"  Random Sharpe Std: {random_sharpes.std():.3f}")
    print(f"  Random Sharpe p5/p50/p95: {np.percentile(random_sharpes, 5):.3f} / {np.percentile(random_sharpes, 50):.3f} / {np.percentile(random_sharpes, 95):.3f}")
    print(f"  Strategy Percentile: {percentile:.1f}%")
    print(f"  PASS CRITERIA: Strategy at >90th percentile")
    print(f"  RESULT: {'PASS' if passed else 'FAIL'}")

    return {
        "test": "random_timing",
        "strategy_sharpe": actual_sharpe,
        "random_mean": round(float(random_sharpes.mean()), 3),
        "random_std": round(float(random_sharpes.std()), 3),
        "random_p5": round(float(np.percentile(random_sharpes, 5)), 3),
        "random_p50": round(float(np.percentile(random_sharpes, 50)), 3),
        "random_p95": round(float(np.percentile(random_sharpes, 95)), 3),
        "percentile": round(percentile, 1),
        "passed": passed,
        "criteria": "Strategy at >90th percentile vs random timing"
    }


def test4_remove_dividends(data, events):
    """Test 4: Re-run strategy excluding dividend income."""
    print("\n" + "="*70)
    print("TEST 4: REMOVE DIVIDENDS")
    print("  Same strategy but exclude dividend income from PnL")
    print("="*70)

    # With dividends
    with_div = simulate_strategy(data, events, ACCOUNT, HOLD_DAYS, SLIPPAGE_RT, direction=1, include_dividends=True)
    # Without dividends
    no_div = simulate_strategy(data, events, ACCOUNT, HOLD_DAYS, SLIPPAGE_RT, direction=1, include_dividends=False)

    passed = no_div["sharpe"] > 0

    print(f"  Sharpe WITH dividends: {with_div['sharpe']}")
    print(f"  Sharpe WITHOUT dividends: {no_div['sharpe']}")
    print(f"  Return WITH dividends: {with_div['total_return']*100:.1f}%")
    print(f"  Return WITHOUT dividends: {no_div['total_return']*100:.1f}%")
    print(f"  Dividend contribution to Sharpe: {with_div['sharpe'] - no_div['sharpe']:.3f}")
    print(f"  PASS CRITERIA: Sharpe without dividends > 0 (price timing has value)")
    print(f"  RESULT: {'PASS' if passed else 'FAIL'}")

    return {
        "test": "remove_dividends",
        "sharpe_with_div": with_div["sharpe"],
        "sharpe_no_div": no_div["sharpe"],
        "return_with_div": with_div["total_return"],
        "return_no_div": no_div["total_return"],
        "div_sharpe_contribution": round(with_div["sharpe"] - no_div["sharpe"], 3),
        "passed": passed,
        "criteria": "Sharpe without dividends > 0"
    }


def test5_cost_sensitivity(data, events):
    """Test 5: Cost sensitivity at increasing slippage levels."""
    print("\n" + "="*70)
    print("TEST 5: COST SENSITIVITY")
    print("  Test at 0.05%, 0.10%, 0.15%, 0.20% round-trip slippage")
    print("="*70)

    cost_levels = [0.0005, 0.0010, 0.0015, 0.0020]
    cost_results = {}
    break_level = None

    for cost in cost_levels:
        result = simulate_strategy(data, events, ACCOUNT, HOLD_DAYS, cost, direction=1, include_dividends=True)
        label = f"{cost*100:.2f}%"
        cost_results[label] = {
            "sharpe": result["sharpe"],
            "return": result["total_return"],
            "win_rate": result["win_rate"],
        }
        print(f"  Slippage {label}: Sharpe={result['sharpe']:.3f}, Return={result['total_return']*100:.1f}%, WR={result['win_rate']*100:.1f}%")

        if result["sharpe"] < 0.5 and break_level is None:
            break_level = cost

    # Pass if robust to at least 0.10%
    passed = cost_results["0.10%"]["sharpe"] >= 0.5

    print(f"\n  Break level (Sharpe < 0.5): {break_level*100:.2f}%" if break_level else "\n  Strategy never breaks at tested levels")
    print(f"  PASS CRITERIA: Sharpe >= 0.5 at 0.10% slippage")
    print(f"  RESULT: {'PASS' if passed else 'FAIL'}")

    return {
        "test": "cost_sensitivity",
        "cost_results": cost_results,
        "break_level": f"{break_level*100:.2f}%" if break_level else "never",
        "passed": passed,
        "criteria": "Sharpe >= 0.5 at 0.10% slippage"
    }


def test6_subperiod_stability(data, events):
    """Test 6: Split OOT into 4 equal sub-periods."""
    print("\n" + "="*70)
    print("TEST 6: SUB-PERIOD STABILITY")
    print("  Split OOT into 4 ~13-month periods, check profitability")
    print("="*70)

    oot_start = pd.Timestamp(OOT_START)
    oot_end = pd.Timestamp(OOT_END)
    total_days = (oot_end - oot_start).days
    period_days = total_days // 4

    periods = []
    for i in range(4):
        p_start = oot_start + timedelta(days=i * period_days)
        p_end = oot_start + timedelta(days=(i + 1) * period_days) if i < 3 else oot_end
        periods.append((p_start, p_end))

    period_results = {}
    profitable_count = 0

    for i, (p_start, p_end) in enumerate(periods):
        label = f"P{i+1}: {p_start.strftime('%Y-%m')} to {p_end.strftime('%Y-%m')}"

        # Filter events to this sub-period
        sub_events = [e for e in events if p_start <= e["exdate"] <= p_end]

        result = simulate_strategy(data, sub_events, ACCOUNT, HOLD_DAYS, SLIPPAGE_RT, direction=1, include_dividends=True)

        profitable = result["total_return"] > 0
        if profitable:
            profitable_count += 1

        period_results[label] = {
            "sharpe": result["sharpe"],
            "return": result["total_return"],
            "n_trades": result["n_trades"],
            "win_rate": result["win_rate"],
            "profitable": profitable,
        }
        status = "PROFITABLE" if profitable else "LOSS"
        print(f"  {label}: Sharpe={result['sharpe']:.3f}, Return={result['total_return']*100:.1f}%, Trades={result['n_trades']}, WR={result['win_rate']*100:.1f}% [{status}]")

    passed = profitable_count >= 3

    print(f"\n  Profitable periods: {profitable_count}/4")
    print(f"  PASS CRITERIA: At least 3/4 periods profitable")
    print(f"  RESULT: {'PASS' if passed else 'FAIL'}")

    return {
        "test": "subperiod_stability",
        "period_results": period_results,
        "profitable_periods": profitable_count,
        "total_periods": 4,
        "passed": passed,
        "criteria": "At least 3/4 periods profitable"
    }


def main():
    print("=" * 70)
    print("DIVIDEND CAPTURE - VARIANT F - ADVERSARIAL VALIDATION")
    print(f"Tickers: {', '.join(TICKERS)}")
    print(f"OOT: {OOT_START} to {OOT_END}")
    print(f"Account: ${ACCOUNT}")
    print(f"Original Sharpe: {ORIGINAL_SHARPE}, WR: 65.3%, MaxDD: -2.82%")
    print("=" * 70)

    # Download data
    data = download_data()

    if len([k for k in data if not k.startswith("_")]) < 5:
        print("ERROR: Too few tickers with data. Aborting.")
        return

    # Get ex-dividend events
    events = get_exdiv_dates(data, OOT_START, OOT_END)
    print(f"\nTotal ex-dividend events in OOT: {len(events)}")

    # Run all 6 tests
    results = {}

    results["test1"] = test1_inverse_direction(data, events)
    results["test2"] = test2_buy_and_hold(data)
    results["test3"] = test3_random_timing(data, events)
    results["test4"] = test4_remove_dividends(data, events)
    results["test5"] = test5_cost_sensitivity(data, events)
    results["test6"] = test6_subperiod_stability(data, events)

    # Summary
    print("\n" + "=" * 70)
    print("ADVERSARIAL VALIDATION SUMMARY")
    print("=" * 70)

    all_passed = True
    for key, res in results.items():
        status = "PASS" if res["passed"] else "FAIL"
        if not res["passed"]:
            all_passed = False
        print(f"  {key}: {res['test']:25s} → {status}  ({res['criteria']})")

    overall = "ALL TESTS PASSED" if all_passed else "SOME TESTS FAILED"
    n_passed = sum(1 for r in results.values() if r["passed"])
    print(f"\n  OVERALL: {n_passed}/6 tests passed — {overall}")

    if not all_passed:
        print("\n  ⚠ STRATEGY HAS ADVERSARIAL VULNERABILITIES")
        failed = [r["test"] for r in results.values() if not r["passed"]]
        print(f"  Failed tests: {', '.join(failed)}")
    else:
        print("\n  ✓ Strategy survived all adversarial tests")

    # Save results
    results["summary"] = {
        "strategy": "Dividend Capture Variant F (Sector-Diversified)",
        "tests_passed": n_passed,
        "tests_total": 6,
        "all_passed": all_passed,
        "timestamp": datetime.now().isoformat(),
        "original_sharpe": ORIGINAL_SHARPE,
        "tickers": TICKERS,
        "oot_period": f"{OOT_START} to {OOT_END}",
        "account": ACCOUNT,
    }

    # Convert non-serializable items
    serializable = {}
    for k, v in results.items():
        if isinstance(v, dict):
            clean = {}
            for kk, vv in v.items():
                if isinstance(vv, (pd.Series, pd.DataFrame)):
                    continue
                elif isinstance(vv, np.floating):
                    clean[kk] = float(vv)
                elif isinstance(vv, np.integer):
                    clean[kk] = int(vv)
                elif isinstance(vv, np.bool_):
                    clean[kk] = bool(vv)
                else:
                    clean[kk] = vv
            serializable[k] = clean

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(serializable, f, indent=2, default=str)
    print(f"\nResults saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
