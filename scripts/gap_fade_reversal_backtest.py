#!/usr/bin/env python3
"""
Gap Fade Reversal Backtest
--------------------------
Buy quality stocks that gap DOWN at the open, betting on mean reversion.
6 variants (A-F) with 5-gate validation.

OOT: Jan 2022 - Jul 2026 | Starting capital: $645
"""

import json
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from scipy import stats

warnings.filterwarnings("ignore")

# ── Config ──────────────────────────────────────────────────────────────────
UNIVERSE = [
    "AAPL", "MSFT", "AVGO", "JPM", "JNJ", "PG", "KO", "PEP", "HD", "COST",
    "UNH", "LLY", "V", "MA", "ABBV", "MRK", "WMT", "AMZN", "GOOGL", "META",
]
START_DATE = "2021-10-01"  # extra lookback for RSI/SMA
OOT_START = "2022-01-01"
OOT_END = "2026-07-31"
STARTING_CAPITAL = 645.0
SLIPPAGE_PCT = 0.0002  # 0.02% each way
MAX_POS_VALUE = 200.0
MAX_CONCURRENT = 3
N_PERMUTATIONS = 1000
RSI_PERIOD = 14
SPY_SMA_PERIOD = 200

OUTPUT_PATH = "/home/jupiter/Lvl3Quant/data/gap_fade_reversal_results.json"


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def download_data():
    """Download OHLCV for universe + SPY + VIX."""
    tickers = UNIVERSE + ["SPY", "^VIX"]
    print(f"Downloading data for {len(tickers)} tickers...")
    data = yf.download(tickers, start=START_DATE, end=OOT_END, auto_adjust=True, progress=False)
    return data


def get_earnings_dates(tickers):
    """Fetch earnings dates from yfinance for each ticker."""
    earnings = {}
    for t in tickers:
        try:
            tk = yf.Ticker(t)
            ed = tk.earnings_dates
            if ed is not None and len(ed) > 0:
                dates = set(ed.index.normalize().date)
                earnings[t] = dates
            else:
                earnings[t] = set()
        except Exception:
            earnings[t] = set()
    return earnings


def prepare_frames(data):
    """Extract per-ticker OHLCV DataFrames plus SPY/VIX series."""
    frames = {}
    for t in UNIVERSE:
        try:
            df = pd.DataFrame({
                "Open": data["Open"][t],
                "High": data["High"][t],
                "Low": data["Low"][t],
                "Close": data["Close"][t],
                "Volume": data["Volume"][t],
            }).dropna()
            df["prev_close"] = df["Close"].shift(1)
            df["gap_pct"] = (df["Open"] - df["prev_close"]) / df["prev_close"]
            df["rsi"] = compute_rsi(df["Close"], RSI_PERIOD)
            df["avg_volume_20"] = df["Volume"].rolling(20).mean()
            frames[t] = df
        except Exception:
            pass

    spy_close = data["Close"]["SPY"].dropna()
    spy_sma200 = spy_close.rolling(SPY_SMA_PERIOD).mean()

    try:
        vix_close = data["Close"]["^VIX"].dropna()
    except Exception:
        vix_close = pd.Series(dtype=float)

    return frames, spy_close, spy_sma200, vix_close


def is_bull(date, spy_close, spy_sma200):
    """Bull if SPY > 200-SMA at date."""
    if date in spy_close.index and date in spy_sma200.index:
        return spy_close.loc[date] > spy_sma200.loc[date]
    # fallback: find nearest prior date
    prior = spy_close.index[spy_close.index <= date]
    if len(prior) == 0:
        return True
    d = prior[-1]
    if d in spy_sma200.index:
        return spy_close.loc[d] > spy_sma200.loc[d]
    return True


def run_variant(variant_name, frames, spy_close, spy_sma200, vix_close, earnings_dates):
    """
    Run a single variant. Returns dict with trades, equity curve, stats.
    """
    trades = []
    oot_start = pd.Timestamp(OOT_START)
    oot_end = pd.Timestamp(OOT_END)

    for ticker, df in frames.items():
        for i in range(1, len(df)):
            date = df.index[i]
            if date < oot_start or date > oot_end:
                continue

            row = df.iloc[i]
            prev_row = df.iloc[i - 1]
            gap = row["gap_pct"]

            if pd.isna(gap) or gap >= 0:
                continue  # no gap-down

            gap_down_pct = -gap  # positive number

            # Variant-specific entry filters + hold period
            if variant_name == "A":
                if gap_down_pct < 0.01:
                    continue
                hold_days = 0  # sell at close same day
            elif variant_name == "B":
                if gap_down_pct < 0.02:
                    continue
                hold_days = 3
            elif variant_name == "C":
                if gap_down_pct < 0.01:
                    continue
                # Earnings filter: skip if gap is on or day after earnings
                entry_date = date.date()
                day_before = (date - timedelta(days=1)).date()
                earn_set = earnings_dates.get(ticker, set())
                if entry_date in earn_set or day_before in earn_set:
                    continue
                hold_days = 2
            elif variant_name == "D":
                if gap_down_pct < 0.015:
                    continue
                prev_rsi = prev_row.get("rsi", np.nan)
                if pd.isna(prev_rsi) or prev_rsi <= 40:
                    continue
                hold_days = 3
            elif variant_name == "E":
                if gap_down_pct < 0.01:
                    continue
                # VIX < 25
                if date in vix_close.index:
                    vix_val = vix_close.loc[date]
                else:
                    prior_vix = vix_close.index[vix_close.index <= date]
                    if len(prior_vix) == 0:
                        continue
                    vix_val = vix_close.iloc[-1]
                if pd.isna(vix_val) or vix_val >= 25:
                    continue
                hold_days = 2
            elif variant_name == "F":
                if gap_down_pct < 0.02:
                    continue
                vol = row.get("Volume", np.nan)
                avg_vol = row.get("avg_volume_20", np.nan)
                if pd.isna(vol) or pd.isna(avg_vol) or avg_vol == 0:
                    continue
                if vol >= 1.5 * avg_vol:
                    continue  # high volume = real selling
                hold_days = 3
            else:
                continue

            # Entry price = open + slippage
            entry_price = row["Open"] * (1 + SLIPPAGE_PCT)

            # Exit price
            if hold_days == 0:
                # Sell at close same day
                exit_price = row["Close"] * (1 - SLIPPAGE_PCT)
                exit_date = date
            else:
                exit_idx = min(i + hold_days, len(df) - 1)
                exit_row = df.iloc[exit_idx]
                exit_price = exit_row["Close"] * (1 - SLIPPAGE_PCT)
                exit_date = df.index[exit_idx]

            shares = max(1, int(MAX_POS_VALUE / entry_price))
            pnl = (exit_price - entry_price) * shares
            ret = (exit_price / entry_price) - 1.0

            regime = "bull" if is_bull(date, spy_close, spy_sma200) else "bear"

            trades.append({
                "ticker": ticker,
                "entry_date": str(date.date()),
                "exit_date": str(exit_date.date()),
                "entry_price": round(entry_price, 4),
                "exit_price": round(exit_price, 4),
                "shares": shares,
                "pnl": round(pnl, 2),
                "return": round(ret, 6),
                "regime": regime,
                "gap_pct": round(gap, 4),
            })

    return trades


def simulate_portfolio(trades, starting_capital):
    """
    Simulate portfolio with max concurrent position limit.
    Returns equity curve and filtered trades.
    """
    if not trades:
        return [], [], starting_capital, 0

    trade_df = pd.DataFrame(trades)
    trade_df["entry_date"] = pd.to_datetime(trade_df["entry_date"])
    trade_df["exit_date"] = pd.to_datetime(trade_df["exit_date"])
    trade_df = trade_df.sort_values("entry_date").reset_index(drop=True)

    equity = starting_capital
    active_positions = []  # list of exit_dates
    executed_trades = []
    equity_curve = [{"date": str(trade_df["entry_date"].iloc[0].date()), "equity": equity}]

    for _, tr in trade_df.iterrows():
        # Remove expired positions
        active_positions = [ed for ed in active_positions if ed > tr["entry_date"]]

        if len(active_positions) >= MAX_CONCURRENT:
            continue  # skip trade

        cost = tr["entry_price"] * tr["shares"]
        if cost > equity:
            continue  # not enough capital

        active_positions.append(tr["exit_date"])
        equity += tr["pnl"]
        executed_trades.append(tr.to_dict())
        equity_curve.append({"date": str(tr["exit_date"].date()), "equity": round(equity, 2)})

    return executed_trades, equity_curve, equity, len(executed_trades)


def compute_stats(trades, starting_capital):
    """Compute performance statistics."""
    if not trades or len(trades) < 2:
        return {
            "n_trades": len(trades) if trades else 0,
            "sharpe": 0, "sortino": 0, "pf": 0, "wr": 0,
            "total_return_pct": 0, "max_dd_pct": 0,
            "avg_return_pct": 0, "median_return_pct": 0,
        }

    returns = [t["return"] for t in trades]
    pnls = [t["pnl"] for t in trades]
    returns = np.array(returns)
    pnls = np.array(pnls)

    n = len(returns)
    avg_ret = np.mean(returns)
    std_ret = np.std(returns, ddof=1) if n > 1 else 1e-9

    # Annualize: assume ~252 trading days, avg ~1 trade per few days
    trades_per_year = max(1, n / 4.5)  # rough
    sharpe = (avg_ret / std_ret) * np.sqrt(trades_per_year) if std_ret > 0 else 0

    downside = returns[returns < 0]
    downside_std = np.std(downside, ddof=1) if len(downside) > 1 else 1e-9
    sortino = (avg_ret / downside_std) * np.sqrt(trades_per_year) if downside_std > 0 else 0

    gross_profit = pnls[pnls > 0].sum() if (pnls > 0).any() else 0
    gross_loss = abs(pnls[pnls < 0].sum()) if (pnls < 0).any() else 1e-9
    pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    wr = (pnls > 0).sum() / n if n > 0 else 0

    # Equity curve for drawdown
    equity = starting_capital
    peak = equity
    max_dd = 0
    for p in pnls:
        equity += p
        peak = max(peak, equity)
        dd = (equity - peak) / peak
        max_dd = min(max_dd, dd)

    total_return = (equity - starting_capital) / starting_capital

    return {
        "n_trades": n,
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "pf": round(pf, 3),
        "wr": round(wr, 3),
        "total_return_pct": round(total_return * 100, 2),
        "max_dd_pct": round(max_dd * 100, 2),
        "avg_return_pct": round(avg_ret * 100, 4),
        "median_return_pct": round(np.median(returns) * 100, 4),
        "final_equity": round(equity, 2),
    }


def permutation_test(trades, n_perms=1000):
    """
    Permutation test: shuffle entry dates (break date-signal association).
    Returns p-value = fraction of shuffled Sharpes >= actual Sharpe.
    """
    if not trades or len(trades) < 5:
        return 1.0

    returns = np.array([t["return"] for t in trades])
    n = len(returns)
    actual_mean = np.mean(returns)

    count_ge = 0
    for _ in range(n_perms):
        shuffled = np.random.permutation(returns)
        if np.mean(shuffled) >= actual_mean:
            count_ge += 1

    return count_ge / n_perms


def regime_gap_test(trades):
    """
    Compute regime gap: |Sharpe_bull - Sharpe_bear| / max(|Sharpe_bull|, |Sharpe_bear|).
    Returns gap value. < 0.5 passes.
    """
    if not trades:
        return 1.0

    bull_rets = [t["return"] for t in trades if t["regime"] == "bull"]
    bear_rets = [t["return"] for t in trades if t["regime"] == "bear"]

    def _sharpe(rets):
        if len(rets) < 2:
            return 0
        r = np.array(rets)
        std = np.std(r, ddof=1)
        if std == 0:
            return 0
        return np.mean(r) / std

    s_bull = _sharpe(bull_rets)
    s_bear = _sharpe(bear_rets)

    denom = max(abs(s_bull), abs(s_bear))
    if denom == 0:
        return 0

    return abs(s_bull - s_bear) / denom


def validate_5gates(stats, trades, variant_name):
    """Run 5-gate validation. Returns dict of gate results."""
    gates = {}

    # Gate 1: Sharpe > 0.5
    gates["sharpe_gt_0.5"] = {"value": stats["sharpe"], "pass": stats["sharpe"] > 0.5}

    # Gate 2: Permutation test p < 0.05
    p_val = permutation_test(trades, N_PERMUTATIONS)
    gates["perm_p_lt_0.05"] = {"value": round(p_val, 4), "pass": p_val < 0.05}

    # Gate 3: Regime gap < 0.5
    rgap = regime_gap_test(trades)
    gates["regime_gap_lt_0.5"] = {"value": round(rgap, 4), "pass": rgap < 0.5}

    # Gate 4: MaxDD > -50%
    gates["maxdd_gt_neg50"] = {"value": stats["max_dd_pct"], "pass": stats["max_dd_pct"] > -50}

    # Gate 5: >= 20 trades
    gates["trades_ge_20"] = {"value": stats["n_trades"], "pass": stats["n_trades"] >= 20}

    all_pass = all(g["pass"] for g in gates.values())
    gates["ALL_PASS"] = all_pass

    return gates


def main():
    np.random.seed(42)

    # Download data
    data = download_data()

    # Prepare per-ticker frames
    print("Preparing data frames...")
    frames, spy_close, spy_sma200, vix_close = prepare_frames(data)
    print(f"  Loaded {len(frames)} tickers")

    # Get earnings dates for variant C
    print("Fetching earnings dates (for variant C filter)...")
    earnings_dates = get_earnings_dates(UNIVERSE)
    n_earn = sum(len(v) for v in earnings_dates.values())
    print(f"  Found {n_earn} earnings dates across {len(earnings_dates)} tickers")

    variants = ["A", "B", "C", "D", "E", "F"]
    results = {}

    for v in variants:
        print(f"\n{'='*60}")
        print(f"  VARIANT {v}")
        print(f"{'='*60}")

        raw_trades = run_variant(v, frames, spy_close, spy_sma200, vix_close, earnings_dates)
        print(f"  Raw signals: {len(raw_trades)}")

        executed_trades, equity_curve, final_eq, n_exec = simulate_portfolio(
            raw_trades, STARTING_CAPITAL
        )
        print(f"  Executed trades (after concurrency limit): {n_exec}")

        st = compute_stats(executed_trades, STARTING_CAPITAL)
        gates = validate_5gates(st, executed_trades, v)

        results[v] = {
            "variant": v,
            "description": get_variant_description(v),
            "stats": st,
            "gates": gates,
            "n_raw_signals": len(raw_trades),
            "n_executed": n_exec,
            "equity_curve_endpoints": {
                "start": STARTING_CAPITAL,
                "end": st.get("final_equity", STARTING_CAPITAL),
            },
        }

        print_variant_summary(v, st, gates)

    # Print final comparison table
    print_comparison_table(results)

    # Save results
    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {OUTPUT_PATH}")


def get_variant_description(v):
    descs = {
        "A": "Gap-down >1%, sell at close same day (intraday reversal)",
        "B": "Gap-down >2%, hold 3 days",
        "C": "Gap-down >1%, NOT earnings day, hold 2 days",
        "D": "Gap-down >1.5% + RSI(14)>40, hold 3 days",
        "E": "Gap-down >1% + VIX<25, hold 2 days",
        "F": "Gap-down >2% + volume<1.5x avg, hold 3 days",
    }
    return descs.get(v, "")


def print_variant_summary(v, stats, gates):
    print(f"\n  --- Variant {v} Results ---")
    print(f"  Trades:  {stats['n_trades']}")
    print(f"  Sharpe:  {stats['sharpe']:.3f}")
    print(f"  Sortino: {stats['sortino']:.3f}")
    print(f"  PF:      {stats['pf']:.3f}")
    print(f"  WR:      {stats['wr']:.1%}")
    print(f"  Return:  {stats['total_return_pct']:.2f}%")
    print(f"  MaxDD:   {stats['max_dd_pct']:.2f}%")
    print(f"  Final:   ${stats.get('final_equity', 0):.2f}")
    print()
    for gname, gval in gates.items():
        if gname == "ALL_PASS":
            status = "PASS" if gval else "FAIL"
            print(f"  >> 5-GATE: {status}")
        else:
            status = "PASS" if gval["pass"] else "FAIL"
            print(f"  Gate {gname}: {gval['value']} [{status}]")


def print_comparison_table(results):
    print(f"\n{'='*90}")
    print(f"  GAP FADE REVERSAL — VARIANT COMPARISON")
    print(f"{'='*90}")
    header = f"{'Var':>3} | {'Trades':>6} | {'Sharpe':>7} | {'Sortino':>7} | {'PF':>6} | {'WR':>6} | {'Ret%':>8} | {'MaxDD%':>7} | {'Final$':>8} | {'5-Gate':>6}"
    print(header)
    print("-" * 90)
    for v in ["A", "B", "C", "D", "E", "F"]:
        r = results[v]
        s = r["stats"]
        gate_status = "PASS" if r["gates"]["ALL_PASS"] else "FAIL"
        print(
            f"  {v:>1} | {s['n_trades']:>6} | {s['sharpe']:>7.3f} | {s['sortino']:>7.3f} | "
            f"{s['pf']:>6.2f} | {s['wr']:>5.1%} | {s['total_return_pct']:>7.2f}% | "
            f"{s['max_dd_pct']:>6.2f}% | ${s.get('final_equity', 0):>7.2f} | {gate_status:>6}"
        )
    print("-" * 90)

    # Highlight best
    passing = [v for v in results if results[v]["gates"]["ALL_PASS"]]
    if passing:
        best = max(passing, key=lambda v: results[v]["stats"]["sharpe"])
        print(f"\n  BEST PASSING VARIANT: {best} (Sharpe={results[best]['stats']['sharpe']:.3f})")
    else:
        print("\n  NO VARIANT PASSED ALL 5 GATES")


if __name__ == "__main__":
    main()
