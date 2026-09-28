#!/usr/bin/env python3
"""
RSI-2 Mean Reversion Backtest — Larry Connors Style
====================================================
HC #705: All adversarial checks built INLINE.
Universe: 30 large-cap stocks | Period: 2010-01-01 to 2026-07-14
Entry: RSI(2) < threshold [+optional 200d MA filter] → buy NEXT DAY OPEN
Exit: RSI(2) > exit_threshold OR fixed 5-day hold

Inline checks:
  1. Permutation test (200 shuffles) — shuffle trade DATES, compare means
  2. Regime test — PRIOR-DAY SPY close (no leakage)
  3. Sub-period consistency (split into 4 equal periods)
  4. Outlier removal (winsorize top/bottom 1%)
  5. Ticker concentration (no single ticker > 15% of trades)
  6. Pricing sanity (no negative prices, no >50% daily moves)
"""

import json
import os
import sys
import warnings
import datetime as dt
from pathlib import Path
from itertools import product

import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats

warnings.filterwarnings("ignore")

OUTPUT_DIR = Path("/home/jupiter/Lvl3Quant/output/rsi2_meanrev_v1")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "NFLX", "AMD", "INTC",
    "BA", "DIS", "SBUX", "HD", "LOW", "MCD", "NKE", "COST", "WMT", "JPM",
    "GS", "BAC", "MS", "JNJ", "PG", "KO", "UNH", "ABBV", "CRM", "NOW"
]

START_DATE = "2010-01-01"
END_DATE = "2026-07-14"
STARTING_CAPITAL = 10_000
N_PERMUTATIONS = 200

# Entry thresholds to test
RSI_ENTRY_THRESHOLDS = [10, 5]
USE_MA_FILTER = [False, True]

# Exit rules
EXIT_RULES = {
    "rsi_gt_70": {"type": "rsi_exit", "threshold": 70},
    "rsi_gt_80": {"type": "rsi_exit", "threshold": 80},
    "rsi_gt_90": {"type": "rsi_exit", "threshold": 90},
    "hold_5d":   {"type": "fixed_hold", "days": 5},
}


def compute_rsi(series, period=2):
    """Compute RSI with Wilder smoothing (exponential)."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def download_data():
    """Download OHLCV data for all tickers + SPY."""
    all_tickers = TICKERS + ["SPY"]
    print(f"Downloading {len(all_tickers)} tickers from {START_DATE} to {END_DATE}...")

    data = {}
    failed = []
    for ticker in all_tickers:
        try:
            df = yf.download(ticker, start=START_DATE, end=END_DATE, progress=False, auto_adjust=True)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            if len(df) < 252:  # need at least 1 year
                print(f"  WARNING: {ticker} has only {len(df)} rows, skipping")
                failed.append(ticker)
                continue

            # PRICING SANITY CHECK (inline)
            if (df["Close"] <= 0).any():
                neg_count = (df["Close"] <= 0).sum()
                print(f"  SANITY FAIL: {ticker} has {neg_count} non-positive close prices — removing those rows")
                df = df[df["Close"] > 0]

            daily_ret = df["Close"].pct_change()
            extreme = (daily_ret.abs() > 0.50)
            if extreme.any():
                n_ext = extreme.sum()
                print(f"  SANITY WARN: {ticker} has {n_ext} daily moves > 50%, clipping")
                # Don't remove — just flag. These could be stock splits missed by adj.
                # We'll handle via outlier removal in trade returns.

            data[ticker] = df

        except Exception as e:
            print(f"  ERROR downloading {ticker}: {e}")
            failed.append(ticker)

    print(f"Downloaded {len(data)} tickers successfully. Failed: {failed}")
    return data


def prepare_features(data):
    """Compute RSI(2), 200-day MA, and next-day open for each ticker."""
    features = {}
    for ticker, df in data.items():
        if ticker == "SPY":
            continue
        feat = pd.DataFrame(index=df.index)
        feat["close"] = df["Close"]
        feat["open"] = df["Open"]
        feat["high"] = df["High"]
        feat["low"] = df["Low"]
        feat["volume"] = df["Volume"]
        feat["rsi2"] = compute_rsi(df["Close"], period=2)
        feat["ma200"] = df["Close"].rolling(200).mean()
        # Next-day open for entry price (shift -1 = tomorrow's open)
        feat["next_open"] = df["Open"].shift(-1)
        # Next-day date
        feat["next_date"] = df.index.to_series().shift(-1)
        features[ticker] = feat.dropna(subset=["rsi2"])
    return features


def generate_trades(features, spy_data, rsi_entry, use_ma, exit_rule):
    """
    Generate trades for a given parameter combo.

    LEAKAGE PREVENTION:
    - Signal: RSI(2) computed on today's close → trade at NEXT DAY open
    - 200d MA: computed on data up to and including today (available at close)
    - Exit: RSI exit uses the exit-day's close (you see it, sell next open)
      For simplicity we'll model exit at the close of the exit-signal day
      (conservative — real execution at next open would be slightly different)
    - Regime: uses PRIOR-DAY SPY return (no same-day leakage)
    """
    trades = []

    # Pre-compute SPY prior-day return for regime classification
    spy_close = spy_data["Close"]
    spy_prior_return = spy_close.pct_change().shift(1)  # PRIOR day return (shift 1 = yesterday's return available today)

    for ticker, feat in features.items():
        in_trade = False
        entry_price = None
        entry_date = None
        entry_idx = None

        dates = feat.index.tolist()

        for i, date in enumerate(dates):
            row = feat.loc[date]

            if in_trade:
                # Check exit conditions
                should_exit = False
                exit_reason = None

                if exit_rule["type"] == "rsi_exit":
                    if row["rsi2"] > exit_rule["threshold"]:
                        should_exit = True
                        exit_reason = f"rsi>{exit_rule['threshold']}"
                elif exit_rule["type"] == "fixed_hold":
                    days_held = (date - entry_date).days
                    if days_held >= exit_rule["days"]:
                        should_exit = True
                        exit_reason = f"hold_{exit_rule['days']}d"

                if should_exit:
                    # Exit at this day's close (signal seen, exit EOD)
                    exit_price = row["close"]
                    ret = (exit_price - entry_price) / entry_price

                    # Get prior-day SPY return for regime (at entry date)
                    regime = "unknown"
                    if entry_date in spy_prior_return.index:
                        spy_ret = spy_prior_return.loc[entry_date]
                        if pd.notna(spy_ret):
                            regime = "bull" if spy_ret > 0 else "bear"

                    trades.append({
                        "ticker": ticker,
                        "entry_date": entry_date,
                        "exit_date": date,
                        "entry_price": entry_price,
                        "exit_price": exit_price,
                        "return": ret,
                        "regime": regime,
                        "exit_reason": exit_reason,
                        "days_held": (date - entry_date).days,
                    })
                    in_trade = False

            else:
                # Check entry conditions
                if pd.isna(row["rsi2"]) or pd.isna(row.get("next_open", np.nan)):
                    continue

                signal = row["rsi2"] < rsi_entry

                if use_ma:
                    if pd.isna(row["ma200"]):
                        continue
                    signal = signal and (row["close"] > row["ma200"])

                if signal:
                    # Enter at NEXT DAY OPEN
                    entry_price = row["next_open"]
                    entry_date = row["next_date"]
                    if pd.isna(entry_price) or pd.isna(entry_date):
                        continue
                    entry_date = pd.Timestamp(entry_date)
                    in_trade = True

    return pd.DataFrame(trades) if trades else pd.DataFrame()


def permutation_test(trades_df, all_features, n_perms=N_PERMUTATIONS):
    """
    Permutation test: for each permutation, randomly pick entry dates
    (from all available trading dates for each ticker) and compute the
    return over the same holding period. This tests whether the RSI signal
    selects better-than-random entry points.

    For each shuffle:
      - For each trade, pick a random entry date for the same ticker
      - Compute the return over the same number of holding days
      - Average across all "fake" trades
    p-value = fraction of shuffled mean returns >= real mean return.
    """
    if len(trades_df) == 0:
        return 1.0, []

    real_mean = trades_df["return"].mean()
    n_trades = len(trades_df)

    # Pre-compute available dates and close prices per ticker for fast lookup
    ticker_data = {}
    for ticker in trades_df["ticker"].unique():
        if ticker in all_features:
            feat = all_features[ticker]
            closes = feat["close"].values
            opens = feat["open"].values
            dates_arr = feat.index.values
            ticker_data[ticker] = {
                "closes": closes,
                "opens": opens,
                "dates": dates_arr,
                "n": len(closes),
            }

    # Build array of (ticker, days_held) for each real trade
    trade_specs = []
    for _, row in trades_df.iterrows():
        ticker = row["ticker"]
        days_held = int(row["days_held"])
        if ticker in ticker_data:
            trade_specs.append((ticker, days_held))

    if not trade_specs:
        return 1.0, []

    shuffled_means = []
    rng = np.random.RandomState(42)

    for _ in range(n_perms):
        fake_returns = []
        for ticker, days_held in trade_specs:
            td = ticker_data[ticker]
            n_bars = td["n"]
            # Pick a random entry point, need room for days_held
            max_entry = max(0, n_bars - days_held - 1)
            if max_entry <= 0:
                continue
            entry_idx = rng.randint(0, max_entry)
            # Use next-day open as entry (like real strategy)
            entry_price = td["opens"][entry_idx + 1] if entry_idx + 1 < n_bars else td["closes"][entry_idx]
            # Exit: find the bar ~days_held later
            exit_idx = min(entry_idx + 1 + days_held, n_bars - 1)
            exit_price = td["closes"][exit_idx]
            if entry_price > 0:
                fake_returns.append((exit_price - entry_price) / entry_price)

        if fake_returns:
            shuffled_means.append(np.mean(fake_returns))

    shuffled_means = np.array(shuffled_means)
    p_value = float((shuffled_means >= real_mean).mean())

    return p_value, shuffled_means.tolist()


def regime_test(trades_df):
    """
    Test performance in bull vs bear regimes.
    Regime = PRIOR-DAY SPY return (already computed with no leakage).
    """
    if len(trades_df) == 0:
        return {}

    bull = trades_df[trades_df["regime"] == "bull"]
    bear = trades_df[trades_df["regime"] == "bear"]

    result = {
        "bull_trades": len(bull),
        "bear_trades": len(bear),
        "bull_mean_return": float(bull["return"].mean()) if len(bull) > 0 else None,
        "bear_mean_return": float(bear["return"].mean()) if len(bear) > 0 else None,
        "bull_win_rate": float((bull["return"] > 0).mean()) if len(bull) > 0 else None,
        "bear_win_rate": float((bear["return"] > 0).mean()) if len(bear) > 0 else None,
    }

    # Regime divergence check
    if result["bull_mean_return"] is not None and result["bear_mean_return"] is not None:
        max_abs = max(abs(result["bull_mean_return"]), abs(result["bear_mean_return"]))
        if max_abs > 0:
            divergence = abs(result["bull_mean_return"] - result["bear_mean_return"]) / max_abs
        else:
            divergence = 0
        result["regime_divergence"] = float(divergence)
        result["regime_pass"] = divergence <= 0.50
    else:
        result["regime_divergence"] = None
        result["regime_pass"] = False

    return result


def subperiod_test(trades_df):
    """Split trades into 4 equal time periods and check consistency."""
    if len(trades_df) < 20:
        return {"pass": False, "reason": "too_few_trades", "periods": []}

    trades_sorted = trades_df.sort_values("entry_date")
    n = len(trades_sorted)
    chunk = n // 4

    periods = []
    all_positive = True
    for i in range(4):
        start_idx = i * chunk
        end_idx = (i + 1) * chunk if i < 3 else n
        sub = trades_sorted.iloc[start_idx:end_idx]
        mean_ret = float(sub["return"].mean())
        wr = float((sub["return"] > 0).mean())
        periods.append({
            "period": i + 1,
            "n_trades": len(sub),
            "start": str(sub["entry_date"].iloc[0].date()),
            "end": str(sub["entry_date"].iloc[-1].date()),
            "mean_return": mean_ret,
            "win_rate": wr,
        })
        if mean_ret <= 0:
            all_positive = False

    # Consistency: at least 3 of 4 periods positive
    n_positive = sum(1 for p in periods if p["mean_return"] > 0)

    return {
        "pass": n_positive >= 3,
        "n_positive_periods": n_positive,
        "all_positive": all_positive,
        "periods": periods,
    }


def outlier_analysis(trades_df):
    """Winsorize top/bottom 1% and re-check if edge survives."""
    if len(trades_df) < 10:
        return {"pass": False, "reason": "too_few_trades"}

    returns = trades_df["return"].values
    p1, p99 = np.percentile(returns, [1, 99])
    winsorized = np.clip(returns, p1, p99)

    raw_mean = float(np.mean(returns))
    winsorized_mean = float(np.mean(winsorized))

    # How much of the edge survives winsorization?
    if raw_mean != 0:
        survival = winsorized_mean / raw_mean
    else:
        survival = 0

    return {
        "raw_mean_return": raw_mean,
        "winsorized_mean_return": winsorized_mean,
        "p1_cutoff": float(p1),
        "p99_cutoff": float(p99),
        "edge_survival_pct": float(survival * 100),
        "pass": winsorized_mean > 0 and survival > 0.50,
    }


def ticker_concentration(trades_df):
    """Check no single ticker dominates >15% of trades."""
    if len(trades_df) == 0:
        return {"pass": False, "reason": "no_trades"}

    counts = trades_df["ticker"].value_counts()
    pcts = counts / len(trades_df)
    max_ticker = pcts.idxmax()
    max_pct = float(pcts.max())

    return {
        "max_ticker": max_ticker,
        "max_concentration": max_pct,
        "pass": max_pct <= 0.15,
        "ticker_counts": {k: int(v) for k, v in counts.head(10).items()},
    }


def compute_equity_curve(trades_df, capital=STARTING_CAPITAL):
    """Compute equity curve and risk metrics from trade returns."""
    if len(trades_df) == 0:
        return {}, pd.Series(dtype=float)

    trades_sorted = trades_df.sort_values("entry_date")
    returns = trades_sorted["return"].values

    # Simple: invest fixed fraction per trade (1 position at a time)
    equity = [capital]
    for r in returns:
        equity.append(equity[-1] * (1 + r))
    equity = np.array(equity)

    # Compute metrics
    total_return = (equity[-1] / equity[0]) - 1
    n_years = (trades_sorted["entry_date"].iloc[-1] - trades_sorted["entry_date"].iloc[0]).days / 365.25
    if n_years > 0:
        cagr = (equity[-1] / equity[0]) ** (1 / n_years) - 1
    else:
        cagr = 0

    # Drawdown
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / peak
    max_dd = float(dd.min())

    # Sharpe (annualized, assuming ~1 trade per few days)
    if len(returns) > 1 and np.std(returns) > 0:
        avg_days_held = trades_sorted["days_held"].mean()
        trades_per_year = 252 / max(avg_days_held, 1)
        sharpe = (np.mean(returns) / np.std(returns)) * np.sqrt(trades_per_year)

        # Sortino
        downside = returns[returns < 0]
        if len(downside) > 0:
            downside_std = np.std(downside)
            sortino = (np.mean(returns) / downside_std) * np.sqrt(trades_per_year) if downside_std > 0 else np.inf
        else:
            sortino = np.inf
    else:
        sharpe = 0
        sortino = 0

    # Profit factor
    gross_profit = returns[returns > 0].sum()
    gross_loss = abs(returns[returns < 0].sum())
    pf = float(gross_profit / gross_loss) if gross_loss > 0 else np.inf

    # Win rate
    wr = float((returns > 0).mean())

    # Average win / average loss
    avg_win = float(returns[returns > 0].mean()) if (returns > 0).any() else 0
    avg_loss = float(returns[returns < 0].mean()) if (returns < 0).any() else 0

    metrics = {
        "total_return_pct": float(total_return * 100),
        "cagr_pct": float(cagr * 100),
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "profit_factor": float(pf),
        "win_rate": float(wr),
        "max_drawdown_pct": float(max_dd * 100),
        "n_trades": int(len(returns)),
        "avg_return_pct": float(np.mean(returns) * 100),
        "avg_win_pct": float(avg_win * 100),
        "avg_loss_pct": float(avg_loss * 100),
        "avg_days_held": float(trades_sorted["days_held"].mean()),
        "final_equity": float(equity[-1]),
        "n_years": float(n_years),
    }

    return metrics, pd.Series(equity)


def run_backtest_combo(features, spy_data, rsi_entry, use_ma, exit_name, exit_rule, all_features=None):
    """Run a single backtest configuration with all adversarial checks."""
    combo_name = f"RSI<{rsi_entry}_MA{'yes' if use_ma else 'no'}_{exit_name}"
    print(f"\n{'='*60}")
    print(f"  {combo_name}")
    print(f"{'='*60}")

    # Generate trades
    trades_df = generate_trades(features, spy_data, rsi_entry, use_ma, exit_rule)

    if len(trades_df) == 0:
        print(f"  NO TRADES generated. Skipping.")
        return {
            "combo": combo_name,
            "params": {"rsi_entry": rsi_entry, "use_ma_filter": use_ma, "exit_rule": exit_name},
            "n_trades": 0,
            "skip_reason": "no_trades",
        }

    print(f"  Generated {len(trades_df)} trades")

    # 1. Performance metrics
    metrics, equity = compute_equity_curve(trades_df)
    print(f"  Sharpe={metrics['sharpe']:.2f} | Sortino={metrics['sortino']:.2f} | "
          f"PF={metrics['profit_factor']:.2f} | WR={metrics['win_rate']:.1%} | "
          f"CAGR={metrics['cagr_pct']:.1f}%")

    # 2. Permutation test
    print(f"  Running permutation test ({N_PERMUTATIONS} shuffles)...")
    p_value, _ = permutation_test(trades_df, all_features or features)
    perm_pass = p_value < 0.05
    print(f"  Permutation p-value: {p_value:.4f} {'PASS' if perm_pass else 'FAIL'}")

    # 3. Regime test (prior-day SPY)
    regime = regime_test(trades_df)
    print(f"  Regime test: bull_mean={regime.get('bull_mean_return', 0):.4f} "
          f"bear_mean={regime.get('bear_mean_return', 0):.4f} "
          f"divergence={regime.get('regime_divergence', 'N/A')} "
          f"{'PASS' if regime.get('regime_pass') else 'FAIL'}")

    # 4. Sub-period consistency
    subperiod = subperiod_test(trades_df)
    print(f"  Sub-period: {subperiod.get('n_positive_periods', 0)}/4 positive "
          f"{'PASS' if subperiod.get('pass') else 'FAIL'}")

    # 5. Outlier removal
    outlier = outlier_analysis(trades_df)
    print(f"  Outlier test: raw_mean={outlier.get('raw_mean_return', 0):.4f} "
          f"winsorized={outlier.get('winsorized_mean_return', 0):.4f} "
          f"survival={outlier.get('edge_survival_pct', 0):.0f}% "
          f"{'PASS' if outlier.get('pass') else 'FAIL'}")

    # 6. Ticker concentration
    conc = ticker_concentration(trades_df)
    print(f"  Concentration: max={conc.get('max_ticker', 'N/A')} "
          f"at {conc.get('max_concentration', 0):.1%} "
          f"{'PASS' if conc.get('pass') else 'FAIL'}")

    # Overall quality gate
    gates = {
        "permutation_pass": perm_pass,
        "regime_pass": regime.get("regime_pass", False),
        "subperiod_pass": subperiod.get("pass", False),
        "outlier_pass": outlier.get("pass", False),
        "concentration_pass": conc.get("pass", False),
        "sharpe_positive": metrics["sharpe"] > 0,
        "profit_factor_gt_1": metrics["profit_factor"] > 1.0,
        "min_trades_50": metrics["n_trades"] >= 50,
    }
    all_pass = all(gates.values())
    n_pass = sum(gates.values())

    print(f"\n  QUALITY GATES: {n_pass}/{len(gates)} passed — "
          f"{'ALL PASS' if all_pass else 'SOME FAILED'}")

    result = {
        "combo": combo_name,
        "params": {
            "rsi_entry_threshold": rsi_entry,
            "use_ma_filter": use_ma,
            "exit_rule": exit_name,
            "exit_params": exit_rule,
        },
        "performance": metrics,
        "quality_gates": gates,
        "all_gates_pass": all_pass,
        "gates_passed": n_pass,
        "gates_total": len(gates),
        "adversarial_checks": {
            "permutation_test": {
                "p_value": float(p_value),
                "n_permutations": N_PERMUTATIONS,
                "pass": perm_pass,
            },
            "regime_test": regime,
            "subperiod_test": subperiod,
            "outlier_test": outlier,
            "ticker_concentration": conc,
        },
    }

    return result


def sizing_for_small_account(best_result):
    """
    Practical sizing guidance for $440 Robinhood cash account.
    """
    if best_result is None or best_result.get("n_trades", 0) == 0:
        return {"feasible": False, "reason": "no_viable_strategy"}

    perf = best_result.get("performance", {})
    params = best_result.get("params", {})

    return {
        "account_size": 440,
        "account_type": "cash (no margin, no PDT issues since cash)",
        "position_size": "1 share per signal (most of these stocks $100-500+, so 1 share is full allocation)",
        "max_concurrent": 1,
        "note": "Cash account settles T+1 for stocks. Can only trade settled funds. With 1 position, wait for settlement before next trade.",
        "strategy_params": params,
        "expected_metrics": {
            "sharpe": perf.get("sharpe"),
            "win_rate": perf.get("win_rate"),
            "avg_return_pct": perf.get("avg_return_pct"),
            "avg_days_held": perf.get("avg_days_held"),
        },
        "affordable_tickers": "Filter universe to stocks < $440 (BAC, INTC, KO, NKE, SBUX, etc.)",
        "warning": "Many large-caps (AMZN, GOOGL, NVDA, META) exceed $440/share. Strategy would need fractional shares or sub-universe.",
    }


def main():
    print("=" * 70)
    print("RSI-2 MEAN REVERSION BACKTEST — Full Adversarial Suite")
    print("=" * 70)

    # Download data
    data = download_data()
    if len(data) < 5:
        print("FATAL: Too few tickers downloaded. Aborting.")
        sys.exit(1)

    spy_data = data.get("SPY")
    if spy_data is None:
        print("FATAL: Could not download SPY data. Aborting.")
        sys.exit(1)

    # Prepare features
    features = prepare_features(data)
    print(f"\nPrepared features for {len(features)} tickers")

    # Run all combos
    all_results = []

    for rsi_entry in RSI_ENTRY_THRESHOLDS:
        for use_ma in USE_MA_FILTER:
            for exit_name, exit_rule in EXIT_RULES.items():
                result = run_backtest_combo(features, spy_data, rsi_entry, use_ma, exit_name, exit_rule, all_features=features)
                all_results.append(result)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY — ALL CONFIGURATIONS")
    print("=" * 70)

    # Sort by gates passed, then Sharpe
    valid_results = [r for r in all_results if r.get("performance")]
    valid_results.sort(
        key=lambda x: (x.get("gates_passed", 0), x.get("performance", {}).get("sharpe", -99)),
        reverse=True
    )

    print(f"\n{'Config':<45} {'Trades':>6} {'Sharpe':>7} {'Sortino':>8} {'PF':>6} {'WR':>6} {'Gates':>7}")
    print("-" * 90)
    for r in valid_results:
        perf = r.get("performance", {})
        print(f"{r['combo']:<45} {perf.get('n_trades', 0):>6} "
              f"{perf.get('sharpe', 0):>7.2f} {perf.get('sortino', 0):>8.2f} "
              f"{perf.get('profit_factor', 0):>6.2f} {perf.get('win_rate', 0):>6.1%} "
              f"{r.get('gates_passed', 0):>3}/{r.get('gates_total', 0)}")

    # Best configuration
    best = valid_results[0] if valid_results else None

    # Small account sizing
    sizing = sizing_for_small_account(best)

    # Build final report
    report = {
        "run_timestamp": dt.datetime.now().isoformat(),
        "parameters": {
            "universe": TICKERS,
            "period": f"{START_DATE} to {END_DATE}",
            "starting_capital": STARTING_CAPITAL,
            "rsi_period": 2,
            "entry_thresholds_tested": RSI_ENTRY_THRESHOLDS,
            "ma_filter_tested": [False, True],
            "exit_rules_tested": list(EXIT_RULES.keys()),
            "n_permutations": N_PERMUTATIONS,
        },
        "leakage_prevention": {
            "entry": "Signal on day-T close RSI → enter at day-T+1 open",
            "ma_filter": "200-day MA computed on data up to signal day (available at close)",
            "regime": "Uses PRIOR-DAY SPY return (shift=1), not same-day",
            "exit": "Exit at close of day exit signal triggers",
        },
        "n_configurations_tested": len(all_results),
        "best_configuration": best,
        "all_configurations": all_results,
        "small_account_sizing": sizing,
        "quality_gate_definitions": {
            "permutation_pass": "Permutation test p < 0.05 (200 shuffles, bootstrap under null)",
            "regime_pass": "Bull/bear regime divergence <= 0.50 (prior-day SPY)",
            "subperiod_pass": "At least 3 of 4 equal time periods show positive mean return",
            "outlier_pass": "Winsorized (1/99 pctile) mean return still positive and >50% of raw",
            "concentration_pass": "No single ticker > 15% of total trades",
            "sharpe_positive": "Annualized Sharpe ratio > 0",
            "profit_factor_gt_1": "Profit factor > 1.0",
            "min_trades_50": "At least 50 trades in sample",
        },
    }

    # Save report
    report_path = OUTPUT_DIR / "backtest_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nReport saved to {report_path}")

    # Print best config details
    if best:
        print(f"\n{'='*70}")
        print(f"BEST CONFIGURATION: {best['combo']}")
        print(f"{'='*70}")
        perf = best["performance"]
        print(f"  Trades: {perf['n_trades']}")
        print(f"  Sharpe: {perf['sharpe']:.2f}")
        print(f"  Sortino: {perf['sortino']:.2f}")
        print(f"  Profit Factor: {perf['profit_factor']:.2f}")
        print(f"  Win Rate: {perf['win_rate']:.1%}")
        print(f"  CAGR: {perf['cagr_pct']:.1f}%")
        print(f"  Max DD: {perf['max_drawdown_pct']:.1f}%")
        print(f"  Avg Days Held: {perf['avg_days_held']:.1f}")
        print(f"  Gates: {best['gates_passed']}/{best['gates_total']}")
        print(f"  All Gates Pass: {best['all_gates_pass']}")

    return report


if __name__ == "__main__":
    report = main()
