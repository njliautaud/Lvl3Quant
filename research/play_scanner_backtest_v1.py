"""
Play Scanner Historical Backtest v1
====================================
Tests the 3 play scanner setup types on historical data.

Instead of pricing options, we test the DIRECTIONAL SIGNAL:
- Does the stock move favorably after setup fires?
- What's the win rate at 5/10/21 day horizons?
- How big is the favorable move vs adverse move?

If the directional signal is real, options on it will be profitable.
If not, no amount of options engineering will save it.

Setup types (from play_scanner_v2.py):
  A) Momentum continuation: above all SMAs, RSI 55-70, MFI>60, OBV rising, 1-3% pullback
  B) Oversold bounce: RSI<35, below 20 SMA, BUT above 200 SMA, volume spike
  C) Flow divergence: OBV rising while price flat/down, MFI turning up from <30

Validation:
  - 200-shuffle permutation test (randomize entry dates)
  - Regime stratification (SPY up/down months)
  - Sub-period stability
"""

import json
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path("/home/jupiter/Lvl3Quant")

# ---------------------------------------------------------------------------
# Universe
# ---------------------------------------------------------------------------
SECTOR_ETFS = ["XLK", "XLF", "XLV", "XLE", "XLI", "XLC", "XLY", "XLP", "XLU", "XLRE", "XLB"]
INDEX_ETFS = ["SPY", "QQQ", "IWM"]
TOP_STOCKS = [
    "AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "META", "JPM",
    "TSLA", "V", "XOM", "MA", "COST", "HD", "WMT", "BAC",
    "CRM", "NFLX", "AMD", "KO", "PEP",
]
ALL_TICKERS = SECTOR_ETFS + INDEX_ETFS + TOP_STOCKS

HORIZONS = [5, 10, 21]  # Forward return horizons (trading days)
MIN_HISTORY = 252  # Need 252 days of history for 200 SMA


# ---------------------------------------------------------------------------
# Indicator Functions (mirror play_scanner_v2.py)
# ---------------------------------------------------------------------------
def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def compute_mfi(high, low, close, volume, period=14):
    typical_price = (high + low + close) / 3
    raw_money_flow = typical_price * volume
    delta = typical_price.diff()
    pos_flow = raw_money_flow.where(delta > 0, 0.0)
    neg_flow = raw_money_flow.where(delta <= 0, 0.0)
    pos_sum = pos_flow.rolling(period).sum()
    neg_sum = neg_flow.rolling(period).sum()
    mfr = pos_sum / neg_sum.replace(0, np.nan)
    return 100 - (100 / (1 + mfr))


def compute_obv(close, volume):
    direction = np.sign(close.diff()).fillna(0)
    return (direction * volume).cumsum()


# ---------------------------------------------------------------------------
# Setup Detection (mirror play_scanner_v2.py logic)
# ---------------------------------------------------------------------------
def detect_setups(df):
    """
    For each day, detect if setup A/B/C fires.
    Returns DataFrame with columns: date, ticker, setup_type
    """
    if len(df) < MIN_HISTORY:
        return pd.DataFrame()

    close = df["Close"].copy()
    high = df["High"].copy()
    low = df["Low"].copy()
    volume = df["Volume"].copy()

    # Indicators
    sma_10 = close.rolling(10).mean()
    sma_20 = close.rolling(20).mean()
    sma_50 = close.rolling(50).mean()
    sma_200 = close.rolling(200).mean()
    rsi = compute_rsi(close)
    mfi = compute_mfi(high, low, close, volume)
    obv = compute_obv(close, volume)
    obv_sma = obv.rolling(20).mean()

    # Returns
    ret_1d = close.pct_change()
    ret_5d = close.pct_change(5)

    # Volume
    vol_sma_20 = volume.rolling(20).mean()
    vol_ratio = volume / vol_sma_20.replace(0, np.nan)

    signals = []

    # Convert to numpy for speed
    c = close.values.astype(float)
    s10 = sma_10.values.astype(float)
    s20 = sma_20.values.astype(float)
    s50 = sma_50.values.astype(float)
    s200 = sma_200.values.astype(float)
    r = rsi.values.astype(float)
    m = mfi.values.astype(float)
    o = obv.values.astype(float)
    os = obv_sma.values.astype(float)
    r1 = ret_1d.values.astype(float)
    r5 = ret_5d.values.astype(float)
    vr = vol_ratio.values.astype(float)

    for i in range(MIN_HISTORY, len(df)):
        date = df.index[i]

        # Setup A: Momentum Continuation
        above_smas = (c[i] > s10[i] and c[i] > s20[i] and
                      c[i] > s50[i] and c[i] > s200[i])
        rsi_range = 55 <= r[i] <= 70
        mfi_strong = m[i] > 60
        obv_rising = o[i] > os[i]
        pullback = -0.03 <= r5[i] <= -0.01

        if above_smas and rsi_range and mfi_strong and obv_rising and pullback:
            signals.append({"date": date, "setup": "momentum_continuation"})

        # Setup B: Oversold Bounce
        oversold = r[i] < 35
        below_20 = c[i] < s20[i]
        above_200 = c[i] > s200[i]
        vol_spike = vr[i] > 1.5

        if oversold and below_20 and above_200 and vol_spike:
            signals.append({"date": date, "setup": "oversold_bounce"})

        # Setup C: Flow Divergence
        price_flat_down = r5[i] <= 0.005
        obv_diverging = o[i] > o[max(0, i-10)]
        mfi_low = m[i] < 40
        mfi_turning = m[i] > m[max(0, i-3)]

        if price_flat_down and obv_diverging and mfi_low and mfi_turning:
            signals.append({"date": date, "setup": "flow_divergence"})

    return pd.DataFrame(signals)


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------
def load_price_data():
    """Load cached price data or download via yfinance."""
    cache_path = ROOT / "research/cache/scanner_backtest_prices_v2.parquet"

    if cache_path.exists():
        import os
        mtime = os.path.getmtime(cache_path)
        age_hours = (datetime.now().timestamp() - mtime) / 3600
        if age_hours < 24:
            print(f"  Using cached prices ({age_hours:.1f}h old)")
            return pd.read_parquet(cache_path)

    print("  Downloading price data via yfinance (one-by-one)...")
    import yfinance as yf

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    frames = []

    for ticker in ALL_TICKERS:
        try:
            df = yf.download(ticker, period="5y", progress=False)
            if len(df) < MIN_HISTORY:
                print(f"  {ticker}: only {len(df)} rows, skipping")
                continue
            # Flatten multi-level columns if needed
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [c[0] for c in df.columns]
            df = df[["Close", "High", "Low", "Open", "Volume"]].copy()
            df["ticker"] = ticker
            frames.append(df)
        except Exception as e:
            print(f"  {ticker}: FAILED ({e})")

    combined = pd.concat(frames)
    combined.to_parquet(cache_path)
    print(f"  Saved {len(frames)} tickers to cache")
    return combined


# ---------------------------------------------------------------------------
# Backtest
# ---------------------------------------------------------------------------
def compute_forward_returns(prices_df, signals_df, ticker):
    """Compute forward returns at each horizon for signal dates."""
    ticker_prices = prices_df[prices_df["ticker"] == ticker].copy()
    if len(ticker_prices) == 0:
        return pd.DataFrame()

    ticker_prices = ticker_prices.sort_index()
    close = ticker_prices["Close"]

    results = []
    ticker_signals = signals_df[signals_df.index.isin(close.index)].copy()

    for sig_date in ticker_signals.index:
        idx = close.index.get_loc(sig_date)
        entry_price = close.iloc[idx]

        row = {"date": sig_date, "setup": ticker_signals.loc[sig_date, "setup"],
               "ticker": ticker, "entry_price": entry_price}

        for h in HORIZONS:
            if idx + h < len(close):
                exit_price = close.iloc[idx + h]
                fwd_ret = (exit_price / entry_price - 1)
                row[f"ret_{h}d"] = fwd_ret

                # MFE/MAE within horizon
                future_prices = close.iloc[idx+1:idx+h+1]
                if len(future_prices) > 0:
                    row[f"mfe_{h}d"] = (future_prices.max() / entry_price - 1)
                    row[f"mae_{h}d"] = (future_prices.min() / entry_price - 1)
            else:
                row[f"ret_{h}d"] = np.nan
                row[f"mfe_{h}d"] = np.nan
                row[f"mae_{h}d"] = np.nan

        results.append(row)

    return pd.DataFrame(results)


def run_backtest():
    """Main backtest logic."""
    print("Loading price data...")
    prices = load_price_data()
    tickers_available = prices["ticker"].unique()
    print(f"  {len(tickers_available)} tickers available")

    # Detect setups for each ticker
    print("\nDetecting setups...")
    all_signals = []

    for ticker in tickers_available:
        ticker_df = prices[prices["ticker"] == ticker].copy()
        if len(ticker_df) < MIN_HISTORY:
            continue

        signals = detect_setups(ticker_df)
        if len(signals) > 0:
            signals["ticker"] = ticker
            all_signals.append(signals)

    if not all_signals:
        print("NO SIGNALS FOUND! Check setup logic.")
        return None

    all_signals_df = pd.concat(all_signals, ignore_index=True)
    all_signals_df["date"] = pd.to_datetime(all_signals_df["date"])
    print(f"  Total signals: {len(all_signals_df)}")
    for setup, count in all_signals_df["setup"].value_counts().items():
        print(f"    {setup}: {count}")

    # Compute forward returns
    print("\nComputing forward returns...")
    all_returns = []

    for ticker in all_signals_df["ticker"].unique():
        ticker_sigs = all_signals_df[all_signals_df["ticker"] == ticker].set_index("date")
        fwd = compute_forward_returns(prices, ticker_sigs, ticker)
        if len(fwd) > 0:
            all_returns.append(fwd)

    returns_df = pd.concat(all_returns, ignore_index=True)
    print(f"  {len(returns_df)} signals with forward returns")

    return returns_df, all_signals_df, prices


def analyze_results(returns_df):
    """Analyze setup performance."""
    print("\n" + "=" * 70)
    print("SETUP PERFORMANCE ANALYSIS")
    print("=" * 70)

    results = {}

    for setup in ["momentum_continuation", "oversold_bounce", "flow_divergence"]:
        sub = returns_df[returns_df["setup"] == setup].copy()
        if len(sub) < 10:
            print(f"\n{setup}: Only {len(sub)} signals, skipping")
            continue

        print(f"\n{'─' * 50}")
        print(f"SETUP: {setup.upper()} ({len(sub)} signals)")
        print(f"{'─' * 50}")

        setup_result = {"n_signals": len(sub)}

        for h in HORIZONS:
            col = f"ret_{h}d"
            mfe_col = f"mfe_{h}d"
            mae_col = f"mae_{h}d"

            valid = sub[col].dropna()
            if len(valid) < 5:
                continue

            mean_ret = valid.mean() * 100
            median_ret = valid.median() * 100
            wr = (valid > 0).mean() * 100
            avg_win = valid[valid > 0].mean() * 100 if (valid > 0).any() else 0
            avg_loss = valid[valid <= 0].mean() * 100 if (valid <= 0).any() else 0
            pf = abs(valid[valid > 0].sum() / valid[valid <= 0].sum()) if (valid <= 0).any() and valid[valid <= 0].sum() != 0 else float("inf")

            # MFE/MAE
            mfe = sub[mfe_col].dropna()
            mae = sub[mae_col].dropna()
            avg_mfe = mfe.mean() * 100 if len(mfe) > 0 else 0
            avg_mae = mae.mean() * 100 if len(mae) > 0 else 0
            p90_mfe = mfe.quantile(0.90) * 100 if len(mfe) > 0 else 0

            print(f"\n  {h}-day horizon ({len(valid)} trades):")
            print(f"    Mean return:  {mean_ret:+.2f}%")
            print(f"    Median return: {median_ret:+.2f}%")
            print(f"    Win rate:     {wr:.1f}%")
            print(f"    Avg win:      {avg_win:+.2f}%")
            print(f"    Avg loss:     {avg_loss:+.2f}%")
            print(f"    Profit factor: {pf:.2f}")
            print(f"    Avg MFE:      {avg_mfe:+.2f}%")
            print(f"    Avg MAE:      {avg_mae:+.2f}%")
            print(f"    P90 MFE:      {p90_mfe:+.2f}%")

            setup_result[f"{h}d"] = {
                "mean_ret": round(mean_ret, 2),
                "median_ret": round(median_ret, 2),
                "win_rate": round(wr, 1),
                "avg_win": round(avg_win, 2),
                "avg_loss": round(avg_loss, 2),
                "profit_factor": round(pf, 2),
                "avg_mfe": round(avg_mfe, 2),
                "avg_mae": round(avg_mae, 2),
                "p90_mfe": round(p90_mfe, 2),
                "n_trades": len(valid),
            }

        # Per-ticker breakdown (top 5 by count)
        top_tickers = sub["ticker"].value_counts().head(5)
        print(f"\n  Top tickers: {dict(top_tickers)}")

        results[setup] = setup_result

    return results


def regime_analysis(returns_df, prices):
    """Stratify by SPY regime."""
    print("\n" + "=" * 50)
    print("REGIME STRATIFICATION")
    print("=" * 50)

    spy = prices[prices["ticker"] == "SPY"].copy()
    if len(spy) == 0:
        print("No SPY data for regime analysis")
        return {}

    spy = spy.sort_index()
    spy["sma_200"] = spy["Close"].rolling(200).mean()
    spy["regime"] = np.where(spy["Close"] > spy["sma_200"], "bull", "bear")

    regime_results = {}

    for setup in returns_df["setup"].unique():
        sub = returns_df[returns_df["setup"] == setup].copy()
        sub["date"] = pd.to_datetime(sub["date"])

        # Match regime
        sub_with_regime = sub.copy()
        regimes = []
        for _, row in sub.iterrows():
            d = row["date"]
            spy_near = spy[spy.index <= d]
            if len(spy_near) > 0:
                regimes.append(spy_near["regime"].iloc[-1])
            else:
                regimes.append("unknown")
        sub_with_regime["regime"] = regimes

        print(f"\n{setup}:")
        for regime in ["bull", "bear"]:
            rsub = sub_with_regime[sub_with_regime["regime"] == regime]
            if len(rsub) < 5:
                print(f"  {regime}: {len(rsub)} signals (too few)")
                continue

            for h in [10]:
                col = f"ret_{h}d"
                valid = rsub[col].dropna()
                if len(valid) < 3:
                    continue
                mean_ret = valid.mean() * 100
                wr = (valid > 0).mean() * 100
                print(f"  {regime} ({len(valid)} sigs): {h}d mean={mean_ret:+.2f}%, WR={wr:.0f}%")

                regime_results[f"{setup}_{regime}_{h}d"] = {
                    "mean_ret": round(mean_ret, 2),
                    "win_rate": round(wr, 1),
                    "n": len(valid),
                }

    return regime_results


def permutation_test(returns_df, prices, n_perms=200):
    """
    Shuffle entry dates to test if setup timing adds value.
    For each permutation, randomly assign entry dates to the same ticker
    and compare forward returns.
    """
    print(f"\nPermutation test ({n_perms} shuffles)...")

    real_means = {}
    for setup in returns_df["setup"].unique():
        sub = returns_df[returns_df["setup"] == setup]
        for h in [10]:
            col = f"ret_{h}d"
            valid = sub[col].dropna()
            if len(valid) >= 10:
                real_means[f"{setup}_{h}d"] = valid.mean()

    perm_results = {k: [] for k in real_means}
    rng = np.random.default_rng(42)

    for p in range(n_perms):
        if p % 50 == 0:
            print(f"  Perm {p}/{n_perms}...")

        for setup in returns_df["setup"].unique():
            sub = returns_df[returns_df["setup"] == setup].copy()

            for ticker in sub["ticker"].unique():
                ticker_sigs = sub[sub["ticker"] == ticker]
                n_sigs = len(ticker_sigs)

                # Get all valid dates for this ticker
                ticker_prices = prices[prices["ticker"] == ticker]
                if len(ticker_prices) < MIN_HISTORY + max(HORIZONS):
                    continue

                valid_dates = ticker_prices.index[MIN_HISTORY:-max(HORIZONS)]
                if len(valid_dates) < n_sigs:
                    continue

                # Random entry dates
                random_dates = rng.choice(valid_dates, size=n_sigs, replace=False)

                for h in [10]:
                    key = f"{setup}_{h}d"
                    if key not in perm_results:
                        continue

                    close = ticker_prices["Close"]
                    for rd in random_dates:
                        idx = close.index.get_loc(rd)
                        if idx + h < len(close):
                            fwd = close.iloc[idx + h] / close.iloc[idx] - 1
                            perm_results[key].append(fwd)

    # Compute p-values
    print("\nPermutation results:")
    perm_summary = {}
    for key, real_mean in real_means.items():
        perm_vals = perm_results[key]
        if not perm_vals:
            continue

        # Compare per-permutation means
        chunk_size = len(perm_vals) // n_perms if n_perms > 0 else len(perm_vals)
        if chunk_size == 0:
            continue

        perm_means = []
        for i in range(0, len(perm_vals) - chunk_size + 1, chunk_size):
            perm_means.append(np.mean(perm_vals[i:i+chunk_size]))

        p_value = np.mean([pm >= real_mean for pm in perm_means])
        random_mean = np.mean(perm_means)

        print(f"  {key}: real={real_mean*100:+.2f}%, random={random_mean*100:+.2f}%, "
              f"p={p_value:.3f} ({'PASS' if p_value < 0.05 else 'FAIL'})")

        perm_summary[key] = {
            "real_mean_pct": round(real_mean * 100, 3),
            "random_mean_pct": round(random_mean * 100, 3),
            "p_value": round(p_value, 4),
            "pass": p_value < 0.05,
        }

    return perm_summary


def main():
    print("=" * 70)
    print("PLAY SCANNER HISTORICAL BACKTEST v1")
    print("=" * 70)
    print(f"Start: {datetime.now()}")
    print(f"Universe: {len(ALL_TICKERS)} tickers")
    print(f"Setups: momentum continuation, oversold bounce, flow divergence")
    print(f"Horizons: {HORIZONS} days")
    print()

    result = run_backtest()
    if result is None:
        print("BACKTEST FAILED")
        return

    returns_df, signals_df, prices = result

    # Analyze
    setup_results = analyze_results(returns_df)

    # Regime
    regime_results = regime_analysis(returns_df, prices)

    # Permutation
    perm_results = permutation_test(returns_df, prices, n_perms=200)

    # Save
    output = {
        "strategy": "play_scanner_backtest_v1",
        "timestamp": datetime.now().isoformat(),
        "universe_size": len(ALL_TICKERS),
        "total_signals": len(signals_df),
        "setup_results": setup_results,
        "regime_results": regime_results,
        "permutation_results": perm_results,
        "signal_counts": dict(signals_df["setup"].value_counts()),
    }

    out_path = ROOT / "research/findings/play_scanner_backtest_v1_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    # Verdict
    print("\n" + "=" * 50)
    print("VERDICT:")

    for setup in ["momentum_continuation", "oversold_bounce", "flow_divergence"]:
        if setup not in setup_results:
            continue
        sr = setup_results[setup]
        pkey = f"{setup}_10d"
        perm = perm_results.get(pkey, {})

        gates = []
        # Gate 1: Positive mean return at 10d
        if sr.get("10d", {}).get("mean_ret", 0) > 0:
            gates.append("Positive 10d return PASS")
        else:
            gates.append("Positive 10d return FAIL")

        # Gate 2: Win rate > 52%
        if sr.get("10d", {}).get("win_rate", 0) > 52:
            gates.append("Win rate >52% PASS")
        else:
            gates.append("Win rate >52% FAIL")

        # Gate 3: Permutation
        if perm.get("pass", False):
            gates.append("Permutation PASS")
        else:
            gates.append("Permutation FAIL")

        # Gate 4: Profit factor > 1.1
        if sr.get("10d", {}).get("profit_factor", 0) > 1.1:
            gates.append("PF>1.1 PASS")
        else:
            gates.append("PF>1.1 FAIL")

        n_pass = sum(1 for g in gates if "PASS" in g)
        status = "✅" if n_pass >= 3 else "⚠️" if n_pass >= 2 else "❌"
        print(f"\n  {status} {setup}: {n_pass}/{len(gates)} gates")
        for g in gates:
            print(f"      {g}")

    print(f"\nDone: {datetime.now()}")
    print(f"Results: {out_path}")


if __name__ == "__main__":
    main()
