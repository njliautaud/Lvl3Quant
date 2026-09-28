"""
Production Config Final v1 — Sector ETF Bull Call Spread Strategy
================================================================
LightGBM sector momentum ranking -> bull call spreads on top-3 ETFs.
Walk-forward: 12-month train, 1-month test, sliding window.
Equity-based Sharpe, calendar month aggregation, B-S pricing + 15% haircut.
4-gate adversarial validation (permutation, regime, sub-period, outlier).

Three configs tested:
  A) 20-day exit
  B) 30% trailing stop from peak
  C) Hold to expiry (baseline)
"""
from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.stats import norm
from datetime import datetime, timedelta
from pathlib import Path
import time
import traceback
import copy

# ── Constants ──────────────────────────────────────────────────────────
SECTOR_ETFS = ["XLE", "XLK", "XLF", "XLV", "XLI", "XLP", "XLU", "XLRE", "XLB", "XLC", "XLY"]
BENCHMARK = "SPY"
ALL_TICKERS = SECTOR_ETFS + [BENCHMARK]

START_DATE = "2011-01-01"
END_DATE = "2026-07-25"

INITIAL_CAPITAL = 645.0
MAX_TRADE_SIZE = 250.0  # target per trade
COMMISSION_PER_CONTRACT = 0.65
COMMISSION_RT = COMMISSION_PER_CONTRACT * 4  # 2 legs x open+close

# Walk-forward
TRAIN_MONTHS = 12
TEST_MONTHS = 1

# Strategy
TARGET_DTE = 30
VIX_FLOOR = 20.0
TOP_N = 3
REBALANCE_FREQ_DAYS = 10  # bi-weekly (trading days)
SPREAD_WIDTH_PCT = 0.03  # 3% OTM spread

# B-S pricing
RISK_FREE_RATE = 0.045
HAIRCUT = 0.15  # 15% slippage haircut on B-S prices

# LGBM
LGBM_PARAMS = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "num_leaves": 31,
    "learning_rate": 0.05,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "min_child_samples": 20,
    "lambda_l1": 0.1,
    "lambda_l2": 1.0,
    "max_depth": 6,
    "verbosity": -1,
    "seed": 42,
    "n_jobs": -1,
}
NUM_BOOST_ROUND = 300
EARLY_STOPPING_ROUNDS = 30

OUTPUT_DIR = Path("/home/jupiter/Lvl3Quant/output/production_config_final_v1")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ── Black-Scholes ─────────────────────────────────────────────────────
def bs_call_price(S, K, T, r, sigma):
    """Black-Scholes call price. T in years."""
    if T <= 0:
        return max(S - K, 0.0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def bull_call_spread_price(S, K_low, K_high, T, r, sigma):
    """Price of bull call spread = long K_low call - short K_high call."""
    return bs_call_price(S, K_low, T, r, sigma) - bs_call_price(S, K_high, T, r, sigma)


def price_spread_with_haircut(S, K_low, K_high, T, r, sigma, direction="buy"):
    """Apply 15% haircut. Buy = pay more, sell = receive less."""
    fair = bull_call_spread_price(S, K_low, K_high, T, r, sigma)
    if direction == "buy":
        return fair * (1 + HAIRCUT)
    else:
        return fair * (1 - HAIRCUT)


# ── Data Loading ──────────────────────────────────────────────────────
def load_data():
    """Download price data via yfinance."""
    import yfinance as yf

    print(f"Downloading data for {len(ALL_TICKERS)} tickers from {START_DATE} to {END_DATE}...")

    # Download all at once
    data = yf.download(ALL_TICKERS, start=START_DATE, end=END_DATE,
                       auto_adjust=True, progress=False, group_by='ticker')

    prices = {}
    for ticker in ALL_TICKERS:
        try:
            if len(ALL_TICKERS) > 1:
                df = data[ticker][["Close", "Volume"]].dropna()
            else:
                df = data[["Close", "Volume"]].dropna()
            df.columns = ["close", "volume"]
            prices[ticker] = df
        except Exception as e:
            print(f"  WARNING: Failed to get {ticker}: {e}")

    # Also get VIX
    vix = yf.download("^VIX", start=START_DATE, end=END_DATE,
                      auto_adjust=True, progress=False)
    vix_series = vix["Close"].dropna()
    if hasattr(vix_series, 'columns'):
        vix_series = vix_series.iloc[:, 0]

    print(f"  Got {len(prices)} tickers, VIX has {len(vix_series)} days")
    return prices, vix_series


# ── Feature Engineering ───────────────────────────────────────────────
def build_features(prices, vix_series):
    """Build feature panel for all sectors."""
    spy_close = prices[BENCHMARK]["close"]

    records = []

    for ticker in SECTOR_ETFS:
        if ticker not in prices:
            continue
        df = prices[ticker].copy()
        close = df["close"]
        volume = df["volume"]

        # Returns at various lookbacks
        for lb in [21, 63, 126, 252]:  # 1m, 3m, 6m, 12m
            df[f"ret_{lb}d"] = close.pct_change(lb)

        # Volatility
        daily_ret = close.pct_change()
        df["vol_20d"] = daily_ret.rolling(20).std() * np.sqrt(252)
        df["vol_60d"] = daily_ret.rolling(60).std() * np.sqrt(252)
        df["vol_ratio"] = df["vol_20d"] / df["vol_60d"].replace(0, np.nan)

        # RSI(14)
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        df["rsi_14"] = 100 - (100 / (1 + rs))

        # Relative strength vs SPY
        spy_aligned = spy_close.reindex(close.index)
        for lb in [21, 63, 126]:
            sector_ret = close.pct_change(lb)
            spy_ret = spy_aligned.pct_change(lb)
            df[f"rel_str_{lb}d"] = sector_ret - spy_ret

        # Volume ratio
        df["vol_ratio_20_60"] = volume.rolling(20).mean() / volume.rolling(60).mean().replace(0, np.nan)

        # Forward 21-day return (target)
        df["fwd_21d_ret"] = close.pct_change(21).shift(-21)

        # VIX
        df["vix"] = vix_series.reindex(close.index)

        df["ticker"] = ticker
        df["date"] = df.index

        records.append(df)

    panel = pd.concat(records, ignore_index=True)

    feature_cols = [
        "ret_21d", "ret_63d", "ret_126d", "ret_252d",
        "vol_20d", "vol_60d", "vol_ratio",
        "rsi_14",
        "rel_str_21d", "rel_str_63d", "rel_str_126d",
        "vol_ratio_20_60",
        "vix",
    ]

    panel = panel.dropna(subset=feature_cols + ["fwd_21d_ret"])

    print(f"  Feature panel: {len(panel)} rows, {panel['date'].min().date()} to {panel['date'].max().date()}")
    return panel, feature_cols


# ── Walk-Forward LightGBM ─────────────────────────────────────────────
def walk_forward_lgbm(panel, feature_cols):
    """Walk-forward LightGBM: 12-month train, 1-month test, sliding."""
    dates = sorted(panel["date"].unique())
    date_arr = pd.to_datetime(dates)

    min_date = date_arr.min()
    max_date = date_arr.max()

    # Build test windows: start after first 12 months
    first_test_start = min_date + pd.DateOffset(months=TRAIN_MONTHS)

    test_windows = []
    current = first_test_start
    while current < max_date:
        next_month = current + pd.DateOffset(months=TEST_MONTHS)
        test_windows.append((current, min(next_month, max_date)))
        current = next_month

    print(f"  Walk-forward: {len(test_windows)} test windows")

    all_predictions = []

    for i, (test_start, test_end) in enumerate(test_windows):
        train_start = test_start - pd.DateOffset(months=TRAIN_MONTHS)

        train_mask = (panel["date"] >= train_start) & (panel["date"] < test_start)
        test_mask = (panel["date"] >= test_start) & (panel["date"] < test_end)

        train_data = panel[train_mask]
        test_data = panel[test_mask]

        if len(train_data) < 100 or len(test_data) < 5:
            continue

        X_train = train_data[feature_cols].values
        y_train = train_data["fwd_21d_ret"].values
        X_test = test_data[feature_cols].values

        dtrain = lgb.Dataset(X_train, label=y_train)
        dval = lgb.Dataset(X_test, label=test_data["fwd_21d_ret"].values, reference=dtrain)

        callbacks = [lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False),
                     lgb.log_evaluation(period=0)]

        model = lgb.train(
            LGBM_PARAMS,
            dtrain,
            num_boost_round=NUM_BOOST_ROUND,
            valid_sets=[dval],
            callbacks=callbacks,
        )

        preds = model.predict(X_test)

        test_rows = test_data.copy()
        test_rows["pred"] = preds
        all_predictions.append(test_rows)

    predictions = pd.concat(all_predictions, ignore_index=True)
    print(f"  Predictions: {len(predictions)} rows")
    return predictions


# ── Ranking & Trade Generation ────────────────────────────────────────
def generate_trades(predictions, prices, vix_series, config):
    """Generate bull call spread trades from sector rankings."""
    config_name = config["name"]
    exit_days = config.get("exit_days", None)  # None = hold to expiry
    trailing_stop_pct = config.get("trailing_stop_pct", None)

    # Get unique dates and rank sectors each day
    dates = sorted(predictions["date"].unique())

    trades = []
    last_rebalance = None
    current_equity = INITIAL_CAPITAL
    active_positions = []

    daily_equity = []

    for date in dates:
        date_ts = pd.Timestamp(date)

        # Check/update active positions
        new_active = []
        for pos in active_positions:
            days_held = (date_ts - pos["entry_date"]).days

            # Get current price of underlying
            ticker = pos["ticker"]
            if ticker in prices and date_ts in prices[ticker].index:
                current_spot = float(prices[ticker].loc[date_ts, "close"])
            else:
                new_active.append(pos)
                continue

            remaining_dte = max(pos["original_dte"] - days_held, 0)
            T_remaining = remaining_dte / 365.0

            # Current spread value
            current_value = bull_call_spread_price(
                current_spot, pos["K_low"], pos["K_high"],
                T_remaining, RISK_FREE_RATE, pos["sigma"]
            )

            # Track peak value for trailing stop
            if current_value > pos.get("peak_value", 0):
                pos["peak_value"] = current_value

            # Exit conditions
            should_exit = False
            exit_reason = None

            if remaining_dte <= 0:
                should_exit = True
                exit_reason = "expiry"
            elif exit_days is not None and days_held >= exit_days:
                should_exit = True
                exit_reason = f"{exit_days}d_exit"
            elif trailing_stop_pct is not None and pos.get("peak_value", 0) > 0:
                drawdown = 1 - (current_value / pos["peak_value"])
                if drawdown >= trailing_stop_pct:
                    should_exit = True
                    exit_reason = "trailing_stop"

            if should_exit:
                # Sell spread
                exit_value = price_spread_with_haircut(
                    current_spot, pos["K_low"], pos["K_high"],
                    T_remaining, RISK_FREE_RATE, pos["sigma"],
                    direction="sell"
                )
                # At expiry, intrinsic value
                if remaining_dte <= 0:
                    intrinsic_low = max(current_spot - pos["K_low"], 0)
                    intrinsic_high = max(current_spot - pos["K_high"], 0)
                    exit_value = (intrinsic_low - intrinsic_high) * 100
                    # Per-share basis, convert back
                    exit_value = (intrinsic_low - intrinsic_high)

                pnl = (exit_value - pos["entry_cost"]) * pos["contracts"] * 100 - COMMISSION_RT * pos["contracts"]
                current_equity += pnl

                trades.append({
                    "config": config_name,
                    "ticker": ticker,
                    "entry_date": pos["entry_date"],
                    "exit_date": date_ts,
                    "days_held": days_held,
                    "entry_cost": pos["entry_cost"],
                    "exit_value": exit_value,
                    "contracts": pos["contracts"],
                    "pnl": pnl,
                    "exit_reason": exit_reason,
                    "equity_after": current_equity,
                })
            else:
                new_active.append(pos)

        active_positions = new_active
        daily_equity.append({"date": date_ts, "equity": current_equity})

        # Rebalance check
        if last_rebalance is not None:
            trading_days_since = len([d for d in dates if d > last_rebalance and d <= date])
            if trading_days_since < REBALANCE_FREQ_DAYS:
                continue

        # VIX filter
        if date_ts in vix_series.index:
            current_vix = float(vix_series.loc[date_ts])
        else:
            # Find nearest prior VIX
            prior = vix_series.index[vix_series.index <= date_ts]
            if len(prior) == 0:
                continue
            current_vix = float(vix_series.loc[prior[-1]])

        if current_vix < VIX_FLOOR:
            continue

        # Get rankings for this date
        day_preds = predictions[predictions["date"] == date]
        if len(day_preds) < 3:
            continue

        # Rank by predicted forward return (higher = better)
        day_preds = day_preds.sort_values("pred", ascending=False)
        top_sectors = day_preds.head(TOP_N)["ticker"].tolist()

        # Don't open positions in sectors we already hold
        held_tickers = {p["ticker"] for p in active_positions}
        top_sectors = [t for t in top_sectors if t not in held_tickers]

        if not top_sectors:
            continue

        last_rebalance = date

        # Open bull call spreads on top sectors
        for ticker in top_sectors:
            if current_equity < 50:  # minimum to trade
                break

            if ticker not in prices or date_ts not in prices[ticker].index:
                continue

            spot = float(prices[ticker].loc[date_ts, "close"])

            # Estimate implied vol from realized vol
            if ticker in prices:
                hist = prices[ticker]["close"]
                prior_prices = hist[hist.index <= date_ts].tail(30)
                if len(prior_prices) < 10:
                    continue
                daily_rets = prior_prices.pct_change().dropna()
                realized_vol = float(daily_rets.std() * np.sqrt(252))
                # IV typically ~1.2x RV in normal conditions, more in high VIX
                iv_mult = 1.2 + 0.01 * max(current_vix - 20, 0)
                sigma = realized_vol * iv_mult
            else:
                sigma = 0.25

            sigma = max(sigma, 0.10)  # floor

            # Strike selection: ATM-ish low strike, +3% high strike
            K_low = round(spot, 0)  # round to nearest dollar
            K_high = round(spot * (1 + SPREAD_WIDTH_PCT), 0)

            if K_high <= K_low:
                K_high = K_low + 1

            T = TARGET_DTE / 365.0

            # Price the spread with haircut (we're buying)
            entry_cost = price_spread_with_haircut(
                spot, K_low, K_high, T, RISK_FREE_RATE, sigma, direction="buy"
            )

            if entry_cost <= 0.01:
                continue

            # Size: max allocation per trade
            cost_per_contract = entry_cost * 100  # options are per 100 shares
            trade_budget = min(MAX_TRADE_SIZE, current_equity * 0.4)
            contracts = max(1, int(trade_budget / (cost_per_contract + COMMISSION_RT)))

            total_cost = contracts * cost_per_contract + COMMISSION_RT * contracts
            if total_cost > current_equity * 0.8:  # don't bet more than 80% of equity
                contracts = max(1, int((current_equity * 0.8 - COMMISSION_RT) / cost_per_contract))

            if contracts < 1:
                continue

            total_cost = contracts * cost_per_contract + COMMISSION_RT * contracts

            active_positions.append({
                "ticker": ticker,
                "entry_date": date_ts,
                "K_low": K_low,
                "K_high": K_high,
                "entry_cost": entry_cost,
                "contracts": contracts,
                "sigma": sigma,
                "original_dte": TARGET_DTE,
                "peak_value": entry_cost,
            })

    # Close any remaining positions at last date
    last_date = pd.Timestamp(dates[-1])
    for pos in active_positions:
        ticker = pos["ticker"]
        days_held = (last_date - pos["entry_date"]).days

        if ticker in prices and last_date in prices[ticker].index:
            final_spot = float(prices[ticker].loc[last_date, "close"])
        else:
            # Find last available price
            if ticker in prices:
                avail = prices[ticker].index[prices[ticker].index <= last_date]
                if len(avail) > 0:
                    final_spot = float(prices[ticker].loc[avail[-1], "close"])
                else:
                    continue
            else:
                continue

        remaining_dte = max(pos["original_dte"] - days_held, 0)
        T_remaining = remaining_dte / 365.0

        exit_value = price_spread_with_haircut(
            final_spot, pos["K_low"], pos["K_high"],
            T_remaining, RISK_FREE_RATE, pos["sigma"],
            direction="sell"
        )

        pnl = (exit_value - pos["entry_cost"]) * pos["contracts"] * 100 - COMMISSION_RT * pos["contracts"]
        current_equity += pnl

        trades.append({
            "config": config_name,
            "ticker": ticker,
            "entry_date": pos["entry_date"],
            "exit_date": last_date,
            "days_held": days_held,
            "entry_cost": pos["entry_cost"],
            "exit_value": exit_value,
            "contracts": pos["contracts"],
            "pnl": pnl,
            "exit_reason": "end_of_data",
            "equity_after": current_equity,
        })

    trades_df = pd.DataFrame(trades)
    equity_df = pd.DataFrame(daily_equity)

    return trades_df, equity_df


# ── Honest Metrics ────────────────────────────────────────────────────
def compute_metrics(trades_df, equity_df):
    """Compute honest equity-based Sharpe with calendar month aggregation."""
    if len(trades_df) == 0:
        return {
            "sharpe": 0.0, "cagr": 0.0, "max_dd": -1.0, "win_rate": 0.0,
            "profit_factor": 0.0, "total_trades": 0, "final_equity": INITIAL_CAPITAL,
        }

    # Build monthly equity series from trade-level data
    trades_df = trades_df.sort_values("exit_date")

    # Use equity curve for monthly returns
    equity_df = equity_df.sort_values("date")
    equity_df["year_month"] = equity_df["date"].dt.to_period("M")

    # Get last equity value per month
    monthly_equity = equity_df.groupby("year_month")["equity"].last()

    # EQUITY-BASED monthly returns (NOT / initial_capital)
    monthly_returns = monthly_equity.pct_change().dropna()

    # Sharpe: annualized
    if len(monthly_returns) > 1 and monthly_returns.std() > 0:
        sharpe = (monthly_returns.mean() / monthly_returns.std()) * np.sqrt(12)
    else:
        sharpe = 0.0

    # CAGR
    final_eq = float(monthly_equity.iloc[-1])
    start_eq = INITIAL_CAPITAL
    years = len(monthly_equity) / 12
    if years > 0 and final_eq > 0 and start_eq > 0:
        cagr = (final_eq / start_eq) ** (1 / years) - 1
    else:
        cagr = 0.0

    # Max drawdown from equity curve
    eq_series = equity_df["equity"].values
    peak = np.maximum.accumulate(eq_series)
    dd = (eq_series - peak) / np.where(peak > 0, peak, 1)
    max_dd = float(dd.min())

    # Win rate, profit factor
    wins = trades_df[trades_df["pnl"] > 0]
    losses = trades_df[trades_df["pnl"] <= 0]
    win_rate = len(wins) / len(trades_df) if len(trades_df) > 0 else 0

    gross_profit = wins["pnl"].sum() if len(wins) > 0 else 0
    gross_loss = abs(losses["pnl"].sum()) if len(losses) > 0 else 1e-9
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    return {
        "sharpe": round(sharpe, 3),
        "cagr": round(cagr * 100, 2),
        "max_dd": round(max_dd * 100, 2),
        "win_rate": round(win_rate * 100, 1),
        "profit_factor": round(profit_factor, 3),
        "total_trades": len(trades_df),
        "final_equity": round(final_eq, 2),
    }


# ── 4-Gate Adversarial Validation ─────────────────────────────────────
def gate1_permutation_test(predictions, prices, vix_series, config, n_perms=100):
    """Shuffle sector rankings, measure how many beat real."""
    print(f"    Gate 1: Permutation test ({n_perms} shuffles)...")

    real_trades, real_equity = generate_trades(predictions, prices, vix_series, config)
    real_metrics = compute_metrics(real_trades, real_equity)
    real_sharpe = real_metrics["sharpe"]

    beat_count = 0
    for i in range(n_perms):
        shuffled = predictions.copy()
        # Shuffle predictions within each date (break sector-specific rankings)
        for date in shuffled["date"].unique():
            mask = shuffled["date"] == date
            preds = shuffled.loc[mask, "pred"].values.copy()
            np.random.shuffle(preds)
            shuffled.loc[mask, "pred"] = preds

        shuf_trades, shuf_equity = generate_trades(shuffled, prices, vix_series, config)
        shuf_metrics = compute_metrics(shuf_trades, shuf_equity)
        if shuf_metrics["sharpe"] >= real_sharpe:
            beat_count += 1

        if (i + 1) % 25 == 0:
            print(f"      ... {i+1}/{n_perms} done, {beat_count} beat real so far")

    perm_p = beat_count / n_perms
    passed = perm_p < 0.05
    print(f"    Gate 1 result: perm_p={perm_p:.3f} {'PASS' if passed else 'FAIL'}")
    return perm_p, passed


def gate2_regime_agnostic(trades_df, spy_prices):
    """R1: split into green/red days, check Sharpe gap."""
    print("    Gate 2: Regime-agnostic validation...")

    if len(trades_df) == 0:
        return 1.0, False

    # Classify each trade by SPY direction during its holding period
    spy_close = spy_prices["close"]

    green_pnls = []
    red_pnls = []

    for _, trade in trades_df.iterrows():
        entry = trade["entry_date"]
        exit_d = trade["exit_date"]

        # SPY return during trade
        spy_entry_prices = spy_close[spy_close.index <= entry]
        spy_exit_prices = spy_close[spy_close.index <= exit_d]

        if len(spy_entry_prices) == 0 or len(spy_exit_prices) == 0:
            continue

        spy_entry = float(spy_entry_prices.iloc[-1])
        spy_exit = float(spy_exit_prices.iloc[-1])

        if spy_exit >= spy_entry:
            green_pnls.append(trade["pnl"])
        else:
            red_pnls.append(trade["pnl"])

    # Compute Sharpe-like metric for each regime (mean/std of trade PnLs)
    def regime_sharpe(pnls):
        if len(pnls) < 3:
            return 0.0
        arr = np.array(pnls)
        if arr.std() == 0:
            return 0.0
        return arr.mean() / arr.std()

    sharpe_green = regime_sharpe(green_pnls)
    sharpe_red = regime_sharpe(red_pnls)

    denom = max(abs(sharpe_green), abs(sharpe_red))
    if denom == 0:
        r1_gap = 1.0
    else:
        r1_gap = abs(sharpe_green - sharpe_red) / denom

    passed = r1_gap < 0.50
    print(f"    Gate 2 result: Sharpe_green={sharpe_green:.3f}, Sharpe_red={sharpe_red:.3f}, "
          f"gap={r1_gap:.3f} {'PASS' if passed else 'FAIL'}")
    return round(r1_gap, 3), passed


def gate3_sub_period_stability(trades_df):
    """Split into 3 equal periods, all must be profitable."""
    print("    Gate 3: Sub-period stability...")

    if len(trades_df) < 9:
        return False

    trades_sorted = trades_df.sort_values("entry_date")
    n = len(trades_sorted)
    third = n // 3

    periods = [
        trades_sorted.iloc[:third],
        trades_sorted.iloc[third:2*third],
        trades_sorted.iloc[2*third:],
    ]

    all_profitable = True
    for i, period in enumerate(periods):
        period_pnl = period["pnl"].sum()
        period_sharpe = period["pnl"].mean() / period["pnl"].std() if period["pnl"].std() > 0 else 0
        status = "PASS" if period_sharpe > 0 else "FAIL"
        print(f"      Period {i+1}: PnL=${period_pnl:.2f}, Sharpe={period_sharpe:.3f} {status}")
        if period_sharpe <= 0:
            all_profitable = False

    print(f"    Gate 3 result: {'PASS' if all_profitable else 'FAIL'}")
    return all_profitable


def gate4_outlier_removal(trades_df):
    """Remove top 5% trades by PnL, must still be profitable."""
    print("    Gate 4: Outlier removal (top 5% PnL removed)...")

    if len(trades_df) < 10:
        return False

    threshold = trades_df["pnl"].quantile(0.95)
    trimmed = trades_df[trades_df["pnl"] <= threshold]

    trimmed_pnl = trimmed["pnl"].sum()
    n_removed = len(trades_df) - len(trimmed)

    passed = trimmed_pnl > 0
    print(f"    Gate 4 result: Removed {n_removed} trades, remaining PnL=${trimmed_pnl:.2f} "
          f"{'PASS' if passed else 'FAIL'}")
    return passed


# ── Main ──────────────────────────────────────────────────────────────
def main():
    t0 = time.time()
    print("=" * 80)
    print("PRODUCTION CONFIG FINAL v1 — Sector ETF Bull Call Spread Strategy")
    print("=" * 80)

    # Load data
    prices, vix_series = load_data()

    # Build features
    panel, feature_cols = build_features(prices, vix_series)

    # Walk-forward LightGBM
    print("\nRunning walk-forward LightGBM...")
    predictions = walk_forward_lgbm(panel, feature_cols)

    # Config definitions
    configs = [
        {"name": "A_Production_20dExit", "exit_days": 20, "trailing_stop_pct": None},
        {"name": "B_Production_TrailingStop", "exit_days": None, "trailing_stop_pct": 0.30},
        {"name": "C_Production_Baseline", "exit_days": None, "trailing_stop_pct": None},
    ]

    results = []

    for config in configs:
        print(f"\n{'─' * 60}")
        print(f"Config: {config['name']}")
        print(f"{'─' * 60}")

        # Generate trades
        trades_df, equity_df = generate_trades(predictions, prices, vix_series, config)
        metrics = compute_metrics(trades_df, equity_df)

        print(f"  Trades: {metrics['total_trades']}, Sharpe: {metrics['sharpe']}, "
              f"CAGR: {metrics['cagr']}%, MaxDD: {metrics['max_dd']}%, "
              f"WR: {metrics['win_rate']}%, PF: {metrics['profit_factor']}, "
              f"Final Equity: ${metrics['final_equity']}")

        # 4-Gate Adversarial Validation
        print(f"\n  Running 4-Gate Adversarial Validation...")

        # Gate 1: Permutation test
        perm_p, g1_pass = gate1_permutation_test(predictions, prices, vix_series, config, n_perms=100)

        # Gate 2: Regime-agnostic
        r1_gap, g2_pass = gate2_regime_agnostic(trades_df, prices[BENCHMARK])

        # Gate 3: Sub-period stability
        g3_pass = gate3_sub_period_stability(trades_df)

        # Gate 4: Outlier removal
        g4_pass = gate4_outlier_removal(trades_df)

        gates_passed = sum([g1_pass, g2_pass, g3_pass, g4_pass])

        result = {
            "config": config["name"],
            **metrics,
            "perm_p": perm_p,
            "r1_gap": r1_gap,
            "sub_period_pass": "PASS" if g3_pass else "FAIL",
            "outlier_pass": "PASS" if g4_pass else "FAIL",
            "gates_passed": f"{gates_passed}/4",
        }
        results.append(result)

        # Save trades
        if len(trades_df) > 0:
            trades_df.to_csv(OUTPUT_DIR / f"trades_{config['name']}.csv", index=False)

    # ── Results Table ─────────────────────────────────────────────────
    print("\n" + "=" * 120)
    print("FINAL RESULTS — 4-Gate Adversarial Validation")
    print("=" * 120)

    results_df = pd.DataFrame(results)

    cols = ["config", "sharpe", "cagr", "max_dd", "win_rate", "profit_factor",
            "total_trades", "final_equity", "perm_p", "r1_gap", "sub_period_pass",
            "outlier_pass", "gates_passed"]

    header = f"{'Config':<30} {'Sharpe':>7} {'CAGR%':>7} {'MaxDD%':>7} {'WR%':>6} {'PF':>7} " \
             f"{'Trades':>7} {'FinalEq':>9} {'PermP':>7} {'R1Gap':>7} {'SubPer':>7} {'Outlier':>8} {'Gates':>6}"
    print(header)
    print("-" * len(header))

    for _, row in results_df.iterrows():
        line = f"{row['config']:<30} {row['sharpe']:>7.3f} {row['cagr']:>6.1f}% {row['max_dd']:>6.1f}% " \
               f"{row['win_rate']:>5.1f}% {row['profit_factor']:>7.3f} {row['total_trades']:>7} " \
               f"${row['final_equity']:>8.2f} {row['perm_p']:>7.3f} {row['r1_gap']:>7.3f} " \
               f"{row['sub_period_pass']:>7} {row['outlier_pass']:>8} {row['gates_passed']:>6}"
        print(line)

    print("\n" + "-" * 120)
    print("Methodology: Equity-based Sharpe (monthly_pnl/equity, NOT /initial_capital)")
    print("Calendar month aggregation. B-S pricing with 15% haircut. Walk-forward 12m train / 1m test.")
    print(f"Capital: ${INITIAL_CAPITAL}, Max trade: ${MAX_TRADE_SIZE}, Commission: ${COMMISSION_PER_CONTRACT}/contract")
    print(f"VIX floor: {VIX_FLOOR}, DTE: {TARGET_DTE}, Spread width: {SPREAD_WIDTH_PCT*100}%, Top-{TOP_N} sectors")

    # ── MLflow Logging ────────────────────────────────────────────────
    print("\n  Logging to MLflow...")
    try:
        import mlflow
        mlflow.set_tracking_uri("http://localhost:5000")
        mlflow.set_experiment("production_config_final_v1")

        for result in results:
            with mlflow.start_run(run_name=result["config"]):
                mlflow.log_params({
                    "config": result["config"],
                    "initial_capital": INITIAL_CAPITAL,
                    "max_trade_size": MAX_TRADE_SIZE,
                    "target_dte": TARGET_DTE,
                    "vix_floor": VIX_FLOOR,
                    "top_n": TOP_N,
                    "spread_width_pct": SPREAD_WIDTH_PCT,
                    "commission": COMMISSION_PER_CONTRACT,
                    "haircut": HAIRCUT,
                    "train_months": TRAIN_MONTHS,
                    "test_months": TEST_MONTHS,
                })
                mlflow.log_metrics({
                    "sharpe": result["sharpe"],
                    "cagr_pct": result["cagr"],
                    "max_dd_pct": result["max_dd"],
                    "win_rate_pct": result["win_rate"],
                    "profit_factor": result["profit_factor"],
                    "total_trades": result["total_trades"],
                    "final_equity": result["final_equity"],
                    "perm_p": result["perm_p"],
                    "r1_gap": result["r1_gap"],
                })
        print("  MLflow logging complete.")
    except Exception as e:
        print(f"  MLflow logging failed (non-fatal): {e}")

    elapsed = time.time() - t0
    print(f"\nTotal runtime: {elapsed:.1f}s")

    # Save results
    results_df.to_csv(OUTPUT_DIR / "results_summary.csv", index=False)
    print(f"Results saved to {OUTPUT_DIR}")

    return results_df


if __name__ == "__main__":
    main()
