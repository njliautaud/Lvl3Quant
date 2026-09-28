#!/usr/bin/env python3
"""
Sector Backtest Engine — Standardized Bull Call Spread Backtesting
===================================================================

SINGLE SOURCE OF TRUTH for all sector spread backtests.
No more 5 different scripts with 5 different Sharpe calculations.

Uses:
  - options_pricer module for consistent pricing with bid-ask haircuts
  - adversarial_validator module for honest 5-gate validation
  - Random baseline comparison by default (QUANT_KNOWLEDGE_BASE finding #16)

The standard strategy:
  1. Walk-forward LGBM ranking of sector ETFs (12-month train, 1-month test)
  2. Bull call spreads on top-N sectors when VIX > threshold
  3. 30 DTE, 3% spread width, hold to expiry (configurable)
  4. Biweekly rebalancing

Usage:
    from research.tools.sector_backtest import run_sector_backtest

    results = run_sector_backtest(
        config={"vix_floor": 20, "spread_width_pct": 0.03, "top_n": 3},
        start_date="2011-01-01",
        end_date="2026-07-25",
    )
    # results contains: trades, metrics, validation, random_baseline
"""
from __future__ import annotations

import json
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from .options_pricer import (
    price_bull_call_spread,
    exit_spread_value,
    spread_pnl,
    estimate_iv,
    COMMISSION_RT_SPREAD,
)
from .adversarial_validator import validate_trades, ValidationResult


# ─── Default Configuration ───────────────────────────────────────────

DEFAULT_CONFIG = {
    # Universe
    "sector_etfs": ["XLE", "XLK", "XLF", "XLV", "XLI", "XLP", "XLU", "XLRE", "XLB", "XLC", "XLY"],
    "benchmark": "SPY",

    # Dates
    "start_date": "2011-01-01",
    "end_date": "2026-07-25",

    # Capital
    "initial_capital": 645.0,
    "max_trade_pct": 0.40,       # max % of equity per trade
    "max_total_pct": 0.80,       # max % of equity in all positions

    # Strategy
    "target_dte": 30,
    "vix_floor": 20.0,
    "top_n": 3,
    "rebalance_freq_days": 10,   # trading days between rebalances
    "spread_width_pct": 0.03,    # 3% OTM spread width

    # Pricing
    "haircut": 0.15,
    "commission_rt": COMMISSION_RT_SPREAD,

    # Walk-forward LGBM
    "train_months": 12,
    "test_months": 1,
    "lgbm_params": {
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
    },
    "num_boost_round": 300,
    "early_stopping_rounds": 30,

    # Validation
    "n_perms": 500,              # permutations for validation (2000 for final, 500 for exploration)
    "run_random_baseline": True, # always compare against random
}


# ─── Data Loading ────────────────────────────────────────────────────

def load_market_data(
    tickers: list[str],
    start_date: str,
    end_date: str,
) -> tuple[dict, pd.Series]:
    """
    Load price data and VIX from yfinance.

    Returns:
        (prices_dict, vix_series): prices_dict maps ticker -> DataFrame with 'close', 'high', 'low', 'volume'.
        vix_series is pd.Series of VIX close values.
    """
    import yfinance as yf

    all_tickers = list(set(tickers))
    print(f"  Loading {len(all_tickers)} tickers from {start_date} to {end_date}...")

    data = yf.download(
        all_tickers, start=start_date, end=end_date,
        auto_adjust=True, progress=False, group_by="ticker"
    )

    prices = {}
    for ticker in all_tickers:
        try:
            if len(all_tickers) > 1:
                df = data[ticker][["Close", "High", "Low", "Volume"]].dropna()
            else:
                df = data[["Close", "High", "Low", "Volume"]].dropna()
            df.columns = ["close", "high", "low", "volume"]
            if len(df) > 60:
                prices[ticker] = df
        except Exception:
            pass

    # VIX
    vix_data = yf.download("^VIX", start=start_date, end=end_date,
                           auto_adjust=True, progress=False)
    vix_series = vix_data["Close"].dropna()
    if hasattr(vix_series, "columns"):
        vix_series = vix_series.iloc[:, 0]

    print(f"  Got {len(prices)} tickers, VIX has {len(vix_series)} days")
    return prices, vix_series


# ─── Feature Engineering ────────────────────────────────────────────

def build_features(
    prices: dict,
    vix_series: pd.Series,
    sector_etfs: list[str],
    benchmark: str = "SPY",
) -> pd.DataFrame:
    """Build feature panel for LGBM sector ranking."""
    if benchmark not in prices:
        raise ValueError(f"Benchmark {benchmark} not in price data")

    spy_close = prices[benchmark]["close"]
    records = []

    for ticker in sector_etfs:
        if ticker not in prices:
            continue
        df = prices[ticker].copy()
        close = df["close"]
        volume = df["volume"]

        # Momentum features
        for lookback in [5, 10, 21, 63]:
            df[f"mom_{lookback}d"] = close.pct_change(lookback)

        # Relative strength vs SPY
        for lookback in [21, 63]:
            spy_ret = spy_close.pct_change(lookback)
            ticker_ret = close.pct_change(lookback)
            aligned = pd.DataFrame({"ticker": ticker_ret, "spy": spy_ret}).dropna()
            df[f"rs_vs_spy_{lookback}d"] = ticker_ret - spy_ret.reindex(ticker_ret.index)

        # Volatility
        df["vol_21d"] = close.pct_change().rolling(21).std() * np.sqrt(252)
        df["vol_63d"] = close.pct_change().rolling(63).std() * np.sqrt(252)
        df["vol_ratio"] = df["vol_21d"] / (df["vol_63d"] + 1e-8)

        # Volume trend
        df["vol_sma_ratio"] = volume / volume.rolling(21).mean()

        # RSI
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(alpha=1 / 14, min_periods=14).mean()
        avg_loss = loss.ewm(alpha=1 / 14, min_periods=14).mean()
        df["rsi_14"] = 100 - (100 / (1 + avg_gain / (avg_loss + 1e-8)))

        # SMA position
        df["sma_50_dist"] = close / close.rolling(50).mean() - 1
        df["sma_200_dist"] = close / close.rolling(200).mean() - 1

        # Forward return (target)
        df["fwd_return_21d"] = close.pct_change(21).shift(-21)

        # VIX
        vix_aligned = vix_series.reindex(df.index, method="ffill")
        df["vix"] = vix_aligned

        # ATR
        tr1 = df["high"] - df["low"]
        tr2 = (df["high"] - close.shift(1)).abs()
        tr3 = (df["low"] - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df["atr_14"] = tr.ewm(alpha=1 / 14, min_periods=14).mean()

        for idx, row in df.iterrows():
            record = {"date": idx, "ticker": ticker}
            feature_cols = [c for c in df.columns if c not in
                           ["close", "high", "low", "volume", "fwd_return_21d", "atr_14", "vix"]]
            for col in feature_cols:
                record[col] = row[col]
            record["target"] = row["fwd_return_21d"]
            record["close"] = row["close"]
            record["atr_14"] = row["atr_14"]
            record["vix"] = row["vix"]
            records.append(record)

    panel = pd.DataFrame(records).dropna(subset=["target"])
    print(f"  Feature panel: {len(panel)} rows, {len(panel.columns)} columns")
    return panel


# ─── Walk-Forward LGBM ──────────────────────────────────────────────

def walk_forward_lgbm(
    panel: pd.DataFrame,
    config: dict,
) -> pd.DataFrame:
    """
    Walk-forward LGBM ranking. Sliding window (not expanding).
    Returns DataFrame with columns: date, ticker, pred, close, atr_14, vix.
    """
    import lightgbm as lgb

    feature_cols = [c for c in panel.columns if c not in
                    ["date", "ticker", "target", "close", "atr_14", "vix"]]

    panel = panel.sort_values("date")
    dates = sorted(panel["date"].unique())

    train_months = config.get("train_months", 12)
    test_months = config.get("test_months", 1)

    predictions = []
    n_folds = 0

    for i in range(train_months, len(dates), max(1, int(21 * test_months))):
        if i >= len(dates):
            break

        # Train window: sliding
        train_start_idx = max(0, i - train_months * 21)
        train_dates = dates[train_start_idx:i]
        test_end_idx = min(len(dates), i + test_months * 21)
        test_dates = dates[i:test_end_idx]

        if len(train_dates) < 100 or len(test_dates) == 0:
            continue

        train_mask = panel["date"].isin(train_dates)
        test_mask = panel["date"].isin(test_dates)

        X_train = panel.loc[train_mask, feature_cols].values
        y_train = panel.loc[train_mask, "target"].values
        X_test = panel.loc[test_mask, feature_cols].values

        # Handle NaN/inf
        X_train = np.nan_to_num(X_train, nan=0, posinf=0, neginf=0)
        y_train = np.nan_to_num(y_train, nan=0, posinf=0, neginf=0)
        X_test = np.nan_to_num(X_test, nan=0, posinf=0, neginf=0)

        try:
            dtrain = lgb.Dataset(X_train, label=y_train)
            dval = lgb.Dataset(X_train[-len(X_train) // 5:],
                               label=y_train[-len(y_train) // 5:])

            model = lgb.train(
                config.get("lgbm_params", DEFAULT_CONFIG["lgbm_params"]),
                dtrain,
                num_boost_round=config.get("num_boost_round", 300),
                valid_sets=[dval],
                callbacks=[lgb.early_stopping(
                    config.get("early_stopping_rounds", 30),
                    verbose=False
                )],
            )

            preds = model.predict(X_test)
            test_data = panel.loc[test_mask, ["date", "ticker", "close", "atr_14", "vix"]].copy()
            test_data["pred"] = preds
            predictions.append(test_data)
            n_folds += 1

        except Exception:
            continue

    if not predictions:
        return pd.DataFrame()

    result = pd.concat(predictions, ignore_index=True)
    print(f"  Walk-forward: {n_folds} folds, {len(result)} predictions")
    return result


# ─── Trade Generation ────────────────────────────────────────────────

def generate_trades(
    predictions: pd.DataFrame,
    prices: dict,
    vix_series: pd.Series,
    config: dict,
    use_random_selection: bool = False,
) -> list[dict]:
    """
    Generate bull call spread trades from LGBM predictions.

    If use_random_selection=True, shuffles sector rankings at each rebalance
    date to create a random baseline for comparison.

    Returns list of trade dicts with: ticker, entry_date, exit_date, pnl,
    entry_cost, exit_value, contracts, K1, K2, exit_reason.
    """
    initial_capital = config.get("initial_capital", 645.0)
    vix_floor = config.get("vix_floor", 20.0)
    top_n = config.get("top_n", 3)
    target_dte = config.get("target_dte", 30)
    rebalance_freq = config.get("rebalance_freq_days", 10)
    spread_width_pct = config.get("spread_width_pct", 0.03)
    haircut = config.get("haircut", 0.15)
    commission_rt = config.get("commission_rt", COMMISSION_RT_SPREAD)
    max_trade_pct = config.get("max_trade_pct", 0.40)
    max_total_pct = config.get("max_total_pct", 0.80)

    current_equity = initial_capital
    active_positions = []
    trades = []
    last_rebalance = None

    pred_dates = sorted(predictions["date"].unique())

    for date in pred_dates:
        date_ts = pd.Timestamp(date)

        # Check and exit existing positions
        new_active = []
        for pos in active_positions:
            days_held = (date_ts - pos["entry_date"]).days
            ticker = pos["ticker"]

            if ticker not in prices or date_ts not in prices[ticker].index:
                new_active.append(pos)
                continue

            spot = float(prices[ticker].loc[date_ts, "close"])
            remaining_dte = max(pos["original_dte"] - days_held, 0)

            if remaining_dte <= 0:
                # Expiry: intrinsic value (no haircut)
                exit_val = exit_spread_value(
                    spot, pos["K1"], pos["K2"], 0, pos["original_dte"],
                    pos["atr"], pos.get("vix", 20.0), haircut
                )
                pnl = spread_pnl(pos["entry_cost"], exit_val, pos["contracts"], commission_rt)
                current_equity += pnl

                trades.append({
                    "ticker": ticker,
                    "entry_date": pos["entry_date"],
                    "exit_date": date_ts,
                    "days_held": days_held,
                    "entry_cost": pos["entry_cost"],
                    "exit_value": exit_val,
                    "contracts": pos["contracts"],
                    "K1": pos["K1"],
                    "K2": pos["K2"],
                    "pnl": pnl,
                    "exit_reason": "expiry",
                    "equity_after": current_equity,
                })
            else:
                new_active.append(pos)

        active_positions = new_active

        # Rebalance check
        if last_rebalance is not None:
            trading_days_since = len([d for d in pred_dates if d > last_rebalance and d <= date])
            if trading_days_since < rebalance_freq:
                continue

        # VIX filter
        if date_ts in vix_series.index:
            current_vix = float(vix_series.loc[date_ts])
        else:
            prior = vix_series.index[vix_series.index <= date_ts]
            if len(prior) == 0:
                continue
            current_vix = float(vix_series.loc[prior[-1]])

        if current_vix < vix_floor:
            continue

        # Get rankings
        day_preds = predictions[predictions["date"] == date].copy()
        if len(day_preds) < 3:
            continue

        if use_random_selection:
            # Shuffle predictions to create random baseline
            preds_shuffled = day_preds["pred"].values.copy()
            np.random.shuffle(preds_shuffled)
            day_preds["pred"] = preds_shuffled

        day_preds = day_preds.sort_values("pred", ascending=False)
        top_sectors = day_preds.head(top_n)["ticker"].tolist()

        # Don't duplicate positions
        held_tickers = {p["ticker"] for p in active_positions}
        top_sectors = [t for t in top_sectors if t not in held_tickers]

        if not top_sectors:
            continue

        last_rebalance = date

        for ticker in top_sectors:
            if current_equity < 50:
                break

            if ticker not in prices or date_ts not in prices[ticker].index:
                continue

            spot = float(prices[ticker].loc[date_ts, "close"])

            # Get ATR
            hist = prices[ticker]
            prior = hist[hist.index <= date_ts].tail(30)
            if len(prior) < 14:
                continue

            from .options_pricer import compute_atr as _compute_atr
            atr_val = _compute_atr(prior["high"], prior["low"], prior["close"])

            # Strikes
            K1 = round(spot, 0)
            K2 = round(spot * (1 + spread_width_pct), 0)
            if K2 <= K1:
                K2 = K1 + 1

            # Price the spread
            entry_cost, max_profit = price_bull_call_spread(
                spot, K1, K2, target_dte, atr_val, current_vix, haircut
            )

            if entry_cost <= 0.01 or max_profit <= 0:
                continue

            # Position sizing
            cost_per_contract = entry_cost * 100
            trade_budget = min(current_equity * max_trade_pct, 250.0)
            contracts = max(1, int(trade_budget / (cost_per_contract + commission_rt)))

            total_cost = contracts * cost_per_contract + commission_rt * contracts
            if total_cost > current_equity * max_total_pct:
                contracts = max(1, int(
                    (current_equity * max_total_pct - commission_rt) / cost_per_contract
                ))

            if contracts < 1:
                continue

            active_positions.append({
                "ticker": ticker,
                "entry_date": date_ts,
                "K1": K1,
                "K2": K2,
                "entry_cost": entry_cost,
                "contracts": contracts,
                "original_dte": target_dte,
                "atr": atr_val,
                "vix": current_vix,
            })

    # Force-close any remaining positions at last date
    last_date = pd.Timestamp(pred_dates[-1]) if pred_dates else pd.Timestamp.now()
    for pos in active_positions:
        ticker = pos["ticker"]
        days_held = (last_date - pos["entry_date"]).days
        remaining_dte = max(pos["original_dte"] - days_held, 0)

        if ticker in prices:
            avail = prices[ticker].index[prices[ticker].index <= last_date]
            if len(avail) > 0:
                spot = float(prices[ticker].loc[avail[-1], "close"])
                exit_val = exit_spread_value(
                    spot, pos["K1"], pos["K2"], remaining_dte, pos["original_dte"],
                    pos["atr"], pos.get("vix", 20.0), haircut
                )
                pnl = spread_pnl(pos["entry_cost"], exit_val, pos["contracts"], commission_rt)
                current_equity += pnl

                trades.append({
                    "ticker": ticker,
                    "entry_date": pos["entry_date"],
                    "exit_date": last_date,
                    "days_held": days_held,
                    "entry_cost": pos["entry_cost"],
                    "exit_value": exit_val,
                    "contracts": pos["contracts"],
                    "K1": pos["K1"],
                    "K2": pos["K2"],
                    "pnl": pnl,
                    "exit_reason": "end_of_data",
                    "equity_after": current_equity,
                })

    return trades


# ─── Main Entry Point ───────────────────────────────────────────────

def run_sector_backtest(
    config: dict | None = None,
    prices: dict | None = None,
    vix_series: pd.Series | None = None,
    predictions: pd.DataFrame | None = None,
    spy_prices: pd.Series | None = None,
    verbose: bool = True,
) -> dict:
    """
    Run a complete sector bull call spread backtest with adversarial validation.

    You can provide pre-loaded data (prices, vix_series, predictions) to skip
    data loading and model training. Or pass just a config and let this function
    handle everything.

    Args:
        config: Strategy configuration dict (merged with DEFAULT_CONFIG).
        prices: Pre-loaded price data dict (ticker -> DataFrame).
        vix_series: Pre-loaded VIX series.
        predictions: Pre-computed LGBM predictions DataFrame.
        spy_prices: SPY close series for regime classification.
        verbose: Print progress.

    Returns:
        dict with keys:
            - 'trades': list of trade dicts
            - 'metrics': honest metrics dict
            - 'validation': ValidationResult object
            - 'random_baseline': ValidationResult for random selection (or None)
            - 'config': the config used
            - 'ml_vs_random': dict comparing ML vs random Sharpe
    """
    # Merge with defaults
    cfg = {**DEFAULT_CONFIG}
    if config:
        cfg.update(config)

    all_tickers = cfg["sector_etfs"] + [cfg["benchmark"]]

    # Load data if not provided
    if prices is None or vix_series is None:
        if verbose:
            print("Step 1: Loading market data...")
        prices, vix_series = load_market_data(all_tickers, cfg["start_date"], cfg["end_date"])

    # Build features and run LGBM if predictions not provided
    if predictions is None:
        if verbose:
            print("Step 2: Building features...")
        panel = build_features(prices, vix_series, cfg["sector_etfs"], cfg["benchmark"])

        if verbose:
            print("Step 3: Walk-forward LGBM ranking...")
        predictions = walk_forward_lgbm(panel, cfg)

        if len(predictions) == 0:
            return {
                "trades": [],
                "metrics": {"error": "No predictions generated"},
                "validation": None,
                "random_baseline": None,
                "config": cfg,
                "ml_vs_random": None,
            }

    # Get SPY close for regime classification
    if spy_prices is None and cfg["benchmark"] in prices:
        spy_prices = prices[cfg["benchmark"]]["close"]

    # Generate trades
    if verbose:
        print("Step 4: Generating trades...")
    trades = generate_trades(predictions, prices, vix_series, cfg, use_random_selection=False)

    if verbose:
        print(f"  Generated {len(trades)} trades")

    # Validate
    if verbose:
        print("Step 5: Adversarial validation...")
    validation = validate_trades(
        trades,
        initial_capital=cfg["initial_capital"],
        spy_prices=spy_prices,
        strategy_name="Sector Bull Call Spreads (LGBM)",
        n_perms=cfg["n_perms"],
    )

    if verbose:
        validation.print_summary()

    # Random baseline
    random_validation = None
    ml_vs_random = None

    if cfg.get("run_random_baseline", True) and len(trades) >= 10:
        if verbose:
            print("\nStep 6: Random baseline comparison...")

        random_trades = generate_trades(
            predictions, prices, vix_series, cfg, use_random_selection=True
        )

        if len(random_trades) >= 10:
            random_validation = validate_trades(
                random_trades,
                initial_capital=cfg["initial_capital"],
                spy_prices=spy_prices,
                strategy_name="Random Sector Selection (baseline)",
                n_perms=min(cfg["n_perms"], 200),  # fewer perms for baseline
            )

            if verbose:
                random_validation.print_summary()

            # Compare
            ml_sharpe = validation.sharpe
            rand_sharpe = random_validation.sharpe
            incremental = ((ml_sharpe - rand_sharpe) / abs(rand_sharpe) * 100
                          if abs(rand_sharpe) > 0.01 else float("inf"))

            ml_vs_random = {
                "ml_sharpe": round(ml_sharpe, 3),
                "random_sharpe": round(rand_sharpe, 3),
                "incremental_pct": round(incremental, 1),
                "ml_wins": ml_sharpe > rand_sharpe,
                "interpretation": (
                    f"ML adds {incremental:.0f}% incremental Sharpe over random. "
                    f"{'ML ranking adds genuine value.' if incremental > 15 else 'Most edge is structural (VIX filter + spread structure).'}"
                ),
            }

            if verbose:
                print(f"\n  ML vs Random: ML Sharpe={ml_sharpe:.2f}, Random Sharpe={rand_sharpe:.2f}")
                print(f"  {ml_vs_random['interpretation']}")

    return {
        "trades": trades,
        "metrics": validation.to_dict() if validation else {},
        "validation": validation,
        "random_baseline": random_validation,
        "config": {k: v for k, v in cfg.items() if k != "lgbm_params"},  # skip large nested dict
        "ml_vs_random": ml_vs_random,
    }


# ─── Self-Test ───────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 65)
    print("  SECTOR BACKTEST ENGINE — Self-Test")
    print("=" * 65)

    # Quick test with reduced parameters
    results = run_sector_backtest(
        config={
            "start_date": "2018-01-01",  # shorter for speed
            "end_date": "2026-07-25",
            "n_perms": 100,  # fewer for speed
        },
        verbose=True,
    )

    # Save results
    output_path = Path("/home/jupiter/Lvl3Quant/output/sector_backtest_selftest.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    serializable = {
        "n_trades": len(results["trades"]),
        "metrics": results["metrics"],
        "ml_vs_random": results["ml_vs_random"],
        "config": results["config"],
    }

    with open(output_path, "w") as f:
        json.dump(serializable, f, indent=2, default=str)

    print(f"\nResults saved to {output_path}")
