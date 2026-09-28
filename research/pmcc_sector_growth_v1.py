#!/usr/bin/env python3
"""
PMCC Sector Growth Strategy v1 — Poor Man's Covered Call on Sector ETFs
=======================================================================
Buy deep-ITM LEAP calls on top-ranked sectors (LGBM ranking),
sell near-term OTM calls for income.  Monthly rebalance.

Capital: $645, max $300/position.
Walk-forward: 60-day sliding train, 1-day OOT, LGBM ranking.
6 variants tested with 5 mandatory validation gates.

Author: Claude Opus 4.6 for Lvl3Quant
Date: 2026-07-27
"""

import json, warnings, time, os, sys
from datetime import datetime, timedelta
from math import log, sqrt, exp
from scipy.stats import norm
import numpy as np
import pandas as pd
import lightgbm as lgb
import yfinance as yf

warnings.filterwarnings("ignore")
np.random.seed(42)

# ── Constants ──────────────────────────────────────────────────────────
SECTOR_ETFS = ["XLK", "XLV", "XLY", "XLC", "XLF", "XLE", "XLB", "XLI", "XLP", "XLU", "XLRE"]
SPY = "SPY"
RISK_FREE = 0.05          # annualized
COMMISSION = 0.65          # per contract, each way
BS_MULT = 1.10             # BS calibration multiplier (KB #282)
BS_ADD = 6.60              # BS calibration additive ($)
STARTING_CAPITAL = 645.0
MAX_PER_POSITION = 300.0
CONTRACT_MULT = 100        # shares per options contract

# Variant definitions
VARIANTS = {
    "A": dict(top_n=1, dte_long=180, dte_short=14, momentum_filter=False, trailing_stop=None),
    "B": dict(top_n=2, dte_long=180, dte_short=14, momentum_filter=False, trailing_stop=None),
    "C": dict(top_n=1, dte_long=365, dte_short=30, momentum_filter=False, trailing_stop=None),
    "D": dict(top_n=2, dte_long=365, dte_short=30, momentum_filter=False, trailing_stop=None),
    "E": dict(top_n=1, dte_long=180, dte_short=14, momentum_filter=True,  trailing_stop=None),
    "F": dict(top_n=2, dte_long=180, dte_short=14, momentum_filter=False, trailing_stop=0.15),
}

# ── Black-Scholes with calibration ────────────────────────────────────
def bs_call_price(S, K, T, r, sigma):
    """BS call price with KB #282 calibration."""
    if T <= 0 or sigma <= 0:
        return max(S - K, 0)
    d1 = (log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    raw = S * norm.cdf(d1) - K * exp(-r * T) * norm.cdf(d2)
    # Calibration: multiply by 1.10 and add $6.60
    calibrated = raw * BS_MULT + BS_ADD
    return max(calibrated, max(S - K, 0) + 0.01)  # at least intrinsic + small premium

def bs_call_delta(S, K, T, r, sigma):
    """BS call delta."""
    if T <= 0 or sigma <= 0:
        return 1.0 if S > K else 0.0
    d1 = (log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt(T))
    return norm.cdf(d1)

def find_strike_for_delta(S, target_delta, T, r, sigma, call=True, itm=True):
    """Find strike that gives approximately the target delta."""
    # Binary search
    if itm:  # deep ITM call → low strike
        lo, hi = S * 0.50, S * 1.0
    else:    # OTM call → high strike
        lo, hi = S * 1.0, S * 1.50
    for _ in range(50):
        mid = (lo + hi) / 2
        d = bs_call_delta(S, mid, T, r, sigma)
        if itm:
            if d > target_delta:
                lo = mid
            else:
                hi = mid
        else:
            if d > target_delta:
                lo = mid
            else:
                hi = mid
    strike = (lo + hi) / 2
    # Round to nearest 0.50
    strike = round(strike * 2) / 2
    return strike

# ── Data Download ─────────────────────────────────────────────────────
print("Downloading data...")
t0 = time.time()
end_date = datetime(2026, 7, 25)
start_date = end_date - timedelta(days=365*2 + 90)  # extra 90d for feature warmup

tickers = SECTOR_ETFS + [SPY]
data = {}
for t in tickers:
    df = yf.download(t, start=start_date.strftime("%Y-%m-%d"),
                     end=end_date.strftime("%Y-%m-%d"), progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    data[t] = df
    print(f"  {t}: {len(df)} rows")

print(f"Data download: {time.time()-t0:.1f}s")

# ── Feature Engineering ───────────────────────────────────────────────
def build_features(prices_dict, spy_prices):
    """Build features for LGBM sector ranking."""
    features = {}
    for ticker in SECTOR_ETFS:
        df = prices_dict[ticker].copy()
        close = df["Close"]

        feat = pd.DataFrame(index=df.index)
        # Momentum features
        for w in [5, 10, 20, 60]:
            feat[f"ret_{w}d"] = close.pct_change(w)
        # Relative strength vs SPY
        spy_c = spy_prices["Close"].reindex(df.index)
        for w in [5, 20]:
            feat[f"rs_spy_{w}d"] = close.pct_change(w) - spy_c.pct_change(w)
        # Volatility
        ret_1d = close.pct_change()
        for w in [10, 20]:
            feat[f"vol_{w}d"] = ret_1d.rolling(w).std()
        # Mean reversion
        feat["dist_sma20"] = close / close.rolling(20).mean() - 1
        feat["dist_sma60"] = close / close.rolling(60).mean() - 1
        # Volume trend
        if "Volume" in df.columns:
            vol = df["Volume"]
            feat["vol_ratio_5_20"] = vol.rolling(5).mean() / vol.rolling(20).mean()

        # Forward return (label) — next 20 trading days
        feat["fwd_ret_20d"] = close.pct_change(20).shift(-20)

        feat["ticker"] = ticker
        features[ticker] = feat

    return pd.concat(features.values()).sort_index()

print("Building features...")
all_features = build_features(data, data[SPY])
feature_cols = [c for c in all_features.columns if c not in ["fwd_ret_20d", "ticker"]]
all_features = all_features.dropna(subset=feature_cols)

# Build historical volatility lookup (for BS pricing)
hist_vol = {}
for ticker in SECTOR_ETFS:
    close = data[ticker]["Close"]
    hv = close.pct_change().rolling(20).std() * sqrt(252)
    hist_vol[ticker] = hv

# ── Walk-Forward LGBM Ranking ────────────────────────────────────────
print("Running walk-forward LGBM ranking...")

# Get trading dates (after warmup)
spy_dates = data[SPY].index
min_date = all_features.index.min() + timedelta(days=80)  # ensure enough warmup
trading_dates = spy_dates[spy_dates >= min_date]

TRAIN_WINDOW = 60  # trading days

# Monthly rebalance dates (first trading day of each month)
rebal_dates = []
current_month = None
for d in trading_dates:
    ym = (d.year, d.month)
    if ym != current_month:
        current_month = ym
        rebal_dates.append(d)

print(f"  Total trading dates: {len(trading_dates)}")
print(f"  Rebalance dates: {len(rebal_dates)}")

# For each rebalance date, train LGBM and rank sectors
rankings = {}  # date → list of (ticker, predicted_ret) sorted best-first

for i, rdate in enumerate(rebal_dates):
    # Training data: last TRAIN_WINDOW days before rdate (SLIDING)
    train_end = rdate
    train_start_idx = max(0, spy_dates.get_loc(rdate) - TRAIN_WINDOW)
    train_start = spy_dates[train_start_idx]

    train_data = all_features[
        (all_features.index >= train_start) &
        (all_features.index < train_end)
    ].dropna(subset=["fwd_ret_20d"])

    if len(train_data) < 50:
        continue

    X_train = train_data[feature_cols].values
    y_train = train_data["fwd_ret_20d"].values

    # Train LGBM
    ds = lgb.Dataset(X_train, label=y_train, free_raw_data=False)
    params = {
        "objective": "regression",
        "metric": "rmse",
        "num_leaves": 15,
        "learning_rate": 0.05,
        "feature_fraction": 0.7,
        "bagging_fraction": 0.7,
        "bagging_freq": 5,
        "verbose": -1,
        "seed": 42,
    }
    model = lgb.train(params, ds, num_boost_round=100)

    # Predict for each sector at rebalance date
    preds = {}
    for ticker in SECTOR_ETFS:
        ticker_data = all_features[
            (all_features["ticker"] == ticker) &
            (all_features.index == rdate)
        ]
        if len(ticker_data) == 0:
            # Use closest prior date
            ticker_data = all_features[
                (all_features["ticker"] == ticker) &
                (all_features.index <= rdate)
            ].tail(1)
        if len(ticker_data) > 0:
            X_pred = ticker_data[feature_cols].values
            pred = model.predict(X_pred)[0]
            preds[ticker] = pred

    # Rank by predicted return (descending)
    ranked = sorted(preds.items(), key=lambda x: x[1], reverse=True)
    rankings[rdate] = ranked

print(f"  Generated rankings for {len(rankings)} rebalance dates")

# ── PMCC Backtest Engine ──────────────────────────────────────────────
class PMCCPosition:
    """Tracks a single PMCC position (long LEAP + short near-term call)."""
    def __init__(self, ticker, entry_date, spot, leap_strike, leap_dte,
                 short_strike, short_dte, leap_cost, short_premium,
                 leap_sigma, cost_mult=1.0):
        self.ticker = ticker
        self.entry_date = entry_date
        self.spot_at_entry = spot
        self.leap_strike = leap_strike
        self.leap_dte_initial = leap_dte
        self.short_strike = short_strike
        self.short_dte_initial = short_dte
        self.leap_cost = leap_cost          # debit paid for LEAP (per contract, after calibration)
        self.short_premium = short_premium  # credit received for short call
        self.net_debit = leap_cost - short_premium
        self.leap_sigma = leap_sigma
        self.cost_mult = cost_mult
        self.commission_paid = 2 * COMMISSION * cost_mult  # open both legs
        self.closed = False
        self.close_date = None
        self.pnl = 0.0
        self.peak_leap_value = leap_cost

    def mark_to_market(self, date, spot, days_elapsed):
        """Return current value of PMCC position."""
        leap_dte_remaining = max(self.leap_dte_initial - days_elapsed, 1) / 365.0
        short_dte_remaining = max(self.short_dte_initial - days_elapsed, 0) / 365.0

        leap_value = bs_call_price(spot, self.leap_strike, leap_dte_remaining,
                                   RISK_FREE, self.leap_sigma)
        if short_dte_remaining > 0:
            short_value = bs_call_price(spot, self.short_strike, short_dte_remaining,
                                        RISK_FREE, self.leap_sigma)
        else:
            # Short call expired
            short_value = max(spot - self.short_strike, 0)  # intrinsic at expiry

        position_value = leap_value - short_value
        return position_value, leap_value, short_value

    def close(self, date, spot, days_elapsed):
        """Close the position, compute FIFO P&L."""
        pos_value, leap_val, short_val = self.mark_to_market(date, spot, days_elapsed)
        close_commission = 2 * COMMISSION * self.cost_mult
        self.pnl = (pos_value - self.net_debit) - self.commission_paid - close_commission
        self.closed = True
        self.close_date = date
        return self.pnl


def run_pmcc_backtest(variant_params, rankings, data, hist_vol,
                       cost_multiplier=1.0, use_random_rankings=False,
                       random_seed=None):
    """Run PMCC backtest for a given variant configuration."""
    top_n = variant_params["top_n"]
    dte_long = variant_params["dte_long"]
    dte_short = variant_params["dte_short"]
    momentum_filter = variant_params["momentum_filter"]
    trailing_stop = variant_params["trailing_stop"]

    capital = STARTING_CAPITAL
    positions = []       # active positions
    closed_trades = []   # completed trades
    equity_curve = []

    rng = np.random.RandomState(random_seed) if random_seed is not None else None

    sorted_rdates = sorted(rankings.keys())

    for ri, rdate in enumerate(sorted_rdates):
        # Get next rebalance date (or end of data)
        if ri + 1 < len(sorted_rdates):
            next_rdate = sorted_rdates[ri + 1]
        else:
            next_rdate = data[SPY].index[-1]

        # Close existing positions at rebalance
        for pos in positions:
            if not pos.closed:
                spot = data[pos.ticker]["Close"].asof(rdate)
                days_elapsed = (rdate - pos.entry_date).days
                pnl = pos.close(rdate, spot, days_elapsed)
                capital += pos.net_debit + pnl  # return capital + pnl
                closed_trades.append(pos)
        positions = []

        # Get rankings
        if use_random_rankings:
            # Random sector picks
            tickers_avail = list(SECTOR_ETFS)
            rng.shuffle(tickers_avail)
            selected = tickers_avail[:top_n]
        else:
            ranked = rankings[rdate]
            selected = [t for t, _ in ranked[:top_n]]

        # Momentum filter (variant E)
        if momentum_filter:
            filtered = []
            for ticker in selected:
                close = data[ticker]["Close"]
                loc = close.index.get_indexer([rdate], method="ffill")[0]
                if loc >= 20:
                    mom_20d = close.iloc[loc] / close.iloc[loc - 20] - 1
                    if mom_20d > 0:
                        filtered.append(ticker)
            selected = filtered

        if not selected:
            equity_curve.append((rdate, capital))
            continue

        # Allocate capital
        per_position = min(MAX_PER_POSITION, capital / len(selected))

        for ticker in selected:
            spot = data[ticker]["Close"].asof(rdate)
            if pd.isna(spot) or spot <= 0:
                continue

            sigma = hist_vol[ticker].asof(rdate)
            if pd.isna(sigma) or sigma <= 0:
                sigma = 0.25  # default

            T_long = dte_long / 365.0
            T_short = dte_short / 365.0

            # Find LEAP strike (deep ITM, delta ~0.80)
            leap_strike = find_strike_for_delta(spot, 0.80, T_long, RISK_FREE, sigma, itm=True)
            leap_price = bs_call_price(spot, leap_strike, T_long, RISK_FREE, sigma)

            # Check if LEAP cost fits budget (price is per share, multiply by 100 for contract)
            leap_contract_cost = leap_price  # This IS per-share price from BS
            # For a small account, we might need to think in terms of per-share
            # But options trade in contracts of 100 shares
            # A LEAP on a $50 ETF with delta 0.80 might cost ~$15/share = $1500/contract
            # That's way over $300 budget, so we need to think about mini/fractional
            # For this backtest, we'll normalize to per-share economics
            # (equivalent to trading 1 share worth of the option spread)

            # Normalize: how many "shares worth" can we buy with per_position budget?
            if leap_price <= 0:
                continue
            num_units = per_position / leap_price  # fractional units
            if num_units <= 0:
                continue

            # Find short call strike (OTM, delta ~0.30)
            short_strike = find_strike_for_delta(spot, 0.30, T_short, RISK_FREE, sigma, itm=False)
            short_price = bs_call_price(spot, short_strike, T_short, RISK_FREE, sigma)

            # Scale everything by num_units
            scaled_leap_cost = leap_price * num_units
            scaled_short_premium = short_price * num_units

            pos = PMCCPosition(
                ticker=ticker,
                entry_date=rdate,
                spot=spot,
                leap_strike=leap_strike,
                leap_dte=dte_long,
                short_strike=short_strike,
                short_dte=dte_short,
                leap_cost=scaled_leap_cost,
                short_premium=scaled_short_premium,
                leap_sigma=sigma,
                cost_mult=cost_multiplier,
            )
            pos._num_units = num_units

            capital -= pos.net_debit + pos.commission_paid
            positions.append(pos)

        # Track equity through the holding period
        holding_dates = data[SPY].index[
            (data[SPY].index >= rdate) & (data[SPY].index < next_rdate)
        ]
        for hdate in holding_dates:
            pos_value = 0
            for pos in positions:
                if not pos.closed:
                    spot = data[pos.ticker]["Close"].asof(hdate)
                    days_el = (hdate - pos.entry_date).days
                    val, leap_val, _ = pos.mark_to_market(hdate, spot, days_el)

                    # Trailing stop check (variant F)
                    if trailing_stop is not None:
                        scaled_leap_val = leap_val * pos._num_units
                        if scaled_leap_val > pos.peak_leap_value:
                            pos.peak_leap_value = scaled_leap_val
                        if scaled_leap_val < pos.peak_leap_value * (1 - trailing_stop):
                            pnl = pos.close(hdate, spot, days_el)
                            capital += pos.net_debit + pnl
                            closed_trades.append(pos)
                            continue

                    pos_value += val * pos._num_units

            equity_curve.append((hdate, capital + pos_value))

    # Close any remaining positions
    final_date = data[SPY].index[-1]
    for pos in positions:
        if not pos.closed:
            spot = data[pos.ticker]["Close"].asof(final_date)
            days_el = (final_date - pos.entry_date).days
            pnl = pos.close(final_date, spot, days_el)
            capital += pos.net_debit + pnl
            closed_trades.append(pos)

    return closed_trades, equity_curve, capital


def compute_metrics(closed_trades, equity_curve, starting_capital=STARTING_CAPITAL):
    """Compute strategy performance metrics."""
    if not equity_curve:
        return {
            "sharpe": 0, "sortino": 0, "cagr": 0, "mdd": -1,
            "win_rate": 0, "profit_factor": 0, "num_trades": 0,
            "final_capital": starting_capital, "total_return_pct": 0,
        }

    eq_df = pd.DataFrame(equity_curve, columns=["date", "equity"])
    eq_df = eq_df.drop_duplicates(subset="date", keep="last").set_index("date").sort_index()

    if len(eq_df) < 2:
        return {
            "sharpe": 0, "sortino": 0, "cagr": 0, "mdd": -1,
            "win_rate": 0, "profit_factor": 0, "num_trades": 0,
            "final_capital": starting_capital, "total_return_pct": 0,
        }

    returns = eq_df["equity"].pct_change().dropna()
    returns = returns.replace([np.inf, -np.inf], 0).fillna(0)

    # Sharpe (annualized)
    if returns.std() > 0:
        sharpe = (returns.mean() / returns.std()) * sqrt(252)
    else:
        sharpe = 0

    # Sortino
    downside = returns[returns < 0]
    if len(downside) > 0 and downside.std() > 0:
        sortino = (returns.mean() / downside.std()) * sqrt(252)
    else:
        sortino = sharpe

    # CAGR
    total_days = (eq_df.index[-1] - eq_df.index[0]).days
    if total_days > 0 and eq_df["equity"].iloc[-1] > 0:
        cagr = (eq_df["equity"].iloc[-1] / starting_capital) ** (365 / total_days) - 1
    else:
        cagr = 0

    # Max Drawdown
    peak = eq_df["equity"].expanding().max()
    dd = (eq_df["equity"] - peak) / peak
    mdd = dd.min()

    # Trade-level metrics
    pnls = [t.pnl for t in closed_trades]
    num_trades = len(pnls)
    if num_trades > 0:
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        win_rate = len(wins) / num_trades
        gross_profit = sum(wins) if wins else 0
        gross_loss = abs(sum(losses)) if losses else 0.001
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 999
    else:
        win_rate = 0
        profit_factor = 0

    return {
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "cagr": round(cagr * 100, 2),
        "mdd": round(mdd * 100, 2),
        "win_rate": round(win_rate * 100, 1),
        "profit_factor": round(profit_factor, 2),
        "num_trades": num_trades,
        "final_capital": round(eq_df["equity"].iloc[-1], 2),
        "total_return_pct": round((eq_df["equity"].iloc[-1] / starting_capital - 1) * 100, 2),
    }


# ── Validation Gates ──────────────────────────────────────────────────

def gate1_permutation_test(variant_params, rankings, data, hist_vol, actual_sharpe, n_perms=100):
    """Gate 1: Permutation test — shuffle sector labels, p < 0.05."""
    print(f"    Gate 1: Permutation test ({n_perms} shuffles)...")
    perm_sharpes = []
    for i in range(n_perms):
        # Shuffle rankings at each date
        shuffled_rankings = {}
        rng = np.random.RandomState(i + 1000)
        for d, ranked in rankings.items():
            tickers = [t for t, s in ranked]
            scores = [s for t, s in ranked]
            rng.shuffle(tickers)
            shuffled_rankings[d] = list(zip(tickers, scores))

        trades, eq, _ = run_pmcc_backtest(variant_params, shuffled_rankings, data, hist_vol)
        m = compute_metrics(trades, eq)
        perm_sharpes.append(m["sharpe"])

    perm_sharpes = np.array(perm_sharpes)
    p_value = np.mean(perm_sharpes >= actual_sharpe)
    return {
        "pass": p_value < 0.05,
        "p_value": round(float(p_value), 4),
        "perm_mean_sharpe": round(float(np.mean(perm_sharpes)), 3),
        "perm_std_sharpe": round(float(np.std(perm_sharpes)), 3),
        "actual_sharpe": actual_sharpe,
    }

def gate2_regime_balance(equity_curve, spy_data):
    """Gate 2: |Sharpe_bull - Sharpe_bear| / max < 0.50."""
    print("    Gate 2: Regime balance...")
    spy_close = spy_data["Close"]
    spy_ret_20d = spy_close.pct_change(20)

    eq_df = pd.DataFrame(equity_curve, columns=["date", "equity"])
    eq_df = eq_df.drop_duplicates(subset="date", keep="last").set_index("date").sort_index()
    eq_returns = eq_df["equity"].pct_change().dropna()

    bull_returns = []
    bear_returns = []

    for date, ret in eq_returns.items():
        spy_mom = spy_ret_20d.asof(date)
        if pd.isna(spy_mom):
            continue
        if spy_mom > 0:
            bull_returns.append(ret)
        else:
            bear_returns.append(ret)

    bull_returns = pd.Series(bull_returns)
    bear_returns = pd.Series(bear_returns)

    sharpe_bull = (bull_returns.mean() / bull_returns.std() * sqrt(252)) if len(bull_returns) > 5 and bull_returns.std() > 0 else 0
    sharpe_bear = (bear_returns.mean() / bear_returns.std() * sqrt(252)) if len(bear_returns) > 5 and bear_returns.std() > 0 else 0

    max_sharpe = max(abs(sharpe_bull), abs(sharpe_bear), 0.001)
    imbalance = abs(sharpe_bull - sharpe_bear) / max_sharpe

    return {
        "pass": imbalance < 0.50,
        "sharpe_bull": round(float(sharpe_bull), 3),
        "sharpe_bear": round(float(sharpe_bear), 3),
        "imbalance_ratio": round(float(imbalance), 3),
        "threshold": 0.50,
    }

def gate3_subperiod_stability(equity_curve):
    """Gate 3: Split into 4 quarters, all must be profitable."""
    print("    Gate 3: Sub-period stability...")
    eq_df = pd.DataFrame(equity_curve, columns=["date", "equity"])
    eq_df = eq_df.drop_duplicates(subset="date", keep="last").set_index("date").sort_index()

    n = len(eq_df)
    quarter_size = n // 4
    quarters_profitable = []
    quarter_returns = []

    for q in range(4):
        start_idx = q * quarter_size
        end_idx = (q + 1) * quarter_size if q < 3 else n
        chunk = eq_df.iloc[start_idx:end_idx]
        if len(chunk) < 2:
            quarters_profitable.append(False)
            quarter_returns.append(0)
            continue
        q_return = chunk["equity"].iloc[-1] / chunk["equity"].iloc[0] - 1
        quarters_profitable.append(q_return > 0)
        quarter_returns.append(round(float(q_return * 100), 2))

    return {
        "pass": all(quarters_profitable),
        "quarter_returns_pct": quarter_returns,
        "all_profitable": all(quarters_profitable),
    }

def gate4_random_baseline(variant_params, rankings, data, hist_vol, actual_sharpe, n_trials=100):
    """Gate 4: Random sector picks, strategy must beat p95."""
    print(f"    Gate 4: Random baseline ({n_trials} trials)...")
    random_sharpes = []
    for i in range(n_trials):
        trades, eq, _ = run_pmcc_backtest(
            variant_params, rankings, data, hist_vol,
            use_random_rankings=True, random_seed=i + 2000
        )
        m = compute_metrics(trades, eq)
        random_sharpes.append(m["sharpe"])

    random_sharpes = np.array(random_sharpes)
    p95 = np.percentile(random_sharpes, 95)

    return {
        "pass": actual_sharpe > p95,
        "actual_sharpe": actual_sharpe,
        "random_p95_sharpe": round(float(p95), 3),
        "random_mean_sharpe": round(float(np.mean(random_sharpes)), 3),
        "random_median_sharpe": round(float(np.median(random_sharpes)), 3),
    }

def gate5_cost_sensitivity(variant_params, rankings, data, hist_vol):
    """Gate 5: 2x costs, strategy still profitable?"""
    print("    Gate 5: Cost sensitivity (2x costs)...")
    trades, eq, _ = run_pmcc_backtest(variant_params, rankings, data, hist_vol, cost_multiplier=2.0)
    m = compute_metrics(trades, eq)
    return {
        "pass": m["total_return_pct"] > 0,
        "sharpe_2x_costs": m["sharpe"],
        "total_return_2x_costs_pct": m["total_return_pct"],
        "final_capital_2x_costs": m["final_capital"],
    }

# ── Run All Variants ──────────────────────────────────────────────────
print("\n" + "="*70)
print("RUNNING PMCC SECTOR GROWTH BACKTEST — 6 VARIANTS")
print("="*70)

results = {}
best_sharpe = -999
best_variant = None

for variant_name, params in VARIANTS.items():
    print(f"\n{'─'*50}")
    print(f"Variant {variant_name}: top_{params['top_n']}, LEAP={params['dte_long']}d, "
          f"short={params['dte_short']}d, mom={params['momentum_filter']}, "
          f"stop={params['trailing_stop']}")
    print(f"{'─'*50}")

    t1 = time.time()
    trades, eq_curve, final_cap = run_pmcc_backtest(params, rankings, data, hist_vol)
    metrics = compute_metrics(trades, eq_curve)
    elapsed = time.time() - t1

    print(f"  Metrics: Sharpe={metrics['sharpe']}, Sortino={metrics['sortino']}, "
          f"CAGR={metrics['cagr']}%, MDD={metrics['mdd']}%, WR={metrics['win_rate']}%, "
          f"PF={metrics['profit_factor']}, Trades={metrics['num_trades']}, "
          f"Final=${metrics['final_capital']}")

    # Run validation gates
    print(f"  Running validation gates...")
    g1 = gate1_permutation_test(params, rankings, data, hist_vol, metrics["sharpe"], n_perms=100)
    g2 = gate2_regime_balance(eq_curve, data[SPY])
    g3 = gate3_subperiod_stability(eq_curve)
    g4 = gate4_random_baseline(params, rankings, data, hist_vol, metrics["sharpe"], n_trials=100)
    g5 = gate5_cost_sensitivity(params, rankings, data, hist_vol)

    gates_passed = sum([g1["pass"], g2["pass"], g3["pass"], g4["pass"], g5["pass"]])

    variant_result = {
        "params": params,
        "metrics": metrics,
        "gates": {
            "gate1_permutation": g1,
            "gate2_regime_balance": g2,
            "gate3_subperiod_stability": g3,
            "gate4_random_baseline": g4,
            "gate5_cost_sensitivity": g5,
        },
        "gates_passed": f"{gates_passed}/5",
        "all_gates_pass": gates_passed == 5,
        "elapsed_seconds": round(elapsed, 1),
    }
    results[f"variant_{variant_name}"] = variant_result

    print(f"  Gates: {gates_passed}/5 passed — "
          f"Perm={'✓' if g1['pass'] else '✗'}(p={g1['p_value']}) "
          f"Regime={'✓' if g2['pass'] else '✗'}({g2['imbalance_ratio']:.2f}) "
          f"Stability={'✓' if g3['pass'] else '✗'} "
          f"Random={'✓' if g4['pass'] else '✗'} "
          f"Cost={'✓' if g5['pass'] else '✗'}")

    if metrics["sharpe"] > best_sharpe:
        best_sharpe = metrics["sharpe"]
        best_variant = variant_name

# ── Summary ───────────────────────────────────────────────────────────
print("\n" + "="*70)
print("FINAL SUMMARY")
print("="*70)

summary = {
    "strategy": "PMCC Sector Growth v1",
    "description": "Poor Man's Covered Call on LGBM-ranked sector ETFs",
    "backtest_period": f"{trading_dates[0].strftime('%Y-%m-%d')} to {trading_dates[-1].strftime('%Y-%m-%d')}",
    "starting_capital": STARTING_CAPITAL,
    "sectors": SECTOR_ETFS,
    "best_variant": best_variant,
    "best_sharpe": best_sharpe,
    "variants": results,
    "methodology": {
        "pricing": "Black-Scholes with KB#282 calibration (1.10x + $6.60)",
        "walk_forward": "60-day sliding window LGBM, monthly rebalance",
        "commission": f"${COMMISSION}/contract each way",
        "pnl_method": "FIFO",
        "validation_gates": [
            "Permutation test (100 shuffles, p<0.05)",
            "Regime balance (|Sharpe_bull-Sharpe_bear|/max < 0.50)",
            "Sub-period stability (4 quarters, all profitable)",
            "Random baseline (100 trials, beat p95)",
            "Cost sensitivity (2x costs, still profitable)",
        ],
    },
    "timestamp": datetime.now().isoformat(),
}

for vname in sorted(results.keys()):
    v = results[vname]
    m = v["metrics"]
    print(f"\n  {vname}: Sharpe={m['sharpe']}, Sortino={m['sortino']}, "
          f"CAGR={m['cagr']}%, MDD={m['mdd']}%, WR={m['win_rate']}%, "
          f"PF={m['profit_factor']}, Trades={m['num_trades']}, "
          f"Final=${m['final_capital']} | Gates: {v['gates_passed']}")

# ── Save Results ──────────────────────────────────────────────────────
output_path = "/home/jupiter/Lvl3Quant/research/findings/pmcc_sector_growth_v1_results.json"
with open(output_path, "w") as f:
    json.dump(summary, f, indent=2, default=str)
print(f"\nResults saved to {output_path}")

# ── MLflow Logging ────────────────────────────────────────────────────
try:
    import mlflow
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("pmcc_sector_growth_v1")

    with mlflow.start_run(run_name="pmcc_sector_growth_v1_full"):
        mlflow.log_param("strategy", "PMCC Sector Growth")
        mlflow.log_param("starting_capital", STARTING_CAPITAL)
        mlflow.log_param("sectors", ",".join(SECTOR_ETFS))
        mlflow.log_param("train_window", TRAIN_WINDOW)
        mlflow.log_param("bs_calibration", f"{BS_MULT}x + ${BS_ADD}")

        for vname, vdata in results.items():
            m = vdata["metrics"]
            for metric_name, metric_val in m.items():
                if isinstance(metric_val, (int, float)):
                    mlflow.log_metric(f"{vname}_{metric_name}", metric_val)
            mlflow.log_metric(f"{vname}_gates_passed", int(vdata["gates_passed"].split("/")[0]))

        mlflow.log_metric("best_sharpe", best_sharpe)
        mlflow.log_param("best_variant", best_variant)

        mlflow.log_artifact(output_path)

    print("MLflow logging complete.")
except Exception as e:
    print(f"MLflow logging failed (non-fatal): {e}")

print("\n" + "="*70)
print("BACKTEST COMPLETE")
print("="*70)
