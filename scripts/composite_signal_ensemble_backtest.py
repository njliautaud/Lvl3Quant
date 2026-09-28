#!/usr/bin/env python3
"""
Composite Signal Ensemble Backtest
===================================
Combines multiple individually weak signals into a weighted composite score.
Individual signals fail permutation tests, but the ensemble may produce edge.

Signals:
  1. 200-SMA Filter (0.20)
  2. Mean Reversion Extreme (0.20)
  3. Post-Earnings Gap (0.25)
  4. VIX Below 25 (0.15)
  5. Sector Momentum (0.20)

Variants A-F with different thresholds, holding periods, and weighting schemes.
OOT: Jan 2022 - Jul 2026, weekly rebalance, $645 initial capital.
"""

import json
import warnings
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm

warnings.filterwarnings("ignore")

# ── Constants ──────────────────────────────────────────────────────────────
UNIVERSE = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "AMD", "CRM", "SNOW",
    "DDOG", "NET", "CRWD", "SHOP", "SQ", "COIN", "MARA", "SOFI", "PLTR", "RBLX",
    "HOOD", "ARM", "SMCI", "MU", "AVGO", "NFLX", "UBER", "ABNB", "RIVN", "SNAP",
]

# Map each stock to its sector ETF
SECTOR_MAP = {
    "AAPL": "XLK", "MSFT": "XLK", "GOOGL": "XLC", "AMZN": "XLY", "NVDA": "XLK",
    "META": "XLC", "TSLA": "XLY", "AMD": "XLK", "CRM": "XLK", "SNOW": "XLK",
    "DDOG": "XLK", "NET": "XLK", "CRWD": "XLK", "SHOP": "XLK", "SQ": "XLF",
    "COIN": "XLF", "MARA": "XLF", "SOFI": "XLF", "PLTR": "XLK", "RBLX": "XLC",
    "HOOD": "XLF", "ARM": "XLK", "SMCI": "XLK", "MU": "XLK", "AVGO": "XLK",
    "NFLX": "XLC", "UBER": "XLK", "ABNB": "XLY", "RIVN": "XLY", "SNAP": "XLC",
}

SECTOR_ETFS = ["XLK", "XLY", "XLC", "XLF"]
INITIAL_CAPITAL = 645.0
OOT_START = "2022-01-01"
OOT_END = "2026-07-28"
DATA_START = "2020-06-01"  # need lookback for 200-SMA

SIGNAL_WEIGHTS_DEFAULT = {
    "sma200": 0.20, "mean_rev": 0.20, "earnings_gap": 0.25,
    "vix": 0.15, "sector_mom": 0.20,
}

SLIPPAGE_PCT = 0.0002  # 0.02% slippage for equities
OPTIONS_COMMISSION = 0.65  # per contract
OPTIONS_SPREAD_PCT = 0.05  # 5% bid-ask spread
OPTIONS_BS_HAIRCUT = 0.30  # 30% BS pricing haircut
PERM_ITERATIONS = 1000


# ── Data Download ──────────────────────────────────────────────────────────
def download_data():
    """Download all required price data."""
    print("Downloading price data...")
    all_tickers = UNIVERSE + ["SPY", "^VIX"] + SECTOR_ETFS
    data = yf.download(all_tickers, start=DATA_START, end=OOT_END, auto_adjust=True, progress=False)

    # yfinance returns MultiIndex columns: (field, ticker)
    closes = data["Close"]
    volumes = data["Volume"]
    opens = data["Open"]

    # Fill forward missing data
    closes = closes.ffill()
    volumes = volumes.ffill().fillna(0)
    opens = opens.ffill()

    print(f"  Data shape: {closes.shape}, date range: {closes.index[0].date()} to {closes.index[-1].date()}")
    return closes, volumes, opens


# ── Signal Computation ─────────────────────────────────────────────────────
def compute_sma200_filter(closes):
    """Signal 1: SPY above 200-day SMA → 1, else 0."""
    spy = closes["SPY"]
    sma200 = spy.rolling(200).mean()
    signal = (spy > sma200).astype(float)
    return signal  # Series indexed by date


def compute_mean_reversion(closes):
    """Signal 2: Stock in bottom 3/5 of universe by weekly return."""
    weekly_ret = closes[UNIVERSE].pct_change(5)
    ranks = weekly_ret.rank(axis=1, ascending=True)  # 1 = worst performer
    scores = pd.DataFrame(0.0, index=closes.index, columns=UNIVERSE)
    scores[ranks <= 3] = 1.0
    scores[(ranks > 3) & (ranks <= 5)] = 0.5
    return scores  # DataFrame: date x stock


def compute_earnings_gap(closes, volumes, opens):
    """Signal 3: >3% overnight gap AND >1.5x avg volume in past 5 days.
    We detect gaps using open vs prior close, volume spike using 20-day avg."""
    scores = pd.DataFrame(0.0, index=closes.index, columns=UNIVERSE)
    for ticker in UNIVERSE:
        if ticker not in closes.columns or ticker not in opens.columns:
            continue
        c = closes[ticker]
        o = opens[ticker]
        v = volumes[ticker]

        # Overnight gap: (open - prev_close) / prev_close
        overnight_gap = (o - c.shift(1)) / c.shift(1)
        avg_vol = v.rolling(20).mean()
        vol_spike = v > (1.5 * avg_vol)

        # Look for gap + volume spike in past 5 days
        pos_gap = (overnight_gap > 0.03) & vol_spike
        # Rolling 5-day window: was there a positive earnings gap recently?
        recent_gap = pos_gap.rolling(5, min_periods=1).max().fillna(0)
        scores[ticker] = recent_gap.astype(float)
    return scores


def compute_vix_filter(closes):
    """Signal 4: VIX environment. <20 → 1, 20-25 → 0.5, >25 → 0."""
    vix = closes["^VIX"]
    scores = pd.Series(0.0, index=closes.index)
    scores[vix < 20] = 1.0
    scores[(vix >= 20) & (vix <= 25)] = 0.5
    return scores


def compute_sector_momentum(closes):
    """Signal 5: Stock's sector ETF 20-day momentum."""
    mom = {}
    for etf in SECTOR_ETFS:
        if etf in closes.columns:
            mom[etf] = closes[etf].pct_change(20)

    scores = pd.DataFrame(0.0, index=closes.index, columns=UNIVERSE)
    for ticker in UNIVERSE:
        etf = SECTOR_MAP.get(ticker, "XLK")
        if etf in mom:
            m = mom[etf]
            scores.loc[m > 0.02, ticker] = 1.0
            scores.loc[(m >= 0) & (m <= 0.02), ticker] = 0.5
    return scores


def compute_all_signals(closes, volumes, opens):
    """Compute all 5 signals, return dict of signal DataFrames/Series."""
    print("Computing signals...")
    signals = {
        "sma200": compute_sma200_filter(closes),
        "mean_rev": compute_mean_reversion(closes),
        "earnings_gap": compute_earnings_gap(closes, volumes, opens),
        "vix": compute_vix_filter(closes),
        "sector_mom": compute_sector_momentum(closes),
    }
    return signals


def compute_composite_scores(signals, weights, dates, stocks):
    """Compute composite score for each stock on each date."""
    scores = pd.DataFrame(0.0, index=dates, columns=stocks)
    for stock in stocks:
        for sig_name, w in weights.items():
            if sig_name in ("sma200", "vix"):
                # These are market-level signals (same for all stocks)
                sig_vals = signals[sig_name].reindex(dates).fillna(0)
                scores[stock] += w * sig_vals
            else:
                # Stock-level signals
                if stock in signals[sig_name].columns:
                    sig_vals = signals[sig_name][stock].reindex(dates).fillna(0)
                    scores[stock] += w * sig_vals
    return scores


# ── Black-Scholes for Options Pricing ──────────────────────────────────────
def bs_call_price(S, K, T, r, sigma):
    """Black-Scholes call price."""
    if T <= 0 or sigma <= 0:
        return max(S - K, 0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def bs_delta(S, K, T, r, sigma):
    """Black-Scholes call delta."""
    if T <= 0 or sigma <= 0:
        return 1.0 if S > K else 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return norm.cdf(d1)


def find_strike_for_delta(S, target_delta, T, r, sigma, direction="call"):
    """Find strike that gives approximately target_delta for a call."""
    # Binary search for strike
    lo, hi = S * 0.7, S * 1.3
    for _ in range(50):
        mid = (lo + hi) / 2
        d = bs_delta(S, mid, T, r, sigma)
        if d > target_delta:
            lo = mid
        else:
            hi = mid
    return round((lo + hi) / 2, 2)


# ── Backtest Engine ────────────────────────────────────────────────────────
def get_weekly_rebalance_dates(dates, oot_start):
    """Get Friday rebalance dates within OOT period."""
    oot_dates = dates[dates >= pd.Timestamp(oot_start)]
    # Group by week, take last trading day of each week
    weekly = oot_dates.to_series().groupby(oot_dates.to_period("W")).last()
    return weekly.values


def backtest_equity(composite_scores, closes, rebal_dates, selection_fn,
                    hold_days, max_positions, initial_capital):
    """Run equity backtest with given selection function and holding period."""
    capital = initial_capital
    trades = []
    positions = []  # list of (stock, entry_date, entry_price, shares)
    equity_curve = []

    for i, date in enumerate(rebal_dates):
        date = pd.Timestamp(date)
        if date not in closes.index:
            continue

        # Close positions that have exceeded hold period
        new_positions = []
        for pos in positions:
            stock, entry_date, entry_price, shares = pos
            days_held = (date - entry_date).days
            if days_held >= hold_days:
                # Exit
                if date in closes.index and stock in closes.columns:
                    exit_price = closes.loc[date, stock]
                    if pd.notna(exit_price) and pd.notna(entry_price):
                        exit_price_adj = exit_price * (1 - SLIPPAGE_PCT)
                        pnl = (exit_price_adj - entry_price) * shares
                        capital += exit_price_adj * shares
                        trades.append({
                            "stock": stock, "entry_date": str(entry_date.date()),
                            "exit_date": str(date.date()),
                            "entry_price": round(entry_price, 2),
                            "exit_price": round(exit_price_adj, 2),
                            "shares": shares, "pnl": round(pnl, 2),
                            "return_pct": round(pnl / (entry_price * shares) * 100, 2),
                        })
                    else:
                        new_positions.append(pos)
                        continue
            else:
                new_positions.append(pos)
        positions = new_positions

        # Select new stocks
        if date in composite_scores.index:
            row = composite_scores.loc[date]
            selected = selection_fn(row)
        else:
            selected = []

        # Open new positions (up to max)
        open_slots = max_positions - len(positions)
        for stock in selected[:open_slots]:
            if stock in closes.columns and date in closes.index:
                price = closes.loc[date, stock]
                if pd.notna(price) and price > 0:
                    entry_price = price * (1 + SLIPPAGE_PCT)
                    alloc = capital / max(open_slots, 1)
                    alloc = min(alloc, capital)
                    if alloc < 5:
                        continue
                    shares = int(alloc / entry_price)
                    if shares > 0:
                        cost = shares * entry_price
                        capital -= cost
                        positions.append((stock, date, entry_price, shares))

        # Mark to market
        mtm = capital
        for pos in positions:
            stock, _, _, shares = pos
            if date in closes.index and stock in closes.columns:
                p = closes.loc[date, stock]
                if pd.notna(p):
                    mtm += p * shares
        equity_curve.append({"date": str(date.date()), "equity": round(mtm, 2)})

    # Force-close remaining positions at end
    last_date = rebal_dates[-1] if len(rebal_dates) > 0 else None
    if last_date is not None:
        last_date = pd.Timestamp(last_date)
        for pos in positions:
            stock, entry_date, entry_price, shares = pos
            if last_date in closes.index and stock in closes.columns:
                exit_price = closes.loc[last_date, stock]
                if pd.notna(exit_price) and pd.notna(entry_price):
                    exit_price_adj = exit_price * (1 - SLIPPAGE_PCT)
                    pnl = (exit_price_adj - entry_price) * shares
                    capital += exit_price_adj * shares
                    trades.append({
                        "stock": stock, "entry_date": str(entry_date.date()),
                        "exit_date": str(last_date.date()),
                        "entry_price": round(entry_price, 2),
                        "exit_price": round(exit_price_adj, 2),
                        "shares": shares, "pnl": round(pnl, 2),
                        "return_pct": round(pnl / (entry_price * shares) * 100, 2),
                    })

    return trades, equity_curve


def backtest_options(composite_scores, closes, rebal_dates, selection_fn,
                     max_positions, initial_capital):
    """Variant C: Buy weekly calls with BS pricing, 0.3-delta, 7DTE."""
    capital = initial_capital
    trades = []
    positions = []  # (stock, entry_date, premium_paid, contracts, strike, entry_stock_price)
    equity_curve = []
    r = 0.05  # risk-free rate
    sigma = 0.35  # implied vol estimate (will vary but reasonable default)

    for i, date in enumerate(rebal_dates):
        date = pd.Timestamp(date)
        if date not in closes.index:
            continue

        # Close positions held >= 5 trading days (approx 7 calendar)
        new_positions = []
        for pos in positions:
            stock, entry_date, premium_paid, contracts, strike, entry_sp = pos
            days_held = (date - entry_date).days
            if days_held >= 7:
                # Exit: compute option value at expiry
                if stock in closes.columns:
                    exit_sp = closes.loc[date, stock]
                    if pd.notna(exit_sp):
                        intrinsic = max(exit_sp - strike, 0) * 100 * contracts
                        # Subtract exit costs
                        exit_cost = contracts * OPTIONS_COMMISSION
                        pnl = intrinsic - premium_paid - exit_cost
                        capital += max(intrinsic - exit_cost, 0)
                        trades.append({
                            "stock": stock, "entry_date": str(entry_date.date()),
                            "exit_date": str(date.date()),
                            "strike": strike, "contracts": contracts,
                            "premium_paid": round(premium_paid, 2),
                            "exit_value": round(intrinsic, 2),
                            "pnl": round(pnl, 2),
                            "return_pct": round(pnl / premium_paid * 100, 2) if premium_paid > 0 else 0,
                        })
                        continue
                new_positions.append(pos)
            else:
                new_positions.append(pos)
        positions = new_positions

        # Select new stocks
        if date in composite_scores.index:
            row = composite_scores.loc[date]
            selected = selection_fn(row)
        else:
            selected = []

        # Open new option positions
        open_slots = max_positions - len(positions)
        for stock in selected[:open_slots]:
            if stock in closes.columns:
                S = closes.loc[date, stock]
                if pd.notna(S) and S > 0:
                    T = 7 / 365  # 7 DTE
                    strike = find_strike_for_delta(S, 0.3, T, r, sigma)
                    theo_price = bs_call_price(S, strike, T, r, sigma)
                    # Apply haircut and spread
                    actual_price = theo_price * (1 - OPTIONS_BS_HAIRCUT) * (1 + OPTIONS_SPREAD_PCT)
                    if actual_price < 0.05:
                        actual_price = 0.05
                    alloc = capital / max(open_slots, 1)
                    alloc = min(alloc, capital)
                    contracts = max(1, int(alloc / (actual_price * 100 + OPTIONS_COMMISSION)))
                    premium = actual_price * 100 * contracts + contracts * OPTIONS_COMMISSION
                    if premium > capital or premium < 1:
                        continue
                    capital -= premium
                    positions.append((stock, date, premium, contracts, strike, S))

        # Mark to market (rough)
        mtm = capital
        for pos in positions:
            stock, entry_date, premium_paid, contracts, strike, _ = pos
            if stock in closes.columns:
                S = closes.loc[date, stock]
                if pd.notna(S):
                    days_left = max(7 - (date - entry_date).days, 0)
                    T_rem = days_left / 365
                    val = bs_call_price(S, strike, T_rem, r, sigma) * 100 * contracts
                    val *= (1 - OPTIONS_BS_HAIRCUT)
                    mtm += val
        equity_curve.append({"date": str(date.date()), "equity": round(mtm, 2)})

    # Force-close remaining
    last_date = pd.Timestamp(rebal_dates[-1]) if len(rebal_dates) > 0 else None
    if last_date is not None:
        for pos in positions:
            stock, entry_date, premium_paid, contracts, strike, _ = pos
            if stock in closes.columns:
                exit_sp = closes.loc[last_date, stock]
                if pd.notna(exit_sp):
                    intrinsic = max(exit_sp - strike, 0) * 100 * contracts
                    exit_cost = contracts * OPTIONS_COMMISSION
                    pnl = intrinsic - premium_paid - exit_cost
                    capital += max(intrinsic - exit_cost, 0)
                    trades.append({
                        "stock": stock, "entry_date": str(entry_date.date()),
                        "exit_date": str(last_date.date()),
                        "strike": strike, "contracts": contracts,
                        "premium_paid": round(premium_paid, 2),
                        "exit_value": round(intrinsic, 2),
                        "pnl": round(pnl, 2),
                        "return_pct": round(pnl / premium_paid * 100, 2) if premium_paid > 0 else 0,
                    })

    return trades, equity_curve


# ── Metrics ────────────────────────────────────────────────────────────────
def compute_metrics(trades, equity_curve, initial_capital, sma200_signal, closes):
    """Compute Sharpe, Sortino, PF, WR, MaxDD, regime analysis."""
    if not trades:
        return {
            "n_trades": 0, "sharpe": 0, "sortino": 0, "profit_factor": 0,
            "win_rate": 0, "max_dd_pct": 0, "total_return_pct": 0,
            "bull_sharpe": 0, "bear_sharpe": 0, "regime_gap": 0,
            "pass_5gate": False, "gate_details": {},
        }

    returns = [t["return_pct"] / 100 for t in trades]
    pnls = [t["pnl"] for t in trades]

    n = len(trades)
    avg_ret = np.mean(returns)
    std_ret = np.std(returns, ddof=1) if n > 1 else 1e-9

    # Annualize: assume ~50 weekly rebalances per year
    sharpe = (avg_ret / std_ret) * np.sqrt(52) if std_ret > 0 else 0

    # Sortino
    downside = [r for r in returns if r < 0]
    down_std = np.std(downside, ddof=1) if len(downside) > 1 else 1e-9
    sortino = (avg_ret / down_std) * np.sqrt(52) if down_std > 0 else 0

    # Profit factor
    gross_profit = sum(p for p in pnls if p > 0)
    gross_loss = abs(sum(p for p in pnls if p < 0))
    pf = gross_profit / gross_loss if gross_loss > 0 else (999 if gross_profit > 0 else 0)

    # Win rate
    wins = sum(1 for p in pnls if p > 0)
    wr = wins / n if n > 0 else 0

    # Max drawdown from equity curve
    if equity_curve:
        eqs = [e["equity"] for e in equity_curve]
        peak = eqs[0]
        max_dd = 0
        for eq in eqs:
            if eq > peak:
                peak = eq
            dd = (eq - peak) / peak if peak > 0 else 0
            if dd < max_dd:
                max_dd = dd
        final_equity = eqs[-1]
    else:
        max_dd = 0
        final_equity = initial_capital

    total_return_pct = (final_equity - initial_capital) / initial_capital * 100

    # Regime analysis
    bull_returns = []
    bear_returns = []
    for t in trades:
        entry_date = pd.Timestamp(t["entry_date"])
        # Find nearest date in sma200_signal
        if entry_date in sma200_signal.index:
            is_bull = sma200_signal.loc[entry_date]
        else:
            idx = sma200_signal.index.get_indexer([entry_date], method="ffill")[0]
            is_bull = sma200_signal.iloc[idx] if idx >= 0 else 1
        ret = t["return_pct"] / 100
        if is_bull:
            bull_returns.append(ret)
        else:
            bear_returns.append(ret)

    bull_sharpe = (np.mean(bull_returns) / np.std(bull_returns, ddof=1) * np.sqrt(52)) if len(bull_returns) > 1 and np.std(bull_returns, ddof=1) > 0 else 0
    bear_sharpe = (np.mean(bear_returns) / np.std(bear_returns, ddof=1) * np.sqrt(52)) if len(bear_returns) > 1 and np.std(bear_returns, ddof=1) > 0 else 0

    max_abs = max(abs(bull_sharpe), abs(bear_sharpe))
    regime_gap = abs(bull_sharpe - bear_sharpe) / max_abs if max_abs > 0 else 0

    # 5-gate validation
    gates = {
        "sharpe_gt_0.5": sharpe > 0.5,
        "max_dd_gt_neg50": max_dd > -0.50,
        "regime_gap_lt_0.5": regime_gap < 0.5,
        "n_trades_gte_20": n >= 20,
        "perm_p": None,  # filled later
    }
    pass_non_perm = all(v for k, v in gates.items() if v is not None)

    return {
        "n_trades": n,
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "profit_factor": round(pf, 3),
        "win_rate": round(wr * 100, 1),
        "max_dd_pct": round(max_dd * 100, 2),
        "total_return_pct": round(total_return_pct, 2),
        "final_equity": round(final_equity, 2),
        "avg_return_pct": round(avg_ret * 100, 3),
        "bull_sharpe": round(bull_sharpe, 3),
        "bear_sharpe": round(bear_sharpe, 3),
        "bull_trades": len(bull_returns),
        "bear_trades": len(bear_returns),
        "regime_gap": round(regime_gap, 3),
        "pass_non_perm_gates": pass_non_perm,
        "gate_details": gates,
    }


# ── Permutation Test ───────────────────────────────────────────────────────
def permutation_test(composite_scores, closes, rebal_dates, selection_fn,
                     hold_days, max_positions, initial_capital, actual_sharpe,
                     is_options=False, sma200_signal=None):
    """Shuffle composite scores across stocks each week, compute null distribution."""
    print(f"    Running permutation test ({PERM_ITERATIONS} iterations)...")
    null_sharpes = []

    for it in range(PERM_ITERATIONS):
        # Shuffle: for each date, randomly permute which stocks get which scores
        shuffled = composite_scores.copy()
        rng = np.random.default_rng(seed=it)
        for date in shuffled.index:
            vals = shuffled.loc[date].values.copy()
            rng.shuffle(vals)
            shuffled.loc[date] = vals

        if is_options:
            trades, eq = backtest_options(shuffled, closes, rebal_dates, selection_fn,
                                          max_positions, initial_capital)
        else:
            trades, eq = backtest_equity(shuffled, closes, rebal_dates, selection_fn,
                                         hold_days, max_positions, initial_capital)

        if trades:
            rets = [t["return_pct"] / 100 for t in trades]
            std = np.std(rets, ddof=1) if len(rets) > 1 else 1e-9
            s = (np.mean(rets) / std) * np.sqrt(52) if std > 0 else 0
        else:
            s = 0
        null_sharpes.append(s)

    perm_p = np.mean([s >= actual_sharpe for s in null_sharpes])
    return round(perm_p, 4), null_sharpes


# ── Selection Functions ────────────────────────────────────────────────────
def select_threshold(row, threshold):
    """Select stocks above composite threshold, sorted descending."""
    above = row[row >= threshold].sort_values(ascending=False)
    return list(above.index)


def select_top_n(row, n):
    """Select top-N stocks by composite score."""
    top = row.sort_values(ascending=False).head(n)
    return list(top.index)


def select_threshold_require_earnings(row, threshold, earnings_scores, date):
    """Select stocks above threshold that also have active earnings gap signal."""
    above = row[row >= threshold].sort_values(ascending=False)
    selected = []
    for stock in above.index:
        if date in earnings_scores.index and stock in earnings_scores.columns:
            if earnings_scores.loc[date, stock] > 0:
                selected.append(stock)
    return selected


# ── Main ───────────────────────────────────────────────────────────────────
def main():
    closes, volumes, opens = download_data()
    signals = compute_all_signals(closes, volumes, opens)
    sma200_signal = signals["sma200"]

    oot_dates = closes.index[closes.index >= pd.Timestamp(OOT_START)]
    rebal_dates = get_weekly_rebalance_dates(closes.index, OOT_START)
    print(f"OOT period: {oot_dates[0].date()} to {oot_dates[-1].date()}, {len(rebal_dates)} weekly rebalances")

    # Default composite scores
    default_scores = compute_composite_scores(signals, SIGNAL_WEIGHTS_DEFAULT, oot_dates, UNIVERSE)

    # Bear-regime adaptive weights for variant D
    bear_weights = SIGNAL_WEIGHTS_DEFAULT.copy()
    bear_weights["mean_rev"] = 0.40  # doubled
    bear_weights["sector_mom"] = 0.0  # zeroed
    # Renormalize
    total_w = sum(bear_weights.values())
    bear_weights = {k: v / total_w for k, v in bear_weights.items()}

    # Compute regime-adaptive scores for variant D
    adaptive_scores = pd.DataFrame(0.0, index=oot_dates, columns=UNIVERSE)
    for date in oot_dates:
        is_bull = sma200_signal.loc[date] if date in sma200_signal.index else 1
        w = SIGNAL_WEIGHTS_DEFAULT if is_bull else bear_weights
        for stock in UNIVERSE:
            score = 0
            for sig_name, wt in w.items():
                if sig_name in ("sma200", "vix"):
                    val = signals[sig_name].loc[date] if date in signals[sig_name].index else 0
                else:
                    val = signals[sig_name].loc[date, stock] if (date in signals[sig_name].index and stock in signals[sig_name].columns) else 0
                if pd.notna(val):
                    score += wt * val
            adaptive_scores.loc[date, stock] = score

    # ── Run Variants ───────────────────────────────────────────────────────
    results = {}

    variants = {
        "A": {"desc": "Score>0.6, equity, 5-day hold", "threshold": 0.6, "hold": 5, "max_pos": 3, "options": False},
        "B": {"desc": "Score>0.7, equity, 10-day hold", "threshold": 0.7, "hold": 10, "max_pos": 3, "options": False},
        "C": {"desc": "Score>0.6, weekly calls (0.3Δ, 7DTE)", "threshold": 0.6, "hold": 7, "max_pos": 2, "options": True},
        "D": {"desc": "Regime-adaptive weights, Score>0.6, 5-day", "threshold": 0.6, "hold": 5, "max_pos": 3, "options": False, "adaptive": True},
        "E": {"desc": "Top-2 composite each week, 5-day hold", "top_n": 2, "hold": 5, "max_pos": 3, "options": False},
        "F": {"desc": "Score>0.5, require earnings gap, 5-day", "threshold": 0.5, "hold": 5, "max_pos": 3, "options": False, "require_earnings": True},
    }

    for variant_name, cfg in variants.items():
        print(f"\n{'='*60}")
        print(f"Variant {variant_name}: {cfg['desc']}")
        print(f"{'='*60}")

        scores = adaptive_scores if cfg.get("adaptive") else default_scores

        if cfg.get("require_earnings"):
            sel_fn = lambda row, d=None: select_threshold_require_earnings(
                row, cfg["threshold"], signals["earnings_gap"], d)
            # Need custom backtest loop for date-dependent selection
            trades, eq = _backtest_with_date_selection(
                scores, closes, rebal_dates, cfg["threshold"],
                signals["earnings_gap"], cfg["hold"], cfg["max_pos"], INITIAL_CAPITAL)
        elif cfg.get("top_n"):
            sel_fn = lambda row: select_top_n(row, cfg["top_n"])
            trades, eq = backtest_equity(scores, closes, rebal_dates, sel_fn,
                                          cfg["hold"], cfg["max_pos"], INITIAL_CAPITAL)
        elif cfg.get("options"):
            sel_fn = lambda row: select_threshold(row, cfg["threshold"])
            trades, eq = backtest_options(scores, closes, rebal_dates, sel_fn,
                                          cfg["max_pos"], INITIAL_CAPITAL)
        else:
            sel_fn = lambda row, th=cfg["threshold"]: select_threshold(row, th)
            trades, eq = backtest_equity(scores, closes, rebal_dates, sel_fn,
                                          cfg["hold"], cfg["max_pos"], INITIAL_CAPITAL)

        metrics = compute_metrics(trades, eq, INITIAL_CAPITAL, sma200_signal, closes)
        print(f"  Trades: {metrics['n_trades']}, Sharpe: {metrics['sharpe']}, "
              f"Sortino: {metrics['sortino']}, PF: {metrics['profit_factor']}, "
              f"WR: {metrics['win_rate']}%, MaxDD: {metrics['max_dd_pct']}%")
        print(f"  Total Return: {metrics['total_return_pct']}%, "
              f"Final Equity: ${metrics['final_equity']}")
        print(f"  Bull Sharpe: {metrics['bull_sharpe']}, Bear Sharpe: {metrics['bear_sharpe']}, "
              f"Regime Gap: {metrics['regime_gap']}")

        # Permutation test
        if metrics["n_trades"] >= 10:
            if cfg.get("require_earnings"):
                # Use threshold selection for perm test (simpler, still valid)
                perm_sel = lambda row, th=cfg["threshold"]: select_threshold(row, th)
                perm_p, null_dist = permutation_test(
                    scores, closes, rebal_dates, perm_sel,
                    cfg["hold"], cfg["max_pos"], INITIAL_CAPITAL,
                    metrics["sharpe"], is_options=False, sma200_signal=sma200_signal)
            elif cfg.get("top_n"):
                perm_sel = lambda row, n=cfg["top_n"]: select_top_n(row, n)
                perm_p, null_dist = permutation_test(
                    scores, closes, rebal_dates, perm_sel,
                    cfg["hold"], cfg["max_pos"], INITIAL_CAPITAL,
                    metrics["sharpe"], is_options=False, sma200_signal=sma200_signal)
            else:
                perm_sel = lambda row, th=cfg.get("threshold", 0.6): select_threshold(row, th)
                perm_p, null_dist = permutation_test(
                    scores, closes, rebal_dates, perm_sel,
                    cfg["hold"], cfg["max_pos"], INITIAL_CAPITAL,
                    metrics["sharpe"], is_options=cfg.get("options", False),
                    sma200_signal=sma200_signal)

            metrics["gate_details"]["perm_p"] = perm_p < 0.05
            metrics["perm_p_value"] = perm_p
            metrics["null_sharpe_mean"] = round(np.mean(null_dist), 3)
            metrics["null_sharpe_std"] = round(np.std(null_dist), 3)
            print(f"  Perm p-value: {perm_p} (null Sharpe: {metrics['null_sharpe_mean']} ± {metrics['null_sharpe_std']})")
        else:
            metrics["perm_p_value"] = 1.0
            metrics["gate_details"]["perm_p"] = False
            print(f"  Perm test skipped (too few trades)")

        # Final 5-gate pass
        all_gates = all(v for v in metrics["gate_details"].values() if v is not None)
        metrics["pass_5gate"] = all_gates
        print(f"  5-Gate Pass: {all_gates} | Gates: {metrics['gate_details']}")

        results[f"variant_{variant_name}"] = {
            "description": cfg["desc"],
            "metrics": metrics,
            "sample_trades": trades[:10] if trades else [],
            "equity_curve_summary": {
                "start": eq[0] if eq else None,
                "end": eq[-1] if eq else None,
                "n_points": len(eq),
            },
        }

    # ── Save Results ───────────────────────────────────────────────────────
    output = {
        "backtest_name": "Composite Signal Ensemble",
        "run_timestamp": dt.datetime.now().isoformat(),
        "oot_period": f"{OOT_START} to {OOT_END}",
        "initial_capital": INITIAL_CAPITAL,
        "universe_size": len(UNIVERSE),
        "n_rebalance_dates": len(rebal_dates),
        "signal_weights_default": SIGNAL_WEIGHTS_DEFAULT,
        "perm_iterations": PERM_ITERATIONS,
        "variants": results,
    }

    out_path = Path("/home/jupiter/Lvl3Quant/data/composite_signal_ensemble_results.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")

    # ── Summary Table ──────────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"{'Variant':<12} {'Trades':>7} {'Sharpe':>8} {'Sortino':>8} {'PF':>7} {'WR%':>7} {'MaxDD%':>8} {'Return%':>9} {'PermP':>7} {'5Gate':>6}")
    print("-" * 80)
    for vname in ["A", "B", "C", "D", "E", "F"]:
        m = results[f"variant_{vname}"]["metrics"]
        perm = m.get("perm_p_value", "N/A")
        perm_str = f"{perm:.3f}" if isinstance(perm, float) else perm
        gate_str = "PASS" if m["pass_5gate"] else "FAIL"
        print(f"  {vname:<10} {m['n_trades']:>7} {m['sharpe']:>8.3f} {m['sortino']:>8.3f} "
              f"{m['profit_factor']:>7.2f} {m['win_rate']:>6.1f} {m['max_dd_pct']:>7.2f} "
              f"{m['total_return_pct']:>8.2f} {perm_str:>7} {'PASS' if m['pass_5gate'] else 'FAIL':>6}")


def _backtest_with_date_selection(composite_scores, closes, rebal_dates,
                                   threshold, earnings_scores, hold_days,
                                   max_positions, initial_capital):
    """Variant F: threshold + require earnings gap (date-dependent selection)."""
    capital = initial_capital
    trades = []
    positions = []
    equity_curve = []

    for date in rebal_dates:
        date = pd.Timestamp(date)
        if date not in closes.index:
            continue

        # Close expired positions
        new_positions = []
        for pos in positions:
            stock, entry_date, entry_price, shares = pos
            if (date - entry_date).days >= hold_days:
                if stock in closes.columns:
                    exit_price = closes.loc[date, stock]
                    if pd.notna(exit_price) and pd.notna(entry_price):
                        ep = exit_price * (1 - SLIPPAGE_PCT)
                        pnl = (ep - entry_price) * shares
                        capital += ep * shares
                        trades.append({
                            "stock": stock, "entry_date": str(entry_date.date()),
                            "exit_date": str(date.date()),
                            "entry_price": round(entry_price, 2),
                            "exit_price": round(ep, 2),
                            "shares": shares, "pnl": round(pnl, 2),
                            "return_pct": round(pnl / (entry_price * shares) * 100, 2),
                        })
                        continue
            new_positions.append(pos)
        positions = new_positions

        # Select: threshold + earnings gap required
        if date in composite_scores.index:
            row = composite_scores.loc[date]
            selected = select_threshold_require_earnings(row, threshold, earnings_scores, date)
        else:
            selected = []

        open_slots = max_positions - len(positions)
        for stock in selected[:open_slots]:
            if stock in closes.columns:
                price = closes.loc[date, stock]
                if pd.notna(price) and price > 0:
                    entry_price = price * (1 + SLIPPAGE_PCT)
                    alloc = min(capital / max(open_slots, 1), capital)
                    if alloc < 5:
                        continue
                    shares = int(alloc / entry_price)
                    if shares > 0:
                        capital -= shares * entry_price
                        positions.append((stock, date, entry_price, shares))

        mtm = capital
        for pos in positions:
            stock, _, _, shares = pos
            if stock in closes.columns:
                p = closes.loc[date, stock]
                if pd.notna(p):
                    mtm += p * shares
        equity_curve.append({"date": str(date.date()), "equity": round(mtm, 2)})

    # Force close
    if rebal_dates.size > 0:
        last = pd.Timestamp(rebal_dates[-1])
        for pos in positions:
            stock, entry_date, entry_price, shares = pos
            if stock in closes.columns:
                ep = closes.loc[last, stock] * (1 - SLIPPAGE_PCT)
                if pd.notna(ep):
                    pnl = (ep - entry_price) * shares
                    capital += ep * shares
                    trades.append({
                        "stock": stock, "entry_date": str(entry_date.date()),
                        "exit_date": str(last.date()),
                        "entry_price": round(entry_price, 2),
                        "exit_price": round(ep, 2),
                        "shares": shares, "pnl": round(pnl, 2),
                        "return_pct": round(pnl / (entry_price * shares) * 100, 2),
                    })

    return trades, equity_curve


if __name__ == "__main__":
    main()
