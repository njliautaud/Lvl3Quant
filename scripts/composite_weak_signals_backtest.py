#!/usr/bin/env python3
"""
Composite Weak Signal Aggregation Backtest
==========================================
Combines 7 individually sub-threshold signals into a composite score.
"Wisdom of crowds" — multiple weak signals agreeing should amplify edge.

Signals (daily, SPY/QQQ):
  1. VIX regime (+1/-1/0)
  2. Trend: SPY vs 50-SMA (+1/-1)
  3. Breadth: SPY 20d return (+1/-1/0)
  4. Bond signal: TLT 20d momentum (+1/-1/0)
  5. Mean reversion: RSI(14) (+2/-1/0)
  6. VIX spike fade (+1/0)
  7. Seasonal: month-end/start (+0.5/0)

Variants A-F with different thresholds, instruments, position sizing.
OOT: Jan 2022 - Jul 2026. Walk-forward daily. $645 initial capital.
5-gate validation: Sharpe>0.5, perm p<0.05, regime gap<0.5, MaxDD>-50%, >=20 trades.
"""

import sys
import json
import warnings
import datetime as dt
from pathlib import Path
from calendar import monthrange

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)

# ── Constants ──────────────────────────────────────────────────────────────
INITIAL_CAPITAL = 645.0
OOT_START = "2022-01-01"
OOT_END = "2026-07-28"
DATA_START = "2020-06-01"
SLIPPAGE_PCT = 0.0002  # 0.02%
PERM_ITERATIONS = 1000

SECTOR_ETFS = ["XLK", "XLF", "XLE", "XLV", "XLC", "XLY", "XLI", "XLB", "XLRE", "XLU", "XLP"]
GROWTH_STOCKS = ["SOFI", "SNAP", "HOOD", "RBLX", "PLTR", "PINS"]


# ── Data Download ──────────────────────────────────────────────────────────
def download_data():
    """Download all required price data."""
    print("Downloading price data...")
    tickers = ["SPY", "QQQ", "^VIX", "TLT", "SH"] + SECTOR_ETFS + GROWTH_STOCKS
    data = yf.download(tickers, start=DATA_START, end=OOT_END, auto_adjust=True, progress=False)
    closes = data["Close"].ffill()
    return closes


# ── Signal Computation ─────────────────────────────────────────────────────
def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def compute_signals(closes):
    """Compute all 7 signals as daily series."""
    spy = closes["SPY"]
    vix = closes["^VIX"]
    tlt = closes["TLT"]

    signals = pd.DataFrame(index=closes.index)

    # 1. VIX regime
    signals["vix_regime"] = 0.0
    signals.loc[vix < 20, "vix_regime"] = 1.0
    signals.loc[vix > 25, "vix_regime"] = -1.0

    # 2. Trend: SPY vs 50-SMA
    sma50 = spy.rolling(50).mean()
    signals["trend"] = 0.0
    signals.loc[spy > sma50, "trend"] = 1.0
    signals.loc[spy < sma50, "trend"] = -1.0

    # 3. Breadth: SPY 20d return
    ret_20d = spy.pct_change(20)
    signals["breadth"] = 0.0
    signals.loc[ret_20d > 0, "breadth"] = 1.0
    signals.loc[ret_20d < -0.05, "breadth"] = -1.0

    # 4. Bond signal: TLT 20d momentum
    tlt_mom = tlt.pct_change(20)
    signals["bond"] = 0.0
    signals.loc[tlt_mom > 0, "bond"] = 1.0
    signals.loc[tlt_mom < -0.03, "bond"] = -1.0

    # 5. Mean reversion: RSI(14)
    rsi = compute_rsi(spy, 14)
    signals["mean_rev"] = 0.0
    signals.loc[rsi < 30, "mean_rev"] = 2.0
    signals.loc[rsi > 70, "mean_rev"] = -1.0

    # 6. VIX spike fade: VIX dropped >15% from 5d high
    vix_5d_high = vix.rolling(5).max()
    vix_drop = (vix - vix_5d_high) / vix_5d_high
    signals["vix_fade"] = 0.0
    signals.loc[vix_drop < -0.15, "vix_fade"] = 1.0

    # 7. Seasonal: last 3 days of month + first 3 of next month (vectorized)
    days = signals.index.day
    # Get last day of each month
    last_days = pd.Series([monthrange(d.year, d.month)[1] for d in signals.index], index=signals.index)
    is_month_boundary = (days >= last_days - 2) | (days <= 3)
    signals["seasonal"] = np.where(is_month_boundary, 0.5, 0.0)

    # Composite score
    signals["composite"] = signals[["vix_regime", "trend", "breadth", "bond",
                                     "mean_rev", "vix_fade", "seasonal"]].sum(axis=1)

    return signals


# ── Pre-compute helpers for variants E and F ────────────────────────────
def precompute_sector_picks(closes):
    """Pre-compute best sector ETF by 20d momentum for each date."""
    mom = pd.DataFrame()
    for etf in SECTOR_ETFS:
        if etf in closes.columns:
            mom[etf] = closes[etf].pct_change(20)
    best_sector = mom.idxmax(axis=1)
    return best_sector


def precompute_growth_picks(closes):
    """Pre-compute cheapest affordable growth stock above 200-SMA for each date."""
    sma200 = {}
    for stk in GROWTH_STOCKS:
        if stk in closes.columns:
            sma200[stk] = closes[stk].rolling(200).mean()

    picks = pd.Series(index=closes.index, dtype=object)
    for date in closes.index:
        candidates = []
        for stk in GROWTH_STOCKS:
            if stk in closes.columns and stk in sma200:
                price = closes[stk].loc[date]
                sma_val = sma200[stk].loc[date]
                if not np.isnan(price) and not np.isnan(sma_val) and price > sma_val:
                    if price <= INITIAL_CAPITAL * 0.95:  # rough affordability
                        candidates.append((stk, price))
        if candidates:
            candidates.sort(key=lambda x: x[1])
            picks.loc[date] = candidates[0][0]
        else:
            picks.loc[date] = None
    return picks


# ── Fast Backtest (vectorized where possible) ──────────────────────────
def backtest_variant_fast(composite_scores, closes, variant,
                          sector_picks=None, growth_picks=None,
                          initial_capital=INITIAL_CAPITAL):
    """
    Run a single variant backtest. Returns equity array and trade list.
    Optimized for speed in permutation testing.
    """
    oot_mask = composite_scores.index >= OOT_START
    dates = composite_scores.index[oot_mask]
    scores = composite_scores.loc[oot_mask].values

    n = len(dates)
    if n == 0:
        return np.array([initial_capital]), []

    # Pre-fetch price arrays for speed
    qqq_prices = closes["QQQ"].reindex(dates).values
    sh_prices = closes["SH"].reindex(dates).values if "SH" in closes.columns else np.zeros(n)

    # For E/F, get the relevant ticker prices
    if variant == "E" and sector_picks is not None:
        e_tickers = sector_picks.reindex(dates).values
    if variant == "F" and growth_picks is not None:
        f_tickers = growth_picks.reindex(dates).values

    cash = initial_capital
    shares = 0
    ticker_held = None
    entry_price = None
    entry_date_idx = None
    equity = np.empty(n)
    trades = []

    def get_price(ticker, i):
        if ticker == "QQQ":
            return qqq_prices[i]
        elif ticker == "SH":
            return sh_prices[i]
        elif ticker in closes.columns:
            return closes[ticker].iloc[closes.index.get_indexer([dates[i]], method='nearest')[0]]
        return 0.0

    # Pre-cache all possible ticker prices for the OOT period
    price_cache = {}
    price_cache["QQQ"] = qqq_prices
    price_cache["SH"] = sh_prices
    for etf in SECTOR_ETFS:
        if etf in closes.columns:
            price_cache[etf] = closes[etf].reindex(dates).values
    for stk in GROWTH_STOCKS:
        if stk in closes.columns:
            price_cache[stk] = closes[stk].reindex(dates).values

    def get_price_fast(ticker, i):
        """Fast price lookup using pre-cached arrays."""
        if ticker is None:
            return 0.0
        arr = price_cache.get(ticker)
        if arr is not None:
            val = arr[i]
            if not np.isnan(val):
                return val
        return 0.0

    for i in range(n):
        score = scores[i]

        # Current price of held asset
        current_price = get_price_fast(ticker_held, i) if ticker_held else 0.0
        port_value = cash + shares * current_price

        # Determine target
        target_ticker = None
        target_frac = 0.0

        if variant == "A":
            if score >= 2:
                target_ticker, target_frac = "QQQ", 1.0
            elif score < 0:
                target_ticker, target_frac = None, 0.0
            else:
                target_ticker = ticker_held
                target_frac = 1.0 if ticker_held else 0.0

        elif variant == "B":
            if score >= 3:
                target_ticker, target_frac = "QQQ", 1.0
            elif score < 0:
                target_ticker, target_frac = None, 0.0
            else:
                target_ticker = ticker_held
                target_frac = 1.0 if ticker_held else 0.0

        elif variant == "C":
            if score >= 5:
                target_ticker, target_frac = "QQQ", 1.0
            elif score >= 3:
                target_ticker, target_frac = "QQQ", 0.75
            elif score >= 1:
                target_ticker, target_frac = "QQQ", 0.25
            elif score < 0:
                target_ticker, target_frac = None, 0.0
            else:
                target_ticker = ticker_held
                target_frac = 0.25 if ticker_held else 0.0

        elif variant == "D":
            if score >= 3:
                target_ticker, target_frac = "QQQ", 1.0
            elif score <= -2:
                target_ticker, target_frac = "SH", 1.0
            else:
                target_ticker, target_frac = None, 0.0

        elif variant == "E":
            if score >= 2:
                t = e_tickers[i] if sector_picks is not None else None
                if t and not (isinstance(t, float) and np.isnan(t)):
                    target_ticker, target_frac = t, 1.0
                else:
                    target_ticker, target_frac = "QQQ", 1.0  # fallback
            elif score < 0:
                target_ticker, target_frac = None, 0.0
            else:
                target_ticker = ticker_held
                target_frac = 1.0 if ticker_held else 0.0

        elif variant == "F":
            if score >= 3:
                t = f_tickers[i] if growth_picks is not None else None
                if t and t is not None and not (isinstance(t, float) and np.isnan(t)):
                    target_ticker, target_frac = t, 1.0
                else:
                    target_ticker = ticker_held
                    target_frac = 1.0 if ticker_held else 0.0
            elif score < 0:
                target_ticker, target_frac = None, 0.0
            else:
                target_ticker = ticker_held
                target_frac = 1.0 if ticker_held else 0.0

        # Execute trades
        need_sell = (ticker_held is not None) and (target_ticker != ticker_held or target_frac == 0)
        need_buy = (target_ticker is not None) and (target_ticker != ticker_held or (ticker_held is None and target_frac > 0))

        # Rebalance for C
        if variant == "C" and ticker_held == target_ticker and ticker_held is not None:
            current_invested = shares * current_price
            target_invested = port_value * target_frac
            if abs(current_invested - target_invested) / max(port_value, 1) > 0.10:
                need_sell = True
                need_buy = True

        if need_sell and ticker_held is not None and shares > 0:
            sell_price = current_price * (1 - SLIPPAGE_PCT)
            proceeds = shares * sell_price
            cash += proceeds
            if entry_price is not None:
                pnl_pct = (sell_price - entry_price) / entry_price
                trades.append({
                    "entry_date": str(dates[entry_date_idx].date()) if entry_date_idx is not None else "",
                    "exit_date": str(dates[i].date()),
                    "ticker": ticker_held,
                    "entry_price": round(entry_price, 2),
                    "exit_price": round(sell_price, 2),
                    "pnl_pct": round(pnl_pct * 100, 2),
                    "pnl_dollar": round(proceeds - shares * entry_price, 2),
                })
            shares = 0
            ticker_held = None
            entry_price = None
            entry_date_idx = None

        if need_buy and target_ticker is not None and target_frac > 0:
            port_value = cash
            buy_amount = port_value * target_frac
            raw_price = get_price_fast(target_ticker, i)
            if raw_price > 0:
                buy_price = raw_price * (1 + SLIPPAGE_PCT)
                shares_to_buy = int(buy_amount / buy_price)
                if shares_to_buy > 0:
                    cost = shares_to_buy * buy_price
                    cash -= cost
                    shares = shares_to_buy
                    ticker_held = target_ticker
                    entry_price = buy_price
                    entry_date_idx = i

        # Record equity — re-fetch current price after any trades
        if ticker_held and shares > 0:
            cur_p = get_price_fast(ticker_held, i)
            final_value = cash + shares * cur_p
        else:
            final_value = cash
        # Safety: equity should never be negative or absurdly low
        if final_value < 0:
            final_value = cash
        equity[i] = final_value

    # Close open position at end
    if ticker_held and shares > 0:
        last_price = get_price_fast(ticker_held, n - 1) * (1 - SLIPPAGE_PCT)
        proceeds = shares * last_price
        if entry_price:
            pnl_pct = (last_price - entry_price) / entry_price
            trades.append({
                "entry_date": str(dates[entry_date_idx].date()) if entry_date_idx is not None else "",
                "exit_date": str(dates[n - 1].date()),
                "ticker": ticker_held,
                "entry_price": round(entry_price, 2),
                "exit_price": round(last_price, 2),
                "pnl_pct": round(pnl_pct * 100, 2),
                "pnl_dollar": round(proceeds - shares * entry_price, 2),
            })

    return equity, trades


# ── Metrics (from equity array) ────────────────────────────────────────────
def compute_metrics_fast(equity, trades, initial_capital=INITIAL_CAPITAL):
    """Compute performance metrics from equity numpy array."""
    if len(equity) < 2:
        return {"sharpe": 0, "sortino": 0, "max_drawdown_pct": 0,
                "total_return_pct": 0, "n_trades": 0, "win_rate": 0,
                "profit_factor": 0, "final_equity": initial_capital,
                "annualized_return_pct": 0, "annualized_vol_pct": 0,
                "avg_win_pct": 0, "avg_loss_pct": 0, "total_pnl": 0}

    daily_returns = np.diff(equity) / equity[:-1]
    daily_returns = daily_returns[~np.isnan(daily_returns)]

    final_equity = equity[-1]
    total_return = (final_equity - initial_capital) / initial_capital

    n_years = len(daily_returns) / 252
    ann_return = (1 + total_return) ** (1 / max(n_years, 0.01)) - 1
    ann_vol = np.std(daily_returns) * np.sqrt(252)
    sharpe = ann_return / max(ann_vol, 1e-6)

    downside = daily_returns[daily_returns < 0]
    downside_vol = np.std(downside) * np.sqrt(252) if len(downside) > 0 else 1e-6
    sortino = ann_return / max(downside_vol, 1e-6)

    cummax = np.maximum.accumulate(equity)
    drawdown = (equity - cummax) / cummax
    max_dd = np.min(drawdown)

    n_trades = len(trades)
    if n_trades > 0:
        wins = [t for t in trades if t["pnl_pct"] > 0]
        losses = [t for t in trades if t["pnl_pct"] <= 0]
        win_rate = len(wins) / n_trades
        avg_win = np.mean([t["pnl_pct"] for t in wins]) if wins else 0
        avg_loss = np.mean([abs(t["pnl_pct"]) for t in losses]) if losses else 1e-6
        win_pnl = sum(t["pnl_dollar"] for t in wins)
        loss_pnl = abs(sum(t["pnl_dollar"] for t in losses))
        profit_factor = win_pnl / max(loss_pnl, 0.01) if losses else (np.inf if win_pnl > 0 else 0)
        total_pnl = sum(t["pnl_dollar"] for t in trades)
    else:
        win_rate = avg_win = avg_loss = profit_factor = total_pnl = 0

    return {
        "final_equity": round(float(final_equity), 2),
        "total_return_pct": round(total_return * 100, 2),
        "annualized_return_pct": round(ann_return * 100, 2),
        "annualized_vol_pct": round(ann_vol * 100, 2),
        "sharpe": round(float(sharpe), 3),
        "sortino": round(float(sortino), 3),
        "max_drawdown_pct": round(float(max_dd) * 100, 2),
        "n_trades": n_trades,
        "win_rate": round(win_rate * 100, 1),
        "avg_win_pct": round(float(avg_win), 2),
        "avg_loss_pct": round(float(avg_loss), 2),
        "profit_factor": round(float(min(profit_factor, 999)), 3),
        "total_pnl": round(float(total_pnl), 2),
    }


def sharpe_from_equity(equity):
    """Fast Sharpe calculation from equity array."""
    if len(equity) < 10:
        return 0.0
    daily_returns = np.diff(equity) / equity[:-1]
    daily_returns = daily_returns[~np.isnan(daily_returns)]
    if len(daily_returns) < 10:
        return 0.0
    total_return = (equity[-1] - equity[0]) / equity[0]
    n_years = len(daily_returns) / 252
    ann_return = (1 + total_return) ** (1 / max(n_years, 0.01)) - 1
    ann_vol = np.std(daily_returns) * np.sqrt(252)
    return ann_return / max(ann_vol, 1e-6)


# ── Regime Analysis ───────────────────────────────────────────────────────
def regime_analysis(equity, dates, closes):
    """Compute Sharpe in bull vs bear regimes."""
    spy = closes["SPY"]
    sma200 = spy.rolling(200).mean()

    daily_ret = np.diff(equity) / equity[:-1]
    ret_dates = dates[1:]

    is_bull = spy.reindex(ret_dates).values > sma200.reindex(ret_dates).values

    bull_rets = daily_ret[is_bull]
    bear_rets = daily_ret[~is_bull]

    bull_sharpe = (np.mean(bull_rets) / max(np.std(bull_rets), 1e-6)) * np.sqrt(252) if len(bull_rets) > 20 else 0
    bear_sharpe = (np.mean(bear_rets) / max(np.std(bear_rets), 1e-6)) * np.sqrt(252) if len(bear_rets) > 20 else 0

    gap = abs(bull_sharpe - bear_sharpe) / max(abs(bull_sharpe), abs(bear_sharpe), 1e-6)

    return {
        "bull_sharpe": round(float(bull_sharpe), 3),
        "bear_sharpe": round(float(bear_sharpe), 3),
        "regime_gap": round(float(gap), 3),
        "bull_days": int(np.sum(is_bull)),
        "bear_days": int(np.sum(~is_bull)),
    }


# ── Permutation Test (optimized) ──────────────────────────────────────────
def permutation_test(signals_df, closes, variant, actual_sharpe,
                     sector_picks=None, growth_picks=None,
                     n_perms=PERM_ITERATIONS):
    """
    Shuffle composite scores across OOT dates, re-run backtest.
    Tests if timing entries by composite score is better than random.
    """
    print(f"  Running {n_perms} permutations...", end=" ")
    perm_sharpes = np.empty(n_perms)

    oot_mask = signals_df.index >= OOT_START
    oot_scores = signals_df.loc[oot_mask, "composite"].values.copy()

    # Create a working copy of composite scores
    composite_series = signals_df["composite"].copy()

    for k in range(n_perms):
        # Shuffle OOT scores in place
        shuffled_scores = oot_scores.copy()
        np.random.shuffle(shuffled_scores)
        composite_series.loc[oot_mask] = shuffled_scores

        eq, _ = backtest_variant_fast(composite_series, closes, variant,
                                       sector_picks, growth_picks)
        perm_sharpes[k] = sharpe_from_equity(eq)

        if (k + 1) % 250 == 0:
            print(f"{k+1}", end=" ")

    p_value = np.mean(perm_sharpes >= actual_sharpe)
    print(f"done. p={p_value:.4f}")

    return {
        "p_value": round(float(p_value), 4),
        "perm_mean_sharpe": round(float(np.mean(perm_sharpes)), 3),
        "perm_std_sharpe": round(float(np.std(perm_sharpes)), 3),
        "actual_vs_perm_mean": round(float(actual_sharpe - np.mean(perm_sharpes)), 3),
    }


# ── 5-Gate Validation ─────────────────────────────────────────────────────
def validate_5gate(metrics, regime, perm):
    gates = {
        "sharpe_gt_0.5": metrics["sharpe"] > 0.5,
        "perm_p_lt_0.05": perm["p_value"] < 0.05,
        "regime_gap_lt_0.5": regime["regime_gap"] < 0.5,
        "max_dd_gt_neg50": metrics["max_drawdown_pct"] > -50,
        "min_20_trades": metrics["n_trades"] >= 20,
    }
    gates["all_passed"] = all(gates.values())
    return gates


# ── Main ───────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("COMPOSITE WEAK SIGNAL AGGREGATION BACKTEST")
    print("=" * 70)

    closes = download_data()
    print(f"Data shape: {closes.shape}, range: {closes.index[0].date()} to {closes.index[-1].date()}")

    signals = compute_signals(closes)
    print(f"\nComposite score stats (OOT period):")
    oot_comp = signals.loc[signals.index >= OOT_START, "composite"]
    print(f"  Mean: {oot_comp.mean():.2f}, Std: {oot_comp.std():.2f}")
    print(f"  Min: {oot_comp.min():.1f}, Max: {oot_comp.max():.1f}")
    print(f"  Score >= 2: {(oot_comp >= 2).sum()} days ({(oot_comp >= 2).mean()*100:.1f}%)")
    print(f"  Score >= 3: {(oot_comp >= 3).sum()} days ({(oot_comp >= 3).mean()*100:.1f}%)")
    print(f"  Score < 0:  {(oot_comp < 0).sum()} days ({(oot_comp < 0).mean()*100:.1f}%)")

    print(f"\nIndividual signal means (OOT):")
    for col in ["vix_regime", "trend", "breadth", "bond", "mean_rev", "vix_fade", "seasonal"]:
        vals = signals.loc[signals.index >= OOT_START, col]
        print(f"  {col:15s}: mean={vals.mean():+.3f}, bullish={((vals > 0).sum()/len(vals))*100:.1f}%")

    # Pre-compute helpers for variants E and F
    print("\nPre-computing sector & growth picks...")
    sector_picks = precompute_sector_picks(closes)
    growth_picks = precompute_growth_picks(closes)

    variants = {
        "A": "Threshold 2+ (QQQ)",
        "B": "Threshold 3+ (QQQ, high conviction)",
        "C": "Scaled position sizing",
        "D": "Long/Short (QQQ + SH)",
        "E": "Sector rotation (top momentum ETF)",
        "F": "Growth stock (cheapest above 200-SMA)",
    }

    oot_dates = signals.index[signals.index >= OOT_START]
    results = {}
    np.random.seed(42)

    for var_code, var_name in variants.items():
        print(f"\n{'─' * 60}")
        print(f"Variant {var_code}: {var_name}")
        print(f"{'─' * 60}")

        eq, trades = backtest_variant_fast(signals["composite"], closes, var_code,
                                           sector_picks, growth_picks)
        metrics = compute_metrics_fast(eq, trades)
        regime = regime_analysis(eq, oot_dates, closes)
        perm = permutation_test(signals, closes, var_code, metrics["sharpe"],
                                sector_picks, growth_picks)
        gates = validate_5gate(metrics, regime, perm)

        print(f"  Final equity:     ${metrics['final_equity']:.2f} (from $645)")
        print(f"  Total return:     {metrics['total_return_pct']:.1f}%")
        print(f"  Sharpe:           {metrics['sharpe']:.3f}")
        print(f"  Sortino:          {metrics['sortino']:.3f}")
        print(f"  Max drawdown:     {metrics['max_drawdown_pct']:.1f}%")
        print(f"  Win rate:         {metrics['win_rate']:.1f}% ({metrics['n_trades']} trades)")
        print(f"  Profit factor:    {metrics['profit_factor']:.3f}")
        print(f"  Regime gap:       {regime['regime_gap']:.3f} (bull={regime['bull_sharpe']:.3f}, bear={regime['bear_sharpe']:.3f})")
        print(f"  Perm p-value:     {perm['p_value']:.4f}")

        status = "PASS" if gates["all_passed"] else "FAIL"
        failed = [k for k, v in gates.items() if not v and k != "all_passed"]
        print(f"  5-Gate:           {status}" + (f" (failed: {', '.join(failed)})" if failed else ""))

        # Convert equity to serializable list-of-dicts (sampled for JSON size)
        eq_sampled = [{"date": str(oot_dates[i].date()), "equity": round(float(eq[i]), 2)}
                      for i in range(0, len(eq), max(1, len(eq) // 50))]

        results[f"variant_{var_code}"] = {
            "name": var_name,
            "metrics": metrics,
            "regime": regime,
            "permutation": perm,
            "five_gate": gates,
            "top_5_trades": sorted(trades, key=lambda t: t["pnl_pct"], reverse=True)[:5] if trades else [],
            "bottom_5_trades": sorted(trades, key=lambda t: t["pnl_pct"])[:5] if trades else [],
            "equity_curve_sampled": eq_sampled,
        }

    # Buy-and-hold QQQ benchmark
    print(f"\n{'─' * 60}")
    print(f"BENCHMARK: Buy & Hold QQQ")
    print(f"{'─' * 60}")
    qqq = closes["QQQ"]
    qqq_oot = qqq.loc[qqq.index >= OOT_START]
    qqq_start = qqq_oot.iloc[0]
    shares_bnh = int(INITIAL_CAPITAL / qqq_start)
    cash_bnh = INITIAL_CAPITAL - shares_bnh * qqq_start
    bnh_final = shares_bnh * qqq_oot.iloc[-1] + cash_bnh
    bnh_ret = (bnh_final - INITIAL_CAPITAL) / INITIAL_CAPITAL
    bnh_daily = (qqq_oot / qqq_oot.shift(1) - 1).dropna()
    bnh_sharpe = (bnh_daily.mean() / bnh_daily.std()) * np.sqrt(252)
    bnh_dd = ((qqq_oot - qqq_oot.cummax()) / qqq_oot.cummax()).min()
    print(f"  Final equity:     ${bnh_final:.2f}")
    print(f"  Total return:     {bnh_ret*100:.1f}%")
    print(f"  Sharpe:           {bnh_sharpe:.3f}")
    print(f"  Max drawdown:     {bnh_dd*100:.1f}%")

    results["benchmark_qqq_bnh"] = {
        "final_equity": round(float(bnh_final), 2),
        "total_return_pct": round(float(bnh_ret * 100), 2),
        "sharpe": round(float(bnh_sharpe), 3),
        "max_drawdown_pct": round(float(bnh_dd * 100), 2),
    }

    # Summary
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    passing = []
    for var_code in variants:
        key = f"variant_{var_code}"
        passed = results[key]["five_gate"]["all_passed"]
        sharpe = results[key]["metrics"]["sharpe"]
        ret = results[key]["metrics"]["total_return_pct"]
        perm_p = results[key]["permutation"]["p_value"]
        tag = "PASS" if passed else "FAIL"
        print(f"  {var_code}: Sharpe={sharpe:.3f} Ret={ret:.1f}% p={perm_p:.3f} [{tag}]")
        if passed:
            passing.append(var_code)

    if passing:
        print(f"\n  >>> PASSING VARIANTS: {', '.join(passing)}")
        best = max(passing, key=lambda v: results[f"variant_{v}"]["metrics"]["sharpe"])
        print(f"  >>> BEST: Variant {best} -- {variants[best]}")
        print(f"      Sharpe={results[f'variant_{best}']['metrics']['sharpe']:.3f}, "
              f"Return={results[f'variant_{best}']['metrics']['total_return_pct']:.1f}%, "
              f"WR={results[f'variant_{best}']['metrics']['win_rate']:.1f}%")
    else:
        print(f"\n  >>> NO VARIANTS PASSED ALL 5 GATES")
        print(f"  >>> Composite approach may need signal refinement or different aggregation.")

    # Save results
    results["metadata"] = {
        "backtest_date": str(dt.datetime.now()),
        "oot_period": f"{OOT_START} to {OOT_END}",
        "initial_capital": INITIAL_CAPITAL,
        "slippage_pct": SLIPPAGE_PCT,
        "perm_iterations": PERM_ITERATIONS,
        "signals": [
            "VIX regime (<20/+1, >25/-1)",
            "SPY vs 50-SMA (+1/-1)",
            "SPY 20d return (>0/+1, <-5%/-1)",
            "TLT 20d momentum (>0/+1, <-3%/-1)",
            "RSI(14) (<30/+2, >70/-1)",
            "VIX spike fade (>15% drop from 5d high/+1)",
            "Seasonal (month boundary/+0.5)",
        ],
    }

    out_path = Path("/home/jupiter/Lvl3Quant/data/composite_weak_signals_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
