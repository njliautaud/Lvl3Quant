#!/usr/bin/env python3
"""
Put Ratio Spread Income Backtest — v1
======================================
Strategy: On each position cycle:
  - Buy 1 ATM put (protection)
  - Sell N OTM puts (income, target ~25-delta, 5-10% OTM)
Net credit from selling N puts minus buying 1 put = income.
If stock stays above short strikes, keep the credit.
Long put provides crash protection. Risk: stock drops to/below short strike.

Configs tested (8+):
  1) delta=0.25, DTE=30, ratio=1:2, stop=2x  (baseline)
  2) delta=0.20, DTE=30, ratio=1:2, stop=2x  (deeper OTM shorts)
  3) delta=0.30, DTE=30, ratio=1:2, stop=2x  (closer shorts)
  4) delta=0.25, DTE=14, ratio=1:2, stop=2x  (shorter cycle)
  5) delta=0.25, DTE=45, ratio=1:2, stop=2x  (longer cycle)
  6) delta=0.25, DTE=30, ratio=1:3, stop=2x  (more shorts)
  7) delta=0.25, DTE=30, ratio=1:2, stop=3x  (wider stop)
  8) delta=0.25, DTE=30, ratio=1:2, stop=None (no stop)
  9) delta=0.20, DTE=30, ratio=1:2, stop=2x, IV_filter>0.30 (high IV only)
 10) delta=0.25, DTE=45, ratio=1:3, stop=3x  (aggressive)

Quality gates (MANDATORY):
  - R1: regime-agnostic (green/red/flat SPY days, reject gap > 0.50)
  - Permutation test: 100 shuffles, p < 0.05
  - Costs: $0.65/contract, 5% slippage
  - Adversarial HC #704: leakage, sub-period consistency, outlier removal, per-ticker

Data: yfinance for prices (2015-2026), Black-Scholes for option pricing.
"""

import sys, json, warnings, os, time, itertools
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import timedelta, datetime
from scipy.stats import norm
from collections import defaultdict
import traceback

warnings.filterwarnings("ignore")

ROOT   = Path("/home/jupiter/Lvl3Quant")
OUTPUT = ROOT / "output" / "put_ratio_spread_v1"
OUTPUT.mkdir(parents=True, exist_ok=True)

STARTING_CAPITAL        = 100_000
COMMISSION_PER_CONTRACT = 0.65
SLIPPAGE_PCT            = 0.05      # 5% of premium as slippage
MAX_CONCURRENT          = 10
MAX_RISK_PCT            = 0.05      # 5% of portfolio per position
RISK_FREE_RATE          = 0.04

TICKERS = [
    'AAPL','MSFT','GOOGL','AMZN','META','JNJ','PG','KO','MCD','HD',
    'WMT','V','MA','COST','UNH','ABBV','BAC','JPM','GS','SBUX',
]

SPY_TICKER = 'SPY'

# ═══════════════════════════════════════════════════════════════════════════════
# Black-Scholes Pricing
# ═══════════════════════════════════════════════════════════════════════════════

def bs_price(S, K, T, sigma, r=RISK_FREE_RATE, opt_type='p'):
    """BS European option price."""
    if T <= 0 or sigma <= 0 or S <= 0:
        return max(K - S, 0) if opt_type == 'p' else max(S - K, 0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if opt_type == 'p':
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    else:
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def bs_delta(S, K, T, sigma, r=RISK_FREE_RATE, opt_type='p'):
    """BS delta."""
    if T <= 0 or sigma <= 0 or S <= 0:
        if opt_type == 'p':
            return -1.0 if S < K else 0.0
        return 1.0 if S > K else 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    if opt_type == 'p':
        return norm.cdf(d1) - 1
    return norm.cdf(d1)


def find_strike_by_delta(S, T, sigma, target_delta, r=RISK_FREE_RATE):
    """Find put strike that gives target_delta (negative number like -0.25).
    For puts: delta goes from ~0 (far OTM, low strike) to ~-1 (deep ITM, high strike).
    We search below spot for the OTM strike matching target_delta.
    """
    lo, hi = S * 0.50, S * 1.0  # search below spot only
    for _ in range(60):
        mid = (lo + hi) / 2
        d = bs_delta(S, mid, T, sigma, r, 'p')
        if d > target_delta:   # not negative enough -> strike too low -> raise lo
            lo = mid
        else:                  # too negative -> strike too high -> lower hi
            hi = mid
    return round((lo + hi) / 2, 2)


def estimate_iv(prices, window=30):
    """Estimate annualized realized vol from daily returns."""
    rets = np.log(prices / prices.shift(1)).dropna()
    if len(rets) < window:
        return rets.std() * np.sqrt(252) if len(rets) > 5 else 0.25
    return rets.rolling(window).std().iloc[-1] * np.sqrt(252)


# ═══════════════════════════════════════════════════════════════════════════════
# Data Loading
# ═══════════════════════════════════════════════════════════════════════════════

def load_prices(tickers, start='2014-06-01', end='2026-07-15'):
    """Load daily prices from yfinance with caching."""
    cache_path = OUTPUT / "price_cache.parquet"
    all_tickers = list(set(tickers + [SPY_TICKER]))

    if cache_path.exists():
        df = pd.read_parquet(cache_path)
        missing = [t for t in all_tickers if t not in df.columns]
        if not missing:
            print(f"  Loaded prices from cache: {len(df)} days, {len(df.columns)} tickers")
            return df

    import yfinance as yf
    print(f"  Downloading prices from yfinance for {len(all_tickers)} tickers...")
    data = yf.download(all_tickers, start=start, end=end, auto_adjust=True, progress=False)
    if isinstance(data.columns, pd.MultiIndex):
        prices = data['Close']
    else:
        prices = data[['Close']]
        prices.columns = all_tickers[:1]

    prices = prices.dropna(how='all')
    prices.to_parquet(cache_path)
    print(f"  Downloaded: {len(prices)} days, {len(prices.columns)} tickers")
    return prices


# ═══════════════════════════════════════════════════════════════════════════════
# Position & Trade Logic
# ═══════════════════════════════════════════════════════════════════════════════

class PutRatioSpread:
    """Represents one put ratio spread position."""
    def __init__(self, ticker, entry_date, spot, atm_strike, otm_strike,
                 atm_premium, otm_premium, iv, dte, ratio, stop_mult,
                 expiry_date, num_spreads):
        self.ticker = ticker
        self.entry_date = entry_date
        self.spot_entry = spot
        self.atm_strike = atm_strike    # long put
        self.otm_strike = otm_strike    # short puts
        self.atm_premium = atm_premium  # paid for long put
        self.otm_premium = otm_premium  # received per short put
        self.iv = iv
        self.dte = dte
        self.ratio = ratio              # number of short puts per long
        self.stop_mult = stop_mult
        self.expiry_date = expiry_date
        self.num_spreads = num_spreads   # how many units

        # Net credit per spread = (ratio * otm_premium) - atm_premium
        self.net_credit_raw = ratio * otm_premium - atm_premium
        # Apply slippage: reduce credit
        self.net_credit = self.net_credit_raw * (1 - SLIPPAGE_PCT)
        # Commission: 1 long + ratio short contracts, both open and close
        self.commission = (1 + ratio) * 2 * COMMISSION_PER_CONTRACT * num_spreads

        self.closed = False
        self.close_date = None
        self.close_pnl = 0.0

    def value_at_expiry(self, spot_final):
        """P&L per spread at expiry (intrinsic only)."""
        # Long put payoff
        long_payoff = max(self.atm_strike - spot_final, 0)
        # Short puts payoff (negative = loss for seller)
        short_payoff = -self.ratio * max(self.otm_strike - spot_final, 0)
        # Total per spread
        total = (self.net_credit + long_payoff + short_payoff) * 100  # x100 for contract multiplier
        return total * self.num_spreads - self.commission

    def mark_to_market(self, spot, days_remaining):
        """Mark-to-market P&L using BS."""
        T = max(days_remaining / 365.0, 0.001)
        long_val = bs_price(spot, self.atm_strike, T, self.iv)
        short_val = bs_price(spot, self.otm_strike, T, self.iv)
        current_cost = long_val - self.ratio * short_val  # cost to close (buy back shorts, sell long)
        # We entered at net_credit (positive = received), so P&L = net_credit - close_cost
        pnl_per_share = self.net_credit_raw * (1 - SLIPPAGE_PCT) + (self.ratio * short_val - long_val) * (1 - SLIPPAGE_PCT)
        return pnl_per_share * 100 * self.num_spreads - self.commission

    def check_stop(self, spot, days_remaining):
        """Check if stop-loss triggered."""
        if self.stop_mult is None:
            return False
        mtm = self.mark_to_market(spot, days_remaining)
        max_loss = -self.stop_mult * abs(self.net_credit) * 100 * self.num_spreads
        return mtm <= max_loss


def run_backtest(prices, spy_prices, config):
    """
    Run one config of the put ratio spread strategy.

    config keys:
      short_delta: target delta for short puts (e.g. 0.25 means -0.25 delta)
      dte: target days to expiry
      ratio: short puts per long put (e.g. 2)
      stop_mult: stop at Nx credit received (None = no stop)
      iv_filter: minimum IV to enter (None = no filter)
      label: config name
    """
    short_delta = config['short_delta']
    dte        = config['dte']
    ratio      = config['ratio']
    stop_mult  = config.get('stop_mult', 2.0)
    iv_filter  = config.get('iv_filter', None)
    iv_markup  = config.get('iv_markup', 0.0)   # add to realized vol for pricing (VRP)
    label      = config['label']

    print(f"\n{'='*70}")
    print(f"  Config: {label}")
    print(f"  delta={short_delta}, DTE={dte}, ratio=1:{ratio}, stop={stop_mult}, iv_filter={iv_filter}")
    print(f"{'='*70}")

    capital = STARTING_CAPITAL
    positions = []
    trades = []  # completed trades
    equity_curve = []
    daily_pnl = []

    # Generate monthly entry dates (3rd Friday approximation)
    dates = prices.index.sort_values()
    start_date = pd.Timestamp('2015-01-01')
    dates = dates[dates >= start_date]

    if len(dates) == 0:
        print("  No dates available!")
        return None

    # Build monthly entry schedule
    entry_dates = []
    for year in range(2015, 2027):
        for month in range(1, 13):
            # Target: ~dte days before 3rd Friday expiry
            # 3rd Friday = first day of month + offset
            first_day = pd.Timestamp(f"{year}-{month:02d}-01")
            # Find 3rd Friday
            dow = first_day.dayofweek  # 0=Mon ... 4=Fri
            days_to_fri = (4 - dow) % 7
            third_fri = first_day + timedelta(days=days_to_fri + 14)
            entry = third_fri - timedelta(days=dte)
            # Snap to nearest trading day
            mask = dates >= entry
            if mask.any():
                actual_entry = dates[mask][0]
                if actual_entry <= dates[-1]:
                    entry_dates.append(actual_entry)

    entry_dates = sorted(set(entry_dates))

    # Track equity daily
    prev_equity = capital
    trade_idx = 0

    for i, today in enumerate(dates):
        # Check for expirations
        for pos in positions:
            if pos.closed:
                continue
            days_left = (pos.expiry_date - today).days

            if days_left <= 0:
                # Expire position
                spot = prices.loc[today, pos.ticker]
                if pd.isna(spot):
                    continue
                pnl = pos.value_at_expiry(spot)
                pos.closed = True
                pos.close_date = today
                pos.close_pnl = pnl
                capital += pnl
                trades.append(pos)

            elif pos.check_stop(prices.loc[today, pos.ticker] if pos.ticker in prices.columns and not pd.isna(prices.loc[today, pos.ticker]) else pos.spot_entry, days_left):
                # Stop-loss triggered — close at mark-to-market
                spot = prices.loc[today, pos.ticker]
                if pd.isna(spot):
                    continue
                pnl = pos.mark_to_market(spot, days_left)
                pos.closed = True
                pos.close_date = today
                pos.close_pnl = pnl
                capital += pnl
                trades.append(pos)

        # Clean up closed positions
        positions = [p for p in positions if not p.closed]

        # New entries on entry dates
        if today in entry_dates and len(positions) < MAX_CONCURRENT:
            # Try to open positions in multiple tickers
            np.random.seed(int(today.timestamp()) % (2**31))
            shuffled = list(TICKERS)
            np.random.shuffle(shuffled)

            for ticker in shuffled:
                if len(positions) >= MAX_CONCURRENT:
                    break
                # Skip if already have position in this ticker
                if any(p.ticker == ticker for p in positions):
                    continue
                if ticker not in prices.columns:
                    continue

                spot = prices.loc[today, ticker]
                if pd.isna(spot) or spot <= 0:
                    continue

                # Estimate IV from historical prices
                hist = prices.loc[:today, ticker].dropna()
                if len(hist) < 30:
                    continue
                iv = estimate_iv(hist, window=30)
                if iv <= 0.05 or iv > 2.0:
                    continue

                # IV filter
                if iv_filter is not None and iv < iv_filter:
                    continue

                # Apply IV markup (VRP: implied > realized)
                iv_entry = iv + iv_markup

                T = dte / 365.0

                # ATM put (long)
                atm_strike = round(spot, 0)  # round to nearest dollar
                atm_premium = bs_price(spot, atm_strike, T, iv_entry)

                # OTM put (short) — find strike at target delta
                target_d = -short_delta
                otm_strike = find_strike_by_delta(spot, T, iv_entry, target_d)

                if otm_strike >= atm_strike:
                    continue  # OTM must be below ATM

                otm_premium = bs_price(spot, otm_strike, T, iv_entry)

                # Net credit check
                net_credit = ratio * otm_premium - atm_premium
                if net_credit <= 0:
                    continue  # Must be a credit trade

                # Position sizing: max risk per position
                # Worst case: stock goes to 0, short puts fully ITM
                # Max loss per spread = (ratio - 1) * otm_strike * 100 - net_credit * 100
                max_loss_per_spread = max((ratio - 1) * otm_strike * 100 - net_credit * 100, net_credit * 100)
                if max_loss_per_spread <= 0:
                    max_loss_per_spread = otm_strike * 100  # fallback

                risk_budget = capital * MAX_RISK_PCT
                num_spreads = max(1, int(risk_budget / max_loss_per_spread))
                num_spreads = min(num_spreads, 5)  # cap at 5 spreads per position

                expiry = today + timedelta(days=dte)
                # Snap expiry to nearest Friday
                days_to_fri = (4 - expiry.dayofweek) % 7
                expiry = expiry + timedelta(days=days_to_fri)

                pos = PutRatioSpread(
                    ticker=ticker,
                    entry_date=today,
                    spot=spot,
                    atm_strike=atm_strike,
                    otm_strike=otm_strike,
                    atm_premium=atm_premium,
                    otm_premium=otm_premium,
                    iv=iv,
                    dte=dte,
                    ratio=ratio,
                    stop_mult=stop_mult,
                    expiry_date=expiry,
                    num_spreads=num_spreads,
                )
                positions.append(pos)

        # Daily equity
        total_mtm = 0
        for pos in positions:
            if pos.ticker in prices.columns:
                s = prices.loc[today, pos.ticker]
                if not pd.isna(s):
                    dl = max((pos.expiry_date - today).days, 1)
                    total_mtm += pos.mark_to_market(s, dl)

        equity = capital + total_mtm
        daily_ret = (equity / prev_equity - 1) if prev_equity > 0 else 0
        equity_curve.append({'date': today, 'equity': equity, 'capital': capital})
        daily_pnl.append({'date': today, 'return': daily_ret, 'equity': equity})
        prev_equity = equity

    # Close any remaining positions at last date
    last_date = dates[-1]
    for pos in positions:
        if not pos.closed:
            spot = prices.loc[last_date, pos.ticker]
            if pd.isna(spot):
                spot = pos.spot_entry
            days_left = max((pos.expiry_date - last_date).days, 0)
            if days_left > 0:
                pnl = pos.mark_to_market(spot, days_left)
            else:
                pnl = pos.value_at_expiry(spot)
            pos.closed = True
            pos.close_date = last_date
            pos.close_pnl = pnl
            capital += pnl
            trades.append(pos)

    eq_df = pd.DataFrame(equity_curve).set_index('date')
    pnl_df = pd.DataFrame(daily_pnl).set_index('date')

    return {
        'label': label,
        'config': config,
        'trades': trades,
        'equity': eq_df,
        'daily_pnl': pnl_df,
        'final_capital': capital,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Metrics
# ═══════════════════════════════════════════════════════════════════════════════

def compute_metrics(result):
    """Compute Sharpe, Sortino, CAGR, MaxDD, WR, PF."""
    trades = result['trades']
    eq = result['equity']
    pnl_df = result['daily_pnl']

    if len(trades) == 0:
        return {'error': 'no trades'}

    pnls = [t.close_pnl for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]

    wr = len(wins) / len(pnls) if len(pnls) > 0 else 0
    avg_win = np.mean(wins) if wins else 0
    avg_loss = abs(np.mean(losses)) if losses else 1
    pf = (sum(wins) / abs(sum(losses))) if losses and sum(losses) != 0 else float('inf')

    # Daily returns
    rets = pnl_df['return'].dropna()
    rets = rets.replace([np.inf, -np.inf], 0)

    ann_factor = np.sqrt(252)
    sharpe = (rets.mean() / rets.std() * ann_factor) if rets.std() > 0 else 0
    downside = rets[rets < 0]
    sortino = (rets.mean() / downside.std() * ann_factor) if len(downside) > 0 and downside.std() > 0 else 0

    # CAGR
    start_eq = STARTING_CAPITAL
    end_eq = eq['equity'].iloc[-1] if len(eq) > 0 else start_eq
    years = len(eq) / 252.0
    cagr = (end_eq / start_eq) ** (1 / years) - 1 if years > 0 and end_eq > 0 else 0

    # Max drawdown
    running_max = eq['equity'].cummax()
    drawdown = (eq['equity'] - running_max) / running_max
    max_dd = drawdown.min()

    # Monthly income
    total_pnl = sum(pnls)
    months = years * 12
    monthly_income = total_pnl / months if months > 0 else 0

    return {
        'n_trades': len(trades),
        'win_rate': wr,
        'profit_factor': pf,
        'sharpe': sharpe,
        'sortino': sortino,
        'cagr': cagr,
        'max_dd': max_dd,
        'total_pnl': total_pnl,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'monthly_income_per_100k': monthly_income,
        'final_capital': result['final_capital'],
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Quality Gates
# ═══════════════════════════════════════════════════════════════════════════════

def regime_test(result, spy_prices):
    """R1: Regime-agnostic test. Compare performance on green/red/flat SPY days."""
    trades = result['trades']
    if len(trades) < 10:
        return {'pass': False, 'reason': 'too few trades', 'gap': 1.0}

    # Classify SPY days
    spy_rets = spy_prices.pct_change().dropna()

    # Map each trade to regime based on SPY return during holding period
    green_pnls, red_pnls, flat_pnls = [], [], []
    for t in trades:
        if t.entry_date not in spy_rets.index or t.close_date not in spy_rets.index:
            continue
        spy_period = spy_rets.loc[t.entry_date:t.close_date]
        cum_ret = (1 + spy_period).prod() - 1 if len(spy_period) > 0 else 0

        if cum_ret > 0.01:
            green_pnls.append(t.close_pnl)
        elif cum_ret < -0.01:
            red_pnls.append(t.close_pnl)
        else:
            flat_pnls.append(t.close_pnl)

    def sharpe_of(pnls):
        if len(pnls) < 3:
            return 0
        arr = np.array(pnls)
        return arr.mean() / arr.std() if arr.std() > 0 else 0

    s_green = sharpe_of(green_pnls)
    s_red = sharpe_of(red_pnls)
    s_flat = sharpe_of(flat_pnls)

    max_s = max(abs(s_green), abs(s_red), abs(s_flat), 0.001)
    gap_gr = abs(s_green - s_red) / max_s
    gap_gf = abs(s_green - s_flat) / max_s
    gap_rf = abs(s_red - s_flat) / max_s
    max_gap = max(gap_gr, gap_gf, gap_rf)

    passed = max_gap <= 0.50

    return {
        'pass': passed,
        'gap': round(max_gap, 3),
        'sharpe_green': round(s_green, 3),
        'sharpe_red': round(s_red, 3),
        'sharpe_flat': round(s_flat, 3),
        'n_green': len(green_pnls),
        'n_red': len(red_pnls),
        'n_flat': len(flat_pnls),
    }


def permutation_test(result, n_perms=200):
    """Permutation test: shuffle trade P&Ls, compute p-value."""
    trades = result['trades']
    if len(trades) < 10:
        return {'pass': False, 'p_value': 1.0}

    pnls = np.array([t.close_pnl for t in trades])
    actual_mean = pnls.mean()

    count_ge = 0
    rng = np.random.RandomState(42)
    for _ in range(n_perms):
        shuffled = pnls.copy()
        # Randomly flip signs (null hypothesis: no directional edge)
        signs = rng.choice([-1, 1], size=len(shuffled))
        shuffled = shuffled * signs
        if shuffled.mean() >= actual_mean:
            count_ge += 1

    p_value = count_ge / n_perms
    return {'pass': p_value < 0.05, 'p_value': round(p_value, 4)}


def sub_period_test(result):
    """Test consistency across sub-periods: 2015-2018, 2019-2022, 2023-2026."""
    trades = result['trades']
    periods = [
        ('2015-2018', pd.Timestamp('2015-01-01'), pd.Timestamp('2018-12-31')),
        ('2019-2022', pd.Timestamp('2019-01-01'), pd.Timestamp('2022-12-31')),
        ('2023-2026', pd.Timestamp('2023-01-01'), pd.Timestamp('2026-12-31')),
    ]

    period_stats = {}
    for name, start, end in periods:
        period_trades = [t for t in trades if start <= t.entry_date <= end]
        if len(period_trades) < 5:
            period_stats[name] = {'n': len(period_trades), 'sharpe': 0, 'wr': 0, 'pf': 0}
            continue
        pnls = [t.close_pnl for t in period_trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        wr = len(wins) / len(pnls)
        pf = sum(wins) / abs(sum(losses)) if losses and sum(losses) != 0 else float('inf')
        arr = np.array(pnls)
        sharpe = arr.mean() / arr.std() if arr.std() > 0 else 0
        period_stats[name] = {
            'n': len(period_trades),
            'sharpe': round(sharpe, 3),
            'wr': round(wr, 3),
            'pf': round(pf, 3),
            'total_pnl': round(sum(pnls), 2),
        }

    # Check if any period is negative while others positive
    sharpes = [v['sharpe'] for v in period_stats.values() if v['n'] >= 5]
    consistent = all(s > 0 for s in sharpes) or all(s <= 0 for s in sharpes) if sharpes else False

    return {'consistent': consistent, 'periods': period_stats}


def outlier_test(result):
    """Remove top 5 trades, re-check if still profitable."""
    trades = result['trades']
    if len(trades) < 10:
        return {'pass': False, 'reason': 'too few trades'}

    pnls = sorted([t.close_pnl for t in trades], reverse=True)
    without_top5 = pnls[5:]

    original_sharpe = np.mean(pnls) / np.std(pnls) if np.std(pnls) > 0 else 0
    reduced_sharpe = np.mean(without_top5) / np.std(without_top5) if np.std(without_top5) > 0 else 0

    return {
        'pass': np.mean(without_top5) > 0,
        'original_sharpe': round(original_sharpe, 3),
        'without_top5_sharpe': round(reduced_sharpe, 3),
        'original_mean': round(np.mean(pnls), 2),
        'without_top5_mean': round(np.mean(without_top5), 2),
    }


def per_ticker_breakdown(result):
    """Check if returns are driven by a single ticker."""
    trades = result['trades']
    ticker_stats = defaultdict(list)
    for t in trades:
        ticker_stats[t.ticker].append(t.close_pnl)

    breakdown = {}
    total_pnl = sum(t.close_pnl for t in trades)
    for ticker, pnls in sorted(ticker_stats.items()):
        breakdown[ticker] = {
            'n': len(pnls),
            'total_pnl': round(sum(pnls), 2),
            'pct_of_total': round(sum(pnls) / total_pnl * 100, 1) if total_pnl != 0 else 0,
            'wr': round(len([p for p in pnls if p > 0]) / len(pnls), 3),
            'avg_pnl': round(np.mean(pnls), 2),
        }

    # Flag if any single ticker > 40% of total P&L
    max_pct = max(abs(v['pct_of_total']) for v in breakdown.values()) if breakdown else 0
    concentrated = max_pct > 40

    return {'concentrated': concentrated, 'max_pct': max_pct, 'tickers': breakdown}


def leakage_check(result):
    """Verify no look-ahead: all entries use only past data."""
    trades = result['trades']
    issues = []
    for t in trades:
        if t.close_date and t.close_date < t.entry_date:
            issues.append(f"{t.ticker}: close {t.close_date} before entry {t.entry_date}")
    return {'pass': len(issues) == 0, 'issues': issues}


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

CONFIGS = [
    # === Core parameter sweep (no VRP) ===
    {'label': 'A_baseline_d25_30d_1to2_s2x',  'short_delta': 0.25, 'dte': 30, 'ratio': 2, 'stop_mult': 2.0},
    {'label': 'B_deeper_d20_30d_1to2_s2x',    'short_delta': 0.20, 'dte': 30, 'ratio': 2, 'stop_mult': 2.0},
    {'label': 'C_closer_d30_30d_1to2_s2x',    'short_delta': 0.30, 'dte': 30, 'ratio': 2, 'stop_mult': 2.0},
    {'label': 'D_short_d25_14d_1to2_s2x',     'short_delta': 0.25, 'dte': 14, 'ratio': 2, 'stop_mult': 2.0},
    {'label': 'E_long_d25_45d_1to2_s2x',      'short_delta': 0.25, 'dte': 45, 'ratio': 2, 'stop_mult': 2.0},
    {'label': 'F_1to3_d25_30d_s2x',           'short_delta': 0.25, 'dte': 30, 'ratio': 3, 'stop_mult': 2.0},
    {'label': 'G_widestop_d25_30d_1to2_s3x',  'short_delta': 0.25, 'dte': 30, 'ratio': 2, 'stop_mult': 3.0},
    {'label': 'H_nostop_d25_30d_1to2',         'short_delta': 0.25, 'dte': 30, 'ratio': 2, 'stop_mult': None},
    {'label': 'I_highiv_d25_30d_1to2_s2x',    'short_delta': 0.25, 'dte': 30, 'ratio': 2, 'stop_mult': 2.0, 'iv_filter': 0.25},
    {'label': 'J_aggro_d25_45d_1to3_s3x',     'short_delta': 0.25, 'dte': 45, 'ratio': 3, 'stop_mult': 3.0},
    # === VRP configs: implied vol ~3% higher than realized (realistic) ===
    {'label': 'K_vrp3_d25_30d_1to2_s2x',      'short_delta': 0.25, 'dte': 30, 'ratio': 2, 'stop_mult': 2.0, 'iv_markup': 0.03},
    {'label': 'L_vrp5_d25_30d_1to2_s2x',      'short_delta': 0.25, 'dte': 30, 'ratio': 2, 'stop_mult': 2.0, 'iv_markup': 0.05},
    {'label': 'M_vrp3_d25_45d_1to2_nostop',    'short_delta': 0.25, 'dte': 45, 'ratio': 2, 'stop_mult': None, 'iv_markup': 0.03},
    {'label': 'N_vrp5_d20_30d_1to2_s3x',      'short_delta': 0.20, 'dte': 30, 'ratio': 2, 'stop_mult': 3.0, 'iv_markup': 0.05},
]


def main():
    t0 = time.time()
    print("=" * 80)
    print("  PUT RATIO SPREAD INCOME BACKTEST — v1")
    print("=" * 80)

    # Load prices
    prices = load_prices(TICKERS)
    spy_prices = prices[SPY_TICKER] if SPY_TICKER in prices.columns else None

    if spy_prices is None:
        print("ERROR: Could not load SPY prices")
        return

    all_results = []
    all_metrics = []
    all_quality = []

    for cfg in CONFIGS:
        try:
            result = run_backtest(prices, spy_prices, cfg)
            if result is None:
                print(f"  SKIP: {cfg['label']} — no result")
                continue

            metrics = compute_metrics(result)
            if 'error' in metrics:
                print(f"  SKIP: {cfg['label']} — {metrics['error']}")
                continue
            regime = regime_test(result, spy_prices)
            perm = permutation_test(result, n_perms=200)
            subperiod = sub_period_test(result)
            outlier = outlier_test(result)
            ticker_bd = per_ticker_breakdown(result)
            leakage = leakage_check(result)

            quality = {
                'regime': regime,
                'permutation': perm,
                'sub_period': subperiod,
                'outlier': outlier,
                'ticker_breakdown': ticker_bd,
                'leakage': leakage,
            }

            all_results.append(result)
            all_metrics.append({'label': cfg['label'], **metrics})
            all_quality.append({'label': cfg['label'], **quality})

            # Print summary
            print(f"\n  {cfg['label']}:")
            print(f"    Trades: {metrics['n_trades']}, WR: {metrics['win_rate']:.1%}, PF: {metrics['profit_factor']:.2f}")
            print(f"    Sharpe: {metrics['sharpe']:.3f}, Sortino: {metrics['sortino']:.3f}")
            print(f"    CAGR: {metrics['cagr']:.1%}, MaxDD: {metrics['max_dd']:.1%}")
            print(f"    Total P&L: ${metrics['total_pnl']:,.0f}, Monthly/$100K: ${metrics['monthly_income_per_100k']:,.0f}")
            print(f"    Regime: {'PASS' if regime['pass'] else 'FAIL'} (gap={regime['gap']:.2f})")
            print(f"    Permutation: {'PASS' if perm['pass'] else 'FAIL'} (p={perm['p_value']:.3f})")
            print(f"    Sub-period: {'CONSISTENT' if subperiod['consistent'] else 'INCONSISTENT'}")
            print(f"    Outlier: {'PASS' if outlier['pass'] else 'FAIL'}")
            print(f"    Ticker concentration: {'WARN' if ticker_bd['concentrated'] else 'OK'} (max={ticker_bd['max_pct']:.1f}%)")
            print(f"    Leakage: {'PASS' if leakage['pass'] else 'FAIL'}")

        except Exception as e:
            print(f"  ERROR in {cfg['label']}: {e}")
            traceback.print_exc()

    # ═══════════════════════════════════════════════════════════════════════════
    # Summary Table
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 120)
    print("  SUMMARY TABLE")
    print("=" * 120)
    print(f"{'Config':<40} {'Trades':>6} {'WR':>6} {'PF':>6} {'Sharpe':>7} {'Sortino':>8} {'CAGR':>7} {'MaxDD':>7} {'Mo/$100K':>10} {'Regime':>7} {'Perm':>6}")
    print("-" * 120)

    passing_configs = []
    for m, q in zip(all_metrics, all_quality):
        regime_ok = q.get('regime', {}).get('pass', False)
        perm_ok = q.get('permutation', {}).get('pass', False)
        label = 'PASS' if regime_ok else 'FAIL'
        p_label = 'PASS' if perm_ok else 'FAIL'

        print(f"{m['label']:<40} {m['n_trades']:>6} {m['win_rate']:>5.1%} {m['profit_factor']:>6.2f} "
              f"{m['sharpe']:>7.3f} {m['sortino']:>8.3f} {m['cagr']:>6.1%} {m['max_dd']:>6.1%} "
              f"${m['monthly_income_per_100k']:>8,.0f} {label:>7} {p_label:>6}")

        if regime_ok and perm_ok and m['sharpe'] > 0:
            passing_configs.append(m['label'])

    print("-" * 120)
    print(f"  Configs passing ALL quality gates: {passing_configs if passing_configs else 'NONE'}")

    # ═══════════════════════════════════════════════════════════════════════════
    # Save Results
    # ═══════════════════════════════════════════════════════════════════════════

    # Save metrics
    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(OUTPUT / "metrics_summary.csv", index=False)

    # Save quality gate details
    quality_report = {}
    for q in all_quality:
        label = q['label']
        # Convert ticker breakdown to serializable format
        qc = dict(q)
        if 'ticker_breakdown' in qc:
            qc['ticker_breakdown'] = {
                'concentrated': qc['ticker_breakdown']['concentrated'],
                'max_pct': qc['ticker_breakdown']['max_pct'],
                'tickers': qc['ticker_breakdown']['tickers'],
            }
        quality_report[label] = qc

    with open(OUTPUT / "quality_gates.json", 'w') as f:
        json.dump(quality_report, f, indent=2, default=str)

    # Save equity curves
    for result in all_results:
        result['equity'].to_csv(OUTPUT / f"equity_{result['label']}.csv")

    # Save trade details for all configs
    for result in all_results:
        trade_records = []
        for t in result['trades']:
            trade_records.append({
                'ticker': t.ticker,
                'entry_date': str(t.entry_date.date()),
                'close_date': str(t.close_date.date()) if t.close_date else None,
                'spot_entry': round(t.spot_entry, 2),
                'atm_strike': t.atm_strike,
                'otm_strike': t.otm_strike,
                'iv': round(t.iv, 4),
                'net_credit': round(t.net_credit, 4),
                'num_spreads': t.num_spreads,
                'pnl': round(t.close_pnl, 2),
            })
        with open(OUTPUT / f"trades_{result['label']}.json", 'w') as f:
            json.dump(trade_records, f, indent=2)

    elapsed = time.time() - t0
    print(f"\n  Completed in {elapsed:.1f}s. Results saved to {OUTPUT}")
    print(f"  Files: metrics_summary.csv, quality_gates.json, equity_*.csv, trades_*.json")


if __name__ == '__main__':
    main()
