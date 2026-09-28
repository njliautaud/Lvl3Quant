#!/usr/bin/env python3
"""
Relative Strength Regime Switch Strategy v1
=============================================

Behavioral bias: When market regime changes (bull->bear or bear->bull),
the FIRST stocks to break out/break down tend to LEAD the next regime.
This is "leadership rotation" — institutional money flows to new leaders first.

Detection: SPY 50-day vs 200-day MA crossover (golden cross / death cross)
Signal: At crossover, rank stocks by 20-day relative strength vs SPY
Trade: Buy top-5 RS stocks (regime leaders), hold 20-40 trading days

VARIANTS:
A — Golden Cross Leaders (buy top-5 RS at golden cross, hold 30d, equity)
B — Death Cross Leaders (short top-5 RS at death cross, hold 20d, equity)
C — Both Directions (long on golden, short on death, equity)
D — Golden Cross + Cheap Calls ($1-3 premium, hold 30d)
E — Momentum Confirmation (require 5d follow-through after cross)
F — Sector-Diversified (max 2 per sector from top-5)

Walk-forward: sliding OOT, Jan 2022 - Jul 2026
Capital: $645
Cost: $0.65/contract options, $0 equity (Robinhood)
"""

import json
import logging
import math
import os
import sys
import time
import warnings
from datetime import datetime, timedelta
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.stats import norm

warnings.filterwarnings('ignore')

# ── Paths ──
for root in ['/home/jupiter/Lvl3Quant', '/home/nick/Lvl3Quant']:
    if os.path.isdir(root):
        LVL3_ROOT = root
        break
else:
    LVL3_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

sys.path.insert(0, LVL3_ROOT)

LOG_DIR = os.path.join(LVL3_ROOT, 'scripts', 'growth_research', 'logs')
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, 'relative_strength_regime_v1.log')),
        logging.StreamHandler(),
    ]
)
log = logging.getLogger(__name__)

# ==================== CONFIG ====================

STOCK_UNIVERSE = [
    # Large-cap growth
    'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'NVDA', 'TSLA', 'AMD',
    'CRM', 'NFLX', 'PYPL', 'SQ', 'SHOP', 'UBER', 'NET', 'CRWD',
    'DDOG', 'ZS', 'MDB', 'ROKU',
    # Large-cap value/cyclical
    'JPM', 'BAC', 'GS', 'XOM', 'CVX', 'CAT', 'DE', 'UNH',
    'LLY', 'JNJ', 'PG', 'KO', 'WMT', 'HD', 'LOW',
    # Mid-cap growth
    'SNAP', 'PINS', 'TTD', 'BILL', 'HUBS', 'VEEV', 'PANW', 'MNDY',
    'SMCI', 'MELI', 'SE', 'COIN', 'PLTR', 'SNOW', 'ABNB',
]

SECTOR_MAP = {
    'AAPL': 'Tech', 'MSFT': 'Tech', 'AMZN': 'Consumer', 'GOOGL': 'Comms',
    'META': 'Comms', 'NVDA': 'Tech', 'TSLA': 'Consumer', 'AMD': 'Tech',
    'CRM': 'Tech', 'NFLX': 'Comms', 'PYPL': 'Fintech', 'SQ': 'Fintech',
    'SHOP': 'Tech', 'UBER': 'Tech', 'NET': 'Tech', 'CRWD': 'Tech',
    'DDOG': 'Tech', 'ZS': 'Tech', 'MDB': 'Tech', 'ROKU': 'Comms',
    'JPM': 'Finance', 'BAC': 'Finance', 'GS': 'Finance', 'XOM': 'Energy',
    'CVX': 'Energy', 'CAT': 'Industrial', 'DE': 'Industrial', 'UNH': 'Health',
    'LLY': 'Health', 'JNJ': 'Health', 'PG': 'Staples', 'KO': 'Staples',
    'WMT': 'Staples', 'HD': 'Consumer', 'LOW': 'Consumer',
    'SNAP': 'Comms', 'PINS': 'Comms', 'TTD': 'Tech', 'BILL': 'Fintech',
    'HUBS': 'Tech', 'VEEV': 'Health', 'PANW': 'Tech', 'MNDY': 'Tech',
    'SMCI': 'Tech', 'MELI': 'Consumer', 'SE': 'Consumer', 'COIN': 'Fintech',
    'PLTR': 'Tech', 'SNOW': 'Tech', 'ABNB': 'Consumer',
}

INITIAL_CAPITAL = 645.0
MAX_POS_PCT = 0.18  # ~18% per position, 5 positions = ~90% deployed
COMMISSION_OPTION = 0.65
COMMISSION_EQUITY = 0.0
RISK_FREE_RATE = 0.05
BS_HAIRCUT = 0.85

OOT_START = '2022-01-01'
OOT_END = '2026-07-28'

# ==================== BLACK-SCHOLES ====================

def bs_call(S, K, T, r, sigma):
    if T <= 1e-8: return max(S - K, 0.0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

def bs_put(S, K, T, r, sigma):
    if T <= 1e-8: return max(K - S, 0.0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


# ==================== DATA LOADING ====================

def load_all_data():
    """Load price data for universe + SPY."""
    import yfinance as yf

    cache_dir = os.path.join(LVL3_ROOT, 'data')
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, 'rs_regime_v1_cache.parquet')

    all_tickers = list(set(STOCK_UNIVERSE + ['SPY']))

    if os.path.exists(cache_file):
        age_hours = (time.time() - os.path.getmtime(cache_file)) / 3600
        if age_hours < 24:
            log.info("Loading cached data")
            return pd.read_parquet(cache_file)

    log.info(f"Downloading data for {len(all_tickers)} tickers...")
    data_frames = {}
    for ticker in all_tickers:
        try:
            t = yf.Ticker(ticker)
            hist = t.history(start='2020-01-01', end=OOT_END, auto_adjust=True)
            if len(hist) > 50:
                hist.index = hist.index.tz_localize(None)
                data_frames[ticker] = hist[['Close', 'Volume', 'High', 'Low']].copy()
                log.info(f"  {ticker}: {len(hist)} days")
        except Exception as e:
            log.warning(f"  {ticker} failed: {e}")
        time.sleep(0.3)

    if not data_frames:
        raise RuntimeError("No data downloaded")

    combined = pd.concat(data_frames, axis=1)
    combined.to_parquet(cache_file)
    log.info(f"Cached {len(data_frames)} tickers")
    return combined


# ==================== REGIME DETECTION ====================

def detect_regime_crossovers(spy_close):
    """
    Detect golden cross (50 > 200) and death cross (50 < 200) events.
    Returns list of (date, type) tuples.
    """
    ma50 = spy_close.rolling(50).mean()
    ma200 = spy_close.rolling(200).mean()

    crossovers = []
    prev_state = None

    for i in range(200, len(spy_close)):
        date = spy_close.index[i]
        if date < pd.Timestamp(OOT_START) or date > pd.Timestamp(OOT_END):
            continue

        if pd.isna(ma50.iloc[i]) or pd.isna(ma200.iloc[i]):
            continue

        current_state = 'bull' if ma50.iloc[i] > ma200.iloc[i] else 'bear'

        if prev_state is not None and current_state != prev_state:
            cross_type = 'golden_cross' if current_state == 'bull' else 'death_cross'
            crossovers.append((date, cross_type))
            log.info(f"  Regime change: {cross_type} on {date.date()} "
                     f"(MA50={ma50.iloc[i]:.1f} vs MA200={ma200.iloc[i]:.1f})")

        prev_state = current_state

    return crossovers


def rank_relative_strength(prices_df, date, spy_close, lookback=20):
    """
    Rank stocks by relative strength vs SPY over lookback days.
    Returns sorted list of (ticker, rs_score) tuples.
    """
    spy_idx = spy_close.index.get_indexer([date], method='ffill')[0]
    if spy_idx < lookback:
        return []

    spy_ret = (spy_close.iloc[spy_idx] / spy_close.iloc[spy_idx - lookback]) - 1.0

    rs_scores = []
    for ticker in STOCK_UNIVERSE:
        col = (ticker, 'Close')
        if col not in prices_df.columns:
            continue
        series = prices_df[col].dropna()
        try:
            idx = series.index.get_indexer([date], method='ffill')[0]
            if idx < lookback or idx < 0:
                continue
            stock_ret = (series.iloc[idx] / series.iloc[idx - lookback]) - 1.0
            rs = stock_ret - spy_ret  # relative strength
            rs_scores.append((ticker, rs, series.iloc[idx]))
        except:
            continue

    rs_scores.sort(key=lambda x: x[1], reverse=True)
    return rs_scores


# ==================== STRATEGY VARIANTS ====================

def run_variant_A(prices_df, spy_close, crossovers):
    """Golden Cross Leaders: buy top-5 RS at golden cross, hold 30d."""
    trades = []
    for date, cross_type in crossovers:
        if cross_type != 'golden_cross':
            continue

        rs = rank_relative_strength(prices_df, date, spy_close)
        top5 = rs[:5]

        for ticker, rs_score, entry_price in top5:
            col = (ticker, 'Close')
            series = prices_df[col].dropna()
            idx = series.index.get_indexer([date], method='ffill')[0]
            if idx + 30 >= len(series):
                continue

            exit_price = series.iloc[idx + 30]
            exit_date = series.index[idx + 30]

            n_shares = max(1, int((INITIAL_CAPITAL * MAX_POS_PCT) / entry_price))
            cost = n_shares * entry_price
            if cost > INITIAL_CAPITAL:
                n_shares = max(1, int(INITIAL_CAPITAL / entry_price))
                cost = n_shares * entry_price

            pnl = n_shares * (exit_price - entry_price)
            trades.append({
                'ticker': ticker, 'entry_date': date, 'exit_date': exit_date,
                'entry_price': entry_price, 'exit_price': exit_price,
                'pnl': pnl, 'cost': cost, 'ret': pnl / cost if cost > 0 else 0,
                'rs_score': rs_score, 'cross_type': cross_type,
                'direction': 'long', 'type': 'equity', 'variant': 'A',
            })

    return trades


def run_variant_B(prices_df, spy_close, crossovers):
    """Death Cross Leaders: short top-5 RS at death cross, hold 20d."""
    trades = []
    for date, cross_type in crossovers:
        if cross_type != 'death_cross':
            continue

        rs = rank_relative_strength(prices_df, date, spy_close)
        # Short the weakest (bottom 5)
        bottom5 = rs[-5:]

        for ticker, rs_score, entry_price in bottom5:
            if entry_price < 5:
                continue

            col = (ticker, 'Close')
            series = prices_df[col].dropna()
            idx = series.index.get_indexer([date], method='ffill')[0]
            if idx + 20 >= len(series):
                continue

            exit_price = series.iloc[idx + 20]
            exit_date = series.index[idx + 20]

            n_shares = max(1, int((INITIAL_CAPITAL * MAX_POS_PCT) / entry_price))
            cost = n_shares * entry_price
            if cost > INITIAL_CAPITAL:
                n_shares = max(1, int(INITIAL_CAPITAL / entry_price))
                cost = n_shares * entry_price

            pnl = n_shares * (entry_price - exit_price)  # short
            trades.append({
                'ticker': ticker, 'entry_date': date, 'exit_date': exit_date,
                'entry_price': entry_price, 'exit_price': exit_price,
                'pnl': pnl, 'cost': cost, 'ret': pnl / cost if cost > 0 else 0,
                'rs_score': rs_score, 'cross_type': cross_type,
                'direction': 'short', 'type': 'equity', 'variant': 'B',
            })

    return trades


def run_variant_C(prices_df, spy_close, crossovers):
    """Both directions: long on golden, short on death."""
    long_trades = run_variant_A(prices_df, spy_close, crossovers)
    short_trades = run_variant_B(prices_df, spy_close, crossovers)
    for t in long_trades:
        t['variant'] = 'C'
    for t in short_trades:
        t['variant'] = 'C'
    return long_trades + short_trades


def run_variant_D(prices_df, spy_close, crossovers):
    """Golden Cross + Cheap Calls ($1-3 premium, hold 30d)."""
    trades = []
    for date, cross_type in crossovers:
        if cross_type != 'golden_cross':
            continue

        rs = rank_relative_strength(prices_df, date, spy_close)
        top5 = rs[:5]

        for ticker, rs_score, spot_price in top5:
            col = (ticker, 'Close')
            series = prices_df[col].dropna()
            idx = series.index.get_indexer([date], method='ffill')[0]
            if idx + 30 >= len(series) or idx < 21:
                continue

            # Calculate IV proxy
            vol = series.pct_change().iloc[max(0, idx - 21):idx].std() * np.sqrt(252)
            if pd.isna(vol) or vol < 0.15:
                vol = 0.35

            # Find OTM strike that gives $1-3 call price
            # Try strikes 5-15% OTM
            T = 45 / 252  # ~45 DTE
            best_strike = None
            best_call_price = None
            for pct_otm in np.arange(0.03, 0.25, 0.02):
                K = round(spot_price * (1 + pct_otm))
                cp = bs_call(spot_price, K, T, RISK_FREE_RATE, vol) * BS_HAIRCUT
                if 1.0 <= cp <= 3.0:
                    best_strike = K
                    best_call_price = cp
                    break

            if best_strike is None:
                # Try ATM if nothing in range
                best_strike = round(spot_price)
                best_call_price = bs_call(spot_price, best_strike, T, RISK_FREE_RATE, vol) * BS_HAIRCUT
                if best_call_price > 5.0 or best_call_price < 0.20:
                    continue

            # Exit price after 30 days
            exit_spot = series.iloc[idx + 30]
            exit_date = series.index[idx + 30]
            T_exit = max(1 / 252, T - 30 / 252)
            exit_call_price = bs_call(exit_spot, best_strike, T_exit, RISK_FREE_RATE, vol) * BS_HAIRCUT

            # Position sizing: buy contracts
            contract_cost = best_call_price * 100
            n_contracts = max(1, int((INITIAL_CAPITAL * MAX_POS_PCT) / contract_cost))
            total_cost = n_contracts * contract_cost + COMMISSION_OPTION * n_contracts * 2
            if total_cost > INITIAL_CAPITAL:
                n_contracts = max(1, int((INITIAL_CAPITAL - COMMISSION_OPTION * 2) / contract_cost))
                total_cost = n_contracts * contract_cost + COMMISSION_OPTION * n_contracts * 2

            proceeds = n_contracts * exit_call_price * 100
            pnl = proceeds - total_cost

            trades.append({
                'ticker': ticker, 'entry_date': date, 'exit_date': exit_date,
                'entry_price': best_call_price, 'exit_price': exit_call_price,
                'spot_entry': spot_price, 'spot_exit': exit_spot,
                'strike': best_strike, 'n_contracts': n_contracts,
                'pnl': pnl, 'cost': total_cost, 'ret': pnl / total_cost if total_cost > 0 else 0,
                'rs_score': rs_score, 'cross_type': cross_type,
                'direction': 'long', 'type': 'call_option', 'variant': 'D',
            })

    return trades


def run_variant_E(prices_df, spy_close, crossovers):
    """Momentum Confirmation: require 5-day follow-through after cross."""
    trades = []
    for date, cross_type in crossovers:
        # Wait 5 days and confirm the cross held
        spy_idx = spy_close.index.get_indexer([date], method='ffill')[0]
        if spy_idx + 5 >= len(spy_close):
            continue

        confirm_date = spy_close.index[spy_idx + 5]
        ma50_5d = spy_close.iloc[spy_idx + 5 - 49:spy_idx + 6].mean()
        ma200_5d = spy_close.iloc[spy_idx + 5 - 199:spy_idx + 6].mean()

        # Check cross held
        if cross_type == 'golden_cross' and ma50_5d <= ma200_5d:
            log.info(f"  Cross failed confirmation at {confirm_date.date()}")
            continue
        if cross_type == 'death_cross' and ma50_5d >= ma200_5d:
            log.info(f"  Cross failed confirmation at {confirm_date.date()}")
            continue

        # Also require SPY moved in expected direction
        spy_5d_ret = (spy_close.iloc[spy_idx + 5] / spy_close.iloc[spy_idx]) - 1.0
        if cross_type == 'golden_cross' and spy_5d_ret < 0:
            continue
        if cross_type == 'death_cross' and spy_5d_ret > 0:
            continue

        # Now rank RS and trade
        rs = rank_relative_strength(prices_df, confirm_date, spy_close)

        if cross_type == 'golden_cross':
            selected = rs[:5]
            direction = 'long'
            hold_days = 30
        else:
            selected = rs[-5:]
            direction = 'short'
            hold_days = 20

        for ticker, rs_score, entry_price in selected:
            if direction == 'short' and entry_price < 5:
                continue

            col = (ticker, 'Close')
            series = prices_df[col].dropna()
            idx = series.index.get_indexer([confirm_date], method='ffill')[0]
            if idx + hold_days >= len(series):
                continue

            exit_price = series.iloc[idx + hold_days]
            exit_date = series.index[idx + hold_days]

            n_shares = max(1, int((INITIAL_CAPITAL * MAX_POS_PCT) / entry_price))
            cost = n_shares * entry_price
            if cost > INITIAL_CAPITAL:
                n_shares = max(1, int(INITIAL_CAPITAL / entry_price))
                cost = n_shares * entry_price

            if direction == 'long':
                pnl = n_shares * (exit_price - entry_price)
            else:
                pnl = n_shares * (entry_price - exit_price)

            trades.append({
                'ticker': ticker, 'entry_date': confirm_date, 'exit_date': exit_date,
                'entry_price': entry_price, 'exit_price': exit_price,
                'pnl': pnl, 'cost': cost, 'ret': pnl / cost if cost > 0 else 0,
                'rs_score': rs_score, 'cross_type': cross_type,
                'direction': direction, 'type': 'equity', 'variant': 'E',
            })

    return trades


def run_variant_F(prices_df, spy_close, crossovers):
    """Sector-Diversified: max 2 per sector from top RS."""
    trades = []
    for date, cross_type in crossovers:
        if cross_type != 'golden_cross':
            continue

        rs = rank_relative_strength(prices_df, date, spy_close)

        # Select top RS but with sector diversification
        selected = []
        sector_count = defaultdict(int)
        for ticker, rs_score, price in rs:
            sector = SECTOR_MAP.get(ticker, 'Other')
            if sector_count[sector] >= 2:
                continue
            selected.append((ticker, rs_score, price))
            sector_count[sector] += 1
            if len(selected) >= 5:
                break

        for ticker, rs_score, entry_price in selected:
            col = (ticker, 'Close')
            series = prices_df[col].dropna()
            idx = series.index.get_indexer([date], method='ffill')[0]
            if idx + 30 >= len(series):
                continue

            exit_price = series.iloc[idx + 30]
            exit_date = series.index[idx + 30]

            n_shares = max(1, int((INITIAL_CAPITAL * MAX_POS_PCT) / entry_price))
            cost = n_shares * entry_price
            if cost > INITIAL_CAPITAL:
                n_shares = max(1, int(INITIAL_CAPITAL / entry_price))
                cost = n_shares * entry_price

            pnl = n_shares * (exit_price - entry_price)
            trades.append({
                'ticker': ticker, 'entry_date': date, 'exit_date': exit_date,
                'entry_price': entry_price, 'exit_price': exit_price,
                'pnl': pnl, 'cost': cost, 'ret': pnl / cost if cost > 0 else 0,
                'rs_score': rs_score, 'cross_type': cross_type,
                'direction': 'long', 'type': 'equity', 'variant': 'F',
            })

    return trades


# ==================== WALK-FORWARD BACKTEST ====================

def backtest_variant(trades, variant_name):
    """Run walk-forward backtest with capital constraints."""
    if not trades:
        return None

    trades_df = pd.DataFrame(trades)
    trades_df['entry_date'] = pd.to_datetime(trades_df['entry_date'])
    trades_df['exit_date'] = pd.to_datetime(trades_df['exit_date'])
    trades_df = trades_df.sort_values('entry_date').reset_index(drop=True)

    capital = INITIAL_CAPITAL
    peak_capital = capital
    max_dd = 0
    realized_trades = []
    active_positions = []

    all_dates = pd.bdate_range(OOT_START, OOT_END)
    equity_curve = pd.Series(index=all_dates, dtype=float)
    equity_curve.iloc[0] = capital

    trade_idx = 0
    for i, date in enumerate(all_dates):
        # Close expired positions
        new_active = []
        for pos in active_positions:
            if date >= pos['exit_date']:
                capital += pos['pnl'] + pos['cost']
                realized_trades.append(pos)
            else:
                new_active.append(pos)
        active_positions = new_active

        # Open new positions
        while trade_idx < len(trades_df) and trades_df.iloc[trade_idx]['entry_date'] <= date:
            t = trades_df.iloc[trade_idx].to_dict()
            trade_idx += 1
            if len(active_positions) >= 5:
                continue
            if t['cost'] > capital:
                continue
            capital -= t['cost']
            active_positions.append(t)

        # Mark-to-market (linear interpolation)
        total_value = capital + sum(
            p['cost'] + p['pnl'] * min(1.0, max(0.0,
                (date - p['entry_date']).days / max(1, (p['exit_date'] - p['entry_date']).days)))
            for p in active_positions)
        equity_curve.iloc[i] = total_value
        peak_capital = max(peak_capital, total_value)
        dd = (total_value - peak_capital) / peak_capital if peak_capital > 0 else 0
        max_dd = min(max_dd, dd)

    equity_curve = equity_curve.dropna().ffill()
    if len(equity_curve) < 20:
        return None

    daily_rets = equity_curve.pct_change().dropna()
    daily_rets = daily_rets.replace([np.inf, -np.inf], 0)

    n_trades = len(realized_trades)
    if n_trades == 0:
        return None

    wins = sum(1 for t in realized_trades if t['pnl'] > 0)
    win_rate = wins / n_trades
    total_pnl = sum(t['pnl'] for t in realized_trades)
    gross_profit = sum(t['pnl'] for t in realized_trades if t['pnl'] > 0)
    gross_loss = abs(sum(t['pnl'] for t in realized_trades if t['pnl'] < 0))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    ann_ret = daily_rets.mean() * 252
    ann_vol = daily_rets.std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0

    downside = daily_rets[daily_rets < 0].std() * np.sqrt(252)
    sortino = ann_ret / downside if downside > 0 else 0

    return {
        'variant': variant_name,
        'n_trades': n_trades,
        'win_rate': win_rate,
        'total_pnl': total_pnl,
        'final_capital': equity_curve.iloc[-1],
        'sharpe': sharpe,
        'sortino': sortino,
        'profit_factor': profit_factor,
        'max_dd': max_dd,
        'ann_return': ann_ret,
        'ann_vol': ann_vol,
        'trades': realized_trades,
        'equity_curve': equity_curve,
        'daily_returns': daily_rets,
    }


# ==================== VALIDATION ====================

def five_gate_validation(result):
    """5-gate validation."""
    if result is None:
        return {'pass': False, 'reason': 'No trades'}

    gates = {}

    # Gate 1: Sharpe > 0.5
    gates['sharpe_gt_0.5'] = result['sharpe'] > 0.5

    # Gate 2: Permutation test
    daily_rets = result['daily_returns']
    if len(daily_rets) > 30:
        observed_mean = daily_rets.mean()
        n_perms = 5000
        rets_arr = daily_rets.values.copy()
        perm_means = []
        for _ in range(n_perms):
            signs = np.random.choice([-1, 1], size=len(rets_arr))
            perm_means.append((rets_arr * signs).mean())
        perm_p = np.mean(np.array(perm_means) >= observed_mean)
        gates['perm_p_lt_0.05'] = perm_p < 0.05
        result['perm_p'] = perm_p
    else:
        gates['perm_p_lt_0.05'] = False
        result['perm_p'] = 1.0

    # Gate 3: Beats random
    if result['n_trades'] >= 5:
        random_sharpes = []
        for _ in range(1000):
            random_rets = np.random.choice(daily_rets.values, size=len(daily_rets), replace=True)
            np.random.shuffle(random_rets)
            r_mean = random_rets.mean() * 252
            r_vol = random_rets.std() * np.sqrt(252)
            random_sharpes.append(r_mean / r_vol if r_vol > 0 else 0)
        pct_beaten = np.mean(result['sharpe'] > np.array(random_sharpes))
        gates['beats_random'] = pct_beaten > 0.95
        result['pct_beats_random'] = pct_beaten
    else:
        gates['beats_random'] = False
        result['pct_beats_random'] = 0

    # Gate 4: Regime gap
    daily_rets_arr = daily_rets.values
    mid = len(daily_rets_arr) // 2
    first_half_sharpe = daily_rets_arr[:mid].mean() / (daily_rets_arr[:mid].std() + 1e-10) * np.sqrt(252)
    second_half_sharpe = daily_rets_arr[mid:].mean() / (daily_rets_arr[mid:].std() + 1e-10) * np.sqrt(252)
    max_abs = max(abs(first_half_sharpe), abs(second_half_sharpe), 0.01)
    regime_gap = abs(first_half_sharpe - second_half_sharpe) / max_abs
    gates['regime_gap_lt_0.5'] = regime_gap < 0.50
    result['regime_gap'] = regime_gap

    # Gate 5: MDD > -50%
    gates['mdd_gt_neg50'] = result['max_dd'] > -0.50

    result['gates'] = gates
    result['gates_passed'] = sum(gates.values())
    result['all_gates_pass'] = all(gates.values())

    return result


# ==================== MLFLOW ====================

def log_to_mlflow(results):
    try:
        import mlflow
        mlflow.set_tracking_uri('http://localhost:5000')
        mlflow.set_experiment('growth_research_rs_regime_v1')

        for res in results:
            if res is None:
                continue
            with mlflow.start_run(run_name=f"rs_regime_{res['variant']}"):
                mlflow.log_param('strategy', 'relative_strength_regime_v1')
                mlflow.log_param('variant', res['variant'])
                mlflow.log_param('n_trades', res['n_trades'])
                mlflow.log_param('initial_capital', INITIAL_CAPITAL)

                mlflow.log_metric('sharpe', res['sharpe'])
                mlflow.log_metric('sortino', res['sortino'])
                mlflow.log_metric('profit_factor', min(res['profit_factor'], 99))
                mlflow.log_metric('win_rate', res['win_rate'])
                mlflow.log_metric('total_pnl', res['total_pnl'])
                mlflow.log_metric('final_capital', res['final_capital'])
                mlflow.log_metric('max_dd', res['max_dd'])
                mlflow.log_metric('ann_return', res['ann_return'])
                mlflow.log_metric('perm_p', res.get('perm_p', 1.0))
                mlflow.log_metric('regime_gap', res.get('regime_gap', 1.0))
                mlflow.log_metric('gates_passed', res.get('gates_passed', 0))
                mlflow.set_tag('all_gates_pass', str(res.get('all_gates_pass', False)))

        log.info(f"Logged {len([r for r in results if r])} variants to MLflow")
    except Exception as e:
        log.warning(f"MLflow logging failed: {e}")


# ==================== MAIN ====================

def main():
    log.info("=" * 70)
    log.info("Relative Strength Regime Switch v1 — Starting backtest")
    log.info("=" * 70)

    prices_df = load_all_data()
    log.info(f"Data loaded: {prices_df.shape}")

    # Extract SPY close
    spy_close = prices_df[('SPY', 'Close')].dropna()
    log.info(f"SPY data: {len(spy_close)} days, {spy_close.index[0].date()} to {spy_close.index[-1].date()}")

    # Detect regime crossovers
    log.info("\n--- Regime Crossover Detection ---")
    crossovers = detect_regime_crossovers(spy_close)
    log.info(f"Found {len(crossovers)} crossovers in OOT period")

    if not crossovers:
        log.warning("No crossovers found! Strategy cannot trade.")
        # Use near-crosses as fallback: when MA50/MA200 gap < 1%
        log.info("Checking for near-crosses (MA gap < 2%)...")
        ma50 = spy_close.rolling(50).mean()
        ma200 = spy_close.rolling(200).mean()
        for i in range(200, len(spy_close)):
            date = spy_close.index[i]
            if date < pd.Timestamp(OOT_START) or date > pd.Timestamp(OOT_END):
                continue
            if pd.isna(ma50.iloc[i]) or pd.isna(ma200.iloc[i]):
                continue
            gap = abs(ma50.iloc[i] - ma200.iloc[i]) / ma200.iloc[i]
            if gap < 0.02:
                cross_type = 'golden_cross' if ma50.iloc[i] > ma200.iloc[i] else 'death_cross'
                # Only add one per month
                if not crossovers or (date - crossovers[-1][0]).days > 30:
                    crossovers.append((date, cross_type))
                    log.info(f"  Near-cross: {cross_type} on {date.date()} (gap={gap:.3%})")

    results = []

    # Variant A
    log.info("\n=== Variant A: Golden Cross Leaders (top-5 RS, hold 30d, equity) ===")
    trades_a = run_variant_A(prices_df, spy_close, crossovers)
    log.info(f"  Generated {len(trades_a)} trades")
    res_a = backtest_variant(trades_a, 'A')
    if res_a:
        res_a = five_gate_validation(res_a)
        results.append(res_a)

    # Variant B
    log.info("\n=== Variant B: Death Cross Leaders (short bottom-5, hold 20d) ===")
    trades_b = run_variant_B(prices_df, spy_close, crossovers)
    log.info(f"  Generated {len(trades_b)} trades")
    res_b = backtest_variant(trades_b, 'B')
    if res_b:
        res_b = five_gate_validation(res_b)
        results.append(res_b)

    # Variant C
    log.info("\n=== Variant C: Both Directions ===")
    trades_c = run_variant_C(prices_df, spy_close, crossovers)
    log.info(f"  Generated {len(trades_c)} trades")
    res_c = backtest_variant(trades_c, 'C')
    if res_c:
        res_c = five_gate_validation(res_c)
        results.append(res_c)

    # Variant D
    log.info("\n=== Variant D: Golden Cross + Cheap Calls ===")
    trades_d = run_variant_D(prices_df, spy_close, crossovers)
    log.info(f"  Generated {len(trades_d)} trades")
    res_d = backtest_variant(trades_d, 'D')
    if res_d:
        res_d = five_gate_validation(res_d)
        results.append(res_d)

    # Variant E
    log.info("\n=== Variant E: Momentum Confirmation (5d follow-through) ===")
    trades_e = run_variant_E(prices_df, spy_close, crossovers)
    log.info(f"  Generated {len(trades_e)} trades")
    res_e = backtest_variant(trades_e, 'E')
    if res_e:
        res_e = five_gate_validation(res_e)
        results.append(res_e)

    # Variant F
    log.info("\n=== Variant F: Sector-Diversified Leaders ===")
    trades_f = run_variant_F(prices_df, spy_close, crossovers)
    log.info(f"  Generated {len(trades_f)} trades")
    res_f = backtest_variant(trades_f, 'F')
    if res_f:
        res_f = five_gate_validation(res_f)
        results.append(res_f)

    # ==================== RESULTS ====================
    log.info("\n" + "=" * 90)
    log.info("RELATIVE STRENGTH REGIME SWITCH v1 — RESULTS SUMMARY")
    log.info("=" * 90)
    log.info(f"{'Var':<4} {'Trades':<7} {'WR':<7} {'Sharpe':<8} {'Sortino':<9} {'PF':<7} "
             f"{'PnL':>8} {'Final$':>8} {'MDD':>7} {'Gates':>6} {'Pass':>5}")
    log.info("-" * 90)

    for r in results:
        if r is None:
            continue
        log.info(f"{r['variant']:<4} {r['n_trades']:<7} {r['win_rate']:<7.1%} "
                 f"{r['sharpe']:<8.2f} {r['sortino']:<9.2f} {r['profit_factor']:<7.2f} "
                 f"${r['total_pnl']:>7.2f} ${r['final_capital']:>7.2f} "
                 f"{r['max_dd']:>6.1%} {r['gates_passed']:>4}/5 "
                 f"{'YES' if r.get('all_gates_pass') else 'NO':>5}")

    log.info("\n--- Gate Details ---")
    for r in results:
        if r is None:
            continue
        gates = r.get('gates', {})
        log.info(f"\nVariant {r['variant']}:")
        for gate, passed in gates.items():
            status = 'PASS' if passed else 'FAIL'
            log.info(f"  [{status}] {gate}")
        log.info(f"  perm_p={r.get('perm_p', 'N/A'):.4f}, "
                 f"regime_gap={r.get('regime_gap', 'N/A'):.3f}, "
                 f"beats_random={r.get('pct_beats_random', 'N/A'):.1%}")

    if not results:
        log.warning("\nNO VARIANTS PRODUCED RESULTS. Likely too few regime crossovers in OOT period.")

    # Trade details
    for r in results:
        if r is None:
            continue
        log.info(f"\n--- Variant {r['variant']} Trade Details ---")
        for t in r['trades'][:10]:
            log.info(f"  {t['ticker']}: {t.get('direction', 'long')} "
                     f"{t['entry_date'].strftime('%Y-%m-%d')} -> {t['exit_date'].strftime('%Y-%m-%d')} "
                     f"${t['entry_price']:.2f} -> ${t['exit_price']:.2f} "
                     f"PnL=${t['pnl']:.2f} ({t['ret']:+.1%}) RS={t.get('rs_score', 0):.3f}")

    # MLflow
    log_to_mlflow(results)

    # Save results
    output_file = os.path.join(LOG_DIR, 'relative_strength_regime_v1_results.json')
    save_results = []
    for r in results:
        if r is None:
            continue
        save_r = {k: v for k, v in r.items()
                  if k not in ('trades', 'equity_curve', 'daily_returns')}
        save_r['gates'] = r.get('gates', {})
        # Save trade summary
        save_r['trade_details'] = [{
            'ticker': t['ticker'], 'direction': t.get('direction', 'long'),
            'entry': str(t['entry_date'].date()), 'exit': str(t['exit_date'].date()),
            'pnl': round(t['pnl'], 2), 'ret': round(t['ret'], 4),
        } for t in r['trades']]
        save_results.append(save_r)

    with open(output_file, 'w') as f:
        json.dump(save_results, f, indent=2, default=str)

    log.info(f"\nResults saved. Done.")
    return results


if __name__ == '__main__':
    main()
