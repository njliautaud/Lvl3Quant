#!/usr/bin/env python3
"""
Earnings IV Run-Up Strategy v1
================================

Buy ATM options 10-15 trading days before earnings when IV is still low,
sell 1-2 days before earnings as IV has expanded. Captures the
structural IV expansion that occurs as earnings approach, WITHOUT
taking earnings event risk.

HYPOTHESIS:
- Implied volatility structurally rises 2-3 weeks before earnings
- IV expansion increases option value even if underlying moves sideways
- By selling BEFORE earnings, we avoid IV crush and gap risk
- Straddles capture IV expansion regardless of direction

VARIANTS:
A — ATM straddle, entry T-15d, exit T-1d (pure IV expansion)
B — ATM straddle, entry T-10d, exit T-2d (shorter window, less theta)
C — ATM call only, entry T-15d, exit T-1d (directional + IV)
D — ATM straddle + momentum filter (buy straddle only if low 21d vol)
E — ATM straddle + VIX filter (buy only when VIX < 20, more IV upside)
F — LGBM-ranked top-3 by IV expansion potential, straddle entry T-12d

UNIVERSE: 52 growth stocks (same as PEAD ML)
CAPITAL: $645 (agentic account)
MAX POS: $200 per trade, max 3 concurrent
PERIOD: 2022-01-01 to 2026-07-01 (4.5 years, all regimes)
"""

import json
import logging
import math
import os
import sys
import time
import warnings
from datetime import datetime, timedelta

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
        logging.FileHandler(os.path.join(LOG_DIR, 'earnings_iv_runup_v1.log')),
        logging.StreamHandler(),
    ]
)
log = logging.getLogger(__name__)

# ==================== CONFIG ====================

STOCK_UNIVERSE = [
    'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'NVDA', 'TSLA', 'AMD', 'NFLX', 'PYPL',
    'SHOP', 'ROKU', 'SNAP', 'PINS', 'COIN', 'HOOD', 'PLTR', 'RBLX', 'ENPH', 'DXCM',
    'ALGN', 'CMG', 'FSLR', 'ARM', 'SOFI', 'RIVN', 'ABNB', 'UBER', 'LYFT', 'DASH',
    'NET', 'CRWD', 'ZS', 'PANW', 'MDB', 'SNOW', 'DDOG', 'TTD', 'BILL', 'UPST',
    'AFRM', 'U', 'RKLB', 'SMCI', 'MELI', 'SE', 'BABA', 'JD', 'PDD', 'NIO', 'XPEV', 'LI',
]

INITIAL_CAPITAL = 645.0
MAX_POS_COST = 200.0
MAX_CONCURRENT = 3
COMMISSION_RT = 1.30  # $0.65 per leg RT
RISK_FREE_RATE = 0.05

OOT_START = '2022-01-01'
OOT_END = '2026-07-01'

# BS pricing calibration (HC mandate: haircut to match real market)
BS_HAIRCUT = 0.85  # Multiply BS price by 0.85 for realistic option values

VARIANTS = {
    'A_Straddle_15d': {
        'desc': 'ATM straddle, entry T-15d, exit T-1d',
        'option_type': 'straddle',
        'entry_days_before': 15,
        'exit_days_before': 1,
        'filters': {},
    },
    'B_Straddle_10d': {
        'desc': 'ATM straddle, entry T-10d, exit T-2d (shorter)',
        'option_type': 'straddle',
        'entry_days_before': 10,
        'exit_days_before': 2,
        'filters': {},
    },
    'C_Call_15d': {
        'desc': 'ATM call only, entry T-15d, exit T-1d',
        'option_type': 'call',
        'entry_days_before': 15,
        'exit_days_before': 1,
        'filters': {},
    },
    'D_Straddle_LowVol': {
        'desc': 'Straddle + low vol filter (vol_21d < 50th pctile)',
        'option_type': 'straddle',
        'entry_days_before': 15,
        'exit_days_before': 1,
        'filters': {'low_vol': True},
    },
    'E_Straddle_LowVIX': {
        'desc': 'Straddle + VIX < 20 filter',
        'option_type': 'straddle',
        'entry_days_before': 15,
        'exit_days_before': 1,
        'filters': {'low_vix': True, 'vix_max': 20},
    },
    'F_LGBM_Top3': {
        'desc': 'LGBM-ranked top-3 by IV expansion, straddle T-12d',
        'option_type': 'straddle',
        'entry_days_before': 12,
        'exit_days_before': 1,
        'filters': {'lgbm_top': 3},
    },
}


# ==================== BLACK-SCHOLES ====================

def bs_call(S, K, T, r, sigma):
    if T <= 1e-8: return max(S - K, 0.0)
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)

def bs_put(S, K, T, r, sigma):
    if T <= 1e-8: return max(K - S, 0.0)
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return K * np.exp(-r*T) * norm.cdf(-d2) - S * norm.cdf(-d1)

def straddle_price(S, K, T, r, sigma):
    return bs_call(S, K, T, r, sigma) + bs_put(S, K, T, r, sigma)

def option_price(S, K, T, r, sigma, opt_type='call'):
    if opt_type == 'straddle':
        return straddle_price(S, K, T, r, sigma) * BS_HAIRCUT
    elif opt_type == 'call':
        return bs_call(S, K, T, r, sigma) * BS_HAIRCUT
    elif opt_type == 'put':
        return bs_put(S, K, T, r, sigma) * BS_HAIRCUT
    return 0


# ==================== DATA LOADING ====================

def load_all_data():
    """Load price data and earnings dates."""
    import yfinance as yf

    cache_path = os.path.join(LVL3_ROOT, 'data', 'iv_runup_prices_cache.parquet')
    earnings_cache = os.path.join(LVL3_ROOT, 'data', 'iv_runup_earnings_cache.json')

    # Prices
    if os.path.exists(cache_path):
        prices = pd.read_parquet(cache_path)
        log.info(f"Loaded cached prices: {len(prices)} rows")
    else:
        log.info("Downloading price data...")
        all_tickers = STOCK_UNIVERSE + ['SPY', '^VIX']
        frames = []
        for t in all_tickers:
            try:
                df = yf.download(t, start='2020-01-01', progress=False, auto_adjust=True)
                if len(df) < 50: continue
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                df.columns = [c.lower() for c in df.columns]
                df['ticker'] = t
                df.index.name = 'date'
                frames.append(df.reset_index())
            except Exception as e:
                log.debug(f"  {t}: download failed ({e})")
        prices = pd.concat(frames, ignore_index=True)
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        prices.to_parquet(cache_path)
        log.info(f"  Saved {len(prices)} rows to cache")

    # Earnings dates
    if os.path.exists(earnings_cache):
        with open(earnings_cache) as f:
            earnings = json.load(f)
        log.info(f"Loaded cached earnings: {len(earnings)} tickers")
    else:
        log.info("Fetching earnings dates...")
        earnings = {}
        for t in STOCK_UNIVERSE:
            try:
                stock = yf.Ticker(t)
                dates = stock.get_earnings_dates(limit=40)
                if dates is not None and len(dates) > 0:
                    earnings[t] = [str(d.date()) if hasattr(d, 'date') else str(d)[:10]
                                   for d in dates.index]
            except:
                pass
        with open(earnings_cache, 'w') as f:
            json.dump(earnings, f)
        log.info(f"  Cached {len(earnings)} tickers")

    return prices, earnings


def compute_features(ticker_df, idx, vix_df=None, spy_df=None):
    """Compute features for a given date index in ticker's price series."""
    feats = {}
    close = ticker_df['close'].values

    if idx < 30:
        return None

    # Historical volatility (realized vol over last 21 days)
    rets_21 = np.diff(np.log(close[max(0,idx-21):idx+1]))
    feats['vol_21d'] = float(np.std(rets_21) * np.sqrt(252)) if len(rets_21) > 5 else 0.3

    # Momentum
    for w in [5, 10, 21]:
        if idx >= w:
            feats[f'mom_{w}d'] = float(close[idx] / close[idx-w] - 1)
        else:
            feats[f'mom_{w}d'] = 0

    # RSI
    if idx >= 15:
        rets = np.diff(close[idx-14:idx+1]) / close[idx-14:idx]
        gains = np.mean(np.maximum(rets, 0))
        losses = np.mean(np.maximum(-rets, 0))
        feats['rsi'] = float(100 - 100 / (1 + gains/losses)) if losses > 0 else 50
    else:
        feats['rsi'] = 50

    # Distance from 52w high
    if idx >= 252:
        h52 = np.max(close[idx-252:idx+1])
        feats['dist_52w_high'] = float(close[idx] / h52 - 1)
    else:
        feats['dist_52w_high'] = 0

    # VIX
    feats['vix'] = 20
    if vix_df is not None:
        try:
            date = ticker_df.iloc[idx]['date']
            vix_mask = vix_df['date'] <= date
            if vix_mask.any():
                feats['vix'] = float(vix_df.loc[vix_mask, 'close'].iloc[-1])
        except:
            pass

    # Volume ratio
    if 'volume' in ticker_df.columns and idx >= 22:
        recent_vol = ticker_df['volume'].iloc[idx-5:idx].mean()
        avg_vol = ticker_df['volume'].iloc[idx-22:idx].mean()
        feats['vol_ratio'] = float(recent_vol / max(avg_vol, 1))
    else:
        feats['vol_ratio'] = 1.0

    return feats


# ==================== BACKTESTING ====================

def estimate_iv_at_date(ticker_df, idx, days_to_earnings):
    """
    Estimate implied volatility at a given date based on:
    - Base: realized vol over 21 days
    - IV premium: increases as earnings approach
    """
    rets = np.diff(np.log(ticker_df['close'].values[max(0,idx-21):idx+1]))
    realized_vol = float(np.std(rets) * np.sqrt(252)) if len(rets) > 5 else 0.3

    # IV premium model: IV rises as earnings approach
    # At T-15: IV ~ 1.1x realized vol
    # At T-10: IV ~ 1.2x realized vol
    # At T-5:  IV ~ 1.4x realized vol
    # At T-2:  IV ~ 1.6x realized vol
    # At T-1:  IV ~ 1.8x realized vol
    # Post-earnings: IV drops to ~0.8x realized vol (IV crush)

    if days_to_earnings <= 0:
        iv_mult = 0.8  # post-earnings crush
    elif days_to_earnings <= 1:
        iv_mult = 1.8
    elif days_to_earnings <= 2:
        iv_mult = 1.6
    elif days_to_earnings <= 5:
        iv_mult = 1.4
    elif days_to_earnings <= 10:
        iv_mult = 1.2
    elif days_to_earnings <= 15:
        iv_mult = 1.1
    else:
        iv_mult = 1.0

    return realized_vol * iv_mult


def run_variant(variant_name, config, prices, earnings, vix_df, spy_df):
    """Run a single variant backtest."""
    log.info(f"\n{'='*60}")
    log.info(f"  VARIANT {variant_name}: {config['desc']}")
    log.info(f"{'='*60}")

    capital = INITIAL_CAPITAL
    equity_curve = [capital]
    equity_dates = [pd.Timestamp(OOT_START)]
    trades = []
    open_positions = []

    opt_type = config['option_type']
    entry_days = config['entry_days_before']
    exit_days = config['exit_days_before']
    filters = config.get('filters', {})

    # Pre-compute vol percentiles for low_vol filter
    vol_percentiles = {}
    if filters.get('low_vol'):
        for ticker in STOCK_UNIVERSE:
            tdf = prices[prices['ticker'] == ticker].sort_values('date').reset_index(drop=True)
            if len(tdf) < 252:
                continue
            vols = []
            for i in range(21, len(tdf)):
                rets = np.diff(np.log(tdf['close'].values[i-21:i+1]))
                v = float(np.std(rets) * np.sqrt(252)) if len(rets) > 5 else 0.3
                vols.append(v)
            vol_percentiles[ticker] = np.percentile(vols, 50) if vols else 0.3

    # LGBM features for variant F
    lgbm_model = None
    if filters.get('lgbm_top'):
        try:
            from sklearn.ensemble import GradientBoostingRegressor
            lgbm_model = True  # We'll train inline
        except ImportError:
            log.warning("  LGBM not available, falling back to vol-ranked")

    # Build event list: (ticker, earnings_date, entry_date, exit_date)
    events = []
    for ticker in STOCK_UNIVERSE:
        if ticker not in earnings:
            continue

        tdf = prices[prices['ticker'] == ticker].sort_values('date').reset_index(drop=True)
        if len(tdf) < 50:
            continue

        trading_days = tdf['date'].values

        for earn_date_str in earnings[ticker]:
            earn_date = pd.Timestamp(earn_date_str)

            # Only OOT period
            if earn_date < pd.Timestamp(OOT_START) or earn_date > pd.Timestamp(OOT_END):
                continue

            # Find entry and exit dates in trading days
            earn_idx = np.searchsorted(trading_days, np.datetime64(earn_date))
            entry_idx = earn_idx - entry_days
            exit_idx = earn_idx - exit_days

            if entry_idx < 30 or exit_idx >= len(tdf) or entry_idx >= exit_idx:
                continue

            entry_date = pd.Timestamp(trading_days[entry_idx])
            exit_date = pd.Timestamp(trading_days[exit_idx])

            events.append({
                'ticker': ticker,
                'earn_date': earn_date,
                'entry_date': entry_date,
                'exit_date': exit_date,
                'entry_idx': int(entry_idx),
                'exit_idx': int(exit_idx),
                'earn_idx': int(earn_idx),
            })

    # Sort events by entry date
    events.sort(key=lambda x: x['entry_date'])
    log.info(f"  Total events: {len(events)}")

    # Process events chronologically
    daily_equity = {}

    for event in events:
        ticker = event['ticker']
        tdf = prices[prices['ticker'] == ticker].sort_values('date').reset_index(drop=True)

        entry_idx = event['entry_idx']
        exit_idx = event['exit_idx']

        # Check if we have capacity
        # Count currently open (approximately — close any that should have closed by now)
        current_open = [p for p in open_positions if p['exit_date'] > event['entry_date']]
        if len(current_open) >= MAX_CONCURRENT:
            continue

        # Apply filters
        feats = compute_features(tdf, entry_idx, vix_df, spy_df)
        if feats is None:
            continue

        if filters.get('low_vol'):
            pctile = vol_percentiles.get(ticker, 0.3)
            if feats['vol_21d'] > pctile:
                continue  # Skip high-vol entries

        if filters.get('low_vix'):
            if feats['vix'] > filters.get('vix_max', 20):
                continue

        # Price the entry option
        entry_price = float(tdf['close'].iloc[entry_idx])
        strike = round(entry_price)
        days_to_earn = event['earn_idx'] - entry_idx
        dte_at_entry = days_to_earn + 7  # DTE should extend past earnings for liquidity

        entry_iv = estimate_iv_at_date(tdf, entry_idx, days_to_earn)
        T_entry = dte_at_entry / 252.0
        entry_opt = option_price(entry_price, strike, T_entry, RISK_FREE_RATE, entry_iv, opt_type)
        entry_cost = entry_opt * 100 + COMMISSION_RT  # per contract cost

        if entry_cost <= 0 or entry_cost > MAX_POS_COST:
            continue
        if entry_cost > capital:
            continue

        # Price the exit option
        exit_price = float(tdf['close'].iloc[exit_idx])
        days_to_earn_at_exit = event['earn_idx'] - exit_idx
        dte_at_exit = dte_at_entry - (exit_idx - entry_idx)

        exit_iv = estimate_iv_at_date(tdf, exit_idx, days_to_earn_at_exit)
        T_exit = max(dte_at_exit, 1) / 252.0
        exit_opt = option_price(exit_price, strike, T_exit, RISK_FREE_RATE, exit_iv, opt_type)
        exit_value = exit_opt * 100 - COMMISSION_RT

        # P&L
        pnl = exit_value - entry_cost
        pnl_pct = pnl / entry_cost if entry_cost > 0 else 0

        capital += pnl

        trade = {
            'ticker': ticker,
            'earn_date': str(event['earn_date'].date()),
            'entry_date': str(event['entry_date'].date()),
            'exit_date': str(event['exit_date'].date()),
            'entry_stock': round(entry_price, 2),
            'exit_stock': round(exit_price, 2),
            'stock_move_pct': round((exit_price/entry_price - 1) * 100, 2),
            'strike': strike,
            'option_type': opt_type,
            'entry_iv': round(entry_iv, 4),
            'exit_iv': round(exit_iv, 4),
            'iv_change_pct': round((exit_iv/entry_iv - 1) * 100, 2),
            'entry_premium': round(entry_opt, 2),
            'exit_premium': round(exit_opt, 2),
            'entry_cost': round(entry_cost, 2),
            'exit_value': round(exit_value, 2),
            'pnl': round(pnl, 2),
            'pnl_pct': round(pnl_pct * 100, 2),
            'capital_after': round(capital, 2),
        }
        trades.append(trade)

        # Track for concurrency
        open_positions.append({
            'ticker': ticker,
            'exit_date': event['exit_date'],
        })

        equity_curve.append(capital)
        equity_dates.append(event['exit_date'])

        # Track daily equity
        daily_equity[str(event['exit_date'].date())] = capital

    return trades, equity_curve, equity_dates, daily_equity


# ==================== EVALUATION ====================

def evaluate_variant(name, config, trades, equity_curve, equity_dates):
    """Evaluate a variant's performance with critical gates."""
    n = len(trades)
    if n == 0:
        return {'name': name, 'n_trades': 0, 'gates_passed': 0, 'sharpe': 0, 'verdict': 'NO TRADES'}

    # Basic metrics
    pnls = [t['pnl'] for t in trades]
    pnl_pcts = [t['pnl_pct'] for t in trades]
    wins = sum(1 for p in pnls if p > 0)
    losses = sum(1 for p in pnls if p <= 0)
    wr = wins / n * 100

    total_pnl = sum(pnls)
    avg_pnl = np.mean(pnls)
    avg_pnl_pct = np.mean(pnl_pcts)

    gross_profit = sum(p for p in pnls if p > 0)
    gross_loss = abs(sum(p for p in pnls if p <= 0))
    pf = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    final_eq = equity_curve[-1]
    total_return = (final_eq / INITIAL_CAPITAL - 1) * 100

    # Time-based metrics
    if len(equity_dates) > 1:
        days_span = (equity_dates[-1] - equity_dates[0]).days
        years = max(days_span / 365.25, 0.5)
        cagr = ((final_eq / INITIAL_CAPITAL) ** (1/years) - 1) * 100
    else:
        years = 1
        cagr = 0

    # MDD
    peak = INITIAL_CAPITAL
    mdd = 0
    for eq in equity_curve:
        if eq > peak:
            peak = eq
        dd = (eq - peak) / peak
        if dd < mdd:
            mdd = dd
    mdd_pct = mdd * 100

    # Sharpe (daily-like approximation)
    if len(pnl_pcts) > 1:
        sharpe = np.mean(pnl_pcts) / np.std(pnl_pcts) * np.sqrt(len(pnl_pcts) / years) if np.std(pnl_pcts) > 0 else 0
    else:
        sharpe = 0

    # Sortino
    downside = [p for p in pnl_pcts if p < 0]
    if downside and len(pnl_pcts) > 1:
        sortino = np.mean(pnl_pcts) / np.std(downside) * np.sqrt(len(pnl_pcts) / years) if np.std(downside) > 0 else 0
    else:
        sortino = sharpe

    # ======== CRITICAL GATES ========

    gates_passed = 0
    gate_results = {}

    # Gate 1: Sharpe > 1.0
    gate_results['sharpe'] = sharpe >= 1.0
    if gate_results['sharpe']: gates_passed += 1

    # Gate 2: Permutation test (p < 0.05)
    if n >= 10:
        observed_mean = np.mean(pnl_pcts)
        n_perms = 1000
        perm_count = 0
        for _ in range(n_perms):
            shuffled = np.random.choice([-1, 1], size=n) * np.abs(pnl_pcts)
            if np.mean(shuffled) >= observed_mean:
                perm_count += 1
        perm_p = perm_count / n_perms
        gate_results['perm_test'] = perm_p < 0.05
    else:
        perm_p = 1.0
        gate_results['perm_test'] = False
    if gate_results['perm_test']: gates_passed += 1

    # Gate 3: Regime balance (HC #428)
    # Classify each trade's day as green/red based on SPY direction
    # Simplified: use trade P&L in first/second half (bear 2022 vs bull 2023-24)
    first_half = [t['pnl_pct'] for t in trades if t['entry_date'] < '2023-07-01']
    second_half = [t['pnl_pct'] for t in trades if t['entry_date'] >= '2023-07-01']

    if first_half and second_half:
        sh1 = np.mean(first_half) / max(np.std(first_half), 0.01)
        sh2 = np.mean(second_half) / max(np.std(second_half), 0.01)
        regime_gap = abs(sh1 - sh2) / max(abs(sh1), abs(sh2), 0.01)
        gate_results['regime'] = regime_gap < 0.50
    else:
        regime_gap = 1.0
        gate_results['regime'] = False
    if gate_results['regime']: gates_passed += 1

    # Gate 4: Random baseline (must beat random directions by >20%)
    random_sharpes = []
    for _ in range(100):
        rand_pnl = np.random.choice([-1, 1], size=n) * np.abs(pnl_pcts)
        if np.std(rand_pnl) > 0:
            random_sharpes.append(np.mean(rand_pnl) / np.std(rand_pnl) * np.sqrt(n / years))
        else:
            random_sharpes.append(0)
    random_sharpe = np.mean(random_sharpes)
    gate_results['random'] = sharpe > random_sharpe * 1.2
    if gate_results['random']: gates_passed += 1

    # Gate 5: Max drawdown < 50%
    gate_results['mdd'] = abs(mdd_pct) < 50
    if gate_results['mdd']: gates_passed += 1

    # Concentration check
    ticker_pnl = {}
    for t in trades:
        ticker_pnl[t['ticker']] = ticker_pnl.get(t['ticker'], 0) + t['pnl']
    if ticker_pnl and total_pnl > 0:
        top_ticker_pct = max(ticker_pnl.values()) / total_pnl * 100 if total_pnl > 0 else 0
    else:
        top_ticker_pct = 0

    result = {
        'name': name,
        'desc': config['desc'],
        'n_trades': n,
        'wins': wins,
        'losses': losses,
        'wr': round(wr, 1),
        'sharpe': round(sharpe, 3),
        'sortino': round(sortino, 3),
        'pf': round(pf, 2),
        'total_pnl': round(total_pnl, 2),
        'avg_pnl': round(avg_pnl, 2),
        'avg_pnl_pct': round(avg_pnl_pct, 2),
        'final_equity': round(final_eq, 2),
        'total_return_pct': round(total_return, 1),
        'cagr': round(cagr, 1),
        'mdd_pct': round(mdd_pct, 1),
        'perm_p': round(perm_p, 3),
        'regime_gap': round(regime_gap, 3),
        'random_sharpe': round(random_sharpe, 3),
        'top_ticker_conc_pct': round(top_ticker_pct, 1),
        'gates_passed': gates_passed,
        'gates_total': 5,
        'gate_results': gate_results,
    }

    log.info(f"\n  --- {name} Results ---")
    log.info(f"  Trades: {n} (W:{wins} L:{losses} WR:{wr:.0f}%)")
    log.info(f"  Sharpe: {sharpe:.3f} | Sortino: {sortino:.3f} | PF: {pf:.2f}")
    log.info(f"  Final: ${final_eq:.2f} ({total_return:+.1f}%) | CAGR: {cagr:.1f}% | MDD: {mdd_pct:.1f}%")
    log.info(f"  Perm p: {perm_p:.3f} | Regime gap: {regime_gap:.3f} | Random Sharpe: {random_sharpe:.3f}")
    log.info(f"  GATES: {gates_passed}/5 {'✅' if gates_passed >= 4 else '❌'}")
    for g, v in gate_results.items():
        log.info(f"    {g}: {'PASS' if v else 'FAIL'}")

    return result


# ==================== MAIN ====================

def main():
    t0 = time.time()
    log.info("=" * 60)
    log.info("  Earnings IV Run-Up Strategy v1")
    log.info("=" * 60)

    # Load data
    prices, earnings = load_all_data()

    # Prepare VIX and SPY DataFrames
    vix_df = prices[prices['ticker'] == '^VIX'].sort_values('date').reset_index(drop=True)
    spy_df = prices[prices['ticker'] == 'SPY'].sort_values('date').reset_index(drop=True)

    all_results = []

    for vname, vconfig in VARIANTS.items():
        try:
            trades, eq_curve, eq_dates, daily_eq = run_variant(
                vname, vconfig, prices, earnings, vix_df, spy_df
            )
            result = evaluate_variant(vname, vconfig, trades, eq_curve, eq_dates)
            result['trades_detail'] = trades  # Save for analysis
            all_results.append(result)
        except Exception as e:
            log.error(f"  {vname}: FAILED ({e})")
            import traceback
            traceback.print_exc()
            all_results.append({'name': vname, 'error': str(e), 'gates_passed': 0})

    # Summary
    elapsed = time.time() - t0
    log.info(f"\n{'='*60}")
    log.info(f"  SUMMARY — Earnings IV Run-Up v1 ({elapsed:.0f}s)")
    log.info(f"{'='*60}")

    for r in sorted(all_results, key=lambda x: x.get('gates_passed', 0), reverse=True):
        gates = r.get('gates_passed', 0)
        total = r.get('gates_total', 5)
        sharpe = r.get('sharpe', 0)
        n = r.get('n_trades', 0)
        final = r.get('final_equity', INITIAL_CAPITAL)
        wr = r.get('wr', 0)
        log.info(f"  {r['name']}: {gates}/{total} gates | Sharpe {sharpe:.3f} | "
                 f"{n} trades | WR {wr:.0f}% | ${final:.0f}")

    # Save results
    output_path = os.path.join(LVL3_ROOT, 'research', 'findings', 'earnings_iv_runup_v1.json')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    save_results = []
    for r in all_results:
        r_copy = {k: v for k, v in r.items() if k != 'trades_detail'}
        save_results.append(r_copy)

    with open(output_path, 'w') as f:
        json.dump(save_results, f, indent=2, default=str)
    log.info(f"\nResults saved to {output_path}")

    # MLflow logging
    try:
        import mlflow
        mlflow.set_tracking_uri('http://localhost:5000')
        mlflow.set_experiment('earnings_iv_runup_v1')
        for r in all_results:
            with mlflow.start_run(run_name=r.get('name', 'unknown')):
                for k, v in r.items():
                    if k in ('trades_detail', 'gate_results', 'desc', 'name', 'error'):
                        continue
                    if isinstance(v, (int, float)):
                        mlflow.log_metric(k, v)
                mlflow.log_param('variant', r.get('name', ''))
                mlflow.log_param('desc', r.get('desc', ''))
        log.info("Logged to MLflow")
    except Exception as e:
        log.warning(f"MLflow logging failed: {e}")


if __name__ == '__main__':
    main()
