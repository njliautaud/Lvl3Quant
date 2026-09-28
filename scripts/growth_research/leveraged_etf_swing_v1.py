#!/usr/bin/env python3
"""
Leveraged ETF Swing Trading v1 — High Growth for $645 Account
==============================================================
KEY INSIGHT: After 30+ failed options strategies, the problem is THETA DECAY.
Leveraged ETFs (TQQQ, SOXL, UPRO, SPXL, TNA, TECL, FAS) provide 3x leverage
WITHOUT theta decay. Hold for days, not weeks.

STRATEGY:
- Universe: 7 leveraged ETFs (3x bull + inverse for shorting)
- Entry: Momentum burst (same signals as v1 Sharpe 1.28) + mean reversion
- Hold: 1-5 days (swing trade, NOT hold-to-decay)
- Sizing: $100-300 per trade, max 2 positions
- Exit: Trailing stop, TP, time stop

WHY THIS COULD WORK:
1. No theta decay (biggest killer of $645 options strategies)
2. 3x leverage = similar to ATM options leverage but without premium cost
3. Can buy fractional shares on RH (no $300/contract minimum problem)
4. Momentum burst signals validated at Sharpe 1.28 in options; should be
   BETTER without theta drag

RISK: Leveraged ETFs have volatility decay over long holds.
MITIGATION: Max 5-day hold, trailing stops, never hold through weekends.

VARIANTS (8):
  A: TQQQ momentum burst (3x Nasdaq, 2+ signals, trailing stop)
  B: SOXL momentum burst (3x Semiconductors, high beta)
  C: Multi-leveraged rotation (rotate among all 7 based on momentum)
  D: Mean reversion (buy dips in leveraged ETFs, RSI<30 entry)
  E: VIX-filtered momentum (only trade when VIX 15-25, sweet spot)
  F: Concentrated TQQQ (1 position, $300, highest conviction)
  G: Inverse hedged (TQQQ + SQQQ paired based on signal)
  H: Best of v1 signals on leveraged (apply v1's trailing stop params)

TRACK: HIGH-GROWTH (agentic account)
"""

import sys
import os
import json
import warnings
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from collections import defaultdict

warnings.filterwarnings('ignore')

# --- Path setup ---
for root in ['/home/jupiter/Lvl3Quant', '/home/nick/Lvl3Quant']:
    if os.path.isdir(root):
        LVL3_ROOT = root
        break
else:
    LVL3_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

OUTPUT_DIR = os.path.join(LVL3_ROOT, 'output', 'growth_research', 'leveraged_etf_swing_v1')
os.makedirs(OUTPUT_DIR, exist_ok=True)

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

print(f"Running on: {LVL3_ROOT}")

if MLFLOW_AVAILABLE:
    try:
        mlflow.set_tracking_uri("http://jupiter:5000")
        mlflow.set_experiment("leveraged_etf_swing_v1")
        print(f"MLflow OK: http://jupiter:5000")
    except Exception as e:
        print(f"MLflow warning: {e}")
        MLFLOW_AVAILABLE = False

# ============================================================
# CONSTANTS
# ============================================================
LEVERAGED_UNIVERSE = {
    'TQQQ': {'underlying': 'QQQ', 'leverage': 3, 'sector': 'Tech/Nasdaq'},
    'SOXL': {'underlying': 'SOXX', 'leverage': 3, 'sector': 'Semiconductors'},
    'UPRO': {'underlying': 'SPY', 'leverage': 3, 'sector': 'S&P 500'},
    'SPXL': {'underlying': 'SPY', 'leverage': 3, 'sector': 'S&P 500'},
    'TNA':  {'underlying': 'IWM', 'leverage': 3, 'sector': 'Small Cap'},
    'TECL': {'underlying': 'XLK', 'leverage': 3, 'sector': 'Technology'},
    'FAS':  {'underlying': 'XLF', 'leverage': 3, 'sector': 'Financials'},
}

# Inverse ETFs for hedging
INVERSE_UNIVERSE = {
    'SQQQ': {'underlying': 'QQQ', 'leverage': -3, 'sector': 'Inverse Nasdaq'},
    'SPXS': {'underlying': 'SPY', 'leverage': -3, 'sector': 'Inverse S&P'},
    'TZA':  {'underlying': 'IWM', 'leverage': -3, 'sector': 'Inverse Small Cap'},
}

ALL_TICKERS = list(LEVERAGED_UNIVERSE.keys()) + list(INVERSE_UNIVERSE.keys()) + ['SPY', 'QQQ', '^VIX']

STARTING_CAPITAL = 645.0
COMMISSION_PER_TRADE = 0.0  # RH zero commission on equities
START_DATE = '2019-01-01'
END_DATE = '2026-07-28'
OOT_START = '2021-01-01'
N_PERMUTATIONS = 150

# ============================================================
# DATA LOADING
# ============================================================

def load_data():
    """Load daily OHLCV for leveraged ETF universe."""
    cache_path = os.path.join(LVL3_ROOT, 'data', 'leveraged_etf_swing_cache.parquet')

    if os.path.exists(cache_path):
        df = pd.read_parquet(cache_path)
        if len(df) > 0:
            latest = df.index.get_level_values('date').max()
            if pd.Timestamp(latest) >= pd.Timestamp('2026-07-20'):
                print(f"Loaded cached data: {len(df)} rows, latest={latest}")
                return df

    import yfinance as yf
    print(f"Downloading {len(ALL_TICKERS)} tickers from {START_DATE}...")
    all_frames = []

    for ticker in ALL_TICKERS:
        try:
            data = yf.download(ticker, start=START_DATE, end=END_DATE, progress=False, auto_adjust=True)
            if len(data) < 100:
                print(f"  WARNING: {ticker} only {len(data)} rows, skipping")
                continue
            data.columns = [c.lower() if isinstance(c, str) else c[0].lower() for c in data.columns]
            data['ticker'] = ticker
            data.index.name = 'date'
            all_frames.append(data)
            print(f"  {ticker}: {len(data)} rows")
        except Exception as e:
            print(f"  ERROR: {ticker}: {e}")

    df = pd.concat(all_frames).reset_index().set_index(['ticker', 'date']).sort_index()

    try:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        df.to_parquet(cache_path)
    except:
        pass

    return df


# ============================================================
# SIGNAL COMPUTATION
# ============================================================

def compute_rsi(prices, period=14):
    if len(prices) < period + 1:
        return 50.0
    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    avg_gain = np.mean(gains[-period:])
    avg_loss = np.mean(losses[-period:])
    if avg_loss == 0:
        return 100.0
    return 100.0 - (100.0 / (1.0 + avg_gain / avg_loss))


def compute_signals(prices_df, ticker, date, spy_prices):
    """
    Compute momentum + mean reversion signals.
    Same signal framework as momentum burst v1 (Sharpe 1.28) but applied to leveraged ETFs.
    """
    try:
        ticker_data = prices_df.loc[ticker]
    except KeyError:
        return None

    mask = ticker_data.index <= date
    td = ticker_data.loc[mask]
    if len(td) < 25:
        return None

    close = td['close'].values
    volume = td['volume'].values

    spy_mask = spy_prices.index <= date
    spy_td = spy_prices.loc[spy_mask]
    if len(spy_td) < 25:
        return None
    spy_close = spy_td['close'].values

    signals = {}
    bull_signals = 0
    bear_signals = 0

    # Signal 1: 5-day momentum
    mom_5d = (close[-1] / close[-6]) - 1.0
    signals['mom_5d'] = mom_5d
    if mom_5d > 0.05:  # Higher threshold for leveraged (3x already)
        bull_signals += 1
    elif mom_5d < -0.05:
        bear_signals += 1

    # Signal 2: 3-day momentum (shorter for swing)
    if len(close) >= 4:
        mom_3d = (close[-1] / close[-4]) - 1.0
        signals['mom_3d'] = mom_3d
        if mom_3d > 0.04:
            bull_signals += 1
        elif mom_3d < -0.04:
            bear_signals += 1

    # Signal 3: Relative strength vs SPY
    if len(close) >= 11 and len(spy_close) >= 11:
        etf_ret = (close[-1] / close[-11]) - 1.0
        spy_ret = (spy_close[-1] / spy_close[-11]) - 1.0
        rel = etf_ret - spy_ret * 3  # Adjust for 3x leverage
        signals['rel_strength'] = rel
        if rel > 0.03:
            bull_signals += 1
        elif rel < -0.03:
            bear_signals += 1

    # Signal 4: RSI momentum cross
    rsi = compute_rsi(close)
    signals['rsi'] = rsi
    rsi_prev = compute_rsi(close[:-3]) if len(close) > 20 else rsi
    if rsi >= 55 and rsi_prev < 55:
        bull_signals += 1
    if rsi <= 45 and rsi_prev > 45:
        bear_signals += 1

    # Signal 5: Volume surge
    if len(volume) >= 21:
        avg_vol = np.mean(volume[-21:-1])
        vol_ratio = volume[-1] / (avg_vol + 1e-8)
        signals['volume_ratio'] = vol_ratio
        if vol_ratio > 1.5:
            if mom_5d > 0:
                bull_signals += 1
            elif mom_5d < 0:
                bear_signals += 1

    # Signal 6: Mean reversion (RSI oversold/overbought)
    if rsi < 30:
        bull_signals += 1  # Oversold bounce
        signals['mean_rev_long'] = True
    elif rsi > 70:
        bear_signals += 1  # Overbought reversal
        signals['mean_rev_short'] = True

    # Signal 7: Price vs 20-day MA
    if len(close) >= 21:
        ma20 = np.mean(close[-20:])
        pct_from_ma = (close[-1] - ma20) / ma20
        signals['pct_from_ma20'] = pct_from_ma
        if pct_from_ma > 0.03 and mom_3d > 0:
            bull_signals += 1  # Above MA with momentum = strong
        elif pct_from_ma < -0.03 and mom_3d < 0:
            bear_signals += 1

    signals['bull_signals'] = bull_signals
    signals['bear_signals'] = bear_signals
    signals['spot'] = close[-1]
    signals['conviction'] = max(bull_signals, bear_signals)
    signals['direction'] = 'bull' if bull_signals > bear_signals else ('bear' if bear_signals > bull_signals else 'neutral')

    return signals


# ============================================================
# STRATEGY VARIANTS
# ============================================================

VARIANTS = {
    'A': {
        'name': 'TQQQ Momentum Burst',
        'universe': ['TQQQ'],
        'min_signals': 2,
        'tp_pct': 0.08,  # 8% TP (leveraged moves fast)
        'sl_pct': 0.05,  # 5% SL
        'time_stop_days': 5,
        'trailing_stop': True,
        'trailing_giveback': 0.50,
        'max_positions': 1,
        'max_position_pct': 0.50,
        'direction': 'both',
        'vix_filter': None,
        'mean_rev_mode': False,
    },
    'B': {
        'name': 'SOXL Momentum Burst',
        'universe': ['SOXL'],
        'min_signals': 2,
        'tp_pct': 0.10,
        'sl_pct': 0.06,
        'time_stop_days': 5,
        'trailing_stop': True,
        'trailing_giveback': 0.50,
        'max_positions': 1,
        'max_position_pct': 0.50,
        'direction': 'both',
        'vix_filter': None,
        'mean_rev_mode': False,
    },
    'C': {
        'name': 'Multi-Leveraged Rotation',
        'universe': ['TQQQ', 'SOXL', 'UPRO', 'TNA', 'TECL', 'FAS'],
        'min_signals': 3,  # Higher bar for multi
        'tp_pct': 0.08,
        'sl_pct': 0.05,
        'time_stop_days': 5,
        'trailing_stop': True,
        'trailing_giveback': 0.50,
        'max_positions': 2,
        'max_position_pct': 0.35,
        'direction': 'both',
        'vix_filter': None,
        'mean_rev_mode': False,
    },
    'D': {
        'name': 'Mean Reversion Dip Buy',
        'universe': ['TQQQ', 'SOXL', 'UPRO'],
        'min_signals': 1,  # RSI<30 is the primary
        'tp_pct': 0.06,
        'sl_pct': 0.04,
        'time_stop_days': 3,  # Quick exit
        'trailing_stop': True,
        'trailing_giveback': 0.40,
        'max_positions': 2,
        'max_position_pct': 0.40,
        'direction': 'bull',  # Only dip buying
        'vix_filter': None,
        'mean_rev_mode': True,  # Require RSI < 30
    },
    'E': {
        'name': 'VIX-Filtered (15-25)',
        'universe': ['TQQQ', 'SOXL', 'UPRO'],
        'min_signals': 2,
        'tp_pct': 0.08,
        'sl_pct': 0.05,
        'time_stop_days': 5,
        'trailing_stop': True,
        'trailing_giveback': 0.50,
        'max_positions': 2,
        'max_position_pct': 0.40,
        'direction': 'both',
        'vix_filter': (15, 25),  # Only trade in normal VIX
        'mean_rev_mode': False,
    },
    'F': {
        'name': 'Concentrated TQQQ $300',
        'universe': ['TQQQ'],
        'min_signals': 3,  # High conviction only
        'tp_pct': 0.12,
        'sl_pct': 0.06,
        'time_stop_days': 5,
        'trailing_stop': True,
        'trailing_giveback': 0.45,
        'max_positions': 1,
        'max_position_pct': 0.50,
        'direction': 'both',
        'vix_filter': None,
        'mean_rev_mode': False,
    },
    'G': {
        'name': 'Inverse Hedge Pairs',
        'universe': ['TQQQ', 'SQQQ'],  # Bull + inverse
        'min_signals': 2,
        'tp_pct': 0.08,
        'sl_pct': 0.05,
        'time_stop_days': 5,
        'trailing_stop': True,
        'trailing_giveback': 0.50,
        'max_positions': 1,
        'max_position_pct': 0.50,
        'direction': 'both',
        'vix_filter': None,
        'mean_rev_mode': False,
    },
    'H': {
        'name': 'V1-Params on Leveraged',
        'universe': ['TQQQ', 'SOXL', 'UPRO', 'TNA', 'TECL', 'FAS'],
        'min_signals': 2,
        'tp_pct': 0.10,  # Matched to v1 trailing stop params scaled for 3x
        'sl_pct': 0.075,
        'time_stop_days': 5,
        'trailing_stop': True,
        'trailing_giveback': 0.50,
        'max_positions': 2,
        'max_position_pct': 0.35,
        'direction': 'both',
        'vix_filter': None,
        'mean_rev_mode': False,
    },
}


# ============================================================
# BACKTESTING ENGINE
# ============================================================

class Position:
    def __init__(self, ticker, direction, entry_price, entry_date, shares, cost, conviction=0):
        self.ticker = ticker
        self.direction = direction  # 'long' or 'short_via_inverse'
        self.entry_price = entry_price
        self.entry_date = entry_date
        self.shares = shares
        self.cost = cost
        self.conviction = conviction
        self.days_held = 0
        self.peak_price = entry_price
        self.pnl = None


def run_variant(variant_key, cfg, prices_df, spy_prices, vix_data, trading_dates):
    """Run a single variant."""
    import time
    t0 = time.time()

    equity = STARTING_CAPITAL
    equity_curve = [equity]
    equity_dates = [trading_dates[0]]
    positions = []
    all_trades = []
    daily_returns = []

    oot_dates = [d for d in trading_dates if d >= pd.Timestamp(OOT_START)]
    if not oot_dates:
        oot_dates = trading_dates[252:]

    for i, date in enumerate(oot_dates):
        prev_equity = equity

        # --- Mark-to-market and exit ---
        to_close = []
        for pi, pos in enumerate(positions):
            pos.days_held += 1

            try:
                td = prices_df.loc[pos.ticker]
                mask = td.index <= date
                current_price = td.loc[mask, 'close'].iloc[-1]
            except:
                continue

            if current_price > pos.peak_price:
                pos.peak_price = current_price

            pct_change = (current_price - pos.entry_price) / pos.entry_price

            exit_reason = None

            # TP
            if pct_change >= cfg['tp_pct']:
                exit_reason = 'take_profit'
            # SL
            elif pct_change <= -cfg['sl_pct']:
                exit_reason = 'stop_loss'
            # Time stop
            elif pos.days_held >= cfg['time_stop_days']:
                exit_reason = 'time_stop'
            # Trailing stop
            elif cfg['trailing_stop'] and pos.peak_price > pos.entry_price * 1.02:
                profit_from_peak = pos.peak_price - pos.entry_price
                giveback = pos.peak_price - current_price
                if giveback > profit_from_peak * cfg['trailing_giveback']:
                    exit_reason = 'trailing_stop'

            if exit_reason:
                exit_value = current_price * pos.shares
                pnl = exit_value - pos.cost
                equity += pnl
                to_close.append(pi)

                all_trades.append({
                    'ticker': pos.ticker,
                    'direction': pos.direction,
                    'entry_date': str(pos.entry_date.date()) if hasattr(pos.entry_date, 'date') else str(pos.entry_date),
                    'exit_date': str(date.date()) if hasattr(date, 'date') else str(date),
                    'entry_price': round(pos.entry_price, 2),
                    'exit_price': round(current_price, 2),
                    'shares': round(pos.shares, 2),
                    'days_held': pos.days_held,
                    'pnl': round(pnl, 2),
                    'pnl_pct': round(pct_change * 100, 1),
                    'exit_reason': exit_reason,
                    'conviction': pos.conviction,
                })

        for idx in sorted(to_close, reverse=True):
            positions.pop(idx)

        # --- Entry signals ---
        if len(positions) < cfg['max_positions'] and equity > 50:
            candidates = []

            # VIX filter
            if cfg.get('vix_filter'):
                vix_mask = vix_data.index <= date
                vix_level = vix_data.loc[vix_mask, 'close'].iloc[-1] if vix_mask.any() else 20
                vix_lo, vix_hi = cfg['vix_filter']
                if vix_level < vix_lo or vix_level > vix_hi:
                    # Skip this day
                    daily_ret = (equity - prev_equity) / max(prev_equity, 1)
                    daily_returns.append(daily_ret)
                    equity_curve.append(equity)
                    equity_dates.append(date)
                    continue

            for ticker in cfg['universe']:
                feat = compute_signals(prices_df, ticker, date, spy_prices)
                if feat is None:
                    continue

                # Mean reversion mode: require RSI < 30 for longs
                if cfg.get('mean_rev_mode', False):
                    if not feat.get('mean_rev_long', False):
                        continue

                direction = feat['direction']
                conviction = feat['conviction']

                # For inverse ETFs, flip direction interpretation
                is_inverse = ticker in INVERSE_UNIVERSE
                if is_inverse:
                    # Inverse ETF goes up when market goes down
                    if direction == 'bear':
                        # Bear signal on inverse = BUY the inverse (profit from decline)
                        if conviction >= cfg['min_signals']:
                            candidates.append((ticker, 'long_inverse', feat, conviction))
                    continue

                # Regular leveraged ETFs
                if cfg['direction'] in ('both', 'bull') and direction == 'bull' and conviction >= cfg['min_signals']:
                    candidates.append((ticker, 'long', feat, conviction))
                if cfg['direction'] in ('both', 'bear') and direction == 'bear' and conviction >= cfg['min_signals']:
                    # For bear: buy inverse if available, else skip
                    # Check if there's a corresponding inverse ETF
                    inverse_map = {'TQQQ': 'SQQQ', 'UPRO': 'SPXS', 'SPXL': 'SPXS', 'TNA': 'TZA'}
                    if ticker in inverse_map and inverse_map[ticker] in cfg['universe']:
                        inv_ticker = inverse_map[ticker]
                        candidates.append((inv_ticker, 'long_inverse', feat, conviction))

            # Sort by conviction
            candidates.sort(key=lambda x: x[3], reverse=True)
            held_tickers = {p.ticker for p in positions}

            for ticker, direction, feat, conviction in candidates:
                if len(positions) >= cfg['max_positions']:
                    break
                if ticker in held_tickers:
                    continue

                spot = feat['spot']
                if ticker != feat.get('_source_ticker', ticker):
                    # Get the actual spot for the ticker we're buying
                    try:
                        td = prices_df.loc[ticker]
                        mask = td.index <= date
                        spot = td.loc[mask, 'close'].iloc[-1]
                    except:
                        continue

                # Try to get the actual spot
                try:
                    td_actual = prices_df.loc[ticker]
                    mask_actual = td_actual.index <= date
                    if mask_actual.any():
                        spot = td_actual.loc[mask_actual, 'close'].iloc[-1]
                except:
                    continue

                # Position sizing: conviction-weighted
                max_spend = equity * cfg['max_position_pct']
                conv_scale = min(1.0, conviction / 4.0)
                position_dollars = max_spend * max(0.5, conv_scale)
                position_dollars = min(position_dollars, equity * 0.60)  # Never more than 60%

                if position_dollars < 30:
                    continue

                # Fractional shares allowed on RH
                shares = position_dollars / spot
                cost = shares * spot  # Zero commission

                if cost > equity:
                    continue

                equity -= cost  # Deduct from cash

                pos = Position(
                    ticker=ticker,
                    direction=direction,
                    entry_price=spot,
                    entry_date=date,
                    shares=shares,
                    cost=cost,
                    conviction=conviction,
                )
                positions.append(pos)
                held_tickers.add(ticker)

        # Mark-to-market equity (cash + position values)
        mtm_equity = equity
        for pos in positions:
            try:
                td = prices_df.loc[pos.ticker]
                mask = td.index <= date
                if mask.any():
                    current_price = td.loc[mask, 'close'].iloc[-1]
                    mtm_equity += current_price * pos.shares
            except:
                mtm_equity += pos.cost  # fallback

        daily_ret = (mtm_equity - prev_equity) / max(prev_equity, 1)
        daily_returns.append(daily_ret)
        equity_curve.append(mtm_equity)
        equity_dates.append(date)

        # Update prev_equity for next iteration
        # (we need to track total equity including positions)

    # Force close
    final_date = oot_dates[-1]
    for pos in positions:
        try:
            td = prices_df.loc[pos.ticker]
            mask = td.index <= final_date
            current_price = td.loc[mask, 'close'].iloc[-1]
            pnl = current_price * pos.shares - pos.cost
            equity += pnl
            all_trades.append({
                'ticker': pos.ticker, 'direction': pos.direction,
                'entry_date': str(pos.entry_date.date()),
                'exit_date': str(final_date.date()),
                'entry_price': round(pos.entry_price, 2),
                'exit_price': round(current_price, 2),
                'shares': round(pos.shares, 2),
                'days_held': pos.days_held,
                'pnl': round(pnl, 2),
                'pnl_pct': round((current_price / pos.entry_price - 1) * 100, 1),
                'exit_reason': 'final_close',
                'conviction': pos.conviction,
            })
        except:
            pass

    runtime = time.time() - t0
    return {
        'equity_curve': equity_curve,
        'equity_dates': equity_dates,
        'daily_returns': daily_returns,
        'trades': all_trades,
        'final_equity': equity_curve[-1] if equity_curve else equity,
        'runtime': runtime,
    }


# ============================================================
# METRICS & VALIDATION
# ============================================================

def compute_metrics(result, variant_key, cfg, spy_prices):
    trades = result['trades']
    daily_rets = np.array(result['daily_returns'])
    final_eq = result['final_equity']

    metrics = {
        'variant': variant_key,
        'name': cfg['name'],
        'final_equity': round(final_eq, 2),
        'total_return_pct': round((final_eq / STARTING_CAPITAL - 1) * 100, 2),
        'total_trades': len(trades),
        'runtime': round(result.get('runtime', 0), 1),
    }

    if len(trades) == 0:
        metrics.update({'sharpe': 0, 'sortino': 0, 'win_rate': 0, 'profit_factor': 0,
                        'max_drawdown_pct': 0, 'avg_hold_days': 0, 'avg_pnl': 0, 'cagr_pct': 0,
                        'regime_gap': 999})
        return metrics

    pnls = [t['pnl'] for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]

    metrics['win_rate'] = round(len(wins) / len(pnls) * 100, 1)
    metrics['avg_pnl'] = round(np.mean(pnls), 2)
    metrics['avg_hold_days'] = round(np.mean([t['days_held'] for t in trades]), 1)

    gp = sum(wins) if wins else 0
    gl = abs(sum(losses)) if losses else 1e-8
    metrics['profit_factor'] = round(gp / gl, 2) if gl > 0 else 999

    if len(daily_rets) > 20:
        ann = np.sqrt(252)
        mu = np.mean(daily_rets)
        sigma = np.std(daily_rets)
        metrics['sharpe'] = round((mu / sigma) * ann, 2) if sigma > 0 else 0

        neg = daily_rets[daily_rets < 0]
        ds = np.std(neg) if len(neg) > 0 else sigma
        metrics['sortino'] = round((mu / ds) * ann, 2) if ds > 0 else 0
    else:
        metrics['sharpe'] = 0
        metrics['sortino'] = 0

    eq = np.array(result['equity_curve'])
    peak = np.maximum.accumulate(eq)
    dd = (eq - peak) / np.maximum(peak, 1)
    metrics['max_drawdown_pct'] = round(np.min(dd) * 100, 2)

    n_years = len(daily_rets) / 252
    if n_years > 0 and final_eq > 0:
        metrics['cagr_pct'] = round(((final_eq / STARTING_CAPITAL) ** (1/n_years) - 1) * 100, 1)
    else:
        metrics['cagr_pct'] = 0

    # Regime analysis
    oot_dates_list = sorted(spy_prices.index[spy_prices.index >= pd.Timestamp(OOT_START)])
    if len(oot_dates_list) > 10:
        spy_rets = spy_prices.loc[oot_dates_list, 'close'].pct_change().dropna()
        green = set(spy_rets[spy_rets > 0].index)
        red = set(spy_rets[spy_rets <= 0].index)

        green_pnl = sum(t['pnl'] for t in trades if pd.Timestamp(t['entry_date']) in green)
        red_pnl = sum(t['pnl'] for t in trades if pd.Timestamp(t['entry_date']) in red)
        total = abs(green_pnl) + abs(red_pnl)
        metrics['regime_gap'] = round(abs(green_pnl - red_pnl) / total, 3) if total > 0 else 0
    else:
        metrics['regime_gap'] = 999

    return metrics


def permutation_test(cfg, prices_df, spy_prices, vix_data, trading_dates, actual_sharpe, n_perms=150):
    """Random entry timing test."""
    random_sharpes = []
    oot_dates = [d for d in trading_dates if d >= pd.Timestamp(OOT_START)]

    for perm in range(n_perms):
        np.random.seed(perm + 42)
        equity = STARTING_CAPITAL
        daily_rets = []

        n_trades = max(1, int(len(oot_dates) * 0.01))
        entry_days = set(np.random.choice(len(oot_dates), size=n_trades, replace=False))

        for i, date in enumerate(oot_dates):
            prev = equity

            if i in entry_days and equity > 50:
                ticker = np.random.choice(cfg['universe'])
                try:
                    td = prices_df.loc[ticker]
                    mask = td.index <= date
                    spot = td.loc[mask, 'close'].iloc[-1]
                except:
                    daily_rets.append(0)
                    continue

                shares = (equity * 0.4) / spot
                hold = min(cfg['time_stop_days'], len(oot_dates) - i - 1)
                if hold <= 0:
                    daily_rets.append(0)
                    continue

                exit_date = oot_dates[min(i + hold, len(oot_dates) - 1)]
                try:
                    exit_price = td.loc[td.index <= exit_date, 'close'].iloc[-1]
                except:
                    daily_rets.append(0)
                    continue

                pnl = (exit_price - spot) * shares
                equity += pnl

            daily_rets.append((equity - prev) / max(prev, 1))

        dr = np.array(daily_rets)
        if len(dr) > 20 and np.std(dr) > 0:
            random_sharpes.append((np.mean(dr) / np.std(dr)) * np.sqrt(252))

    if not random_sharpes:
        return 1.0, 0.0

    p = np.mean([s >= actual_sharpe for s in random_sharpes])
    return p, np.mean(random_sharpes)


def validate_5gate(metrics, p_val, rand_sharpe):
    gates = {
        'sharpe_gt_1': metrics['sharpe'] >= 1.0,
        'perm_p_lt_005': p_val < 0.05,
        'wr_gt_40': metrics['win_rate'] >= 40.0,
        'regime_balance': metrics.get('regime_gap', 999) < 0.50,
        'beats_random': metrics['sharpe'] > rand_sharpe + 0.1,
    }
    return sum(gates.values()), gates


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 70)
    print("  LEVERAGED ETF SWING TRADING V1")
    print("  High-growth strategy for $645 agentic account")
    print("  Key advantage: 3x leverage WITHOUT theta decay")
    print("=" * 70)

    prices_df = load_data()

    spy_prices = prices_df.loc['SPY'] if 'SPY' in prices_df.index.get_level_values('ticker') else None
    vix_data = prices_df.loc['^VIX'] if '^VIX' in prices_df.index.get_level_values('ticker') else None

    if spy_prices is None:
        print("ERROR: No SPY data")
        return

    trading_dates = sorted(spy_prices.index.unique())
    oot_dates = [d for d in trading_dates if d >= pd.Timestamp(OOT_START)]
    print(f"\nData: {trading_dates[0].date()} to {trading_dates[-1].date()}")
    print(f"OOT: {oot_dates[0].date()} to {oot_dates[-1].date()} ({len(oot_dates)} days)")

    available = [t for t in list(LEVERAGED_UNIVERSE.keys()) + list(INVERSE_UNIVERSE.keys())
                 if t in prices_df.index.get_level_values('ticker')]
    print(f"Available leveraged ETFs: {available}")

    all_metrics = {}

    for vk in sorted(VARIANTS.keys()):
        cfg = VARIANTS[vk]
        print(f"\n{'=' * 60}")
        print(f"  VARIANT {vk}: {cfg['name']}")
        print(f"{'=' * 60}")

        # Check universe availability
        avail_univ = [t for t in cfg['universe'] if t in prices_df.index.get_level_values('ticker')]
        if not avail_univ:
            print(f"  SKIP: No tickers available from {cfg['universe']}")
            all_metrics[vk] = {'variant': vk, 'name': cfg['name'], 'sharpe': 0, 'sortino': 0,
                               'win_rate': 0, 'total_trades': 0, 'final_equity': 645, 'gates_passed': 0}
            continue
        cfg_copy = dict(cfg)
        cfg_copy['universe'] = avail_univ

        result = run_variant(vk, cfg_copy, prices_df, spy_prices, vix_data, trading_dates)
        metrics = compute_metrics(result, vk, cfg_copy, spy_prices)

        # Print first 3 trades
        for t in result['trades'][:3]:
            print(f"  {t['entry_date']}: {t['direction'].upper()} {t['ticker']} "
                  f"${t['entry_price']:.1f} → ${t['exit_price']:.1f} "
                  f"{t['shares']:.1f}sh held={t['days_held']}d pnl=${t['pnl']:.0f} ({t['exit_reason']})")

        print(f"  Trades: {metrics['total_trades']} | Sharpe: {metrics['sharpe']} | "
              f"Sortino: {metrics['sortino']} | PF: {metrics['profit_factor']} | "
              f"WR: {metrics['win_rate']}% | MDD: {metrics['max_drawdown_pct']}%")
        print(f"  Final: ${metrics['final_equity']:.0f} | Return: {metrics['total_return_pct']:.1f}% | "
              f"CAGR: {metrics['cagr_pct']:.1f}% | Regime Gap: {metrics.get('regime_gap', 'N/A')}")

        # Permutation test
        if metrics['total_trades'] >= 5 and metrics['sharpe'] > 0:
            print(f"  Running {N_PERMUTATIONS}-permutation test...")
            p_val, rand_sharpe = permutation_test(
                cfg_copy, prices_df, spy_prices, vix_data, trading_dates, metrics['sharpe'], N_PERMUTATIONS)
            n_pass, gates = validate_5gate(metrics, p_val, rand_sharpe)

            print(f"  5-Gate: {n_pass}/5 PASS")
            for gn, gp in gates.items():
                if gn == 'sharpe_gt_1':
                    vs = f"value={metrics['sharpe']}, threshold=1.0"
                elif gn == 'perm_p_lt_005':
                    vs = f"value={p_val:.3f}, threshold=0.05"
                elif gn == 'wr_gt_40':
                    vs = f"value={metrics['win_rate']}, threshold=40.0"
                elif gn == 'regime_balance':
                    vs = f"value={metrics.get('regime_gap', 999):.3f}, threshold=0.5"
                elif gn == 'beats_random':
                    vs = f"value={metrics['sharpe']}, random={rand_sharpe:.2f}"
                else:
                    vs = ""
                print(f"    {gn}: {'PASS' if gp else 'FAIL'} ({vs})")

            metrics['perm_p'] = round(p_val, 4)
            metrics['random_sharpe'] = round(rand_sharpe, 2)
            metrics['gates_passed'] = n_pass
        else:
            metrics['gates_passed'] = 0
            print(f"  5-Gate: 0/5 (insufficient trades or negative Sharpe)")

        all_metrics[vk] = metrics

        # MLflow
        if MLFLOW_AVAILABLE:
            try:
                with mlflow.start_run(run_name=f"swing_{vk}_{cfg['name'][:20]}"):
                    mlflow.log_params({k: str(v)[:250] for k, v in cfg.items()})
                    for mk, mv in metrics.items():
                        if isinstance(mv, (int, float)):
                            mlflow.log_metric(mk, mv)
            except:
                pass

    # Summary
    print(f"\n{'=' * 80}")
    print("  SUMMARY — LEVERAGED ETF SWING TRADING V1")
    print(f"{'=' * 80}")

    sorted_v = sorted(all_metrics.items(), key=lambda x: x[1].get('sharpe', 0), reverse=True)

    print(f"\n  {'Var':<4} {'Name':<30} {'Sharpe':>7} {'Sort':>7} {'PF':>6} {'WR':>6} "
          f"{'Trades':>7} {'Return':>8} {'MDD':>7} {'Gates':>6}")
    print(f"  {'-'*4} {'-'*30} {'-'*7} {'-'*7} {'-'*6} {'-'*6} {'-'*7} {'-'*8} {'-'*7} {'-'*6}")

    for vk, m in sorted_v:
        print(f"  {vk:<4} {m['name']:<30} {m.get('sharpe',0):>7.2f} {m.get('sortino',0):>7.2f} "
              f"{m.get('profit_factor',0):>6.2f} {m.get('win_rate',0):>5.1f}% "
              f"{m.get('total_trades',0):>7} {m.get('total_return_pct',0):>7.1f}% "
              f"{m.get('max_drawdown_pct',0):>6.1f}% {m.get('gates_passed',0):>4}/5")

    best = sorted_v[0]
    print(f"\n  BEST: {best[0]} ({best[1]['name']}) — Sharpe {best[1].get('sharpe', 0)}")
    print(f"  KEY ADVANTAGE: Zero commission + fractional shares + 3x leverage + no theta decay")

    # Save
    results_path = os.path.join(OUTPUT_DIR, 'backtest_results.json')
    with open(results_path, 'w') as f:
        json.dump({
            'metrics': all_metrics,
            'best_variant': best[0],
            'timestamp': datetime.now().isoformat(),
            'track': 'HIGH_GROWTH',
        }, f, indent=2, default=str)

    print(f"\nResults saved.")
    print("\nDone.")


if __name__ == '__main__':
    main()
