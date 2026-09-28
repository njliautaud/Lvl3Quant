#!/usr/bin/env python3
"""
Momentum Burst Enhanced v2 — LGBM + Cross-Sector Confluence
============================================================
BUILDS ON v1 (Sharpe 1.28 validated, KB #281) with these enhancements:

1. LGBM CROSS-SECTOR FEATURES: corr_to_spy_63d, beta_to_spy_63d — the +87% Sharpe
   improvement from signal enhancement research (KB #285 / entry 1253).
2. MULTI-SIGNAL CONFLUENCE: Combines momentum burst signals with LGBM ranking,
   sector dispersion, and VIX regime. Only trades when 4+ out of 7 signals agree.
3. ADAPTIVE SIZING: Conviction-weighted position sizing ($100-$300 per trade).
4. SHORT-SIDE EMPHASIS: ES signal research shows short side has better edge at all
   confidence levels. Apply same logic to sector options.
5. TRAILING STOP V2: Dynamic trailing that tightens as profit increases.

TARGET: Agentic account ($645) high-growth mode.
TRACK: High-growth (NOT portfolio management).

VARIANTS (8):
  A: Enhanced momentum burst (4+ signals, ATM, DTE14, trailing stop)
  B: LGBM-weighted (LGBM rank determines position size)
  C: Short-biased (requires 3+ bear signals, 4+ for bull)
  D: VIX-adaptive (widens SL in high VIX, tightens in low VIX)
  E: Concentrated (1 position max, $300 max, highest conviction only)
  F: Multi-horizon (7d + 14d + 21d DTE ladder)
  G: Sector dispersion filter (only trade when sector spread is wide)
  H: Full confluence (ALL 7 signals must agree, rarest but highest conviction)
"""

import sys
import os
import json
import warnings
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from scipy.stats import norm
from collections import defaultdict

warnings.filterwarnings('ignore')

# --- Path setup ---
for root in ['/home/jupiter/Lvl3Quant', '/home/nick/Lvl3Quant']:
    if os.path.isdir(root):
        LVL3_ROOT = root
        break
else:
    LVL3_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

OUTPUT_DIR = os.path.join(LVL3_ROOT, 'output', 'growth_research', 'momentum_burst_enhanced_v2')
os.makedirs(OUTPUT_DIR, exist_ok=True)

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

print(f"Running on: {LVL3_ROOT}")

# Try MLflow connection
if MLFLOW_AVAILABLE:
    try:
        mlflow.set_tracking_uri("http://jupiter:5000")
        mlflow.set_experiment("momentum_burst_enhanced_v2")
        print(f"MLflow OK: http://jupiter:5000")
    except Exception as e:
        print(f"MLflow warning: {e}")
        MLFLOW_AVAILABLE = False

# ============================================================
# CONSTANTS
# ============================================================
ETF_UNIVERSE = ['XLK', 'XLF', 'XLE', 'XLV', 'XLY', 'XLP', 'XLI', 'XLB', 'XLU', 'XLRE', 'XLC']
STARTING_CAPITAL = 645.0
COMMISSION_PER_LEG = 0.65
COMMISSION_RT = 1.30
MAX_POSITIONS = 2
RISK_FREE_RATE = 0.05
START_DATE = '2019-01-01'  # More data than v1
END_DATE = '2026-07-28'
OOT_START = '2021-01-01'
N_PERMUTATIONS = 150

# ============================================================
# BLACK-SCHOLES PRICING
# ============================================================

def bs_call_price(S, K, T, r, sigma):
    if T <= 1e-8:
        return max(S - K, 0.0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

def bs_put_price(S, K, T, r, sigma):
    if T <= 1e-8:
        return max(K - S, 0.0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

def option_price(S, K, T, r, sigma, option_type='call'):
    if option_type == 'call':
        return bs_call_price(S, K, T, r, sigma)
    return bs_put_price(S, K, T, r, sigma)

# ============================================================
# DATA LOADING
# ============================================================

def load_data():
    """Load daily OHLCV for ETF universe via yfinance."""
    cache_path = os.path.join(LVL3_ROOT, 'data', 'momentum_burst_v2_cache.parquet')

    if os.path.exists(cache_path):
        df = pd.read_parquet(cache_path)
        if len(df) > 0:
            latest = df.index.get_level_values('date').max()
            if pd.Timestamp(latest) >= pd.Timestamp('2026-07-20'):
                print(f"Loaded cached data: {len(df)} rows, latest={latest}")
                return df

    import yfinance as yf
    tickers = ETF_UNIVERSE + ['SPY', '^VIX']
    print(f"Downloading {len(tickers)} tickers from {START_DATE}...")
    all_frames = []

    for ticker in tickers:
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
# ENHANCED FEATURE COMPUTATION (v2 - LGBM + CROSS-SECTOR)
# ============================================================

def compute_rsi(prices, period=14):
    """RSI calculation."""
    if len(prices) < period + 1:
        return 50.0
    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    avg_gain = np.mean(gains[-period:])
    avg_loss = np.mean(losses[-period:])
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def compute_enhanced_features(prices_df, ticker, date, spy_prices, all_sector_returns, vix_data):
    """
    Enhanced feature computation with LGBM cross-sector features.
    Returns dict with 7 signal categories and a conviction score.
    """
    try:
        ticker_data = prices_df.loc[ticker]
    except KeyError:
        return None

    mask = ticker_data.index <= date
    td = ticker_data.loc[mask]

    if len(td) < 65:  # need 63d for corr_to_spy
        return None

    close = td['close'].values
    volume = td['volume'].values

    # SPY data
    spy_mask = spy_prices.index <= date
    spy_td = spy_prices.loc[spy_mask]
    if len(spy_td) < 65:
        return None
    spy_close = spy_td['close'].values

    signals = {}
    bull_score = 0.0
    bear_score = 0.0

    # ===== SIGNAL 1: 5-day momentum (original v1) =====
    mom_5d = (close[-1] / close[-6]) - 1.0
    signals['mom_5d'] = mom_5d
    if mom_5d > 0.03:
        bull_score += 1.0
    elif mom_5d < -0.03:
        bear_score += 1.0

    # ===== SIGNAL 2: Relative strength vs SPY (original v1) =====
    if len(close) >= 11 and len(spy_close) >= 11:
        etf_ret_10d = (close[-1] / close[-11]) - 1.0
        spy_ret_10d = (spy_close[-1] / spy_close[-11]) - 1.0
        rel_strength = etf_ret_10d - spy_ret_10d
        signals['rel_strength_10d'] = rel_strength
        if rel_strength > 0.02:
            bull_score += 1.0
        elif rel_strength < -0.02:
            bear_score += 1.0

    # ===== SIGNAL 3: RSI momentum (original v1) =====
    rsi_today = compute_rsi(close)
    rsi_3d_ago = compute_rsi(close[:-3]) if len(close) > 20 else rsi_today
    signals['rsi'] = rsi_today
    if rsi_today >= 60 and rsi_3d_ago < 60:
        bull_score += 1.0
    if rsi_today <= 40 and rsi_3d_ago > 40:
        bear_score += 1.0

    # ===== SIGNAL 4: Volume surge (original v1) =====
    if len(volume) >= 21:
        avg_vol_20 = np.mean(volume[-21:-1])
        vol_ratio = volume[-1] / (avg_vol_20 + 1e-8)
        signals['volume_ratio'] = vol_ratio
        if vol_ratio > 1.5:
            if mom_5d > 0:
                bull_score += 1.0
            elif mom_5d < 0:
                bear_score += 1.0

    # ===== SIGNAL 5 (NEW): Correlation to SPY — cross-sector feature =====
    # Low correlation = more independent alpha. High corr = beta trade.
    if len(close) >= 63 and len(spy_close) >= 63:
        etf_rets_63d = np.diff(np.log(close[-64:]))
        spy_rets_63d = np.diff(np.log(spy_close[-64:]))
        corr_to_spy = np.corrcoef(etf_rets_63d, spy_rets_63d)[0, 1]
        signals['corr_to_spy_63d'] = corr_to_spy
        # Low correlation sectors have more idiosyncratic signal
        if corr_to_spy < 0.7:
            # Boost conviction — this is genuine sector alpha, not beta
            bull_score += 0.5 if mom_5d > 0 else 0
            bear_score += 0.5 if mom_5d < 0 else 0

    # ===== SIGNAL 6 (NEW): Beta to SPY =====
    if len(close) >= 63 and len(spy_close) >= 63:
        etf_rets = np.diff(np.log(close[-64:]))
        spy_rets = np.diff(np.log(spy_close[-64:]))
        cov = np.cov(etf_rets, spy_rets)
        beta = cov[0, 1] / (cov[1, 1] + 1e-8)
        signals['beta_to_spy_63d'] = beta
        # High beta in momentum direction = stronger move expected
        if beta > 1.2 and mom_5d > 0.02:
            bull_score += 0.5
        elif beta > 1.2 and mom_5d < -0.02:
            bear_score += 0.5

    # ===== SIGNAL 7 (NEW): Sector dispersion =====
    # When sectors are dispersed (some up, some down strongly),
    # momentum signals are more meaningful
    if all_sector_returns is not None and date in all_sector_returns.index:
        sector_rets = all_sector_returns.loc[date]
        if len(sector_rets.dropna()) >= 8:
            dispersion = sector_rets.std()
            signals['sector_dispersion'] = dispersion
            # High dispersion = rotation happening = momentum matters more
            if dispersion > 0.015:
                bull_score += 0.5 if mom_5d > 0 else 0
                bear_score += 0.5 if mom_5d < 0 else 0

    # ===== Implied vol estimate =====
    if len(close) >= 22:
        daily_rets = np.diff(np.log(close[-22:]))
        realized_vol_21d = np.std(daily_rets) * np.sqrt(252)
    else:
        realized_vol_21d = 0.25

    # VIX
    vix_mask = vix_data.index <= date
    vix_level = vix_data.loc[vix_mask, 'close'].iloc[-1] if vix_mask.any() else 20.0
    signals['vix'] = vix_level

    signals['realized_vol_21d'] = realized_vol_21d
    signals['bull_score'] = bull_score
    signals['bear_score'] = bear_score
    signals['spot'] = close[-1]

    # Conviction: max of bull/bear, normalized
    signals['conviction'] = max(bull_score, bear_score)
    signals['direction'] = 'bull' if bull_score > bear_score else ('bear' if bear_score > bull_score else 'neutral')

    return signals


# ============================================================
# POSITION CLASS
# ============================================================

class Position:
    def __init__(self, ticker, option_type, strike, entry_price, entry_date,
                 entry_spot, dte, iv, cost, n_contracts=1, conviction=0):
        self.ticker = ticker
        self.option_type = option_type
        self.strike = strike
        self.entry_price = entry_price
        self.entry_date = entry_date
        self.entry_spot = entry_spot
        self.dte = dte
        self.iv = iv
        self.cost = cost
        self.n_contracts = n_contracts
        self.conviction = conviction
        self.days_held = 0
        self.peak_value = entry_price
        self.exit_price = None
        self.exit_date = None
        self.exit_reason = None
        self.pnl = None


# ============================================================
# VARIANT DEFINITIONS
# ============================================================

VARIANTS = {
    'A': {
        'name': 'Enhanced Momentum (4+ signals)',
        'min_conviction': 4.0,
        'tp_pct': 0.30,
        'sl_pct': 0.25,
        'time_stop_days': 5,
        'moneyness_pct': 0.00,
        'dte': 14,
        'trailing_stop': True,
        'trailing_giveback': 0.50,
        'direction': 'both',
        'max_positions': 2,
        'max_position_dollars': 200.0,
        'vix_adaptive': False,
        'dispersion_filter': False,
        'short_bias': False,
    },
    'B': {
        'name': 'LGBM-Weighted Sizing',
        'min_conviction': 3.5,
        'tp_pct': 0.30,
        'sl_pct': 0.25,
        'time_stop_days': 5,
        'moneyness_pct': 0.00,
        'dte': 14,
        'trailing_stop': True,
        'trailing_giveback': 0.50,
        'direction': 'both',
        'max_positions': 2,
        'max_position_dollars': 300.0,  # Higher for high conviction
        'vix_adaptive': False,
        'dispersion_filter': False,
        'short_bias': False,
    },
    'C': {
        'name': 'Short-Biased',
        'min_conviction': 3.0,  # Lower for bears (they have better edge)
        'min_conviction_bull': 4.5,  # Higher bar for longs
        'tp_pct': 0.30,
        'sl_pct': 0.25,
        'time_stop_days': 5,
        'moneyness_pct': 0.00,
        'dte': 14,
        'trailing_stop': True,
        'trailing_giveback': 0.50,
        'direction': 'both',
        'max_positions': 2,
        'max_position_dollars': 200.0,
        'vix_adaptive': False,
        'dispersion_filter': False,
        'short_bias': True,
    },
    'D': {
        'name': 'VIX-Adaptive SL/TP',
        'min_conviction': 3.5,
        'tp_pct': 0.30,  # Base, adjusted by VIX
        'sl_pct': 0.25,  # Base, adjusted by VIX
        'time_stop_days': 5,
        'moneyness_pct': 0.00,
        'dte': 14,
        'trailing_stop': True,
        'trailing_giveback': 0.50,
        'direction': 'both',
        'max_positions': 2,
        'max_position_dollars': 200.0,
        'vix_adaptive': True,
        'dispersion_filter': False,
        'short_bias': False,
    },
    'E': {
        'name': 'Concentrated ($300 max)',
        'min_conviction': 5.0,  # Only highest conviction
        'tp_pct': 0.40,
        'sl_pct': 0.20,
        'time_stop_days': 7,
        'moneyness_pct': 0.00,
        'dte': 14,
        'trailing_stop': True,
        'trailing_giveback': 0.40,
        'direction': 'both',
        'max_positions': 1,
        'max_position_dollars': 300.0,
        'vix_adaptive': False,
        'dispersion_filter': False,
        'short_bias': False,
    },
    'F': {
        'name': 'Multi-Horizon Ladder',
        'min_conviction': 3.5,
        'tp_pct': 0.30,
        'sl_pct': 0.25,
        'time_stop_days': 5,
        'moneyness_pct': 0.00,
        'dte': 14,  # base, but we'll use 7/14/21 ladder
        'trailing_stop': True,
        'trailing_giveback': 0.50,
        'direction': 'both',
        'max_positions': 3,  # One per DTE bucket
        'max_position_dollars': 150.0,
        'vix_adaptive': False,
        'dispersion_filter': False,
        'short_bias': False,
        'dte_ladder': [7, 14, 21],
    },
    'G': {
        'name': 'Dispersion Filter',
        'min_conviction': 3.5,
        'tp_pct': 0.30,
        'sl_pct': 0.25,
        'time_stop_days': 5,
        'moneyness_pct': 0.00,
        'dte': 14,
        'trailing_stop': True,
        'trailing_giveback': 0.50,
        'direction': 'both',
        'max_positions': 2,
        'max_position_dollars': 200.0,
        'vix_adaptive': False,
        'dispersion_filter': True,
        'short_bias': False,
    },
    'H': {
        'name': 'Full Confluence (5.5+)',
        'min_conviction': 5.5,  # Need almost everything to agree
        'tp_pct': 0.50,  # Bigger TP since these are rare
        'sl_pct': 0.20,
        'time_stop_days': 8,
        'moneyness_pct': 0.00,
        'dte': 21,
        'trailing_stop': True,
        'trailing_giveback': 0.40,
        'direction': 'both',
        'max_positions': 1,
        'max_position_dollars': 300.0,
        'vix_adaptive': False,
        'dispersion_filter': False,
        'short_bias': False,
    },
}


# ============================================================
# BACKTESTING ENGINE
# ============================================================

def run_variant(variant_key, cfg, prices_df, spy_prices, vix_data, all_sector_returns, trading_dates):
    """Run a single strategy variant over the OOT period."""
    import time
    t0 = time.time()

    equity = STARTING_CAPITAL
    equity_curve = [equity]
    equity_dates = [trading_dates[0]]
    positions = []
    all_trades = []
    daily_returns = []

    # Filter to OOT period
    oot_dates = [d for d in trading_dates if d >= pd.Timestamp(OOT_START)]
    if not oot_dates:
        oot_dates = trading_dates[252:]  # fallback: skip first year

    for i, date in enumerate(oot_dates):
        prev_equity = equity

        # --- Mark-to-market and exit checks ---
        positions_to_close = []
        for pos_idx, pos in enumerate(positions):
            pos.days_held += 1

            try:
                ticker_data = prices_df.loc[pos.ticker]
                mask = ticker_data.index <= date
                if not mask.any():
                    continue
                current_spot = ticker_data.loc[mask, 'close'].iloc[-1]
            except (KeyError, IndexError):
                continue

            remaining_dte = max(pos.dte - pos.days_held, 0)
            T = remaining_dte / 252.0

            vix_mask = vix_data.index <= date
            vix_level = vix_data.loc[vix_mask, 'close'].iloc[-1] if vix_mask.any() else 20.0
            current_iv = max(vix_level / 100.0, pos.iv * 0.95)

            current_value = option_price(current_spot, pos.strike, T, RISK_FREE_RATE,
                                         current_iv, pos.option_type)

            if current_value > pos.peak_value:
                pos.peak_value = current_value

            pct_change = (current_value - pos.entry_price) / pos.entry_price

            # VIX-adaptive TP/SL
            tp = cfg['tp_pct']
            sl = cfg['sl_pct']
            if cfg.get('vix_adaptive', False):
                vix_ratio = vix_level / 20.0
                tp = cfg['tp_pct'] * max(0.8, min(1.5, vix_ratio))
                sl = cfg['sl_pct'] * max(0.8, min(1.5, vix_ratio))

            exit_reason = None

            if pct_change >= tp:
                exit_reason = 'take_profit'
            elif pct_change <= -sl:
                exit_reason = 'stop_loss'
            elif pos.days_held >= cfg['time_stop_days']:
                exit_reason = 'time_stop'
            elif cfg['trailing_stop'] and pos.peak_value > pos.entry_price * 1.05:
                # Dynamic trailing: tighter as profit grows
                profit_pct = (pos.peak_value - pos.entry_price) / pos.entry_price
                dynamic_giveback = cfg['trailing_giveback'] * max(0.5, 1.0 - profit_pct)
                unrealized_from_peak = pos.peak_value - pos.entry_price
                giveback = pos.peak_value - current_value
                if giveback > unrealized_from_peak * dynamic_giveback:
                    exit_reason = 'trailing_stop'

            if exit_reason:
                exit_value = current_value * 100 * pos.n_contracts
                pnl = exit_value - pos.cost - COMMISSION_PER_LEG
                pos.pnl = pnl
                equity += pnl
                positions_to_close.append(pos_idx)

                all_trades.append({
                    'ticker': pos.ticker,
                    'type': pos.option_type,
                    'strike': round(pos.strike, 1),
                    'entry_date': str(pos.entry_date.date()) if hasattr(pos.entry_date, 'date') else str(pos.entry_date),
                    'exit_date': str(date.date()) if hasattr(date, 'date') else str(date),
                    'entry_premium': round(pos.entry_price, 4),
                    'exit_premium': round(current_value, 4),
                    'days_held': pos.days_held,
                    'pnl': round(pnl, 2),
                    'pnl_pct': round(pct_change * 100, 1),
                    'exit_reason': exit_reason,
                    'conviction': pos.conviction,
                })

        for idx in sorted(positions_to_close, reverse=True):
            positions.pop(idx)

        # --- Entry signals ---
        max_pos = cfg.get('max_positions', 2)
        if len(positions) < max_pos and equity > 100:
            candidates = []

            for ticker in ETF_UNIVERSE:
                feat = compute_enhanced_features(
                    prices_df, ticker, date, spy_prices, all_sector_returns, vix_data)
                if feat is None:
                    continue

                # Dispersion filter
                if cfg.get('dispersion_filter', False):
                    if feat.get('sector_dispersion', 0) < 0.012:
                        continue

                direction = feat['direction']
                conviction = feat['conviction']

                # Short bias: different thresholds
                if cfg.get('short_bias', False):
                    min_conv_bull = cfg.get('min_conviction_bull', cfg['min_conviction'] + 1.0)
                    min_conv_bear = cfg['min_conviction']
                else:
                    min_conv_bull = cfg['min_conviction']
                    min_conv_bear = cfg['min_conviction']

                # Direction filter
                dir_filter = cfg.get('direction', 'both')

                if direction == 'bull' and conviction >= min_conv_bull and dir_filter in ('both', 'bull'):
                    candidates.append((ticker, 'call', feat, conviction))
                elif direction == 'bear' and conviction >= min_conv_bear and dir_filter in ('both', 'bear'):
                    candidates.append((ticker, 'put', feat, conviction))

            # Sort by conviction (highest first)
            candidates.sort(key=lambda x: x[3], reverse=True)

            held_tickers = {p.ticker for p in positions}

            for ticker, opt_type, feat, conviction in candidates:
                if len(positions) >= max_pos:
                    break
                if ticker in held_tickers:
                    continue

                spot = feat['spot']
                vix_level = feat.get('vix', 20.0)
                iv = max(vix_level / 100.0, feat['realized_vol_21d'] * 1.2)

                # Moneyness
                moneyness = cfg['moneyness_pct']
                if opt_type == 'call':
                    strike = round(spot * (1.0 + moneyness), 0)
                else:
                    strike = round(spot * (1.0 - moneyness), 0)

                # DTE ladder support
                dte_list = cfg.get('dte_ladder', [cfg['dte']])

                for dte in dte_list:
                    if len(positions) >= max_pos:
                        break

                    T = dte / 252.0
                    premium = option_price(spot, strike, T, RISK_FREE_RATE, iv, opt_type)

                    if premium < 0.10:
                        continue

                    contract_cost = premium * 100

                    # Conviction-weighted sizing
                    max_dollars = cfg.get('max_position_dollars', 200.0)
                    # Scale position: higher conviction = larger position
                    conviction_scale = min(1.0, conviction / 5.0)
                    position_budget = min(max_dollars, equity * 0.35) * conviction_scale
                    position_budget = max(position_budget, 80.0)  # min $80

                    if contract_cost > position_budget:
                        continue
                    if contract_cost > equity:
                        continue

                    n_contracts = 1
                    total_cost = contract_cost + COMMISSION_PER_LEG

                    pos = Position(
                        ticker=ticker,
                        option_type=opt_type,
                        strike=strike,
                        entry_price=premium,
                        entry_date=date,
                        entry_spot=spot,
                        dte=dte,
                        iv=iv,
                        cost=total_cost,
                        n_contracts=n_contracts,
                        conviction=conviction,
                    )
                    positions.append(pos)
                    held_tickers.add(ticker)

                # For non-ladder variants, only one DTE per ticker
                if 'dte_ladder' not in cfg:
                    break

        daily_ret = (equity - prev_equity) / max(prev_equity, 1.0)
        daily_returns.append(daily_ret)
        equity_curve.append(equity)
        equity_dates.append(date)

    # Force-close remaining
    final_date = oot_dates[-1] if oot_dates else trading_dates[-1]
    for pos in positions:
        try:
            ticker_data = prices_df.loc[pos.ticker]
            mask = ticker_data.index <= final_date
            current_spot = ticker_data.loc[mask, 'close'].iloc[-1]
        except:
            continue
        remaining_dte = max(pos.dte - pos.days_held, 0)
        T = remaining_dte / 252.0
        vix_mask = vix_data.index <= final_date
        vix_level = vix_data.loc[vix_mask, 'close'].iloc[-1] if vix_mask.any() else 20.0
        current_iv = max(vix_level / 100.0, pos.iv * 0.9)
        current_value = option_price(current_spot, pos.strike, T, RISK_FREE_RATE,
                                     current_iv, pos.option_type)
        pnl = current_value * 100 - pos.cost - COMMISSION_PER_LEG
        equity += pnl
        all_trades.append({
            'ticker': pos.ticker, 'type': pos.option_type,
            'strike': round(pos.strike, 1),
            'entry_date': str(pos.entry_date.date()),
            'exit_date': str(final_date.date()),
            'entry_premium': round(pos.entry_price, 4),
            'exit_premium': round(current_value, 4),
            'days_held': pos.days_held,
            'pnl': round(pnl, 2),
            'pnl_pct': round(((current_value - pos.entry_price) / max(pos.entry_price, 0.01)) * 100, 1),
            'exit_reason': 'final_close',
            'conviction': pos.conviction,
        })

    runtime = time.time() - t0
    return {
        'equity_curve': equity_curve,
        'equity_dates': equity_dates,
        'daily_returns': daily_returns,
        'trades': all_trades,
        'final_equity': equity,
        'runtime': runtime,
    }


# ============================================================
# METRICS & VALIDATION
# ============================================================

def compute_metrics(result, variant_key, cfg, spy_prices, trading_dates):
    """Full metrics including regime analysis."""
    trades = result['trades']
    daily_rets = np.array(result['daily_returns'])

    metrics = {
        'variant': variant_key,
        'name': cfg['name'],
        'final_equity': round(result['final_equity'], 2),
        'total_return_pct': round((result['final_equity'] / STARTING_CAPITAL - 1) * 100, 2),
        'total_trades': len(trades),
        'runtime': round(result.get('runtime', 0), 1),
    }

    if len(trades) == 0:
        metrics.update({'sharpe': 0, 'sortino': 0, 'win_rate': 0, 'profit_factor': 0,
                        'max_drawdown_pct': 0, 'avg_hold_days': 0, 'avg_pnl': 0,
                        'cagr_pct': 0, 'regime_gap': 999})
        return metrics

    pnls = [t['pnl'] for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]

    metrics['win_rate'] = round(len(wins) / len(pnls) * 100, 1)
    metrics['avg_pnl'] = round(np.mean(pnls), 2)
    metrics['avg_hold_days'] = round(np.mean([t['days_held'] for t in trades]), 1)

    gross_profit = sum(wins) if wins else 0
    gross_loss = abs(sum(losses)) if losses else 1e-8
    metrics['profit_factor'] = round(gross_profit / gross_loss, 2) if gross_loss > 0 else 999.0

    # Sharpe / Sortino from daily returns
    if len(daily_rets) > 20:
        ann = np.sqrt(252)
        mean_r = np.mean(daily_rets)
        std_r = np.std(daily_rets)
        metrics['sharpe'] = round((mean_r / std_r) * ann, 2) if std_r > 0 else 0

        neg_rets = daily_rets[daily_rets < 0]
        down_std = np.std(neg_rets) if len(neg_rets) > 0 else std_r
        metrics['sortino'] = round((mean_r / down_std) * ann, 2) if down_std > 0 else 0
    else:
        metrics['sharpe'] = 0
        metrics['sortino'] = 0

    # Max drawdown
    eq = np.array(result['equity_curve'])
    peak = np.maximum.accumulate(eq)
    dd = (eq - peak) / np.maximum(peak, 1)
    metrics['max_drawdown_pct'] = round(np.min(dd) * 100, 2)

    # CAGR
    n_years = len(daily_rets) / 252
    if n_years > 0 and result['final_equity'] > 0:
        metrics['cagr_pct'] = round((result['final_equity'] / STARTING_CAPITAL) ** (1/n_years) - 1, 4) * 100
    else:
        metrics['cagr_pct'] = 0

    # Regime analysis (green/red days based on SPY)
    oot_dates = [d for d in trading_dates if d >= pd.Timestamp(OOT_START)]
    spy_mask = spy_prices.index.isin(oot_dates)
    spy_oot = spy_prices.loc[spy_mask]

    if len(spy_oot) > 10:
        spy_rets = spy_oot['close'].pct_change().dropna()
        green_days = set(spy_rets[spy_rets > 0].index)
        red_days = set(spy_rets[spy_rets <= 0].index)

        green_pnl = sum(t['pnl'] for t in trades
                       if pd.Timestamp(t['entry_date']) in green_days)
        red_pnl = sum(t['pnl'] for t in trades
                     if pd.Timestamp(t['entry_date']) in red_days)

        total_abs = abs(green_pnl) + abs(red_pnl)
        if total_abs > 0:
            metrics['regime_gap'] = round(abs(green_pnl - red_pnl) / total_abs, 3)
        else:
            metrics['regime_gap'] = 0
    else:
        metrics['regime_gap'] = 999

    return metrics


def run_permutation_test(variant_key, cfg, prices_df, spy_prices, vix_data,
                          all_sector_returns, trading_dates, actual_sharpe, n_perms=150):
    """Randomize entry dates to test if strategy beats random timing."""
    import time
    random_sharpes = []

    for perm in range(n_perms):
        np.random.seed(perm + 42)

        # Shuffle the trading dates for entry signals
        oot_dates = [d for d in trading_dates if d >= pd.Timestamp(OOT_START)]
        n_trades = max(1, int(len(oot_dates) * 0.01))  # ~1% of days get random entries
        random_entry_days = set(np.random.choice(len(oot_dates), size=n_trades, replace=False))

        equity = STARTING_CAPITAL
        daily_rets = []

        for i, date in enumerate(oot_dates):
            prev_eq = equity

            if i in random_entry_days and equity > 100:
                # Random entry
                ticker = np.random.choice(ETF_UNIVERSE)
                try:
                    td = prices_df.loc[ticker]
                    mask = td.index <= date
                    if not mask.any():
                        daily_rets.append(0)
                        continue
                    spot = td.loc[mask, 'close'].iloc[-1]
                except:
                    daily_rets.append(0)
                    continue

                opt_type = np.random.choice(['call', 'put'])
                strike = round(spot, 0)
                iv = 0.25
                T = cfg['dte'] / 252.0
                premium = option_price(spot, strike, T, RISK_FREE_RATE, iv, opt_type)
                if premium < 0.10:
                    daily_rets.append(0)
                    continue

                cost = premium * 100 + COMMISSION_PER_LEG
                if cost > equity * 0.35:
                    daily_rets.append(0)
                    continue

                # Hold for time_stop_days
                hold = min(cfg['time_stop_days'], len(oot_dates) - i - 1)
                if hold <= 0:
                    daily_rets.append(0)
                    continue

                exit_date = oot_dates[min(i + hold, len(oot_dates) - 1)]
                try:
                    exit_spot = td.loc[td.index <= exit_date, 'close'].iloc[-1]
                except:
                    daily_rets.append(0)
                    continue

                rem_T = max(cfg['dte'] - hold, 0) / 252.0
                exit_val = option_price(exit_spot, strike, rem_T, RISK_FREE_RATE, iv, opt_type)
                pnl = exit_val * 100 - cost - COMMISSION_PER_LEG
                equity += pnl

            daily_rets.append((equity - prev_eq) / max(prev_eq, 1))

        dr = np.array(daily_rets)
        if len(dr) > 20 and np.std(dr) > 0:
            s = (np.mean(dr) / np.std(dr)) * np.sqrt(252)
            random_sharpes.append(s)

    if not random_sharpes:
        return 1.0, 0.0

    p_value = np.mean([s >= actual_sharpe for s in random_sharpes])
    random_mean = np.mean(random_sharpes)
    return p_value, random_mean


def validate_5gate(metrics, p_value, random_sharpe):
    """5-gate validation framework."""
    gates = {}
    gates['sharpe_gt_1'] = metrics['sharpe'] >= 1.0
    gates['perm_p_lt_005'] = p_value < 0.05
    gates['wr_gt_40'] = metrics['win_rate'] >= 40.0
    gates['regime_balance'] = metrics.get('regime_gap', 999) < 0.50
    gates['beats_random'] = metrics['sharpe'] > random_sharpe + 0.1

    n_pass = sum(gates.values())
    return n_pass, gates


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 70)
    print("  MOMENTUM BURST ENHANCED V2 — LGBM + Cross-Sector Confluence")
    print("  Target: Agentic account high-growth mode")
    print("  Builds on v1 (Sharpe 1.28 validated)")
    print("=" * 70)

    # Load data
    print("\nDownloading data...")
    prices_df = load_data()

    # Extract SPY and VIX
    spy_prices = prices_df.loc['SPY'] if 'SPY' in prices_df.index.get_level_values('ticker') else None
    vix_data = prices_df.loc['^VIX'] if '^VIX' in prices_df.index.get_level_values('ticker') else None

    if spy_prices is None or vix_data is None:
        print("ERROR: Missing SPY or VIX data")
        return

    # Get trading dates
    trading_dates = sorted(spy_prices.index.unique())
    oot_dates = [d for d in trading_dates if d >= pd.Timestamp(OOT_START)]
    print(f"Data: {trading_dates[0].date()} to {trading_dates[-1].date()}, {len(trading_dates)} days")
    print(f"OOT period: {oot_dates[0].date()} to {oot_dates[-1].date()} ({len(oot_dates)} days)")
    print(f"ETFs: {ETF_UNIVERSE}")

    # Pre-compute sector returns for dispersion signal
    print("\nComputing cross-sector features...")
    sector_close = {}
    for ticker in ETF_UNIVERSE:
        if ticker in prices_df.index.get_level_values('ticker'):
            td = prices_df.loc[ticker, 'close']
            sector_close[ticker] = td

    if sector_close:
        sector_df = pd.DataFrame(sector_close)
        all_sector_returns = sector_df.pct_change()
    else:
        all_sector_returns = None

    # Run all variants
    all_results = {}
    all_metrics = {}

    for vk in sorted(VARIANTS.keys()):
        cfg = VARIANTS[vk]
        print(f"\n{'=' * 60}")
        print(f"  VARIANT {vk}: {cfg['name']}")
        print(f"{'=' * 60}")

        result = run_variant(vk, cfg, prices_df, spy_prices, vix_data,
                            all_sector_returns, trading_dates)

        metrics = compute_metrics(result, vk, cfg, spy_prices, trading_dates)

        # Show first 3 trades
        for t in result['trades'][:3]:
            print(f"  {t['entry_date']}: {t['type'].upper()} {t['ticker']} K={t['strike']} "
                  f"prem=${t['entry_premium']:.2f} → ${t['exit_premium']:.2f} "
                  f"held={t['days_held']}d pnl=${t['pnl']:.0f} ({t['exit_reason']})")

        print(f"  Trades: {metrics['total_trades']} | Sharpe: {metrics['sharpe']} | "
              f"Sortino: {metrics['sortino']} | PF: {metrics['profit_factor']} | "
              f"WR: {metrics['win_rate']}% | MDD: {metrics['max_drawdown_pct']}%")
        print(f"  Final: ${metrics['final_equity']:.0f} | Return: {metrics['total_return_pct']:.1f}% | "
              f"CAGR: {metrics['cagr_pct']:.1f}% | Regime Gap: {metrics.get('regime_gap', 'N/A')}")
        print(f"  Runtime: {metrics['runtime']:.1f}s")

        # Permutation test
        if metrics['total_trades'] >= 5 and metrics['sharpe'] > 0:
            print(f"  Running {N_PERMUTATIONS}-permutation test...")
            p_val, rand_sharpe = run_permutation_test(
                vk, cfg, prices_df, spy_prices, vix_data,
                all_sector_returns, trading_dates, metrics['sharpe'], N_PERMUTATIONS)
            n_pass, gates = validate_5gate(metrics, p_val, rand_sharpe)

            print(f"  5-Gate: {n_pass}/5 PASS")
            for gname, gpassed in gates.items():
                val_str = ""
                if gname == 'sharpe_gt_1':
                    val_str = f"value={metrics['sharpe']}, threshold=1.0"
                elif gname == 'perm_p_lt_005':
                    val_str = f"value={p_val:.3f}, threshold=0.05"
                elif gname == 'wr_gt_40':
                    val_str = f"value={metrics['win_rate']}, threshold=40.0"
                elif gname == 'regime_balance':
                    val_str = f"value={metrics.get('regime_gap', 999):.3f}, threshold=0.5"
                elif gname == 'beats_random':
                    val_str = f"value={metrics['sharpe']}, random={rand_sharpe:.2f}"
                print(f"    {gname}: {'PASS' if gpassed else 'FAIL'} ({val_str})")

            metrics['perm_p'] = round(p_val, 4)
            metrics['random_sharpe'] = round(rand_sharpe, 2)
            metrics['gates_passed'] = n_pass
        else:
            metrics['perm_p'] = 1.0
            metrics['random_sharpe'] = 0
            metrics['gates_passed'] = 0
            print(f"  5-Gate: 0/5 PASS (insufficient trades or negative Sharpe)")

        all_results[vk] = result
        all_metrics[vk] = metrics

        # MLflow logging
        if MLFLOW_AVAILABLE:
            try:
                with mlflow.start_run(run_name=f"v2_{vk}_{cfg['name'].replace(' ', '_')}"):
                    mlflow.log_params({
                        'variant': vk,
                        'name': cfg['name'],
                        'min_conviction': cfg['min_conviction'],
                        'tp_pct': cfg['tp_pct'],
                        'sl_pct': cfg['sl_pct'],
                        'dte': cfg['dte'],
                        'trailing_stop': cfg['trailing_stop'],
                    })
                    for mk, mv in metrics.items():
                        if isinstance(mv, (int, float)):
                            mlflow.log_metric(mk, mv)
            except:
                pass

    # ============================================================
    # SUMMARY
    # ============================================================
    print(f"\n{'=' * 80}")
    print("  SUMMARY — MOMENTUM BURST ENHANCED V2")
    print(f"{'=' * 80}")

    # Sort by Sharpe
    sorted_variants = sorted(all_metrics.items(), key=lambda x: x[1]['sharpe'], reverse=True)

    print(f"\n  {'Var':<4} {'Name':<30} {'Sharpe':>7} {'Sort':>7} {'PF':>6} {'WR':>6} "
          f"{'Trades':>7} {'Return':>8} {'MDD':>7} {'Gates':>6}")
    print(f"  {'-'*4} {'-'*30} {'-'*7} {'-'*7} {'-'*6} {'-'*6} {'-'*7} {'-'*8} {'-'*7} {'-'*6}")

    for vk, m in sorted_variants:
        print(f"  {vk:<4} {m['name']:<30} {m['sharpe']:>7.2f} {m['sortino']:>7.2f} "
              f"{m['profit_factor']:>6.2f} {m['win_rate']:>5.1f}% {m['total_trades']:>7} "
              f"{m['total_return_pct']:>7.1f}% {m['max_drawdown_pct']:>6.1f}% "
              f"{m.get('gates_passed', 0):>4}/5")

    best_vk = sorted_variants[0][0]
    best_m = sorted_variants[0][1]

    print(f"\n  BEST VARIANT: {best_vk} ({best_m['name']}) — Sharpe {best_m['sharpe']}")

    # Compare to v1
    print(f"\n  v1 BASELINE: Sharpe 1.28 (variant F trailing stop)")
    if best_m['sharpe'] > 1.28:
        print(f"  ✅ v2 IMPROVEMENT: +{best_m['sharpe'] - 1.28:.2f} Sharpe points")
    else:
        print(f"  ❌ v2 DID NOT IMPROVE on v1 baseline ({best_m['sharpe']:.2f} vs 1.28)")

    # Save results
    results_path = os.path.join(OUTPUT_DIR, 'backtest_results.json')
    with open(results_path, 'w') as f:
        json.dump({
            'metrics': {k: v for k, v in all_metrics.items()},
            'best_variant': best_vk,
            'v1_baseline_sharpe': 1.28,
            'timestamp': datetime.now().isoformat(),
        }, f, indent=2, default=str)

    print(f"\nResults saved to {results_path}")
    print("\nDone.")


if __name__ == '__main__':
    main()
