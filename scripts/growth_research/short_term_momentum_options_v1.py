#!/usr/bin/env python3
"""
Short-Term Momentum Burst Options Strategy v1
==============================================
Single-leg calls/puts on sector ETFs with SHORT hold periods (1-5 days).

KEY INSIGHT (from real trading):
- Monthly sector rotation + hold-to-expiry = total failure (Sharpe -2 to -4). Theta kills edge.
- But XLE call held 3 days on momentum = +65.8% return.
- Single-leg options CAN work if: (a) trade short-term momentum, (b) exit within days,
  (c) use higher confidence thresholds.

STRATEGY CONCEPT - "Momentum Burst":
- Entry requires 2+ momentum signals (5d breakout, relative strength, RSI cross, volume surge)
- DTE 14-21 (shorter = less theta bleed)
- ATM or slightly OTM
- Exit: profit target, stop loss, time stop, trailing stop
- Max 2 positions, max $200/trade, $645 account

TESTS 8 VARIANTS (A-H) covering TP/SL/time-stop/moneyness/confidence/direction combos.

PRICING: Inline Black-Scholes. IV = max(VIX/100, realized_vol_21d * 1.2).
DATA: yfinance daily OHLCV, 2020-01-01 to 2026-07-25.
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

# --- Path setup (works on Jupiter and Neptune) ---
for root in ['/home/jupiter/Lvl3Quant', '/home/nick/Lvl3Quant']:
    if os.path.isdir(root):
        LVL3_ROOT = root
        break
else:
    LVL3_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

OUTPUT_DIR = os.path.join(LVL3_ROOT, 'output', 'growth_research', 'short_term_momentum')
os.makedirs(OUTPUT_DIR, exist_ok=True)

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

# ============================================================
# CONSTANTS
# ============================================================
ETF_UNIVERSE = ['XLK', 'XLF', 'XLE', 'XLV', 'XLY', 'XLP', 'XLI', 'XLB', 'XLU', 'XLRE', 'XLC']
STARTING_CAPITAL = 645.0
COMMISSION_PER_LEG = 0.65
COMMISSION_RT = 1.30  # round-trip per contract
MAX_POSITIONS = 2
MAX_POSITION_DOLLARS = 200.0
MAX_POSITION_PCT = 0.30
RISK_FREE_RATE = 0.05  # annualized
START_DATE = '2020-01-01'
END_DATE = '2026-07-25'
N_PERMUTATIONS = 100

# ============================================================
# BLACK-SCHOLES PRICING (inline, self-contained)
# ============================================================

def bs_d1(S, K, T, r, sigma):
    """d1 in Black-Scholes."""
    if T <= 0 or sigma <= 0:
        return 0.0
    return (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))


def bs_d2(S, K, T, r, sigma):
    """d2 in Black-Scholes."""
    if T <= 0 or sigma <= 0:
        return 0.0
    return bs_d1(S, K, T, r, sigma) - sigma * np.sqrt(T)


def bs_call_price(S, K, T, r, sigma):
    """European call price via Black-Scholes."""
    if T <= 1e-8:
        return max(S - K, 0.0)
    d1 = bs_d1(S, K, T, r, sigma)
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def bs_put_price(S, K, T, r, sigma):
    """European put price via Black-Scholes."""
    if T <= 1e-8:
        return max(K - S, 0.0)
    d1 = bs_d1(S, K, T, r, sigma)
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def option_price(S, K, T, r, sigma, option_type='call'):
    """Price a call or put."""
    if option_type == 'call':
        return bs_call_price(S, K, T, r, sigma)
    else:
        return bs_put_price(S, K, T, r, sigma)


# ============================================================
# DATA LOADING
# ============================================================

def load_data():
    """Load daily OHLCV for ETF universe via yfinance with caching."""
    cache_path = os.path.join(LVL3_ROOT, 'data', 'short_term_momentum_cache.parquet')

    if os.path.exists(cache_path):
        df = pd.read_parquet(cache_path)
        # Check if cache is reasonably recent
        if len(df) > 0:
            latest = df.index.get_level_values('date').max()
            if pd.Timestamp(latest) >= pd.Timestamp('2026-07-20'):
                print(f"Loaded cached data: {len(df)} rows, latest={latest}")
                return df

    import yfinance as yf

    print(f"Downloading data for {len(ETF_UNIVERSE)} ETFs + SPY + ^VIX...")
    tickers = ETF_UNIVERSE + ['SPY', '^VIX']
    all_frames = []

    for ticker in tickers:
        try:
            data = yf.download(ticker, start=START_DATE, end=END_DATE, progress=False, auto_adjust=True)
            if len(data) < 100:
                print(f"  WARNING: {ticker} has only {len(data)} rows, skipping")
                continue
            data.columns = [c.lower() if isinstance(c, str) else c[0].lower() for c in data.columns]
            data['ticker'] = ticker
            data.index.name = 'date'
            all_frames.append(data)
            print(f"  {ticker}: {len(data)} rows")
        except Exception as e:
            print(f"  ERROR downloading {ticker}: {e}")

    df = pd.concat(all_frames)
    df = df.reset_index().set_index(['ticker', 'date']).sort_index()

    # Save cache
    try:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        df.to_parquet(cache_path)
        print(f"Cached data to {cache_path}")
    except Exception as e:
        print(f"Warning: Could not cache data: {e}")

    return df


def compute_features(prices_df, ticker, date, spy_prices):
    """
    Compute momentum signals for a given ticker on a given date.
    Returns dict of signal values and a boolean for bull/bear direction.
    """
    try:
        ticker_data = prices_df.loc[ticker]
    except KeyError:
        return None

    # Get data up to this date
    mask = ticker_data.index <= date
    td = ticker_data.loc[mask]

    if len(td) < 25:
        return None

    close = td['close'].values
    volume = td['volume'].values

    # Need at least 21 trading days of history
    if len(close) < 21:
        return None

    signals = {}
    bull_signals = 0
    bear_signals = 0

    # --- Signal 1: 5-day momentum > 3% ---
    if len(close) >= 6:
        mom_5d = (close[-1] / close[-6]) - 1.0
        signals['mom_5d'] = mom_5d
        if mom_5d > 0.03:
            bull_signals += 1
        elif mom_5d < -0.03:
            bear_signals += 1

    # --- Signal 2: Relative strength vs SPY > 2% over 10 days ---
    spy_mask = spy_prices.index <= date
    spy_td = spy_prices.loc[spy_mask]
    if len(spy_td) >= 11 and len(close) >= 11:
        etf_ret_10d = (close[-1] / close[-11]) - 1.0
        spy_close = spy_td['close'].values
        spy_ret_10d = (spy_close[-1] / spy_close[-11]) - 1.0
        rel_strength = etf_ret_10d - spy_ret_10d
        signals['rel_strength_10d'] = rel_strength
        if rel_strength > 0.02:
            bull_signals += 1
        elif rel_strength < -0.02:
            bear_signals += 1

    # --- Signal 3: RSI crossed above 60 (bull) or below 40 (bear) within last 3 days ---
    if len(close) >= 18:
        # Compute RSI(14) for last few days
        def compute_rsi(prices, period=14):
            deltas = np.diff(prices)
            gains = np.where(deltas > 0, deltas, 0)
            losses = np.where(deltas < 0, -deltas, 0)
            avg_gain = np.mean(gains[-period:])
            avg_loss = np.mean(losses[-period:])
            if avg_loss == 0:
                return 100.0
            rs = avg_gain / avg_loss
            return 100.0 - (100.0 / (1.0 + rs))

        rsi_today = compute_rsi(close)
        rsi_3d_ago = compute_rsi(close[:-3]) if len(close) > 20 else rsi_today
        signals['rsi'] = rsi_today

        # Bull: RSI crossed above 60 in last 3 days
        if rsi_today >= 60 and rsi_3d_ago < 60:
            bull_signals += 1
        # Bear: RSI crossed below 40 in last 3 days
        if rsi_today <= 40 and rsi_3d_ago > 40:
            bear_signals += 1

    # --- Signal 4: Volume surge > 1.5x 20-day average ---
    if len(volume) >= 21:
        avg_vol_20 = np.mean(volume[-21:-1])
        vol_ratio = volume[-1] / (avg_vol_20 + 1e-8)
        signals['volume_ratio'] = vol_ratio
        if vol_ratio > 1.5:
            # Volume surge supports whatever direction momentum is going
            if signals.get('mom_5d', 0) > 0:
                bull_signals += 1
            elif signals.get('mom_5d', 0) < 0:
                bear_signals += 1

    # --- Compute IV proxy ---
    if len(close) >= 22:
        daily_rets = np.diff(np.log(close[-22:]))
        realized_vol_21d = np.std(daily_rets) * np.sqrt(252)
    else:
        realized_vol_21d = 0.25  # default

    signals['realized_vol_21d'] = realized_vol_21d
    signals['bull_signals'] = bull_signals
    signals['bear_signals'] = bear_signals
    signals['spot'] = close[-1]

    return signals


# ============================================================
# STRATEGY VARIANTS
# ============================================================

VARIANTS = {
    'A': {
        'name': 'Base',
        'tp_pct': 0.30,
        'sl_pct': 0.25,
        'time_stop_days': 5,
        'moneyness_pct': 0.00,  # ATM
        'dte': 14,
        'trailing_stop': False,
        'trailing_giveback': 0.50,
        'min_signals': 2,
        'direction': 'both',
    },
    'B': {
        'name': 'Aggressive TP',
        'tp_pct': 0.50,
        'sl_pct': 0.25,
        'time_stop_days': 5,
        'moneyness_pct': 0.00,
        'dte': 14,
        'trailing_stop': False,
        'trailing_giveback': 0.50,
        'min_signals': 2,
        'direction': 'both',
    },
    'C': {
        'name': 'Tight SL',
        'tp_pct': 0.30,
        'sl_pct': 0.15,
        'time_stop_days': 5,
        'moneyness_pct': 0.00,
        'dte': 14,
        'trailing_stop': False,
        'trailing_giveback': 0.50,
        'min_signals': 2,
        'direction': 'both',
    },
    'D': {
        'name': 'Longer Hold',
        'tp_pct': 0.30,
        'sl_pct': 0.25,
        'time_stop_days': 8,
        'moneyness_pct': 0.00,
        'dte': 21,
        'trailing_stop': False,
        'trailing_giveback': 0.50,
        'min_signals': 2,
        'direction': 'both',
    },
    'E': {
        'name': 'OTM 2%',
        'tp_pct': 0.30,
        'sl_pct': 0.25,
        'time_stop_days': 5,
        'moneyness_pct': 0.02,  # 2% OTM
        'dte': 14,
        'trailing_stop': False,
        'trailing_giveback': 0.50,
        'min_signals': 2,
        'direction': 'both',
    },
    'F': {
        'name': 'Trailing Stop',
        'tp_pct': 0.30,
        'sl_pct': 0.25,
        'time_stop_days': 5,
        'moneyness_pct': 0.00,
        'dte': 14,
        'trailing_stop': True,
        'trailing_giveback': 0.50,
        'min_signals': 2,
        'direction': 'both',
    },
    'G': {
        'name': 'High Conf Only',
        'tp_pct': 0.30,
        'sl_pct': 0.25,
        'time_stop_days': 5,
        'moneyness_pct': 0.00,
        'dte': 14,
        'trailing_stop': False,
        'trailing_giveback': 0.50,
        'min_signals': 3,
        'direction': 'both',
    },
    'H': {
        'name': 'Puts Only',
        'tp_pct': 0.30,
        'sl_pct': 0.25,
        'time_stop_days': 5,
        'moneyness_pct': 0.00,
        'dte': 14,
        'trailing_stop': False,
        'trailing_giveback': 0.50,
        'min_signals': 2,
        'direction': 'bear',
    },
}


# ============================================================
# BACKTESTING ENGINE
# ============================================================

class Position:
    """Tracks an open option position."""
    def __init__(self, ticker, option_type, strike, entry_price, entry_date,
                 entry_spot, dte, iv, cost, n_contracts=1):
        self.ticker = ticker
        self.option_type = option_type  # 'call' or 'put'
        self.strike = strike
        self.entry_price = entry_price  # option premium per share
        self.entry_date = entry_date
        self.entry_spot = entry_spot
        self.dte = dte
        self.iv = iv
        self.cost = cost  # total cost including commission
        self.n_contracts = n_contracts
        self.days_held = 0
        self.peak_value = entry_price  # for trailing stop
        self.exit_price = None
        self.exit_date = None
        self.exit_reason = None
        self.pnl = None


def get_vix_for_date(vix_data, date):
    """Get VIX level for a given date."""
    mask = vix_data.index <= date
    if mask.any():
        return vix_data.loc[mask, 'close'].iloc[-1]
    return 20.0  # default


def estimate_iv(vix_level, realized_vol):
    """Estimate implied volatility. IV = max(VIX/100, realized_vol * 1.2)."""
    return max(vix_level / 100.0, realized_vol * 1.2)


def run_variant(variant_key, variant_cfg, prices_df, spy_prices, vix_data, trading_dates):
    """Run a single strategy variant over the full backtest period."""
    equity = STARTING_CAPITAL
    equity_curve = [equity]
    equity_dates = [trading_dates[0]]
    positions = []  # open positions
    all_trades = []
    daily_returns = []

    # Pre-compute signal dates to avoid scanning every ticker every day
    # We'll scan for signals on each trading day
    for i, date in enumerate(trading_dates):
        if i < 25:  # need lookback
            equity_curve.append(equity)
            equity_dates.append(date)
            daily_returns.append(0.0)
            continue

        prev_equity = equity

        # --- Mark-to-market open positions ---
        positions_to_close = []
        for pos_idx, pos in enumerate(positions):
            pos.days_held += 1

            # Get current spot price
            try:
                ticker_data = prices_df.loc[pos.ticker]
                mask = ticker_data.index <= date
                if not mask.any():
                    continue
                current_spot = ticker_data.loc[mask, 'close'].iloc[-1]
            except (KeyError, IndexError):
                continue

            # Remaining DTE
            remaining_dte = max(pos.dte - pos.days_held, 0)
            T = remaining_dte / 252.0

            # Get current VIX for IV update
            vix_level = get_vix_for_date(vix_data, date)
            current_iv = estimate_iv(vix_level, pos.iv * 0.9)  # slight decay assumption

            # Recalculate option value
            current_value = option_price(current_spot, pos.strike, T, RISK_FREE_RATE,
                                         current_iv, pos.option_type)

            # Track peak for trailing stop
            if current_value > pos.peak_value:
                pos.peak_value = current_value

            # --- Exit checks ---
            exit_reason = None
            pct_change = (current_value - pos.entry_price) / pos.entry_price

            # Take profit
            if pct_change >= variant_cfg['tp_pct']:
                exit_reason = 'take_profit'

            # Stop loss
            elif pct_change <= -variant_cfg['sl_pct']:
                exit_reason = 'stop_loss'

            # Time stop
            elif pos.days_held >= variant_cfg['time_stop_days']:
                exit_reason = 'time_stop'

            # Trailing stop (only if enabled and we've had some gain)
            elif variant_cfg['trailing_stop'] and pos.peak_value > pos.entry_price:
                unrealized_from_peak = pos.peak_value - pos.entry_price
                giveback = pos.peak_value - current_value
                if giveback > unrealized_from_peak * variant_cfg['trailing_giveback']:
                    exit_reason = 'trailing_stop'

            if exit_reason:
                # Close position
                exit_value = current_value * 100 * pos.n_contracts  # option value in dollars
                entry_cost = pos.cost
                pnl = exit_value - entry_cost - COMMISSION_PER_LEG  # exit leg commission
                pos.pnl = pnl
                pos.exit_price = current_value
                pos.exit_date = date
                pos.exit_reason = exit_reason
                equity += pnl
                positions_to_close.append(pos_idx)

                all_trades.append({
                    'ticker': pos.ticker,
                    'type': pos.option_type,
                    'strike': pos.strike,
                    'entry_date': str(pos.entry_date.date()) if hasattr(pos.entry_date, 'date') else str(pos.entry_date),
                    'exit_date': str(date.date()) if hasattr(date, 'date') else str(date),
                    'entry_premium': round(pos.entry_price, 4),
                    'exit_premium': round(current_value, 4),
                    'days_held': pos.days_held,
                    'pnl': round(pnl, 2),
                    'pnl_pct': round(pct_change * 100, 1),
                    'exit_reason': exit_reason,
                })

        # Remove closed positions (reverse order to preserve indices)
        for idx in sorted(positions_to_close, reverse=True):
            positions.pop(idx)

        # --- Generate new entry signals (if we have capacity) ---
        if len(positions) < MAX_POSITIONS:
            candidates = []

            for ticker in ETF_UNIVERSE:
                feat = compute_features(prices_df, ticker, date, spy_prices)
                if feat is None:
                    continue

                # Check direction filter
                direction = variant_cfg['direction']
                bull_ok = direction in ('both', 'bull')
                bear_ok = direction in ('both', 'bear')

                # Bull entry
                if bull_ok and feat['bull_signals'] >= variant_cfg['min_signals']:
                    candidates.append((ticker, 'call', feat))

                # Bear entry
                if bear_ok and feat['bear_signals'] >= variant_cfg['min_signals']:
                    candidates.append((ticker, 'put', feat))

            # Sort by signal count (strongest first)
            candidates.sort(key=lambda x: x[2].get('bull_signals', 0) + x[2].get('bear_signals', 0),
                            reverse=True)

            # Don't open position in a ticker we already hold
            held_tickers = {p.ticker for p in positions}

            for ticker, opt_type, feat in candidates:
                if len(positions) >= MAX_POSITIONS:
                    break
                if ticker in held_tickers:
                    continue

                spot = feat['spot']
                vix_level = get_vix_for_date(vix_data, date)
                iv = estimate_iv(vix_level, feat['realized_vol_21d'])

                # Strike: ATM or OTM
                moneyness = variant_cfg['moneyness_pct']
                if opt_type == 'call':
                    strike = spot * (1.0 + moneyness)
                else:
                    strike = spot * (1.0 - moneyness)

                # Round strike to nearest dollar
                strike = round(strike, 0)

                # Price the option
                T = variant_cfg['dte'] / 252.0
                premium = option_price(spot, strike, T, RISK_FREE_RATE, iv, opt_type)

                if premium < 0.10:
                    continue  # too cheap, probably deep OTM or no value

                # Position sizing
                contract_cost = premium * 100  # 1 contract = 100 shares
                max_spend = min(MAX_POSITION_DOLLARS, equity * MAX_POSITION_PCT)

                if contract_cost > max_spend:
                    continue  # can't afford even 1 contract at this size cap

                n_contracts = 1  # Level 2 small account, always 1 contract
                total_cost = contract_cost + COMMISSION_PER_LEG  # entry leg

                if total_cost > equity:
                    continue  # insufficient capital

                # Open position
                pos = Position(
                    ticker=ticker,
                    option_type=opt_type,
                    strike=strike,
                    entry_price=premium,
                    entry_date=date,
                    entry_spot=spot,
                    dte=variant_cfg['dte'],
                    iv=iv,
                    cost=total_cost,
                    n_contracts=n_contracts,
                )
                positions.append(pos)
                held_tickers.add(ticker)

        # Daily return
        daily_ret = (equity - prev_equity) / max(prev_equity, 1.0)
        daily_returns.append(daily_ret)
        equity_curve.append(equity)
        equity_dates.append(date)

    # Force-close any remaining positions at final date
    final_date = trading_dates[-1]
    for pos in positions:
        try:
            ticker_data = prices_df.loc[pos.ticker]
            mask = ticker_data.index <= final_date
            if not mask.any():
                continue
            current_spot = ticker_data.loc[mask, 'close'].iloc[-1]
        except (KeyError, IndexError):
            continue

        remaining_dte = max(pos.dte - pos.days_held, 0)
        T = remaining_dte / 252.0
        vix_level = get_vix_for_date(vix_data, final_date)
        current_iv = estimate_iv(vix_level, pos.iv * 0.9)
        current_value = option_price(current_spot, pos.strike, T, RISK_FREE_RATE,
                                     current_iv, pos.option_type)
        exit_value = current_value * 100 * pos.n_contracts
        pnl = exit_value - pos.cost - COMMISSION_PER_LEG
        equity += pnl

        all_trades.append({
            'ticker': pos.ticker,
            'type': pos.option_type,
            'strike': pos.strike,
            'entry_date': str(pos.entry_date.date()) if hasattr(pos.entry_date, 'date') else str(pos.entry_date),
            'exit_date': str(final_date.date()) if hasattr(final_date, 'date') else str(final_date),
            'entry_premium': round(pos.entry_price, 4),
            'exit_premium': round(current_value, 4),
            'days_held': pos.days_held,
            'pnl': round(pnl, 2),
            'pnl_pct': round(((current_value - pos.entry_price) / pos.entry_price) * 100, 1),
            'exit_reason': 'final_close',
        })

    return {
        'equity_curve': equity_curve,
        'equity_dates': equity_dates,
        'daily_returns': daily_returns,
        'trades': all_trades,
        'final_equity': equity,
    }


# ============================================================
# ANALYSIS & VALIDATION
# ============================================================

def compute_metrics(result, variant_key, variant_cfg):
    """Compute Sharpe, Sortino, WR, PF, max DD, avg hold, avg PnL."""
    trades = result['trades']
    daily_rets = np.array(result['daily_returns'])

    metrics = {
        'variant': variant_key,
        'name': variant_cfg['name'],
        'final_equity': round(result['final_equity'], 2),
        'total_return_pct': round((result['final_equity'] / STARTING_CAPITAL - 1) * 100, 2),
        'total_trades': len(trades),
    }

    if len(trades) == 0:
        metrics.update({
            'sharpe': 0.0, 'sortino': 0.0, 'win_rate': 0.0,
            'profit_factor': 0.0, 'max_drawdown_pct': 0.0,
            'avg_hold_days': 0.0, 'avg_pnl': 0.0, 'avg_pnl_pct': 0.0,
        })
        return metrics

    # Trade-level stats
    pnls = [t['pnl'] for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]

    metrics['win_rate'] = round(len(wins) / len(pnls) * 100, 1) if pnls else 0.0
    metrics['avg_pnl'] = round(np.mean(pnls), 2)
    metrics['avg_pnl_pct'] = round(np.mean([t['pnl_pct'] for t in trades]), 1)
    metrics['avg_hold_days'] = round(np.mean([t['days_held'] for t in trades]), 1)

    gross_profit = sum(wins) if wins else 0
    gross_loss = abs(sum(losses)) if losses else 0
    metrics['profit_factor'] = round(gross_profit / gross_loss, 2) if gross_loss > 0 else (
        999.0 if gross_profit > 0 else 0.0)

    # Daily returns stats
    nonzero_rets = daily_rets[daily_rets != 0]
    if len(nonzero_rets) > 5:
        ann_factor = np.sqrt(252)
        mean_ret = np.mean(daily_rets)
        std_ret = np.std(daily_rets)
        metrics['sharpe'] = round((mean_ret / std_ret) * ann_factor, 2) if std_ret > 0 else 0.0

        downside_rets = daily_rets[daily_rets < 0]
        downside_std = np.std(downside_rets) if len(downside_rets) > 1 else std_ret
        metrics['sortino'] = round((mean_ret / downside_std) * ann_factor, 2) if downside_std > 0 else 0.0
    else:
        metrics['sharpe'] = 0.0
        metrics['sortino'] = 0.0

    # Max drawdown from equity curve
    eq = np.array(result['equity_curve'])
    peak = np.maximum.accumulate(eq)
    dd = (eq - peak) / peak
    metrics['max_drawdown_pct'] = round(np.min(dd) * 100, 2)

    # Exit reason breakdown
    reasons = defaultdict(int)
    for t in trades:
        reasons[t['exit_reason']] += 1
    metrics['exit_reasons'] = dict(reasons)

    return metrics


def permutation_test(result, n_perms=N_PERMUTATIONS):
    """Shuffle trade P&Ls to test if the strategy has real edge."""
    trades = result['trades']
    if len(trades) < 5:
        return {'p_value': 1.0, 'actual_sharpe': 0.0, 'perm_sharpes_mean': 0.0}

    pnls = np.array([t['pnl'] for t in trades])
    actual_mean = np.mean(pnls)

    # Permutation: shuffle signs (null hypothesis: no directional edge)
    rng = np.random.RandomState(42)
    perm_means = []
    for _ in range(n_perms):
        signs = rng.choice([-1, 1], size=len(pnls))
        perm_means.append(np.mean(pnls * signs))

    perm_means = np.array(perm_means)
    p_value = np.mean(perm_means >= actual_mean)

    return {
        'p_value': round(p_value, 4),
        'actual_mean_pnl': round(actual_mean, 2),
        'perm_mean_pnl_mean': round(np.mean(perm_means), 2),
        'perm_mean_pnl_std': round(np.std(perm_means), 2),
    }


def regime_analysis(result, spy_prices, trading_dates):
    """Split performance by green/red SPY days."""
    trades = result['trades']
    if len(trades) < 5:
        return {'green_day': {}, 'red_day': {}}

    # Classify each trade by SPY regime during its holding period
    spy_daily = spy_prices['close'].pct_change()

    green_pnls = []
    red_pnls = []

    for t in trades:
        entry = pd.Timestamp(t['entry_date'])
        exit_dt = pd.Timestamp(t['exit_date'])

        # Get SPY return during hold period
        mask = (spy_daily.index >= entry) & (spy_daily.index <= exit_dt)
        spy_period_ret = spy_daily.loc[mask].sum() if mask.any() else 0

        if spy_period_ret >= 0:
            green_pnls.append(t['pnl'])
        else:
            red_pnls.append(t['pnl'])

    def summarize(pnls, label):
        if not pnls:
            return {}
        pnls = np.array(pnls)
        return {
            'count': len(pnls),
            'win_rate': round(np.mean(pnls > 0) * 100, 1),
            'avg_pnl': round(np.mean(pnls), 2),
            'total_pnl': round(np.sum(pnls), 2),
        }

    return {
        'green_day': summarize(green_pnls, 'green'),
        'red_day': summarize(red_pnls, 'red'),
    }


def monthly_returns(result):
    """Compute monthly P&L distribution."""
    trades = result['trades']
    if not trades:
        return {}

    monthly = defaultdict(float)
    for t in trades:
        month_key = t['exit_date'][:7]  # YYYY-MM
        monthly[month_key] += t['pnl']

    monthly_vals = list(monthly.values())
    return {
        'n_months_active': len(monthly),
        'avg_monthly_pnl': round(np.mean(monthly_vals), 2),
        'median_monthly_pnl': round(np.median(monthly_vals), 2),
        'best_month': round(max(monthly_vals), 2),
        'worst_month': round(min(monthly_vals), 2),
        'pct_months_positive': round(np.mean(np.array(monthly_vals) > 0) * 100, 1),
        'monthly_pnl': {k: round(v, 2) for k, v in sorted(monthly.items())},
    }


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 70)
    print("SHORT-TERM MOMENTUM BURST OPTIONS STRATEGY v1")
    print("=" * 70)
    print(f"Account: ${STARTING_CAPITAL:.0f} | Universe: {len(ETF_UNIVERSE)} sector ETFs")
    print(f"Period: {START_DATE} to {END_DATE}")
    print(f"Variants: {len(VARIANTS)}")
    print()

    # Load data
    prices_df = load_data()

    # Extract SPY and VIX
    try:
        spy_prices = prices_df.loc['SPY'].copy()
    except KeyError:
        print("ERROR: SPY data not found")
        return
    try:
        vix_data = prices_df.loc['^VIX'].copy()
    except KeyError:
        print("WARNING: VIX data not found, using default IV=0.20")
        vix_data = spy_prices.copy()
        vix_data['close'] = 20.0

    # Get trading dates from SPY
    trading_dates = sorted(spy_prices.index)
    print(f"Trading dates: {len(trading_dates)} ({trading_dates[0].date()} to {trading_dates[-1].date()})")
    print()

    # Run all variants
    all_results = {}
    all_metrics = {}
    all_perm_tests = {}
    all_regime = {}
    all_monthly = {}

    for vk, vcfg in VARIANTS.items():
        print(f"--- Running Variant {vk}: {vcfg['name']} ---")
        result = run_variant(vk, vcfg, prices_df, spy_prices, vix_data, trading_dates)
        metrics = compute_metrics(result, vk, vcfg)
        perm = permutation_test(result)
        regime = regime_analysis(result, spy_prices, trading_dates)
        monthly = monthly_returns(result)

        all_results[vk] = result
        all_metrics[vk] = metrics
        all_perm_tests[vk] = perm
        all_regime[vk] = regime
        all_monthly[vk] = monthly

        print(f"  Trades: {metrics['total_trades']} | WR: {metrics['win_rate']}% | "
              f"Sharpe: {metrics['sharpe']} | Sortino: {metrics['sortino']} | "
              f"PF: {metrics['profit_factor']} | MaxDD: {metrics['max_drawdown_pct']}%")
        print(f"  Final equity: ${metrics['final_equity']:.2f} | "
              f"Total return: {metrics['total_return_pct']}% | "
              f"Avg hold: {metrics['avg_hold_days']}d | Avg PnL: ${metrics['avg_pnl']:.2f}")
        print(f"  Perm test p-value: {perm['p_value']}")
        if regime.get('green_day'):
            print(f"  Green days: {regime['green_day'].get('count', 0)} trades, "
                  f"WR={regime['green_day'].get('win_rate', 0)}%")
        if regime.get('red_day'):
            print(f"  Red days: {regime['red_day'].get('count', 0)} trades, "
                  f"WR={regime['red_day'].get('win_rate', 0)}%")
        print()

    # ============================================================
    # SUMMARY TABLE
    # ============================================================
    print("=" * 90)
    print("SUMMARY TABLE")
    print("=" * 90)
    header = f"{'Var':>3} {'Name':<16} {'Trades':>6} {'WR%':>5} {'Sharpe':>7} {'Sortino':>7} " \
             f"{'PF':>5} {'MaxDD%':>7} {'Final$':>7} {'Ret%':>7} {'PermP':>6}"
    print(header)
    print("-" * 90)
    for vk in sorted(all_metrics.keys()):
        m = all_metrics[vk]
        p = all_perm_tests[vk]
        print(f"  {vk:>1} {m['name']:<16} {m['total_trades']:>6} {m['win_rate']:>5.1f} "
              f"{m['sharpe']:>7.2f} {m['sortino']:>7.2f} {m['profit_factor']:>5.2f} "
              f"{m['max_drawdown_pct']:>7.2f} {m['final_equity']:>7.0f} "
              f"{m['total_return_pct']:>7.1f} {p['p_value']:>6.3f}")
    print()

    # ============================================================
    # SAVE RESULTS
    # ============================================================
    output = {
        'strategy': 'short_term_momentum_burst_options_v1',
        'run_date': datetime.now().isoformat(),
        'starting_capital': STARTING_CAPITAL,
        'universe': ETF_UNIVERSE,
        'period': {'start': START_DATE, 'end': END_DATE},
        'variants': {},
    }

    for vk in sorted(VARIANTS.keys()):
        output['variants'][vk] = {
            'config': VARIANTS[vk],
            'metrics': all_metrics[vk],
            'permutation_test': all_perm_tests[vk],
            'regime_analysis': all_regime[vk],
            'monthly_returns': all_monthly[vk],
            'trades': all_results[vk]['trades'],
        }

    results_path = os.path.join(OUTPUT_DIR, 'results_v1.json')
    with open(results_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"Results saved to {results_path}")

    # Save equity curves
    eq_path = os.path.join(OUTPUT_DIR, 'equity_curves_v1.json')
    eq_data = {}
    for vk in sorted(VARIANTS.keys()):
        eq_data[vk] = {
            'dates': [str(d.date()) if hasattr(d, 'date') else str(d)
                      for d in all_results[vk]['equity_dates']],
            'equity': [round(e, 2) for e in all_results[vk]['equity_curve']],
        }
    with open(eq_path, 'w') as f:
        json.dump(eq_data, f, indent=2)
    print(f"Equity curves saved to {eq_path}")

    # ============================================================
    # MLFLOW LOGGING
    # ============================================================
    if MLFLOW_AVAILABLE:
        try:
            mlflow.set_tracking_uri('http://localhost:5000')
            mlflow.set_experiment('short_term_momentum_options')

            for vk in sorted(VARIANTS.keys()):
                m = all_metrics[vk]
                p = all_perm_tests[vk]
                with mlflow.start_run(run_name=f"variant_{vk}_{VARIANTS[vk]['name']}"):
                    # Log config
                    for k, v in VARIANTS[vk].items():
                        mlflow.log_param(f"cfg_{k}", v)
                    mlflow.log_param('starting_capital', STARTING_CAPITAL)
                    mlflow.log_param('variant', vk)

                    # Log metrics
                    mlflow.log_metric('sharpe', m['sharpe'])
                    mlflow.log_metric('sortino', m['sortino'])
                    mlflow.log_metric('win_rate', m['win_rate'])
                    mlflow.log_metric('profit_factor', m['profit_factor'])
                    mlflow.log_metric('max_drawdown_pct', m['max_drawdown_pct'])
                    mlflow.log_metric('total_trades', m['total_trades'])
                    mlflow.log_metric('total_return_pct', m['total_return_pct'])
                    mlflow.log_metric('final_equity', m['final_equity'])
                    mlflow.log_metric('avg_hold_days', m['avg_hold_days'])
                    mlflow.log_metric('avg_pnl', m['avg_pnl'])
                    mlflow.log_metric('perm_test_p_value', p['p_value'])

                    # Log artifacts
                    mlflow.log_artifact(results_path)

            print("MLflow logging complete.")
        except Exception as e:
            print(f"MLflow logging failed (non-fatal): {e}")

    # ============================================================
    # FINAL VERDICT
    # ============================================================
    print()
    print("=" * 70)
    print("VERDICT")
    print("=" * 70)

    # Find best variant by Sharpe
    best_vk = max(all_metrics.keys(), key=lambda k: all_metrics[k]['sharpe'])
    best = all_metrics[best_vk]
    best_perm = all_perm_tests[best_vk]

    print(f"Best variant: {best_vk} ({best['name']})")
    print(f"  Sharpe: {best['sharpe']} | Sortino: {best['sortino']} | "
          f"WR: {best['win_rate']}% | PF: {best['profit_factor']}")
    print(f"  Total return: {best['total_return_pct']}% | Max DD: {best['max_drawdown_pct']}%")
    print(f"  Permutation p-value: {best_perm['p_value']}")

    if best['sharpe'] > 0.5 and best_perm['p_value'] < 0.10 and best['win_rate'] > 45:
        print("\n  >>> PROMISING — Short-term momentum burst shows edge. Worth paper trading.")
    elif best['sharpe'] > 0 and best['win_rate'] > 40:
        print("\n  >>> MARGINAL — Some edge but not strong. Needs refinement or different parameters.")
    else:
        print("\n  >>> NO EDGE — Short-term momentum burst does not overcome costs in this form.")

    # Regime robustness check
    regime = all_regime[best_vk]
    if regime.get('green_day') and regime.get('red_day'):
        g_wr = regime['green_day'].get('win_rate', 0)
        r_wr = regime['red_day'].get('win_rate', 0)
        if abs(g_wr - r_wr) > 25:
            print(f"  WARNING: Regime-dependent (Green WR={g_wr}% vs Red WR={r_wr}%). "
                  f"Not robust across market conditions.")
        else:
            print(f"  GOOD: Regime-balanced (Green WR={g_wr}% vs Red WR={r_wr}%)")

    print()
    print("Done.")


if __name__ == '__main__':
    main()
