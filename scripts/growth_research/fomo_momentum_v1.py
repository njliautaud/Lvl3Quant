#!/usr/bin/env python3
"""
FOMO Momentum Strategy v1 — IPO Drift + OpEx Pin Effect
==========================================================

Two behavioral biases combined:
A) Post-IPO Quiet Period Drift: Stocks that IPO and rise >20% in month 1 tend to
   cool off days 30-60 then resume uptrend days 60-120 as analyst coverage begins.
B) Options Expiration Pin Release: After OpEx Friday, stocks with heavy OI "unpin"
   and tend to move directionally. Buy breakout direction Monday after OpEx.

VARIANTS:
A — IPO Drift Pure (buy day 60, sell day 120)
B — IPO Drift Conservative (buy day 45, sell day 90, require >30% first-month gain)
C — OpEx Pin Release (buy Monday after OpEx, hold 5 days)
D — OpEx Pin Release + Volume (require volume surge on OpEx Friday)
E — Combined IPO + OpEx (both signals active)
F — IPO Drift Options (buy cheap calls day 55, sell day 100)

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
from scipy.stats import norm, permutation_test

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
        logging.FileHandler(os.path.join(LOG_DIR, 'fomo_momentum_v1.log')),
        logging.StreamHandler(),
    ]
)
log = logging.getLogger(__name__)

# ==================== CONFIG ====================

# Growth stock universe — mix of recent-ish IPOs and established growth
STOCK_UNIVERSE = [
    # Recent IPOs (2019-2024)
    'SNOW', 'PLTR', 'ABNB', 'DASH', 'COIN', 'RBLX', 'RIVN', 'LCID',
    'HOOD', 'SOFI', 'AFRM', 'PATH', 'DUOL', 'ARM', 'BIRK', 'CART',
    'CAVA', 'TOST', 'PINS', 'U',
    # Established growth
    'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'NVDA', 'TSLA', 'AMD',
    'CRM', 'NFLX', 'SHOP', 'SQ', 'UBER', 'NET', 'CRWD', 'DDOG',
    'ZS', 'MDB', 'ROKU', 'SNAP',
    # Additional growth
    'ENPH', 'SMCI', 'MELI', 'SE', 'TTD', 'BILL', 'HUBS', 'VEEV',
    'PANW', 'MNDY',
]

INITIAL_CAPITAL = 645.0
MAX_POS_PCT = 0.20  # max 20% of capital per position
COMMISSION_OPTION = 0.65  # per contract
COMMISSION_EQUITY = 0.0
RISK_FREE_RATE = 0.05
BS_HAIRCUT = 0.85

OOT_START = '2022-01-01'
OOT_END = '2026-07-28'

# OpEx dates are 3rd Friday of each month
def get_opex_dates(start, end):
    """Generate options expiration dates (3rd Friday of each month)."""
    dates = []
    current = pd.Timestamp(start).replace(day=1)
    end_dt = pd.Timestamp(end)
    while current <= end_dt:
        # Find 3rd Friday
        first_day = current.replace(day=1)
        # dayofweek: Monday=0, Friday=4
        days_to_friday = (4 - first_day.dayofweek) % 7
        first_friday = first_day + timedelta(days=days_to_friday)
        third_friday = first_friday + timedelta(weeks=2)
        if third_friday <= end_dt:
            dates.append(third_friday)
        # Next month
        if current.month == 12:
            current = current.replace(year=current.year + 1, month=1)
        else:
            current = current.replace(month=current.month + 1)
    return dates


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
    """Load price + volume data for all stocks and SPY."""
    import yfinance as yf

    cache_dir = os.path.join(LVL3_ROOT, 'data')
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, 'fomo_momentum_v1_cache.parquet')

    all_tickers = list(set(STOCK_UNIVERSE + ['SPY', '^VIX']))

    if os.path.exists(cache_file):
        age_hours = (time.time() - os.path.getmtime(cache_file)) / 3600
        if age_hours < 24:
            log.info("Loading cached price data")
            df = pd.read_parquet(cache_file)
            return df

    log.info(f"Downloading price data for {len(all_tickers)} tickers...")
    # Download with longer history to detect IPO dates
    data_frames = {}
    for ticker in all_tickers:
        try:
            t = yf.Ticker(ticker)
            hist = t.history(start='2018-01-01', end=OOT_END, auto_adjust=True)
            if len(hist) > 20:
                hist.index = hist.index.tz_localize(None)
                data_frames[ticker] = hist[['Close', 'Volume', 'High', 'Low']].copy()
                log.info(f"  {ticker}: {len(hist)} days, first={hist.index[0].date()}")
        except Exception as e:
            log.warning(f"  {ticker} failed: {e}")
        time.sleep(0.3)

    if not data_frames:
        raise RuntimeError("No data downloaded")

    # Combine into multi-level DataFrame
    combined = pd.concat(data_frames, axis=1)
    combined.to_parquet(cache_file)
    log.info(f"Cached {len(data_frames)} tickers")
    return combined


def detect_ipo_date(prices_df, ticker):
    """Detect IPO date as first trading date in our data."""
    try:
        col = (ticker, 'Close')
        if col not in prices_df.columns:
            return None
        series = prices_df[col].dropna()
        if len(series) < 60:
            return None
        return series.index[0]
    except:
        return None


def get_first_month_return(prices_df, ticker, ipo_date):
    """Calculate return in first 21 trading days after IPO."""
    try:
        col = (ticker, 'Close')
        series = prices_df[col].dropna()
        mask = series.index >= ipo_date
        first_month = series[mask].iloc[:21]
        if len(first_month) < 15:
            return None
        return (first_month.iloc[-1] / first_month.iloc[0]) - 1.0
    except:
        return None


# ==================== STRATEGY LOGIC ====================

def run_ipo_drift(prices_df, variant='A'):
    """
    Post-IPO Drift strategy.
    Buy after quiet period, sell during analyst-coverage-driven uptrend.
    """
    trades = []

    if variant == 'A':
        entry_day, exit_day, min_first_month = 60, 120, 0.20
    elif variant == 'B':
        entry_day, exit_day, min_first_month = 45, 90, 0.30
    elif variant == 'F':
        # Options variant
        entry_day, exit_day, min_first_month = 55, 100, 0.20
    else:
        return trades

    for ticker in STOCK_UNIVERSE:
        ipo_date = detect_ipo_date(prices_df, ticker)
        if ipo_date is None:
            continue

        first_month_ret = get_first_month_return(prices_df, ticker, ipo_date)
        if first_month_ret is None or first_month_ret < min_first_month:
            continue

        col = (ticker, 'Close')
        series = prices_df[col].dropna()
        mask = series.index >= ipo_date
        post_ipo = series[mask]

        if len(post_ipo) < exit_day + 5:
            continue

        entry_price = post_ipo.iloc[entry_day]
        exit_price = post_ipo.iloc[exit_day]
        entry_date = post_ipo.index[entry_day]
        exit_date = post_ipo.index[exit_day]

        # Only include if in OOT window
        if entry_date < pd.Timestamp(OOT_START) or entry_date > pd.Timestamp(OOT_END):
            continue

        if variant == 'F':
            # Buy call option: strike = ATM, expiry ~45 days out
            T_entry = 45 / 252
            T_exit = max(1 / 252, T_entry - (exit_day - entry_day) / 252)
            vol = series.pct_change().rolling(21).std().iloc[entry_day] * np.sqrt(252)
            if pd.isna(vol) or vol < 0.1:
                vol = 0.40
            K = round(entry_price)
            call_entry = bs_call(entry_price, K, T_entry, RISK_FREE_RATE, vol) * BS_HAIRCUT
            call_exit = bs_call(exit_price, K, T_exit, RISK_FREE_RATE, vol) * BS_HAIRCUT
            if call_entry < 0.50:
                continue  # too cheap, not realistic
            n_contracts = max(1, int((INITIAL_CAPITAL * MAX_POS_PCT) / (call_entry * 100)))
            cost = n_contracts * call_entry * 100 + COMMISSION_OPTION * n_contracts * 2
            proceeds = n_contracts * call_exit * 100
            pnl = proceeds - cost
            trades.append({
                'ticker': ticker, 'entry_date': entry_date, 'exit_date': exit_date,
                'entry_price': call_entry, 'exit_price': call_exit,
                'pnl': pnl, 'cost': cost, 'ret': pnl / cost if cost > 0 else 0,
                'type': 'call_option', 'variant': variant,
            })
        else:
            # Equity trade
            n_shares = max(1, int((INITIAL_CAPITAL * MAX_POS_PCT) / entry_price))
            cost = n_shares * entry_price
            if cost > INITIAL_CAPITAL:
                n_shares = max(1, int(INITIAL_CAPITAL / entry_price))
                cost = n_shares * entry_price
            pnl = n_shares * (exit_price - entry_price)
            trades.append({
                'ticker': ticker, 'entry_date': entry_date, 'exit_date': exit_date,
                'entry_price': entry_price, 'exit_price': exit_price,
                'pnl': pnl, 'cost': cost, 'ret': pnl / cost if cost > 0 else 0,
                'type': 'equity', 'variant': variant,
            })

    return trades


def run_opex_pin_release(prices_df, variant='C'):
    """
    Options Expiration Pin Release strategy.
    After OpEx Friday, stocks tend to "unpin" and move directionally.
    Buy breakout direction Monday after OpEx, hold 5 days.
    """
    trades = []
    opex_dates = get_opex_dates(OOT_START, OOT_END)

    vol_filter = (variant == 'D')

    for opex_date in opex_dates:
        # Monday after OpEx
        monday = opex_date + timedelta(days=3)
        # Adjust if holiday
        for ticker in STOCK_UNIVERSE:
            col_close = (ticker, 'Close')
            col_vol = (ticker, 'Volume')
            if col_close not in prices_df.columns:
                continue

            series = prices_df[col_close].dropna()
            vol_series = prices_df[col_vol].dropna() if col_vol in prices_df.columns else None

            # Find nearest trading day to Monday
            valid_dates = series.index[series.index >= monday]
            if len(valid_dates) < 6:
                continue
            actual_monday = valid_dates[0]
            if (actual_monday - monday).days > 3:
                continue  # holiday week, skip

            # Need OpEx Friday price too
            pre_dates = series.index[series.index <= opex_date]
            if len(pre_dates) < 21:
                continue
            opex_idx = len(pre_dates) - 1

            opex_price = series.iloc[opex_idx]
            entry_price = series[actual_monday]

            # Volume filter: require OpEx Friday volume > 1.5x 20-day avg
            if vol_filter and vol_series is not None:
                pre_vol = vol_series.iloc[max(0, opex_idx - 20):opex_idx]
                opex_vol = vol_series.iloc[opex_idx] if opex_idx < len(vol_series) else 0
                if len(pre_vol) > 0 and opex_vol < 1.5 * pre_vol.mean():
                    continue

            # Determine direction: if stock moved against its trend on OpEx week,
            # expect reversal (unpin). Use 5-day pre-OpEx trend.
            pre_5d = series.iloc[max(0, opex_idx - 5):opex_idx + 1]
            if len(pre_5d) < 3:
                continue
            opex_week_ret = (pre_5d.iloc[-1] / pre_5d.iloc[0]) - 1.0

            # Buy in the direction of 20-day trend (pin release restores trend)
            pre_20d = series.iloc[max(0, opex_idx - 20):opex_idx + 1]
            trend_20d = (pre_20d.iloc[-1] / pre_20d.iloc[0]) - 1.0 if len(pre_20d) > 5 else 0

            go_long = trend_20d > 0

            # Exit after 5 trading days
            exit_idx = list(series.index).index(actual_monday) + 5
            if exit_idx >= len(series):
                continue
            exit_date = series.index[exit_idx]
            exit_price = series.iloc[exit_idx]

            # Calculate P&L
            n_shares = max(1, int((INITIAL_CAPITAL * MAX_POS_PCT * 0.5) / entry_price))
            if go_long:
                pnl = n_shares * (exit_price - entry_price)
            else:
                # Short: need to check if stock is above $5 for shorting
                if entry_price < 5:
                    continue
                pnl = n_shares * (entry_price - exit_price)

            cost = n_shares * entry_price
            if cost > INITIAL_CAPITAL:
                continue

            trades.append({
                'ticker': ticker, 'entry_date': actual_monday, 'exit_date': exit_date,
                'entry_price': entry_price, 'exit_price': exit_price,
                'pnl': pnl, 'cost': cost, 'ret': pnl / cost if cost > 0 else 0,
                'direction': 'long' if go_long else 'short',
                'type': 'equity', 'variant': variant,
            })

    return trades


def run_combined(prices_df):
    """Variant E: Both IPO drift and OpEx signals active."""
    ipo_trades = run_ipo_drift(prices_df, 'A')
    opex_trades = run_opex_pin_release(prices_df, 'C')
    # Tag them
    for t in ipo_trades:
        t['variant'] = 'E'
        t['sub'] = 'ipo'
    for t in opex_trades:
        t['variant'] = 'E'
        t['sub'] = 'opex'
    return ipo_trades + opex_trades


# ==================== WALK-FORWARD BACKTEST ====================

def backtest_variant(trades, variant_name):
    """Run walk-forward backtest with sliding capital management."""
    if not trades:
        return None

    trades_df = pd.DataFrame(trades)
    trades_df['entry_date'] = pd.to_datetime(trades_df['entry_date'])
    trades_df['exit_date'] = pd.to_datetime(trades_df['exit_date'])
    trades_df = trades_df.sort_values('entry_date').reset_index(drop=True)

    # Walk-forward: simulate with capital constraints
    capital = INITIAL_CAPITAL
    peak_capital = capital
    max_dd = 0
    daily_returns = []
    realized_trades = []
    active_positions = []

    # Create daily equity curve
    all_dates = pd.bdate_range(OOT_START, OOT_END)
    equity_curve = pd.Series(index=all_dates, dtype=float)
    equity_curve.iloc[0] = capital

    trade_idx = 0
    for i, date in enumerate(all_dates):
        # Close expired positions
        new_active = []
        for pos in active_positions:
            if date >= pos['exit_date']:
                capital += pos['pnl'] + pos['cost']  # return capital + P&L
                realized_trades.append(pos)
            else:
                new_active.append(pos)
        active_positions = new_active

        # Open new positions (max 3 concurrent)
        while trade_idx < len(trades_df) and trades_df.iloc[trade_idx]['entry_date'] <= date:
            t = trades_df.iloc[trade_idx].to_dict()
            trade_idx += 1
            if len(active_positions) >= 3:
                continue
            if t['cost'] > capital:
                continue
            capital -= t['cost']
            active_positions.append(t)

        # Mark-to-market
        total_value = capital + sum(p['cost'] + p['pnl'] * min(1.0, max(0.0,
            (date - p['entry_date']).days / max(1, (p['exit_date'] - p['entry_date']).days)))
            for p in active_positions)
        equity_curve.iloc[i] = total_value
        peak_capital = max(peak_capital, total_value)
        dd = (total_value - peak_capital) / peak_capital if peak_capital > 0 else 0
        max_dd = min(max_dd, dd)

    # Calculate metrics
    equity_curve = equity_curve.dropna().ffill()
    if len(equity_curve) < 20:
        return None

    daily_rets = equity_curve.pct_change().dropna()
    daily_rets = daily_rets.replace([np.inf, -np.inf], 0)

    n_trades = len(realized_trades)
    if n_trades == 0:
        return None

    wins = sum(1 for t in realized_trades if t['pnl'] > 0)
    losses = sum(1 for t in realized_trades if t['pnl'] <= 0)
    win_rate = wins / n_trades if n_trades > 0 else 0
    total_pnl = sum(t['pnl'] for t in realized_trades)
    gross_profit = sum(t['pnl'] for t in realized_trades if t['pnl'] > 0)
    gross_loss = abs(sum(t['pnl'] for t in realized_trades if t['pnl'] < 0))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    ann_ret = daily_rets.mean() * 252
    ann_vol = daily_rets.std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0

    downside = daily_rets[daily_rets < 0].std() * np.sqrt(252)
    sortino = ann_ret / downside if downside > 0 else 0

    final_capital = equity_curve.iloc[-1]

    # SPY regime analysis
    spy_daily = equity_curve.copy()  # placeholder for regime analysis

    return {
        'variant': variant_name,
        'n_trades': n_trades,
        'win_rate': win_rate,
        'total_pnl': total_pnl,
        'final_capital': final_capital,
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
    """Run 5-gate validation on backtest results."""
    if result is None:
        return {'pass': False, 'reason': 'No trades'}

    gates = {}

    # Gate 1: Sharpe > 0.5
    gates['sharpe_gt_0.5'] = result['sharpe'] > 0.5

    # Gate 2: Permutation test p < 0.05
    daily_rets = result['daily_returns']
    if len(daily_rets) > 30:
        observed_mean = daily_rets.mean()
        n_perms = 5000
        perm_means = []
        rets_arr = daily_rets.values.copy()
        for _ in range(n_perms):
            signs = np.random.choice([-1, 1], size=len(rets_arr))
            perm_means.append((rets_arr * signs).mean())
        perm_p = np.mean(np.array(perm_means) >= observed_mean)
        gates['perm_p_lt_0.05'] = perm_p < 0.05
        result['perm_p'] = perm_p
    else:
        gates['perm_p_lt_0.05'] = False
        result['perm_p'] = 1.0

    # Gate 3: Beats random (compare to random entry/exit with same holding period)
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

    # Gate 4: Regime gap < 0.50
    # Classify days by SPY direction
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


# ==================== MLFLOW LOGGING ====================

def log_to_mlflow(results):
    """Log all variant results to MLflow."""
    try:
        import mlflow
        mlflow.set_tracking_uri('http://localhost:5000')
        experiment_name = 'growth_research_fomo_momentum_v1'
        mlflow.set_experiment(experiment_name)

        for res in results:
            if res is None:
                continue
            with mlflow.start_run(run_name=f"fomo_momentum_{res['variant']}"):
                mlflow.log_param('strategy', 'fomo_momentum_v1')
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
                mlflow.set_tag('oot_period', f"{OOT_START} to {OOT_END}")

        log.info(f"Logged {len([r for r in results if r])} variants to MLflow")
    except Exception as e:
        log.warning(f"MLflow logging failed: {e}")


# ==================== MAIN ====================

def main():
    log.info("=" * 70)
    log.info("FOMO Momentum Strategy v1 — Starting backtest")
    log.info("=" * 70)

    # Load data
    prices_df = load_all_data()
    log.info(f"Data loaded: {prices_df.shape}")

    # Detect IPOs in our universe
    log.info("\n--- IPO Detection ---")
    for ticker in STOCK_UNIVERSE:
        ipo = detect_ipo_date(prices_df, ticker)
        if ipo:
            ret = get_first_month_return(prices_df, ticker, ipo)
            if ret is not None:
                log.info(f"  {ticker}: IPO ~{ipo.date()}, 1st month return: {ret:+.1%}")

    results = []

    # Variant A: IPO Drift Pure
    log.info("\n=== Variant A: IPO Drift Pure (buy day 60, sell day 120) ===")
    trades_a = run_ipo_drift(prices_df, 'A')
    log.info(f"  Generated {len(trades_a)} trades")
    res_a = backtest_variant(trades_a, 'A')
    if res_a:
        res_a = five_gate_validation(res_a)
        results.append(res_a)

    # Variant B: IPO Drift Conservative
    log.info("\n=== Variant B: IPO Drift Conservative (buy day 45, sell day 90, >30% gain) ===")
    trades_b = run_ipo_drift(prices_df, 'B')
    log.info(f"  Generated {len(trades_b)} trades")
    res_b = backtest_variant(trades_b, 'B')
    if res_b:
        res_b = five_gate_validation(res_b)
        results.append(res_b)

    # Variant C: OpEx Pin Release
    log.info("\n=== Variant C: OpEx Pin Release (buy Monday after OpEx, hold 5d) ===")
    trades_c = run_opex_pin_release(prices_df, 'C')
    log.info(f"  Generated {len(trades_c)} trades")
    res_c = backtest_variant(trades_c, 'C')
    if res_c:
        res_c = five_gate_validation(res_c)
        results.append(res_c)

    # Variant D: OpEx + Volume Filter
    log.info("\n=== Variant D: OpEx Pin Release + Volume Surge ===")
    trades_d = run_opex_pin_release(prices_df, 'D')
    log.info(f"  Generated {len(trades_d)} trades")
    res_d = backtest_variant(trades_d, 'D')
    if res_d:
        res_d = five_gate_validation(res_d)
        results.append(res_d)

    # Variant E: Combined
    log.info("\n=== Variant E: Combined IPO + OpEx ===")
    trades_e = run_combined(prices_df)
    log.info(f"  Generated {len(trades_e)} trades")
    res_e = backtest_variant(trades_e, 'E')
    if res_e:
        res_e = five_gate_validation(res_e)
        results.append(res_e)

    # Variant F: IPO Drift Options
    log.info("\n=== Variant F: IPO Drift with Call Options ===")
    trades_f = run_ipo_drift(prices_df, 'F')
    log.info(f"  Generated {len(trades_f)} trades")
    res_f = backtest_variant(trades_f, 'F')
    if res_f:
        res_f = five_gate_validation(res_f)
        results.append(res_f)

    # ==================== RESULTS ====================
    log.info("\n" + "=" * 90)
    log.info("FOMO MOMENTUM v1 — RESULTS SUMMARY")
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

    # Log to MLflow
    log_to_mlflow(results)

    # Save results
    output_file = os.path.join(LVL3_ROOT, 'scripts', 'growth_research', 'logs',
                               'fomo_momentum_v1_results.json')
    save_results = []
    for r in results:
        if r is None:
            continue
        save_r = {k: v for k, v in r.items()
                  if k not in ('trades', 'equity_curve', 'daily_returns')}
        save_r['gates'] = r.get('gates', {})
        save_results.append(save_r)
    with open(output_file, 'w') as f:
        json.dump(save_results, f, indent=2, default=str)

    log.info(f"\nResults saved. Done.")
    return results


if __name__ == '__main__':
    main()
