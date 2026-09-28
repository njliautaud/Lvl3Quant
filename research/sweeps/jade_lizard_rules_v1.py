#!/usr/bin/env python3
"""
Jade Lizard Rules Optimization v1
===================================
No ML — simple rule tweaks on top of the proven baseline (Sharpe 1.03, WR 81%, PF 1.46).

Tests: IV rank thresholds, VIX gates, position sizing modes, TP/SL levels,
       DTE, max positions, sector diversification.

Reuses Black-Scholes pricing and backtest engine from jade_lizard_expanded_v1.py.
Output: /home/nick/Lvl3Quant/output/jade_lizard_rules_v1/
MLflow: http://jupiter:5000, experiment "jade_lizard_rules"
"""

import os, sys, json, time, warnings, logging, itertools
from datetime import datetime, timedelta
from pathlib import Path
from copy import deepcopy

import numpy as np
import pandas as pd
from scipy.stats import norm
import mlflow

warnings.filterwarnings('ignore')

# ── Config ──────────────────────────────────────────────────────────────────
OUTPUT_DIR = Path("/home/nick/Lvl3Quant/output/jade_lizard_rules_v1")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_FILE = Path("/home/nick/Lvl3Quant/output/jade_lizard_expanded_v1/price_data_cache.parquet")

LOG_FILE = OUTPUT_DIR / "jade_lizard_rules.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("jade_lizard_rules")

MLFLOW_URI = "http://jupiter:5000"
EXPERIMENT_NAME = "jade_lizard_rules"

FULL_UNIVERSE = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "JPM", "GS", "BAC",
    "V", "MA", "UNH", "JNJ", "PG", "KO", "PEP", "MRK", "ABBV", "LLY",
    "HD", "COST", "WMT", "CRM", "AMD", "NFLX", "ADBE", "INTC", "CSCO", "QCOM",
    "XOM", "CVX", "PFE", "TMO", "ABT", "AVGO", "TXN", "MCD", "NKE", "DIS",
    "CMCSA", "T", "VZ", "NEE", "SO", "SHW", "LMT", "RTX", "CAT", "DE",
]

# Sector mapping for diversification rule
SECTOR_MAP = {
    "AAPL": "Tech", "MSFT": "Tech", "GOOGL": "Tech", "AMZN": "ConsDisc",
    "META": "Tech", "NVDA": "Tech", "TSLA": "ConsDisc", "JPM": "Fin",
    "GS": "Fin", "BAC": "Fin", "V": "Fin", "MA": "Fin",
    "UNH": "Health", "JNJ": "Health", "PG": "ConsStap", "KO": "ConsStap",
    "PEP": "ConsStap", "MRK": "Health", "ABBV": "Health", "LLY": "Health",
    "HD": "ConsDisc", "COST": "ConsStap", "WMT": "ConsStap", "CRM": "Tech",
    "AMD": "Tech", "NFLX": "Tech", "ADBE": "Tech", "INTC": "Tech",
    "CSCO": "Tech", "QCOM": "Tech", "XOM": "Energy", "CVX": "Energy",
    "PFE": "Health", "TMO": "Health", "ABT": "Health", "AVGO": "Tech",
    "TXN": "Tech", "MCD": "ConsDisc", "NKE": "ConsDisc", "DIS": "ConsDisc",
    "CMCSA": "Comm", "T": "Comm", "VZ": "Comm", "NEE": "Util",
    "SO": "Util", "SHW": "Materials", "LMT": "Indust", "RTX": "Indust",
    "CAT": "Indust", "DE": "Indust",
}

# Baseline params (from jade_lizard_expanded_v1)
BASELINE = {
    'put_delta': 0.30,
    'call_short_delta': 0.20,
    'call_long_delta': 0.10,
    'dte': 30,
    'iv_premium': 1.10,
    'risk_free_rate': 0.04,
    'max_positions': 5,
    'max_per_stock': 1,
    'position_risk_pct': 0.05,
    'commission_per_contract': 0.65,
    'contracts_per_trade': 1,
    'initial_capital': 100_000,
    'take_profit_pct': 0.50,
    'stop_loss_mult': 2.0,
    'vix_pause_threshold': 999,  # baseline has no VIX gate effectively
    'iv_rank_min': 0.0,          # baseline: no IV rank filter
    'sizing_mode': 'equal',      # equal, iv_proportional, vix_inverse
    'max_per_sector': 999,       # baseline: no sector cap
    'train_window': 504,
    'rebalance_freq': 21,
}


# ── Black-Scholes (copied from v1 to avoid import dependency) ──────────────
def bs_price(S, K, T, r, sigma, option_type="put"):
    if T <= 0 or sigma <= 0:
        return max(0, (K - S) if option_type == "put" else (S - K))
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == "call":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def bs_delta(S, K, T, r, sigma, option_type="put"):
    if T <= 0 or sigma <= 0:
        if option_type == "put":
            return -1.0 if S < K else 0.0
        return 1.0 if S > K else 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    if option_type == "call":
        return norm.cdf(d1)
    return norm.cdf(d1) - 1.0


def find_strike_for_delta(S, T, r, sigma, target_delta, option_type="put", tol=0.001):
    if option_type == "put":
        lo, hi = S * 0.70, S * 1.0
        target = -abs(target_delta)
    else:
        lo, hi = S * 1.0, S * 1.50
        target = abs(target_delta)
    for _ in range(100):
        mid = (lo + hi) / 2
        d = bs_delta(S, mid, T, r, sigma, option_type)
        if option_type == "put":
            if d < target:
                lo = mid
            else:
                hi = mid
        else:
            if d > target:
                lo = mid
            else:
                hi = mid
        if abs(d - target) < tol:
            break
    return mid


# ── Backtest Engine (parameterized version) ────────────────────────────────
class JadeLizardBacktest:
    def __init__(self, params):
        self.p = params
        self.initial_capital = params['initial_capital']
        self.capital = self.initial_capital
        self.positions = []
        self.trades = []
        self.equity_curve = []

    def _commission_cost(self, contracts=1):
        return 3 * self.p['commission_per_contract'] * contracts

    def _sector_count(self, sector):
        """Count open positions in a given sector."""
        return sum(1 for p in self.positions if SECTOR_MAP.get(p['ticker'], '?') == sector)

    def open_position(self, date, ticker, spot, iv, iv_rank, vix_val):
        if len(self.positions) >= self.p['max_positions']:
            return None
        if any(p['ticker'] == ticker for p in self.positions):
            return None

        # Sector diversification check
        sector = SECTOR_MAP.get(ticker, 'Unknown')
        if self._sector_count(sector) >= self.p['max_per_sector']:
            return None

        dte = self.p['dte']
        T = dte / 365.0
        r = self.p['risk_free_rate']

        put_strike = find_strike_for_delta(spot, T, r, iv, self.p['put_delta'], "put")
        call_short_strike = find_strike_for_delta(spot, T, r, iv, self.p['call_short_delta'], "call")
        call_long_strike = find_strike_for_delta(spot, T, r, iv, self.p['call_long_delta'], "call")

        if call_long_strike <= call_short_strike:
            call_long_strike = call_short_strike * 1.05

        put_premium = bs_price(spot, put_strike, T, r, iv, "put")
        call_short_premium = bs_price(spot, call_short_strike, T, r, iv, "call")
        call_long_premium = bs_price(spot, call_long_strike, T, r, iv, "call")

        net_credit = put_premium + call_short_premium - call_long_premium
        if net_credit <= 0:
            return None

        put_risk = (put_strike - net_credit) * 100
        call_spread_risk = (call_long_strike - call_short_strike) * 100 - net_credit * 100
        max_risk = max(put_risk, max(0, call_spread_risk))
        if max_risk <= 0:
            return None

        # Position sizing
        sizing = self.p['sizing_mode']
        risk_pct = self.p['position_risk_pct']
        if sizing == 'iv_proportional':
            # Scale up with IV rank (higher IV = bigger position)
            scale = 0.5 + iv_rank  # range [0.5, 1.5]
            risk_pct = self.p['position_risk_pct'] * scale
        elif sizing == 'vix_inverse':
            # Scale down with VIX (higher VIX = smaller position)
            scale = max(0.3, min(1.5, 20.0 / (vix_val + 1e-10)))
            risk_pct = self.p['position_risk_pct'] * scale

        max_contracts = max(1, int(self.capital * risk_pct / max_risk))
        contracts = min(max_contracts, self.p['contracts_per_trade'])

        commission = self._commission_cost(contracts)
        net_credit_total = net_credit * 100 * contracts - commission

        position = {
            'open_date': date,
            'ticker': ticker,
            'spot_at_open': spot,
            'put_strike': put_strike,
            'call_short_strike': call_short_strike,
            'call_long_strike': call_long_strike,
            'iv_at_open': iv,
            'iv_rank_at_open': iv_rank,
            'net_credit': net_credit,
            'net_credit_total': net_credit_total,
            'contracts': contracts,
            'max_risk': max_risk * contracts,
            'dte_remaining': dte,
            'commission': commission,
        }
        self.positions.append(position)
        return position

    def mark_to_market(self, date, spot_dict, days_elapsed=1):
        closed = []
        for pos in self.positions[:]:
            ticker = pos['ticker']
            if ticker not in spot_dict:
                continue

            spot = spot_dict[ticker]
            pos['dte_remaining'] -= days_elapsed

            T = max(pos['dte_remaining'] / 365.0, 1/365.0)
            iv = pos['iv_at_open']
            r = self.p['risk_free_rate']

            put_val = bs_price(spot, pos['put_strike'], T, r, iv, "put")
            call_short_val = bs_price(spot, pos['call_short_strike'], T, r, iv, "call")
            call_long_val = bs_price(spot, pos['call_long_strike'], T, r, iv, "call")

            current_debit = put_val + call_short_val - call_long_val
            pnl_per_share = pos['net_credit'] - current_debit
            pnl_total = pnl_per_share * 100 * pos['contracts']

            exit_reason = None

            # Take profit
            if pnl_total >= pos['net_credit_total'] * self.p['take_profit_pct']:
                exit_reason = "take_profit"
            # Stop loss
            elif pnl_total < -pos['net_credit_total'] * self.p['stop_loss_mult']:
                exit_reason = "stop_loss"
            # Expiration
            elif pos['dte_remaining'] <= 0:
                put_intrinsic = max(0, pos['put_strike'] - spot)
                call_short_intrinsic = max(0, spot - pos['call_short_strike'])
                call_long_intrinsic = max(0, spot - pos['call_long_strike'])
                pnl_per_share = pos['net_credit'] - (put_intrinsic + call_short_intrinsic - call_long_intrinsic)
                pnl_total = pnl_per_share * 100 * pos['contracts']
                exit_reason = "expiration"

            if exit_reason:
                close_commission = self._commission_cost(pos['contracts']) if exit_reason != "expiration" else 0
                pnl_total -= close_commission

                trade = {
                    'open_date': pos['open_date'],
                    'close_date': date,
                    'ticker': ticker,
                    'spot_at_open': pos['spot_at_open'],
                    'spot_at_close': spot,
                    'put_strike': pos['put_strike'],
                    'call_short_strike': pos['call_short_strike'],
                    'call_long_strike': pos['call_long_strike'],
                    'iv_at_open': pos['iv_at_open'],
                    'net_credit': pos['net_credit'],
                    'pnl': pnl_total,
                    'pnl_pct': pnl_total / (pos['max_risk'] + 1e-10),
                    'exit_reason': exit_reason,
                    'dte_at_close': pos['dte_remaining'],
                    'contracts': pos['contracts'],
                    'put_expired_worthless': 1 if spot > pos['put_strike'] else 0,
                }
                self.trades.append(trade)
                self.capital += pnl_total
                self.positions.remove(pos)
                closed.append(trade)

        self.equity_curve.append({'date': date, 'equity': self.capital, 'n_positions': len(self.positions)})
        return closed


# ── Feature Computation (same as v1, no ML) ────────────────────────────────
def compute_features(prices_df, ticker, vix_series, iv_premium=1.10, dte=30):
    try:
        if isinstance(prices_df.columns, pd.MultiIndex):
            close = prices_df[(ticker, 'Close')].dropna()
        else:
            close = prices_df['Close'].dropna()
    except (KeyError, TypeError):
        return None

    if len(close) < 300:
        return None

    df = pd.DataFrame(index=close.index)
    df['close'] = close
    df['ticker'] = ticker

    df['ret_1d'] = close.pct_change()
    df['rv_20d'] = df['ret_1d'].rolling(20).std() * np.sqrt(252)
    df['rv_60d'] = df['ret_1d'].rolling(60).std() * np.sqrt(252)

    # IV rank
    df['iv_rank'] = df['rv_20d'].rolling(252).apply(
        lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min() + 1e-10) if len(x) == 252 else np.nan
    )

    # VIX
    if vix_series is not None:
        df['vix'] = vix_series.reindex(df.index).ffill()
    else:
        df['vix'] = 20.0

    # Lag features by 1 day
    for col in ['rv_20d', 'rv_60d', 'iv_rank', 'vix']:
        df[col] = df[col].shift(1)

    return df


# ── Run Single Backtest ────────────────────────────────────────────────────
def run_backtest(all_features, params, label=""):
    """Run a full walk-forward backtest with given params. No ML."""
    bt = JadeLizardBacktest(params)
    df = all_features.dropna(subset=['rv_20d', 'iv_rank', 'vix'])
    dates = sorted(df.index.unique())

    start_idx = params['train_window']  # skip train window for consistency with v1
    if start_idx >= len(dates):
        return None

    rebal_counter = 0
    selected_tickers = set()

    for i in range(start_idx, len(dates)):
        date = dates[i]
        rebal_counter += 1

        # Monthly rebalance: select stocks with highest IV rank
        if rebal_counter >= params['rebalance_freq'] or not selected_tickers:
            rebal_counter = 0
            today_data = df[df.index == date].copy()
            if len(today_data) == 0:
                continue

            # Filter by IV rank threshold
            eligible = today_data[today_data['iv_rank'] >= params['iv_rank_min']]
            if len(eligible) == 0:
                eligible = today_data  # fallback: use all if none pass

            top = eligible.nlargest(params['max_positions'] * 2, 'iv_rank')
            selected_tickers = set(top['ticker'].values)

        # Daily: mark to market
        day_data = df[df.index == date]
        spot_dict = {}
        for _, row in day_data.iterrows():
            spot_dict[row['ticker']] = row['close']

        bt.mark_to_market(date, spot_dict)

        # VIX gate
        vix_val = day_data['vix'].iloc[0] if len(day_data) > 0 else 20
        if pd.isna(vix_val):
            vix_val = 20
        if vix_val > params['vix_pause_threshold']:
            continue

        # Open new positions
        for ticker in selected_tickers:
            if len(bt.positions) >= params['max_positions']:
                break
            if any(p['ticker'] == ticker for p in bt.positions):
                continue

            ticker_data = day_data[day_data['ticker'] == ticker]
            if len(ticker_data) == 0:
                continue

            row = ticker_data.iloc[0]
            spot = row['close']
            rv = row.get('rv_20d', 0.25)
            if pd.isna(rv) or rv <= 0:
                rv = 0.25
            iv = rv * params['iv_premium']
            iv_rank = row.get('iv_rank', 0.5)
            if pd.isna(iv_rank):
                iv_rank = 0.5

            bt.open_position(date, ticker, spot, iv, iv_rank, vix_val)

    # Close remaining at expiration
    if bt.positions:
        last_date = dates[-1]
        day_data = df[df.index == last_date]
        spot_dict = {row['ticker']: row['close'] for _, row in day_data.iterrows()}
        for pos in bt.positions[:]:
            pos['dte_remaining'] = 0
        bt.mark_to_market(last_date, spot_dict)

    return bt


# ── Analyze Results ────────────────────────────────────────────────────────
def analyze_results(bt, label=""):
    if not bt or not bt.trades:
        return {"label": label, "n_trades": 0, "sharpe": 0, "win_rate": 0,
                "profit_factor": 0, "max_drawdown": 0, "sortino": 0,
                "total_pnl": 0, "return_pct": 0}

    trades_df = pd.DataFrame(bt.trades)
    equity_df = pd.DataFrame(bt.equity_curve)

    n_trades = len(trades_df)
    win_rate = (trades_df['pnl'] > 0).mean()
    total_pnl = trades_df['pnl'].sum()
    avg_win = trades_df[trades_df['pnl'] > 0]['pnl'].mean() if (trades_df['pnl'] > 0).any() else 0
    avg_loss = trades_df[trades_df['pnl'] <= 0]['pnl'].mean() if (trades_df['pnl'] <= 0).any() else 0
    profit_factor = abs(avg_win * (trades_df['pnl'] > 0).sum()) / (abs(avg_loss * (trades_df['pnl'] <= 0).sum()) + 1e-10)

    if len(equity_df) > 1:
        equity_df['ret'] = equity_df['equity'].pct_change().fillna(0)
        daily_ret = equity_df['ret']
        sharpe = daily_ret.mean() / (daily_ret.std() + 1e-10) * np.sqrt(252)
        downside = daily_ret[daily_ret < 0].std()
        sortino = daily_ret.mean() / (downside + 1e-10) * np.sqrt(252)
        max_dd = (equity_df['equity'] / equity_df['equity'].cummax() - 1).min()
    else:
        sharpe = sortino = max_dd = 0

    exit_reasons = trades_df['exit_reason'].value_counts().to_dict()

    return {
        'label': label,
        'n_trades': n_trades,
        'total_pnl': round(total_pnl, 2),
        'win_rate': round(win_rate, 4),
        'profit_factor': round(profit_factor, 2),
        'sharpe': round(sharpe, 3),
        'sortino': round(sortino, 3),
        'max_drawdown': round(max_dd, 4),
        'return_pct': round((bt.capital / bt.initial_capital - 1) * 100, 2),
        'final_equity': round(bt.capital, 2),
        'exit_reasons': exit_reasons,
    }


# ── Regime Analysis ────────────────────────────────────────────────────────
def regime_analysis(bt, spy_returns):
    if not bt or not bt.trades or spy_returns is None or len(spy_returns) == 0:
        return {}

    trades_df = pd.DataFrame(bt.trades)
    trades_df['close_date'] = pd.to_datetime(trades_df['close_date'])

    spy_63d = spy_returns.rolling(63).sum()
    spy_63d = spy_63d[~spy_63d.index.duplicated(keep='first')]

    bull_pnls, bear_pnls = [], []
    for _, trade in trades_df.iterrows():
        dt = trade['close_date']
        idx_pos = spy_63d.index.get_indexer([dt], method='nearest')
        if len(idx_pos) > 0 and idx_pos[0] >= 0:
            regime_ret = float(spy_63d.iloc[idx_pos[0]])
            if regime_ret > 0:
                bull_pnls.append(trade['pnl'])
            else:
                bear_pnls.append(trade['pnl'])

    results = {
        'bull_n': len(bull_pnls),
        'bull_wr': round(np.mean([1 if p > 0 else 0 for p in bull_pnls]), 4) if bull_pnls else 0,
        'bear_n': len(bear_pnls),
        'bear_wr': round(np.mean([1 if p > 0 else 0 for p in bear_pnls]), 4) if bear_pnls else 0,
    }
    if bull_pnls and bear_pnls:
        bull_sharpe = np.mean(bull_pnls) / (np.std(bull_pnls) + 1e-10)
        bear_sharpe = np.mean(bear_pnls) / (np.std(bear_pnls) + 1e-10)
        disparity = abs(bull_sharpe - bear_sharpe) / (max(abs(bull_sharpe), abs(bear_sharpe)) + 1e-10)
        results['regime_disparity'] = round(disparity, 4)
        results['regime_agnostic'] = disparity < 0.50
    return results


def sub_period_analysis(bt):
    if not bt or not bt.trades:
        return {}
    trades_df = pd.DataFrame(bt.trades).sort_values('close_date')
    n = len(trades_df)
    if n < 20:
        return {'too_few_trades': True}
    third = n // 3
    results = {}
    for i, name in enumerate(['p1', 'p2', 'p3']):
        chunk = trades_df.iloc[i*third:(i+1)*third] if i < 2 else trades_df.iloc[2*third:]
        results[name] = {
            'n': len(chunk),
            'wr': round((chunk['pnl'] > 0).mean(), 4),
            'pnl': round(chunk['pnl'].sum(), 2),
        }
    results['all_profitable'] = all(results[p]['pnl'] > 0 for p in ['p1', 'p2', 'p3'])
    return results


def permutation_test(all_features, params, observed_sharpe, n_perms=200):
    """Shuffle entry timing to get null distribution."""
    log.info(f"Permutation test: {n_perms} shuffles vs observed Sharpe {observed_sharpe:.3f}")
    null_sharpes = []
    for i in range(n_perms):
        if i % 50 == 0:
            log.info(f"  Perm {i}/{n_perms}")
        # Shuffle IV rank to randomize stock selection
        shuffled = all_features.copy()
        shuffled['iv_rank'] = np.random.permutation(shuffled['iv_rank'].values)
        bt = run_backtest(shuffled, params, f"perm_{i}")
        res = analyze_results(bt)
        null_sharpes.append(res['sharpe'])

    null_sharpes = np.array(null_sharpes)
    p_value = (null_sharpes >= observed_sharpe).mean()
    return {
        'p_value': round(p_value, 4),
        'null_mean': round(null_sharpes.mean(), 3),
        'null_std': round(null_sharpes.std(), 3),
        'observed': round(observed_sharpe, 3),
        'sig_05': bool(p_value < 0.05),
        'sig_01': bool(p_value < 0.01),
    }


# ── Define All Experiments ─────────────────────────────────────────────────
def build_experiments():
    """Build list of (label, params_override) tuples."""
    experiments = []

    # 0. Baseline
    experiments.append(("baseline", {}))

    # 1. IV Rank thresholds
    for iv_min in [0.30, 0.40, 0.50, 0.60, 0.70, 0.80]:
        experiments.append((f"ivrank_min_{int(iv_min*100)}", {'iv_rank_min': iv_min}))

    # 2. VIX gates
    for vix_max in [30, 35, 40]:
        experiments.append((f"vix_gate_{vix_max}", {'vix_pause_threshold': vix_max}))

    # 3. Position sizing modes
    for mode in ['iv_proportional', 'vix_inverse']:
        experiments.append((f"sizing_{mode}", {'sizing_mode': mode}))

    # 4. Take profit levels
    for tp in [0.40, 0.60, 0.75]:
        experiments.append((f"tp_{int(tp*100)}", {'take_profit_pct': tp}))

    # 5. Stop loss levels
    for sl in [1.5, 2.5, 3.0]:
        experiments.append((f"sl_{sl}x", {'stop_loss_mult': sl}))

    # 6. DTE
    for dte in [21, 45]:
        experiments.append((f"dte_{dte}", {'dte': dte}))

    # 7. Max positions
    for mp in [3, 7, 10]:
        experiments.append((f"maxpos_{mp}", {'max_positions': mp}))

    # 8. Sector diversification
    experiments.append(("sector_max2", {'max_per_sector': 2}))

    # Combo experiments (top potential combos)
    experiments.append(("combo_ivr50_vix35", {'iv_rank_min': 0.50, 'vix_pause_threshold': 35}))
    experiments.append(("combo_ivr50_tp60", {'iv_rank_min': 0.50, 'take_profit_pct': 0.60}))
    experiments.append(("combo_ivr50_sector2", {'iv_rank_min': 0.50, 'max_per_sector': 2}))
    experiments.append(("combo_ivr50_maxpos7", {'iv_rank_min': 0.50, 'max_positions': 7}))
    experiments.append(("combo_tp60_sl25", {'take_profit_pct': 0.60, 'stop_loss_mult': 2.5}))
    experiments.append(("combo_ivr50_tp60_sector2", {'iv_rank_min': 0.50, 'take_profit_pct': 0.60, 'max_per_sector': 2}))
    experiments.append(("combo_ivr50_tp60_maxpos7", {'iv_rank_min': 0.50, 'take_profit_pct': 0.60, 'max_positions': 7}))
    experiments.append(("combo_full_best", {'iv_rank_min': 0.50, 'take_profit_pct': 0.60, 'stop_loss_mult': 2.5,
                                             'max_per_sector': 2, 'max_positions': 7}))

    return experiments


# ── Main ───────────────────────────────────────────────────────────────────
def main():
    start_time = time.time()
    log.info("=" * 80)
    log.info("JADE LIZARD RULES OPTIMIZATION v1")
    log.info("No ML — simple rule tweaks on proven baseline")
    log.info("=" * 80)

    # Load cached price data
    log.info(f"Loading price data from {CACHE_FILE}")
    if not CACHE_FILE.exists():
        log.error("Cache file not found! Run jade_lizard_expanded_v1.py first.")
        sys.exit(1)
    prices = pd.read_parquet(CACHE_FILE)
    log.info(f"  Price data shape: {prices.shape}")

    # Extract VIX
    try:
        if isinstance(prices.columns, pd.MultiIndex):
            vix = prices[("^VIX", "Close")].dropna()
        else:
            vix = None
    except:
        vix = None

    # Compute features for all tickers
    log.info("Computing features for all tickers...")
    all_features = []
    for ticker in FULL_UNIVERSE:
        feats = compute_features(prices, ticker, vix)
        if feats is not None:
            all_features.append(feats)
        else:
            log.warning(f"  Skipped {ticker}: insufficient data")

    all_features_df = pd.concat(all_features)
    log.info(f"  Total rows: {len(all_features_df)}, tickers: {all_features_df['ticker'].nunique()}")

    # Fetch SPY for regime analysis
    log.info("Fetching SPY for regime analysis...")
    try:
        import yfinance as yf
        spy = yf.download("SPY", start="2012-01-01", end="2026-07-21", auto_adjust=True, progress=False)
        if isinstance(spy.columns, pd.MultiIndex):
            spy_close = spy[('Close', 'SPY')].dropna()
        else:
            spy_close = spy['Close'].dropna()
        spy_returns = spy_close.pct_change().dropna()
    except Exception as e:
        log.warning(f"SPY fetch failed: {e}")
        spy_returns = None

    # Setup MLflow
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    # Run all experiments
    experiments = build_experiments()
    log.info(f"\nRunning {len(experiments)} experiments...")

    all_results = []
    for idx, (label, overrides) in enumerate(experiments):
        params = {**BASELINE, **overrides}
        log.info(f"\n[{idx+1}/{len(experiments)}] {label}")
        log.info(f"  Overrides: {overrides if overrides else 'none (baseline)'}")

        bt = run_backtest(all_features_df, params, label)
        results = analyze_results(bt, label)
        results['params_override'] = overrides
        all_results.append(results)

        log.info(f"  -> Sharpe={results['sharpe']}, WR={results['win_rate']}, "
                 f"PF={results['profit_factor']}, MaxDD={results['max_drawdown']}, "
                 f"N={results['n_trades']}, Ret={results['return_pct']}%")

        # Log to MLflow
        with mlflow.start_run(run_name=f"rules_{label}", nested=False):
            mlflow.log_params({k: str(v) for k, v in params.items() if k != 'sizing_mode'})
            mlflow.log_param("sizing_mode", params['sizing_mode'])
            mlflow.log_param("variant", label)
            for metric_key in ['sharpe', 'sortino', 'win_rate', 'profit_factor',
                               'max_drawdown', 'n_trades', 'total_pnl', 'return_pct']:
                mlflow.log_metric(metric_key, results.get(metric_key, 0))

    # ── Rank Results ───────────────────────────────────────────────────
    log.info("\n" + "=" * 80)
    log.info("ALL RESULTS RANKED BY SHARPE")
    log.info("=" * 80)
    ranked = sorted(all_results, key=lambda x: x['sharpe'], reverse=True)

    log.info(f"{'Rank':<5} {'Label':<35} {'Sharpe':>7} {'WR':>7} {'PF':>6} {'MaxDD':>8} {'N':>5} {'Ret%':>8}")
    log.info("-" * 85)
    for i, r in enumerate(ranked):
        log.info(f"{i+1:<5} {r['label']:<35} {r['sharpe']:>7.3f} {r['win_rate']:>7.1%} "
                 f"{r['profit_factor']:>6.2f} {r['max_drawdown']:>8.2%} {r['n_trades']:>5} {r['return_pct']:>7.1f}%")

    # Save full results
    results_path = OUTPUT_DIR / "all_results.json"
    with open(results_path, 'w') as f:
        json.dump(ranked, f, indent=2, default=str)
    log.info(f"\nAll results saved to {results_path}")

    # ── Validate Top 3 ─────────────────────────────────────────────────
    log.info("\n" + "=" * 80)
    log.info("VALIDATION OF TOP 3 VARIANTS")
    log.info("=" * 80)

    top3_validated = []
    for rank, r in enumerate(ranked[:3]):
        label = r['label']
        overrides = r.get('params_override', {})
        params = {**BASELINE, **overrides}
        log.info(f"\n--- Validating #{rank+1}: {label} (Sharpe={r['sharpe']}) ---")

        # Re-run to get backtest object
        bt = run_backtest(all_features_df, params, label)

        # 1. Permutation test
        perm = permutation_test(all_features_df, params, r['sharpe'], n_perms=200)
        log.info(f"  Permutation: p={perm['p_value']}, null_mean={perm['null_mean']}, sig@5%={perm['sig_05']}")

        # 2. Regime analysis
        regime = regime_analysis(bt, spy_returns)
        log.info(f"  Regime: bull_wr={regime.get('bull_wr',0):.1%}, bear_wr={regime.get('bear_wr',0):.1%}, "
                 f"disparity={regime.get('regime_disparity','N/A')}, agnostic={regime.get('regime_agnostic','N/A')}")

        # 3. Sub-period stability
        subperiod = sub_period_analysis(bt)
        log.info(f"  Sub-periods: {subperiod}")

        validated = {
            'rank': rank + 1,
            'label': label,
            'metrics': r,
            'permutation': perm,
            'regime': regime,
            'sub_period': subperiod,
            'passes_validation': (
                perm.get('sig_05', False) and
                regime.get('regime_agnostic', False) and
                subperiod.get('all_profitable', False)
            ),
        }
        top3_validated.append(validated)

        # Save trades for top 3
        if bt and bt.trades:
            trades_path = OUTPUT_DIR / f"trades_{label}.csv"
            pd.DataFrame(bt.trades).to_csv(trades_path, index=False)
            log.info(f"  Trades saved: {trades_path}")

        if bt and bt.equity_curve:
            eq_path = OUTPUT_DIR / f"equity_{label}.csv"
            pd.DataFrame(bt.equity_curve).to_csv(eq_path, index=False)

    # Save validation results
    val_path = OUTPUT_DIR / "top3_validation.json"
    with open(val_path, 'w') as f:
        json.dump(top3_validated, f, indent=2, default=str)

    # Log top3 to MLflow
    with mlflow.start_run(run_name="rules_summary"):
        for v in top3_validated:
            prefix = f"top{v['rank']}"
            mlflow.log_metric(f"{prefix}_sharpe", v['metrics']['sharpe'])
            mlflow.log_metric(f"{prefix}_wr", v['metrics']['win_rate'])
            mlflow.log_metric(f"{prefix}_pf", v['metrics']['profit_factor'])
            mlflow.log_metric(f"{prefix}_maxdd", v['metrics']['max_drawdown'])
            mlflow.log_param(f"{prefix}_label", v['label'])
            mlflow.log_metric(f"{prefix}_perm_pval", v['permutation'].get('p_value', 1))
            mlflow.log_metric(f"{prefix}_regime_disp", v['regime'].get('regime_disparity', 1))
            mlflow.log_metric(f"{prefix}_passes", 1 if v['passes_validation'] else 0)
        mlflow.log_artifact(str(results_path))
        mlflow.log_artifact(str(val_path))

    # ── Final Summary ──────────────────────────────────────────────────
    elapsed = (time.time() - start_time) / 60
    log.info("\n" + "=" * 80)
    log.info("FINAL SUMMARY")
    log.info("=" * 80)
    log.info(f"Total experiments: {len(experiments)}")
    log.info(f"Runtime: {elapsed:.1f} minutes")

    log.info("\nTop 3 with validation status:")
    for v in top3_validated:
        status = "PASS" if v['passes_validation'] else "FAIL"
        log.info(f"  #{v['rank']} {v['label']}: Sharpe={v['metrics']['sharpe']}, "
                 f"WR={v['metrics']['win_rate']:.1%}, PF={v['metrics']['profit_factor']}, "
                 f"perm_p={v['permutation']['p_value']}, regime_ok={v['regime'].get('regime_agnostic','?')}, "
                 f"subperiod_ok={v['sub_period'].get('all_profitable','?')} => [{status}]")

    baseline_res = next(r for r in all_results if r['label'] == 'baseline')
    best_res = ranked[0]
    log.info(f"\nBaseline: Sharpe={baseline_res['sharpe']}, WR={baseline_res['win_rate']:.1%}, PF={baseline_res['profit_factor']}")
    log.info(f"Best:     Sharpe={best_res['sharpe']}, WR={best_res['win_rate']:.1%}, PF={best_res['profit_factor']} ({best_res['label']})")
    improvement = best_res['sharpe'] - baseline_res['sharpe']
    log.info(f"Improvement: {improvement:+.3f} Sharpe")

    log.info("\nDone.")


if __name__ == "__main__":
    main()
