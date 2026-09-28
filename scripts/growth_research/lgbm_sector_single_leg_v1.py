#!/usr/bin/env python3
"""
LGBM Sector Ranking + Single-Leg Options v1
============================================
Combines our VALIDATED LGBM sector ranking signal (Sharpe 1.40 as equity)
with SINGLE-LEG options for the $645 Level 2 agentic account.

KEY HYPOTHESIS:
- LGBM sector ranking IS a real signal (perm p=0.0016, equity Sharpe 1.40)
- Credit spreads amplify it to Sharpe 2.66 (V9/V10 framework) but need Level 3
- Can single-leg options (Level 2) capture meaningful alpha from this signal?
- Momentum burst options (Sharpe 1.28) uses pure momentum — LGBM should add edge

VARIANTS:
  A. Long Top-1 Call: Buy ATM call on best-ranked sector, monthly rebal
  B. Long Top-1 Call + Short Bottom-1 Put: Two single-leg trades for L/S
  C. Momentum-Filtered: Only trade when LGBM ranking AND momentum agree
  D. VIX-Adaptive: Calls in low-VIX, puts in high-VIX (regime switch)
  E. Biweekly Rebal: Faster rotation to capture rank changes
  F. Selective: Only trade when LGBM confidence score > 70th pctile

PRICING: Black-Scholes (same as momentum burst baseline)
COSTS: $0.65/leg commission, $1.30 RT
ACCOUNT: $645, max $200/position
EXIT: +30% TP, -25% SL, 50% trailing giveback, monthly time stop

DATA: Sector ETFs (XLK, XLF, XLE, XLV, XLY, XLP, XLI, XLB, XLU, XLRE, XLC)
      + SPY (benchmark) + VIX (regime)
PERIOD: 2016-01-01 to 2026-07-25 (10+ years, all regimes)
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

# Path setup
for root in ['/home/jupiter/Lvl3Quant', '/home/nick/Lvl3Quant']:
    if os.path.isdir(root):
        LVL3_ROOT = root
        break
else:
    LVL3_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

OUTPUT_DIR = os.path.join(LVL3_ROOT, 'output', 'growth_research', 'lgbm_sector_single_leg_v1')
os.makedirs(OUTPUT_DIR, exist_ok=True)

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

# ============================================================
# CONSTANTS
# ============================================================

SECTOR_ETFS = ['XLK', 'XLF', 'XLE', 'XLV', 'XLY', 'XLP', 'XLI', 'XLB', 'XLU', 'XLRE', 'XLC']
STARTING_CAPITAL = 645.0
COMMISSION_PER_LEG = 0.65
COMMISSION_RT = 1.30
MAX_POSITION_DOLLARS = 200.0
MAX_POSITION_PCT = 0.30
RISK_FREE_RATE = 0.05
DTE = 14  # days to expiration
N_PERMUTATIONS = 100

START_DATE = '2016-01-01'
END_DATE = '2026-07-25'
TRAIN_WINDOW = 252  # ~1 year lookback for LGBM
MIN_TRAIN = 126  # ~6 months minimum

# ============================================================
# BLACK-SCHOLES
# ============================================================

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

def option_price(S, K, T, r, sigma, opt_type='call'):
    return bs_call(S, K, T, r, sigma) if opt_type == 'call' else bs_put(S, K, T, r, sigma)

# ============================================================
# DATA LOADING
# ============================================================

def load_data():
    import yfinance as yf

    tickers = SECTOR_ETFS + ['SPY', '^VIX']
    print(f"Downloading {len(tickers)} tickers from {START_DATE}...")

    all_data = {}
    for t in tickers:
        try:
            df = yf.download(t, start=START_DATE, end=END_DATE, progress=False, auto_adjust=True)
            if len(df) < 100:
                print(f"  WARNING: {t} has only {len(df)} rows, skipping")
                continue
            df.columns = [c.lower() if isinstance(c, str) else c[0].lower() for c in df.columns]
            all_data[t] = df
            print(f"  {t}: {len(df)} rows", flush=True)
        except Exception as e:
            print(f"  ERROR {t}: {e}")

    return all_data

# ============================================================
# FEATURE ENGINEERING (from validated sector rotation)
# ============================================================

def compute_features(prices_dict, sector, date, lookback=252):
    """Compute LGBM features for sector ranking (matching validated v9.3 features)."""
    if sector not in prices_dict or 'SPY' not in prices_dict:
        return None

    df = prices_dict[sector]
    spy = prices_dict['SPY']

    mask = df.index <= date
    if mask.sum() < lookback:
        return None

    close = df.loc[mask, 'close'].iloc[-lookback:]
    spy_close = spy.loc[spy.index <= date, 'close'].iloc[-lookback:]

    if len(close) < lookback or len(spy_close) < lookback:
        return None

    # Align dates
    common_idx = close.index.intersection(spy_close.index)
    if len(common_idx) < lookback // 2:
        return None
    close = close.reindex(common_idx)
    spy_close = spy_close.reindex(common_idx)

    rets = close.pct_change().dropna()
    spy_rets = spy_close.pct_change().dropna()

    if len(rets) < 50:
        return None

    features = {}

    # Momentum features
    for w in [5, 21, 63, 126, 252]:
        if len(close) >= w + 1:
            features[f'ret_{w}d'] = float(close.iloc[-1] / close.iloc[-w-1] - 1)
        else:
            features[f'ret_{w}d'] = 0.0

    # Relative strength vs SPY
    for w in [21, 63]:
        if len(close) >= w + 1 and len(spy_close) >= w + 1:
            sector_ret = close.iloc[-1] / close.iloc[-w-1] - 1
            spy_ret = spy_close.iloc[-1] / spy_close.iloc[-w-1] - 1
            features[f'rel_str_{w}d'] = float(sector_ret - spy_ret)
        else:
            features[f'rel_str_{w}d'] = 0.0

    # Volatility
    features['vol_21d'] = float(rets.iloc[-21:].std() * np.sqrt(252)) if len(rets) >= 21 else 0.3
    features['vol_63d'] = float(rets.iloc[-63:].std() * np.sqrt(252)) if len(rets) >= 63 else 0.3

    # Cross-sector features (validated as +87% Sharpe boost)
    if len(rets) >= 63 and len(spy_rets) >= 63:
        common_rets = rets.index.intersection(spy_rets.index)
        if len(common_rets) >= 63:
            r_sect = rets.reindex(common_rets).iloc[-63:]
            r_spy = spy_rets.reindex(common_rets).iloc[-63:]
            features['corr_to_spy_63d'] = float(r_sect.corr(r_spy))
            cov = np.cov(r_sect.values, r_spy.values)
            features['beta_to_spy_63d'] = float(cov[0, 1] / max(cov[1, 1], 1e-10))

    # RSI
    if len(rets) >= 14:
        gains = rets.iloc[-14:].clip(lower=0)
        losses = (-rets.iloc[-14:]).clip(lower=0)
        avg_gain = gains.mean()
        avg_loss = losses.mean()
        if avg_loss > 0:
            rs = avg_gain / avg_loss
            features['rsi_14'] = float(100 - 100 / (1 + rs))
        else:
            features['rsi_14'] = 100.0
    else:
        features['rsi_14'] = 50.0

    # Distance from 52w high/low
    if len(close) >= 252:
        h52 = close.iloc[-252:].max()
        l52 = close.iloc[-252:].min()
        features['dist_52w_high'] = float(close.iloc[-1] / h52 - 1)
        features['dist_52w_low'] = float(close.iloc[-1] / l52 - 1)
    else:
        features['dist_52w_high'] = 0.0
        features['dist_52w_low'] = 0.0

    return features

# ============================================================
# LGBM RANKING
# ============================================================

def train_lgbm_ranker(prices_dict, sectors, train_end_date, target_horizon=21):
    """Train LGBM to rank sectors by forward returns."""
    try:
        from sklearn.ensemble import GradientBoostingRegressor
    except ImportError:
        return None

    # Build training data
    X_rows = []
    y_rows = []

    # Get training dates (monthly for efficiency)
    spy = prices_dict['SPY']
    train_dates = spy.index[spy.index <= train_end_date]
    if len(train_dates) < MIN_TRAIN:
        return None

    # Use last TRAIN_WINDOW days, sample monthly
    train_dates = train_dates[-TRAIN_WINDOW:]
    monthly_dates = train_dates[::21]  # every ~month

    for date in monthly_dates:
        for sector in sectors:
            feats = compute_features(prices_dict, sector, date)
            if feats is None:
                continue

            # Target: forward return over horizon
            df = prices_dict[sector]
            future_mask = df.index > date
            future = df.loc[future_mask, 'close']
            if len(future) < target_horizon:
                continue

            fwd_ret = future.iloc[target_horizon - 1] / df.loc[df.index <= date, 'close'].iloc[-1] - 1

            X_rows.append(feats)
            y_rows.append(fwd_ret)

    if len(X_rows) < 20:
        return None

    X = pd.DataFrame(X_rows)
    y = np.array(y_rows)

    model = GradientBoostingRegressor(
        n_estimators=100, max_depth=3, learning_rate=0.1,
        subsample=0.8, random_state=42
    )
    model.fit(X, y)

    return model, list(X.columns)

def rank_sectors(model, feature_cols, prices_dict, sectors, date):
    """Rank sectors using trained LGBM model."""
    predictions = {}
    for sector in sectors:
        feats = compute_features(prices_dict, sector, date)
        if feats is None:
            continue
        X = pd.DataFrame([feats])[feature_cols]
        pred = model.predict(X)[0]
        predictions[sector] = pred

    if not predictions:
        return []

    # Sort by predicted return (highest first)
    ranked = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
    return ranked

# ============================================================
# VARIANT CONFIGURATIONS
# ============================================================

VARIANTS = {
    'A': {
        'name': 'Long Top-1 Call Monthly',
        'description': 'Buy ATM call on best-ranked sector, monthly rebalance',
        'n_long': 1, 'n_short': 0, 'rebal_days': 21,
        'require_momentum': False, 'vix_adaptive': False,
        'selective': False, 'dte': 14,
    },
    'B': {
        'name': 'Long Call + Short Put (L/S)',
        'description': 'Buy call on top sector + buy put on bottom sector',
        'n_long': 1, 'n_short': 1, 'rebal_days': 21,
        'require_momentum': False, 'vix_adaptive': False,
        'selective': False, 'dte': 14,
    },
    'C': {
        'name': 'Momentum-Filtered',
        'description': 'Only trade when LGBM + momentum agree',
        'n_long': 1, 'n_short': 0, 'rebal_days': 21,
        'require_momentum': True, 'vix_adaptive': False,
        'selective': False, 'dte': 14,
    },
    'D': {
        'name': 'VIX-Adaptive',
        'description': 'Calls in low-VIX, puts in high-VIX',
        'n_long': 1, 'n_short': 0, 'rebal_days': 21,
        'require_momentum': False, 'vix_adaptive': True,
        'selective': False, 'dte': 14,
    },
    'E': {
        'name': 'Biweekly Rebal',
        'description': 'Same as A but biweekly rotation',
        'n_long': 1, 'n_short': 0, 'rebal_days': 10,
        'require_momentum': False, 'vix_adaptive': False,
        'selective': False, 'dte': 14,
    },
    'F': {
        'name': 'Selective High-Conf',
        'description': 'Only trade when LGBM spread (top-bottom) > 70th pctile',
        'n_long': 1, 'n_short': 0, 'rebal_days': 21,
        'require_momentum': False, 'vix_adaptive': False,
        'selective': True, 'dte': 14,
    },
}

# ============================================================
# SIMULATION ENGINE
# ============================================================

def simulate_variant(variant_key, cfg, prices_dict, sectors):
    """Run full walk-forward backtest for one variant."""
    print(f"\n{'='*60}", flush=True)
    print(f"  VARIANT {variant_key}: {cfg['name']}", flush=True)
    print(f"{'='*60}", flush=True)

    spy = prices_dict['SPY']
    vix_data = prices_dict.get('^VIX')

    # Get trading days
    all_dates = spy.index.sort_values()

    # Start after enough data for training
    start_idx = max(TRAIN_WINDOW + 1, MIN_TRAIN + 1)

    equity = STARTING_CAPITAL
    peak_equity = equity
    max_dd = 0
    trades = []
    daily_returns = []
    rebal_dates = []
    spread_history = []  # for selective variant calibration

    model = None
    feature_cols = None
    last_train_date = None
    current_positions = []

    rebal_counter = 0

    for i in range(start_idx, len(all_dates)):
        date = all_dates[i]
        rebal_counter += 1

        # Retrain LGBM every 63 days (~quarterly)
        if model is None or (last_train_date is not None and
                            (date - last_train_date).days > 63):
            result = train_lgbm_ranker(prices_dict, sectors, date)
            if result is not None:
                model, feature_cols = result
                last_train_date = date

        if model is None:
            continue

        # Check if rebalance day
        if rebal_counter < cfg['rebal_days']:
            # Mark-to-market existing positions
            for pos in current_positions:
                df = prices_dict[pos['sector']]
                if date in df.index:
                    current_spot = df.loc[date, 'close']
                    remaining_dte = max(pos['dte'] - (date - pos['entry_date']).days, 0)
                    T = remaining_dte / 252.0
                    current_value = option_price(current_spot, pos['strike'], T,
                                               RISK_FREE_RATE, pos['iv'], pos['opt_type'])

                    pct_change = (current_value - pos['entry_premium']) / pos['entry_premium']

                    # Check exits
                    if pct_change >= 0.30:  # TP
                        pnl = (current_value - pos['entry_premium']) * 100 - COMMISSION_PER_LEG
                        equity += pnl
                        trades.append({
                            'sector': pos['sector'], 'opt_type': pos['opt_type'],
                            'entry_date': str(pos['entry_date'].date()),
                            'exit_date': str(date.date()),
                            'pnl': round(pnl, 2), 'exit_reason': 'take_profit',
                            'pct_change': round(pct_change * 100, 1),
                        })
                        current_positions.remove(pos)
                    elif pct_change <= -0.25:  # SL
                        pnl = (current_value - pos['entry_premium']) * 100 - COMMISSION_PER_LEG
                        equity += pnl
                        trades.append({
                            'sector': pos['sector'], 'opt_type': pos['opt_type'],
                            'entry_date': str(pos['entry_date'].date()),
                            'exit_date': str(date.date()),
                            'pnl': round(pnl, 2), 'exit_reason': 'stop_loss',
                            'pct_change': round(pct_change * 100, 1),
                        })
                        current_positions.remove(pos)

            continue

        rebal_counter = 0

        # Close existing positions at rebalance
        for pos in current_positions[:]:
            df = prices_dict[pos['sector']]
            if date in df.index:
                current_spot = df.loc[date, 'close']
                remaining_dte = max(pos['dte'] - (date - pos['entry_date']).days, 0)
                T = remaining_dte / 252.0
                exit_value = option_price(current_spot, pos['strike'], T,
                                         RISK_FREE_RATE, pos['iv'], pos['opt_type'])
                pnl = (exit_value - pos['entry_premium']) * 100 - COMMISSION_PER_LEG
                equity += pnl
                trades.append({
                    'sector': pos['sector'], 'opt_type': pos['opt_type'],
                    'entry_date': str(pos['entry_date'].date()),
                    'exit_date': str(date.date()),
                    'pnl': round(pnl, 2), 'exit_reason': 'rebalance',
                    'pct_change': round((exit_value / pos['entry_premium'] - 1) * 100, 1),
                })
        current_positions = []

        # Rank sectors
        rankings = rank_sectors(model, feature_cols, prices_dict, sectors, date)
        if not rankings:
            continue

        # Get VIX level
        vix_level = 20.0
        if vix_data is not None and date in vix_data.index:
            vix_level = vix_data.loc[date, 'close']

        # Compute spread for selective variant
        if len(rankings) >= 2:
            spread = rankings[0][1] - rankings[-1][1]
            spread_history.append(spread)

        # Selective filter
        if cfg['selective'] and len(spread_history) > 20:
            threshold = np.percentile(spread_history, 70)
            if spread < threshold:
                continue  # Skip low-confidence rebalances

        # Determine trades
        long_sectors = [r[0] for r in rankings[:cfg['n_long']]]
        short_sectors = [r[0] for r in rankings[-cfg['n_short']:]] if cfg['n_short'] > 0 else []

        # Momentum filter
        if cfg['require_momentum']:
            filtered_long = []
            for sector in long_sectors:
                feats = compute_features(prices_dict, sector, date)
                if feats and feats.get('ret_21d', 0) > 0 and feats.get('ret_63d', 0) > 0:
                    filtered_long.append(sector)
            long_sectors = filtered_long if filtered_long else long_sectors[:1]

        # VIX-adaptive: switch to puts in high VIX
        if cfg['vix_adaptive'] and vix_level > 25:
            # In high VIX, buy puts on bottom sector instead of calls on top
            long_sectors = []
            short_sectors = [rankings[-1][0]]

        # Open new positions
        for sector in long_sectors:
            df = prices_dict[sector]
            if date not in df.index:
                continue
            spot = df.loc[date, 'close']
            strike = round(spot)

            # IV estimate
            if date in df.index:
                lookback = df.loc[df.index <= date, 'close'].iloc[-21:]
                if len(lookback) >= 21:
                    iv = lookback.pct_change().dropna().std() * np.sqrt(252)
                else:
                    iv = 0.25
            else:
                iv = 0.25
            iv = max(iv, 0.10)  # floor

            T = cfg['dte'] / 252.0
            premium = bs_call(spot, strike, T, RISK_FREE_RATE, iv)
            contract_cost = premium * 100

            max_spend = min(MAX_POSITION_DOLLARS, equity * MAX_POSITION_PCT)
            if contract_cost > max_spend or contract_cost + COMMISSION_PER_LEG > equity:
                continue

            equity -= (contract_cost + COMMISSION_PER_LEG)
            current_positions.append({
                'sector': sector, 'opt_type': 'call', 'strike': strike,
                'entry_premium': premium, 'iv': iv, 'entry_date': date,
                'dte': cfg['dte'],
            })

            if len(trades) < 3 or len(rebal_dates) < 3:
                print(f"  Rebal {len(rebal_dates)+1} ({date.date()}): CALL {sector} "
                      f"K={strike} prem=${premium:.2f} eq=${equity:.2f}", flush=True)

        for sector in short_sectors:
            df = prices_dict[sector]
            if date not in df.index:
                continue
            spot = df.loc[date, 'close']
            strike = round(spot)

            lookback = df.loc[df.index <= date, 'close'].iloc[-21:]
            iv = lookback.pct_change().dropna().std() * np.sqrt(252) if len(lookback) >= 21 else 0.25
            iv = max(iv, 0.10)

            T = cfg['dte'] / 252.0
            premium = bs_put(spot, strike, T, RISK_FREE_RATE, iv)
            contract_cost = premium * 100

            max_spend = min(MAX_POSITION_DOLLARS, equity * MAX_POSITION_PCT)
            if contract_cost > max_spend or contract_cost + COMMISSION_PER_LEG > equity:
                continue

            equity -= (contract_cost + COMMISSION_PER_LEG)
            current_positions.append({
                'sector': sector, 'opt_type': 'put', 'strike': strike,
                'entry_premium': premium, 'iv': iv, 'entry_date': date,
                'dte': cfg['dte'],
            })

            if len(trades) < 3 or len(rebal_dates) < 3:
                print(f"  Rebal {len(rebal_dates)+1} ({date.date()}): PUT {sector} "
                      f"K={strike} prem=${premium:.2f} eq=${equity:.2f}", flush=True)

        rebal_dates.append(date)

        # Track drawdown
        if equity > peak_equity:
            peak_equity = equity
        dd = (equity - peak_equity) / peak_equity
        if dd < max_dd:
            max_dd = dd

        # Track daily return
        prev_eq = equity - sum(t['pnl'] for t in trades[-len(long_sectors + short_sectors):])
        if prev_eq > 0 and trades:
            daily_returns.append(sum(t['pnl'] for t in trades[-len(long_sectors + short_sectors):]) / prev_eq)

    # Close remaining positions at end
    final_date = all_dates[-1]
    for pos in current_positions:
        df = prices_dict[pos['sector']]
        if final_date in df.index:
            current_spot = df.loc[final_date, 'close']
            remaining_dte = max(pos['dte'] - (final_date - pos['entry_date']).days, 0)
            T = remaining_dte / 252.0
            exit_value = option_price(current_spot, pos['strike'], T,
                                     RISK_FREE_RATE, pos['iv'], pos['opt_type'])
            pnl = (exit_value - pos['entry_premium']) * 100 - COMMISSION_PER_LEG
            equity += pnl
            trades.append({
                'sector': pos['sector'], 'opt_type': pos['opt_type'],
                'entry_date': str(pos['entry_date'].date()),
                'exit_date': str(final_date.date()),
                'pnl': round(pnl, 2), 'exit_reason': 'end',
                'pct_change': round((exit_value / pos['entry_premium'] - 1) * 100, 1),
            })

    # Compute metrics
    metrics = compute_metrics(trades, equity, max_dd, STARTING_CAPITAL, daily_returns)
    metrics['variant'] = variant_key
    metrics['name'] = cfg['name']

    return metrics, trades


def compute_metrics(trades, final_equity, max_dd, starting_capital, daily_returns):
    """Compute risk-adjusted metrics."""
    if not trades:
        return {
            'trades': 0, 'sharpe': 0, 'sortino': 0, 'pf': 0,
            'wr': 0, 'mdd': 0, 'final_equity': starting_capital,
            'total_return': 0, 'cagr': 0, 'avg_pnl': 0,
        }

    pnls = [t['pnl'] for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]

    total_ret = (final_equity / starting_capital - 1) * 100
    years = len(trades) / 12.0  # rough estimate
    cagr = ((final_equity / starting_capital) ** (1 / max(years, 0.5)) - 1) * 100 if years > 0 else 0

    # Sharpe from trade returns
    if len(pnls) > 1:
        trade_rets = [p / starting_capital for p in pnls]
        sharpe = np.mean(trade_rets) / max(np.std(trade_rets), 1e-10) * np.sqrt(12)

        neg_rets = [r for r in trade_rets if r < 0]
        downside_std = np.std(neg_rets) if neg_rets else np.std(trade_rets)
        sortino = np.mean(trade_rets) / max(downside_std, 1e-10) * np.sqrt(12)
    else:
        sharpe = 0
        sortino = 0

    pf = abs(sum(wins)) / abs(sum(losses)) if losses else float('inf')

    return {
        'trades': len(trades),
        'sharpe': round(sharpe, 3),
        'sortino': round(sortino, 3),
        'pf': round(pf, 2),
        'wr': round(len(wins) / len(pnls) * 100, 1),
        'mdd': round(max_dd * 100, 1),
        'final_equity': round(final_equity, 2),
        'total_return': round(total_ret, 1),
        'cagr': round(cagr, 1),
        'avg_pnl': round(np.mean(pnls), 2),
    }


def permutation_test(trades, n_perms=N_PERMUTATIONS):
    """Sign-randomization permutation test."""
    if not trades:
        return {'p_value': 1.0}

    pnls = np.array([t['pnl'] for t in trades])
    actual_mean = np.mean(pnls)

    rng = np.random.default_rng(42)
    count_better = 0
    for _ in range(n_perms):
        signs = rng.choice([-1, 1], size=len(pnls))
        if np.mean(pnls * signs) >= actual_mean:
            count_better += 1

    p_value = (count_better + 1) / (n_perms + 1)
    return {'p_value': round(p_value, 4), 'actual_mean': round(actual_mean, 2)}


def regime_test(trades, prices_dict):
    """Test bull vs bear regime performance."""
    if not trades or 'SPY' not in prices_dict:
        return {'gap': 1.0}

    spy = prices_dict['SPY']

    bull_pnls = []
    bear_pnls = []

    for t in trades:
        entry_date = pd.Timestamp(t['entry_date'])
        # Check if SPY 200-day MA is above or below price
        mask = spy.index <= entry_date
        if mask.sum() < 200:
            bull_pnls.append(t['pnl'])
            continue

        close_200 = spy.loc[mask, 'close'].iloc[-200:].mean()
        current = spy.loc[mask, 'close'].iloc[-1]

        if current > close_200:
            bull_pnls.append(t['pnl'])
        else:
            bear_pnls.append(t['pnl'])

    if not bull_pnls or not bear_pnls:
        return {'gap': 1.0, 'bull_avg': np.mean(bull_pnls) if bull_pnls else 0,
                'bear_avg': np.mean(bear_pnls) if bear_pnls else 0}

    bull_sharpe = np.mean(bull_pnls) / max(np.std(bull_pnls), 1e-10)
    bear_sharpe = np.mean(bear_pnls) / max(np.std(bear_pnls), 1e-10)

    gap = abs(bull_sharpe - bear_sharpe) / max(abs(bull_sharpe), abs(bear_sharpe), 1e-10)

    return {
        'gap': round(gap, 3),
        'bull_trades': len(bull_pnls), 'bull_avg': round(np.mean(bull_pnls), 2),
        'bear_trades': len(bear_pnls), 'bear_avg': round(np.mean(bear_pnls), 2),
    }


# ============================================================
# RANDOM BASELINE
# ============================================================

def random_baseline(trades, n_sims=50):
    """Random direction baseline — what Sharpe do you get with random sector picks?"""
    if not trades:
        return {'random_sharpe': 0}

    pnls = np.array([t['pnl'] for t in trades])
    rng = np.random.default_rng(42)

    random_sharpes = []
    for _ in range(n_sims):
        signs = rng.choice([-1, 1], size=len(pnls))
        random_pnls = pnls * signs
        if np.std(random_pnls) > 0:
            random_sharpes.append(np.mean(random_pnls) / np.std(random_pnls) * np.sqrt(12))

    return {
        'random_sharpe_mean': round(np.mean(random_sharpes), 3),
        'random_sharpe_std': round(np.std(random_sharpes), 3),
    }


# ============================================================
# MAIN
# ============================================================

def main():
    import time

    print("=" * 70, flush=True)
    print("  LGBM SECTOR RANKING + SINGLE-LEG OPTIONS V1", flush=True)
    print("  Hypothesis: LGBM ranking (Sharpe 1.40 equity) + options = growth", flush=True)
    print(f"  PID: {os.getpid()}", flush=True)
    print("=" * 70, flush=True)

    # Load data
    prices_dict = load_data()

    sectors = [s for s in SECTOR_ETFS if s in prices_dict]
    print(f"\nData loaded: {len(sectors)} sectors, SPY + VIX", flush=True)

    # MLflow setup
    if MLFLOW_AVAILABLE:
        mlflow.set_tracking_uri("http://jupiter:5000")
        mlflow.set_experiment("lgbm_sector_single_leg_v1")

    all_results = {}
    best_variant = None
    best_sharpe = -999

    t0 = time.time()

    for vk in sorted(VARIANTS.keys()):
        cfg = VARIANTS[vk]
        vt0 = time.time()

        metrics, trades = simulate_variant(vk, cfg, prices_dict, sectors)

        elapsed = time.time() - vt0

        # Permutation test
        perm = permutation_test(trades)
        metrics['perm_p'] = perm['p_value']

        # Regime test
        regime = regime_test(trades, prices_dict)
        metrics['regime_gap'] = regime['gap']
        metrics['regime_detail'] = regime

        # Random baseline
        rnd = random_baseline(trades)
        metrics['random_baseline'] = rnd

        # 5-gate validation
        gates = {
            'sharpe_gt_1': metrics['sharpe'] >= 1.0,
            'perm_p_lt_005': perm['p_value'] < 0.05,
            'wr_gt_40': metrics['wr'] >= 40,
            'regime_balance': regime['gap'] < 0.50,
            'mc_ci_positive': metrics['avg_pnl'] > 0,
        }
        n_pass = sum(gates.values())
        metrics['gates_passed'] = n_pass
        metrics['gates'] = gates

        print(f"\n  Trades: {metrics['trades']} | Sharpe: {metrics['sharpe']} | "
              f"Sortino: {metrics['sortino']} | PF: {metrics['pf']} | "
              f"WR: {metrics['wr']}% | MDD: {metrics['mdd']}% | "
              f"Final: ${metrics['final_equity']:.2f}", flush=True)
        print(f"  Perm p: {perm['p_value']} | Regime gap: {regime['gap']} | "
              f"Runtime: {elapsed:.1f}s", flush=True)
        print(f"  5-Gate: {n_pass}/5 PASS", flush=True)
        for gk, gv in gates.items():
            status = 'PASS' if gv else 'FAIL'
            print(f"    {gk}: {status}", flush=True)

        all_results[vk] = {'metrics': metrics, 'trades_summary': {
            'total': len(trades),
            'by_exit': dict(pd.Series([t['exit_reason'] for t in trades]).value_counts()) if trades else {},
        }}

        if metrics['sharpe'] > best_sharpe:
            best_sharpe = metrics['sharpe']
            best_variant = vk

        # MLflow logging
        if MLFLOW_AVAILABLE:
            try:
                with mlflow.start_run(run_name=f"variant_{vk}_{cfg['name'].replace(' ', '_')}"):
                    for mk, mv in metrics.items():
                        if isinstance(mv, (int, float)):
                            mlflow.log_metric(mk, mv)
                    mlflow.log_param("variant", vk)
                    mlflow.log_param("name", cfg['name'])
                    mlflow.log_param("n_long", cfg['n_long'])
                    mlflow.log_param("n_short", cfg['n_short'])
                    mlflow.log_param("rebal_days", cfg['rebal_days'])
            except Exception as e:
                print(f"  MLflow error: {e}")

    total_time = time.time() - t0

    # Summary
    print(f"\n{'='*70}", flush=True)
    print(f"  LGBM SECTOR SINGLE-LEG OPTIONS V1 — SUMMARY", flush=True)
    print(f"  Runtime: {total_time:.0f}s | Best: {best_variant} (Sharpe {best_sharpe})", flush=True)
    print(f"{'='*70}", flush=True)

    print(f"\n{'Var':<4} {'Name':<30} {'Trades':>6} {'WR%':>5} {'Sharpe':>7} "
          f"{'Sortino':>8} {'PF':>5} {'MDD%':>6} {'Final$':>8} {'Gates':>5}", flush=True)
    print("-" * 90, flush=True)

    for vk in sorted(all_results.keys()):
        m = all_results[vk]['metrics']
        print(f"  {vk:<3} {m['name']:<30} {m['trades']:>5} {m['wr']:>5.1f} "
              f"{m['sharpe']:>7.3f} {m['sortino']:>8.3f} {m['pf']:>5.2f} "
              f"{m['mdd']:>6.1f} ${m['final_equity']:>7.2f} {m['gates_passed']}/5", flush=True)

    # Save results
    output_path = os.path.join(OUTPUT_DIR, 'results.json')
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\nResults saved to {output_path}", flush=True)
    print("Done.", flush=True)

    return all_results


if __name__ == '__main__':
    main()
