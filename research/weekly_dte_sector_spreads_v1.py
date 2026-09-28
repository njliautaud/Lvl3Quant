#!/usr/bin/env python3
"""
Weekly DTE Sector Bull Call Spreads v1
======================================
Research question: Do 5-day DTE bull call spreads with weekly sector momentum rotation
outperform our validated 30-day DTE monthly rotation?

Hypothesis: Weekly DTE gives:
- Smaller max loss per trade (spread is narrower relative to price)
- Faster compounding (52 cycles/year vs 12)
- Better momentum capture (weekly signal more responsive)
- Lower capital tie-up per position

Uses same LightGBM sector ranking as validated strategies.
Realistic pricing: ATR-based spread widths, 15% haircut, $4.70 RT commission.

HC compliance:
- HC #428: Regime-agnostic + MFE-within-horizon
- HC #433: Plain English in Discord
- HC #750: Multi-signal confluence
- Permutation test + 4-gate adversarial audit
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from datetime import datetime
import json
import os
import sys
import traceback

# MLflow
try:
    import mlflow
    mlflow.set_tracking_uri("http://localhost:5000")
    HAS_MLFLOW = True
except:
    HAS_MLFLOW = False

# LightGBM
try:
    import lightgbm as lgb
    HAS_LGB = True
except:
    HAS_LGB = False

###############################################################################
# CONFIG
###############################################################################
UNIVERSE = ['XLK', 'XLF', 'XLE', 'XLV', 'XLY', 'XLI', 'XLP', 'XLU', 'XLRE', 'XLB', 'XLC']
START_CAP = 645.0
COMMISSION_RT = 4.70  # AMP round-trip
HAIRCUT = 0.15  # 15% slippage on B-S theoretical
RISK_FREE = 0.05

# Variants to test
CONFIGS = {
    'A_5d_Top3_Weekly': {
        'dte': 5, 'spread_pct': 2.0, 'top_k': 3, 'rebal_days': 5,
        'vix_min': 0, 'train_periods': 26, 'desc': 'Baseline weekly DTE'
    },
    'B_5d_Top3_VIX20': {
        'dte': 5, 'spread_pct': 2.0, 'top_k': 3, 'rebal_days': 5,
        'vix_min': 20, 'train_periods': 26, 'desc': '5d DTE + VIX>20 filter'
    },
    'C_5d_Top2_VIX20': {
        'dte': 5, 'spread_pct': 2.0, 'top_k': 2, 'rebal_days': 5,
        'vix_min': 20, 'train_periods': 26, 'desc': 'Concentrated top-2 + VIX filter'
    },
    'D_5d_Top3_3pct': {
        'dte': 5, 'spread_pct': 3.0, 'top_k': 3, 'rebal_days': 5,
        'vix_min': 0, 'train_periods': 26, 'desc': 'Wider 3% spreads'
    },
    'E_7d_Top3_Weekly': {
        'dte': 7, 'spread_pct': 2.0, 'top_k': 3, 'rebal_days': 5,
        'vix_min': 0, 'train_periods': 26, 'desc': '7-day DTE for more theta'
    },
    'F_5d_Top3_VIX25': {
        'dte': 5, 'spread_pct': 2.0, 'top_k': 3, 'rebal_days': 5,
        'vix_min': 25, 'train_periods': 26, 'desc': 'High VIX only (>25)'
    },
    'G_5d_Top1_Best': {
        'dte': 5, 'spread_pct': 2.0, 'top_k': 1, 'rebal_days': 5,
        'vix_min': 20, 'train_periods': 26, 'desc': 'Single best sector + VIX filter'
    },
}

###############################################################################
# DATA LOADING
###############################################################################
def load_data():
    """Load ETF price data + VIX from Yahoo via cache or download."""
    cache_dir = '/home/jupiter/Lvl3Quant/research/cache'
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, 'sector_etf_weekly_data.parquet')

    # Check if cached data is fresh (< 1 day old)
    if os.path.exists(cache_file):
        mtime = os.path.getmtime(cache_file)
        age_hours = (datetime.now().timestamp() - mtime) / 3600
        if age_hours < 24:
            print(f"Loading cached data ({age_hours:.1f}h old)")
            df = pd.read_parquet(cache_file)
            # Load VIX separately
            vix_file = os.path.join(cache_dir, 'vix_data.parquet')
            vix = pd.read_parquet(vix_file) if os.path.exists(vix_file) else None
            return df, vix

    import yfinance as yf

    tickers = UNIVERSE + ['^VIX']
    print(f"Downloading {len(tickers)} tickers...")
    data = yf.download(tickers, start='2010-01-01', progress=False, auto_adjust=True)

    # Handle MultiIndex columns
    if isinstance(data.columns, pd.MultiIndex):
        close = data['Close']
        volume = data['Volume']
    else:
        close = data[['Close']]
        volume = data[['Volume']]

    # Separate VIX
    vix_col = '^VIX' if '^VIX' in close.columns else 'VIX'
    if vix_col in close.columns:
        vix = close[[vix_col]].rename(columns={vix_col: 'VIX'})
        close = close.drop(columns=[vix_col], errors='ignore')
        volume = volume.drop(columns=[vix_col], errors='ignore')
    else:
        vix = None

    # Save cache
    close.to_parquet(cache_file)
    if vix is not None:
        vix.to_parquet(os.path.join(cache_dir, 'vix_data.parquet'))

    return close, vix

###############################################################################
# FEATURE ENGINEERING
###############################################################################
def build_features(close, vix, lookback=26):
    """Build weekly features for LightGBM sector ranking."""
    # Resample to weekly
    weekly_close = close.resample('W-FRI').last().dropna(how='all')
    weekly_vix = vix.resample('W-FRI').last().dropna() if vix is not None else None

    features_list = []

    for ticker in UNIVERSE:
        if ticker not in weekly_close.columns:
            continue

        px = weekly_close[ticker].dropna()
        if len(px) < lookback + 10:
            continue

        feat = pd.DataFrame(index=px.index)
        feat['ticker'] = ticker
        feat['price'] = px

        # Momentum features
        for w in [1, 2, 4, 8, 12, 26]:
            feat[f'ret_{w}w'] = px.pct_change(w)

        # Volatility
        ret_1w = px.pct_change()
        for w in [4, 12, 26]:
            feat[f'vol_{w}w'] = ret_1w.rolling(w).std()

        # Risk-adjusted momentum
        feat['sharpe_12w'] = feat['ret_12w'] / (feat['vol_12w'] + 1e-8)
        feat['sharpe_26w'] = feat['ret_26w'] / (feat['vol_26w'] + 1e-8)

        # Mean reversion
        feat['rsi_14'] = compute_rsi(px, 14)

        # Trend strength
        for w in [10, 20, 50]:
            sma = px.rolling(w).mean()
            feat[f'above_sma{w}'] = (px > sma).astype(float)

        # Relative strength vs equal-weight
        ew_ret = weekly_close[UNIVERSE].pct_change(4).mean(axis=1)
        feat['rel_strength_4w'] = feat['ret_4w'] - ew_ret

        # Calmar-like (return / max drawdown)
        rolling_max = px.rolling(26).max()
        drawdown = (px - rolling_max) / rolling_max
        feat['max_dd_26w'] = drawdown.rolling(26).min()
        feat['calmar_26w'] = feat['ret_26w'] / (-feat['max_dd_26w'] + 1e-8)

        # VIX context
        if weekly_vix is not None and 'VIX' in weekly_vix.columns:
            vix_aligned = weekly_vix['VIX'].reindex(feat.index, method='ffill')
            feat['vix'] = vix_aligned
            feat['vix_z'] = (vix_aligned - vix_aligned.rolling(52).mean()) / (vix_aligned.rolling(52).std() + 1e-8)

        # Target: next-week return (for training)
        feat['target'] = px.pct_change().shift(-1)

        features_list.append(feat)

    all_features = pd.concat(features_list)
    return all_features, weekly_close

def compute_rsi(prices, period=14):
    delta = prices.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / (loss + 1e-8)
    return 100 - (100 / (1 + rs))

###############################################################################
# BLACK-SCHOLES PRICING (simplified)
###############################################################################
def bs_call_price(S, K, T, sigma, r=0.05):
    """Black-Scholes call price."""
    from scipy.stats import norm
    if T <= 0 or sigma <= 0:
        return max(S - K, 0)
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)

def spread_pnl(S_entry, S_exit, K_long, K_short, T_entry, T_exit, sigma, haircut=0.15, commission=4.70):
    """
    Bull call spread: buy K_long call, sell K_short call (K_short > K_long).
    Returns PnL per spread.
    """
    # Entry prices
    T_rem_entry = max(T_entry, 1/365)
    long_entry = bs_call_price(S_entry, K_long, T_rem_entry, sigma)
    short_entry = bs_call_price(S_entry, K_short, T_rem_entry, sigma)
    debit = long_entry - short_entry  # pay this to enter

    # Exit prices (at expiry or near-expiry)
    T_rem_exit = max(T_exit, 0.001)
    if T_exit <= 0.01:  # At expiry
        long_exit = max(S_exit - K_long, 0)
        short_exit = max(S_exit - K_short, 0)
    else:
        long_exit = bs_call_price(S_exit, K_long, T_rem_exit, sigma * 0.95)  # slight vol decay
        short_exit = bs_call_price(S_exit, K_short, T_rem_exit, sigma * 0.95)

    credit = long_exit - short_exit  # receive this on exit

    # Apply haircut (slippage vs theoretical)
    debit *= (1 + haircut)  # pay more
    credit *= (1 - haircut)  # receive less

    # PnL = what we get back - what we paid - commissions
    pnl = (credit - debit) * 100  # options are in lots of 100
    pnl -= commission  # RT commission

    return pnl, debit * 100  # pnl, cost to enter

###############################################################################
# BACKTEST ENGINE
###############################################################################
def run_backtest(features, weekly_close, vix, config, name):
    """Walk-forward LightGBM ranking + weekly bull call spread trading."""

    dte = config['dte']
    spread_pct = config['spread_pct'] / 100.0
    top_k = config['top_k']
    rebal_days = config['rebal_days']
    vix_min = config['vix_min']
    train_periods = config['train_periods']

    # Feature columns
    feat_cols = [c for c in features.columns if c not in ['ticker', 'price', 'target', 'vix']]

    # Get unique dates
    dates = sorted(features.index.unique())

    # Walk-forward: train on train_periods weeks, predict next week
    equity = START_CAP
    trades = []
    equity_curve = []

    min_train = train_periods

    for i in range(min_train, len(dates) - 1):
        date = dates[i]

        # VIX filter
        if vix_min > 0 and vix is not None:
            current_vix = vix.loc[:date, 'VIX'].iloc[-1] if date in vix.index or len(vix.loc[:date]) > 0 else 15
            if current_vix < vix_min:
                equity_curve.append({'date': date, 'equity': equity})
                continue
        else:
            current_vix = 15

        # Train data: last train_periods weeks
        train_start = dates[max(0, i - train_periods)]
        train_mask = (features.index >= train_start) & (features.index < date)
        train_data = features[train_mask].copy()

        # Predict data: current week
        pred_mask = features.index == date
        pred_data = features[pred_mask].copy()

        if len(train_data) < 20 or len(pred_data) == 0:
            equity_curve.append({'date': date, 'equity': equity})
            continue

        # Clean features
        train_X = train_data[feat_cols].replace([np.inf, -np.inf], np.nan)
        train_y = train_data['target'].values
        pred_X = pred_data[feat_cols].replace([np.inf, -np.inf], np.nan)

        valid_train = ~(train_X.isna().any(axis=1) | np.isnan(train_y))
        train_X = train_X[valid_train]
        train_y = train_y[valid_train]

        if len(train_X) < 10:
            equity_curve.append({'date': date, 'equity': equity})
            continue

        # Fill NaN in pred
        pred_X = pred_X.fillna(0)

        # Train LightGBM
        try:
            model = lgb.LGBMRegressor(
                n_estimators=100, max_depth=4, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, min_child_samples=5,
                verbose=-1, random_state=42
            )
            model.fit(train_X, train_y)
            predictions = model.predict(pred_X)
        except:
            equity_curve.append({'date': date, 'equity': equity})
            continue

        # Rank sectors by predicted return
        pred_df = pred_data[['ticker', 'price']].copy()
        pred_df['pred_ret'] = predictions
        pred_df = pred_df.sort_values('pred_ret', ascending=False)

        # Select top-K
        top_sectors = pred_df.head(top_k)

        if len(top_sectors) == 0:
            equity_curve.append({'date': date, 'equity': equity})
            continue

        # Execute trades on each top sector
        next_date = dates[i + 1]

        for _, row in top_sectors.iterrows():
            ticker = row['ticker']
            S_entry = row['price']

            # Get exit price
            if ticker in weekly_close.columns and next_date in weekly_close.index:
                S_exit = weekly_close.loc[next_date, ticker]
            else:
                continue

            if pd.isna(S_entry) or pd.isna(S_exit) or S_entry <= 0:
                continue

            # Bull call spread strikes
            K_long = S_entry * (1 - spread_pct / 2)  # slightly ITM
            K_short = S_entry * (1 + spread_pct / 2)  # slightly OTM

            # Implied vol estimate from recent realized vol
            ticker_data = features[(features['ticker'] == ticker) & (features.index <= date)]
            if 'vol_4w' in ticker_data.columns and len(ticker_data) > 0:
                realized_vol = ticker_data['vol_4w'].iloc[-1]
                if pd.isna(realized_vol) or realized_vol <= 0:
                    realized_vol = 0.25
                # Annualize (weekly vol * sqrt(52))
                iv = realized_vol * np.sqrt(52) * 1.1  # IV typically 10% above realized
            else:
                iv = 0.25

            # Time to expiry
            T_entry = dte / 365.0
            T_exit = max(0, (dte - rebal_days) / 365.0)

            # Position sizing: max trade size relative to equity
            max_trade = min(equity * 0.35 / top_k, 300)  # max $300/trade, split across top_k

            # Calculate spread cost and PnL
            pnl, cost = spread_pnl(S_entry, S_exit, K_long, K_short, T_entry, T_exit, iv, HAIRCUT, COMMISSION_RT)

            if cost <= 0 or cost > max_trade * 100:  # cost is already * 100
                continue

            # Scale to position size
            n_contracts = max(1, int(max_trade / (cost / 100 + 0.01)))
            if n_contracts * cost / 100 > equity * 0.5:  # don't risk more than 50% on one period
                n_contracts = max(1, int(equity * 0.5 / (cost / 100 + 0.01)))

            actual_pnl = pnl * n_contracts
            actual_cost = cost * n_contracts / 100

            # Cap loss at cost
            if actual_pnl < -actual_cost:
                actual_pnl = -actual_cost

            equity += actual_pnl

            # Determine regime
            spy_ret = None
            if 'XLK' in weekly_close.columns:  # use as market proxy
                mkt = weekly_close[UNIVERSE].mean(axis=1)
                if date in mkt.index and next_date in mkt.index:
                    spy_ret = (mkt.loc[next_date] - mkt.loc[date]) / mkt.loc[date]

            trades.append({
                'date': date,
                'exit_date': next_date,
                'ticker': ticker,
                'entry_price': S_entry,
                'exit_price': S_exit,
                'pnl': actual_pnl,
                'cost': actual_cost,
                'n_contracts': n_contracts,
                'pred_ret': row['pred_ret'],
                'vix': current_vix,
                'regime': 'bull' if (spy_ret is not None and spy_ret > 0) else 'bear',
                'equity_after': equity,
            })

        equity_curve.append({'date': date, 'equity': equity})

    return trades, equity_curve

###############################################################################
# ADVERSARIAL VALIDATION (4 gates)
###############################################################################
def adversarial_audit(trades, equity_curve, name):
    """Run 4-gate adversarial validation."""
    if len(trades) < 10:
        return {
            'name': name, 'n_trades': len(trades), 'gates_passed': 0,
            'error': 'Too few trades'
        }

    trade_df = pd.DataFrame(trades)
    pnls = trade_df['pnl'].values

    # Basic metrics
    n = len(pnls)
    wins = (pnls > 0).sum()
    wr = wins / n * 100
    avg_win = pnls[pnls > 0].mean() if wins > 0 else 0
    avg_loss = pnls[pnls < 0].mean() if (pnls < 0).sum() > 0 else 0
    total_pnl = pnls.sum()
    pf = abs(pnls[pnls > 0].sum() / pnls[pnls < 0].sum()) if (pnls < 0).sum() != 0 else 999

    # Monthly returns for Sharpe (using equity curve)
    eq_df = pd.DataFrame(equity_curve)
    eq_df['date'] = pd.to_datetime(eq_df['date'])
    eq_df = eq_df.set_index('date')

    # Calendar month aggregation (honest method)
    monthly_eq = eq_df['equity'].resample('ME').last().dropna()
    monthly_ret = monthly_eq.pct_change().dropna()

    if len(monthly_ret) > 1:
        sharpe = monthly_ret.mean() / (monthly_ret.std() + 1e-8) * np.sqrt(12)
        sortino_denom = monthly_ret[monthly_ret < 0].std()
        sortino = monthly_ret.mean() / (sortino_denom + 1e-8) * np.sqrt(12) if sortino_denom > 0 else sharpe * 1.5
    else:
        sharpe = 0
        sortino = 0

    # CAGR
    final_eq = eq_df['equity'].iloc[-1] if len(eq_df) > 0 else START_CAP
    start_date = eq_df.index[0] if len(eq_df) > 0 else pd.Timestamp('2010-01-01')
    end_date = eq_df.index[-1] if len(eq_df) > 0 else pd.Timestamp('2026-01-01')
    years = (end_date - start_date).days / 365.25
    cagr = (final_eq / START_CAP) ** (1/max(years, 0.1)) - 1

    # Max drawdown
    eq_series = eq_df['equity']
    rolling_max = eq_series.cummax()
    drawdown = (eq_series - rolling_max) / rolling_max
    maxdd = drawdown.min()

    # Calmar
    calmar = cagr / (-maxdd + 1e-8) if maxdd < 0 else 0

    # === GATE 1: Permutation test ===
    real_sharpe = sharpe
    n_perms = 500
    perm_sharpes = []
    for _ in range(n_perms):
        shuffled = pnls.copy()
        np.random.shuffle(shuffled)
        eq_perm = np.cumsum(shuffled) + START_CAP
        # Simple monthly approximation
        n_months = max(1, len(eq_perm) // 4)  # ~4 trades per month
        monthly_chunks = np.array_split(eq_perm, n_months)
        monthly_rets_perm = []
        prev_val = START_CAP
        for chunk in monthly_chunks:
            if len(chunk) > 0:
                ret = (chunk[-1] - prev_val) / prev_val
                monthly_rets_perm.append(ret)
                prev_val = chunk[-1]
        if len(monthly_rets_perm) > 1:
            arr = np.array(monthly_rets_perm)
            perm_sharpe = arr.mean() / (arr.std() + 1e-8) * np.sqrt(12)
        else:
            perm_sharpe = 0
        perm_sharpes.append(perm_sharpe)

    perm_p = np.mean(np.array(perm_sharpes) >= real_sharpe)
    g1_pass = perm_p < 0.05

    # === GATE 2: R1 Regime-agnostic ===
    bull_trades = trade_df[trade_df['regime'] == 'bull']['pnl'].values
    bear_trades = trade_df[trade_df['regime'] == 'bear']['pnl'].values

    if len(bull_trades) > 5 and len(bear_trades) > 5:
        bull_wr = (bull_trades > 0).mean() * 100
        bear_wr = (bear_trades > 0).mean() * 100
        bull_sharpe_approx = bull_trades.mean() / (bull_trades.std() + 1e-8) * np.sqrt(52)
        bear_sharpe_approx = bear_trades.mean() / (bear_trades.std() + 1e-8) * np.sqrt(52)
        r1_gap = abs(bull_sharpe_approx - bear_sharpe_approx) / max(abs(bull_sharpe_approx), abs(bear_sharpe_approx), 1e-8)
    else:
        bull_wr = bear_wr = wr
        bull_sharpe_approx = bear_sharpe_approx = sharpe
        r1_gap = 0

    g2_pass = r1_gap < 0.50

    # === GATE 3: Sub-period stability ===
    third = len(trade_df) // 3
    sub_sharpes = []
    for start, end in [(0, third), (third, 2*third), (2*third, len(trade_df))]:
        sub_pnls = trade_df.iloc[start:end]['pnl'].values
        if len(sub_pnls) > 5:
            sub_s = sub_pnls.mean() / (sub_pnls.std() + 1e-8) * np.sqrt(52)
            sub_sharpes.append(sub_s)

    g3_pass = len(sub_sharpes) >= 2 and all(s > 0 for s in sub_sharpes)

    # === GATE 4: Outlier removal ===
    p95 = np.percentile(pnls, 95)
    p5 = np.percentile(pnls, 5)
    trimmed = pnls[(pnls >= p5) & (pnls <= p95)]
    if len(trimmed) > 5:
        trimmed_wr = (trimmed > 0).mean() * 100
        trimmed_sharpe = trimmed.mean() / (trimmed.std() + 1e-8) * np.sqrt(52)
        g4_pass = trimmed_sharpe > 0
    else:
        trimmed_sharpe = 0
        trimmed_wr = 0
        g4_pass = False

    gates_passed = sum([g1_pass, g2_pass, g3_pass, g4_pass])

    result = {
        'name': name,
        'n_trades': n,
        'win_rate': round(wr, 1),
        'avg_win': round(avg_win, 2),
        'avg_loss': round(avg_loss, 2),
        'total_pnl': round(total_pnl, 2),
        'final_equity': round(final_eq, 2),
        'cagr_pct': round(cagr * 100, 1),
        'sharpe': round(sharpe, 2),
        'sortino': round(sortino, 2),
        'maxdd_pct': round(maxdd * 100, 1),
        'calmar': round(calmar, 2),
        'pf': round(pf, 2),
        'r1_gap': round(r1_gap, 3),
        'r1_pass': g2_pass,
        'bull_wr': round(bull_wr, 1),
        'bear_wr': round(bear_wr, 1),
        'bull_trades': len(bull_trades),
        'bear_trades': len(bear_trades),
        'bull_sharpe': round(bull_sharpe_approx, 2),
        'bear_sharpe': round(bear_sharpe_approx, 2),
        'perm_p': round(perm_p, 4),
        'g1_pass': g1_pass,
        'g2_pass': g2_pass,
        'g3_pass': g3_pass,
        'g4_pass': g4_pass,
        'sub_sharpes': [round(s, 2) for s in sub_sharpes],
        'gates_passed': gates_passed,
    }

    return result

###############################################################################
# RANDOM DIRECTION BASELINE (artifact check)
###############################################################################
def random_direction_baseline(features, weekly_close, vix, config, n_trials=10):
    """Test if random sector selection is also profitable (structural edge check)."""
    random_sharpes = []

    for trial in range(n_trials):
        rng = np.random.RandomState(trial + 100)

        # Same as backtest but with random rankings
        dates = sorted(features.index.unique())
        equity = START_CAP
        equity_curve = []

        dte = config['dte']
        spread_pct = config['spread_pct'] / 100.0
        top_k = config['top_k']
        vix_min = config['vix_min']

        for i in range(config['train_periods'], len(dates) - 1):
            date = dates[i]
            next_date = dates[i + 1]

            # VIX filter
            if vix_min > 0 and vix is not None:
                current_vix = vix.loc[:date, 'VIX'].iloc[-1] if len(vix.loc[:date]) > 0 else 15
                if current_vix < vix_min:
                    equity_curve.append({'date': date, 'equity': equity})
                    continue

            # Random sector selection
            available = [t for t in UNIVERSE if t in weekly_close.columns and date in weekly_close.index]
            if len(available) < top_k:
                equity_curve.append({'date': date, 'equity': equity})
                continue

            selected = rng.choice(available, min(top_k, len(available)), replace=False)

            for ticker in selected:
                S_entry = weekly_close.loc[date, ticker]
                if next_date in weekly_close.index:
                    S_exit = weekly_close.loc[next_date, ticker]
                else:
                    continue

                if pd.isna(S_entry) or pd.isna(S_exit) or S_entry <= 0:
                    continue

                K_long = S_entry * (1 - spread_pct / 2)
                K_short = S_entry * (1 + spread_pct / 2)

                iv = 0.25
                T_entry = dte / 365.0
                T_exit = max(0, (dte - config['rebal_days']) / 365.0)

                max_trade = min(equity * 0.35 / top_k, 300)
                pnl, cost = spread_pnl(S_entry, S_exit, K_long, K_short, T_entry, T_exit, iv, HAIRCUT, COMMISSION_RT)

                if cost <= 0:
                    continue

                n_contracts = max(1, int(max_trade / (cost / 100 + 0.01)))
                actual_pnl = pnl * n_contracts
                actual_cost = cost * n_contracts / 100

                if actual_pnl < -actual_cost:
                    actual_pnl = -actual_cost

                equity += actual_pnl

            equity_curve.append({'date': date, 'equity': equity})

        # Calculate Sharpe
        if len(equity_curve) > 10:
            eq_df = pd.DataFrame(equity_curve)
            eq_df['date'] = pd.to_datetime(eq_df['date'])
            eq_df = eq_df.set_index('date')
            monthly_eq = eq_df['equity'].resample('ME').last().dropna()
            monthly_ret = monthly_eq.pct_change().dropna()
            if len(monthly_ret) > 1:
                s = monthly_ret.mean() / (monthly_ret.std() + 1e-8) * np.sqrt(12)
                random_sharpes.append(s)

    return random_sharpes

###############################################################################
# MAIN
###############################################################################
def main():
    print("=" * 70)
    print("Weekly DTE Sector Bull Call Spreads v1")
    print("=" * 70)

    t0 = datetime.now()

    # Load data
    close, vix = load_data()
    print(f"Data: {close.shape[0]} days, {close.shape[1]} tickers, {close.index[0].date()} to {close.index[-1].date()}")

    # Build features
    features, weekly_close = build_features(close, vix)
    print(f"Features: {len(features)} rows, {features.columns.tolist()[:5]}...")

    # MLflow
    if HAS_MLFLOW:
        exp = mlflow.set_experiment("weekly_dte_sector_spreads_v1")
        exp_id = exp.experiment_id

    all_results = []
    best_result = None
    best_sharpe = -999

    for name, config in CONFIGS.items():
        print(f"\n{'='*60}")
        print(f"Testing: {name} — {config['desc']}")
        print(f"{'='*60}")

        trades, equity_curve = run_backtest(features, weekly_close, vix, config, name)

        if len(trades) < 10:
            print(f"  SKIP: Only {len(trades)} trades")
            all_results.append({
                'name': name, 'n_trades': len(trades),
                'error': 'Too few trades', 'gates_passed': 0
            })
            continue

        result = adversarial_audit(trades, equity_curve, name)

        # Random baseline check
        print(f"  Running random baseline check...")
        random_sharpes = random_direction_baseline(features, weekly_close, vix, config, n_trials=10)
        if len(random_sharpes) > 0:
            result['random_sharpe_mean'] = round(np.mean(random_sharpes), 2)
            result['random_sharpe_std'] = round(np.std(random_sharpes), 2)
            result['ml_vs_random'] = round(result['sharpe'] - np.mean(random_sharpes), 2)
            result['random_also_profitable'] = np.mean(np.array(random_sharpes) > 0) > 0.5

        all_results.append(result)

        gates = f"{result['gates_passed']}/4"
        status = "✅" if result['gates_passed'] == 4 else "⚠️" if result['gates_passed'] >= 2 else "❌"

        print(f"  {status} {gates} gates | Sharpe {result['sharpe']:.2f} | WR {result['win_rate']:.1f}% | "
              f"CAGR {result['cagr_pct']:.1f}% | MDD {result['maxdd_pct']:.1f}% | "
              f"PF {result['pf']:.2f} | {result['n_trades']} trades | ${START_CAP}→${result['final_equity']:,.0f}")
        print(f"  R1 gap {result['r1_gap']:.3f} | Bull WR {result['bull_wr']:.1f}% | Bear WR {result['bear_wr']:.1f}% | "
              f"Perm p={result['perm_p']:.4f}")
        if 'random_sharpe_mean' in result:
            print(f"  Random baseline: Sharpe {result['random_sharpe_mean']:.2f} ± {result['random_sharpe_std']:.2f} | "
                  f"ML advantage: {result['ml_vs_random']:+.2f}")
            if result['random_also_profitable']:
                print(f"  ⚠️ STRUCTURAL EDGE WARNING: Random selection also profitable")

        # Log to MLflow
        if HAS_MLFLOW:
            with mlflow.start_run(experiment_id=exp_id, run_name=name):
                for k, v in result.items():
                    if isinstance(v, (int, float)):
                        mlflow.log_metric(k, v)
                mlflow.log_params({k: str(v) for k, v in config.items()})

        if result['sharpe'] > best_sharpe:
            best_sharpe = result['sharpe']
            best_result = result

    # Summary
    runtime = (datetime.now() - t0).total_seconds()

    print(f"\n{'='*70}")
    print(f"SUMMARY — Weekly DTE Sector Bull Call Spreads v1")
    print(f"{'='*70}")

    pass_4 = [r for r in all_results if r.get('gates_passed', 0) == 4]
    pass_any = [r for r in all_results if r.get('gates_passed', 0) >= 2]

    print(f"Configs tested: {len(all_results)}")
    print(f"4/4 gates: {len(pass_4)}")
    print(f"≥2/4 gates: {len(pass_any)}")
    print(f"Runtime: {runtime:.0f}s")

    if best_result:
        print(f"\nBEST: {best_result['name']}")
        print(f"  Sharpe {best_result['sharpe']:.2f} | Sortino {best_result['sortino']:.2f} | "
              f"CAGR {best_result['cagr_pct']:.1f}% | MDD {best_result['maxdd_pct']:.1f}%")
        print(f"  WR {best_result['win_rate']:.1f}% | PF {best_result['pf']:.2f} | "
              f"{best_result['n_trades']} trades | ${START_CAP}→${best_result['final_equity']:,.0f}")

    # Check structural edge
    structural = any(r.get('random_also_profitable', False) for r in all_results if r.get('gates_passed', 0) >= 2)
    if structural:
        print("\n⚠️ WARNING: Random sector selection is also profitable.")
        print("   Edge may be STRUCTURAL (bull call spreads + VIX filter) not ML-specific.")
        print("   This is consistent with our previous finding (HC #428 focused adversarial).")

    # Save results
    findings_dir = '/home/jupiter/Lvl3Quant/research/findings'
    os.makedirs(findings_dir, exist_ok=True)

    save_data = {
        'timestamp': datetime.now().isoformat(),
        'experiment': 'weekly_dte_sector_spreads_v1',
        'universe': UNIVERSE,
        'capital': START_CAP,
        'configs': CONFIGS,
        'results': all_results,
        'best': best_result,
        'structural_edge_warning': structural,
        'runtime_s': runtime,
    }

    with open(os.path.join(findings_dir, 'weekly_dte_sector_spreads_v1_results.json'), 'w') as f:
        json.dump(save_data, f, indent=2, default=str)

    print(f"\nResults saved.")
    return all_results

if __name__ == '__main__':
    results = main()
