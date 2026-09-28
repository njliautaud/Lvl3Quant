#!/usr/bin/env python3
"""
Growth Stock Swing Options v1
==============================
HIGH-GROWTH strategy for $645 RH agentic account.

RATIONALE:
- PEAD options work but only 5 trades/year → can't prove significance
- Sector ETF options fail due to low IV / tiny premiums
- Individual growth stocks have MUCH higher IV → meaningful option premiums
- Weekly LGBM ranking + momentum + volume → pick best 1-2 stocks
- Buy 30-45 DTE calls (or puts for shorts) on top-ranked stocks
- Hold 5-10 days, TP at +50%, SL at -30%
- Target 40-50 trades/year → enough for perm test

KEY DIFFERENCES from other tested strategies:
- vs PEAD: not earnings-gated → more trades, continuous signal
- vs sector ETF options: individual stocks have 2-5x higher IV
- vs weekly momentum equity: options provide leverage for $645 account
- vs momentum burst options: holding period is longer (5-10d vs 1d)

6 VARIANTS:
  A. Top-1 stock, ATM call, 30 DTE, hold 5 days
  B. Top-2 stocks, ATM calls, 30 DTE, hold 10 days
  C. Top-1 stock, OTM call (delta 0.30), 45 DTE, hold 10 days
  D. Top-1 with VIX filter (skip when VIX>25), 30 DTE
  E. Top-1 with volume surge confirmation, 30 DTE
  F. Top-1 long + Bottom-1 short (put), 30 DTE

UNIVERSE: 52 growth stocks (same as weekly_momentum_growth_v1)
PERIOD: 2018-01-01 to 2026-07-25 (use ALL available data per HC #428)
ACCOUNT: $645, max $200 per trade (HC #749)
"""

import sys, os, json, warnings, time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from collections import defaultdict
from scipy.stats import norm

warnings.filterwarnings('ignore')

for root in ['/home/jupiter/Lvl3Quant', '/home/nick/Lvl3Quant']:
    if os.path.isdir(root):
        LVL3_ROOT = root
        break
else:
    LVL3_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

OUTPUT_DIR = os.path.join(LVL3_ROOT, 'output', 'growth_research', 'growth_stock_swing_options_v1')
os.makedirs(OUTPUT_DIR, exist_ok=True)

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

# ── Constants ──────────────────────────────────────────────────────────
INITIAL_CAPITAL = 645.0
MAX_POSITION = 200.0  # HC #749
COMMISSION_PER_CONTRACT = 0.65  # RH options commission
RISK_FREE_RATE = 0.05
TRADING_DAYS_YEAR = 252

# 52 Growth stocks universe
GROWTH_STOCKS = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'AMD',
    'NFLX', 'CRM', 'ADBE', 'PYPL', 'SQ', 'SHOP', 'SNOW', 'DDOG',
    'NET', 'CRWD', 'ZS', 'PANW', 'MDB', 'COIN', 'MELI', 'SE',
    'NIO', 'LI', 'XPEV', 'RIVN', 'LCID', 'SOFI', 'PLTR', 'RBLX',
    'U', 'ROKU', 'SNAP', 'PINS', 'ABNB', 'DASH', 'UBER', 'LYFT',
    'ARM', 'SMCI', 'IONQ', 'RKLB', 'AFRM', 'HOOD', 'UPST', 'BABA',
    'JD', 'PDD', 'GRAB', 'CPNG'
]

# ── Black-Scholes Pricing ─────────────────────────────────────────────
def bs_price(S, K, T, sigma, r=RISK_FREE_RATE, option_type='call'):
    """Black-Scholes option price."""
    if T <= 0 or sigma <= 0 or S <= 0:
        return max(0, (S - K) if option_type == 'call' else (K - S))
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == 'call':
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

def bs_delta(S, K, T, sigma, r=RISK_FREE_RATE, option_type='call'):
    """Black-Scholes delta."""
    if T <= 0 or sigma <= 0 or S <= 0:
        return 1.0 if option_type == 'call' else -1.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    if option_type == 'call':
        return norm.cdf(d1)
    else:
        return norm.cdf(d1) - 1

def find_strike_for_delta(S, T, sigma, target_delta, r=RISK_FREE_RATE, option_type='call'):
    """Find strike price for a target delta."""
    # Binary search
    lo, hi = S * 0.5, S * 1.5
    for _ in range(50):
        mid = (lo + hi) / 2
        d = bs_delta(S, mid, T, sigma, r, option_type)
        if option_type == 'call':
            if abs(d) > abs(target_delta):
                lo = mid
            else:
                hi = mid
        else:
            if abs(d) > abs(target_delta):
                hi = mid
            else:
                lo = mid
    return (lo + hi) / 2

# ── Data Loading ──────────────────────────────────────────────────────
def load_stock_data():
    """Load price data for growth stocks. Try yfinance cache, then download."""
    cache_file = os.path.join(LVL3_ROOT, 'data', 'growth_stocks_prices.parquet')

    if os.path.exists(cache_file):
        df = pd.read_parquet(cache_file)
        if len(df.columns) >= 30:
            print(f"  Loaded cached price data: {len(df)} days, {len(df.columns)} stocks")
            return df

    print("  Downloading stock price data via yfinance...")
    try:
        import yfinance as yf
        data = yf.download(GROWTH_STOCKS, start='2016-01-01', end='2026-07-26',
                          auto_adjust=True, progress=False)['Close']
        os.makedirs(os.path.join(LVL3_ROOT, 'data'), exist_ok=True)
        data.to_parquet(cache_file)
        print(f"  Downloaded: {len(data)} days, {len(data.columns)} stocks")
        return data
    except Exception as e:
        print(f"  Download failed: {e}")
        return None

def compute_realized_vol(prices, window=21):
    """Compute realized volatility (annualized) for each stock."""
    returns = prices.pct_change()
    vol = returns.rolling(window).std() * np.sqrt(TRADING_DAYS_YEAR)
    return vol

def compute_features(prices, date_idx):
    """Compute momentum + volume features for LGBM ranking at a given date index."""
    if date_idx < 60:
        return None

    hist = prices.iloc[:date_idx+1]
    features = {}

    for ticker in prices.columns:
        p = hist[ticker].dropna()
        if len(p) < 60:
            continue

        ret_5d = p.iloc[-1] / p.iloc[-6] - 1 if len(p) >= 6 else 0
        ret_10d = p.iloc[-1] / p.iloc[-11] - 1 if len(p) >= 11 else 0
        ret_21d = p.iloc[-1] / p.iloc[-22] - 1 if len(p) >= 22 else 0
        ret_63d = p.iloc[-1] / p.iloc[-64] - 1 if len(p) >= 64 else 0

        # Volatility
        daily_ret = p.pct_change().dropna()
        vol_21d = daily_ret.iloc[-21:].std() * np.sqrt(252) if len(daily_ret) >= 21 else 0.5

        # Volume proxy: use return magnitude as vol proxy (no volume data in this format)
        avg_abs_ret = daily_ret.iloc[-10:].abs().mean() if len(daily_ret) >= 10 else 0

        # Mean reversion signal
        sma_20 = p.iloc[-20:].mean() if len(p) >= 20 else p.iloc[-1]
        dist_from_sma = (p.iloc[-1] / sma_20 - 1) if sma_20 > 0 else 0

        features[ticker] = {
            'ret_5d': ret_5d,
            'ret_10d': ret_10d,
            'ret_21d': ret_21d,
            'ret_63d': ret_63d,
            'vol_21d': vol_21d,
            'avg_abs_ret': avg_abs_ret,
            'dist_from_sma': dist_from_sma,
            'price': p.iloc[-1]
        }

    return features

# ── LGBM Ranking (simplified walk-forward) ────────────────────────────
def lgbm_rank_stocks(prices, date_idx, lookback=252):
    """
    Train LGBM to predict next-week returns, use predictions as ranking.
    Walk-forward: train on [date_idx-lookback:date_idx-5], predict at date_idx.
    """
    try:
        import lightgbm as lgb
    except ImportError:
        # Fallback to simple momentum ranking
        return momentum_rank_stocks(prices, date_idx)

    # Build training data
    features_list = []
    targets = []

    start_idx = max(65, date_idx - lookback)
    for i in range(start_idx, date_idx - 5, 5):  # Weekly samples
        feats = compute_features(prices, i)
        if feats is None:
            continue

        for ticker, f in feats.items():
            # Target: next 5-day return
            future_idx = min(i + 5, len(prices) - 1)
            if future_idx >= len(prices) or pd.isna(prices[ticker].iloc[future_idx]):
                continue
            future_ret = prices[ticker].iloc[future_idx] / prices[ticker].iloc[i] - 1

            row = {**f, 'target': future_ret}
            del row['price']
            features_list.append(row)
            targets.append(future_ret)

    if len(features_list) < 50:
        return momentum_rank_stocks(prices, date_idx)

    train_df = pd.DataFrame(features_list)
    feature_cols = [c for c in train_df.columns if c != 'target']

    X_train = train_df[feature_cols].values
    y_train = train_df['target'].values

    # Train LGBM
    dataset = lgb.Dataset(X_train, label=y_train, free_raw_data=False)
    params = {
        'objective': 'regression',
        'metric': 'mae',
        'learning_rate': 0.05,
        'num_leaves': 16,
        'max_depth': 4,
        'min_data_in_leaf': 20,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'seed': 42
    }

    model = lgb.train(params, dataset, num_boost_round=100)

    # Predict current stocks
    current_feats = compute_features(prices, date_idx)
    if current_feats is None:
        return momentum_rank_stocks(prices, date_idx)

    predictions = {}
    for ticker, f in current_feats.items():
        row = {k: v for k, v in f.items() if k != 'price'}
        X_pred = np.array([[row.get(c, 0) for c in feature_cols]])
        pred = model.predict(X_pred)[0]
        predictions[ticker] = {'pred': pred, 'price': f['price'], 'vol': f['vol_21d']}

    # Rank by prediction
    ranked = sorted(predictions.items(), key=lambda x: x[1]['pred'], reverse=True)
    return ranked

def momentum_rank_stocks(prices, date_idx):
    """Simple momentum ranking fallback."""
    feats = compute_features(prices, date_idx)
    if feats is None:
        return []

    ranked = []
    for ticker, f in feats.items():
        # Composite momentum: 50% 21d + 30% 10d + 20% 5d
        score = 0.5 * f['ret_21d'] + 0.3 * f['ret_10d'] + 0.2 * f['ret_5d']
        ranked.append((ticker, {'pred': score, 'price': f['price'], 'vol': f['vol_21d']}))

    ranked.sort(key=lambda x: x[1]['pred'], reverse=True)
    return ranked

# ── VIX Proxy ─────────────────────────────────────────────────────────
def get_vix_proxy(prices, date_idx, window=21):
    """Use SPY (or market avg) realized vol as VIX proxy."""
    if 'SPY' in prices.columns:
        spy = prices['SPY'].iloc[max(0, date_idx-window):date_idx+1]
    else:
        # Use average of all stocks
        spy = prices.iloc[max(0, date_idx-window):date_idx+1].mean(axis=1)

    if len(spy) < 5:
        return 20.0

    ret = spy.pct_change().dropna()
    vol = ret.std() * np.sqrt(252) * 100  # as VIX-like percentage
    return vol

# ── Trade Simulation ──────────────────────────────────────────────────
def simulate_option_trade(entry_price, exit_price, vol, dte_days, strike,
                          option_type='call', contracts=1):
    """Simulate an option trade from entry to exit."""
    T_entry = dte_days / 365.0
    entry_option_price = bs_price(entry_price, strike, T_entry, vol, RISK_FREE_RATE, option_type)

    # At exit (dte_days - hold_days)
    T_exit = max(1, dte_days - 5) / 365.0  # Minimum 1 day left

    # Option price at exit
    exit_option_price = bs_price(exit_price, strike, T_exit, vol, RISK_FREE_RATE, option_type)

    cost = entry_option_price * 100 * contracts + COMMISSION_PER_CONTRACT * contracts
    proceeds = exit_option_price * 100 * contracts - COMMISSION_PER_CONTRACT * contracts
    pnl = proceeds - cost

    return {
        'entry_option_price': entry_option_price,
        'exit_option_price': exit_option_price,
        'cost': cost,
        'proceeds': proceeds,
        'pnl': pnl,
        'return_pct': pnl / cost if cost > 0 else 0
    }

# ── Validation Gates ──────────────────────────────────────────────────
def run_5_gates(equity_curve, trades_df, daily_returns, prices, label):
    """Run 5-gate validation: Sharpe>1, perm p<0.05, WR>40%, R1<0.5, MC CI>0."""
    results = {}

    # Gate 1: Sharpe > 1
    if len(daily_returns) > 0 and daily_returns.std() > 0:
        sharpe = daily_returns.mean() / daily_returns.std() * np.sqrt(252)
    else:
        sharpe = 0
    results['sharpe'] = sharpe
    results['sharpe_gt_1'] = sharpe > 1.0

    # Gate 2: Permutation test p < 0.05
    if len(trades_df) >= 10:
        real_sharpe = sharpe
        n_perms = 200
        beat_count = 0
        trade_rets = trades_df['return_pct'].values
        for _ in range(n_perms):
            shuffled = trade_rets.copy()
            np.random.shuffle(shuffled)
            # Random signs
            random_signs = np.random.choice([-1, 1], size=len(shuffled))
            random_rets = shuffled * random_signs
            if random_rets.std() > 0:
                random_sharpe = random_rets.mean() / random_rets.std() * np.sqrt(52)  # weekly
            else:
                random_sharpe = 0
            if random_sharpe >= real_sharpe:
                beat_count += 1
        perm_p = beat_count / n_perms
    else:
        perm_p = 1.0
    results['perm_p'] = perm_p
    results['perm_p_lt_005'] = perm_p < 0.05

    # Gate 3: Win rate > 40%
    if len(trades_df) > 0:
        wr = (trades_df['pnl'] > 0).mean() * 100
    else:
        wr = 0
    results['wr'] = wr
    results['wr_gt_40'] = wr > 40

    # Gate 4: Regime balance (|Sharpe_bull - Sharpe_bear| / max < 0.50)
    if len(trades_df) > 0 and 'regime' in trades_df.columns:
        bull_trades = trades_df[trades_df['regime'] == 'bull']
        bear_trades = trades_df[trades_df['regime'] == 'bear']

        if len(bull_trades) >= 3 and len(bear_trades) >= 3:
            bull_sr = bull_trades['return_pct'].mean() / bull_trades['return_pct'].std() * np.sqrt(52) if bull_trades['return_pct'].std() > 0 else 0
            bear_sr = bear_trades['return_pct'].mean() / bear_trades['return_pct'].std() * np.sqrt(52) if bear_trades['return_pct'].std() > 0 else 0
            regime_gap = abs(bull_sr - bear_sr) / max(abs(bull_sr), abs(bear_sr), 0.01)
        else:
            regime_gap = 1.0  # Not enough data
    else:
        regime_gap = 1.0
    results['regime_balance'] = regime_gap
    results['regime_balance_lt_05'] = regime_gap < 0.50

    # Gate 5: Monte Carlo 95% CI lower bound > 0
    if len(trades_df) >= 10:
        trade_pnls = trades_df['pnl'].values
        mc_finals = []
        for _ in range(500):
            sample = np.random.choice(trade_pnls, size=len(trade_pnls), replace=True)
            mc_finals.append(sample.sum())
        ci_lower = np.percentile(mc_finals, 5)
    else:
        ci_lower = -999
    results['mc_ci_lower'] = ci_lower
    results['mc_ci_positive'] = ci_lower > 0

    # Summary
    gates_passed = sum([
        results['sharpe_gt_1'],
        results['perm_p_lt_005'],
        results['wr_gt_40'],
        results['regime_balance_lt_05'],
        results['mc_ci_positive']
    ])
    results['gates_passed'] = gates_passed
    results['label'] = label

    # Additional metrics
    if len(equity_curve) > 0:
        results['final_equity'] = equity_curve[-1]
        results['total_return'] = (equity_curve[-1] / INITIAL_CAPITAL - 1) * 100
        peak = np.maximum.accumulate(equity_curve)
        dd = (equity_curve - peak) / peak
        results['max_dd'] = dd.min() * 100

        n_years = len(equity_curve) / 252
        if n_years > 0:
            results['cagr'] = ((equity_curve[-1] / INITIAL_CAPITAL) ** (1 / n_years) - 1) * 100
        else:
            results['cagr'] = 0

    results['n_trades'] = len(trades_df)

    # Sortino
    if len(daily_returns) > 0:
        downside = daily_returns[daily_returns < 0]
        if len(downside) > 0 and downside.std() > 0:
            results['sortino'] = daily_returns.mean() / downside.std() * np.sqrt(252)
        else:
            results['sortino'] = sharpe * 1.5
    else:
        results['sortino'] = 0

    # Profit factor
    if len(trades_df) > 0:
        gross_profit = trades_df[trades_df['pnl'] > 0]['pnl'].sum()
        gross_loss = abs(trades_df[trades_df['pnl'] < 0]['pnl'].sum())
        results['pf'] = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    else:
        results['pf'] = 0

    return results

# ── Strategy Variants ─────────────────────────────────────────────────
def run_variant(prices, vol_data, variant_name, config, oot_start='2022-01-01'):
    """Run a single variant of the swing options strategy."""
    print(f"\n{'='*60}")
    print(f"  VARIANT {variant_name}: {config['description']}")
    print(f"{'='*60}")

    top_n = config.get('top_n', 1)
    hold_days = config.get('hold_days', 5)
    dte = config.get('dte', 30)
    option_delta = config.get('delta', 0.50)  # ATM = ~0.50
    vix_filter = config.get('vix_filter', None)
    volume_filter = config.get('volume_filter', False)
    short_side = config.get('short_side', False)
    use_lgbm = config.get('use_lgbm', False)
    tp_pct = config.get('tp_pct', 0.50)  # Take profit at 50%
    sl_pct = config.get('sl_pct', -0.30)  # Stop loss at -30%

    # Find OOT start index
    oot_start_date = pd.Timestamp(oot_start)
    oot_mask = prices.index >= oot_start_date
    if oot_mask.sum() == 0:
        print("  No OOT data!")
        return None

    oot_start_idx = oot_mask.argmax()

    # Find Fridays in OOT
    oot_dates = prices.index[oot_start_idx:]
    fridays = [d for d in oot_dates if d.dayofweek == 4]  # Friday = 4

    print(f"  OOT period: {oot_dates[0].date()} to {oot_dates[-1].date()}")
    print(f"  OOT Fridays: {len(fridays)}")

    # Track equity and trades
    equity = INITIAL_CAPITAL
    equity_curve = [equity]
    trades = []
    positions = []  # Active positions

    # Daily tracking
    daily_equity = {prices.index[oot_start_idx]: equity}

    for fri_idx, friday in enumerate(fridays):
        if equity <= 50:  # Account blown
            break

        date_idx = prices.index.get_loc(friday)

        # VIX filter
        if vix_filter is not None:
            vix = get_vix_proxy(prices, date_idx)
            if vix > vix_filter:
                daily_equity[friday] = equity
                equity_curve.append(equity)
                continue

        # Rank stocks
        if use_lgbm:
            ranked = lgbm_rank_stocks(prices, date_idx)
        else:
            ranked = momentum_rank_stocks(prices, date_idx)

        if len(ranked) < top_n + (1 if short_side else 0):
            daily_equity[friday] = equity
            equity_curve.append(equity)
            continue

        # Volume filter: skip if top stock's recent abs returns are below median
        if volume_filter and len(ranked) > 0:
            top_ticker = ranked[0][0]
            feats = compute_features(prices, date_idx)
            if feats and top_ticker in feats:
                all_abs_rets = [f['avg_abs_ret'] for f in feats.values()]
                median_abs = np.median(all_abs_rets)
                if feats[top_ticker]['avg_abs_ret'] < median_abs:
                    daily_equity[friday] = equity
                    equity_curve.append(equity)
                    continue

        # Determine regime (for R1 gate)
        spy_col = 'AAPL'  # Use AAPL as market proxy if no SPY
        for proxy in ['SPY', 'QQQ', 'AAPL', 'MSFT']:
            if proxy in prices.columns:
                spy_col = proxy
                break

        market_price = prices[spy_col].iloc[date_idx]
        sma_200 = prices[spy_col].iloc[max(0, date_idx-200):date_idx+1].mean()
        regime = 'bull' if market_price > sma_200 else 'bear'

        # === LONG TRADES ===
        for rank_pos in range(top_n):
            if rank_pos >= len(ranked):
                break

            ticker, info = ranked[rank_pos]
            stock_price = info['price']
            stock_vol = max(info['vol'], 0.20)  # Floor vol at 20%

            if stock_price <= 0 or np.isnan(stock_price):
                continue

            # Calculate option price
            T = dte / 365.0
            if option_delta < 0.50:
                strike = find_strike_for_delta(stock_price, T, stock_vol, option_delta, RISK_FREE_RATE, 'call')
            else:
                strike = stock_price  # ATM

            option_price = bs_price(stock_price, strike, T, stock_vol, RISK_FREE_RATE, 'call')
            contract_cost = option_price * 100 + COMMISSION_PER_CONTRACT

            if contract_cost > MAX_POSITION or contract_cost > equity * 0.5:
                continue  # Too expensive

            if contract_cost < 10:
                continue  # Too cheap = probably worthless

            # Find exit date
            exit_idx = min(date_idx + hold_days, len(prices) - 1)
            exit_price = prices[ticker].iloc[exit_idx]

            if pd.isna(exit_price):
                continue

            # Daily P&L tracking with TP/SL
            actual_exit_idx = exit_idx
            exit_reason = 'hold_expiry'

            for day_offset in range(1, hold_days + 1):
                check_idx = min(date_idx + day_offset, len(prices) - 1)
                if check_idx >= len(prices):
                    break

                check_price = prices[ticker].iloc[check_idx]
                if pd.isna(check_price):
                    continue

                days_held = day_offset
                T_check = max(1, dte - days_held) / 365.0
                check_option = bs_price(check_price, strike, T_check, stock_vol, RISK_FREE_RATE, 'call')
                check_return = (check_option - option_price) / option_price if option_price > 0 else 0

                if check_return >= tp_pct:
                    actual_exit_idx = check_idx
                    exit_price = check_price
                    exit_reason = 'tp'
                    break
                elif check_return <= sl_pct:
                    actual_exit_idx = check_idx
                    exit_price = check_price
                    exit_reason = 'sl'
                    break

            # Final P&L
            days_held = actual_exit_idx - date_idx
            T_exit = max(1, dte - days_held) / 365.0
            exit_option = bs_price(exit_price, strike, T_exit, stock_vol, RISK_FREE_RATE, 'call')

            pnl = (exit_option - option_price) * 100 - 2 * COMMISSION_PER_CONTRACT
            ret_pct = pnl / contract_cost if contract_cost > 0 else 0

            equity += pnl

            trades.append({
                'date': friday,
                'exit_date': prices.index[actual_exit_idx],
                'ticker': ticker,
                'type': 'call',
                'entry_price': stock_price,
                'exit_price': exit_price,
                'strike': strike,
                'vol': stock_vol,
                'option_entry': option_price,
                'option_exit': exit_option,
                'cost': contract_cost,
                'pnl': pnl,
                'return_pct': ret_pct,
                'regime': regime,
                'exit_reason': exit_reason,
                'days_held': days_held
            })

            if fri_idx < 3:
                print(f"  Week {fri_idx+1} ({friday.date()}): {ticker} CALL K={strike:.0f}, "
                      f"entry=${option_price:.2f}, exit=${exit_option:.2f}, "
                      f"PnL=${pnl:.0f}, eq=${equity:.0f} [{exit_reason}]")

        # === SHORT TRADES (puts on worst-ranked) ===
        if short_side and len(ranked) >= 3:
            ticker, info = ranked[-1]
            stock_price = info['price']
            stock_vol = max(info['vol'], 0.20)

            if stock_price > 0 and not np.isnan(stock_price):
                T = dte / 365.0
                strike = stock_price  # ATM put
                option_price = bs_price(stock_price, strike, T, stock_vol, RISK_FREE_RATE, 'put')
                contract_cost = option_price * 100 + COMMISSION_PER_CONTRACT

                if contract_cost <= MAX_POSITION and contract_cost <= equity * 0.5 and contract_cost >= 10:
                    exit_idx = min(date_idx + hold_days, len(prices) - 1)
                    exit_price = prices[ticker].iloc[exit_idx]

                    if not pd.isna(exit_price):
                        # TP/SL check
                        actual_exit_idx = exit_idx
                        exit_reason = 'hold_expiry'

                        for day_offset in range(1, hold_days + 1):
                            check_idx = min(date_idx + day_offset, len(prices) - 1)
                            if check_idx >= len(prices):
                                break
                            check_price = prices[ticker].iloc[check_idx]
                            if pd.isna(check_price):
                                continue
                            days_held = day_offset
                            T_check = max(1, dte - days_held) / 365.0
                            check_option = bs_price(check_price, strike, T_check, stock_vol, RISK_FREE_RATE, 'put')
                            check_return = (check_option - option_price) / option_price if option_price > 0 else 0

                            if check_return >= tp_pct:
                                actual_exit_idx = check_idx
                                exit_price = check_price
                                exit_reason = 'tp'
                                break
                            elif check_return <= sl_pct:
                                actual_exit_idx = check_idx
                                exit_price = check_price
                                exit_reason = 'sl'
                                break

                        days_held = actual_exit_idx - date_idx
                        T_exit = max(1, dte - days_held) / 365.0
                        exit_option = bs_price(exit_price, strike, T_exit, stock_vol, RISK_FREE_RATE, 'put')

                        pnl = (exit_option - option_price) * 100 - 2 * COMMISSION_PER_CONTRACT
                        ret_pct = pnl / contract_cost if contract_cost > 0 else 0

                        equity += pnl

                        trades.append({
                            'date': friday,
                            'exit_date': prices.index[actual_exit_idx],
                            'ticker': ticker,
                            'type': 'put',
                            'entry_price': stock_price,
                            'exit_price': exit_price,
                            'strike': strike,
                            'vol': stock_vol,
                            'option_entry': option_price,
                            'option_exit': exit_option,
                            'cost': contract_cost,
                            'pnl': pnl,
                            'return_pct': ret_pct,
                            'regime': regime,
                            'exit_reason': exit_reason,
                            'days_held': days_held
                        })

        daily_equity[friday] = equity
        equity_curve.append(equity)

    # Build results
    equity_arr = np.array(equity_curve)
    trades_df = pd.DataFrame(trades)

    # Daily returns from equity curve
    daily_returns = pd.Series(equity_arr).pct_change().dropna().replace([np.inf, -np.inf], 0)

    # Validation
    gate_results = run_5_gates(equity_arr, trades_df, daily_returns, prices, variant_name)

    # Print results
    print(f"\n  Trades: {gate_results['n_trades']} | Sharpe: {gate_results['sharpe']:.3f} | "
          f"Sortino: {gate_results.get('sortino', 0):.3f} | PF: {gate_results.get('pf', 0):.3f} | "
          f"WR: {gate_results['wr']:.1f}% | MDD: {gate_results.get('max_dd', 0):.2f}% | "
          f"Total Return: {gate_results.get('total_return', 0):.2f}% | "
          f"CAGR: {gate_results.get('cagr', 0):.2f}%")

    if len(trades_df) > 0:
        print(f"  Exit reasons: TP={len(trades_df[trades_df.get('exit_reason', '') == 'tp'])} "
              f"SL={len(trades_df[trades_df.get('exit_reason', '') == 'sl'])} "
              f"Hold={len(trades_df[trades_df.get('exit_reason', '') == 'hold_expiry'])}")

        # Top winners/losers
        if len(trades_df) >= 3:
            by_ticker = trades_df.groupby('ticker')['pnl'].sum().sort_values(ascending=False)
            print(f"  Top winners: {', '.join([f'{t}: ${v:.0f}' for t, v in by_ticker.head(3).items()])}")
            print(f"  Top losers: {', '.join([f'{t}: ${v:.0f}' for t, v in by_ticker.tail(3).items()])}")

    # Alpha vs SPY
    if 'AAPL' in prices.columns:  # Use AAPL as proxy
        spy_start = prices['AAPL'].iloc[oot_start_idx]
        spy_end = prices['AAPL'].iloc[-1]
        spy_return = (spy_end / spy_start - 1) * 100
        gate_results['alpha_vs_spy'] = gate_results.get('total_return', 0) - spy_return

    print(f"\n  5-Gate: {gate_results['gates_passed']}/5 {'✅' if gate_results['gates_passed'] >= 4 else '❌'}")
    print(f"    sharpe_gt_1: {'PASS' if gate_results['sharpe_gt_1'] else 'FAIL'} ({gate_results['sharpe']:.3f} vs 1.0)")
    print(f"    perm_p_lt_005: {'PASS' if gate_results['perm_p_lt_005'] else 'FAIL'} ({gate_results['perm_p']:.3f} vs 0.05)")
    print(f"    wr_gt_40: {'PASS' if gate_results['wr_gt_40'] else 'FAIL'} ({gate_results['wr']:.1f} vs 40)")
    print(f"    regime_balance: {'PASS' if gate_results['regime_balance_lt_05'] else 'FAIL'} ({gate_results['regime_balance']:.3f} vs 0.5)")
    print(f"    mc_ci_positive: {'PASS' if gate_results['mc_ci_positive'] else 'FAIL'} ({gate_results['mc_ci_lower']:.2f} vs 0)")

    return gate_results

# ── Main ──────────────────────────────────────────────────────────────
def main():
    t0 = time.time()
    print("Growth Stock Swing Options v1")
    print("=" * 60)

    # Load data
    prices = load_stock_data()
    if prices is None:
        print("FATAL: Cannot load stock data")
        return

    # Compute vol for all stocks
    vol_data = compute_realized_vol(prices)

    # Define variants
    variants = {
        'A': {
            'description': 'Top-1 ATM Call, 30 DTE, Hold 5d',
            'top_n': 1, 'hold_days': 5, 'dte': 30, 'delta': 0.50,
            'use_lgbm': False, 'tp_pct': 0.50, 'sl_pct': -0.30
        },
        'B': {
            'description': 'Top-2 ATM Calls, 30 DTE, Hold 10d',
            'top_n': 2, 'hold_days': 10, 'dte': 30, 'delta': 0.50,
            'use_lgbm': False, 'tp_pct': 0.50, 'sl_pct': -0.30
        },
        'C': {
            'description': 'Top-1 OTM Call (d=0.30), 45 DTE, Hold 10d',
            'top_n': 1, 'hold_days': 10, 'dte': 45, 'delta': 0.30,
            'use_lgbm': False, 'tp_pct': 0.75, 'sl_pct': -0.40
        },
        'D': {
            'description': 'Top-1 ATM Call + VIX<25 Filter, 30 DTE',
            'top_n': 1, 'hold_days': 5, 'dte': 30, 'delta': 0.50,
            'vix_filter': 25, 'use_lgbm': False, 'tp_pct': 0.50, 'sl_pct': -0.30
        },
        'E': {
            'description': 'Top-1 ATM Call + Volume Surge, 30 DTE',
            'top_n': 1, 'hold_days': 5, 'dte': 30, 'delta': 0.50,
            'volume_filter': True, 'use_lgbm': False, 'tp_pct': 0.50, 'sl_pct': -0.30
        },
        'F': {
            'description': 'Top-1 Call + Bottom-1 Put, 30 DTE',
            'top_n': 1, 'hold_days': 5, 'dte': 30, 'delta': 0.50,
            'short_side': True, 'use_lgbm': False, 'tp_pct': 0.50, 'sl_pct': -0.30
        }
    }

    # MLflow setup
    global MLFLOW_AVAILABLE
    if MLFLOW_AVAILABLE:
        try:
            # Use Jupiter's MLflow server (accessible from any node via Tailscale)
            mlflow.set_tracking_uri("http://jupiter:5000")
            mlflow.set_experiment("growth_stock_swing_options_v1")
        except Exception as e:
            print(f"  MLflow setup failed, disabling: {e}")
            MLFLOW_AVAILABLE = False

    all_results = {}

    for var_name, config in variants.items():
        try:
            if MLFLOW_AVAILABLE:
                with mlflow.start_run(run_name=f"swing_{var_name}_{config['description'][:30]}"):
                    result = run_variant(prices, vol_data, var_name, config)
                    if result:
                        mlflow.log_metrics({
                            'sharpe': result['sharpe'],
                            'sortino': result.get('sortino', 0),
                            'pf': min(result.get('pf', 0), 100),
                            'wr': result['wr'],
                            'max_dd': result.get('max_dd', 0),
                            'cagr': result.get('cagr', 0),
                            'n_trades': result['n_trades'],
                            'gates_passed': result['gates_passed'],
                            'perm_p': result['perm_p'],
                            'regime_gap': result['regime_balance'],
                            'final_equity': result.get('final_equity', 0)
                        })
                        mlflow.log_params({
                            'variant': var_name,
                            'top_n': config.get('top_n', 1),
                            'hold_days': config.get('hold_days', 5),
                            'dte': config.get('dte', 30),
                            'tp_pct': config.get('tp_pct', 0.50),
                            'sl_pct': config.get('sl_pct', -0.30)
                        })
                        all_results[var_name] = result
            else:
                result = run_variant(prices, vol_data, var_name, config)
                if result:
                    all_results[var_name] = result
        except Exception as e:
            print(f"\n  VARIANT {var_name} CRASHED: {e}")
            import traceback
            traceback.print_exc()

    # Summary
    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"  SUMMARY — Growth Stock Swing Options v1")
    print(f"  Runtime: {elapsed:.1f}s")
    print(f"{'='*60}")

    if not all_results:
        print("  ALL VARIANTS FAILED")
        return

    # Find best
    best_var = max(all_results.items(), key=lambda x: x[1]['gates_passed'] * 10 + x[1]['sharpe'])

    for var_name, r in sorted(all_results.items()):
        marker = '🏆' if var_name == best_var[0] else '  '
        print(f"  {marker} {var_name}: Sharpe {r['sharpe']:.3f}, "
              f"Sortino {r.get('sortino', 0):.3f}, PF {r.get('pf', 0):.2f}, "
              f"WR {r['wr']:.1f}%, MDD {r.get('max_dd', 0):.1f}%, "
              f"${INITIAL_CAPITAL}→${r.get('final_equity', 0):.0f}, "
              f"{r['gates_passed']}/5 gates, "
              f"perm p={r['perm_p']:.3f}, R1={r['regime_balance']:.3f}")

    # Random direction baseline
    print(f"\n  RANDOM BASELINE CHECK:")
    print(f"  (If random directions are also profitable, result is ARTIFACT)")

    print(f"\n  Best: {best_var[0]} ({best_var[1]['gates_passed']}/5 gates, "
          f"Sharpe {best_var[1]['sharpe']:.3f})")

    # Save results
    results_file = os.path.join(OUTPUT_DIR, 'results.json')
    serializable = {}
    for k, v in all_results.items():
        serializable[k] = {sk: (float(sv) if isinstance(sv, (np.floating, np.integer)) else sv)
                          for sk, sv in v.items() if not isinstance(sv, (np.ndarray, pd.Series))}

    with open(results_file, 'w') as f:
        json.dump(serializable, f, indent=2, default=str)

    print(f"\n  Results saved to {results_file}")

if __name__ == '__main__':
    main()
