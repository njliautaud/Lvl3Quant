#!/usr/bin/env python3
"""
Agentic Growth Engine v1 — Combines ALL Validated Strategies
=============================================================
META-STRATEGY: Instead of testing yet another standalone approach,
this COMBINES the three validated strategies into an optimal whole:

1. SECTOR ROTATION (Sharpe 1.40): LGBM ranking of 11 sector ETFs
   - Buy/sell top/bottom sectors with equity shares
   - Monthly rebalance, enhanced features (corr_to_spy, beta_to_spy)
   - PORTFOLIO MANAGEMENT TRACK (but also used for agentic sizing)

2. FACTOR ETF ROTATION (Sharpe 1.13): LGBM ranking of 10 factor ETFs
   - Complementary alpha source (different from sector momentum)
   - Biweekly rebalance with trailing stop

3. MOMENTUM BURST (Sharpe 1.28): Short-term options on high-conviction signals
   - 2+ momentum signals, ATM, DTE14, trailing stop
   - Only triggered when LGBM ranking has extreme conviction

COMBINATION VARIANTS:
  A: Equity rotation base + options overlay on high conviction
  B: 60/40 split between sector and factor rotation (equity only)
  C: Signal-weighted allocation (more capital to stronger signal)
  D: Sequential growth (equity rotation until $1500, then add options)
  E: Risk-managed combo (stop trading when DD > 15%, resume at recovery)
  F: Aggressive combo (higher allocation, lower conviction threshold)
  G: Momentum timing (only trade when VIX 12-25 and SPY above 50d MA)
  H: Full conviction combo (all three running simultaneously)

TRACK: HIGH-GROWTH (agentic account $645)
"""

import sys, os, json, warnings
import numpy as np
import pandas as pd
from datetime import datetime
from scipy.stats import norm
warnings.filterwarnings('ignore')

for root in ['/home/jupiter/Lvl3Quant', '/home/nick/Lvl3Quant']:
    if os.path.isdir(root):
        LVL3_ROOT = root
        break
else:
    LVL3_ROOT = '.'

OUTPUT_DIR = os.path.join(LVL3_ROOT, 'output', 'growth_research', 'agentic_growth_engine_v1')
os.makedirs(OUTPUT_DIR, exist_ok=True)

try:
    import mlflow
    MLFLOW_AVAILABLE = True
    mlflow.set_tracking_uri("http://jupiter:5000")
    mlflow.set_experiment("agentic_growth_engine_v1")
    print("MLflow OK")
except:
    MLFLOW_AVAILABLE = False

print(f"Running on: {LVL3_ROOT}")

# ============================================================
# CONSTANTS
# ============================================================
SECTOR_ETFS = ['XLK', 'XLF', 'XLE', 'XLV', 'XLY', 'XLP', 'XLI', 'XLB', 'XLU', 'XLRE', 'XLC']
FACTOR_ETFS = ['MTUM', 'VLUE', 'QUAL', 'SIZE', 'USMV', 'VTV', 'VUG', 'MOAT', 'COWZ', 'NOBL']
ALL_TICKERS = list(set(SECTOR_ETFS + FACTOR_ETFS + ['SPY', '^VIX']))

STARTING_CAPITAL = 645.0
COMMISSION_PER_LEG = 0.65
START_DATE = '2017-01-01'
END_DATE = '2026-07-28'
OOT_START = '2019-01-01'
N_PERMUTATIONS = 150


# ============================================================
# DATA
# ============================================================
def load_data():
    cache_path = os.path.join(LVL3_ROOT, 'data', 'agentic_growth_engine_cache.parquet')
    if os.path.exists(cache_path):
        df = pd.read_parquet(cache_path)
        if len(df) > 0 and pd.Timestamp(df.index.get_level_values('date').max()) >= pd.Timestamp('2026-07-20'):
            print(f"Cached: {len(df)} rows")
            return df

    import yfinance as yf
    print(f"Downloading {len(ALL_TICKERS)} tickers...")
    frames = []
    for t in ALL_TICKERS:
        try:
            d = yf.download(t, start=START_DATE, end=END_DATE, progress=False, auto_adjust=True)
            if len(d) < 100: continue
            d.columns = [c.lower() if isinstance(c, str) else c[0].lower() for c in d.columns]
            d['ticker'] = t
            d.index.name = 'date'
            frames.append(d)
            print(f"  {t}: {len(d)} rows")
        except:
            pass
    df = pd.concat(frames).reset_index().set_index(['ticker', 'date']).sort_index()
    try:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        df.to_parquet(cache_path)
    except: pass
    return df


# ============================================================
# LGBM RANKING (simplified walk-forward)
# ============================================================
def lgbm_rank_etfs(prices_df, etf_universe, date, lookback=400, predict_horizon=21):
    """
    Simplified LGBM ranking: train on lookback days, predict forward returns.
    Returns dict: {ticker: predicted_return_rank}
    """
    from sklearn.ensemble import GradientBoostingRegressor

    features_all = {}
    targets_all = {}

    for ticker in etf_universe:
        try:
            td = prices_df.loc[ticker]
        except KeyError:
            continue

        mask = td.index <= date
        hist = td.loc[mask]
        if len(hist) < lookback + predict_horizon + 60:
            continue

        close = hist['close'].values
        volume = hist['volume'].values
        dates_idx = hist.index

        # Build features for each date in lookback window
        for offset in range(predict_horizon, lookback):
            idx = len(close) - offset
            if idx < 65:
                continue

            c = close[:idx]
            v = volume[:idx]

            # 17 base features + 2 cross-sector (simplified)
            feats = {}
            feats['ret_5d'] = (c[-1] / c[-6]) - 1 if len(c) >= 6 else 0
            feats['ret_10d'] = (c[-1] / c[-11]) - 1 if len(c) >= 11 else 0
            feats['ret_21d'] = (c[-1] / c[-22]) - 1 if len(c) >= 22 else 0
            feats['ret_63d'] = (c[-1] / c[-64]) - 1 if len(c) >= 64 else 0
            feats['vol_21d'] = np.std(np.diff(np.log(c[-22:]))) * np.sqrt(252) if len(c) >= 22 else 0.2
            feats['vol_63d'] = np.std(np.diff(np.log(c[-64:]))) * np.sqrt(252) if len(c) >= 64 else 0.2

            ma20 = np.mean(c[-20:]) if len(c) >= 20 else c[-1]
            ma50 = np.mean(c[-50:]) if len(c) >= 50 else c[-1]
            feats['pct_above_ma20'] = (c[-1] - ma20) / ma20
            feats['pct_above_ma50'] = (c[-1] - ma50) / ma50
            feats['ma20_slope'] = (np.mean(c[-5:]) - np.mean(c[-10:-5])) / (np.mean(c[-10:-5]) + 1e-8) if len(c) >= 10 else 0

            # RSI
            if len(c) >= 15:
                d = np.diff(c[-15:])
                g = np.mean(np.where(d > 0, d, 0))
                l = np.mean(np.where(d < 0, -d, 0))
                feats['rsi_14'] = 100 - 100 / (1 + g / (l + 1e-8))
            else:
                feats['rsi_14'] = 50

            # Volume features
            if len(v) >= 21:
                feats['vol_ratio'] = v[-1] / (np.mean(v[-21:-1]) + 1e-8)
            else:
                feats['vol_ratio'] = 1.0

            feats['ret_252d'] = (c[-1] / c[-min(252, len(c)-1)]) - 1

            # Cross-sector features (correlation + beta to SPY proxy)
            # Use the return series for correlation
            if len(c) >= 64:
                spy_mask = prices_df.loc['SPY'].index <= dates_idx[idx-1]
                spy_hist = prices_df.loc['SPY'].loc[spy_mask, 'close'].values
                if len(spy_hist) >= 64:
                    etf_rets = np.diff(np.log(c[-64:]))
                    spy_rets = np.diff(np.log(spy_hist[-64:]))
                    if len(etf_rets) == len(spy_rets):
                        feats['corr_to_spy_63d'] = np.corrcoef(etf_rets, spy_rets)[0, 1]
                        cov = np.cov(etf_rets, spy_rets)
                        feats['beta_to_spy_63d'] = cov[0, 1] / (cov[1, 1] + 1e-8)
                    else:
                        feats['corr_to_spy_63d'] = 0.8
                        feats['beta_to_spy_63d'] = 1.0
                else:
                    feats['corr_to_spy_63d'] = 0.8
                    feats['beta_to_spy_63d'] = 1.0
            else:
                feats['corr_to_spy_63d'] = 0.8
                feats['beta_to_spy_63d'] = 1.0

            # Target: forward return
            fwd_idx = min(idx + predict_horizon, len(close) - 1)
            target = (close[fwd_idx] / close[idx-1]) - 1

            key = (ticker, offset)
            features_all[key] = feats
            targets_all[key] = target

    if len(features_all) < 50:
        return None

    # Build training matrix
    feat_names = sorted(list(list(features_all.values())[0].keys()))
    X = np.array([[f.get(fn, 0) for fn in feat_names] for f in features_all.values()])
    y = np.array(list(targets_all.values()))

    # Handle NaN/Inf
    X = np.nan_to_num(X, nan=0, posinf=1, neginf=-1)
    y = np.nan_to_num(y, nan=0)

    # Train LGBM (using sklearn's GBR as lightweight alternative)
    model = GradientBoostingRegressor(
        n_estimators=100, max_depth=4, learning_rate=0.1,
        subsample=0.8, random_state=42
    )
    model.fit(X, y)

    # Predict for current date
    predictions = {}
    for ticker in etf_universe:
        try:
            td = prices_df.loc[ticker]
            mask = td.index <= date
            hist = td.loc[mask]
            if len(hist) < 65:
                continue

            close = hist['close'].values
            volume = hist['volume'].values

            feats = {}
            feats['ret_5d'] = (close[-1] / close[-6]) - 1 if len(close) >= 6 else 0
            feats['ret_10d'] = (close[-1] / close[-11]) - 1 if len(close) >= 11 else 0
            feats['ret_21d'] = (close[-1] / close[-22]) - 1 if len(close) >= 22 else 0
            feats['ret_63d'] = (close[-1] / close[-64]) - 1 if len(close) >= 64 else 0
            feats['vol_21d'] = np.std(np.diff(np.log(close[-22:]))) * np.sqrt(252) if len(close) >= 22 else 0.2
            feats['vol_63d'] = np.std(np.diff(np.log(close[-64:]))) * np.sqrt(252) if len(close) >= 64 else 0.2

            ma20 = np.mean(close[-20:])
            ma50 = np.mean(close[-50:]) if len(close) >= 50 else close[-1]
            feats['pct_above_ma20'] = (close[-1] - ma20) / ma20
            feats['pct_above_ma50'] = (close[-1] - ma50) / ma50
            feats['ma20_slope'] = (np.mean(close[-5:]) - np.mean(close[-10:-5])) / (np.mean(close[-10:-5]) + 1e-8)

            d = np.diff(close[-15:])
            g = np.mean(np.where(d > 0, d, 0))
            l_val = np.mean(np.where(d < 0, -d, 0))
            feats['rsi_14'] = 100 - 100 / (1 + g / (l_val + 1e-8))

            feats['vol_ratio'] = volume[-1] / (np.mean(volume[-21:-1]) + 1e-8) if len(volume) >= 21 else 1.0
            feats['ret_252d'] = (close[-1] / close[-min(252, len(close)-1)]) - 1

            spy_hist = prices_df.loc['SPY']
            spy_mask = spy_hist.index <= date
            spy_c = spy_hist.loc[spy_mask, 'close'].values
            if len(spy_c) >= 64 and len(close) >= 64:
                er = np.diff(np.log(close[-64:]))
                sr = np.diff(np.log(spy_c[-64:]))
                if len(er) == len(sr):
                    feats['corr_to_spy_63d'] = np.corrcoef(er, sr)[0, 1]
                    cov = np.cov(er, sr)
                    feats['beta_to_spy_63d'] = cov[0, 1] / (cov[1, 1] + 1e-8)
                else:
                    feats['corr_to_spy_63d'] = 0.8
                    feats['beta_to_spy_63d'] = 1.0
            else:
                feats['corr_to_spy_63d'] = 0.8
                feats['beta_to_spy_63d'] = 1.0

            x_pred = np.array([[feats.get(fn, 0) for fn in feat_names]])
            x_pred = np.nan_to_num(x_pred, nan=0, posinf=1, neginf=-1)
            pred = model.predict(x_pred)[0]
            predictions[ticker] = pred
        except:
            continue

    if not predictions:
        return None

    # Rank by predicted return
    ranked = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
    return {ticker: (rank + 1, pred) for rank, (ticker, pred) in enumerate(ranked)}


# ============================================================
# BS OPTIONS PRICING
# ============================================================
def bs_call(S, K, T, r, sigma):
    if T <= 1e-8: return max(S - K, 0.0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

def bs_put(S, K, T, r, sigma):
    if T <= 1e-8: return max(K - S, 0.0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


# ============================================================
# STRATEGY VARIANTS
# ============================================================

VARIANTS = {
    'A': {
        'name': 'Equity Rotation + Options Overlay',
        'sector_alloc': 0.60,  # 60% to sector rotation
        'factor_alloc': 0.20,  # 20% to factor rotation
        'options_alloc': 0.20,  # 20% reserved for momentum burst options
        'options_conviction_threshold': 3,  # Need 3+ momentum signals for options
        'sector_top_n': 2,
        'factor_top_n': 1,
        'rebalance_days': 21,  # Monthly
        'trailing_stop_pct': 0.15,
        'use_options': True,
    },
    'B': {
        'name': '60/40 Sector/Factor (Equity Only)',
        'sector_alloc': 0.60,
        'factor_alloc': 0.40,
        'options_alloc': 0.00,
        'sector_top_n': 2,
        'factor_top_n': 2,
        'rebalance_days': 21,
        'trailing_stop_pct': 0.15,
        'use_options': False,
    },
    'C': {
        'name': 'Signal-Weighted (Stronger Signal Gets More)',
        'sector_alloc': 0.50,
        'factor_alloc': 0.30,
        'options_alloc': 0.20,
        'sector_top_n': 2,
        'factor_top_n': 1,
        'rebalance_days': 21,
        'trailing_stop_pct': 0.15,
        'use_options': True,
        'signal_weight': True,  # Allocate more to stronger predictions
    },
    'D': {
        'name': 'Sequential Growth (equity→options at $1500)',
        'sector_alloc': 0.80,
        'factor_alloc': 0.20,
        'options_alloc': 0.00,  # Start with 0, add options at threshold
        'options_threshold': 1500,  # Start options at $1500
        'sector_top_n': 2,
        'factor_top_n': 1,
        'rebalance_days': 21,
        'trailing_stop_pct': 0.15,
        'use_options': True,
    },
    'E': {
        'name': 'Risk-Managed (stop at 15% DD)',
        'sector_alloc': 0.60,
        'factor_alloc': 0.20,
        'options_alloc': 0.20,
        'sector_top_n': 2,
        'factor_top_n': 1,
        'rebalance_days': 21,
        'trailing_stop_pct': 0.15,
        'dd_threshold': 0.15,  # Stop trading at 15% drawdown
        'dd_recovery': 0.05,   # Resume when DD < 5%
        'use_options': True,
    },
    'F': {
        'name': 'Aggressive (90% invested, top-1 only)',
        'sector_alloc': 0.70,
        'factor_alloc': 0.20,
        'options_alloc': 0.10,
        'sector_top_n': 1,  # Concentrated
        'factor_top_n': 1,
        'rebalance_days': 14,  # Biweekly
        'trailing_stop_pct': 0.12,
        'use_options': True,
    },
    'G': {
        'name': 'VIX-Timed (trade only VIX 12-25)',
        'sector_alloc': 0.60,
        'factor_alloc': 0.20,
        'options_alloc': 0.20,
        'sector_top_n': 2,
        'factor_top_n': 1,
        'rebalance_days': 21,
        'trailing_stop_pct': 0.15,
        'vix_range': (12, 25),
        'use_options': True,
    },
    'H': {
        'name': 'Full Combo (sector+factor+options)',
        'sector_alloc': 0.40,
        'factor_alloc': 0.30,
        'options_alloc': 0.30,
        'sector_top_n': 2,
        'factor_top_n': 2,
        'rebalance_days': 21,
        'trailing_stop_pct': 0.15,
        'use_options': True,
    },
}


# ============================================================
# COMBINED BACKTEST ENGINE
# ============================================================

def compute_momentum_signals(prices_df, ticker, date):
    """Same momentum burst signals as v1 (Sharpe 1.28)."""
    try:
        td = prices_df.loc[ticker]
    except KeyError:
        return 0, 'neutral'

    mask = td.index <= date
    hist = td.loc[mask]
    if len(hist) < 25:
        return 0, 'neutral'

    close = hist['close'].values
    bull = 0
    bear = 0

    # 5d momentum
    if len(close) >= 6:
        mom = (close[-1] / close[-6]) - 1
        if mom > 0.03: bull += 1
        elif mom < -0.03: bear += 1

    # RSI cross
    if len(close) >= 18:
        d = np.diff(close[-15:])
        g = np.mean(np.where(d > 0, d, 0))
        l = np.mean(np.where(d < 0, -d, 0))
        rsi = 100 - 100 / (1 + g / (l + 1e-8))
        d_prev = np.diff(close[-18:-3])
        g_p = np.mean(np.where(d_prev > 0, d_prev, 0))
        l_p = np.mean(np.where(d_prev < 0, -d_prev, 0))
        rsi_prev = 100 - 100 / (1 + g_p / (l_p + 1e-8))
        if rsi >= 60 and rsi_prev < 60: bull += 1
        if rsi <= 40 and rsi_prev > 40: bear += 1

    # Volume surge
    v = hist['volume'].values
    if len(v) >= 21:
        if v[-1] > np.mean(v[-21:-1]) * 1.5:
            if len(close) >= 6 and (close[-1] / close[-6] - 1) > 0: bull += 1
            elif len(close) >= 6: bear += 1

    direction = 'bull' if bull > bear else ('bear' if bear > bull else 'neutral')
    return max(bull, bear), direction


def run_variant(vk, cfg, prices_df, spy_prices, vix_data, trading_dates):
    """Run combined strategy."""
    import time
    t0 = time.time()

    equity = STARTING_CAPITAL
    equity_curve = [equity]
    trades = []
    holdings = {}  # {ticker: {shares, entry_price, entry_date}}
    option_positions = []  # [{ticker, type, strike, entry_premium, entry_date, dte, iv, cost}]

    oot_dates = [d for d in trading_dates if d >= pd.Timestamp(OOT_START)]
    if not oot_dates:
        return None

    # Track drawdown for risk management
    peak_equity = equity
    in_drawdown = False
    last_rebalance = 0

    for i, date in enumerate(oot_dates):
        prev_equity = equity

        # Mark-to-market holdings
        holdings_value = 0
        for ticker, h in list(holdings.items()):
            try:
                td = prices_df.loc[ticker]
                if date in td.index:
                    price = td.loc[date, 'close']
                    if isinstance(price, pd.Series): price = price.iloc[0]
                    holdings_value += h['shares'] * price

                    # Trailing stop check
                    pct_change = (price - h['entry_price']) / h['entry_price']
                    if pct_change < -cfg['trailing_stop_pct']:
                        # Sell
                        sell_value = h['shares'] * price
                        pnl = sell_value - h['cost']
                        equity += sell_value
                        trades.append({
                            'ticker': ticker, 'type': 'equity',
                            'entry_date': str(h['entry_date'].date()) if hasattr(h['entry_date'], 'date') else str(h['entry_date']),
                            'exit_date': str(date.date()),
                            'pnl': round(pnl, 2),
                            'pnl_pct': round(pct_change * 100, 1),
                            'exit_reason': 'trailing_stop',
                        })
                        del holdings[ticker]
                        holdings_value -= h['shares'] * price  # Adjust
                else:
                    holdings_value += h['shares'] * h['entry_price']
            except:
                holdings_value += h.get('cost', 0)

        # Mark-to-market options
        options_value = 0
        opts_to_close = []
        for oi, opt in enumerate(option_positions):
            opt['days_held'] = (date - opt['entry_date']).days
            try:
                td = prices_df.loc[opt['ticker']]
                if date in td.index:
                    spot = td.loc[date, 'close']
                    if isinstance(spot, pd.Series): spot = spot.iloc[0]
                    rem_dte = max(opt['dte'] - opt['days_held'], 0)
                    T = rem_dte / 252.0
                    vix_mask = vix_data.index <= date
                    vix = vix_data.loc[vix_mask, 'close'].iloc[-1] if vix_mask.any() else 20
                    if isinstance(vix, pd.Series): vix = vix.iloc[0]
                    iv = max(vix / 100, opt['iv'] * 0.95)

                    if opt['option_type'] == 'call':
                        val = bs_call(spot, opt['strike'], T, 0.05, iv)
                    else:
                        val = bs_put(spot, opt['strike'], T, 0.05, iv)

                    pct_change = (val - opt['entry_premium']) / opt['entry_premium']

                    # Exit conditions
                    exit_reason = None
                    if pct_change >= 0.30:  # 30% TP
                        exit_reason = 'take_profit'
                    elif pct_change <= -0.25:  # 25% SL
                        exit_reason = 'stop_loss'
                    elif opt['days_held'] >= 5:  # Time stop
                        exit_reason = 'time_stop'

                    if exit_reason:
                        exit_value = val * 100
                        pnl = exit_value - opt['cost'] - COMMISSION_PER_LEG
                        equity += exit_value - COMMISSION_PER_LEG
                        trades.append({
                            'ticker': opt['ticker'], 'type': f"option_{opt['option_type']}",
                            'entry_date': str(opt['entry_date'].date()),
                            'exit_date': str(date.date()),
                            'pnl': round(pnl, 2),
                            'pnl_pct': round(pct_change * 100, 1),
                            'exit_reason': exit_reason,
                        })
                        opts_to_close.append(oi)
                    else:
                        options_value += val * 100
            except:
                options_value += opt.get('cost', 0)

        for oi in sorted(opts_to_close, reverse=True):
            option_positions.pop(oi)

        total_value = equity + holdings_value + options_value

        # Drawdown management
        if total_value > peak_equity:
            peak_equity = total_value
        dd = (total_value - peak_equity) / peak_equity

        if cfg.get('dd_threshold') and dd < -cfg['dd_threshold']:
            in_drawdown = True
        if cfg.get('dd_recovery') and in_drawdown and dd > -cfg.get('dd_recovery', 0.05):
            in_drawdown = False

        # VIX filter
        skip_trading = False
        if cfg.get('vix_range'):
            vix_mask = vix_data.index <= date
            vix = vix_data.loc[vix_mask, 'close'].iloc[-1] if vix_mask.any() else 20
            if isinstance(vix, pd.Series): vix = vix.iloc[0]
            lo, hi = cfg['vix_range']
            if vix < lo or vix > hi:
                skip_trading = True

        # Rebalance
        if i - last_rebalance >= cfg['rebalance_days'] and not in_drawdown and not skip_trading:
            last_rebalance = i

            # Sequential growth: check if we should add options
            options_alloc = cfg['options_alloc']
            if cfg.get('options_threshold') and total_value < cfg['options_threshold']:
                options_alloc = 0  # No options until threshold
                sector_alloc = cfg['sector_alloc'] + cfg['options_alloc'] * 0.7
                factor_alloc = cfg['factor_alloc'] + cfg['options_alloc'] * 0.3
            else:
                sector_alloc = cfg['sector_alloc']
                factor_alloc = cfg['factor_alloc']

            # 1. SELL ALL HOLDINGS
            for ticker, h in list(holdings.items()):
                try:
                    td = prices_df.loc[ticker]
                    if date in td.index:
                        price = td.loc[date, 'close']
                        if isinstance(price, pd.Series): price = price.iloc[0]
                        sell_value = h['shares'] * price
                        pnl = sell_value - h['cost']
                        equity += sell_value
                        trades.append({
                            'ticker': ticker, 'type': 'equity',
                            'entry_date': str(h['entry_date'].date()) if hasattr(h['entry_date'], 'date') else str(h['entry_date']),
                            'exit_date': str(date.date()),
                            'pnl': round(pnl, 2),
                            'pnl_pct': round((price / h['entry_price'] - 1) * 100, 1),
                            'exit_reason': 'rebalance',
                        })
                except:
                    pass
            holdings = {}

            # Recalculate total value
            total_value = equity + options_value

            # 2. LGBM SECTOR RANKING
            sector_ranking = lgbm_rank_etfs(prices_df, SECTOR_ETFS, date)
            if sector_ranking:
                sector_budget = total_value * sector_alloc
                top_sectors = sorted(sector_ranking.items(), key=lambda x: x[1][0])[:cfg['sector_top_n']]

                per_sector = sector_budget / len(top_sectors) if top_sectors else 0
                for ticker, (rank, pred) in top_sectors:
                    try:
                        td = prices_df.loc[ticker]
                        if date in td.index:
                            price = td.loc[date, 'close']
                            if isinstance(price, pd.Series): price = price.iloc[0]

                            # Signal weighting
                            alloc = per_sector
                            if cfg.get('signal_weight') and pred > 0:
                                alloc *= min(2.0, 1 + pred * 10)

                            shares = alloc / price
                            cost = shares * price
                            if cost > equity:
                                shares = equity * 0.95 / price
                                cost = shares * price

                            if cost > 10:
                                equity -= cost
                                holdings[ticker] = {
                                    'shares': shares,
                                    'entry_price': price,
                                    'entry_date': date,
                                    'cost': cost,
                                }
                    except:
                        pass

            # 3. LGBM FACTOR RANKING
            avail_factors = [f for f in FACTOR_ETFS if f in prices_df.index.get_level_values('ticker')]
            if avail_factors:
                factor_ranking = lgbm_rank_etfs(prices_df, avail_factors, date)
                if factor_ranking:
                    factor_budget = total_value * factor_alloc
                    top_factors = sorted(factor_ranking.items(), key=lambda x: x[1][0])[:cfg['factor_top_n']]

                    per_factor = factor_budget / len(top_factors) if top_factors else 0
                    for ticker, (rank, pred) in top_factors:
                        try:
                            td = prices_df.loc[ticker]
                            if date in td.index:
                                price = td.loc[date, 'close']
                                if isinstance(price, pd.Series): price = price.iloc[0]
                                shares = per_factor / price
                                cost = shares * price
                                if cost > equity:
                                    shares = equity * 0.95 / price
                                    cost = shares * price
                                if cost > 10:
                                    equity -= cost
                                    holdings[ticker] = {
                                        'shares': shares,
                                        'entry_price': price,
                                        'entry_date': date,
                                        'cost': cost,
                                    }
                        except:
                            pass

            # 4. OPTIONS OVERLAY (if applicable)
            if cfg['use_options'] and options_alloc > 0 and len(option_positions) == 0:
                options_budget = total_value * options_alloc
                if options_budget > 80:
                    # Find highest conviction momentum burst signal
                    best_signal = None
                    best_conv = 0
                    best_dir = None

                    for ticker in SECTOR_ETFS:
                        conv, direction = compute_momentum_signals(prices_df, ticker, date)
                        threshold = cfg.get('options_conviction_threshold', 2)
                        if conv >= threshold and conv > best_conv:
                            best_signal = ticker
                            best_conv = conv
                            best_dir = direction

                    if best_signal and best_dir != 'neutral':
                        try:
                            td = prices_df.loc[best_signal]
                            if date in td.index:
                                spot = td.loc[date, 'close']
                                if isinstance(spot, pd.Series): spot = spot.iloc[0]

                                vix_mask = vix_data.index <= date
                                vix = vix_data.loc[vix_mask, 'close'].iloc[-1] if vix_mask.any() else 20
                                if isinstance(vix, pd.Series): vix = vix.iloc[0]

                                hist_close = td.loc[td.index <= date, 'close'].values
                                rv = np.std(np.diff(np.log(hist_close[-22:]))) * np.sqrt(252) if len(hist_close) >= 22 else 0.25
                                iv = max(vix / 100, rv * 1.2)

                                strike = round(spot, 0)
                                T = 14 / 252.0
                                opt_type = 'call' if best_dir == 'bull' else 'put'
                                if opt_type == 'call':
                                    prem = bs_call(spot, strike, T, 0.05, iv)
                                else:
                                    prem = bs_put(spot, strike, T, 0.05, iv)

                                contract_cost = prem * 100
                                if 10 < contract_cost < min(options_budget, equity):
                                    equity -= contract_cost + COMMISSION_PER_LEG
                                    option_positions.append({
                                        'ticker': best_signal,
                                        'option_type': opt_type,
                                        'strike': strike,
                                        'entry_premium': prem,
                                        'entry_date': date,
                                        'dte': 14,
                                        'iv': iv,
                                        'cost': contract_cost + COMMISSION_PER_LEG,
                                        'days_held': 0,
                                    })
                        except:
                            pass

        # Update equity curve
        holdings_val = 0
        for ticker, h in holdings.items():
            try:
                td = prices_df.loc[ticker]
                if date in td.index:
                    p = td.loc[date, 'close']
                    if isinstance(p, pd.Series): p = p.iloc[0]
                    holdings_val += h['shares'] * p
                else:
                    holdings_val += h['cost']
            except:
                holdings_val += h.get('cost', 0)

        opts_val = 0
        for opt in option_positions:
            opts_val += opt.get('cost', 0)  # Approximate

        total = equity + holdings_val + opts_val
        equity_curve.append(total)

    runtime = time.time() - t0
    return {
        'equity_curve': equity_curve,
        'trades': trades,
        'final_equity': equity_curve[-1],
        'runtime': runtime,
    }


# ============================================================
# METRICS
# ============================================================

def compute_metrics(result, vk, cfg, spy_prices, oot_dates):
    eq = np.array(result['equity_curve'])
    trades = result['trades']
    final = result['final_equity']

    m = {
        'variant': vk, 'name': cfg['name'],
        'final_equity': round(final, 2),
        'total_return_pct': round((final / STARTING_CAPITAL - 1) * 100, 1),
        'total_trades': len(trades),
        'runtime': round(result.get('runtime', 0), 1),
    }

    if len(eq) < 10:
        m.update({'sharpe': 0, 'sortino': 0, 'win_rate': 0, 'profit_factor': 0,
                  'max_drawdown_pct': 0, 'cagr_pct': 0, 'regime_gap': 999})
        return m

    dr = np.diff(eq) / np.maximum(eq[:-1], 1)
    ann = np.sqrt(252)
    mu = np.mean(dr)
    sigma = np.std(dr)
    m['sharpe'] = round((mu / sigma) * ann, 2) if sigma > 1e-10 else 0

    neg = dr[dr < 0]
    ds = np.std(neg) if len(neg) > 0 else sigma
    m['sortino'] = round((mu / ds) * ann, 2) if ds > 1e-10 else 0

    peak = np.maximum.accumulate(eq)
    dd = (eq - peak) / np.maximum(peak, 1)
    m['max_drawdown_pct'] = round(np.min(dd) * 100, 1)

    n_years = len(dr) / 252
    m['cagr_pct'] = round(((final / STARTING_CAPITAL) ** (1/max(n_years, 0.1)) - 1) * 100, 1)

    if trades:
        pnls = [t['pnl'] for t in trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        m['win_rate'] = round(len(wins) / len(pnls) * 100, 1) if pnls else 0
        gp = sum(wins) if wins else 0
        gl = abs(sum(losses)) if losses else 1e-8
        m['profit_factor'] = round(gp / gl, 2) if gl > 0 else 999

        equity_trades = [t for t in trades if t['type'] == 'equity']
        option_trades = [t for t in trades if 'option' in t['type']]
        m['equity_trades'] = len(equity_trades)
        m['option_trades'] = len(option_trades)
    else:
        m['win_rate'] = 0
        m['profit_factor'] = 0
        m['equity_trades'] = 0
        m['option_trades'] = 0

    # Regime
    try:
        spy_rets = spy_prices.loc[spy_prices.index.isin(oot_dates), 'close'].pct_change().dropna()
        green = set(spy_rets[spy_rets > 0].index)
        red = set(spy_rets[spy_rets <= 0].index)
        gp = sum(t['pnl'] for t in trades if pd.Timestamp(t['entry_date']) in green)
        rp = sum(t['pnl'] for t in trades if pd.Timestamp(t['entry_date']) in red)
        total = abs(gp) + abs(rp)
        m['regime_gap'] = round(abs(gp - rp) / total, 3) if total > 0 else 0
    except:
        m['regime_gap'] = 999

    # SPY comparison
    try:
        spy_start = spy_prices.loc[spy_prices.index >= pd.Timestamp(OOT_START), 'close'].iloc[0]
        spy_end = spy_prices.loc[spy_prices.index >= pd.Timestamp(OOT_START), 'close'].iloc[-1]
        m['alpha_vs_spy'] = round(m['total_return_pct'] - (spy_end / spy_start - 1) * 100, 1)
    except:
        m['alpha_vs_spy'] = 0

    return m


def permutation_test(cfg, prices_df, spy_prices, vix_data, oot_dates, actual_sharpe, n_perms=150):
    """Random ETF selection baseline."""
    random_sharpes = []
    all_etfs = [t for t in SECTOR_ETFS + FACTOR_ETFS if t in prices_df.index.get_level_values('ticker')]

    for perm in range(n_perms):
        np.random.seed(perm + 42)
        eq = STARTING_CAPITAL
        curve = [eq]

        for i, date in enumerate(oot_dates):
            if i % cfg['rebalance_days'] == 0:
                # Random selection
                picks = np.random.choice(all_etfs, size=min(3, len(all_etfs)), replace=False)
                alloc_per = eq * 0.80 / len(picks)

                holdings = {}
                cash = eq * 0.20
                for t in picks:
                    try:
                        td = prices_df.loc[t]
                        if date in td.index:
                            p = td.loc[date, 'close']
                            if isinstance(p, pd.Series): p = p.iloc[0]
                            holdings[t] = {'shares': alloc_per / p, 'price': p}
                    except:
                        pass

            # MTM
            val = cash if 'cash' in dir() else eq * 0.20
            for t, h in holdings.items() if 'holdings' in dir() else {}:
                try:
                    td = prices_df.loc[t]
                    if date in td.index:
                        p = td.loc[date, 'close']
                        if isinstance(p, pd.Series): p = p.iloc[0]
                        val += h['shares'] * p
                except:
                    val += h.get('shares', 0) * h.get('price', 0)

            if 'holdings' in dir() and holdings:
                eq = val
            curve.append(eq)

        c = np.array(curve)
        dr = np.diff(c) / np.maximum(c[:-1], 1)
        if len(dr) > 20 and np.std(dr) > 1e-10:
            random_sharpes.append((np.mean(dr) / np.std(dr)) * np.sqrt(252))

    if not random_sharpes: return 1.0, 0.0
    return np.mean([s >= actual_sharpe for s in random_sharpes]), np.mean(random_sharpes)


# ============================================================
# MAIN
# ============================================================

def main():
    import time
    print("=" * 70)
    print("  AGENTIC GROWTH ENGINE V1")
    print("  Combines validated strategies: sector rotation + factor rotation + momentum burst")
    print("  Target: $645 agentic account high-growth mode")
    print("=" * 70)

    prices_df = load_data()
    spy_prices = prices_df.loc['SPY']
    vix_data = prices_df.loc['^VIX'] if '^VIX' in prices_df.index.get_level_values('ticker') else None

    trading_dates = sorted(spy_prices.index.unique())
    oot_dates = [d for d in trading_dates if d >= pd.Timestamp(OOT_START)]
    avail_sectors = [t for t in SECTOR_ETFS if t in prices_df.index.get_level_values('ticker')]
    avail_factors = [t for t in FACTOR_ETFS if t in prices_df.index.get_level_values('ticker')]

    print(f"\nOOT: {oot_dates[0].date()} to {oot_dates[-1].date()} ({len(oot_dates)} days)")
    print(f"Sectors: {len(avail_sectors)}, Factors: {len(avail_factors)}")

    all_metrics = {}

    for vk in sorted(VARIANTS.keys()):
        cfg = VARIANTS[vk]
        print(f"\n{'=' * 60}")
        print(f"  VARIANT {vk}: {cfg['name']}")
        print(f"{'=' * 60}")

        result = run_variant(vk, cfg, prices_df, spy_prices, vix_data, trading_dates)
        if result is None:
            print(f"  SKIP: No result")
            all_metrics[vk] = {'variant': vk, 'name': cfg['name'], 'sharpe': 0, 'gates_passed': 0}
            continue

        metrics = compute_metrics(result, vk, cfg, spy_prices, oot_dates)

        for t in result['trades'][:3]:
            print(f"  {t.get('entry_date', '?')} → {t.get('exit_date', '?')}: "
                  f"{t['ticker']} ({t['type']}) pnl=${t['pnl']:.0f} ({t['pnl_pct']:.1f}%) [{t['exit_reason']}]")

        print(f"  Trades: {metrics['total_trades']} (equity={metrics.get('equity_trades',0)}, "
              f"options={metrics.get('option_trades',0)})")
        print(f"  Sharpe: {metrics['sharpe']} | Sortino: {metrics['sortino']} | "
              f"PF: {metrics['profit_factor']} | WR: {metrics['win_rate']}%")
        print(f"  $645 → ${metrics['final_equity']:.0f} | Return: {metrics['total_return_pct']:.1f}% | "
              f"CAGR: {metrics['cagr_pct']:.1f}% | MDD: {metrics['max_drawdown_pct']}% | "
              f"Alpha: {metrics.get('alpha_vs_spy', 0):.1f}%")
        print(f"  Runtime: {metrics['runtime']:.1f}s")

        # Permutation test
        if metrics['total_trades'] >= 3 and metrics['sharpe'] > 0:
            print(f"  Running {N_PERMUTATIONS}-permutation test...")
            p_val, rand_sharpe = permutation_test(
                cfg, prices_df, spy_prices, vix_data, oot_dates, metrics['sharpe'], N_PERMUTATIONS)

            gates = {
                'sharpe_gt_1': metrics['sharpe'] >= 1.0,
                'perm_p_lt_005': p_val < 0.05,
                'wr_gt_40': metrics['win_rate'] >= 40.0,
                'regime_balance': metrics.get('regime_gap', 999) < 0.50,
                'beats_random': metrics['sharpe'] > rand_sharpe + 0.1,
            }
            n_pass = sum(gates.values())

            print(f"  5-Gate: {n_pass}/5 PASS")
            for gn, gp in gates.items():
                if gn == 'sharpe_gt_1': vs = f"value={metrics['sharpe']}"
                elif gn == 'perm_p_lt_005': vs = f"value={p_val:.3f}"
                elif gn == 'wr_gt_40': vs = f"value={metrics['win_rate']}"
                elif gn == 'regime_balance': vs = f"value={metrics.get('regime_gap', 999):.3f}"
                elif gn == 'beats_random': vs = f"value={metrics['sharpe']}, random={rand_sharpe:.2f}"
                else: vs = ""
                print(f"    {gn}: {'PASS' if gp else 'FAIL'} ({vs})")

            metrics['perm_p'] = round(p_val, 4)
            metrics['random_sharpe'] = round(rand_sharpe, 2)
            metrics['gates_passed'] = n_pass
        else:
            metrics['gates_passed'] = 0
            print(f"  5-Gate: 0/5 PASS")

        all_metrics[vk] = metrics

        if MLFLOW_AVAILABLE:
            try:
                with mlflow.start_run(run_name=f"engine_{vk}_{cfg['name'][:20]}"):
                    for mk, mv in metrics.items():
                        if isinstance(mv, (int, float)):
                            mlflow.log_metric(mk, mv)
            except: pass

    # Summary
    print(f"\n{'=' * 90}")
    print("  SUMMARY — AGENTIC GROWTH ENGINE V1")
    print(f"{'=' * 90}")

    sv = sorted(all_metrics.items(), key=lambda x: x[1].get('sharpe', 0), reverse=True)

    print(f"\n  {'V':<3} {'Name':<35} {'Sharpe':>7} {'Sort':>7} {'PF':>6} {'WR':>6} "
          f"{'Trd':>5} {'Return':>8} {'CAGR':>7} {'MDD':>7} {'Alpha':>7} {'Gate':>5}")

    for vk, m in sv:
        print(f"  {vk:<3} {m['name']:<35} {m.get('sharpe',0):>7.2f} {m.get('sortino',0):>7.2f} "
              f"{m.get('profit_factor',0):>6.2f} {m.get('win_rate',0):>5.1f}% "
              f"{m.get('total_trades',0):>5} {m.get('total_return_pct',0):>7.1f}% "
              f"{m.get('cagr_pct',0):>6.1f}% {m.get('max_drawdown_pct',0):>6.1f}% "
              f"{m.get('alpha_vs_spy',0):>6.1f}% {m.get('gates_passed',0):>3}/5")

    best = sv[0]
    print(f"\n  BEST: {best[0]} ({best[1]['name']}) — Sharpe {best[1].get('sharpe', 0)}, "
          f"$645 → ${best[1].get('final_equity', 0):.0f}")

    with open(os.path.join(OUTPUT_DIR, 'backtest_results.json'), 'w') as f:
        json.dump({'metrics': all_metrics, 'best': best[0],
                   'timestamp': datetime.now().isoformat(), 'track': 'HIGH_GROWTH'}, f, indent=2, default=str)

    print("\nDone.")


if __name__ == '__main__':
    main()
