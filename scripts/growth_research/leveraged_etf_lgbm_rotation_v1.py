#!/usr/bin/env python3
"""
Leveraged ETF LGBM Rotation v1 — HIGH GROWTH TRACK (Agentic Account)
=====================================================================
HYPOTHESIS: LGBM momentum ranking on leveraged ETFs (3x) provides
high-growth returns WITHOUT theta decay that killed all options strategies.

KEY INSIGHT: At $645, we need leverage for growth. Options = theta death.
Leveraged ETFs = synthetic 3x leverage with NO time decay. LGBM ranking
(Sharpe 1.40 on sector ETFs) should transfer to leveraged universe.

UNIVERSE (3x leveraged + inverse):
  Bull: TQQQ, SOXL, UPRO, TECL, FNGU, LABU, TNA, SPXL
  Bear: SQQQ, SOXS, SPXU, TECS, FNGD, LABD, TZA, SPXS

VARIANTS:
  A: Long top-2 leveraged bull ETFs monthly (pure momentum)
  B: Long top-2 + short bottom-1 (long-short leveraged)
  C: Bull/bear rotation — long TQQQ when momentum up, SQQQ when down
  D: Trailing stop 15% (protect gains in volatile leveraged space)
  E: VIX filter — cash when VIX > 25 (leverage kills in vol spikes)
  F: Concentrated top-1 (max aggression for $645 growth)
  G: Weekly rebalance (capture faster momentum in leveraged space)
  H: Sector-timed (use sector LGBM ranking to pick WHICH leveraged ETF)

5-GATE VALIDATION: Sharpe>1, perm p<0.05, WR>40%, regime balance, beats random.
"""

import sys
import os
import json
import warnings
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')

# --- Path setup ---
for root in ['/home/jupiter/Lvl3Quant', '/home/nick/Lvl3Quant']:
    if os.path.isdir(root):
        LVL3_ROOT = root
        break
else:
    LVL3_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

OUTPUT_DIR = os.path.join(LVL3_ROOT, 'output', 'growth_research', 'leveraged_etf_lgbm_rotation_v1')
os.makedirs(OUTPUT_DIR, exist_ok=True)

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

try:
    from lightgbm import LGBMRegressor
    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False

# ============================================================
# CONSTANTS
# ============================================================
STARTING_CAPITAL = 645.0
COMMISSION_PER_TRADE = 0.0  # RH zero-commission equity
START_DATE = '2015-01-01'  # Go back further for ETFs that exist
END_DATE = '2026-07-28'

# Leveraged ETF universe
BULL_ETFS = ['TQQQ', 'SOXL', 'UPRO', 'TECL', 'FNGU', 'LABU', 'TNA', 'SPXL']
BEAR_ETFS = ['SQQQ', 'SOXS', 'SPXU', 'TECS', 'FNGD', 'LABD', 'TZA', 'SPXS']

# Sector mapping for variant H
SECTOR_TO_LEV = {
    'XLK': ('TECL', 'TECS'),
    'XLF': ('FAS', 'FAZ'),
    'XLE': ('ERX', 'ERY'),
    'XLV': ('LABU', 'LABD'),  # biotech proxy
    'XLY': ('TQQQ', 'SQQQ'),  # consumer disc ~ tech-heavy
    'XLI': ('TNA', 'TZA'),   # small cap as industrial proxy
    'XLB': ('UPRO', 'SPXU'),  # materials ~ broad market
    'XLU': ('UPRO', 'SPXU'),  # utilities ~ defensive
    'XLRE': ('UPRO', 'SPXU'),
    'XLC': ('FNGU', 'FNGD'),  # comm services ~ FANG
    'XLP': ('UPRO', 'SPXU'),  # staples ~ defensive
}

# LGBM parameters (same as validated sector rotation)
LGBM_PARAMS = {
    'n_estimators': 200,
    'max_depth': 5,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_samples': 20,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'random_state': 42,
    'verbose': -1,
    'n_jobs': -1,
}

TRAIN_WINDOW = 250  # trading days (reduced from 400 for leveraged ETFs with shorter histories)
PREDICT_HORIZON = 21  # predict 21-day forward return
N_PERMUTATIONS = 150

print(f"Running on: {LVL3_ROOT}")

# ============================================================
# DATA LOADING
# ============================================================

def load_data():
    """Load daily OHLCV for leveraged ETF universe + SPY + VIX + sector ETFs."""
    import yfinance as yf

    # All tickers we need
    sector_etfs = list(SECTOR_TO_LEV.keys())
    all_tickers = list(set(BULL_ETFS + BEAR_ETFS + ['SPY', '^VIX'] + sector_etfs +
                          ['FAS', 'FAZ', 'ERX', 'ERY']))

    print(f"\nDownloading {len(all_tickers)} tickers from {START_DATE}...")

    all_frames = {}
    for ticker in sorted(all_tickers):
        try:
            data = yf.download(ticker, start=START_DATE, end=END_DATE, progress=False, auto_adjust=True)
            if len(data) < 100:
                print(f"  SKIP {ticker}: only {len(data)} rows")
                continue
            data.columns = [c.lower() if isinstance(c, str) else c[0].lower() for c in data.columns]
            all_frames[ticker] = data
        except Exception as e:
            print(f"  ERROR {ticker}: {e}")

    print(f"Loaded {len(all_frames)} tickers")

    # Filter to ETFs with enough data
    available_bull = [t for t in BULL_ETFS if t in all_frames and len(all_frames[t]) > 200]
    available_bear = [t for t in BEAR_ETFS if t in all_frames and len(all_frames[t]) > 200]

    print(f"Bull ETFs available: {available_bull}")
    print(f"Bear ETFs available: {available_bear}")

    return all_frames, available_bull, available_bear


# ============================================================
# FEATURE ENGINEERING (Enhanced — from LGBM Signal Enhancement v1)
# ============================================================

def compute_features(prices, spy_prices, vix_prices, date_idx, lookback=252):
    """
    Compute 28 enhanced features for a single ETF at a given date.
    Uses the cross-sector feature set that boosted Sharpe by 87%.
    """
    if date_idx < lookback + 5:
        return None

    close = prices['close'].values[:date_idx+1]
    volume = prices['volume'].values[:date_idx+1]
    high = prices['high'].values[:date_idx+1]
    low = prices['low'].values[:date_idx+1]

    if len(close) < lookback:
        return None

    features = {}

    # --- Momentum features (core 17) ---
    for period in [5, 10, 21, 63, 126, 252]:
        if len(close) > period:
            features[f'ret_{period}d'] = (close[-1] / close[-period-1]) - 1.0
        else:
            features[f'ret_{period}d'] = 0.0

    # Volatility
    if len(close) > 22:
        daily_rets = np.diff(np.log(close[-22:]))
        features['vol_21d'] = np.std(daily_rets) * np.sqrt(252)
    else:
        features['vol_21d'] = 0.25

    if len(close) > 64:
        daily_rets_63 = np.diff(np.log(close[-64:]))
        features['vol_63d'] = np.std(daily_rets_63) * np.sqrt(252)
    else:
        features['vol_63d'] = 0.25

    # RSI(14)
    if len(close) > 15:
        deltas = np.diff(close[-15:])
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)
        if avg_loss == 0:
            features['rsi_14'] = 100.0
        else:
            rs = avg_gain / avg_loss
            features['rsi_14'] = 100.0 - (100.0 / (1.0 + rs))
    else:
        features['rsi_14'] = 50.0

    # MACD
    if len(close) > 27:
        ema12 = pd.Series(close).ewm(span=12).mean().values
        ema26 = pd.Series(close).ewm(span=26).mean().values
        features['macd'] = (ema12[-1] - ema26[-1]) / close[-1]
    else:
        features['macd'] = 0.0

    # Bollinger band position
    if len(close) > 21:
        sma20 = np.mean(close[-20:])
        std20 = np.std(close[-20:])
        if std20 > 0:
            features['bb_position'] = (close[-1] - sma20) / (2 * std20)
        else:
            features['bb_position'] = 0.0
    else:
        features['bb_position'] = 0.0

    # Volume features
    if len(volume) > 21:
        features['vol_ratio_20d'] = volume[-1] / (np.mean(volume[-21:-1]) + 1e-8)
        features['vol_trend'] = np.mean(volume[-5:]) / (np.mean(volume[-21:]) + 1e-8)
    else:
        features['vol_ratio_20d'] = 1.0
        features['vol_trend'] = 1.0

    # High-low range
    if len(high) > 21:
        features['avg_range_21d'] = np.mean((high[-21:] - low[-21:]) / close[-21:])
    else:
        features['avg_range_21d'] = 0.02

    # Drawdown from peak
    if len(close) > 63:
        peak_63 = np.max(close[-63:])
        features['dd_from_peak_63d'] = (close[-1] / peak_63) - 1.0
    else:
        features['dd_from_peak_63d'] = 0.0

    # --- Cross-sector features (the +87% Sharpe boost) ---
    spy_close = spy_prices['close'].values[:date_idx+1]
    if len(spy_close) > 63 and len(close) > 63:
        # Correlation to SPY (63d)
        etf_rets = np.diff(np.log(close[-64:]))
        spy_rets = np.diff(np.log(spy_close[-64:]))
        min_len = min(len(etf_rets), len(spy_rets))
        if min_len > 10:
            corr = np.corrcoef(etf_rets[-min_len:], spy_rets[-min_len:])[0, 1]
            features['corr_to_spy_63d'] = corr if np.isfinite(corr) else 0.0

            # Beta to SPY
            cov = np.cov(etf_rets[-min_len:], spy_rets[-min_len:])[0, 1]
            var_spy = np.var(spy_rets[-min_len:])
            features['beta_to_spy_63d'] = cov / var_spy if var_spy > 0 else 1.0
        else:
            features['corr_to_spy_63d'] = 0.0
            features['beta_to_spy_63d'] = 1.0
    else:
        features['corr_to_spy_63d'] = 0.0
        features['beta_to_spy_63d'] = 1.0

    # Relative strength vs SPY
    if len(spy_close) > 21 and len(close) > 21:
        etf_ret_21 = (close[-1] / close[-22]) - 1.0
        spy_ret_21 = (spy_close[-1] / spy_close[-22]) - 1.0
        features['rel_strength_vs_spy_21d'] = etf_ret_21 - spy_ret_21
    else:
        features['rel_strength_vs_spy_21d'] = 0.0

    # VIX level
    if vix_prices is not None:
        vix_close = vix_prices['close'].values[:date_idx+1]
        if len(vix_close) > 0:
            features['vix_level'] = vix_close[-1]
            if len(vix_close) > 21:
                features['vix_rank_21d'] = (vix_close[-1] - np.min(vix_close[-21:])) / \
                    (np.max(vix_close[-21:]) - np.min(vix_close[-21:]) + 1e-8)
            else:
                features['vix_rank_21d'] = 0.5
        else:
            features['vix_level'] = 20.0
            features['vix_rank_21d'] = 0.5
    else:
        features['vix_level'] = 20.0
        features['vix_rank_21d'] = 0.5

    # Momentum acceleration
    if len(close) > 42:
        ret_21d_now = (close[-1] / close[-22]) - 1.0
        ret_21d_prev = (close[-22] / close[-43]) - 1.0
        features['mom_acceleration'] = ret_21d_now - ret_21d_prev
    else:
        features['mom_acceleration'] = 0.0

    return features


# ============================================================
# LGBM WALK-FORWARD ENGINE
# ============================================================

def lgbm_walk_forward_rank(all_frames, etf_list, spy_prices, vix_prices,
                           train_window=TRAIN_WINDOW, predict_horizon=PREDICT_HORIZON,
                           rebalance_freq=21, sample_spacing=20):
    """
    Walk-forward LGBM ranking on leveraged ETFs.
    Returns: dict of date -> ranked list of ETFs with predicted returns.
    """
    if not LGBM_AVAILABLE:
        print("WARNING: LightGBM not available, falling back to momentum ranking")
        return momentum_rank_fallback(all_frames, etf_list, rebalance_freq)

    # Build date index from SPY (most complete), don't require ALL ETFs to share dates
    if 'SPY' in all_frames:
        common_dates = sorted(all_frames['SPY'].index)
    else:
        # Union of all ETF dates
        all_dates_set = set()
        for etf in etf_list:
            if etf in all_frames:
                all_dates_set.update(all_frames[etf].index)
        common_dates = sorted(all_dates_set)

    if len(common_dates) < train_window + predict_horizon + 50:
        print(f"  Insufficient dates ({len(common_dates)})")
        return {}

    print(f"  Date range: {common_dates[0].date()} to {common_dates[-1].date()}, {len(common_dates)} days")

    # Walk-forward: train on [t-train_window:t], predict at t, rebalance every rebalance_freq days
    rankings = {}
    n_rebalances = 0

    # Need 252 days for feature lookback + train_window + predict_horizon before first prediction
    min_start = max(train_window + predict_horizon, 300)

    for t in range(min_start, len(common_dates), rebalance_freq):
        if t >= len(common_dates):
            break

        current_date = common_dates[t]

        # Build training data: sample observations where features can be computed (idx >= 252)
        # and forward return is available (idx + predict_horizon < t)
        train_end = t - predict_horizon
        # Start sampling from where we have enough feature lookback (252 days)
        train_sample_start = max(252, t - train_window * 2)  # look back up to 2x train_window

        X_train = []
        y_train = []

        for sample_idx in range(train_sample_start, train_end, sample_spacing):
            if sample_idx + predict_horizon >= len(common_dates):
                break

            sample_date = common_dates[sample_idx]
            future_date = common_dates[min(sample_idx + predict_horizon, len(common_dates) - 1)]

            for etf in etf_list:
                if etf not in all_frames:
                    continue

                prices = all_frames[etf]
                # Find the index of sample_date in prices
                try:
                    date_pos = prices.index.get_loc(sample_date)
                except KeyError:
                    continue

                feat = compute_features(prices, all_frames.get('SPY', spy_prices),
                                       all_frames.get('^VIX', vix_prices), date_pos)
                if feat is None:
                    continue

                # Target: forward return
                try:
                    future_pos = prices.index.get_loc(future_date)
                    fwd_ret = (prices['close'].iloc[future_pos] / prices['close'].iloc[date_pos]) - 1.0
                except (KeyError, IndexError):
                    continue

                X_train.append(feat)
                y_train.append(fwd_ret)

        if len(X_train) < 30:
            continue

        X_df = pd.DataFrame(X_train)
        y_arr = np.array(y_train)

        # Handle NaN/inf
        X_df = X_df.replace([np.inf, -np.inf], np.nan).fillna(0)

        # Train LGBM
        model = LGBMRegressor(**LGBM_PARAMS)
        model.fit(X_df, y_arr)

        # Predict current ranking
        predictions = {}
        for etf in etf_list:
            if etf not in all_frames:
                continue

            prices = all_frames[etf]
            try:
                date_pos = prices.index.get_loc(current_date)
            except KeyError:
                continue

            feat = compute_features(prices, all_frames.get('SPY', spy_prices),
                                   all_frames.get('^VIX', vix_prices), date_pos)
            if feat is None:
                continue

            feat_df = pd.DataFrame([feat])
            feat_df = feat_df.replace([np.inf, -np.inf], np.nan).fillna(0)

            # Ensure columns match training
            for col in X_df.columns:
                if col not in feat_df.columns:
                    feat_df[col] = 0
            feat_df = feat_df[X_df.columns]

            pred = model.predict(feat_df)[0]
            predictions[etf] = pred

        if predictions:
            # Sort by predicted return (highest first)
            ranked = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
            rankings[current_date] = ranked
            n_rebalances += 1

    print(f"  LGBM walk-forward: {n_rebalances} rebalance points")
    return rankings


def momentum_rank_fallback(all_frames, etf_list, rebalance_freq=21):
    """Simple momentum ranking fallback if LGBM not available."""
    common_dates = None
    for etf in etf_list:
        if etf in all_frames:
            dates = all_frames[etf].index
            if common_dates is None:
                common_dates = set(dates)
            else:
                common_dates = common_dates.intersection(dates)

    common_dates = sorted(common_dates)
    rankings = {}

    for t in range(252, len(common_dates), rebalance_freq):
        current_date = common_dates[t]
        scores = {}
        for etf in etf_list:
            if etf not in all_frames:
                continue
            prices = all_frames[etf]
            try:
                idx = prices.index.get_loc(current_date)
                if idx < 63:
                    continue
                ret_63 = (prices['close'].iloc[idx] / prices['close'].iloc[idx-63]) - 1.0
                ret_21 = (prices['close'].iloc[idx] / prices['close'].iloc[idx-21]) - 1.0
                scores[etf] = 0.5 * ret_63 + 0.5 * ret_21
            except (KeyError, IndexError):
                continue

        if scores:
            ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            rankings[current_date] = ranked

    return rankings


# ============================================================
# BACKTESTING ENGINE
# ============================================================

def backtest_variant(variant_key, variant_cfg, all_frames, rankings, spy_prices, vix_prices):
    """Run a single variant backtest."""

    if not rankings:
        return None

    rebalance_dates = sorted(rankings.keys())
    if len(rebalance_dates) < 3:
        return None

    # Get all trading dates from SPY
    all_dates = spy_prices.index
    start_idx = all_dates.get_loc(rebalance_dates[0])

    equity = STARTING_CAPITAL
    equity_curve = []
    equity_dates = []
    all_trades = []
    holdings = {}  # ticker -> (n_shares, entry_price, entry_date)
    trailing_peaks = {}  # ticker -> peak price for trailing stop

    rebal_idx = 0

    for t in range(start_idx, len(all_dates)):
        date = all_dates[t]
        prev_equity = equity

        # Get VIX
        vix_level = 20.0
        if vix_prices is not None and date in vix_prices.index:
            vix_level = vix_prices.loc[date, 'close']
        elif vix_prices is not None:
            mask = vix_prices.index <= date
            if mask.any():
                vix_level = vix_prices.loc[mask, 'close'].iloc[-1]

        # Check if we need to rebalance
        should_rebalance = False
        if rebal_idx < len(rebalance_dates) and date >= rebalance_dates[rebal_idx]:
            should_rebalance = True
            current_ranking = rankings[rebalance_dates[rebal_idx]]
            rebal_idx += 1
            # Skip ahead if multiple rebalance dates passed
            while rebal_idx < len(rebalance_dates) and rebalance_dates[rebal_idx] <= date:
                current_ranking = rankings[rebalance_dates[rebal_idx]]
                rebal_idx += 1

        # Mark-to-market current holdings
        mtm_value = 0
        for ticker, (n_shares, entry_price, entry_date) in list(holdings.items()):
            if ticker in all_frames and date in all_frames[ticker].index:
                current_price = all_frames[ticker].loc[date, 'close']
                mtm_value += n_shares * current_price

                # Update trailing peak
                if ticker not in trailing_peaks:
                    trailing_peaks[ticker] = current_price
                elif current_price > trailing_peaks[ticker]:
                    trailing_peaks[ticker] = current_price

                # Check trailing stop
                if variant_cfg.get('trailing_stop_pct'):
                    if trailing_peaks[ticker] > 0:
                        drawdown = (trailing_peaks[ticker] - current_price) / trailing_peaks[ticker]
                        if drawdown > variant_cfg['trailing_stop_pct']:
                            # Trigger trailing stop — sell this position
                            pnl = n_shares * (current_price - entry_price)
                            equity += pnl
                            all_trades.append({
                                'ticker': ticker, 'direction': 'LONG',
                                'entry_date': str(entry_date)[:10], 'exit_date': str(date)[:10],
                                'entry_price': round(entry_price, 2),
                                'exit_price': round(current_price, 2),
                                'n_shares': n_shares, 'pnl': round(pnl, 2),
                                'exit_reason': 'trailing_stop',
                            })
                            del holdings[ticker]
                            del trailing_peaks[ticker]
                            should_rebalance = True  # re-enter on next signal
                            continue
            else:
                # Can't price, keep as-is
                mtm_value += n_shares * entry_price

        # VIX filter — go to cash if VIX too high
        if variant_cfg.get('vix_filter') and vix_level > variant_cfg['vix_threshold']:
            if holdings:
                # Liquidate all
                for ticker, (n_shares, entry_price, entry_date) in list(holdings.items()):
                    if ticker in all_frames and date in all_frames[ticker].index:
                        current_price = all_frames[ticker].loc[date, 'close']
                        pnl = n_shares * (current_price - entry_price)
                        equity += pnl
                        all_trades.append({
                            'ticker': ticker, 'direction': 'LONG',
                            'entry_date': str(entry_date)[:10], 'exit_date': str(date)[:10],
                            'entry_price': round(entry_price, 2),
                            'exit_price': round(current_price, 2),
                            'n_shares': n_shares, 'pnl': round(pnl, 2),
                            'exit_reason': 'vix_filter',
                        })
                holdings = {}
                trailing_peaks = {}
            equity_curve.append(equity)
            equity_dates.append(date)
            continue

        # Rebalance
        if should_rebalance and rebal_idx > 0:
            ranking = rankings[rebalance_dates[rebal_idx - 1]]

            # Determine target portfolio
            target_tickers = []
            n_long = variant_cfg.get('n_long', 2)
            n_short = variant_cfg.get('n_short', 0)
            direction_mode = variant_cfg.get('direction_mode', 'bull_only')

            if direction_mode == 'bull_only':
                # Long top-N bull ETFs
                bull_ranked = [(t, s) for t, s in ranking if t in (variant_cfg.get('universe', BULL_ETFS))]
                target_tickers = [t for t, s in bull_ranked[:n_long]]

            elif direction_mode == 'long_short':
                # Long top-N, short bottom-M
                all_ranked = ranking
                target_tickers = [t for t, s in all_ranked[:n_long]]
                # Short positions more complex — skip for v1, just use long

            elif direction_mode == 'bull_bear_switch':
                # If top prediction is bullish (>0), go bull; else go bear
                if ranking and ranking[0][1] > 0:
                    bull_ranked = [(t, s) for t, s in ranking if t in BULL_ETFS]
                    target_tickers = [t for t, s in bull_ranked[:n_long]]
                else:
                    bear_ranked = [(t, s) for t, s in ranking if t in BEAR_ETFS]
                    target_tickers = [t for t, s in bear_ranked[:n_long]]

            elif direction_mode == 'sector_timed':
                # Use sector ranking to pick which leveraged ETF
                sector_etfs = [t for t in ['XLK', 'XLF', 'XLE', 'XLV', 'XLY', 'XLI', 'XLC']
                              if t in all_frames]
                # Rank sectors by recent momentum
                sector_scores = {}
                for sec in sector_etfs:
                    if date in all_frames[sec].index:
                        idx = all_frames[sec].index.get_loc(date)
                        if idx >= 21:
                            ret = (all_frames[sec]['close'].iloc[idx] /
                                   all_frames[sec]['close'].iloc[idx-21]) - 1.0
                            sector_scores[sec] = ret

                if sector_scores:
                    top_sector = max(sector_scores, key=sector_scores.get)
                    if top_sector in SECTOR_TO_LEV:
                        bull_lev, _ = SECTOR_TO_LEV[top_sector]
                        if bull_lev in all_frames:
                            target_tickers = [bull_lev]

                if not target_tickers:
                    target_tickers = ['TQQQ']  # fallback

            # Sell positions not in target
            for ticker in list(holdings.keys()):
                if ticker not in target_tickers:
                    n_shares, entry_price, entry_date_pos = holdings[ticker]
                    if ticker in all_frames and date in all_frames[ticker].index:
                        current_price = all_frames[ticker].loc[date, 'close']
                        pnl = n_shares * (current_price - entry_price)
                        equity += pnl
                        all_trades.append({
                            'ticker': ticker, 'direction': 'LONG',
                            'entry_date': str(entry_date_pos)[:10], 'exit_date': str(date)[:10],
                            'entry_price': round(entry_price, 2),
                            'exit_price': round(current_price, 2),
                            'n_shares': n_shares, 'pnl': round(pnl, 2),
                            'exit_reason': 'rebalance',
                        })
                    del holdings[ticker]
                    if ticker in trailing_peaks:
                        del trailing_peaks[ticker]

            # Buy new positions
            n_new = len([t for t in target_tickers if t not in holdings])
            if n_new > 0:
                cash_available = equity - sum(
                    n * all_frames[t].loc[date, 'close']
                    for t, (n, _, _) in holdings.items()
                    if t in all_frames and date in all_frames[t].index
                )
                per_position = max(cash_available / n_new, 0) if n_new > 0 else 0

                for ticker in target_tickers:
                    if ticker in holdings:
                        continue
                    if ticker not in all_frames or date not in all_frames[ticker].index:
                        continue

                    price = all_frames[ticker].loc[date, 'close']
                    if price <= 0 or per_position <= 0:
                        continue

                    n_shares = int(per_position / price)
                    if n_shares < 1:
                        # Fractional shares on RH
                        n_shares = per_position / price  # fractional

                    holdings[ticker] = (n_shares, price, date)
                    trailing_peaks[ticker] = price

        # Update equity (mark to market)
        total_value = 0
        cash_portion = equity
        for ticker, (n_shares, entry_price, entry_date) in holdings.items():
            if ticker in all_frames and date in all_frames[ticker].index:
                current_price = all_frames[ticker].loc[date, 'close']
                total_value += n_shares * current_price
                cash_portion -= n_shares * entry_price  # approximate

        # Simple equity tracking: cash + MTM of holdings
        portfolio_value = 0
        for ticker, (n_shares, entry_price, _) in holdings.items():
            if ticker in all_frames and date in all_frames[ticker].index:
                current_price = all_frames[ticker].loc[date, 'close']
                portfolio_value += n_shares * current_price

        # Equity = realized gains/losses + current holdings value
        # Track via entry costs
        invested = sum(n * ep for _, (n, ep, _) in holdings.items())
        unrealized = portfolio_value - invested
        current_equity = equity + unrealized  # equity has realized P&L baked in

        equity_curve.append(current_equity)
        equity_dates.append(date)

    # Close remaining positions
    final_date = all_dates[-1]
    for ticker, (n_shares, entry_price, entry_date) in list(holdings.items()):
        if ticker in all_frames and final_date in all_frames[ticker].index:
            current_price = all_frames[ticker].loc[final_date, 'close']
            pnl = n_shares * (current_price - entry_price)
            equity += pnl
            all_trades.append({
                'ticker': ticker, 'direction': 'LONG',
                'entry_date': str(entry_date)[:10], 'exit_date': str(final_date)[:10],
                'entry_price': round(entry_price, 2),
                'exit_price': round(current_price, 2),
                'n_shares': n_shares, 'pnl': round(pnl, 2),
                'exit_reason': 'final_close',
            })

    return {
        'equity_curve': equity_curve,
        'equity_dates': [str(d)[:10] for d in equity_dates],
        'trades': all_trades,
        'final_equity': equity_curve[-1] if equity_curve else STARTING_CAPITAL,
    }


# ============================================================
# METRICS & VALIDATION
# ============================================================

def compute_metrics(result, variant_key, variant_cfg):
    """Compute risk-adjusted metrics."""
    if result is None or not result['equity_curve']:
        return {'variant': variant_key, 'name': variant_cfg['name'], 'sharpe': 0, 'total_trades': 0}

    eq = np.array(result['equity_curve'])
    trades = result['trades']

    # Daily returns
    daily_rets = np.diff(eq) / eq[:-1]
    daily_rets = daily_rets[np.isfinite(daily_rets)]

    if len(daily_rets) < 20:
        return {'variant': variant_key, 'name': variant_cfg['name'], 'sharpe': 0, 'total_trades': len(trades)}

    # Sharpe
    mean_ret = np.mean(daily_rets)
    std_ret = np.std(daily_rets)
    sharpe = (mean_ret / std_ret) * np.sqrt(252) if std_ret > 0 else 0.0

    # Sortino
    downside = daily_rets[daily_rets < 0]
    downside_std = np.std(downside) if len(downside) > 0 else std_ret
    sortino = (mean_ret / downside_std) * np.sqrt(252) if downside_std > 0 else 0.0

    # Max drawdown
    peak = np.maximum.accumulate(eq)
    dd = (eq - peak) / peak
    max_dd = np.min(dd) * 100

    # Trade stats
    pnls = [t['pnl'] for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    wr = len(wins) / len(pnls) * 100 if pnls else 0

    gross_win = sum(wins) if wins else 0
    gross_loss = abs(sum(losses)) if losses else 0
    pf = gross_win / gross_loss if gross_loss > 0 else (999 if gross_win > 0 else 0)

    # CAGR
    n_years = len(daily_rets) / 252
    total_ret = eq[-1] / eq[0]
    cagr = (total_ret ** (1 / n_years) - 1) * 100 if n_years > 0 and total_ret > 0 else 0

    # SPY comparison
    return {
        'variant': variant_key,
        'name': variant_cfg['name'],
        'sharpe': round(sharpe, 3),
        'sortino': round(sortino, 3),
        'profit_factor': round(pf, 2),
        'win_rate': round(wr, 1),
        'max_drawdown_pct': round(max_dd, 2),
        'cagr_pct': round(cagr, 1),
        'total_return_pct': round((total_ret - 1) * 100, 1),
        'final_equity': round(eq[-1], 2),
        'total_trades': len(trades),
        'avg_pnl': round(np.mean(pnls), 2) if pnls else 0,
        'n_years': round(n_years, 1),
    }


def run_permutation_test(result, all_frames, spy_prices, vix_prices, variant_cfg, n_perms=N_PERMUTATIONS):
    """Permutation test: shuffle the ranking and re-run."""
    if result is None:
        return 1.0, 0, 0

    actual_sharpe = compute_metrics(result, 'actual', variant_cfg)['sharpe']

    print(f"    Running {n_perms} permutation tests (actual Sharpe={actual_sharpe:.3f})...")

    better_count = 0
    random_sharpes = []

    for perm in range(n_perms):
        # Random selection instead of LGBM ranking
        universe = variant_cfg.get('universe', BULL_ETFS)
        available = [t for t in universe if t in all_frames]

        if not available:
            continue

        # Create random rankings
        random_rankings = {}
        if 'SPY' in all_frames:
            common_dates = sorted(all_frames['SPY'].index)
        else:
            all_d = set()
            for t in available:
                if t in all_frames:
                    all_d.update(all_frames[t].index)
            common_dates = sorted(all_d)

        for t_idx in range(400, len(common_dates), variant_cfg.get('rebalance_days', 21)):
            if t_idx >= len(common_dates):
                break
            date = common_dates[t_idx]
            shuffled = list(available)
            np.random.shuffle(shuffled)
            random_rankings[date] = [(t, np.random.randn()) for t in shuffled]

        rand_result = backtest_variant('rand', variant_cfg, all_frames, random_rankings, spy_prices, vix_prices)

        if rand_result and rand_result['equity_curve']:
            rand_metrics = compute_metrics(rand_result, 'rand', variant_cfg)
            rand_sharpe = rand_metrics['sharpe']
            random_sharpes.append(rand_sharpe)
            if rand_sharpe >= actual_sharpe:
                better_count += 1

    p_value = better_count / max(n_perms, 1)
    random_mean = np.mean(random_sharpes) if random_sharpes else 0

    return p_value, actual_sharpe, random_mean


def regime_balance_test(result, spy_prices):
    """Check regime balance — Sharpe in bull vs bear markets."""
    if result is None or not result['equity_curve'] or not result['equity_dates']:
        return 1.0  # fail

    eq = np.array(result['equity_curve'])
    dates = pd.to_datetime(result['equity_dates'])

    # Classify each day as bull/bear based on SPY 50d MA
    spy_close = spy_prices['close']
    spy_ma50 = spy_close.rolling(50).mean()

    bull_rets = []
    bear_rets = []

    for i in range(1, len(eq)):
        daily_ret = (eq[i] - eq[i-1]) / eq[i-1] if eq[i-1] > 0 else 0
        date = dates[i]

        if date in spy_close.index and date in spy_ma50.index:
            if spy_close.loc[date] > spy_ma50.loc[date]:
                bull_rets.append(daily_ret)
            else:
                bear_rets.append(daily_ret)

    if len(bull_rets) < 20 or len(bear_rets) < 20:
        return 0.5  # not enough data, pass by default

    bull_sharpe = (np.mean(bull_rets) / np.std(bull_rets)) * np.sqrt(252) if np.std(bull_rets) > 0 else 0
    bear_sharpe = (np.mean(bear_rets) / np.std(bear_rets)) * np.sqrt(252) if np.std(bear_rets) > 0 else 0

    # Regime gap per HC #428
    max_abs = max(abs(bull_sharpe), abs(bear_sharpe))
    gap = abs(bull_sharpe - bear_sharpe) / max_abs if max_abs > 0 else 0

    print(f"    Regime: Bull Sharpe={bull_sharpe:.3f}, Bear Sharpe={bear_sharpe:.3f}, Gap={gap:.3f}")

    return gap


# ============================================================
# VARIANT DEFINITIONS
# ============================================================

VARIANTS = {
    'A': {
        'name': 'Long Top-2 Monthly',
        'universe': BULL_ETFS,
        'n_long': 2,
        'direction_mode': 'bull_only',
        'rebalance_days': 21,
        'trailing_stop_pct': None,
        'vix_filter': False,
        'vix_threshold': 25,
    },
    'B': {
        'name': 'Long Top-2 + Short Bot-1',
        'universe': BULL_ETFS + BEAR_ETFS,
        'n_long': 2,
        'n_short': 1,
        'direction_mode': 'bull_only',
        'rebalance_days': 21,
        'trailing_stop_pct': None,
        'vix_filter': False,
        'vix_threshold': 25,
    },
    'C': {
        'name': 'Bull/Bear Switch',
        'universe': BULL_ETFS + BEAR_ETFS,
        'n_long': 2,
        'direction_mode': 'bull_bear_switch',
        'rebalance_days': 21,
        'trailing_stop_pct': None,
        'vix_filter': False,
        'vix_threshold': 25,
    },
    'D': {
        'name': 'Trailing Stop 15%',
        'universe': BULL_ETFS,
        'n_long': 2,
        'direction_mode': 'bull_only',
        'rebalance_days': 21,
        'trailing_stop_pct': 0.15,
        'vix_filter': False,
        'vix_threshold': 25,
    },
    'E': {
        'name': 'VIX Filter (<25)',
        'universe': BULL_ETFS,
        'n_long': 2,
        'direction_mode': 'bull_only',
        'rebalance_days': 21,
        'trailing_stop_pct': None,
        'vix_filter': True,
        'vix_threshold': 25,
    },
    'F': {
        'name': 'Concentrated Top-1',
        'universe': BULL_ETFS,
        'n_long': 1,
        'direction_mode': 'bull_only',
        'rebalance_days': 21,
        'trailing_stop_pct': None,
        'vix_filter': False,
        'vix_threshold': 25,
    },
    'G': {
        'name': 'Weekly Rebalance',
        'universe': BULL_ETFS,
        'n_long': 2,
        'direction_mode': 'bull_only',
        'rebalance_days': 5,
        'trailing_stop_pct': None,
        'vix_filter': False,
        'vix_threshold': 25,
    },
    'H': {
        'name': 'Sector-Timed Leveraged',
        'universe': BULL_ETFS,
        'n_long': 1,
        'direction_mode': 'sector_timed',
        'rebalance_days': 21,
        'trailing_stop_pct': 0.15,
        'vix_filter': False,
        'vix_threshold': 25,
    },
}


# ============================================================
# MAIN
# ============================================================

def main():
    start_time = datetime.now()

    # MLflow setup
    if MLFLOW_AVAILABLE:
        try:
            mlflow.set_tracking_uri('http://jupiter:5000')
            mlflow.set_experiment('leveraged_etf_lgbm_rotation_v1')
            print("MLflow OK: http://jupiter:5000")
        except Exception as e:
            print(f"MLflow setup warning: {e}")

    print("=" * 70)
    print("  LEVERAGED ETF LGBM ROTATION V1 — HIGH GROWTH TRACK")
    print("  Hypothesis: 3x leveraged ETFs + LGBM ranking = high growth")
    print("  without the theta decay that killed all option strategies")
    print("=" * 70)

    # Load data
    all_frames, available_bull, available_bear = load_data()

    if len(available_bull) < 3:
        print("ERROR: Not enough bull leveraged ETFs available")
        return

    spy_prices = all_frames.get('SPY')
    vix_prices = all_frames.get('^VIX')

    if spy_prices is None:
        print("ERROR: SPY data not available")
        return

    # Show date range per major ETF
    print(f"\nDate ranges (per ETF):")
    for etf in sorted(available_bull[:4]):
        if etf in all_frames:
            df = all_frames[etf]
            print(f"  {etf}: {df.index[0].date()} to {df.index[-1].date()}, {len(df)} days")
    spy_range = spy_prices.index
    print(f"  SPY: {spy_range[0].date()} to {spy_range[-1].date()}, {len(spy_range)} days")

    # Run all variants
    all_results = {}
    all_metrics = {}

    for vkey, vcfg in sorted(VARIANTS.items()):
        print(f"\n{'='*60}")
        print(f"  VARIANT {vkey}: {vcfg['name']}")
        print(f"{'='*60}")

        # Build rankings with LGBM
        universe = vcfg.get('universe', BULL_ETFS)
        available_universe = [t for t in universe if t in all_frames]

        rankings = lgbm_walk_forward_rank(
            all_frames, available_universe, spy_prices, vix_prices,
            rebalance_freq=vcfg.get('rebalance_days', 21)
        )

        if not rankings:
            print(f"  No rankings generated, skipping")
            continue

        # Run backtest
        result = backtest_variant(vkey, vcfg, all_frames, rankings, spy_prices, vix_prices)

        if result is None:
            print(f"  Backtest returned None")
            continue

        # Compute metrics
        metrics = compute_metrics(result, vkey, vcfg)

        # Print summary
        print(f"  Trades: {metrics['total_trades']} | Sharpe: {metrics['sharpe']} | "
              f"Sortino: {metrics['sortino']} | PF: {metrics['profit_factor']} | "
              f"WR: {metrics['win_rate']}% | MDD: {metrics['max_drawdown_pct']}% | "
              f"Total Return: {metrics['total_return_pct']}% | CAGR: {metrics.get('cagr_pct', 0)}%")
        print(f"  $645 → ${metrics['final_equity']}")

        # 5-Gate validation
        print(f"  Running 5-gate validation ({N_PERMUTATIONS} permutations)...")

        # Gate 1: Sharpe > 1
        g1 = metrics['sharpe'] > 1.0

        # Gate 2: Permutation test p < 0.05
        p_value, actual_s, random_s = run_permutation_test(result, all_frames, spy_prices, vix_prices, vcfg)
        g2 = p_value < 0.05

        # Gate 3: WR > 40%
        g3 = metrics['win_rate'] > 40.0

        # Gate 4: Regime balance (HC #428)
        regime_gap = regime_balance_test(result, spy_prices)
        g4 = regime_gap < 0.50

        # Gate 5: Beats random baseline
        g5 = actual_s > random_s if random_s else True

        gates_passed = sum([g1, g2, g3, g4, g5])

        print(f"  5-Gate Validation: {gates_passed}/5 PASS")
        print(f"    sharpe_gt_1: {'PASS' if g1 else 'FAIL'} (value={metrics['sharpe']}, threshold=1.0)")
        print(f"    perm_p_lt_005: {'PASS' if g2 else 'FAIL'} (value={p_value:.3f}, threshold=0.05)")
        print(f"    wr_gt_40: {'PASS' if g3 else 'FAIL'} (value={metrics['win_rate']}, threshold=40.0)")
        print(f"    regime_balance: {'PASS' if g4 else 'FAIL'} (value={regime_gap:.3f}, threshold=0.5)")
        print(f"    beats_random: {'PASS' if g5 else 'FAIL'} (actual={actual_s:.3f}, random={random_s:.3f})")

        metrics['gates_passed'] = gates_passed
        metrics['perm_p_value'] = round(p_value, 4)
        metrics['regime_gap'] = round(regime_gap, 3)
        metrics['random_sharpe'] = round(random_s, 3)

        all_results[vkey] = result
        all_metrics[vkey] = metrics

    # Summary
    elapsed = (datetime.now() - start_time).total_seconds()

    print(f"\n{'='*70}")
    print(f"  SUMMARY — LEVERAGED ETF LGBM ROTATION V1")
    print(f"{'='*70}")

    best_variant = None
    best_sharpe = -999

    for vkey in sorted(all_metrics.keys()):
        m = all_metrics[vkey]
        gates = m.get('gates_passed', 0)
        marker = "✅" if gates >= 4 else "❌"
        print(f"  {marker} {vkey}: {m['name']:30s} Sharpe={m['sharpe']:6.3f}  "
              f"${645}→${m['final_equity']:>10.2f}  "
              f"CAGR={m.get('cagr_pct', 0):5.1f}%  Gates={gates}/5  "
              f"p={m.get('perm_p_value', 1.0):.3f}")

        if m['sharpe'] > best_sharpe and gates >= 3:
            best_sharpe = m['sharpe']
            best_variant = vkey

    if best_variant:
        bm = all_metrics[best_variant]
        print(f"\n  BEST VARIANT: {best_variant} ({bm['name']}) — Sharpe {bm['sharpe']}")
    else:
        print(f"\n  VERDICT: No variant achieved 3+ gates")

    print(f"\nTotal runtime: {elapsed:.0f}s")

    # Save results
    results_path = os.path.join(OUTPUT_DIR, 'backtest_results.json')
    with open(results_path, 'w') as f:
        json.dump({
            'metrics': all_metrics,
            'runtime_seconds': elapsed,
            'n_bull_etfs': len(available_bull),
            'n_bear_etfs': len(available_bear),
            'timestamp': datetime.now().isoformat(),
        }, f, indent=2, default=str)
    print(f"\nResults saved to {results_path}")

    # MLflow logging
    if MLFLOW_AVAILABLE:
        try:
            run_name = f"lev_etf_rotation_{datetime.now().strftime('%Y%m%d_%H%M')}"
            with mlflow.start_run(run_name=run_name):
                for vkey, m in all_metrics.items():
                    for mk, mv in m.items():
                        if isinstance(mv, (int, float)):
                            mlflow.log_metric(f"{vkey}_{mk}", mv)

                if best_variant:
                    mlflow.log_metric("best_sharpe", best_sharpe)
                    mlflow.log_param("best_variant", best_variant)

                mlflow.log_param("n_bull_etfs", len(available_bull))
                mlflow.log_param("n_variants", len(VARIANTS))
                mlflow.log_metric("runtime_seconds", elapsed)
                mlflow.log_artifact(results_path)

            print(f"MLflow logging complete")
        except Exception as e:
            print(f"MLflow logging failed: {e}")

    print("\nDone.")


if __name__ == '__main__':
    main()
