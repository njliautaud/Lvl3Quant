#!/usr/bin/env python3
"""
Market-Neutral Sector Equity Rotation Research v1
==================================================
Research Question: Does our LGBM sector ranking work for long/short equity pairs?
No options, no theta decay, no BS pricing assumptions — pure equity rotation.

6 Variants:
  A. Long-Only Top-2 (reproduce KB #285 Sharpe ~1.40)
  B. Long Top-2 / Short Bottom-2 (4 positions, equal weight, market neutral)
  C. Long Top-3 / Short Bottom-3 (6 positions, more diversified)
  D. Long Top-2 / Short SPY (sector alpha with market hedge)
  E. Long Top-2 / Short Bottom-2 + SPY Hedge (dollar-neutral with SPY overlay)
  F. Rank-Weighted L/S (position sizes proportional to LGBM rank score distance from median)

MLflow experiment: market_neutral_equity_rotation
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import lightgbm as lgb
import yfinance as yf
import logging
import time
import sys
import os
from datetime import datetime

# --- MLflow setup ---
MLFLOW_URI = "http://jupiter:5000"
EXPERIMENT_NAME = "market_neutral_equity_rotation"
USE_MLFLOW = True

try:
    import mlflow
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
except Exception as e:
    print(f"[WARN] MLflow unavailable: {e}. Continuing without tracking.")
    USE_MLFLOW = False

# --- Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
log = logging.getLogger(__name__)

# --- Constants ---
SECTOR_ETFS = ['XLB', 'XLC', 'XLE', 'XLF', 'XLI', 'XLK', 'XLP', 'XLRE', 'XLU', 'XLV', 'XLY']
BENCHMARK = 'SPY'
ALL_TICKERS = SECTOR_ETFS + [BENCHMARK]
TRAIN_WINDOW = 500       # trading days
REBAL_PERIOD = 21        # monthly rebalance
STARTING_CAPITAL = 10000
N_PERMUTATIONS = 100
FEATURE_COLS = [
    'ret_5d', 'ret_10d', 'ret_21d', 'ret_63d',
    'vol_21d', 'vol_ratio', 'rsi_14', 'macd', 'macd_signal', 'bb_pct',
    'obv_slope', 'atr_pct', 'sector_rel_strength',
    'skew_21d', 'kurt_21d', 'max_dd_21d', 'up_down_vol_ratio'
]


def download_data(start='2005-01-01', end=None):
    """Download all ETF data via yfinance."""
    log.info(f"Downloading data for {len(ALL_TICKERS)} tickers from {start}...")
    if end is None:
        end = datetime.now().strftime('%Y-%m-%d')
    data = yf.download(ALL_TICKERS, start=start, end=end, auto_adjust=True, progress=False)
    # Handle multi-level columns
    if isinstance(data.columns, pd.MultiIndex):
        close = data['Close']
        volume = data['Volume']
        high = data['High']
        low = data['Low']
    else:
        close = data[['Close']].copy()
        volume = data[['Volume']].copy()
        high = data[['High']].copy()
        low = data[['Low']].copy()
    log.info(f"Downloaded {len(close)} days, {close.shape[1]} tickers. Range: {close.index[0].date()} to {close.index[-1].date()}")
    return close, volume, high, low


def compute_features(close, volume, high, low):
    """Compute 17 features for each sector ETF on each date."""
    log.info("Computing features...")
    spy_ret = close[BENCHMARK].pct_change()
    records = []

    for ticker in SECTOR_ETFS:
        c = close[ticker]
        v = volume[ticker]
        h = high[ticker]
        lo = low[ticker]
        ret = c.pct_change()

        # Momentum
        ret_5d = c.pct_change(5)
        ret_10d = c.pct_change(10)
        ret_21d = c.pct_change(21)
        ret_63d = c.pct_change(63)

        # Volatility
        vol_21d = ret.rolling(21).std()
        vol_5d = ret.rolling(5).std()
        vol_ratio = vol_5d / vol_21d

        # RSI
        delta = ret.copy()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi_14 = 100 - (100 / (1 + rs))

        # MACD
        ema12 = c.ewm(span=12).mean()
        ema26 = c.ewm(span=26).mean()
        macd = ema12 - ema26
        macd_signal = macd.ewm(span=9).mean()

        # Bollinger %
        sma20 = c.rolling(20).mean()
        std20 = c.rolling(20).std()
        bb_pct = (c - sma20) / (2 * std20)

        # OBV slope
        obv = (np.sign(ret) * v).cumsum()
        obv_slope = obv.rolling(21).apply(
            lambda x: np.polyfit(np.arange(len(x)), x, 1)[0] if len(x) == 21 else np.nan,
            raw=True
        )
        # Normalize OBV slope
        obv_slope = obv_slope / v.rolling(21).mean()

        # ATR %
        tr = pd.concat([h - lo, (h - c.shift(1)).abs(), (lo - c.shift(1)).abs()], axis=1).max(axis=1)
        atr14 = tr.rolling(14).mean()
        atr_pct = atr14 / c

        # Sector relative strength
        sector_rel_strength = ret_21d - spy_ret.rolling(21).apply(lambda x: (1 + x).prod() - 1 if len(x) == 21 else np.nan, raw=False)

        # Higher moments
        skew_21d = ret.rolling(21).skew()
        kurt_21d = ret.rolling(21).kurt()

        # Max drawdown 21d
        def max_dd_window(x):
            cum = (1 + x).cumprod()
            peak = cum.cummax()
            dd = (cum / peak - 1)
            return dd.min()
        max_dd_21d = ret.rolling(21).apply(max_dd_window, raw=False)

        # Up/down volume ratio
        up_vol = (ret.clip(lower=0) * v).rolling(21).sum()
        down_vol = ((-ret.clip(upper=0)) * v).rolling(21).sum()
        up_down_vol_ratio = up_vol / down_vol.replace(0, np.nan)

        # Forward return (target)
        fwd_ret_21d = c.pct_change(21).shift(-21)

        df = pd.DataFrame({
            'date': c.index,
            'ticker': ticker,
            'ret_5d': ret_5d.values,
            'ret_10d': ret_10d.values,
            'ret_21d': ret_21d.values,
            'ret_63d': ret_63d.values,
            'vol_21d': vol_21d.values,
            'vol_ratio': vol_ratio.values,
            'rsi_14': rsi_14.values,
            'macd': macd.values,
            'macd_signal': macd_signal.values,
            'bb_pct': bb_pct.values,
            'obv_slope': obv_slope.values,
            'atr_pct': atr_pct.values,
            'sector_rel_strength': sector_rel_strength.values,
            'skew_21d': skew_21d.values,
            'kurt_21d': kurt_21d.values,
            'max_dd_21d': max_dd_21d.values,
            'up_down_vol_ratio': up_down_vol_ratio.values,
            'fwd_ret_21d': fwd_ret_21d.values,
            'close': c.values,
        })
        records.append(df)

    features_df = pd.concat(records, ignore_index=True)
    features_df = features_df.dropna(subset=FEATURE_COLS + ['fwd_ret_21d'])
    log.info(f"Feature matrix: {len(features_df)} rows, {len(FEATURE_COLS)} features")
    return features_df


def walk_forward_ranking(features_df, close):
    """Walk-forward LGBM ranking with sliding 500-day window, monthly rebalance."""
    log.info("Running walk-forward LGBM ranking...")
    dates = sorted(features_df['date'].unique())

    # Find rebalance dates (every 21 trading days starting after train window)
    all_close_dates = close.index.tolist()
    # Map feature dates to close dates index
    valid_start_idx = TRAIN_WINDOW + 63 + 21  # warmup for features + forward return
    rebal_dates = []
    for i in range(valid_start_idx, len(all_close_dates), REBAL_PERIOD):
        d = all_close_dates[i]
        if d in dates:
            rebal_dates.append(d)

    log.info(f"Rebalance dates: {len(rebal_dates)} (first: {rebal_dates[0].date()}, last: {rebal_dates[-1].date()})")

    rankings = []  # list of (date, {ticker: predicted_rank_score})

    for rebal_date in rebal_dates:
        # Training data: last TRAIN_WINDOW days before rebal_date
        mask_train = (features_df['date'] < rebal_date)
        train_pool = features_df[mask_train].copy()

        # Take last TRAIN_WINDOW days worth of data (sliding window)
        train_dates = sorted(train_pool['date'].unique())
        if len(train_dates) < TRAIN_WINDOW:
            continue
        cutoff_date = train_dates[-TRAIN_WINDOW]
        train_pool = train_pool[train_pool['date'] >= cutoff_date]

        # Prediction data: features on rebal_date
        pred_pool = features_df[features_df['date'] == rebal_date].copy()
        if len(pred_pool) < len(SECTOR_ETFS) * 0.5:
            continue

        X_train = train_pool[FEATURE_COLS].values
        y_train = train_pool['fwd_ret_21d'].values
        X_pred = pred_pool[FEATURE_COLS].values

        # Replace inf/nan
        X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
        y_train = np.nan_to_num(y_train, nan=0.0)
        X_pred = np.nan_to_num(X_pred, nan=0.0, posinf=0.0, neginf=0.0)

        # Train LGBM
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'n_jobs': -1,
            'seed': 42,
        }
        train_ds = lgb.Dataset(X_train, label=y_train)
        callbacks = [lgb.log_evaluation(period=-1)]
        model = lgb.train(params, train_ds, num_boost_round=200, callbacks=callbacks)

        # Predict
        preds = model.predict(X_pred)
        ticker_scores = dict(zip(pred_pool['ticker'].values, preds))
        rankings.append((rebal_date, ticker_scores))

    log.info(f"Generated {len(rankings)} rebalance rankings")
    return rankings


def backtest_variant(variant_name, rankings, close, spy_close):
    """Backtest a single variant given rankings and return equity curve + metrics."""
    equity = STARTING_CAPITAL
    equity_curve = []
    trade_returns = []

    for i in range(len(rankings) - 1):
        rebal_date, scores = rankings[i]
        next_date = rankings[i + 1][0]

        # Sort sectors by predicted score
        sorted_sectors = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        tickers_ranked = [t for t, s in sorted_sectors]
        scores_dict = dict(sorted_sectors)

        # Determine positions based on variant
        positions = {}  # ticker: weight (positive=long, negative=short)

        if variant_name == 'A':
            # Long-Only Top-2
            for t in tickers_ranked[:2]:
                positions[t] = 0.5

        elif variant_name == 'B':
            # Long Top-2 / Short Bottom-2
            for t in tickers_ranked[:2]:
                positions[t] = 0.25
            for t in tickers_ranked[-2:]:
                positions[t] = -0.25

        elif variant_name == 'C':
            # Long Top-3 / Short Bottom-3
            for t in tickers_ranked[:3]:
                positions[t] = 1.0 / 6.0
            for t in tickers_ranked[-3:]:
                positions[t] = -1.0 / 6.0

        elif variant_name == 'D':
            # Long Top-2 / Short SPY
            for t in tickers_ranked[:2]:
                positions[t] = 0.5
            positions[BENCHMARK] = -1.0

        elif variant_name == 'E':
            # Long Top-2 / Short Bottom-2 + SPY Hedge
            for t in tickers_ranked[:2]:
                positions[t] = 0.25
            for t in tickers_ranked[-2:]:
                positions[t] = -0.25
            # Add SPY hedge: short SPY for net dollar exposure = 0
            # Long = 0.5, Short sectors = -0.5, already neutral
            # SPY overlay: short additional 0.25 SPY for extra hedge
            positions[BENCHMARK] = -0.25

        elif variant_name == 'F':
            # Rank-Weighted L/S
            all_scores = np.array([s for _, s in sorted_sectors])
            median_score = np.median(all_scores)
            distances = {t: s - median_score for t, s in sorted_sectors}
            total_pos = sum(d for d in distances.values() if d > 0)
            total_neg = sum(abs(d) for d in distances.values() if d < 0)

            if total_pos > 0 and total_neg > 0:
                for t, d in distances.items():
                    if d > 0:
                        positions[t] = 0.5 * (d / total_pos)
                    elif d < 0:
                        positions[t] = -0.5 * (abs(d) / total_neg)

        # Calculate period return
        period_return = 0.0
        for ticker, weight in positions.items():
            try:
                if ticker == BENCHMARK:
                    p0 = spy_close.loc[rebal_date]
                    p1 = spy_close.loc[next_date]
                else:
                    p0 = close[ticker].loc[rebal_date]
                    p1 = close[ticker].loc[next_date]
                ret = (p1 / p0) - 1.0
                period_return += weight * ret
            except (KeyError, TypeError):
                continue

        equity *= (1 + period_return)
        equity_curve.append({'date': next_date, 'equity': equity, 'return': period_return})
        trade_returns.append(period_return)

    return pd.DataFrame(equity_curve), np.array(trade_returns)


def compute_metrics(equity_df, trade_returns, spy_close, variant_name):
    """Compute all performance metrics for a variant."""
    if len(equity_df) == 0 or len(trade_returns) == 0:
        return {}

    returns = trade_returns
    n_periods = len(returns)
    periods_per_year = 252 / REBAL_PERIOD  # ~12 months

    # Sharpe (annualized)
    if returns.std() > 0:
        sharpe = (returns.mean() / returns.std()) * np.sqrt(periods_per_year)
    else:
        sharpe = 0.0

    # Sortino
    downside = returns[returns < 0]
    if len(downside) > 0 and downside.std() > 0:
        sortino = (returns.mean() / downside.std()) * np.sqrt(periods_per_year)
    else:
        sortino = 0.0

    # Profit Factor
    gains = returns[returns > 0].sum()
    losses = abs(returns[returns < 0].sum())
    pf = gains / losses if losses > 0 else float('inf')

    # Win Rate
    wr = (returns > 0).sum() / len(returns) * 100

    # MDD
    cum_ret = (1 + pd.Series(returns)).cumprod()
    peak = cum_ret.cummax()
    dd = (cum_ret / peak - 1)
    mdd = dd.min() * 100

    # CAGR
    total_return = equity_df['equity'].iloc[-1] / STARTING_CAPITAL
    years = n_periods / periods_per_year
    cagr = (total_return ** (1 / years) - 1) * 100 if years > 0 else 0

    # Beta to SPY
    dates = equity_df['date'].values
    spy_returns = []
    for i in range(1, len(dates)):
        try:
            s0 = spy_close.loc[dates[i-1]] if i > 0 else spy_close.iloc[0]
            # Use the rebal period returns from SPY
        except:
            pass
    # Simpler beta calc: use matching period returns
    spy_period_rets = []
    for i in range(len(dates) - 1):
        try:
            s0 = spy_close.loc[dates[i]]
            s1 = spy_close.loc[dates[i + 1]]
            spy_period_rets.append(s1 / s0 - 1)
        except:
            spy_period_rets.append(0)

    # Align lengths
    min_len = min(len(returns), len(spy_period_rets))
    if min_len > 2:
        r = returns[:min_len]
        s = np.array(spy_period_rets[:min_len])
        cov_matrix = np.cov(r, s)
        beta = cov_matrix[0, 1] / cov_matrix[1, 1] if cov_matrix[1, 1] > 0 else 0
        spy_cagr = ((1 + s).prod() ** (1 / years) - 1) * 100 if years > 0 else 0
        alpha = cagr - spy_cagr
    else:
        beta = 0
        alpha = 0
        spy_cagr = 0

    return {
        'variant': variant_name,
        'sharpe': round(sharpe, 2),
        'sortino': round(sortino, 2),
        'pf': round(pf, 2),
        'wr': round(wr, 1),
        'mdd': round(mdd, 1),
        'cagr': round(cagr, 1),
        'beta': round(beta, 2),
        'alpha_vs_spy': round(alpha, 1),
        'n_rebalances': n_periods,
        'total_return': round((total_return - 1) * 100, 1),
    }


def permutation_test(features_df, close, spy_close, variant_name, rankings, actual_sharpe, n_perms=N_PERMUTATIONS):
    """Shuffle sector rankings randomly and compute Sharpe for each permutation."""
    log.info(f"  Permutation test ({n_perms} shuffles) for variant {variant_name}...")
    perm_sharpes = []

    for p in range(n_perms):
        shuffled_rankings = []
        for date, scores in rankings:
            tickers = list(scores.keys())
            shuffled_scores = list(scores.values())
            np.random.shuffle(shuffled_scores)
            shuffled_rankings.append((date, dict(zip(tickers, shuffled_scores))))

        _, perm_returns = backtest_variant(variant_name, shuffled_rankings, close, spy_close)
        if len(perm_returns) > 0 and perm_returns.std() > 0:
            periods_per_year = 252 / REBAL_PERIOD
            perm_sharpe = (perm_returns.mean() / perm_returns.std()) * np.sqrt(periods_per_year)
            perm_sharpes.append(perm_sharpe)

    perm_sharpes = np.array(perm_sharpes)
    p_value = (perm_sharpes >= actual_sharpe).sum() / len(perm_sharpes) if len(perm_sharpes) > 0 else 1.0
    return p_value, perm_sharpes


def regime_analysis(equity_df, trade_returns, spy_close):
    """Stratify Sharpe by bull/bear/flat regimes (SPY monthly: >2% bull, <-2% bear, else flat)."""
    if len(equity_df) == 0:
        return {}

    dates = equity_df['date'].values
    regimes = []

    for i, d in enumerate(dates):
        # Look back ~21 days for SPY return
        try:
            idx = spy_close.index.get_loc(d)
            if idx >= 21:
                spy_ret = spy_close.iloc[idx] / spy_close.iloc[idx - 21] - 1
                if spy_ret > 0.02:
                    regimes.append('bull')
                elif spy_ret < -0.02:
                    regimes.append('bear')
                else:
                    regimes.append('flat')
            else:
                regimes.append('flat')
        except:
            regimes.append('flat')

    regimes = np.array(regimes)
    periods_per_year = 252 / REBAL_PERIOD
    result = {}

    for regime in ['bull', 'bear', 'flat']:
        mask = regimes == regime
        if mask.sum() > 1:
            r = trade_returns[mask]
            if r.std() > 0:
                s = (r.mean() / r.std()) * np.sqrt(periods_per_year)
            else:
                s = 0
            result[f'sharpe_{regime}'] = round(s, 2)
            result[f'n_{regime}'] = int(mask.sum())
        else:
            result[f'sharpe_{regime}'] = np.nan
            result[f'n_{regime}'] = 0

    return result


def main():
    t0 = time.time()
    log.info("=" * 70)
    log.info("Market-Neutral Sector Equity Rotation Research v1")
    log.info("=" * 70)

    # 1. Download data
    close, volume, high, low = download_data(start='2005-01-01')
    spy_close = close[BENCHMARK].dropna()

    # 2. Compute features
    features_df = compute_features(close, volume, high, low)

    # 3. Walk-forward ranking
    rankings = walk_forward_ranking(features_df, close)

    if len(rankings) < 5:
        log.error("Not enough rankings generated. Aborting.")
        return

    # 4. Backtest all 6 variants
    variant_names = {
        'A': 'Long-Only Top-2',
        'B': 'L2/S2 Equal Weight',
        'C': 'L3/S3 Diversified',
        'D': 'L2 / Short SPY',
        'E': 'L2/S2 + SPY Hedge',
        'F': 'Rank-Weighted L/S',
    }

    all_results = []

    for vkey, vname in variant_names.items():
        log.info(f"\n--- Variant {vkey}: {vname} ---")
        equity_df, trade_returns = backtest_variant(vkey, rankings, close, spy_close)

        if len(trade_returns) == 0:
            log.warning(f"  No trades for variant {vkey}")
            continue

        metrics = compute_metrics(equity_df, trade_returns, spy_close, f"{vkey}: {vname}")

        # Permutation test
        p_value, perm_sharpes = permutation_test(
            features_df, close, spy_close, vkey, rankings, metrics['sharpe']
        )
        metrics['p_value'] = round(p_value, 4)

        # Regime analysis
        regime_metrics = regime_analysis(equity_df, trade_returns, spy_close)
        metrics.update(regime_metrics)

        all_results.append(metrics)

        log.info(f"  Sharpe={metrics['sharpe']}, Sortino={metrics['sortino']}, "
                 f"PF={metrics['pf']}, WR={metrics['wr']}%, MDD={metrics['mdd']}%, "
                 f"CAGR={metrics['cagr']}%, Beta={metrics['beta']}, "
                 f"Alpha={metrics['alpha_vs_spy']}%, p={metrics['p_value']}")
        log.info(f"  Regime: Bull={regime_metrics.get('sharpe_bull', 'N/A')} "
                 f"({regime_metrics.get('n_bull', 0)}), "
                 f"Bear={regime_metrics.get('sharpe_bear', 'N/A')} "
                 f"({regime_metrics.get('n_bear', 0)}), "
                 f"Flat={regime_metrics.get('sharpe_flat', 'N/A')} "
                 f"({regime_metrics.get('n_flat', 0)})")

    # 5. Summary table
    elapsed = time.time() - t0
    log.info(f"\n{'=' * 90}")
    log.info(f"FINAL SUMMARY — Market-Neutral Equity Rotation ({elapsed:.0f}s)")
    log.info(f"{'=' * 90}")

    if all_results:
        df = pd.DataFrame(all_results)
        cols = ['variant', 'sharpe', 'sortino', 'pf', 'wr', 'mdd', 'cagr',
                'beta', 'alpha_vs_spy', 'p_value', 'n_rebalances', 'total_return',
                'sharpe_bull', 'sharpe_bear', 'sharpe_flat']
        available_cols = [c for c in cols if c in df.columns]
        print("\n" + df[available_cols].to_string(index=False))

    # Regime balance check
    log.info(f"\n--- Regime Balance Check ---")
    for r in all_results:
        sb = r.get('sharpe_bull', 0) or 0
        sr = r.get('sharpe_bear', 0) or 0
        max_s = max(abs(sb), abs(sr))
        if max_s > 0:
            imbalance = abs(sb - sr) / max_s
        else:
            imbalance = 0
        passes = "PASS" if imbalance <= 0.50 else "FAIL"
        log.info(f"  {r['variant']}: |Bull-Bear|/max = {imbalance:.2f} → {passes}")

    # 6. MLflow logging
    if USE_MLFLOW:
        try:
            with mlflow.start_run(run_name=f"equity_rotation_6var_{datetime.now().strftime('%Y%m%d_%H%M')}"):
                mlflow.log_param("n_sectors", len(SECTOR_ETFS))
                mlflow.log_param("train_window", TRAIN_WINDOW)
                mlflow.log_param("rebal_period", REBAL_PERIOD)
                mlflow.log_param("n_features", len(FEATURE_COLS))
                mlflow.log_param("n_permutations", N_PERMUTATIONS)
                mlflow.log_param("n_variants", 6)

                for r in all_results:
                    v = r['variant'].split(':')[0].strip()
                    mlflow.log_metric(f"{v}_sharpe", r['sharpe'])
                    mlflow.log_metric(f"{v}_sortino", r['sortino'])
                    mlflow.log_metric(f"{v}_pf", r['pf'])
                    mlflow.log_metric(f"{v}_wr", r['wr'])
                    mlflow.log_metric(f"{v}_mdd", r['mdd'])
                    mlflow.log_metric(f"{v}_cagr", r['cagr'])
                    mlflow.log_metric(f"{v}_beta", r['beta'])
                    mlflow.log_metric(f"{v}_alpha", r['alpha_vs_spy'])
                    mlflow.log_metric(f"{v}_pvalue", r['p_value'])

                mlflow.log_metric("elapsed_seconds", elapsed)
                log.info("MLflow run logged successfully.")
        except Exception as e:
            log.warning(f"MLflow logging failed: {e}")

    # Best variant summary
    if all_results:
        best = max(all_results, key=lambda x: x['sharpe'])
        log.info(f"\n*** BEST VARIANT: {best['variant']} — Sharpe {best['sharpe']}, "
                 f"Alpha {best['alpha_vs_spy']}% vs SPY, p={best['p_value']} ***")

    log.info(f"\nCompleted in {elapsed:.1f}s")


if __name__ == '__main__':
    main()
