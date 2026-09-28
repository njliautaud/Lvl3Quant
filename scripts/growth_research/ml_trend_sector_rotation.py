#!/usr/bin/env python3
"""
ML Trend Following — SECTOR ROTATION VARIANT
==============================================
Takes the PROVEN v2 framework (Sharpe 2.90 on 8-asset CTA) and applies it
to 11 SPDR sector ETFs. Tests whether the ML trend filter generalizes
to a different universe.

Hypothesis: If v2's edge is from ML learning WHICH trends persist (real alpha),
it should work across asset universes. If it only works on the specific 8-asset
set, the edge may be universe-specific/overfit.

HC #713: Fixed capital, no DCA.
HC #0: Sliding 252d walk-forward.
Adversarial: permutation 100x, sub-period 4-block, outlier, R1 regime.
"""

import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier

warnings.filterwarnings('ignore')
np.random.seed(42)

# ─────────────────────────────────────────────────────────────────────────────
INITIAL_CAPITAL  = 100_000
TRAIN_WINDOW     = 252
N_PERMUTATIONS   = 100
REBAL_COST_BPS   = 10
MA_SHORT         = 20
MA_LONG          = 100
ML_THRESHOLD     = 0.55
TARGET_VOL       = 0.10

BASE   = Path("/home/jupiter/Lvl3Quant")
OUTPUT = BASE / "output" / "ml_trend_sectors"
OUTPUT.mkdir(parents=True, exist_ok=True)

# 11 SPDR Sector ETFs (inception ~1998 for most)
UNIVERSE = {
    'XLK': 'Technology',
    'XLF': 'Financials',
    'XLE': 'Energy',
    'XLV': 'Health Care',
    'XLY': 'Consumer Disc',
    'XLP': 'Consumer Staples',
    'XLI': 'Industrials',
    'XLB': 'Materials',
    'XLU': 'Utilities',
    'XLRE': 'Real Estate',
    'XLC': 'Communication',
}


def download_data():
    print("=" * 80)
    print("STEP 1: DOWNLOADING SECTOR DATA")
    print("=" * 80)

    all_tickers = list(UNIVERSE.keys()) + ['^VIX', 'SPY']

    cache = BASE / "data" / "cache" / "sector_trend_data.parquet"
    if cache.exists() and (time.time() - cache.stat().st_mtime) < 3600:
        df = pd.read_parquet(cache)
        print(f"  Cached: {df.shape}, {df.index[0].date()} → {df.index[-1].date()}")
        return df

    print("  Downloading...")
    raw = yf.download(all_tickers, start='2000-01-01', auto_adjust=True, progress=False)
    closes = raw['Close'] if isinstance(raw.columns, pd.MultiIndex) else raw
    if '^VIX' in closes.columns:
        closes = closes.rename(columns={'^VIX': 'VIX'})
    closes = closes.ffill().dropna(thresh=len(closes.columns) - 3)
    cache.parent.mkdir(parents=True, exist_ok=True)
    closes.to_parquet(cache)
    print(f"  Shape: {closes.shape}, {closes.index[0].date()} → {closes.index[-1].date()}")
    return closes


def generate_trend_signals(df):
    """Generate trend signals for each sector using dual-MA crossover."""
    print("\n" + "=" * 80)
    print("STEP 2: SECTOR TREND SIGNALS")
    print("=" * 80)

    signals = {}
    for ticker in UNIVERSE:
        if ticker not in df.columns:
            print(f"  SKIP {ticker} — not in data")
            continue
        price = df[ticker]
        ma_short = price.rolling(MA_SHORT).mean()
        ma_long = price.rolling(MA_LONG).mean()

        trend = pd.Series(0.0, index=df.index)
        trend[ma_short > ma_long] = 1.0
        trend[ma_short < ma_long] = -1.0

        signals[ticker] = {
            'trend': trend,
            'ma_short': ma_short,
            'ma_long': ma_long,
            'price': price,
        }

        up_pct = (trend == 1).mean() * 100
        dn_pct = (trend == -1).mean() * 100
        print(f"  {ticker} ({UNIVERSE[ticker]:18s}): up {up_pct:.0f}%, down {dn_pct:.0f}%")

    return signals


def build_ml_features(df, signals):
    """Build ML features — VECTORIZED for speed."""
    print("\n" + "=" * 80)
    print("STEP 3: ML FEATURES (vectorized)")
    print("=" * 80)

    start_idx = MA_LONG + 60
    all_frames = []

    # Pre-compute cross-sector trend alignment matrix
    trend_matrix = pd.DataFrame({t: s['trend'] for t, s in signals.items()})

    # Pre-compute sector dispersion and breadth
    mom20_matrix = pd.DataFrame({t: s['price'].pct_change(20) for t, s in signals.items()})
    sector_dispersion = mom20_matrix.std(axis=1)
    sector_breadth = (mom20_matrix > 0).mean(axis=1)

    # Pre-compute SPY momentum
    spy_mom20 = df['SPY'].pct_change(20) if 'SPY' in df.columns else None
    spy_mom60 = df['SPY'].pct_change(60) if 'SPY' in df.columns else None

    # VIX features
    vix = df['VIX'] if 'VIX' in df.columns else None
    vix_ma20 = df['VIX'].rolling(20).mean() if 'VIX' in df.columns else None

    for ticker, sig in signals.items():
        print(f"  Building features for {ticker}...", flush=True)
        price = sig['price']
        trend = sig['trend']
        ma_s = sig['ma_short']
        ma_l = sig['ma_long']
        ret = price.pct_change()

        # Vectorized trend duration: count consecutive same-direction days
        trend_changed = trend.diff().ne(0)
        trend_groups = trend_changed.cumsum()
        trend_duration = trend_groups.groupby(trend_groups).cumcount() + 1

        # Vectorized momentum
        mom5 = price.pct_change(5)
        mom20 = price.pct_change(20)
        mom60 = price.pct_change(60)

        # Vectorized volatility
        vol20 = ret.rolling(20).std() * np.sqrt(252)
        vol60 = ret.rolling(60).std() * np.sqrt(252)

        # Vectorized drawdown
        rolling_max = price.rolling(252, min_periods=1).max()
        drawdown = price / rolling_max - 1

        # Vectorized skew and kurtosis
        skew20 = ret.rolling(20).skew()
        kurt20 = ret.rolling(20).kurt()

        # Cross-sector alignment for this ticker
        other_tickers = [t for t in signals if t != ticker]
        if other_tickers:
            same_dir_count = (trend_matrix[other_tickers].eq(trend, axis=0)).sum(axis=1)
            cross_align = same_dir_count / len(other_tickers)
        else:
            cross_align = pd.Series(0.5, index=df.index)

        # MA distance
        ma_dist = (ma_s - ma_l) / (price + 1e-8)

        # Build dataframe for this ticker (only rows where trend != 0 and i >= start_idx)
        mask = (trend != 0)
        mask.iloc[:start_idx] = False

        idx = df.index[mask]
        if len(idx) == 0:
            continue

        ticker_df = pd.DataFrame({
            'ticker': ticker,
            'date': idx,
            'direction': trend[mask].values,
            'ma_dist': ma_dist[mask].values,
            'trend_duration': trend_duration[mask].values.clip(max=252),
            'mom_5d': mom5[mask].values,
            'mom_20d': mom20[mask].values,
            'mom_60d': mom60[mask].values,
            'vol_20d': vol20[mask].values,
            'vol_60d': vol60[mask].values,
            'drawdown': drawdown[mask].values,
            'cross_align': cross_align[mask].values,
            'skew_20d': skew20[mask].values,
            'kurt_20d': kurt20[mask].values,
            'sector_dispersion': sector_dispersion[mask].values,
            'sector_breadth': sector_breadth[mask].values,
        })

        if vix is not None:
            ticker_df['vix'] = vix[mask].values
            ticker_df['vix_ma20'] = vix_ma20[mask].values

        if spy_mom20 is not None:
            ticker_df['rel_strength_20d'] = mom20[mask].values - spy_mom20[mask].values
            ticker_df['rel_strength_60d'] = mom60[mask].values - spy_mom60[mask].values

        # Target: trend continues profitably over next 20 days?
        future_price = price.shift(-20)
        future_ret = (future_price / price - 1) * trend
        ticker_df['target'] = (future_ret[mask] > 0).astype(float).values
        ticker_df.loc[future_price[mask].isna().values, 'target'] = np.nan
        ticker_df['future_ret'] = future_ret[mask].values

        all_frames.append(ticker_df)

    features_df = pd.concat(all_frames, ignore_index=True)
    print(f"  Total observations: {len(features_df)}")
    print(f"  Positive target rate: {features_df['target'].mean():.1%}")
    print(f"  Assets: {features_df['ticker'].nunique()}")

    return features_df


def train_ml_filter(features_df):
    """Walk-forward GBM — OPTIMIZED with pre-indexed date lookups."""
    print("\n" + "=" * 80)
    print("STEP 4: WALK-FORWARD ML FILTER (optimized)")
    print("=" * 80)

    feature_cols = [c for c in features_df.columns
                   if c not in ['ticker', 'date', 'direction', 'target', 'future_ret']]

    features_df = features_df.sort_values('date').reset_index(drop=True)
    features_df['ml_prob'] = np.nan

    dates = sorted(features_df['date'].unique())

    # Pre-build date→row index for O(1) lookups instead of O(n) scans
    date_to_rows = features_df.groupby('date').apply(lambda g: g.index.tolist()).to_dict()

    # Pre-extract numpy arrays for speed
    X_all = features_df[feature_cols].fillna(0).values
    y_all = features_df['target'].values
    ml_probs = np.full(len(features_df), np.nan)

    n_folds = 0
    RETRAIN_EVERY = 5  # Retrain model every 5 days (weekly), reuse for intervening days
    total_folds = len(dates) - TRAIN_WINDOW
    t0 = time.time()

    current_model = None

    for i in range(TRAIN_WINDOW, len(dates)):
        # Only retrain every RETRAIN_EVERY days
        need_retrain = (current_model is None) or ((i - TRAIN_WINDOW) % RETRAIN_EVERY == 0)

        if need_retrain:
            train_start_idx = max(0, i - TRAIN_WINDOW)

            # Collect train rows from date index
            train_rows = []
            for d_idx in range(train_start_idx, i):
                train_rows.extend(date_to_rows.get(dates[d_idx], []))

            if not train_rows:
                continue

            train_rows = np.array(train_rows)
            X_train = X_all[train_rows]
            y_train = y_all[train_rows]

            valid = ~np.isnan(y_train)
            X_train = X_train[valid]
            y_train = y_train[valid]

            if len(X_train) < 50:
                continue

            current_model = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=3,
                learning_rate=0.05,
                subsample=0.8,
                random_state=42
            )
            current_model.fit(X_train, y_train)
            n_folds += 1

        # Predict for today using current model
        test_rows = date_to_rows.get(dates[i], [])
        if not test_rows or current_model is None:
            continue

        test_rows = np.array(test_rows)
        X_test = X_all[test_rows]
        probs = current_model.predict_proba(X_test)[:, 1]
        ml_probs[test_rows] = probs

        # Progress every 200 retrains
        if n_folds > 0 and n_folds % 200 == 0 and need_retrain:
            elapsed = time.time() - t0
            est_total_retrains = total_folds // RETRAIN_EVERY
            rate = n_folds / elapsed
            remaining = (est_total_retrains - n_folds) / rate if rate > 0 else 0
            print(f"    Retrain {n_folds}/~{est_total_retrains} ({n_folds/est_total_retrains*100:.0f}%) — "
                  f"{elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining", flush=True)

    features_df['ml_prob'] = ml_probs

    valid_mask = ~np.isnan(ml_probs) & ~np.isnan(y_all)
    if valid_mask.sum() > 0:
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(y_all[valid_mask], ml_probs[valid_mask])
        accuracy = ((ml_probs[valid_mask] > 0.5) == y_all[valid_mask]).mean()
        print(f"  Walk-forward folds: {n_folds}")
        print(f"  OOS AUC: {auc:.3f}")
        print(f"  OOS Accuracy: {accuracy:.1%}")
        print(f"  Total time: {(time.time() - t0)/60:.1f} min")

    return features_df


def backtest_strategy(features_df, df):
    """Backtest — IDENTICAL to v2."""
    print("\n" + "=" * 80)
    print("STEP 5: BACKTEST")
    print("=" * 80)

    pred_df = features_df.dropna(subset=['ml_prob']).copy()
    dates = sorted(pred_df['date'].unique())

    portfolio_returns = []

    for date in dates:
        day_signals = pred_df[pred_df['date'] == date]
        high_conf = day_signals[day_signals['ml_prob'] > ML_THRESHOLD]

        if len(high_conf) == 0:
            portfolio_returns.append({'date': date, 'return': 0.0, 'n_positions': 0})
            continue

        positions = []
        for _, row in high_conf.iterrows():
            ticker = row['ticker']
            direction = row['direction']
            vol = max(row['vol_20d'], 0.05)
            weight = (TARGET_VOL / vol) / len(high_conf)
            weight = min(weight, 0.5)

            if ticker in df.columns:
                ticker_dates = df.index
                date_loc = ticker_dates.get_loc(date) if date in ticker_dates else None
                if date_loc is not None and date_loc + 1 < len(ticker_dates):
                    next_ret = df[ticker].iloc[date_loc + 1] / df[ticker].iloc[date_loc] - 1
                    positions.append({
                        'ticker': ticker,
                        'direction': direction,
                        'weight': weight,
                        'return': next_ret * direction * weight,
                    })

        if positions:
            day_ret = sum(p['return'] for p in positions)
            turnover = sum(p['weight'] for p in positions)
            cost = turnover * REBAL_COST_BPS / 10000
            day_ret -= cost * 0.1
            portfolio_returns.append({
                'date': date,
                'return': day_ret,
                'n_positions': len(positions),
            })
        else:
            portfolio_returns.append({'date': date, 'return': 0.0, 'n_positions': 0})

    ret_df = pd.DataFrame(portfolio_returns).set_index('date')
    ret_series = ret_df['return']

    print(f"  Trading days: {len(ret_series)}")
    print(f"  Mean positions/day: {ret_df['n_positions'].mean():.1f}")
    print(f"  Days with positions: {(ret_df['n_positions'] > 0).sum()} ({(ret_df['n_positions'] > 0).mean():.1%})")

    return ret_series, ret_df


def compute_metrics(returns, name="Strategy"):
    r = returns.dropna()
    if len(r) < 30:
        return {}

    mu = r.mean() * 252
    sigma = r.std() * np.sqrt(252)
    sharpe = mu / (sigma + 1e-8)

    downside = r[r < 0].std() * np.sqrt(252)
    sortino = mu / (downside + 1e-8)

    cumret = (1 + r).cumprod()
    total_return = cumret.iloc[-1] - 1
    years = len(r) / 252
    cagr = (cumret.iloc[-1]) ** (1 / years) - 1 if years > 0 else 0

    running_max = cumret.cummax()
    drawdown = cumret / running_max - 1
    max_dd = drawdown.min()
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0

    gains = r[r > 0].sum()
    losses = abs(r[r < 0].sum())
    pf = gains / (losses + 1e-8)
    wr = (r > 0).mean()

    return {
        'name': name, 'sharpe': round(sharpe, 3), 'sortino': round(sortino, 3),
        'cagr': round(cagr * 100, 1), 'max_dd': round(max_dd * 100, 1),
        'calmar': round(calmar, 3), 'total_return': round(total_return * 100, 1),
        'profit_factor': round(pf, 3), 'win_rate': round(wr * 100, 1),
        'annual_vol': round(sigma * 100, 1), 'years': round(years, 1),
    }


def run_adversarial(ret_series, features_df, df):
    """Full adversarial validation — IDENTICAL to v2."""
    print("\n" + "=" * 80)
    print("ADVERSARIAL VALIDATION")
    print("=" * 80)

    results = {}
    real = compute_metrics(ret_series, "real")
    real_sharpe = real.get('sharpe', 0)

    # 1. PERMUTATION TEST
    print("\n  [1/4] Permutation test...")
    perm_sharpes = []
    for trial in range(N_PERMUTATIONS):
        perm_df = features_df.copy()
        perm_df['ml_prob'] = np.random.permutation(perm_df['ml_prob'].values)

        pred_df = perm_df.dropna(subset=['ml_prob']).copy()
        dates = sorted(pred_df['date'].unique())
        perm_rets = []

        for date in dates:
            day_signals = pred_df[pred_df['date'] == date]
            high_conf = day_signals[day_signals['ml_prob'] > ML_THRESHOLD]

            if len(high_conf) == 0:
                perm_rets.append(0.0)
                continue

            day_ret = 0.0
            for _, row in high_conf.iterrows():
                ticker = row['ticker']
                direction = row['direction']
                vol = max(row.get('vol_20d', 0.2), 0.05)
                weight = (TARGET_VOL / vol) / len(high_conf)
                weight = min(weight, 0.5)

                if ticker in df.columns:
                    ticker_dates = df.index
                    date_loc = ticker_dates.get_loc(date) if date in ticker_dates else None
                    if date_loc is not None and date_loc + 1 < len(ticker_dates):
                        next_ret = df[ticker].iloc[date_loc + 1] / df[ticker].iloc[date_loc] - 1
                        day_ret += next_ret * direction * weight

            perm_rets.append(day_ret)

        perm_ret_series = pd.Series(perm_rets, index=dates)
        m = compute_metrics(perm_ret_series, f"perm_{trial}")
        if m:
            perm_sharpes.append(m['sharpe'])

    perm_p = np.mean([s >= real_sharpe for s in perm_sharpes]) if perm_sharpes else 1.0
    perm_pass = perm_p < 0.05
    print(f"    Real Sharpe: {real_sharpe:.3f}")
    print(f"    Perm mean: {np.mean(perm_sharpes):.3f} ± {np.std(perm_sharpes):.3f}")
    print(f"    p-value: {perm_p:.3f} → {'PASS' if perm_pass else 'FAIL'}")
    results['permutation'] = {'p_value': float(perm_p), 'pass': bool(perm_pass)}

    # 2. SUB-PERIOD
    print("\n  [2/4] Sub-period consistency...")
    n = len(ret_series)
    block_size = n // 4
    block_sharpes = []
    for b in range(4):
        start = b * block_size
        end = (b + 1) * block_size if b < 3 else n
        m = compute_metrics(ret_series.iloc[start:end], f"block_{b}")
        if m:
            block_sharpes.append(m['sharpe'])
            print(f"    Block {b+1}: Sharpe {m['sharpe']:.3f}")

    cv = np.std(block_sharpes) / abs(np.mean(block_sharpes)) if block_sharpes and np.mean(block_sharpes) != 0 else 999
    sub_pass = cv < 0.50
    print(f"    CV: {cv:.3f} → {'PASS' if sub_pass else 'FAIL'}")
    results['sub_period'] = {'cv': round(cv, 3), 'pass': bool(sub_pass)}

    # 3. OUTLIER
    print("\n  [3/4] Outlier robustness...")
    p95 = ret_series.quantile(0.95)
    p05 = ret_series.quantile(0.05)
    trimmed = ret_series[(ret_series > p05) & (ret_series < p95)]
    m_full = compute_metrics(ret_series)
    m_trim = compute_metrics(trimmed)
    if m_full.get('sharpe', 0) != 0:
        deg = (m_full['sharpe'] - m_trim['sharpe']) / abs(m_full['sharpe'])
    else:
        deg = 999
    outlier_pass = abs(deg) < 0.30
    print(f"    Degradation: {deg:.1%} → {'PASS' if outlier_pass else 'FAIL'}")
    results['outlier'] = {'degradation': round(deg, 3), 'pass': bool(outlier_pass)}

    # 4. R1 REGIME
    print("\n  [4/4] R1 regime test...")
    spy_ret = df['SPY'].pct_change()
    spy_20d = spy_ret.rolling(20).sum()

    green_mask = ret_series.index.map(lambda x: spy_20d.loc[x] > 0 if x in spy_20d.index else True)
    m_green = compute_metrics(ret_series[green_mask], "green")
    m_red = compute_metrics(ret_series[~green_mask], "red")

    if m_green and m_red:
        s_g, s_r = m_green['sharpe'], m_red['sharpe']
        gap = abs(s_g - s_r) / max(abs(s_g), abs(s_r), 0.01)
        r1_pass = gap < 0.50
        print(f"    Green: {s_g:.3f}, Red: {s_r:.3f}, Gap: {gap:.3f} → {'PASS' if r1_pass else 'FAIL'}")
        results['r1_regime'] = {'gap': round(gap, 3), 'pass': bool(r1_pass),
                                'green_sharpe': s_g, 'red_sharpe': s_r}
    else:
        r1_pass = False
        results['r1_regime'] = {'pass': False}

    gates = sum([results.get(k, {}).get('pass', False) for k in ['permutation', 'sub_period', 'outlier', 'r1_regime']])
    results['summary'] = {'gates_passed': int(gates), 'total': 4, 'verdict': 'PASS' if gates >= 3 else 'FAIL'}
    print(f"\n  SUMMARY: {gates}/4 gates → {results['summary']['verdict']}")

    return results


def main():
    t0 = time.time()
    print("=" * 80)
    print("ML TREND FOLLOWING — SECTOR ROTATION VARIANT")
    print("Tests v2 framework generalizability on 11 SPDR sector ETFs")
    print("=" * 80)

    df = download_data()
    signals = generate_trend_signals(df)
    features_df = build_ml_features(df, signals)
    features_df = train_ml_filter(features_df)
    ret_series, ret_df = backtest_strategy(features_df, df)

    # Compute metrics
    m = compute_metrics(ret_series, "ML Sector Rotation")
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)

    # Benchmarks
    idx = ret_series.index
    spy_ret = df['SPY'].pct_change().reindex(idx).fillna(0)
    bm_spy = compute_metrics(spy_ret, "SPY B&H")

    # Simple trend (no ML filter)
    all_signals_df = features_df.copy()
    all_signals_df['ml_prob'] = 1.0
    simple_ret, _ = backtest_strategy(all_signals_df, df)
    simple_ret = simple_ret.reindex(idx).fillna(0)
    bm_simple = compute_metrics(simple_ret, "Simple Sector Trend")

    for label, metrics in [("ML Sector Rotation", m), ("SPY B&H", bm_spy), ("Simple Trend (no ML)", bm_simple)]:
        if metrics:
            print(f"\n  {label}:")
            print(f"    Sharpe: {metrics['sharpe']:.3f}  |  Sortino: {metrics['sortino']:.3f}")
            print(f"    CAGR: {metrics['cagr']:.1f}%  |  MaxDD: {metrics['max_dd']:.1f}%")
            print(f"    PF: {metrics['profit_factor']:.3f}  |  WR: {metrics['win_rate']:.1f}%")
            print(f"    Calmar: {metrics['calmar']:.3f}  |  Vol: {metrics['annual_vol']:.1f}%")

    # Correlation with v2 (if v2 results exist)
    v2_results_path = BASE / "output" / "ml_trend_following" / "results.json"
    if v2_results_path.exists():
        print(f"\n  V2 COMPARISON (8-asset CTA):")
        v2 = json.loads(v2_results_path.read_text())
        v2m = v2['metrics']['portfolio']
        print(f"    V2 Sharpe: {v2m['sharpe']:.3f}  |  Sector Sharpe: {m.get('sharpe', 'N/A')}")
        print(f"    V2 Sortino: {v2m['sortino']:.3f}  |  Sector Sortino: {m.get('sortino', 'N/A')}")

    # Adversarial
    adv = run_adversarial(ret_series, features_df, df)

    # Equity curve
    try:
        cumret = (1 + ret_series).cumprod()
        spy_cumret = (1 + spy_ret).cumprod()

        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(cumret.index, cumret.values, label=f"ML Sector Rotation (Sharpe {m.get('sharpe', '?')})", linewidth=2)
        ax.plot(spy_cumret.index, spy_cumret.values, label=f"SPY B&H (Sharpe {bm_spy.get('sharpe', '?')})", alpha=0.7)
        ax.set_title("ML Sector Rotation — Equity Curve")
        ax.set_ylabel("Growth of $1")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.savefig(OUTPUT / "equity_curve.png", dpi=150, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"  Chart error: {e}")

    # Save results
    result = {
        'strategy': 'ML Sector Rotation',
        'metrics': {
            'portfolio': m,
            'spy': bm_spy,
            'simple_trend': bm_simple,
        },
        'adversarial': adv,
        'parameters': {
            'ma_short': MA_SHORT, 'ma_long': MA_LONG,
            'ml_threshold': ML_THRESHOLD, 'target_vol': TARGET_VOL,
            'train_window': TRAIN_WINDOW,
            'universe': list(UNIVERSE.keys()),
        },
        'runtime_seconds': round(time.time() - t0, 1),
    }
    (OUTPUT / "results.json").write_text(json.dumps(result, indent=2, default=str))
    print(f"\n  Results saved to {OUTPUT}")
    print(f"  Runtime: {(time.time() - t0)/60:.1f} minutes")

    # Print verdict
    print("\n" + "=" * 80)
    sharpe = m.get('sharpe', 0)
    gates = adv.get('summary', {}).get('gates_passed', 0)
    if sharpe > 1.5 and gates >= 3:
        print(f"✅ SECTOR ROTATION WORKS — Sharpe {sharpe}, {gates}/4 gates. V2 framework GENERALIZES.")
        print("   This is a SECOND validated strategy. Check correlation with v2 for portfolio construction.")
    elif sharpe > 1.0:
        print(f"🟡 MODERATE — Sharpe {sharpe}, {gates}/4 gates. Decent but not as strong as v2.")
    else:
        print(f"❌ SECTOR ROTATION FAILS — Sharpe {sharpe}, {gates}/4 gates.")
        print("   V2's edge may be universe-specific (broad macro assets, not sectors).")
    print("=" * 80)


if __name__ == '__main__':
    main()
