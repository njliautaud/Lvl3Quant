#!/usr/bin/env python3
"""
CTA / Managed Futures Trend-Following Strategy v2
====================================================

Classic trend-following applied to diversified asset classes via ETFs.
Uses LightGBM to combine multiple trend signals (MA crossovers, breakouts,
time-series momentum) into a composite signal.

Universe: SPY, TLT, GLD, DBC, UUP, EFA, EEM, IWM, HYG, VNQ
  - Equities (US large/small, intl developed, EM)
  - Bonds (long treasury, high yield)
  - Commodities (broad, gold)
  - Currency (dollar index)
  - Real estate

This is designed to be UNCORRELATED with equity-only strategies.
The core thesis: trend-following captures crisis alpha and works across
asset classes because of behavioral persistence (herding, slow info diffusion).

Method:
  - 252d train / 21d OOT sliding walk-forward (NEVER expanding)
  - LightGBM cross-sectional ranking of trend signals
  - Top 3 assets, equal weight, monthly rebalance
  - 10 bps round-trip cost
  - 4 adversarial gates: permutation, regime R1, sub-period, outlier

Run: python3 cta_trend_following_v2.py
"""

import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from scipy.stats import norm
import warnings
warnings.filterwarnings('ignore')

def fprint(*args, **kwargs):
    print(*args, **kwargs)
    sys.stdout.flush()

RESULTS_DIR = Path(__file__).resolve().parent / 'research' / 'findings'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_PATH = RESULTS_DIR / 'cta_trend_following_v2_results.json'

# MLflow
MLFLOW_OK = False
try:
    import urllib.request
    urllib.request.urlopen('http://jupiter:5000/', timeout=2)
    import mlflow
    mlflow.set_tracking_uri('http://jupiter:5000')
    MLFLOW_OK = True
except:
    pass

# Diversified universe: covers equities, bonds, commodities, currencies, real estate
UNIVERSE = [
    'SPY',   # US Large Cap Equities
    'IWM',   # US Small Cap
    'EFA',   # International Developed
    'EEM',   # Emerging Markets
    'TLT',   # Long-term US Treasury
    'HYG',   # High Yield Corporate Bonds
    'GLD',   # Gold
    'DBC',   # Broad Commodities
    'UUP',   # US Dollar Index
    'VNQ',   # US Real Estate
]

TOP_K = 3


def build_trend_features(close, volume):
    """
    Build trend-following features for a single asset.

    These capture the core phenomena that drive CTA returns:
    1. Time-series momentum (absolute returns over various lookbacks)
    2. MA crossover signals (trend direction)
    3. Breakout signals (new highs/lows)
    4. Volatility-adjusted momentum (risk parity perspective)
    5. Mean reversion indicators (to avoid whipsaws)
    """
    lr = np.log(close / close.shift(1))

    feat = pd.DataFrame(index=close.index)

    # === TIME-SERIES MOMENTUM (TSMOM) ===
    # The core trend signal: past returns predict future returns
    feat['tsmom_5d'] = close.pct_change(5)
    feat['tsmom_10d'] = close.pct_change(10)
    feat['tsmom_21d'] = close.pct_change(21)
    feat['tsmom_63d'] = close.pct_change(63)
    feat['tsmom_126d'] = close.pct_change(126)
    feat['tsmom_252d'] = close.pct_change(252)

    # 12-1 momentum (skip most recent month to avoid reversal)
    feat['mom_12_1'] = close.pct_change(252) - close.pct_change(21)

    # === MA CROSSOVER SIGNALS ===
    # Classic trend signals used by CTAs
    ma_10 = close.rolling(10).mean()
    ma_20 = close.rolling(20).mean()
    ma_50 = close.rolling(50).mean()
    ma_100 = close.rolling(100).mean()
    ma_200 = close.rolling(200).mean()

    # Binary crossover signals (normalized by price for cross-asset comparability)
    feat['ma_10_50_cross'] = (ma_10 - ma_50) / close
    feat['ma_20_100_cross'] = (ma_20 - ma_100) / close
    feat['ma_50_200_cross'] = (ma_50 - ma_200) / close

    # Price relative to MAs (trend positioning)
    feat['price_vs_ma50'] = (close - ma_50) / ma_50
    feat['price_vs_ma200'] = (close - ma_200) / ma_200

    # === BREAKOUT SIGNALS ===
    feat['high_52w_pct'] = close / close.rolling(252).max()
    feat['low_52w_pct'] = close / close.rolling(252).min()
    feat['high_13w_pct'] = close / close.rolling(63).max()

    # Donchian channel position (0 to 1)
    roll_high_20 = close.rolling(20).max()
    roll_low_20 = close.rolling(20).min()
    feat['donchian_pos'] = (close - roll_low_20) / (roll_high_20 - roll_low_20 + 1e-8)

    # === VOLATILITY FEATURES ===
    feat['vol_20d'] = lr.rolling(20).std() * np.sqrt(252)
    feat['vol_60d'] = lr.rolling(60).std() * np.sqrt(252)
    feat['vol_ratio'] = lr.rolling(20).std() / lr.rolling(60).std()

    # Vol-adjusted momentum (Sharpe-like, what CTAs actually use)
    feat['vol_adj_mom_63d'] = close.pct_change(63) / (lr.rolling(63).std() * np.sqrt(252) + 1e-8)
    feat['vol_adj_mom_126d'] = close.pct_change(126) / (lr.rolling(126).std() * np.sqrt(252) + 1e-8)

    # === ACCELERATION / TREND STRENGTH ===
    feat['mom_accel'] = close.pct_change(63) - close.pct_change(63).shift(63)
    feat['trend_strength'] = abs(close.pct_change(63)) / (lr.rolling(63).std() * np.sqrt(252) + 1e-8)

    # === MAX DRAWDOWN (RISK) ===
    roll_max_63 = close.rolling(63).max()
    feat['maxdd_63d'] = (close / roll_max_63 - 1).rolling(63).min()

    # === VOLUME FEATURES (if available) ===
    if volume is not None and not volume.isna().all():
        feat['vol_rel'] = volume / volume.rolling(20).mean()
        feat['vol_trend'] = volume.rolling(5).mean() / volume.rolling(20).mean()
    else:
        feat['vol_rel'] = 0
        feat['vol_trend'] = 0

    # === HIGHER MOMENTS ===
    feat['skew_63d'] = lr.rolling(63).skew()
    feat['kurt_63d'] = lr.rolling(63).kurt()

    # === MEAN REVERSION INDICATOR (avoid whipsaws) ===
    # RSI-like indicator
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    feat['rsi_14'] = 100 - (100 / (1 + gain / (loss + 1e-8)))

    return feat


def run_backtest(universe=UNIVERSE, top_k=TOP_K, start_year='2008', cost_bps=10):
    """Run walk-forward LightGBM trend-following backtest."""
    import yfinance as yf
    try:
        import lightgbm as lgb
    except ImportError:
        fprint("ERROR: LightGBM not available")
        return None

    fprint(f"\n{'='*60}")
    fprint(f"CTA TREND-FOLLOWING v2 -- Walk-Forward Backtest")
    fprint(f"{'='*60}")
    fprint(f"Universe: {universe}")
    fprint(f"Top K: {top_k}, Cost: {cost_bps} bps RT")
    fprint(f"Method: 252d train / 21d OOT SLIDING walk-forward")
    fprint(f"{'='*60}\n")

    # Download data
    fprint(f"Downloading {len(universe)} assets...")
    all_data = {}
    for t in universe:
        try:
            df = yf.download(t, start=f'{start_year}-01-01', end='2026-07-24', progress=False)
            if df.index.tz is not None:
                df.index = df.index.tz_convert(None)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            if len(df) > 252:
                all_data[t] = df
                fprint(f"  {t}: {len(df)} days")
        except Exception as e:
            fprint(f"  {t}: FAILED ({e})")

    # Download SPY benchmark
    spy = yf.download('SPY', start=f'{start_year}-01-01', end='2026-07-24', progress=False)
    if spy.index.tz is not None:
        spy.index = spy.index.tz_convert(None)
    if isinstance(spy.columns, pd.MultiIndex):
        spy.columns = spy.columns.get_level_values(0)

    fprint(f"\nLoaded {len(all_data)} assets with sufficient history")

    # Find common dates
    common = None
    for t, df in all_data.items():
        common = df.index if common is None else common.intersection(df.index)

    fprint(f"Common trading days: {len(common)}")

    # Build feature panels
    fprint("\nBuilding trend features...")
    feat_dict = {}
    close_dict = {}
    for t, df in all_data.items():
        df_c = df.loc[common]
        close_dict[t] = df_c['Close']
        vol = df_c['Volume'] if 'Volume' in df_c.columns else None
        feat_dict[t] = build_trend_features(df_c['Close'], vol)

    # Forward returns (target)
    fwd_ret = {}
    for t in all_data:
        fwd_ret[t] = close_dict[t].pct_change(21).shift(-21)

    # Walk-forward
    TRAIN_DAYS = 252
    OOT_DAYS = 21
    tickers = list(all_data.keys())
    FEAT_COLS = feat_dict[tickers[0]].columns.tolist()
    n_feat = len(FEAT_COLS)

    dates = sorted(common)
    min_start = TRAIN_DAYS + 252 + 21  # Need warmup for features
    n_dates = len(dates)

    # Pre-convert everything to numpy for speed
    fprint("Converting to numpy arrays...")
    feat_arrays = {}  # ticker -> (n_dates, n_feat) numpy array
    fwd_arrays = {}   # ticker -> (n_dates,) numpy array
    close_arrays = {} # ticker -> (n_dates,) numpy array
    for t in tickers:
        feat_arrays[t] = feat_dict[t].values.astype(np.float64)
        fwd_arrays[t] = fwd_ret[t].values.astype(np.float64)
        close_arrays[t] = close_dict[t].values.astype(np.float64)

    fprint(f"\nFeatures: {n_feat}")
    fprint(f"Walk-forward: {TRAIN_DAYS}d train, {OOT_DAYS}d OOT, sliding")
    fprint(f"Starting from index {min_start}, total dates: {n_dates}")

    monthly_returns = []
    monthly_picks = []

    step = 0
    i = min_start
    while i + OOT_DAYS <= n_dates:
        train_start = i - TRAIN_DAYS
        train_end = i

        oot_start = i
        oot_end = min(oot_start + OOT_DAYS, n_dates)

        # Build training data from pre-computed numpy arrays
        X_parts = []
        y_parts = []
        for t in tickers:
            X_slice = feat_arrays[t][train_start:train_end]
            y_slice = fwd_arrays[t][train_start:train_end]
            # Mask: no NaN in either
            mask = ~(np.isnan(X_slice).any(axis=1) | np.isnan(y_slice))
            if mask.sum() > 0:
                X_parts.append(X_slice[mask])
                y_parts.append(y_slice[mask])

        if not X_parts:
            i += OOT_DAYS
            continue

        X_train = np.vstack(X_parts)
        y_train = np.concatenate(y_parts)

        if len(X_train) < 50:
            i += OOT_DAYS
            continue

        # Train LightGBM
        model = lgb.LGBMRegressor(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_samples=10,
            reg_alpha=0.1,
            reg_lambda=1.0,
            verbose=-1,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)

        # Predict at OOT start for all assets
        scores = {}
        for t in tickers:
            x_oot = feat_arrays[t][oot_start]
            if np.isnan(x_oot).any():
                continue
            pred = model.predict(x_oot.reshape(1, -1))[0]
            scores[t] = pred

        if len(scores) < top_k:
            i += OOT_DAYS
            continue

        # Select top K
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        picks = [t for t, _ in ranked[:top_k]]

        # Calculate realized return over OOT window
        if oot_end - oot_start < 2:
            i += OOT_DAYS
            continue

        port_ret = 0
        for t in picks:
            start_price = close_arrays[t][oot_start]
            end_price = close_arrays[t][oot_end - 1]
            ret = end_price / start_price - 1
            port_ret += ret / top_k

        # Apply cost
        cost = cost_bps / 10000 * 2  # round-trip cost per rebalance
        net_ret = port_ret - cost

        oot_date = dates[oot_start]
        monthly_returns.append({
            'date': str(oot_date.date()),
            'return': float(net_ret),
            'gross_return': float(port_ret),
            'picks': picks,
        })
        monthly_picks.append({
            'date': str(oot_date.date()),
            'picks': picks,
            'return': float(net_ret),
        })

        step += 1
        if step % 12 == 0:
            cumret = np.prod([1 + m['return'] for m in monthly_returns]) - 1
            fprint(f"  Step {step}: {oot_date.date()} | Cum return: {cumret*100:+.1f}% | Picks: {picks}")

        i += OOT_DAYS

    if len(monthly_returns) == 0:
        fprint("ERROR: No monthly returns generated")
        return None

    # Compute metrics
    rets = np.array([m['return'] for m in monthly_returns])
    n_months = len(rets)

    cum_equity = np.cumprod(1 + rets) * CAPITAL_INITIAL
    final_equity = cum_equity[-1]

    ann_ret = (final_equity / CAPITAL_INITIAL) ** (12 / n_months) - 1
    ann_vol = np.std(rets) * np.sqrt(12)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0

    downside = rets[rets < 0]
    downside_vol = np.std(downside) * np.sqrt(12) if len(downside) > 0 else 1e-6
    sortino = ann_ret / downside_vol

    # Max drawdown
    peak = np.maximum.accumulate(cum_equity)
    dd = (cum_equity - peak) / peak
    maxdd = float(np.min(dd) * 100)

    win_rate = float(np.mean(rets > 0) * 100)

    gross_gains = rets[rets > 0].sum()
    gross_losses = abs(rets[rets < 0].sum())
    pf = float(gross_gains / gross_losses) if gross_losses > 0 else 999

    cagr = ann_ret * 100
    calmar = cagr / abs(maxdd) if maxdd != 0 else 999

    CAPITAL_INITIAL_VAL = 100000

    fprint(f"\n{'='*60}")
    fprint(f"RESULTS: CTA Trend-Following v2")
    fprint(f"{'='*60}")
    fprint(f"Months:        {n_months}")
    fprint(f"Final equity:  ${final_equity:,.2f}")
    fprint(f"CAGR:          {cagr:.1f}%")
    fprint(f"Sharpe:        {sharpe:.2f}")
    fprint(f"Sortino:       {sortino:.2f}")
    fprint(f"Max DD:        {maxdd:.1f}%")
    fprint(f"Win Rate:      {win_rate:.1f}%")
    fprint(f"Profit Factor: {pf:.2f}")
    fprint(f"Calmar:        {calmar:.2f}")

    # SPY benchmark
    spy_monthly = []
    i = min_start
    while i + OOT_DAYS <= n_dates:
        oot_start = i
        oot_end = min(oot_start + OOT_DAYS, n_dates)
        oot_dates_range = dates[oot_start:oot_end]
        if len(oot_dates_range) >= 2:
            s = spy.loc[oot_dates_range[0]:oot_dates_range[-1], 'Close']
            if len(s) >= 2:
                spy_monthly.append(float(s.iloc[-1]) / float(s.iloc[0]) - 1)
        i += OOT_DAYS

    spy_rets = np.array(spy_monthly[:n_months])
    spy_cum = np.cumprod(1 + spy_rets) * CAPITAL_INITIAL_VAL
    spy_ann_ret = (spy_cum[-1] / CAPITAL_INITIAL_VAL) ** (12 / len(spy_rets)) - 1
    spy_ann_vol = np.std(spy_rets) * np.sqrt(12)
    spy_sharpe = spy_ann_ret / spy_ann_vol if spy_ann_vol > 0 else 0
    spy_peak = np.maximum.accumulate(spy_cum)
    spy_dd = float(np.min((spy_cum - spy_peak) / spy_peak) * 100)
    spy_wr = float(np.mean(spy_rets > 0) * 100)
    spy_gains = spy_rets[spy_rets > 0].sum()
    spy_losses = abs(spy_rets[spy_rets < 0].sum())
    spy_pf = float(spy_gains / spy_losses) if spy_losses > 0 else 999
    spy_sortino_d = spy_rets[spy_rets < 0]
    spy_dsv = np.std(spy_sortino_d) * np.sqrt(12) if len(spy_sortino_d) > 0 else 1e-6
    spy_sortino = spy_ann_ret / spy_dsv

    fprint(f"\nSPY Benchmark:")
    fprint(f"  Sharpe: {spy_sharpe:.2f}, Sortino: {spy_sortino:.2f}")
    fprint(f"  CAGR: {spy_ann_ret*100:.1f}%, Max DD: {spy_dd:.1f}%")
    fprint(f"  WR: {spy_wr:.1f}%, PF: {spy_pf:.2f}")

    # Correlation with SPY
    min_len = min(len(rets), len(spy_rets))
    corr = float(np.corrcoef(rets[:min_len], spy_rets[:min_len])[0, 1])
    fprint(f"\n  Strategy-SPY correlation: {corr:.3f}")

    # Feature importance
    feat_imp = dict(zip(FEAT_COLS, [int(x) for x in model.feature_importances_]))

    metrics = {
        'name': 'CTA Trend-Following v2',
        'sharpe': round(sharpe, 2),
        'sortino': round(sortino, 2),
        'cagr': round(cagr, 1),
        'maxdd': round(maxdd, 1),
        'wr': round(win_rate, 1),
        'pf': round(pf, 2),
        'calmar': round(calmar, 2),
        'n_months': n_months,
        'final_equity': round(final_equity, 2),
        'spy_correlation': round(corr, 3),
    }

    benchmark = {
        'name': 'SPY Benchmark',
        'sharpe': round(spy_sharpe, 2),
        'sortino': round(spy_sortino, 2),
        'cagr': round(spy_ann_ret * 100, 1),
        'maxdd': round(spy_dd, 1),
        'wr': round(spy_wr, 1),
        'pf': round(spy_pf, 2),
    }

    return {
        'metrics': metrics,
        'benchmark': benchmark,
        'feature_importance': feat_imp,
        'monthly_returns': rets,
        'spy_returns': spy_rets,
        'monthly_picks': monthly_picks,
    }


def run_adversarial_gates(results):
    """Run all 4 adversarial validation gates."""
    fprint(f"\n{'='*60}")
    fprint(f"ADVERSARIAL GATES")
    fprint(f"{'='*60}")

    rets = results['monthly_returns']
    spy_rets = results['spy_returns']
    gates = {}

    # === GATE 1: PERMUTATION TEST ===
    fprint("\n[Gate 1] Permutation Test (1000 shuffles)...")
    real_sharpe = results['metrics']['sharpe']
    n_better = 0
    N_PERM = 1000
    for _ in range(N_PERM):
        shuffled = np.random.permutation(rets)
        s_ann = np.mean(shuffled) * 12
        s_vol = np.std(shuffled) * np.sqrt(12)
        s_sharpe = s_ann / s_vol if s_vol > 0 else 0
        if s_sharpe >= real_sharpe:
            n_better += 1
    p_val = n_better / N_PERM
    perm_pass = p_val < 0.05
    gates['permutation'] = {
        'p_value': round(p_val, 4),
        'pass': str(perm_pass),
        'real_sharpe': real_sharpe,
    }
    fprint(f"  Real Sharpe: {real_sharpe:.2f}, p-value: {p_val:.4f} -> {'PASS' if perm_pass else 'FAIL'}")

    # === GATE 2: REGIME R1 TEST ===
    fprint("\n[Gate 2] Regime R1 Test (bull vs bear months)...")
    min_len = min(len(rets), len(spy_rets))
    r = rets[:min_len]
    s = spy_rets[:min_len]

    bull_mask = s > 0
    bear_mask = s <= 0

    bull_r = r[bull_mask]
    bear_r = r[bear_mask]

    bull_sharpe = (np.mean(bull_r) * 12) / (np.std(bull_r) * np.sqrt(12)) if len(bull_r) > 3 else 0
    bear_sharpe = (np.mean(bear_r) * 12) / (np.std(bear_r) * np.sqrt(12)) if len(bear_r) > 3 else 0

    max_s = max(abs(bull_sharpe), abs(bear_sharpe))
    gap = abs(bull_sharpe - bear_sharpe) / max_s if max_s > 0 else 0
    regime_pass = gap <= 0.50

    gates['regime_r1'] = {
        'bull_sharpe': round(bull_sharpe, 2),
        'bear_sharpe': round(bear_sharpe, 2),
        'gap': round(gap, 3),
        'pass': str(regime_pass),
        'bull_months': int(bull_mask.sum()),
        'bear_months': int(bear_mask.sum()),
    }
    fprint(f"  Bull Sharpe: {bull_sharpe:.2f} ({int(bull_mask.sum())} months)")
    fprint(f"  Bear Sharpe: {bear_sharpe:.2f} ({int(bear_mask.sum())} months)")
    fprint(f"  Gap: {gap:.3f} -> {'PASS' if regime_pass else 'FAIL'}")

    # === GATE 3: SUB-PERIOD STABILITY ===
    fprint("\n[Gate 3] Sub-period Stability (H1 vs H2)...")
    mid = len(rets) // 2
    h1 = rets[:mid]
    h2 = rets[mid:]

    h1_sharpe = (np.mean(h1) * 12) / (np.std(h1) * np.sqrt(12)) if np.std(h1) > 0 else 0
    h2_sharpe = (np.mean(h2) * 12) / (np.std(h2) * np.sqrt(12)) if np.std(h2) > 0 else 0

    sub_pass = h1_sharpe > 0 and h2_sharpe > 0 and min(h1_sharpe, h2_sharpe) > 0.5

    gates['sub_period'] = {
        'h1_sharpe': round(h1_sharpe, 2),
        'h2_sharpe': round(h2_sharpe, 2),
        'h1_months': len(h1),
        'h2_months': len(h2),
        'pass': str(sub_pass),
    }
    fprint(f"  H1 Sharpe: {h1_sharpe:.2f} ({len(h1)} months)")
    fprint(f"  H2 Sharpe: {h2_sharpe:.2f} ({len(h2)} months)")
    fprint(f"  -> {'PASS' if sub_pass else 'FAIL'}")

    # === GATE 4: OUTLIER SENSITIVITY ===
    fprint("\n[Gate 4] Outlier Sensitivity (remove top 3 months)...")
    sorted_rets = np.sort(rets)
    trimmed = sorted_rets[:-3]  # remove top 3

    trim_ann = np.mean(trimmed) * 12
    trim_vol = np.std(trimmed) * np.sqrt(12)
    trim_sharpe = trim_ann / trim_vol if trim_vol > 0 else 0

    outlier_pass = trim_sharpe > 0.5

    gates['outlier'] = {
        'trimmed_sharpe': round(trim_sharpe, 2),
        'n_removed': 3,
        'pass': str(outlier_pass),
    }
    fprint(f"  Original Sharpe: {real_sharpe:.2f}")
    fprint(f"  Trimmed Sharpe: {trim_sharpe:.2f} (top 3 months removed)")
    fprint(f"  -> {'PASS' if outlier_pass else 'FAIL'}")

    # Summary
    n_pass = sum(1 for g in gates.values() if g['pass'] == 'True')
    fprint(f"\n{'='*60}")
    fprint(f"GATES: {n_pass}/4 PASS")
    for name, g in gates.items():
        fprint(f"  {name}: {'PASS' if g['pass'] == 'True' else 'FAIL'}")
    fprint(f"{'='*60}")

    return gates


def analyze_diversification(results):
    """Analyze how this strategy diversifies a portfolio."""
    fprint(f"\n{'='*60}")
    fprint(f"DIVERSIFICATION ANALYSIS")
    fprint(f"{'='*60}")

    rets = results['monthly_returns']
    spy_rets = results['spy_returns']
    min_len = min(len(rets), len(spy_rets))
    r = rets[:min_len]
    s = spy_rets[:min_len]

    # Correlation
    corr = np.corrcoef(r, s)[0, 1]
    fprint(f"Strategy-SPY correlation: {corr:.3f}")

    # Crisis alpha: performance during worst SPY months
    spy_sorted_idx = np.argsort(s)
    worst_10pct = spy_sorted_idx[:max(1, int(len(s) * 0.1))]
    crisis_ret = np.mean(r[worst_10pct])
    fprint(f"Avg return during worst 10% SPY months: {crisis_ret*100:+.2f}%")

    # Combined portfolio (60% SPY + 40% strategy)
    combo_60_40 = 0.6 * s + 0.4 * r
    combo_ann = np.mean(combo_60_40) * 12
    combo_vol = np.std(combo_60_40) * np.sqrt(12)
    combo_sharpe = combo_ann / combo_vol if combo_vol > 0 else 0

    spy_ann = np.mean(s) * 12
    spy_vol = np.std(s) * np.sqrt(12)
    spy_only_sharpe = spy_ann / spy_vol if spy_vol > 0 else 0

    fprint(f"\n60% SPY + 40% Strategy:")
    fprint(f"  Combined Sharpe: {combo_sharpe:.2f} (vs SPY-only: {spy_only_sharpe:.2f})")
    fprint(f"  Sharpe improvement: {(combo_sharpe/spy_only_sharpe - 1)*100:+.1f}%")

    # Equal weight combination
    combo_eq = 0.5 * s + 0.5 * r
    eq_ann = np.mean(combo_eq) * 12
    eq_vol = np.std(combo_eq) * np.sqrt(12)
    eq_sharpe = eq_ann / eq_vol if eq_vol > 0 else 0
    fprint(f"\n50/50 SPY + Strategy:")
    fprint(f"  Combined Sharpe: {eq_sharpe:.2f}")

    # Max drawdown of combined vs standalone
    spy_cum = np.cumprod(1 + s)
    spy_peak = np.maximum.accumulate(spy_cum)
    spy_maxdd = float(np.min((spy_cum - spy_peak) / spy_peak) * 100)

    combo_cum = np.cumprod(1 + combo_60_40)
    combo_peak = np.maximum.accumulate(combo_cum)
    combo_maxdd = float(np.min((combo_cum - combo_peak) / combo_peak) * 100)

    fprint(f"\nMax Drawdown comparison:")
    fprint(f"  SPY only: {spy_maxdd:.1f}%")
    fprint(f"  60/40 combo: {combo_maxdd:.1f}%")
    fprint(f"  DD reduction: {(1 - combo_maxdd/spy_maxdd)*100:.0f}%")

    return {
        'spy_correlation': round(corr, 3),
        'crisis_alpha_avg': round(crisis_ret * 100, 2),
        'combo_60_40_sharpe': round(combo_sharpe, 2),
        'combo_50_50_sharpe': round(eq_sharpe, 2),
        'spy_only_sharpe': round(spy_only_sharpe, 2),
        'spy_maxdd': round(spy_maxdd, 1),
        'combo_maxdd': round(combo_maxdd, 1),
    }


def main():
    CAPITAL_INITIAL_VAL = 100000

    # MLflow tracking
    mlflow_run = None
    if MLFLOW_OK:
        try:
            mlflow.set_experiment("CTA-Trend-Following-v2")
            mlflow_run = mlflow.start_run(run_name=f"cta_trend_v2_{datetime.now():%Y%m%d_%H%M}")
        except:
            pass

    # Run backtest
    results = run_backtest()
    if results is None:
        fprint("Backtest failed")
        return

    # Run adversarial gates
    gates = run_adversarial_gates(results)

    # Diversification analysis
    div_analysis = analyze_diversification(results)

    # Most selected assets
    pick_counts = {}
    for m in results['monthly_picks']:
        for p in m['picks']:
            pick_counts[p] = pick_counts.get(p, 0) + 1
    most_selected = dict(sorted(pick_counts.items(), key=lambda x: x[1], reverse=True))

    fprint(f"\nMost frequently selected assets:")
    for t, cnt in list(most_selected.items())[:10]:
        fprint(f"  {t}: {cnt} times")

    # Save results
    output = {
        'strategy': 'CTA Trend-Following v2',
        'run_date': str(datetime.now()),
        'universe': UNIVERSE,
        'top_k': TOP_K,
        'method': '252d/21d sliding walk-forward LightGBM',
        'note': 'Diversified trend-following across equities, bonds, commodities, currencies, real estate',
        'metrics': results['metrics'],
        'benchmark': results['benchmark'],
        'gates': gates,
        'diversification': div_analysis,
        'feature_importance': results['feature_importance'],
        'most_selected': most_selected,
        'monthly_picks': results['monthly_picks'],
    }

    with open(RESULTS_PATH, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    fprint(f"\nResults saved to {RESULTS_PATH}")

    # MLflow logging
    if mlflow_run:
        try:
            for k, v in results['metrics'].items():
                if isinstance(v, (int, float)):
                    mlflow.log_metric(k, v)
            for gate_name, gate_data in gates.items():
                for k, v in gate_data.items():
                    if isinstance(v, (int, float)):
                        mlflow.log_metric(f"gate_{gate_name}_{k}", v)
            for k, v in div_analysis.items():
                if isinstance(v, (int, float)):
                    mlflow.log_metric(f"div_{k}", v)
            mlflow.log_params({
                'universe': ','.join(UNIVERSE),
                'top_k': TOP_K,
                'train_days': 252,
                'oot_days': 21,
                'cost_bps': 10,
            })
            n_pass = sum(1 for g in gates.values() if g['pass'] == 'True')
            mlflow.log_metric('gates_passed', n_pass)
            mlflow.end_run()
            fprint("MLflow run logged.")
        except Exception as e:
            fprint(f"MLflow logging failed: {e}")

    n_pass = sum(1 for g in gates.values() if g['pass'] == 'True')
    fprint(f"\n{'='*60}")
    fprint(f"FINAL VERDICT: {n_pass}/4 gates pass")
    fprint(f"Sharpe: {results['metrics']['sharpe']}, Sortino: {results['metrics']['sortino']}")
    fprint(f"SPY correlation: {results['metrics']['spy_correlation']}")
    fprint(f"{'='*60}")


if __name__ == '__main__':
    main()
