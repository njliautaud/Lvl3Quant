#!/usr/bin/env python3
"""
ML Risk-Budgeted Portfolio — Dynamic Allocation Using Vol Prediction
=====================================================================
INSIGHT: ML works for continuous sizing (vol targeting R²=0.65) but fails
at discrete prediction (sector rotation, factor timing all fail perm tests).

This strategy exploits that insight by:
  1. Using our proven ML vol model to predict RISK (not direction)
  2. Allocating a fixed risk budget across uncorrelated return streams
  3. Dynamic rebalancing: when predicted vol is high → reduce risky assets,
     increase defensive; when low → increase growth allocation

Return streams (uncorrelated sources):
  A) Equity momentum: UPRO/SPY via v4.4 signal (proven Sharpe ~1.0)
  B) Bond carry: TLT when yield curve steep, SHY when flat/inverted
  C) Gold/commodity: GLD as inflation hedge + crisis alpha
  D) Low-vol: USMV/XLU for stable income in all regimes

Risk budget: target portfolio vol = 12% annualized
  - Each stream gets risk budget proportional to its recent Sharpe
  - ML vol model adjusts total risk exposure (leverage up/down)
  - Monthly rebalance with 5bps transaction costs

Walk-forward: SLIDING 252d (HC #0). Fixed $100K, NO DCA (HC #713).
Full adversarial: permutation 100x, sub-period 4-block, outlier, R1 regime (HC #705).

Output: /home/jupiter/Lvl3Quant/output/ml_risk_budget/
"""

import os
import json
import time
import warnings
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor

warnings.filterwarnings('ignore')
np.random.seed(42)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
INITIAL_CAPITAL  = 100_000
TARGET_VOL       = 0.12        # 12% annualized portfolio vol target
TRAIN_WINDOW     = 252         # sliding window (HC #0)
REBAL_COST_BPS   = 5           # one-way switching cost
N_PERMUTATIONS   = 100
VIX_THRESHOLD    = 25.0
VIX_SMA_SHORT    = 20
VIX_SMA_LONG     = 200
LOOKBACK_SHARPE  = 63          # 3-month rolling Sharpe for risk budgeting

BASE   = Path("/home/jupiter/Lvl3Quant")
OUTPUT = BASE / "output" / "ml_risk_budget"
OUTPUT.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: DATA
# ─────────────────────────────────────────────────────────────────────────────
def download_data():
    print("=" * 80)
    print("STEP 1: DOWNLOADING DATA")
    print("=" * 80)

    tickers = {
        'SPY':  'SPY',   # equity benchmark
        'UPRO': 'UPRO',  # 3x leveraged S&P
        'TLT':  'TLT',   # long-term treasuries
        'SHY':  'SHY',   # short-term treasuries (risk-free proxy)
        'GLD':  'GLD',   # gold
        'USMV': 'USMV',  # low-vol ETF
        'XLU':  'XLU',   # utilities (income/defensive)
        'UUP':  'UUP',   # dollar index
        'XLF':  'XLF',   # financials (for cross-asset features)
        'HYG':  'HYG',   # high yield (credit spread proxy)
        'IEF':  'IEF',   # intermediate treasuries
    }

    # Also need VIX
    all_tickers = list(tickers.values()) + ['^VIX']

    cache = BASE / "data" / "cache" / "risk_budget_data.parquet"
    if cache.exists() and (time.time() - cache.stat().st_mtime) < 3600:
        print(f"  Using cached data: {cache}")
        df = pd.read_parquet(cache)
        print(f"  Shape: {df.shape}, range: {df.index[0].date()} → {df.index[-1].date()}")
        return df

    print("  Downloading from yfinance...")
    raw = yf.download(all_tickers, start='2010-01-01', auto_adjust=True, progress=False)

    if isinstance(raw.columns, pd.MultiIndex):
        closes = raw['Close']
    else:
        closes = raw

    # Rename VIX column
    if '^VIX' in closes.columns:
        closes = closes.rename(columns={'^VIX': 'VIX'})

    # Forward fill, drop rows with too many NaN
    closes = closes.ffill().dropna(thresh=len(closes.columns) - 2)

    cache.parent.mkdir(parents=True, exist_ok=True)
    closes.to_parquet(cache)
    print(f"  Shape: {closes.shape}, range: {closes.index[0].date()} → {closes.index[-1].date()}")
    return closes


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: FEATURES FOR VOL PREDICTION
# ─────────────────────────────────────────────────────────────────────────────
def build_features(df):
    print("\n" + "=" * 80)
    print("STEP 2: BUILDING FEATURES")
    print("=" * 80)

    feat = pd.DataFrame(index=df.index)

    spy = df['SPY']
    spy_ret = spy.pct_change()

    # Realized vol features (various windows)
    for w in [5, 10, 20, 60]:
        feat[f'spy_rvol_{w}d'] = spy_ret.rolling(w).std() * np.sqrt(252)

    # VIX features
    if 'VIX' in df.columns:
        feat['vix_level'] = df['VIX']
        feat['vix_sma20'] = df['VIX'].rolling(20).mean()
        feat['vix_sma200'] = df['VIX'].rolling(200).mean()
        feat['vix_rv_ratio'] = df['VIX'] / (feat['spy_rvol_20d'] * 100 + 1e-8)
        feat['vol_of_vol_20d'] = df['VIX'].pct_change().rolling(20).std()

    # Cross-asset features
    for ticker in ['GLD', 'TLT', 'UUP', 'XLF', 'HYG']:
        if ticker in df.columns:
            ret = df[ticker].pct_change()
            feat[f'{ticker.lower()}_mom_20d'] = df[ticker].pct_change(20)
            feat[f'{ticker.lower()}_vol_20d'] = ret.rolling(20).std() * np.sqrt(252)

    # SPY momentum and drawdown
    feat['spy_mom_20d'] = spy.pct_change(20)
    feat['spy_mom_60d'] = spy.pct_change(60)
    feat['spy_drawdown'] = spy / spy.rolling(252).max() - 1

    # SPY higher moments
    feat['spy_skew_20d'] = spy_ret.rolling(20).skew()
    feat['spy_kurt_20d'] = spy_ret.rolling(20).kurt()
    feat['spy_abs_ret_5d_avg'] = spy_ret.abs().rolling(5).mean()

    # Credit spread proxy (SPY-HYG correlation / HYG vol)
    if 'HYG' in df.columns:
        hyg_ret = df['HYG'].pct_change()
        feat['spy_hyg_corr_20d'] = spy_ret.rolling(20).corr(hyg_ret)

    # Bond-equity correlation (regime indicator)
    if 'TLT' in df.columns:
        tlt_ret = df['TLT'].pct_change()
        feat['spy_tlt_corr_20d'] = spy_ret.rolling(20).corr(tlt_ret)
        feat['spy_tlt_corr_60d'] = spy_ret.rolling(60).corr(tlt_ret)

    # Yield curve proxy (TLT/IEF ratio as slope)
    if 'TLT' in df.columns and 'IEF' in df.columns:
        feat['yield_slope'] = (df['TLT'] / df['IEF']).pct_change(20)

    feat = feat.dropna()
    print(f"  Features: {feat.shape[1]}, rows: {len(feat)}")
    return feat


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3: RETURN STREAMS DEFINITION
# ─────────────────────────────────────────────────────────────────────────────
def compute_stream_returns(df):
    """Compute daily returns for each uncorrelated return stream."""
    print("\n" + "=" * 80)
    print("STEP 3: DEFINING RETURN STREAMS")
    print("=" * 80)

    streams = pd.DataFrame(index=df.index)

    # Stream A: Equity momentum (v4.4 signal → UPRO or SPY)
    spy_ret = df['SPY'].pct_change()
    upro_ret = df['UPRO'].pct_change() if 'UPRO' in df.columns else spy_ret * 3

    vix = df['VIX'] if 'VIX' in df.columns else pd.Series(20.0, index=df.index)
    vix_sma20 = vix.rolling(VIX_SMA_SHORT).mean()
    vix_sma200 = vix.rolling(VIX_SMA_LONG).mean()
    risk_on = (vix_sma20 < vix_sma200) & (vix < VIX_THRESHOLD)

    streams['equity_momentum'] = np.where(risk_on, upro_ret, spy_ret)

    # Stream B: Bond carry (TLT when yield curve steep, SHY when flat)
    tlt_ret = df['TLT'].pct_change() if 'TLT' in df.columns else pd.Series(0, index=df.index)
    shy_ret = df['SHY'].pct_change() if 'SHY' in df.columns else pd.Series(0, index=df.index)

    if 'TLT' in df.columns and 'IEF' in df.columns:
        # Yield slope proxy: when TLT outperforms IEF → curve steepening
        slope = (df['TLT'] / df['IEF']).rolling(60).mean()
        slope_z = (slope - slope.rolling(252).mean()) / (slope.rolling(252).std() + 1e-8)
        bond_risk_on = slope_z > -0.5  # steep or normal curve
        streams['bond_carry'] = np.where(bond_risk_on, tlt_ret, shy_ret)
    else:
        streams['bond_carry'] = tlt_ret

    # Stream C: Gold (always hold as diversifier — crisis alpha)
    gld_ret = df['GLD'].pct_change() if 'GLD' in df.columns else pd.Series(0, index=df.index)
    streams['gold'] = gld_ret

    # Stream D: Low vol / defensive (USMV or XLU)
    usmv_ret = df['USMV'].pct_change() if 'USMV' in df.columns else spy_ret
    streams['low_vol'] = usmv_ret

    streams = streams.dropna()

    # Print correlation matrix
    print("\n  Stream correlation matrix:")
    corr = streams.corr()
    for i, col in enumerate(corr.columns):
        vals = [f"{corr.iloc[i, j]:+.2f}" for j in range(len(corr.columns))]
        print(f"    {col:20s}: {' '.join(vals)}")

    return streams


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4: ML VOL PREDICTION (WALK-FORWARD)
# ─────────────────────────────────────────────────────────────────────────────
def predict_vol_walkforward(features, spy_ret_series):
    """Walk-forward GBM vol prediction. Returns predicted vol series."""
    print("\n" + "=" * 80)
    print("STEP 4: WALK-FORWARD VOL PREDICTION")
    print("=" * 80)

    # Target: realized vol over next 5 days
    target = spy_ret_series.rolling(5).std().shift(-5) * np.sqrt(252)

    # Align
    common_idx = features.index.intersection(target.dropna().index)
    X = features.loc[common_idx]
    y = target.loc[common_idx]

    pred_vol = pd.Series(np.nan, index=X.index)

    n = len(X)
    n_folds = 0

    for i in range(TRAIN_WINDOW, n):
        train_idx = slice(i - TRAIN_WINDOW, i)
        X_train = X.iloc[train_idx]
        y_train = y.iloc[train_idx]

        if y_train.isna().sum() > TRAIN_WINDOW * 0.1:
            continue

        X_train_clean = X_train[~y_train.isna()]
        y_train_clean = y_train[~y_train.isna()]

        if len(X_train_clean) < 100:
            continue

        model = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42
        )
        model.fit(X_train_clean, y_train_clean)

        pred_vol.iloc[i] = model.predict(X.iloc[[i]])[0]
        n_folds += 1

    pred_vol = pred_vol.dropna()
    actual_vol = target.loc[pred_vol.index].dropna()
    common = pred_vol.index.intersection(actual_vol.index)

    if len(common) > 50:
        r2 = np.corrcoef(pred_vol[common], actual_vol[common])[0, 1] ** 2
        corr = np.corrcoef(pred_vol[common], actual_vol[common])[0, 1]
        print(f"  Walk-forward folds: {n_folds}")
        print(f"  Prediction R²: {r2:.3f}, Corr: {corr:.3f}")

    return pred_vol


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5: RISK-BUDGETED PORTFOLIO CONSTRUCTION
# ─────────────────────────────────────────────────────────────────────────────
def build_risk_budget_portfolio(streams, pred_vol, features):
    """
    Allocate risk budget across streams using:
    1. Rolling Sharpe of each stream → determines relative weight
    2. ML vol prediction → determines total risk exposure
    """
    print("\n" + "=" * 80)
    print("STEP 5: RISK-BUDGETED PORTFOLIO")
    print("=" * 80)

    common_idx = streams.index.intersection(pred_vol.index)
    streams_aligned = streams.loc[common_idx]
    pred_vol_aligned = pred_vol.loc[common_idx]

    n_streams = streams_aligned.shape[1]
    stream_names = streams_aligned.columns.tolist()

    # Equal risk budget as baseline
    equal_weight = 1.0 / n_streams

    portfolio_returns = pd.Series(0.0, index=common_idx)
    weights_history = pd.DataFrame(0.0, index=common_idx, columns=stream_names)

    for i in range(LOOKBACK_SHARPE + 1, len(common_idx)):
        idx = common_idx[i]

        # 1. Compute rolling Sharpe for each stream
        lookback = streams_aligned.iloc[max(0, i - LOOKBACK_SHARPE):i]

        sharpes = {}
        vols = {}
        for col in stream_names:
            stream_ret = lookback[col]
            mu = stream_ret.mean() * 252
            sigma = stream_ret.std() * np.sqrt(252)
            sharpes[col] = mu / (sigma + 1e-8)
            vols[col] = sigma

        # 2. Risk budget: proportional to max(Sharpe, 0.1) — floor to prevent zero allocation
        raw_budgets = {k: max(v, 0.1) for k, v in sharpes.items()}
        total_budget = sum(raw_budgets.values())
        risk_weights = {k: v / total_budget for k, v in raw_budgets.items()}

        # 3. Convert risk weights to capital weights using inverse vol
        cap_weights = {}
        for col in stream_names:
            vol = vols[col] if vols[col] > 0.01 else 0.20  # floor vol at 1%
            # Risk weight / vol = capital weight (risk parity logic)
            cap_weights[col] = risk_weights[col] / vol

        # Normalize to sum to 1.0
        total_cap = sum(cap_weights.values())
        cap_weights = {k: v / total_cap for k, v in cap_weights.items()}

        # 4. ML vol scaling: adjust total exposure based on predicted vol
        pred_v = pred_vol_aligned.iloc[i] if i < len(pred_vol_aligned) else 0.15
        vol_scale = TARGET_VOL / max(pred_v, 0.05)  # target / predicted
        vol_scale = np.clip(vol_scale, 0.3, 1.5)     # cap leverage

        # 5. Final weights
        for col in stream_names:
            w = cap_weights[col] * vol_scale
            weights_history.loc[idx, col] = w

        # Portfolio return (with rebalancing cost approximation)
        port_ret = 0.0
        for col in stream_names:
            port_ret += weights_history.loc[idx, col] * streams_aligned.loc[idx, col]

        # Subtract rebalancing cost (approximate: cost proportional to weight change)
        if i > LOOKBACK_SHARPE + 1:
            prev_idx = common_idx[i - 1]
            turnover = (weights_history.loc[idx] - weights_history.loc[prev_idx]).abs().sum()
            port_ret -= turnover * REBAL_COST_BPS / 10000

        portfolio_returns.loc[idx] = port_ret

    portfolio_returns = portfolio_returns.iloc[LOOKBACK_SHARPE + 1:]
    weights_history = weights_history.iloc[LOOKBACK_SHARPE + 1:]

    print(f"  Portfolio days: {len(portfolio_returns)}")
    print(f"  Mean weights:")
    for col in stream_names:
        print(f"    {col:20s}: {weights_history[col].mean():.3f} "
              f"(min={weights_history[col].min():.3f}, max={weights_history[col].max():.3f})")

    return portfolio_returns, weights_history


# ─────────────────────────────────────────────────────────────────────────────
# STEP 6: BENCHMARKS
# ─────────────────────────────────────────────────────────────────────────────
def compute_benchmarks(df, portfolio_returns):
    """Compute benchmark returns for comparison."""
    print("\n" + "=" * 80)
    print("STEP 6: BENCHMARKS")
    print("=" * 80)

    idx = portfolio_returns.index
    benchmarks = {}

    # SPY buy & hold
    spy_ret = df['SPY'].pct_change().loc[idx]
    benchmarks['SPY B&H'] = spy_ret

    # v4.4 pure
    vix = df['VIX'] if 'VIX' in df.columns else pd.Series(20.0, index=df.index)
    vix_sma20 = vix.rolling(VIX_SMA_SHORT).mean()
    vix_sma200 = vix.rolling(VIX_SMA_LONG).mean()
    risk_on = (vix_sma20 < vix_sma200) & (vix < VIX_THRESHOLD)

    upro_ret = df['UPRO'].pct_change() if 'UPRO' in df.columns else df['SPY'].pct_change() * 3
    v44_ret = np.where(risk_on.loc[idx], upro_ret.loc[idx], spy_ret)
    benchmarks['v4.4 pure'] = pd.Series(v44_ret, index=idx)

    # Equal weight buy & hold (SPY/TLT/GLD/USMV each 25%)
    ew_ret = pd.Series(0.0, index=idx)
    for ticker, weight in [('SPY', 0.25), ('TLT', 0.25), ('GLD', 0.25), ('USMV', 0.25)]:
        if ticker in df.columns:
            ew_ret += df[ticker].pct_change().loc[idx] * weight
    benchmarks['Equal Weight B&H'] = ew_ret

    # 60/40 SPY/TLT
    bm_6040 = df['SPY'].pct_change().loc[idx] * 0.6 + df['TLT'].pct_change().loc[idx] * 0.4
    benchmarks['60/40'] = bm_6040

    return benchmarks


# ─────────────────────────────────────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────────────────────────────────────
def compute_metrics(returns, name="Strategy"):
    """Compute risk-adjusted metrics."""
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

    # Profit factor
    gains = r[r > 0].sum()
    losses = abs(r[r < 0].sum())
    pf = gains / (losses + 1e-8)

    # Win rate (daily)
    wr = (r > 0).mean()

    return {
        'name': name,
        'sharpe': round(sharpe, 3),
        'sortino': round(sortino, 3),
        'cagr': round(cagr * 100, 1),
        'max_dd': round(max_dd * 100, 1),
        'calmar': round(calmar, 3),
        'total_return': round(total_return * 100, 1),
        'profit_factor': round(pf, 3),
        'win_rate': round(wr * 100, 1),
        'annual_vol': round(sigma * 100, 1),
        'years': round(years, 1),
    }


# ─────────────────────────────────────────────────────────────────────────────
# ADVERSARIAL VALIDATION (HC #705)
# ─────────────────────────────────────────────────────────────────────────────
def run_adversarial(portfolio_returns, streams, pred_vol, features):
    """Full adversarial validation suite."""
    print("\n" + "=" * 80)
    print("ADVERSARIAL VALIDATION")
    print("=" * 80)

    results = {}

    # 1. PERMUTATION TEST (HC #659 R3)
    print("\n  [1/4] Permutation test (100 shuffles)...")
    real_sharpe = compute_metrics(portfolio_returns, "real")['sharpe']
    perm_sharpes = []

    for trial in range(N_PERMUTATIONS):
        # Shuffle the vol predictions → randomize the risk scaling
        shuffled_vol = pred_vol.copy()
        shuffled_vol.values[:] = np.random.permutation(shuffled_vol.values)

        # Also shuffle stream allocation signals (rolling Sharpe)
        perm_ret = pd.Series(0.0, index=portfolio_returns.index)
        n_streams = streams.shape[1]

        for i, idx in enumerate(portfolio_returns.index):
            if idx in streams.index:
                # Random weights (Dirichlet)
                rng_weights = np.random.dirichlet(np.ones(n_streams))
                # Random vol scale
                vol_scale = np.clip(TARGET_VOL / max(np.random.choice(shuffled_vol.values), 0.05), 0.3, 1.5)
                stream_rets = streams.loc[idx].values
                perm_ret.iloc[i] = np.sum(rng_weights * stream_rets * vol_scale)

        perm_m = compute_metrics(perm_ret, f"perm_{trial}")
        if perm_m:
            perm_sharpes.append(perm_m['sharpe'])

    perm_p = np.mean([s >= real_sharpe for s in perm_sharpes]) if perm_sharpes else 1.0
    perm_pass = perm_p < 0.05
    print(f"    Real Sharpe: {real_sharpe:.3f}")
    print(f"    Perm mean Sharpe: {np.mean(perm_sharpes):.3f} ± {np.std(perm_sharpes):.3f}")
    print(f"    p-value: {perm_p:.3f} → {'PASS' if perm_pass else 'FAIL'}")
    results['permutation'] = {'p_value': perm_p, 'pass': perm_pass,
                               'real_sharpe': real_sharpe, 'perm_mean': round(np.mean(perm_sharpes), 3)}

    # 2. SUB-PERIOD CONSISTENCY
    print("\n  [2/4] Sub-period consistency (4 blocks)...")
    n = len(portfolio_returns)
    block_size = n // 4
    block_sharpes = []

    for b in range(4):
        start = b * block_size
        end = (b + 1) * block_size if b < 3 else n
        block_ret = portfolio_returns.iloc[start:end]
        m = compute_metrics(block_ret, f"block_{b}")
        if m:
            block_sharpes.append(m['sharpe'])
            print(f"    Block {b+1}: Sharpe {m['sharpe']:.3f}, CAGR {m['cagr']:.1f}%")

    if block_sharpes and np.mean(block_sharpes) != 0:
        cv = np.std(block_sharpes) / abs(np.mean(block_sharpes))
    else:
        cv = 999
    sub_pass = cv < 0.50
    print(f"    CV: {cv:.3f} → {'PASS' if sub_pass else 'FAIL'}")
    results['sub_period'] = {'block_sharpes': block_sharpes, 'cv': round(cv, 3), 'pass': sub_pass}

    # 3. OUTLIER ROBUSTNESS
    print("\n  [3/4] Outlier robustness...")
    p95 = portfolio_returns.quantile(0.95)
    p05 = portfolio_returns.quantile(0.05)
    trimmed = portfolio_returns[(portfolio_returns > p05) & (portfolio_returns < p95)]
    m_full = compute_metrics(portfolio_returns, "full")
    m_trimmed = compute_metrics(trimmed, "trimmed")

    if m_full and m_trimmed and m_full['sharpe'] != 0:
        degradation = (m_full['sharpe'] - m_trimmed['sharpe']) / abs(m_full['sharpe'])
    else:
        degradation = 999

    outlier_pass = abs(degradation) < 0.30
    print(f"    Full Sharpe: {m_full['sharpe']:.3f}, Trimmed: {m_trimmed['sharpe']:.3f}")
    print(f"    Degradation: {degradation:.1%} → {'PASS' if outlier_pass else 'FAIL'}")
    results['outlier'] = {'degradation': round(degradation, 3), 'pass': outlier_pass}

    # 4. R1 REGIME TEST (HC #428)
    print("\n  [4/4] R1 regime test...")
    # Classify days by SPY daily return (green = up, red = down)
    spy_daily = portfolio_returns.index.map(
        lambda x: 'green' if x in features.index and
                  features.loc[x, 'spy_mom_20d'] > 0 else 'red'
    )

    green_ret = portfolio_returns[spy_daily == 'green']
    red_ret = portfolio_returns[spy_daily == 'red']

    m_green = compute_metrics(green_ret, "green")
    m_red = compute_metrics(red_ret, "red")

    if m_green and m_red:
        s_g = m_green['sharpe']
        s_r = m_red['sharpe']
        gap = abs(s_g - s_r) / max(abs(s_g), abs(s_r), 0.01)
        r1_pass = gap < 0.50
        print(f"    Green Sharpe: {s_g:.3f} ({len(green_ret)} days)")
        print(f"    Red Sharpe: {s_r:.3f} ({len(red_ret)} days)")
        print(f"    Gap: {gap:.3f} → {'PASS' if r1_pass else 'FAIL'}")
        results['r1_regime'] = {'green_sharpe': s_g, 'red_sharpe': s_r,
                                 'gap': round(gap, 3), 'pass': r1_pass}
    else:
        r1_pass = False
        results['r1_regime'] = {'pass': False, 'note': 'insufficient data'}

    # Summary
    gates_passed = sum([
        results.get('permutation', {}).get('pass', False),
        results.get('sub_period', {}).get('pass', False),
        results.get('outlier', {}).get('pass', False),
        results.get('r1_regime', {}).get('pass', False),
    ])

    results['summary'] = {
        'gates_passed': gates_passed,
        'total_gates': 4,
        'verdict': 'PASS' if gates_passed >= 3 else 'FAIL'
    }

    print(f"\n  ADVERSARIAL SUMMARY: {gates_passed}/4 gates passed → {results['summary']['verdict']}")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    t0 = time.time()
    print("=" * 80)
    print("ML RISK-BUDGETED PORTFOLIO")
    print("Dynamic allocation using vol prediction + risk parity")
    print("=" * 80)

    # Step 1: Data
    df = download_data()

    # Step 2: Features
    features = build_features(df)

    # Step 3: Return streams
    streams = compute_stream_returns(df)

    # Step 4: Vol prediction
    spy_ret = df['SPY'].pct_change()
    pred_vol = predict_vol_walkforward(features, spy_ret)

    # Step 5: Risk-budgeted portfolio
    portfolio_returns, weights_history = build_risk_budget_portfolio(streams, pred_vol, features)

    # Step 6: Benchmarks
    benchmarks = compute_benchmarks(df, portfolio_returns)

    # Metrics comparison
    print("\n" + "=" * 80)
    print("RESULTS COMPARISON")
    print("=" * 80)

    all_results = {}

    # Portfolio
    m = compute_metrics(portfolio_returns, "ML Risk Budget")
    all_results['portfolio'] = m
    print(f"\n  {'Strategy':<25} {'Sharpe':>8} {'Sortino':>8} {'CAGR':>8} {'MaxDD':>8} {'Calmar':>8}")
    print(f"  {'-'*65}")
    print(f"  {'ML Risk Budget':<25} {m['sharpe']:>8.3f} {m['sortino']:>8.3f} {m['cagr']:>7.1f}% {m['max_dd']:>7.1f}% {m['calmar']:>8.3f}")

    # Benchmarks
    for name, bm_ret in benchmarks.items():
        bm = compute_metrics(bm_ret, name)
        all_results[name] = bm
        print(f"  {name:<25} {bm['sharpe']:>8.3f} {bm['sortino']:>8.3f} {bm['cagr']:>7.1f}% {bm['max_dd']:>7.1f}% {bm['calmar']:>8.3f}")

    # Step 7: Adversarial validation
    adversarial = run_adversarial(portfolio_returns, streams, pred_vol, features)
    all_results['adversarial'] = adversarial

    # Save results
    output = {
        'strategy': 'ML Risk-Budgeted Portfolio',
        'description': 'Dynamic risk allocation across 4 uncorrelated streams using ML vol prediction',
        'metrics': all_results,
        'parameters': {
            'target_vol': TARGET_VOL,
            'train_window': TRAIN_WINDOW,
            'lookback_sharpe': LOOKBACK_SHARPE,
            'rebal_cost_bps': REBAL_COST_BPS,
            'n_permutations': N_PERMUTATIONS,
        },
        'streams': streams.columns.tolist(),
        'runtime_seconds': round(time.time() - t0, 1),
    }

    with open(OUTPUT / 'results.json', 'w') as f:
        json.dump(output, f, indent=2, default=str)

    # Save equity curve plot
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))

    # Equity curves
    ax = axes[0]
    cum = (1 + portfolio_returns).cumprod() * INITIAL_CAPITAL
    ax.plot(cum.index, cum.values, label='ML Risk Budget', linewidth=2, color='blue')
    for name, bm_ret in benchmarks.items():
        bm_cum = (1 + bm_ret).cumprod() * INITIAL_CAPITAL
        ax.plot(bm_cum.index, bm_cum.values, label=name, alpha=0.7)
    ax.set_title('Equity Curves (Fixed $100K, No DCA)')
    ax.legend()
    ax.set_ylabel('Portfolio Value ($)')
    ax.grid(True, alpha=0.3)

    # Drawdown
    ax = axes[1]
    running_max = cum.cummax()
    dd = cum / running_max - 1
    ax.fill_between(dd.index, dd.values, 0, alpha=0.5, color='red')
    ax.set_title('Drawdown')
    ax.set_ylabel('Drawdown %')
    ax.grid(True, alpha=0.3)

    # Weights over time
    ax = axes[2]
    for col in weights_history.columns:
        ax.plot(weights_history.index, weights_history[col].rolling(20).mean(), label=col, alpha=0.8)
    ax.set_title('Stream Weights (20d smoothed)')
    ax.legend()
    ax.set_ylabel('Weight')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT / 'equity_curve.png', dpi=150)
    plt.close()

    elapsed = time.time() - t0
    print(f"\n{'='*80}")
    print(f"COMPLETE in {elapsed:.0f}s")
    print(f"Output: {OUTPUT}")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
