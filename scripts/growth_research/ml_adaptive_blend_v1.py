#!/usr/bin/env python3
"""
ML Adaptive Blend v1 — Dynamic VIX/Trend Allocation
=====================================================
HYPOTHESIS: ML regime detector can tell us WHEN VIX leverage is safe (bull)
vs when we should shelter in regime-agnostic trend following (bear/uncertain).

Strategy:
  - Component A: VIX Leverage (thresholds 12/16/22 → UPRO/SPY/SHY). Sharpe 3.9 but R1 FAIL.
  - Component B: ML Trend Following v2 (8-asset CTA). Sharpe 2.9, R1 PASS.
  - ML Allocator: GBM predicts P(favorable for leverage) using cross-asset features.
    - High confidence → deploy VIX leverage (target: capture the 3.9 Sharpe)
    - Low confidence → deploy ML Trend (target: preserve capital in adverse regimes)
    - This should give us R1 PASS (because trend following protects in bear) + higher
      Sharpe than pure trend (because VIX leverage dominates in bull).

Walk-forward: 504d sliding train, daily test (HC #0).
Fixed $100K, NO DCA (HC #713).
Full adversarial: permutation 100x, sub-period 4-block, outlier, R1 regime.

SUCCESS CRITERIA: Sharpe > 3.0 AND R1 PASS AND perm PASS.
"""

import json
import time
import warnings
from pathlib import Path
from functools import partial

print = partial(print, flush=True)
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from scipy import stats

np.random.seed(42)

# ─────────────────────────────────────────────────────────────────────────────
INITIAL_CAPITAL   = 100_000
TRAIN_WINDOW      = 504          # 2 years sliding
ML_THRESHOLD_HIGH = 0.65         # high confidence → VIX leverage
ML_THRESHOLD_LOW  = 0.40         # low confidence → pure trend
N_PERMUTATIONS    = 100
REBAL_COST_BPS    = 10

BASE   = Path("/home/jupiter/Lvl3Quant")
OUTPUT = BASE / "output" / "ml_adaptive_blend"
OUTPUT.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# DATA
# ─────────────────────────────────────────────────────────────────────────────

UNIVERSE_TREND = ['SPY', 'TLT', 'GLD', 'UUP', 'EEM', 'VNQ', 'HYG', 'XLE']
UNIVERSE_VIX   = ['SPY', 'UPRO', 'SHY']
ALL_TICKERS    = list(set(UNIVERSE_TREND + UNIVERSE_VIX + ['QQQ', 'IWM', 'TLT', 'IEF', 'USO']))

def download_data():
    """Download all needed data with caching."""
    cache = BASE / "data" / "cache" / "adaptive_blend_data.parquet"
    cache.parent.mkdir(parents=True, exist_ok=True)

    if cache.exists() and (time.time() - cache.stat().st_mtime) < 3600:
        print(f"  Using cached data: {cache}")
        return pd.read_parquet(cache)

    print("  Downloading from yfinance...")
    tickers = list(set(ALL_TICKERS + ['^VIX']))
    data = yf.download(tickers, start='2006-01-01', end='2026-07-18',
                       auto_adjust=True, progress=False)

    closes = data['Close'].copy()
    closes.columns = [c.replace('^', '') for c in closes.columns]
    closes = closes.ffill().dropna(how='all')
    closes.to_parquet(cache)
    print(f"  Data: {len(closes)} days, {closes.shape[1]} assets")
    return closes


# ─────────────────────────────────────────────────────────────────────────────
# COMPONENT A: VIX LEVERAGE STRATEGY
# ─────────────────────────────────────────────────────────────────────────────

def vix_leverage_signal(vix_level):
    """
    VIX threshold → leverage decision.
    VIX < 12: 3x (UPRO)
    VIX 12-16: 3x (UPRO)
    VIX 16-22: 1x (SPY)
    VIX 22-30: cash (SHY)
    VIX > 30: cash (SHY)
    """
    if vix_level < 16:
        return 'UPRO'
    elif vix_level < 22:
        return 'SPY'
    else:
        return 'SHY'


def run_vix_leverage(closes):
    """Run standalone VIX leverage strategy, return daily returns."""
    vix = closes['VIX'].copy()
    spy_ret = closes['SPY'].pct_change()
    upro_ret = closes.get('UPRO', closes['SPY'] * 3).pct_change()  # fallback: 3x SPY
    shy_ret = closes['SHY'].pct_change() if 'SHY' in closes.columns else pd.Series(0.0001, index=closes.index)

    # If UPRO not available for early dates, simulate as 3x SPY - 0.01%/day (expense)
    if 'UPRO' not in closes.columns:
        upro_ret = spy_ret * 3 - 0.0001

    daily_ret = pd.Series(0.0, index=closes.index)
    for i in range(1, len(closes)):
        prev_vix = vix.iloc[i-1] if not np.isnan(vix.iloc[i-1]) else 20
        signal = vix_leverage_signal(prev_vix)
        if signal == 'UPRO':
            daily_ret.iloc[i] = upro_ret.iloc[i] if not np.isnan(upro_ret.iloc[i]) else 0
        elif signal == 'SPY':
            daily_ret.iloc[i] = spy_ret.iloc[i] if not np.isnan(spy_ret.iloc[i]) else 0
        else:
            daily_ret.iloc[i] = shy_ret.iloc[i] if not np.isnan(shy_ret.iloc[i]) else 0.0001

    return daily_ret


# ─────────────────────────────────────────────────────────────────────────────
# COMPONENT B: ML TREND FOLLOWING (simplified inline version)
# ─────────────────────────────────────────────────────────────────────────────

def run_ml_trend(closes):
    """
    Simplified ML Trend Following.
    Uses GBM to predict trend continuation for 8-asset universe.
    Returns daily portfolio returns.
    """
    ma_short, ma_long = 20, 100
    ml_threshold = 0.55
    target_vol = 0.10

    assets = [t for t in UNIVERSE_TREND if t in closes.columns]
    returns = closes[assets].pct_change()

    # Build signals
    signals = {}
    for asset in assets:
        ma_s = closes[asset].rolling(ma_short).mean()
        ma_l = closes[asset].rolling(ma_long).mean()
        signals[asset] = (ma_s > ma_l).astype(int) * 2 - 1  # +1 long, -1 flat/short

    signals_df = pd.DataFrame(signals)

    # Features for ML
    def build_features(closes, i, lookback=60):
        feats = {}
        for asset in assets:
            price = closes[asset].iloc[max(0,i-lookback):i]
            if len(price) < lookback:
                return None
            ret = price.pct_change().dropna()
            feats[f'{asset}_mom20'] = price.iloc[-1] / price.iloc[-20] - 1 if len(price) > 20 else 0
            feats[f'{asset}_vol20'] = ret.iloc[-20:].std() if len(ret) >= 20 else 0
            feats[f'{asset}_trend'] = 1 if price.iloc[-1] > price.rolling(50).mean().iloc[-1] else 0

        if 'VIX' in closes.columns:
            vix = closes['VIX'].iloc[max(0,i-lookback):i]
            feats['vix_level'] = vix.iloc[-1] if len(vix) > 0 else 20
            feats['vix_ma20'] = vix.rolling(20).mean().iloc[-1] if len(vix) >= 20 else 20
            feats['vix_pctile'] = stats.percentileofscore(vix.dropna(), vix.iloc[-1]) if len(vix.dropna()) > 10 else 50

        feats['n_trending'] = sum(1 for a in assets if signals_df[a].iloc[i] > 0)
        return feats

    # Walk-forward ML filtering
    portfolio_ret = pd.Series(0.0, index=closes.index)
    train_size = 252

    for i in range(train_size + ma_long, len(closes)):
        # Retrain every 21 days
        if (i - train_size - ma_long) % 21 == 0:
            # Build training data
            X_train, y_train = [], []
            for j in range(max(ma_long+60, i-train_size), i):
                feats = build_features(closes, j)
                if feats is None:
                    continue
                # Label: was the average trend signal profitable over next 5 days?
                fwd_ret = returns.iloc[j:j+5].mean().mean() if j+5 < len(returns) else 0
                X_train.append(feats)
                y_train.append(1 if fwd_ret > 0 else 0)

            if len(X_train) < 50:
                continue

            X_train = pd.DataFrame(X_train).fillna(0)
            y_train = np.array(y_train)

            model = GradientBoostingClassifier(
                n_estimators=50, max_depth=3, learning_rate=0.1,
                subsample=0.8, random_state=42
            )
            model.fit(X_train, y_train)

        # Predict
        feats = build_features(closes, i)
        if feats is None or 'model' not in dir():
            continue

        try:
            X_pred = pd.DataFrame([feats]).fillna(0)
            # Ensure same columns
            for col in X_train.columns:
                if col not in X_pred.columns:
                    X_pred[col] = 0
            X_pred = X_pred[X_train.columns]

            prob = model.predict_proba(X_pred)[0][1]
        except:
            prob = 0.5

        # Position sizing with vol targeting
        if prob > ml_threshold:
            # Go long the trending assets
            trending = [a for a in assets if signals_df[a].iloc[i] > 0]
            if trending:
                # Equal weight trending assets, vol-targeted
                n = len(trending)
                daily_ret = returns[trending].iloc[i].mean() if i < len(returns) else 0
                portfolio_ret.iloc[i] = daily_ret * min(1.0, target_vol / (returns[trending].iloc[max(0,i-20):i].std().mean() + 1e-6))
        # else: stay in cash (0 return)

    return portfolio_ret


# ─────────────────────────────────────────────────────────────────────────────
# ML REGIME ALLOCATOR — decides blend between VIX leverage vs Trend
# ─────────────────────────────────────────────────────────────────────────────

def build_regime_features(closes, i, lookback=120):
    """Cross-asset features to predict whether VIX leverage will outperform."""
    if i < lookback:
        return None

    feats = {}

    # VIX features
    if 'VIX' in closes.columns:
        vix = closes['VIX'].iloc[max(0,i-lookback):i].dropna()
        if len(vix) < 20:
            return None
        feats['vix_level'] = vix.iloc[-1]
        feats['vix_sma20'] = vix.rolling(20).mean().iloc[-1]
        feats['vix_sma50'] = vix.rolling(50).mean().iloc[-1] if len(vix) >= 50 else vix.mean()
        feats['vix_percentile'] = stats.percentileofscore(vix, vix.iloc[-1])
        feats['vix_slope'] = (vix.iloc[-1] - vix.iloc[-5]) / 5 if len(vix) > 5 else 0
        feats['vix_vol'] = vix.pct_change().std()

    # SPY features
    if 'SPY' in closes.columns:
        spy = closes['SPY'].iloc[max(0,i-lookback):i].dropna()
        if len(spy) < 50:
            return None
        feats['spy_above_sma50'] = 1 if spy.iloc[-1] > spy.rolling(50).mean().iloc[-1] else 0
        feats['spy_above_sma200'] = 1 if len(spy) >= 200 and spy.iloc[-1] > spy.rolling(200).mean().iloc[-1] else 0
        feats['spy_mom20'] = spy.iloc[-1] / spy.iloc[-20] - 1
        feats['spy_mom60'] = spy.iloc[-1] / spy.iloc[-60] - 1 if len(spy) >= 60 else 0
        feats['spy_vol20'] = spy.pct_change().iloc[-20:].std()
        feats['spy_drawdown'] = spy.iloc[-1] / spy.rolling(50).max().iloc[-1] - 1

    # Credit spread proxy (HYG vs TLT)
    if 'HYG' in closes.columns and 'TLT' in closes.columns:
        hyg = closes['HYG'].iloc[max(0,i-lookback):i].dropna()
        tlt = closes['TLT'].iloc[max(0,i-lookback):i].dropna()
        if len(hyg) >= 20 and len(tlt) >= 20:
            spread = (hyg / tlt).dropna()
            if len(spread) >= 20:
                feats['credit_spread_z'] = (spread.iloc[-1] - spread.mean()) / (spread.std() + 1e-8)
                feats['credit_mom20'] = spread.iloc[-1] / spread.iloc[-20] - 1

    # Breadth proxy (IWM vs SPY)
    if 'IWM' in closes.columns and 'SPY' in closes.columns:
        iwm = closes['IWM'].iloc[max(0,i-lookback):i].dropna()
        spy = closes['SPY'].iloc[max(0,i-lookback):i].dropna()
        if len(iwm) >= 20 and len(spy) >= 20:
            breadth = (iwm / spy).dropna()
            if len(breadth) >= 20:
                feats['breadth_mom20'] = breadth.iloc[-1] / breadth.iloc[-20] - 1

    # Gold as fear gauge
    if 'GLD' in closes.columns:
        gld = closes['GLD'].iloc[max(0,i-lookback):i].dropna()
        if len(gld) >= 20:
            feats['gold_mom20'] = gld.iloc[-1] / gld.iloc[-20] - 1

    # Trend alignment count
    n_bullish = 0
    for asset in UNIVERSE_TREND:
        if asset in closes.columns:
            px = closes[asset].iloc[max(0,i-lookback):i].dropna()
            if len(px) >= 50:
                if px.iloc[-1] > px.rolling(50).mean().iloc[-1]:
                    n_bullish += 1
    feats['n_bullish_assets'] = n_bullish
    feats['pct_bullish'] = n_bullish / len(UNIVERSE_TREND)

    return feats


def run_adaptive_blend(closes, vix_ret, trend_ret, threshold_high=0.65, threshold_low=0.40):
    """
    ML allocator: predicts whether VIX leverage will be favorable.
    High confidence → VIX leverage. Low → Trend following. Middle → 50/50.
    """
    train_window = TRAIN_WINDOW

    portfolio_ret = pd.Series(0.0, index=closes.index)
    allocations = pd.DataFrame(index=closes.index, columns=['vix_wt', 'trend_wt', 'prob'])

    model = None
    last_train = 0

    for i in range(train_window + 120, len(closes)):
        # Retrain every 21 days
        if model is None or (i - last_train) >= 21:
            X_train, y_train = [], []

            for j in range(max(120, i - train_window), i - 21):
                feats = build_regime_features(closes, j)
                if feats is None:
                    continue
                # Label: VIX leverage outperforms trend over next 21 days
                vix_fwd = vix_ret.iloc[j:j+21].sum()
                trend_fwd = trend_ret.iloc[j:j+21].sum()
                y = 1 if vix_fwd > trend_fwd else 0
                X_train.append(feats)
                y_train.append(y)

            if len(X_train) < 60:
                # Not enough data, default 50/50
                portfolio_ret.iloc[i] = 0.5 * vix_ret.iloc[i] + 0.5 * trend_ret.iloc[i]
                continue

            X_train_df = pd.DataFrame(X_train).fillna(0)
            y_train_arr = np.array(y_train)

            model = GradientBoostingClassifier(
                n_estimators=80, max_depth=3, learning_rate=0.05,
                subsample=0.8, min_samples_leaf=10, random_state=42
            )
            model.fit(X_train_df, y_train_arr)
            feature_cols = X_train_df.columns.tolist()
            last_train = i

        # Predict
        feats = build_regime_features(closes, i)
        if feats is None:
            portfolio_ret.iloc[i] = 0.5 * vix_ret.iloc[i] + 0.5 * trend_ret.iloc[i]
            continue

        X_pred = pd.DataFrame([feats]).fillna(0)
        for col in feature_cols:
            if col not in X_pred.columns:
                X_pred[col] = 0
        X_pred = X_pred[feature_cols]

        try:
            prob = model.predict_proba(X_pred)[0][1]  # P(VIX leverage better)
        except:
            prob = 0.5

        # Allocation decision
        if prob >= threshold_high:
            vix_wt, trend_wt = 0.80, 0.20
        elif prob <= threshold_low:
            vix_wt, trend_wt = 0.20, 0.80
        else:
            # Linear interpolation
            frac = (prob - threshold_low) / (threshold_high - threshold_low)
            vix_wt = 0.20 + frac * 0.60
            trend_wt = 1.0 - vix_wt

        portfolio_ret.iloc[i] = vix_wt * vix_ret.iloc[i] + trend_wt * trend_ret.iloc[i]
        allocations.loc[closes.index[i]] = [vix_wt, trend_wt, prob]

    return portfolio_ret, allocations


# ─────────────────────────────────────────────────────────────────────────────
# ADVERSARIAL VALIDATION
# ─────────────────────────────────────────────────────────────────────────────

def compute_sharpe(returns, rf=0.0):
    """Annualized Sharpe from daily returns."""
    excess = returns - rf/252
    if excess.std() == 0:
        return 0
    return excess.mean() / excess.std() * np.sqrt(252)


def compute_metrics(returns, name="Strategy"):
    """Full metrics suite."""
    if returns.std() == 0 or len(returns) == 0:
        return {'name': name, 'sharpe': 0, 'sortino': 0, 'cagr': 0, 'max_dd': 0, 'calmar': 0}

    sharpe = compute_sharpe(returns)
    downside = returns[returns < 0].std()
    sortino = returns.mean() / downside * np.sqrt(252) if downside > 0 else 0

    cum = (1 + returns).cumprod()
    years = len(returns) / 252
    cagr = (cum.iloc[-1] ** (1/years) - 1) * 100 if years > 0 and cum.iloc[-1] > 0 else 0

    running_max = cum.cummax()
    drawdown = (cum - running_max) / running_max
    max_dd = drawdown.min() * 100

    calmar = cagr / abs(max_dd) if max_dd != 0 else 0

    pf_pos = returns[returns > 0].sum()
    pf_neg = abs(returns[returns < 0].sum())
    profit_factor = pf_pos / pf_neg if pf_neg > 0 else 99

    win_rate = (returns > 0).mean() * 100

    return {
        'name': name, 'sharpe': round(sharpe, 3), 'sortino': round(sortino, 3),
        'cagr': round(cagr, 1), 'max_dd': round(max_dd, 1), 'calmar': round(calmar, 3),
        'profit_factor': round(profit_factor, 3), 'win_rate': round(win_rate, 1),
        'total_return': round((cum.iloc[-1] - 1) * 100, 1),
        'annual_vol': round(returns.std() * np.sqrt(252) * 100, 1)
    }


def permutation_test(closes, vix_ret, trend_ret, real_sharpe, n_perms=100):
    """Shuffle VIX levels → re-run allocator → compare Sharpe distribution."""
    print(f"\n  Running {n_perms} permutations...")
    perm_sharpes = []

    for p in range(n_perms):
        if (p+1) % 20 == 0:
            print(f"    Perm {p+1}/{n_perms}...")

        # Shuffle VIX levels (break signal relationship)
        closes_shuffled = closes.copy()
        vix_vals = closes_shuffled['VIX'].dropna().values.copy()
        np.random.shuffle(vix_vals)
        closes_shuffled.loc[closes_shuffled['VIX'].notna(), 'VIX'] = vix_vals

        # Also shuffle SPY momentum (break trend signal)
        spy_vals = closes_shuffled['SPY'].values.copy()
        # Block shuffle (preserve autocorrelation structure)
        block_size = 21
        n_blocks = len(spy_vals) // block_size
        block_indices = np.arange(n_blocks)
        np.random.shuffle(block_indices)
        shuffled_spy = np.concatenate([spy_vals[i*block_size:(i+1)*block_size] for i in block_indices])
        closes_shuffled['SPY'] = np.nan
        closes_shuffled.iloc[:len(shuffled_spy), closes_shuffled.columns.get_loc('SPY')] = shuffled_spy
        closes_shuffled['SPY'] = closes_shuffled['SPY'].ffill()

        # Re-run allocator with shuffled signals
        perm_ret, _ = run_adaptive_blend(closes_shuffled, vix_ret, trend_ret)
        perm_sharpes.append(compute_sharpe(perm_ret.dropna()))

    perm_sharpes = np.array(perm_sharpes)
    p_value = (perm_sharpes >= real_sharpe).mean()

    print(f"  Real Sharpe: {real_sharpe:.3f}")
    print(f"  Perm mean: {np.mean(perm_sharpes):.3f} ± {np.std(perm_sharpes):.3f}")
    print(f"  P-value: {p_value:.4f}")

    return {'p_value': float(p_value), 'perm_mean': float(np.mean(perm_sharpes)),
            'perm_std': float(np.std(perm_sharpes)), 'pass': p_value < 0.05}


def sub_period_test(returns, n_blocks=4):
    """Split into n blocks, check Sharpe consistency."""
    block_size = len(returns) // n_blocks
    block_sharpes = []

    for b in range(n_blocks):
        block = returns.iloc[b*block_size:(b+1)*block_size]
        block_sharpes.append(compute_sharpe(block))

    cv = np.std(block_sharpes) / (np.mean(block_sharpes) + 1e-8)

    print(f"  Sub-period Sharpes: {[f'{s:.2f}' for s in block_sharpes]}")
    print(f"  CV: {cv:.3f} (PASS < 0.50)")

    return {'block_sharpes': [float(s) for s in block_sharpes],
            'cv': float(cv), 'pass': cv < 0.50}


def r1_regime_test(returns, closes):
    """Check if strategy works in both green (bull) and red (bear) days."""
    spy_ret = closes['SPY'].pct_change()

    # Align indices
    common = returns.index.intersection(spy_ret.index)
    returns_aligned = returns.loc[common]
    spy_aligned = spy_ret.loc[common]

    # Monthly regime (rolling 21d SPY return)
    spy_monthly = spy_aligned.rolling(21).sum()

    green_mask = spy_monthly > 0
    red_mask = spy_monthly <= 0

    green_sharpe = compute_sharpe(returns_aligned[green_mask].dropna())
    red_sharpe = compute_sharpe(returns_aligned[red_mask].dropna())

    gap = abs(green_sharpe - red_sharpe) / max(abs(green_sharpe), abs(red_sharpe), 0.01)

    print(f"  Green Sharpe: {green_sharpe:.3f}")
    print(f"  Red Sharpe: {red_sharpe:.3f}")
    print(f"  Gap: {gap:.3f} (PASS < 0.50)")

    return {'green_sharpe': float(green_sharpe), 'red_sharpe': float(red_sharpe),
            'gap': float(gap), 'pass': gap < 0.50}


def outlier_test(returns, pctile=95):
    """Remove top/bottom outlier days, check if Sharpe persists."""
    full_sharpe = compute_sharpe(returns)

    lower = np.percentile(returns.dropna(), 100 - pctile)
    upper = np.percentile(returns.dropna(), pctile)
    trimmed = returns[(returns >= lower) & (returns <= upper)]
    trimmed_sharpe = compute_sharpe(trimmed)

    degradation = (trimmed_sharpe - full_sharpe) / (abs(full_sharpe) + 1e-8)

    print(f"  Full Sharpe: {full_sharpe:.3f}")
    print(f"  Trimmed Sharpe (no top/bottom {100-pctile}%): {trimmed_sharpe:.3f}")
    print(f"  Degradation: {degradation:.1%} (PASS > -0.40)")

    return {'full_sharpe': float(full_sharpe), 'trimmed_sharpe': float(trimmed_sharpe),
            'degradation': float(degradation), 'pass': degradation > -0.40}


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    start_time = time.time()
    print("=" * 80)
    print("ML ADAPTIVE BLEND v1 — VIX Leverage + ML Trend Following")
    print("=" * 80)

    # Step 1: Data
    print("\n[1/6] DOWNLOADING DATA...")
    closes = download_data()

    # Step 2: Run component strategies
    print("\n[2/6] RUNNING COMPONENT STRATEGIES...")
    print("  Running VIX Leverage...")
    vix_ret = run_vix_leverage(closes)
    vix_metrics = compute_metrics(vix_ret.dropna(), "VIX Leverage")
    print(f"  VIX Leverage: Sharpe={vix_metrics['sharpe']}, CAGR={vix_metrics['cagr']}%")

    print("  Running ML Trend Following...")
    trend_ret = run_ml_trend(closes)
    trend_metrics = compute_metrics(trend_ret.dropna(), "ML Trend")
    print(f"  ML Trend: Sharpe={trend_metrics['sharpe']}, CAGR={trend_metrics['cagr']}%")

    # Step 3: Run ML adaptive blend
    print("\n[3/6] RUNNING ML ADAPTIVE BLEND...")
    blend_ret, allocations = run_adaptive_blend(closes, vix_ret, trend_ret)
    blend_ret_clean = blend_ret[blend_ret != 0].dropna()
    blend_metrics = compute_metrics(blend_ret_clean, "ML Adaptive Blend")
    print(f"  Blend: Sharpe={blend_metrics['sharpe']}, CAGR={blend_metrics['cagr']}%")
    print(f"         MaxDD={blend_metrics['max_dd']}%, Sortino={blend_metrics['sortino']}")

    # Also compute naive 50/50 blend for comparison
    naive_ret = 0.5 * vix_ret + 0.5 * trend_ret
    naive_metrics = compute_metrics(naive_ret.dropna(), "Naive 50/50")
    print(f"  Naive 50/50: Sharpe={naive_metrics['sharpe']}")

    # SPY benchmark
    spy_ret = closes['SPY'].pct_change().dropna()
    spy_metrics = compute_metrics(spy_ret, "SPY B&H")

    # Step 4: Adversarial validation
    print("\n[4/6] ADVERSARIAL VALIDATION...")

    print("\n  --- Permutation Test ---")
    perm_result = permutation_test(closes, vix_ret, trend_ret, blend_metrics['sharpe'], N_PERMUTATIONS)

    print("\n  --- Sub-Period Consistency ---")
    subp_result = sub_period_test(blend_ret_clean)

    print("\n  --- R1 Regime Test ---")
    r1_result = r1_regime_test(blend_ret_clean, closes)

    print("\n  --- Outlier Sensitivity ---")
    outlier_result = outlier_test(blend_ret_clean)

    # Step 5: Summary
    gates_passed = sum([perm_result['pass'], subp_result['pass'],
                       r1_result['pass'], outlier_result['pass']])

    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    print(f"\n  ML Adaptive Blend v1:")
    print(f"    Sharpe:  {blend_metrics['sharpe']}")
    print(f"    Sortino: {blend_metrics['sortino']}")
    print(f"    CAGR:    {blend_metrics['cagr']}%")
    print(f"    MaxDD:   {blend_metrics['max_dd']}%")
    print(f"    Calmar:  {blend_metrics['calmar']}")
    print(f"    PF:      {blend_metrics['profit_factor']}")
    print(f"    WR:      {blend_metrics['win_rate']}%")
    print(f"\n  Adversarial Gates: {gates_passed}/4")
    print(f"    Permutation: {'✅ PASS' if perm_result['pass'] else '❌ FAIL'} (p={perm_result['p_value']:.4f})")
    print(f"    Sub-period:  {'✅ PASS' if subp_result['pass'] else '❌ FAIL'} (CV={subp_result['cv']:.3f})")
    print(f"    R1 Regime:   {'✅ PASS' if r1_result['pass'] else '❌ FAIL'} (gap={r1_result['gap']:.3f})")
    print(f"    Outlier:     {'✅ PASS' if outlier_result['pass'] else '❌ FAIL'} (deg={outlier_result['degradation']:.1%})")

    print(f"\n  Comparison:")
    print(f"    VIX Leverage alone: Sharpe {vix_metrics['sharpe']}")
    print(f"    ML Trend alone:     Sharpe {trend_metrics['sharpe']}")
    print(f"    Naive 50/50:        Sharpe {naive_metrics['sharpe']}")
    print(f"    ML Adaptive Blend:  Sharpe {blend_metrics['sharpe']}")
    print(f"    SPY B&H:            Sharpe {spy_metrics['sharpe']}")

    verdict = "PASS" if gates_passed >= 3 and blend_metrics['sharpe'] > 3.0 else "NEEDS WORK" if gates_passed >= 2 else "FAIL"
    print(f"\n  VERDICT: {verdict} ({gates_passed}/4 gates, target: Sharpe>3.0 + R1 PASS)")

    # Step 6: Save
    print("\n[6/6] SAVING RESULTS...")

    results = {
        'strategy': 'ML Adaptive Blend v1',
        'hypothesis': 'ML regime detector allocates between VIX leverage (bull-biased, high Sharpe) and trend following (regime-agnostic)',
        'metrics': {
            'blend': blend_metrics,
            'vix_leverage': vix_metrics,
            'ml_trend': trend_metrics,
            'naive_5050': naive_metrics,
            'spy': spy_metrics
        },
        'adversarial': {
            'permutation': perm_result,
            'sub_period': subp_result,
            'r1_regime': r1_result,
            'outlier': outlier_result,
            'gates_passed': gates_passed,
            'total': 4,
            'verdict': verdict
        },
        'parameters': {
            'train_window': TRAIN_WINDOW,
            'threshold_high': ML_THRESHOLD_HIGH,
            'threshold_low': ML_THRESHOLD_LOW,
            'n_permutations': N_PERMUTATIONS
        },
        'runtime_seconds': round(time.time() - start_time, 1)
    }

    with open(OUTPUT / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)

    # Equity curve plot
    cum_blend = (1 + blend_ret).cumprod() * INITIAL_CAPITAL
    cum_vix = (1 + vix_ret).cumprod() * INITIAL_CAPITAL
    cum_trend = (1 + trend_ret).cumprod() * INITIAL_CAPITAL
    cum_spy = (1 + spy_ret).cumprod() * INITIAL_CAPITAL

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

    ax1.plot(cum_blend.index, cum_blend.values, label=f'ML Adaptive Blend (Sharpe {blend_metrics["sharpe"]})', linewidth=2)
    ax1.plot(cum_vix.index, cum_vix.values, label=f'VIX Leverage (Sharpe {vix_metrics["sharpe"]})', alpha=0.7)
    ax1.plot(cum_trend.index, cum_trend.values, label=f'ML Trend (Sharpe {trend_metrics["sharpe"]})', alpha=0.7)
    ax1.plot(cum_spy.index, cum_spy.values, label=f'SPY B&H (Sharpe {spy_metrics["sharpe"]})', alpha=0.5, color='gray')
    ax1.set_yscale('log')
    ax1.set_title('ML Adaptive Blend v1 — Equity Curves')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylabel('Portfolio Value ($)')

    # Allocation over time
    alloc_clean = allocations.dropna()
    if len(alloc_clean) > 0:
        ax2.fill_between(alloc_clean.index, 0, alloc_clean['vix_wt'].astype(float),
                        alpha=0.6, label='VIX Leverage Weight', color='green')
        ax2.fill_between(alloc_clean.index, alloc_clean['vix_wt'].astype(float), 1.0,
                        alpha=0.6, label='Trend Following Weight', color='blue')
        ax2.set_title('Dynamic Allocation (ML-Driven)')
        ax2.legend()
        ax2.set_ylabel('Weight')
        ax2.set_ylim(0, 1)
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT / 'equity_curve.png', dpi=100, bbox_inches='tight')
    plt.close()

    elapsed = time.time() - start_time
    print(f"\n  Done in {elapsed:.0f}s. Results saved to {OUTPUT}")
    print(f"\n{'=' * 80}")

    return results


if __name__ == '__main__':
    results = main()
