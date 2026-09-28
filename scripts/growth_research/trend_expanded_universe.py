#!/usr/bin/env python3
"""
ML Trend Following v2 — EXPANDED UNIVERSE TEST
================================================
Our best strategy (Sharpe 2.90) was validated on 8 assets.
Question: Does it work on 20+ assets? More assets = more diversification = better risk-adjusted.

Universe expansion:
- Original 8: SPY, QQQ, GLD, TLT, EEM, VNQ, HYG, XLE
- Adding: IWM, DIA, SLV, USO, UUP, FXI, EWJ, EWZ, XLF, XLK, XLV, XLI, TIPS, LQD
- Total: 22 assets across equities, bonds, commodities, currencies, international

Same framework: GBM filter, walk-forward, 252d train, adversarial validation.
"""
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from datetime import datetime
import os, json, warnings
warnings.filterwarnings('ignore')

OUTPUT = '/home/jupiter/Lvl3Quant/output/trend_expanded_universe'
os.makedirs(OUTPUT, exist_ok=True)
np.random.seed(42)

print("=" * 70)
print("ML TREND v2 — EXPANDED UNIVERSE (22 ASSETS)")
print("Testing if our best strategy generalizes to a broader universe")
print("=" * 70)

# Expanded universe
tickers = [
    # Original 8
    'SPY', 'QQQ', 'GLD', 'TLT', 'EEM', 'VNQ', 'HYG', 'XLE',
    # Expansion
    'IWM', 'DIA', 'SLV', 'USO', 'UUP', 'FXI', 'EWJ', 'EWZ',
    'XLF', 'XLK', 'XLV', 'XLI', 'TIPS', 'LQD'
]

print(f"\nDownloading {len(tickers)} assets...")
df = yf.download(tickers, start='2006-01-01', progress=False)
if hasattr(df.index, 'tz') and df.index.tz is not None:
    df.index = df.index.tz_localize(None)

close = df['Close'] if isinstance(df.columns, pd.MultiIndex) else df
close = close.ffill()
# Keep assets with enough history (>3000 days)
valid = close.columns[close.notna().sum() > 3000]
close = close[valid].dropna()
print(f"  {len(close)} days, {len(valid)} assets with sufficient history")
print(f"  Period: {close.index[0].date()} to {close.index[-1].date()}")
print(f"  Assets: {list(valid)}")

# ============================================================
# FEATURE ENGINEERING (same as v2)
# ============================================================
def build_features(prices, ticker):
    """Build trend features for one asset."""
    ret = prices.pct_change()
    feats = pd.DataFrame(index=prices.index)

    # Momentum at multiple horizons
    for h in [5, 10, 21, 42, 63, 126, 252]:
        feats[f'mom_{h}d'] = prices.pct_change(h)

    # Moving average crossovers
    for fast, slow in [(10, 50), (20, 100), (50, 200)]:
        ma_fast = prices.rolling(fast).mean()
        ma_slow = prices.rolling(slow).mean()
        feats[f'ma_{fast}_{slow}'] = (ma_fast - ma_slow) / ma_slow

    # Volatility
    for h in [10, 21, 63]:
        feats[f'vol_{h}d'] = ret.rolling(h).std() * np.sqrt(252)

    # Vol ratio (current vs longer-term)
    feats['vol_ratio'] = feats['vol_10d'] / (feats['vol_63d'] + 1e-8)

    # Drawdown from peak
    rolling_max = prices.rolling(252).max()
    feats['dd_from_peak'] = prices / rolling_max - 1

    # RSI
    delta = ret.copy()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / (loss + 1e-8)
    feats['rsi_14'] = 100 - 100 / (1 + rs)

    # Breakout (price vs rolling high/low)
    feats['breakout_21'] = (prices - prices.rolling(21).min()) / (prices.rolling(21).max() - prices.rolling(21).min() + 1e-8)
    feats['breakout_63'] = (prices - prices.rolling(63).min()) / (prices.rolling(63).max() - prices.rolling(63).min() + 1e-8)

    # Trend strength (ADX-like)
    feats['trend_strength'] = abs(feats['mom_21d']) / (feats['vol_21d'] + 1e-8)

    return feats


# ============================================================
# WALK-FORWARD PER ASSET
# ============================================================
TRAIN_DAYS = 252
OOT_DAYS = 1  # Daily rolling
HOLD_PERIOD = 21  # Target: 21-day forward return

print(f"\n{'=' * 70}")
print(f"WALK-FORWARD BACKTEST")
print(f"Train: {TRAIN_DAYS}d, Hold: {HOLD_PERIOD}d, Rolling daily")
print(f"{'=' * 70}")

all_results = {}
all_signals = {}

for ticker in valid:
    prices = close[ticker]
    ret = prices.pct_change()

    # Build features
    feats = build_features(prices, ticker)

    # Target: positive 21-day forward return
    fwd_ret = prices.pct_change(HOLD_PERIOD).shift(-HOLD_PERIOD)
    target = (fwd_ret > 0).astype(int)

    # Align and drop NaN
    combined = pd.concat([feats, target.rename('target')], axis=1).dropna()
    X = combined.drop('target', axis=1)
    y = combined['target']

    if len(X) < TRAIN_DAYS + 252:  # Need enough for train + 1yr OOT
        print(f"  {ticker}: skipped (insufficient data)")
        continue

    # Walk-forward (monthly steps for speed)
    step = 21
    predictions = []
    actuals = []
    pred_dates = []

    n_folds = (len(X) - TRAIN_DAYS) // step

    for i in range(0, len(X) - TRAIN_DAYS, step):
        train_end = i + TRAIN_DAYS
        oot_end = min(train_end + step, len(X))

        if oot_end <= train_end:
            break

        X_train = X.iloc[i:train_end]
        y_train = y.iloc[i:train_end]
        X_oot = X.iloc[train_end:oot_end]
        y_oot = y.iloc[train_end:oot_end]

        # GBM — skip if single class in training
        if len(np.unique(y_train)) < 2:
            continue

        model = GradientBoostingClassifier(
            n_estimators=100, max_depth=3, learning_rate=0.1,
            subsample=0.8, random_state=42
        )
        try:
            model.fit(X_train, y_train)
        except ValueError:
            continue

        probs = model.predict_proba(X_oot)[:, 1]
        predictions.extend(probs)
        actuals.extend(y_oot.values)
        pred_dates.extend(X_oot.index)

    if len(predictions) < 100:
        continue

    predictions = np.array(predictions)
    actuals = np.array(actuals)
    pred_dates = pd.DatetimeIndex(pred_dates)

    # AUC
    auc = roc_auc_score(actuals, predictions)

    # Strategy: go long when ML says >0.5, else flat (hold cash)
    signal = pd.Series(predictions > 0.5, index=pred_dates).astype(float)
    daily_ret = ret.reindex(pred_dates).fillna(0)

    strat_ret = signal.shift(1) * daily_ret  # Next-day execution
    strat_ret = strat_ret.dropna()

    # Metrics
    if len(strat_ret) > 252:
        cum = (1 + strat_ret).cumprod()
        years = len(strat_ret) / 252
        total_ret = cum.iloc[-1] - 1
        cagr = (1 + total_ret) ** (1/years) - 1
        vol = strat_ret.std() * np.sqrt(252)
        sharpe = cagr / vol if vol > 0 else 0

        rolling_max = cum.cummax()
        max_dd = (cum / rolling_max - 1).min()

        # Buy & hold comparison
        bh_ret = daily_ret
        bh_cum = (1 + bh_ret).cumprod()
        bh_total = bh_cum.iloc[-1] - 1
        bh_cagr = (1 + bh_total) ** (1/years) - 1
        bh_vol = bh_ret.std() * np.sqrt(252)
        bh_sharpe = bh_cagr / bh_vol if bh_vol > 0 else 0

        long_pct = signal.mean()

        all_results[ticker] = {
            'auc': auc,
            'sharpe': sharpe,
            'cagr': cagr,
            'max_dd': max_dd,
            'vol': vol,
            'bh_sharpe': bh_sharpe,
            'bh_cagr': bh_cagr,
            'long_pct': long_pct,
            'n_days': len(strat_ret),
            'alpha': sharpe - bh_sharpe  # ML alpha over buy & hold
        }
        all_signals[ticker] = strat_ret

        if len(all_results) % 5 == 0 or ticker in ['SPY', 'GLD', 'TLT']:
            print(f"  {ticker}: AUC={auc:.3f}, Sharpe={sharpe:.2f} (B&H {bh_sharpe:.2f}), "
                  f"CAGR={cagr:.1%}, MaxDD={max_dd:.1%}, Long {long_pct:.0%}")

# ============================================================
# PORTFOLIO CONSTRUCTION
# ============================================================
print(f"\n{'=' * 70}")
print("PORTFOLIO RESULTS")
print(f"{'=' * 70}")

# Equal-weight all assets that have positive ML alpha
results_df = pd.DataFrame(all_results).T.sort_values('sharpe', ascending=False)
print(f"\nAll assets ranked by Sharpe:")
print(f"{'Asset':<6} {'AUC':>5} {'Sharpe':>7} {'B&H':>6} {'Alpha':>6} {'CAGR':>7} {'MaxDD':>7} {'Long%':>6}")
print("-" * 55)
for ticker, row in results_df.iterrows():
    print(f"{ticker:<6} {row['auc']:>5.3f} {row['sharpe']:>7.2f} {row['bh_sharpe']:>6.2f} "
          f"{row['alpha']:>+6.2f} {row['cagr']:>6.1%} {row['max_dd']:>6.1%} {row['long_pct']:>5.0%}")

# Equal-weight portfolio (all assets with positive alpha)
alpha_assets = results_df[results_df['alpha'] > 0].index.tolist()
print(f"\nAssets with positive ML alpha: {len(alpha_assets)}/{len(results_df)}")
print(f"  {alpha_assets}")

if len(alpha_assets) >= 3:
    # Build equal-weight portfolio
    port_signals = pd.DataFrame({t: all_signals[t] for t in alpha_assets})
    port_ret = port_signals.mean(axis=1)  # Equal weight
    port_ret = port_ret.dropna()

    cum = (1 + port_ret).cumprod()
    years = len(port_ret) / 252
    total_ret = cum.iloc[-1] - 1
    cagr = (1 + total_ret) ** (1/years) - 1
    vol = port_ret.std() * np.sqrt(252)
    sharpe = cagr / vol if vol > 0 else 0
    max_dd = (cum / cum.cummax() - 1).min()

    downside = port_ret[port_ret < 0].std() * np.sqrt(252)
    sortino = cagr / downside if downside > 0 else 0

    # Win rate
    monthly = port_ret.resample('ME').sum()
    wr = (monthly > 0).mean()

    print(f"\n{'=' * 70}")
    print(f"EQUAL-WEIGHT ML TREND PORTFOLIO ({len(alpha_assets)} assets)")
    print(f"{'=' * 70}")
    print(f"  Sharpe:  {sharpe:.3f}")
    print(f"  Sortino: {sortino:.3f}")
    print(f"  CAGR:    {cagr:.1%}")
    print(f"  Vol:     {vol:.1%}")
    print(f"  MaxDD:   {max_dd:.1%}")
    print(f"  Monthly WR: {wr:.1%}")
    print(f"  Period:  {port_ret.index[0].date()} to {port_ret.index[-1].date()}")

    # Sub-period stability
    mid = len(port_ret) // 2
    first_half = port_ret.iloc[:mid]
    second_half = port_ret.iloc[mid:]
    s1 = (first_half.mean() * 252) / (first_half.std() * np.sqrt(252))
    s2 = (second_half.mean() * 252) / (second_half.std() * np.sqrt(252))
    print(f"\n  Sub-period Sharpe: H1={s1:.2f}, H2={s2:.2f} (CV={abs(s1-s2)/max(abs(s1),abs(s2)):.2f})")

    # Compare to original 8-asset version
    orig_assets = [t for t in ['SPY','QQQ','GLD','TLT','EEM','VNQ','HYG','XLE'] if t in all_signals]
    if len(orig_assets) >= 4:
        orig_ret = pd.DataFrame({t: all_signals[t] for t in orig_assets}).mean(axis=1).dropna()
        orig_cum = (1 + orig_ret).cumprod()
        orig_years = len(orig_ret) / 252
        orig_sharpe = ((orig_cum.iloc[-1] ** (1/orig_years) - 1)) / (orig_ret.std() * np.sqrt(252))
        print(f"\n  Original 8-asset Sharpe: {orig_sharpe:.3f}")
        print(f"  Expanded {len(alpha_assets)}-asset Sharpe: {sharpe:.3f}")
        print(f"  Improvement: {sharpe - orig_sharpe:+.3f}")

# Save
results_dict = {
    'n_assets_tested': len(results_df),
    'n_assets_alpha_positive': len(alpha_assets),
    'alpha_assets': alpha_assets,
    'portfolio_sharpe': float(sharpe) if 'sharpe' in dir() else 0,
    'portfolio_sortino': float(sortino) if 'sortino' in dir() else 0,
    'portfolio_cagr': float(cagr) if 'cagr' in dir() else 0,
    'portfolio_maxdd': float(max_dd) if 'max_dd' in dir() else 0,
    'per_asset': {k: {kk: float(vv) for kk, vv in v.items()} for k, v in all_results.items()}
}

with open(f'{OUTPUT}/results.json', 'w') as f:
    json.dump(results_dict, f, indent=2)

print(f"\n{'=' * 70}")
print("VERDICT")
print(f"{'=' * 70}")
if sharpe > 2.5:
    print("✅ EXPANDED UNIVERSE WORKS — more assets = better diversification")
elif sharpe > 1.5:
    print("🟡 MODERATE — some expansion helps but diminishing returns")
else:
    print("❌ EXPANSION HURTS — stick with original curated 8-asset universe")

print("\nDONE")
