#!/usr/bin/env python3
"""
Stock Prediction for Asymmetric Upside — Options-Enhanced Growth
================================================================
Uses ML to predict which mega-cap stocks will have large upside moves,
then sizes positions with options for asymmetric payoff (risk $1 to make $5+).

Approach:
1. ML GBM predicts 21-day forward returns for mega-caps
2. Top-confidence longs get option overlays (long calls / risk reversals)
3. Walk-forward validation, FIFO regime-agnostic

Key insight from prior research (HC #709):
- Stock prediction v3 had precision 43% at 80% threshold
- Long stock: Sharpe 1.13, passes R1
- OTM calls: 23.7% CAGR but FAILS R1
- Risk reversal: 10.2% CAGR, Sharpe 1.01, MaxDD -7.2%
- "Stock prediction signal is the REAL edge; options add cost/complexity"

THIS SCRIPT: Tests if we can get asymmetric upside by combining:
- ML stock picking (which stocks go up)
- Position sizing based on confidence (higher conf = more size)
- Synthetic option overlay pricing (for the OTM call upside)
- Strict regime check and adversarial validation
"""
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import roc_auc_score
from datetime import datetime
from scipy.stats import norm
import os, json, warnings
warnings.filterwarnings('ignore')

OUTPUT = '/home/jupiter/Lvl3Quant/output/stock_prediction_asymmetric'
os.makedirs(OUTPUT, exist_ok=True)
np.random.seed(42)

print("=" * 70)
print("STOCK PREDICTION — ASYMMETRIC UPSIDE VIA OPTIONS OVERLAY")
print("ML stock picker + confidence-sized positions + call overlay")
print("=" * 70)

# Universe: mega-caps with liquid options
UNIVERSE = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA',
            'JPM', 'BAC', 'GS', 'V', 'MA',
            'JNJ', 'UNH', 'PFE', 'LLY',
            'XOM', 'CVX',
            'HD', 'WMT', 'COST', 'TGT',
            'DIS', 'NFLX', 'CRM', 'ORCL']

print(f"\nDownloading {len(UNIVERSE)} mega-cap stocks + SPY/VIX...")
all_tickers = UNIVERSE + ['SPY', '^VIX']
data = yf.download(all_tickers, start='2015-01-01', progress=False)
if hasattr(data.index, 'tz') and data.index.tz is not None:
    data.index = data.index.tz_localize(None)

close = data['Close'] if isinstance(data.columns, pd.MultiIndex) else data
close = close.ffill()
volume = data['Volume'] if isinstance(data.columns, pd.MultiIndex) else None
high = data['High'] if isinstance(data.columns, pd.MultiIndex) else None
low = data['Low'] if isinstance(data.columns, pd.MultiIndex) else None

vix = close['^VIX'] if '^VIX' in close.columns else None
spy = close['SPY']
close = close.drop(['^VIX', 'SPY'], axis=1, errors='ignore')

# Keep stocks with enough data
valid = close.columns[close.notna().sum() > 2000]
close = close[valid].dropna()
print(f"  {len(close)} days, {len(valid)} stocks")
print(f"  Period: {close.index[0].date()} to {close.index[-1].date()}")
print(f"  Stocks: {list(valid)}")

# ============================================================
# FEATURE ENGINEERING (per stock)
# ============================================================
def build_stock_features(prices, spy_prices, vix_data, vol_data=None):
    """Rich feature set for stock prediction."""
    ret = prices.pct_change()
    spy_ret = spy_prices.pct_change()

    feats = pd.DataFrame(index=prices.index)

    # Momentum
    for h in [5, 10, 21, 42, 63, 126]:
        feats[f'mom_{h}d'] = prices.pct_change(h)

    # Relative strength vs SPY
    for h in [5, 21, 63]:
        feats[f'rs_spy_{h}d'] = prices.pct_change(h) - spy_prices.pct_change(h)

    # Volatility
    for h in [10, 21, 63]:
        feats[f'vol_{h}d'] = ret.rolling(h).std() * np.sqrt(252)

    # Vol ratio
    feats['vol_ratio'] = feats['vol_10d'] / (feats['vol_63d'] + 1e-8)

    # Drawdown
    rolling_max = prices.rolling(252).max()
    feats['dd_from_peak'] = prices / rolling_max - 1

    # Mean reversion
    feats['dist_from_ma50'] = prices / prices.rolling(50).mean() - 1
    feats['dist_from_ma200'] = prices / prices.rolling(200).mean() - 1

    # RSI
    delta = ret.copy()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / (loss + 1e-8)
    feats['rsi_14'] = 100 - 100 / (1 + rs)

    # Beta
    feats['beta_63d'] = ret.rolling(63).cov(spy_ret) / (spy_ret.rolling(63).var() + 1e-8)

    # Correlation with market
    feats['corr_spy_63d'] = ret.rolling(63).corr(spy_ret)

    # VIX features
    if vix_data is not None:
        feats['vix'] = vix_data
        feats['vix_change_5d'] = vix_data.pct_change(5)
        feats['vix_rank'] = vix_data.rolling(252).rank(pct=True)

    # Market regime
    feats['spy_mom_21d'] = spy_prices.pct_change(21)
    feats['spy_vol_21d'] = spy_ret.rolling(21).std() * np.sqrt(252)

    # Return distribution
    feats['skew_21d'] = ret.rolling(21).skew()
    feats['kurt_21d'] = ret.rolling(21).kurt()

    # Volume features (if available)
    if vol_data is not None:
        vol_ma = vol_data.rolling(20).mean()
        feats['vol_ratio_20d'] = vol_data / (vol_ma + 1)

    return feats


# ============================================================
# WALK-FORWARD STOCK PREDICTION
# ============================================================
TRAIN_WINDOW = 252  # 1 year
HOLD_PERIOD = 21    # 3-week hold
TOP_K = 5           # Top 5 stocks per period
THRESHOLD = 0.6     # Confidence threshold

print(f"\n{'='*70}")
print(f"WALK-FORWARD STOCK PREDICTION")
print(f"Train: {TRAIN_WINDOW}d, Hold: {HOLD_PERIOD}d, Top-K: {TOP_K}")
print(f"{'='*70}")

# Build features for all stocks
all_features = {}
for ticker in valid:
    vol_col = volume[ticker] if volume is not None and ticker in volume.columns else None
    feats = build_stock_features(close[ticker], spy, vix, vol_col)
    all_features[ticker] = feats

# Walk-forward with monthly steps
step = HOLD_PERIOD
all_periods = []
dates = close.index

for t in range(TRAIN_WINDOW, len(dates) - HOLD_PERIOD, step):
    train_start = t - TRAIN_WINDOW
    train_end = t
    oot_start = t
    oot_end = min(t + HOLD_PERIOD, len(dates))

    # Collect training data across all stocks
    X_train_all = []
    y_train_all = []

    for ticker in valid:
        feats = all_features[ticker]
        prices = close[ticker]

        # Forward return target
        fwd_ret = prices.pct_change(HOLD_PERIOD).shift(-HOLD_PERIOD)

        # Training window
        train_feats = feats.iloc[train_start:train_end]
        train_target = fwd_ret.iloc[train_start:train_end]

        valid_mask = train_feats.notna().all(axis=1) & train_target.notna()
        if valid_mask.sum() < 50:
            continue

        X_train_all.append(train_feats[valid_mask].values)
        y_train_all.append(train_target[valid_mask].values)

    if not X_train_all:
        continue

    X_train = np.vstack(X_train_all)
    y_train = np.concatenate(y_train_all)

    # Train regressor (predict return magnitude)
    model_reg = GradientBoostingRegressor(
        n_estimators=100, max_depth=4, learning_rate=0.1,
        subsample=0.8, random_state=42
    )
    model_reg.fit(X_train, y_train)

    # Also train classifier (predict direction)
    y_class = (y_train > 0).astype(int)
    model_cls = GradientBoostingClassifier(
        n_estimators=100, max_depth=4, learning_rate=0.1,
        subsample=0.8, random_state=42
    )
    model_cls.fit(X_train, y_class)

    # Predict for each stock at the decision point
    predictions = {}
    for ticker in valid:
        feats = all_features[ticker]
        if oot_start >= len(feats):
            continue
        feat_row = feats.iloc[oot_start:oot_start+1]
        if feat_row.isna().any(axis=1).values[0]:
            continue

        pred_ret = model_reg.predict(feat_row.values)[0]
        pred_prob = model_cls.predict_proba(feat_row.values)[0][1]  # Prob of going up

        # Actual forward return
        if oot_end < len(close):
            actual_ret = (close[ticker].iloc[oot_end-1] / close[ticker].iloc[oot_start]) - 1
        else:
            actual_ret = np.nan

        predictions[ticker] = {
            'pred_ret': pred_ret,
            'pred_prob': pred_prob,
            'actual_ret': actual_ret,
            'confidence': pred_prob * abs(pred_ret)  # Combined confidence
        }

    if not predictions:
        continue

    # Select top-K stocks by combined confidence (bullish only)
    bullish = {k: v for k, v in predictions.items() if v['pred_prob'] > THRESHOLD}
    sorted_picks = sorted(bullish.items(), key=lambda x: x[1]['confidence'], reverse=True)
    top_picks = sorted_picks[:TOP_K]

    # Record period results
    period_date = dates[oot_start]
    spy_fwd = (spy.iloc[oot_end-1] / spy.iloc[oot_start]) - 1 if oot_end < len(spy) else 0
    vix_level = vix.iloc[oot_start] if vix is not None else 15

    # Portfolio return (equal-weight top picks)
    if top_picks:
        pick_rets = [p[1]['actual_ret'] for p in top_picks if not np.isnan(p[1]['actual_ret'])]
        port_ret = np.mean(pick_rets) if pick_rets else 0

        # Confidence-weighted version
        total_conf = sum(p[1]['confidence'] for p in top_picks)
        if total_conf > 0:
            conf_rets = [p[1]['actual_ret'] * p[1]['confidence'] / total_conf
                        for p in top_picks if not np.isnan(p[1]['actual_ret'])]
            conf_port_ret = sum(conf_rets)
        else:
            conf_port_ret = port_ret

        # OPTIONS OVERLAY: For top confidence picks, simulate OTM call overlay
        # Buy 5% OTM calls (1-month), risk premium, capture upside asymmetry
        option_ret = 0
        n_option = 0
        for ticker, pred in top_picks:
            if pred['pred_prob'] > 0.7 and not np.isnan(pred['actual_ret']):
                # Simplified: 5% OTM call, premium = BS estimate
                stock_vol = all_features[ticker]['vol_21d'].iloc[oot_start] if oot_start < len(all_features[ticker]) else 0.25
                if np.isnan(stock_vol) or stock_vol <= 0:
                    stock_vol = 0.25

                # BS call price approximation for 5% OTM, 21 days
                T = 21/252
                d1 = (np.log(1/1.05) + (0.02 + stock_vol**2/2)*T) / (stock_vol*np.sqrt(T))
                d2 = d1 - stock_vol*np.sqrt(T)
                call_price = norm.cdf(d1) - 1.05*np.exp(-0.02*T)*norm.cdf(d2)
                call_price = max(call_price, 0.005)  # Floor at 0.5% of stock

                # Payoff: max(actual_ret - 0.05, 0) / call_price - 1
                if pred['actual_ret'] > 0.05:
                    call_payoff = (pred['actual_ret'] - 0.05) / call_price - 1
                else:
                    call_payoff = -1  # Total loss of premium

                option_ret += call_payoff
                n_option += 1

        avg_option_ret = option_ret / n_option if n_option > 0 else 0

        all_periods.append({
            'date': period_date,
            'port_ret': port_ret,
            'conf_port_ret': conf_port_ret,
            'option_overlay_ret': avg_option_ret,
            'spy_ret': spy_fwd,
            'n_picks': len(top_picks),
            'n_options': n_option,
            'vix': vix_level,
            'picks': [p[0] for p in top_picks],
            'avg_confidence': np.mean([p[1]['pred_prob'] for p in top_picks])
        })
    else:
        # No picks above threshold — flat
        all_periods.append({
            'date': period_date,
            'port_ret': 0,
            'conf_port_ret': 0,
            'option_overlay_ret': 0,
            'spy_ret': spy_fwd,
            'n_picks': 0,
            'n_options': 0,
            'vix': vix_level,
            'picks': [],
            'avg_confidence': 0
        })

    if len(all_periods) % 20 == 0:
        cum_ret = np.prod([1 + p['port_ret'] for p in all_periods]) - 1
        print(f"  Period {len(all_periods)}: cum_ret={cum_ret:+.1%}, "
              f"picks={len(top_picks)}, date={period_date.date()}")

# ============================================================
# RESULTS
# ============================================================
print(f"\n{'='*70}")
print("RESULTS")
print(f"{'='*70}")

df = pd.DataFrame(all_periods)
df['date'] = pd.to_datetime(df['date'])
df = df.set_index('date')

# Strategy variants
strategies = {
    'ML Stock Pick (EW)': df['port_ret'],
    'ML Stock Pick (Conf-Weighted)': df['conf_port_ret'],
    'ML Stock + OTM Call Overlay': df['port_ret'] * 0.7 + df['option_overlay_ret'] * 0.3,
    'SPY Buy & Hold': df['spy_ret']
}

print(f"\nBacktest: {df.index[0].date()} to {df.index[-1].date()}, {len(df)} periods ({HOLD_PERIOD}d each)")
print(f"\n{'Strategy':<35} {'CAGR':>7} {'Sharpe':>7} {'Sortino':>8} {'MaxDD':>7} {'WR':>5} {'PF':>5}")
print("-" * 80)

results = {}
for name, rets in strategies.items():
    rets = rets.dropna()
    cum = np.cumprod(1 + rets)
    total = cum.iloc[-1] - 1
    years = len(rets) * HOLD_PERIOD / 252
    cagr = (1 + total) ** (1/years) - 1 if years > 0 else 0
    vol = rets.std() * np.sqrt(252/HOLD_PERIOD)
    sharpe = cagr / vol if vol > 0 else 0

    peak = np.maximum.accumulate(cum)
    max_dd = (cum / peak - 1).min()

    downside = rets[rets < 0].std() * np.sqrt(252/HOLD_PERIOD) if (rets < 0).sum() > 0 else 1e-8
    sortino = cagr / downside

    wr = (rets > 0).mean()
    avg_win = rets[rets > 0].mean() if (rets > 0).sum() > 0 else 0
    avg_loss = abs(rets[rets < 0].mean()) if (rets < 0).sum() > 0 else 1e-8
    pf = (avg_win * (rets > 0).sum()) / (avg_loss * (rets < 0).sum()) if (rets < 0).sum() > 0 else 999

    print(f"{name:<35} {cagr:>+6.1%} {sharpe:>7.2f} {sortino:>8.2f} {max_dd:>6.1%} {wr:>4.0%} {pf:>5.2f}")

    results[name] = {
        'cagr': float(cagr), 'sharpe': float(sharpe), 'sortino': float(sortino),
        'max_dd': float(max_dd), 'wr': float(wr), 'pf': float(pf),
        'total_return': float(total), 'years': float(years)
    }

# ============================================================
# REGIME TEST (R1)
# ============================================================
print(f"\n{'='*70}")
print("R1 REGIME TEST")
print(f"{'='*70}")

# Green: SPY up over trailing 21d, Red: SPY down
spy_trailing = spy.pct_change(21)
df['regime'] = 'green'
for i, date in enumerate(df.index):
    loc = spy_trailing.index.get_indexer([date], method='nearest')[0]
    if spy_trailing.iloc[loc] < -0.02:
        df.loc[date, 'regime'] = 'red'

green = df[df['regime'] == 'green']['port_ret']
red = df[df['regime'] == 'red']['port_ret']

green_sharpe = green.mean() / (green.std() + 1e-8) * np.sqrt(252/HOLD_PERIOD)
red_sharpe = red.mean() / (red.std() + 1e-8) * np.sqrt(252/HOLD_PERIOD)
regime_gap = abs(green_sharpe - red_sharpe) / max(abs(green_sharpe), abs(red_sharpe), 0.01)

print(f"  Green periods: {len(green)}, Sharpe: {green_sharpe:.3f}")
print(f"  Red periods:   {len(red)}, Sharpe: {red_sharpe:.3f}")
print(f"  Regime gap:    {regime_gap:.3f} (threshold 0.50)")
print(f"  {'✅ R1 PASS' if regime_gap < 0.50 else '❌ R1 FAIL'}")

# ============================================================
# ASYMMETRY ANALYSIS
# ============================================================
print(f"\n{'='*70}")
print("ASYMMETRY ANALYSIS")
print(f"{'='*70}")

port_rets = df['port_ret'].dropna()
avg_win = port_rets[port_rets > 0].mean()
avg_loss = abs(port_rets[port_rets < 0].mean())
asymmetry = avg_win / avg_loss if avg_loss > 0 else 0

# Tail analysis
p90_win = port_rets[port_rets > 0].quantile(0.9)
p10_loss = abs(port_rets[port_rets < 0].quantile(0.1))
tail_asymmetry = p90_win / p10_loss if p10_loss > 0 else 0

# Options overlay asymmetry
opt_rets = df['option_overlay_ret'].dropna()
opt_rets_nonzero = opt_rets[opt_rets != 0]
if len(opt_rets_nonzero) > 10:
    opt_avg_win = opt_rets_nonzero[opt_rets_nonzero > 0].mean()
    opt_avg_loss = abs(opt_rets_nonzero[opt_rets_nonzero < 0].mean())
    opt_asymmetry = opt_avg_win / opt_avg_loss if opt_avg_loss > 0 else 0
    opt_wr = (opt_rets_nonzero > 0).mean()
    print(f"  Options overlay: WR={opt_wr:.0%}, Avg Win={opt_avg_win:.1%}, "
          f"Avg Loss={opt_avg_loss:.1%}, Asymmetry={opt_asymmetry:.1f}x")

print(f"\n  Stock picks: Avg Win={avg_win:.2%}, Avg Loss={port_rets[port_rets<0].mean():.2%}")
print(f"  Win/Loss asymmetry: {asymmetry:.2f}x")
print(f"  Tail asymmetry (P90 win / P10 loss): {tail_asymmetry:.2f}x")
print(f"  Best period:  {port_rets.max():+.1%}")
print(f"  Worst period: {port_rets.min():+.1%}")

# ============================================================
# PERMUTATION TEST
# ============================================================
print(f"\nPermutation test (50 shuffles)...")
real_sharpe = results['ML Stock Pick (EW)']['sharpe']
perm_sharpes = []
for p in range(50):
    shuffled = np.random.permutation(port_rets.values)
    cum = np.cumprod(1 + shuffled)
    total = cum[-1] - 1
    years = len(shuffled) * HOLD_PERIOD / 252
    cagr = (1 + total) ** (1/years) - 1 if years > 0 else 0
    vol = np.std(shuffled) * np.sqrt(252/HOLD_PERIOD)
    perm_sharpes.append(cagr / vol if vol > 0 else 0)

perm_p = np.mean([ps >= real_sharpe for ps in perm_sharpes])
print(f"  p-value: {perm_p:.3f} (real Sharpe {real_sharpe:.3f} vs perm mean {np.mean(perm_sharpes):.3f})")
print(f"  {'✅ PERM PASS' if perm_p < 0.05 else '❌ PERM FAIL'}")

# ============================================================
# YEAR-BY-YEAR
# ============================================================
print(f"\n{'='*70}")
print("YEAR-BY-YEAR RETURNS")
print(f"{'='*70}")

yearly = df.groupby(df.index.year)['port_ret'].apply(lambda x: np.prod(1+x)-1)
spy_yearly = df.groupby(df.index.year)['spy_ret'].apply(lambda x: np.prod(1+x)-1)

print(f"  {'Year':<6} {'Strategy':>9} {'SPY':>9} {'Alpha':>9}")
print(f"  {'-'*35}")
for year in sorted(yearly.index):
    s = yearly[year]
    b = spy_yearly.get(year, 0)
    print(f"  {year:<6} {s:>+8.1%} {b:>+8.1%} {s-b:>+8.1%}")

# Save
results['regime'] = {
    'green_sharpe': float(green_sharpe),
    'red_sharpe': float(red_sharpe),
    'gap': float(regime_gap),
    'r1_pass': regime_gap < 0.50
}
results['asymmetry'] = {
    'win_loss_ratio': float(asymmetry),
    'tail_asymmetry': float(tail_asymmetry),
    'options_asymmetry': float(opt_asymmetry) if 'opt_asymmetry' in dir() else 0
}
results['adversarial'] = {
    'perm_p': float(perm_p),
    'perm_pass': perm_p < 0.05,
    'r1_pass': regime_gap < 0.50
}
results['year_by_year'] = {str(y): float(r) for y, r in yearly.items()}

def convert_to_serializable(obj):
    """Convert numpy types to Python native for JSON serialization."""
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(v) for v in obj]
    return obj

with open(f'{OUTPUT}/results.json', 'w') as f:
    json.dump(convert_to_serializable(results), f, indent=2)

print(f"\n{'='*70}")
print("VERDICT")
print(f"{'='*70}")
ew_sharpe = results['ML Stock Pick (EW)']['sharpe']
ew_cagr = results['ML Stock Pick (EW)']['cagr']
if ew_sharpe > 1.0 and perm_p < 0.05:
    print(f"✅ STRONG — Stock prediction works (Sharpe {ew_sharpe:.2f}, CAGR {ew_cagr:.1%})")
    print(f"   Asymmetry: {asymmetry:.1f}x win/loss ratio")
elif ew_sharpe > 0.5:
    print(f"🟡 MODERATE — Signal exists but needs enhancement (Sharpe {ew_sharpe:.2f})")
else:
    print(f"❌ WEAK — Stock prediction doesn't add enough over market")

print("\nDONE")
