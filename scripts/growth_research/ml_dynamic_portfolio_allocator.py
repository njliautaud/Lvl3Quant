#!/usr/bin/env python3
"""
ML Dynamic Portfolio Allocator
=================================
Instead of finding NEW strategies, use ML to dynamically allocate across
our 12 validated strategies based on regime conditions.

Key insight: Static optimal weights (Max Sharpe) assume stationarity.
In reality, different strategies shine in different regimes:
- CTA/Trend: best in trending markets (VIX moderate, clear direction)
- Tail Risk: best in volatile/crash periods (VIX high)
- Gold/Silver: best in uncertainty (flight to safety, metals diverge)
- Yield Curve: best in rate transition periods
- Currency Carry: best in calm, risk-on environments
- Income strategies: best in low-vol sideways markets

ML learns WHEN each strategy outperforms and tilts allocation accordingly.

Walk-forward: 252d train, 21d advance, sliding window (HC #0)
Adversarial: 4-gate validation
Capital: Fixed $100K (HC #713)
"""

import numpy as np
import pandas as pd
import warnings
import json
from pathlib import Path
from datetime import datetime

warnings.filterwarnings('ignore')
import yfinance as yf
import lightgbm as lgb

OUTPUT = Path("/home/jupiter/Lvl3Quant/output/ml_dynamic_portfolio")
OUTPUT.mkdir(parents=True, exist_ok=True)

print(f"{'='*60}")
print(f"ML DYNAMIC PORTFOLIO ALLOCATOR — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
print(f"{'='*60}")

# ─── Strategy ETF Proxies ───
# Map our 12 validated strategies to tradeable ETF proxies
STRATEGIES = {
    'CTA_Trend': ['SPY', 'EFA', 'EEM', 'GLD', 'TLT'],  # Multi-asset trend
    'Sector_Rotation': ['XLK', 'XLF', 'XLV', 'XLE', 'XLI'],  # Top 5 sectors
    'Commodity_Trend': ['GLD', 'SLV', 'USO', 'DBC', 'DBA'],  # Commodity basket
    'Tail_Risk': ['TAIL', 'SHY'],  # Tail hedge ETF + cash proxy
    'Currency_Carry': ['FXA', 'FXB', 'UUP'],  # High carry + USD
    'Gold_Silver': ['GLD', 'SLV'],  # Precious metals ratio
    'Yield_Curve': ['SHY', 'TLT'],  # Duration trade
    'Bond_Duration': ['TLT', 'SHY', 'IEF'],  # Duration timing
    'Vol_Breakout': ['SPY', 'QQQ'],  # Straddle proxy (use equity for backtest)
    'StatArb': ['SPY'],  # Market neutral → proxy as low-beta SPY
    'Credit_Timing': ['HYG', 'LQD', 'SHY'],  # Credit rotation
    'Carry_Momentum': ['SCHD', 'QQQ', 'SHY'],  # Dividend/growth/safety
}

# For backtesting: simulate each strategy with its simple rule
# CTA = 12m-1m momentum on each asset, go long if positive
# Others = simple relative value / momentum rules

MACRO_TICKERS = ['^VIX', 'UUP', 'SPY', 'TLT', 'GLD', 'HYG', 'LQD', '^TNX', 'IEF', 'DBC', 'EEM', 'QQQ']

TRAIN_WINDOW = 252
ADVANCE_DAYS = 21
CAPITAL = 100000
TOP_K = 4  # Allocate to top 4 strategies

# ─── Download data ───
print("Downloading data...")
start_date = '2012-01-01'  # TAIL ETF starts ~2017, but many others earlier

# Collect all unique tickers
all_tickers_set = set(MACRO_TICKERS)
for strat_tickers in STRATEGIES.values():
    all_tickers_set.update(strat_tickers)

all_tickers = list(all_tickers_set)
data = {}
for t in all_tickers:
    try:
        df = yf.download(t, start=start_date, progress=False, auto_adjust=True)
        if len(df) > 50:
            if isinstance(df.columns, pd.MultiIndex):
                close = df[('Close', t)].copy()
            else:
                close = df['Close'].copy()
            clean = t.replace('^', '')
            close.name = clean
            data[clean] = close
    except:
        pass

prices = pd.DataFrame(data).dropna(how='all').ffill().dropna()
returns = prices.pct_change()
print(f"Data: {len(prices)} days, {prices.index[0].date()} to {prices.index[-1].date()}")
print(f"Available: {list(prices.columns)}")

# ─── Simulate each strategy's returns ───
print("\nBuilding strategy return streams...")

strategy_returns = pd.DataFrame(index=prices.index)

# CTA Trend: equal weight momentum — long if 12m-1m > 0, else cash (SHY)
cta_tickers = [t for t in ['SPY', 'EFA', 'EEM', 'GLD', 'TLT'] if t in prices.columns]
for t in cta_tickers:
    mom = prices[t].pct_change(252) - prices[t].pct_change(21)
    signal = (mom > 0).astype(float)
    strategy_returns[f'cta_{t}'] = returns[t] * signal
strategy_returns['CTA_Trend'] = strategy_returns[[f'cta_{t}' for t in cta_tickers]].mean(axis=1)
strategy_returns.drop([f'cta_{t}' for t in cta_tickers], axis=1, inplace=True)

# Sector Rotation: top 2 of 5 sectors by 63d momentum
sector_tickers = [t for t in ['XLK', 'XLF', 'XLV', 'XLE', 'XLI'] if t in prices.columns]
if len(sector_tickers) >= 3:
    sector_mom = pd.DataFrame({t: prices[t].pct_change(63) for t in sector_tickers})
    def top_k_return(row, k=2):
        top = row.nlargest(k).index
        return float(returns.loc[row.name, top].mean()) if row.name in returns.index else 0
    strategy_returns['Sector_Rotation'] = sector_mom.apply(top_k_return, axis=1)

# Commodity Trend: top 2 of available commodities by 63d momentum
comm_tickers = [t for t in ['GLD', 'SLV', 'USO', 'DBC', 'DBA'] if t in prices.columns]
if len(comm_tickers) >= 2:
    comm_mom = pd.DataFrame({t: prices[t].pct_change(63) for t in comm_tickers})
    strategy_returns['Commodity_Trend'] = comm_mom.apply(lambda row: float(returns.loc[row.name, row.nlargest(2).index].mean()) if row.name in returns.index else 0, axis=1)

# Gold/Silver Ratio: long gold when ratio z-score high, long silver when low
if 'GLD' in prices.columns and 'SLV' in prices.columns:
    gs_ratio = prices['GLD'] / prices['SLV']
    gs_z = (gs_ratio - gs_ratio.rolling(63).mean()) / gs_ratio.rolling(63).std()
    gold_signal = (gs_z > 0.5).astype(float)
    silver_signal = (gs_z < -0.5).astype(float)
    neutral = 1 - gold_signal - silver_signal
    strategy_returns['Gold_Silver'] = returns['GLD'] * gold_signal + returns['SLV'] * silver_signal + 0.5*(returns['GLD']+returns['SLV']) * neutral

# Yield Curve: long SHY/short TLT when curve steepening, reverse when flattening
if 'SHY' in prices.columns and 'TLT' in prices.columns:
    curve = prices['TLT'] / prices['SHY']
    curve_mom = curve.pct_change(63)
    steep_signal = (curve_mom < 0).astype(float)  # Steepening = long short end
    strategy_returns['Yield_Curve'] = returns['SHY'] * steep_signal + returns['TLT'] * (1 - steep_signal)

# Bond Duration: long TLT when rates falling, SHY when rising
if 'TLT' in prices.columns and 'SHY' in prices.columns and 'TNX' in prices.columns:
    rate_trend = prices['TNX'].pct_change(63)
    long_dur = (rate_trend < 0).astype(float)
    strategy_returns['Bond_Duration'] = returns['TLT'] * long_dur + returns['SHY'] * (1 - long_dur)

# Currency Carry: high-carry FX when VIX low, USD when VIX high
if 'UUP' in prices.columns:
    carry_fx = [t for t in ['FXA', 'FXB'] if t in prices.columns]
    if carry_fx and 'VIX' in prices.columns:
        vix_low = (prices['VIX'] < prices['VIX'].rolling(63).mean()).astype(float)
        carry_ret = returns[carry_fx].mean(axis=1) if carry_fx else returns['UUP'] * 0
        strategy_returns['Currency_Carry'] = carry_ret * vix_low + returns['UUP'] * (1 - vix_low)

# Tail Risk: long TAIL when VIX high and rising, SHY otherwise
if 'VIX' in prices.columns and 'SHY' in prices.columns:
    vix_high = (prices['VIX'] > 25).astype(float)
    vix_rising = (prices['VIX'].pct_change(5) > 0.1).astype(float)
    tail_signal = (vix_high * vix_rising).clip(0, 1)
    if 'TAIL' in prices.columns:
        strategy_returns['Tail_Risk'] = returns['TAIL'] * tail_signal + returns['SHY'] * (1 - tail_signal)
    else:
        # Proxy: inverse SPY during stress
        strategy_returns['Tail_Risk'] = -returns['SPY'] * tail_signal + returns['SHY'] * (1 - tail_signal)

# Credit Timing: HYG when spreads tightening, SHY when widening
if 'HYG' in prices.columns and 'LQD' in prices.columns and 'SHY' in prices.columns:
    credit_ratio = prices['HYG'] / prices['LQD']
    credit_mom = credit_ratio.pct_change(21)
    risk_on = (credit_mom > 0).astype(float)
    strategy_returns['Credit_Timing'] = returns['HYG'] * risk_on + returns['SHY'] * (1 - risk_on)

# Carry Momentum: SCHD when yield high & stable, QQQ when growth strong, SHY when defensive
if 'QQQ' in prices.columns and 'SHY' in prices.columns:
    schd_proxy = 'SCHD' if 'SCHD' in prices.columns else 'SPY'
    qqq_mom = prices['QQQ'].pct_change(63)
    spy_vol = returns['SPY'].rolling(21).std() if 'SPY' in returns.columns else returns['QQQ'].rolling(21).std()
    growth_on = (qqq_mom > 0).astype(float) * (spy_vol < spy_vol.rolling(63).mean()).astype(float)
    defensive = (spy_vol > spy_vol.rolling(63).quantile(0.75)).astype(float)
    income = 1 - growth_on - defensive
    income = income.clip(0, 1)
    strategy_returns['Carry_Momentum'] = returns[schd_proxy] * income + returns['QQQ'] * growth_on + returns['SHY'] * defensive

# StatArb proxy: low-beta SPY (beta = 0.3)
if 'SPY' in prices.columns:
    strategy_returns['StatArb'] = returns['SPY'] * 0.3

# Vol Breakout proxy: straddle-like payoff
if 'SPY' in prices.columns:
    spy_vol = returns['SPY'].rolling(21).std()
    vol_expanding = (spy_vol > spy_vol.rolling(63).mean()).astype(float)
    strategy_returns['Vol_Breakout'] = returns['SPY'].abs() * vol_expanding - returns['SPY'].abs() * (1-vol_expanding) * 0.3

# Clean up
strategy_returns = strategy_returns.dropna()
available_strats = [c for c in strategy_returns.columns if c in STRATEGIES.keys()]
strategy_returns = strategy_returns[available_strats]
print(f"Strategy streams: {len(available_strats)} strategies, {len(strategy_returns)} days")
print(f"Available: {available_strats}")

# ─── Build ML features ───
features = pd.DataFrame(index=strategy_returns.index)

# Strategy-level features (recent performance of each)
for s in available_strats:
    features[f'{s}_ret_21d'] = strategy_returns[s].rolling(21).sum()
    features[f'{s}_ret_63d'] = strategy_returns[s].rolling(63).sum()
    features[f'{s}_vol_21d'] = strategy_returns[s].rolling(21).std()
    features[f'{s}_sharpe_63d'] = strategy_returns[s].rolling(63).mean() / strategy_returns[s].rolling(63).std().clip(lower=1e-6) * np.sqrt(252)

# Macro features
if 'VIX' in prices.columns:
    features['vix_level'] = prices['VIX'].reindex(features.index)
    features['vix_zscore'] = ((prices['VIX'] - prices['VIX'].rolling(63).mean()) / prices['VIX'].rolling(63).std()).reindex(features.index)
    features['vix_mom_21d'] = prices['VIX'].pct_change(21).reindex(features.index)

if 'TNX' in prices.columns:
    features['tnx_level'] = prices['TNX'].reindex(features.index)
    features['tnx_mom_21d'] = prices['TNX'].pct_change(21).reindex(features.index)

if 'SPY' in prices.columns:
    features['spy_mom_21d'] = prices['SPY'].pct_change(21).reindex(features.index)
    features['spy_vol_21d'] = returns['SPY'].rolling(21).std().reindex(features.index)
    features['spy_mom_63d'] = prices['SPY'].pct_change(63).reindex(features.index)

if 'HYG' in prices.columns and 'LQD' in prices.columns:
    features['credit_spread'] = ((returns['HYG'] - returns['LQD']).rolling(21).mean()).reindex(features.index)

if 'UUP' in prices.columns:
    features['usd_mom_21d'] = prices['UUP'].pct_change(21).reindex(features.index)

if 'GLD' in prices.columns:
    features['gld_mom_21d'] = prices['GLD'].pct_change(21).reindex(features.index)

if 'TLT' in prices.columns:
    features['tlt_mom_21d'] = prices['TLT'].pct_change(21).reindex(features.index)

features = features.dropna()
print(f"Features: {features.shape[1]} cols, {len(features)} rows")

# ─── Walk-forward: predict best strategy ───
print("\nRunning walk-forward backtest...")

# Forward 21d return for each strategy
fwd_strat = {}
for s in available_strats:
    fwd_strat[s] = strategy_returns[s].rolling(ADVANCE_DAYS).sum().shift(-ADVANCE_DAYS)

all_dates = []
all_actuals = []
all_picks = []

for i in range(TRAIN_WINDOW + 126, len(features) - ADVANCE_DAYS, ADVANCE_DAYS):
    train_start = max(0, i - TRAIN_WINDOW)
    train_idx = features.index[train_start:i]
    test_idx = features.index[i:min(i + ADVANCE_DAYS, len(features))]

    # Train one model per strategy: predict if it's in top-K
    strat_scores = {}
    for s in available_strats:
        # Label: 1 if this strategy is in top-K for this period
        fwd_df = pd.DataFrame({ss: fwd_strat[ss] for ss in available_strats})
        ranks = fwd_df.rank(axis=1, ascending=False)
        labels = (ranks[s] <= TOP_K).astype(int).reindex(train_idx).dropna()
        common = train_idx.intersection(labels.index)

        if len(common) < 50:
            continue

        X_tr = np.nan_to_num(features.loc[common].values, nan=0, posinf=0, neginf=0)
        y_tr = labels.loc[common].values

        if len(np.unique(y_tr)) < 2:
            continue

        model = lgb.LGBMClassifier(
            n_estimators=80, max_depth=3, learning_rate=0.05,
            min_child_samples=5, subsample=0.8, colsample_bytree=0.8,
            verbose=-1, n_jobs=1
        )
        model.fit(X_tr, y_tr)

        X_test = np.nan_to_num(features.loc[test_idx].values, nan=0, posinf=0, neginf=0)
        probs = model.predict_proba(X_test)[:, 1]
        strat_scores[s] = pd.Series(probs, index=test_idx)

    if len(strat_scores) < TOP_K:
        continue

    score_df = pd.DataFrame(strat_scores)
    for dt in test_idx:
        if dt not in strategy_returns.index or dt not in score_df.index:
            continue

        row = score_df.loc[dt].dropna()
        if len(row) < TOP_K:
            continue

        top_strats = row.nlargest(TOP_K).index.tolist()
        daily_ret = float(strategy_returns.loc[dt, top_strats].mean())

        all_dates.append(dt)
        all_actuals.append(daily_ret)
        all_picks.append(','.join(top_strats))

print(f"Walk-forward: {len(all_dates)} trading days")

if len(all_dates) < 100:
    print("ERROR: Too few trading days")
    exit(1)

# ─── Benchmarks ───
# Equal weight all strategies
ew_rets = strategy_returns[available_strats].mean(axis=1)

# Static optimal (in-sample for comparison only)
# Just use equal weight as static benchmark

# ─── Metrics ───
equity = pd.Series(index=all_dates, data=np.cumprod(1 + np.array(all_actuals)) * CAPITAL)
daily_rets = pd.Series(index=all_dates, data=all_actuals)

ann_ret = float(daily_rets.mean() * 252)
ann_vol = float(daily_rets.std() * np.sqrt(252))
sharpe = float(ann_ret / ann_vol) if ann_vol > 0 else 0
neg_vol = float(daily_rets[daily_rets < 0].std() * np.sqrt(252)) if len(daily_rets[daily_rets < 0]) > 0 else 1
sortino = float(ann_ret / neg_vol)
max_dd = float((equity / equity.cummax() - 1).min())
calmar = float(ann_ret / abs(max_dd)) if max_dd != 0 else 0
years = len(daily_rets) / 252
cagr = float((equity.iloc[-1] / CAPITAL) ** (1/years) - 1) if years > 0 else 0
win_rate = float((daily_rets > 0).mean())
pf = float(abs(daily_rets[daily_rets > 0].sum() / daily_rets[daily_rets < 0].sum())) if daily_rets[daily_rets < 0].sum() != 0 else 0

spy_rets = returns['SPY'].reindex(daily_rets.index).fillna(0) if 'SPY' in returns.columns else daily_rets * 0
spy_corr = float(daily_rets.corr(spy_rets))

# Equal weight benchmark
ew_bench = ew_rets.reindex(daily_rets.index).fillna(0)
ew_sharpe = float(ew_bench.mean() / ew_bench.std() * np.sqrt(252)) if ew_bench.std() > 0 else 0
ew_cagr = float((np.cumprod(1 + ew_bench.values)[-1]) ** (1/years) - 1) if years > 0 else 0

print(f"\n{'='*60}")
print(f"STRATEGY METRICS")
print(f"{'='*60}")
print(f"ML Dynamic:  CAGR {cagr:.1%}, Sharpe {sharpe:.3f}, Sortino {sortino:.3f}, MaxDD {max_dd:.1%}")
print(f"Equal Weight: CAGR {ew_cagr:.1%}, Sharpe {ew_sharpe:.3f}")
print(f"ML vs EW:    Sharpe improvement {(sharpe/ew_sharpe - 1)*100:.0f}%" if ew_sharpe > 0 else "")
print(f"Calmar: {calmar:.2f}, PF: {pf:.2f}, WR: {win_rate:.1%}")
print(f"SPY Corr: {spy_corr:.3f}")

# Strategy pick frequency
from collections import Counter
all_individual = [p for picks_str in all_picks for p in picks_str.split(',')]
print(f"\nStrategy pick frequency:")
for s, cnt in Counter(all_individual).most_common():
    print(f"  {s}: {cnt} ({cnt/len(all_dates)*100:.0f}%)")

# Yearly returns
yearly = daily_rets.groupby(daily_rets.index.year).apply(lambda x: float((1+x).prod()-1)*100)
print(f"\nYearly returns: {yearly.round(1).to_dict()}")

# ─── Adversarial ───
print(f"\n{'='*60}")
print(f"ADVERSARIAL VALIDATION")
print(f"{'='*60}")

gates_passed = 0

# Gate 1
perm_sharpes = []
for _ in range(200):
    shuffled = daily_rets.sample(frac=1, replace=False).values
    ps = float(np.mean(shuffled) / np.std(shuffled) * np.sqrt(252)) if np.std(shuffled) > 0 else 0
    perm_sharpes.append(ps)
p_value = float(np.mean([ps >= sharpe for ps in perm_sharpes]))
perm_pass = p_value < 0.05
if perm_pass: gates_passed += 1
print(f"G1 Perm: p={p_value:.3f} {'PASS' if perm_pass else 'FAIL'}")

# Gate 2
block_size = len(daily_rets) // 4
block_sharpes = []
for b in range(4):
    block = daily_rets.iloc[b*block_size:(b+1)*block_size]
    bs = float(block.mean() / block.std() * np.sqrt(252)) if block.std() > 0 else 0
    block_sharpes.append(bs)
all_pos = all(s > 0 for s in block_sharpes)
cv = float(np.std(block_sharpes) / np.mean(block_sharpes)) if np.mean(block_sharpes) != 0 else 999
sub_pass = all_pos and cv < 1.0
if sub_pass: gates_passed += 1
print(f"G2 Sub: blocks={[round(s,2) for s in block_sharpes]}, CV={cv:.3f} {'PASS' if sub_pass else 'FAIL'}")

# Gate 3
p5, p95 = daily_rets.quantile(0.05), daily_rets.quantile(0.95)
trimmed = daily_rets[(daily_rets >= p5) & (daily_rets <= p95)]
t_sharpe = float(trimmed.mean() / trimmed.std() * np.sqrt(252)) if trimmed.std() > 0 else 0
degrad = float(1 - t_sharpe / sharpe) if sharpe != 0 else 0
o_pass = abs(degrad) < 0.30
if o_pass: gates_passed += 1
print(f"G3 Outlier: degrad={degrad:.1%} {'PASS' if o_pass else 'FAIL'}")

# Gate 4
green_rets = daily_rets[spy_rets > 0]
red_rets = daily_rets[spy_rets < 0]
gs = float(green_rets.mean() / green_rets.std() * np.sqrt(252)) if green_rets.std() > 0 else 0
rs = float(red_rets.mean() / red_rets.std() * np.sqrt(252)) if red_rets.std() > 0 else 0
gap = float(abs(gs - rs) / max(abs(gs), abs(rs), 0.01))
r1_pass = gap < 0.50
if r1_pass: gates_passed += 1
print(f"G4 R1: green={gs:.3f}, red={rs:.3f}, gap={gap:.3f} {'PASS' if r1_pass else 'FAIL'}")

verdict = "PASS" if gates_passed >= 4 else "FAIL"
print(f"\nVERDICT: {gates_passed}/4 — {verdict}")

# ─── Save ───
results = {
    'strategy': 'ML Dynamic Portfolio Allocator',
    'n_strategies': len(available_strats),
    'top_k': TOP_K,
    'ml_sharpe': round(sharpe, 3), 'ml_sortino': round(sortino, 3),
    'ml_cagr': round(cagr * 100, 1), 'ml_max_dd': round(max_dd * 100, 1),
    'ml_calmar': round(calmar, 2), 'ml_pf': round(pf, 2),
    'ml_wr': round(win_rate * 100, 1),
    'ew_sharpe': round(ew_sharpe, 3), 'ew_cagr': round(ew_cagr * 100, 1),
    'spy_corr': round(spy_corr, 3),
    'gates_passed': gates_passed, 'verdict': verdict,
    'adversarial': {
        'permutation': {'p_value': round(p_value, 3), 'pass': bool(perm_pass)},
        'sub_period': {'cv': round(cv, 3), 'all_positive': bool(all_pos), 'pass': bool(sub_pass)},
        'outlier': {'degradation': round(degrad, 3), 'pass': bool(o_pass)},
        'r1_regime': {'green': round(gs, 3), 'red': round(rs, 3), 'gap': round(gap, 3), 'pass': bool(r1_pass)},
    },
}

with open(OUTPUT / 'results.json', 'w') as f:
    json.dump(results, f, indent=2)

equity.to_csv(OUTPUT / 'equity_curve.csv')
print(f"\nSaved. Done.")
