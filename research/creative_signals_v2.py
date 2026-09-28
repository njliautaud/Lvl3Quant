#!/usr/bin/env python3
"""
Creative Sector ETF Signals v2 — Three unconventional signal ideas
1. Sector Dispersion (mean-rev vs momentum regime switch)
2. Put-Call Skew / VIX sentiment rotation
3. Volume Divergence (accumulation/distribution detection)

5-day hold, 0.03% RT cost, permutation tested.
"""

import json
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
from pathlib import Path

warnings.filterwarnings('ignore')
np.random.seed(42)

# ── Config ──────────────────────────────────────────────────────────────
SECTOR_TICKERS = ['XLF','XLE','XLU','XLK','XLY','XLP','XLRE','XLV','XLI','XLB','XLC']
BENCHMARK = 'SPY'
VIX_TICKER = '^VIX'
ALL_TICKERS = SECTOR_TICKERS + [BENCHMARK, VIX_TICKER]
HOLD_DAYS = 5
RT_COST = 0.0003  # 0.03% round-trip
PERM_ITERS = 1000
LOOKBACK_DISP = 20  # days for dispersion z-score
LOOKBACK_VOL = 20   # days for volume normalization
DEFENSIVE = ['XLU', 'XLP', 'XLV']
CYCLICAL = ['XLF', 'XLE', 'XLK', 'XLY', 'XLI', 'XLB', 'XLC', 'XLRE']

# ── Data Download ───────────────────────────────────────────────────────
print("Downloading data...")
data = yf.download(ALL_TICKERS, start='2015-01-01', end='2026-08-15', auto_adjust=True)

close = data['Close'].copy()
volume = data['Volume'].copy()

# Handle VIX ticker name
if '^VIX' in close.columns:
    close = close.rename(columns={'^VIX': 'VIX'})
    volume = volume.rename(columns={'^VIX': 'VIX'})

close = close.dropna(how='all')
volume = volume.dropna(how='all')

# Forward-fill small gaps
close = close.ffill().dropna()
volume = volume.ffill().fillna(0)

print(f"Data: {close.index[0].date()} to {close.index[-1].date()}, {len(close)} trading days")

# ── Helper Functions ────────────────────────────────────────────────────
def compute_forward_returns(prices, days=5):
    """5-day forward returns for each column."""
    return prices.shift(-days) / prices - 1

def sharpe_ratio(returns):
    if len(returns) < 10 or returns.std() == 0:
        return 0.0
    return returns.mean() / returns.std() * np.sqrt(252 / HOLD_DAYS)

def profit_factor(returns):
    gross_profit = returns[returns > 0].sum()
    gross_loss = abs(returns[returns < 0].sum())
    if gross_loss == 0:
        return float('inf') if gross_profit > 0 else 0.0
    return gross_profit / gross_loss

def win_rate(returns):
    if len(returns) == 0:
        return 0.0
    return (returns > 0).mean()

def permutation_test(signal_returns, n_iter=1000):
    """Permutation test: shuffle signal assignments, compute Sharpe distribution."""
    observed_sharpe = sharpe_ratio(signal_returns)
    count_better = 0
    shuffled_rets = signal_returns.values.copy()
    for _ in range(n_iter):
        np.random.shuffle(shuffled_rets)
        if sharpe_ratio(pd.Series(shuffled_rets)) >= observed_sharpe:
            count_better += 1
    p_value = count_better / n_iter
    return observed_sharpe, p_value

def regime_stratify(returns, spy_returns_aligned):
    """Split returns by green (SPY up) vs red (SPY down) days."""
    green_mask = spy_returns_aligned > 0
    red_mask = spy_returns_aligned <= 0
    green_rets = returns[green_mask]
    red_rets = returns[red_mask]
    return {
        'green_sharpe': round(sharpe_ratio(green_rets), 3),
        'red_sharpe': round(sharpe_ratio(red_rets), 3),
        'green_n': int(green_mask.sum()),
        'red_n': int(red_mask.sum()),
        'green_wr': round(win_rate(green_rets), 3),
        'red_wr': round(win_rate(red_rets), 3),
    }

# ── Compute shared data ────────────────────────────────────────────────
sector_close = close[SECTOR_TICKERS]
spy_close = close[BENCHMARK]
spy_5d_ret = compute_forward_returns(spy_close, HOLD_DAYS)

sector_1d_ret = sector_close.pct_change()
sector_5d_fwd = compute_forward_returns(sector_close, HOLD_DAYS)

results = {}

# ════════════════════════════════════════════════════════════════════════
# SIGNAL 1: SECTOR DISPERSION — High disp = mean-revert, Low disp = momentum
# ════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("SIGNAL 1: SECTOR DISPERSION")
print("="*70)

# Cross-sectional dispersion: std of sector returns each day
cross_disp = sector_1d_ret.std(axis=1)
disp_mean = cross_disp.rolling(LOOKBACK_DISP).mean()
disp_std = cross_disp.rolling(LOOKBACK_DISP).std()
disp_z = (cross_disp - disp_mean) / disp_std

# Sector 20-day momentum (past returns)
sector_20d_past = sector_close.pct_change(20)

# For each day, rank sectors by past 20d return
sector_rank = sector_20d_past.rank(axis=1, pct=True)

# Signal logic:
# High dispersion (z > 1): mean-revert → go long losers (low rank), short winners (high rank)
# Low dispersion (z < -1): momentum → go long winners (high rank), short losers (low rank)
# Neutral: no trade

signal_returns_disp = []
signal_dates_disp = []

for date in sector_5d_fwd.index:
    if date not in disp_z.index or pd.isna(disp_z.loc[date]):
        continue
    z = disp_z.loc[date]
    if date not in sector_rank.index:
        continue
    ranks = sector_rank.loc[date].dropna()
    fwd = sector_5d_fwd.loc[date].dropna()
    common = ranks.index.intersection(fwd.index)
    if len(common) < 6:
        continue
    ranks = ranks[common]
    fwd = fwd[common]

    if z > 1.0:  # High dispersion → mean-revert
        # Long bottom 3, short top 3
        bottom3 = ranks.nsmallest(3).index
        top3 = ranks.nlargest(3).index
        long_ret = fwd[bottom3].mean()
        short_ret = fwd[top3].mean()
        trade_ret = (long_ret - short_ret) / 2 - RT_COST
        signal_returns_disp.append(trade_ret)
        signal_dates_disp.append(date)
    elif z < -1.0:  # Low dispersion → momentum
        # Long top 3, short bottom 3
        top3 = ranks.nlargest(3).index
        bottom3 = ranks.nsmallest(3).index
        long_ret = fwd[top3].mean()
        short_ret = fwd[bottom3].mean()
        trade_ret = (long_ret - short_ret) / 2 - RT_COST
        signal_returns_disp.append(trade_ret)
        signal_dates_disp.append(date)

disp_rets = pd.Series(signal_returns_disp, index=signal_dates_disp)
print(f"  Trades: {len(disp_rets)}")
print(f"  Sharpe: {sharpe_ratio(disp_rets):.3f}")
print(f"  WR: {win_rate(disp_rets):.1%}")
print(f"  PF: {profit_factor(disp_rets):.2f}")
print(f"  Avg ret: {disp_rets.mean()*100:.3f}%")

# Permutation test
disp_sharpe, disp_pval = permutation_test(disp_rets, PERM_ITERS)
print(f"  Permutation p-value: {disp_pval:.3f}")

# Regime stratification
spy_5d_aligned = spy_5d_ret.reindex(disp_rets.index)
disp_regime = regime_stratify(disp_rets, spy_5d_aligned)
print(f"  Green regime Sharpe: {disp_regime['green_sharpe']}, Red: {disp_regime['red_sharpe']}")

# Sub-analysis: high-disp (mean-rev) vs low-disp (momentum) separately
high_disp_dates = [d for d in signal_dates_disp if disp_z.loc[d] > 1.0]
low_disp_dates = [d for d in signal_dates_disp if disp_z.loc[d] < -1.0]
high_disp_rets = disp_rets.loc[high_disp_dates] if high_disp_dates else pd.Series(dtype=float)
low_disp_rets = disp_rets.loc[low_disp_dates] if low_disp_dates else pd.Series(dtype=float)

print(f"\n  High-disp (mean-rev) trades: {len(high_disp_rets)}, Sharpe: {sharpe_ratio(high_disp_rets):.3f}, WR: {win_rate(high_disp_rets):.1%}")
print(f"  Low-disp (momentum) trades: {len(low_disp_rets)}, Sharpe: {sharpe_ratio(low_disp_rets):.3f}, WR: {win_rate(low_disp_rets):.1%}")

results['sector_dispersion'] = {
    'description': 'High dispersion = mean-revert losers/winners, Low dispersion = momentum',
    'n_trades': len(disp_rets),
    'sharpe': round(sharpe_ratio(disp_rets), 3),
    'win_rate': round(win_rate(disp_rets), 3),
    'profit_factor': round(profit_factor(disp_rets), 3),
    'avg_return_pct': round(disp_rets.mean() * 100, 4),
    'permutation_p_value': round(disp_pval, 3),
    'regime': disp_regime,
    'sub_signals': {
        'high_disp_meanrev': {
            'n_trades': len(high_disp_rets),
            'sharpe': round(sharpe_ratio(high_disp_rets), 3),
            'win_rate': round(win_rate(high_disp_rets), 3),
        },
        'low_disp_momentum': {
            'n_trades': len(low_disp_rets),
            'sharpe': round(sharpe_ratio(low_disp_rets), 3),
            'win_rate': round(win_rate(low_disp_rets), 3),
        }
    }
}


# ════════════════════════════════════════════════════════════════════════
# SIGNAL 2: VIX SENTIMENT → SECTOR ROTATION (Defensive vs Cyclical)
# ════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("SIGNAL 2: VIX SENTIMENT ROTATION")
print("="*70)

vix = close['VIX'] if 'VIX' in close.columns else None

if vix is not None:
    vix_5d_chg = vix.pct_change(5)  # 5-day VIX change
    vix_5d_z = (vix_5d_chg - vix_5d_chg.rolling(60).mean()) / vix_5d_chg.rolling(60).std()

    # Compute defensive vs cyclical forward returns
    def_tickers = [t for t in DEFENSIVE if t in sector_5d_fwd.columns]
    cyc_tickers = [t for t in CYCLICAL if t in sector_5d_fwd.columns]

    def_fwd = sector_5d_fwd[def_tickers].mean(axis=1)
    cyc_fwd = sector_5d_fwd[cyc_tickers].mean(axis=1)

    signal_returns_vix = []
    signal_dates_vix = []

    for date in sector_5d_fwd.index:
        if date not in vix_5d_z.index or pd.isna(vix_5d_z.loc[date]):
            continue
        z = vix_5d_z.loc[date]
        if pd.isna(def_fwd.loc[date]) or pd.isna(cyc_fwd.loc[date]):
            continue

        if z > 1.5:  # VIX spiking → fear → long defensive, short cyclical
            trade_ret = (def_fwd.loc[date] - cyc_fwd.loc[date]) / 2 - RT_COST
            signal_returns_vix.append(trade_ret)
            signal_dates_vix.append(date)
        elif z < -1.5:  # VIX dropping → greed → long cyclical, short defensive
            trade_ret = (cyc_fwd.loc[date] - def_fwd.loc[date]) / 2 - RT_COST
            signal_returns_vix.append(trade_ret)
            signal_dates_vix.append(date)

    vix_rets = pd.Series(signal_returns_vix, index=signal_dates_vix)
    print(f"  Trades: {len(vix_rets)}")
    print(f"  Sharpe: {sharpe_ratio(vix_rets):.3f}")
    print(f"  WR: {win_rate(vix_rets):.1%}")
    print(f"  PF: {profit_factor(vix_rets):.2f}")
    print(f"  Avg ret: {vix_rets.mean()*100:.3f}%")

    vix_sharpe, vix_pval = permutation_test(vix_rets, PERM_ITERS)
    print(f"  Permutation p-value: {vix_pval:.3f}")

    spy_5d_aligned_v = spy_5d_ret.reindex(vix_rets.index)
    vix_regime = regime_stratify(vix_rets, spy_5d_aligned_v)
    print(f"  Green regime Sharpe: {vix_regime['green_sharpe']}, Red: {vix_regime['red_sharpe']}")

    # Sub-analysis by direction
    fear_dates = [d for d in signal_dates_vix if vix_5d_z.loc[d] > 1.5]
    greed_dates = [d for d in signal_dates_vix if vix_5d_z.loc[d] < -1.5]
    fear_rets = vix_rets.loc[fear_dates] if fear_dates else pd.Series(dtype=float)
    greed_rets = vix_rets.loc[greed_dates] if greed_dates else pd.Series(dtype=float)

    print(f"\n  Fear trades (long defensive): {len(fear_rets)}, Sharpe: {sharpe_ratio(fear_rets):.3f}, WR: {win_rate(fear_rets):.1%}")
    print(f"  Greed trades (long cyclical): {len(greed_rets)}, Sharpe: {sharpe_ratio(greed_rets):.3f}, WR: {win_rate(greed_rets):.1%}")

    # Per-sector breakdown for VIX signal
    per_sector_vix = {}
    for ticker in SECTOR_TICKERS:
        if ticker not in sector_5d_fwd.columns:
            continue
        t_rets = []
        for date in signal_dates_vix:
            if pd.isna(sector_5d_fwd.loc[date, ticker]):
                continue
            z = vix_5d_z.loc[date]
            if z > 1.5:  # Fear: long if defensive, short if cyclical
                if ticker in DEFENSIVE:
                    t_rets.append(sector_5d_fwd.loc[date, ticker] - RT_COST/2)
                else:
                    t_rets.append(-sector_5d_fwd.loc[date, ticker] - RT_COST/2)
            elif z < -1.5:  # Greed: long if cyclical, short if defensive
                if ticker in CYCLICAL:
                    t_rets.append(sector_5d_fwd.loc[date, ticker] - RT_COST/2)
                else:
                    t_rets.append(-sector_5d_fwd.loc[date, ticker] - RT_COST/2)
        t_rets = pd.Series(t_rets)
        per_sector_vix[ticker] = {
            'n_trades': len(t_rets),
            'sharpe': round(sharpe_ratio(t_rets), 3),
            'win_rate': round(win_rate(t_rets), 3),
        }

    results['vix_sentiment_rotation'] = {
        'description': 'VIX spike = long defensive/short cyclical, VIX drop = long cyclical/short defensive',
        'n_trades': len(vix_rets),
        'sharpe': round(sharpe_ratio(vix_rets), 3),
        'win_rate': round(win_rate(vix_rets), 3),
        'profit_factor': round(profit_factor(vix_rets), 3),
        'avg_return_pct': round(vix_rets.mean() * 100, 4),
        'permutation_p_value': round(vix_pval, 3),
        'regime': vix_regime,
        'sub_signals': {
            'fear_long_defensive': {
                'n_trades': len(fear_rets),
                'sharpe': round(sharpe_ratio(fear_rets), 3),
                'win_rate': round(win_rate(fear_rets), 3),
            },
            'greed_long_cyclical': {
                'n_trades': len(greed_rets),
                'sharpe': round(sharpe_ratio(greed_rets), 3),
                'win_rate': round(win_rate(greed_rets), 3),
            }
        },
        'per_sector': per_sector_vix,
    }
else:
    print("  VIX data not available!")
    results['vix_sentiment_rotation'] = {'error': 'VIX data not available'}


# ════════════════════════════════════════════════════════════════════════
# SIGNAL 3: VOLUME DIVERGENCE (price-volume mismatch)
# ════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("SIGNAL 3: VOLUME DIVERGENCE")
print("="*70)

# For each sector: compute normalized volume change vs price change
# Volume surge + flat price → accumulation → go long
# Price move + flat volume → weak move → fade

signal_returns_voldiv = []
signal_dates_voldiv = []
signal_tickers_voldiv = []

per_sector_voldiv = {}

for ticker in SECTOR_TICKERS:
    if ticker not in close.columns or ticker not in volume.columns:
        continue

    px = close[ticker]
    vol = volume[ticker]

    # 1-day returns and volume change
    px_ret_1d = px.pct_change()
    vol_norm = vol / vol.rolling(LOOKBACK_VOL).mean()  # volume relative to 20d avg

    # Z-score of volume
    vol_z = (vol_norm - vol_norm.rolling(LOOKBACK_VOL).mean()) / vol_norm.rolling(LOOKBACK_VOL).std()
    # Z-score of abs price change
    abs_ret_z = (px_ret_1d.abs() - px_ret_1d.abs().rolling(LOOKBACK_VOL).mean()) / px_ret_1d.abs().rolling(LOOKBACK_VOL).std()

    # Divergence = vol_z - abs_ret_z
    # Positive: volume surging relative to price move → accumulation
    # Negative: price moving without volume → weak move
    divergence = vol_z - abs_ret_z

    ticker_rets = []

    fwd_ret = compute_forward_returns(px, HOLD_DAYS)

    for date in fwd_ret.index:
        if date not in divergence.index or pd.isna(divergence.loc[date]) or pd.isna(fwd_ret.loc[date]):
            continue

        d = divergence.loc[date]

        if d > 2.0:  # Strong volume, weak price → accumulation → long
            trade_ret = fwd_ret.loc[date] - RT_COST
            signal_returns_voldiv.append(trade_ret)
            signal_dates_voldiv.append(date)
            signal_tickers_voldiv.append(ticker)
            ticker_rets.append(trade_ret)
        elif d < -2.0:  # Weak volume, strong price → fade → short
            trade_ret = -fwd_ret.loc[date] - RT_COST
            signal_returns_voldiv.append(trade_ret)
            signal_dates_voldiv.append(date)
            signal_tickers_voldiv.append(ticker)
            ticker_rets.append(trade_ret)

    ticker_rets = pd.Series(ticker_rets)
    per_sector_voldiv[ticker] = {
        'n_trades': len(ticker_rets),
        'sharpe': round(sharpe_ratio(ticker_rets), 3),
        'win_rate': round(win_rate(ticker_rets), 3),
        'profit_factor': round(profit_factor(ticker_rets), 3) if len(ticker_rets) > 0 else 0,
    }

voldiv_rets = pd.Series(signal_returns_voldiv, index=signal_dates_voldiv)
print(f"  Total trades: {len(voldiv_rets)}")
print(f"  Sharpe: {sharpe_ratio(voldiv_rets):.3f}")
print(f"  WR: {win_rate(voldiv_rets):.1%}")
print(f"  PF: {profit_factor(voldiv_rets):.2f}")
print(f"  Avg ret: {voldiv_rets.mean()*100:.3f}%")

voldiv_sharpe, voldiv_pval = permutation_test(voldiv_rets, PERM_ITERS)
print(f"  Permutation p-value: {voldiv_pval:.3f}")

spy_5d_aligned_vd = spy_5d_ret.reindex(voldiv_rets.index)
voldiv_regime = regime_stratify(voldiv_rets, spy_5d_aligned_vd)
print(f"  Green regime Sharpe: {voldiv_regime['green_sharpe']}, Red: {voldiv_regime['red_sharpe']}")

# Sub-analysis: accumulation vs fade
accum_mask = pd.Series(signal_tickers_voldiv, index=signal_dates_voldiv)
# Rebuild with direction info
accum_rets_list = []
fade_rets_list = []
for i, date in enumerate(signal_dates_voldiv):
    ticker = signal_tickers_voldiv[i]
    px = close[ticker]
    vol = volume[ticker]
    px_ret_1d = px.pct_change()
    vol_norm = vol / vol.rolling(LOOKBACK_VOL).mean()
    vol_z = (vol_norm - vol_norm.rolling(LOOKBACK_VOL).mean()) / vol_norm.rolling(LOOKBACK_VOL).std()
    abs_ret_z = (px_ret_1d.abs() - px_ret_1d.abs().rolling(LOOKBACK_VOL).mean()) / px_ret_1d.abs().rolling(LOOKBACK_VOL).std()
    d = vol_z.loc[date] - abs_ret_z.loc[date]
    if d > 2.0:
        accum_rets_list.append(signal_returns_voldiv[i])
    else:
        fade_rets_list.append(signal_returns_voldiv[i])

accum_rets = pd.Series(accum_rets_list)
fade_rets = pd.Series(fade_rets_list)
print(f"\n  Accumulation (long) trades: {len(accum_rets)}, Sharpe: {sharpe_ratio(accum_rets):.3f}, WR: {win_rate(accum_rets):.1%}")
print(f"  Fade (short) trades: {len(fade_rets)}, Sharpe: {sharpe_ratio(fade_rets):.3f}, WR: {win_rate(fade_rets):.1%}")

print("\n  Per-sector breakdown:")
for t, stats in sorted(per_sector_voldiv.items(), key=lambda x: x[1]['sharpe'], reverse=True):
    print(f"    {t}: {stats['n_trades']} trades, Sharpe={stats['sharpe']:.3f}, WR={stats['win_rate']:.1%}")

results['volume_divergence'] = {
    'description': 'Volume surge + flat price = long (accumulation), Price move + flat volume = short (fade)',
    'n_trades': len(voldiv_rets),
    'sharpe': round(sharpe_ratio(voldiv_rets), 3),
    'win_rate': round(win_rate(voldiv_rets), 3),
    'profit_factor': round(profit_factor(voldiv_rets), 3),
    'avg_return_pct': round(voldiv_rets.mean() * 100, 4),
    'permutation_p_value': round(voldiv_pval, 3),
    'regime': voldiv_regime,
    'sub_signals': {
        'accumulation_long': {
            'n_trades': len(accum_rets),
            'sharpe': round(sharpe_ratio(accum_rets), 3),
            'win_rate': round(win_rate(accum_rets), 3),
        },
        'fade_short': {
            'n_trades': len(fade_rets),
            'sharpe': round(sharpe_ratio(fade_rets), 3),
            'win_rate': round(win_rate(fade_rets), 3),
        }
    },
    'per_sector': per_sector_voldiv,
}


# ════════════════════════════════════════════════════════════════════════
# SUMMARY
# ════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("SUMMARY")
print("="*70)

summary = {}
for name, res in results.items():
    if 'error' in res:
        continue
    verdict = "PASS" if res['permutation_p_value'] < 0.05 and res['sharpe'] > 0.3 else "FAIL"
    # Check regime robustness
    if 'regime' in res:
        g = res['regime']['green_sharpe']
        r = res['regime']['red_sharpe']
        max_sr = max(abs(g), abs(r))
        if max_sr > 0:
            regime_skew = abs(g - r) / max_sr
        else:
            regime_skew = 0
        if regime_skew > 0.5:
            verdict = "FAIL (regime-dependent)"

    summary[name] = verdict
    print(f"  {name}: Sharpe={res['sharpe']:.3f}, WR={res['win_rate']:.1%}, PF={res['profit_factor']:.2f}, "
          f"p-val={res['permutation_p_value']:.3f} → {verdict}")

results['summary'] = summary
results['metadata'] = {
    'run_date': datetime.now().isoformat(),
    'data_start': str(close.index[0].date()),
    'data_end': str(close.index[-1].date()),
    'n_trading_days': len(close),
    'hold_days': HOLD_DAYS,
    'rt_cost': RT_COST,
    'permutation_iterations': PERM_ITERS,
}

# Save results
outpath = Path('/home/jupiter/Lvl3Quant/research/creative_signals_v2_results.json')
with open(outpath, 'w') as f:
    json.dump(results, f, indent=2, default=str)
print(f"\nResults saved to {outpath}")
