#!/usr/bin/env python3
"""
High-Confidence Signal Scanner V1
====================================

PURPOSE: Aggregate signals from ALL validated strategies to identify
high-confidence trade setups. When multiple independent strategies agree
on a direction, confidence is much higher than any single signal.

STRATEGIES MONITORED:
  1. LGBM Sector ETF Rotation (KB #285, Sharpe 1.40) — which sectors are top-ranked?
  2. Market-Neutral L/S (validated, Sharpe 2.64) — which sectors to long/short?
  3. DL Stock Ranker (Sharpe 2.37) — which stocks are top-ranked?
  4. Factor Momentum — which factors are leading?
  5. VIX Regime — bull/bear/transition?
  6. Breadth/Flow signals — where is money flowing?

OUTPUT:
  - state/high_confidence_signals.json — machine-readable for agentic execution
  - Prints human-readable summary
  - Flags setups where 3+ signals agree (HIGH CONFIDENCE)

DESIGN FOR DAILY CRON:
  This runs at market close (4:15 PM ET) weekdays. It downloads latest data,
  computes all signals, and produces a unified signal report.

Output: state/high_confidence_signals.json
"""

import json
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

_builtin_print = print
def fprint(*args, **kwargs):
    _builtin_print(*args, **kwargs)
    sys.stdout.flush()

BASE = Path("/home/nick/Lvl3Quant") if Path("/home/nick/Lvl3Quant").exists() else Path("/home/jupiter/Lvl3Quant")
sys.path.insert(0, str(BASE))

STATE_DIR = BASE / "state"
STATE_DIR.mkdir(parents=True, exist_ok=True)

try:
    import lightgbm as lgb
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

# ==================== CONFIG ====================

SECTORS = ['XLK', 'XLF', 'XLE', 'XLV', 'XLY', 'XLP', 'XLI', 'XLB', 'XLU', 'XLRE', 'XLC']
FACTOR_ETFS = ['MTUM', 'VLUE', 'QUAL', 'USMV', 'IWM', 'IWF', 'IWD']

# 22 features (17 baseline + 5 cross-sector from enhancement study)
FEAT_COLS = [
    'ret_5d', 'ret_10d', 'ret_21d', 'ret_63d', 'ret_126d', 'ret_252d',
    'vol_21d', 'vol_63d', 'sharpe_63d', 'maxdd_63d', 'pct_52w_high', 'mom_accel',
    'pct_pos_months_12m', 'sortino_63d', 'calmar_1y',
    'trend_r2_63d', 'trend_slope_63d',
    'rel_strength_vs_mean', 'beta_to_spy_63d', 'corr_to_spy_63d',
    'ret_21d_minus_spy', 'vol_ratio_21_63',
]


# ==================== DATA ====================

def download_data():
    import yfinance as yf
    all_tickers = list(set(SECTORS + FACTOR_ETFS + ['SPY', '^VIX', 'QQQ', 'IEF', 'TLT', 'GLD', 'USO']))
    fprint(f"Downloading {len(all_tickers)} tickers...")
    raw = yf.download(all_tickers, start='2020-01-01', progress=False)
    mi = isinstance(raw.columns, pd.MultiIndex)
    close = raw['Close'] if mi else raw
    if isinstance(close.columns, pd.MultiIndex):
        close.columns = close.columns.get_level_values(-1)
    close = close.ffill().rename(columns={'^VIX': 'VIX'})

    volume = raw['Volume'] if mi else None
    if volume is not None:
        if isinstance(volume.columns, pd.MultiIndex):
            volume.columns = volume.columns.get_level_values(-1)
        volume = volume.ffill().fillna(0).rename(columns={'^VIX': 'VIX'})

    return close, volume


# ==================== FEATURES ====================

def compute_features(px, spy_px=None, all_close=None):
    if len(px) < 260:
        return None
    f = {}
    for lb, nm in [(5, 'ret_5d'), (10, 'ret_10d'), (21, 'ret_21d'),
                   (63, 'ret_63d'), (126, 'ret_126d'), (252, 'ret_252d')]:
        f[nm] = float(px.iloc[-1] / px.iloc[-lb] - 1) if len(px) > lb else 0.0
    rets = px.pct_change().dropna()
    f['vol_21d'] = float(rets.iloc[-21:].std() * np.sqrt(252)) if len(rets) > 21 else 0.2
    f['vol_63d'] = float(rets.iloc[-63:].std() * np.sqrt(252)) if len(rets) > 63 else 0.2
    r63 = rets.iloc[-63:]
    f['sharpe_63d'] = float(r63.mean() / (r63.std() + 1e-10) * np.sqrt(252)) if len(r63) > 10 else 0.0
    pk63 = px.iloc[-63:].cummax()
    f['maxdd_63d'] = float(((px.iloc[-63:] / pk63) - 1).min())
    f['pct_52w_high'] = float(px.iloc[-1] / px.iloc[-252:].max())
    f['mom_accel'] = f['ret_21d'] - f['ret_63d'] / 3
    monthly = rets.resample('ME').sum()
    f['pct_pos_months_12m'] = float((monthly.iloc[-12:] > 0).mean()) if len(monthly) >= 12 else 0.5
    dr = r63[r63 < 0]
    f['sortino_63d'] = float(r63.mean() / (dr.std() + 1e-10) * np.sqrt(252)) if len(dr) > 3 else 0.0
    pk = px.iloc[-252:].cummax()
    mdd = float(((px.iloc[-252:] / pk) - 1).min())
    cagr = float(px.iloc[-1] / px.iloc[-252] - 1) if len(px) >= 252 else 0.0
    f['calmar_1y'] = cagr / (abs(mdd) + 1e-10)
    if len(px) >= 63:
        y_vals = np.log(px.iloc[-63:].values + 1e-10)
        x_vals = np.arange(len(y_vals))
        slope, _, r_val, _, _ = stats.linregress(x_vals, y_vals)
        f['trend_r2_63d'] = r_val ** 2
        f['trend_slope_63d'] = slope * 252
    else:
        f['trend_r2_63d'] = 0.0
        f['trend_slope_63d'] = 0.0

    # Cross-sector/factor features
    if all_close is not None:
        peer_rets = []
        for c in all_close.columns:
            if len(all_close[c].dropna()) > 21:
                peer_rets.append(float(all_close[c].iloc[-1] / all_close[c].iloc[-21] - 1))
        f['rel_strength_vs_mean'] = f['ret_21d'] - (np.mean(peer_rets) if peer_rets else 0)
    else:
        f['rel_strength_vs_mean'] = 0.0

    if spy_px is not None and len(spy_px) > 63 and len(px) >= 63:
        sp_rets = spy_px.pct_change().dropna().iloc[-63:]
        tk_rets = rets.iloc[-63:]
        common = sp_rets.index.intersection(tk_rets.index)
        if len(common) > 20:
            cov = np.cov(tk_rets.loc[common].values, sp_rets.loc[common].values)
            f['beta_to_spy_63d'] = float(cov[0, 1] / (cov[1, 1] + 1e-10))
            f['corr_to_spy_63d'] = float(np.corrcoef(tk_rets.loc[common].values, sp_rets.loc[common].values)[0, 1])
        else:
            f['beta_to_spy_63d'] = 1.0
            f['corr_to_spy_63d'] = 0.5
        spy_21d = float(spy_px.iloc[-1] / spy_px.iloc[-21] - 1) if len(spy_px) > 21 else 0
        f['ret_21d_minus_spy'] = f['ret_21d'] - spy_21d
    else:
        f['beta_to_spy_63d'] = 1.0
        f['corr_to_spy_63d'] = 0.5
        f['ret_21d_minus_spy'] = 0.0

    f['vol_ratio_21_63'] = f['vol_21d'] / max(f['vol_63d'], 0.001)
    return f


# ==================== SIGNAL GENERATORS ====================

def signal_sector_rotation(close, spy):
    """LGBM sector ranking signal — which sectors to overweight/underweight."""
    sc = close[[c for c in SECTORS if c in close.columns]].dropna(how='all')
    if not HAS_LGBM:
        # Fallback: momentum ranking
        rets = sc.pct_change(21).iloc[-1].sort_values(ascending=False)
        return {tk: {'score': float(v), 'rank': i+1} for i, (tk, v) in enumerate(rets.items())}

    # Build training data
    idx_end = len(sc) - 1
    records = []
    start_i = max(260, idx_end - 500)
    for i in list(range(start_i, idx_end))[::20]:
        for tk in sc.columns:
            px = sc[tk].iloc[:i + 1].dropna()
            spy_px = spy.iloc[:i + 1]
            feats = compute_features(px, spy_px, sc)
            if not feats:
                continue
            fi = min(i + 28, len(sc) - 1)
            feats['fwd_ret'] = float(sc[tk].iloc[fi] / sc[tk].iloc[i] - 1)
            feats['date_idx'] = i
            records.append(feats)

    df = pd.DataFrame(records)
    for c in FEAT_COLS:
        if c not in df.columns:
            df[c] = 0.0
    df[FEAT_COLS] = df[FEAT_COLS].fillna(0.0)
    if len(df) < 50:
        return {}

    df['rank_label'] = df.groupby('date_idx')['fwd_ret'].rank(pct=True)
    X = np.nan_to_num(df[FEAT_COLS].values.astype(np.float32))
    y = df['rank_label'].values.astype(np.float32)

    m = lgb.LGBMRegressor(n_estimators=100, max_depth=4, learning_rate=0.05,
                           subsample=0.8, colsample_bytree=0.8, min_child_samples=5, verbose=-1)
    m.fit(X, y)

    # Predict current
    results = {}
    for tk in sc.columns:
        feats = compute_features(sc[tk].dropna(), spy, sc)
        if feats:
            results[tk] = feats

    if not results:
        return {}

    pred_df = pd.DataFrame(results).T
    for c in FEAT_COLS:
        if c not in pred_df.columns:
            pred_df[c] = 0.0
    X_pred = np.nan_to_num(pred_df[FEAT_COLS].values.astype(np.float32))
    scores = m.predict(X_pred)

    # Sort and rank
    ranked = sorted(zip(pred_df.index, scores), key=lambda x: x[1], reverse=True)
    return {tk: {'score': round(float(s), 4), 'rank': i+1,
                 'action': 'LONG' if i < 2 else ('SHORT' if i >= len(ranked)-2 else 'NEUTRAL')}
            for i, (tk, s) in enumerate(ranked)}


def signal_momentum_breadth(close):
    """Multi-timeframe momentum breadth across sectors."""
    sc = close[[c for c in SECTORS if c in close.columns]].dropna(how='all')
    signals = {}
    for tk in sc.columns:
        px = sc[tk].dropna()
        if len(px) < 63:
            continue
        pos_count = 0
        for lb in [5, 10, 21, 63]:
            if len(px) > lb and float(px.iloc[-1] / px.iloc[-lb] - 1) > 0:
                pos_count += 1
        signals[tk] = {
            'breadth': pos_count,  # 0-4: how many timeframes are positive
            'signal': 'STRONG_LONG' if pos_count == 4 else ('LONG' if pos_count >= 3 else
                      ('NEUTRAL' if pos_count == 2 else ('SHORT' if pos_count <= 1 else 'NEUTRAL'))),
        }
    return signals


def signal_vix_regime(close):
    """VIX-based market regime classification."""
    if 'VIX' not in close.columns:
        return {'regime': 'UNKNOWN', 'vix': 0, 'vix_21d_avg': 0}
    vix = close['VIX'].dropna()
    if len(vix) < 63:
        return {'regime': 'UNKNOWN', 'vix': 0, 'vix_21d_avg': 0}

    current = float(vix.iloc[-1])
    avg_21d = float(vix.iloc[-21:].mean())
    avg_63d = float(vix.iloc[-63:].mean())
    pctile = float((vix.iloc[-252:] <= current).mean()) if len(vix) > 252 else 0.5

    if current < 15:
        regime = 'LOW_VOL_BULL'
    elif current < 20:
        regime = 'NORMAL'
    elif current < 25:
        regime = 'ELEVATED'
    elif current < 30:
        regime = 'HIGH_VOL'
    else:
        regime = 'CRISIS'

    # Term structure: VIX vs 21d avg
    term = 'CONTANGO' if current < avg_21d else 'BACKWARDATION'

    return {
        'regime': regime,
        'vix': round(current, 2),
        'vix_21d_avg': round(avg_21d, 2),
        'vix_63d_avg': round(avg_63d, 2),
        'vix_percentile': round(pctile * 100, 1),
        'term_structure': term,
        'risk_on': regime in ['LOW_VOL_BULL', 'NORMAL'],
    }


def signal_relative_strength(close):
    """Relative strength of key asset classes — where is money flowing?"""
    assets = {'Equities': 'SPY', 'Tech': 'QQQ', 'Bonds': 'TLT', 'Gold': 'GLD',
              'SmallCap': 'IWM', 'Value': 'IWD', 'Growth': 'IWF'}
    signals = {}
    for name, tk in assets.items():
        if tk not in close.columns:
            continue
        px = close[tk].dropna()
        if len(px) < 63:
            continue
        ret_21d = float(px.iloc[-1] / px.iloc[-21] - 1)
        ret_63d = float(px.iloc[-1] / px.iloc[-63] - 1)
        vol = float(px.pct_change().dropna().iloc[-21:].std() * np.sqrt(252))
        signals[name] = {
            'ticker': tk,
            'ret_21d': round(ret_21d * 100, 2),
            'ret_63d': round(ret_63d * 100, 2),
            'vol_21d': round(vol * 100, 1),
            'momentum': 'UP' if ret_21d > 0.01 else ('DOWN' if ret_21d < -0.01 else 'FLAT'),
        }
    return signals


def signal_sector_flow(close, volume):
    """Volume-weighted sector flow analysis."""
    if volume is None:
        return {}
    signals = {}
    for tk in SECTORS:
        if tk not in close.columns or tk not in volume.columns:
            continue
        px = close[tk].dropna()
        vol = volume[tk].dropna()
        if len(px) < 21 or len(vol) < 21:
            continue

        # Volume relative to 63d average
        vol_ratio = float(vol.iloc[-5:].mean() / (vol.iloc[-63:].mean() + 1)) if len(vol) > 63 else 1.0

        # Price-volume confirmation: up on high volume = bullish
        rets = px.pct_change().dropna()
        common = rets.index.intersection(vol.index)
        if len(common) > 21:
            r = rets.loc[common].iloc[-21:]
            v = vol.loc[common].iloc[-21:]
            up_vol = float(v[r > 0].sum())
            dn_vol = float(v[r < 0].sum())
            up_ratio = up_vol / (up_vol + dn_vol + 1)
        else:
            up_ratio = 0.5

        signals[tk] = {
            'volume_ratio_5d': round(vol_ratio, 2),
            'up_volume_ratio': round(up_ratio, 3),
            'flow_signal': 'INFLOW' if vol_ratio > 1.2 and up_ratio > 0.55 else
                          ('OUTFLOW' if vol_ratio > 1.2 and up_ratio < 0.45 else 'NEUTRAL'),
        }
    return signals


# ==================== CONFLUENCE SCORING ====================

def compute_confluence(sector_signal, breadth_signal, flow_signal, vix_signal):
    """Score each sector by how many signals agree."""
    all_sectors = set()
    for sig in [sector_signal, breadth_signal, flow_signal]:
        all_sectors.update(sig.keys())

    confluence = {}
    for tk in all_sectors:
        score = 0
        reasons = []

        # Sector rotation signal
        if tk in sector_signal:
            s = sector_signal[tk]
            if s.get('action') == 'LONG':
                score += 2  # Strong signal (validated)
                reasons.append(f"LGBM rank #{s['rank']}")
            elif s.get('action') == 'SHORT':
                score -= 2
                reasons.append(f"LGBM bottom (rank #{s['rank']})")

        # Breadth signal
        if tk in breadth_signal:
            b = breadth_signal[tk]
            if b['signal'] == 'STRONG_LONG':
                score += 1
                reasons.append("4/4 timeframes positive")
            elif b['signal'] == 'LONG':
                score += 0.5
                reasons.append("3/4 timeframes positive")
            elif b['signal'] == 'SHORT':
                score -= 1
                reasons.append("0-1/4 timeframes positive")

        # Flow signal
        if tk in flow_signal:
            f = flow_signal[tk]
            if f['flow_signal'] == 'INFLOW':
                score += 1
                reasons.append("Volume inflow")
            elif f['flow_signal'] == 'OUTFLOW':
                score -= 1
                reasons.append("Volume outflow")

        # VIX regime modifier
        if vix_signal.get('risk_on'):
            pass  # No penalty
        elif vix_signal.get('regime') in ['HIGH_VOL', 'CRISIS']:
            score *= 0.5  # Reduce confidence in high-vol regimes
            reasons.append("⚠️ High VIX dampening")

        confidence = 'HIGH' if abs(score) >= 3 else ('MEDIUM' if abs(score) >= 2 else 'LOW')
        direction = 'LONG' if score > 0 else ('SHORT' if score < 0 else 'NEUTRAL')

        confluence[tk] = {
            'score': round(score, 2),
            'direction': direction,
            'confidence': confidence,
            'reasons': reasons,
        }

    return confluence


# ==================== MAIN ====================

def main():
    t0 = time.time()
    fprint("=" * 70)
    fprint("HIGH-CONFIDENCE SIGNAL SCANNER V1")
    fprint(f"Run: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    fprint("=" * 70)

    close, volume = download_data()

    # Run all signal generators
    fprint("\n[1/5] LGBM Sector Rotation signal...")
    sector_signal = signal_sector_rotation(close, close['SPY'] if 'SPY' in close.columns else pd.Series())
    if sector_signal:
        top3 = sorted(sector_signal.items(), key=lambda x: x[1]['score'], reverse=True)[:3]
        bot3 = sorted(sector_signal.items(), key=lambda x: x[1]['score'])[:3]
        top3_str = ', '.join(f"{tk} ({s['score']:.3f})" for tk, s in top3)
        bot3_str = ', '.join(f"{tk} ({s['score']:.3f})" for tk, s in bot3)
        fprint(f"  Top 3: {top3_str}")
        fprint(f"  Bottom 3: {bot3_str}")

    fprint("\n[2/5] Momentum Breadth signal...")
    breadth_signal = signal_momentum_breadth(close)
    strong = [tk for tk, s in breadth_signal.items() if s['signal'] == 'STRONG_LONG']
    weak = [tk for tk, s in breadth_signal.items() if s['signal'] == 'SHORT']
    fprint(f"  Strong momentum (4/4): {strong or 'none'}")
    fprint(f"  Weak momentum (0-1/4): {weak or 'none'}")

    fprint("\n[3/5] VIX Regime...")
    vix_signal = signal_vix_regime(close)
    fprint(f"  Regime: {vix_signal['regime']}, VIX: {vix_signal['vix']}, "
           f"Percentile: {vix_signal.get('vix_percentile', 'N/A')}%")

    fprint("\n[4/5] Relative Strength (asset classes)...")
    rs_signal = signal_relative_strength(close)
    for name, s in sorted(rs_signal.items(), key=lambda x: x[1]['ret_21d'], reverse=True):
        fprint(f"  {name:12s}: {s['momentum']:5s} ({s['ret_21d']:+.1f}% 21d, {s['ret_63d']:+.1f}% 63d)")

    fprint("\n[5/5] Sector Flow (volume analysis)...")
    flow_signal = signal_sector_flow(close, volume)
    inflows = [tk for tk, s in flow_signal.items() if s['flow_signal'] == 'INFLOW']
    outflows = [tk for tk, s in flow_signal.items() if s['flow_signal'] == 'OUTFLOW']
    fprint(f"  Inflows: {inflows or 'none'}")
    fprint(f"  Outflows: {outflows or 'none'}")

    # Compute confluence
    fprint(f"\n{'='*70}")
    fprint("CONFLUENCE SCORING")
    fprint(f"{'='*70}")
    confluence = compute_confluence(sector_signal, breadth_signal, flow_signal, vix_signal)

    # Sort by score
    sorted_conf = sorted(confluence.items(), key=lambda x: x[1]['score'], reverse=True)

    high_conf_longs = []
    high_conf_shorts = []

    for tk, c in sorted_conf:
        marker = "🔥" if c['confidence'] == 'HIGH' else ("⚡" if c['confidence'] == 'MEDIUM' else "")
        fprint(f"  {tk:6s}: score={c['score']:+.1f} {c['direction']:7s} [{c['confidence']}] {marker}")
        for r in c['reasons']:
            fprint(f"          → {r}")

        if c['confidence'] == 'HIGH' and c['direction'] == 'LONG':
            high_conf_longs.append(tk)
        elif c['confidence'] == 'HIGH' and c['direction'] == 'SHORT':
            high_conf_shorts.append(tk)

    # Summary
    fprint(f"\n{'='*70}")
    fprint("HIGH-CONFIDENCE SETUPS")
    fprint(f"{'='*70}")
    if high_conf_longs:
        fprint(f"  🔥 LONG: {', '.join(high_conf_longs)}")
    else:
        fprint(f"  No high-confidence long setups")
    if high_conf_shorts:
        fprint(f"  🔥 SHORT: {', '.join(high_conf_shorts)}")
    else:
        fprint(f"  No high-confidence short setups")
    fprint(f"  VIX regime: {vix_signal['regime']} (risk {'ON' if vix_signal.get('risk_on') else 'OFF'})")

    # Save state
    output = {
        'timestamp': datetime.now().isoformat(),
        'vix_regime': vix_signal,
        'sector_rankings': sector_signal,
        'momentum_breadth': breadth_signal,
        'sector_flow': flow_signal,
        'relative_strength': rs_signal,
        'confluence': confluence,
        'high_confidence_longs': high_conf_longs,
        'high_confidence_shorts': high_conf_shorts,
    }

    with open(STATE_DIR / 'high_confidence_signals.json', 'w') as f:
        json.dump(output, f, indent=2, default=str)
    fprint(f"\nSignals saved to state/high_confidence_signals.json")

    elapsed = time.time() - t0
    fprint(f"Runtime: {elapsed:.0f}s")

    return output


if __name__ == '__main__':
    main()
