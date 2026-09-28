#!/usr/bin/env python3
"""
Daily High-Confidence Signal Scanner
======================================

Comprehensive scanner that checks ALL validated signals across sectors
every market day and identifies high-confidence trade setups.

Signals checked:
  1. LGBM Sector Ranking (17 momentum features, walk-forward)
  2. Market-Neutral long/short signals
  3. Momentum Burst (5d, 10d, 21d)
  4. VIX Regime (elevated >25, low <15)
  5. Cross-Sector Dispersion (rotation edge)
  6. Trend Strength (R-squared of 63d log-price trend)
  7. Earnings Proximity (major constituent earnings this week)

Confluence scoring: 0-100 per sector.
  HIGH CONFIDENCE = score > 70 (3+ signals agree)
  VERY HIGH CONFIDENCE = score > 85 (4+ signals agree)

Output:
  - state/daily_signals.json
  - state/high_confidence_alert.txt (if any HC setup found)
  - MLflow run in experiment "daily_signal_scanner"

Usage:
  python daily_signal_scanner.py          # run scan
  python daily_signal_scanner.py --dry    # no MLflow logging
"""

import json
import math
import sys
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

# ── Paths ──
BASE = Path("/home/jupiter/Lvl3Quant")
STATE_DIR = BASE / "state"
STATE_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_FILE = STATE_DIR / "daily_signals.json"
ALERT_FILE = STATE_DIR / "high_confidence_alert.txt"

# ── MLflow ──
MLFLOW_URI = "http://jupiter:5000"
EXPERIMENT_NAME = "daily_signal_scanner"
MLFLOW_OK = False
DRY_RUN = "--dry" in sys.argv

if not DRY_RUN:
    try:
        import mlflow
        mlflow.set_tracking_uri(MLFLOW_URI)
        mlflow.set_experiment(EXPERIMENT_NAME)
        MLFLOW_OK = True
    except Exception as e:
        print(f"[WARN] MLflow unavailable: {e}")

# ── LightGBM ──
try:
    import lightgbm as lgb
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False
    print("[WARN] LightGBM not available. Using momentum fallback.")

# ==================== CONFIG ====================

SECTORS = ['XLK', 'XLF', 'XLE', 'XLV', 'XLY', 'XLP', 'XLI', 'XLB', 'XLU', 'XLRE', 'XLC']
BROAD_MARKET = ['SPY', 'QQQ', 'IWM']
VIX_TICKER = '^VIX'
ACCOUNT_CAPITAL = 645.0

SECTOR_NAMES = {
    'XLK': 'Technology', 'XLF': 'Financials', 'XLE': 'Energy',
    'XLV': 'Health Care', 'XLY': 'Consumer Disc', 'XLP': 'Consumer Staples',
    'XLI': 'Industrials', 'XLB': 'Materials', 'XLU': 'Utilities',
    'XLRE': 'Real Estate', 'XLC': 'Communication',
}

# Top constituents per sector ETF for earnings check
SECTOR_TOP_CONSTITUENTS = {
    'XLK': ['AAPL', 'MSFT', 'NVDA', 'AVGO', 'CRM', 'ORCL', 'ADBE', 'ACN', 'AMD', 'CSCO'],
    'XLF': ['JPM', 'V', 'MA', 'BAC', 'WFC', 'GS', 'MS', 'BLK', 'SCHW', 'AXP'],
    'XLE': ['XOM', 'CVX', 'COP', 'SLB', 'EOG', 'MPC', 'PSX', 'VLO', 'OXY', 'WMB'],
    'XLV': ['UNH', 'JNJ', 'LLY', 'ABBV', 'MRK', 'TMO', 'ABT', 'DHR', 'PFE', 'AMGN'],
    'XLY': ['AMZN', 'TSLA', 'HD', 'MCD', 'LOW', 'NKE', 'SBUX', 'TJX', 'BKNG', 'ORLY'],
    'XLP': ['PG', 'PEP', 'KO', 'COST', 'WMT', 'PM', 'MO', 'CL', 'MDLZ', 'KDP'],
    'XLI': ['CAT', 'DE', 'UNP', 'HON', 'RTX', 'BA', 'GE', 'LMT', 'UPS', 'FDX'],
    'XLB': ['LIN', 'APD', 'SHW', 'FCX', 'NEM', 'NUE', 'DOW', 'DD', 'ECL', 'PPG'],
    'XLU': ['NEE', 'SO', 'DUK', 'D', 'AEP', 'SRE', 'EXC', 'XEL', 'WEC', 'ED'],
    'XLRE': ['PLD', 'AMT', 'CCI', 'EQIX', 'SPG', 'PSA', 'WELL', 'O', 'DLR', 'VICI'],
    'XLC': ['META', 'GOOGL', 'NFLX', 'DIS', 'CMCSA', 'T', 'VZ', 'TMUS', 'CHTR', 'EA'],
}

# 17 LGBM momentum features (identical to V10 production)
FEAT_COLS = [
    'ret_5d', 'ret_10d', 'ret_21d', 'ret_63d', 'ret_126d', 'ret_252d',
    'vol_21d', 'vol_63d', 'sharpe_63d', 'maxdd_63d', 'pct_52w_high', 'mom_accel',
    'pct_pos_months_12m', 'sortino_63d', 'calmar_1y',
    'trend_r2_63d', 'trend_slope_63d',
]

# ==================== DATA ====================

def download_data():
    """Download sector ETFs, broad market, and VIX via yfinance."""
    import yfinance as yf

    all_tickers = SECTORS + BROAD_MARKET + [VIX_TICKER]
    print(f"Downloading {len(all_tickers)} tickers...")
    raw = yf.download(all_tickers, start='2023-01-01', progress=False)
    mi = isinstance(raw.columns, pd.MultiIndex)

    close = raw['Close'] if mi else raw
    if isinstance(close.columns, pd.MultiIndex):
        close.columns = close.columns.get_level_values(-1)

    close = close.ffill()

    rename_map = {'^VIX': 'VIX'}
    close = close.rename(columns=rename_map)

    vix_col = 'VIX' if 'VIX' in close.columns else None
    if vix_col is None:
        raise ValueError("VIX data not downloaded")

    vix = close[vix_col].dropna()
    spy = close['SPY'].dropna()
    sector_close = close[[c for c in SECTORS if c in close.columns]].dropna(how='all')
    broad_close = close[[c for c in BROAD_MARKET if c in close.columns]].dropna(how='all')

    ix = sector_close.index.intersection(vix.index).intersection(spy.index)
    return sector_close.loc[ix], broad_close.loc[ix], spy.loc[ix], vix.loc[ix]


# ==================== FEATURE ENGINEERING (17 features, identical to V10) ====================

def compute_features(px):
    """Compute the 17 momentum features for a single ETF price series."""
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
        y = np.log(px.iloc[-63:].values + 1e-10)
        x = np.arange(len(y))
        slope, _, r_val, _, _ = stats.linregress(x, y)
        f['trend_r2_63d'] = r_val ** 2
        f['trend_slope_63d'] = slope * 252
    else:
        f['trend_r2_63d'] = 0.0
        f['trend_slope_63d'] = 0.0

    return f


# ==================== SIGNAL 1: LGBM SECTOR RANKING ====================

def compute_lgbm_ranking(sc):
    """Walk-forward LGBM ranking using sliding window. Returns dict of {ticker: predicted_rank_score}."""
    idx_end = len(sc) - 1

    if not HAS_LGBM:
        rets_21d = sc.pct_change(21).iloc[-1]
        ranked = rets_21d.rank(pct=True)
        return dict(ranked)

    # Build training samples: go back ~400 trading days, sample every 20 days
    records = []
    start_i = max(260, idx_end - 400)
    all_idx = list(range(start_i, idx_end))
    rebal_idx = all_idx[::20]

    for i in rebal_idx[:-1]:
        for tk in sc.columns:
            px = sc[tk].iloc[:i + 1].dropna()
            feats = compute_features(px)
            if not feats:
                continue
            fi = min(i + 28, len(sc) - 1)
            feats.update({
                'date_idx': i, 'ticker': tk,
                'fwd_ret': float(sc[tk].iloc[fi] / sc[tk].iloc[i] - 1)
            })
            records.append(feats)

    df = pd.DataFrame(records)
    for c in FEAT_COLS:
        if c not in df.columns:
            df[c] = 0.0
    df[FEAT_COLS] = df[FEAT_COLS].fillna(0.0)

    if len(df) < 50:
        rets_21d = sc.pct_change(21).iloc[-1]
        return dict(rets_21d.rank(pct=True))

    df['rank_label'] = df.groupby('date_idx')['fwd_ret'].rank(pct=True)
    X_train = np.nan_to_num(df[FEAT_COLS].values.astype(np.float32))
    y_train = df['rank_label'].values.astype(np.float32)

    m = lgb.LGBMRegressor(
        n_estimators=100, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_samples=5, verbose=-1
    )
    m.fit(X_train, y_train)

    # Predict current ranks
    current_feats = {}
    for tk in sc.columns:
        px = sc[tk].iloc[:idx_end + 1].dropna()
        feats = compute_features(px)
        if feats:
            current_feats[tk] = feats

    if not current_feats:
        return {}

    pred_df = pd.DataFrame(current_feats).T
    for c in FEAT_COLS:
        if c not in pred_df.columns:
            pred_df[c] = 0.0
    X_pred = np.nan_to_num(pred_df[FEAT_COLS].values.astype(np.float32))
    scores = m.predict(X_pred)
    return dict(zip(pred_df.index, scores))


# ==================== SIGNAL 2: MARKET-NEUTRAL LONG/SHORT ====================

def compute_market_neutral_signal(sc, spy):
    """Identify sectors with strong relative strength vs SPY for long/short pairs."""
    result = {}
    for tk in sc.columns:
        if len(sc[tk].dropna()) < 63:
            result[tk] = 0.0
            continue
        # Relative strength: sector return minus SPY return over 21d
        sec_ret = float(sc[tk].iloc[-1] / sc[tk].iloc[-21] - 1)
        spy_ret = float(spy.iloc[-1] / spy.iloc[-21] - 1)
        rel_strength = sec_ret - spy_ret

        # Also check 63d relative strength
        sec_ret_63 = float(sc[tk].iloc[-1] / sc[tk].iloc[-63] - 1)
        spy_ret_63 = float(spy.iloc[-1] / spy.iloc[-63] - 1)
        rel_strength_63 = sec_ret_63 - spy_ret_63

        # Combined relative strength (weight recent more)
        result[tk] = 0.6 * rel_strength + 0.4 * rel_strength_63
    return result


# ==================== SIGNAL 3: MOMENTUM BURST ====================

def compute_momentum_burst(sc):
    """Detect strong short-term momentum (5d, 10d, 21d)."""
    result = {}
    for tk in sc.columns:
        px = sc[tk].dropna()
        if len(px) < 30:
            result[tk] = {'score': 0.0, 'ret_5d': 0.0, 'ret_10d': 0.0, 'ret_21d': 0.0}
            continue
        r5 = float(px.iloc[-1] / px.iloc[-5] - 1)
        r10 = float(px.iloc[-1] / px.iloc[-10] - 1)
        r21 = float(px.iloc[-1] / px.iloc[-21] - 1)

        # Compute historical vol for z-score normalization
        daily_rets = px.pct_change().dropna().iloc[-63:]
        vol = float(daily_rets.std()) + 1e-10

        # Z-scores of momentum
        z5 = r5 / (vol * np.sqrt(5))
        z10 = r10 / (vol * np.sqrt(10))
        z21 = r21 / (vol * np.sqrt(21))

        # Composite momentum burst score
        score = 0.5 * z5 + 0.3 * z10 + 0.2 * z21
        result[tk] = {'score': score, 'ret_5d': r5, 'ret_10d': r10, 'ret_21d': r21}
    return result


# ==================== SIGNAL 4: VIX REGIME ====================

def compute_vix_regime(vix_series):
    """Determine VIX regime and trading implications."""
    current_vix = float(vix_series.iloc[-1])
    vix_5d_ago = float(vix_series.iloc[-5]) if len(vix_series) > 5 else current_vix
    vix_21d_avg = float(vix_series.iloc[-21:].mean()) if len(vix_series) > 21 else current_vix
    vix_pctile = float((vix_series.iloc[-252:] <= current_vix).mean()) if len(vix_series) > 252 else 0.5

    if current_vix < 15:
        regime = 'LOW_VOL'
        bias = 'LONG'
        description = f'VIX {current_vix:.1f} — low vol, favorable for longs'
    elif current_vix < 20:
        regime = 'NORMAL'
        bias = 'NEUTRAL'
        description = f'VIX {current_vix:.1f} — normal range, selective positioning'
    elif current_vix < 25:
        regime = 'ELEVATED'
        bias = 'CAUTIOUS'
        description = f'VIX {current_vix:.1f} — elevated, reduce size / hedge'
    else:
        regime = 'HIGH_VOL'
        bias = 'SHORT_BIAS'
        description = f'VIX {current_vix:.1f} — high vol, favor short/hedge setups'

    vix_change = current_vix - vix_5d_ago
    vix_rising = vix_change > 1.0

    return {
        'current': current_vix,
        'regime': regime,
        'bias': bias,
        'description': description,
        'vix_21d_avg': vix_21d_avg,
        'vix_percentile': vix_pctile,
        'vix_5d_change': vix_change,
        'vix_rising': vix_rising,
    }


# ==================== SIGNAL 5: CROSS-SECTOR DISPERSION ====================

def compute_dispersion(sc):
    """Measure cross-sector return dispersion. Higher = more rotation edge."""
    rets_21d = {}
    for tk in sc.columns:
        px = sc[tk].dropna()
        if len(px) > 21:
            rets_21d[tk] = float(px.iloc[-1] / px.iloc[-21] - 1)
    if len(rets_21d) < 5:
        return {'dispersion': 0.0, 'percentile': 0.5, 'interpretation': 'insufficient data'}

    vals = list(rets_21d.values())
    dispersion = float(np.std(vals))

    # Historical dispersion for percentile
    hist_dispersions = []
    for lookback in range(21, min(252, len(sc)), 21):
        rets_t = {}
        for tk in sc.columns:
            px = sc[tk].dropna()
            end = len(px) - lookback + 21
            if end > 21 and end < len(px):
                rets_t[tk] = float(px.iloc[end] / px.iloc[end - 21] - 1)
        if len(rets_t) >= 5:
            hist_dispersions.append(float(np.std(list(rets_t.values()))))

    pctile = 0.5
    if hist_dispersions:
        pctile = float(np.mean([1 if d <= dispersion else 0 for d in hist_dispersions]))

    if pctile > 0.75:
        interp = 'HIGH — wide dispersion, rotation has strong edge'
    elif pctile > 0.5:
        interp = 'MODERATE — some rotation edge'
    else:
        interp = 'LOW — sectors moving together, less rotation edge'

    return {
        'dispersion': dispersion,
        'percentile': pctile,
        'spread': float(max(vals) - min(vals)),
        'interpretation': interp,
    }


# ==================== SIGNAL 6: TREND STRENGTH ====================

def compute_trend_strength(sc):
    """R-squared of 63d log-price trend for each sector."""
    result = {}
    for tk in sc.columns:
        px = sc[tk].dropna()
        if len(px) < 63:
            result[tk] = {'r2': 0.0, 'slope': 0.0, 'direction': 'FLAT'}
            continue
        y = np.log(px.iloc[-63:].values + 1e-10)
        x = np.arange(len(y))
        slope, _, r_val, _, _ = stats.linregress(x, y)
        r2 = r_val ** 2
        ann_slope = slope * 252

        if r2 > 0.7 and ann_slope > 0.05:
            direction = 'STRONG_UP'
        elif r2 > 0.5 and ann_slope > 0.02:
            direction = 'UP'
        elif r2 > 0.7 and ann_slope < -0.05:
            direction = 'STRONG_DOWN'
        elif r2 > 0.5 and ann_slope < -0.02:
            direction = 'DOWN'
        else:
            direction = 'FLAT'

        result[tk] = {'r2': float(r2), 'slope': float(ann_slope), 'direction': direction}
    return result


# ==================== SIGNAL 7: EARNINGS PROXIMITY ====================

def check_earnings_proximity():
    """Check if major constituents have earnings this week.
    Uses yfinance earnings calendar when available, falls back to saved data."""
    result = {}
    today = datetime.now()
    week_end = today + timedelta(days=7)

    for sector_etf, constituents in SECTOR_TOP_CONSTITUENTS.items():
        earnings_this_week = []
        for stock in constituents[:5]:  # Check top 5 to save API calls
            try:
                import yfinance as yf
                tk = yf.Ticker(stock)
                cal = tk.calendar
                if cal is not None and not (isinstance(cal, pd.DataFrame) and cal.empty):
                    if isinstance(cal, dict):
                        ed = cal.get('Earnings Date', [])
                        if ed:
                            for d in (ed if isinstance(ed, list) else [ed]):
                                try:
                                    dt = pd.Timestamp(d)
                                    if today <= dt.to_pydatetime().replace(tzinfo=None) <= week_end:
                                        earnings_this_week.append(stock)
                                except Exception:
                                    pass
                    elif isinstance(cal, pd.DataFrame):
                        for col in cal.columns:
                            try:
                                dt = pd.Timestamp(cal[col].iloc[0]) if len(cal[col]) > 0 else None
                                if dt and today <= dt.to_pydatetime().replace(tzinfo=None) <= week_end:
                                    earnings_this_week.append(stock)
                            except Exception:
                                pass
            except Exception:
                pass

        result[sector_etf] = {
            'has_earnings': len(earnings_this_week) > 0,
            'earnings_stocks': earnings_this_week,
            'count': len(earnings_this_week),
        }
    return result


# ==================== CONFLUENCE SCORING ====================

def compute_confluence_scores(lgbm_ranks, mn_signals, momentum, vix_regime,
                               dispersion, trend, earnings):
    """Score each sector 0-100 based on signal confluence.

    Scoring weights:
      - LGBM Rank:        25 pts (top 3 = full, bottom 3 = 0, middle scaled)
      - Market-Neutral:   20 pts (relative strength vs SPY)
      - Momentum Burst:   20 pts (z-score based)
      - Trend Strength:   15 pts (R2 + direction)
      - VIX Regime:       10 pts (favorable regime = bonus for longs)
      - Dispersion:        5 pts (high dispersion = rotation edge)
      - Earnings:          5 pts (no imminent earnings = safer, deduct if earnings week)
    """
    scores = {}
    sorted_lgbm = sorted(lgbm_ranks.items(), key=lambda x: x[1], reverse=True)
    n_sectors = len(sorted_lgbm)

    for i, (tk, rank_score) in enumerate(sorted_lgbm):
        s = {'ticker': tk, 'name': SECTOR_NAMES.get(tk, tk), 'components': {}}

        # ── 1. LGBM Rank (25 pts) ──
        rank_pct = 1.0 - (i / max(n_sectors - 1, 1))  # 1.0 = top, 0.0 = bottom
        lgbm_pts = rank_pct * 25.0
        s['components']['lgbm_rank'] = round(lgbm_pts, 1)
        s['lgbm_rank_position'] = i + 1

        # ── 2. Market-Neutral (20 pts) ──
        mn_val = mn_signals.get(tk, 0.0)
        # Scale: +5% relative = full 20 pts, -5% = 0 pts
        mn_pts = max(0, min(20, (mn_val + 0.05) / 0.10 * 20))
        s['components']['market_neutral'] = round(mn_pts, 1)

        # ── 3. Momentum Burst (20 pts) ──
        mom = momentum.get(tk, {})
        mom_z = mom.get('score', 0.0) if isinstance(mom, dict) else 0.0
        # Z-score of 2.0 = full 20 pts, -2.0 = 0 pts
        mom_pts = max(0, min(20, (mom_z + 2.0) / 4.0 * 20))
        s['components']['momentum_burst'] = round(mom_pts, 1)

        # ── 4. Trend Strength (15 pts) ──
        tr = trend.get(tk, {})
        r2 = tr.get('r2', 0.0)
        slope = tr.get('slope', 0.0)
        direction = tr.get('direction', 'FLAT')
        if direction in ('STRONG_UP', 'UP'):
            trend_pts = min(15, r2 * 15 + (5 if direction == 'STRONG_UP' else 0))
        elif direction in ('STRONG_DOWN', 'DOWN'):
            trend_pts = 0.0  # downtrend = no long points
        else:
            trend_pts = r2 * 7.5  # flat gets partial credit
        s['components']['trend_strength'] = round(trend_pts, 1)

        # ── 5. VIX Regime (10 pts) ──
        if vix_regime['regime'] == 'LOW_VOL':
            vix_pts = 10.0
        elif vix_regime['regime'] == 'NORMAL':
            vix_pts = 7.0
        elif vix_regime['regime'] == 'ELEVATED':
            vix_pts = 3.0
        else:  # HIGH_VOL
            vix_pts = 0.0
        s['components']['vix_regime'] = round(vix_pts, 1)

        # ── 6. Dispersion (5 pts) ──
        disp_pctile = dispersion.get('percentile', 0.5)
        disp_pts = disp_pctile * 5.0
        s['components']['dispersion'] = round(disp_pts, 1)

        # ── 7. Earnings (5 pts) ──
        earn = earnings.get(tk, {})
        if earn.get('has_earnings', False):
            earn_pts = 0.0  # Earnings this week = risk, no points
            s['earnings_warning'] = f"Earnings: {', '.join(earn.get('earnings_stocks', []))}"
        else:
            earn_pts = 5.0
        s['components']['earnings_safety'] = round(earn_pts, 1)

        # ── Total ──
        total = sum(s['components'].values())
        s['total_score'] = round(total, 1)

        # ── Direction ──
        if total >= 70:
            s['direction'] = 'LONG'
            s['confidence'] = 'VERY HIGH' if total >= 85 else 'HIGH'
        elif total <= 25:
            s['direction'] = 'SHORT'
            s['confidence'] = 'VERY HIGH' if total <= 10 else 'HIGH'
        else:
            s['direction'] = 'NEUTRAL'
            s['confidence'] = 'LOW'

        # ── Momentum detail ──
        if isinstance(mom, dict):
            s['ret_5d'] = round(mom.get('ret_5d', 0) * 100, 2)
            s['ret_10d'] = round(mom.get('ret_10d', 0) * 100, 2)
            s['ret_21d'] = round(mom.get('ret_21d', 0) * 100, 2)

        s['trend_r2'] = round(r2, 3)
        s['trend_direction'] = direction
        s['relative_strength'] = round(mn_val * 100, 2) if mn_val else 0.0

        scores[tk] = s

    # ── Also compute SHORT scores (inverse scoring for bottom-ranked) ──
    for tk, s in scores.items():
        inv_score = 100 - s['total_score']
        if inv_score >= 70 and s['direction'] != 'LONG':
            s['direction'] = 'SHORT'
            s['short_score'] = round(inv_score, 1)
            s['confidence'] = 'VERY HIGH' if inv_score >= 85 else 'HIGH'

    return scores


# ==================== POSITION SIZING ====================

def compute_position_sizing(scores, vix_regime):
    """Compute suggested position sizes for $645 account."""
    trades = []
    for tk, s in scores.items():
        if s['confidence'] not in ('HIGH', 'VERY HIGH'):
            continue

        direction = s['direction']
        total = s['total_score']

        if direction == 'LONG':
            if total >= 85:
                size_pct = 0.40
            else:
                size_pct = 0.25
        elif direction == 'SHORT':
            short_score = s.get('short_score', 100 - total)
            if short_score >= 85:
                size_pct = 0.40
            else:
                size_pct = 0.25
        else:
            continue

        # VIX regime adjustment (VIX-timed research validated: VIX<20 = +$18.5 avg, VIX≥20 = -$7.6 avg)
        if vix_regime['regime'] == 'HIGH_VOL':
            size_pct *= 0.25  # Quarter position in high vol (backtest shows -$12 avg)
        elif vix_regime['regime'] == 'ELEVATED':
            size_pct *= 0.50  # Half position (VIX 20-25: negative EV per backtest)
        elif vix_regime['regime'] == 'NORMAL' and vix_regime['current'] >= 18:
            size_pct *= 0.80  # Slightly reduce near VIX 20 boundary

        dollar_size = ACCOUNT_CAPITAL * size_pct

        # Estimate ATM option cost (simple BS approximation with 15% uplift)
        px = None
        try:
            import yfinance as yf
            tkr = yf.Ticker(tk)
            hist = tkr.history(period='1d')
            if not hist.empty:
                px = float(hist['Close'].iloc[-1])
        except Exception:
            pass

        option_type = 'CALL' if direction == 'LONG' else 'PUT'
        est_cost = None
        if px:
            # Simplified BS: ATM option ~ S * 0.04 * sqrt(T) * vol
            # Assume 30d option, annualized vol from VIX
            T = 30 / 365
            vol = vix_regime['current'] / 100.0
            est_cost = px * 0.4 * np.sqrt(T) * vol * 100  # per contract, *100 shares
            est_cost *= 1.15  # 15% real-pricing uplift
            est_cost = round(est_cost, 2)

        trades.append({
            'ticker': tk,
            'name': s['name'],
            'direction': direction,
            'confidence': s['confidence'],
            'score': s['total_score'],
            'size_pct': round(size_pct * 100, 1),
            'dollar_size': round(dollar_size, 2),
            'option_type': option_type,
            'instrument': f"ATM {option_type} on {tk}",
            'estimated_option_cost': est_cost,
            'current_price': px,
            'earnings_warning': s.get('earnings_warning', None),
        })

    trades.sort(key=lambda x: x['score'] if x['direction'] == 'LONG' else (100 - x['score']),
                reverse=True)
    return trades


# ==================== BROAD MARKET CONTEXT ====================

def compute_broad_market_context(broad_close, vix_regime):
    """Assess broad market conditions from SPY, QQQ, IWM."""
    context = {}
    for tk in ['SPY', 'QQQ', 'IWM']:
        if tk not in broad_close.columns:
            continue
        px = broad_close[tk].dropna()
        if len(px) < 63:
            continue
        r5 = float(px.iloc[-1] / px.iloc[-5] - 1) * 100
        r21 = float(px.iloc[-1] / px.iloc[-21] - 1) * 100
        r63 = float(px.iloc[-1] / px.iloc[-63] - 1) * 100

        # SMA checks
        sma50 = float(px.iloc[-50:].mean()) if len(px) >= 50 else px.iloc[-1]
        sma200 = float(px.iloc[-200:].mean()) if len(px) >= 200 else px.iloc[-1]
        current = float(px.iloc[-1])

        above_50 = current > sma50
        above_200 = current > sma200

        if above_50 and above_200:
            trend = 'BULLISH'
        elif above_200:
            trend = 'CONSOLIDATING'
        elif not above_50 and not above_200:
            trend = 'BEARISH'
        else:
            trend = 'MIXED'

        context[tk] = {
            'price': round(current, 2),
            'ret_5d': round(r5, 2),
            'ret_21d': round(r21, 2),
            'ret_63d': round(r63, 2),
            'above_50sma': above_50,
            'above_200sma': above_200,
            'trend': trend,
        }

    # Overall market assessment
    trends = [v['trend'] for v in context.values()]
    if all(t == 'BULLISH' for t in trends):
        market_bias = 'BULLISH'
    elif all(t == 'BEARISH' for t in trends):
        market_bias = 'BEARISH'
    elif trends.count('BULLISH') >= 2:
        market_bias = 'LEAN_BULLISH'
    elif trends.count('BEARISH') >= 2:
        market_bias = 'LEAN_BEARISH'
    else:
        market_bias = 'MIXED'

    context['overall_bias'] = market_bias
    context['vix_regime'] = vix_regime['regime']
    return context


# ==================== MAIN ====================

def run_scanner():
    """Run the full daily signal scanner."""
    start_time = datetime.now()
    print("=" * 70)
    print(f"  DAILY HIGH-CONFIDENCE SIGNAL SCANNER")
    print(f"  {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # ── 1. Download data ──
    print("\n[1/8] Downloading market data...")
    sc, broad, spy, vix = download_data()
    print(f"  Data: {len(sc)} days, {len(sc.columns)} sectors, last date: {sc.index[-1].strftime('%Y-%m-%d')}")

    # ── 2. VIX Regime ──
    print("\n[2/8] Analyzing VIX regime...")
    vix_regime = compute_vix_regime(vix)
    print(f"  {vix_regime['description']}")
    print(f"  Percentile: {vix_regime['vix_percentile']:.0%} | 21d avg: {vix_regime['vix_21d_avg']:.1f} | 5d change: {vix_regime['vix_5d_change']:+.1f}")

    # ── 3. LGBM Ranking ──
    print("\n[3/8] Running LGBM sector ranking (walk-forward)...")
    lgbm_ranks = compute_lgbm_ranking(sc)
    sorted_ranks = sorted(lgbm_ranks.items(), key=lambda x: x[1], reverse=True)
    print("  Rankings (best to worst):")
    for i, (tk, score) in enumerate(sorted_ranks):
        print(f"    {i+1}. {tk} ({SECTOR_NAMES.get(tk, tk)}): {score:.4f}")

    # ── 4. Market-Neutral ──
    print("\n[4/8] Computing market-neutral relative strength...")
    mn_signals = compute_market_neutral_signal(sc, spy)

    # ── 5. Momentum Burst ──
    print("\n[5/8] Scanning for momentum bursts...")
    momentum = compute_momentum_burst(sc)

    # ── 6. Cross-Sector Dispersion ──
    print("\n[6/8] Measuring cross-sector dispersion...")
    dispersion = compute_dispersion(sc)
    print(f"  Dispersion: {dispersion['dispersion']:.4f} ({dispersion['interpretation']})")

    # ── 7. Trend Strength ──
    print("\n[7/8] Computing trend strength (63d R-squared)...")
    trend = compute_trend_strength(sc)

    # ── 8. Earnings Proximity ──
    print("\n[8/8] Checking earnings calendar...")
    earnings = check_earnings_proximity()
    earn_count = sum(1 for v in earnings.values() if v.get('has_earnings', False))
    print(f"  {earn_count} sectors with earnings this week")

    # ── Confluence Scoring ──
    print("\n" + "=" * 70)
    print("  CONFLUENCE SCORING")
    print("=" * 70)
    scores = compute_confluence_scores(lgbm_ranks, mn_signals, momentum,
                                        vix_regime, dispersion, trend, earnings)

    # ── Broad Market Context ──
    market_ctx = compute_broad_market_context(broad, vix_regime)

    # ── Position Sizing ──
    trades = compute_position_sizing(scores, vix_regime)

    # ── Display Results ──
    print("\n" + "=" * 70)
    print("  TODAY'S SECTOR SCORES (sorted by score)")
    print("=" * 70)
    sorted_scores = sorted(scores.values(), key=lambda x: x['total_score'], reverse=True)
    for s in sorted_scores:
        conf_tag = ""
        if s['confidence'] == 'VERY HIGH':
            conf_tag = " *** VERY HIGH CONFIDENCE ***"
        elif s['confidence'] == 'HIGH':
            conf_tag = " ** HIGH CONFIDENCE **"
        dir_str = s['direction']
        if dir_str == 'SHORT':
            dir_str = f"SHORT (inv={s.get('short_score', 100-s['total_score']):.0f})"

        print(f"  {s['ticker']:5s} {s['name']:20s}  Score: {s['total_score']:5.1f}  Dir: {dir_str:10s}{conf_tag}")
        print(f"         LGBM:{s['components']['lgbm_rank']:4.1f}  MN:{s['components']['market_neutral']:4.1f}  "
              f"Mom:{s['components']['momentum_burst']:4.1f}  Trend:{s['components']['trend_strength']:4.1f}  "
              f"VIX:{s['components']['vix_regime']:4.1f}  Disp:{s['components']['dispersion']:4.1f}  "
              f"Earn:{s['components']['earnings_safety']:4.1f}")
        if s.get('earnings_warning'):
            print(f"         !! {s['earnings_warning']}")

    # ── High Confidence Setups ──
    hc_setups = [t for t in trades if t['confidence'] in ('HIGH', 'VERY HIGH')]

    print("\n" + "=" * 70)
    if hc_setups:
        print("  TODAY'S HIGH CONFIDENCE SETUPS")
        print("=" * 70)
        for t in hc_setups:
            print(f"\n  {t['confidence']} CONFIDENCE {t['direction']}: {t['ticker']} ({t['name']})")
            print(f"    Score: {t['score']:.1f} | Instrument: {t['instrument']}")
            print(f"    Size: {t['size_pct']:.0f}% of capital = ${t['dollar_size']:.0f}")
            if t['current_price']:
                print(f"    Current price: ${t['current_price']:.2f}")
            if t['estimated_option_cost']:
                print(f"    Est. ATM option cost: ${t['estimated_option_cost']:.2f} per contract")
            if t.get('earnings_warning'):
                print(f"    WARNING: {t['earnings_warning']}")
    else:
        print("  NO HIGH CONFIDENCE SETUP TODAY")
        print("=" * 70)
        print("  No sector passed the 70-point confluence threshold.")
        print("  Patience is a position. Wait for better setups.")

    # ── Broad Market Context ──
    print(f"\n  BROAD MARKET: {market_ctx.get('overall_bias', 'N/A')} | VIX: {vix_regime['regime']}")
    for tk in ['SPY', 'QQQ', 'IWM']:
        if tk in market_ctx:
            m = market_ctx[tk]
            print(f"    {tk}: ${m['price']:.2f}  5d:{m['ret_5d']:+.1f}%  21d:{m['ret_21d']:+.1f}%  "
                  f"Trend:{m['trend']}  >50SMA:{'Y' if m['above_50sma'] else 'N'}  >200SMA:{'Y' if m['above_200sma'] else 'N'}")

    # ── Save Results ──
    output = {
        'scan_timestamp': start_time.isoformat(),
        'scan_date': start_time.strftime('%Y-%m-%d'),
        'vix_regime': vix_regime,
        'market_context': {k: v for k, v in market_ctx.items()},
        'dispersion': dispersion,
        'sector_scores': {tk: s for tk, s in scores.items()},
        'high_confidence_trades': hc_setups,
        'all_trades': trades,
        'lgbm_rankings': {tk: round(float(v), 4) for tk, v in lgbm_ranks.items()},
    }

    with open(OUTPUT_FILE, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Results saved to {OUTPUT_FILE}")

    # ── High Confidence Alert File ──
    if hc_setups:
        alert_lines = [f"HIGH CONFIDENCE SIGNAL ALERT - {start_time.strftime('%Y-%m-%d %H:%M')}"]
        for t in hc_setups:
            alert_lines.append(f"{t['confidence']} {t['direction']}: {t['ticker']} ({t['name']}) score={t['score']:.0f} size={t['size_pct']:.0f}%")
        with open(ALERT_FILE, 'w') as f:
            f.write('\n'.join(alert_lines) + '\n')
        print(f"  Alert written to {ALERT_FILE}")
    else:
        # Remove stale alert file
        if ALERT_FILE.exists():
            ALERT_FILE.unlink()

    # ── MLflow Logging ──
    if MLFLOW_OK:
        try:
            with mlflow.start_run(run_name=f"scan_{start_time.strftime('%Y%m%d_%H%M')}"):
                mlflow.log_param("scan_date", start_time.strftime('%Y-%m-%d'))
                mlflow.log_param("vix_regime", vix_regime['regime'])
                mlflow.log_param("market_bias", market_ctx.get('overall_bias', 'N/A'))
                mlflow.log_param("n_high_confidence", len(hc_setups))

                mlflow.log_metric("vix_current", vix_regime['current'])
                mlflow.log_metric("vix_percentile", vix_regime['vix_percentile'])
                mlflow.log_metric("dispersion", dispersion['dispersion'])
                mlflow.log_metric("dispersion_percentile", dispersion['percentile'])
                mlflow.log_metric("n_hc_setups", len(hc_setups))

                # Log top/bottom sector scores
                for s in sorted_scores[:3]:
                    mlflow.log_metric(f"score_top_{s['ticker']}", s['total_score'])
                for s in sorted_scores[-3:]:
                    mlflow.log_metric(f"score_bot_{s['ticker']}", s['total_score'])

                # Log sector rankings
                for tk, rank in lgbm_ranks.items():
                    mlflow.log_metric(f"lgbm_rank_{tk}", float(rank))

                mlflow.log_artifact(str(OUTPUT_FILE))
                print("  MLflow run logged successfully.")
        except Exception as e:
            print(f"  [WARN] MLflow logging failed: {e}")

    elapsed = (datetime.now() - start_time).total_seconds()
    print(f"\n  Scanner completed in {elapsed:.1f}s")
    print("=" * 70)

    return output


if __name__ == '__main__':
    run_scanner()
