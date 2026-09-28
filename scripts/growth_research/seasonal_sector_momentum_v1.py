#!/usr/bin/env python3
"""Seasonal Sector Momentum v1 — Multi-signal confluence with seasonality.

NEW SIGNAL: Historical sector seasonality patterns.
- Computes average monthly returns for each sector over trailing 10 years
- "Seasonal alignment" = when current month is historically strong for a sector
  AND momentum/quality signals agree → higher conviction entry

Hypothesis: Sector seasonality is well-documented (energy winter rally,
tech Q4, financials Q1 etc). Combining with our validated LGBM momentum
should produce higher-confidence entries per HC #750.

Variants:
  A: Baseline bull+bear v1 (no seasonality) — control
  B: Seasonal filter — only trade sectors in their top 4 seasonal months
  C: Seasonal boost — score += seasonal_z_score for composite ranking
  D: Seasonal + momentum acceleration — require positive momentum acceleration
     in seasonal window
  E: Anti-seasonal contrarian — trade sectors in their WORST seasonal months
     when momentum is strong (contrarian signal)
  F: Full confluence (seasonality + momentum + quality + rel_str + RSI +
     VIX + earnings_proximity) — maximum HC #750 compliance

All variants: $645 starting capital, bull call spreads (VIX>=20) + bear put
spreads (VIX<20), 20d exit, ATR pricing + 15% haircut, 4-gate adversarial.
"""
import json, sys, numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')
from pathlib import Path
from datetime import datetime
from scipy import stats
import lightgbm as lgb

BASE = Path(__file__).resolve().parents[2]
RESULTS_DIR = BASE / 'research' / 'findings'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_PATH = RESULTS_DIR / 'seasonal_sector_momentum_v1_results.json'
def fprint(*a, **kw): print(*a, **kw, flush=True)

MLFLOW_OK = False
try:
    import urllib.request; urllib.request.urlopen('http://jupiter:5000/', timeout=2)
    import mlflow; mlflow.set_tracking_uri('http://jupiter:5000'); MLFLOW_OK = True
except Exception: fprint("MLflow unavailable")

SECTORS = ['XLK','XLF','XLE','XLV','XLY','XLP','XLI','XLB','XLU','XLRE','XLC']
CAP = 645.0; LEG_COMM = 0.65; SPREAD_COMM = 4*LEG_COMM; HAIRCUT = 0.15

# ==================== FEATURES ====================
MOM_COLS = ['ret_5d','ret_10d','ret_21d','ret_63d','ret_126d','ret_252d',
            'vol_21d','vol_63d','sharpe_63d','maxdd_63d','pct_52w_high','mom_accel']
QUALITY_COLS = ['pct_pos_months_12m','sortino_63d','calmar_1y','up_capture','dn_capture',
                'trend_r2_63d','trend_slope_63d','rel_vol_21d']
SEASONAL_COLS = ['seasonal_z','seasonal_rank','seasonal_hit_rate']
ALL_FEAT_COLS = MOM_COLS + QUALITY_COLS + SEASONAL_COLS

def download_data():
    import yfinance as yf
    fprint("Downloading data...")
    tickers = SECTORS + ['SPY', '^VIX']
    raw = yf.download(tickers, start='2006-01-01', end='2026-07-26', progress=False)
    mi = isinstance(raw.columns, pd.MultiIndex)
    close = raw['Close'] if mi else raw
    high, low = (raw['High'], raw['Low']) if mi else (raw, raw)
    volume = raw['Volume'] if mi else raw
    vc = '^VIX' if '^VIX' in close.columns else 'VIX'
    vix, spy = close[vc].dropna(), close['SPY'].dropna()
    sc = close[[c for c in SECTORS if c in close.columns]].dropna(how='all')
    sh = high[[c for c in SECTORS if c in high.columns]].dropna(how='all')
    sl = low[[c for c in SECTORS if c in low.columns]].dropna(how='all')
    ix = sc.index.intersection(vix.index).intersection(spy.index).intersection(sh.index).intersection(sl.index)
    fprint(f"Data: {len(ix)} days, {len(sc.columns)} sectors")
    return sc.loc[ix], sh.loc[ix], sl.loc[ix], spy.loc[ix], vix.loc[ix]

# ==================== SEASONALITY ENGINE ====================
def compute_sector_seasonality(sc, as_of_date, lookback_years=10):
    """Compute monthly seasonal patterns for each sector using trailing data.

    Returns dict: {ticker: {month: {'avg_ret': float, 'hit_rate': float, 'z_score': float}}}
    """
    idx = sc.index.get_indexer([as_of_date], method='ffill')[0]
    cutoff_days = lookback_years * 252
    start_idx = max(0, idx - cutoff_days)
    hist = sc.iloc[start_idx:idx+1]

    if len(hist) < 252:  # Need at least 1 year
        return {}

    monthly_rets = hist.resample('ME').last().pct_change().dropna()
    if len(monthly_rets) < 12:
        return {}

    result = {}
    for tk in hist.columns:
        tk_rets = monthly_rets[tk].dropna()
        if len(tk_rets) < 12:
            continue
        result[tk] = {}
        all_mean = tk_rets.mean()
        all_std = tk_rets.std()
        for month in range(1, 13):
            m_rets = tk_rets[tk_rets.index.month == month]
            if len(m_rets) < 3:
                result[tk][month] = {'avg_ret': 0, 'hit_rate': 0.5, 'z_score': 0}
                continue
            avg = m_rets.mean()
            hit = (m_rets > 0).mean()
            z = (avg - all_mean) / (all_std + 1e-10)
            result[tk][month] = {'avg_ret': float(avg), 'hit_rate': float(hit), 'z_score': float(z)}
    return result

def get_seasonal_features(tk, dt, seasonality_cache, sc):
    """Get seasonal features for a ticker at a given date."""
    month = dt.month
    if tk not in seasonality_cache or month not in seasonality_cache[tk]:
        return {'seasonal_z': 0, 'seasonal_rank': 0.5, 'seasonal_hit_rate': 0.5}

    s = seasonality_cache[tk][month]
    # Rank this month among all months for this sector
    all_z = [seasonality_cache[tk].get(m, {}).get('z_score', 0) for m in range(1, 13)]
    rank = sum(1 for z in all_z if z <= s['z_score']) / 12.0

    return {
        'seasonal_z': s['z_score'],
        'seasonal_rank': rank,
        'seasonal_hit_rate': s['hit_rate']
    }

def is_seasonal_top(tk, dt, seasonality_cache, top_n=4):
    """Is current month in the top N seasonal months for this sector?"""
    month = dt.month
    if tk not in seasonality_cache:
        return False
    all_months = [(m, seasonality_cache[tk].get(m, {}).get('z_score', 0)) for m in range(1, 13)]
    all_months.sort(key=lambda x: x[1], reverse=True)
    top_months = [m for m, _ in all_months[:top_n]]
    return month in top_months

def is_seasonal_bottom(tk, dt, seasonality_cache, bottom_n=4):
    """Is current month in the bottom N seasonal months for this sector?"""
    month = dt.month
    if tk not in seasonality_cache:
        return False
    all_months = [(m, seasonality_cache[tk].get(m, {}).get('z_score', 0)) for m in range(1, 13)]
    all_months.sort(key=lambda x: x[1])
    bot_months = [m for m, _ in all_months[:bottom_n]]
    return month in bot_months

# ==================== STANDARD FEATURES ====================
def compute_features(px, spy_slice):
    if len(px) < 260: return None
    f = {}
    for lb, nm in [(5,'ret_5d'),(10,'ret_10d'),(21,'ret_21d'),(63,'ret_63d'),(126,'ret_126d'),(252,'ret_252d')]:
        f[nm] = float(px.iloc[-1]/px.iloc[-lb]-1) if len(px) > lb else 0.0
    rets = px.pct_change().dropna()
    f['vol_21d'] = float(rets.iloc[-21:].std()*np.sqrt(252)) if len(rets) > 21 else 0.2
    f['vol_63d'] = float(rets.iloc[-63:].std()*np.sqrt(252)) if len(rets) > 63 else 0.2
    r63 = rets.iloc[-63:]
    f['sharpe_63d'] = float(r63.mean()/(r63.std()+1e-10)*np.sqrt(252)) if len(r63) > 10 else 0.0
    pk63 = px.iloc[-63:].cummax()
    f['maxdd_63d'] = float(((px.iloc[-63:]/pk63)-1).min())
    f['pct_52w_high'] = float(px.iloc[-1]/px.iloc[-252:].max())
    f['mom_accel'] = f['ret_21d'] - f['ret_63d']/3

    monthly = rets.resample('ME').sum()
    f['pct_pos_months_12m'] = float((monthly.iloc[-12:] > 0).mean()) if len(monthly) >= 12 else 0.5
    dr = r63[r63 < 0]
    f['sortino_63d'] = float(r63.mean()/(dr.std()+1e-10)*np.sqrt(252)) if len(dr) > 3 else 0.0
    pk = px.iloc[-252:].cummax()
    mdd = float(((px.iloc[-252:]/pk)-1).min())
    cagr = float(px.iloc[-1]/px.iloc[-252]-1) if len(px) >= 252 else 0.0
    f['calmar_1y'] = cagr / (abs(mdd) + 1e-10)
    up_days = rets[rets > 0]; dn_days = rets[rets < 0]
    f['up_capture'] = float(up_days.iloc[-63:].mean() / (up_days.mean()+1e-10)) if len(up_days) > 10 else 1.0
    f['dn_capture'] = float(dn_days.iloc[-63:].mean() / (dn_days.mean()+1e-10)) if len(dn_days) > 10 else 1.0
    if len(px) >= 63:
        y = np.log(px.iloc[-63:].values + 1e-10); x = np.arange(len(y))
        slope, _, r_val, _, _ = stats.linregress(x, y)
        f['trend_r2_63d'] = r_val**2; f['trend_slope_63d'] = slope*252
    else:
        f['trend_r2_63d'] = 0.0; f['trend_slope_63d'] = 0.0
    f['rel_vol_21d'] = 1.0
    return f

# ==================== LGBM RANKING ====================
def build_rankings(sc, spy, rebal_dates, feat_cols, seasonality_by_date=None):
    fprint(f"  Building LGBM rankings ({len(feat_cols)} features)...")
    records = []
    for dt in rebal_dates:
        idx = sc.index.get_indexer([dt], method='ffill')[0]
        if idx < 260: continue
        seasonal = seasonality_by_date.get(dt, {}) if seasonality_by_date else {}
        for tk in sc.columns:
            px = sc[tk].iloc[:idx+1].dropna()
            spy_s = spy.iloc[:idx+1]
            feats = compute_features(px, spy_s)
            if not feats: continue
            # Add seasonal features
            sf = get_seasonal_features(tk, dt, seasonal, sc)
            feats.update(sf)
            fi = min(idx+14, len(sc)-1)
            feats.update({'date': dt, 'ticker': tk, 'fwd_ret': float(sc[tk].iloc[fi]/sc[tk].iloc[idx]-1)})
            records.append(feats)
    df = pd.DataFrame(records)
    for c in feat_cols:
        if c not in df.columns: df[c] = 0.0
    df[feat_cols] = df[feat_cols].fillna(0.0)
    if len(df) < 100: return {}
    df['rank_label'] = df.groupby('date')['fwd_ret'].rank(pct=True)
    dates = sorted(df['date'].unique()); rankings = {}
    for i in range(12, len(dates)):
        td = dates[max(0,i-12):i]; test_date = dates[i]
        tr = df[df['date'].isin(td)]; te = df[df['date']==test_date].copy()
        if len(te) < 3 or len(tr) < 50: continue
        Xt = np.nan_to_num(tr[feat_cols].values.astype(np.float32))
        yt = tr['rank_label'].values.astype(np.float32)
        Xe = np.nan_to_num(te[feat_cols].values.astype(np.float32))
        try:
            m = lgb.LGBMRegressor(n_estimators=100, max_depth=4, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, min_child_samples=5, verbose=-1)
            m.fit(Xt, yt)
            te['score'] = m.predict(Xe)
            rankings[test_date] = dict(zip(te['ticker'], te['score']))
        except: continue
    fprint(f"    {len(rankings)} ranking dates")
    return rankings

# ==================== CONFLUENCE GATE (HC #750) ====================
def confluence_check(tk, dt, sc, spy, vix, mode='bull', seasonality_cache=None, level='standard'):
    """Multi-signal confluence. Returns (passes, n_signals, signal_list)."""
    signals = []
    idx = sc.index.get_indexer([dt], method='ffill')[0]
    if idx < 63: return False, 0, []
    px = sc[tk].iloc[:idx+1].dropna()
    if len(px) < 63: return False, 0, []

    # Signal 1: Momentum (21d return in direction)
    ret_21d = float(px.iloc[-1]/px.iloc[-21]-1)
    if mode == 'bull' and ret_21d > 0: signals.append('mom_21d')
    elif mode == 'bear' and ret_21d < 0: signals.append('mom_21d_neg')

    # Signal 2: Trend (price vs 50-SMA)
    sma50 = px.rolling(50).mean()
    if not pd.isna(sma50.iloc[-1]):
        if mode == 'bull' and px.iloc[-1] > sma50.iloc[-1]: signals.append('above_sma50')
        elif mode == 'bear' and px.iloc[-1] < sma50.iloc[-1]: signals.append('below_sma50')

    # Signal 3: Relative strength vs SPY
    spy_s = spy.iloc[:idx+1]
    if len(spy_s) > 21:
        rel = px / spy_s
        rel_ret = float(rel.iloc[-1]/rel.iloc[-21]-1)
        if mode == 'bull' and rel_ret > 0: signals.append('rel_str_pos')
        elif mode == 'bear' and rel_ret < 0: signals.append('rel_str_neg')

    # Signal 4: RSI check (not overbought/oversold depending on direction)
    rets = px.pct_change().dropna()
    if len(rets) > 14:
        gains = rets.clip(lower=0).rolling(14).mean()
        losses = (-rets.clip(upper=0)).rolling(14).mean()
        rs = gains/(losses+1e-10); rsi = 100-100/(1+rs)
        r = float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else 50
        if mode == 'bull' and r < 80: signals.append('rsi_ok')
        elif mode == 'bear' and r > 20: signals.append('rsi_ok')

    # Signal 5: VIX context
    cv = float(vix.loc[dt]) if dt in vix.index else 20
    if mode == 'bull' and cv >= 18: signals.append('vix_premium')
    elif mode == 'bear' and cv < 22: signals.append('vix_calm')

    # Signal 6: Momentum acceleration
    if len(px) > 63:
        ret_63d = float(px.iloc[-1]/px.iloc[-63]-1)
        accel = ret_21d - ret_63d/3
        if mode == 'bull' and accel > 0: signals.append('mom_accel_pos')
        elif mode == 'bear' and accel < 0: signals.append('mom_accel_neg')

    # Signal 7: Seasonality (NEW)
    if seasonality_cache:
        if mode == 'bull' and is_seasonal_top(tk, dt, seasonality_cache, top_n=4):
            signals.append('seasonal_top')
        elif mode == 'bear' and is_seasonal_bottom(tk, dt, seasonality_cache, bottom_n=4):
            signals.append('seasonal_bottom')

    min_signals = 2 if level == 'standard' else 3
    return len(signals) >= min_signals, len(signals), signals

# ==================== OPTIONS PRICING ====================
def compute_atr(h, l, c, period=14):
    tr = pd.DataFrame({'hl': h-l, 'hc': abs(h-c.shift(1)), 'lc': abs(l-c.shift(1))}).max(axis=1)
    return tr.rolling(period).mean()

def atr_premium(S, K, dte, atr, vix_val, opt='call'):
    T = dte/252.0
    if T <= 0: return max(0, S-K) if opt=='call' else max(0, K-S)
    intrinsic = max(0, S-K) if opt=='call' else max(0, K-S)
    vol_factor = max(0.3, vix_val/20.0)
    return intrinsic + atr*np.sqrt(T)*vol_factor*np.exp(-3.0*abs(S-K)/S)

# ==================== SIMULATION ====================
def simulate(name, rankings, sc, sh, sl, spy, vix,
             spread_pct=3.0, dte=30, top_k=3,
             vix_min=None, vix_max=None,
             use_confluence=True, confluence_level='standard',
             mode='bull', early_exit_day=20,
             seasonality_by_date=None,
             seasonal_filter=False, seasonal_anti=False,
             seasonal_boost=False, mom_accel_required=False):

    sma200 = spy.rolling(200).mean()
    atr_d = {tk: compute_atr(sh[tk], sl[tk], sc[tk]) for tk in sc.columns if tk in sh.columns}
    equity, trades, eq_curve = CAP, [], [CAP]

    for dt in sorted(rankings.keys()):
        if dt not in spy.index or dt not in vix.index: continue
        cv = float(vix.loc[dt])

        # VIX filter
        if vix_min is not None and cv < vix_min: continue
        if vix_max is not None and cv > vix_max: continue

        scores = rankings[dt]
        if not scores: continue

        # Get seasonality for this date
        seasonal_cache = seasonality_by_date.get(dt, {}) if seasonality_by_date else {}

        if mode == 'bull':
            ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        else:  # bear
            ranked = sorted(scores.items(), key=lambda x: x[1])
        picks = [t for t, _ in ranked[:top_k]]

        max_pos = min(200, equity/3)
        if max_pos < 30: eq_curve.append(equity); continue

        n_ent = 0
        for tk in picks:
            if tk not in sc.columns or tk not in atr_d or n_ent >= 3: continue

            # Seasonal filter
            if seasonal_filter:
                if mode == 'bull' and not is_seasonal_top(tk, dt, seasonal_cache, top_n=4):
                    continue
                if mode == 'bear' and not is_seasonal_bottom(tk, dt, seasonal_cache, bottom_n=4):
                    continue

            # Anti-seasonal (contrarian)
            if seasonal_anti:
                if mode == 'bull' and not is_seasonal_bottom(tk, dt, seasonal_cache, bottom_n=4):
                    continue

            # Momentum acceleration required
            if mom_accel_required:
                idx2 = sc.index.get_indexer([dt], method='ffill')[0]
                px2 = sc[tk].iloc[:idx2+1].dropna()
                if len(px2) > 63:
                    r21 = float(px2.iloc[-1]/px2.iloc[-21]-1)
                    r63 = float(px2.iloc[-1]/px2.iloc[-63]-1)
                    accel = r21 - r63/3
                    if mode == 'bull' and accel <= 0: continue
                    if mode == 'bear' and accel >= 0: continue

            # Confluence gate
            if use_confluence:
                passes, n_sig, sigs = confluence_check(
                    tk, dt, sc, spy, vix, mode=mode,
                    seasonality_cache=seasonal_cache, level=confluence_level)
                if not passes: continue

            S = float(sc[tk].loc[dt])
            av = float(atr_d[tk].loc[dt]) if dt in atr_d[tk].index and not pd.isna(atr_d[tk].loc[dt]) else S*0.015
            di = sc.index.get_loc(dt)
            ei = min(di + early_exit_day, len(sc)-1)

            if mode == 'bull':
                K1, K2 = round(S), round(S*(1+spread_pct/100))
                lp = atr_premium(S, K1, dte, av, cv, 'call')*(1+HAIRCUT)
                sp = atr_premium(S, K2, dte, av, cv, 'call')*(1-HAIRCUT)
                val = lp - sp; width = K2 - K1
            else:  # bear put spread
                K1, K2 = round(S*(1-spread_pct/100)), round(S)
                lp = atr_premium(S, K2, dte, av, cv, 'put')*(1+HAIRCUT)
                sp = atr_premium(S, K1, dte, av, cv, 'put')*(1-HAIRCUT)
                val = lp - sp; width = K2 - K1

            cost = val*100 + SPREAD_COMM; mx_prof = (width-val)*100 - SPREAD_COMM
            if cost <= 0 or cost > max_pos or cost > equity*0.40: continue

            # Seasonal score boost for ranking adjustment
            boost = 0
            if seasonal_boost and seasonal_cache:
                sf = get_seasonal_features(tk, dt, seasonal_cache, sc)
                boost = sf['seasonal_z'] * 0.1  # Small boost

            pnl = None
            for ci in range(di+1, ei+1):
                if ci >= len(sc): break
                Sc = float(sc[tk].iloc[ci])
                dh = ci - di
                rd = max(0, dte - dh)
                ac = float(atr_d[tk].iloc[ci]) if ci < len(atr_d[tk]) else av
                tm = np.sqrt(rd/max(dte,1))

                if mode == 'bull':
                    intrinsic = (max(0, Sc-K1) - max(0, Sc-K2)) * 100
                else:
                    intrinsic = (max(0, K2-Sc) - max(0, K1-Sc)) * 100
                time_val = ac * tm * 0.3 * 100
                current_val = intrinsic + time_val
                cp = current_val - cost

                if cp >= mx_prof * 0.50:
                    pnl = cp; break
                if ci == ei:
                    pnl = cp; break

            if pnl is None:
                Se = float(sc[tk].iloc[ei])
                if mode == 'bull':
                    pnl = (max(0, Se-K1) - max(0, Se-K2))*100 - cost
                else:
                    pnl = (max(0, K2-Se) - max(0, K1-Se))*100 - cost

            equity += pnl
            trades.append({'date': str(dt.date()), 'ticker': tk, 'mode': mode,
                           'cost': round(cost,2), 'pnl': round(pnl,2),
                           'equity': round(equity,2)})
            eq_curve.append(equity)
            n_ent += 1

    return equity, trades, eq_curve

# ==================== ADVERSARIAL GATES ====================
def permutation_test(equity_real, sc, sh, sl, spy, vix, rankings,
                     n_perms=200, **sim_kwargs):
    """Randomize sector selection to test if LGBM ranking adds value."""
    fprint(f"    Permutation test ({n_perms} perms)...")
    random_equities = []
    for i in range(n_perms):
        rand_rankings = {}
        for dt, scores in rankings.items():
            tickers = list(scores.keys())
            np.random.shuffle(tickers)
            rand_rankings[dt] = dict(zip(tickers, scores.values()))
        eq, _, _ = simulate(f"perm_{i}", rand_rankings, sc, sh, sl, spy, vix, **sim_kwargs)
        random_equities.append(eq)
    p_val = np.mean([re >= equity_real for re in random_equities])
    return p_val, np.mean(random_equities), np.std(random_equities)

def compute_metrics(trades, eq_curve):
    if not trades: return {'sharpe': 0, 'cagr': 0, 'mdd': -1, 'wr': 0, 'pf': 0, 'sortino': 0, 'n_trades': 0}
    df = pd.DataFrame(trades)
    df['date'] = pd.to_datetime(df['date'])

    # Monthly PnL for Sharpe
    df['month'] = df['date'].dt.to_period('M')
    monthly = df.groupby('month')['pnl'].sum()
    eq_at_month = df.groupby('month')['equity'].last()
    monthly_ret = monthly / (eq_at_month.shift(1).fillna(CAP))

    sharpe = float(monthly_ret.mean() / (monthly_ret.std()+1e-10) * np.sqrt(12))

    # Sortino
    downside = monthly_ret[monthly_ret < 0]
    sortino = float(monthly_ret.mean() / (downside.std()+1e-10) * np.sqrt(12)) if len(downside) > 1 else sharpe

    # CAGR
    final_eq = eq_curve[-1]
    years = len(monthly) / 12
    cagr = (final_eq/CAP)**(1/max(years,0.5)) - 1 if final_eq > 0 else -1

    # MaxDD
    peak = pd.Series(eq_curve).cummax()
    dd = (pd.Series(eq_curve) - peak) / peak
    mdd = float(dd.min())

    # Win rate and profit factor
    wins = df[df['pnl'] > 0]
    losses = df[df['pnl'] <= 0]
    wr = len(wins) / len(df) if len(df) > 0 else 0
    pf = abs(wins['pnl'].sum()) / (abs(losses['pnl'].sum()) + 1e-10) if len(losses) > 0 else 99

    # Calmar
    calmar = cagr / (abs(mdd) + 1e-10)

    return {
        'sharpe': round(sharpe, 2), 'sortino': round(sortino, 2),
        'cagr': round(cagr*100, 1), 'mdd': round(mdd*100, 1),
        'wr': round(wr*100, 1), 'pf': round(pf, 2),
        'calmar': round(calmar, 1),
        'n_trades': len(df), 'final_equity': round(final_eq, 0)
    }

def regime_r1_check(trades, spy):
    """Check Sharpe gap between bull and bear market regimes."""
    if not trades: return 1.0, 0, 0
    df = pd.DataFrame(trades)
    df['date'] = pd.to_datetime(df['date'])

    spy_monthly = spy.resample('ME').last().pct_change()

    bull_pnl, bear_pnl = [], []
    for _, row in df.iterrows():
        dt = row['date']
        closest = spy_monthly.index[spy_monthly.index.get_indexer([dt], method='ffill')[0]]
        if spy_monthly.loc[closest] >= 0:
            bull_pnl.append(row['pnl'])
        else:
            bear_pnl.append(row['pnl'])

    if not bull_pnl or not bear_pnl: return 1.0, 0, 0

    bull_sr = np.mean(bull_pnl) / (np.std(bull_pnl) + 1e-10)
    bear_sr = np.mean(bear_pnl) / (np.std(bear_pnl) + 1e-10)

    gap = abs(bull_sr - bear_sr) / (max(abs(bull_sr), abs(bear_sr)) + 1e-10)
    return round(gap, 3), round(bull_sr, 3), round(bear_sr, 3)

def subperiod_check(trades):
    """Split trades into halves and check stability."""
    if len(trades) < 20: return False, 0, 0
    df = pd.DataFrame(trades)
    mid = len(df) // 2
    h1_wr = (df.iloc[:mid]['pnl'] > 0).mean()
    h2_wr = (df.iloc[mid:]['pnl'] > 0).mean()
    return abs(h1_wr - h2_wr) < 0.15, round(h1_wr, 3), round(h2_wr, 3)

def outlier_check(trades):
    """Remove top 5% trades and check profitability."""
    if len(trades) < 20: return False, 0
    df = pd.DataFrame(trades)
    thresh = df['pnl'].quantile(0.95)
    trimmed = df[df['pnl'] <= thresh]
    trimmed_pf = trimmed[trimmed['pnl'] > 0]['pnl'].sum() / (abs(trimmed[trimmed['pnl'] <= 0]['pnl'].sum()) + 1e-10)
    return trimmed_pf > 1.0, round(trimmed_pf, 2)

# ==================== MAIN ====================
def main():
    fprint("=" * 70)
    fprint("SEASONAL SECTOR MOMENTUM v1 — Multi-Signal Confluence + Seasonality")
    fprint("=" * 70)

    sc, sh, sl, spy, vix = download_data()

    # Generate biweekly rebalance dates
    all_dates = sc.index[sc.index >= '2010-01-01']
    rebal_dates = all_dates[::10]  # ~biweekly
    fprint(f"Rebalance dates: {len(rebal_dates)}")

    # Pre-compute seasonality for each rebal date
    fprint("Computing seasonal patterns...")
    seasonality_by_date = {}
    for dt in rebal_dates:
        seasonality_by_date[dt] = compute_sector_seasonality(sc, dt, lookback_years=10)
    fprint(f"  Seasonality computed for {len(seasonality_by_date)} dates")

    # Build rankings with and without seasonal features
    rankings_base = build_rankings(sc, spy, rebal_dates, MOM_COLS + QUALITY_COLS, None)
    rankings_seasonal = build_rankings(sc, spy, rebal_dates, ALL_FEAT_COLS, seasonality_by_date)

    if MLFLOW_OK:
        exp = mlflow.set_experiment("seasonal_sector_momentum_v1")

    # ==================== VARIANTS ====================
    variants = {
        'A_Baseline': {
            'desc': 'Bull+Bear v1 (no seasonality) — control',
            'rankings_type': 'base',
            'seasonal_filter': False, 'seasonal_anti': False,
            'seasonal_boost': False, 'mom_accel_required': False,
            'confluence_level': 'standard',
        },
        'B_SeasonalFilter': {
            'desc': 'Only trade sectors in their top 4 seasonal months',
            'rankings_type': 'base',
            'seasonal_filter': True, 'seasonal_anti': False,
            'seasonal_boost': False, 'mom_accel_required': False,
            'confluence_level': 'standard',
        },
        'C_SeasonalBoost': {
            'desc': 'LGBM score + seasonal z-score boost in composite ranking',
            'rankings_type': 'seasonal',
            'seasonal_filter': False, 'seasonal_anti': False,
            'seasonal_boost': True, 'mom_accel_required': False,
            'confluence_level': 'standard',
        },
        'D_SeasonalAccel': {
            'desc': 'Seasonal filter + momentum acceleration required',
            'rankings_type': 'seasonal',
            'seasonal_filter': True, 'seasonal_anti': False,
            'seasonal_boost': False, 'mom_accel_required': True,
            'confluence_level': 'standard',
        },
        'E_AntiSeasonal': {
            'desc': 'Contrarian: trade worst seasonal months when momentum strong',
            'rankings_type': 'base',
            'seasonal_filter': False, 'seasonal_anti': True,
            'seasonal_boost': False, 'mom_accel_required': False,
            'confluence_level': 'standard',
        },
        'F_FullConfluence': {
            'desc': 'Full confluence: seasonality + mom + quality + RSI + VIX (min 3 signals)',
            'rankings_type': 'seasonal',
            'seasonal_filter': False, 'seasonal_anti': False,
            'seasonal_boost': True, 'mom_accel_required': False,
            'confluence_level': 'high',  # min 3 signals
        },
    }

    results = {}
    for vname, cfg in variants.items():
        fprint(f"\n{'='*60}")
        fprint(f"Variant {vname}: {cfg['desc']}")
        fprint(f"{'='*60}")

        rnk = rankings_seasonal if cfg['rankings_type'] == 'seasonal' else rankings_base

        common_kwargs = dict(
            spread_pct=3.0, dte=30, top_k=3,
            use_confluence=True, early_exit_day=20,
            seasonality_by_date=seasonality_by_date,
            seasonal_filter=cfg['seasonal_filter'],
            seasonal_anti=cfg['seasonal_anti'],
            seasonal_boost=cfg['seasonal_boost'],
            mom_accel_required=cfg['mom_accel_required'],
            confluence_level=cfg['confluence_level'],
        )

        # Bull side (VIX >= 20)
        eq_bull, trades_bull, curve_bull = simulate(
            f"{vname}_bull", rnk, sc, sh, sl, spy, vix,
            vix_min=20, mode='bull', **common_kwargs)

        # Bear side (VIX < 20)
        eq_bear, trades_bear, curve_bear = simulate(
            f"{vname}_bear", rnk, sc, sh, sl, spy, vix,
            vix_max=20, mode='bear', **common_kwargs)

        # Combined equity curve
        all_trades = trades_bull + trades_bear
        all_trades.sort(key=lambda x: x['date'])

        # Rebuild combined equity curve
        combined_equity = CAP
        combined_curve = [CAP]
        for t in all_trades:
            combined_equity += t['pnl']
            t['equity'] = round(combined_equity, 2)
            combined_curve.append(combined_equity)

        metrics = compute_metrics(all_trades, combined_curve)
        fprint(f"  Trades: {metrics['n_trades']} (bull {len(trades_bull)}, bear {len(trades_bear)})")
        fprint(f"  Sharpe: {metrics['sharpe']}, CAGR: {metrics['cagr']}%, MDD: {metrics['mdd']}%")
        fprint(f"  WR: {metrics['wr']}%, PF: {metrics['pf']}, Sortino: {metrics['sortino']}")
        fprint(f"  Final equity: ${metrics['final_equity']}")

        # === ADVERSARIAL GATES ===
        gates = {'perm': False, 'r1': False, 'subperiod': False, 'outlier': False}

        # Gate 1: Permutation
        if metrics['n_trades'] >= 20:
            p_val, rand_mean, rand_std = permutation_test(
                combined_curve[-1], sc, sh, sl, spy, vix, rnk, n_perms=200,
                spread_pct=3.0, dte=30, top_k=3,
                use_confluence=True, early_exit_day=20, mode='bull',
                vix_min=20, seasonality_by_date=seasonality_by_date,
                seasonal_filter=cfg['seasonal_filter'], seasonal_anti=cfg['seasonal_anti'],
                seasonal_boost=cfg['seasonal_boost'], mom_accel_required=cfg['mom_accel_required'],
                confluence_level=cfg['confluence_level'])
            gates['perm'] = p_val < 0.05
            fprint(f"  Perm: p={p_val:.3f} (rand=${rand_mean:.0f}) → {'PASS' if gates['perm'] else 'FAIL'}")
            rand_sharpe = 0
            if rand_mean > CAP:
                rand_sharpe = round((rand_mean/CAP - 1) / (rand_std/CAP + 1e-10), 2)
        else:
            p_val, rand_mean = 1.0, CAP
            rand_sharpe = 0
            fprint(f"  Perm: SKIP (too few trades)")

        # Gate 2: R1 regime check
        r1_gap, bull_sr, bear_sr = regime_r1_check(all_trades, spy)
        gates['r1'] = r1_gap < 0.50
        fprint(f"  R1: gap={r1_gap} (bull_sr={bull_sr}, bear_sr={bear_sr}) → {'PASS' if gates['r1'] else 'FAIL'}")

        # Gate 3: Sub-period stability
        sp_ok, h1_wr, h2_wr = subperiod_check(all_trades)
        gates['subperiod'] = sp_ok
        fprint(f"  SubPeriod: h1_wr={h1_wr}, h2_wr={h2_wr} → {'PASS' if gates['subperiod'] else 'FAIL'}")

        # Gate 4: Outlier robustness
        out_ok, trimmed_pf = outlier_check(all_trades)
        gates['outlier'] = out_ok
        fprint(f"  Outlier: trimmed_PF={trimmed_pf} → {'PASS' if gates['outlier'] else 'FAIL'}")

        gates_passed = sum(gates.values())
        fprint(f"  GATES: {gates_passed}/4 {'✅ PASS' if gates_passed == 4 else '❌'}")

        results[vname] = {
            'desc': cfg['desc'],
            'metrics': metrics,
            'gates': gates,
            'gates_passed': gates_passed,
            'perm_p': round(p_val, 3),
            'rand_equity': round(rand_mean, 0),
            'r1_gap': r1_gap,
            'bull_trades': len(trades_bull),
            'bear_trades': len(trades_bear),
        }

        # MLflow logging
        if MLFLOW_OK:
            with mlflow.start_run(run_name=vname):
                mlflow.log_params({
                    'variant': vname, 'seasonal_filter': cfg['seasonal_filter'],
                    'seasonal_anti': cfg['seasonal_anti'], 'seasonal_boost': cfg['seasonal_boost'],
                    'mom_accel': cfg['mom_accel_required'], 'confluence_level': cfg['confluence_level'],
                })
                mlflow.log_metrics({
                    'sharpe': metrics['sharpe'], 'sortino': metrics['sortino'],
                    'cagr': metrics['cagr'], 'mdd': metrics['mdd'],
                    'wr': metrics['wr'], 'pf': metrics['pf'],
                    'n_trades': metrics['n_trades'], 'final_equity': metrics['final_equity'],
                    'gates_passed': gates_passed, 'perm_p': p_val,
                    'r1_gap': r1_gap,
                })

    # ==================== SUMMARY ====================
    fprint(f"\n{'='*70}")
    fprint("SUMMARY — SEASONAL SECTOR MOMENTUM v1")
    fprint(f"{'='*70}")
    fprint(f"{'Variant':<25} {'Sharpe':>7} {'CAGR':>7} {'MDD':>7} {'WR':>6} {'Trades':>7} {'Gates':>6} {'Final$':>8}")
    fprint("-" * 75)
    for vname, r in sorted(results.items()):
        m = r['metrics']
        fprint(f"{vname:<25} {m['sharpe']:>7.2f} {m['cagr']:>6.1f}% {m['mdd']:>6.1f}% {m['wr']:>5.1f}% {m['n_trades']:>7} {r['gates_passed']:>4}/4 ${m['final_equity']:>7.0f}")

    # Save results
    with open(RESULTS_PATH, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    fprint(f"\nResults saved to {RESULTS_PATH}")

    # Determine verdict
    baseline = results.get('A_Baseline', {}).get('metrics', {})
    best_seasonal = None
    best_sharpe = 0
    for vname, r in results.items():
        if vname == 'A_Baseline': continue
        if r['gates_passed'] >= 4 and r['metrics']['sharpe'] > best_sharpe:
            best_sharpe = r['metrics']['sharpe']
            best_seasonal = vname

    fprint(f"\nBaseline Sharpe: {baseline.get('sharpe', 0)}")
    if best_seasonal:
        bs = results[best_seasonal]['metrics']
        improvement = (bs['sharpe'] - baseline.get('sharpe', 0)) / (baseline.get('sharpe', 1)) * 100
        fprint(f"Best seasonal variant: {best_seasonal} (Sharpe {bs['sharpe']}, {improvement:+.1f}% vs baseline)")
        if improvement > 5:
            fprint("VERDICT: Seasonality IMPROVES strategy → ADOPT")
        else:
            fprint("VERDICT: Seasonality is MARGINAL → baseline is fine")
    else:
        fprint("VERDICT: No seasonal variant passes all gates → seasonality adds NO value")

    fprint("\nDONE")

if __name__ == '__main__':
    main()
