#!/usr/bin/env python3
"""Multi-Factor Sector Options v1 — Extend sector rotation with quality+value+flow factors.

Our best strategy is bi-weekly sector bull spreads (Sharpe 3.76, momentum-only ranking).
This tests whether adding quality, value, and breadth factors improves selection.

Variants:
  A: Momentum-only baseline (replicate F_BiWeekly from rotation v1)
  B: Multi-factor (mom + quality + value + vol-adjusted)
  C: Quality-weighted momentum (momentum * quality score)
  D: Contrarian-value (buy beaten-down sectors with quality)
  E: Adaptive (switch factor weights based on VIX regime)
  F: Ensemble (average ranks across all factor models)
  G: Top-2 concentrated (fewer positions, higher conviction)
"""
import json, sys, os, numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')
from pathlib import Path
from datetime import datetime
from scipy import stats
import lightgbm as lgb

BASE = Path(__file__).resolve().parents[2]
RESULTS_DIR = BASE / 'research' / 'findings'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_PATH = RESULTS_DIR / 'multifactor_sector_options_v1_results.json'
def fprint(*a, **kw): print(*a, **kw, flush=True)

MLFLOW_OK = False
try:
    import urllib.request; urllib.request.urlopen('http://jupiter:5000/', timeout=2)
    import mlflow; mlflow.set_tracking_uri('http://jupiter:5000'); MLFLOW_OK = True
except Exception: fprint("MLflow unavailable — will log locally only")

# Sector ETFs + benchmark
SECTORS = ['XLK','XLF','XLE','XLV','XLY','XLP','XLI','XLB','XLU','XLRE','XLC']
CAP = 645.0
LEG_COMM = 0.65
SPREAD_COMM = 4 * LEG_COMM  # $2.60 round-trip
HAIRCUT = 0.15  # 15% bid-ask haircut

# ==================== DATA ====================
def download_data():
    import yfinance as yf
    fprint("Downloading sector + macro data...")
    tickers = SECTORS + ['SPY', '^VIX', '^TNX']  # Add 10yr yield for value context
    raw = yf.download(tickers, start='2008-01-01', end='2026-07-26', progress=False)
    mi = isinstance(raw.columns, pd.MultiIndex)
    close = raw['Close'] if mi else raw
    high = raw['High'] if mi else raw
    low = raw['Low'] if mi else raw
    volume = raw['Volume'] if mi else raw

    vc = '^VIX' if '^VIX' in close.columns else 'VIX'
    vix = close[vc].dropna()
    spy = close['SPY'].dropna()
    tnx_col = '^TNX' if '^TNX' in close.columns else None
    tnx = close[tnx_col].dropna() if tnx_col and tnx_col in close.columns else None

    sc = close[[c for c in SECTORS if c in close.columns]].dropna(how='all')
    sh = high[[c for c in SECTORS if c in high.columns]].dropna(how='all')
    sl = low[[c for c in SECTORS if c in low.columns]].dropna(how='all')
    sv = volume[[c for c in SECTORS if c in volume.columns]].dropna(how='all')

    ix = sc.index.intersection(vix.index).intersection(spy.index)
    ix = ix.intersection(sh.index).intersection(sl.index)
    fprint(f"Data: {len(ix)} days, {len(sc.columns)} sectors")
    return sc.loc[ix], sh.loc[ix], sl.loc[ix], sv.reindex(ix), spy.loc[ix], vix.loc[ix], tnx

# ==================== FEATURE ENGINEERING ====================
def compute_momentum_features(px, idx):
    """Standard momentum features (baseline)."""
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
    return f

def compute_quality_features(px, vol_data=None):
    """Quality features: consistency, risk-adjusted momentum, drawdown recovery."""
    if len(px) < 260: return {}
    rets = px.pct_change().dropna()
    f = {}
    # Consistency: % of positive months in last year
    monthly = rets.resample('ME').sum()
    f['pct_pos_months_12m'] = float((monthly.iloc[-12:] > 0).mean()) if len(monthly) >= 12 else 0.5
    # Sortino ratio (downside deviation)
    r63 = rets.iloc[-63:]
    dr = r63[r63 < 0]
    f['sortino_63d'] = float(r63.mean()/(dr.std()+1e-10)*np.sqrt(252)) if len(dr) > 3 else 0.0
    # Calmar ratio
    pk = px.iloc[-252:].cummax()
    mdd = float(((px.iloc[-252:]/pk)-1).min())
    cagr = float(px.iloc[-1]/px.iloc[-252]-1) if len(px) >= 252 else 0.0
    f['calmar_1y'] = cagr / (abs(mdd) + 1e-10)
    # Up/Down capture vs simple average
    avg_ret = rets.mean()
    up_days = rets[rets > 0]
    dn_days = rets[rets < 0]
    f['up_capture'] = float(up_days.iloc[-63:].mean() / (up_days.mean()+1e-10)) if len(up_days) > 10 else 1.0
    f['dn_capture'] = float(dn_days.iloc[-63:].mean() / (dn_days.mean()+1e-10)) if len(dn_days) > 10 else 1.0
    # Trend strength (R-squared of price vs time)
    if len(px) >= 63:
        y = np.log(px.iloc[-63:].values + 1e-10)
        x = np.arange(len(y))
        slope, _, r_val, _, _ = stats.linregress(x, y)
        f['trend_r2_63d'] = r_val**2
        f['trend_slope_63d'] = slope * 252  # annualized
    else:
        f['trend_r2_63d'] = 0.0
        f['trend_slope_63d'] = 0.0
    # Volume trend (relative volume)
    if vol_data is not None and len(vol_data) >= 63:
        f['rel_vol_21d'] = float(vol_data.iloc[-21:].mean() / (vol_data.iloc[-63:].mean()+1e-10))
    else:
        f['rel_vol_21d'] = 1.0
    return f

def compute_value_features(px):
    """Value features: reversion signals, distance from fair value estimates."""
    if len(px) < 260: return {}
    f = {}
    # Distance from 200-day SMA (mean reversion potential)
    sma200 = px.rolling(200).mean()
    f['dist_sma200'] = float(px.iloc[-1] / sma200.iloc[-1] - 1) if not pd.isna(sma200.iloc[-1]) else 0.0
    # Distance from 50-day SMA
    sma50 = px.rolling(50).mean()
    f['dist_sma50'] = float(px.iloc[-1] / sma50.iloc[-1] - 1) if not pd.isna(sma50.iloc[-1]) else 0.0
    # RSI (14-day)
    rets = px.pct_change().dropna()
    gains = rets.clip(lower=0).rolling(14).mean()
    losses = (-rets.clip(upper=0)).rolling(14).mean()
    rs = gains / (losses + 1e-10)
    rsi = 100 - 100 / (1 + rs)
    f['rsi_14'] = float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else 50.0
    # Bollinger Band position
    sma20 = px.rolling(20).mean()
    std20 = px.rolling(20).std()
    if not pd.isna(sma20.iloc[-1]) and not pd.isna(std20.iloc[-1]) and std20.iloc[-1] > 0:
        f['bb_pct'] = float((px.iloc[-1] - sma20.iloc[-1]) / (2 * std20.iloc[-1]))
    else:
        f['bb_pct'] = 0.0
    # 1-year z-score of price
    if len(px) >= 252:
        f['price_zscore'] = float((px.iloc[-1] - px.iloc[-252:].mean()) / (px.iloc[-252:].std()+1e-10))
    else:
        f['price_zscore'] = 0.0
    return f

def compute_breadth_features(sector_px, spy_px, idx):
    """Cross-sector breadth features."""
    f = {}
    # Sector-vs-SPY relative strength
    if len(sector_px) >= 63 and len(spy_px) >= 63:
        rel = sector_px / spy_px
        f['rel_str_21d'] = float(rel.iloc[-1] / rel.iloc[-21] - 1) if len(rel) > 21 else 0.0
        f['rel_str_63d'] = float(rel.iloc[-1] / rel.iloc[-63] - 1) if len(rel) > 63 else 0.0
    else:
        f['rel_str_21d'] = 0.0
        f['rel_str_63d'] = 0.0
    return f

# ==================== LGBM VARIANTS ====================
MOM_COLS = ['ret_5d','ret_10d','ret_21d','ret_63d','ret_126d','ret_252d',
            'vol_21d','vol_63d','sharpe_63d','maxdd_63d','pct_52w_high','mom_accel']

QUALITY_COLS = ['pct_pos_months_12m','sortino_63d','calmar_1y','up_capture','dn_capture',
                'trend_r2_63d','trend_slope_63d','rel_vol_21d']

VALUE_COLS = ['dist_sma200','dist_sma50','rsi_14','bb_pct','price_zscore']

BREADTH_COLS = ['rel_str_21d','rel_str_63d']

ALL_COLS = MOM_COLS + QUALITY_COLS + VALUE_COLS + BREADTH_COLS

def build_dataset(sc, sv, spy, rebal_dates, feature_set='momentum'):
    """Build training dataset with specified feature set."""
    fprint(f"  Building dataset ({feature_set})...")
    records = []
    col_map = {
        'momentum': MOM_COLS,
        'multi': ALL_COLS,
        'quality_mom': MOM_COLS + QUALITY_COLS,
        'value': VALUE_COLS + MOM_COLS[:4],  # value + short-term momentum
    }
    use_cols = col_map.get(feature_set, ALL_COLS)

    for dt in rebal_dates:
        idx = sc.index.get_indexer([dt], method='ffill')[0]
        if idx < 260: continue
        for tk in sc.columns:
            px = sc[tk].iloc[:idx+1].dropna()
            if len(px) < 260: continue

            feats = compute_momentum_features(px, idx)
            if feats is None: continue

            vol_d = sv[tk].iloc[:idx+1] if tk in sv.columns else None
            feats.update(compute_quality_features(px, vol_d))
            feats.update(compute_value_features(px))
            feats.update(compute_breadth_features(px, spy.iloc[:idx+1], idx))

            fi = min(idx+14, len(sc)-1)  # 2-week forward return (bi-weekly)
            feats.update({
                'date': dt, 'ticker': tk,
                'fwd_ret': float(sc[tk].iloc[fi]/sc[tk].iloc[idx]-1)
            })
            records.append(feats)

    df = pd.DataFrame(records)
    # Fill NaN features with 0
    for c in use_cols:
        if c not in df.columns:
            df[c] = 0.0
    df[use_cols] = df[use_cols].fillna(0.0)
    return df, use_cols

def lgbm_wf(df, feat_cols, train_periods=12, variant_name=''):
    """Walk-forward LightGBM ranking."""
    fprint(f"  LGBM walk-forward ({variant_name}, {len(feat_cols)} features)...")
    df['rank_label'] = df.groupby('date')['fwd_ret'].rank(pct=True)
    dates = sorted(df['date'].unique())
    rankings = {}

    for i in range(train_periods, len(dates)):
        td = dates[max(0, i-train_periods):i]
        test_date = dates[i]
        tr = df[df['date'].isin(td)]
        te = df[df['date']==test_date].copy()
        if len(te) < 3 or len(tr) < 50: continue

        Xt = np.nan_to_num(tr[feat_cols].values.astype(np.float32))
        yt = tr['rank_label'].values.astype(np.float32)
        Xe = np.nan_to_num(te[feat_cols].values.astype(np.float32))

        try:
            try:
                m = lgb.LGBMRegressor(n_estimators=100, max_depth=4, learning_rate=0.05,
                    subsample=0.8, colsample_bytree=0.8, min_child_samples=5, device='gpu', verbose=-1)
                m.fit(Xt, yt)
            except Exception:
                m = lgb.LGBMRegressor(n_estimators=100, max_depth=4, learning_rate=0.05,
                    subsample=0.8, colsample_bytree=0.8, min_child_samples=5, verbose=-1)
                m.fit(Xt, yt)
            te['score'] = m.predict(Xe)
            rankings[test_date] = dict(zip(te['ticker'], te['score']))
        except Exception as e:
            continue

    fprint(f"    Rankings for {len(rankings)} dates")
    return rankings

def ensemble_rankings(*ranking_dicts):
    """Average rank across multiple models."""
    all_dates = set()
    for rd in ranking_dicts:
        all_dates.update(rd.keys())

    combined = {}
    for dt in sorted(all_dates):
        scores = {}
        for rd in ranking_dicts:
            if dt not in rd: continue
            for tk, s in rd[dt].items():
                if tk not in scores:
                    scores[tk] = []
                scores[tk].append(s)
        if scores:
            combined[dt] = {tk: np.mean(vals) for tk, vals in scores.items()}
    return combined

def contrarian_rankings(rankings):
    """Flip rankings — buy lowest-ranked (beaten-down) sectors."""
    flipped = {}
    for dt, scores in rankings.items():
        if scores:
            max_s = max(scores.values())
            flipped[dt] = {tk: max_s - s for tk, s in scores.items()}
    return flipped

def adaptive_rankings(rankings_mom, rankings_multi, vix, spy):
    """Switch between momentum and multi-factor based on VIX regime."""
    sma200 = spy.rolling(200).mean()
    combined = {}
    all_dates = set(rankings_mom.keys()) | set(rankings_multi.keys())

    for dt in sorted(all_dates):
        if dt not in vix.index: continue
        v = float(vix.loc[dt])
        s = float(spy.loc[dt])
        sm = float(sma200.loc[dt]) if dt in sma200.index and not pd.isna(sma200.loc[dt]) else s

        # High VIX (>25) or bear market → use multi-factor (quality matters more)
        # Low VIX (<20) and bull → use momentum (trend works)
        # In between → blend
        if v > 25 or s < sm:
            combined[dt] = rankings_multi.get(dt, rankings_mom.get(dt, {}))
        elif v < 20 and s >= sm:
            combined[dt] = rankings_mom.get(dt, rankings_multi.get(dt, {}))
        else:
            # Blend 50/50
            mom = rankings_mom.get(dt, {})
            multi = rankings_multi.get(dt, {})
            blended = {}
            all_tk = set(mom.keys()) | set(multi.keys())
            for tk in all_tk:
                ms = mom.get(tk, 0.5)
                mts = multi.get(tk, 0.5)
                blended[tk] = 0.5 * ms + 0.5 * mts
            combined[dt] = blended
    return combined

# ==================== OPTIONS PRICING & SIMULATION ====================
def compute_atr(h, l, c, period=14):
    tr = pd.DataFrame({'hl': h-l, 'hc': abs(h-c.shift(1)), 'lc': abs(l-c.shift(1))}).max(axis=1)
    return tr.rolling(period).mean()

def atr_premium(S, K, dte, atr, vix_val, opt='call'):
    T = dte / 252.0
    if T <= 0: return max(0, S-K) if opt=='call' else max(0, K-S)
    intrinsic = max(0, S-K) if opt=='call' else max(0, K-S)
    vol_factor = max(0.3, vix_val / 20.0)
    time_prem = atr * np.sqrt(T) * vol_factor * np.exp(-3.0 * abs(S-K)/S)
    return intrinsic + time_prem

def price_bull_spread(S, spread_pct, dte, atr, vix_val):
    K1, K2 = round(S), round(S*(1+spread_pct/100))
    lp = atr_premium(S, K1, dte, atr, vix_val, 'call') * (1+HAIRCUT)
    sp = atr_premium(S, K2, dte, atr, vix_val, 'call') * (1-HAIRCUT)
    debit = lp - sp
    width = K2 - K1
    return debit, (width-debit)*100-SPREAD_COMM, debit*100+SPREAD_COMM, K1, K2

def simulate(name, rankings, sc, sh, sl, spy, vix, spread_pct=3.0, dte=30, top_k=3):
    fprint(f"\n--- {name} ---")
    sma200 = spy.rolling(200).mean()
    atr_d = {tk: compute_atr(sh[tk], sl[tk], sc[tk]) for tk in sc.columns if tk in sh.columns}
    equity, trades, eq_curve = CAP, [], [CAP]

    for dt in sorted(rankings.keys()):
        if dt not in spy.index or dt not in vix.index: continue
        sv = float(spy.loc[dt])
        sm = float(sma200.loc[dt]) if dt in sma200.index and not pd.isna(sma200.loc[dt]) else sv
        bull = sv >= sm
        cv = float(vix.loc[dt])
        scores = rankings[dt]
        if not scores: continue
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        picks = [t for t, _ in ranked[:top_k]]
        max_pos = min(200, equity/3)
        if max_pos < 30:
            eq_curve.append(equity); continue

        n_ent = 0
        for tk in picks:
            if tk not in sc.columns or tk not in atr_d or n_ent >= 3: continue
            S = float(sc[tk].loc[dt])
            av = float(atr_d[tk].loc[dt]) if dt in atr_d[tk].index and not pd.isna(atr_d[tk].loc[dt]) else S*0.015
            di = sc.index.get_loc(dt)
            ei = min(di + dte, len(sc) - 1)
            Se = float(sc[tk].iloc[ei])
            val, mx_prof, mx_loss, K1, K2 = price_bull_spread(S, spread_pct, dte, av, cv)
            cost = val * 100 + SPREAD_COMM
            if cost <= 0 or cost > max_pos or cost > equity * 0.40: continue

            # Walk forward: early exit at 50% profit or DTE<7
            pnl, aei = None, ei
            for ci in range(di+7, ei+1):
                Sc = float(sc[tk].iloc[ci])
                dh = ci - di
                rd = max(0, dte - dh)
                ac = float(atr_d[tk].iloc[ci]) if ci < len(atr_d[tk]) else av
                tm = np.sqrt(rd / max(dte, 1))
                si = (max(0, Sc-K1) - max(0, Sc-K2)) * 100 + ac * tm * 0.3 * 100
                cp = si - (val*100 + SPREAD_COMM)
                if cp >= mx_prof * 0.50 or rd < 7:
                    pnl = cp; aei = ci; break

            if pnl is None:
                pnl = (max(0, Se-K1) - max(0, Se-K2)) * 100 - val*100 - SPREAD_COMM

            equity += pnl
            n_ent += 1
            trades.append({
                'entry': str(dt.date()), 'exit': str(sc.index[aei].date()),
                'ticker': tk, 'pnl': round(pnl, 2), 'win': pnl > 0,
                'regime': 'bull' if bull else 'bear',
                'vix': round(cv, 1)
            })
        eq_curve.append(equity)
    return trades, equity, eq_curve

# ==================== VALIDATION ====================
def metrics(trades, final_eq, eq_curve, name):
    if not trades:
        fprint(f"  {name}: No trades"); return None
    n = len(trades); wins = sum(1 for t in trades if t['win']); wr = wins/n*100
    pnls = [t['pnl'] for t in trades]
    tdf = pd.DataFrame(trades)
    tdf['m'] = pd.to_datetime(tdf['entry']).dt.to_period('M')
    mr = tdf.groupby('m')['pnl'].sum() / CAP
    ny = max(len(mr)/12, 0.5)
    sh = (mr.mean()*12)/(mr.std()*np.sqrt(12)+1e-10) if len(mr) > 3 else 0
    dn = mr[mr<0]
    so = (mr.mean()*12)/(dn.std()*np.sqrt(12)+1e-10) if len(dn) > 1 else 0
    cagr = (final_eq/CAP)**(1/ny)-1
    eq = np.array(eq_curve)
    pk = np.maximum.accumulate(eq)
    mdd = float(((eq-pk)/(pk+1e-10)).min())
    gp = sum(p for p in pnls if p > 0)
    gl = abs(sum(p for p in pnls if p <= 0))
    pf = gp/(gl+1e-10)
    ap = np.mean(pnls)
    bt = [t for t in trades if t['regime']=='bull']
    brt = [t for t in trades if t['regime']=='bear']
    bw = sum(1 for t in bt if t['win'])/max(len(bt),1)*100
    brw = sum(1 for t in brt if t['win'])/max(len(brt),1)*100
    r = {
        'name': name, 'n_trades': n, 'win_rate': round(wr,1),
        'sharpe': round(sh,2), 'sortino': round(so,2),
        'cagr_pct': round(cagr*100,1), 'maxdd_pct': round(mdd*100,1),
        'pf': round(pf,2), 'avg_pnl': round(ap,2),
        'avg_trades_yr': round(n/ny,1), 'final_equity': round(final_eq,2),
        'total_pnl': round(sum(pnls),2),
        'bull_wr': round(bw,1), 'bear_wr': round(brw,1),
        'bull_n': len(bt), 'bear_n': len(brt),
        'monthly_returns': mr.values.tolist()
    }
    fprint(f"  {name}: {n} trades | WR {wr:.1f}% | Sharpe {sh:.2f} | Sort {so:.2f} | "
           f"CAGR {cagr*100:.1f}% | MaxDD {mdd*100:.1f}% | PF {pf:.2f} | ${CAP:.0f}->${final_eq:.0f}")
    return r

def validate(r):
    if r is None: return r
    rets = np.array(r['monthly_returns'])
    if len(rets) < 10:
        r.update({'gates':0,'perm_p':1.0,'r1_gap':1.0}); return r
    gates = 0
    rs = np.mean(rets)/(np.std(rets)+1e-10)

    # G1: Permutation (direction shuffle)
    n_perm = 2000
    pp = sum(1 for _ in range(n_perm) if np.mean(rets*np.random.choice([-1,1],len(rets)))/(np.std(rets)+1e-10)>=rs)/n_perm
    g1 = pp < 0.05; gates += g1

    # G2: Regime balance (R1)
    rg = abs(r['bull_wr']-r['bear_wr'])/max(r['bull_wr'],r['bear_wr'],1)
    g2 = rg < 0.50; gates += g2

    # G3: Sub-period stability
    mid = len(rets)//2
    h1 = np.mean(rets[:mid])/(np.std(rets[:mid])+1e-10) if mid > 3 else 0
    h2 = np.mean(rets[mid:])/(np.std(rets[mid:])+1e-10) if len(rets)-mid > 3 else 0
    g3 = h1 > 0 and h2 > 0; gates += g3

    # G4: Outlier removal
    g4 = False
    if len(rets) > 5:
        tr = np.sort(rets)[:-1]
        g4 = np.mean(tr)/(np.std(tr)+1e-10) > 0
    gates += g4

    r.update({
        'gates': gates, 'perm_p': round(pp,4), 'r1_gap': round(rg,3),
        'g1_perm': g1, 'g2_regime': g2, 'g3_sub': g3, 'g4_outlier': g4,
        'h1_sh': round(h1,2), 'h2_sh': round(h2,2)
    })
    fprint(f"  Gates: G1={'P' if g1 else 'F'}(p={pp:.4f}) G2={'P' if g2 else 'F'}(gap={rg:.3f}) "
           f"G3={'P' if g3 else 'F'}({h1:.2f}/{h2:.2f}) G4={'P' if g4 else 'F'} => {gates}/4")
    return r

# ==================== MAIN ====================
def main():
    t0 = datetime.now()
    fprint(f"Multi-Factor Sector Options v1 — {t0.strftime('%Y-%m-%d %H:%M:%S')}")
    fprint(f"{'='*80}")
    fprint(f"Capital: ${CAP:.0f} | Pricing: ATR+{HAIRCUT:.0%} haircut | Comm: ${SPREAD_COMM}/spread RT")
    fprint(f"Testing: momentum vs multi-factor vs quality-weighted vs contrarian vs adaptive vs ensemble")
    fprint(f"{'='*80}")

    sc, sh, sl, sv, spy, vix, tnx = download_data()
    bd = pd.DatetimeIndex(sc.index.to_series().resample('2W-FRI').last().dropna().values)

    # Build datasets with different feature sets
    df_mom, cols_mom = build_dataset(sc, sv, spy, bd, 'momentum')
    df_multi, cols_multi = build_dataset(sc, sv, spy, bd, 'multi')
    df_qm, cols_qm = build_dataset(sc, sv, spy, bd, 'quality_mom')
    df_val, cols_val = build_dataset(sc, sv, spy, bd, 'value')

    fprint(f"\nDataset sizes: mom={len(df_mom)}, multi={len(df_multi)}, qm={len(df_qm)}, val={len(df_val)}")

    # Build rankings for each variant
    fprint("\n=== Building Rankings ===")
    rank_mom = lgbm_wf(df_mom, cols_mom, variant_name='A_Momentum')
    rank_multi = lgbm_wf(df_multi, cols_multi, variant_name='B_MultiF')
    rank_qm = lgbm_wf(df_qm, cols_qm, variant_name='C_QualityMom')
    rank_val = lgbm_wf(df_val, cols_val, variant_name='D_Value')

    # Derived rankings
    rank_contrarian = contrarian_rankings(rank_val)
    rank_adaptive = adaptive_rankings(rank_mom, rank_multi, vix, spy)
    rank_ensemble = ensemble_rankings(rank_mom, rank_multi, rank_qm)

    # Simulate all variants
    fprint("\n=== Simulating Variants ===")
    results = []
    configs = [
        ('A_Momentum_Baseline', rank_mom, 3.0, 30, 3),
        ('B_MultiFactor', rank_multi, 3.0, 30, 3),
        ('C_QualityMomentum', rank_qm, 3.0, 30, 3),
        ('D_Contrarian_Value', rank_contrarian, 3.0, 30, 3),
        ('E_Adaptive_Regime', rank_adaptive, 3.0, 30, 3),
        ('F_Ensemble_Avg', rank_ensemble, 3.0, 30, 3),
        ('G_Concentrated_Top2', rank_multi, 3.0, 30, 2),  # Fewer positions
    ]

    for nm, rnk, sp, dt, tk in configs:
        if not rnk:
            fprint(f"\n--- {nm} --- SKIPPED (no rankings)")
            continue
        tr, eq, cu = simulate(nm, rnk, sc, sh, sl, spy, vix, sp, dt, tk)
        r = metrics(tr, eq, cu, nm)
        if r:
            results.append(validate(r))

    if not results:
        fprint("FATAL: No results produced")
        return

    # Feature importance analysis (from multi-factor model)
    fprint(f"\n{'='*90}")
    fprint(f"FEATURE IMPORTANCE (Multi-Factor Model)")
    fprint(f"{'='*90}")
    # Train a final model on all data to get feature importance
    if len(df_multi) > 100:
        df_multi['rank_label'] = df_multi.groupby('date')['fwd_ret'].rank(pct=True)
        X_all = np.nan_to_num(df_multi[cols_multi].values.astype(np.float32))
        y_all = df_multi['rank_label'].values.astype(np.float32)
        try:
            m_fi = lgb.LGBMRegressor(n_estimators=100, max_depth=4, verbose=-1)
            m_fi.fit(X_all, y_all)
            imp = dict(zip(cols_multi, m_fi.feature_importances_))
            imp_sorted = sorted(imp.items(), key=lambda x: x[1], reverse=True)
            fprint(f"{'Feature':<25} {'Importance':>10}")
            fprint("-" * 37)
            for feat, val in imp_sorted[:15]:
                fprint(f"{feat:<25} {val:>10.0f}")
        except Exception as e:
            fprint(f"  Feature importance failed: {e}")

    # Summary table
    fprint(f"\n{'='*95}")
    fprint(f"SUMMARY — Multi-Factor Sector Options v1")
    fprint(f"{'='*95}")
    fprint(f"{'Variant':<25} {'#':>5} {'WR':>6} {'Sh':>6} {'So':>6} {'CAGR':>7} {'MDD':>7} {'PF':>6} {'Final$':>9} {'G':>4}")
    fprint("-"*95)
    for r in sorted(results, key=lambda x: x['sharpe'], reverse=True):
        fprint(f"{r['name']:<25} {r['n_trades']:>5} {r['win_rate']:>5.1f}% {r['sharpe']:>6.2f} "
               f"{r['sortino']:>6.2f} {r['cagr_pct']:>6.1f}% {r['maxdd_pct']:>6.1f}% "
               f"{r['pf']:>6.2f} ${r['final_equity']:>8.0f} {r['gates']:>3}/4")

    # Analysis
    bs = max(results, key=lambda x: x['sharpe'])
    vl = [r for r in results if r['gates'] >= 3]
    mom_r = next((r for r in results if r['name'] == 'A_Momentum_Baseline'), None)

    fprint(f"\nBEST SHARPE: {bs['name']} — Sharpe {bs['sharpe']}, CAGR {bs['cagr_pct']}%, Gates {bs['gates']}/4")
    if vl:
        bv = max(vl, key=lambda x: x['sharpe'])
        fprint(f"BEST VALIDATED (3+): {bv['name']} — Sharpe {bv['sharpe']}, CAGR {bv['cagr_pct']}%")

    if mom_r:
        fprint(f"\nMOMENTUM BASELINE: Sharpe {mom_r['sharpe']}, CAGR {mom_r['cagr_pct']}%")
        for r in results:
            if r['name'] != 'A_Momentum_Baseline':
                delta_sh = r['sharpe'] - mom_r['sharpe']
                delta_cagr = r['cagr_pct'] - mom_r['cagr_pct']
                better = "BETTER" if delta_sh > 0 else "WORSE" if delta_sh < 0 else "SAME"
                fprint(f"  {r['name']:<25} vs baseline: Sharpe {delta_sh:+.2f}, CAGR {delta_cagr:+.1f}% [{better}]")

    elapsed = (datetime.now() - t0).total_seconds()
    fprint(f"\nRuntime: {elapsed:.0f}s ({elapsed/60:.1f}m)")

    # Save results
    save_data = {
        'timestamp': t0.isoformat(),
        'capital': CAP,
        'pricing': 'ATR',
        'haircut': HAIRCUT,
        'commission_rt': SPREAD_COMM,
        'results': results,
        'runtime_s': round(elapsed, 1)
    }
    with open(RESULTS_PATH, 'w') as f:
        json.dump(save_data, f, indent=2, default=str)
    fprint(f"\nResults saved to {RESULTS_PATH}")

    # MLflow logging
    if MLFLOW_OK:
        try:
            en = 'multifactor_sector_options_v1'
            try:
                exp = mlflow.get_experiment_by_name(en)
                if not exp: mlflow.create_experiment(en)
            except Exception: pass
            mlflow.set_experiment(en)
            with mlflow.start_run(run_name=f"mf_opts_{t0.strftime('%Y%m%d_%H%M')}"):
                mlflow.log_params({
                    'pricing': 'ATR', 'haircut': HAIRCUT, 'capital': CAP,
                    'n_sectors': len(SECTORS), 'n_variants': len(results),
                    'spread_pct': 3.0, 'dte': 30, 'rebal': 'biweekly'
                })
                for r in results:
                    pref = r['name'][:20].replace(' ','_')
                    mlflow.log_metrics({
                        f'{pref}_sharpe': r['sharpe'],
                        f'{pref}_cagr': r['cagr_pct'],
                        f'{pref}_maxdd': r['maxdd_pct'],
                        f'{pref}_wr': r['win_rate'],
                        f'{pref}_pf': r['pf'],
                        f'{pref}_gates': r['gates'],
                        f'{pref}_perm_p': r.get('perm_p', 1.0),
                        f'{pref}_r1_gap': r.get('r1_gap', 1.0),
                    })
                if bs:
                    mlflow.log_metrics({
                        'best_sharpe': bs['sharpe'],
                        'best_cagr': bs['cagr_pct'],
                        'best_gates': bs['gates'],
                    })
                mlflow.log_artifact(str(RESULTS_PATH))
                fprint(f"\n🏃 MLflow run logged")
        except Exception as e:
            fprint(f"MLflow logging failed: {e}")

    fprint(f"\n{'='*80}")
    fprint("DONE — Multi-Factor Sector Options v1")
    fprint(f"{'='*80}")

if __name__ == '__main__':
    main()
