#!/usr/bin/env python3
"""Simple vs ML Sector Selection — The Most Important Test.

QUESTION: Does our LightGBM sector ranking add ANY edge over simple rules?
Permutation tests showed random selection gives similar Sharpe (~3.5-4.5)
to our ML model (3.88). This suggests the edge comes from TRADE STRUCTURE
(bull call spreads on sector ETFs when VIX>=20), not sector SELECTION.

METHODS TESTED:
  1. RANDOM        — Pick 3 random sectors each period (avg of 10 seeds)
  2. EQUAL_WEIGHT  — Buy ALL 11 sectors (no selection at all)
  3. SIMPLE_MOM    — Top 3 by 63-day return (no ML)
  4. SIMPLE_QUAL   — Top 3 by 63-day Sharpe (no ML)
  5. LGBM          — Full LightGBM quality-momentum ranking (production model)

ALL use identical trade execution: bull call spreads, 3% OTM, 30d DTE,
ATR pricing + 15% haircut, $2.60 commission, VIX>=20, bi-weekly rebalance.

HONEST Sharpe = mean(monthly_pnl / equity_at_start_of_month) / std(same) * sqrt(12)
"""
import json, time, numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')
from pathlib import Path
from datetime import datetime
from scipy import stats
import lightgbm as lgb

BASE = Path(__file__).resolve().parents[2]
RESULTS_DIR = BASE / 'research' / 'findings'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_PATH = RESULTS_DIR / 'simple_vs_ml_sector_results.json'
def fprint(*a, **kw): print(*a, **kw, flush=True)

# MLflow setup
MLFLOW_OK = False
try:
    import urllib.request; urllib.request.urlopen('http://jupiter:5000/', timeout=2)
    import mlflow; mlflow.set_tracking_uri('http://jupiter:5000'); MLFLOW_OK = True
    fprint("MLflow connected")
except Exception:
    fprint("MLflow unavailable — results saved locally only")

# ── Constants ──
SECTORS = ['XLK','XLF','XLE','XLV','XLY','XLP','XLI','XLB','XLU','XLRE','XLC']
CAP = 645.0
LEG_COMM = 0.65
SPREAD_COMM = 4 * LEG_COMM  # $2.60 RT
HAIRCUT = 0.15
TRAIN_PERIODS = 24  # bi-weekly periods for WF train (≈12 months)
N_RANDOM_SEEDS = 10

QM_COLS = ['ret_5d','ret_10d','ret_21d','ret_63d','ret_126d','ret_252d',
           'vol_21d','vol_63d','sharpe_63d','maxdd_63d','pct_52w_high','mom_accel',
           'pct_pos_months_12m','sortino_63d','calmar_1y','up_capture','dn_capture',
           'trend_r2_63d','trend_slope_63d','rel_vol_21d']


# ══════════════════════════════════════════════════════════════
# DATA
# ══════════════════════════════════════════════════════════════

def download_data():
    import yfinance as yf
    tickers = SECTORS + ['SPY', '^VIX']
    fprint(f"Downloading {len(tickers)} tickers...")
    raw = yf.download(tickers, start='2008-01-01', end='2026-07-26', progress=False)
    mi = isinstance(raw.columns, pd.MultiIndex)
    close = raw['Close'] if mi else raw
    high = raw['High'] if mi else raw
    low = raw['Low'] if mi else raw
    volume = raw['Volume'] if mi else raw
    vc = '^VIX' if '^VIX' in close.columns else 'VIX'
    vix = close[vc].dropna()
    spy = close['SPY'].dropna()
    sc = close[[c for c in SECTORS if c in close.columns]].dropna(how='all')
    sh = high[[c for c in SECTORS if c in high.columns]].dropna(how='all')
    sl = low[[c for c in SECTORS if c in low.columns]].dropna(how='all')
    sv = volume[[c for c in SECTORS if c in volume.columns]].dropna(how='all')
    ix = sc.index.intersection(vix.index).intersection(spy.index).intersection(sh.index).intersection(sl.index)
    sc = sc.loc[ix]; sh = sh.loc[ix]; sl = sl.loc[ix]; sv = sv.reindex(ix)
    fprint(f"  Data: {len(ix)} days, {len(sc.columns)} sectors, {sc.index[0].date()} to {sc.index[-1].date()}")
    return sc, sh, sl, sv, spy.loc[ix], vix.loc[ix]


# ══════════════════════════════════════════════════════════════
# FEATURES (for LGBM)
# ══════════════════════════════════════════════════════════════

def compute_features(px, vol_data=None):
    if len(px) < 260:
        return None
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
    up = rets[rets > 0]; dn = rets[rets < 0]
    f['up_capture'] = float(up.iloc[-63:].mean()/(up.mean()+1e-10)) if len(up) > 10 else 1.0
    f['dn_capture'] = float(dn.iloc[-63:].mean()/(dn.mean()+1e-10)) if len(dn) > 10 else 1.0
    if len(px) >= 63:
        y = np.log(px.iloc[-63:].values+1e-10); x = np.arange(len(y))
        slope, _, r_val, _, _ = stats.linregress(x, y)
        f['trend_r2_63d'] = r_val**2; f['trend_slope_63d'] = slope*252
    else:
        f['trend_r2_63d'] = 0.0; f['trend_slope_63d'] = 0.0
    f['rel_vol_21d'] = float(vol_data.iloc[-21:].mean()/(vol_data.iloc[-63:].mean()+1e-10)) if vol_data is not None and len(vol_data) >= 63 else 1.0
    return f


# ══════════════════════════════════════════════════════════════
# SECTOR SELECTION METHODS
# ══════════════════════════════════════════════════════════════

def build_lgbm_rankings(sc, sv, rebal_dates, train_periods=24):
    """Walk-forward LightGBM ranking — production model."""
    fprint("  Building LGBM rankings...")
    records = []
    for dt in rebal_dates:
        idx = sc.index.get_indexer([dt], method='ffill')[0]
        if idx < 260:
            continue
        for tk in sc.columns:
            px = sc[tk].iloc[:idx+1].dropna()
            vol_d = sv[tk].iloc[:idx+1] if tk in sv.columns else None
            feats = compute_features(px, vol_d)
            if not feats:
                continue
            fi = min(idx+14, len(sc)-1)
            feats.update({'date': dt, 'ticker': tk, 'fwd_ret': float(sc[tk].iloc[fi]/sc[tk].iloc[idx]-1)})
            records.append(feats)

    df = pd.DataFrame(records)
    for c in QM_COLS:
        if c not in df.columns:
            df[c] = 0.0
    df[QM_COLS] = df[QM_COLS].fillna(0.0)
    if len(df) < 100:
        return {}

    df['rank_label'] = df.groupby('date')['fwd_ret'].rank(pct=True)
    dates = sorted(df['date'].unique())
    rankings = {}

    for i in range(train_periods, len(dates)):
        td = dates[max(0, i-train_periods):i]
        test_date = dates[i]
        tr = df[df['date'].isin(td)]
        te = df[df['date'] == test_date].copy()
        if len(te) < 3 or len(tr) < 50:
            continue
        Xt = np.nan_to_num(tr[QM_COLS].values.astype(np.float32))
        yt = tr['rank_label'].values.astype(np.float32)
        Xe = np.nan_to_num(te[QM_COLS].values.astype(np.float32))
        try:
            m = lgb.LGBMRegressor(n_estimators=100, max_depth=4, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, min_child_samples=5, verbose=-1)
            m.fit(Xt, yt)
            te['score'] = m.predict(Xe)
            rankings[test_date] = dict(zip(te['ticker'], te['score']))
        except:
            continue

    fprint(f"    {len(rankings)} ranking dates")
    return rankings


def build_simple_momentum_rankings(sc, rebal_dates, lookback=63):
    """Sort by trailing return, assign rank scores. No ML."""
    fprint(f"  Building simple momentum rankings (lookback={lookback}d)...")
    rankings = {}
    for dt in rebal_dates:
        idx = sc.index.get_indexer([dt], method='ffill')[0]
        if idx < lookback + 10:
            continue
        scores = {}
        for tk in sc.columns:
            px = sc[tk].iloc[:idx+1].dropna()
            if len(px) < lookback + 1:
                continue
            ret = float(px.iloc[-1] / px.iloc[-lookback] - 1)
            scores[tk] = ret
        if len(scores) >= 3:
            rankings[dt] = scores
    fprint(f"    {len(rankings)} ranking dates")
    return rankings


def build_simple_quality_rankings(sc, rebal_dates, lookback=63):
    """Sort by trailing Sharpe ratio. No ML."""
    fprint(f"  Building simple quality rankings (lookback={lookback}d)...")
    rankings = {}
    for dt in rebal_dates:
        idx = sc.index.get_indexer([dt], method='ffill')[0]
        if idx < lookback + 10:
            continue
        scores = {}
        for tk in sc.columns:
            px = sc[tk].iloc[:idx+1].dropna()
            if len(px) < lookback + 1:
                continue
            rets = px.pct_change().dropna().iloc[-lookback:]
            if len(rets) < 20:
                continue
            sharpe = float(rets.mean() / (rets.std() + 1e-10) * np.sqrt(252))
            scores[tk] = sharpe
        if len(scores) >= 3:
            rankings[dt] = scores
    fprint(f"    {len(rankings)} ranking dates")
    return rankings


def build_random_rankings(sc, rebal_dates, seed=42):
    """Random scores for each sector. Same dates as other methods."""
    rng = np.random.RandomState(seed)
    rankings = {}
    for dt in rebal_dates:
        idx = sc.index.get_indexer([dt], method='ffill')[0]
        if idx < 260:
            continue
        available = [tk for tk in sc.columns if not pd.isna(sc[tk].iloc[idx])]
        if len(available) < 3:
            continue
        scores = {tk: rng.random() for tk in available}
        rankings[dt] = scores
    return rankings


def build_equal_weight_rankings(sc, rebal_dates):
    """All sectors get equal score — buy all 11, not top 3."""
    rankings = {}
    for dt in rebal_dates:
        idx = sc.index.get_indexer([dt], method='ffill')[0]
        if idx < 260:
            continue
        available = [tk for tk in sc.columns if not pd.isna(sc[tk].iloc[idx])]
        if len(available) < 3:
            continue
        # All equal scores
        rankings[dt] = {tk: 1.0 for tk in available}
    return rankings


# ══════════════════════════════════════════════════════════════
# OPTIONS PRICING + TRADE SIMULATION (identical for all methods)
# ══════════════════════════════════════════════════════════════

def compute_atr(h, l, c, period=14):
    tr = pd.DataFrame({'hl': h-l, 'hc': abs(h-c.shift(1)), 'lc': abs(l-c.shift(1))}).max(axis=1)
    return tr.rolling(period).mean()


def atr_premium(S, K, dte, atr, vix_val, opt='call'):
    T = dte / 252.0
    if T <= 0:
        return max(0, S-K) if opt == 'call' else max(0, K-S)
    intrinsic = max(0, S-K) if opt == 'call' else max(0, K-S)
    vol_factor = max(0.3, vix_val / 20.0)
    return intrinsic + atr * np.sqrt(T) * vol_factor * np.exp(-3.0 * abs(S-K) / S)


def simulate(rankings, sc, sh, sl, spy, vix, atr_d, top_k=3, vix_min=20):
    """Run strategy simulation. Identical execution for all selection methods."""
    sma200 = spy.rolling(200).mean()
    equity = CAP
    trades = []
    monthly_start_equity = {}

    for dt in sorted(rankings.keys()):
        if dt not in spy.index or dt not in vix.index:
            continue
        cv = float(vix.loc[dt])
        if vix_min is not None and cv < vix_min:
            continue

        sv_val = float(spy.loc[dt])
        sm = float(sma200.loc[dt]) if dt in sma200.index and not pd.isna(sma200.loc[dt]) else sv_val
        bull = sv_val >= sm

        scores = rankings[dt]
        if not scores:
            continue

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        picks = [t for t, _ in ranked[:top_k]]
        max_pos = min(200, equity / 3)
        if max_pos < 30:
            continue

        # Track month-start equity
        month_key = dt.to_period('M')
        if month_key not in monthly_start_equity:
            monthly_start_equity[month_key] = equity

        n_ent = 0
        for tk in picks:
            if tk not in sc.columns or tk not in atr_d or n_ent >= 3:
                continue
            S = float(sc[tk].loc[dt])
            av = float(atr_d[tk].loc[dt]) if dt in atr_d[tk].index and not pd.isna(atr_d[tk].loc[dt]) else S * 0.015
            di = sc.index.get_loc(dt)
            ei = min(di + 30, len(sc) - 1)
            Se = float(sc[tk].iloc[ei])
            K1, K2 = round(S), round(S * 1.03)
            lp = atr_premium(S, K1, 30, av, cv, 'call') * (1 + HAIRCUT)
            sp = atr_premium(S, K2, 30, av, cv, 'call') * (1 - HAIRCUT)
            val = lp - sp
            width = K2 - K1
            cost = val * 100 + SPREAD_COMM
            mx_prof = (width - val) * 100 - SPREAD_COMM
            if cost <= 0 or cost > max_pos or cost > equity * 0.40:
                continue

            pnl = None
            for ci in range(di + 7, ei + 1):
                Sc = float(sc[tk].iloc[ci])
                dh = ci - di
                rd = max(0, 30 - dh)
                ac = float(atr_d[tk].iloc[ci]) if ci < len(atr_d[tk]) else av
                tm = np.sqrt(rd / 30.0)
                si = (max(0, Sc - K1) - max(0, Sc - K2)) * 100 + ac * tm * 0.3 * 100
                cp = si - cost
                if cp >= mx_prof * 0.50 or rd < 7:
                    pnl = cp
                    break
            if pnl is None:
                pnl = (max(0, Se - K1) - max(0, Se - K2)) * 100 - val * 100 - SPREAD_COMM

            equity += pnl
            n_ent += 1
            trades.append({
                'date': str(dt.date()),
                'month': str(month_key),
                'ticker': tk,
                'pnl': round(pnl, 2),
                'win': pnl > 0,
                'regime': 'bull' if bull else 'bear',
                'year': dt.year,
            })

    return trades, equity, monthly_start_equity


# ══════════════════════════════════════════════════════════════
# HONEST SHARPE + METRICS
# ══════════════════════════════════════════════════════════════

def honest_sharpe(trades, monthly_start_equity):
    """HONEST annualized Sharpe: mean(monthly_ret)/std(monthly_ret)*sqrt(12)
    where monthly_ret = monthly_pnl / equity_at_start_of_month."""
    if not trades:
        return 0.0, []
    tdf = pd.DataFrame(trades)
    tdf['month'] = pd.to_datetime(tdf['date']).dt.to_period('M')
    monthly_pnl = tdf.groupby('month')['pnl'].sum()
    monthly_rets = []
    for m in monthly_pnl.index:
        pnl = monthly_pnl[m]
        start_eq = monthly_start_equity.get(m, CAP)
        if start_eq <= 0:
            start_eq = CAP
        monthly_rets.append(pnl / start_eq)
    rets = np.array(monthly_rets)
    if len(rets) < 3:
        return 0.0, rets.tolist()
    sharpe = float(rets.mean() / (rets.std() + 1e-10) * np.sqrt(12))
    return sharpe, rets.tolist()


def compute_metrics(trades, final_equity, monthly_start_equity, name):
    if not trades:
        return None

    n = len(trades)
    wins = sum(1 for t in trades if t['win'])
    wr = wins / n * 100
    pnls = [t['pnl'] for t in trades]

    sharpe, monthly_rets = honest_sharpe(trades, monthly_start_equity)

    # Sortino
    rets = np.array(monthly_rets)
    downside = rets[rets < 0]
    if len(downside) >= 2:
        sortino = float(rets.mean() / (downside.std() + 1e-10) * np.sqrt(12))
    else:
        sortino = 99.0  # no downside months

    # CAGR
    n_years = max(len(monthly_rets) / 12, 0.5)
    cagr = (final_equity / CAP) ** (1 / n_years) - 1

    # Max drawdown from equity curve
    eq = [CAP]
    for t in trades:
        eq.append(eq[-1] + t['pnl'])
    eq = np.array(eq)
    pk = np.maximum.accumulate(eq)
    maxdd = float(((eq - pk) / (pk + 1e-10)).min())

    # Profit factor
    gp = sum(p for p in pnls if p > 0)
    gl = abs(sum(p for p in pnls if p <= 0))
    pf = gp / (gl + 1e-10)

    # Bull/bear split
    bull_trades = [t for t in trades if t['regime'] == 'bull']
    bear_trades = [t for t in trades if t['regime'] == 'bear']
    bull_wr = sum(1 for t in bull_trades if t['win']) / max(len(bull_trades), 1) * 100
    bear_wr = sum(1 for t in bear_trades if t['win']) / max(len(bear_trades), 1) * 100

    # Bull/bear Sharpe
    bull_pnls_by_month = {}
    bear_pnls_by_month = {}
    for t in trades:
        m = t['month']
        if t['regime'] == 'bull':
            bull_pnls_by_month[m] = bull_pnls_by_month.get(m, 0) + t['pnl']
        else:
            bear_pnls_by_month[m] = bear_pnls_by_month.get(m, 0) + t['pnl']

    def sharpe_from_dict(pnl_dict):
        if len(pnl_dict) < 3:
            return 0.0
        vals = np.array(list(pnl_dict.values()))
        avg_eq = np.mean([monthly_start_equity.get(pd.Period(m, 'M'), CAP) for m in pnl_dict.keys()])
        r = vals / max(avg_eq, 1)
        return float(r.mean() / (r.std() + 1e-10) * np.sqrt(12))

    bull_sharpe = sharpe_from_dict(bull_pnls_by_month)
    bear_sharpe = sharpe_from_dict(bear_pnls_by_month)

    # Yearly Sharpe
    tdf = pd.DataFrame(trades)
    yearly_sharpes = {}
    for yr in sorted(tdf['year'].unique()):
        yr_trades = [t for t in trades if t['year'] == yr]
        yr_monthly = {}
        for t in yr_trades:
            m = t['month']
            yr_monthly[m] = yr_monthly.get(m, 0) + t['pnl']
        if len(yr_monthly) >= 3:
            vals = np.array(list(yr_monthly.values()))
            avg_eq = np.mean([monthly_start_equity.get(pd.Period(m, 'M'), CAP) for m in yr_monthly.keys()])
            r = vals / max(avg_eq, 1)
            yearly_sharpes[int(yr)] = round(float(r.mean() / (r.std() + 1e-10) * np.sqrt(12)), 2)

    return {
        'name': name,
        'n_trades': n,
        'win_rate': round(wr, 1),
        'honest_sharpe': round(sharpe, 2),
        'sortino': round(sortino, 2),
        'cagr_pct': round(cagr * 100, 1),
        'maxdd_pct': round(maxdd * 100, 1),
        'profit_factor': round(pf, 2),
        'final_equity': round(final_equity, 2),
        'avg_pnl': round(np.mean(pnls), 2),
        'bull_trades': len(bull_trades),
        'bear_trades': len(bear_trades),
        'bull_wr': round(bull_wr, 1),
        'bear_wr': round(bear_wr, 1),
        'bull_sharpe': round(bull_sharpe, 2),
        'bear_sharpe': round(bear_sharpe, 2),
        'yearly_sharpes': yearly_sharpes,
        'n_months': len(monthly_rets),
    }


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    fprint(f"Simple vs ML Sector Selection — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    fprint(f"{'='*90}")
    fprint("QUESTION: Does LightGBM sector ranking add edge over simple rules?")
    fprint("All methods use IDENTICAL trade execution (bull call spreads, VIX>=20)")
    fprint(f"{'='*90}\n")

    # Download data once
    sc, sh, sl, sv, spy, vix = download_data()

    # ATR (shared by all methods)
    atr_d = {}
    for tk in sc.columns:
        if tk in sh.columns and tk in sl.columns:
            atr_d[tk] = compute_atr(sh[tk], sl[tk], sc[tk])

    # Bi-weekly rebalance dates
    bd = pd.DatetimeIndex(sc.index.to_series().resample('2W-FRI').last().dropna().values)
    fprint(f"Rebalance dates: {len(bd)} bi-weekly periods\n")

    results = {}

    # ── 1. RANDOM (average of 10 seeds) ──
    fprint(f"\n{'='*70}")
    fprint("METHOD 1: RANDOM — Pick 3 random sectors each period")
    fprint(f"{'='*70}")
    random_results = []
    for seed in range(N_RANDOM_SEEDS):
        rankings = build_random_rankings(sc, bd, seed=42 + seed)
        trades, final_eq, monthly_start_eq = simulate(
            rankings, sc, sh, sl, spy, vix, atr_d, top_k=3, vix_min=20)
        m = compute_metrics(trades, final_eq, monthly_start_eq, f"RANDOM_seed{seed}")
        if m:
            random_results.append(m)
            fprint(f"  Seed {seed}: Sharpe {m['honest_sharpe']}, WR {m['win_rate']}%, "
                   f"CAGR {m['cagr_pct']}%, n={m['n_trades']}")

    if random_results:
        avg_sharpe = np.mean([r['honest_sharpe'] for r in random_results])
        avg_wr = np.mean([r['win_rate'] for r in random_results])
        avg_cagr = np.mean([r['cagr_pct'] for r in random_results])
        avg_pf = np.mean([r['profit_factor'] for r in random_results])
        avg_mdd = np.mean([r['maxdd_pct'] for r in random_results])
        avg_n = np.mean([r['n_trades'] for r in random_results])
        avg_bull_wr = np.mean([r['bull_wr'] for r in random_results])
        avg_bear_wr = np.mean([r['bear_wr'] for r in random_results])
        avg_bull_sh = np.mean([r['bull_sharpe'] for r in random_results])
        avg_bear_sh = np.mean([r['bear_sharpe'] for r in random_results])

        results['RANDOM'] = {
            'honest_sharpe': round(avg_sharpe, 2),
            'win_rate': round(avg_wr, 1),
            'cagr_pct': round(avg_cagr, 1),
            'maxdd_pct': round(avg_mdd, 1),
            'profit_factor': round(avg_pf, 2),
            'n_trades': round(avg_n, 0),
            'bull_wr': round(avg_bull_wr, 1),
            'bear_wr': round(avg_bear_wr, 1),
            'bull_sharpe': round(avg_bull_sh, 2),
            'bear_sharpe': round(avg_bear_sh, 2),
            'sharpe_std': round(np.std([r['honest_sharpe'] for r in random_results]), 2),
            'sharpe_min': round(min(r['honest_sharpe'] for r in random_results), 2),
            'sharpe_max': round(max(r['honest_sharpe'] for r in random_results), 2),
            'all_seeds': [{'seed': i, 'sharpe': r['honest_sharpe'], 'wr': r['win_rate'],
                           'cagr': r['cagr_pct']} for i, r in enumerate(random_results)],
        }
        fprint(f"\n  RANDOM AVERAGE: Sharpe {avg_sharpe:.2f} (std {results['RANDOM']['sharpe_std']:.2f}, "
               f"range [{results['RANDOM']['sharpe_min']:.2f}, {results['RANDOM']['sharpe_max']:.2f}])")
        fprint(f"  WR {avg_wr:.1f}% | CAGR {avg_cagr:.1f}% | MDD {avg_mdd:.1f}% | PF {avg_pf:.2f} | n={avg_n:.0f}")

    # ── 2. EQUAL WEIGHT (all 11 sectors) ──
    fprint(f"\n{'='*70}")
    fprint("METHOD 2: EQUAL_WEIGHT — Buy ALL 11 sectors (no selection)")
    fprint(f"{'='*70}")
    rankings = build_equal_weight_rankings(sc, bd)
    # For equal weight, use top_k=11 (buy all)
    trades, final_eq, monthly_start_eq = simulate(
        rankings, sc, sh, sl, spy, vix, atr_d, top_k=11, vix_min=20)
    m = compute_metrics(trades, final_eq, monthly_start_eq, "EQUAL_WEIGHT")
    if m:
        results['EQUAL_WEIGHT'] = m
        fprint(f"  Sharpe {m['honest_sharpe']} | WR {m['win_rate']}% | CAGR {m['cagr_pct']}% | "
               f"MDD {m['maxdd_pct']}% | PF {m['profit_factor']} | n={m['n_trades']}")
        fprint(f"  Bull: WR {m['bull_wr']}% ({m['bull_trades']} trades, Sharpe {m['bull_sharpe']})")
        fprint(f"  Bear: WR {m['bear_wr']}% ({m['bear_trades']} trades, Sharpe {m['bear_sharpe']})")

    # ── 3. SIMPLE MOMENTUM (63-day return, top 3) ──
    fprint(f"\n{'='*70}")
    fprint("METHOD 3: SIMPLE_MOMENTUM — Top 3 by 63-day return")
    fprint(f"{'='*70}")
    rankings = build_simple_momentum_rankings(sc, bd, lookback=63)
    trades, final_eq, monthly_start_eq = simulate(
        rankings, sc, sh, sl, spy, vix, atr_d, top_k=3, vix_min=20)
    m = compute_metrics(trades, final_eq, monthly_start_eq, "SIMPLE_MOMENTUM")
    if m:
        results['SIMPLE_MOMENTUM'] = m
        fprint(f"  Sharpe {m['honest_sharpe']} | WR {m['win_rate']}% | CAGR {m['cagr_pct']}% | "
               f"MDD {m['maxdd_pct']}% | PF {m['profit_factor']} | n={m['n_trades']}")
        fprint(f"  Bull: WR {m['bull_wr']}% ({m['bull_trades']} trades, Sharpe {m['bull_sharpe']})")
        fprint(f"  Bear: WR {m['bear_wr']}% ({m['bear_trades']} trades, Sharpe {m['bear_sharpe']})")
        fprint(f"  Yearly Sharpes: {m['yearly_sharpes']}")

    # ── 4. SIMPLE QUALITY (63-day Sharpe, top 3) ──
    fprint(f"\n{'='*70}")
    fprint("METHOD 4: SIMPLE_QUALITY — Top 3 by 63-day Sharpe ratio")
    fprint(f"{'='*70}")
    rankings = build_simple_quality_rankings(sc, bd, lookback=63)
    trades, final_eq, monthly_start_eq = simulate(
        rankings, sc, sh, sl, spy, vix, atr_d, top_k=3, vix_min=20)
    m = compute_metrics(trades, final_eq, monthly_start_eq, "SIMPLE_QUALITY")
    if m:
        results['SIMPLE_QUALITY'] = m
        fprint(f"  Sharpe {m['honest_sharpe']} | WR {m['win_rate']}% | CAGR {m['cagr_pct']}% | "
               f"MDD {m['maxdd_pct']}% | PF {m['profit_factor']} | n={m['n_trades']}")
        fprint(f"  Bull: WR {m['bull_wr']}% ({m['bull_trades']} trades, Sharpe {m['bull_sharpe']})")
        fprint(f"  Bear: WR {m['bear_wr']}% ({m['bear_trades']} trades, Sharpe {m['bear_sharpe']})")
        fprint(f"  Yearly Sharpes: {m['yearly_sharpes']}")

    # ── 5. LGBM (full production model) ──
    fprint(f"\n{'='*70}")
    fprint("METHOD 5: LGBM — Full LightGBM quality-momentum ranking")
    fprint(f"{'='*70}")
    rankings = build_lgbm_rankings(sc, sv, bd, train_periods=TRAIN_PERIODS)
    trades, final_eq, monthly_start_eq = simulate(
        rankings, sc, sh, sl, spy, vix, atr_d, top_k=3, vix_min=20)
    m = compute_metrics(trades, final_eq, monthly_start_eq, "LGBM")
    if m:
        results['LGBM'] = m
        fprint(f"  Sharpe {m['honest_sharpe']} | WR {m['win_rate']}% | CAGR {m['cagr_pct']}% | "
               f"MDD {m['maxdd_pct']}% | PF {m['profit_factor']} | n={m['n_trades']}")
        fprint(f"  Bull: WR {m['bull_wr']}% ({m['bull_trades']} trades, Sharpe {m['bull_sharpe']})")
        fprint(f"  Bear: WR {m['bear_wr']}% ({m['bear_trades']} trades, Sharpe {m['bear_sharpe']})")
        fprint(f"  Yearly Sharpes: {m['yearly_sharpes']}")

    # ══════════════════════════════════════════════════════════════
    # COMPARISON TABLE
    # ══════════════════════════════════════════════════════════════
    elapsed = time.time() - t0
    fprint(f"\n\n{'='*100}")
    fprint(f"FINAL COMPARISON — Simple vs ML Sector Selection")
    fprint(f"{'='*100}")
    fprint(f"{'Method':<20} {'Sharpe':>8} {'WR':>7} {'CAGR':>8} {'MDD':>8} {'PF':>7} {'#Trades':>8} "
           f"{'Bull WR':>8} {'Bear WR':>8} {'Bull Sh':>8} {'Bear Sh':>8}")
    fprint("-" * 105)

    # Sort by honest_sharpe descending
    method_order = ['RANDOM', 'EQUAL_WEIGHT', 'SIMPLE_MOMENTUM', 'SIMPLE_QUALITY', 'LGBM']
    for method in method_order:
        if method not in results:
            continue
        r = results[method]
        sh = r['honest_sharpe']
        wr = r['win_rate']
        cagr = r['cagr_pct']
        mdd = r['maxdd_pct']
        pf = r['profit_factor']
        nt = r['n_trades']
        bw = r['bull_wr']
        brw = r['bear_wr']
        bsh = r['bull_sharpe']
        brsh = r['bear_sharpe']
        marker = " <-- ML" if method == 'LGBM' else ""
        fprint(f"{method:<20} {sh:>8.2f} {wr:>6.1f}% {cagr:>7.1f}% {mdd:>7.1f}% {pf:>7.2f} {nt:>8.0f} "
               f"{bw:>7.1f}% {brw:>7.1f}% {bsh:>8.2f} {brsh:>8.2f}{marker}")

    # ── Statistical significance test ──
    fprint(f"\n{'='*70}")
    fprint("STATISTICAL ANALYSIS")
    fprint(f"{'='*70}")

    lgbm_sharpe = results.get('LGBM', {}).get('honest_sharpe', 0)

    if 'RANDOM' in results:
        random_sharpes = [r['honest_sharpe'] for r in random_results]
        pct_random_beats_ml = sum(1 for s in [r['honest_sharpe'] for r in random_results]
                                   if s >= lgbm_sharpe) / len(random_results) * 100
        fprint(f"  Random beats LGBM: {pct_random_beats_ml:.0f}% of seeds ({sum(1 for s in [r['honest_sharpe'] for r in random_results] if s >= lgbm_sharpe)}/{len(random_results)})")
        fprint(f"  Random Sharpe range: [{results['RANDOM']['sharpe_min']:.2f}, {results['RANDOM']['sharpe_max']:.2f}]")
        fprint(f"  LGBM Sharpe: {lgbm_sharpe:.2f}")

    for method in ['SIMPLE_MOMENTUM', 'SIMPLE_QUALITY', 'EQUAL_WEIGHT']:
        if method in results:
            delta = results[method]['honest_sharpe'] - lgbm_sharpe
            pct = delta / max(abs(lgbm_sharpe), 0.01) * 100
            better = "BETTER" if delta > 0 else "WORSE"
            fprint(f"  {method} vs LGBM: {delta:+.2f} Sharpe ({pct:+.1f}%) — {better}")

    # ── Conclusion ──
    fprint(f"\n{'='*70}")
    fprint("CONCLUSION")
    fprint(f"{'='*70}")

    all_sharpes = {k: v['honest_sharpe'] for k, v in results.items()}
    best_method = max(all_sharpes, key=all_sharpes.get)
    worst_method = min(all_sharpes, key=all_sharpes.get)
    spread = max(all_sharpes.values()) - min(all_sharpes.values())

    fprint(f"  Best method: {best_method} (Sharpe {all_sharpes[best_method]:.2f})")
    fprint(f"  Worst method: {worst_method} (Sharpe {all_sharpes[worst_method]:.2f})")
    fprint(f"  Sharpe spread across all methods: {spread:.2f}")

    if spread < 1.0:
        fprint(f"\n  >>> VERDICT: TRADE STRUCTURE IS THE EDGE, NOT SECTOR SELECTION <<<")
        fprint(f"  All methods produce similar Sharpes (spread={spread:.2f} < 1.0)")
        fprint(f"  The edge comes from: VIX>=20 filter + bull call spreads on sector ETFs")
        fprint(f"  ML adds NO significant value for sector picking")
    elif lgbm_sharpe > max(v for k, v in all_sharpes.items() if k != 'LGBM') + 0.5:
        fprint(f"\n  >>> VERDICT: ML ADDS MEANINGFUL EDGE <<<")
        fprint(f"  LGBM outperforms best simple method by {lgbm_sharpe - max(v for k, v in all_sharpes.items() if k != 'LGBM'):.2f} Sharpe")
    else:
        fprint(f"\n  >>> VERDICT: MIXED — ML provides marginal benefit <<<")
        fprint(f"  Consider simplicity: simple methods may be more robust")

    fprint(f"\nRuntime: {elapsed:.0f}s ({elapsed/60:.1f}m)")

    # ── Save results ──
    save_data = {
        'timestamp': datetime.now().isoformat(),
        'description': 'Simple vs ML sector selection comparison — does LGBM add edge?',
        'methodology': {
            'honest_sharpe': 'mean(monthly_ret)/std(monthly_ret)*sqrt(12) where monthly_ret = monthly_pnl/equity_at_start_of_month',
            'trade_structure': 'Bull call spreads, 3% OTM, 30d DTE, ATR pricing + 15% haircut',
            'filter': 'VIX >= 20',
            'rebalance': 'Bi-weekly',
            'capital': CAP,
            'commission': f'${SPREAD_COMM:.2f} per spread RT (${LEG_COMM}/leg)',
            'sectors': SECTORS,
        },
        'results': {},
        'conclusion': {
            'best_method': best_method,
            'best_sharpe': round(all_sharpes[best_method], 2),
            'lgbm_sharpe': round(lgbm_sharpe, 2),
            'sharpe_spread': round(spread, 2),
            'ml_adds_value': spread >= 1.0 and lgbm_sharpe == all_sharpes[best_method],
        },
        'runtime_s': round(elapsed, 1),
    }

    # Clean results for JSON serialization
    for method, r in results.items():
        clean = {}
        for k, v in r.items():
            if k == 'all_seeds':
                clean[k] = v
            elif k == 'yearly_sharpes':
                clean[k] = {str(yr): s for yr, s in v.items()} if isinstance(v, dict) else v
            else:
                clean[k] = v
        save_data['results'][method] = clean

    with open(RESULTS_PATH, 'w') as f:
        json.dump(save_data, f, indent=2, default=str)
    fprint(f"\nResults saved to {RESULTS_PATH}")

    # ── MLflow logging ──
    if MLFLOW_OK:
        try:
            en = 'simple_vs_ml_sector'
            try:
                exp = mlflow.get_experiment_by_name(en)
                if not exp:
                    mlflow.create_experiment(en)
            except:
                pass
            mlflow.set_experiment(en)
            with mlflow.start_run(run_name=f"simple_vs_ml_{datetime.now().strftime('%Y%m%d_%H%M')}"):
                mlflow.log_params({
                    'capital': CAP,
                    'n_sectors': len(SECTORS),
                    'vix_filter': 20,
                    'rebalance': 'bi-weekly',
                    'n_random_seeds': N_RANDOM_SEEDS,
                    'lgbm_train_periods': TRAIN_PERIODS,
                })
                for method, r in results.items():
                    p = method[:15]
                    mlflow.log_metrics({
                        f'{p}_sharpe': r['honest_sharpe'],
                        f'{p}_wr': r['win_rate'],
                        f'{p}_cagr': r['cagr_pct'],
                        f'{p}_mdd': r['maxdd_pct'],
                        f'{p}_pf': r['profit_factor'],
                    })
                mlflow.log_metrics({
                    'sharpe_spread': round(spread, 2),
                    'ml_is_best': 1.0 if best_method == 'LGBM' else 0.0,
                })
                mlflow.log_artifact(str(RESULTS_PATH))
            fprint("MLflow run logged successfully")
        except Exception as e:
            fprint(f"MLflow logging failed: {e}")

    fprint(f"\n{'='*90}")
    fprint("DONE — Simple vs ML Sector Selection")
    fprint(f"{'='*90}")


if __name__ == '__main__':
    main()
