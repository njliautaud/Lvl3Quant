#!/usr/bin/env python3
"""Sector Bear Put Spreads v1 — Short the weakest sectors via put spreads.

ALL prior sector momentum options research was BULLISH (bull call spreads on
top-ranked sectors). This tests the BEAR side: buy put spreads on bottom-ranked
sectors (worst momentum/quality).

Hypothesis: If momentum persists on the downside, bottom-ranked sectors should
continue falling. We profit from put spreads on them.

Why this might work:
- Momentum anomaly is well-documented on BOTH sides (winners continue, losers continue)
- VIX>20 filter is lifted here — bear trades work in any VIX regime
- When VIX<20 (our bull strategy is sidelined), we might still find bear plays

Why this might fail:
- Mean reversion: weakest sectors bounce back
- Put spreads cost money (theta negative)
- Sector ETFs are diversified — individual stocks drop more than sector ETFs

Variants:
  A: Bottom-3 sectors, VIX any, 30d hold
  B: Bottom-3, VIX any, 20d exit
  C: Bottom-1 concentrated, VIX any, 20d exit
  D: Bottom-3, VIX<20 only (trade when bull strategy inactive)
  E: Bottom-3, SPY<200SMA only (bear regime)
  F: Bottom-3, no confluence (compare effect)

$645 starting capital, ATR pricing, honest Sharpe, 4-gate validation.
"""
import json, numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')
from pathlib import Path
from datetime import datetime
from scipy import stats
import lightgbm as lgb

BASE = Path(__file__).resolve().parents[2]
RESULTS_DIR = BASE / 'research' / 'findings'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_PATH = RESULTS_DIR / 'sector_bear_puts_v1_results.json'
def fprint(*a, **kw): print(*a, **kw, flush=True)

MLFLOW_OK = False
try:
    import urllib.request; urllib.request.urlopen('http://jupiter:5000/', timeout=2)
    import mlflow; mlflow.set_tracking_uri('http://jupiter:5000'); MLFLOW_OK = True
except Exception: fprint("MLflow unavailable")

SECTORS = ['XLK','XLF','XLE','XLV','XLY','XLP','XLI','XLB','XLU','XLRE','XLC']
CAP = 645.0; LEG_COMM = 0.65; SPREAD_COMM = 4*LEG_COMM; HAIRCUT = 0.15

MOM_COLS = ['ret_5d','ret_10d','ret_21d','ret_63d','ret_126d','ret_252d',
            'vol_21d','vol_63d','sharpe_63d','maxdd_63d','pct_52w_high','mom_accel']
QUALITY_COLS = ['pct_pos_months_12m','sortino_63d','calmar_1y','up_capture','dn_capture',
                'trend_r2_63d','trend_slope_63d','rel_vol_21d']
QM_COLS = MOM_COLS + QUALITY_COLS

def download_data():
    import yfinance as yf
    fprint("Downloading data...")
    tickers = SECTORS + ['SPY', '^VIX']
    raw = yf.download(tickers, start='2008-01-01', end='2026-07-26', progress=False)
    mi = isinstance(raw.columns, pd.MultiIndex)
    close = raw['Close'] if mi else raw
    high, low = (raw['High'], raw['Low']) if mi else (raw, raw)
    volume = raw['Volume'] if mi else raw
    vc = '^VIX' if '^VIX' in close.columns else 'VIX'
    vix, spy = close[vc].dropna(), close['SPY'].dropna()
    sc = close[[c for c in SECTORS if c in close.columns]].dropna(how='all')
    sh = high[[c for c in SECTORS if c in high.columns]].dropna(how='all')
    sl = low[[c for c in SECTORS if c in low.columns]].dropna(how='all')
    sv = volume[[c for c in SECTORS if c in volume.columns]].dropna(how='all')
    ix = sc.index.intersection(vix.index).intersection(spy.index).intersection(sh.index)
    fprint(f"Data: {len(ix)} days, {len(sc.columns)} sectors")
    return sc.loc[ix], sh.loc[ix], sl.loc[ix], sv.reindex(ix), spy.loc[ix], vix.loc[ix]

def compute_features(px, vol_data, spy_slice):
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
    pk = px.iloc[-252:].cummax(); mdd = float(((px.iloc[-252:]/pk)-1).min())
    cagr_1y = float(px.iloc[-1]/px.iloc[-252]-1) if len(px) >= 252 else 0.0
    f['calmar_1y'] = cagr_1y / (abs(mdd) + 1e-10)
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

def compute_atr(h, l, c, period=14):
    tr = pd.DataFrame({'hl': h-l, 'hc': abs(h-c.shift(1)), 'lc': abs(l-c.shift(1))}).max(axis=1)
    return tr.rolling(period).mean()

def atr_premium(S, K, dte, atr, vix_val, opt='call'):
    T = dte/252.0
    if T <= 0: return max(0, S-K) if opt=='call' else max(0, K-S)
    intrinsic = max(0, S-K) if opt=='call' else max(0, K-S)
    vol_factor = max(0.3, vix_val/20.0)
    return intrinsic + atr*np.sqrt(T)*vol_factor*np.exp(-3.0*abs(S-K)/S)

def build_rankings(sc, sv, spy, rebal_dates, feat_cols):
    fprint("  Building LGBM rankings...")
    records = []
    for dt in rebal_dates:
        idx = sc.index.get_indexer([dt], method='ffill')[0]
        if idx < 260: continue
        for tk in sc.columns:
            px = sc[tk].iloc[:idx+1].dropna()
            vol_d = sv[tk].iloc[:idx+1] if tk in sv.columns else None
            feats = compute_features(px, vol_d, spy.iloc[:idx+1])
            if not feats: continue
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
            try:
                m = lgb.LGBMRegressor(n_estimators=100, max_depth=4, learning_rate=0.05,
                    subsample=0.8, colsample_bytree=0.8, min_child_samples=5, device='gpu', verbose=-1)
                m.fit(Xt, yt)
            except:
                m = lgb.LGBMRegressor(n_estimators=100, max_depth=4, learning_rate=0.05,
                    subsample=0.8, colsample_bytree=0.8, min_child_samples=5, verbose=-1)
                m.fit(Xt, yt)
            te['score'] = m.predict(Xe)
            rankings[test_date] = dict(zip(te['ticker'], te['score']))
        except: continue
    fprint(f"    {len(rankings)} ranking dates")
    return rankings

def bear_confluence(tk, dt, sc, spy, vix):
    """Bearish confluence: min 2 confirming bearish signals."""
    signals = []
    idx = sc.index.get_indexer([dt], method='ffill')[0]
    if idx < 63: return False, 0
    px = sc[tk].iloc[:idx+1].dropna()
    if len(px) < 63: return False, 0

    # Signal 1: Negative momentum (21d return < 0)
    if float(px.iloc[-1]/px.iloc[-21]-1) < 0: signals.append('neg_mom')

    # Signal 2: Below 50-SMA (downtrend)
    sma50 = px.rolling(50).mean()
    if not pd.isna(sma50.iloc[-1]) and px.iloc[-1] < sma50.iloc[-1]:
        signals.append('below_sma50')

    # Signal 3: Negative relative strength vs SPY
    spy_s = spy.iloc[:idx+1]
    if len(spy_s) > 21:
        rel = px / spy_s
        if len(rel) > 21 and float(rel.iloc[-1]/rel.iloc[-21]-1) < 0:
            signals.append('neg_rel_str')

    # Signal 4: RSI not oversold (RSI > 20) — don't short into extreme oversold
    rets = px.pct_change().dropna()
    if len(rets) > 14:
        gains = rets.clip(lower=0).rolling(14).mean()
        losses = (-rets.clip(upper=0)).rolling(14).mean()
        rs = gains/(losses+1e-10); rsi = 100-100/(1+rs)
        r = float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else 50
        if r > 20 and r < 50: signals.append('rsi_bearish')

    return len(signals) >= 2, len(signals)

# ==================== SIMULATION ====================
def simulate(name, rankings, sc, sh, sl, spy, vix,
             spread_pct=3.0, dte=30, bottom_k=3,
             vix_min=None, vix_max=None,
             spy_below_sma=False, use_confluence=True,
             early_exit_day=None, sizing='fixed'):

    sma200 = spy.rolling(200).mean()
    atr_d = {tk: compute_atr(sh[tk], sl[tk], sc[tk]) for tk in sc.columns if tk in sh.columns}
    equity, trades, eq_curve = CAP, [], [CAP]

    for dt in sorted(rankings.keys()):
        if dt not in spy.index or dt not in vix.index: continue
        cv = float(vix.loc[dt])

        # VIX filters
        if vix_min is not None and cv < vix_min: continue
        if vix_max is not None and cv > vix_max: continue

        # SPY trend filter
        sv_val = float(spy.loc[dt])
        sm = float(sma200.loc[dt]) if dt in sma200.index and not pd.isna(sma200.loc[dt]) else sv_val
        bull = sv_val >= sm

        if spy_below_sma and bull: continue  # Only trade in bear regime

        scores = rankings[dt]
        if not scores: continue
        # BOTTOM ranked sectors (worst expected performance)
        ranked = sorted(scores.items(), key=lambda x: x[1])  # Ascending = worst first
        picks = [t for t, _ in ranked[:bottom_k]]

        max_pos = min(200, equity/3) if sizing == 'fixed' else min(500, 200*np.sqrt(equity/CAP))
        if max_pos < 30: eq_curve.append(equity); continue

        n_ent = 0
        for tk in picks:
            if tk not in sc.columns or tk not in atr_d or n_ent >= 3: continue

            if use_confluence:
                passes, _ = bear_confluence(tk, dt, sc, spy, vix)
                if not passes: continue

            S = float(sc[tk].loc[dt])
            av = float(atr_d[tk].loc[dt]) if dt in atr_d[tk].index and not pd.isna(atr_d[tk].loc[dt]) else S*0.015
            di = sc.index.get_loc(dt)

            # Bear put spread: buy ATM put, sell OTM put
            K1 = round(S)  # Buy ATM put
            K2 = round(S * (1 - spread_pct/100))  # Sell OTM put
            if K2 >= K1: continue

            lp = atr_premium(S, K1, dte, av, cv, 'put') * (1 + HAIRCUT)  # Buy expensive
            sp = atr_premium(S, K2, dte, av, cv, 'put') * (1 - HAIRCUT)  # Sell cheap
            debit = lp - sp
            width = K1 - K2
            cost = debit * 100 + SPREAD_COMM
            mx_prof = (width - debit) * 100 - SPREAD_COMM

            if cost <= 0 or cost > max_pos or cost > equity * 0.40: continue
            if mx_prof <= 0: continue

            max_hold = early_exit_day if early_exit_day else dte
            ei = min(di + max_hold, len(sc) - 1)

            pnl, aei = None, ei
            for ci in range(di + 1, ei + 1):
                if ci >= len(sc): break
                Sc = float(sc[tk].iloc[ci])
                dh = ci - di
                rd = max(0, dte - dh)
                ac = float(atr_d[tk].iloc[ci]) if ci < len(atr_d[tk]) else av
                tm = np.sqrt(rd / max(dte, 1))

                # Bear put spread value: max(0, K1-Sc) - max(0, K2-Sc) + time value
                intrinsic = (max(0, K1 - Sc) - max(0, K2 - Sc)) * 100
                time_val = ac * tm * 0.3 * 100
                current_val = intrinsic + time_val

                cp = current_val - cost

                # Take profit at 50% of max
                if cp >= mx_prof * 0.50:
                    pnl = cp; aei = ci; break

                # Time exit
                if ci == ei:
                    if early_exit_day and rd > 0:
                        pnl = cp
                    else:
                        pnl = intrinsic - cost
                    aei = ci; break

            if pnl is None:
                Se = float(sc[tk].iloc[ei]) if ei < len(sc) else S
                pnl = (max(0, K1 - Se) - max(0, K2 - Se)) * 100 - cost

            equity += pnl; n_ent += 1
            trades.append({
                'entry': str(dt.date()), 'exit': str(sc.index[aei].date()),
                'ticker': tk, 'pnl': round(pnl, 2), 'win': pnl > 0,
                'hold_days': aei - di,
                'regime': 'bull' if bull else 'bear',
                'vix': round(cv, 1), 'equity_at_trade': round(equity, 2)
            })
        eq_curve.append(equity)

    return trades, equity, eq_curve

def compute_honest_sharpe(trades):
    if not trades: return 0.0, 0.0, []
    tdf = pd.DataFrame(trades)
    tdf['entry_dt'] = pd.to_datetime(tdf['entry'])
    tdf['month'] = tdf['entry_dt'].dt.to_period('M')
    monthly = []
    for mo in sorted(tdf['month'].unique()):
        mt = tdf[tdf['month'] == mo]
        pnl = mt['pnl'].sum()
        eq = max(mt['equity_at_trade'].iloc[0] - mt['pnl'].iloc[0], 100)
        monthly.append(pnl / eq)
    rets = np.array(monthly)
    if len(rets) < 4: return 0.0, 0.0, rets
    sh = float(np.mean(rets) / (np.std(rets) + 1e-10) * np.sqrt(12))
    dn = rets[rets < 0]
    so = float(np.mean(rets) / (np.std(dn) + 1e-10) * np.sqrt(12)) if len(dn) > 1 else 0.0
    return sh, so, rets

def validate(trades, final_eq, eq_curve, name):
    if not trades: fprint(f"  {name}: No trades"); return None
    n = len(trades); wins = sum(1 for t in trades if t['win']); wr = wins/n*100
    pnls = [t['pnl'] for t in trades]
    sh, so, rets = compute_honest_sharpe(trades)
    ny = max(len(rets)/12, 0.5)
    cagr = (final_eq/CAP)**(1/ny)-1
    eq = np.array(eq_curve); pk = np.maximum.accumulate(eq); mdd = float(((eq-pk)/(pk+1e-10)).min())
    gp = sum(p for p in pnls if p>0); gl = abs(sum(p for p in pnls if p<=0))
    pf = gp/(gl+1e-10)
    bt = [t for t in trades if t['regime']=='bull']; brt = [t for t in trades if t['regime']=='bear']
    bw = sum(1 for t in bt if t['win'])/max(len(bt),1)*100
    brw = sum(1 for t in brt if t['win'])/max(len(brt),1)*100

    gates = 0; pp, rg, g1, g2, g3, g4, h1, h2 = 1.0, 1.0, False, False, False, False, 0, 0
    if len(rets) >= 10:
        rs = np.mean(rets)/(np.std(rets)+1e-10)
        pp = sum(1 for _ in range(2000) if np.mean(rets*np.random.choice([-1,1],len(rets)))/(np.std(rets)+1e-10)>=rs)/2000
        g1 = pp < 0.05; gates += g1
        rg = abs(bw-brw)/max(bw,brw,1); g2 = rg < 0.50; gates += g2
        mid = len(rets)//2
        h1 = np.mean(rets[:mid])/(np.std(rets[:mid])+1e-10) if mid > 3 else 0
        h2 = np.mean(rets[mid:])/(np.std(rets[mid:])+1e-10) if len(rets)-mid > 3 else 0
        g3 = h1 > 0 and h2 > 0; gates += g3
        tr = np.sort(rets)[:-1]; g4 = np.mean(tr)/(np.std(tr)+1e-10) > 0 if len(rets) > 5 else False; gates += g4

    r = {'name': name, 'n_trades': n, 'win_rate': round(wr,1),
         'sharpe': round(sh,2), 'sortino': round(so,2),
         'cagr_pct': round(cagr*100,1), 'maxdd_pct': round(mdd*100,1),
         'pf': round(pf,2), 'avg_pnl': round(np.mean(pnls),2),
         'final_equity': round(final_eq,2),
         'bull_wr': round(bw,1), 'bear_wr': round(brw,1),
         'bull_n': len(bt), 'bear_n': len(brt),
         'gates': gates, 'perm_p': round(pp,4), 'r1_gap': round(rg,3),
         'g1_perm': g1, 'g2_regime': g2, 'g3_sub': g3, 'g4_outlier': g4}
    fprint(f"  {name}: {n} trades | WR {wr:.1f}% | Sh {sh:.2f} | Sort {so:.2f} | "
           f"CAGR {cagr*100:.1f}% | MDD {mdd*100:.1f}% | PF {pf:.2f} | "
           f"Bull/Bear: {bw:.0f}%({len(bt)})/{brw:.0f}%({len(brt)}) | "
           f"${CAP}->${final_eq:.0f} | Gates {gates}/4")
    fprint(f"  G1={'P' if g1 else 'F'}(p={pp:.4f}) G2={'P' if g2 else 'F'}(gap={rg:.3f}) "
           f"G3={'P' if g3 else 'F'}({h1:.2f}/{h2:.2f}) G4={'P' if g4 else 'F'}")
    return r

def main():
    t0 = datetime.now()
    fprint(f"Sector Bear Put Spreads v1 — {t0.strftime('%Y-%m-%d %H:%M:%S')}")
    fprint(f"{'='*80}")
    fprint(f"SHORT weakest sectors via put spreads — NOVEL (all prior was bullish)")
    fprint(f"{'='*80}")

    sc, sh, sl, sv, spy, vix = download_data()
    bd = pd.DatetimeIndex(sc.index.to_series().resample('2W-FRI').last().dropna().values)
    rankings = build_rankings(sc, sv, spy, bd, QM_COLS)
    if not rankings: fprint("FATAL: No rankings"); return

    configs = [
        # name, bottom_k, vix_min, vix_max, spy_below_sma, confluence, exit_day, sizing
        ('A_Bottom3_AnyVIX',     3, None, None, False, True,  None, 'fixed'),
        ('B_Bottom3_20dExit',    3, None, None, False, True,  20,   'fixed'),
        ('C_Bottom1_Conc',       1, None, None, False, True,  20,   'fixed'),
        ('D_LowVIX_Only',        3, None, 20,   False, True,  20,   'fixed'),
        ('E_BearRegime',         3, None, None, True,  True,  20,   'fixed'),
        ('F_NoConfluence',       3, None, None, False, False, 20,   'fixed'),
    ]

    results = []
    for nm, bk, vmin, vmax, spy_below, conf, exit_day, sz in configs:
        fprint(f"\n--- {nm} ---")
        tr, eq, cu = simulate(nm, rankings, sc, sh, sl, spy, vix,
                               spread_pct=3.0, dte=30, bottom_k=bk,
                               vix_min=vmin, vix_max=vmax,
                               spy_below_sma=spy_below, use_confluence=conf,
                               early_exit_day=exit_day, sizing=sz)
        r = validate(tr, eq, cu, nm)
        if r: results.append(r)

    if not results: fprint("No results"); return

    # Random control
    fprint(f"\n=== RANDOM CONTROL (random sectors as 'bottom') ===")
    np.random.seed(42)
    random_sharpes = []
    for trial in range(5):
        rand_rankings = {}
        for dt, scores in rankings.items():
            rand_rankings[dt] = {tk: np.random.random() for tk in scores}
        tr, eq, cu = simulate(f'Rand_{trial}', rand_rankings, sc, sh, sl, spy, vix,
                               spread_pct=3.0, dte=30, bottom_k=3,
                               use_confluence=True, early_exit_day=20)
        if tr:
            sh_r, _, _ = compute_honest_sharpe(tr)
            random_sharpes.append(sh_r)
            fprint(f"  Random {trial}: Sharpe {sh_r:.2f}, ${CAP}->${eq:.0f}")

    # Summary
    fprint(f"\n{'='*100}")
    fprint(f"SUMMARY — Sector Bear Put Spreads v1")
    fprint(f"{'='*100}")
    fprint(f"{'Variant':<22} {'#':>5} {'WR':>6} {'Sh':>7} {'CAGR':>7} {'MDD':>7} {'PF':>6} {'AvgPnl':>7} {'Final$':>8} {'G':>4}")
    fprint("-"*100)
    for r in sorted(results, key=lambda x: x['sharpe'], reverse=True):
        fprint(f"{r['name']:<22} {r['n_trades']:>5} {r['win_rate']:>5.1f}% {r['sharpe']:>7.2f} "
               f"{r['cagr_pct']:>6.1f}% {r['maxdd_pct']:>6.1f}% "
               f"{r['pf']:>6.2f} ${r['avg_pnl']:>6.1f} ${r['final_equity']:>7.0f} {r['gates']:>3}/4")

    if random_sharpes:
        best = max(results, key=lambda x: x['sharpe'])
        fprint(f"\nRandom mean Sharpe: {np.mean(random_sharpes):.2f} vs best real {best['sharpe']:.2f}")

    # Compare to BULL side
    fprint(f"\n=== BEAR vs BULL COMPARISON ===")
    fprint(f"Bull call spreads (production v3): Sharpe 4.73, CAGR 68%, WR 88.7%, 4/4 gates")
    best = max(results, key=lambda x: x['sharpe']) if results else None
    if best:
        fprint(f"Bear put spreads (this):          Sharpe {best['sharpe']}, CAGR {best['cagr_pct']}%, WR {best['win_rate']}%, {best['gates']}/4 gates")

    elapsed = (datetime.now() - t0).total_seconds()
    fprint(f"\nRuntime: {elapsed:.0f}s ({elapsed/60:.1f}m)")

    save_data = {'timestamp': t0.isoformat(), 'capital': CAP,
                 'results': results, 'random_sharpes': random_sharpes, 'runtime_s': round(elapsed,1)}
    with open(RESULTS_PATH, 'w') as f:
        json.dump(save_data, f, indent=2, default=str)

    if MLFLOW_OK:
        try:
            en = 'sector_bear_puts_v1'
            try:
                exp = mlflow.get_experiment_by_name(en)
                if not exp: mlflow.create_experiment(en)
            except: pass
            mlflow.set_experiment(en)
            with mlflow.start_run(run_name=f"bear_puts_{t0.strftime('%Y%m%d_%H%M')}"):
                mlflow.log_params({'capital': CAP, 'direction': 'BEAR', 'pricing': 'ATR'})
                for r in results:
                    p = r['name'][:18].replace(' ','_')
                    mlflow.log_metrics({f'{p}_sh': r['sharpe'], f'{p}_wr': r['win_rate'],
                                        f'{p}_cagr': r['cagr_pct'], f'{p}_gates': r['gates']})
                mlflow.log_artifact(str(RESULTS_PATH))
                fprint("MLflow logged")
        except Exception as e:
            fprint(f"MLflow failed: {e}")

    fprint(f"\n{'='*80}\nDONE — Sector Bear Put Spreads v1\n{'='*80}")

if __name__ == '__main__':
    main()
