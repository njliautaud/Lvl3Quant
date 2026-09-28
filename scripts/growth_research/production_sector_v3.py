#!/usr/bin/env python3
"""Production Sector Momentum v3 — Combines ALL validated improvements.

NEW in v3 (not tested together before):
1. 20-day early exit (doubles Sharpe per exit_optimization_v1)
2. Trailing stop at 30% from peak (per exit_optimization_v1)
3. Tiered/sqrt position sizing (per dynamic_position_scaling_v1)
4. HONEST equity-based Sharpe (monthly PnL / current equity, not initial capital)
5. Calendar month aggregation (per optimized_sector_v1 Sharpe fix)

Kept from v2:
- Multi-factor LGBM ranking (20 features)
- VIX>20 entry filter
- HC #750 confluence gating (min 2 signals)
- ATR pricing + 15% haircut
- $645 starting capital

Variants tested:
  A: Baseline (30d hold, fixed sizing) — control
  B: 20d early exit + fixed sizing
  C: 20d exit + tiered sizing
  D: 20d exit + sqrt sizing
  E: Trailing 30% stop + tiered sizing
  F: 20d exit + trailing stop + tiered (full stack)
  G: 20d exit + sqrt sizing + VIX>25 (high-conviction)
  H: Conservative (20d exit + fixed + VIX 20-35)
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
RESULTS_PATH = RESULTS_DIR / 'production_sector_v3_results.json'
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
VALUE_COLS = ['dist_sma200','dist_sma50','rsi_14','bb_pct','price_zscore']
BREADTH_COLS = ['rel_str_21d','rel_str_63d']
QM_COLS = MOM_COLS + QUALITY_COLS  # Quality-momentum subset (best per v2)

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
    ix = sc.index.intersection(vix.index).intersection(spy.index).intersection(sh.index).intersection(sl.index)
    fprint(f"Data: {len(ix)} days, {len(sc.columns)} sectors")
    return sc.loc[ix], sh.loc[ix], sl.loc[ix], sv.reindex(ix), spy.loc[ix], vix.loc[ix]

def compute_features(px, vol_data, spy_slice):
    """Compute quality-momentum features for a single ticker."""
    if len(px) < 260: return None
    f = {}
    # Momentum
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

    # Quality
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
    f['rel_vol_21d'] = float(vol_data.iloc[-21:].mean()/(vol_data.iloc[-63:].mean()+1e-10)) if vol_data is not None and len(vol_data) >= 63 else 1.0

    return f

# ==================== LGBM RANKING ====================
def build_rankings(sc, sv, spy, rebal_dates, feat_cols):
    fprint(f"  Building LGBM rankings ({len(feat_cols)} features)...")
    records = []
    for dt in rebal_dates:
        idx = sc.index.get_indexer([dt], method='ffill')[0]
        if idx < 260: continue
        for tk in sc.columns:
            px = sc[tk].iloc[:idx+1].dropna()
            vol_d = sv[tk].iloc[:idx+1] if tk in sv.columns else None
            spy_s = spy.iloc[:idx+1]
            feats = compute_features(px, vol_d, spy_s)
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

# ==================== CONFLUENCE GATE (HC #750) ====================
def confluence_check(tk, dt, sc, spy, vix):
    """Min 2 confirming signals required."""
    signals = []
    idx = sc.index.get_indexer([dt], method='ffill')[0]
    if idx < 21: return False, 0, []
    px = sc[tk].iloc[:idx+1].dropna()
    if len(px) < 63: return False, 0, []

    # Signal 1: Momentum (21d return > 0)
    if float(px.iloc[-1]/px.iloc[-21]-1) > 0: signals.append('mom_21d')

    # Signal 2: Trend (price > 50-SMA)
    sma50 = px.rolling(50).mean()
    if not pd.isna(sma50.iloc[-1]) and px.iloc[-1] > sma50.iloc[-1]:
        signals.append('above_sma50')

    # Signal 3: Relative strength vs SPY
    spy_s = spy.iloc[:idx+1]
    if len(spy_s) > 21:
        rel = px / spy_s
        if len(rel) > 21 and float(rel.iloc[-1]/rel.iloc[-21]-1) > 0:
            signals.append('rel_str_pos')

    # Signal 4: RSI not overbought (RSI < 80)
    rets = px.pct_change().dropna()
    if len(rets) > 14:
        gains = rets.clip(lower=0).rolling(14).mean()
        losses = (-rets.clip(upper=0)).rolling(14).mean()
        rs = gains/(losses+1e-10); rsi = 100-100/(1+rs)
        r = float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else 50
        if r < 80: signals.append('rsi_ok')

    # Signal 5: VIX context (VIX > 15 = enough premium)
    cv = float(vix.loc[dt]) if dt in vix.index else 20
    if cv > 15: signals.append('vix_premium')

    return len(signals) >= 2, len(signals), signals

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

# ==================== SIMULATION (v3: exit management) ====================
def simulate(name, rankings, sc, sh, sl, spy, vix,
             spread_pct=3.0, dte=30, top_k=3,
             vix_min=None, vix_max=None,
             use_confluence=True, sizing='fixed',
             early_exit_day=None, trailing_stop_pct=None):
    """
    NEW in v3:
    - early_exit_day: exit position after N trading days (e.g., 20)
    - trailing_stop_pct: exit if value drops X% from peak (e.g., 0.30)
    """
    sma200 = spy.rolling(200).mean()
    atr_d = {tk: compute_atr(sh[tk], sl[tk], sc[tk]) for tk in sc.columns if tk in sh.columns}
    equity, trades, eq_curve = CAP, [], [CAP]
    confluence_blocks, confluence_passes = 0, 0

    for dt in sorted(rankings.keys()):
        if dt not in spy.index or dt not in vix.index: continue
        cv = float(vix.loc[dt])

        # VIX filter
        if vix_min is not None and cv < vix_min: continue
        if vix_max is not None and cv > vix_max: continue

        sv_val = float(spy.loc[dt])
        sm = float(sma200.loc[dt]) if dt in sma200.index and not pd.isna(sma200.loc[dt]) else sv_val
        bull = sv_val >= sm
        scores = rankings[dt]
        if not scores: continue
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        picks = [t for t, _ in ranked[:top_k]]

        # Position sizing
        if sizing == 'fixed':
            max_pos = min(200, equity/3)
        elif sizing == 'sqrt':
            max_pos = min(2000, 200 * np.sqrt(equity/CAP))
        elif sizing == 'tiered':
            if equity < 2000: max_pos = 200
            elif equity < 10000: max_pos = 500
            elif equity < 50000: max_pos = 1000
            else: max_pos = 2000
        else:
            max_pos = min(200, equity/3)

        if max_pos < 30: eq_curve.append(equity); continue
        n_ent = 0
        for tk in picks:
            if tk not in sc.columns or tk not in atr_d or n_ent >= 3: continue

            # Confluence gate (HC #750)
            if use_confluence:
                passes, n_sig, sigs = confluence_check(tk, dt, sc, spy, vix)
                if not passes:
                    confluence_blocks += 1; continue
                confluence_passes += 1

            S = float(sc[tk].loc[dt])
            av = float(atr_d[tk].loc[dt]) if dt in atr_d[tk].index and not pd.isna(atr_d[tk].loc[dt]) else S*0.015
            di = sc.index.get_loc(dt)

            # Determine max hold period
            max_hold = dte
            if early_exit_day is not None:
                max_hold = early_exit_day

            ei = min(di + max_hold, len(sc)-1)
            K1, K2 = round(S), round(S*(1+spread_pct/100))
            lp = atr_premium(S, K1, dte, av, cv, 'call')*(1+HAIRCUT)
            sp = atr_premium(S, K2, dte, av, cv, 'call')*(1-HAIRCUT)
            val = lp - sp; width = K2 - K1
            cost = val*100 + SPREAD_COMM; mx_prof = (width-val)*100 - SPREAD_COMM
            if cost <= 0 or cost > max_pos or cost > equity*0.40: continue

            pnl, aei = None, ei
            peak_value = cost  # Track peak position value for trailing stop

            for ci in range(di+1, ei+1):
                if ci >= len(sc): break
                Sc = float(sc[tk].iloc[ci])
                dh = ci - di
                rd = max(0, dte - dh)  # Remaining days to original expiry
                ac = float(atr_d[tk].iloc[ci]) if ci < len(atr_d[tk]) else av
                tm = np.sqrt(rd/max(dte,1))

                # Current spread value = intrinsic + time value
                intrinsic = (max(0, Sc-K1) - max(0, Sc-K2)) * 100
                time_val = ac * tm * 0.3 * 100
                current_val = intrinsic + time_val

                # Track peak for trailing stop
                peak_value = max(peak_value, current_val)

                cp = current_val - cost

                # Exit conditions:
                # 1. Take profit at 50% of max profit
                if cp >= mx_prof * 0.50:
                    pnl = cp; aei = ci; break

                # 2. Trailing stop (if enabled)
                if trailing_stop_pct is not None and peak_value > cost * 1.1:
                    if current_val < peak_value * (1 - trailing_stop_pct):
                        pnl = cp; aei = ci; break

                # 3. Time exit (at early_exit_day or dte)
                if ci == ei:
                    # At exit day, compute residual value
                    if early_exit_day is not None and rd > 0:
                        # Still has time value — sell the spread
                        pnl = cp
                    else:
                        # At expiry — intrinsic only
                        pnl = intrinsic - cost
                    aei = ci; break

            if pnl is None:
                Se = float(sc[tk].iloc[ei]) if ei < len(sc) else S
                pnl = (max(0,Se-K1)-max(0,Se-K2))*100 - cost

            equity += pnl; n_ent += 1
            trades.append({'entry': str(dt.date()), 'exit': str(sc.index[aei].date()),
                           'ticker': tk, 'pnl': round(pnl,2), 'win': pnl>0,
                           'hold_days': aei-di,
                           'regime': 'bull' if bull else 'bear', 'vix': round(cv,1),
                           'equity_at_trade': round(equity,2)})
        eq_curve.append(equity)

    return trades, equity, eq_curve

# ==================== HONEST METRICS (equity-based Sharpe) ====================
def compute_honest_sharpe(trades):
    """Calendar-month Sharpe using equity-based returns (not fixed capital)."""
    if not trades: return 0.0, 0.0, []
    tdf = pd.DataFrame(trades)
    tdf['entry_dt'] = pd.to_datetime(tdf['entry'])
    tdf['month'] = tdf['entry_dt'].dt.to_period('M')

    # Get equity at start of each month
    monthly = []
    for mo in sorted(tdf['month'].unique()):
        mo_trades = tdf[tdf['month'] == mo]
        mo_pnl = mo_trades['pnl'].sum()
        # Use the equity at the START of the month's first trade
        eq_start = mo_trades['equity_at_trade'].iloc[0] - mo_trades['pnl'].iloc[0]
        eq_start = max(eq_start, 100)  # Floor to avoid division by tiny numbers
        mo_return = mo_pnl / eq_start
        monthly.append({'month': mo, 'pnl': mo_pnl, 'equity': eq_start, 'return': mo_return})

    mdf = pd.DataFrame(monthly)
    rets = mdf['return'].values
    if len(rets) < 4: return 0.0, 0.0, rets

    sharpe = float(np.mean(rets) / (np.std(rets) + 1e-10) * np.sqrt(12))
    dn = rets[rets < 0]
    sortino = float(np.mean(rets) / (np.std(dn) + 1e-10) * np.sqrt(12)) if len(dn) > 1 else 0.0

    return sharpe, sortino, rets

def metrics_validate(trades, final_eq, eq_curve, name):
    if not trades: fprint(f"  {name}: No trades"); return None
    n = len(trades); wins = sum(1 for t in trades if t['win']); wr = wins/n*100
    pnls = [t['pnl'] for t in trades]
    hold_days = [t['hold_days'] for t in trades]
    avg_hold = np.mean(hold_days)

    # HONEST Sharpe (equity-based, calendar months)
    sh, so, monthly_rets = compute_honest_sharpe(trades)

    ny = max(len(monthly_rets)/12, 0.5)
    cagr = (final_eq/CAP)**(1/ny)-1
    eq = np.array(eq_curve); pk = np.maximum.accumulate(eq); mdd = float(((eq-pk)/(pk+1e-10)).min())
    gp = sum(p for p in pnls if p>0); gl = abs(sum(p for p in pnls if p<=0))
    pf = gp/(gl+1e-10)

    bt = [t for t in trades if t['regime']=='bull']; brt = [t for t in trades if t['regime']=='bear']
    bw = sum(1 for t in bt if t['win'])/max(len(bt),1)*100
    brw = sum(1 for t in brt if t['win'])/max(len(brt),1)*100

    # 4-gate validation on HONEST monthly returns
    rets = monthly_rets; gates = 0
    pp, rg, g1, g2, g3, g4, h1, h2 = 1.0, 1.0, False, False, False, False, 0, 0
    if len(rets) >= 10:
        rs = np.mean(rets)/(np.std(rets)+1e-10)
        pp = sum(1 for _ in range(2000) if np.mean(rets*np.random.choice([-1,1],len(rets)))/(np.std(rets)+1e-10)>=rs)/2000
        g1 = pp < 0.05; gates += g1
        rg = abs(bw-brw)/max(bw,brw,1); g2 = rg < 0.50; gates += g2
        mid = len(rets)//2
        h1s = rets[:mid]; h2s = rets[mid:]
        h1 = np.mean(h1s)/(np.std(h1s)+1e-10) if len(h1s) > 3 else 0
        h2 = np.mean(h2s)/(np.std(h2s)+1e-10) if len(h2s) > 3 else 0
        g3 = h1 > 0 and h2 > 0; gates += g3
        tr = np.sort(rets)[:-1]; g4 = np.mean(tr)/(np.std(tr)+1e-10) > 0 if len(rets) > 5 else False; gates += g4

    r = {'name': name, 'n_trades': n, 'win_rate': round(wr,1),
         'sharpe': round(sh,2), 'sortino': round(so,2),
         'cagr_pct': round(cagr*100,1), 'maxdd_pct': round(mdd*100,1),
         'pf': round(pf,2), 'avg_pnl': round(np.mean(pnls),2),
         'avg_hold_days': round(avg_hold,1),
         'final_equity': round(final_eq,2),
         'bull_wr': round(bw,1), 'bear_wr': round(brw,1), 'bull_n': len(bt), 'bear_n': len(brt),
         'gates': gates, 'perm_p': round(pp,4), 'r1_gap': round(rg,3),
         'g1_perm': g1, 'g2_regime': g2, 'g3_sub': g3, 'g4_outlier': g4,
         'h1_sh': round(h1,2), 'h2_sh': round(h2,2)}

    fprint(f"  {name}: {n} trades | WR {wr:.1f}% | Sh {sh:.2f} | Sort {so:.2f} | "
           f"CAGR {cagr*100:.1f}% | MDD {mdd*100:.1f}% | PF {pf:.2f} | "
           f"AvgHold {avg_hold:.1f}d | ${CAP:.0f}->${final_eq:.0f}")
    fprint(f"  Gates: G1={'P' if g1 else 'F'}(p={pp:.4f}) G2={'P' if g2 else 'F'}(gap={rg:.3f}) "
           f"G3={'P' if g3 else 'F'}({h1:.2f}/{h2:.2f}) G4={'P' if g4 else 'F'} => {gates}/4")
    return r

# ==================== RANDOM DIRECTION CONTROL ====================
def random_direction_test(rankings, sc, sh, sl, spy, vix, best_config):
    """Test if random direction (random tickers) also produces similar returns."""
    fprint(f"\n=== RANDOM DIRECTION CONTROL TEST ===")
    np.random.seed(42)
    random_sharpes = []
    for trial in range(5):
        # Shuffle ticker rankings randomly
        rand_rankings = {}
        all_tickers = list(sc.columns)
        for dt, scores in rankings.items():
            shuffled = {tk: np.random.random() for tk in scores.keys()}
            rand_rankings[dt] = shuffled

        tr, eq, cu = simulate(f'Random_{trial}', rand_rankings, sc, sh, sl, spy, vix,
                               **best_config)
        if tr:
            sh_r, _, _ = compute_honest_sharpe(tr)
            random_sharpes.append(sh_r)
            fprint(f"  Random trial {trial}: Sharpe {sh_r:.2f}, ${CAP}->${eq:.0f}")

    if random_sharpes:
        fprint(f"  Random mean Sharpe: {np.mean(random_sharpes):.2f} (vs real LGBM)")
        fprint(f"  If random is also profitable => structural edge, NOT ML edge")
    return random_sharpes

# ==================== MAIN ====================
def main():
    t0 = datetime.now()
    fprint(f"Production Sector Momentum v3 — {t0.strftime('%Y-%m-%d %H:%M:%S')}")
    fprint(f"{'='*80}")
    fprint(f"NEW: 20d early exit + trailing stop + tiered sizing + HONEST Sharpe")
    fprint(f"Capital: ${CAP:.0f} | ATR+{HAIRCUT:.0%} haircut | Comm: ${SPREAD_COMM}/spread")
    fprint(f"{'='*80}")

    sc, sh, sl, sv, spy, vix = download_data()
    bd = pd.DatetimeIndex(sc.index.to_series().resample('2W-FRI').last().dropna().values)

    # Build rankings (quality-momentum was best per v2)
    rankings = build_rankings(sc, sv, spy, bd, QM_COLS)
    if not rankings: fprint("FATAL: No rankings"); return

    # 8 Variants testing exit management + sizing combinations
    fprint(f"\n=== Simulating 8 Production Variants ===")
    results = []
    configs = [
        # name, early_exit_day, trailing_stop_pct, sizing, vix_min, vix_max
        ('A_Baseline_30d',        None,  None,  'fixed',  20, None),   # Control: v2 style
        ('B_Exit20d_Fixed',       20,    None,  'fixed',  20, None),   # NEW: 20d exit
        ('C_Exit20d_Tiered',      20,    None,  'tiered', 20, None),   # NEW: 20d + tiered
        ('D_Exit20d_Sqrt',        20,    None,  'sqrt',   20, None),   # NEW: 20d + sqrt
        ('E_Trail30_Tiered',      None,  0.30,  'tiered', 20, None),   # NEW: trailing stop + tiered
        ('F_FullStack',           20,    0.30,  'tiered', 20, None),   # NEW: 20d + trail + tiered
        ('G_HighConv_Sqrt',       20,    None,  'sqrt',   25, None),   # NEW: VIX>25 + 20d + sqrt
        ('H_Conservative',        20,    None,  'fixed',  20, 35),     # NEW: VIX 20-35 band
    ]

    for nm, exit_day, trail, sz, vmin, vmax in configs:
        tr, eq, cu = simulate(nm, rankings, sc, sh, sl, spy, vix,
                               spread_pct=3.0, dte=30, top_k=3,
                               vix_min=vmin, vix_max=vmax,
                               use_confluence=True, sizing=sz,
                               early_exit_day=exit_day, trailing_stop_pct=trail)
        r = metrics_validate(tr, eq, cu, nm)
        if r: results.append(r)

    if not results: fprint("No results"); return

    # Random direction control test on best config
    best = max(results, key=lambda x: x['sharpe'])
    best_idx = next(i for i, (nm,*_) in enumerate(configs) if nm == best['name'])
    best_cfg = configs[best_idx]
    random_sharpes = random_direction_test(rankings, sc, sh, sl, spy, vix, {
        'spread_pct': 3.0, 'dte': 30, 'top_k': 3,
        'vix_min': best_cfg[4], 'vix_max': best_cfg[5],
        'use_confluence': True, 'sizing': best_cfg[3],
        'early_exit_day': best_cfg[1], 'trailing_stop_pct': best_cfg[2]
    })

    # Summary table
    fprint(f"\n{'='*120}")
    fprint(f"SUMMARY — Production Sector Momentum v3 (HONEST SHARPE)")
    fprint(f"{'='*120}")
    fprint(f"{'Variant':<22} {'#':>5} {'WR':>6} {'Sharpe':>7} {'Sort':>6} {'CAGR':>7} {'MDD':>7} {'PF':>6} {'Hold':>5} {'Final$':>9} {'G':>4}")
    fprint("-"*120)
    for r in sorted(results, key=lambda x: x['sharpe'], reverse=True):
        fprint(f"{r['name']:<22} {r['n_trades']:>5} {r['win_rate']:>5.1f}% {r['sharpe']:>7.2f} "
               f"{r['sortino']:>6.2f} {r['cagr_pct']:>6.1f}% {r['maxdd_pct']:>6.1f}% "
               f"{r['pf']:>6.2f} {r['avg_hold_days']:>4.0f}d ${r['final_equity']:>8.0f} {r['gates']:>3}/4")

    # v3 vs v2 comparison
    a_result = next((r for r in results if r['name'] == 'A_Baseline_30d'), None)
    fprint(f"\n=== v3 IMPROVEMENTS vs v2 BASELINE ===")
    if a_result:
        for r in sorted(results, key=lambda x: x['sharpe'], reverse=True):
            if r['name'] == 'A_Baseline_30d': continue
            sh_chg = (r['sharpe'] - a_result['sharpe']) / max(abs(a_result['sharpe']), 0.01) * 100
            eq_chg = (r['final_equity'] - a_result['final_equity']) / max(a_result['final_equity'], 1) * 100
            fprint(f"  {r['name']:<22}: Sharpe {r['sharpe']:+.2f} ({sh_chg:+.0f}%), "
                   f"Final ${r['final_equity']:.0f} ({eq_chg:+.0f}%)")

    if random_sharpes:
        fprint(f"\n  Random control: mean Sharpe {np.mean(random_sharpes):.2f} vs LGBM {best['sharpe']:.2f}")
        if np.mean(random_sharpes) > best['sharpe'] * 0.8:
            fprint(f"  WARNING: Random is >80% of real => mostly STRUCTURAL edge, ML adds marginal value")
        else:
            fprint(f"  GOOD: ML adds significant value over random selection")

    fprint(f"\n=== PRODUCTION RECOMMENDATION ===")
    fprint(f"BEST: {best['name']} — Sharpe {best['sharpe']}, CAGR {best['cagr_pct']}%, "
           f"MaxDD {best['maxdd_pct']}%, Gates {best['gates']}/4")

    elapsed = (datetime.now() - t0).total_seconds()
    fprint(f"\nRuntime: {elapsed:.0f}s ({elapsed/60:.1f}m)")

    # Save results
    save_data = {
        'timestamp': t0.isoformat(),
        'capital': CAP,
        'version': 'v3_production_optimized',
        'improvements': ['20d_early_exit', 'trailing_stop', 'tiered_sizing', 'honest_sharpe', 'random_control'],
        'results': results,
        'random_control_sharpes': random_sharpes,
        'runtime_s': round(elapsed,1)
    }
    with open(RESULTS_PATH, 'w') as f:
        json.dump(save_data, f, indent=2, default=str)
    fprint(f"Results saved")

    # MLflow logging
    if MLFLOW_OK:
        try:
            en = 'production_sector_v3'
            try:
                exp = mlflow.get_experiment_by_name(en)
                if not exp: mlflow.create_experiment(en)
            except: pass
            mlflow.set_experiment(en)
            with mlflow.start_run(run_name=f"prod_v3_{t0.strftime('%Y%m%d_%H%M')}"):
                mlflow.log_params({
                    'pricing': 'ATR', 'haircut': HAIRCUT, 'capital': CAP,
                    'features': 'quality_momentum_20', 'vix_filter': '20+',
                    'confluence': 'HC750', 'sharpe_method': 'honest_equity_based',
                    'new_features': '20d_exit+trailing_stop+tiered_sizing'
                })
                for r in results:
                    p = r['name'][:18].replace(' ','_')
                    mlflow.log_metrics({
                        f'{p}_sh': r['sharpe'], f'{p}_cagr': r['cagr_pct'],
                        f'{p}_mdd': r['maxdd_pct'], f'{p}_wr': r['win_rate'],
                        f'{p}_pf': r['pf'], f'{p}_gates': r['gates'],
                        f'{p}_hold': r['avg_hold_days']
                    })
                if random_sharpes:
                    mlflow.log_metrics({'random_mean_sharpe': np.mean(random_sharpes),
                                        'random_std_sharpe': np.std(random_sharpes)})
                mlflow.log_artifact(str(RESULTS_PATH))
                fprint("MLflow logged")
        except Exception as e:
            fprint(f"MLflow failed: {e}")

    fprint(f"\n{'='*80}\nDONE — Production Sector Momentum v3\n{'='*80}")

if __name__ == '__main__':
    main()
