#!/usr/bin/env python3
"""Regime Integration v1 — Tests GRU continuous regime score vs crude VIX>20 binary filter
for sector bull call spread strategy.

Key question: Does the GRU continuous regime score REDUCE whipsaw and IMPROVE Sharpe
vs the crude VIX>20 binary?

6 Variants:
  A. BASELINE: Binary VIX>20 filter (current approach)
  B. REGIME_SCORE > 0.5: Only trade when GRU regime > 0.5
  C. REGIME_SCORE > 0.4: More permissive threshold
  D. REGIME_SCORE > 0.6: More conservative threshold
  E. ADAPTIVE_SIZE: Trade when regime>0.3, scale size by score
  F. COMBINED: Trade when BOTH VIX>18 AND regime>0.4

$645 starting capital, LGBM sector ranking, hold to expiry (30d),
15% bid-ask haircut on BOTH entry AND exit, walk-forward LGBM, honest Sharpe.
Full adversarial validation (permutation, regime split, sub-period, yearly).

MLflow: regime_integration_v1 at http://jupiter:5000
"""

import json, warnings, time, sys
from pathlib import Path
from datetime import datetime
import numpy as np, pandas as pd
from scipy import stats
import lightgbm as lgb
warnings.filterwarnings('ignore')
np.random.seed(42)

def fprint(*a, **kw): print(*a, **kw, flush=True)

BASE = Path(__file__).resolve().parents[2]
OUT_DIR = BASE / 'research' / 'findings'; OUT_DIR.mkdir(parents=True, exist_ok=True)

SECTORS = ['XLK','XLF','XLE','XLV','XLY','XLP','XLI','XLB','XLU','XLRE','XLC']
CAP = 645.0
SPREAD_COMM = 2.60
HAIRCUT = 0.15
MAX_POS = 200
MAX_CONC = 3
HOLD_DAYS = 30  # hold to expiry
SPREAD_WIDTH_PCT = 3.0

LGBM_FEAT = ['ret_5d','ret_10d','ret_21d','ret_63d','ret_126d','ret_252d',
             'vol_21d','vol_63d','sharpe_63d','maxdd_63d','pct_52w_high','mom_accel']

# MLflow
MLFLOW_OK = False
try:
    import urllib.request; urllib.request.urlopen('http://jupiter:5000/', timeout=3)
    import mlflow; mlflow.set_tracking_uri('http://jupiter:5000'); MLFLOW_OK = True
    fprint("MLflow: connected")
except Exception as e:
    fprint(f"MLflow: unavailable ({e})")


# ==================== 1. LOAD REGIME PREDICTIONS ====================

def load_regime_predictions():
    fprint("[1/7] Loading GRU regime predictions...")
    npz_path = BASE / 'output' / 'regime_detector_v1' / 'regime_predictions_v1.npz'
    d = np.load(str(npz_path), allow_pickle=True)
    dates = pd.DatetimeIndex(d['dates'])
    scores = d['regime_scores']
    regime = pd.Series(scores, index=dates, name='regime_score')
    # Remove any duplicates (keep last)
    regime = regime[~regime.index.duplicated(keep='last')]
    fprint(f"  Loaded {len(regime)} regime scores, range [{scores.min():.3f}, {scores.max():.3f}]")
    fprint(f"  Date range: {dates[0].strftime('%Y-%m-%d')} to {dates[-1].strftime('%Y-%m-%d')}")
    fprint(f"  Mean={scores.mean():.3f}, Std={scores.std():.3f}")
    fprint(f"  Quintiles: " + ", ".join(f"Q{i+1}={np.percentile(scores, (i+1)*20):.3f}" for i in range(4)))
    return regime


# ==================== 2. DOWNLOAD MARKET DATA ====================

def download_data():
    import yfinance as yf
    fprint("[2/7] Downloading sector ETF + market data...")
    tickers = SECTORS + ['SPY', '^VIX']
    raw = yf.download(tickers, start='2008-01-01', end='2026-07-26', progress=False)
    mi = isinstance(raw.columns, pd.MultiIndex)
    close = raw['Close'] if mi else raw
    high = raw['High'] if mi else raw
    low = raw['Low'] if mi else raw
    vc = '^VIX' if '^VIX' in close.columns else 'VIX'
    vix = close[vc].dropna()
    spy = close['SPY'].dropna()
    sc = close[[c for c in SECTORS if c in close.columns]].dropna(how='all')
    sh = high[[c for c in SECTORS if c in high.columns]].dropna(how='all')
    sl = low[[c for c in SECTORS if c in low.columns]].dropna(how='all')
    ix = sc.index
    for s in [vix, spy, sh, sl]:
        ix = ix.intersection(s.index)
    fprint(f"  {len(ix)} trading days, {len(sc.columns)} sectors")
    return sc.loc[ix], sh.loc[ix], sl.loc[ix], spy.loc[ix], vix.loc[ix]


# ==================== 3. LGBM SECTOR RANKING (Walk-Forward) ====================

def sector_features(px, idx, tk):
    p = px[tk].iloc[:idx+1].dropna()
    if len(p) < 260:
        return None
    f = {}
    for lb, nm in [(5,'ret_5d'),(10,'ret_10d'),(21,'ret_21d'),(63,'ret_63d'),(126,'ret_126d'),(252,'ret_252d')]:
        f[nm] = float(p.iloc[-1]/p.iloc[-lb]-1) if len(p) > lb else 0.0
    r = p.pct_change().dropna()
    f['vol_21d'] = float(r.iloc[-21:].std()*np.sqrt(252)) if len(r) > 21 else 0.2
    f['vol_63d'] = float(r.iloc[-63:].std()*np.sqrt(252)) if len(r) > 63 else 0.2
    r63 = r.iloc[-63:]
    f['sharpe_63d'] = float(r63.mean()/(r63.std()+1e-10)*np.sqrt(252)) if len(r63) > 10 else 0
    p63 = p.iloc[-63:]
    f['maxdd_63d'] = float(((p63/p63.cummax())-1).min())
    f['pct_52w_high'] = float(p.iloc[-1]/p.iloc[-252:].max())
    f['mom_accel'] = f['ret_21d'] - f['ret_63d']/3
    return f


def lgbm_wf(px, dates, tp=12):
    fprint("[3/7] LightGBM walk-forward ranking...")
    recs = []
    for dt in dates:
        idx = px.index.get_indexer([dt], method='ffill')[0]
        if idx < 260:
            continue
        for tk in px.columns:
            f = sector_features(px, idx, tk)
            if not f:
                continue
            fi = min(idx+21, len(px)-1)
            f.update({'date': dt, 'ticker': tk, 'fwd_ret': float(px[tk].iloc[fi]/px[tk].iloc[idx]-1)})
            recs.append(f)
    df = pd.DataFrame(recs)
    if len(df) < 100:
        return {}
    df['rank_label'] = df.groupby('date')['fwd_ret'].rank(pct=True)
    udates = sorted(df['date'].unique())
    ranks = {}
    for i in range(tp, len(udates)):
        td = udates[max(0, i-tp):i]
        tdate = udates[i]
        tr = df[df['date'].isin(td)]
        te = df[df['date'] == tdate].copy()
        if len(te) < 3 or len(tr) < 50:
            continue
        Xt = np.nan_to_num(tr[LGBM_FEAT].values.astype(np.float32))
        Xe = np.nan_to_num(te[LGBM_FEAT].values.astype(np.float32))
        try:
            try:
                m = lgb.LGBMRegressor(n_estimators=100, max_depth=4, learning_rate=0.05,
                    subsample=0.8, colsample_bytree=0.8, min_child_samples=5,
                    device='gpu', verbose=-1)
                m.fit(Xt, tr['rank_label'].values)
            except:
                m = lgb.LGBMRegressor(n_estimators=100, max_depth=4, learning_rate=0.05,
                    subsample=0.8, colsample_bytree=0.8, min_child_samples=5, verbose=-1)
                m.fit(Xt, tr['rank_label'].values)
            te['score'] = m.predict(Xe)
            ranks[tdate] = dict(zip(te['ticker'], te['score']))
        except:
            continue
    fprint(f"  Rankings generated for {len(ranks)} dates")
    return ranks


# ==================== 4. ATR OPTIONS PRICING ====================

def compute_atr(h, l, c, p=14):
    tr = pd.DataFrame({'hl': h-l, 'hc': abs(h-c.shift(1)), 'lc': abs(l-c.shift(1))}).max(axis=1)
    return tr.rolling(p).mean()


def atr_prem(S, K, dte, atr, vx, opt='call'):
    T = dte / 252.0
    if T <= 0:
        return max(0, S-K) if opt == 'call' else max(0, K-S)
    intrinsic = max(0, S-K) if opt == 'call' else max(0, K-S)
    extrinsic = atr * np.sqrt(T) * max(0.3, vx/20) * np.exp(-3*abs(S-K)/S)
    return intrinsic + extrinsic


def price_spread(S, spct, dte, atr, vx):
    K1 = round(S)
    K2 = round(S * (1 + spct/100))
    lp = atr_prem(S, K1, dte, atr, vx, 'call') * (1 + HAIRCUT)  # buy long: pay more
    sp = atr_prem(S, K2, dte, atr, vx, 'call') * (1 - HAIRCUT)  # sell short: receive less
    debit = lp - sp
    max_profit = (K2 - K1 - debit) * 100 - SPREAD_COMM
    max_loss = debit * 100 + SPREAD_COMM
    return debit, max_profit, max_loss, K1, K2


# ==================== 5. SIMULATION ====================

def get_regime_for_date(regime_scores, dt):
    """Get regime score for a date, with forward-fill for missing dates."""
    if dt in regime_scores.index:
        return float(regime_scores.loc[dt])
    # Find most recent available score
    mask = regime_scores.index <= dt
    if mask.any():
        return float(regime_scores.loc[regime_scores.index[mask][-1]])
    return 0.35  # default to mean


def simulate(variant, ranks, sc, sh, sl, spy, vix, regime_scores, atr_d):
    """Run backtest for a specific variant."""
    eq = CAP
    trades = []
    curve = [CAP]
    sma200 = spy.rolling(200).mean()
    whipsaw_count = 0  # track filter on/off transitions
    prev_trading = None

    for dt in sorted(ranks.keys()):
        if dt not in spy.index or dt not in vix.index:
            continue

        cv = float(vix.loc[dt])
        scores = ranks[dt]
        if not scores:
            curve.append(eq)
            continue

        rs = get_regime_for_date(regime_scores, dt)
        size_mult = 1.0
        should_trade = True

        # Apply variant filter logic
        if variant == 'A_BASELINE':
            # Current approach: only trade when VIX > 20
            should_trade = cv > 20

        elif variant == 'B_REGIME_GT_0.5':
            # Replace VIX filter with regime score > 0.5
            should_trade = rs > 0.5

        elif variant == 'C_REGIME_GT_0.4':
            # More permissive regime threshold
            should_trade = rs > 0.4

        elif variant == 'D_REGIME_GT_0.6':
            # More conservative regime threshold
            should_trade = rs > 0.6

        elif variant == 'E_ADAPTIVE_SIZE':
            # Always trade if regime > 0.3, scale size by score
            should_trade = rs > 0.3
            # Scale: regime 0.3 -> 0.3x, regime 0.8 -> 1.6x
            size_mult = rs * 2.0

        elif variant == 'F_COMBINED':
            # Both signals: VIX > 18 AND regime > 0.4
            should_trade = cv > 18 and rs > 0.4

        # Track whipsaw
        if prev_trading is not None and should_trade != prev_trading:
            whipsaw_count += 1
        prev_trading = should_trade

        if not should_trade:
            curve.append(eq)
            continue

        # Pick top sectors from LGBM ranking
        picks = [t for t, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)[:3]]
        mx = min(MAX_POS * size_mult, eq / 3)
        if mx < 30:
            curve.append(eq)
            continue

        bull = float(spy.loc[dt]) >= (float(sma200.loc[dt]) if dt in sma200.index and not pd.isna(sma200.loc[dt]) else float(spy.loc[dt]))
        n_entered = 0

        for tk in picks:
            if tk not in sc.columns or tk not in atr_d or n_entered >= MAX_CONC:
                continue

            S = float(sc[tk].loc[dt])
            di = sc.index.get_loc(dt)
            dte = HOLD_DAYS

            av = float(atr_d[tk].loc[dt]) if dt in atr_d[tk].index and not pd.isna(atr_d[tk].loc[dt]) else S * 0.015

            ei = min(di + dte, len(sc) - 1)
            Se = float(sc[tk].iloc[ei])

            debit, max_profit, max_loss, K1, K2 = price_spread(S, SPREAD_WIDTH_PCT, dte, av, cv)
            cost = debit * 100 + SPREAD_COMM

            if cost <= 0 or cost > mx or cost > eq * 0.40:
                continue

            # Hold to expiry — compute P&L at expiration
            pnl = (max(0, Se - K1) - max(0, Se - K2)) * 100 - cost

            # Apply exit haircut (15% on exit value too)
            exit_val = (max(0, Se - K1) - max(0, Se - K2)) * 100
            if exit_val > 0:
                exit_val_after_haircut = exit_val * (1 - HAIRCUT)
                pnl = exit_val_after_haircut - cost

            eq += pnl
            n_entered += 1
            trades.append({
                'entry': str(dt.date()),
                'exit': str(sc.index[ei].date()),
                'ticker': tk,
                'pnl': round(pnl, 2),
                'win': pnl > 0,
                'regime': 'bull' if bull else 'bear',
                'vix': round(cv, 1),
                'regime_score': round(rs, 3),
            })

        curve.append(eq)

    return trades, eq, curve, whipsaw_count


# ==================== 6. METRICS ====================

def compute_metrics(trades, feq, curve, name, whipsaw_count=0):
    if not trades:
        fprint(f"  {name}: No trades")
        return None

    n = len(trades)
    wins = sum(1 for t in trades if t['win'])
    wr = wins / n * 100
    pnls = [t['pnl'] for t in trades]

    # Honest Sharpe: equity-based, calendar month
    tdf = pd.DataFrame(trades)
    tdf['m'] = pd.to_datetime(tdf['entry']).dt.to_period('M')
    mr = tdf.groupby('m')['pnl'].sum() / CAP
    ny = max(len(mr) / 12, 0.5)
    sharpe = (mr.mean() * 12) / (mr.std() * np.sqrt(12) + 1e-10) if len(mr) > 3 else 0
    dn = mr[mr < 0]
    sortino = (mr.mean() * 12) / (dn.std() * np.sqrt(12) + 1e-10) if len(dn) > 1 else 0
    cagr = (feq / CAP) ** (1 / ny) - 1

    eq_arr = np.array(curve)
    pk = np.maximum.accumulate(eq_arr)
    mdd = float(((eq_arr - pk) / (pk + 1e-10)).min())

    gp = sum(p for p in pnls if p > 0)
    gl = abs(sum(p for p in pnls if p <= 0))
    pf = gp / (gl + 1e-10)

    # Max consecutive losses
    mcl = c = 0
    for t in trades:
        if not t['win']:
            c += 1
            mcl = max(mcl, c)
        else:
            c = 0

    # Regime split
    bt = [t for t in trades if t.get('regime') == 'bull']
    brt = [t for t in trades if t.get('regime') == 'bear']
    bw = sum(1 for t in bt if t['win']) / max(len(bt), 1) * 100
    brw = sum(1 for t in brt if t['win']) / max(len(brt), 1) * 100

    # Avg regime score
    avg_rs = np.mean([t.get('regime_score', 0) for t in trades])

    r = {
        'name': name, 'n_trades': n, 'win_rate': round(wr, 1),
        'sharpe': round(sharpe, 2), 'sortino': round(sortino, 2),
        'cagr_pct': round(cagr * 100, 1), 'maxdd_pct': round(mdd * 100, 1),
        'pf': round(pf, 2), 'avg_pnl': round(np.mean(pnls), 2),
        'final_equity': round(feq, 2), 'total_pnl': round(sum(pnls), 2),
        'max_consec_loss': mcl, 'bull_wr': round(bw, 1), 'bear_wr': round(brw, 1),
        'bull_n': len(bt), 'bear_n': len(brt),
        'whipsaw_count': whipsaw_count,
        'avg_regime_score': round(avg_rs, 3),
        'monthly_returns': mr.values.tolist(),
    }

    fprint(f"  {name:22s} | {n:4d} trades | WR {wr:5.1f}% | Sh {sharpe:5.2f} | So {sortino:5.2f} | "
           f"CAGR {cagr*100:5.1f}% | MDD {mdd*100:5.1f}% | PF {pf:5.2f} | "
           f"${CAP:.0f}->${feq:.0f} | MCL {mcl} | Whip {whipsaw_count}")
    return r


# ==================== 7. ADVERSARIAL VALIDATION ====================

def adversarial_validation(r):
    """Full adversarial validation: permutation, regime split, sub-period, yearly."""
    if r is None:
        return r

    rets = np.array(r['monthly_returns'])
    if len(rets) < 10:
        r.update({'gates': 0, 'perm_p': 1.0, 'r1_gap': 1.0})
        fprint(f"    {r['name']}: Too few months ({len(rets)}) for validation")
        return r

    gates = 0
    rs_actual = np.mean(rets) / (np.std(rets) + 1e-10)

    # Gate 1: Permutation test (p < 0.05)
    N_PERM = 2000
    perm_sharpes = np.zeros(N_PERM)
    for i in range(N_PERM):
        shuffled = rets * np.random.choice([-1, 1], len(rets))
        perm_sharpes[i] = np.mean(shuffled) / (np.std(shuffled) + 1e-10)
    perm_p = (perm_sharpes >= rs_actual).sum() / N_PERM
    g1 = perm_p < 0.05
    gates += g1

    # Gate 2: Regime gap < 0.50
    rg = abs(r['bull_wr'] - r['bear_wr']) / max(r['bull_wr'], r['bear_wr'], 1)
    g2 = rg < 0.50
    gates += g2

    # Gate 3: Sub-period stability (both halves positive Sharpe)
    mid = len(rets) // 2
    h1_sh = np.mean(rets[:mid]) / (np.std(rets[:mid]) + 1e-10) if mid > 3 else 0
    h2_sh = np.mean(rets[mid:]) / (np.std(rets[mid:]) + 1e-10) if len(rets) - mid > 3 else 0
    g3 = h1_sh > 0 and h2_sh > 0
    gates += g3

    # Gate 4: Outlier robustness (Sharpe > 0 after removing best month)
    sorted_rets = np.sort(rets)
    trimmed = sorted_rets[:-1]  # remove best month
    g4 = np.mean(trimmed) / (np.std(trimmed) + 1e-10) > 0 if len(trimmed) > 3 else False
    gates += g4

    # Gate 5: Yearly consistency — at least 60% of years profitable
    tdf_months = pd.Series(rets, index=pd.period_range(start='2012', periods=len(rets), freq='M'))
    yearly = tdf_months.groupby(tdf_months.index.year).sum()
    pct_profitable_years = (yearly > 0).mean() if len(yearly) > 0 else 0
    g5 = pct_profitable_years >= 0.60
    gates += g5

    r.update({
        'gates': gates,
        'perm_p': round(perm_p, 4),
        'r1_gap': round(rg, 3),
        'g1_perm': g1, 'g2_regime': g2, 'g3_sub': g3, 'g4_outlier': g4, 'g5_yearly': g5,
        'h1_sh': round(h1_sh, 2), 'h2_sh': round(h2_sh, 2),
        'pct_profitable_years': round(pct_profitable_years * 100, 1),
    })

    fprint(f"    Gates: Perm={'Y' if g1 else 'N'}(p={perm_p:.4f}) "
           f"Regime={'Y' if g2 else 'N'}({rg:.2f}) "
           f"Sub={'Y' if g3 else 'N'}({h1_sh:.2f}/{h2_sh:.2f}) "
           f"Outlier={'Y' if g4 else 'N'} "
           f"Yearly={'Y' if g5 else 'N'}({pct_profitable_years*100:.0f}%) "
           f"=> {gates}/5")
    return r


# ==================== 8. RANDOM BASELINE ====================

def random_baseline(ranks, sc, sh, sl, spy, vix, atr_d, n_trials=100):
    """Compare against random entry/exit timing."""
    fprint("[6/7] Random baseline comparison...")
    random_sharpes = []
    dates = sorted(ranks.keys())

    for trial in range(n_trials):
        eq = CAP
        curve = [CAP]
        monthly_pnl = {}

        for dt in dates:
            if dt not in spy.index or dt not in vix.index:
                continue
            # Random: trade with 50% probability
            if np.random.random() < 0.5:
                curve.append(eq)
                continue

            cv = float(vix.loc[dt])
            scores = ranks[dt]
            if not scores:
                curve.append(eq)
                continue

            # Random sector pick
            available = [tk for tk in scores.keys() if tk in sc.columns and tk in atr_d]
            if len(available) < 1:
                curve.append(eq)
                continue
            picks = list(np.random.choice(available, min(3, len(available)), replace=False))
            mx = min(MAX_POS, eq / 3)
            if mx < 30:
                curve.append(eq)
                continue

            for tk in picks[:MAX_CONC]:
                S = float(sc[tk].loc[dt])
                di = sc.index.get_loc(dt)
                dte = HOLD_DAYS
                av = float(atr_d[tk].loc[dt]) if dt in atr_d[tk].index and not pd.isna(atr_d[tk].loc[dt]) else S * 0.015
                ei = min(di + dte, len(sc) - 1)
                Se = float(sc[tk].iloc[ei])
                debit, max_profit, max_loss, K1, K2 = price_spread(S, SPREAD_WIDTH_PCT, dte, av, cv)
                cost = debit * 100 + SPREAD_COMM
                if cost <= 0 or cost > mx or cost > eq * 0.40:
                    continue
                exit_val = (max(0, Se - K1) - max(0, Se - K2)) * 100
                if exit_val > 0:
                    exit_val *= (1 - HAIRCUT)
                pnl = exit_val - cost
                eq += pnl
                m_key = dt.strftime('%Y-%m')
                monthly_pnl[m_key] = monthly_pnl.get(m_key, 0) + pnl

            curve.append(eq)

        if monthly_pnl:
            mr = np.array(list(monthly_pnl.values())) / CAP
            if len(mr) > 3 and np.std(mr) > 0:
                sh = (np.mean(mr) * 12) / (np.std(mr) * np.sqrt(12))
                random_sharpes.append(sh)

    if random_sharpes:
        fprint(f"  Random baseline: mean Sharpe={np.mean(random_sharpes):.2f}, "
               f"std={np.std(random_sharpes):.2f}, "
               f"p95={np.percentile(random_sharpes, 95):.2f}")
    return random_sharpes


# ==================== MAIN ====================

def main():
    t0 = time.time()
    fprint(f"\n{'='*80}")
    fprint(f"Regime Integration v1 — GRU Score vs VIX>20 Binary Filter")
    fprint(f"Started: {datetime.now():%Y-%m-%d %H:%M:%S}")
    fprint(f"Cap: ${CAP:.0f} | Haircut: {HAIRCUT:.0%} entry+exit | Comm: ${SPREAD_COMM}/spread | Hold: {HOLD_DAYS}d")
    fprint(f"{'='*80}\n")

    # 1. Load regime predictions
    regime_scores = load_regime_predictions()

    # 2. Download market data
    sc, sh, sl, spy, vix = download_data()

    # 3. LGBM walk-forward ranking
    rdates = pd.DatetimeIndex(sc.index.to_series().resample('2W-FRI').last().dropna().values)
    ranks = lgbm_wf(sc, rdates)
    if not ranks:
        fprint("FATAL: No rankings generated")
        return

    # Compute ATR for each sector
    atr_d = {tk: compute_atr(sh[tk], sl[tk], sc[tk]) for tk in sc.columns if tk in sh.columns}

    # 4. Run all 6 variants
    fprint(f"\n[4/7] Simulating 6 variants...")
    variants = [
        'A_BASELINE',       # VIX > 20 (current)
        'B_REGIME_GT_0.5',  # Regime score > 0.5
        'C_REGIME_GT_0.4',  # Regime score > 0.4
        'D_REGIME_GT_0.6',  # Regime score > 0.6
        'E_ADAPTIVE_SIZE',  # Scale size by regime
        'F_COMBINED',       # VIX > 18 AND regime > 0.4
    ]

    results = []
    all_trades = {}
    for v in variants:
        tr, eq, cu, whip = simulate(v, ranks, sc, sh, sl, spy, vix, regime_scores, atr_d)
        r = compute_metrics(tr, eq, cu, v, whip)
        if r:
            results.append(r)
            all_trades[v] = tr

    # 5. Adversarial validation
    fprint(f"\n[5/7] Adversarial validation...")
    for i, r in enumerate(results):
        results[i] = adversarial_validation(r)

    # 6. Random baseline
    rand_sharpes = random_baseline(ranks, sc, sh, sl, spy, vix, atr_d)

    # ==================== SUMMARY TABLE ====================
    fprint(f"\n{'='*120}")
    fprint(f"{'Variant':<22} {'#Tr':>5} {'WR':>6} {'Sharpe':>7} {'Sort':>7} {'CAGR':>7} {'MDD':>7} {'PF':>5} "
           f"{'$Final':>8} {'MCL':>4} {'Whip':>5} {'Gates':>6} {'Perm_p':>7}")
    fprint(f"{'-'*120}")
    for r in sorted(results, key=lambda x: x['sharpe'], reverse=True):
        fprint(f"{r['name']:<22} {r['n_trades']:>5} {r['win_rate']:>5.1f}% {r['sharpe']:>7.2f} {r['sortino']:>7.2f} "
               f"{r['cagr_pct']:>6.1f}% {r['maxdd_pct']:>6.1f}% {r['pf']:>5.2f} "
               f"${r['final_equity']:>7.0f} {r['max_consec_loss']:>3} {r['whipsaw_count']:>5} "
               f"{r.get('gates',0):>4}/5 {r.get('perm_p',1.0):>7.4f}")

    # ==================== KEY COMPARISONS ====================
    bl = next((r for r in results if r['name'] == 'A_BASELINE'), None)
    if bl:
        fprint(f"\n{'='*80}")
        fprint("KEY COMPARISONS vs BASELINE (VIX>20):")
        fprint(f"{'='*80}")
        for r in results:
            if r['name'] == 'A_BASELINE':
                continue
            dsh = r['sharpe'] - bl['sharpe']
            dwr = r['win_rate'] - bl['win_rate']
            dmdd = r['maxdd_pct'] - bl['maxdd_pct']  # less negative = better
            dwhip = r['whipsaw_count'] - bl['whipsaw_count']
            fprint(f"  {r['name']:22s}: Sharpe {dsh:+.2f} | WR {dwr:+.1f}% | MDD {dmdd:+.1f}% | "
                   f"Whipsaw {dwhip:+d} | Trades {r['n_trades']-bl['n_trades']:+d}")

    # Best variant
    best = max(results, key=lambda x: x['sharpe'])
    valid = [r for r in results if r.get('gates', 0) >= 3]
    best_valid = max(valid, key=lambda x: x['sharpe']) if valid else None

    fprint(f"\nBEST OVERALL: {best['name']} (Sharpe={best['sharpe']}, Gates={best.get('gates',0)}/5)")
    if best_valid:
        fprint(f"BEST VALIDATED (3+): {best_valid['name']} (Sharpe={best_valid['sharpe']}, Gates={best_valid.get('gates',0)}/5)")

    # Random baseline comparison
    if rand_sharpes:
        fprint(f"\nRANDOM BASELINE: mean Sharpe={np.mean(rand_sharpes):.2f}, "
               f"p95={np.percentile(rand_sharpes, 95):.2f}")
        for r in results:
            pct_beat = (np.array(rand_sharpes) < r['sharpe']).mean() * 100
            fprint(f"  {r['name']:22s} beats {pct_beat:.1f}% of random")

    # Whipsaw analysis
    fprint(f"\nWHIPSAW ANALYSIS (filter on/off transitions):")
    for r in sorted(results, key=lambda x: x['whipsaw_count']):
        fprint(f"  {r['name']:22s}: {r['whipsaw_count']:4d} transitions | "
               f"Efficiency: ${r['total_pnl']:.0f} / {r['whipsaw_count']+1} = "
               f"${r['total_pnl']/(r['whipsaw_count']+1):.1f}/transition")

    # ==================== ANSWER THE KEY QUESTION ====================
    fprint(f"\n{'='*80}")
    fprint("VERDICT: Does GRU continuous regime score beat crude VIX>20 binary?")
    fprint(f"{'='*80}")
    if bl:
        improvements = [r for r in results if r['name'] != 'A_BASELINE' and r['sharpe'] > bl['sharpe']]
        if improvements:
            best_imp = max(improvements, key=lambda x: x['sharpe'])
            fprint(f"  YES — {best_imp['name']} improves Sharpe by {best_imp['sharpe']-bl['sharpe']:+.2f} "
                   f"({bl['sharpe']:.2f} -> {best_imp['sharpe']:.2f})")
            if best_imp['whipsaw_count'] < bl['whipsaw_count']:
                fprint(f"  + Reduces whipsaw by {bl['whipsaw_count']-best_imp['whipsaw_count']} transitions")
            fprint(f"  + Win rate: {bl['win_rate']:.1f}% -> {best_imp['win_rate']:.1f}%")
            fprint(f"  + Max drawdown: {bl['maxdd_pct']:.1f}% -> {best_imp['maxdd_pct']:.1f}%")
            fprint(f"  + Validation: {best_imp.get('gates',0)}/5 gates passed")
        else:
            fprint(f"  NO — VIX>20 binary (Sharpe={bl['sharpe']:.2f}) still best. "
                   f"GRU regime score does NOT improve the strategy.")
    else:
        fprint("  Cannot determine — baseline had no trades")

    # ==================== 7. MLFLOW LOGGING ====================
    fprint(f"\n[7/7] Logging to MLflow...")
    if MLFLOW_OK:
        try:
            en = 'regime_integration_v1'
            try:
                if not mlflow.get_experiment_by_name(en):
                    mlflow.create_experiment(en)
            except:
                pass
            mlflow.set_experiment(en)
            with mlflow.start_run(run_name=f"regime_integ_{datetime.now():%Y%m%d_%H%M}"):
                mlflow.log_params({
                    'capital': CAP, 'haircut': HAIRCUT, 'hold_days': HOLD_DAYS,
                    'spread_width_pct': SPREAD_WIDTH_PCT, 'n_variants': len(variants),
                    'regime_score_source': 'GRU_regime_detector_v1',
                    'regime_accuracy': '83.9%',
                    'best_variant': best['name'],
                    'best_valid_variant': best_valid['name'] if best_valid else 'NONE',
                })
                for r in results:
                    for k in ['sharpe','sortino','cagr_pct','maxdd_pct','win_rate','pf',
                              'n_trades','whipsaw_count','avg_regime_score']:
                        try:
                            mlflow.log_metric(f"{r['name']}_{k}", r.get(k, 0))
                        except:
                            pass
                    for k in ['gates','perm_p','r1_gap']:
                        try:
                            mlflow.log_metric(f"{r['name']}_{k}", r.get(k, 0))
                        except:
                            pass
                if rand_sharpes:
                    mlflow.log_metric('random_baseline_mean_sharpe', np.mean(rand_sharpes))
                    mlflow.log_metric('random_baseline_p95_sharpe', np.percentile(rand_sharpes, 95))
            fprint("  MLflow: logged successfully")
        except Exception as e:
            fprint(f"  MLflow: {e}")
    else:
        fprint("  MLflow: skipped (unavailable)")

    # Save results
    op = OUT_DIR / 'regime_integration_v1_results.json'
    save = {
        'strategy': 'Regime Integration v1 — GRU Score vs VIX>20',
        'run_date': datetime.now().isoformat(),
        'capital': CAP, 'haircut': HAIRCUT, 'hold_days': HOLD_DAYS,
        'regime_source': 'GRU_regime_detector_v1 (83.9% accuracy)',
        'variants': [{k: v for k, v in r.items() if k != 'monthly_returns'} for r in results],
        'best': best['name'],
        'best_valid': best_valid['name'] if best_valid else 'NONE',
        'random_baseline': {
            'mean_sharpe': round(np.mean(rand_sharpes), 3) if rand_sharpes else None,
            'p95_sharpe': round(np.percentile(rand_sharpes, 95), 3) if rand_sharpes else None,
        },
        'verdict': 'GRU regime score improves strategy' if (bl and any(r['sharpe'] > bl['sharpe'] for r in results if r['name'] != 'A_BASELINE')) else 'VIX>20 binary still better',
    }
    with open(op, 'w') as f:
        json.dump(save, f, indent=2, default=lambda o: float(o) if hasattr(o, '__float__') else str(o))
    fprint(f"\nResults saved to {op}")

    elapsed = time.time() - t0
    fprint(f"\nDone — {elapsed:.0f}s ({elapsed/60:.1f}min)")


if __name__ == '__main__':
    main()
