#!/usr/bin/env python3
"""Low-Vol Overlay Research v1 — Non-options equity overlays for VIX<20 idle capital.

Problem: Our sector bull call spread strategy sits in cash ~64% of the time (VIX<20).
We proved selling premium in low VIX is catastrophic. This tests simple equity
overlays that deploy idle capital WITHOUT options risk.

6 overlays tested during VIX<20 periods:
  A. CASH        — Do nothing (baseline)
  B. T_BILLS     — Park in SHY (short-term treasuries)
  C. SPY_MOMENTUM — Buy SPY when above 50d SMA
  D. SECTOR_MOMENTUM — Buy top LGBM-ranked sector ETF (equity, not options)
  E. BOND_ROTATION — Best 21d momentum of TLT/GLD/SHY
  F. RISK_PARITY — Equal split SPY+TLT+GLD

VIX>=20 periods: all strategies run the same sector bull call spread engine.
Combined on a single $645 equity curve.

MLflow: low_vol_overlay_v1 at http://jupiter:5000
"""
import json, warnings, time, sys
from pathlib import Path
from datetime import datetime
import numpy as np, pandas as pd
from scipy import stats
import lightgbm as lgb
warnings.filterwarnings('ignore')

def fprint(*a, **kw): print(*a, **kw, flush=True)
np.random.seed(42)

BASE = Path(__file__).resolve().parents[2]
OUT_DIR = BASE / 'research' / 'findings'; OUT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_PATH = OUT_DIR / 'low_vol_overlay_v1_results.json'

SECTORS = ['XLK','XLF','XLE','XLV','XLY','XLP','XLI','XLB','XLU','XLRE','XLC']
CAP = 645.0
SPREAD_COMM = 2.60
HAIRCUT = 0.15
MAX_POS, MAX_CONC = 200, 3
VIX_THRESHOLD = 20.0

LGBM_FEAT = ['ret_5d','ret_10d','ret_21d','ret_63d','ret_126d','ret_252d',
             'vol_21d','vol_63d','sharpe_63d','maxdd_63d','pct_52w_high','mom_accel']

MLFLOW_OK = False
try:
    import urllib.request; urllib.request.urlopen('http://jupiter:5000/', timeout=2)
    import mlflow; mlflow.set_tracking_uri('http://jupiter:5000'); MLFLOW_OK = True
    fprint("MLflow connected")
except Exception:
    fprint("MLflow unavailable — will log locally only")


# ============================================================
# 1. DATA DOWNLOAD
# ============================================================
def download_data():
    import yfinance as yf
    fprint("[1/7] Downloading data (2009-2026)...")
    tickers = SECTORS + ['SPY','QQQ','IWM','TLT','GLD','SHY','BIL','^VIX']
    raw = yf.download(tickers, start='2009-01-01', end='2026-07-26', progress=False)
    mi = isinstance(raw.columns, pd.MultiIndex)
    close = raw['Close'] if mi else raw
    high = raw['High'] if mi else raw
    low = raw['Low'] if mi else raw

    vc = '^VIX' if '^VIX' in close.columns else 'VIX'
    vix = close[vc].dropna()
    spy = close['SPY'].dropna()

    sector_cols = [c for c in SECTORS if c in close.columns]
    sc = close[sector_cols].dropna(how='all')
    sh = high[sector_cols].dropna(how='all')
    sl = low[sector_cols].dropna(how='all')

    overlay_tickers = ['SPY','QQQ','IWM','TLT','GLD','SHY','BIL']
    overlay = {}
    for tk in overlay_tickers:
        if tk in close.columns:
            overlay[tk] = close[tk].dropna()

    # Common index
    ix = sc.index
    for s in [vix, spy] + list(overlay.values()):
        ix = ix.intersection(s.index)

    fprint(f"  {len(ix)} trading days, {len(sector_cols)} sectors, {len(overlay)} overlay assets")
    return (sc.loc[ix], sh.loc[ix], sl.loc[ix], spy.loc[ix], vix.loc[ix],
            {k: v.loc[ix] for k, v in overlay.items()})


# ============================================================
# 2. LGBM SECTOR RANKING (walk-forward)
# ============================================================
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


def lgbm_wf_rankings(px, dates, tp=12):
    fprint("[2/7] LightGBM walk-forward sector ranking...")
    records = []
    for dt in dates:
        idx = px.index.get_indexer([dt], method='ffill')[0]
        if idx < 260:
            continue
        for tk in px.columns:
            f = sector_features(px, idx, tk)
            if f is None:
                continue
            fi = min(idx+21, len(px)-1)
            f.update({'date': dt, 'ticker': tk, 'fwd_ret': float(px[tk].iloc[fi]/px[tk].iloc[idx]-1)})
            records.append(f)
    df = pd.DataFrame(records)
    if len(df) < 100:
        fprint("  WARNING: too few records for LGBM")
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
            m = lgb.LGBMRegressor(n_estimators=100, max_depth=4, learning_rate=0.05,
                                  subsample=0.8, colsample_bytree=0.8, min_child_samples=5,
                                  verbose=-1)
            m.fit(Xt, tr['rank_label'].values)
            te['score'] = m.predict(Xe)
            ranks[tdate] = dict(zip(te['ticker'], te['score']))
        except Exception:
            continue
    fprint(f"  Rankings: {len(ranks)} dates")
    return ranks


# ============================================================
# 3. ATR OPTIONS PRICING (for VIX>=20 bull call spread leg)
# ============================================================
def compute_atr(h, l, c, period=14):
    tr = pd.DataFrame({'hl': h-l, 'hc': abs(h-c.shift(1)), 'lc': abs(l-c.shift(1))}).max(axis=1)
    return tr.rolling(period).mean()


def atr_premium(S, K, dte, atr, vx, opt='call'):
    T = dte / 252.0
    if T <= 0:
        return max(0, S-K) if opt == 'call' else max(0, K-S)
    intrinsic = max(0, S-K) if opt == 'call' else max(0, K-S)
    vol_factor = max(0.3, vx / 20.0)
    return intrinsic + atr * np.sqrt(T) * vol_factor * np.exp(-3.0 * abs(S-K) / S)


def price_spread(S, spct, dte, atr, vx):
    K1 = round(S)
    K2 = round(S * (1 + spct / 100))
    lp = atr_premium(S, K1, dte, atr, vx, 'call') * (1 + HAIRCUT)
    sp = atr_premium(S, K2, dte, atr, vx, 'call') * (1 - HAIRCUT)
    deb = lp - sp
    w = K2 - K1
    return deb, (w - deb) * 100 - SPREAD_COMM, deb * 100 + SPREAD_COMM, K1, K2


# ============================================================
# 4. VIX>=20 BULL CALL SPREAD ENGINE (shared across all overlays)
# ============================================================
def run_options_period(dt, eq, ranks, sc, sh, sl, vix, atr_d):
    """Run standard bull call spread for a VIX>=20 date. Returns (pnl, trades)."""
    if dt not in ranks:
        return 0.0, []
    scores = ranks[dt]
    if not scores:
        return 0.0, []

    cv = float(vix.loc[dt])
    di = sc.index.get_loc(dt)
    sma200 = sc.rolling(200).mean()

    picks = [t for t, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)[:3]]
    mx = min(MAX_POS, eq / 3)
    if mx < 30:
        return 0.0, []

    total_pnl = 0.0
    trades = []
    ne = 0

    for tk in picks:
        if tk not in sc.columns or tk not in atr_d or ne >= MAX_CONC:
            continue
        S = float(sc[tk].loc[dt])
        dte = 14
        av = float(atr_d[tk].loc[dt]) if dt in atr_d[tk].index and not pd.isna(atr_d[tk].loc[dt]) else S * 0.015
        ei = min(di + dte, len(sc) - 1)
        Se = float(sc[tk].iloc[ei])

        val, mxp, mxl, K1, K2 = price_spread(S, 3.0, dte, av, cv)
        cost = val * 100 + SPREAD_COMM
        if cost <= 0 or cost > mx or cost > eq * 0.40:
            continue

        # Early exit scan
        pnl = None
        aei = ei
        for ci in range(di + 3, ei + 1):
            Sc = float(sc[tk].iloc[ci])
            rd = max(0, dte - (ci - di))
            ac = float(atr_d[tk].iloc[ci]) if ci < len(atr_d[tk]) else av
            tm = np.sqrt(rd / max(dte, 1))
            si = (max(0, Sc - K1) - max(0, Sc - K2)) * 100 + ac * tm * 0.3 * 100
            if si - cost >= mxp * 0.5 or rd < 7:
                pnl = si - cost
                aei = ci
                break
        if pnl is None:
            pnl = (max(0, Se - K1) - max(0, Se - K2)) * 100 - cost

        total_pnl += pnl
        ne += 1
        trades.append({
            'entry': str(dt.date()), 'exit': str(sc.index[aei].date()),
            'ticker': tk, 'pnl': round(pnl, 2), 'win': pnl > 0,
            'type': 'bull_call_spread', 'vix': round(cv, 1)
        })

    return total_pnl, trades


# ============================================================
# 5. EQUITY OVERLAY ENGINES (VIX<20 only)
# ============================================================
def overlay_cash(dt, eq, **kwargs):
    """A. CASH — do nothing."""
    return 0.0, []


def overlay_tbills(dt, eq, overlay, sc, **kwargs):
    """B. T-BILLS — park in SHY."""
    if 'SHY' not in overlay:
        return 0.0, []
    shy = overlay['SHY']
    di = sc.index.get_loc(dt)
    # Hold for ~14 days (same cadence as options)
    ei = min(di + 14, len(sc) - 1)
    if ei <= di:
        return 0.0, []
    ret = float(shy.iloc[ei] / shy.iloc[di] - 1)
    pnl = eq * ret  # deploy full idle capital
    return pnl, [{'entry': str(dt.date()), 'exit': str(sc.index[ei].date()),
                  'ticker': 'SHY', 'pnl': round(pnl, 2), 'win': pnl > 0,
                  'type': 'tbill_overlay', 'vix': round(float(kwargs.get('vix_val', 0)), 1)}]


def overlay_spy_momentum(dt, eq, overlay, sc, **kwargs):
    """C. SPY_MOMENTUM — buy SPY if above 50d SMA."""
    spy = overlay.get('SPY')
    if spy is None:
        return 0.0, []
    di = sc.index.get_loc(dt)
    if di < 50:
        return 0.0, []
    sma50 = float(spy.iloc[di-49:di+1].mean())
    price = float(spy.iloc[di])
    if price <= sma50:
        return 0.0, []  # Below SMA, stay cash
    ei = min(di + 14, len(sc) - 1)
    if ei <= di:
        return 0.0, []
    ret = float(spy.iloc[ei] / spy.iloc[di] - 1)
    pnl = eq * ret
    return pnl, [{'entry': str(dt.date()), 'exit': str(sc.index[ei].date()),
                  'ticker': 'SPY', 'pnl': round(pnl, 2), 'win': pnl > 0,
                  'type': 'spy_momentum_overlay', 'vix': round(float(kwargs.get('vix_val', 0)), 1)}]


def overlay_sector_momentum(dt, eq, overlay, sc, ranks, **kwargs):
    """D. SECTOR_MOMENTUM — buy the top LGBM-ranked sector ETF (equity, not options)."""
    if dt not in ranks or not ranks[dt]:
        return 0.0, []
    scores = ranks[dt]
    top_sector = max(scores, key=scores.get)
    if top_sector not in sc.columns:
        return 0.0, []
    di = sc.index.get_loc(dt)
    ei = min(di + 14, len(sc) - 1)
    if ei <= di:
        return 0.0, []
    ret = float(sc[top_sector].iloc[ei] / sc[top_sector].iloc[di] - 1)
    pnl = eq * ret
    return pnl, [{'entry': str(dt.date()), 'exit': str(sc.index[ei].date()),
                  'ticker': top_sector, 'pnl': round(pnl, 2), 'win': pnl > 0,
                  'type': 'sector_equity_overlay', 'vix': round(float(kwargs.get('vix_val', 0)), 1)}]


def overlay_bond_rotation(dt, eq, overlay, sc, **kwargs):
    """E. BOND_ROTATION — buy whichever of TLT/GLD/SHY has best 21d momentum."""
    candidates = {}
    di = sc.index.get_loc(dt)
    if di < 21:
        return 0.0, []
    for tk in ['TLT', 'GLD', 'SHY']:
        if tk in overlay:
            mom = float(overlay[tk].iloc[di] / overlay[tk].iloc[di - 21] - 1)
            candidates[tk] = mom
    if not candidates:
        return 0.0, []
    best = max(candidates, key=candidates.get)
    ei = min(di + 14, len(sc) - 1)
    if ei <= di:
        return 0.0, []
    ret = float(overlay[best].iloc[ei] / overlay[best].iloc[di] - 1)
    pnl = eq * ret
    return pnl, [{'entry': str(dt.date()), 'exit': str(sc.index[ei].date()),
                  'ticker': best, 'pnl': round(pnl, 2), 'win': pnl > 0,
                  'type': 'bond_rotation_overlay', 'vix': round(float(kwargs.get('vix_val', 0)), 1)}]


def overlay_risk_parity(dt, eq, overlay, sc, **kwargs):
    """F. RISK_PARITY — equal split SPY+TLT+GLD."""
    assets = ['SPY', 'TLT', 'GLD']
    available = [tk for tk in assets if tk in overlay]
    if not available:
        return 0.0, []
    di = sc.index.get_loc(dt)
    ei = min(di + 14, len(sc) - 1)
    if ei <= di:
        return 0.0, []
    alloc = eq / len(available)
    total_pnl = 0.0
    trades = []
    for tk in available:
        ret = float(overlay[tk].iloc[ei] / overlay[tk].iloc[di] - 1)
        pnl = alloc * ret
        total_pnl += pnl
        trades.append({'entry': str(dt.date()), 'exit': str(sc.index[ei].date()),
                       'ticker': tk, 'pnl': round(pnl, 2), 'win': pnl > 0,
                       'type': 'risk_parity_overlay', 'vix': round(float(kwargs.get('vix_val', 0)), 1)})
    return total_pnl, trades


OVERLAYS = {
    'A_CASH': overlay_cash,
    'B_T_BILLS': overlay_tbills,
    'C_SPY_MOMENTUM': overlay_spy_momentum,
    'D_SECTOR_MOMENTUM': overlay_sector_momentum,
    'E_BOND_ROTATION': overlay_bond_rotation,
    'F_RISK_PARITY': overlay_risk_parity,
}


# ============================================================
# 6. COMBINED SIMULATION
# ============================================================
def simulate_combined(overlay_name, overlay_fn, ranks, sc, sh, sl, spy, vix, overlay_data, atr_d):
    """Run combined strategy: VIX>=20 = bull call spreads, VIX<20 = overlay."""
    eq = CAP
    all_trades = []
    curve = [CAP]
    dates_used = []

    # Biweekly rebalance dates (every 10 trading days)
    all_dates = sorted(ranks.keys())
    if not all_dates:
        return [], eq, [CAP], {'vix_high_days': 0, 'vix_low_days': 0}

    rebal_dates = all_dates[::10]  # Every ~2 weeks
    vix_high_days = 0
    vix_low_days = 0
    options_trades = 0
    overlay_trades = 0

    for dt in rebal_dates:
        if dt not in spy.index or dt not in vix.index:
            continue
        cv = float(vix.loc[dt])

        if cv >= VIX_THRESHOLD:
            # VIX >= 20: run bull call spreads
            vix_high_days += 1
            pnl, trades = run_options_period(dt, eq, ranks, sc, sh, sl, vix, atr_d)
            eq += pnl
            all_trades.extend(trades)
            options_trades += len(trades)
        else:
            # VIX < 20: run the overlay
            vix_low_days += 1
            pnl, trades = overlay_fn(dt, eq, overlay=overlay_data, sc=sc,
                                     ranks=ranks, vix_val=cv)
            eq += pnl
            all_trades.extend(trades)
            overlay_trades += len(trades)

        curve.append(eq)

    regime_stats = {
        'vix_high_periods': vix_high_days,
        'vix_low_periods': vix_low_days,
        'pct_low_vol': round(vix_low_days / max(vix_high_days + vix_low_days, 1) * 100, 1),
        'options_trades': options_trades,
        'overlay_trades': overlay_trades,
    }

    return all_trades, eq, curve, regime_stats


# ============================================================
# 7. METRICS & VALIDATION
# ============================================================
def compute_metrics(trades, final_eq, curve, name, regime_stats=None):
    if not trades:
        fprint(f"  {name}: No trades")
        return None

    n = len(trades)
    wins = sum(1 for t in trades if t['win'])
    wr = wins / n * 100
    pnls = [t['pnl'] for t in trades]

    # Monthly returns for honest Sharpe
    tdf = pd.DataFrame(trades)
    tdf['month'] = pd.to_datetime(tdf['entry']).dt.to_period('M')
    monthly_ret = tdf.groupby('month')['pnl'].sum() / CAP
    n_years = max(len(monthly_ret) / 12, 0.5)

    sharpe = (monthly_ret.mean() * 12) / (monthly_ret.std() * np.sqrt(12) + 1e-10) if len(monthly_ret) > 3 else 0
    dn = monthly_ret[monthly_ret < 0]
    sortino = (monthly_ret.mean() * 12) / (dn.std() * np.sqrt(12) + 1e-10) if len(dn) > 1 else 0
    cagr = (final_eq / CAP) ** (1 / n_years) - 1

    eq = np.array(curve)
    pk = np.maximum.accumulate(eq)
    maxdd = float(((eq - pk) / (pk + 1e-10)).min())

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

    # Separate options vs overlay performance
    opt_trades = [t for t in trades if t.get('type') == 'bull_call_spread']
    ovl_trades = [t for t in trades if t.get('type') != 'bull_call_spread']
    opt_pnl = sum(t['pnl'] for t in opt_trades)
    ovl_pnl = sum(t['pnl'] for t in ovl_trades)

    result = {
        'name': name,
        'final_equity': round(final_eq, 2),
        'total_return_pct': round((final_eq / CAP - 1) * 100, 1),
        'cagr_pct': round(cagr * 100, 1),
        'sharpe': round(sharpe, 2),
        'sortino': round(sortino, 2),
        'maxdd_pct': round(maxdd * 100, 1),
        'profit_factor': round(pf, 2),
        'win_rate': round(wr, 1),
        'n_trades': n,
        'avg_pnl': round(np.mean(pnls), 2),
        'max_consec_loss': mcl,
        'options_pnl': round(opt_pnl, 2),
        'overlay_pnl': round(ovl_pnl, 2),
        'n_options_trades': len(opt_trades),
        'n_overlay_trades': len(ovl_trades),
    }
    if regime_stats:
        result.update(regime_stats)

    return result


def adversarial_validation(trades, n_perms=1000):
    """Permutation test — is the edge real or random?"""
    if not trades or len(trades) < 20:
        return {'perm_p_value': 1.0, 'perm_sharpe_pctile': 0}
    pnls = np.array([t['pnl'] for t in trades])
    real_sharpe = pnls.mean() / (pnls.std() + 1e-10) * np.sqrt(len(pnls) / max(len(pnls)/252, 0.5))

    rng = np.random.RandomState(42)
    better = 0
    rand_sharpes = []
    for _ in range(n_perms):
        shuffled = rng.permutation(pnls)
        s = shuffled.mean() / (shuffled.std() + 1e-10) * np.sqrt(len(shuffled) / max(len(shuffled)/252, 0.5))
        rand_sharpes.append(s)
        if s >= real_sharpe:
            better += 1

    p_value = better / n_perms
    pctile = (np.array(rand_sharpes) < real_sharpe).mean() * 100
    return {
        'perm_p_value': round(p_value, 4),
        'perm_sharpe_pctile': round(pctile, 1),
        'real_sharpe_raw': round(real_sharpe, 3),
    }


def random_baseline(n_trades, n_sims=1000):
    """What Sharpe would random trading produce?"""
    rng = np.random.RandomState(99)
    sharpes = []
    for _ in range(n_sims):
        pnls = rng.normal(0, 20, n_trades)  # Random noise trades
        if pnls.std() > 0:
            sharpes.append(pnls.mean() / pnls.std() * np.sqrt(min(n_trades, 252)))
    return {
        'random_median_sharpe': round(np.median(sharpes), 3),
        'random_p95_sharpe': round(np.percentile(sharpes, 95), 3),
    }


# ============================================================
# 8. MAIN
# ============================================================
def main():
    t0 = time.time()
    fprint("=" * 70)
    fprint("LOW-VOL OVERLAY RESEARCH v1")
    fprint("Testing non-options equity overlays for VIX<20 idle capital")
    fprint("=" * 70)

    # 1. Download data
    sc, sh, sl, spy, vix, overlay_data = download_data()

    # VIX regime stats
    low_vol_pct = (vix < VIX_THRESHOLD).mean() * 100
    fprint(f"\n  VIX regime breakdown:")
    fprint(f"    VIX < {VIX_THRESHOLD}: {low_vol_pct:.1f}% of days ({(vix < VIX_THRESHOLD).sum()} days)")
    fprint(f"    VIX >= {VIX_THRESHOLD}: {100-low_vol_pct:.1f}% of days ({(vix >= VIX_THRESHOLD).sum()} days)")

    # 2. LGBM rankings
    # Biweekly rebalance dates
    rebal_dates = sc.index[::10].tolist()
    ranks = lgbm_wf_rankings(sc, rebal_dates)

    # 3. ATR for options pricing
    fprint("[3/7] Computing ATR for options pricing...")
    atr_d = {}
    for tk in sc.columns:
        if tk in sh.columns and tk in sl.columns:
            atr_d[tk] = compute_atr(sh[tk], sl[tk], sc[tk])

    # 4. Run all overlay strategies
    fprint("[4/7] Running combined simulations...")
    results = {}
    all_curves = {}
    all_trade_details = {}

    for ov_name, ov_fn in OVERLAYS.items():
        fprint(f"\n  --- {ov_name} ---")
        trades, final_eq, curve, regime_stats = simulate_combined(
            ov_name, ov_fn, ranks, sc, sh, sl, spy, vix, overlay_data, atr_d)
        metrics = compute_metrics(trades, final_eq, curve, ov_name, regime_stats)
        if metrics:
            results[ov_name] = metrics
            all_curves[ov_name] = curve
            all_trade_details[ov_name] = trades
            fprint(f"    Final equity: ${final_eq:,.2f} | Sharpe: {metrics['sharpe']:.2f} | "
                   f"Sortino: {metrics['sortino']:.2f} | MaxDD: {metrics['maxdd_pct']:.1f}% | "
                   f"WR: {metrics['win_rate']:.1f}% | Options P&L: ${metrics['options_pnl']:,.2f} | "
                   f"Overlay P&L: ${metrics['overlay_pnl']:,.2f}")

    # 5. Adversarial validation
    fprint("\n[5/7] Adversarial validation...")
    for ov_name in results:
        adv = adversarial_validation(all_trade_details.get(ov_name, []))
        results[ov_name].update(adv)
        fprint(f"  {ov_name}: p-value={adv['perm_p_value']:.4f}, pctile={adv['perm_sharpe_pctile']:.1f}%")

    # Random baseline
    if results:
        baseline_n = results.get('A_CASH', {}).get('n_trades', 100)
        rb = random_baseline(baseline_n)
        fprint(f"  Random baseline (n={baseline_n}): median Sharpe={rb['random_median_sharpe']:.3f}, "
               f"p95={rb['random_p95_sharpe']:.3f}")

    # 6. Comparison table
    fprint("\n[6/7] RESULTS COMPARISON")
    fprint("=" * 110)
    fprint(f"{'Strategy':<22} {'Final$':>10} {'CAGR%':>7} {'Sharpe':>7} {'Sortino':>8} {'MaxDD%':>7} "
           f"{'WR%':>6} {'PF':>6} {'OptP&L':>10} {'OvlP&L':>10} {'%LowVol':>8}")
    fprint("-" * 110)

    baseline_eq = results.get('A_CASH', {}).get('final_equity', CAP)
    baseline_dd = results.get('A_CASH', {}).get('maxdd_pct', 0)

    for name in ['A_CASH', 'B_T_BILLS', 'C_SPY_MOMENTUM', 'D_SECTOR_MOMENTUM', 'E_BOND_ROTATION', 'F_RISK_PARITY']:
        r = results.get(name)
        if not r:
            continue
        marker = ''
        if r['final_equity'] > baseline_eq and r['maxdd_pct'] >= baseline_dd:
            marker = ' **'  # Better returns without worse drawdown
        elif r['final_equity'] > baseline_eq:
            marker = ' *'  # Better returns but worse drawdown

        fprint(f"{name:<22} {r['final_equity']:>10,.2f} {r['cagr_pct']:>6.1f}% {r['sharpe']:>7.2f} "
               f"{r['sortino']:>8.2f} {r['maxdd_pct']:>6.1f}% {r['win_rate']:>5.1f}% {r['profit_factor']:>5.2f} "
               f"{r['options_pnl']:>10,.2f} {r['overlay_pnl']:>10,.2f} {r.get('pct_low_vol', 0):>7.1f}%{marker}")

    fprint("-" * 110)
    fprint("** = better returns AND same/better drawdown vs CASH")
    fprint("*  = better returns but worse drawdown vs CASH")

    # Key finding
    fprint("\n  KEY QUESTION: Does the overlay improve returns WITHOUT hurting drawdown?")
    for name in ['B_T_BILLS', 'C_SPY_MOMENTUM', 'D_SECTOR_MOMENTUM', 'E_BOND_ROTATION', 'F_RISK_PARITY']:
        r = results.get(name)
        if not r:
            continue
        ret_delta = r['final_equity'] - baseline_eq
        dd_delta = r['maxdd_pct'] - baseline_dd
        verdict = "YES" if ret_delta > 0 and dd_delta >= -1.0 else ("MIXED" if ret_delta > 0 else "NO")
        fprint(f"    {name}: return delta=${ret_delta:+,.2f}, drawdown delta={dd_delta:+.1f}% -> {verdict}")

    # 7. Save and log
    fprint("\n[7/7] Saving results...")
    output = {
        'timestamp': datetime.now().isoformat(),
        'vix_threshold': VIX_THRESHOLD,
        'start_capital': CAP,
        'data_range': f"{sc.index[0].date()} to {sc.index[-1].date()}",
        'n_trading_days': len(sc),
        'pct_low_vol': round(low_vol_pct, 1),
        'results': results,
        'random_baseline': rb if results else {},
    }
    with open(RESULTS_PATH, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    fprint(f"  Saved to {RESULTS_PATH}")

    # MLflow logging
    if MLFLOW_OK:
        fprint("  Logging to MLflow...")
        try:
            exp_name = 'low_vol_overlay_v1'
            mlflow.set_experiment(exp_name)
            for ov_name, r in results.items():
                with mlflow.start_run(run_name=ov_name):
                    mlflow.log_params({
                        'overlay': ov_name,
                        'vix_threshold': VIX_THRESHOLD,
                        'start_capital': CAP,
                        'data_range': output['data_range'],
                    })
                    for k, v in r.items():
                        if isinstance(v, (int, float)):
                            mlflow.log_metric(k, v)
                    # Log the full results JSON as artifact
                    mlflow.log_artifact(str(RESULTS_PATH))
            fprint(f"  MLflow: {len(results)} runs logged to '{exp_name}'")
        except Exception as e:
            fprint(f"  MLflow logging failed: {e}")

    elapsed = time.time() - t0
    fprint(f"\nDone in {elapsed:.0f}s")
    return results


if __name__ == '__main__':
    main()
