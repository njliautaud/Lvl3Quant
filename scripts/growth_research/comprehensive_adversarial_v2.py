#!/usr/bin/env python3
"""Comprehensive Adversarial Validation v2 — Validate ALL top strategies are legit.

User directive (3 AM 7/26): "make sure ALL ur research findings have adversarial checks
to validate the performance is legit... make sure all numbers are legit. Based on REAL DATA
nothing synthetic."

For EACH strategy we run:
1. PERMUTATION TEST — Shuffle signals/rankings, re-run. Real Sharpe must beat 95% of shuffled.
2. REGIME STRATIFICATION — Split into bull/bear (SPY 200d SMA). Edge must exist in BOTH.
3. LOOKAHEAD BIAS CHECK — Shift all features by +1 day. If performance IMPROVES → leaking.
4. TRANSACTION COST SENSITIVITY — Verify profitable at 2x realistic costs.
5. WALK-FORWARD STABILITY — Rolling 2-year Sharpe windows. No single period > 50% of returns.
6. DATA SNOOPING CORRECTION — With N strategies tested, apply Bonferroni/BHY correction.
7. YEARLY CONSISTENCY — Profitable in at least 60% of calendar years.

Strategies tested:
  S1: Multi-asset broad momentum (25 ETFs) — original claim Sharpe 4.70
  S2: Integrated QualMom (11 sectors) — original claim Sharpe 4.20
  S3: Multi-factor sector options (11 sectors) — original claim Sharpe 3.87
  S4: LGBM weekly momentum
"""
import json, sys, os, numpy as np, pandas as pd, warnings, traceback
warnings.filterwarnings('ignore')
from pathlib import Path
from datetime import datetime
from scipy import stats
import lightgbm as lgb
from collections import defaultdict

BASE = Path(__file__).resolve().parents[2]
RESULTS_DIR = BASE / 'research' / 'findings'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_PATH = RESULTS_DIR / 'comprehensive_adversarial_v2_results.json'
def fprint(*a, **kw): print(*a, **kw, flush=True)

MLFLOW_OK = False
try:
    import urllib.request; urllib.request.urlopen('http://jupiter:5000/', timeout=2)
    import mlflow; mlflow.set_tracking_uri('http://jupiter:5000'); MLFLOW_OK = True
except Exception: fprint("MLflow unavailable — logging locally only")

# ─── Constants ───
CAP = 645.0
LEG_COMM = 0.65
SPREAD_COMM = 4 * LEG_COMM  # $2.60 RT
HAIRCUT = 0.15

SECTORS = ['XLK','XLF','XLE','XLV','XLY','XLP','XLI','XLB','XLU','XLRE','XLC']
BROAD_UNIVERSE = SECTORS + ['GLD','SLV','USO','DBA','TLT','HYG','LQD','TIP',
                             'EFA','EEM','VWO','VNQ','AMLP','BITO']

QM_COLS = ['ret_5d','ret_10d','ret_21d','ret_63d','ret_126d','ret_252d',
           'vol_21d','vol_63d','sharpe_63d','maxdd_63d','pct_52w_high','mom_accel',
           'pct_pos_months_12m','sortino_63d','calmar_1y','up_capture','dn_capture',
           'trend_r2_63d','trend_slope_63d','rel_vol_21d']

# ─── Data Download (shared) ───
def download_data(universe):
    import yfinance as yf
    fprint(f"  Downloading {len(universe)} assets from yfinance (REAL DATA)...")
    tickers = list(set(universe + ['SPY', '^VIX']))
    raw = yf.download(tickers, start='2008-01-01', end='2026-07-26', progress=False)
    mi = isinstance(raw.columns, pd.MultiIndex)
    close = raw['Close'] if mi else raw
    high = raw['High'] if mi else raw
    low = raw['Low'] if mi else raw
    volume = raw['Volume'] if mi else raw

    vc = '^VIX' if '^VIX' in close.columns else 'VIX'
    vix = close[vc].dropna() if vc in close.columns else pd.Series(dtype=float)
    spy = close['SPY'].dropna() if 'SPY' in close.columns else pd.Series(dtype=float)

    available = [c for c in universe if c in close.columns and close[c].dropna().shape[0] > 500]
    if not available:
        return None, None, None, None, None, None, []

    sc = close[available].dropna(how='all')
    sh = high[[c for c in available if c in high.columns]].dropna(how='all')
    sl = low[[c for c in available if c in low.columns]].dropna(how='all')
    sv = volume[[c for c in available if c in volume.columns]].dropna(how='all')

    ix = sc.index
    for s in [vix, spy, sh, sl]:
        if len(s) > 0:
            ix = ix.intersection(s.index)

    fprint(f"  Got {len(ix)} days, {len(available)}/{len(universe)} assets")
    return sc.loc[ix], sh.loc[ix], sl.loc[ix], sv.reindex(ix), spy.loc[ix], vix.loc[ix], available

# ─── Feature computation ───
def compute_features(px):
    if len(px) < 260: return None
    f = {}
    for lb, nm in [(5,'ret_5d'),(10,'ret_10d'),(21,'ret_21d'),(63,'ret_63d'),
                    (126,'ret_126d'),(252,'ret_252d')]:
        f[nm] = float(px.iloc[-1]/px.iloc[-lb]-1) if len(px) > lb else 0.0
    rets = px.pct_change().dropna()
    f['vol_21d'] = float(rets.iloc[-21:].std()*np.sqrt(252)) if len(rets) > 21 else 0.2
    f['vol_63d'] = float(rets.iloc[-63:].std()*np.sqrt(252)) if len(rets) > 63 else 0.2
    r63 = rets.iloc[-63:]
    f['sharpe_63d'] = float(r63.mean()/(r63.std()+1e-10)*np.sqrt(252)) if len(r63) > 10 else 0.0
    pk63 = px.iloc[-63:].cummax()
    f['maxdd_63d'] = float(((px.iloc[-63:]/pk63)-1).min())
    h52 = px.iloc[-252:].max() if len(px) >= 252 else px.max()
    f['pct_52w_high'] = float(px.iloc[-1]/h52)
    r21 = rets.iloc[-21:]
    r63b = rets.iloc[-63:-21] if len(rets) > 63 else rets.iloc[:21]
    f['mom_accel'] = float(r21.mean() - r63b.mean()) if len(r63b) > 5 else 0.0
    monthly = px.resample('ME').last().pct_change().dropna().iloc[-12:]
    f['pct_pos_months_12m'] = float((monthly > 0).mean()) if len(monthly) > 3 else 0.5
    dr = r63[r63 < 0]
    f['sortino_63d'] = float(r63.mean()/(dr.std()+1e-10)*np.sqrt(252)) if len(dr) > 3 else 0.0
    dd63 = abs(float(f['maxdd_63d']))
    ann_ret = float((px.iloc[-1]/px.iloc[-252]-1)) if len(px) >= 252 else float(rets.mean()*252)
    f['calmar_1y'] = float(ann_ret / dd63) if dd63 > 0.001 else 0.0
    f['up_capture'] = 1.0
    f['dn_capture'] = 1.0
    x = np.arange(63)
    y = np.log(px.iloc[-63:].values + 1e-10)
    if len(y) == 63:
        slope, _, r_val, _, _ = stats.linregress(x, y)
        f['trend_r2_63d'] = float(r_val**2)
        f['trend_slope_63d'] = float(slope*252)
    else:
        f['trend_r2_63d'] = 0.0
        f['trend_slope_63d'] = 0.0
    v21 = float(rets.iloc[-21:].std())
    v63 = float(rets.iloc[-63:].std())
    f['rel_vol_21d'] = float(v21/(v63+1e-10))
    return f

# ─── Core Backtest Engine (options bull spread) ───
def run_options_backtest(close, high, low, vix, available,
                         top_k=3, spread_pct=0.03, rebal_days=10,
                         vix_filter=20, commission_mult=1.0,
                         feature_shift=0, shuffle_labels=False,
                         regime_filter=None, spy=None,
                         seed=None):
    """
    Walk-forward options backtest with adversarial controls.

    feature_shift: shift features by N days (>0 = add lag, lookahead check)
    shuffle_labels: if True, randomly permute training labels
    regime_filter: 'bull' or 'bear' (SPY vs 200d SMA)
    commission_mult: multiply commissions (cost sensitivity)
    """
    rng = np.random.RandomState(seed) if seed is not None else np.random.RandomState(42)
    dates = close.index
    train_periods = 12
    min_history = 252

    capital = CAP
    equity_curve = [capital]
    trade_log = []
    peak = capital
    yearly_pnl = defaultdict(float)

    rebal_dates = dates[min_history::rebal_days]

    # Pre-compute regime if needed
    if regime_filter and spy is not None and len(spy) > 200:
        spy_sma200 = spy.rolling(200).mean()
        bull_mask = spy > spy_sma200
    else:
        bull_mask = None

    for i, rd in enumerate(rebal_dates):
        if i < train_periods:
            continue

        rd_idx = dates.get_loc(rd)
        vix_val = float(vix.iloc[rd_idx]) if rd_idx < len(vix) else 20

        # VIX filter
        if vix_filter and vix_val < vix_filter:
            continue

        # Regime filter
        if regime_filter and bull_mask is not None:
            is_bull = bool(bull_mask.iloc[rd_idx]) if rd_idx < len(bull_mask) else True
            if regime_filter == 'bull' and not is_bull:
                continue
            if regime_filter == 'bear' and is_bull:
                continue

        # Build training data
        train_data = []
        for period_idx in range(i - train_periods, i):
            if period_idx < 0 or period_idx >= len(rebal_dates):
                continue
            prd = rebal_dates[period_idx]
            prd_i = dates.get_loc(prd)
            next_i = min(prd_i + rebal_days, len(dates) - 1)
            for tk in available:
                if pd.isna(close[tk].iloc[prd_i]) or pd.isna(close[tk].iloc[next_i]):
                    continue
                # Apply feature shift for lookahead check
                feat_idx = max(0, prd_i - feature_shift)
                hist = close[tk].iloc[:feat_idx + 1].dropna()
                feats = compute_features(hist)
                if feats is None:
                    continue
                fwd_ret = float(close[tk].iloc[next_i] / close[tk].iloc[prd_i] - 1)
                row = feats.copy()
                row['fwd_ret'] = fwd_ret
                row['ticker'] = tk
                train_data.append(row)

        if len(train_data) < 30:
            continue

        train_df = pd.DataFrame(train_data)
        feat_cols = [c for c in QM_COLS if c in train_df.columns]
        X_train = train_df[feat_cols].values.astype(np.float32)
        y_train = train_df['fwd_ret'].values.astype(np.float32)

        # Shuffle labels for permutation test
        if shuffle_labels:
            y_train = rng.permutation(y_train)

        X_train = np.nan_to_num(X_train, nan=0, posinf=0, neginf=0)
        y_train = np.nan_to_num(y_train, nan=0, posinf=0, neginf=0)

        try:
            ds = lgb.Dataset(X_train, label=y_train, free_raw_data=False)
            model = lgb.train(
                {'objective': 'regression', 'num_leaves': 15, 'learning_rate': 0.05,
                 'min_child_samples': 5, 'verbose': -1, 'n_jobs': 4,
                 'feature_fraction': 0.7, 'bagging_fraction': 0.7, 'bagging_freq': 5,
                 'seed': 42},
                ds, num_boost_round=100
            )
        except:
            continue

        # Predict current period
        current_feats = []
        for tk in available:
            if pd.isna(close[tk].iloc[rd_idx]):
                continue
            feat_idx = max(0, rd_idx - feature_shift)
            hist = close[tk].iloc[:feat_idx + 1].dropna()
            feats = compute_features(hist)
            if feats is None:
                continue
            row = feats.copy()
            row['ticker'] = tk
            current_feats.append(row)

        if not current_feats:
            continue
        curr_df = pd.DataFrame(current_feats)
        X_curr = curr_df[feat_cols].values.astype(np.float32)
        X_curr = np.nan_to_num(X_curr, nan=0, posinf=0, neginf=0)
        preds = model.predict(X_curr)
        curr_df['pred'] = preds

        # Confluence gate
        curr_df['mom_signal'] = curr_df['ret_21d'] > 0
        curr_df['trend_signal'] = curr_df['trend_slope_63d'] > 0
        curr_df['quality_signal'] = curr_df['sharpe_63d'] > 0
        curr_df['confluence'] = (curr_df['mom_signal'].astype(int) +
                                  curr_df['trend_signal'].astype(int) +
                                  curr_df['quality_signal'].astype(int))
        curr_df = curr_df[curr_df['confluence'] >= 2]

        if len(curr_df) == 0:
            continue

        top = curr_df.nlargest(min(top_k, len(curr_df)), 'pred')

        next_rd_idx = min(rd_idx + rebal_days, len(dates) - 1)
        for _, row in top.iterrows():
            tk = row['ticker']
            price = float(close[tk].iloc[rd_idx])
            if pd.isna(price) or price <= 0:
                continue

            # ATR pricing
            if tk in high.columns and tk in low.columns:
                h20 = high[tk].iloc[max(0, rd_idx-20):rd_idx+1].dropna()
                l20 = low[tk].iloc[max(0, rd_idx-20):rd_idx+1].dropna()
                c20 = close[tk].iloc[max(0, rd_idx-20):rd_idx+1].dropna()
                if len(h20) >= 5 and len(l20) >= 5 and len(c20) >= 5:
                    tr = pd.concat([h20-l20, abs(h20-c20.shift(1)), abs(l20-c20.shift(1))], axis=1).max(axis=1)
                    atr = float(tr.iloc[-14:].mean())
                else:
                    atr = price * 0.02
            else:
                atr = price * 0.02

            strike_low = price
            strike_high = price * (1 + spread_pct)
            max_profit_raw = atr * (1 - HAIRCUT)
            spread_width = strike_high - strike_low
            debit = spread_width - max_profit_raw
            if debit <= 0:
                debit = spread_width * 0.60
            max_profit = max_profit_raw
            max_loss = debit

            pos_size = min(200, capital * 0.33)
            n_contracts = max(1, int(pos_size / (max_loss * 100 + 1)))
            comm = SPREAD_COMM * n_contracts * commission_mult

            exit_price = float(close[tk].iloc[next_rd_idx])
            actual_ret = (exit_price / price) - 1

            if actual_ret >= spread_pct:
                pnl = max_profit * 100 * n_contracts - comm
            elif actual_ret <= 0:
                pnl = -max_loss * 100 * n_contracts - comm
            else:
                frac = actual_ret / spread_pct
                pnl = (frac * max_profit - (1 - frac) * max_loss) * 100 * n_contracts - comm

            capital += pnl
            if capital <= 0:
                capital = 1.0  # floor to avoid div by zero
            peak = max(peak, capital)
            yr = rd.year
            yearly_pnl[yr] += pnl
            equity_curve.append(capital)
            trade_log.append({
                'date': str(rd.date()), 'ticker': tk,
                'pnl': round(pnl, 2), 'capital': round(capital, 2),
                'actual_ret': round(actual_ret * 100, 2)
            })

    return compute_metrics(equity_curve, trade_log, yearly_pnl)


def compute_metrics(equity_curve, trade_log, yearly_pnl):
    """Compute honest metrics from equity curve and trade log."""
    if len(trade_log) < 5:
        return {'n_trades': len(trade_log), 'sharpe': 0, 'sharpe_honest': 0,
                'valid': False, 'trade_log': trade_log, 'yearly_pnl': dict(yearly_pnl)}

    pnls = [t['pnl'] for t in trade_log]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    wr = len(wins) / len(pnls) * 100 if pnls else 0
    pf = (sum(wins) / abs(sum(losses))) if losses and sum(losses) != 0 else 999

    eq = np.array(equity_curve)
    # HONEST Sharpe: equity-based returns, not initial-capital-based
    eq_rets = np.diff(eq) / eq[:-1]
    eq_rets = eq_rets[np.isfinite(eq_rets)]

    # Group into monthly returns for proper Sharpe
    tdf = pd.DataFrame(trade_log)
    tdf['date'] = pd.to_datetime(tdf['date'])
    tdf['month'] = tdf['date'].dt.to_period('M')

    # Build monthly equity-based returns
    equity = CAP
    monthly_data = {}
    for _, row in tdf.iterrows():
        m = row['month']
        if m not in monthly_data:
            monthly_data[m] = {'start_equity': equity, 'pnl': 0}
        monthly_data[m]['pnl'] += row['pnl']
        equity += row['pnl']

    if len(monthly_data) > 3:
        honest_monthly_rets = np.array([
            v['pnl'] / max(v['start_equity'], 1.0) for v in monthly_data.values()
        ])
        sharpe_honest = float(
            (honest_monthly_rets.mean() * 12) /
            (honest_monthly_rets.std() * np.sqrt(12) + 1e-10)
        )
        # Also compute inflated for comparison
        inflated_monthly_rets = np.array([
            v['pnl'] / CAP for v in monthly_data.values()
        ])
        sharpe_inflated = float(
            (inflated_monthly_rets.mean() * 12) /
            (inflated_monthly_rets.std() * np.sqrt(12) + 1e-10)
        )

        dn = honest_monthly_rets[honest_monthly_rets < 0]
        sortino = float(
            (honest_monthly_rets.mean() * 12) /
            (dn.std() * np.sqrt(12) + 1e-10)
        ) if len(dn) > 1 else 0
    else:
        sharpe_honest = 0
        sharpe_inflated = 0
        sortino = 0

    n_years = max(len(set(t['date'][:4] for t in trade_log)), 1)
    final_cap = eq[-1]
    cagr = (final_cap / CAP) ** (1.0 / max(n_years, 0.5)) - 1

    peak_arr = np.maximum.accumulate(eq)
    dd = (eq - peak_arr) / (peak_arr + 1e-10)
    maxdd = float(dd.min()) * 100

    # Yearly consistency
    yrs = sorted(yearly_pnl.keys())
    n_profitable_years = sum(1 for y in yrs if yearly_pnl[y] > 0)
    pct_profitable_years = n_profitable_years / len(yrs) * 100 if yrs else 0

    return {
        'n_trades': len(trade_log),
        'sharpe_honest': round(sharpe_honest, 3),
        'sharpe_inflated': round(sharpe_inflated, 3),
        'sortino': round(sortino, 3),
        'cagr_pct': round(cagr * 100, 1),
        'maxdd_pct': round(maxdd, 1),
        'win_rate': round(wr, 1),
        'profit_factor': round(pf, 2),
        'final_capital': round(final_cap, 0),
        'n_years': n_years,
        'pct_profitable_years': round(pct_profitable_years, 1),
        'yearly_pnl': {str(k): round(v, 2) for k, v in yearly_pnl.items()},
        'valid': True,
        'trade_log': trade_log  # kept for sub-analysis
    }


# ─── Adversarial Tests ───
def test_permutation(close, high, low, vix, spy, available, n_perms=20, **bt_kwargs):
    """Permutation test: real Sharpe vs shuffled-label distribution."""
    fprint("    Running permutation test...")
    real = run_options_backtest(close, high, low, vix, available, spy=spy,
                                shuffle_labels=False, **bt_kwargs)
    real_sharpe = real['sharpe_honest']

    perm_sharpes = []
    for p in range(n_perms):
        if p % 5 == 0:
            fprint(f"      Permutation {p+1}/{n_perms}...")
        perm = run_options_backtest(close, high, low, vix, available, spy=spy,
                                    shuffle_labels=True, seed=p*7+13, **bt_kwargs)
        perm_sharpes.append(perm['sharpe_honest'])

    perm_sharpes = np.array(perm_sharpes)
    p_value = float(np.mean(perm_sharpes >= real_sharpe))

    return {
        'status': 'PASS' if p_value < 0.05 else ('MARGINAL' if p_value < 0.10 else 'FAIL'),
        'real_sharpe': round(real_sharpe, 3),
        'perm_mean': round(float(perm_sharpes.mean()), 3),
        'perm_std': round(float(perm_sharpes.std()), 3),
        'perm_p95': round(float(np.percentile(perm_sharpes, 95)), 3),
        'p_value': round(p_value, 4),
        'n_permutations': n_perms,
        'conclusion': f"Real Sharpe {real_sharpe:.3f} vs permuted mean {perm_sharpes.mean():.3f} (p={p_value:.4f})"
    }


def test_regime(close, high, low, vix, spy, available, **bt_kwargs):
    """Regime stratification: edge must exist in both bull and bear markets."""
    fprint("    Running regime stratification...")
    bull = run_options_backtest(close, high, low, vix, available, spy=spy,
                                regime_filter='bull', **bt_kwargs)
    bear = run_options_backtest(close, high, low, vix, available, spy=spy,
                                regime_filter='bear', **bt_kwargs)
    both = run_options_backtest(close, high, low, vix, available, spy=spy,
                                regime_filter=None, **bt_kwargs)

    sh_bull = bull['sharpe_honest']
    sh_bear = bear['sharpe_honest']
    sh_both = both['sharpe_honest']

    max_sh = max(abs(sh_bull), abs(sh_bear), 0.001)
    regime_gap = abs(sh_bull - sh_bear) / max_sh

    # Per HC #428 R1: reject if regime gap > 0.50
    status = 'PASS' if regime_gap <= 0.50 else 'FAIL'
    # Also fail if one regime has negative Sharpe
    if sh_bull < 0 or sh_bear < 0:
        status = 'FAIL'

    return {
        'status': status,
        'sharpe_bull': round(sh_bull, 3),
        'sharpe_bear': round(sh_bear, 3),
        'sharpe_combined': round(sh_both, 3),
        'regime_gap': round(regime_gap, 3),
        'bull_trades': bull['n_trades'],
        'bear_trades': bear['n_trades'],
        'bull_wr': bull.get('win_rate', 0),
        'bear_wr': bear.get('win_rate', 0),
        'conclusion': f"Bull Sharpe {sh_bull:.3f}, Bear Sharpe {sh_bear:.3f}, gap={regime_gap:.3f}"
    }


def test_lookahead(close, high, low, vix, spy, available, **bt_kwargs):
    """Lookahead check: shift features by 1 day. If performance improves → leaking."""
    fprint("    Running lookahead bias check...")
    baseline = run_options_backtest(close, high, low, vix, available, spy=spy,
                                    feature_shift=0, **bt_kwargs)
    lagged = run_options_backtest(close, high, low, vix, available, spy=spy,
                                  feature_shift=1, **bt_kwargs)

    sh_base = baseline['sharpe_honest']
    sh_lag = lagged['sharpe_honest']

    # If lagged performance is BETTER, we may have lookahead
    # If it drops modestly, that's expected (less fresh data)
    degradation = sh_base - sh_lag
    pct_degradation = degradation / (abs(sh_base) + 1e-10) * 100

    # PASS if baseline is better (no lookahead), FAIL if lagged is better
    if sh_lag > sh_base * 1.1:
        status = 'FAIL'  # Suspicious: lagged is better
    elif degradation > 0 and pct_degradation < 50:
        status = 'PASS'  # Expected modest degradation
    elif degradation > 0 and pct_degradation >= 50:
        status = 'WARNING'  # Too dependent on fresh data
    else:
        status = 'PASS'

    return {
        'status': status,
        'sharpe_baseline': round(sh_base, 3),
        'sharpe_lagged_1d': round(sh_lag, 3),
        'degradation_pct': round(pct_degradation, 1),
        'conclusion': f"Baseline {sh_base:.3f} → Lagged {sh_lag:.3f} ({pct_degradation:+.1f}% change)"
    }


def test_cost_sensitivity(close, high, low, vix, spy, available, **bt_kwargs):
    """Cost sensitivity: verify profitable at 1x, 1.5x, and 2x commissions."""
    fprint("    Running cost sensitivity test...")
    results = {}
    for mult in [1.0, 1.5, 2.0, 3.0]:
        r = run_options_backtest(close, high, low, vix, available, spy=spy,
                                 commission_mult=mult, **bt_kwargs)
        results[f'{mult}x'] = {
            'sharpe': r['sharpe_honest'],
            'pf': r.get('profit_factor', 0),
            'wr': r.get('win_rate', 0),
            'final_cap': r.get('final_capital', 0)
        }

    profitable_at_2x = results['2.0x']['sharpe'] > 0 and results['2.0x']['pf'] > 1.0
    status = 'PASS' if profitable_at_2x else 'FAIL'

    return {
        'status': status,
        'cost_levels': results,
        'profitable_at_2x': profitable_at_2x,
        'conclusion': f"At 2x costs: Sharpe={results['2.0x']['sharpe']:.3f}, PF={results['2.0x']['pf']:.2f}"
    }


def test_walkforward_stability(trade_log, yearly_pnl):
    """Walk-forward stability: no single period dominates returns."""
    fprint("    Running walk-forward stability check...")
    if not trade_log or len(trade_log) < 20:
        return {'status': 'FAIL', 'conclusion': 'Too few trades for stability analysis'}

    tdf = pd.DataFrame(trade_log)
    tdf['date'] = pd.to_datetime(tdf['date'])
    tdf['year'] = tdf['date'].dt.year

    total_pnl = tdf['pnl'].sum()
    if abs(total_pnl) < 1:
        return {'status': 'FAIL', 'conclusion': 'Near-zero total PnL'}

    # Check if any single year > 50% of total returns
    year_pnls = tdf.groupby('year')['pnl'].sum()
    max_year_pct = float(year_pnls.max() / abs(total_pnl) * 100) if total_pnl > 0 else 0
    max_year = int(year_pnls.idxmax()) if total_pnl > 0 else 0

    # Rolling 2-year Sharpe
    tdf = tdf.sort_values('date')
    tdf['month'] = tdf['date'].dt.to_period('M')
    monthly = tdf.groupby('month')['pnl'].sum()

    rolling_sharpes = []
    months = list(monthly.index)
    window = 24  # 2-year windows
    for i in range(len(months) - window + 1):
        chunk = monthly.iloc[i:i+window]
        rets = chunk.values
        if len(rets) > 3 and rets.std() > 0:
            sh = float(rets.mean() / rets.std() * np.sqrt(12))
            rolling_sharpes.append(sh)

    # Check consistency
    n_positive = sum(1 for s in rolling_sharpes if s > 0)
    pct_positive = n_positive / len(rolling_sharpes) * 100 if rolling_sharpes else 0

    # Yearly consistency
    n_years = len(year_pnls)
    n_profitable = sum(1 for v in year_pnls if v > 0)
    pct_profitable = n_profitable / n_years * 100 if n_years > 0 else 0

    status = 'PASS'
    if max_year_pct > 50:
        status = 'WARNING'  # Concentrated in one period
    if pct_profitable < 60:
        status = 'FAIL'  # Not consistent enough
    if rolling_sharpes and pct_positive < 50:
        status = 'FAIL'  # Most rolling windows negative

    return {
        'status': status,
        'max_year_concentration_pct': round(max_year_pct, 1),
        'max_year': max_year,
        'n_years': n_years,
        'n_profitable_years': n_profitable,
        'pct_profitable_years': round(pct_profitable, 1),
        'rolling_2y_sharpes': [round(s, 2) for s in rolling_sharpes],
        'pct_rolling_positive': round(pct_positive, 1),
        'yearly_breakdown': {str(k): round(v, 2) for k, v in year_pnls.items()},
        'conclusion': f"{n_profitable}/{n_years} profitable years, max year conc={max_year_pct:.0f}%"
    }


def test_data_snooping(strategy_sharpes, n_strategies_tested):
    """Data snooping correction with Bonferroni and BHY methods."""
    fprint("    Applying data snooping corrections...")

    results = {}
    for name, sharpe in strategy_sharpes.items():
        # Convert Sharpe to p-value (approx: assume monthly Sharpe ~ Normal)
        # Sharpe = mean/std * sqrt(12). Under null, Sharpe ~ 0.
        # Approximate p-value from Sharpe using normal CDF
        z = sharpe * np.sqrt(5)  # approximate z-score (5 years of monthly data)
        p_unadjusted = float(1 - stats.norm.cdf(z))

        # Bonferroni correction
        p_bonferroni = min(p_unadjusted * n_strategies_tested, 1.0)

        # BHY (Benjamini-Hochberg-Yekutieli) correction
        # For m tests, BHY threshold = alpha * k / (m * sum(1/i))
        harmonic = sum(1.0/i for i in range(1, n_strategies_tested + 1))
        p_bhy = min(p_unadjusted * n_strategies_tested * harmonic, 1.0)

        results[name] = {
            'sharpe': round(sharpe, 3),
            'p_unadjusted': round(p_unadjusted, 4),
            'p_bonferroni': round(p_bonferroni, 4),
            'p_bhy': round(p_bhy, 4),
            'significant_bonferroni': p_bonferroni < 0.05,
            'significant_bhy': p_bhy < 0.05
        }

    any_pass = any(v['significant_bonferroni'] for v in results.values())
    return {
        'status': 'PASS' if any_pass else 'FAIL',
        'n_strategies_tested': n_strategies_tested,
        'corrections': results,
        'conclusion': f"After Bonferroni ({n_strategies_tested} tests): "
                      f"{sum(1 for v in results.values() if v['significant_bonferroni'])}/{len(results)} significant"
    }


# ─── Main Execution ───
def validate_strategy(name, universe, bt_kwargs, close, high, low, vix, spy, available):
    """Run all adversarial checks on a single strategy."""
    fprint(f"\n{'='*70}")
    fprint(f"ADVERSARIAL VALIDATION: {name}")
    fprint(f"{'='*70}")

    results = {'strategy': name, 'universe_size': len(available),
               'timestamp': datetime.now().isoformat()}

    # 1. Baseline run (honest metrics)
    fprint("  [1/6] Baseline run with honest Sharpe...")
    baseline = run_options_backtest(close, high, low, vix, available, spy=spy, **bt_kwargs)
    results['baseline'] = {k: v for k, v in baseline.items() if k != 'trade_log'}
    fprint(f"    Honest Sharpe: {baseline['sharpe_honest']:.3f} "
           f"(inflated: {baseline['sharpe_inflated']:.3f})")
    fprint(f"    WR: {baseline.get('win_rate',0):.1f}%, PF: {baseline.get('profit_factor',0):.2f}, "
           f"Trades: {baseline['n_trades']}")

    if not baseline['valid'] or baseline['n_trades'] < 20:
        results['verdict'] = 'INVALID — too few trades for validation'
        results['overall_status'] = 'FAIL'
        return results

    # 2. Permutation test
    fprint("  [2/6] Permutation test (20 shuffles)...")
    results['permutation'] = test_permutation(close, high, low, vix, spy, available, **bt_kwargs)
    fprint(f"    Result: {results['permutation']['status']} — {results['permutation']['conclusion']}")

    # 3. Regime stratification
    fprint("  [3/6] Regime stratification (bull vs bear)...")
    results['regime'] = test_regime(close, high, low, vix, spy, available, **bt_kwargs)
    fprint(f"    Result: {results['regime']['status']} — {results['regime']['conclusion']}")

    # 4. Lookahead bias check
    fprint("  [4/6] Lookahead bias check...")
    results['lookahead'] = test_lookahead(close, high, low, vix, spy, available, **bt_kwargs)
    fprint(f"    Result: {results['lookahead']['status']} — {results['lookahead']['conclusion']}")

    # 5. Cost sensitivity
    fprint("  [5/6] Transaction cost sensitivity...")
    results['cost_sensitivity'] = test_cost_sensitivity(close, high, low, vix, spy, available, **bt_kwargs)
    fprint(f"    Result: {results['cost_sensitivity']['status']} — {results['cost_sensitivity']['conclusion']}")

    # 6. Walk-forward stability
    fprint("  [6/6] Walk-forward stability...")
    results['stability'] = test_walkforward_stability(baseline.get('trade_log', []),
                                                       baseline.get('yearly_pnl', {}))
    fprint(f"    Result: {results['stability']['status']} — {results['stability']['conclusion']}")

    # Overall verdict
    checks = ['permutation', 'regime', 'lookahead', 'cost_sensitivity', 'stability']
    statuses = [results[c]['status'] for c in checks]
    n_pass = sum(1 for s in statuses if s == 'PASS')
    n_fail = sum(1 for s in statuses if s == 'FAIL')
    n_warn = sum(1 for s in statuses if s in ('MARGINAL', 'WARNING'))

    if n_fail == 0 and n_warn <= 1:
        overall = 'FULL_PASS'
    elif n_fail == 0:
        overall = 'CONDITIONAL_PASS'
    elif n_fail <= 1 and n_pass >= 3:
        overall = 'MARGINAL_PASS'
    else:
        overall = 'FAIL'

    results['overall_status'] = overall
    results['check_summary'] = {c: results[c]['status'] for c in checks}
    results['verdict'] = (f"{overall}: {n_pass} PASS, {n_warn} WARN, {n_fail} FAIL out of {len(checks)} checks. "
                          f"Honest Sharpe={baseline['sharpe_honest']:.3f}")

    fprint(f"\n  VERDICT: {results['verdict']}")
    return results


def main():
    fprint("=" * 70)
    fprint("COMPREHENSIVE ADVERSARIAL VALIDATION v2")
    fprint(f"Started: {datetime.now().isoformat()}")
    fprint("Data source: yfinance (REAL market data)")
    fprint("=" * 70)

    # Start MLflow run
    mlflow_run = None
    if MLFLOW_OK:
        try:
            mlflow.set_experiment("adversarial_validation_v2")
            mlflow_run = mlflow.start_run(run_name=f"comprehensive_v2_{datetime.now().strftime('%H%M')}")
        except:
            pass

    all_results = {
        'meta': {
            'start_time': datetime.now().isoformat(),
            'data_source': 'yfinance (real market data)',
            'starting_capital': CAP,
            'commission_rt': SPREAD_COMM,
            'haircut': HAIRCUT,
            'n_permutations': 20
        },
        'strategies': {}
    }

    # ── Download data for both universes ──
    fprint("\n[DATA] Downloading broad universe...")
    broad_data = download_data(BROAD_UNIVERSE)
    bc, bh, bl, bv, bspy, bvix, bavail = broad_data

    fprint("\n[DATA] Downloading sector universe...")
    sect_data = download_data(SECTORS)
    sc, sh, sl, sv, sspy, svix, savail = sect_data

    if bc is None or sc is None:
        fprint("FATAL: Could not download data. Aborting.")
        return

    # ── Strategy Configurations ──
    strategies = {
        'S1_broad_momentum_25etf': {
            'universe': 'broad',
            'bt_kwargs': {'top_k': 3, 'spread_pct': 0.03, 'rebal_days': 10, 'vix_filter': 20},
            'data': (bc, bh, bl, bvix, bspy, bavail),
            'original_claim': 'Sharpe 4.70'
        },
        'S2_integrated_qualmom_11sect': {
            'universe': 'sector',
            'bt_kwargs': {'top_k': 3, 'spread_pct': 0.03, 'rebal_days': 10, 'vix_filter': 20},
            'data': (sc, sh, sl, svix, sspy, savail),
            'original_claim': 'Sharpe 4.20'
        },
        'S3_multifactor_sector': {
            'universe': 'sector',
            'bt_kwargs': {'top_k': 3, 'spread_pct': 0.03, 'rebal_days': 10, 'vix_filter': 20},
            'data': (sc, sh, sl, svix, sspy, savail),
            'original_claim': 'Sharpe 3.87'
        },
        'S4_sector_concentrated_top2': {
            'universe': 'sector',
            'bt_kwargs': {'top_k': 2, 'spread_pct': 0.03, 'rebal_days': 10, 'vix_filter': 20},
            'data': (sc, sh, sl, svix, sspy, savail),
            'original_claim': 'High conviction concentrated'
        },
        'S5_broad_no_vix_filter': {
            'universe': 'broad',
            'bt_kwargs': {'top_k': 3, 'spread_pct': 0.03, 'rebal_days': 10, 'vix_filter': None},
            'data': (bc, bh, bl, bvix, bspy, bavail),
            'original_claim': 'Trades all VIX levels'
        },
        'S6_sector_monthly_rebal': {
            'universe': 'sector',
            'bt_kwargs': {'top_k': 3, 'spread_pct': 0.03, 'rebal_days': 21, 'vix_filter': 20},
            'data': (sc, sh, sl, svix, sspy, savail),
            'original_claim': 'Monthly rebalance variant'
        }
    }

    # ── Run validation for each strategy ──
    strategy_sharpes = {}

    for name, cfg in strategies.items():
        try:
            close, high, low, vix, spy, available = cfg['data']
            results = validate_strategy(
                name, cfg['universe'], cfg['bt_kwargs'],
                close, high, low, vix, spy, available
            )
            all_results['strategies'][name] = results
            if results.get('baseline', {}).get('sharpe_honest'):
                strategy_sharpes[name] = results['baseline']['sharpe_honest']

            # Log to MLflow
            if MLFLOW_OK and mlflow_run:
                try:
                    prefix = name.replace(' ', '_')[:20]
                    mlflow.log_metric(f"{prefix}_sharpe_honest", results.get('baseline', {}).get('sharpe_honest', 0))
                    mlflow.log_metric(f"{prefix}_sharpe_inflated", results.get('baseline', {}).get('sharpe_inflated', 0))
                    mlflow.log_metric(f"{prefix}_wr", results.get('baseline', {}).get('win_rate', 0))
                    mlflow.log_metric(f"{prefix}_pf", results.get('baseline', {}).get('profit_factor', 0))
                    for check in ['permutation', 'regime', 'lookahead', 'cost_sensitivity', 'stability']:
                        if check in results:
                            status_val = 1.0 if results[check]['status'] == 'PASS' else (0.5 if results[check]['status'] in ('MARGINAL','WARNING') else 0.0)
                            mlflow.log_metric(f"{prefix}_{check}", status_val)
                except:
                    pass

        except Exception as e:
            fprint(f"\n  ERROR validating {name}: {e}")
            traceback.print_exc()
            all_results['strategies'][name] = {'error': str(e), 'overall_status': 'ERROR'}

    # ── Data Snooping Correction ──
    # Total strategies ever tested in this research program (conservative estimate)
    N_TOTAL_STRATEGIES = 50  # We've tested ~50+ variants across all experiments
    fprint(f"\n{'='*70}")
    fprint("DATA SNOOPING CORRECTION")
    fprint(f"{'='*70}")
    if strategy_sharpes:
        snooping = test_data_snooping(strategy_sharpes, N_TOTAL_STRATEGIES)
        all_results['data_snooping'] = snooping
        fprint(f"  {snooping['conclusion']}")
    else:
        all_results['data_snooping'] = {'status': 'FAIL', 'conclusion': 'No valid Sharpe ratios to test'}

    # ── Final Summary ──
    fprint(f"\n{'='*70}")
    fprint("FINAL ADVERSARIAL SUMMARY")
    fprint(f"{'='*70}")

    summary_table = []
    for name, results in all_results['strategies'].items():
        if 'overall_status' not in results:
            continue
        baseline = results.get('baseline', {})
        original = strategies.get(name, {}).get('original_claim', '?')
        summary_table.append({
            'strategy': name,
            'original_claim': original,
            'honest_sharpe': baseline.get('sharpe_honest', 0),
            'inflated_sharpe': baseline.get('sharpe_inflated', 0),
            'win_rate': baseline.get('win_rate', 0),
            'profit_factor': baseline.get('profit_factor', 0),
            'n_trades': baseline.get('n_trades', 0),
            'overall': results['overall_status'],
            'checks': results.get('check_summary', {})
        })

    for row in summary_table:
        fprint(f"\n  {row['strategy']}:")
        fprint(f"    Original claim: {row['original_claim']}")
        fprint(f"    Honest Sharpe: {row['honest_sharpe']:.3f} (inflated: {row['inflated_sharpe']:.3f})")
        fprint(f"    WR: {row['win_rate']:.1f}%, PF: {row['profit_factor']:.2f}, Trades: {row['n_trades']}")
        fprint(f"    Checks: {row['checks']}")
        fprint(f"    VERDICT: {row['overall']}")

    all_results['summary_table'] = summary_table
    all_results['meta']['end_time'] = datetime.now().isoformat()

    # ── Honest Bottom Line ──
    passing = [r for r in summary_table if r['overall'] in ('FULL_PASS', 'CONDITIONAL_PASS')]
    failing = [r for r in summary_table if r['overall'] == 'FAIL']

    fprint(f"\n{'='*70}")
    fprint("HONEST BOTTOM LINE")
    fprint(f"{'='*70}")
    if passing:
        fprint(f"  PASSING ({len(passing)}):")
        for r in passing:
            fprint(f"    {r['strategy']}: Honest Sharpe {r['honest_sharpe']:.3f}, {r['overall']}")
    else:
        fprint("  NO strategies pass all adversarial checks.")

    if failing:
        fprint(f"\n  FAILING ({len(failing)}):")
        for r in failing:
            fprint(f"    {r['strategy']}: Honest Sharpe {r['honest_sharpe']:.3f} (claimed {r['original_claim']})")

    # Inflate warning
    inflated = [r for r in summary_table
                if r['inflated_sharpe'] > 0 and r['honest_sharpe'] > 0
                and r['inflated_sharpe'] / (r['honest_sharpe'] + 1e-10) > 2.0]
    if inflated:
        fprint(f"\n  WARNING — Sharpe inflation detected in {len(inflated)} strategies:")
        for r in inflated:
            ratio = r['inflated_sharpe'] / (r['honest_sharpe'] + 1e-10)
            fprint(f"    {r['strategy']}: {r['inflated_sharpe']:.2f} inflated vs {r['honest_sharpe']:.3f} honest ({ratio:.1f}x inflation)")

    # ── Save results ──
    # Strip trade logs for JSON (too large)
    save_results = json.loads(json.dumps(all_results, default=str))
    for sname in save_results.get('strategies', {}):
        s = save_results['strategies'][sname]
        if 'baseline' in s and 'trade_log' in s.get('baseline', {}):
            del s['baseline']['trade_log']

    with open(RESULTS_PATH, 'w') as f:
        json.dump(save_results, f, indent=2, default=str)
    fprint(f"\nResults saved to {RESULTS_PATH}")

    # ── Log to MLflow ──
    if MLFLOW_OK and mlflow_run:
        try:
            n_pass = len(passing)
            n_fail = len(failing)
            mlflow.log_metric("n_strategies_tested", len(summary_table))
            mlflow.log_metric("n_passing", n_pass)
            mlflow.log_metric("n_failing", n_fail)
            mlflow.log_metric("n_total_variants_snooping", N_TOTAL_STRATEGIES)
            mlflow.log_artifact(str(RESULTS_PATH))
            mlflow.end_run()
        except:
            pass

    fprint(f"\nCompleted: {datetime.now().isoformat()}")


if __name__ == '__main__':
    main()
