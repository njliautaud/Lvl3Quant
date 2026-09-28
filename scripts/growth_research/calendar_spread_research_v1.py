#!/usr/bin/env python3
"""
Calendar Spread Research v1 — Time Spread Income on Sector ETFs
================================================================

Tests whether calendar spreads (sell near-term, buy longer-term at same strike)
produce a viable income/growth strategy at $645. Calendar spreads profit from
the faster theta decay of the near-term option relative to the far-term option.

6 variants tested:
  A. ATM call calendar on TOP 3 LGBM sectors, regime>0.4, 21/45 DTE
  B. ATM call calendar on TOP 3, regime>0.4, 14/45 DTE (wider time gap)
  C. ATM put calendar on BOTTOM 3 LGBM sectors, regime<0.2, 21/45 DTE
  D. ATM call calendar, VIX>25 only (elevated IV = richer front premium)
  E. ATM call calendar, all VIX levels (no filter)
  F. Double calendar (call + put at same strike) on top 3 sectors

Walk-forward LGBM with 21 features (18 legacy + 3 cross-asset).
$645 capital, max $200/trade, $2.60 commission RT.
Full 5-gate adversarial validation.

MLflow: calendar_spread_research_v1 @ http://jupiter:5000
"""

import sys
import json
import warnings
import time
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings('ignore')

sys.path.insert(0, '/home/jupiter/Lvl3Quant')
from research.tools.options_pricer import (
    bs_call_price, bs_put_price, estimate_iv, compute_atr,
    COMMISSION_RT_SPREAD, DEFAULT_HAIRCUT, RISK_FREE_RATE,
)
from research.tools.adversarial_validator import validate_trades

def fprint(*a, **kw):
    print(*a, **kw, flush=True)

# ─── MLflow ─────────────────────────────────────────────────────────
MLFLOW_OK = False
try:
    import urllib.request
    urllib.request.urlopen('http://jupiter:5000/', timeout=2)
    import mlflow
    mlflow.set_tracking_uri('http://jupiter:5000')
    MLFLOW_OK = True
except Exception:
    fprint("MLflow unavailable — results saved to JSON only")

# ─── Constants ──────────────────────────────────────────────────────
BASE = Path('/home/jupiter/Lvl3Quant')
OUT_DIR = BASE / 'research' / 'findings'
OUT_DIR.mkdir(parents=True, exist_ok=True)

SECTORS = ['XLK', 'XLF', 'XLE', 'XLV', 'XLY', 'XLP', 'XLI', 'XLB', 'XLU', 'XLRE', 'XLC']
EXTRA_TICKERS = ['SPY', '^VIX', 'TLT', 'SHY', 'HYG', 'GLD']

CAP = 645.0
MAX_PER_TRADE = 200.0
COMMISSION = COMMISSION_RT_SPREAD  # $2.60 RT
HAIRCUT = DEFAULT_HAIRCUT           # 15%
TOP_K = 3
TRAIN_PERIODS = 12  # WF LGBM training periods (months)

# 18 legacy features + 3 cross-asset = 21
LEGACY_FEATURES = [
    'ret_5d', 'ret_10d', 'ret_21d', 'ret_63d', 'ret_126d', 'ret_252d',
    'vol_21d', 'vol_63d', 'sharpe_63d', 'maxdd_63d', 'pct_52w_high', 'mom_accel',
    'pct_pos_months_12m', 'sortino_63d', 'calmar_1y', 'up_capture', 'dn_capture',
    'trend_r2_63d',
]
CROSS_ASSET_FEATURES = ['tlt_spy_corr_63d', 'gld_ret_21d', 'hyg_spread_z']
ALL_FEATURES = LEGACY_FEATURES + CROSS_ASSET_FEATURES

np.random.seed(42)


# ═══════════════════════════════════════════════════════════════════
# 1. DATA
# ═══════════════════════════════════════════════════════════════════

def download_data():
    import yfinance as yf
    fprint("[1/6] Downloading data...")
    tickers = list(set(SECTORS + EXTRA_TICKERS))
    raw = yf.download(tickers, start='2012-01-01', end='2026-07-27', progress=False)
    mi = isinstance(raw.columns, pd.MultiIndex)
    close = raw['Close'] if mi else raw
    high = raw['High'] if mi else raw
    low = raw['Low'] if mi else raw
    volume = raw['Volume'] if mi else raw

    vc = '^VIX' if '^VIX' in close.columns else 'VIX'
    vix = close[vc].dropna()
    spy = close['SPY'].dropna()

    avail = [c for c in SECTORS if c in close.columns and close[c].dropna().shape[0] > 500]
    sc = close[avail].dropna(how='all')
    sh = high[[c for c in avail if c in high.columns]].dropna(how='all')
    sl = low[[c for c in avail if c in low.columns]].dropna(how='all')
    sv = volume[[c for c in avail if c in volume.columns]].dropna(how='all')

    extras = {}
    for tk in ['TLT', 'SHY', 'HYG', 'GLD']:
        if tk in close.columns:
            extras[tk] = close[tk].dropna()

    ix = sc.index.intersection(vix.index).intersection(spy.index)
    for s in [sh, sl] + list(extras.values()):
        ix = ix.intersection(s.index)

    fprint(f"  {len(ix)} days, {len(avail)} sectors, extras: {list(extras.keys())}")
    return (
        sc.loc[ix], sh.loc[ix], sl.loc[ix], sv.reindex(ix),
        spy.loc[ix], vix.loc[ix],
        {k: v.loc[ix] for k, v in extras.items()},
    )


# ═══════════════════════════════════════════════════════════════════
# 2. GRU REGIME PREDICTIONS
# ═══════════════════════════════════════════════════════════════════

def load_regime_data():
    """Load GRU regime predictions. Returns dict date_str -> regime_score."""
    path = BASE / 'output' / 'regime_detector_v1' / 'regime_predictions_v1.npz'
    if not path.exists():
        fprint(f"  WARNING: Regime file not found, will skip regime filters")
        return {}
    d = np.load(path, allow_pickle=True)
    dates = d['dates']
    scores = d['regime_scores']
    fprint(f"  Regime predictions: {len(dates)} dates, score range [{scores.min():.3f}, {scores.max():.3f}]")
    return dict(zip(dates, scores))


# ═══════════════════════════════════════════════════════════════════
# 3. FEATURE ENGINEERING (18 legacy + 3 cross-asset = 21)
# ═══════════════════════════════════════════════════════════════════

def compute_features(px, vol_data=None):
    """Compute 18 legacy features for a single asset."""
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

    pk252 = px.iloc[-252:].cummax()
    mdd = float(((px.iloc[-252:] / pk252) - 1).min())
    cagr = float(px.iloc[-1] / px.iloc[-252] - 1) if len(px) >= 252 else 0.0
    f['calmar_1y'] = cagr / (abs(mdd) + 1e-10)

    up = rets[rets > 0]
    dn = rets[rets < 0]
    f['up_capture'] = float(up.iloc[-63:].mean() / (up.mean() + 1e-10)) if len(up) > 10 else 1.0
    f['dn_capture'] = float(dn.iloc[-63:].mean() / (dn.mean() + 1e-10)) if len(dn) > 10 else 1.0

    if len(px) >= 63:
        y = np.log(px.iloc[-63:].values + 1e-10)
        x = np.arange(len(y))
        slope, _, r_val, _, _ = stats.linregress(x, y)
        f['trend_r2_63d'] = r_val ** 2
    else:
        f['trend_r2_63d'] = 0.0

    return f


def compute_cross_asset_features(date_idx, spy, extras):
    """Compute 3 cross-asset features at a given index."""
    ca = {}
    # TLT-SPY correlation (63d)
    if 'TLT' in extras and date_idx >= 63:
        tlt_r = extras['TLT'].pct_change().iloc[date_idx - 63:date_idx]
        spy_r = spy.pct_change().iloc[date_idx - 63:date_idx]
        ca['tlt_spy_corr_63d'] = float(tlt_r.corr(spy_r)) if len(tlt_r) > 10 else 0.0
    else:
        ca['tlt_spy_corr_63d'] = 0.0

    # GLD 21d return
    if 'GLD' in extras and date_idx >= 21:
        ca['gld_ret_21d'] = float(extras['GLD'].iloc[date_idx] / extras['GLD'].iloc[date_idx - 21] - 1)
    else:
        ca['gld_ret_21d'] = 0.0

    # HYG spread z-score (proxy: HYG 63d return z-scored over 252d)
    if 'HYG' in extras and date_idx >= 252:
        hyg_rets = extras['HYG'].pct_change().iloc[date_idx - 252:date_idx]
        r63 = float(extras['HYG'].iloc[date_idx] / extras['HYG'].iloc[date_idx - 63] - 1)
        ca['hyg_spread_z'] = (r63 - float(hyg_rets.mean() * 63)) / (float(hyg_rets.std() * np.sqrt(63)) + 1e-10)
    else:
        ca['hyg_spread_z'] = 0.0

    return ca


# ═══════════════════════════════════════════════════════════════════
# 4. WALK-FORWARD LGBM RANKING
# ═══════════════════════════════════════════════════════════════════

def build_rankings(sc, sv, spy, extras, rebal_dates, fwd_days=21):
    """Walk-forward LGBM: predict forward returns, rank sectors."""
    import lightgbm as lgb
    fprint("[3/6] Walk-forward LGBM ranking (21 features)...")

    records = []
    for dt in rebal_dates:
        idx = sc.index.get_indexer([dt], method='ffill')[0]
        if idx < 260:
            continue
        ca_feats = compute_cross_asset_features(idx, spy, extras)
        for tk in sc.columns:
            px = sc[tk].iloc[:idx + 1].dropna()
            vol_d = sv[tk].iloc[:idx + 1] if tk in sv.columns else None
            feats = compute_features(px, vol_d)
            if not feats:
                continue
            fi = min(idx + fwd_days, len(sc) - 1)
            feats.update(ca_feats)
            feats.update({
                'date': dt,
                'ticker': tk,
                'fwd_ret': float(sc[tk].iloc[fi] / sc[tk].iloc[idx] - 1),
            })
            records.append(feats)

    df = pd.DataFrame(records)
    for c in ALL_FEATURES:
        if c not in df.columns:
            df[c] = 0.0
    df[ALL_FEATURES] = df[ALL_FEATURES].fillna(0.0)

    if len(df) < 100:
        fprint("  ERROR: not enough data for LGBM")
        return {}

    df['rank_label'] = df.groupby('date')['fwd_ret'].rank(pct=True)
    dates = sorted(df['date'].unique())
    rankings = {}

    for i in range(TRAIN_PERIODS, len(dates)):
        td = dates[max(0, i - TRAIN_PERIODS):i]
        test_date = dates[i]
        tr = df[df['date'].isin(td)]
        te = df[df['date'] == test_date].copy()
        if len(te) < 3 or len(tr) < 50:
            continue

        Xt = np.nan_to_num(tr[ALL_FEATURES].values.astype(np.float32))
        yt = tr['rank_label'].values.astype(np.float32)
        Xe = np.nan_to_num(te[ALL_FEATURES].values.astype(np.float32))

        try:
            m = lgb.LGBMRegressor(
                n_estimators=100, max_depth=4, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, min_child_samples=5, verbose=-1,
            )
            m.fit(Xt, yt)
            te['score'] = m.predict(Xe)
            rankings[test_date] = dict(zip(te['ticker'], te['score']))
        except Exception:
            continue

    fprint(f"  Rankings: {len(rankings)} dates")
    return rankings


# ═══════════════════════════════════════════════════════════════════
# 5. CALENDAR SPREAD PRICING
# ═══════════════════════════════════════════════════════════════════

def price_call_calendar(S, K, front_dte, back_dte, atr, vix,
                        haircut=HAIRCUT, r=RISK_FREE_RATE):
    """
    Price an ATM call calendar spread (debit).
    Buy back-month call, sell front-month call, same strike.

    Returns:
        (net_debit_per_share, front_credit, back_cost)
        net_debit is what we PAY (after haircut).
    """
    sigma = estimate_iv(atr, S, vix)
    T_front = front_dte / 365.0
    T_back = back_dte / 365.0

    # BS fair values
    front_call = bs_call_price(S, K, T_front, r, sigma)
    back_call = bs_call_price(S, K, T_back, r, sigma)

    # Haircut: we BUY back month (pay more), SELL front month (receive less)
    back_cost = back_call * (1 + haircut)    # pay more than fair
    front_credit = front_call * (1 - haircut)  # receive less than fair

    net_debit = back_cost - front_credit
    if net_debit <= 0:
        return None, None, None  # degenerate case
    return net_debit, front_credit, back_cost


def price_put_calendar(S, K, front_dte, back_dte, atr, vix,
                       haircut=HAIRCUT, r=RISK_FREE_RATE):
    """
    Price an ATM put calendar spread (debit).
    Buy back-month put, sell front-month put, same strike.
    """
    sigma = estimate_iv(atr, S, vix)
    T_front = front_dte / 365.0
    T_back = back_dte / 365.0

    front_put = bs_put_price(S, K, T_front, r, sigma)
    back_put = bs_put_price(S, K, T_back, r, sigma)

    back_cost = back_put * (1 + haircut)
    front_credit = front_put * (1 - haircut)

    net_debit = back_cost - front_credit
    if net_debit <= 0:
        return None, None, None
    return net_debit, front_credit, back_cost


def calendar_exit_value(S, K, remaining_dte, option_type, atr, vix,
                        haircut=HAIRCUT, r=RISK_FREE_RATE):
    """
    Value the back-month leg at front-month expiry.
    Front-month leg settles at intrinsic (no haircut — exercise/assignment).
    Back-month leg sold at theoretical - haircut.

    Returns:
        (spread_value, front_intrinsic, back_theoretical)
    """
    sigma = estimate_iv(atr, S, vix)

    # Front leg at expiry: intrinsic only
    if option_type == 'call':
        front_intrinsic = max(S - K, 0)
    else:
        front_intrinsic = max(K - S, 0)

    # Back leg: still has time value, sell with haircut
    T_remaining = remaining_dte / 365.0
    if option_type == 'call':
        back_theo = bs_call_price(S, K, T_remaining, r, sigma)
    else:
        back_theo = bs_put_price(S, K, T_remaining, r, sigma)

    back_exit = back_theo * (1 - haircut)  # receive less than fair

    # Calendar value at exit:
    # We are LONG back, SHORT front
    # We receive back_exit, we owe front_intrinsic
    spread_value = back_exit - front_intrinsic

    return spread_value, front_intrinsic, back_exit


# ═══════════════════════════════════════════════════════════════════
# 6. STRATEGY SIMULATION
# ═══════════════════════════════════════════════════════════════════

def simulate_variant(variant_name, rankings, sc, sh, sl, spy, vix, extras,
                     regime_data,
                     top_k=3, bottom_k=3,
                     front_dte=21, back_dte=45,
                     option_type='call',
                     use_regime_bull=False, regime_bull_thresh=0.4,
                     use_regime_bear=False, regime_bear_thresh=0.2,
                     vix_min=None, vix_max=None,
                     double_calendar=False):
    """
    Simulate a calendar spread variant.

    Calendar spread mechanics:
    - Entry: sell front-month option, buy back-month option at same (ATM) strike
    - Net debit = back_cost - front_credit (we pay this)
    - At front expiry: front settles at intrinsic, back still has time value
    - P&L = spread_value_at_exit - net_debit - commission
    - Max loss = net debit (if spread goes to zero)
    """
    fprint(f"\n  --- {variant_name} ---")
    fprint(f"  Type: {option_type} calendar, front={front_dte}d, back={back_dte}d")
    if double_calendar:
        fprint(f"  Double calendar: call + put at same strike")

    # Pre-compute ATR series
    atr_series = {}
    for tk in sc.columns:
        if tk in sh.columns and tk in sl.columns:
            atr_series[tk] = compute_atr(sh[tk].values, sl[tk].values, sc[tk].values, period=14)

    equity = CAP
    trades = []

    sorted_dates = sorted(rankings.keys())

    for dt in sorted_dates:
        if dt not in spy.index or dt not in vix.index:
            continue

        # VIX filter
        cv = float(vix.loc[dt])
        if vix_min is not None and cv < vix_min:
            continue
        if vix_max is not None and cv > vix_max:
            continue

        # Regime filter
        dt_str = dt.strftime('%Y-%m-%d') if hasattr(dt, 'strftime') else str(dt)
        regime_score = regime_data.get(dt_str, 0.5)

        if use_regime_bull and regime_score <= regime_bull_thresh:
            continue
        if use_regime_bear and regime_score >= regime_bear_thresh:
            continue

        if equity <= 50:
            continue

        scores = rankings[dt]
        if not scores:
            continue

        # Select sectors
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        if use_regime_bear:
            # For bearish put calendars, pick BOTTOM sectors
            picks = [tk for tk, _ in ranked[-bottom_k:] if tk in atr_series]
        else:
            picks = [tk for tk, _ in ranked[:top_k] if tk in atr_series]

        if not picks:
            continue

        per_pick = min(MAX_PER_TRADE, equity / len(picks))

        for tk in picks:
            spot = float(sc[tk].loc[dt])
            atr_val = atr_series.get(tk, spot * 0.02)

            # ATM strike
            K = round(spot)
            if K <= 0:
                continue

            # Price calendar spread
            if option_type == 'call' or double_calendar:
                result = price_call_calendar(spot, K, front_dte, back_dte, atr_val, cv)
                if result[0] is None:
                    continue
                call_debit = result[0]
            if option_type == 'put' or double_calendar:
                result = price_put_calendar(spot, K, front_dte, back_dte, atr_val, cv)
                if result[0] is None:
                    continue
                put_debit = result[0]

            if double_calendar:
                net_debit = call_debit + put_debit
            elif option_type == 'call':
                net_debit = call_debit
            else:
                net_debit = put_debit

            # Cost per contract
            cost_per_contract = net_debit * 100
            if cost_per_contract <= 0 or cost_per_contract > per_pick:
                continue

            n_contracts = max(1, int(per_pick / cost_per_contract))
            total_cost = cost_per_contract * n_contracts

            # Check we can afford it
            if total_cost > equity:
                n_contracts = max(1, int(equity / cost_per_contract))
                total_cost = cost_per_contract * n_contracts
                if total_cost > equity:
                    continue

            # Find exit date (front-month expiry)
            dt_idx = sc.index.get_loc(dt)
            exit_idx = min(dt_idx + front_dte, len(sc) - 1)
            exit_date = sc.index[exit_idx]
            exit_price = float(sc[tk].iloc[exit_idx])

            # Remaining DTE for back leg
            remaining_dte = back_dte - front_dte

            # Re-compute ATR at exit for IV estimation
            exit_atr = atr_val  # approximate
            exit_vix = float(vix.iloc[exit_idx]) if exit_idx < len(vix) else cv

            # Calendar spread value at exit
            if double_calendar:
                call_exit, call_front_intr, call_back_val = calendar_exit_value(
                    exit_price, K, remaining_dte, 'call', exit_atr, exit_vix)
                put_exit, put_front_intr, put_back_val = calendar_exit_value(
                    exit_price, K, remaining_dte, 'put', exit_atr, exit_vix)
                spread_exit = call_exit + put_exit
                comm = COMMISSION * 2 * n_contracts  # 4 legs for double
            else:
                spread_exit, front_intr, back_val = calendar_exit_value(
                    exit_price, K, remaining_dte, option_type, exit_atr, exit_vix)
                comm = COMMISSION * n_contracts

            # P&L per share
            pnl_per_share = spread_exit - net_debit
            pnl = pnl_per_share * 100 * n_contracts - comm

            # Calendar max loss = net debit paid (can't lose more than that)
            if pnl < -total_cost:
                pnl = -total_cost

            equity += pnl

            trades.append({
                'entry_date': str(dt.date()) if hasattr(dt, 'date') else str(dt),
                'exit_date': str(exit_date.date()) if hasattr(exit_date, 'date') else str(exit_date),
                'ticker': tk,
                'spot': spot,
                'exit_price': exit_price,
                'strike': K,
                'price_move': exit_price / spot - 1,
                'net_debit_total': total_cost,
                'spread_exit_total': spread_exit * 100 * n_contracts,
                'pnl': pnl,
                'pnl_pct': pnl / total_cost if total_cost > 0 else 0,
                'n_contracts': n_contracts,
                'vix': cv,
                'regime_score': regime_score,
                'option_type': 'double' if double_calendar else option_type,
                'front_dte': front_dte,
                'back_dte': back_dte,
            })

    if not trades:
        fprint(f"  NO TRADES for {variant_name}")
        return None

    # Compute summary
    pnls = [t['pnl'] for t in trades]
    winners = sum(1 for p in pnls if p > 0)
    total_pnl = sum(pnls)
    wr = winners / len(pnls)
    gp = sum(p for p in pnls if p > 0)
    gl = abs(sum(p for p in pnls if p < 0))
    pf = gp / gl if gl > 0 else float('inf')

    final_eq = CAP + total_pnl

    fprint(f"  Trades: {len(trades)}, WR: {wr:.1%}, PF: {pf:.2f}")
    fprint(f"  Total PnL: ${total_pnl:.2f}, Final equity: ${final_eq:.0f}")

    return {
        'variant': variant_name,
        'trades': trades,
        'n_trades': len(trades),
        'win_rate': wr,
        'profit_factor': pf,
        'total_pnl': total_pnl,
        'final_equity': final_eq,
    }


# ═══════════════════════════════════════════════════════════════════
# 7. MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    fprint("=" * 70)
    fprint("CALENDAR SPREAD RESEARCH v1")
    fprint(f"Capital: ${CAP}, Max/trade: ${MAX_PER_TRADE}")
    fprint(f"Commission: ${COMMISSION} RT, Haircut: {HAIRCUT:.0%}")
    fprint("=" * 70)

    # 1. Download data
    sc, sh, sl, sv, spy, vix, extras = download_data()

    # 2. Load regime data
    fprint("\n[2/6] Loading GRU regime predictions...")
    regime_data = load_regime_data()

    # 3. Build LGBM rankings
    # Bi-weekly rebalance dates
    rebal_dates = sc.index[::14]  # every ~2 weeks
    rankings = build_rankings(sc, sv, spy, extras, rebal_dates)

    if len(rankings) < 20:
        fprint("ERROR: insufficient ranking data")
        return

    # 4. Run 6 variants
    fprint("\n[4/6] Running 6 calendar spread variants...")

    variants = {}

    # A. ATM call calendar, TOP 3, regime>0.4, 21/45 DTE
    variants['A'] = simulate_variant(
        'A: Call Cal, Top3, regime>0.4, 21/45',
        rankings, sc, sh, sl, spy, vix, extras, regime_data,
        top_k=3, front_dte=21, back_dte=45, option_type='call',
        use_regime_bull=True, regime_bull_thresh=0.4,
    )

    # B. ATM call calendar, TOP 3, regime>0.4, 14/45 DTE (wider time gap)
    variants['B'] = simulate_variant(
        'B: Call Cal, Top3, regime>0.4, 14/45',
        rankings, sc, sh, sl, spy, vix, extras, regime_data,
        top_k=3, front_dte=14, back_dte=45, option_type='call',
        use_regime_bull=True, regime_bull_thresh=0.4,
    )

    # C. ATM put calendar, BOTTOM 3, regime<0.2, 21/45 DTE
    variants['C'] = simulate_variant(
        'C: Put Cal, Bot3, regime<0.2, 21/45',
        rankings, sc, sh, sl, spy, vix, extras, regime_data,
        bottom_k=3, front_dte=21, back_dte=45, option_type='put',
        use_regime_bear=True, regime_bear_thresh=0.2,
    )

    # D. ATM call calendar, VIX>25 only
    variants['D'] = simulate_variant(
        'D: Call Cal, Top3, VIX>25, 21/45',
        rankings, sc, sh, sl, spy, vix, extras, regime_data,
        top_k=3, front_dte=21, back_dte=45, option_type='call',
        vix_min=25,
    )

    # E. ATM call calendar, all VIX levels (no filter)
    variants['E'] = simulate_variant(
        'E: Call Cal, Top3, no filter, 21/45',
        rankings, sc, sh, sl, spy, vix, extras, regime_data,
        top_k=3, front_dte=21, back_dte=45, option_type='call',
    )

    # F. Double calendar (call + put at same strike) on top 3
    variants['F'] = simulate_variant(
        'F: Double Cal, Top3, regime>0.4, 21/45',
        rankings, sc, sh, sl, spy, vix, extras, regime_data,
        top_k=3, front_dte=21, back_dte=45,
        use_regime_bull=True, regime_bull_thresh=0.4,
        double_calendar=True,
    )

    # 5. Adversarial validation on each variant
    fprint("\n[5/6] Adversarial validation (5-gate)...")

    spy_close = spy.copy()
    spy_close.index = pd.DatetimeIndex(spy_close.index)

    results_summary = {}
    best_variant = None
    best_sharpe = -999

    for key, v in variants.items():
        if v is None:
            fprint(f"\n  {key}: SKIPPED (no trades)")
            results_summary[key] = {'status': 'NO_TRADES'}
            continue

        validation = validate_trades(
            trades=v['trades'],
            initial_capital=CAP,
            spy_prices=spy_close,
            strategy_name=v['variant'],
            n_perms=2000,
        )
        validation.print_summary()

        v['validation'] = validation
        results_summary[key] = {
            'variant': v['variant'],
            'n_trades': v['n_trades'],
            'win_rate': v['win_rate'],
            'profit_factor': v['profit_factor'],
            'sharpe': validation.sharpe,
            'sortino': validation.sortino,
            'cagr': validation.cagr,
            'max_dd': validation.max_dd,
            'final_equity': validation.final_equity,
            'gates_passed': validation.gates_passed,
            'gates_total': validation.gates_total,
            'all_passed': validation.all_passed,
        }

        if validation.sharpe > best_sharpe:
            best_sharpe = validation.sharpe
            best_variant = key

    # 6. Summary comparison
    fprint("\n[6/6] Results summary...")
    fprint("\n" + "=" * 90)
    fprint(f"{'Variant':<45} {'Sharpe':>7} {'Sort':>7} {'WR':>6} {'PF':>6} {'CAGR':>8} {'MaxDD':>8} {'Gates':>7} {'Final':>8}")
    fprint("-" * 90)

    for key in ['A', 'B', 'C', 'D', 'E', 'F']:
        r = results_summary.get(key, {})
        if r.get('status') == 'NO_TRADES' or 'sharpe' not in r:
            fprint(f"{key}: No trades")
            continue
        fprint(f"{r['variant']:<45} {r['sharpe']:>7.2f} {r['sortino']:>7.2f} "
               f"{r['win_rate']:>5.1%} {r['profit_factor']:>6.2f} "
               f"{r['cagr']:>7.1%} {r['max_dd']:>7.1%} "
               f"{r['gates_passed']:>2}/{r['gates_total']:>1} "
               f"${r['final_equity']:>7.0f}")

    fprint("=" * 90)

    if best_variant:
        bv = results_summary[best_variant]
        fprint(f"\nBEST: {bv['variant']}")
        fprint(f"  Sharpe: {bv['sharpe']:.2f}, Sortino: {bv['sortino']:.2f}")
        fprint(f"  WR: {bv['win_rate']:.1%}, PF: {bv['profit_factor']:.2f}")
        fprint(f"  CAGR: {bv['cagr']:.1%}, MaxDD: {bv['max_dd']:.1%}")
        fprint(f"  Gates: {bv['gates_passed']}/{bv['gates_total']}")
        fprint(f"  Final equity: ${bv['final_equity']:.0f}")

        # Ticker breakdown for best variant
        best_v = variants[best_variant]
        if best_v and best_v['trades']:
            tdf = pd.DataFrame(best_v['trades'])
            fprint(f"\n  Ticker breakdown:")
            for tk, g in tdf.groupby('ticker'):
                if len(g) >= 2:
                    fprint(f"    {tk}: {len(g)}t, WR {(g['pnl'] > 0).mean():.0%}, "
                           f"PnL ${g['pnl'].sum():.0f}, "
                           f"avg move {g['price_move'].mean():.1%}")

            # P&L by year
            tdf['year'] = pd.to_datetime(tdf['exit_date']).dt.year
            fprint(f"\n  Yearly P&L:")
            for yr, g in tdf.groupby('year'):
                fprint(f"    {yr}: {len(g)}t, PnL ${g['pnl'].sum():.0f}, "
                       f"WR {(g['pnl'] > 0).mean():.0%}")

    # Verdict
    fprint("\n" + "=" * 70)
    any_viable = any(
        r.get('sharpe', 0) > 1.0 and r.get('gates_passed', 0) >= 3
        for r in results_summary.values()
    )
    if any_viable:
        fprint("VERDICT: Calendar spreads show POTENTIAL as a complement to bull call spreads.")
        fprint("Recommended: further testing with live option chain data before deployment.")
    else:
        fprint("VERDICT: Calendar spreads at $645 do NOT produce viable risk-adjusted returns.")
        fprint("Stick with bull call spreads (Sharpe 1.87-2.96) as the primary options strategy.")
    fprint("=" * 70)

    # Save results
    save_path = OUT_DIR / 'calendar_spread_research_v1_results.json'
    save_data = {
        'timestamp': datetime.now().isoformat(),
        'config': {
            'capital': CAP,
            'max_per_trade': MAX_PER_TRADE,
            'commission': COMMISSION,
            'haircut': HAIRCUT,
            'features': len(ALL_FEATURES),
            'train_periods': TRAIN_PERIODS,
        },
        'variants': {
            k: {kk: vv for kk, vv in v.items() if kk != 'validation'}
            for k, v in results_summary.items()
        },
        'best_variant': best_variant,
        'verdict': 'VIABLE' if any_viable else 'NOT_VIABLE',
        'runtime_seconds': round(time.time() - t0, 1),
    }
    with open(save_path, 'w') as f:
        json.dump(save_data, f, indent=2, default=str)
    fprint(f"\nResults saved. Runtime: {time.time() - t0:.0f}s")

    # MLflow logging
    if MLFLOW_OK:
        try:
            mlflow.set_experiment('calendar_spread_research_v1')
            with mlflow.start_run(run_name=f"v1_{datetime.now().strftime('%Y%m%d_%H%M')}"):
                mlflow.log_param('capital', CAP)
                mlflow.log_param('max_per_trade', MAX_PER_TRADE)
                mlflow.log_param('commission', COMMISSION)
                mlflow.log_param('haircut', HAIRCUT)
                mlflow.log_param('n_features', len(ALL_FEATURES))
                mlflow.log_param('best_variant', best_variant or 'none')
                mlflow.log_param('verdict', 'VIABLE' if any_viable else 'NOT_VIABLE')

                for key, r in results_summary.items():
                    if 'sharpe' in r:
                        mlflow.log_metric(f'{key}_sharpe', r['sharpe'])
                        mlflow.log_metric(f'{key}_sortino', r['sortino'])
                        mlflow.log_metric(f'{key}_wr', r['win_rate'])
                        mlflow.log_metric(f'{key}_pf', r['profit_factor'])
                        mlflow.log_metric(f'{key}_cagr', r['cagr'])
                        mlflow.log_metric(f'{key}_maxdd', r['max_dd'])
                        mlflow.log_metric(f'{key}_gates', r['gates_passed'])
                        mlflow.log_metric(f'{key}_final_eq', r['final_equity'])

                if best_variant and best_variant in results_summary:
                    bv = results_summary[best_variant]
                    mlflow.log_metric('best_sharpe', bv.get('sharpe', 0))
                    mlflow.log_metric('best_sortino', bv.get('sortino', 0))
                    mlflow.log_metric('best_gates', bv.get('gates_passed', 0))

                mlflow.log_artifact(str(save_path))
            fprint("MLflow run logged.")
        except Exception as e:
            fprint(f"MLflow logging failed: {e}")

    return save_data


if __name__ == '__main__':
    main()
