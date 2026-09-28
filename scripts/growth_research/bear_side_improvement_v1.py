#!/usr/bin/env python3
"""Bear Side Improvement v1 — Research 8 variants to improve bear put spread MDD.

Context: Bull call spread strategy (Sharpe 2.96) only trades when GRU regime > 0.4.
During low-vol (regime < 0.4, ~60% of time), capital sits idle. Bear puts work
(Sharpe 1.39-2.10) but have HIGH MDD (-18% to -49%).

Variants tested:
  A. Baseline bear puts — bottom 3 LGBM sectors, VIX<20, hold to expiry
  B. Tighter stops — exit if spread loses 50% of max loss potential
  C. VIX term structure filter — only when VIX 5d change > 2 (backwardation proxy)
  D. Defensive sector exclusion — never short XLU, XLP, XLRE
  E. Regime-gated bears — only when regime_score < 0.2 (complacency)
  F. Size reduction — half position size on bears
  G. Bear+Bull combined — bull (regime>0.4) + bear (regime<0.2) simultaneously
  H. Momentum acceleration filter — only when 21d < 63d momentum (accelerating down)

Uses standardized tools: options_pricer + adversarial_validator.
Logs to MLflow experiment "bear_side_improvement_v1".
"""
import sys, json, time, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from scipy import stats
import lightgbm as lgb

warnings.filterwarnings('ignore')

# Standardized tools
sys.path.insert(0, "/home/jupiter/Lvl3Quant")
from research.tools.options_pricer import (
    price_bear_put_spread, price_bull_call_spread, spread_pnl,
    estimate_iv, compute_atr, COMMISSION_RT_SPREAD, DEFAULT_HAIRCUT
)
from research.tools.adversarial_validator import validate_trades

BASE = Path("/home/jupiter/Lvl3Quant")
RESULTS_DIR = BASE / 'research' / 'findings'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_PATH = RESULTS_DIR / 'bear_side_improvement_v1_results.json'

def fprint(*a, **kw): print(*a, **kw, flush=True)

# ─── MLflow Setup ───────────────────────────────────────────────────
MLFLOW_OK = False
try:
    import urllib.request
    urllib.request.urlopen('http://jupiter:5000/', timeout=3)
    import mlflow
    mlflow.set_tracking_uri('http://jupiter:5000')
    MLFLOW_OK = True
    fprint("MLflow connected")
except Exception:
    fprint("MLflow unavailable — results saved to disk only")

# ─── Constants ──────────────────────────────────────────────────────
SECTORS = ['XLK', 'XLF', 'XLE', 'XLV', 'XLY', 'XLP', 'XLI', 'XLB', 'XLU', 'XLRE', 'XLC']
DEFENSIVE_SECTORS = {'XLU', 'XLP', 'XLRE'}
CAP = 645.0
DTE = 30
SPREAD_PCT = 3.0  # 3% OTM for put spread width

# 18 legacy features for LGBM
FEATURE_COLS = [
    'ret_5d', 'ret_10d', 'ret_21d', 'ret_63d', 'ret_126d', 'ret_252d',
    'vol_21d', 'vol_63d', 'sharpe_63d', 'maxdd_63d', 'pct_52w_high', 'mom_accel',
    'pct_pos_months_12m', 'sortino_63d', 'calmar_1y', 'up_capture',
    'trend_r2_63d', 'trend_slope_63d',
]


# ═══════════════════════════════════════════════════════════════════
# 1. DATA DOWNLOAD
# ═══════════════════════════════════════════════════════════════════

def download_data():
    import yfinance as yf
    fprint("[1/6] Downloading market data...")
    tickers = SECTORS + ['SPY', '^VIX', 'TLT', 'SHY', 'HYG', 'GLD']
    raw = yf.download(tickers, start='2008-01-01', end='2026-07-27', progress=False)
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

    # Extra assets
    extras = {}
    for tk in ['TLT', 'SHY', 'HYG', 'GLD']:
        if tk in close.columns:
            extras[tk] = close[tk].dropna()

    ix = sc.index.intersection(vix.index).intersection(spy.index).intersection(sh.index)
    fprint(f"  Data: {len(ix)} days, {len(sc.columns)} sectors, {ix[0].date()} to {ix[-1].date()}")
    return sc.loc[ix], sh.loc[ix], sl.loc[ix], sv.reindex(ix), spy.loc[ix], vix.loc[ix], extras


# ═══════════════════════════════════════════════════════════════════
# 2. GRU REGIME PREDICTIONS
# ═══════════════════════════════════════════════════════════════════

def load_regime_predictions():
    fprint("[2/6] Loading GRU regime predictions...")
    regime_path = BASE / 'output' / 'regime_detector_v1' / 'regime_predictions_v1.npz'
    if not regime_path.exists():
        fprint("  WARNING: Regime predictions not found — using VIX proxy")
        return None

    data = np.load(regime_path, allow_pickle=True)
    dates = pd.to_datetime(data['dates'])
    scores = data['regime_scores']
    regime_series = pd.Series(scores, index=dates, name='regime_score')
    if regime_series.index.duplicated().any():
        regime_series = regime_series[~regime_series.index.duplicated(keep='last')]
    fprint(f"  Loaded {len(regime_series)} regime scores, {dates[0].date()} to {dates[-1].date()}")
    fprint(f"  Range: {scores.min():.3f} to {scores.max():.3f}, mean={scores.mean():.3f}")
    fprint(f"  Days regime>0.4: {(scores > 0.4).sum()} ({(scores > 0.4).mean()*100:.1f}%)")
    fprint(f"  Days regime<0.2: {(scores < 0.2).sum()} ({(scores < 0.2).mean()*100:.1f}%)")
    return regime_series


def vix_proxy_regime(vix_series):
    """Fallback: create regime score from VIX. Higher VIX = higher regime score."""
    # Normalize VIX to 0-1 range: VIX 12 -> 0.0, VIX 35+ -> 1.0
    scores = (vix_series - 12) / 23.0
    scores = scores.clip(0, 1)
    return scores.rename('regime_score')


# ═══════════════════════════════════════════════════════════════════
# 3. FEATURE ENGINEERING + LGBM RANKING
# ═══════════════════════════════════════════════════════════════════

def compute_features(px, spy_slice):
    """Compute 18 legacy features for a single sector at a point in time."""
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
    cagr_1y = float(px.iloc[-1] / px.iloc[-252] - 1) if len(px) >= 252 else 0.0
    f['calmar_1y'] = cagr_1y / (abs(mdd) + 1e-10)

    up_days = rets[rets > 0]
    f['up_capture'] = float(up_days.iloc[-63:].mean() / (up_days.mean() + 1e-10)) if len(up_days) > 10 else 1.0

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


def build_lgbm_rankings(sc, spy, vix):
    """Walk-forward LGBM sector ranking with sliding window."""
    fprint("[3/6] Building walk-forward LGBM rankings...")

    # Biweekly rebalancing dates
    all_dates = sc.index[sc.index >= '2010-01-01']
    rebal_dates = all_dates[::10]  # Every ~2 weeks
    fprint(f"  {len(rebal_dates)} rebalance dates")

    # Build feature matrix
    records = []
    for dt in rebal_dates:
        idx = sc.index.get_indexer([dt], method='ffill')[0]
        if idx < 260:
            continue
        for tk in sc.columns:
            px = sc[tk].iloc[:idx + 1].dropna()
            feats = compute_features(px, spy.iloc[:idx + 1])
            if feats is None:
                continue
            # Forward return as label (14 trading days)
            fi = min(idx + 14, len(sc) - 1)
            fwd_ret = float(sc[tk].iloc[fi] / sc[tk].iloc[idx] - 1)
            feats.update({'date': dt, 'ticker': tk, 'fwd_ret': fwd_ret})
            records.append(feats)

    df = pd.DataFrame(records)
    for c in FEATURE_COLS:
        if c not in df.columns:
            df[c] = 0.0
    df[FEATURE_COLS] = df[FEATURE_COLS].fillna(0.0)
    fprint(f"  {len(df)} total feature rows")

    if len(df) < 100:
        fprint("  ERROR: Too few data points for LGBM")
        return {}

    # Rank label (percentile rank of forward return within each date)
    df['rank_label'] = df.groupby('date')['fwd_ret'].rank(pct=True)
    dates = sorted(df['date'].unique())

    # Sliding window walk-forward: 12 periods train, predict next
    rankings = {}
    for i in range(12, len(dates)):
        train_dates = dates[max(0, i - 12):i]
        test_date = dates[i]
        train = df[df['date'].isin(train_dates)]
        test = df[df['date'] == test_date].copy()
        if len(test) < 3 or len(train) < 50:
            continue

        X_train = np.nan_to_num(train[FEATURE_COLS].values.astype(np.float32))
        y_train = train['rank_label'].values.astype(np.float32)
        X_test = np.nan_to_num(test[FEATURE_COLS].values.astype(np.float32))

        try:
            model = lgb.LGBMRegressor(
                n_estimators=100, max_depth=4, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, min_child_samples=5,
                verbose=-1
            )
            model.fit(X_train, y_train)
            test['score'] = model.predict(X_test)
            rankings[test_date] = dict(zip(test['ticker'], test['score']))
        except Exception:
            continue

    fprint(f"  {len(rankings)} ranking dates generated")
    return rankings


# ═══════════════════════════════════════════════════════════════════
# 4. ATR COMPUTATION
# ═══════════════════════════════════════════════════════════════════

def build_atr_dict(sc, sh, sl, period=14):
    """Pre-compute ATR series for all sectors."""
    atr_dict = {}
    for tk in sc.columns:
        if tk in sh.columns and tk in sl.columns:
            h, l, c = sh[tk], sl[tk], sc[tk]
            tr1 = h - l
            tr2 = (h - c.shift(1)).abs()
            tr3 = (l - c.shift(1)).abs()
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr_dict[tk] = tr.ewm(alpha=1 / period, min_periods=period).mean()
    return atr_dict


# ═══════════════════════════════════════════════════════════════════
# 5. SIMULATION ENGINE
# ═══════════════════════════════════════════════════════════════════

def simulate_bear(
    name,
    rankings,
    sc, sh, sl, spy, vix,
    atr_dict,
    regime_scores=None,
    # Variant controls
    bottom_k=3,
    vix_max=20.0,          # A: VIX < 20
    use_tight_stops=False,  # B: exit at 50% max loss
    vix_term_filter=False,  # C: VIX 5d change > 2
    exclude_defensive=False,  # D: no XLU/XLP/XLRE
    regime_gate=None,       # E: only when regime < threshold
    half_size=False,        # F: half position size
    mom_accel_filter=False,  # H: 21d < 63d momentum
):
    """Simulate bear put spread strategy with various filters."""
    equity = CAP
    trades = []
    eq_dates = [sc.index[0]]
    eq_values = [CAP]

    vix_5d_change = vix.diff(5)

    for dt in sorted(rankings.keys()):
        if dt not in spy.index or dt not in vix.index:
            continue

        cv = float(vix.loc[dt])

        # VIX filter (A: only VIX < 20)
        if vix_max is not None and cv > vix_max:
            continue

        # VIX term structure filter (C)
        if vix_term_filter:
            if dt not in vix_5d_change.index or pd.isna(vix_5d_change.loc[dt]):
                continue
            if float(vix_5d_change.loc[dt]) <= 2.0:
                continue

        # Regime gate (E)
        if regime_gate is not None and regime_scores is not None:
            if dt in regime_scores.index:
                rs = float(regime_scores.loc[dt])
                if rs >= regime_gate:
                    continue

        scores = rankings[dt]
        if not scores:
            continue

        # Bottom-ranked sectors (worst expected performance = short candidates)
        ranked = sorted(scores.items(), key=lambda x: x[1])
        picks = [t for t, _ in ranked[:bottom_k]]

        # Defensive exclusion (D)
        if exclude_defensive:
            picks = [t for t in picks if t not in DEFENSIVE_SECTORS]
            if not picks:
                continue

        # Position sizing
        max_pos = min(200, equity / 3)
        if half_size:
            max_pos = max_pos / 2
        if max_pos < 20:
            continue

        n_entered = 0
        for tk in picks:
            if tk not in sc.columns or tk not in atr_dict:
                continue
            if n_entered >= 3:
                break

            S = float(sc[tk].loc[dt])
            di = sc.index.get_loc(dt)

            # Momentum acceleration filter (H)
            if mom_accel_filter and di > 63:
                mom_21d = float(sc[tk].iloc[di] / sc[tk].iloc[di - 21] - 1)
                mom_63d = float(sc[tk].iloc[di] / sc[tk].iloc[di - 63] - 1)
                # Only enter if 21d momentum is worse (more negative) than 63d/3
                # = accelerating downward
                if mom_21d >= mom_63d / 3:
                    continue

            # Get ATR
            if dt in atr_dict[tk].index and not pd.isna(atr_dict[tk].loc[dt]):
                atr_val = float(atr_dict[tk].loc[dt])
            else:
                atr_val = S * 0.015

            # Bear put spread strikes: buy ATM put, sell OTM put
            K2 = round(S, 0)                        # Higher strike = long put (ATM)
            K1 = round(S * (1 - SPREAD_PCT / 100), 0)  # Lower strike = short put (OTM)
            if K1 >= K2:
                continue

            # Price using standardized pricer
            try:
                entry_cost, max_profit = price_bear_put_spread(
                    S=S, K1=K1, K2=K2, dte=DTE, atr=atr_val, vix=cv,
                    haircut=DEFAULT_HAIRCUT
                )
            except (ValueError, Exception):
                continue

            # Per-contract costs
            cost_per_contract = entry_cost * 100 + COMMISSION_RT_SPREAD
            max_loss = cost_per_contract  # max you can lose = debit paid
            max_prof_dollar = max_profit * 100 - COMMISSION_RT_SPREAD

            if cost_per_contract <= 0 or cost_per_contract > max_pos or cost_per_contract > equity * 0.40:
                continue
            if max_prof_dollar <= 0:
                continue

            # Simulate hold period
            pnl = None
            exit_idx = di
            exit_reason = 'expiry'

            for ci in range(di + 1, min(di + DTE + 1, len(sc))):
                Sc = float(sc[tk].iloc[ci])

                # At expiry or last day: intrinsic value
                if ci == di + DTE or ci == len(sc) - 1:
                    intrinsic = max(0, K2 - Sc) - max(0, K1 - Sc)
                    pnl = intrinsic * 100 - cost_per_contract
                    exit_idx = ci
                    exit_reason = 'expiry'
                    break

                # Mid-life checks for early exit
                remaining_dte = DTE - (ci - di)
                intrinsic = max(0, K2 - Sc) - max(0, K1 - Sc)
                # Simple time value decay estimate
                time_frac = np.sqrt(remaining_dte / DTE)
                current_val = intrinsic * 100 + atr_val * time_frac * 30  # rough time value

                unrealized_pnl = current_val - cost_per_contract

                # Take profit at 50% of max profit
                if unrealized_pnl >= max_prof_dollar * 0.50:
                    pnl = unrealized_pnl
                    exit_idx = ci
                    exit_reason = 'take_profit'
                    break

                # Tight stop (B): exit if losing 50% of max loss
                if use_tight_stops and unrealized_pnl <= -max_loss * 0.50:
                    pnl = unrealized_pnl
                    exit_idx = ci
                    exit_reason = 'stop_loss'
                    break

            if pnl is None:
                # Fallback: expiry value
                Se = float(sc[tk].iloc[min(di + DTE, len(sc) - 1)])
                intrinsic = max(0, K2 - Se) - max(0, K1 - Se)
                pnl = intrinsic * 100 - cost_per_contract
                exit_idx = min(di + DTE, len(sc) - 1)

            equity += pnl
            n_entered += 1

            trades.append({
                'pnl': round(pnl, 2),
                'entry_date': str(dt.date()),
                'exit_date': str(sc.index[exit_idx].date()),
                'ticker': tk,
                'side': 'bear',
                'exit_reason': exit_reason,
                'vix_at_entry': round(cv, 1),
            })

            eq_dates.append(sc.index[exit_idx])
            eq_values.append(equity)

    return trades, equity


def simulate_bull(
    rankings,
    sc, sh, sl, spy, vix,
    atr_dict,
    regime_scores=None,
    regime_threshold=0.4,
):
    """Simulate bull call spread strategy (regime > threshold)."""
    equity = CAP
    trades = []

    for dt in sorted(rankings.keys()):
        if dt not in spy.index or dt not in vix.index:
            continue

        cv = float(vix.loc[dt])

        # Regime gate: only trade when regime > threshold
        if regime_scores is not None and dt in regime_scores.index:
            rs = float(regime_scores.loc[dt])
            if rs <= regime_threshold:
                continue
        else:
            # Fallback: VIX > 20 as proxy for elevated vol
            if cv <= 20:
                continue

        scores = rankings[dt]
        if not scores:
            continue

        # TOP ranked sectors (best expected performance)
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        picks = [t for t, _ in ranked[:3]]

        max_pos = min(200, equity / 3)
        if max_pos < 20:
            continue

        n_entered = 0
        for tk in picks:
            if tk not in sc.columns or tk not in atr_dict:
                continue
            if n_entered >= 3:
                break

            S = float(sc[tk].loc[dt])
            di = sc.index.get_loc(dt)

            if dt in atr_dict[tk].index and not pd.isna(atr_dict[tk].loc[dt]):
                atr_val = float(atr_dict[tk].loc[dt])
            else:
                atr_val = S * 0.015

            # Bull call spread: buy ATM call, sell OTM call
            K1 = round(S, 0)
            K2 = round(S * (1 + SPREAD_PCT / 100), 0)
            if K2 <= K1:
                continue

            try:
                entry_cost, max_profit = price_bull_call_spread(
                    S=S, K1=K1, K2=K2, dte=DTE, atr=atr_val, vix=cv,
                    haircut=DEFAULT_HAIRCUT
                )
            except (ValueError, Exception):
                continue

            cost_per_contract = entry_cost * 100 + COMMISSION_RT_SPREAD
            max_prof_dollar = max_profit * 100 - COMMISSION_RT_SPREAD

            if cost_per_contract <= 0 or cost_per_contract > max_pos or cost_per_contract > equity * 0.40:
                continue
            if max_prof_dollar <= 0:
                continue

            # Hold to expiry
            ei = min(di + DTE, len(sc) - 1)
            Se = float(sc[tk].iloc[ei])
            intrinsic = max(0, Se - K1) - max(0, Se - K2)
            pnl = intrinsic * 100 - cost_per_contract

            equity += pnl
            n_entered += 1

            trades.append({
                'pnl': round(pnl, 2),
                'entry_date': str(dt.date()),
                'exit_date': str(sc.index[ei].date()),
                'ticker': tk,
                'side': 'bull',
                'vix_at_entry': round(cv, 1),
            })

    return trades, equity


# ═══════════════════════════════════════════════════════════════════
# 6. MAIN EXECUTION
# ═══════════════════════════════════════════════════════════════════

def run_variant(name, trades, spy_prices, label=""):
    """Validate and report results for a variant."""
    fprint(f"\n{'='*60}")
    fprint(f"  VARIANT {name}: {label}")
    fprint(f"{'='*60}")

    if len(trades) < 10:
        fprint(f"  Only {len(trades)} trades — insufficient for validation")
        return {
            'name': name, 'label': label, 'n_trades': len(trades),
            'sharpe': 0, 'sortino': 0, 'mdd': -1, 'wr': 0, 'pf': 0,
            'final_equity': CAP, 'gates_passed': 0, 'gates_total': 5,
            'all_passed': False, 'error': 'too_few_trades',
        }

    result = validate_trades(
        trades=trades,
        initial_capital=CAP,
        spy_prices=spy_prices,
        strategy_name=f"Bear Improvement {name}",
        n_perms=2000,
    )
    result.print_summary()

    return {
        'name': name,
        'label': label,
        'n_trades': result.n_trades,
        'sharpe': round(result.sharpe, 3),
        'sortino': round(result.sortino, 3),
        'mdd': round(result.max_dd, 4),
        'wr': round(result.win_rate, 4),
        'pf': round(result.profit_factor, 3),
        'cagr': round(result.cagr, 4),
        'final_equity': round(result.final_equity, 2),
        'gates_passed': result.gates_passed,
        'gates_total': result.gates_total,
        'all_passed': result.all_passed,
        'gates': [g.name + (':PASS' if g.passed else ':FAIL') for g in result.gates],
    }


def main():
    t0 = time.time()

    # 1. Download data
    sc, sh, sl, sv, spy, vix, extras = download_data()

    # 2. Load regime predictions
    regime_scores = load_regime_predictions()
    if regime_scores is None:
        fprint("  Using VIX proxy for regime scores")
        regime_scores = vix_proxy_regime(vix)

    # 3. Build LGBM rankings
    rankings = build_lgbm_rankings(sc, spy, vix)
    if len(rankings) < 50:
        fprint("FATAL: Too few ranking dates. Aborting.")
        return

    # 4. Pre-compute ATRs
    fprint("[4/6] Computing ATR series...")
    atr_dict = build_atr_dict(sc, sh, sl)

    # 5. Run all 8 variants
    fprint("\n[5/6] Running 8 bear side variants...")

    results = []

    # A. Baseline bear puts
    trades_a, _ = simulate_bear(
        "A", rankings, sc, sh, sl, spy, vix, atr_dict,
        regime_scores=regime_scores,
        bottom_k=3, vix_max=20.0,
    )
    results.append(run_variant("A", trades_a, spy, "Baseline bear puts (VIX<20, bottom 3)"))

    # B. Tighter stops
    trades_b, _ = simulate_bear(
        "B", rankings, sc, sh, sl, spy, vix, atr_dict,
        regime_scores=regime_scores,
        bottom_k=3, vix_max=20.0, use_tight_stops=True,
    )
    results.append(run_variant("B", trades_b, spy, "Tighter stops (50% max loss exit)"))

    # C. VIX term structure filter
    trades_c, _ = simulate_bear(
        "C", rankings, sc, sh, sl, spy, vix, atr_dict,
        regime_scores=regime_scores,
        bottom_k=3, vix_max=20.0, vix_term_filter=True,
    )
    results.append(run_variant("C", trades_c, spy, "VIX term structure filter (5d change > 2)"))

    # D. Defensive sector exclusion
    trades_d, _ = simulate_bear(
        "D", rankings, sc, sh, sl, spy, vix, atr_dict,
        regime_scores=regime_scores,
        bottom_k=3, vix_max=20.0, exclude_defensive=True,
    )
    results.append(run_variant("D", trades_d, spy, "Defensive exclusion (no XLU/XLP/XLRE)"))

    # E. Regime-gated bears (regime < 0.2 only)
    trades_e, _ = simulate_bear(
        "E", rankings, sc, sh, sl, spy, vix, atr_dict,
        regime_scores=regime_scores,
        bottom_k=3, vix_max=None,  # No VIX filter, regime does the gating
        regime_gate=0.2,
    )
    results.append(run_variant("E", trades_e, spy, "Regime-gated (regime < 0.2 only)"))

    # F. Size reduction (half positions)
    trades_f, _ = simulate_bear(
        "F", rankings, sc, sh, sl, spy, vix, atr_dict,
        regime_scores=regime_scores,
        bottom_k=3, vix_max=20.0, half_size=True,
    )
    results.append(run_variant("F", trades_f, spy, "Half-size bears"))

    # G. Bear + Bull combined portfolio
    fprint("\n  Running combined bull+bear for variant G...")
    trades_bull, _ = simulate_bull(
        rankings, sc, sh, sl, spy, vix, atr_dict,
        regime_scores=regime_scores, regime_threshold=0.4,
    )
    trades_bear_g, _ = simulate_bear(
        "G_bear", rankings, sc, sh, sl, spy, vix, atr_dict,
        regime_scores=regime_scores,
        bottom_k=3, vix_max=None, regime_gate=0.2,
    )
    # Combine trades on a single equity curve
    combined_trades = trades_bull + trades_bear_g
    combined_trades.sort(key=lambda x: x['entry_date'])
    results.append(run_variant("G", combined_trades, spy, "Bear+Bull combined (regime>0.4 bull, regime<0.2 bear)"))

    # H. Momentum acceleration filter
    trades_h, _ = simulate_bear(
        "H", rankings, sc, sh, sl, spy, vix, atr_dict,
        regime_scores=regime_scores,
        bottom_k=3, vix_max=20.0, mom_accel_filter=True,
    )
    results.append(run_variant("H", trades_h, spy, "Momentum acceleration (21d < 63d/3)"))

    # 6. Summary + MLflow logging
    fprint("\n[6/6] Summary and logging...")
    fprint("\n" + "=" * 80)
    fprint(f"{'Var':<4} {'Label':<45} {'N':>5} {'Sharpe':>7} {'MDD':>8} {'WR':>6} {'PF':>6} {'Gates':>6}")
    fprint("-" * 80)
    for r in results:
        mdd_str = f"{r['mdd']*100:.1f}%" if r['mdd'] != -1 else "N/A"
        fprint(f"{r['name']:<4} {r['label'][:44]:<45} {r['n_trades']:>5} {r['sharpe']:>7.2f} "
               f"{mdd_str:>8} {r['wr']*100:>5.1f}% {r['pf']:>6.2f} {r['gates_passed']}/{r['gates_total']}")
    fprint("=" * 80)

    # Best variant by Sharpe (excluding those with too few trades)
    valid = [r for r in results if r['n_trades'] >= 10]
    if valid:
        best = max(valid, key=lambda x: x['sharpe'])
        fprint(f"\nBest by Sharpe: Variant {best['name']} — Sharpe {best['sharpe']:.2f}, "
               f"MDD {best['mdd']*100:.1f}%, {best['n_trades']} trades")

        # Best by MDD (least negative)
        best_mdd = max(valid, key=lambda x: x['mdd'])
        fprint(f"Best by MDD:    Variant {best_mdd['name']} — MDD {best_mdd['mdd']*100:.1f}%, "
               f"Sharpe {best_mdd['sharpe']:.2f}")

    # Save results
    output = {
        'timestamp': datetime.now().isoformat(),
        'variants': results,
        'config': {
            'capital': CAP, 'dte': DTE, 'spread_pct': SPREAD_PCT,
            'haircut': DEFAULT_HAIRCUT, 'commission': COMMISSION_RT_SPREAD,
            'features': FEATURE_COLS,
        },
    }
    with open(RESULTS_PATH, 'w') as f:
        json.dump(output, f, indent=2)
    fprint(f"\nResults saved to {RESULTS_PATH}")

    # MLflow logging
    if MLFLOW_OK:
        try:
            exp = mlflow.set_experiment("bear_side_improvement_v1")
            with mlflow.start_run(run_name=f"bear_improvement_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
                mlflow.log_param("capital", CAP)
                mlflow.log_param("dte", DTE)
                mlflow.log_param("spread_pct", SPREAD_PCT)
                mlflow.log_param("n_variants", len(results))
                mlflow.log_param("n_ranking_dates", len(rankings))

                for r in results:
                    prefix = f"v{r['name']}_"
                    mlflow.log_metric(f"{prefix}sharpe", r['sharpe'])
                    mlflow.log_metric(f"{prefix}sortino", r['sortino'])
                    mlflow.log_metric(f"{prefix}mdd", r['mdd'])
                    mlflow.log_metric(f"{prefix}wr", r['wr'])
                    mlflow.log_metric(f"{prefix}pf", r['pf'])
                    mlflow.log_metric(f"{prefix}n_trades", r['n_trades'])
                    mlflow.log_metric(f"{prefix}gates_passed", r['gates_passed'])

                if valid:
                    best = max(valid, key=lambda x: x['sharpe'])
                    mlflow.log_metric("best_sharpe", best['sharpe'])
                    mlflow.log_param("best_variant", best['name'])

                mlflow.log_artifact(str(RESULTS_PATH))
            fprint("MLflow run logged successfully")
        except Exception as e:
            fprint(f"MLflow logging failed: {e}")

    elapsed = time.time() - t0
    fprint(f"\nTotal runtime: {elapsed:.0f}s ({elapsed/60:.1f}m)")


if __name__ == "__main__":
    main()
