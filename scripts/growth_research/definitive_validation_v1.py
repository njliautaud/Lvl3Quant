#!/usr/bin/env python3
"""Definitive Adversarial Validation v1 — Sector ETF Bull Call Spreads.

PURPOSE: Settle the Sharpe discrepancy between sensitivity_analysis_v2 (1.49)
and production_sector_v3 (4.73). The gap comes from the early exit mechanic:
v3 assumes you can sell the spread mid-life at theoretical value with NO
bid-ask haircut on exit. This script tests variants with and without exit costs.

VARIANTS:
  A: Hold to expiry (30d), no early exit — MOST CONSERVATIVE
  B: Early exit at 20d, WITH 15% exit haircut (honest)
  C: Early exit at 20d, WITHOUT exit haircut (v3 method, for comparison)
  D: Random sector selection (not LGBM), hold to expiry — structural edge test
  E: Random entry dates (shuffle dates), LGBM ranking — timing test

TESTS:
  1. Permutation test (2000 sign-flip trials)
  2. Regime split (bull=SPY>200d SMA, bear=below)
  3. Sub-period stability (first half vs second half)
  4. Yearly Sharpe breakdown
  5. Random control comparison

COSTS: $0.65/leg * 4 legs = $2.60/spread RT. 15% haircut on ATR pricing.
CAPITAL: $645 starting. Fixed position sizing (max $200/trade, max 40% equity).
"""
import numpy as np, pandas as pd, warnings, sys
warnings.filterwarnings('ignore')
from pathlib import Path
from datetime import datetime
from scipy import stats
import lightgbm as lgb

def fprint(*a, **kw): print(*a, **kw, flush=True)

# ============================================================
# CONSTANTS — all in one place, nothing hidden
# ============================================================
SECTORS = ['XLB', 'XLC', 'XLE', 'XLF', 'XLI', 'XLK', 'XLP', 'XLRE', 'XLU', 'XLV', 'XLY']
CAP = 645.0
LEG_COMM = 0.65
SPREAD_COMM = 4 * LEG_COMM  # $2.60 RT
HAIRCUT = 0.15              # 15% on ATR-based pricing
SPREAD_PCT = 3.0            # 3% OTM for short leg
DTE = 30                    # 30 calendar days to expiry
TOP_K = 3                   # top 3 sectors per rebalance
VIX_MIN = 20                # only enter when VIX > 20
N_PERM = 2000               # permutation trials

# Feature columns (quality-momentum, same as v3)
QM_COLS = ['ret_5d', 'ret_10d', 'ret_21d', 'ret_63d', 'ret_126d', 'ret_252d',
           'vol_21d', 'vol_63d', 'sharpe_63d', 'maxdd_63d', 'pct_52w_high', 'mom_accel',
           'pct_pos_months_12m', 'sortino_63d', 'calmar_1y', 'up_capture', 'dn_capture',
           'trend_r2_63d', 'trend_slope_63d', 'rel_vol_21d']


# ============================================================
# DATA DOWNLOAD
# ============================================================
def download_data():
    import yfinance as yf
    fprint("Downloading data...")
    tickers = SECTORS + ['SPY', '^VIX']
    raw = yf.download(tickers, start='2008-01-01', end='2026-07-26', progress=False)
    mi = isinstance(raw.columns, pd.MultiIndex)
    close = raw['Close'] if mi else raw
    high = raw['High'] if mi else raw
    low = raw['Low'] if mi else raw
    volume = raw['Volume'] if mi else raw

    vc = '^VIX' if '^VIX' in close.columns else 'VIX'
    vix = close[vc].dropna()
    spy = close['SPY'].dropna()

    avail = [c for c in SECTORS if c in close.columns]
    sc = close[avail].dropna(how='all')
    sh = high[[c for c in avail if c in high.columns]].dropna(how='all')
    sl = low[[c for c in avail if c in low.columns]].dropna(how='all')
    sv = volume[[c for c in avail if c in volume.columns]].dropna(how='all')

    ix = sc.index.intersection(vix.index).intersection(spy.index)
    ix = ix.intersection(sh.index).intersection(sl.index)

    fprint(f"  Data: {len(ix)} trading days, {len(avail)} sectors")
    fprint(f"  Date range: {ix[0].date()} to {ix[-1].date()}")
    return sc.loc[ix], sh.loc[ix], sl.loc[ix], sv.reindex(ix), spy.loc[ix], vix.loc[ix]


# ============================================================
# FEATURE ENGINEERING (copied from production_sector_v3.py)
# ============================================================
def compute_features(px, vol_data=None):
    """Compute 20 quality-momentum features for one ticker up to a date."""
    if len(px) < 260:
        return None
    f = {}
    # Momentum returns
    for lb, nm in [(5,'ret_5d'),(10,'ret_10d'),(21,'ret_21d'),(63,'ret_63d'),
                   (126,'ret_126d'),(252,'ret_252d')]:
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

    # Quality
    monthly = rets.resample('ME').sum()
    f['pct_pos_months_12m'] = float((monthly.iloc[-12:] > 0).mean()) if len(monthly) >= 12 else 0.5

    dr = r63[r63 < 0]
    f['sortino_63d'] = float(r63.mean() / (dr.std() + 1e-10) * np.sqrt(252)) if len(dr) > 3 else 0.0

    pk = px.iloc[-252:].cummax()
    mdd = float(((px.iloc[-252:] / pk) - 1).min())
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
        f['trend_slope_63d'] = slope * 252
    else:
        f['trend_r2_63d'] = 0.0
        f['trend_slope_63d'] = 0.0

    f['rel_vol_21d'] = 1.0
    if vol_data is not None and len(vol_data) >= 63:
        f['rel_vol_21d'] = float(vol_data.iloc[-21:].mean() / (vol_data.iloc[-63:].mean() + 1e-10))

    return f


# ============================================================
# LGBM RANKING (same walk-forward as v3)
# ============================================================
def build_rankings(sc, sv, rebal_dates):
    """Walk-forward LGBM ranking: train on 12 prior periods, predict next."""
    fprint("  Building LGBM rankings (20 features, 12-period lookback)...")
    records = []
    for dt in rebal_dates:
        idx = sc.index.get_indexer([dt], method='ffill')[0]
        if idx < 260:
            continue
        for tk in sc.columns:
            px = sc[tk].iloc[:idx + 1].dropna()
            vol_d = sv[tk].iloc[:idx + 1] if tk in sv.columns else None
            feats = compute_features(px, vol_d)
            if feats is None:
                continue
            fi = min(idx + 14, len(sc) - 1)
            fwd_ret = float(sc[tk].iloc[fi] / sc[tk].iloc[idx] - 1)
            feats['date'] = dt
            feats['ticker'] = tk
            feats['fwd_ret'] = fwd_ret
            records.append(feats)

    df = pd.DataFrame(records)
    for c in QM_COLS:
        if c not in df.columns:
            df[c] = 0.0
    df[QM_COLS] = df[QM_COLS].fillna(0.0)

    if len(df) < 100:
        fprint("    FATAL: not enough data for LGBM")
        return {}

    df['rank_label'] = df.groupby('date')['fwd_ret'].rank(pct=True)
    dates = sorted(df['date'].unique())
    rankings = {}

    for i in range(12, len(dates)):
        train_dates = dates[max(0, i - 12):i]
        test_date = dates[i]
        tr = df[df['date'].isin(train_dates)]
        te = df[df['date'] == test_date].copy()
        if len(te) < 3 or len(tr) < 50:
            continue

        Xt = np.nan_to_num(tr[QM_COLS].values.astype(np.float32))
        yt = tr['rank_label'].values.astype(np.float32)
        Xe = np.nan_to_num(te[QM_COLS].values.astype(np.float32))

        try:
            m = lgb.LGBMRegressor(n_estimators=100, max_depth=4, learning_rate=0.05,
                                  subsample=0.8, colsample_bytree=0.8,
                                  min_child_samples=5, verbose=-1)
            m.fit(Xt, yt)
            te['score'] = m.predict(Xe)
            rankings[test_date] = dict(zip(te['ticker'], te['score']))
        except Exception:
            continue

    fprint(f"    {len(rankings)} ranking dates produced")
    return rankings


# ============================================================
# OPTIONS PRICING (ATR-based, same as v3)
# ============================================================
def compute_atr(h, l, c, period=14):
    tr = pd.DataFrame({
        'hl': h - l,
        'hc': abs(h - c.shift(1)),
        'lc': abs(l - c.shift(1))
    }).max(axis=1)
    return tr.rolling(period).mean()


def atr_premium(S, K, dte, atr, vix_val, opt='call'):
    """ATR-based option premium estimate."""
    T = dte / 252.0
    if T <= 0:
        return max(0, S - K) if opt == 'call' else max(0, K - S)
    intrinsic = max(0, S - K) if opt == 'call' else max(0, K - S)
    vol_factor = max(0.3, vix_val / 20.0)
    time_component = atr * np.sqrt(T) * vol_factor * np.exp(-3.0 * abs(S - K) / S)
    return intrinsic + time_component


# ============================================================
# SIMULATION — deliberately simple, one function, no tricks
# ============================================================
def simulate_variant(variant_name, rankings, sc, sh, sl, spy, vix,
                     early_exit_day=None, apply_exit_haircut=False):
    """
    Simulate bull call spread strategy.

    Args:
        variant_name: label for this variant
        rankings: {date: {ticker: score}} from LGBM or random
        early_exit_day: if set, exit after N trading days (sell spread back)
        apply_exit_haircut: if True, apply 15% haircut when SELLING spread back
    """
    sma200 = spy.rolling(200).mean()
    atr_d = {}
    for tk in sc.columns:
        if tk in sh.columns and tk in sl.columns:
            atr_d[tk] = compute_atr(sh[tk], sl[tk], sc[tk])

    equity = CAP
    trades = []
    eq_curve = [CAP]

    for dt in sorted(rankings.keys()):
        if dt not in spy.index or dt not in vix.index:
            continue

        cv = float(vix.loc[dt])
        if cv < VIX_MIN:
            continue

        sv_val = float(spy.loc[dt])
        sm = float(sma200.loc[dt]) if dt in sma200.index and not pd.isna(sma200.loc[dt]) else sv_val
        bull = sv_val >= sm

        scores = rankings[dt]
        if not scores:
            continue

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        picks = [t for t, _ in ranked[:TOP_K]]

        max_pos = min(200, equity / 3)
        if max_pos < 30:
            eq_curve.append(equity)
            continue

        n_entered = 0
        for tk in picks:
            if tk not in sc.columns or tk not in atr_d or n_entered >= 3:
                continue

            S = float(sc[tk].loc[dt])
            if np.isnan(S) or S <= 0:
                continue
            di = sc.index.get_loc(dt)

            # ATR for pricing
            if dt in atr_d[tk].index and not pd.isna(atr_d[tk].loc[dt]):
                av = float(atr_d[tk].loc[dt])
            else:
                av = S * 0.015

            # Strikes
            K1 = round(S)                          # ATM long call
            K2 = round(S * (1 + SPREAD_PCT / 100)) # OTM short call

            # ENTRY pricing: buy long call (pay more due to haircut), sell short call (receive less)
            long_premium = atr_premium(S, K1, DTE, av, cv, 'call') * (1 + HAIRCUT)
            short_premium = atr_premium(S, K2, DTE, av, cv, 'call') * (1 - HAIRCUT)
            net_debit = long_premium - short_premium  # per-share debit

            entry_cost = net_debit * 100 + SPREAD_COMM  # total $ cost to enter
            width = K2 - K1
            max_profit = (width - net_debit) * 100 - SPREAD_COMM

            # Skip if cost is negative, too large, or exceeds risk limits
            if entry_cost <= 0 or entry_cost > max_pos or entry_cost > equity * 0.40:
                continue

            # Determine exit index
            if early_exit_day is not None:
                exit_idx = min(di + early_exit_day, len(sc) - 1)
            else:
                exit_idx = min(di + DTE, len(sc) - 1)

            # Compute P&L at exit
            Se = float(sc[tk].iloc[exit_idx])
            days_held = exit_idx - di
            remaining_dte = max(0, DTE - days_held)

            if remaining_dte > 0 and early_exit_day is not None:
                # EARLY EXIT: sell the spread back at theoretical value
                # We close by selling long call + buying back short call
                exit_atr = av
                if exit_idx < len(atr_d[tk]):
                    a_val = atr_d[tk].iloc[exit_idx]
                    if not pd.isna(a_val):
                        exit_atr = float(a_val)

                exit_vix = cv
                if sc.index[exit_idx] in vix.index:
                    exit_vix = float(vix.loc[sc.index[exit_idx]])

                # Theoretical mid-market value of the spread at exit
                long_exit_val = atr_premium(Se, K1, remaining_dte, exit_atr, exit_vix, 'call')
                short_exit_val = atr_premium(Se, K2, remaining_dte, exit_atr, exit_vix, 'call')
                theo_spread_value = long_exit_val - short_exit_val  # per-share

                if apply_exit_haircut:
                    # When CLOSING a spread, you receive less than theoretical.
                    # Apply a single 15% haircut to the NET spread value.
                    # This is the same logic as entry: market makers take their cut.
                    theo_spread_value *= (1 - HAIRCUT)

                spread_exit_value = theo_spread_value * 100

                # P&L = exit proceeds - entry cost - exit commission
                pnl = spread_exit_value - entry_cost - SPREAD_COMM  # extra commission to close

            else:
                # HOLD TO EXPIRY: intrinsic value only
                intrinsic = (max(0, Se - K1) - max(0, Se - K2)) * 100
                pnl = intrinsic - entry_cost

            equity += pnl
            n_entered += 1
            exit_date = sc.index[exit_idx]

            trades.append({
                'entry': str(dt.date()),
                'exit': str(exit_date.date()),
                'ticker': tk,
                'pnl': round(pnl, 2),
                'win': pnl > 0,
                'hold_days': days_held,
                'regime': 'bull' if bull else 'bear',
                'vix': round(cv, 1),
                'equity_after': round(equity, 2),
                'entry_cost': round(entry_cost, 2),
                'year': dt.year,
            })

        eq_curve.append(equity)

    return trades, equity, eq_curve


# ============================================================
# METRICS — honest equity-based Sharpe
# ============================================================
def compute_monthly_returns(trades):
    """Convert trade-level P&L into calendar-month equity-based returns."""
    if not trades:
        return np.array([]), pd.DataFrame()

    tdf = pd.DataFrame(trades)
    tdf['entry_dt'] = pd.to_datetime(tdf['entry'])
    tdf['month'] = tdf['entry_dt'].dt.to_period('M')

    monthly = []
    for mo in sorted(tdf['month'].unique()):
        mo_trades = tdf[tdf['month'] == mo]
        mo_pnl = mo_trades['pnl'].sum()
        # Equity at START of month = equity after last trade minus that trade's P&L contribution
        eq_start = mo_trades['equity_after'].iloc[0] - mo_trades['pnl'].iloc[0]
        eq_start = max(eq_start, 100)  # floor
        mo_return = mo_pnl / eq_start
        monthly.append({
            'month': str(mo),
            'pnl': round(mo_pnl, 2),
            'equity_start': round(eq_start, 2),
            'return': round(mo_return, 4),
        })

    mdf = pd.DataFrame(monthly)
    rets = mdf['return'].values if len(mdf) > 0 else np.array([])
    return rets, mdf


def compute_all_metrics(trades, final_eq, eq_curve, name):
    """Compute comprehensive metrics for a variant."""
    if not trades:
        fprint(f"  {name}: NO TRADES")
        return None

    n = len(trades)
    wins = sum(1 for t in trades if t['win'])
    wr = wins / n * 100
    pnls = [t['pnl'] for t in trades]

    # Monthly returns
    monthly_rets, mdf = compute_monthly_returns(trades)
    if len(monthly_rets) < 4:
        fprint(f"  {name}: Only {len(monthly_rets)} months, too few for Sharpe")
        return None

    # Sharpe and Sortino (annualized from monthly)
    sharpe = float(np.mean(monthly_rets) / (np.std(monthly_rets) + 1e-10) * np.sqrt(12))
    dn_rets = monthly_rets[monthly_rets < 0]
    sortino = float(np.mean(monthly_rets) / (np.std(dn_rets) + 1e-10) * np.sqrt(12)) if len(dn_rets) > 1 else 0.0

    # CAGR
    ny = max(len(monthly_rets) / 12, 0.5)
    cagr = (final_eq / CAP) ** (1 / ny) - 1

    # Max drawdown from equity curve
    eq = np.array(eq_curve)
    pk = np.maximum.accumulate(eq)
    mdd = float(((eq - pk) / (pk + 1e-10)).min())

    # Profit factor
    gp = sum(p for p in pnls if p > 0)
    gl = abs(sum(p for p in pnls if p <= 0))
    pf = gp / (gl + 1e-10)

    # Regime split
    bull_trades = [t for t in trades if t['regime'] == 'bull']
    bear_trades = [t for t in trades if t['regime'] == 'bear']
    bull_wr = sum(1 for t in bull_trades if t['win']) / max(len(bull_trades), 1) * 100
    bear_wr = sum(1 for t in bear_trades if t['win']) / max(len(bear_trades), 1) * 100
    regime_gap = abs(bull_wr - bear_wr) / max(bull_wr, bear_wr, 1)

    # Average hold
    avg_hold = np.mean([t['hold_days'] for t in trades])
    avg_cost = np.mean([t['entry_cost'] for t in trades])

    return {
        'name': name,
        'n_trades': n,
        'win_rate': round(wr, 1),
        'sharpe': round(sharpe, 2),
        'sortino': round(sortino, 2),
        'cagr_pct': round(cagr * 100, 1),
        'maxdd_pct': round(mdd * 100, 1),
        'pf': round(pf, 2),
        'avg_pnl': round(np.mean(pnls), 2),
        'avg_hold': round(avg_hold, 1),
        'avg_cost': round(avg_cost, 2),
        'final_equity': round(final_eq, 2),
        'bull_n': len(bull_trades),
        'bear_n': len(bear_trades),
        'bull_wr': round(bull_wr, 1),
        'bear_wr': round(bear_wr, 1),
        'regime_gap': round(regime_gap, 3),
        'monthly_rets': monthly_rets,
        'monthly_df': mdf,
    }


# ============================================================
# STATISTICAL TESTS
# ============================================================
def permutation_test(monthly_rets, n_perm=N_PERM):
    """Sign-flip permutation test on monthly returns."""
    if len(monthly_rets) < 10:
        return 1.0
    observed = np.mean(monthly_rets) / (np.std(monthly_rets) + 1e-10)
    count = 0
    np.random.seed(42)
    for _ in range(n_perm):
        signs = np.random.choice([-1, 1], len(monthly_rets))
        shuffled = monthly_rets * signs
        stat = np.mean(shuffled) / (np.std(shuffled) + 1e-10)
        if stat >= observed:
            count += 1
    return count / n_perm


def sub_period_test(monthly_rets):
    """Split in half, check both halves are positive Sharpe."""
    mid = len(monthly_rets) // 2
    h1 = monthly_rets[:mid]
    h2 = monthly_rets[mid:]
    h1_sh = float(np.mean(h1) / (np.std(h1) + 1e-10) * np.sqrt(12)) if len(h1) > 3 else 0.0
    h2_sh = float(np.mean(h2) / (np.std(h2) + 1e-10) * np.sqrt(12)) if len(h2) > 3 else 0.0
    return h1_sh, h2_sh, h1_sh > 0 and h2_sh > 0


def yearly_sharpe(trades):
    """Compute Sharpe by year."""
    if not trades:
        return {}
    tdf = pd.DataFrame(trades)
    tdf['entry_dt'] = pd.to_datetime(tdf['entry'])
    tdf['year'] = tdf['entry_dt'].dt.year
    result = {}
    for yr in sorted(tdf['year'].unique()):
        yr_trades = tdf[tdf['year'] == yr]
        yr_pnls = yr_trades['pnl'].values
        if len(yr_pnls) < 3:
            result[yr] = {'sharpe': 0.0, 'n': len(yr_pnls), 'wr': 0.0, 'pnl': 0.0}
            continue
        # Simple: annualize from per-trade stats (since no monthly breakdown per year is easy)
        # Use per-trade Sharpe * sqrt(trades_per_year) as approximation
        mean_r = np.mean(yr_pnls)
        std_r = np.std(yr_pnls) + 1e-10
        n = len(yr_pnls)
        wr = sum(1 for p in yr_pnls if p > 0) / n * 100
        total_pnl = sum(yr_pnls)
        # Monthly-ish: group by month within this year
        yr_trades_copy = yr_trades.copy()
        yr_trades_copy['month'] = yr_trades_copy['entry_dt'].dt.to_period('M')
        mo_pnls = yr_trades_copy.groupby('month')['pnl'].sum().values
        if len(mo_pnls) >= 3:
            sh = float(np.mean(mo_pnls) / (np.std(mo_pnls) + 1e-10) * np.sqrt(12))
        else:
            sh = 0.0
        result[yr] = {'sharpe': round(sh, 2), 'n': n, 'wr': round(wr, 1), 'pnl': round(total_pnl, 2)}
    return result


# ============================================================
# MAIN
# ============================================================
def main():
    t0 = datetime.now()
    fprint("=" * 90)
    fprint("DEFINITIVE ADVERSARIAL VALIDATION v1 — Sector ETF Bull Call Spreads")
    fprint("=" * 90)
    fprint(f"Date: {t0.strftime('%Y-%m-%d %H:%M:%S')}")
    fprint(f"Capital: ${CAP} | Commission: ${SPREAD_COMM}/spread | Haircut: {HAIRCUT:.0%}")
    fprint(f"Universe: {len(SECTORS)} sector ETFs | VIX filter: >{VIX_MIN}")
    fprint(f"Permutation trials: {N_PERM}")
    fprint()

    # ---- Step 1: Download data ----
    sc, sh, sl, sv, spy, vix = download_data()

    # ---- Step 2: Build LGBM rankings ----
    fprint("\nStep 2: Building LGBM rankings...")
    bd = pd.DatetimeIndex(sc.index.to_series().resample('2W-FRI').last().dropna().values)
    rankings = build_rankings(sc, sv, bd)
    if not rankings:
        fprint("FATAL: No rankings produced. Exiting.")
        return
    fprint(f"  Total rebalance dates with rankings: {len(rankings)}")
    fprint(f"  Date range: {min(rankings.keys()).date()} to {max(rankings.keys()).date()}")

    # ---- Step 3: Build random rankings (for variant D) ----
    fprint("\nStep 3: Building random rankings for control...")
    np.random.seed(42)
    random_rankings = {}
    for dt, scores in rankings.items():
        random_rankings[dt] = {tk: np.random.random() for tk in scores.keys()}
    fprint(f"  {len(random_rankings)} random ranking dates")

    # ---- Step 4: Build shuffled-date rankings (for variant E) ----
    fprint("\nStep 4: Building shuffled-date rankings for timing test...")
    all_dates = sorted(rankings.keys())
    shuffled_dates = list(all_dates)
    np.random.seed(123)
    np.random.shuffle(shuffled_dates)
    shuffled_rankings = {}
    for orig_dt, shuf_dt in zip(all_dates, shuffled_dates):
        shuffled_rankings[orig_dt] = rankings[shuf_dt]
    fprint(f"  {len(shuffled_rankings)} shuffled ranking dates")

    # ---- Step 5: Run all 5 variants ----
    fprint("\n" + "=" * 90)
    fprint("Step 5: SIMULATING ALL VARIANTS")
    fprint("=" * 90)

    variants = [
        ('A: Hold-to-Expiry (30d)',     rankings,          None,  False),
        ('B: Exit-20d WITH haircut',    rankings,          20,    True),
        ('C: Exit-20d NO haircut',      rankings,          20,    False),
        ('D: RANDOM sectors, expiry',   random_rankings,   None,  False),
        ('E: Shuffled dates, LGBM',     shuffled_rankings, None,  False),
    ]

    all_results = {}
    for vname, vrank, exit_day, exit_hc in variants:
        fprint(f"\n--- {vname} ---")
        trades, final_eq, eq_curve = simulate_variant(
            vname, vrank, sc, sh, sl, spy, vix,
            early_exit_day=exit_day, apply_exit_haircut=exit_hc
        )
        metrics = compute_all_metrics(trades, final_eq, eq_curve, vname)
        if metrics is None:
            continue

        fprint(f"  Trades: {metrics['n_trades']} | WR: {metrics['win_rate']:.1f}%")
        fprint(f"  Sharpe: {metrics['sharpe']:.2f} | Sortino: {metrics['sortino']:.2f}")
        fprint(f"  CAGR: {metrics['cagr_pct']:.1f}% | MaxDD: {metrics['maxdd_pct']:.1f}%")
        fprint(f"  PF: {metrics['pf']:.2f} | Avg P&L: ${metrics['avg_pnl']:.2f}")
        fprint(f"  Avg Hold: {metrics['avg_hold']:.1f}d | Avg Cost: ${metrics['avg_cost']:.2f}")
        fprint(f"  Final Equity: ${metrics['final_equity']:.2f} (from ${CAP})")
        fprint(f"  Bull: {metrics['bull_n']} trades, {metrics['bull_wr']:.1f}% WR")
        fprint(f"  Bear: {metrics['bear_n']} trades, {metrics['bear_wr']:.1f}% WR")
        fprint(f"  Regime gap: {metrics['regime_gap']:.3f} (< 0.50 = OK)")

        all_results[vname] = {
            'metrics': metrics,
            'trades': trades,
        }

    # ---- Step 6: Statistical tests on each variant ----
    fprint("\n" + "=" * 90)
    fprint("Step 6: STATISTICAL TESTS")
    fprint("=" * 90)

    for vname, data in all_results.items():
        m = data['metrics']
        mr = m['monthly_rets']
        fprint(f"\n--- {vname} ---")

        # Permutation test
        perm_p = permutation_test(mr)
        fprint(f"  Permutation p-value: {perm_p:.4f} {'PASS (p<0.05)' if perm_p < 0.05 else 'FAIL (p>=0.05)'}")

        # Sub-period
        h1_sh, h2_sh, sub_pass = sub_period_test(mr)
        fprint(f"  Sub-period: H1 Sharpe={h1_sh:.2f}, H2 Sharpe={h2_sh:.2f} "
               f"{'PASS (both>0)' if sub_pass else 'FAIL'}")

        # Regime
        regime_ok = m['regime_gap'] < 0.50
        fprint(f"  Regime gap: {m['regime_gap']:.3f} {'PASS (<0.50)' if regime_ok else 'FAIL (>=0.50)'}")

        # Yearly breakdown
        yr_sharpes = yearly_sharpe(data['trades'])
        fprint(f"  Yearly breakdown:")
        for yr, ys in sorted(yr_sharpes.items()):
            fprint(f"    {yr}: Sharpe={ys['sharpe']:>6.2f} | N={ys['n']:>3} | "
                   f"WR={ys['wr']:>5.1f}% | P&L=${ys['pnl']:>8.2f}")

        # Store results
        m['perm_p'] = perm_p
        m['h1_sharpe'] = h1_sh
        m['h2_sharpe'] = h2_sh
        m['sub_period_pass'] = sub_pass
        m['regime_pass'] = regime_ok
        m['yearly'] = yr_sharpes
        gates = sum([perm_p < 0.05, sub_pass, regime_ok])
        m['gates_passed'] = gates
        fprint(f"  GATES: {gates}/3 passed")

    # ---- Step 7: COMPARISON TABLE ----
    fprint("\n" + "=" * 90)
    fprint("Step 7: COMPARISON TABLE")
    fprint("=" * 90)

    header = f"{'Variant':<30} {'N':>5} {'WR':>6} {'Sharpe':>7} {'Sort':>6} {'CAGR':>7} {'MDD':>7} {'PF':>6} {'Final$':>9} {'Gates':>6}"
    fprint(header)
    fprint("-" * len(header))
    for vname, data in all_results.items():
        m = data['metrics']
        fprint(f"{vname:<30} {m['n_trades']:>5} {m['win_rate']:>5.1f}% {m['sharpe']:>7.2f} "
               f"{m['sortino']:>6.2f} {m['cagr_pct']:>6.1f}% {m['maxdd_pct']:>6.1f}% "
               f"{m['pf']:>6.2f} ${m['final_equity']:>8.0f} {m['gates_passed']:>4}/3")

    # ---- Step 8: KEY FINDINGS ----
    fprint("\n" + "=" * 90)
    fprint("Step 8: KEY FINDINGS")
    fprint("=" * 90)

    a = all_results.get('A: Hold-to-Expiry (30d)', {}).get('metrics')
    b = all_results.get('B: Exit-20d WITH haircut', {}).get('metrics')
    c = all_results.get('C: Exit-20d NO haircut', {}).get('metrics')
    d = all_results.get('D: RANDOM sectors, expiry', {}).get('metrics')
    e = all_results.get('E: Shuffled dates, LGBM', {}).get('metrics')

    if a:
        fprint(f"\n1. CONSERVATIVE BASELINE (hold to expiry):")
        fprint(f"   Sharpe: {a['sharpe']:.2f} | This is the FLOOR — no exit assumptions")

    if b and c:
        fprint(f"\n2. EXIT HAIRCUT IMPACT:")
        fprint(f"   With exit haircut:    Sharpe {b['sharpe']:.2f}")
        fprint(f"   Without exit haircut: Sharpe {c['sharpe']:.2f}")
        if c['sharpe'] > 0:
            pct_inflation = (c['sharpe'] - b['sharpe']) / max(abs(b['sharpe']), 0.01) * 100
            fprint(f"   Sharpe inflation from no exit haircut: {pct_inflation:+.0f}%")
            fprint(f"   => The exit haircut {'matters a lot' if abs(pct_inflation) > 30 else 'has moderate impact' if abs(pct_inflation) > 10 else 'has small impact'}")

    if a and d:
        fprint(f"\n3. ML EDGE vs STRUCTURAL EDGE:")
        fprint(f"   LGBM ranking:   Sharpe {a['sharpe']:.2f}")
        fprint(f"   Random sectors: Sharpe {d['sharpe']:.2f}")
        ml_alpha = a['sharpe'] - d['sharpe']
        fprint(f"   ML alpha (Sharpe difference): {ml_alpha:+.2f}")
        if d['sharpe'] > 0 and a['sharpe'] > 0:
            ml_pct = ml_alpha / max(abs(a['sharpe']), 0.01) * 100
            if d['sharpe'] > a['sharpe'] * 0.80:
                fprint(f"   WARNING: Random is >{80:.0f}% of LGBM => edge is mostly STRUCTURAL (bull spreads + VIX filter)")
                fprint(f"   The LGBM ranking adds {'no' if ml_alpha <= 0 else 'marginal'} value")
            else:
                fprint(f"   GOOD: ML adds {ml_pct:.0f}% incremental Sharpe over random")

    if a and e:
        fprint(f"\n4. TIMING EDGE:")
        fprint(f"   Real dates:     Sharpe {a['sharpe']:.2f}")
        fprint(f"   Shuffled dates: Sharpe {e['sharpe']:.2f}")
        if e['sharpe'] > a['sharpe'] * 0.80:
            fprint(f"   => Timing adds little value. Edge is in the STRUCTURE, not WHEN you enter.")
        else:
            fprint(f"   => Timing matters. Real entry dates are meaningfully better.")

    fprint(f"\n5. BOTTOM LINE:")
    if a:
        if a['sharpe'] > 1.0 and a.get('gates_passed', 0) >= 2:
            fprint(f"   Strategy has REAL edge (Sharpe {a['sharpe']:.2f}, {a.get('gates_passed',0)}/3 gates)")
            fprint(f"   But be honest about what drives it:")
            if d and d['sharpe'] > a['sharpe'] * 0.80:
                fprint(f"   - Mostly structural (bull call spreads in high-VIX environments)")
                fprint(f"   - LGBM ranking adds little incremental value")
            else:
                fprint(f"   - ML ranking adds meaningful alpha on top of structural edge")
        elif a['sharpe'] > 0.5:
            fprint(f"   Strategy has MARGINAL edge (Sharpe {a['sharpe']:.2f})")
        else:
            fprint(f"   Strategy edge is WEAK or NONEXISTENT (Sharpe {a['sharpe']:.2f})")

    elapsed = (datetime.now() - t0).total_seconds()
    fprint(f"\nRuntime: {elapsed:.0f}s ({elapsed / 60:.1f}m)")
    fprint("=" * 90)
    fprint("DONE — Definitive Adversarial Validation v1")
    fprint("=" * 90)


if __name__ == '__main__':
    main()
