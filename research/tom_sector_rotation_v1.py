#!/usr/bin/env python3
"""
Turn-of-Month Enhanced Sector Rotation v1
==========================================
Combines two validated edges:
  1. Turn-of-Month (TOM) calendar anomaly: last-2 + first-3 trading days per month
     have structurally higher returns (Sharpe 1.276 vs 0.750 mid-month — SESSION_STATE #1305)
  2. LGBM sector rotation: momentum/quality ranking of sector ETFs (Sharpe 1.40-1.87)

Variants:
  A: TOM-Only Sector Equity   — top-2 sectors only during TOM window, cash mid-month
  B: TOM + Momentum Filter    — skip entry if 21d momentum of top sector is negative
  C: TOM + Anti-TOM Hedge     — long sectors TOM, short SPY (30% notional) mid-month
  D: Full-Month + TOM Sizing  — always invested: 2x during TOM, 0.5x mid-month
  E: TOM + VIX>15 Filter      — only enter TOM when VIX > 15
  F: Plain Sector Rotation    — baseline: top-2 sectors always, no TOM timing

OOT: Jan 2022 — Jul 2026 (4.5 years)
Capital: $645 | HC #428 R1 regime-agnostic | HC #0 sliding walk-forward
"""

import sys, json, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import lightgbm as lgb

warnings.filterwarnings('ignore')

def fprint(*a, **k):
    print(*a, **k)
    sys.stdout.flush()

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT        = Path('/home/jupiter/Lvl3Quant')
PRICES_PATH = ROOT / 'wheel_strategy_v1/data/cache/prices_v2.parquet'
MACRO_PATH  = ROOT / 'wheel_strategy_v1/data/cache/macro.parquet'
OUT_DIR     = ROOT / 'research/findings'
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Config ────────────────────────────────────────────────────────────────────
SECTOR_ETFS   = ['XLK','XLV','XLE','XLF','XLI','XLC','XLY','XLP','XLU','XLB','XLRE']
BENCHMARK     = 'SPY'
TOP_K         = 2
CAPITAL       = 645.0
TXN_BPS       = 5.0          # round-trip bps per trade
OOT_START     = '2022-01-01'
OOT_END       = '2026-07-27'
TRAIN_DAYS    = 60
N_PERM        = 100
VIX_THRESHOLD = 15.0
MOM_LOOKBACK  = 21
RF_ANNUAL     = 0.045
RF_DAILY      = RF_ANNUAL / 252

LGBM_PARAMS = dict(
    objective='regression', metric='rmse', boosting_type='gbdt',
    num_leaves=15, learning_rate=0.05, feature_fraction=0.8,
    bagging_fraction=0.8, bagging_freq=5, min_child_samples=10,
    lambda_l1=0.1, lambda_l2=0.5, max_depth=4,
    verbosity=-1, seed=42, n_jobs=4,
)
NUM_BOOST = 200
EARLY_STOP = 20


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_data():
    fprint('Loading prices...')
    raw = pd.read_parquet(PRICES_PATH)
    universe = SECTOR_ETFS + [BENCHMARK]
    raw = raw[raw['ticker'].isin(universe)].copy()
    raw['date'] = pd.to_datetime(raw['date'])

    close = raw.pivot_table(index='date', columns='ticker', values='close').sort_index()
    # Forward-fill gaps (XLC/XLRE started later; ffill is fine post-inception)
    close = close.ffill()
    # Drop rows where SPY is missing (non-trading days leaked in)
    close = close.dropna(subset=[BENCHMARK])

    fprint(f'  Prices: {close.shape} ({close.index.min().date()} → {close.index.max().date()})')

    # VIX from macro
    fprint('Loading VIX...')
    macro = pd.read_parquet(MACRO_PATH)
    macro['date'] = pd.to_datetime(macro['date'])
    vix = macro.set_index('date')[['vix']].rename(columns={'vix': 'VIX'})
    vix = vix.reindex(close.index).ffill().fillna(20.0)
    fprint(f'  VIX: {vix["VIX"].notna().sum()} days covered')

    return close, vix


# ══════════════════════════════════════════════════════════════════════════════
# TOM CALENDAR
# ══════════════════════════════════════════════════════════════════════════════

def compute_tom_mask(dates: pd.DatetimeIndex) -> pd.Series:
    """
    TOM window: last 2 trading days of month + first 3 trading days of next month.
    Validated edge: Sharpe 1.276 TOM vs 0.750 mid-month (SESSION_STATE #1305).
    """
    tom = pd.Series(False, index=dates)
    ym  = pd.Series(dates).dt.to_period('M')
    months = sorted(ym.unique())

    for i, m in enumerate(months):
        mask_this = (ym == m).values
        days_this = dates[mask_this]
        if len(days_this) < 2:
            continue
        last2 = days_this[-2:]
        if i + 1 < len(months):
            mask_next = (ym == months[i+1]).values
            days_next = dates[mask_next]
            first3 = days_next[:3]
        else:
            first3 = pd.DatetimeIndex([])
        tom_days = last2.append(first3)
        tom[tom_days] = True

    return tom


# ══════════════════════════════════════════════════════════════════════════════
# LGBM FEATURES
# ══════════════════════════════════════════════════════════════════════════════

def build_features(close: pd.DataFrame) -> pd.DataFrame:
    """
    Build cross-sectional momentum panel: (date, ticker, features, fwd_ret_21d).
    HC #0: used in sliding walk-forward only.
    """
    records = []
    spy = close[BENCHMARK]

    for etf in SECTOR_ETFS:
        if etf not in close.columns:
            continue
        c  = close[etf]
        lr = np.log(c / c.shift(1))

        df = pd.DataFrame(index=c.index)
        df['ticker'] = etf

        # Momentum
        for w in [5, 10, 21, 42, 63]:
            df[f'mom_{w}d'] = c.pct_change(w)

        # Momentum acceleration
        df['mom_accel'] = df['mom_21d'].diff(5)

        # Volatility-adjusted momentum (Information Ratio proxy)
        rv20 = lr.rolling(20).std() * np.sqrt(252)
        df['ir_21d'] = df['mom_21d'] / rv20.replace(0, np.nan)

        # Relative strength vs SPY
        df['rs_spy_21d'] = c.pct_change(21) - spy.pct_change(21)
        df['rs_spy_63d'] = c.pct_change(63) - spy.pct_change(63)

        # Vol features
        df['rv_20'] = lr.rolling(20).std() * np.sqrt(252)
        df['rv_60'] = lr.rolling(60).std() * np.sqrt(252)
        df['vol_ratio'] = df['rv_20'] / df['rv_60'].replace(0, np.nan)

        # Distance from moving averages
        sma20 = c.rolling(20).mean()
        sma60 = c.rolling(60).mean()
        df['dist_sma20'] = (c - sma20) / sma20.replace(0, np.nan)
        df['dist_sma60'] = (c - sma60) / sma60.replace(0, np.nan)

        # RSI proxy
        gains  = lr.clip(lower=0).rolling(14).mean()
        losses = (-lr.clip(upper=0)).rolling(14).mean()
        df['rsi14'] = gains / (gains + losses + 1e-9)

        # Forward return target: 21-day (used during training only)
        df['fwd_ret_21d'] = c.pct_change(21).shift(-21)

        df = df.reset_index().rename(columns={'index': 'date', 'date': 'date'})
        if 'index' in df.columns:
            df = df.rename(columns={'index': 'date'})
        records.append(df)

    panel = pd.concat(records, ignore_index=True)
    feat_cols = [c for c in panel.columns
                 if c not in ['ticker', 'date', 'fwd_ret_21d']]
    return panel, feat_cols


# ══════════════════════════════════════════════════════════════════════════════
# WALK-FORWARD LGBM RANKING
# ══════════════════════════════════════════════════════════════════════════════

def walk_forward_rank(panel: pd.DataFrame, feat_cols: list) -> pd.DataFrame:
    """
    Sliding 60-day train → predict top sectors per day.
    HC #0: SLIDING window only (oldest day dropped when window advances).
    Returns DataFrame: date, ticker, score, rank.
    """
    fprint('Running walk-forward LGBM ranking...')
    dates_all = sorted(panel['date'].unique())
    results   = []
    n_fits    = 0

    for i, dt in enumerate(dates_all):
        # Sliding window: last TRAIN_DAYS before this date
        train_idx_start = max(0, i - TRAIN_DAYS)
        train_dates     = dates_all[train_idx_start:i]
        if len(train_dates) < 30:
            continue

        train = panel[panel['date'].isin(train_dates)].copy()
        train = train.dropna(subset=['fwd_ret_21d'] + feat_cols)
        if len(train) < 30:
            continue

        test = panel[panel['date'] == dt].copy()
        if test.empty:
            continue

        X_tr = train[feat_cols].fillna(0)
        y_tr = train['fwd_ret_21d']
        X_te = test[feat_cols].fillna(0)

        dtrain = lgb.Dataset(X_tr, label=y_tr)
        model  = lgb.train(
            LGBM_PARAMS, dtrain, num_boost_round=NUM_BOOST,
            valid_sets=[dtrain],
            callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False),
                       lgb.log_evaluation(-1)]
        )
        scores = model.predict(X_te)
        test   = test.copy()
        test['score'] = scores
        test['rank']  = test['score'].rank(ascending=False)
        results.append(test[['date','ticker','score','rank']])
        n_fits += 1

        if n_fits % 50 == 0:
            fprint(f'  ...{n_fits} WF steps (date={dt.date() if hasattr(dt, "date") else dt})')

    fprint(f'  Walk-forward complete: {n_fits} steps')
    if not results:
        return pd.DataFrame()
    return pd.concat(results, ignore_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# REGIME CLASSIFICATION
# ══════════════════════════════════════════════════════════════════════════════

def classify_regime(close: pd.DataFrame) -> pd.Series:
    """SPY 21d forward return → green/red/flat labels."""
    spy    = close[BENCHMARK]
    fwd21  = spy.pct_change(21).shift(-21)
    regime = pd.Series('flat', index=spy.index)
    regime[fwd21 >  0.01] = 'green'
    regime[fwd21 < -0.01] = 'red'
    return regime


# ══════════════════════════════════════════════════════════════════════════════
# BACKTEST ENGINE — RETURN-BASED (NaN-SAFE)
# ══════════════════════════════════════════════════════════════════════════════

def get_top_sectors(ranking: pd.DataFrame, dt, prev_tops: list) -> list:
    """Get LGBM top-K sectors for date dt."""
    rank_dt = ranking[ranking['date'] == dt]
    if rank_dt.empty:
        return prev_tops if prev_tops else SECTOR_ETFS[:TOP_K]
    return rank_dt.sort_values('rank').head(TOP_K)['ticker'].tolist()


def backtest(
    close: pd.DataFrame,
    vix:   pd.DataFrame,
    ranking: pd.DataFrame,
    tom: pd.Series,
    variant: str = 'A',
    capital: float = CAPITAL,
) -> pd.Series:
    """
    Return-based backtest engine. NaN-safe.
    Each day: compute portfolio return as weighted sum of ETF returns.

    size_mult: position sizing multiplier (1.0 = 100% of capital)
    invested:  True → hold top-K sectors, False → cash
    """
    dates_oot  = tom.index
    oot_mask   = (dates_oot >= OOT_START) & (dates_oot <= OOT_END)
    dates_oot  = dates_oot[oot_mask]

    # Pre-compute daily returns for all sector ETFs
    daily_rets = close[SECTOR_ETFS].pct_change()  # (date × etf)

    equity     = [capital]
    eq_dates   = [dates_oot[0]]
    prev_tops  = SECTOR_ETFS[:TOP_K]
    port_val   = capital

    # For variant D: track 2x leverage. We track a single multiplier.
    # For variant C: track SPY short position P&L mid-month.
    spy_rets   = daily_rets.get('SPY', pd.Series(0.0, index=daily_rets.index))
    if 'SPY' in close.columns:
        spy_rets = close[BENCHMARK].pct_change()

    for i in range(1, len(dates_oot)):
        dt      = dates_oot[i]
        prev_dt = dates_oot[i-1]

        # LGBM top sectors
        tops = get_top_sectors(ranking, dt, prev_tops)

        # TOM flag
        in_tom = bool(tom.get(dt, False))

        # VIX
        vix_val = float(vix.loc[dt, 'VIX']) if dt in vix.index else 20.0
        if np.isnan(vix_val):
            vix_val = 20.0

        # Variant logic
        if variant == 'A':
            invested   = in_tom
            size_mult  = 1.0
            hedge_mult = 0.0

        elif variant == 'B':
            invested = in_tom
            size_mult = 1.0
            hedge_mult = 0.0
            # Skip if top sector 21d momentum is negative
            if invested and tops:
                top_etf = tops[0]
                if top_etf in close.columns:
                    c_etf = close[top_etf]
                    if prev_dt in c_etf.index:
                        idx = c_etf.index.get_loc(prev_dt)
                        if idx >= MOM_LOOKBACK:
                            mom21 = c_etf.iloc[idx] / c_etf.iloc[idx - MOM_LOOKBACK] - 1
                            if mom21 < 0:
                                invested = False

        elif variant == 'C':
            # Long sectors TOM, short SPY 30% notional mid-month
            invested   = in_tom
            size_mult  = 1.0
            hedge_mult = 0.0 if in_tom else -0.30  # negative = short SPY

        elif variant == 'D':
            # Always invested, 2x TOM / 0.5x mid
            invested   = True
            size_mult  = 2.0 if in_tom else 0.5
            hedge_mult = 0.0

        elif variant == 'E':
            # TOM + VIX > 15 filter
            invested   = in_tom and (vix_val > VIX_THRESHOLD)
            size_mult  = 1.0
            hedge_mult = 0.0

        elif variant == 'F':
            # Baseline: always fully invested
            invested   = True
            size_mult  = 1.0
            hedge_mult = 0.0

        else:
            invested   = in_tom
            size_mult  = 1.0
            hedge_mult = 0.0

        # ── Compute daily portfolio return ──────────────────────────────────────
        port_ret = 0.0

        if invested:
            # Equal weight across top-K sectors
            etf_rets = []
            for etf in tops:
                if etf in daily_rets.columns and dt in daily_rets.index:
                    r = daily_rets.loc[dt, etf]
                    if not np.isnan(r):
                        etf_rets.append(r)
            if etf_rets:
                raw_ret  = np.mean(etf_rets)
                # Transaction cost: apply on turnover only
                # Simplified: cost applied as daily drag proportional to rebalance frequency
                txn_drag = TXN_BPS / 10000 / 21  # amortize over ~21-day hold
                port_ret = raw_ret * size_mult - txn_drag * size_mult

        # SPY hedge (variant C mid-month)
        if hedge_mult != 0.0 and dt in daily_rets.index:
            spy_r = spy_rets.loc[dt] if dt in spy_rets.index else 0.0
            if np.isnan(spy_r):
                spy_r = 0.0
            # Short position: gain when SPY falls
            hedge_ret = -spy_r * abs(hedge_mult)
            hedge_cost = TXN_BPS / 10000 / 21
            port_ret  += hedge_ret - hedge_cost

        # Clamp catastrophic losses
        port_ret = max(port_ret, -0.5)

        port_val = port_val * (1 + port_ret)
        port_val = max(port_val, 0.01)  # avoid going negative

        equity.append(port_val)
        eq_dates.append(dt)
        prev_tops = tops

    eq_series = pd.Series(equity, index=pd.DatetimeIndex(eq_dates))
    return eq_series


# ══════════════════════════════════════════════════════════════════════════════
# METRICS
# ══════════════════════════════════════════════════════════════════════════════

def compute_metrics(equity: pd.Series) -> dict:
    rets = equity.pct_change().dropna()
    if len(rets) < 10 or rets.isna().all():
        return dict(sharpe=np.nan, sortino=np.nan, cagr=np.nan,
                    mdd=np.nan, wr=np.nan, pf=np.nan,
                    n_days=len(rets), final_equity=float(equity.iloc[-1]))

    excess   = rets - RF_DAILY
    sharpe   = excess.mean() / (excess.std() + 1e-9) * np.sqrt(252)

    down     = rets[rets < RF_DAILY]
    sortino  = (excess.mean() / (down.std() + 1e-9)) * np.sqrt(252) if len(down) > 5 else np.nan

    n_years  = len(rets) / 252
    cagr     = (equity.iloc[-1] / equity.iloc[0]) ** (1/n_years) - 1 if n_years > 0 else 0

    roll_max = equity.cummax()
    mdd      = ((equity - roll_max) / roll_max).min()

    wins     = (rets > 0).sum()
    wr       = wins / len(rets)
    pos_sum  = rets[rets > 0].sum()
    neg_sum  = abs(rets[rets <= 0].sum())
    pf       = pos_sum / (neg_sum + 1e-9)

    return dict(
        sharpe=round(float(sharpe), 3),
        sortino=round(float(sortino) if not np.isnan(sortino) else np.nan, 3),
        cagr=round(float(cagr)*100, 2),
        mdd=round(float(mdd)*100, 2),
        wr=round(float(wr)*100, 2),
        pf=round(float(pf), 3),
        n_days=int(len(rets)),
        final_equity=round(float(equity.iloc[-1]), 2),
    )


def regime_sharpe(equity: pd.Series, regime: pd.Series) -> dict:
    """Per-regime Sharpe + HC #428 R1 balance check."""
    rets = equity.pct_change().dropna()
    out  = {}
    for reg in ['green', 'red', 'flat']:
        mask  = regime.reindex(rets.index) == reg
        r_sub = rets[mask]
        if len(r_sub) < 5:
            out[f'sharpe_{reg}'] = np.nan
            out[f'n_{reg}'] = 0
        else:
            ex = r_sub - RF_DAILY
            out[f'sharpe_{reg}'] = round(float(ex.mean() / (ex.std()+1e-9) * np.sqrt(252)), 3)
            out[f'n_{reg}'] = int(mask.sum())

    sg = out.get('sharpe_green', np.nan)
    sr = out.get('sharpe_red', np.nan)
    if not (np.isnan(sg) or np.isnan(sr) or sg is None or sr is None):
        denom = max(abs(sg), abs(sr), 1e-9)
        gap   = abs(sg - sr) / denom
        out['regime_gap']      = round(float(gap), 3)
        out['regime_balance_ok'] = bool(gap <= 0.50)
    else:
        out['regime_gap']      = np.nan
        out['regime_balance_ok'] = None
    return out


def permutation_test(equity: pd.Series, n_perm: int = N_PERM) -> dict:
    """
    Shuffle monthly RETURNS (not daily) to preserve autocorrelation structure.
    p-value = fraction of permutations with Sharpe >= actual.
    """
    rets    = equity.pct_change().dropna()
    excess  = rets - RF_DAILY
    actual  = float(excess.mean() / (excess.std()+1e-9) * np.sqrt(252))

    # Aggregate to monthly blocks then shuffle
    monthly = rets.resample('ME').apply(lambda x: (1+x).prod()-1)
    arr     = monthly.values.copy()
    rng     = np.random.default_rng(42)

    perm_sharpes = []
    for _ in range(n_perm):
        shuffled = arr.copy()
        rng.shuffle(shuffled)
        # Reconstruct daily-like excess
        perm_ex = shuffled - (RF_ANNUAL / 12)
        perm_sharpes.append(
            float(perm_ex.mean() / (perm_ex.std()+1e-9) * np.sqrt(12))
        )

    perm_arr = np.array(perm_sharpes)
    pval     = float((perm_arr >= actual).mean())

    return dict(
        actual_sharpe=round(actual, 3),
        perm_mean=round(float(perm_arr.mean()), 3),
        perm_std=round(float(perm_arr.std()), 3),
        p_value=round(pval, 4),
        significant=bool(pval < 0.05),
    )


# ══════════════════════════════════════════════════════════════════════════════
# RANDOM BASELINE
# ══════════════════════════════════════════════════════════════════════════════

def random_sector_baseline(close: pd.DataFrame, tom: pd.Series, n_sims: int = 50) -> dict:
    """Same TOM timing but random sector pairs — measures timing vs selection value."""
    rng      = np.random.default_rng(99)
    daily    = close[SECTOR_ETFS].pct_change()
    oot_mask = (tom.index >= OOT_START) & (tom.index <= OOT_END)
    dates    = tom.index[oot_mask]
    in_tom   = tom[oot_mask]

    sharpes = []
    for _ in range(n_sims):
        port = CAPITAL
        equity = []
        for i in range(1, len(dates)):
            dt = dates[i]
            if not in_tom.iloc[i]:
                equity.append(port)
                continue
            chosen = rng.choice(len(SECTOR_ETFS), size=TOP_K, replace=False)
            rets_today = []
            for idx_e in chosen:
                etf = SECTOR_ETFS[idx_e]
                if dt in daily.index:
                    r = daily.loc[dt, etf]
                    if not np.isnan(r):
                        rets_today.append(r)
            port *= (1 + np.mean(rets_today)) if rets_today else 1.0
            equity.append(port)

        eq_ser = pd.Series(equity)
        rets_s = eq_ser.pct_change().dropna()
        if len(rets_s) > 5:
            ex = rets_s - RF_DAILY
            sharpes.append(float(ex.mean() / (ex.std()+1e-9) * np.sqrt(252)))

    arr = np.array(sharpes)
    return dict(
        random_sharpe_mean=round(float(arr.mean()), 3) if len(arr) else np.nan,
        random_sharpe_std=round(float(arr.std()), 3) if len(arr) else np.nan,
        n_sims=n_sims,
    )


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    fprint('='*70)
    fprint('TURN-OF-MONTH ENHANCED SECTOR ROTATION v1')
    fprint(f'Capital: ${CAPITAL}  |  OOT: {OOT_START} → {OOT_END}')
    fprint(f'Top-K: {TOP_K} sectors  |  TXN: {TXN_BPS} bps  |  RF: {RF_ANNUAL*100:.1f}%')
    fprint('='*70)

    # ── Load data ──────────────────────────────────────────────────────────────
    close, vix = load_data()

    # ── TOM mask ───────────────────────────────────────────────────────────────
    fprint('\nComputing TOM calendar mask...')
    tom       = compute_tom_mask(close.index)
    n_tom     = int(tom.sum())
    n_total   = len(tom)
    fprint(f'  TOM days: {n_tom}/{n_total} = {n_tom/n_total*100:.1f}% of trading days')

    # OOT TOM stats
    oot_tom   = tom[(tom.index >= OOT_START) & (tom.index <= OOT_END)]
    n_tom_oot = int(oot_tom.sum())
    fprint(f'  TOM days in OOT window: {n_tom_oot}/{len(oot_tom)} = {n_tom_oot/len(oot_tom)*100:.1f}%')

    # ── TOM effect on SPY ──────────────────────────────────────────────────────
    spy_rets  = close[BENCHMARK].pct_change().dropna()
    oot_spy   = spy_rets[(spy_rets.index >= OOT_START) & (spy_rets.index <= OOT_END)]
    tom_oot_a = oot_tom.reindex(oot_spy.index).fillna(False)

    tom_spy   = oot_spy[tom_oot_a]
    mid_spy   = oot_spy[~tom_oot_a]

    tom_sharpe = float((tom_spy-RF_DAILY).mean() / ((tom_spy-RF_DAILY).std()+1e-9) * np.sqrt(252))
    mid_sharpe = float((mid_spy-RF_DAILY).mean() / ((mid_spy-RF_DAILY).std()+1e-9) * np.sqrt(252))
    tom_mean   = float(tom_spy.mean() * 252 * 100)
    mid_mean   = float(mid_spy.mean() * 252 * 100)

    fprint(f'\nTOM EFFECT (OOT 2022-2026):')
    fprint(f'  SPY TOM Sharpe: {tom_sharpe:.3f}  annualized ret: {tom_mean:.1f}%')
    fprint(f'  SPY Mid Sharpe: {mid_sharpe:.3f}  annualized ret: {mid_mean:.1f}%')
    fprint(f'  Uplift: {tom_sharpe - mid_sharpe:.3f} Sharpe units from timing alone')

    # ── LGBM Walk-forward ranking ──────────────────────────────────────────────
    fprint('\nBuilding LGBM features...')
    panel, feat_cols = build_features(close)
    fprint(f'  Panel: {panel.shape}  |  features: {len(feat_cols)}')

    # Include warm-up period before OOT start
    panel_train = panel[panel['date'] >= '2021-01-01'].copy()
    fprint(f'  Panel for WF (incl. warm-up): {panel_train.shape}')

    ranking = walk_forward_rank(panel_train, feat_cols)
    if ranking.empty:
        fprint('ERROR: walk-forward ranking failed. Exiting.')
        return

    ranking_oot = ranking[
        (ranking['date'] >= OOT_START) &
        (ranking['date'] <= OOT_END)
    ].copy()
    fprint(f'  Ranking in OOT: {len(ranking_oot)} rows')

    # ── Regime labels ──────────────────────────────────────────────────────────
    regime = classify_regime(close)

    # ── Run all variants ───────────────────────────────────────────────────────
    fprint('\n' + '='*70)
    fprint('RUNNING VARIANTS...')
    fprint('='*70)

    variant_labels = {
        'F': 'Plain Sector Rotation (Baseline)',
        'A': 'TOM-Only Sector Equity',
        'B': 'TOM + Momentum Filter',
        'C': 'TOM + Anti-TOM SPY Hedge',
        'D': 'Full-Month + TOM Sizing (2x/0.5x)',
        'E': 'TOM + VIX>15 Filter',
    }

    results       = {}
    equity_curves = {}

    for var in ['F', 'A', 'B', 'C', 'D', 'E']:
        label = variant_labels[var]
        fprint(f'\n--- Variant {var}: {label} ---')

        eq = backtest(close, vix, ranking_oot, tom, variant=var)
        equity_curves[var] = eq

        m  = compute_metrics(eq)
        rg = regime_sharpe(eq, regime.reindex(eq.index))
        pm = permutation_test(eq)

        reg_ok  = 'YES' if rg.get('regime_balance_ok') else 'NO'
        fprint(f'  Sharpe={m["sharpe"]}  Sortino={m["sortino"]}  '
               f'CAGR={m["cagr"]}%  MDD={m["mdd"]}%  WR={m["wr"]}%  PF={m["pf"]}')
        fprint(f'  Final equity: ${m["final_equity"]:,.2f}  (from ${CAPITAL})')
        fprint(f'  Regime: green={rg.get("sharpe_green")}  red={rg.get("sharpe_red")}  '
               f'gap={rg.get("regime_gap")}  balanced={reg_ok}')
        fprint(f'  Permutation p={pm["p_value"]}  significant={pm["significant"]}  '
               f'(actual={pm["actual_sharpe"]} vs perm_mean={pm["perm_mean"]}±{pm["perm_std"]})')

        results[var] = dict(
            variant=var, label=label,
            metrics=m, regime=rg, permutation=pm,
        )

    # ── Random baseline ────────────────────────────────────────────────────────
    fprint('\nComputing random sector baseline (same TOM timing, random picks)...')
    rand_base = random_sector_baseline(close, tom)
    fprint(f'  Random Sharpe: {rand_base["random_sharpe_mean"]:.3f} ± {rand_base["random_sharpe_std"]:.3f}')

    # ── Summary table ──────────────────────────────────────────────────────────
    fprint('\n' + '='*70)
    fprint('SUMMARY TABLE')
    fprint('='*70)
    hdr = f'{"Var":<4} {"Label":<36} {"Sharpe":>7} {"Sortino":>8} {"CAGR%":>7} '
    hdr += f'{"MDD%":>6} {"WR%":>6} {"PF":>6} {"p-val":>7} {"Regime":>8}'
    fprint(hdr)
    fprint('-'*102)

    for var in ['F', 'A', 'B', 'C', 'D', 'E']:
        r  = results[var]
        m  = r['metrics']
        rg = r['regime']
        pm = r['permutation']
        ok = 'PASS' if rg.get('regime_balance_ok') else 'FAIL'
        sortino_str = f'{m["sortino"]:8.3f}' if m["sortino"] is not None and not (isinstance(m["sortino"], float) and np.isnan(m["sortino"])) else '     N/A'
        fprint(
            f'{var:<4} {r["label"][:36]:<36} '
            f'{m["sharpe"]:>7.3f} {sortino_str} {m["cagr"]:>7.1f} '
            f'{m["mdd"]:>6.1f} {m["wr"]:>6.1f} {m["pf"]:>6.3f} '
            f'{pm["p_value"]:>7.4f} {ok:>8}'
        )

    fprint(f'\nRandom baseline (TOM timing, random sector pairs): '
           f'Sharpe = {rand_base["random_sharpe_mean"]:.3f} ± {rand_base["random_sharpe_std"]:.3f}')

    # ── Key insight ────────────────────────────────────────────────────────────
    sha_a = results['A']['metrics']['sharpe']
    sha_f = results['F']['metrics']['sharpe']
    sha_d = results['D']['metrics']['sharpe']
    fprint('\nKEY FINDINGS:')
    fprint(f'  TOM-only (A) vs Always-in (F): {sha_a:.3f} vs {sha_f:.3f}  '
           f'(lift={sha_a-sha_f:+.3f})')
    fprint(f'  Best variant: ' + max(results.items(), key=lambda x: x[1]['metrics']['sharpe'] or -999)[1]['label'])
    fprint(f'  TOM timing lift over random baseline: '
           f'{sha_a - rand_base["random_sharpe_mean"]:+.3f} Sharpe units')

    fprint(f'\nTOM CALENDAR STATS:')
    fprint(f'  SPY TOM Sharpe in OOT: {tom_sharpe:.3f}  vs  Mid-month: {mid_sharpe:.3f}')

    # ── Save ───────────────────────────────────────────────────────────────────
    out = dict(
        strategy='TOM Enhanced Sector Rotation v1',
        run_date=datetime.now().isoformat(),
        capital=CAPITAL,
        oot_start=OOT_START,
        oot_end=OOT_END,
        top_k=TOP_K,
        tom_effect=dict(
            tom_sharpe_oot=round(tom_sharpe, 3),
            mid_sharpe_oot=round(mid_sharpe, 3),
            tom_pct_days_oot=round(n_tom_oot/len(oot_tom)*100, 1),
            uplift=round(tom_sharpe - mid_sharpe, 3),
        ),
        variants=results,
        random_baseline=rand_base,
    )

    out_path = OUT_DIR / 'tom_sector_rotation_v1_results.json'
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2, default=str)
    fprint(f'\nResults saved → {out_path}')
    fprint('DONE.')
    return out


if __name__ == '__main__':
    main()
