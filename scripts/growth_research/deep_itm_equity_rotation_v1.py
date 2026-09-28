#!/usr/bin/env python3
"""
Deep ITM Call Equity Rotation V1
=================================

HYPOTHESIS: The LGBM sector ranking signal is VALIDATED (KB #285, Sharpe 1.40,
perm p=0.0016) but we can't trade it as pure equity in a $645 OPTIONS-ONLY account.
Deep ITM calls provide stock-like exposure with defined risk and leverage.

WHY DEEP ITM CALLS WORK HERE:
  - Delta ~0.85-0.95: captures 85-95% of stock movement
  - Premium ≈ intrinsic + small time value: BS pricing is MORE ACCURATE for ITM
    (the 72.7% BS underpricing from KB #282 is mainly OTM where time value dominates)
  - Max loss = premium paid (defined risk, Level 2 compatible)
  - $645 account: each sector ETF is ~$40-80, deep ITM call costs ~$300-500/contract
  - Monthly rotation (matching the validated equity rotation frequency)

SIX VARIANTS:
  A: Top-1 Deep ITM (90% delta, DTE 60) — concentrated, one position
  B: Top-2 Deep ITM (90% delta, DTE 60) — split capital between top 2
  C: Top-1 Moderate ITM (80% delta, DTE 45) — cheaper premium, more leverage
  D: Top-1 Deep ITM with trailing stop (90% delta, DTE 60, 15% trailing)
  E: Top-1 with VIX filter (no entry when VIX > 25)
  F: Top-2 with momentum filter (only enter if 21d momentum > 0)

PRICING MODEL:
  - Use BS for deep ITM, with adjustments:
  - Intrinsic value = max(0, S - K) — this is exact, no model needed
  - Time value estimated via BS, then apply calibration factor for realism
  - For deep ITM: time value is typically 5-15% of total premium, so even
    100% error on time value = only 5-15% error on total cost
  - Compare with "worst case" pricing variant that adds 20% to BS premium

VALIDATION:
  - 5 gates: Sharpe>1, perm p<0.05, WR>40%, regime balance, random baseline
  - ALL OOT days (40+), regime-stratified (HC #428)
  - Permutation test: 200 shuffles
  - Random direction baseline comparison

Output: output/growth_research/deep_itm_equity_rotation_v1/
MLflow experiment: deep_itm_equity_rotation_v1
"""

import json
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

_builtin_print = print
def fprint(*args, **kwargs):
    _builtin_print(*args, **kwargs)
    sys.stdout.flush()

# ── Detect environment ──
_JUPITER_BASE = Path("/home/jupiter/Lvl3Quant")
_NEPTUNE_BASE = Path("/home/nick/Lvl3Quant")
BASE = _NEPTUNE_BASE if _NEPTUNE_BASE.exists() else _JUPITER_BASE
fprint(f"Running on: {BASE}")
sys.path.insert(0, str(BASE))

OUTPUT_DIR = BASE / "output" / "growth_research" / "deep_itm_equity_rotation_v1"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── MLflow setup ──
MLFLOW_URI = "http://jupiter:5000"
EXPERIMENT_NAME = "deep_itm_equity_rotation_v1"
MLFLOW_OK = False
try:
    import mlflow
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    MLFLOW_OK = True
    fprint(f"MLflow OK: {MLFLOW_URI}")
except Exception as e:
    fprint(f"MLflow not available: {e}")

# ── LightGBM ──
try:
    import lightgbm as lgb
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False
    fprint("WARNING: LightGBM not available. Using simple momentum ranking.")

# ==================== CONFIG ====================
SECTORS = ['XLK', 'XLF', 'XLE', 'XLV', 'XLY', 'XLP', 'XLI', 'XLB', 'XLU', 'XLRE', 'XLC']
INITIAL_CAPITAL = 645.0
RISK_FREE_RATE = 0.05  # For BS pricing

FEAT_COLS = [
    'ret_5d', 'ret_10d', 'ret_21d', 'ret_63d', 'ret_126d', 'ret_252d',
    'vol_21d', 'vol_63d', 'sharpe_63d', 'maxdd_63d', 'pct_52w_high', 'mom_accel',
    'pct_pos_months_12m', 'sortino_63d', 'calmar_1y',
    'trend_r2_63d', 'trend_slope_63d',
]

# ==================== BS PRICING FOR DEEP ITM CALLS ====================

def bs_call_price(S, K, T, r, sigma):
    """Black-Scholes call price."""
    from scipy.stats import norm
    if T <= 0 or sigma <= 0:
        return max(0, S - K)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

def bs_delta(S, K, T, r, sigma):
    """Black-Scholes call delta."""
    from scipy.stats import norm
    if T <= 0 or sigma <= 0:
        return 1.0 if S > K else 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return norm.cdf(d1)

def price_deep_itm_call(stock_price, target_delta, dte_days, realized_vol,
                         pricing_adjustment=1.0):
    """
    Price a deep ITM call option targeting a specific delta.

    Strategy: find the strike that gives target_delta, then compute premium.
    For deep ITM calls, premium ≈ intrinsic + small time value.

    Args:
        stock_price: current stock price
        target_delta: target delta (e.g., 0.90)
        dte_days: days to expiration
        realized_vol: annualized realized volatility
        pricing_adjustment: multiplier for total premium (1.0 = BS, 1.2 = conservative)

    Returns:
        dict with strike, premium, delta, intrinsic, time_value, leverage
    """
    T = dte_days / 365.0
    sigma = max(realized_vol, 0.10)  # Floor at 10% vol

    # Binary search for strike that gives target delta
    # For deep ITM calls (high delta), strike < stock_price
    lo, hi = stock_price * 0.50, stock_price * 1.10
    for _ in range(50):
        mid = (lo + hi) / 2
        d = bs_delta(stock_price, mid, T, RISK_FREE_RATE, sigma)
        if d > target_delta:
            lo = mid  # Need higher strike (lower delta)
        else:
            hi = mid  # Need lower strike (higher delta)

    strike = (lo + hi) / 2
    # Round strike to nearest $0.50 (typical ETF option strikes)
    strike = round(strike * 2) / 2

    premium = bs_call_price(stock_price, strike, T, RISK_FREE_RATE, sigma)
    delta = bs_delta(stock_price, strike, T, RISK_FREE_RATE, sigma)
    intrinsic = max(0, stock_price - strike)
    time_value = premium - intrinsic

    # Apply pricing adjustment (conservative = higher premium)
    premium *= pricing_adjustment

    # Per-share premium; multiply by 100 for per-contract
    contract_cost = premium * 100

    # Effective leverage: delta * 100 shares * stock_price / contract_cost
    leverage = (delta * 100 * stock_price) / contract_cost if contract_cost > 0 else 0

    return {
        'strike': strike,
        'premium_per_share': premium,
        'contract_cost': contract_cost,
        'delta': delta,
        'intrinsic': intrinsic,
        'time_value': time_value,
        'time_value_pct': time_value / premium if premium > 0 else 0,
        'leverage': leverage,
        'T': T,
        'sigma': sigma,
    }


# ==================== DATA DOWNLOAD ====================

def download_data():
    """Download sector ETF + SPY + VIX data via yfinance."""
    import yfinance as yf
    all_tickers = SECTORS + ['SPY', '^VIX']
    fprint(f"Downloading {len(all_tickers)} tickers...")
    raw = yf.download(all_tickers, start='2020-01-01', progress=False)
    mi = isinstance(raw.columns, pd.MultiIndex)
    close = raw['Close'] if mi else raw
    if isinstance(close.columns, pd.MultiIndex):
        close.columns = close.columns.get_level_values(-1)
    close = close.ffill()
    rename_map = {'^VIX': 'VIX'}
    close = close.rename(columns=rename_map)
    vc = 'VIX' if 'VIX' in close.columns else ('^VIX' if '^VIX' in close.columns else None)
    if vc is None:
        raise ValueError("VIX data not available")
    vix = close[vc].dropna()
    spy = close['SPY'].dropna()
    sc = close[[c for c in SECTORS if c in close.columns]].dropna(how='all')
    ix = sc.index.intersection(vix.index).intersection(spy.index)
    fprint(f"Data: {len(ix)} trading days, {ix[0].strftime('%Y-%m-%d')} to {ix[-1].strftime('%Y-%m-%d')}")
    return sc.loc[ix], spy.loc[ix], vix.loc[ix]


# ==================== FEATURE ENGINEERING (identical to V10/equity rotation) ====================

def compute_features(px):
    """Compute 17 momentum features for a single sector ETF."""
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
    cagr = float(px.iloc[-1] / px.iloc[-252] - 1) if len(px) >= 252 else 0.0
    f['calmar_1y'] = cagr / (abs(mdd) + 1e-10)
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


# ==================== LGBM RANKING ====================

def run_lgbm_ranking_wf(sc, idx_end, train_window=60):
    """Walk-forward LGBM ranking (same as equity rotation V1)."""
    if not HAS_LGBM:
        rets_21d = sc.iloc[:idx_end + 1].pct_change(21).iloc[-1]
        return dict(rets_21d.sort_values(ascending=False))

    records = []
    start_i = max(260, idx_end - 400)
    all_idx = list(range(start_i, idx_end))
    rebal_idx = all_idx[::20]

    for i in rebal_idx[:-1]:
        for tk in sc.columns:
            px = sc[tk].iloc[:i + 1].dropna()
            feats = compute_features(px)
            if not feats:
                continue
            fi = min(i + 28, len(sc) - 1)
            feats.update({
                'date_idx': i, 'ticker': tk,
                'fwd_ret': float(sc[tk].iloc[fi] / sc[tk].iloc[i] - 1)
            })
            records.append(feats)

    df = pd.DataFrame(records)
    for c in FEAT_COLS:
        if c not in df.columns:
            df[c] = 0.0
    df[FEAT_COLS] = df[FEAT_COLS].fillna(0.0)

    if len(df) < 50:
        rets_21d = sc.iloc[:idx_end + 1].pct_change(21).iloc[-1]
        return dict(rets_21d.sort_values(ascending=False))

    df['rank_label'] = df.groupby('date_idx')['fwd_ret'].rank(pct=True)
    X_train = np.nan_to_num(df[FEAT_COLS].values.astype(np.float32))
    y_train = df['rank_label'].values.astype(np.float32)

    m = lgb.LGBMRegressor(
        n_estimators=100, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_samples=5, verbose=-1
    )
    m.fit(X_train, y_train)

    current_feats = {}
    for tk in sc.columns:
        px = sc[tk].iloc[:idx_end + 1].dropna()
        feats = compute_features(px)
        if feats:
            current_feats[tk] = feats

    if not current_feats:
        return {}

    pred_df = pd.DataFrame(current_feats).T
    for c in FEAT_COLS:
        if c not in pred_df.columns:
            pred_df[c] = 0.0
    X_pred = np.nan_to_num(pred_df[FEAT_COLS].values.astype(np.float32))
    scores = m.predict(X_pred)
    return dict(zip(pred_df.index, scores))


# ==================== BACKTEST ENGINE ====================

def run_variant(sc, spy, vix, variant, verbose=True):
    """
    Run a single deep ITM call rotation backtest variant.

    Variants:
        A: Top-1 Deep ITM (90% delta, DTE 60) — concentrated
        B: Top-2 Deep ITM (90% delta, DTE 60) — diversified
        C: Top-1 Moderate ITM (80% delta, DTE 45) — cheaper, more leverage
        D: Top-1 Deep ITM + 15% trailing stop
        E: Top-1 Deep ITM + VIX filter (no entry VIX > 25)
        F: Top-2 Deep ITM + momentum filter (21d mom > 0)
    """
    # Variant config
    configs = {
        'A': {'n_picks': 1, 'target_delta': 0.90, 'dte': 60, 'trailing_stop': None, 'vix_filter': None, 'mom_filter': False, 'pricing_adj': 1.0},
        'B': {'n_picks': 2, 'target_delta': 0.90, 'dte': 60, 'trailing_stop': None, 'vix_filter': None, 'mom_filter': False, 'pricing_adj': 1.0},
        'C': {'n_picks': 1, 'target_delta': 0.80, 'dte': 45, 'trailing_stop': None, 'vix_filter': None, 'mom_filter': False, 'pricing_adj': 1.0},
        'D': {'n_picks': 1, 'target_delta': 0.90, 'dte': 60, 'trailing_stop': 0.15, 'vix_filter': None, 'mom_filter': False, 'pricing_adj': 1.0},
        'E': {'n_picks': 1, 'target_delta': 0.90, 'dte': 60, 'trailing_stop': None, 'vix_filter': 25.0, 'mom_filter': False, 'pricing_adj': 1.0},
        'F': {'n_picks': 2, 'target_delta': 0.90, 'dte': 60, 'trailing_stop': None, 'vix_filter': None, 'mom_filter': True, 'pricing_adj': 1.0},
    }
    cfg = configs[variant]

    dates = sc.index
    n_days = len(dates)
    start_idx = 280

    equity = INITIAL_CAPITAL
    peak_equity = equity
    equity_curve = []
    positions = []  # list of active positions
    closed_trades = []
    last_rebalance_idx = None
    rebalance_interval = 28  # Monthly

    for day_idx in range(start_idx, n_days):
        today = dates[day_idx]

        # ── Mark-to-market existing positions ──
        daily_pnl = 0.0
        expired_positions = []

        for i, pos in enumerate(positions):
            tk = pos['ticker']
            if tk not in sc.columns:
                continue

            current_price = float(sc[tk].iloc[day_idx])
            days_held = (today - pos['entry_date']).days
            remaining_dte = pos['dte'] - days_held

            # Current option value: intrinsic + time value (decaying)
            if remaining_dte <= 0:
                # Expired: value = intrinsic only
                option_value = max(0, current_price - pos['strike']) * 100
                expired_positions.append(i)
            else:
                # Use BS to reprice
                vol = pos['sigma']
                option_value = bs_call_price(current_price, pos['strike'],
                                              remaining_dte/365.0, RISK_FREE_RATE, vol) * 100

            pos_pnl = option_value - pos['contract_cost']
            daily_pnl = pos_pnl  # This is cumulative P&L on position
            pos['current_value'] = option_value
            pos['current_pnl'] = pos_pnl
            pos['current_pnl_pct'] = pos_pnl / pos['contract_cost'] if pos['contract_cost'] > 0 else 0
            pos['peak_value'] = max(pos.get('peak_value', option_value), option_value)

            # ── Trailing stop check ──
            if cfg['trailing_stop'] is not None and pos['peak_value'] > 0:
                drawdown_from_peak = 1 - option_value / pos['peak_value']
                if drawdown_from_peak > cfg['trailing_stop']:
                    expired_positions.append(i)

        # ── Close expired/stopped positions ──
        for i in sorted(set(expired_positions), reverse=True):
            pos = positions.pop(i)
            closed_trades.append({
                'ticker': pos['ticker'],
                'entry_date': pos['entry_date'],
                'exit_date': today,
                'strike': pos['strike'],
                'delta': pos['delta'],
                'entry_cost': pos['contract_cost'],
                'exit_value': pos.get('current_value', 0),
                'pnl': pos.get('current_pnl', 0),
                'pnl_pct': pos.get('current_pnl_pct', 0),
                'days_held': (today - pos['entry_date']).days,
                'reason': 'expired' if (today - pos['entry_date']).days >= pos['dte'] else 'trailing_stop',
            })
            equity += pos.get('current_pnl', 0)

        # ── Rebalance check ──
        should_rebalance = False
        if last_rebalance_idx is None:
            should_rebalance = True
        elif day_idx - last_rebalance_idx >= rebalance_interval:
            should_rebalance = True

        if should_rebalance and day_idx < n_days - 5:  # Don't open near end
            # Get rankings
            ranks = run_lgbm_ranking_wf(sc, day_idx)
            if not ranks:
                equity_curve.append({'date': today, 'equity': equity})
                continue

            sorted_tickers = sorted(ranks.keys(), key=lambda t: ranks[t], reverse=True)

            # ── Apply filters ──
            # VIX filter
            if cfg['vix_filter'] is not None:
                current_vix = float(vix.iloc[day_idx]) if day_idx < len(vix) else 20
                if current_vix > cfg['vix_filter']:
                    # Don't open new positions, but keep existing ones
                    equity_curve.append({'date': today, 'equity': equity})
                    continue

            # Momentum filter
            if cfg['mom_filter']:
                filtered = []
                for tk in sorted_tickers:
                    if day_idx >= 21:
                        mom = float(sc[tk].iloc[day_idx] / sc[tk].iloc[day_idx - 21] - 1)
                        if mom > 0:
                            filtered.append(tk)
                    else:
                        filtered.append(tk)
                sorted_tickers = filtered

            # Close all existing positions before rebalancing
            for pos in positions:
                closed_trades.append({
                    'ticker': pos['ticker'],
                    'entry_date': pos['entry_date'],
                    'exit_date': today,
                    'strike': pos['strike'],
                    'delta': pos['delta'],
                    'entry_cost': pos['contract_cost'],
                    'exit_value': pos.get('current_value', 0),
                    'pnl': pos.get('current_pnl', 0),
                    'pnl_pct': pos.get('current_pnl_pct', 0),
                    'days_held': (today - pos['entry_date']).days,
                    'reason': 'rebalance',
                })
                equity += pos.get('current_pnl', 0)
            positions = []

            # Open new positions
            n_picks = min(cfg['n_picks'], len(sorted_tickers))
            if n_picks == 0:
                equity_curve.append({'date': today, 'equity': equity})
                continue

            capital_per_position = equity / n_picks

            for tk in sorted_tickers[:n_picks]:
                stock_price = float(sc[tk].iloc[day_idx])
                # Get realized vol for pricing
                rets = sc[tk].iloc[max(0, day_idx-63):day_idx].pct_change().dropna()
                realized_vol = float(rets.std() * np.sqrt(252)) if len(rets) > 10 else 0.20

                opt = price_deep_itm_call(
                    stock_price=stock_price,
                    target_delta=cfg['target_delta'],
                    dte_days=cfg['dte'],
                    realized_vol=realized_vol,
                    pricing_adjustment=cfg['pricing_adj'],
                )

                # Can we afford this contract?
                if opt['contract_cost'] > capital_per_position:
                    # Can't afford a full contract — skip
                    # In reality with fractional options this wouldn't happen,
                    # but standard options require 100 shares per contract
                    if verbose and day_idx == start_idx:
                        fprint(f"  {variant}: Can't afford {tk} contract (${opt['contract_cost']:.0f} > ${capital_per_position:.0f})")
                    continue

                n_contracts = int(capital_per_position / opt['contract_cost'])
                if n_contracts < 1:
                    continue
                n_contracts = 1  # Cap at 1 contract per position for risk mgmt

                total_cost = opt['contract_cost'] * n_contracts

                positions.append({
                    'ticker': tk,
                    'entry_date': today,
                    'strike': opt['strike'],
                    'delta': opt['delta'],
                    'premium_per_share': opt['premium_per_share'],
                    'contract_cost': total_cost,
                    'n_contracts': n_contracts,
                    'sigma': opt['sigma'],
                    'dte': cfg['dte'],
                    'entry_stock_price': stock_price,
                    'current_value': total_cost,
                    'current_pnl': 0,
                    'current_pnl_pct': 0,
                    'peak_value': total_cost,
                    'leverage': opt['leverage'],
                    'time_value_pct': opt['time_value_pct'],
                })

            last_rebalance_idx = day_idx

        # Record equity (mark-to-market)
        total_position_value = sum(p.get('current_value', 0) for p in positions)
        total_invested = sum(p['contract_cost'] for p in positions)
        cash = equity - total_invested + sum(p.get('current_pnl', 0) for p in positions)
        # Actually, equity tracking:
        # equity = cash + sum(current_values)
        # After opening: cash = equity_before - total_cost_of_new_positions
        # Mark-to-market equity = cash + sum(current_option_values)
        mtm_equity = equity + sum(p.get('current_pnl', 0) for p in positions)

        equity_curve.append({'date': today, 'equity': mtm_equity})

        if verbose and day_idx == start_idx:
            fprint(f"  {variant}: Start equity ${equity:.0f}, positions: {len(positions)}")

    # Close any remaining positions at end
    for pos in positions:
        closed_trades.append({
            'ticker': pos['ticker'],
            'entry_date': pos['entry_date'],
            'exit_date': dates[-1],
            'strike': pos['strike'],
            'delta': pos['delta'],
            'entry_cost': pos['contract_cost'],
            'exit_value': pos.get('current_value', 0),
            'pnl': pos.get('current_pnl', 0),
            'pnl_pct': pos.get('current_pnl_pct', 0),
            'days_held': (dates[-1] - pos['entry_date']).days,
            'reason': 'end',
        })
        equity += pos.get('current_pnl', 0)

    # Build equity curve DataFrame
    eq_df = pd.DataFrame(equity_curve)
    if eq_df.empty:
        return None
    eq_df['date'] = pd.to_datetime(eq_df['date'])
    eq_df = eq_df.set_index('date')

    return {
        'equity_curve': eq_df,
        'closed_trades': closed_trades,
        'variant': variant,
        'config': cfg,
    }


# ==================== METRICS ====================

def compute_metrics(result, spy):
    """Compute risk-adjusted metrics from backtest result."""
    if result is None:
        return None

    eq = result['equity_curve']['equity']
    trades = result['closed_trades']

    if len(eq) < 20:
        return None

    # Daily returns
    daily_rets = eq.pct_change().dropna()
    if len(daily_rets) < 10:
        return None

    # Core metrics
    total_ret = float(eq.iloc[-1] / eq.iloc[0] - 1)
    n_years = len(daily_rets) / 252
    cagr = float((eq.iloc[-1] / eq.iloc[0]) ** (1/n_years) - 1) if n_years > 0 else 0

    ann_ret = float(daily_rets.mean() * 252)
    ann_vol = float(daily_rets.std() * np.sqrt(252))
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0

    down_rets = daily_rets[daily_rets < 0]
    down_vol = float(down_rets.std() * np.sqrt(252)) if len(down_rets) > 0 else 0.001
    sortino = ann_ret / down_vol

    # Max drawdown
    cummax = eq.cummax()
    drawdown = (eq - cummax) / cummax
    max_dd = float(drawdown.min())

    # Win rate from trades
    if trades:
        wins = sum(1 for t in trades if t['pnl'] > 0)
        wr = wins / len(trades) if trades else 0
        avg_win = np.mean([t['pnl'] for t in trades if t['pnl'] > 0]) if wins > 0 else 0
        avg_loss = np.mean([abs(t['pnl']) for t in trades if t['pnl'] <= 0]) if (len(trades) - wins) > 0 else 0.001
        pf = avg_win * wins / (avg_loss * (len(trades) - wins)) if (len(trades) - wins) > 0 and avg_loss > 0 else 999
    else:
        wr, pf = 0, 0

    # Calmar
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0

    # SPY alpha
    spy_aligned = spy.reindex(eq.index).ffill()
    if len(spy_aligned.dropna()) > 20:
        spy_ret = float(spy_aligned.iloc[-1] / spy_aligned.iloc[0] - 1)
        alpha = total_ret - spy_ret
    else:
        spy_ret = 0
        alpha = total_ret

    # Average leverage and time value %
    avg_leverage = np.mean([t.get('leverage', 1) for t in trades]) if trades else 0

    return {
        'sharpe': round(sharpe, 3),
        'sortino': round(sortino, 3),
        'cagr': round(cagr * 100, 1),
        'total_return': round(total_ret * 100, 1),
        'max_dd': round(max_dd * 100, 1),
        'win_rate': round(wr * 100, 1),
        'profit_factor': round(min(pf, 99), 2),
        'calmar': round(calmar, 2),
        'n_trades': len(trades),
        'alpha_vs_spy': round(alpha * 100, 1),
        'final_equity': round(float(eq.iloc[-1]), 2),
        'ann_vol': round(ann_vol * 100, 1),
    }


# ==================== PERMUTATION TEST ====================

def run_permutation_test(sc, spy, vix, variant, n_shuffles=200):
    """Shuffle sector rankings randomly and compare to real strategy."""
    fprint(f"  Permutation test ({n_shuffles} shuffles)...")

    # Get real result
    real_result = run_variant(sc, spy, vix, variant, verbose=False)
    if real_result is None:
        return None, None, None
    real_metrics = compute_metrics(real_result, spy)
    if real_metrics is None:
        return None, None, None
    real_sharpe = real_metrics['sharpe']

    # Run shuffled versions
    random_sharpes = []
    for i in range(n_shuffles):
        # Temporarily replace LGBM with random ranking
        np.random.seed(i + 42)
        result = run_variant_random(sc, spy, vix, variant)
        if result is None:
            continue
        metrics = compute_metrics(result, spy)
        if metrics:
            random_sharpes.append(metrics['sharpe'])

    if not random_sharpes:
        return real_sharpe, None, None

    p_value = sum(1 for s in random_sharpes if s >= real_sharpe) / len(random_sharpes)
    mean_random = np.mean(random_sharpes)

    return real_sharpe, p_value, mean_random


def run_variant_random(sc, spy, vix, variant):
    """Run variant with random sector rankings instead of LGBM."""
    # Monkey-patch the ranking function temporarily
    global _USE_RANDOM_RANKING
    _USE_RANDOM_RANKING = True
    result = run_variant(sc, spy, vix, variant, verbose=False)
    _USE_RANDOM_RANKING = False
    return result

_USE_RANDOM_RANKING = False

# Override the ranking function to support random mode
_original_lgbm_ranking = run_lgbm_ranking_wf

def run_lgbm_ranking_wf_wrapped(sc, idx_end, train_window=60):
    if _USE_RANDOM_RANKING:
        scores = np.random.randn(len(sc.columns))
        return dict(zip(sc.columns, scores))
    return _original_lgbm_ranking(sc, idx_end, train_window)

# Replace
run_lgbm_ranking_wf = run_lgbm_ranking_wf_wrapped


# ==================== REGIME ANALYSIS ====================

def regime_analysis(result, spy):
    """Classify days as green/red/flat and compute per-regime metrics."""
    if result is None:
        return None

    eq = result['equity_curve']['equity']
    spy_aligned = spy.reindex(eq.index).ffill().dropna()

    # Classify by SPY daily return
    spy_rets = spy_aligned.pct_change().dropna()
    eq_rets = eq.pct_change().dropna()

    common_idx = spy_rets.index.intersection(eq_rets.index)
    spy_rets = spy_rets.loc[common_idx]
    eq_rets = eq_rets.loc[common_idx]

    green = spy_rets > 0.001
    red = spy_rets < -0.001
    flat = ~green & ~red

    regime_metrics = {}
    for name, mask in [('green', green), ('red', red), ('flat', flat)]:
        r = eq_rets[mask]
        if len(r) > 5:
            ann_ret = float(r.mean() * 252)
            ann_vol = float(r.std() * np.sqrt(252)) if r.std() > 0 else 0.001
            regime_metrics[name] = {
                'sharpe': round(ann_ret / ann_vol, 3),
                'n_days': int(mask.sum()),
                'mean_ret': round(float(r.mean()) * 10000, 2),  # bps
            }
        else:
            regime_metrics[name] = {'sharpe': 0, 'n_days': 0, 'mean_ret': 0}

    # Regime balance check (HC #428)
    g_sharpe = abs(regime_metrics['green']['sharpe'])
    r_sharpe = abs(regime_metrics['red']['sharpe'])
    max_sharpe = max(g_sharpe, r_sharpe)
    gap = abs(g_sharpe - r_sharpe) / max_sharpe if max_sharpe > 0 else 0
    regime_metrics['gap_ratio'] = round(gap, 3)
    regime_metrics['regime_pass'] = gap <= 0.50

    return regime_metrics


# ==================== MAIN ====================

def main():
    t0 = time.time()
    fprint("=" * 70)
    fprint("DEEP ITM CALL EQUITY ROTATION V1")
    fprint("=" * 70)
    fprint(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    fprint(f"Capital: ${INITIAL_CAPITAL}")
    fprint()

    # Download data
    sc, spy, vix = download_data()

    all_results = {}
    all_metrics = {}

    variants = ['A', 'B', 'C', 'D', 'E', 'F']

    for v in variants:
        fprint(f"\n{'─'*50}")
        fprint(f"VARIANT {v}")
        fprint(f"{'─'*50}")

        # Run backtest
        result = run_variant(sc, spy, vix, v, verbose=True)

        if result is None:
            fprint(f"  {v}: No result (insufficient data or no trades)")
            all_metrics[v] = None
            continue

        metrics = compute_metrics(result, spy)
        if metrics is None:
            fprint(f"  {v}: Could not compute metrics")
            all_metrics[v] = None
            continue

        fprint(f"  Sharpe: {metrics['sharpe']}, Sortino: {metrics['sortino']}, "
               f"CAGR: {metrics['cagr']}%, MDD: {metrics['max_dd']}%")
        fprint(f"  WR: {metrics['win_rate']}%, PF: {metrics['profit_factor']}, "
               f"Trades: {metrics['n_trades']}, Final: ${metrics['final_equity']:.0f}")
        fprint(f"  Alpha vs SPY: {metrics['alpha_vs_spy']}%")

        # Regime analysis
        regime = regime_analysis(result, spy)
        if regime:
            fprint(f"  Regime: Green Sharpe={regime['green']['sharpe']}, "
                   f"Red Sharpe={regime['red']['sharpe']}, Gap={regime['gap_ratio']}")
            metrics['regime'] = regime
            metrics['regime_pass'] = regime['regime_pass']
        else:
            metrics['regime_pass'] = False

        # Permutation test
        real_s, p_val, mean_rand = run_permutation_test(sc, spy, vix, v, n_shuffles=200)
        if p_val is not None:
            fprint(f"  Permutation: p={p_val:.4f}, real={real_s:.3f}, random_mean={mean_rand:.3f}")
            metrics['perm_p'] = round(p_val, 4)
            metrics['random_sharpe'] = round(mean_rand, 3)
        else:
            metrics['perm_p'] = 1.0
            metrics['random_sharpe'] = 0

        # 5-gate validation
        gates = {
            'sharpe_gt_1': metrics['sharpe'] > 1.0,
            'perm_significant': metrics.get('perm_p', 1.0) < 0.05,
            'wr_gt_40': metrics['win_rate'] > 40,
            'regime_balance': metrics.get('regime_pass', False),
            'beats_random': metrics['sharpe'] > metrics.get('random_sharpe', 0) * 1.5,
        }
        gates_passed = sum(gates.values())
        metrics['gates'] = gates
        metrics['gates_passed'] = gates_passed
        metrics['gates_total'] = len(gates)

        fprint(f"  GATES: {gates_passed}/{len(gates)} — "
               + ", ".join(f"{'✅' if v else '❌'} {k}" for k, v in gates.items()))

        all_results[v] = result
        all_metrics[v] = metrics

        # Log to MLflow
        if MLFLOW_OK:
            try:
                with mlflow.start_run(run_name=f"variant_{v}"):
                    mlflow.log_params({
                        'variant': v,
                        'n_picks': result['config']['n_picks'],
                        'target_delta': result['config']['target_delta'],
                        'dte': result['config']['dte'],
                        'trailing_stop': str(result['config']['trailing_stop']),
                        'pricing_adj': result['config']['pricing_adj'],
                    })
                    for mk, mv in metrics.items():
                        if isinstance(mv, (int, float)):
                            mlflow.log_metric(mk, mv)
                    mlflow.log_metric('gates_passed', gates_passed)
            except Exception as e:
                fprint(f"  MLflow log error: {e}")

    # ── PRICING INSIGHT: How much does ITM time value actually matter? ──
    fprint(f"\n{'='*70}")
    fprint("PRICING INSIGHT: TIME VALUE ANALYSIS")
    fprint(f"{'='*70}")

    # Analyze how much time value contributes to deep ITM costs
    for v, result in all_results.items():
        if result and result['closed_trades']:
            trades = result['closed_trades']
            # We can estimate time value % from the pricing at entry
            fprint(f"  Variant {v}: {len(trades)} trades")

    # ── SUMMARY ──
    fprint(f"\n{'='*70}")
    fprint("SUMMARY")
    fprint(f"{'='*70}")

    best_sharpe = -999
    best_variant = None

    for v in variants:
        m = all_metrics.get(v)
        if m is None:
            fprint(f"  {v}: NO RESULT")
            continue
        fprint(f"  {v}: Sharpe {m['sharpe']}, Sortino {m['sortino']}, "
               f"CAGR {m['cagr']}%, MDD {m['max_dd']}%, WR {m['win_rate']}%, "
               f"Trades {m['n_trades']}, Gates {m['gates_passed']}/{m['gates_total']}, "
               f"Final ${m['final_equity']:.0f}")

        if m['sharpe'] > best_sharpe:
            best_sharpe = m['sharpe']
            best_variant = v

    fprint(f"\n  BEST: Variant {best_variant} (Sharpe {best_sharpe})")

    # Key question: can we actually afford the contracts?
    fprint(f"\n  KEY QUESTION: With ${INITIAL_CAPITAL} capital, can we afford deep ITM contracts?")
    for v, result in all_results.items():
        if result and result['closed_trades']:
            costs = [t['entry_cost'] for t in result['closed_trades']]
            fprint(f"    {v}: Avg contract cost ${np.mean(costs):.0f}, "
                   f"Min ${np.min(costs):.0f}, Max ${np.max(costs):.0f}")

    elapsed = time.time() - t0
    fprint(f"\nTotal runtime: {elapsed:.0f}s ({elapsed/60:.1f} min)")

    # Save results
    summary = {
        'experiment': EXPERIMENT_NAME,
        'timestamp': datetime.now().isoformat(),
        'capital': INITIAL_CAPITAL,
        'runtime_s': round(elapsed, 1),
        'variants': {},
    }
    for v in variants:
        m = all_metrics.get(v)
        if m:
            # Remove non-serializable items
            m_clean = {k: v for k, v in m.items() if k != 'gates'}
            m_clean['gates'] = {k: bool(v) for k, v in m.get('gates', {}).items()}
            summary['variants'][v] = m_clean

    with open(OUTPUT_DIR / 'results.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    fprint(f"Results saved to {OUTPUT_DIR / 'results.json'}")

    return all_metrics


if __name__ == '__main__':
    main()
