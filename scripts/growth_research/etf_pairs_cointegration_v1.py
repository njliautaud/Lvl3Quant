#!/usr/bin/env python3
"""
ETF Pairs Cointegration V1 (2026-07-28)
========================================

HYPOTHESIS: Mean reversion via cointegrated ETF pairs. When historically
correlated ETFs diverge, bet on convergence (long underperformer, short
outperformer, dollar neutral).

UNIVERSE (10 structurally related pairs):
  1. XLK/QQQ   (tech overlap)
  2. EWJ/EWY   (Japan/Korea export economies)
  3. GLD/GDX   (gold vs gold miners)
  4. XLE/OIH   (energy vs oil services)
  5. XLF/KRE   (financials vs regional banks)
  6. XBI/IBB   (biotech equal vs cap weight)
  7. EEM/VWO   (emerging markets duplication)
  8. SPY/IVV   (S&P 500 exact duplication)
  9. XLU/VPU   (utilities overlap)
  10. TLT/IEF  (long vs intermediate treasuries)

APPROACH: Z-score of log(price_A/price_B) vs rolling mean/std.
Entry |z|>2, exit z crosses 0 or stop |z|>3.5, max hold 30 days.

SIX VARIANTS:
  A: Baseline (z>2 entry, z<0.5 exit, 63d lookback)
  B: Faster (z>1.5 entry, z<0.3 exit, 21d lookback)
  C: Slower (z>2.5 entry, z<0.5 exit, 126d lookback)
  D: LGBM enhancement (predict which pairs will converge)
  E: Top-3 pairs simultaneously (portfolio of pairs)
  F: VIX filter (no new trades when VIX>25)

VALIDATION: 5 gates (Sharpe>1, perm p<0.05, WR>40%, regime gap<0.50, MC CI>0)
Walk-forward: 252-day sliding calibration window.
Capital: $645, data from 2015-01-01.

MLflow: etf_pairs_cointegration_v1, URI http://jupiter:5000
Output: output/growth_research/etf_pairs_cointegration_v1/
"""

import json, sys, time, warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

_builtin_print = print
def fprint(*args, **kwargs):
    _builtin_print(*args, **kwargs)
    sys.stdout.flush()

# ── Environment ──
_NEPTUNE_BASE = Path("/home/nick/Lvl3Quant")
_JUPITER_BASE = Path("/home/jupiter/Lvl3Quant")
BASE = _NEPTUNE_BASE if _NEPTUNE_BASE.exists() else _JUPITER_BASE
fprint(f"Running on: {BASE}")

OUTPUT_DIR = BASE / "output" / "growth_research" / "etf_pairs_cointegration_v1"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── MLflow ──
MLFLOW_URI = "http://jupiter:5000"
EXPERIMENT_NAME = "etf_pairs_cointegration_v1"
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
    fprint("WARNING: LightGBM not available, variant D will use fallback.")

# ==================== CONFIG ====================

PAIRS = [
    ('XLK', 'QQQ'), ('EWJ', 'EWY'), ('GLD', 'GDX'), ('XLE', 'OIH'),
    ('XLF', 'KRE'), ('XBI', 'IBB'), ('EEM', 'VWO'), ('SPY', 'IVV'),
    ('XLU', 'VPU'), ('TLT', 'IEF'),
]

ALL_TICKERS = sorted(set(t for p in PAIRS for t in p))
INITIAL_CAPITAL = 645.0
DATA_START = '2015-01-01'
CALIB_WINDOW = 252  # rolling calibration window (sliding, not expanding)
MAX_HOLD = 30       # max holding days

VARIANT_CFG = {
    'A': {'z_entry': 2.0, 'z_exit': 0.5, 'z_stop': 3.5, 'lookback': 63},
    'B': {'z_entry': 1.5, 'z_exit': 0.3, 'z_stop': 3.5, 'lookback': 21},
    'C': {'z_entry': 2.5, 'z_exit': 0.5, 'z_stop': 3.5, 'lookback': 126},
    'D': {'z_entry': 2.0, 'z_exit': 0.5, 'z_stop': 3.5, 'lookback': 63},
    'E': {'z_entry': 2.0, 'z_exit': 0.5, 'z_stop': 3.5, 'lookback': 63},
    'F': {'z_entry': 2.0, 'z_exit': 0.5, 'z_stop': 3.5, 'lookback': 63},
}


# ==================== DATA ====================

def download_data():
    import yfinance as yf
    tickers = ALL_TICKERS + ['^VIX']
    fprint(f"Downloading {len(tickers)} tickers from {DATA_START}...")
    raw = yf.download(tickers, start=DATA_START, progress=False)
    mi = isinstance(raw.columns, pd.MultiIndex)
    close = raw['Close'] if mi else raw
    if isinstance(close.columns, pd.MultiIndex):
        close.columns = close.columns.get_level_values(-1)
    close = close.ffill().rename(columns={'^VIX': 'VIX'})

    vix = close['VIX'].dropna() if 'VIX' in close.columns else pd.Series(dtype=float)
    spy = close['SPY'].dropna()

    avail = [t for t in ALL_TICKERS if t in close.columns]
    missing = [t for t in ALL_TICKERS if t not in close.columns]
    if missing:
        fprint(f"WARNING: Missing tickers: {missing}")

    prices = close[avail].dropna(how='all').ffill()
    ix = prices.index.intersection(spy.index)
    if len(vix) > 0:
        ix = ix.intersection(vix.index)
    prices = prices.loc[ix]
    spy = spy.loc[ix]
    vix = vix.loc[ix] if len(vix) > 0 else pd.Series(20.0, index=ix)

    valid_pairs = [(a, b) for a, b in PAIRS if a in prices.columns and b in prices.columns]
    fprint(f"Data: {ix[0].date()} to {ix[-1].date()}, {len(ix)} days, "
           f"{len(valid_pairs)}/{len(PAIRS)} pairs available")
    return prices, spy, vix, valid_pairs


# ==================== Z-SCORE COMPUTATION ====================

def compute_spread_zscore(prices, a, b, lookback):
    """Compute z-score of log(price_A/price_B) with rolling stats."""
    log_ratio = np.log(prices[a] / prices[b])
    roll_mean = log_ratio.rolling(lookback).mean()
    roll_std = log_ratio.rolling(lookback).std()
    z = (log_ratio - roll_mean) / (roll_std + 1e-10)
    return z, log_ratio


# ==================== LGBM CONVERGENCE PREDICTOR (Variant D) ====================

def build_lgbm_features(prices, a, b, idx):
    """Build features for a pair at a given index for LGBM prediction."""
    log_ratio = np.log(prices[a].iloc[:idx+1] / prices[b].iloc[:idx+1])
    if len(log_ratio) < 126:
        return None
    f = {}
    for lb in [21, 63, 126]:
        r = log_ratio.iloc[-lb:]
        mu, std = r.mean(), r.std()
        f[f'z_{lb}d'] = float((log_ratio.iloc[-1] - mu) / (std + 1e-10))
        f[f'spread_vol_{lb}d'] = float(std)
        f[f'spread_ret_{lb}d'] = float(log_ratio.iloc[-1] - log_ratio.iloc[-lb])

    # Individual asset features
    for tk, label in [(a, 'A'), (b, 'B')]:
        px = prices[tk].iloc[:idx+1]
        rets = px.pct_change().dropna()
        f[f'ret_21d_{label}'] = float(px.iloc[-1] / px.iloc[-21] - 1) if len(px) > 21 else 0
        f[f'vol_21d_{label}'] = float(rets.iloc[-21:].std() * np.sqrt(252)) if len(rets) > 21 else 0.2

    # Correlation between the two assets
    r_a = prices[a].iloc[max(0,idx-63):idx+1].pct_change().dropna()
    r_b = prices[b].iloc[max(0,idx-63):idx+1].pct_change().dropna()
    min_len = min(len(r_a), len(r_b))
    if min_len > 10:
        f['corr_63d'] = float(r_a.iloc[-min_len:].corr(r_b.iloc[-min_len:]))
    else:
        f['corr_63d'] = 0.5

    # Half-life of mean reversion (Ornstein-Uhlenbeck)
    if len(log_ratio) >= 63:
        lr = log_ratio.iloc[-63:]
        delta = lr.diff().dropna()
        lag = lr.shift(1).dropna()
        delta, lag = delta.align(lag, join='inner')
        if len(delta) > 5 and lag.std() > 1e-10:
            from scipy import stats as sp_stats
            slope, _, _, _, _ = sp_stats.linregress(lag.values, delta.values)
            f['half_life'] = float(-np.log(2) / slope) if slope < -0.001 else 999.0
        else:
            f['half_life'] = 999.0
    else:
        f['half_life'] = 999.0

    return f


LGBM_FEAT_COLS = [
    'z_21d', 'z_63d', 'z_126d',
    'spread_vol_21d', 'spread_vol_63d', 'spread_vol_126d',
    'spread_ret_21d', 'spread_ret_63d', 'spread_ret_126d',
    'ret_21d_A', 'ret_21d_B', 'vol_21d_A', 'vol_21d_B',
    'corr_63d', 'half_life',
]


def train_lgbm_convergence(prices, valid_pairs, idx, train_window=500):
    """Train LGBM to predict whether a diverged pair will converge within 30 days."""
    if not HAS_LGBM:
        return None
    records = []
    start_i = max(252, idx - train_window)
    sample_indices = list(range(start_i, idx, 5))

    for si in sample_indices:
        for a, b in valid_pairs:
            feats = build_lgbm_features(prices, a, b, si)
            if feats is None:
                continue
            # Label: did spread converge (z cross 0) within 30 days?
            log_ratio = np.log(prices[a].iloc[:min(si+31, len(prices))] /
                              prices[b].iloc[:min(si+31, len(prices))])
            if len(log_ratio) <= si:
                continue
            z_at_entry = feats['z_63d']
            future = log_ratio.iloc[si+1:si+31]
            if len(future) < 5:
                continue
            mu = log_ratio.iloc[max(0,si-63):si+1].mean()
            # Did it converge toward mean?
            entry_dist = abs(log_ratio.iloc[si] - mu)
            min_future_dist = abs(future - mu).min()
            converged = 1.0 if min_future_dist < entry_dist * 0.5 else 0.0
            feats['label'] = converged
            feats['pair'] = f"{a}_{b}"
            records.append(feats)

    if len(records) < 50:
        return None

    df = pd.DataFrame(records)
    for c in LGBM_FEAT_COLS:
        if c not in df.columns:
            df[c] = 0.0
    X = np.nan_to_num(df[LGBM_FEAT_COLS].values.astype(np.float32))
    y = df['label'].values.astype(np.float32)

    m = lgb.LGBMClassifier(
        n_estimators=80, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_samples=5, verbose=-1,
    )
    m.fit(X, y)
    return m


def lgbm_predict_convergence(model, prices, a, b, idx):
    """Predict probability of convergence for a pair."""
    if model is None:
        return 0.5
    feats = build_lgbm_features(prices, a, b, idx)
    if feats is None:
        return 0.5
    X = np.nan_to_num(np.array([[feats.get(c, 0.0) for c in LGBM_FEAT_COLS]], dtype=np.float32))
    return float(model.predict_proba(X)[0, 1])


# ==================== BACKTEST ENGINE ====================

def run_variant(prices, spy, vix, valid_pairs, variant, verbose=False):
    cfg = VARIANT_CFG[variant]
    z_entry = cfg['z_entry']
    z_exit = cfg['z_exit']
    z_stop = cfg['z_stop']
    lookback = cfg['lookback']

    dates = prices.index
    n_days = len(dates)
    start_idx = CALIB_WINDOW + lookback  # need calibration + lookback warm-up

    equity = INITIAL_CAPITAL
    equity_curve = []
    positions = []  # list of active position dicts
    closed_trades = []
    lgbm_model = None
    last_lgbm_train = 0

    # For variant E: allow up to 3 simultaneous positions
    max_positions = 3 if variant == 'E' else 1

    spy_entry_price = float(spy.iloc[start_idx])
    spy_shares = INITIAL_CAPITAL / spy_entry_price

    for day_idx in range(start_idx, n_days):
        today = dates[day_idx]

        # ── Check exits on existing positions ──
        to_close = []
        for i, pos in enumerate(positions):
            a, b = pos['pair']
            z, _ = compute_spread_zscore(prices.iloc[:day_idx+1], a, b, lookback)
            current_z = float(z.iloc[-1]) if not np.isnan(z.iloc[-1]) else 0.0
            hold_days = day_idx - pos['entry_idx']

            exit_reason = None
            if pos['direction'] == 'long_spread':
                # We went long spread (long A, short B) because z was very negative
                if current_z >= -z_exit:
                    exit_reason = 'convergence'
                elif current_z < -z_stop:
                    exit_reason = 'stop_loss'
            else:
                # We went short spread (short A, long B) because z was very positive
                if current_z <= z_exit:
                    exit_reason = 'convergence'
                elif current_z > z_stop:
                    exit_reason = 'stop_loss'

            if hold_days >= MAX_HOLD:
                exit_reason = 'max_hold'

            if exit_reason:
                to_close.append((i, exit_reason, current_z))

        for i, reason, exit_z in sorted(to_close, reverse=True):
            pos = positions.pop(i)
            a, b = pos['pair']
            price_a = float(prices[a].iloc[day_idx])
            price_b = float(prices[b].iloc[day_idx])

            if pos['direction'] == 'long_spread':
                pnl_a = pos['shares_a'] * (price_a - pos['entry_price_a'])
                pnl_b = pos['shares_b'] * (pos['entry_price_b'] - price_b)
            else:
                pnl_a = pos['shares_a'] * (pos['entry_price_a'] - price_a)
                pnl_b = pos['shares_b'] * (price_b - pos['entry_price_b'])
            pnl = pnl_a + pnl_b

            closed_trades.append({
                'pair': f"{a}/{b}", 'direction': pos['direction'],
                'exit_reason': reason,
                'entry_date': str(pos['entry_date'].date()),
                'exit_date': str(today.date()),
                'entry_z': round(pos['entry_z'], 3),
                'exit_z': round(exit_z, 3),
                'pnl': round(pnl, 4),
                'holding_days': day_idx - pos['entry_idx'],
            })

        # ── Mark-to-market ──
        mtm = 0.0
        for pos in positions:
            a, b = pos['pair']
            pa = float(prices[a].iloc[day_idx])
            pb = float(prices[b].iloc[day_idx])
            if pos['direction'] == 'long_spread':
                mtm += pos['shares_a'] * (pa - pos['entry_price_a'])
                mtm += pos['shares_b'] * (pos['entry_price_b'] - pb)
            else:
                mtm += pos['shares_a'] * (pos['entry_price_a'] - pa)
                mtm += pos['shares_b'] * (pb - pos['entry_price_b'])

        realized = sum(t['pnl'] for t in closed_trades)
        current_equity = INITIAL_CAPITAL + realized + mtm
        spy_eq = spy_shares * float(spy.iloc[day_idx])

        equity_curve.append({
            'date': today, 'equity': current_equity,
            'spy_equity': spy_eq, 'n_positions': len(positions),
        })

        # ── Entry signals ──
        if len(positions) >= max_positions:
            continue

        # VIX filter for variant F
        if variant == 'F' and float(vix.iloc[day_idx]) > 25:
            continue

        # Train LGBM periodically for variant D
        if variant == 'D' and day_idx - last_lgbm_train >= 63:
            lgbm_model = train_lgbm_convergence(prices, valid_pairs, day_idx)
            last_lgbm_train = day_idx

        # Score all pairs
        pair_signals = []
        for a, b in valid_pairs:
            if any(pos['pair'] == (a, b) for pos in positions):
                continue
            z, _ = compute_spread_zscore(prices.iloc[:day_idx+1], a, b, lookback)
            current_z = float(z.iloc[-1]) if not np.isnan(z.iloc[-1]) else 0.0

            if abs(current_z) < z_entry:
                continue

            # For variant D: filter by LGBM convergence probability
            if variant == 'D' and lgbm_model is not None:
                conv_prob = lgbm_predict_convergence(lgbm_model, prices, a, b, day_idx)
                if conv_prob < 0.55:
                    continue

            pair_signals.append((a, b, current_z, abs(current_z)))

        if not pair_signals:
            continue

        # Sort by absolute z-score (strongest divergence first)
        pair_signals.sort(key=lambda x: x[3], reverse=True)

        # For variant E, take up to 3; otherwise take top 1
        n_to_open = min(max_positions - len(positions), len(pair_signals))
        for a, b, current_z, _ in pair_signals[:n_to_open]:
            # Dollar neutral: equal dollar on each leg
            price_a = float(prices[a].iloc[day_idx])
            price_b = float(prices[b].iloc[day_idx])
            avail_capital = (INITIAL_CAPITAL + realized) / max_positions
            half_cap = avail_capital / 2.0
            shares_a = half_cap / price_a
            shares_b = half_cap / price_b

            if current_z > 0:
                direction = 'short_spread'  # A overperformed, short A long B
            else:
                direction = 'long_spread'   # A underperformed, long A short B

            positions.append({
                'pair': (a, b), 'direction': direction,
                'shares_a': shares_a, 'shares_b': shares_b,
                'entry_price_a': price_a, 'entry_price_b': price_b,
                'entry_date': today, 'entry_idx': day_idx,
                'entry_z': current_z,
            })

            if verbose and len(closed_trades) < 3:
                fprint(f"  OPEN {direction} {a}/{b} z={current_z:.2f} @ {today.date()}")

    # ── Close remaining positions ──
    for pos in positions:
        a, b = pos['pair']
        pa = float(prices[a].iloc[-1])
        pb = float(prices[b].iloc[-1])
        if pos['direction'] == 'long_spread':
            pnl = pos['shares_a'] * (pa - pos['entry_price_a']) + \
                  pos['shares_b'] * (pos['entry_price_b'] - pb)
        else:
            pnl = pos['shares_a'] * (pos['entry_price_a'] - pa) + \
                  pos['shares_b'] * (pb - pos['entry_price_b'])
        z, _ = compute_spread_zscore(prices, a, b, lookback)
        closed_trades.append({
            'pair': f"{a}/{b}", 'direction': pos['direction'],
            'exit_reason': 'end_of_backtest',
            'entry_date': str(pos['entry_date'].date()),
            'exit_date': str(dates[-1].date()),
            'entry_z': round(pos['entry_z'], 3),
            'exit_z': round(float(z.iloc[-1]), 3) if not np.isnan(z.iloc[-1]) else 0,
            'pnl': round(pnl, 4),
            'holding_days': len(dates) - 1 - pos['entry_idx'],
        })

    eq_df = pd.DataFrame(equity_curve)
    if len(eq_df) > 0:
        eq_df = eq_df.set_index('date')
        eq_df = eq_df[~eq_df.index.duplicated(keep='last')]

    metrics = compute_metrics(closed_trades, eq_df, variant)
    return {'trades': closed_trades, 'equity_curve': eq_df, 'metrics': metrics, 'variant': variant}


# ==================== METRICS ====================

def compute_metrics(trades, eq_df, variant_name):
    empty = {
        'variant': variant_name, 'n_trades': 0, 'sharpe': 0, 'sortino': 0,
        'pf': 0, 'wr': 0, 'mdd': 0, 'total_return': 0, 'cagr': 0,
        'total_pnl': 0, 'mean_pnl': 0, 'wins': 0, 'losses': 0,
        'spy_total_return': 0, 'spy_sharpe': 0, 'alpha_vs_spy': 0,
        'avg_hold': 0, 'convergence_rate': 0,
    }
    if not trades:
        return empty

    pnls = [t['pnl'] for t in trades]
    n = len(pnls)
    wins = sum(1 for p in pnls if p > 0)
    wr = wins / n if n > 0 else 0

    # Sharpe/Sortino from daily equity returns
    if len(eq_df) > 5:
        daily_rets = eq_df['equity'].pct_change().dropna().replace([np.inf, -np.inf], 0).fillna(0)
        sharpe = float(daily_rets.mean() / (daily_rets.std() + 1e-10) * np.sqrt(252))
        down = daily_rets[daily_rets < 0]
        sortino = float(daily_rets.mean() / (down.std() + 1e-10) * np.sqrt(252)) if len(down) > 1 else sharpe
    else:
        sharpe = sortino = 0.0

    gross_w = sum(p for p in pnls if p > 0)
    gross_l = abs(sum(p for p in pnls if p < 0))
    pf = gross_w / (gross_l + 1e-10)

    if len(eq_df) > 0:
        peak = eq_df['equity'].cummax()
        mdd = float(((eq_df['equity'] - peak) / peak).min())
    else:
        mdd = 0

    total_pnl = sum(pnls)
    total_return = total_pnl / INITIAL_CAPITAL
    if len(eq_df) > 1:
        n_days_bt = (eq_df.index[-1] - eq_df.index[0]).days
        years = n_days_bt / 365.25
        cagr = (1 + total_return) ** (1.0 / years) - 1 if years > 0 and (1 + total_return) > 0 else 0
    else:
        cagr = 0

    spy_total_return = spy_sharpe = alpha_vs_spy = 0
    if 'spy_equity' in eq_df.columns and len(eq_df) > 5:
        spy_total_return = float(eq_df['spy_equity'].iloc[-1] / eq_df['spy_equity'].iloc[0] - 1)
        spy_d = eq_df['spy_equity'].pct_change().dropna().replace([np.inf, -np.inf], 0).fillna(0)
        spy_sharpe = float(spy_d.mean() / (spy_d.std() + 1e-10) * np.sqrt(252))
        alpha_vs_spy = total_return - spy_total_return

    avg_hold = np.mean([t['holding_days'] for t in trades])
    conv_rate = sum(1 for t in trades if t['exit_reason'] == 'convergence') / n if n > 0 else 0

    return {
        'variant': variant_name, 'n_trades': n, 'sharpe': round(sharpe, 3),
        'sortino': round(sortino, 3), 'pf': round(pf, 3), 'wr': round(wr * 100, 1),
        'mdd': round(mdd * 100, 2), 'total_return': round(total_return * 100, 2),
        'cagr': round(cagr * 100, 2), 'total_pnl': round(total_pnl, 2),
        'mean_pnl': round(np.mean(pnls), 4), 'wins': wins, 'losses': n - wins,
        'spy_total_return': round(spy_total_return * 100, 2),
        'spy_sharpe': round(spy_sharpe, 3), 'alpha_vs_spy': round(alpha_vs_spy * 100, 2),
        'avg_hold': round(avg_hold, 1), 'convergence_rate': round(conv_rate * 100, 1),
    }


# ==================== REGIME BREAKDOWN ====================

def compute_regime_breakdown(trades, spy):
    if not trades or len(spy) < 2:
        return {}
    regime_trades = {'green': [], 'red': [], 'flat': []}
    for t in trades:
        entry_date = pd.Timestamp(t['entry_date'])
        exit_date = pd.Timestamp(t['exit_date'])
        mask = (spy.index >= entry_date) & (spy.index <= exit_date)
        period_spy = spy[mask]
        if len(period_spy) >= 2:
            ret = float(period_spy.iloc[-1] / period_spy.iloc[0] - 1)
        else:
            ret = 0.0
        if ret > 0.005:
            regime_trades['green'].append(t)
        elif ret < -0.005:
            regime_trades['red'].append(t)
        else:
            regime_trades['flat'].append(t)

    breakdown = {}
    for regime, rtrades in regime_trades.items():
        if not rtrades:
            breakdown[regime] = {'n': 0, 'sharpe': 0, 'wr': 0, 'mean_pnl': 0, 'total_pnl': 0}
            continue
        pnls = [t['pnl'] for t in rtrades]
        n = len(pnls)
        wins = sum(1 for p in pnls if p > 0)
        mean_pnl = np.mean(pnls)
        std_pnl = np.std(pnls) if n > 1 else 1.0
        sharpe = (mean_pnl / (std_pnl + 1e-10)) * np.sqrt(12) if std_pnl > 0 else 0
        breakdown[regime] = {
            'n': n, 'sharpe': round(sharpe, 2),
            'wr': round(wins / n * 100, 1) if n > 0 else 0,
            'mean_pnl': round(mean_pnl, 4), 'total_pnl': round(sum(pnls), 2),
        }
    return breakdown


# ==================== 5-GATE VALIDATION ====================

def five_gate_validation(result, spy):
    metrics = result['metrics']
    trades = result['trades']
    gates = {}

    # Gate 1: Sharpe > 1.0
    gates['sharpe_gt_1'] = {
        'pass': metrics['sharpe'] > 1.0,
        'value': metrics['sharpe'], 'threshold': 1.0,
    }

    # Gate 2: Permutation test (100 shuffles)
    n_perm = 100
    if len(trades) >= 10:
        pnls = np.array([t['pnl'] for t in trades])
        actual_sharpe = np.mean(pnls) / (np.std(pnls) + 1e-10)
        rng = np.random.RandomState(42)
        count_better = 0
        for _ in range(n_perm):
            shuffled = rng.permutation(pnls)
            # Randomly flip signs to destroy temporal structure
            signs = rng.choice([-1, 1], size=len(pnls))
            perm_pnls = pnls * signs
            perm_sharpe = np.mean(perm_pnls) / (np.std(perm_pnls) + 1e-10)
            if perm_sharpe >= actual_sharpe:
                count_better += 1
        p_value = count_better / n_perm
    else:
        p_value = 1.0
    gates['perm_p_lt_005'] = {
        'pass': p_value < 0.05,
        'value': round(p_value, 4), 'threshold': 0.05,
    }

    # Gate 3: WR > 40%
    gates['wr_gt_40'] = {
        'pass': metrics['wr'] > 40.0,
        'value': metrics['wr'], 'threshold': 40.0,
    }

    # Gate 4: Regime balance
    breakdown = compute_regime_breakdown(trades, spy)
    sg = breakdown.get('green', {}).get('sharpe', 0)
    sr = breakdown.get('red', {}).get('sharpe', 0)
    max_s = max(abs(sg), abs(sr), 0.01)
    regime_skew = abs(sg - sr) / max_s
    gates['regime_balance'] = {
        'pass': regime_skew <= 0.50,
        'value': round(regime_skew, 3), 'threshold': 0.50,
        'sharpe_green': sg, 'sharpe_red': sr,
    }

    # Gate 5: MC 95% CI > 0
    if len(trades) >= 10:
        pnls = np.array([t['pnl'] for t in trades])
        rng = np.random.RandomState(123)
        mc_totals = np.array([np.sum(rng.choice(pnls, size=len(pnls), replace=True)) for _ in range(5000)])
        ci_lo = float(np.percentile(mc_totals, 2.5))
        ci_hi = float(np.percentile(mc_totals, 97.5))
    else:
        ci_lo = ci_hi = 0
    gates['mc_ci_positive'] = {
        'pass': ci_lo > 0,
        'value': round(ci_lo, 2), 'ci_upper': round(ci_hi, 2), 'threshold': 0.0,
    }

    n_pass = sum(1 for g in gates.values() if g['pass'])
    return {
        'gates': gates, 'n_pass': n_pass, 'n_total': len(gates),
        'all_pass': n_pass == len(gates), 'regime_breakdown': breakdown,
    }


# ==================== MAIN ====================

def main():
    t0 = time.time()
    fprint("=" * 70)
    fprint("  ETF PAIRS COINTEGRATION V1 — Mean Reversion")
    fprint("  Hypothesis: Correlated ETF pairs revert when they diverge")
    fprint("=" * 70)

    prices, spy, vix, valid_pairs = download_data()
    oot_days = len(prices) - (CALIB_WINDOW + 126)
    fprint(f"OOT days: {oot_days}, Valid pairs: {len(valid_pairs)}")

    variants = ['A', 'B', 'C', 'D', 'E', 'F']
    variant_names = {
        'A': 'Baseline z>2 63d',
        'B': 'Faster z>1.5 21d',
        'C': 'Slower z>2.5 126d',
        'D': 'LGBM Enhanced',
        'E': 'Top-3 Simultaneous',
        'F': 'VIX<25 Filter',
    }

    results = {}
    validations = {}

    for v in variants:
        fprint(f"\n{'='*60}")
        fprint(f"  VARIANT {v}: {variant_names[v]}")
        fprint(f"{'='*60}")
        ts = time.time()
        results[v] = run_variant(prices, spy, vix, valid_pairs, v, verbose=True)
        elapsed_v = time.time() - ts
        m = results[v]['metrics']
        fprint(f"  Trades: {m['n_trades']} | Sharpe: {m['sharpe']:.3f} | "
               f"Sortino: {m['sortino']:.3f} | PF: {m['pf']:.3f} | "
               f"WR: {m['wr']:.1f}% | MDD: {m['mdd']:.2f}% | "
               f"Return: {m['total_return']:.2f}% | CAGR: {m['cagr']:.2f}%")
        fprint(f"  Avg Hold: {m['avg_hold']:.1f}d | Conv Rate: {m['convergence_rate']:.1f}% | "
               f"Alpha vs SPY: {m['alpha_vs_spy']:.2f}%")
        fprint(f"  Runtime: {elapsed_v:.1f}s")

        val = five_gate_validation(results[v], spy)
        validations[v] = val
        fprint(f"  5-Gate: {val['n_pass']}/{val['n_total']} PASS")
        for name, gate in val['gates'].items():
            status = "PASS" if gate['pass'] else "FAIL"
            fprint(f"    {name}: {status} (value={gate['value']}, threshold={gate['threshold']})")

    # ── Summary ──
    fprint("\n" + "=" * 100)
    fprint("  COMPARISON SUMMARY — ETF Pairs Cointegration V1")
    fprint("=" * 100)
    fprint(f"{'Variant':<25} {'Trades':>6} {'Sharpe':>7} {'Sortino':>8} {'PF':>6} "
           f"{'WR%':>6} {'MDD%':>7} {'Ret%':>7} {'CAGR%':>7} {'AvgHold':>8} {'Conv%':>6} {'Gates':>6}")
    fprint("-" * 100)
    for v in variants:
        m = results[v]['metrics']
        val = validations[v]
        fprint(f"{v}: {variant_names[v]:<22} {m['n_trades']:>5} {m['sharpe']:>7.3f} "
               f"{m['sortino']:>8.3f} {m['pf']:>6.3f} {m['wr']:>6.1f} {m['mdd']:>7.2f} "
               f"{m['total_return']:>7.2f} {m['cagr']:>7.2f} {m['avg_hold']:>8.1f} "
               f"{m['convergence_rate']:>6.1f} {val['n_pass']:>2}/{val['n_total']}")

    spy_m = results['A']['metrics']
    fprint(f"{'SPY Buy-Hold':<28} {'':>5} {spy_m['spy_sharpe']:>7.3f} {'':>8} {'':>6} "
           f"{'':>6} {'':>7} {spy_m['spy_total_return']:>7.2f} {'':>7} {'':>8} {'':>6} {'':>6}")

    # ── Regime breakdowns ──
    fprint("\n" + "=" * 100)
    fprint("  REGIME BREAKDOWNS")
    fprint("=" * 100)
    for v in variants:
        bd = validations[v]['regime_breakdown']
        fprint(f"\n  Variant {v} ({variant_names[v]}):")
        for regime in ['green', 'red', 'flat']:
            b = bd.get(regime, {})
            fprint(f"    {regime:5s}: n={b.get('n',0):3d}  Sharpe={b.get('sharpe',0):6.2f}  "
                   f"WR={b.get('wr',0):5.1f}%  mean=${b.get('mean_pnl',0):8.4f}  "
                   f"total=${b.get('total_pnl',0):8.2f}")

    # ── Pair-level analysis ──
    fprint("\n" + "=" * 100)
    fprint("  PAIR-LEVEL P&L (Variant A)")
    fprint("=" * 100)
    pair_pnl = {}
    for t in results['A']['trades']:
        p = t['pair']
        pair_pnl.setdefault(p, []).append(t['pnl'])
    for p, pnls in sorted(pair_pnl.items(), key=lambda x: sum(x[1]), reverse=True):
        fprint(f"  {p:10s}: n={len(pnls):3d}  total=${sum(pnls):8.2f}  "
               f"mean=${np.mean(pnls):8.4f}  WR={sum(1 for x in pnls if x>0)/len(pnls)*100:5.1f}%")

    # ── Best variant ──
    best_v = max(variants, key=lambda v: results[v]['metrics']['sharpe'])
    best_m = results[best_v]['metrics']
    fprint(f"\n  BEST VARIANT: {best_v} ({variant_names[best_v]}) — Sharpe {best_m['sharpe']:.3f}")

    # ── Conclusions ──
    fprint("\n" + "=" * 100)
    fprint("  KEY CONCLUSIONS")
    fprint("=" * 100)
    any_pass = any(validations[v]['all_pass'] for v in variants)
    if any_pass:
        passing = [v for v in variants if validations[v]['all_pass']]
        fprint(f"  VERDICT: Pairs cointegration WORKS — passes all 5 gates!")
        fprint(f"  Passing: {', '.join(f'{v} ({variant_names[v]})' for v in passing)}")
    elif best_m['sharpe'] > 1.0:
        fprint(f"  VERDICT: Pairs has edge but fails some gates. Best Sharpe: {best_m['sharpe']:.3f}")
    elif best_m['sharpe'] > 0:
        fprint(f"  VERDICT: Weak positive edge. Best Sharpe: {best_m['sharpe']:.3f}")
    else:
        fprint(f"  VERDICT: Pairs cointegration does NOT work with this config.")

    elapsed = time.time() - t0
    fprint(f"\nTotal runtime: {elapsed:.0f}s")

    # ── Save results ──
    save_results = {
        'timestamp': datetime.now().isoformat(),
        'runtime_seconds': round(elapsed, 1),
        'data_range': f"{prices.index[0].date()} to {prices.index[-1].date()}",
        'n_days': len(prices), 'oot_days': oot_days,
        'initial_capital': INITIAL_CAPITAL,
        'pairs': [f"{a}/{b}" for a, b in valid_pairs],
        'calib_window': CALIB_WINDOW, 'max_hold': MAX_HOLD,
        'metrics': {}, 'validations': {}, 'regime_breakdowns': {},
    }
    for v in variants:
        save_results['metrics'][v] = results[v]['metrics']
        val_clean = {
            'n_pass': validations[v]['n_pass'],
            'n_total': validations[v]['n_total'],
            'all_pass': validations[v]['all_pass'],
            'gates': {gn: {k: v2 for k, v2 in gv.items()} for gn, gv in validations[v]['gates'].items()},
        }
        save_results['validations'][v] = val_clean
        save_results['regime_breakdowns'][v] = validations[v]['regime_breakdown']

    results_path = OUTPUT_DIR / "backtest_results.json"
    with open(results_path, 'w') as f:
        json.dump(save_results, f, indent=2, default=str)
    fprint(f"\nResults saved to {results_path}")

    for v in variants:
        trades_path = OUTPUT_DIR / f"trades_variant_{v}.json"
        with open(trades_path, 'w') as f:
            json.dump(results[v]['trades'], f, indent=2, default=str)
        eq_path = OUTPUT_DIR / f"equity_curve_{v}.csv"
        results[v]['equity_curve'].to_csv(eq_path)

    # ── MLflow ──
    if MLFLOW_OK:
        try:
            run_name = f"pairs_coint_{datetime.now():%Y%m%d_%H%M}"
            with mlflow.start_run(run_name=run_name):
                mlflow.log_param("initial_capital", INITIAL_CAPITAL)
                mlflow.log_param("n_pairs", len(valid_pairs))
                mlflow.log_param("pairs", ','.join(f"{a}/{b}" for a, b in valid_pairs))
                mlflow.log_param("calib_window", CALIB_WINDOW)
                mlflow.log_param("max_hold", MAX_HOLD)
                mlflow.log_param("data_range", f"{prices.index[0].date()} to {prices.index[-1].date()}")
                mlflow.log_param("best_variant", f"{best_v}_{variant_names[best_v]}")
                for v in variants:
                    m = results[v]['metrics']
                    pfx = f"v{v}_"
                    for k in ['sharpe','sortino','pf','wr','mdd','total_return','cagr',
                              'n_trades','alpha_vs_spy','avg_hold','convergence_rate']:
                        mlflow.log_metric(f"{pfx}{k}", m[k])
                    mlflow.log_metric(f"{pfx}gates_pass", validations[v]['n_pass'])
                mlflow.log_artifact(str(results_path))
            fprint("MLflow run logged successfully")
        except Exception as e:
            fprint(f"MLflow logging failed: {e}")

    fprint("\nDone.")


if __name__ == '__main__':
    main()
