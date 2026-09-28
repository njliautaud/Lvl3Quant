#!/usr/bin/env python3
"""
Adaptive Portfolio Allocation System v1
========================================
Combines validated strategies into an optimal portfolio using multiple
allocation schemes, all with strict anti-lookahead walk-forward design.

Strategies:
  1. Z-Score Stat Arb (Sharpe ~0.81, low SPY correlation)
  2. UPRO + 200SMA (Sharpe ~0.82, high CAGR but high drawdown)
  3. VIX Spike Mean-Reversion (event-driven, ~2-6 trades/year)

Allocation Schemes:
  a. Equal weight (1/N)
  b. Inverse volatility (trailing 63d vol, T-1)
  c. Risk parity (equal risk contribution)
  d. Min-variance (252d sliding covariance, T-1)
  e. Half-Kelly (walk-forward estimated)

Each tested at 1x and 1.5x leverage.
Monthly rebalancing, 10 bps cost when weights shift > 5%.

HC #724: ALL estimates use STRICTLY T-1 data. Walk-forward only.
"""

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
from itertools import combinations
from scipy.optimize import minimize
import json, os, sys, warnings
warnings.filterwarnings('ignore')

OUTPUT_DIR = '/home/jupiter/Lvl3Quant/output/adaptive_portfolio_v1'
os.makedirs(OUTPUT_DIR, exist_ok=True)

np.random.seed(42)

print("=" * 70)
print("ADAPTIVE PORTFOLIO ALLOCATION v1")
print("Walk-forward, anti-lookahead, with costs")
print("=" * 70)

# ══════════════════════════════════════════════════════════════════════
# 1. DATA DOWNLOAD
# ══════════════════════════════════════════════════════════════════════
print("\n[1/7] Downloading data...")
sys.stdout.flush()

all_tickers = [
    'SPY', 'UPRO', 'TQQQ',
    'XLK', 'XLF', 'XLE', 'XLV', 'XLY', 'XLP', 'XLI', 'XLB', 'XLU', 'XLRE', 'XLC',
    'GLD', 'GDX', 'TLT', 'IEF', 'HYG', 'LQD', 'QQQ', 'IWM', 'EEM', 'DIA',
    'UUP', 'CPER',
]

df = yf.download(all_tickers, start='2012-06-01', progress=False)
if hasattr(df.index, 'tz') and df.index.tz is not None:
    df.index = df.index.tz_localize(None)
close = df['Close'] if isinstance(df.columns, pd.MultiIndex) else df
close = close.ffill()

vix = yf.download('^VIX', start='2012-06-01', progress=False)
if hasattr(vix.index, 'tz') and vix.index.tz is not None:
    vix.index = vix.index.tz_localize(None)
if isinstance(vix.columns, pd.MultiIndex):
    vix.columns = vix.columns.get_level_values(0)
vix_close = vix['Close'].ffill()

# Align
common_idx = close.index.intersection(vix_close.index)
close = close.loc[common_idx]
vix_close = vix_close.loc[common_idx]
returns = close.pct_change()

print(f"  {len(close)} days, {close.index[0].strftime('%Y-%m-%d')} to {close.index[-1].strftime('%Y-%m-%d')}")
sys.stdout.flush()


# ══════════════════════════════════════════════════════════════════════
# 2. STRATEGY 1: UPRO DRAWDOWN PROTECTION (200SMA + signals)
# ══════════════════════════════════════════════════════════════════════
print("\n[2/7] Running UPRO Drawdown Protection...")
sys.stdout.flush()

def run_upro_protection():
    """
    UPRO + 200SMA with 1-day lag.
    Simple rule: Hold UPRO when SPY > 200-day SMA (as of yesterday's close).
    Otherwise hold cash (SHY proxy = ~0% for simplicity).
    10 bps transaction cost on each switch.
    """
    spy = close['SPY']
    upro = close['UPRO']
    upro_ret = upro.pct_change()

    # 200-day SMA
    spy_200sma = spy.rolling(200).mean()

    # Signal: SPY above 200SMA at close -> trade next day (T-1 lag)
    signal = (spy > spy_200sma).astype(int)
    signal_shifted = signal.shift(1).fillna(0)

    daily_ret = pd.Series(0.0, index=spy.index)
    switches = 0
    for i in range(1, len(spy)):
        if signal_shifted.iloc[i] == 1:
            daily_ret.iloc[i] = float(upro_ret.iloc[i])
        # Transaction cost on switch
        if i > 1 and signal_shifted.iloc[i] != signal_shifted.iloc[i-1]:
            daily_ret.iloc[i] -= 0.001  # 10 bps
            switches += 1

    print(f"    UPRO+200SMA: {switches} switches, risk-on {signal_shifted.mean()*100:.0f}% of days")
    return daily_ret


# ══════════════════════════════════════════════════════════════════════
# 3. STRATEGY 2: Z-SCORE STAT ARB
# ══════════════════════════════════════════════════════════════════════
print("\n[3/7] Running Z-Score Stat Arb...")
sys.stdout.flush()

def run_zscore_stat_arb():
    """
    Walk-forward ETF pairs mean-reversion.
    Uses daily MTM for open positions (no double-counting on exits).
    Exit day: only charge transaction costs, the MTM already captured the P&L.
    """
    tickers = ['XLK','XLF','XLE','XLV','XLY','XLP','XLI','XLB','XLU','XLRE','XLC',
               'GLD','GDX','TLT','IEF','HYG','LQD','SPY','QQQ','IWM','EEM','DIA']
    available = [t for t in tickers if t in close.columns]
    prices = close[available].copy()
    rets = prices.pct_change().fillna(0)

    TRAIN_WINDOW = 252
    COINT_LOOKBACK = 126
    ENTRY_Z = 1.5
    EXIT_Z = 0.3
    STOP_Z = 4.0
    MAX_HOLD = 42
    MAX_PAIRS = 8
    POS_SIZE = 1.0 / MAX_PAIRS
    COST_BPS = 10
    RESEL_FREQ = 21

    n_days = len(prices)
    daily_ret = pd.Series(0.0, index=prices.index)
    positions = {}
    selected_pairs = []
    last_reselect = 0

    for i in range(TRAIN_WINDOW, n_days):
        day_pnl = 0.0

        # Re-select pairs monthly
        if i - last_reselect >= RESEL_FREQ or not selected_pairs:
            last_reselect = i
            train = prices.iloc[i-COINT_LOOKBACK:i]
            log_train = np.log(train)

            pair_scores = []
            for a, b in combinations(available, 2):
                try:
                    spread = log_train[a] - log_train[b]
                    spread_std = spread.std()
                    if spread_std < 0.001:
                        continue
                    spread_diff = spread.diff().dropna()
                    spread_lag = spread.shift(1).dropna()
                    if len(spread_diff) < 20:
                        continue
                    common_idx = spread_diff.index.intersection(spread_lag.index)
                    y = spread_diff.loc[common_idx].values
                    x = spread_lag.loc[common_idx].values
                    if len(x) < 20 or np.std(x) < 1e-10:
                        continue
                    phi = np.sum(x * y) / np.sum(x ** 2)
                    if phi >= 0:
                        continue
                    half_life = -np.log(2) / phi
                    if 2 < half_life < 42:
                        pair_scores.append((a, b, half_life, spread_std))
                except:
                    continue

            pair_scores.sort(key=lambda x: x[2])
            selected_pairs = [(a, b) for a, b, _, _ in pair_scores[:MAX_PAIRS * 3]]

        # Mark-to-market ALL open positions (including those about to be closed)
        for pair_key, pos in positions.items():
            a, b = pair_key.split('/')
            if a not in rets.columns or b not in rets.columns:
                continue
            ret_a = float(rets[a].iloc[i])
            ret_b = float(rets[b].iloc[i])
            if pos['side'] == 'long_spread':
                day_pnl += (ret_a - ret_b) * POS_SIZE
            else:
                day_pnl += (ret_b - ret_a) * POS_SIZE

        # Check exits (after MTM so we don't double-count)
        closed = []
        for pair_key, pos in positions.items():
            a, b = pair_key.split('/')
            if a not in prices.columns or b not in prices.columns:
                continue
            lookback = prices.iloc[max(0,i-COINT_LOOKBACK):i]
            spread = np.log(lookback[a]) - np.log(lookback[b])
            z = (spread.iloc[-1] - spread.mean()) / spread.std() if spread.std() > 0 else 0
            hold_days = i - pos['entry_day']

            exit_signal = False
            if abs(z) < EXIT_Z:
                exit_signal = True
            elif abs(z) > STOP_Z:
                exit_signal = True
            elif hold_days >= MAX_HOLD:
                exit_signal = True

            if exit_signal:
                # Only charge exit transaction costs (MTM already captured P&L)
                day_pnl -= 2 * COST_BPS / 10000 * POS_SIZE  # 2 legs to close
                closed.append(pair_key)

        for pk in closed:
            del positions[pk]

        # New entries
        if len(positions) < MAX_PAIRS:
            for a, b in selected_pairs:
                if len(positions) >= MAX_PAIRS:
                    break
                pair_key = f"{a}/{b}"
                rev_key = f"{b}/{a}"
                if pair_key in positions or rev_key in positions:
                    continue

                lookback = prices.iloc[max(0,i-COINT_LOOKBACK):i]
                spread = np.log(lookback[a]) - np.log(lookback[b])
                if spread.std() < 0.001:
                    continue
                z = (spread.iloc[-1] - spread.mean()) / spread.std()

                if abs(z) > ENTRY_Z:
                    side = 'short_spread' if z > ENTRY_Z else 'long_spread'
                    positions[pair_key] = {
                        'side': side,
                        'entry_day': i,
                        'entry_prices': (prices[a].iloc[i], prices[b].iloc[i]),
                    }
                    # Entry transaction costs (2 legs to open)
                    day_pnl -= 2 * COST_BPS / 10000 * POS_SIZE

        daily_ret.iloc[i] = day_pnl

    return daily_ret


# ══════════════════════════════════════════════════════════════════════
# 4. STRATEGY 3: VIX SPIKE MEAN-REVERSION (SPY Calls Proxy)
# ══════════════════════════════════════════════════════════════════════
print("\n[4/7] Running VIX Spike Mean-Reversion...")
sys.stdout.flush()

def run_vix_spike_strategy():
    """
    Event-driven: buy SPY when VIX spikes above 30, sell when VIX drops below 20.
    Uses SPY returns directly as a proxy for call option profits (conservative).
    Staged entries: 50% at VIX>30, 25% at VIX>35, 25% at VIX>40.
    Signal at T, trade at T+1 (anti-lookahead).
    ~2-6 trades per year.
    """
    spy = close['SPY']
    spy_ret = spy.pct_change()

    # T-1 VIX signal
    vix_prev = vix_close.shift(1)

    daily_ret = pd.Series(0.0, index=spy.index)
    exposure = 0.0  # Current fraction deployed
    in_trade = False
    entry_vix = 0

    for i in range(2, len(spy)):
        v = float(vix_prev.iloc[i])
        if pd.isna(v):
            continue

        # Entry logic (staged)
        if not in_trade and v >= 30:
            exposure = 0.50
            in_trade = True
            entry_vix = v
        elif in_trade and v >= 35 and exposure < 0.75:
            exposure = 0.75
        elif in_trade and v >= 40 and exposure < 1.0:
            exposure = 1.0

        # Exit logic
        if in_trade and v < 20:
            in_trade = False
            exposure = 0.0
            entry_vix = 0

        # Daily return = exposure * SPY return (conservative proxy)
        # Scale by 2x to approximate call option leverage (still conservative)
        if in_trade:
            daily_ret.iloc[i] = float(spy_ret.iloc[i]) * exposure * 2.0

    return daily_ret


# ══════════════════════════════════════════════════════════════════════
# 5. GENERATE ALL STRATEGY RETURNS
# ══════════════════════════════════════════════════════════════════════
print("\n[5/7] Generating strategy return series...")
sys.stdout.flush()

upro_ret = run_upro_protection()
stat_arb_ret = run_zscore_stat_arb()
vix_spike_ret = run_vix_spike_strategy()
spy_ret = close['SPY'].pct_change().fillna(0)

# Combine into DataFrame
strat_returns = pd.DataFrame({
    'upro_protection': upro_ret,
    'stat_arb': stat_arb_ret,
    'vix_spike': vix_spike_ret,
}, index=close.index)

# Find overlap period (all strategies have data)
# UPRO starts at index 51 (warmup), stat arb at 252
warmup_end = 252 + 63  # stat arb warmup + extra buffer for vol estimates
strat_returns = strat_returns.iloc[warmup_end:]
spy_returns = spy_ret.iloc[warmup_end:]

print(f"  Overlap period: {strat_returns.index[0].strftime('%Y-%m-%d')} to {strat_returns.index[-1].strftime('%Y-%m-%d')}")
print(f"  {len(strat_returns)} trading days ({len(strat_returns)/252:.1f} years)")

# Quick per-strategy stats
for col in strat_returns.columns:
    r = strat_returns[col]
    ann_ret = r.mean() * 252
    ann_vol = r.std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
    eq = (1 + r).cumprod()
    dd = eq / eq.cummax() - 1
    max_dd = dd.min()
    corr_spy = r.corr(spy_returns)
    print(f"  {col}: Sharpe={sharpe:.2f}, AnnRet={ann_ret*100:.1f}%, MaxDD={max_dd*100:.1f}%, SPY_corr={corr_spy:.2f}")

# Save individual returns
strat_returns.to_csv(os.path.join(OUTPUT_DIR, 'strategy_daily_returns.csv'))
sys.stdout.flush()


# ══════════════════════════════════════════════════════════════════════
# 6. ALLOCATION SCHEMES
# ══════════════════════════════════════════════════════════════════════
print("\n[6/7] Running allocation schemes...")
sys.stdout.flush()

N_STRATS = len(strat_returns.columns)
VOL_LOOKBACK = 63
COV_LOOKBACK = 252
REBAL_COST_BPS = 10  # 10 bps per leg when weights shift > 5%
WEIGHT_SHIFT_THRESHOLD = 0.05  # Only charge costs if weight shifts > 5%


def compute_metrics(returns_series, name=""):
    """Compute standard performance metrics."""
    r = returns_series.dropna()
    if len(r) < 252:
        return {'sharpe': 0, 'sortino': 0, 'cagr': 0, 'max_dd': 0, 'calmar': 0}

    ann_ret = r.mean() * 252
    ann_vol = r.std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
    downside = r[r < 0].std() * np.sqrt(252) if (r < 0).any() else 0.001
    sortino = ann_ret / downside

    eq = (1 + r).cumprod()
    cum_ret = eq.iloc[-1] - 1
    years = len(r) / 252
    cagr = (1 + cum_ret) ** (1/years) - 1 if years > 0 else 0
    dd = eq / eq.cummax() - 1
    max_dd = dd.min()
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0

    return {
        'sharpe': round(sharpe, 3),
        'sortino': round(sortino, 3),
        'cagr': round(cagr, 4),
        'max_dd': round(max_dd, 4),
        'calmar': round(calmar, 3),
    }


def get_rebalance_dates(index):
    """First trading day of each month."""
    dates = []
    current_month = None
    for dt in index:
        ym = (dt.year, dt.month)
        if ym != current_month:
            dates.append(dt)
            current_month = ym
    return dates


def apply_portfolio(strat_returns, weight_func, leverage=1.0, name=""):
    """
    Walk-forward portfolio construction.
    weight_func(returns_history) -> array of weights (sum to 1).
    Rebalance monthly. Charge costs on weight shifts > 5%.
    """
    dates = strat_returns.index
    cols = strat_returns.columns
    n = len(cols)
    rebal_dates = set(get_rebalance_dates(dates))

    portfolio_ret = pd.Series(0.0, index=dates)
    current_weights = np.ones(n) / n  # Start equal weight
    total_cost = 0.0

    for i in range(1, len(dates)):
        dt = dates[i]

        # Rebalance on first trading day of month
        if dt in rebal_dates and i > COV_LOOKBACK:
            # Use STRICTLY T-1 data (up to yesterday)
            hist = strat_returns.iloc[:i]
            try:
                new_weights = weight_func(hist)
                new_weights = np.array(new_weights, dtype=float)
                # Ensure valid weights
                if np.any(np.isnan(new_weights)) or np.sum(new_weights) < 0.01:
                    new_weights = np.ones(n) / n
                new_weights = new_weights / new_weights.sum()  # Normalize
            except Exception as e:
                new_weights = current_weights.copy()

            # Transaction costs: 10 bps per leg where weight shifts > 5%
            weight_changes = np.abs(new_weights - current_weights)
            for j in range(n):
                if weight_changes[j] > WEIGHT_SHIFT_THRESHOLD:
                    cost = REBAL_COST_BPS / 10000 * weight_changes[j] * leverage
                    portfolio_ret.iloc[i] -= cost
                    total_cost += cost

            current_weights = new_weights

        # Portfolio return = weighted sum of strategy returns
        day_returns = strat_returns.iloc[i].values
        portfolio_ret.iloc[i] += np.sum(current_weights * day_returns) * leverage

    return portfolio_ret, total_cost


# --- Weight functions ---

def equal_weight(hist):
    n = hist.shape[1]
    return np.ones(n) / n


def inverse_vol_weight(hist):
    """Inverse volatility using trailing 63d, T-1."""
    recent = hist.iloc[-VOL_LOOKBACK:]
    vols = recent.std() * np.sqrt(252)
    vols = vols.values
    vols = np.maximum(vols, 0.001)  # Floor
    inv_vol = 1.0 / vols
    return inv_vol / inv_vol.sum()


def risk_parity_weight(hist):
    """Equal risk contribution using trailing 252d covariance, T-1."""
    recent = hist.iloc[-COV_LOOKBACK:]
    cov = recent.cov().values * 252
    n = cov.shape[0]

    # Optimization: minimize sum of (w_i * (Cov @ w)_i - target_risk)^2
    def objective(w):
        w = np.abs(w)
        port_vol = np.sqrt(w @ cov @ w)
        if port_vol < 1e-10:
            return 1e10
        marginal_risk = cov @ w
        risk_contrib = w * marginal_risk / port_vol
        target = port_vol / n
        return np.sum((risk_contrib - target) ** 2)

    w0 = np.ones(n) / n
    bounds = [(0.01, 0.8)] * n
    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
    try:
        result = minimize(objective, w0, method='SLSQP', bounds=bounds, constraints=constraints,
                         options={'maxiter': 500, 'ftol': 1e-12})
        if result.success:
            w = np.abs(result.x)
            return w / w.sum()
    except:
        pass
    return np.ones(n) / n


def min_variance_weight(hist):
    """Minimum variance portfolio using trailing 252d covariance, T-1."""
    recent = hist.iloc[-COV_LOOKBACK:]
    cov = recent.cov().values * 252
    n = cov.shape[0]

    def objective(w):
        return w @ cov @ w

    w0 = np.ones(n) / n
    bounds = [(0.01, 0.8)] * n
    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
    try:
        result = minimize(objective, w0, method='SLSQP', bounds=bounds, constraints=constraints,
                         options={'maxiter': 500, 'ftol': 1e-12})
        if result.success:
            w = np.abs(result.x)
            return w / w.sum()
    except:
        pass
    return np.ones(n) / n


def half_kelly_weight(hist):
    """Half-Kelly criterion using trailing 252d data, T-1."""
    recent = hist.iloc[-COV_LOOKBACK:]
    mu = recent.mean().values * 252  # Annualized returns
    cov = recent.cov().values * 252

    try:
        cov_inv = np.linalg.inv(cov + np.eye(len(mu)) * 1e-6)
        kelly = cov_inv @ mu
        # Half-Kelly
        kelly = kelly * 0.5
        # Clip to [0, 1] and normalize
        kelly = np.clip(kelly, 0.0, 1.0)
        if kelly.sum() < 0.01:
            return np.ones(len(mu)) / len(mu)
        return kelly / kelly.sum()
    except:
        return np.ones(len(mu)) / len(mu)


# --- Run all schemes ---
schemes = {
    'equal_weight': equal_weight,
    'inverse_vol': inverse_vol_weight,
    'risk_parity': risk_parity_weight,
    'min_variance': min_variance_weight,
    'half_kelly': half_kelly_weight,
}

results = {}
all_equity_curves = {}

for scheme_name, weight_func in schemes.items():
    for leverage in [1.0, 1.5]:
        label = f"{scheme_name}_{leverage:.1f}x"
        print(f"  Running {label}...", end=" ")
        sys.stdout.flush()

        port_ret, total_cost = apply_portfolio(strat_returns, weight_func, leverage=leverage, name=label)
        metrics = compute_metrics(port_ret, name=label)
        metrics['total_rebal_cost_bps'] = round(total_cost * 10000, 1)
        metrics['leverage'] = leverage
        metrics['scheme'] = scheme_name

        # SPY correlation
        metrics['spy_corr'] = round(port_ret.corr(spy_returns), 3)

        results[label] = metrics
        all_equity_curves[label] = (1 + port_ret).cumprod()

        print(f"Sharpe={metrics['sharpe']:.2f}, CAGR={metrics['cagr']*100:.1f}%, MaxDD={metrics['max_dd']*100:.1f}%")
        sys.stdout.flush()

# SPY benchmark
spy_metrics = compute_metrics(spy_returns, name="SPY")
spy_metrics['spy_corr'] = 1.0
spy_metrics['total_rebal_cost_bps'] = 0
spy_metrics['leverage'] = 1.0
spy_metrics['scheme'] = 'benchmark'
results['spy_benchmark'] = spy_metrics
all_equity_curves['spy_benchmark'] = (1 + spy_returns).cumprod()

print(f"  SPY benchmark: Sharpe={spy_metrics['sharpe']:.2f}, CAGR={spy_metrics['cagr']*100:.1f}%, MaxDD={spy_metrics['max_dd']*100:.1f}%")


# ══════════════════════════════════════════════════════════════════════
# 7. VALIDATION & ANALYSIS
# ══════════════════════════════════════════════════════════════════════
print("\n[7/7] Validation & analysis...")
sys.stdout.flush()

# --- Regime Analysis ---
print("\n  === REGIME ANALYSIS (Green vs Red months) ===")

# Monthly returns for SPY to classify regimes
spy_monthly = spy_returns.resample('ME').apply(lambda x: (1+x).prod() - 1)

regime_results = {}
for label in results:
    if label == 'spy_benchmark':
        port_daily = spy_returns
    else:
        scheme_name = results[label]['scheme']
        leverage = results[label]['leverage']
        weight_func = schemes.get(scheme_name, equal_weight)
        port_daily, _ = apply_portfolio(strat_returns, weight_func, leverage=leverage)

    port_monthly = port_daily.resample('ME').apply(lambda x: (1+x).prod() - 1)

    # Align
    common = spy_monthly.index.intersection(port_monthly.index)
    spy_m = spy_monthly.loc[common]
    port_m = port_monthly.loc[common]

    green_mask = spy_m > 0
    red_mask = spy_m <= 0

    green_months = port_m[green_mask]
    red_months = port_m[red_mask]

    green_sharpe = (green_months.mean() * 12) / (green_months.std() * np.sqrt(12)) if len(green_months) > 1 and green_months.std() > 0 else 0
    red_sharpe = (red_months.mean() * 12) / (red_months.std() * np.sqrt(12)) if len(red_months) > 1 and red_months.std() > 0 else 0
    green_avg = green_months.mean() if len(green_months) > 0 else 0
    red_avg = red_months.mean() if len(red_months) > 0 else 0

    regime_results[label] = {
        'green_months': len(green_months),
        'red_months': len(red_months),
        'green_avg_ret': round(float(green_avg) * 100, 2),
        'red_avg_ret': round(float(red_avg) * 100, 2),
        'green_sharpe': round(float(green_sharpe), 2),
        'red_sharpe': round(float(red_sharpe), 2),
    }

# Print regime summary for key schemes
for label in ['equal_weight_1.0x', 'risk_parity_1.0x', 'min_variance_1.0x', 'half_kelly_1.0x', 'spy_benchmark']:
    if label in regime_results:
        rr = regime_results[label]
        print(f"    {label}: Green={rr['green_avg_ret']:.1f}%/mo (Sharpe {rr['green_sharpe']:.1f}), "
              f"Red={rr['red_avg_ret']:.1f}%/mo (Sharpe {rr['red_sharpe']:.1f})")


# --- Drawdown Analysis ---
print("\n  === WORST 5 DRAWDOWNS ===")

def analyze_drawdowns(eq_curve, top_n=5):
    """Find worst drawdowns with recovery times."""
    dd = eq_curve / eq_curve.cummax() - 1
    drawdowns = []

    in_dd = False
    dd_start = None
    dd_trough = None
    dd_trough_val = 0

    for i in range(len(dd)):
        if dd.iloc[i] < -0.005:  # In drawdown
            if not in_dd:
                dd_start = dd.index[i]
                in_dd = True
                dd_trough = dd.index[i]
                dd_trough_val = dd.iloc[i]
            elif dd.iloc[i] < dd_trough_val:
                dd_trough = dd.index[i]
                dd_trough_val = dd.iloc[i]
        elif in_dd and dd.iloc[i] >= -0.001:
            # Recovered
            recovery_date = dd.index[i]
            duration = (recovery_date - dd_start).days
            drawdowns.append({
                'start': dd_start.strftime('%Y-%m-%d'),
                'trough': dd_trough.strftime('%Y-%m-%d'),
                'recovery': recovery_date.strftime('%Y-%m-%d'),
                'depth': round(float(dd_trough_val) * 100, 2),
                'duration_days': duration,
            })
            in_dd = False

    # Handle open drawdown
    if in_dd:
        drawdowns.append({
            'start': dd_start.strftime('%Y-%m-%d'),
            'trough': dd_trough.strftime('%Y-%m-%d'),
            'recovery': 'ongoing',
            'depth': round(float(dd_trough_val) * 100, 2),
            'duration_days': (dd.index[-1] - dd_start).days,
        })

    drawdowns.sort(key=lambda x: x['depth'])
    return drawdowns[:top_n]


dd_analysis = {}
for label in ['equal_weight_1.0x', 'risk_parity_1.0x', 'min_variance_1.0x', 'half_kelly_1.0x', 'spy_benchmark']:
    if label in all_equity_curves:
        dds = analyze_drawdowns(all_equity_curves[label])
        dd_analysis[label] = dds
        print(f"\n    {label}:")
        for j, d in enumerate(dds):
            print(f"      #{j+1}: {d['depth']:.1f}% ({d['start']} to {d['recovery']}, {d['duration_days']}d)")


# --- Lag Sensitivity Test ---
print("\n  === LAG SENSITIVITY TEST ===")
print("  Testing if results are robust to +/- 1 day lag...")

lag_test_results = {}
for lag in [-1, 0, 1, 2]:
    shifted = strat_returns.shift(lag) if lag != 0 else strat_returns
    shifted = shifted.dropna()
    spy_shifted = spy_returns.loc[shifted.index]

    # Test with risk parity
    port_ret, _ = apply_portfolio(shifted, risk_parity_weight, leverage=1.0)
    port_ret = port_ret.loc[shifted.index]
    m = compute_metrics(port_ret)
    lag_test_results[f"lag_{lag}d"] = m
    print(f"    Lag {lag:+d}d: Sharpe={m['sharpe']:.2f}, CAGR={m['cagr']*100:.1f}%, MaxDD={m['max_dd']*100:.1f}%")


# ══════════════════════════════════════════════════════════════════════
# 8. SAVE RESULTS
# ══════════════════════════════════════════════════════════════════════
print("\n[8] Saving results...")

# Summary table
summary = pd.DataFrame(results).T
summary = summary.sort_values('sharpe', ascending=False)
summary.to_csv(os.path.join(OUTPUT_DIR, 'allocation_comparison.csv'))

# Full results JSON
full_results = {
    'metadata': {
        'date_run': datetime.now().strftime('%Y-%m-%d %H:%M'),
        'overlap_start': strat_returns.index[0].strftime('%Y-%m-%d'),
        'overlap_end': strat_returns.index[-1].strftime('%Y-%m-%d'),
        'n_days': len(strat_returns),
        'n_years': round(len(strat_returns) / 252, 1),
        'rebal_frequency': 'monthly (1st trading day)',
        'cost_model': '10 bps per leg when weight shift > 5%',
        'anti_lookahead': 'ALL covariance/vol estimates use T-1 data only',
    },
    'allocation_results': results,
    'regime_analysis': regime_results,
    'drawdown_analysis': dd_analysis,
    'lag_sensitivity': lag_test_results,
}

with open(os.path.join(OUTPUT_DIR, 'results.json'), 'w') as f:
    json.dump(full_results, f, indent=2, default=str)

# Save equity curves
eq_df = pd.DataFrame(all_equity_curves)
eq_df.to_csv(os.path.join(OUTPUT_DIR, 'equity_curves.csv'))

# Strategy correlation matrix
corr_matrix = strat_returns.corr()
corr_matrix.to_csv(os.path.join(OUTPUT_DIR, 'strategy_correlations.csv'))


# ══════════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("FINAL RESULTS SUMMARY")
print("=" * 70)

print("\n  Strategy Correlations:")
print(corr_matrix.to_string())

print(f"\n  {'Scheme':<30} {'Sharpe':>7} {'Sortino':>8} {'CAGR':>7} {'MaxDD':>7} {'Calmar':>7} {'SPY_r':>6} {'Cost':>6}")
print("  " + "-" * 82)
for label in summary.index:
    r = results[label]
    print(f"  {label:<30} {r['sharpe']:>7.2f} {r['sortino']:>8.2f} {r['cagr']*100:>6.1f}% {r['max_dd']*100:>6.1f}% {r['calmar']:>7.2f} {r['spy_corr']:>6.2f} {r.get('total_rebal_cost_bps',0):>5.0f}bp")

print("\n  Best risk-adjusted (1x leverage):")
best_1x = {k: v for k, v in results.items() if v.get('leverage', 1.0) == 1.0 and k != 'spy_benchmark'}
if best_1x:
    best_key = max(best_1x, key=lambda k: best_1x[k]['sharpe'])
    b = best_1x[best_key]
    print(f"    {best_key}: Sharpe={b['sharpe']:.2f}, CAGR={b['cagr']*100:.1f}%, MaxDD={b['max_dd']*100:.1f}%, Calmar={b['calmar']:.2f}")

print("\n  Best risk-adjusted (1.5x leverage):")
best_15x = {k: v for k, v in results.items() if v.get('leverage', 1.0) == 1.5}
if best_15x:
    best_key = max(best_15x, key=lambda k: best_15x[k]['sharpe'])
    b = best_15x[best_key]
    print(f"    {best_key}: Sharpe={b['sharpe']:.2f}, CAGR={b['cagr']*100:.1f}%, MaxDD={b['max_dd']*100:.1f}%, Calmar={b['calmar']:.2f}")

spy_s = results['spy_benchmark']
print(f"\n  SPY benchmark: Sharpe={spy_s['sharpe']:.2f}, CAGR={spy_s['cagr']*100:.1f}%, MaxDD={spy_s['max_dd']*100:.1f}%")

print(f"\n  Output saved to: {OUTPUT_DIR}")
print("=" * 70)
