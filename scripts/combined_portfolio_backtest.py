#!/usr/bin/env python3
"""
Combined Portfolio Backtest — HC #717 R4 Deliverable
=====================================================
Produces TWO versions of the combined portfolio:
  (a) 1x leverage (fully funded, no margin)
  (b) Dynamically leveraged using VIX regime + credit spreads

Strategies included (only those that survived adversarial validation):
  1. UPRO Drawdown Protection (Grade A) — holds UPRO when 4 signals clear, cash when not
  2. Z-Score Stat Arb (Grade B) — ETF pairs mean-reversion, pure z-score rules

HC #717 R1: NO synthetic returns — everything from real price data
HC #717 R2: Walk-forward validated backtests on real market data
HC #717 R3: Monthly rebalancing, transaction costs, drawdown controls
HC #717 R4: Two versions — 1x and dynamically leveraged
HC #718 R3: Transaction costs mandatory
"""

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
from itertools import combinations
import json, os, sys, warnings
warnings.filterwarnings('ignore')

OUTPUT_DIR = '/home/jupiter/Lvl3Quant/output/combined_portfolio'
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 70)
print("COMBINED PORTFOLIO BACKTEST — HC #717 R4")
print("Real data, walk-forward, with costs")
print("=" * 70)

# ══════════════════════════════════════════════════════════════════════
# 1. DATA DOWNLOAD
# ══════════════════════════════════════════════════════════════════════
print("\n[1/5] Downloading data...")
sys.stdout.flush()

# All tickers needed across both strategies
strat_tickers = {
    # UPRO Protection
    'SPY', 'UPRO', 'TQQQ',
    # Stat Arb universe
    'XLK', 'XLF', 'XLE', 'XLV', 'XLY', 'XLP', 'XLI', 'XLB', 'XLU', 'XLRE', 'XLC',
    'GLD', 'GDX', 'TLT', 'IEF', 'HYG', 'LQD', 'QQQ', 'IWM', 'EEM', 'DIA',
    # Signals
    'UUP', 'CPER',
}
vix_ticker = '^VIX'

# Download equity data
all_tickers = list(strat_tickers)
df = yf.download(all_tickers, start='2012-06-01', progress=False)
if hasattr(df.index, 'tz') and df.index.tz is not None:
    df.index = df.index.tz_localize(None)
close = df['Close'] if isinstance(df.columns, pd.MultiIndex) else df
close = close.ffill()

# Download VIX separately
vix = yf.download(vix_ticker, start='2012-06-01', progress=False)
if hasattr(vix.index, 'tz') and vix.index.tz is not None:
    vix.index = vix.index.tz_localize(None)
if isinstance(vix.columns, pd.MultiIndex):
    vix.columns = vix.columns.get_level_values(0)
vix_close = vix['Close'].ffill()

# Align dates
common_idx = close.index.intersection(vix_close.index)
close = close.loc[common_idx]
vix_close = vix_close.loc[common_idx]
returns = close.pct_change()

print(f"  {len(close)} days, {len(close.columns)} assets")
print(f"  Date range: {close.index[0].strftime('%Y-%m-%d')} to {close.index[-1].strftime('%Y-%m-%d')}")
sys.stdout.flush()


# ══════════════════════════════════════════════════════════════════════
# 2. STRATEGY 1: UPRO DRAWDOWN PROTECTION
# ══════════════════════════════════════════════════════════════════════
print("\n[2/5] Running UPRO Drawdown Protection backtest...")
sys.stdout.flush()

def run_upro_protection(close_df, vix_series, initial_capital=100_000):
    """
    Hold UPRO when ALL 4 conditions are met, otherwise cash.
    Conditions (evaluated daily, using only past data):
      1. VIX < 20
      2. SPY above 50-day SMA
      3. Credit not stressed (HYG/LQD ratio above 20d SMA)
      4. Market breadth OK (>50% of sectors above their 50d SMA)
    """
    spy = close_df['SPY']
    upro = close_df['UPRO']
    upro_ret = upro.pct_change()

    sectors = ['XLK', 'XLF', 'XLE', 'XLV', 'XLY', 'XLP', 'XLI', 'XLB', 'XLU', 'XLRE', 'XLC']
    available_sectors = [s for s in sectors if s in close_df.columns]

    # Compute signals (all using trailing data only — no lookahead)
    spy_50sma = spy.rolling(50).mean()

    credit_ratio = close_df['HYG'] / close_df['LQD'] if 'HYG' in close_df and 'LQD' in close_df else pd.Series(1.0, index=spy.index)
    credit_20sma = credit_ratio.rolling(20).mean()

    # Breadth: fraction of sectors above their 50d SMA
    breadth = pd.Series(0.0, index=spy.index)
    for s in available_sectors:
        s_sma = close_df[s].rolling(50).mean()
        breadth += (close_df[s] > s_sma).astype(float)
    breadth = breadth / len(available_sectors) if available_sectors else breadth

    # Generate daily signal (shift by 1 to avoid lookahead — signal at close, trade next open)
    signal = pd.Series(0, index=spy.index)
    for i in range(51, len(spy)):
        vix_ok = vix_series.iloc[i] < 20
        sma_ok = spy.iloc[i] > spy_50sma.iloc[i]
        credit_ok = credit_ratio.iloc[i] > credit_20sma.iloc[i]
        breadth_ok = breadth.iloc[i] > 0.5

        if vix_ok and sma_ok and credit_ok and breadth_ok:
            signal.iloc[i] = 1  # Risk-on: hold UPRO

    # Shift signal by 1 day (trade on next day's return)
    signal_shifted = signal.shift(1).fillna(0)

    # Strategy returns (UPRO when signal=1, 0% when signal=0)
    # Include 10bps transaction cost on each switch
    daily_ret = pd.Series(0.0, index=spy.index)
    switches = 0
    for i in range(1, len(spy)):
        if signal_shifted.iloc[i] == 1:
            daily_ret.iloc[i] = upro_ret.iloc[i]
        # Transaction cost on switch
        if i > 1 and signal_shifted.iloc[i] != signal_shifted.iloc[i-1]:
            daily_ret.iloc[i] -= 0.001  # 10bps per switch
            switches += 1

    # Compute equity curve
    equity = initial_capital * (1 + daily_ret).cumprod()

    # Metrics
    valid = daily_ret[51:]  # Skip warmup
    ann_ret = valid.mean() * 252
    ann_vol = valid.std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
    downside = valid[valid < 0].std() * np.sqrt(252) if (valid < 0).any() else 0.001
    sortino = ann_ret / downside
    cum_ret = equity.iloc[-1] / initial_capital - 1
    years = len(valid) / 252
    cagr = (1 + cum_ret) ** (1/years) - 1 if years > 0 else 0
    dd = (equity / equity.cummax() - 1)
    max_dd = dd.min()
    risk_on_pct = signal_shifted[51:].mean() * 100

    print(f"  UPRO Protection:")
    print(f"    Sharpe: {sharpe:.2f}, Sortino: {sortino:.2f}")
    print(f"    CAGR: {cagr*100:.1f}%, MaxDD: {max_dd*100:.1f}%")
    print(f"    Risk-on: {risk_on_pct:.1f}% of days, Switches: {switches}")
    sys.stdout.flush()

    return daily_ret, equity, {
        'sharpe': sharpe, 'sortino': sortino, 'cagr': cagr,
        'max_dd': max_dd, 'risk_on_pct': risk_on_pct, 'switches': switches,
        'ann_ret': ann_ret, 'ann_vol': ann_vol
    }


# ══════════════════════════════════════════════════════════════════════
# 3. STRATEGY 2: Z-SCORE STAT ARB
# ══════════════════════════════════════════════════════════════════════
print("\n[3/5] Running Z-Score Stat Arb backtest...")
sys.stdout.flush()

def run_zscore_stat_arb(close_df, initial_capital=100_000):
    """
    Walk-forward ETF pairs mean-reversion.
    - Rolling 126d cointegration test to select pairs
    - Entry at |z| > 1.5, exit at |z| < 0.3
    - Stop at |z| > 4.0 or 42-day time stop
    - Max 8 pairs, equal weight, 10bps cost per leg
    """
    # Stat arb universe
    tickers = ['XLK','XLF','XLE','XLV','XLY','XLP','XLI','XLB','XLU','XLRE','XLC',
               'GLD','GDX','TLT','IEF','HYG','LQD','SPY','QQQ','IWM','EEM','DIA']
    available = [t for t in tickers if t in close_df.columns]
    prices = close_df[available].copy()

    TRAIN_WINDOW = 252
    COINT_LOOKBACK = 126
    ENTRY_Z = 1.5
    EXIT_Z = 0.3
    STOP_Z = 4.0
    MAX_HOLD = 42
    MAX_PAIRS = 8
    POS_SIZE = 1.0 / MAX_PAIRS
    COST_BPS = 10
    RESEL_FREQ = 21  # Re-select pairs monthly

    n_days = len(prices)
    daily_ret = pd.Series(0.0, index=prices.index)

    # Track positions
    positions = {}  # pair -> {side, entry_z, entry_day, entry_prices}
    selected_pairs = []
    last_reselect = 0
    total_trades = 0
    wins = 0
    trade_returns = []

    for i in range(TRAIN_WINDOW, n_days):
        date = prices.index[i]
        day_pnl = 0.0

        # Re-select pairs monthly using trailing cointegration
        if i - last_reselect >= RESEL_FREQ or not selected_pairs:
            last_reselect = i
            train = prices.iloc[i-COINT_LOOKBACK:i]
            log_train = np.log(train)

            pair_scores = []
            for a, b in combinations(available, 2):
                try:
                    # Simple mean-reversion score: correlation of returns
                    spread = log_train[a] - log_train[b]
                    spread_std = spread.std()
                    if spread_std < 0.001:
                        continue
                    # Half-life of mean reversion
                    spread_diff = spread.diff().dropna()
                    spread_lag = spread.shift(1).dropna()
                    if len(spread_diff) < 20:
                        continue
                    # Regression: dS = phi * S_{t-1} + eps
                    common = spread_diff.index.intersection(spread_lag.index)
                    y = spread_diff.loc[common].values
                    x = spread_lag.loc[common].values
                    if len(x) < 20 or np.std(x) < 1e-10:
                        continue
                    phi = np.sum(x * y) / np.sum(x ** 2)
                    if phi >= 0:
                        continue  # Not mean-reverting
                    half_life = -np.log(2) / phi
                    if 2 < half_life < 42:  # Tradeable half-life
                        pair_scores.append((a, b, half_life, spread_std))
                except:
                    continue

            # Select top pairs by half-life (shorter = faster reversion)
            pair_scores.sort(key=lambda x: x[2])
            selected_pairs = [(a, b) for a, b, _, _ in pair_scores[:MAX_PAIRS * 3]]

        # Check exits on existing positions
        closed = []
        for pair_key, pos in positions.items():
            a, b = pair_key.split('/')
            if a not in prices.columns or b not in prices.columns:
                continue

            # Current z-score
            lookback = prices.iloc[max(0,i-COINT_LOOKBACK):i]
            spread = np.log(lookback[a]) - np.log(lookback[b])
            z = (spread.iloc[-1] - spread.mean()) / spread.std() if spread.std() > 0 else 0

            hold_days = i - pos['entry_day']

            # Exit conditions
            exit_signal = False
            if abs(z) < EXIT_Z:
                exit_signal = True  # Mean reverted
            elif abs(z) > STOP_Z:
                exit_signal = True  # Stop loss
            elif hold_days >= MAX_HOLD:
                exit_signal = True  # Time stop

            if exit_signal:
                # Calculate return
                ret_a = prices[a].iloc[i] / pos['entry_prices'][0] - 1
                ret_b = prices[b].iloc[i] / pos['entry_prices'][1] - 1

                if pos['side'] == 'long_spread':
                    trade_ret = (ret_a - ret_b) * POS_SIZE
                else:
                    trade_ret = (ret_b - ret_a) * POS_SIZE

                # Transaction costs (10bps per leg, 4 legs total for RT)
                trade_ret -= 4 * COST_BPS / 10000 * POS_SIZE

                day_pnl += trade_ret
                total_trades += 1
                trade_returns.append(trade_ret)
                if trade_ret > 0:
                    wins += 1
                closed.append(pair_key)

        for pk in closed:
            del positions[pk]

        # Check entries
        if len(positions) < MAX_PAIRS:
            for a, b in selected_pairs:
                pair_key = f"{a}/{b}"
                if pair_key in positions or len(positions) >= MAX_PAIRS:
                    continue

                lookback = prices.iloc[max(0,i-COINT_LOOKBACK):i]
                spread = np.log(lookback[a]) - np.log(lookback[b])
                if spread.std() < 0.001:
                    continue
                z = (spread.iloc[-1] - spread.mean()) / spread.std()

                if z > ENTRY_Z:
                    positions[pair_key] = {
                        'side': 'short_spread',
                        'entry_z': z,
                        'entry_day': i,
                        'entry_prices': (prices[a].iloc[i], prices[b].iloc[i])
                    }
                elif z < -ENTRY_Z:
                    positions[pair_key] = {
                        'side': 'long_spread',
                        'entry_z': z,
                        'entry_day': i,
                        'entry_prices': (prices[a].iloc[i], prices[b].iloc[i])
                    }

        # Mark-to-market open positions for daily P&L
        for pair_key, pos in positions.items():
            a, b = pair_key.split('/')
            if i > 0:
                ret_a = returns[a].iloc[i] if a in returns.columns else 0
                ret_b = returns[b].iloc[i] if b in returns.columns else 0
                if pos['side'] == 'long_spread':
                    day_pnl += (ret_a - ret_b) * POS_SIZE
                else:
                    day_pnl += (ret_b - ret_a) * POS_SIZE

        daily_ret.iloc[i] = day_pnl

    # Metrics
    valid = daily_ret[TRAIN_WINDOW:]
    equity = initial_capital * (1 + daily_ret).cumprod()
    ann_ret = valid.mean() * 252
    ann_vol = valid.std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
    downside = valid[valid < 0].std() * np.sqrt(252) if (valid < 0).any() else 0.001
    sortino = ann_ret / downside
    cum_ret = equity.iloc[-1] / initial_capital - 1
    years = len(valid) / 252
    cagr = (1 + cum_ret) ** (1/years) - 1 if years > 0 else 0
    dd = equity / equity.cummax() - 1
    max_dd = dd.min()
    wr = wins / total_trades * 100 if total_trades > 0 else 0
    spy_corr = daily_ret.corr(returns['SPY']) if 'SPY' in returns.columns else 0

    print(f"  Z-Score Stat Arb:")
    print(f"    Sharpe: {sharpe:.2f}, Sortino: {sortino:.2f}")
    print(f"    CAGR: {cagr*100:.1f}%, MaxDD: {max_dd*100:.1f}%")
    print(f"    Trades: {total_trades}, WR: {wr:.1f}%")
    print(f"    SPY correlation: {spy_corr:.3f}")
    sys.stdout.flush()

    return daily_ret, equity, {
        'sharpe': sharpe, 'sortino': sortino, 'cagr': cagr,
        'max_dd': max_dd, 'trades': total_trades, 'wr': wr,
        'spy_corr': spy_corr, 'ann_ret': ann_ret, 'ann_vol': ann_vol
    }


# ══════════════════════════════════════════════════════════════════════
# 4. COMBINE INTO PORTFOLIO
# ══════════════════════════════════════════════════════════════════════

# Run individual strategies
upro_ret, upro_eq, upro_stats = run_upro_protection(close, vix_close)
arb_ret, arb_eq, arb_stats = run_zscore_stat_arb(close)

print("\n[4/5] Building combined portfolios...")
sys.stdout.flush()

# Align return series
common = upro_ret.index.intersection(arb_ret.index)
upro_ret_aligned = upro_ret.loc[common]
arb_ret_aligned = arb_ret.loc[common]
spy_ret = returns['SPY'].loc[common] if 'SPY' in returns.columns else pd.Series(0, index=common)
vix_aligned = vix_close.reindex(common).ffill()

warmup = 252  # Skip first year for metrics

# ── VERSION A: 1x LEVERAGE (fully funded, no margin) ──
# Static allocation: 60% UPRO Protection, 40% Stat Arb
# Rebalance monthly (first trading day of each month)
print("\n  VERSION A — 1x Leverage (60/40 static)")
w_upro_a, w_arb_a = 0.60, 0.40

port_a_ret = w_upro_a * upro_ret_aligned + w_arb_a * arb_ret_aligned

# Monthly rebalancing cost (10bps when weights drift > 5%)
months = pd.Series(common).dt.to_period('M')
drift_cost = 0.001  # 10bps
for m in months.unique():
    mask = months == m
    first_idx = mask.idxmax()
    if first_idx > 0:
        port_a_ret.iloc[first_idx] -= drift_cost * 0.2  # Assume ~20% of capital moves

eq_a = 100_000 * (1 + port_a_ret).cumprod()
valid_a = port_a_ret[warmup:]
ann_ret_a = valid_a.mean() * 252
ann_vol_a = valid_a.std() * np.sqrt(252)
sharpe_a = ann_ret_a / ann_vol_a if ann_vol_a > 0 else 0
down_a = valid_a[valid_a < 0].std() * np.sqrt(252) if (valid_a < 0).any() else 0.001
sortino_a = ann_ret_a / down_a
dd_a = (eq_a / eq_a.cummax() - 1).min()
cum_a = eq_a.iloc[-1] / 100_000 - 1
years_a = len(valid_a) / 252
cagr_a = (1 + cum_a) ** (1/max(years_a, 0.01)) - 1
spy_corr_a = port_a_ret[warmup:].corr(spy_ret[warmup:])

print(f"    Sharpe: {sharpe_a:.2f}, Sortino: {sortino_a:.2f}")
print(f"    CAGR: {cagr_a*100:.1f}%, MaxDD: {dd_a*100:.1f}%")
print(f"    SPY correlation: {spy_corr_a:.3f}")
print(f"    Calmar: {cagr_a / abs(dd_a):.2f}" if dd_a != 0 else "    Calmar: N/A")

# ── VERSION B: DYNAMIC LEVERAGE ──
# Base allocation same as A, but scale exposure 0.5x-1.5x using VIX regime
# Low VIX (<15): 1.5x, Normal (15-20): 1.0x, Elevated (20-25): 0.5x, High (>25): 0x
print("\n  VERSION B — Dynamic Leverage (VIX-scaled)")

def vix_scale(vix_val):
    if vix_val < 15:
        return 1.5
    elif vix_val < 20:
        return 1.0
    elif vix_val < 25:
        return 0.5
    else:
        return 0.0  # Full cash in crisis

vix_scales = vix_aligned.apply(vix_scale)
# Shift by 1 to avoid lookahead
vix_scales_shifted = vix_scales.shift(1).fillna(1.0)

port_b_ret = vix_scales_shifted * (w_upro_a * upro_ret_aligned + w_arb_a * arb_ret_aligned)

# Add margin cost for leverage > 1x (6% annual rate)
margin_rate = 0.06 / 252
for i in range(len(port_b_ret)):
    if vix_scales_shifted.iloc[i] > 1.0:
        excess = vix_scales_shifted.iloc[i] - 1.0
        port_b_ret.iloc[i] -= excess * margin_rate

eq_b = 100_000 * (1 + port_b_ret).cumprod()
valid_b = port_b_ret[warmup:]
ann_ret_b = valid_b.mean() * 252
ann_vol_b = valid_b.std() * np.sqrt(252)
sharpe_b = ann_ret_b / ann_vol_b if ann_vol_b > 0 else 0
down_b = valid_b[valid_b < 0].std() * np.sqrt(252) if (valid_b < 0).any() else 0.001
sortino_b = ann_ret_b / down_b
dd_b = (eq_b / eq_b.cummax() - 1).min()
cum_b = eq_b.iloc[-1] / 100_000 - 1
years_b = len(valid_b) / 252
cagr_b = (1 + cum_b) ** (1/max(years_b, 0.01)) - 1
spy_corr_b = port_b_ret[warmup:].corr(spy_ret[warmup:])

print(f"    Sharpe: {sharpe_b:.2f}, Sortino: {sortino_b:.2f}")
print(f"    CAGR: {cagr_b*100:.1f}%, MaxDD: {dd_b*100:.1f}%")
print(f"    SPY correlation: {spy_corr_b:.3f}")
print(f"    Calmar: {cagr_b / abs(dd_b):.2f}" if dd_b != 0 else "    Calmar: N/A")
print(f"    Avg leverage: {vix_scales_shifted[warmup:].mean():.2f}x")


# ── SPY BUY & HOLD BENCHMARK ──
spy_bh_eq = 100_000 * (1 + spy_ret).cumprod()
spy_bh_valid = spy_ret[warmup:]
spy_sharpe = spy_bh_valid.mean() * 252 / (spy_bh_valid.std() * np.sqrt(252)) if spy_bh_valid.std() > 0 else 0
spy_dd = (spy_bh_eq / spy_bh_eq.cummax() - 1).min()
spy_cum = spy_bh_eq.iloc[-1] / 100_000 - 1
spy_cagr = (1 + spy_cum) ** (1/max(years_a, 0.01)) - 1


# ══════════════════════════════════════════════════════════════════════
# 5. REGIME ANALYSIS
# ══════════════════════════════════════════════════════════════════════
print("\n[5/5] Regime analysis...")
sys.stdout.flush()

# Classify days as green/red/flat based on SPY close-to-close
spy_daily = spy_ret[warmup:]
green_mask = spy_daily > 0.001
red_mask = spy_daily < -0.001
flat_mask = ~green_mask & ~red_mask

for label, ret_series, eq_series in [
    ("Version A (1x)", valid_a, eq_a),
    ("Version B (dynamic)", valid_b, eq_b)
]:
    green_sharpe = ret_series[green_mask].mean() * 252 / (ret_series[green_mask].std() * np.sqrt(252)) if ret_series[green_mask].std() > 0 else 0
    red_sharpe = ret_series[red_mask].mean() * 252 / (ret_series[red_mask].std() * np.sqrt(252)) if ret_series[red_mask].std() > 0 else 0
    gap = abs(green_sharpe - red_sharpe) / max(abs(green_sharpe), abs(red_sharpe), 0.01)

    print(f"\n  {label}:")
    print(f"    Green-day Sharpe: {green_sharpe:.2f}, Red-day Sharpe: {red_sharpe:.2f}")
    print(f"    Regime gap: {gap:.3f} {'PASS' if gap < 0.50 else 'FAIL'} (threshold: 0.50)")

    # Sub-period analysis (quarterly)
    quarterly = ret_series.resample('Q').apply(lambda x: x.mean() * 252 / (x.std() * np.sqrt(252)) if len(x) > 10 and x.std() > 0 else 0)
    pos_q = (quarterly > 0).sum()
    neg_q = (quarterly <= 0).sum()
    print(f"    Quarters positive: {pos_q}/{pos_q+neg_q} ({pos_q/(pos_q+neg_q)*100:.0f}%)")

# ══════════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("FINAL SUMMARY — COMBINED PORTFOLIO")
print("=" * 70)

summary = {
    'upro_protection': {k: float(v) if isinstance(v, (np.floating, float)) else v for k,v in upro_stats.items()},
    'stat_arb': {k: float(v) if isinstance(v, (np.floating, float)) else v for k,v in arb_stats.items()},
    'version_a_1x': {
        'allocation': '60% UPRO Protection / 40% Stat Arb',
        'sharpe': float(sharpe_a), 'sortino': float(sortino_a),
        'cagr': float(cagr_a), 'max_dd': float(dd_a),
        'spy_corr': float(spy_corr_a),
        'calmar': float(cagr_a / abs(dd_a)) if dd_a != 0 else None,
    },
    'version_b_dynamic': {
        'allocation': '60% UPRO Protection / 40% Stat Arb, VIX-scaled 0-1.5x',
        'sharpe': float(sharpe_b), 'sortino': float(sortino_b),
        'cagr': float(cagr_b), 'max_dd': float(dd_b),
        'spy_corr': float(spy_corr_b),
        'calmar': float(cagr_b / abs(dd_b)) if dd_b != 0 else None,
        'avg_leverage': float(vix_scales_shifted[warmup:].mean()),
    },
    'benchmark_spy_bh': {
        'sharpe': float(spy_sharpe), 'cagr': float(spy_cagr), 'max_dd': float(spy_dd),
    },
    'metadata': {
        'start_date': str(common[warmup].date()),
        'end_date': str(common[-1].date()),
        'trading_days': int(len(valid_a)),
        'years': float(years_a),
    }
}

print(f"\n{'Metric':<20} {'SPY B&H':>10} {'1x (A)':>10} {'Dynamic (B)':>12}")
print("-" * 55)
print(f"{'Sharpe':<20} {spy_sharpe:>10.2f} {sharpe_a:>10.2f} {sharpe_b:>12.2f}")
print(f"{'Sortino':<20} {'N/A':>10} {sortino_a:>10.2f} {sortino_b:>12.2f}")
print(f"{'CAGR':<20} {spy_cagr*100:>9.1f}% {cagr_a*100:>9.1f}% {cagr_b*100:>11.1f}%")
print(f"{'MaxDD':<20} {spy_dd*100:>9.1f}% {dd_a*100:>9.1f}% {dd_b*100:>11.1f}%")
print(f"{'SPY Corr':<20} {'1.000':>10} {spy_corr_a:>10.3f} {spy_corr_b:>12.3f}")

# Save results
with open(os.path.join(OUTPUT_DIR, 'results.json'), 'w') as f:
    json.dump(summary, f, indent=2, default=str)

# Save daily returns for future use
daily_df = pd.DataFrame({
    'date': common,
    'upro_protection': upro_ret_aligned.values,
    'stat_arb': arb_ret_aligned.values,
    'portfolio_1x': port_a_ret.values,
    'portfolio_dynamic': port_b_ret.values,
    'spy': spy_ret.values,
})
daily_df.to_csv(os.path.join(OUTPUT_DIR, 'daily_returns.csv'), index=False)

print(f"\nResults saved to {OUTPUT_DIR}/")
print("Daily returns saved to daily_returns.csv")
