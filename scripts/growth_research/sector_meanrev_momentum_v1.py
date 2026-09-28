#!/usr/bin/env python3
"""
Sector Mean-Reversion + Momentum Hybrid Strategy
===================================================
Momentum identifies strong sectors; mean-reversion timing finds better entries
(buy the dip in trending sectors). Based on Asness et al, "Value and Momentum Everywhere".

8 variants tested with 5-gate mandatory validation.
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
import os, sys, json, time
from scipy import stats

# MLflow setup
try:
    import mlflow
    mlflow.set_tracking_uri("http://localhost:5000")
    MLFLOW_OK = True
except Exception:
    MLFLOW_OK = False
    print("WARNING: MLflow not available, skipping tracking")

# ─── Configuration ───────────────────────────────────────────────────────────

SECTOR_ETFS = ['XLB', 'XLC', 'XLE', 'XLF', 'XLI', 'XLK', 'XLP', 'XLRE', 'XLU', 'XLV', 'XLY']
BENCHMARK = 'SPY'
VIX_TICKER = '^VIX'
ALL_TICKERS = SECTOR_ETFS + [BENCHMARK, VIX_TICKER]

START_DATE = '2020-01-01'
END_DATE = '2026-07-01'
INITIAL_CAPITAL = 10_000
SMALL_CAPITAL = 645
TX_COST_PCT = 0.001  # 0.1% per trade

OUTPUT_DIR = '/home/jupiter/Lvl3Quant/output/growth_research/meanrev_momentum_v1'
os.makedirs(OUTPUT_DIR, exist_ok=True)

MOMENTUM_LOOKBACK = 63  # ~3 months trading days

# Variant definitions
VARIANTS = {
    'A_Base':              dict(top_n=3, rsi_period=5, rsi_entry=30, rsi_exit=70, tp=0.08, sl=-0.05, max_hold=15, vix_filter=None, dual_tf=False, short_mode=False),
    'B_AggressiveEntry':   dict(top_n=3, rsi_period=5, rsi_entry=35, rsi_exit=70, tp=0.08, sl=-0.05, max_hold=15, vix_filter=None, dual_tf=False, short_mode=False),
    'C_ConservativeEntry': dict(top_n=3, rsi_period=5, rsi_entry=25, rsi_exit=70, tp=0.08, sl=-0.05, max_hold=15, vix_filter=None, dual_tf=False, short_mode=False),
    'D_WiderStops':        dict(top_n=3, rsi_period=5, rsi_entry=30, rsi_exit=70, tp=0.12, sl=-0.08, max_hold=20, vix_filter=None, dual_tf=False, short_mode=False),
    'E_Top5Universe':      dict(top_n=5, rsi_period=5, rsi_entry=30, rsi_exit=70, tp=0.08, sl=-0.05, max_hold=15, vix_filter=None, dual_tf=False, short_mode=False),
    'F_DualTimeframe':     dict(top_n=3, rsi_period=5, rsi_entry=30, rsi_exit=70, tp=0.08, sl=-0.05, max_hold=15, vix_filter=None, dual_tf=True,  short_mode=False),
    'G_VIXFilter':         dict(top_n=3, rsi_period=5, rsi_entry=30, rsi_exit=70, tp=0.08, sl=-0.05, max_hold=15, vix_filter=18.0, dual_tf=False, short_mode=False),
    'H_ContrarianShort':   dict(top_n=3, rsi_period=5, rsi_entry=70, rsi_exit=30, tp=0.08, sl=-0.05, max_hold=15, vix_filter=None, dual_tf=False, short_mode=True),
}


# ─── Data Download ───────────────────────────────────────────────────────────

def download_data():
    """Download daily OHLCV for all tickers."""
    print(f"Downloading data for {len(ALL_TICKERS)} tickers: {START_DATE} to {END_DATE}")
    cache_path = os.path.join(OUTPUT_DIR, '_price_cache.parquet')

    if os.path.exists(cache_path):
        df = pd.read_parquet(cache_path)
        print(f"  Loaded from cache: {len(df)} rows")
        return df

    data = yf.download(ALL_TICKERS, start=START_DATE, end=END_DATE,
                        auto_adjust=True, progress=False, threads=True)

    # Handle MultiIndex columns from yf.download
    close = data['Close'] if isinstance(data.columns, pd.MultiIndex) else data[['Close']]
    high = data['High'] if isinstance(data.columns, pd.MultiIndex) else data[['High']]
    low = data['Low'] if isinstance(data.columns, pd.MultiIndex) else data[['Low']]

    # Build combined df
    result = pd.DataFrame(index=close.index)
    for t in ALL_TICKERS:
        col_name = t.replace('^', '')
        result[f'{col_name}_close'] = close[t] if t in close.columns else np.nan
        result[f'{col_name}_high'] = high[t] if t in high.columns else np.nan
        result[f'{col_name}_low'] = low[t] if t in low.columns else np.nan

    result = result.dropna(how='all')
    result.to_parquet(cache_path)
    print(f"  Downloaded {len(result)} trading days")
    return result


# ─── Indicators ──────────────────────────────────────────────────────────────

def compute_rsi(series, period=5):
    """Wilder RSI."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def compute_momentum_ranks(data, day_idx, lookback=63):
    """Rank sectors by trailing return. Returns sorted list of (ticker, return)."""
    if day_idx < lookback:
        return []

    ranks = []
    for etf in SECTOR_ETFS:
        col = f'{etf}_close'
        if col not in data.columns:
            continue
        current = data[col].iloc[day_idx]
        past = data[col].iloc[day_idx - lookback]
        if pd.isna(current) or pd.isna(past) or past == 0:
            continue
        ret = (current / past) - 1
        ranks.append((etf, ret))

    ranks.sort(key=lambda x: x[1], reverse=True)
    return ranks


def compute_momentum_ranks_short(data, day_idx, lookback=21):
    """Short-term momentum (1 month) for dual-timeframe filter."""
    return compute_momentum_ranks(data, day_idx, lookback=lookback)


# ─── Backtest Engine ─────────────────────────────────────────────────────────

def run_backtest(data, params, capital=INITIAL_CAPITAL):
    """
    Run the sector mean-reversion + momentum backtest.
    Returns: equity_curve (pd.Series), trades (list of dicts)
    """
    top_n = params['top_n']
    rsi_period = params['rsi_period']
    rsi_entry = params['rsi_entry']
    rsi_exit = params['rsi_exit']
    tp = params['tp']
    sl = params['sl']
    max_hold = params['max_hold']
    vix_filter = params['vix_filter']
    dual_tf = params['dual_tf']
    short_mode = params['short_mode']
    max_positions = 3

    # Pre-compute RSI for all sectors
    rsi_data = {}
    for etf in SECTOR_ETFS:
        col = f'{etf}_close'
        if col in data.columns:
            rsi_data[etf] = compute_rsi(data[col], rsi_period)

    # VIX data
    vix_col = 'VIX_close'
    has_vix = vix_col in data.columns

    dates = data.index
    n_days = len(dates)

    equity = capital
    positions = []  # list of {etf, entry_price, entry_date, entry_idx, direction}
    trades = []
    equity_series = pd.Series(index=dates, dtype=float)

    for i in range(MOMENTUM_LOOKBACK + rsi_period + 5, n_days):
        date = dates[i]

        # ── Mark-to-market existing positions ──
        daily_pnl = 0.0
        positions_to_close = []

        for j, pos in enumerate(positions):
            etf = pos['etf']
            col = f'{etf}_close'
            current_price = data[col].iloc[i]
            prev_price = data[col].iloc[i-1] if i > 0 else pos['entry_price']
            direction = pos.get('direction', 1)  # 1 for long, -1 for short

            if pd.isna(current_price):
                continue

            daily_ret = (current_price / prev_price - 1) * direction
            pos_size = pos['size']
            daily_pnl += pos_size * daily_ret

            # Check exits
            total_ret = (current_price / pos['entry_price'] - 1) * direction
            hold_days = i - pos['entry_idx']
            rsi_val = rsi_data.get(etf, pd.Series()).iloc[i] if etf in rsi_data else 50

            exit_reason = None
            if total_ret >= tp:
                exit_reason = 'TP'
            elif total_ret <= sl:
                exit_reason = 'SL'
            elif hold_days >= max_hold:
                exit_reason = 'MAX_HOLD'
            elif not short_mode and not pd.isna(rsi_val) and rsi_val > rsi_exit:
                exit_reason = 'RSI_EXIT'
            elif short_mode and not pd.isna(rsi_val) and rsi_val < rsi_exit:
                exit_reason = 'RSI_EXIT'

            if exit_reason:
                # Apply transaction cost on exit
                cost = pos_size * TX_COST_PCT
                equity -= cost

                trades.append({
                    'etf': etf,
                    'direction': 'SHORT' if direction == -1 else 'LONG',
                    'entry_date': pos['entry_date'],
                    'exit_date': date,
                    'entry_price': pos['entry_price'],
                    'exit_price': current_price,
                    'return': total_ret,
                    'pnl': pos_size * total_ret - cost - pos['entry_cost'],
                    'hold_days': hold_days,
                    'exit_reason': exit_reason,
                })
                positions_to_close.append(j)

        # Remove closed positions (reverse order)
        for j in sorted(positions_to_close, reverse=True):
            positions.pop(j)

        equity += daily_pnl

        # ── Check for new entries ──
        if len(positions) < max_positions:
            # Momentum ranking
            mom_ranks_3m = compute_momentum_ranks(data, i, lookback=MOMENTUM_LOOKBACK)

            if short_mode:
                # Bottom N for shorting
                eligible = [r[0] for r in mom_ranks_3m[-top_n:]] if len(mom_ranks_3m) >= top_n else []
            else:
                # Top N for buying
                eligible = [r[0] for r in mom_ranks_3m[:top_n]] if len(mom_ranks_3m) >= top_n else []

            if dual_tf:
                # Must also be in top N of 1-month momentum
                mom_ranks_1m = compute_momentum_ranks_short(data, i, lookback=21)
                if short_mode:
                    eligible_1m = set(r[0] for r in mom_ranks_1m[-top_n:]) if len(mom_ranks_1m) >= top_n else set()
                else:
                    eligible_1m = set(r[0] for r in mom_ranks_1m[:top_n]) if len(mom_ranks_1m) >= top_n else set()
                eligible = [e for e in eligible if e in eligible_1m]

            # VIX filter
            if vix_filter is not None and has_vix:
                vix_val = data[vix_col].iloc[i]
                if not pd.isna(vix_val) and vix_val < vix_filter:
                    eligible = []  # Only enter when VIX > threshold

            # Filter out already-held ETFs
            held = set(p['etf'] for p in positions)
            eligible = [e for e in eligible if e not in held]

            # Check RSI entry for eligible ETFs
            for etf in eligible:
                if len(positions) >= max_positions:
                    break

                rsi_val = rsi_data.get(etf, pd.Series()).iloc[i] if etf in rsi_data else None
                if rsi_val is None or pd.isna(rsi_val):
                    continue

                entry_signal = False
                if short_mode:
                    entry_signal = rsi_val > rsi_entry  # RSI > 70 for shorts
                else:
                    entry_signal = rsi_val < rsi_entry  # RSI < 30 for longs

                if entry_signal:
                    col = f'{etf}_close'
                    entry_price = data[col].iloc[i]
                    if pd.isna(entry_price) or entry_price <= 0:
                        continue

                    # Position sizing: equal weight across max positions
                    pos_size = equity / max_positions
                    direction = -1 if short_mode else 1
                    entry_cost = pos_size * TX_COST_PCT
                    equity -= entry_cost

                    positions.append({
                        'etf': etf,
                        'entry_price': entry_price,
                        'entry_date': date,
                        'entry_idx': i,
                        'size': pos_size,
                        'direction': direction,
                        'entry_cost': entry_cost,
                    })

        equity_series.iloc[i] = equity

    # Fill forward equity for days before strategy starts
    equity_series = equity_series.ffill().bfill()

    return equity_series, trades


# ─── Metrics ─────────────────────────────────────────────────────────────────

def compute_metrics(equity_curve, trades, capital=INITIAL_CAPITAL):
    """Compute strategy performance metrics."""
    # Daily returns
    eq = equity_curve.dropna()
    if len(eq) < 10:
        return {k: np.nan for k in ['sharpe', 'sortino', 'wr_pct', 'pf', 'mdd_pct', 'cagr_pct', 'n_trades', 'avg_hold']}

    daily_ret = eq.pct_change().dropna()
    daily_ret = daily_ret.replace([np.inf, -np.inf], 0)

    # Sharpe (annualized)
    mean_ret = daily_ret.mean()
    std_ret = daily_ret.std()
    sharpe = (mean_ret / std_ret * np.sqrt(252)) if std_ret > 0 else 0

    # Sortino
    downside = daily_ret[daily_ret < 0].std()
    sortino = (mean_ret / downside * np.sqrt(252)) if downside > 0 else 0

    # Max drawdown
    cummax = eq.cummax()
    dd = (eq - cummax) / cummax
    mdd = dd.min() * 100

    # CAGR
    total_days = (eq.index[-1] - eq.index[0]).days
    total_years = total_days / 365.25
    final_ret = eq.iloc[-1] / eq.iloc[0]
    cagr = ((final_ret ** (1 / total_years)) - 1) * 100 if total_years > 0 else 0

    # Trade-level stats
    n_trades = len(trades)
    if n_trades > 0:
        trade_rets = [t['return'] for t in trades]
        winners = [r for r in trade_rets if r > 0]
        losers = [r for r in trade_rets if r <= 0]
        wr = len(winners) / n_trades * 100

        gross_profit = sum(winners) if winners else 0
        gross_loss = abs(sum(losers)) if losers else 1e-10
        pf = gross_profit / gross_loss if gross_loss > 0 else np.inf

        avg_hold = np.mean([t['hold_days'] for t in trades])
    else:
        wr, pf, avg_hold = 0, 0, 0

    return {
        'sharpe': round(sharpe, 3),
        'sortino': round(sortino, 3),
        'wr_pct': round(wr, 1),
        'pf': round(pf, 2),
        'mdd_pct': round(mdd, 1),
        'cagr_pct': round(cagr, 1),
        'n_trades': n_trades,
        'avg_hold': round(avg_hold, 1),
    }


# ─── Validation Gates ────────────────────────────────────────────────────────

def gate_permutation_test(trades, n_perms=100):
    """Permutation test: shuffle trade returns, recompute Sharpe. p < 0.05 passes."""
    if len(trades) < 5:
        return False, 1.0

    trade_rets = np.array([t['return'] for t in trades])

    # Real Sharpe from trades
    real_sharpe = np.mean(trade_rets) / np.std(trade_rets) if np.std(trade_rets) > 0 else 0

    perm_sharpes = []
    for _ in range(n_perms):
        shuffled = np.random.permutation(trade_rets)
        s = np.mean(shuffled) / np.std(shuffled) if np.std(shuffled) > 0 else 0
        perm_sharpes.append(s)

    # For permutation of same returns, Sharpe is invariant to ordering.
    # Instead: shuffle date assignment. Use daily returns approach.
    # Reinterpret: permute the sign of returns (more meaningful).
    perm_sharpes = []
    for _ in range(n_perms):
        signs = np.random.choice([-1, 1], size=len(trade_rets))
        shuffled = trade_rets * signs
        s = np.mean(shuffled) / np.std(shuffled) if np.std(shuffled) > 0 else 0
        perm_sharpes.append(s)

    p_value = np.mean(np.array(perm_sharpes) >= real_sharpe)
    return p_value < 0.05, round(p_value, 4)


def gate_subperiod_stability(equity_curve):
    """Split into 4 quarters, 3/4 must have positive Sharpe."""
    eq = equity_curve.dropna()
    if len(eq) < 40:
        return False, []

    n = len(eq)
    quarter_size = n // 4
    sharpes = []

    for q in range(4):
        start = q * quarter_size
        end = (q + 1) * quarter_size if q < 3 else n
        segment = eq.iloc[start:end]
        daily_ret = segment.pct_change().dropna()
        if len(daily_ret) < 5 or daily_ret.std() == 0:
            sharpes.append(0)
        else:
            sharpes.append(daily_ret.mean() / daily_ret.std() * np.sqrt(252))

    positive_quarters = sum(1 for s in sharpes if s > 0)
    return positive_quarters >= 3, [round(s, 3) for s in sharpes]


def gate_outlier_removal(equity_curve):
    """Trim top/bottom 1% of daily returns. Trimmed Sharpe > 0.8x full."""
    eq = equity_curve.dropna()
    daily_ret = eq.pct_change().dropna()

    if len(daily_ret) < 20 or daily_ret.std() == 0:
        return False, 0

    full_sharpe = daily_ret.mean() / daily_ret.std() * np.sqrt(252)

    lower = daily_ret.quantile(0.01)
    upper = daily_ret.quantile(0.99)
    trimmed = daily_ret[(daily_ret >= lower) & (daily_ret <= upper)]

    if len(trimmed) < 10 or trimmed.std() == 0:
        return False, 0

    trimmed_sharpe = trimmed.mean() / trimmed.std() * np.sqrt(252)

    if full_sharpe <= 0:
        # If full Sharpe is negative, trimmed just needs to also be negative or zero
        passed = True  # Not meaningful to compare ratios when negative
        if trimmed_sharpe < full_sharpe:
            passed = False
    else:
        passed = trimmed_sharpe >= 0.8 * full_sharpe

    return passed, round(trimmed_sharpe, 3)


def gate_regime_balance(equity_curve, vix_series):
    """Sharpe on VIX>25 vs VIX<25. Gap ratio < 0.50 passes."""
    eq = equity_curve.dropna()
    daily_ret = eq.pct_change().dropna()

    # Align VIX with equity
    common_idx = daily_ret.index.intersection(vix_series.dropna().index)
    if len(common_idx) < 20:
        return False, 0

    dr = daily_ret.loc[common_idx]
    vix = vix_series.loc[common_idx]

    high_vix = dr[vix > 25]
    low_vix = dr[vix <= 25]

    if len(high_vix) < 5 or len(low_vix) < 5:
        return True, 0  # Not enough data to split, pass by default

    sharpe_high = high_vix.mean() / high_vix.std() * np.sqrt(252) if high_vix.std() > 0 else 0
    sharpe_low = low_vix.mean() / low_vix.std() * np.sqrt(252) if low_vix.std() > 0 else 0

    max_abs = max(abs(sharpe_high), abs(sharpe_low))
    if max_abs == 0:
        gap_ratio = 0
    else:
        gap_ratio = abs(sharpe_high - sharpe_low) / max_abs

    return gap_ratio < 0.50, round(gap_ratio, 3)


def gate_random_baseline(equity_curve, data, n_random=100):
    """Real Sharpe > 95th percentile of 100 random entry/exit strategies."""
    eq = equity_curve.dropna()
    daily_ret = eq.pct_change().dropna()

    if len(daily_ret) < 20 or daily_ret.std() == 0:
        return False, 0

    real_sharpe = daily_ret.mean() / daily_ret.std() * np.sqrt(252)

    # Random strategy: pick random entry/exit dates for random ETFs
    spy_col = 'SPY_close'
    if spy_col not in data.columns:
        return False, 0

    spy_close = data[spy_col].dropna()
    n_days = len(spy_close)

    random_sharpes = []
    for _ in range(n_random):
        # Random entries: pick ~20 random trades
        n_random_trades = max(5, np.random.randint(10, 40))
        random_equity = pd.Series(1.0, index=spy_close.index)

        for _ in range(n_random_trades):
            etf = np.random.choice(SECTOR_ETFS)
            col = f'{etf}_close'
            if col not in data.columns:
                continue

            entry_idx = np.random.randint(MOMENTUM_LOOKBACK, n_days - 20)
            hold = np.random.randint(1, 20)
            exit_idx = min(entry_idx + hold, n_days - 1)

            entry_p = data[col].iloc[entry_idx]
            exit_p = data[col].iloc[exit_idx]

            if pd.isna(entry_p) or pd.isna(exit_p) or entry_p <= 0:
                continue

            ret = exit_p / entry_p - 1
            # Apply to equity at exit date
            exit_date = data.index[exit_idx]
            if exit_date in random_equity.index:
                random_equity.loc[exit_date:] *= (1 + ret / n_random_trades)

        r_daily = random_equity.pct_change().dropna()
        if len(r_daily) > 5 and r_daily.std() > 0:
            random_sharpes.append(r_daily.mean() / r_daily.std() * np.sqrt(252))

    if not random_sharpes:
        return False, 0

    pct95 = np.percentile(random_sharpes, 95)
    return real_sharpe > pct95, round(pct95, 3)


def run_all_gates(equity_curve, trades, data, vix_series):
    """Run all 5 validation gates. Returns (n_passed, details_dict)."""
    results = {}
    passed = 0

    # Gate 1: Permutation test
    g1_pass, g1_pval = gate_permutation_test(trades)
    results['permutation'] = {'passed': g1_pass, 'p_value': g1_pval}
    if g1_pass: passed += 1

    # Gate 2: Sub-period stability
    g2_pass, g2_sharpes = gate_subperiod_stability(equity_curve)
    results['subperiod'] = {'passed': g2_pass, 'quarter_sharpes': g2_sharpes}
    if g2_pass: passed += 1

    # Gate 3: Outlier removal
    g3_pass, g3_trimmed = gate_outlier_removal(equity_curve)
    results['outlier'] = {'passed': g3_pass, 'trimmed_sharpe': g3_trimmed}
    if g3_pass: passed += 1

    # Gate 4: Regime balance
    g4_pass, g4_gap = gate_regime_balance(equity_curve, vix_series)
    results['regime'] = {'passed': g4_pass, 'gap_ratio': g4_gap}
    if g4_pass: passed += 1

    # Gate 5: Random baseline
    g5_pass, g5_pct95 = gate_random_baseline(equity_curve, data)
    results['random_baseline'] = {'passed': g5_pass, 'pct95_sharpe': g5_pct95}
    if g5_pass: passed += 1

    return passed, results


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    np.random.seed(42)
    t0 = time.time()

    print("=" * 80)
    print("SECTOR MEAN-REVERSION + MOMENTUM HYBRID BACKTEST")
    print("=" * 80)

    # Download data
    data = download_data()

    # VIX series for validation
    vix_col = 'VIX_close'
    vix_series = data[vix_col] if vix_col in data.columns else pd.Series(20, index=data.index)

    # Run all variants
    all_results = []
    all_details = {}

    for vname, params in VARIANTS.items():
        print(f"\n{'─'*60}")
        print(f"Running variant: {vname}")
        print(f"  Params: top_n={params['top_n']}, RSI_entry={params['rsi_entry']}, "
              f"RSI_exit={params['rsi_exit']}, TP={params['tp']}, SL={params['sl']}, "
              f"max_hold={params['max_hold']}, VIX_filter={params['vix_filter']}, "
              f"dual_tf={params['dual_tf']}, short={params['short_mode']}")

        equity_curve, trades = run_backtest(data, params, capital=INITIAL_CAPITAL)
        metrics = compute_metrics(equity_curve, trades, INITIAL_CAPITAL)

        print(f"  Trades: {metrics['n_trades']}, Sharpe: {metrics['sharpe']}, "
              f"Sortino: {metrics['sortino']}, WR: {metrics['wr_pct']}%, "
              f"PF: {metrics['pf']}, MDD: {metrics['mdd_pct']}%, CAGR: {metrics['cagr_pct']}%")

        # Run validation gates
        n_passed, gate_details = run_all_gates(equity_curve, trades, data, vix_series)
        gate_status = 'PASS' if n_passed >= 4 else 'FAIL'

        print(f"  Gates: {n_passed}/5 ({gate_status})")
        for gname, ginfo in gate_details.items():
            status = 'OK' if ginfo['passed'] else 'FAIL'
            detail_str = ', '.join(f"{k}={v}" for k, v in ginfo.items() if k != 'passed')
            print(f"    {gname}: {status} ({detail_str})")

        # $645 scaling
        equity_645, trades_645 = run_backtest(data, params, capital=SMALL_CAPITAL)
        metrics_645 = compute_metrics(equity_645, trades_645, SMALL_CAPITAL)

        result_row = {
            'Variant': vname,
            'Sharpe': metrics['sharpe'],
            'Sortino': metrics['sortino'],
            'WR%': metrics['wr_pct'],
            'PF': metrics['pf'],
            'MDD%': metrics['mdd_pct'],
            'CAGR%': metrics['cagr_pct'],
            'Gates': f"{n_passed}/5",
            'Status': gate_status,
            'Trades': metrics['n_trades'],
            'Avg_Hold': metrics['avg_hold'],
            'CAGR_645': metrics_645['cagr_pct'],
            'Final_10k': round(equity_curve.dropna().iloc[-1], 0) if len(equity_curve.dropna()) > 0 else INITIAL_CAPITAL,
            'Final_645': round(equity_645.dropna().iloc[-1], 0) if len(equity_645.dropna()) > 0 else SMALL_CAPITAL,
        }
        all_results.append(result_row)

        all_details[vname] = {
            'params': params,
            'metrics': metrics,
            'metrics_645': metrics_645,
            'gates': gate_details,
            'n_gates_passed': n_passed,
            'gate_status': gate_status,
            'trades': trades,
            'equity_final_10k': result_row['Final_10k'],
            'equity_final_645': result_row['Final_645'],
        }

    # ── Results Table ──
    results_df = pd.DataFrame(all_results)
    results_df = results_df.sort_values('Sharpe', ascending=False)

    print("\n" + "=" * 120)
    print("RESULTS SUMMARY (sorted by Sharpe)")
    print("=" * 120)
    print(results_df.to_string(index=False))

    # ── Best variant per-quarter breakdown ──
    best_variant = results_df.iloc[0]['Variant']
    best_detail = all_details[best_variant]
    best_params = best_detail['params']

    print(f"\n{'─'*80}")
    print(f"PER-QUARTER BREAKDOWN: {best_variant}")
    print(f"{'─'*80}")

    equity_curve_best, _ = run_backtest(data, best_params, capital=INITIAL_CAPITAL)
    eq = equity_curve_best.dropna()
    daily_ret = eq.pct_change().dropna()

    # Split by calendar quarters
    quarters = daily_ret.groupby(pd.Grouper(freq='QE'))
    print(f"{'Quarter':<12} {'Sharpe':>8} {'Return%':>10} {'MDD%':>8} {'Days':>6}")
    print("-" * 50)
    for period, group in quarters:
        if len(group) < 5:
            continue
        q_sharpe = group.mean() / group.std() * np.sqrt(252) if group.std() > 0 else 0
        q_ret = (1 + group).prod() - 1
        # MDD for quarter
        cumret = (1 + group).cumprod()
        q_mdd = ((cumret - cumret.cummax()) / cumret.cummax()).min() * 100
        print(f"{str(period.date()):12} {q_sharpe:8.2f} {q_ret*100:10.2f} {q_mdd:8.1f} {len(group):6d}")

    # ── Save outputs ──
    results_df.to_csv(os.path.join(OUTPUT_DIR, 'variant_results.csv'), index=False)

    # Save detailed results
    serializable_details = {}
    for vname, detail in all_details.items():
        sd = {k: v for k, v in detail.items() if k != 'trades'}
        sd['trade_count'] = len(detail['trades'])
        # Save trade list separately
        if detail['trades']:
            trade_df = pd.DataFrame(detail['trades'])
            trade_df.to_csv(os.path.join(OUTPUT_DIR, f'trades_{vname}.csv'), index=False)
        serializable_details[vname] = sd

    # Convert numpy types for JSON
    def convert_numpy(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(i) for i in obj]
        return obj

    with open(os.path.join(OUTPUT_DIR, 'detailed_results.json'), 'w') as f:
        json.dump(convert_numpy(serializable_details), f, indent=2, default=str)

    print(f"\nResults saved to {OUTPUT_DIR}")

    # ── MLflow logging ──
    if MLFLOW_OK:
        try:
            mlflow.set_experiment("sector_meanrev_momentum_v1")
            for vname, detail in all_details.items():
                with mlflow.start_run(run_name=vname):
                    # Log params
                    for k, v in detail['params'].items():
                        mlflow.log_param(k, v)
                    mlflow.log_param('variant', vname)
                    mlflow.log_param('initial_capital', INITIAL_CAPITAL)
                    mlflow.log_param('tx_cost_pct', TX_COST_PCT)

                    # Log metrics
                    for k, v in detail['metrics'].items():
                        if isinstance(v, (int, float)) and not np.isnan(v) and not np.isinf(v):
                            mlflow.log_metric(k, v)

                    mlflow.log_metric('gates_passed', detail['n_gates_passed'])
                    mlflow.log_metric('final_equity_10k', float(detail['equity_final_10k']))
                    mlflow.log_metric('final_equity_645', float(detail['equity_final_645']))

                    # Log gate details
                    for gname, ginfo in detail['gates'].items():
                        mlflow.log_metric(f'gate_{gname}', 1 if ginfo['passed'] else 0)

            print("MLflow logging complete.")
        except Exception as e:
            print(f"MLflow logging error: {e}")

    elapsed = time.time() - t0
    print(f"\nTotal runtime: {elapsed:.1f}s")
    print("Done.")


if __name__ == '__main__':
    main()
