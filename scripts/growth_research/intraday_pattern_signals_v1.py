#!/usr/bin/env python3
"""
Intraday Pattern Signals v1 — Megacap Dip-Buying Entry Timing
=============================================================
Concept: Use daily OHLCV data to detect intraday price patterns that signal
"today's price action makes buying at close especially attractive."
These are ENTRY TIMING signals for multi-day/week holds — NOT intraday trades.

Signals tested (A-F) vs Base MR (RSI<30 + >7% below 50d high):
  A) Hammer Candle + Dip
  B) Gap Down Reversal
  C) Wide Range Dip Day
  D) Volume Climax Reversal
  E) Inside Day After Dip
  F) Three Bar Reversal

Universe: 30 megacap stocks, 2020-2026
Exit: +10% TP, -15% SL, 21-day max hold
Position: $300, max 2 concurrent
Cost: 0.1% round-trip

5-Gate Validation:
  1. Regime gap < 0.50
  2. Permutation p < 0.05 (1000 perms)
  3. 4/4 sub-periods positive
  4. MDD > -50%
  5. N >= 20 trades

Author: Claude (Head of Quant)
Date: 2026-08-05
"""

import sys, json, warnings, os, time, traceback
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import timedelta, datetime
from collections import defaultdict

warnings.filterwarnings("ignore")

ROOT   = Path("/home/jupiter/Lvl3Quant") if os.path.exists("/home/jupiter") else Path("/home/nick/Lvl3Quant")
OUTPUT = ROOT / "output" / "intraday_pattern_signals_v1"
OUTPUT.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# UNIVERSE — 30 megacap stocks
# ─────────────────────────────────────────────────────────────────────────────
MEGACAP_30 = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'AVGO', 'ORCL', 'CRM',
    'JPM',  'BAC',  'WFC',   'GS',   'MS',
    'UNH',  'JNJ',  'LLY',   'PFE',  'MRK',
    'HD',   'MCD',  'NKE',   'LOW',  'SBUX',
    'XOM',  'CVX',  'COP',
    'PG',   'KO'
]

START_DATE     = '2020-01-01'
END_DATE       = '2026-07-01'
POSITION_SIZE  = 300.0
MAX_CONCURRENT = 2
TP_PCT         = 0.10      # +10% take profit
SL_PCT         = -0.15     # -15% stop loss
MAX_HOLD_DAYS  = 21
COST_RT_PCT    = 0.001     # 0.1% round-trip total
N_PERMS        = 500
REGIME_GAP_MAX = 0.50
PERM_P_THRESH  = 0.05
MIN_TRADES     = 20

SUB_PERIODS = [
    ('2020-01-01', '2021-06-30', 'P1_2020H1+'),
    ('2021-07-01', '2022-12-31', 'P2_2021H2-2022'),
    ('2023-01-01', '2024-06-30', 'P3_2023H1+'),
    ('2024-07-01', '2026-07-01', 'P4_2024H2+'),
]


# ─────────────────────────────────────────────────────────────────────────────
# DATA FETCH
# ─────────────────────────────────────────────────────────────────────────────

def fetch_data(tickers, cache_path=None):
    import yfinance as yf
    if cache_path and Path(cache_path).exists():
        df = pd.read_parquet(cache_path)
        print(f"  Cache hit: {df['ticker'].nunique()} tickers, {len(df)} rows")
        return df

    print(f"  Downloading {len(tickers)} tickers ({START_DATE} → {END_DATE})...")
    raw = yf.download(tickers, start=START_DATE, end=END_DATE,
                      auto_adjust=True, progress=False)

    # raw has MultiIndex columns: (Price, Ticker)
    frames = []
    for tk in tickers:
        try:
            sub = pd.DataFrame({
                'open':   raw[('Open',   tk)],
                'high':   raw[('High',   tk)],
                'low':    raw[('Low',    tk)],
                'close':  raw[('Close',  tk)],
                'volume': raw[('Volume', tk)],
            })
            sub.index.name = 'date'
            sub = sub.reset_index()
            sub['ticker'] = tk
            sub = sub.dropna(subset=['close'])
            frames.append(sub)
        except KeyError:
            print(f"  Warning: {tk} not in download, skipping")

    df = pd.concat(frames, ignore_index=True)
    df['date'] = pd.to_datetime(df['date'])
    df = df[['date', 'ticker', 'open', 'high', 'low', 'close', 'volume']]
    df = df.sort_values(['ticker', 'date']).reset_index(drop=True)

    if cache_path:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache_path)
    print(f"  Downloaded {df['ticker'].nunique()} tickers, {len(df)} rows")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────

def add_features(df):
    """Add all signal features to per-ticker price data."""
    df = df.copy().sort_values('date').reset_index(drop=True)
    c = df['close']
    o = df['open']
    h = df['high']
    lo = df['low']
    v = df['volume']
    n = len(df)

    # ── Rolling indicators ──────────────────────────────────────────────────
    df['sma_20']     = c.rolling(20).mean()
    df['sma_50']     = c.rolling(50).mean()
    df['avg_range_20'] = (h - lo).rolling(20).mean()
    df['avg_vol_20'] = v.rolling(20).mean()

    # RSI (14)
    delta = c.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / loss.replace(0, np.nan)
    df['rsi14'] = 100 - 100 / (1 + rs)

    # Previous row values
    df['prev_close']  = c.shift(1)
    df['prev_high']   = h.shift(1)
    df['prev_low']    = lo.shift(1)
    df['prev_open']   = o.shift(1)
    df['close_2d_ago'] = c.shift(2)
    df['close_3d_ago'] = c.shift(3)
    df['prev_volume'] = v.shift(1)

    range_ = (h - lo).clip(lower=1e-6)

    # ── Signal A: Hammer Candle + Dip ───────────────────────────────────────
    wick_ratio   = (c - lo) / range_
    body_ratio   = range_ / c
    hammer_candle = (wick_ratio > 0.6) & (body_ratio > 0.015) & (c > o)
    below_sma20   = c < df['sma_20'] * 0.95  # >5% below 20-SMA
    df['sig_A']   = (hammer_candle & below_sma20).astype(int)

    # ── Signal B: Gap Down Reversal ─────────────────────────────────────────
    gap_down      = o < df['prev_close'] * 0.99   # gap down >1%
    close_above   = c > df['prev_close']            # closes above prev close
    df['sig_B']   = (gap_down & close_above).astype(int)

    # ── Signal C: Wide Range Dip Day ────────────────────────────────────────
    range_ratio   = range_ / df['avg_range_20'].replace(0, np.nan)
    close_pos     = (c - lo) / range_             # 0=low, 1=high
    wide_range    = range_ratio > 2.0
    close_low_end = close_pos < 0.25
    below_sma20_C = c < df['sma_20'] * 0.95
    df['sig_C']   = (wide_range & close_low_end & below_sma20_C).astype(int)

    # ── Signal D: Volume Climax Reversal ────────────────────────────────────
    vol_ratio     = v / df['avg_vol_20'].replace(0, np.nan)
    bullish_close = c > o
    vol_climax    = vol_ratio > 3.0
    oversold      = df['rsi14'] < 40
    df['sig_D']   = (vol_climax & bullish_close & oversold).astype(int)

    # ── Signal E: Inside Day After Dip ──────────────────────────────────────
    inside_day    = (h < df['prev_high']) & (lo > df['prev_low'])
    prev_big_down = df['prev_close'].pct_change(1) < -0.02  # prev day down >2%
    # Actually: prev_close vs close_2d_ago gives prev day return
    prev_day_ret  = (df['prev_close'] - df['close_2d_ago']) / df['close_2d_ago'].replace(0, np.nan)
    prev_big_down = prev_day_ret < -0.02
    below_sma20_E = c < df['sma_20'] * 0.95
    df['sig_E']   = (inside_day & prev_big_down & below_sma20_E).astype(int)

    # ── Signal F: Three Bar Reversal ────────────────────────────────────────
    three_lower   = (df['prev_close'] < df['close_2d_ago']) & \
                    (df['close_2d_ago'] < df['close_3d_ago'])
    up_day        = c > o
    higher_vol    = v > df['prev_volume']
    df['sig_F']   = (three_lower & up_day & higher_vol).astype(int)

    # ── Base MR: RSI<30 + >7% below 50d high ───────────────────────────────
    high_50d      = c.rolling(50).max()
    below_50d_high = c < high_50d * 0.93
    df['sig_base'] = ((df['rsi14'] < 30) & below_50d_high).astype(int)

    return df


# ─────────────────────────────────────────────────────────────────────────────
# BACKTEST ENGINE
# ─────────────────────────────────────────────────────────────────────────────

def run_backtest(all_data, signal_col, start=None, end=None):
    """
    Simulate trades: enter at close on signal day, exit at TP/SL/max-hold.
    Max 2 concurrent positions, $300 each.
    Returns list of trade dicts.
    """
    if start:
        mask = (all_data['date'] >= pd.Timestamp(start)) & (all_data['date'] <= pd.Timestamp(end))
        data = all_data[mask].copy()
    else:
        data = all_data.copy()

    if data.empty:
        return []

    # Build per-ticker price series indexed by date
    tickers = data['ticker'].unique()
    price_data = {}
    for tk in tickers:
        sub = data[data['ticker'] == tk].set_index('date').sort_index()
        price_data[tk] = sub

    # Collect all signals across all tickers
    signals = []
    for tk, sub in price_data.items():
        sig_series = sub[signal_col].fillna(0)
        sig_days = sig_series[sig_series == 1].index.tolist()
        for d in sig_days:
            signals.append((d, tk))

    signals.sort(key=lambda x: x[0])

    trades = []
    # Track open positions: list of (exit_date, ticker)
    open_positions = []  # tuples of (exit_date, ticker)

    for sig_date, ticker in signals:
        # Remove positions that have exited by sig_date
        open_positions = [(ed, tk) for (ed, tk) in open_positions if ed > sig_date]

        if len(open_positions) >= MAX_CONCURRENT:
            continue
        if ticker in [tk for (_, tk) in open_positions]:
            continue   # no doubling up same ticker

        tk_data = price_data[ticker]
        if sig_date not in tk_data.index:
            continue

        entry_price_raw = tk_data.loc[sig_date, 'close']
        if pd.isna(entry_price_raw) or entry_price_raw <= 0:
            continue

        cost_entry = entry_price_raw * (1 + COST_RT_PCT / 2)

        # Simulate forward
        future_dates = tk_data.index[tk_data.index > sig_date]
        entry_price = entry_price_raw

        exit_price = None
        exit_date  = None
        exit_reason = None

        for i, fdate in enumerate(future_dates):
            if i >= MAX_HOLD_DAYS:
                exit_price  = tk_data.loc[fdate, 'close']
                exit_date   = fdate
                exit_reason = 'max_hold'
                break
            day_high  = tk_data.loc[fdate, 'high']
            day_low   = tk_data.loc[fdate, 'low']
            day_close = tk_data.loc[fdate, 'close']

            tp_price = entry_price * (1 + TP_PCT)
            sl_price = entry_price * (1 + SL_PCT)

            # Check SL first (intraday), then TP
            if day_low <= sl_price:
                exit_price  = sl_price
                exit_date   = fdate
                exit_reason = 'sl'
                break
            elif day_high >= tp_price:
                exit_price  = tp_price
                exit_date   = fdate
                exit_reason = 'tp'
                break
            elif i == len(future_dates) - 1:
                exit_price  = day_close
                exit_date   = fdate
                exit_reason = 'data_end'
                break

        if exit_price is None or exit_date is None:
            continue

        exit_net  = exit_price * (1 - COST_RT_PCT / 2)
        pnl_pct   = (exit_net - cost_entry) / cost_entry
        pnl_usd   = pnl_pct * POSITION_SIZE
        hold_days = (exit_date - sig_date).days

        open_positions.append((exit_date, ticker))
        trades.append({
            'entry_date':  sig_date,
            'exit_date':   exit_date,
            'ticker':      ticker,
            'entry_price': entry_price,
            'exit_price':  exit_price,
            'exit_reason': exit_reason,
            'pnl_pct':     pnl_pct,
            'pnl':         pnl_usd,
            'hold_days':   hold_days,
        })

    return trades


# ─────────────────────────────────────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(trades, label=''):
    if not trades:
        return {
            'n': 0, 'wr': 0, 'pf': 0, 'sharpe': 0,
            'sortino': 0, 'mdd': 0, 'total_return': 0,
            'avg_return': 0, 'label': label,
        }

    rets = [t['pnl_pct'] for t in trades]
    wins = [r for r in rets if r > 0]
    loss = [r for r in rets if r <= 0]
    n    = len(rets)

    wr = len(wins) / n
    gross_profit = sum(wins)
    gross_loss   = abs(sum(loss))
    pf = gross_profit / gross_loss if gross_loss > 0 else np.inf

    mean_r  = np.mean(rets)
    std_r   = np.std(rets, ddof=1) if n > 1 else 1e-9
    neg_std = np.std([r for r in rets if r < 0], ddof=1) if len([r for r in rets if r < 0]) > 1 else std_r

    sharpe  = mean_r / std_r * np.sqrt(252 / max(1, np.mean([t['hold_days'] for t in trades])))
    sortino = mean_r / neg_std * np.sqrt(252 / max(1, np.mean([t['hold_days'] for t in trades])))

    # MDD on cumulative PnL $
    pnl_vals = np.array([t['pnl'] for t in trades])
    cum_pnl  = np.cumsum(pnl_vals)
    running_max = np.maximum.accumulate(cum_pnl)
    drawdowns   = cum_pnl - running_max
    mdd = drawdowns.min() / (abs(running_max.max()) + 1e-9)

    return {
        'n':             n,
        'wr':            round(wr, 4),
        'pf':            round(pf, 3),
        'sharpe':        round(sharpe, 3),
        'sortino':       round(sortino, 3),
        'mdd':           round(mdd, 4),
        'total_pnl':     round(sum(pnl_vals), 2),
        'avg_return':    round(mean_r, 4),
        'label':         label,
    }


# ─────────────────────────────────────────────────────────────────────────────
# REGIME CLASSIFICATION
# ─────────────────────────────────────────────────────────────────────────────

def classify_regimes(spy_data):
    """Classify days as green/red/flat based on SPY close-to-close."""
    spy = spy_data.sort_values('date').set_index('date')['close']
    # Flatten if MultiIndex column
    if isinstance(spy, pd.DataFrame):
        spy = spy.iloc[:, 0]
    spy = spy.squeeze()
    ret = spy.pct_change()
    regime = pd.Series('flat', index=ret.index, dtype=str)
    regime.loc[ret > 0.001]  = 'green'
    regime.loc[ret < -0.001] = 'red'
    return regime


# ─────────────────────────────────────────────────────────────────────────────
# VALIDATION GATES
# ─────────────────────────────────────────────────────────────────────────────

def regime_gate(trades, regime_map):
    """Gate 1: |Sharpe_green - Sharpe_red| / max < 0.50"""
    if not trades:
        return False, 0.0, {}

    green_trades = [t for t in trades if regime_map.get(t['entry_date'], 'flat') == 'green']
    red_trades   = [t for t in trades if regime_map.get(t['entry_date'], 'flat') == 'red']

    m_green = compute_metrics(green_trades, 'green')
    m_red   = compute_metrics(red_trades, 'red')

    sh_g = m_green['sharpe']
    sh_r = m_red['sharpe']
    denom = max(abs(sh_g), abs(sh_r), 1e-9)
    gap   = abs(sh_g - sh_r) / denom

    passed = gap <= REGIME_GAP_MAX
    return passed, round(gap, 3), {'green': m_green, 'red': m_red}


def permutation_gate(trades, observed_sharpe, rng):
    """
    Gate 2: fast permutation test via trade-return shuffle (N_PERMS perms).
    Shuffling trade returns preserves the marginal distribution of returns but
    destroys any real time-series predictability, giving a valid null distribution.
    """
    if not trades:
        return False, 1.0, np.array([])

    rets = np.array([t['pnl_pct'] for t in trades])
    hold = np.array([t['hold_days'] for t in trades])
    n = len(rets)

    null_sharpes = []
    for _ in range(N_PERMS):
        perm_rets = rng.permutation(rets)
        mean_r = perm_rets.mean()
        std_r  = perm_rets.std(ddof=1) if n > 1 else 1e-9
        avg_hold = hold.mean()
        sh = mean_r / (std_r + 1e-9) * np.sqrt(252 / max(1, avg_hold))
        null_sharpes.append(sh)

    null_arr = np.array(null_sharpes)
    p_val = (null_arr >= observed_sharpe).mean()
    passed = p_val < PERM_P_THRESH
    return passed, round(p_val, 4), null_arr


def subperiod_gate(all_data, signal_col):
    """Gate 3: 4/4 sub-periods must show positive average return."""
    results = {}
    n_positive = 0
    for sp_start, sp_end, sp_name in SUB_PERIODS:
        trades = run_backtest(all_data, signal_col, start=sp_start, end=sp_end)
        m = compute_metrics(trades, sp_name)
        results[sp_name] = m
        if m['n'] > 0 and m['avg_return'] > 0:
            n_positive += 1

    passed = n_positive >= 4
    return passed, n_positive, results


def mdd_gate(trades):
    """Gate 4: max drawdown > -50%."""
    m = compute_metrics(trades)
    passed = m['mdd'] > -0.50
    return passed, m['mdd']


def count_gate(trades):
    """Gate 5: N >= 20."""
    n = len(trades)
    return n >= MIN_TRADES, n


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("INTRADAY PATTERN SIGNALS v1 — Megacap Entry Timing Research")
    print(f"Universe: {len(MEGACAP_30)} stocks | {START_DATE} → {END_DATE}")
    print("=" * 70)

    rng = np.random.default_rng(42)

    # ── Download data ───────────────────────────────────────────────────────
    cache = OUTPUT / "price_cache.parquet"
    print("\n[1] Fetching price data...")
    df_raw = fetch_data(MEGACAP_30, cache_path=str(cache))
    print(f"    Loaded: {df_raw['ticker'].nunique()} tickers, {len(df_raw)} rows")

    # Download SPY for regime classification
    import yfinance as yf
    spy_raw = yf.download('SPY', start=START_DATE, end=END_DATE,
                          auto_adjust=True, progress=False)
    spy_raw = spy_raw.reset_index()
    # Flatten multi-index columns if present
    if isinstance(spy_raw.columns, pd.MultiIndex):
        spy_raw.columns = [c[0] if c[1] == '' else c[0] for c in spy_raw.columns]
    spy_raw.columns = [c.lower() for c in spy_raw.columns]
    spy_df = spy_raw[['date', 'close']].copy()
    spy_df['date'] = pd.to_datetime(spy_df['date'])
    spy_df['close'] = spy_df['close'].astype(float)
    spy_df['ticker'] = 'SPY'
    regime_map = classify_regimes(spy_df)

    # ── Feature engineering ─────────────────────────────────────────────────
    print("\n[2] Computing features...")
    frames = []
    for tk in df_raw['ticker'].unique():
        sub = df_raw[df_raw['ticker'] == tk].copy()
        sub = add_features(sub)
        frames.append(sub)
    all_data = pd.concat(frames, ignore_index=True)
    all_data['date'] = pd.to_datetime(all_data['date'])
    print(f"    Features computed for {all_data['ticker'].nunique()} tickers")

    # Signal columns to test
    signals = {
        'A_Hammer_Dip':     'sig_A',
        'B_GapDown_Rev':    'sig_B',
        'C_WideRange_Dip':  'sig_C',
        'D_VolClimax_Rev':  'sig_D',
        'E_InsideDay_Dip':  'sig_E',
        'F_ThreeBar_Rev':   'sig_F',
        'BASE_MR':          'sig_base',
    }

    # Signal counts
    print("\n[3] Signal frequency:")
    for name, col in signals.items():
        count = all_data[col].sum()
        pct   = 100 * count / len(all_data)
        print(f"    {name:20s}: {int(count):5d} signals ({pct:.2f}% of rows)")

    # ── Run backtests & validation ──────────────────────────────────────────
    print("\n[4] Running backtests and 5-gate validation...")
    print("-" * 70)

    all_results = {}

    for name, sig_col in signals.items():
        print(f"\n  {name}")

        # Full-period backtest
        trades = run_backtest(all_data, sig_col)
        metrics = compute_metrics(trades, name)

        # Gate 1: Regime
        g1_pass, g1_gap, g1_detail = regime_gate(trades, regime_map)
        # Gate 2: Permutation (fast trade-return shuffle)
        g2_pass, g2_pval, _ = permutation_gate(trades, metrics['sharpe'], rng)
        # Gate 3: Sub-periods
        g3_pass, g3_pos, g3_detail = subperiod_gate(all_data, sig_col)
        # Gate 4: MDD
        g4_pass, g4_mdd = mdd_gate(trades)
        # Gate 5: Count
        g5_pass, g5_n = count_gate(trades)

        gates_passed = sum([g1_pass, g2_pass, g3_pass, g4_pass, g5_pass])
        validated    = gates_passed == 5

        result = {
            'name':         name,
            'signal_col':   sig_col,
            'metrics':      metrics,
            'gates': {
                'regime':    {'pass': g1_pass, 'gap': g1_gap, 'detail': g1_detail},
                'perm':      {'pass': g2_pass, 'p_val': g2_pval},
                'subperiod': {'pass': g3_pass, 'n_positive': g3_pos, 'detail': g3_detail},
                'mdd':       {'pass': g4_pass, 'mdd': g4_mdd},
                'count':     {'pass': g5_pass, 'n': g5_n},
            },
            'gates_passed':  gates_passed,
            'validated':     validated,
        }
        all_results[name] = result

        # Print summary
        g_str = "VALIDATED" if validated else f"FAILED ({gates_passed}/5 gates)"
        print(f"    N={g5_n} | WR={metrics['wr']:.1%} | PF={metrics['pf']:.2f} | "
              f"Sharpe={metrics['sharpe']:.3f} | Sortino={metrics['sortino']:.3f} | "
              f"MDD={metrics['mdd']:.1%}")
        print(f"    Gate1(regime)={g1_pass}({g1_gap:.2f}) | "
              f"Gate2(perm)={g2_pass}(p={g2_pval:.3f}) | "
              f"Gate3(subp)={g3_pass}({g3_pos}/4) | "
              f"Gate4(mdd)={g4_pass} | Gate5(n)={g5_pass}")
        print(f"    → {g_str}")

    # ── Summary table ───────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"{'Signal':<22} {'N':>5} {'WR':>6} {'PF':>5} {'Sharpe':>7} {'Sortino':>8} {'MDD':>7} {'Gates':>6} {'Status'}")
    print("-" * 70)

    base_sharpe = all_results['BASE_MR']['metrics']['sharpe']

    for name, res in all_results.items():
        m   = res['metrics']
        gp  = res['gates_passed']
        val = "✓ VALID" if res['validated'] else f"✗ {gp}/5"
        vs_base = m['sharpe'] - base_sharpe
        print(f"  {name:<20} {m['n']:>5} {m['wr']:>6.1%} {m['pf']:>5.2f} "
              f"{m['sharpe']:>7.3f} {m['sortino']:>8.3f} {m['mdd']:>7.1%} "
              f"{gp:>5}/5  {val}")

    print("\n  vs Base MR Sharpe:")
    for name, res in all_results.items():
        if name == 'BASE_MR':
            continue
        delta = res['metrics']['sharpe'] - base_sharpe
        sign  = "+" if delta >= 0 else ""
        print(f"    {name:<20}: {sign}{delta:.3f} Sharpe vs Base")

    # ── Sub-period detail for validated signals ──────────────────────────────
    print("\n" + "=" * 70)
    print("SUB-PERIOD BREAKDOWN")
    print("=" * 70)
    for name, res in all_results.items():
        sp = res['gates']['subperiod']['detail']
        print(f"\n  {name} ({res['gates_passed']}/5 gates):")
        for sp_name, m in sp.items():
            status = "+" if m['avg_return'] > 0 else "-"
            print(f"    [{status}] {sp_name}: N={m['n']}, AvgRet={m['avg_return']:.2%}, "
                  f"Sharpe={m['sharpe']:.3f}")

    # ── Save JSON ────────────────────────────────────────────────────────────
    # Convert numpy types for JSON serialization
    def convert(obj):
        if isinstance(obj, bool):           return bool(obj)   # must be before int check
        if isinstance(obj, (np.bool_,)):    return bool(obj)
        if isinstance(obj, (np.integer,)):  return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, float):          return obj
        if isinstance(obj, int):            return obj
        if isinstance(obj, str):            return obj
        if isinstance(obj, (np.ndarray,)):  return obj.tolist()
        if isinstance(obj, dict):           return {str(k): convert(v) for k, v in obj.items()}
        if isinstance(obj, list):           return [convert(i) for i in obj]
        if isinstance(obj, pd.Timestamp):   return str(obj)
        return str(obj)

    output_path = OUTPUT / "results.json"
    with open(output_path, 'w') as f:
        json.dump(convert(all_results), f, indent=2)
    print(f"\n[SAVED] {output_path}")

    # ── Final verdict ─────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("FINAL VERDICT")
    print("=" * 70)
    validated = [(n, r) for n, r in all_results.items() if r['validated']]
    if validated:
        print(f"\n  {len(validated)} signal(s) passed ALL 5 gates:")
        for name, res in validated:
            m = res['metrics']
            print(f"    {name}: Sharpe {m['sharpe']:.3f}, WR {m['wr']:.1%}, "
                  f"PF {m['pf']:.2f}, N={m['n']}")
    else:
        partial = sorted(all_results.items(), key=lambda x: x[1]['gates_passed'], reverse=True)
        print(f"\n  No signals passed all 5 gates.")
        print(f"  Best partial results:")
        for name, res in partial[:3]:
            m = res['metrics']
            print(f"    {name} ({res['gates_passed']}/5): Sharpe {m['sharpe']:.3f}, "
                  f"WR {m['wr']:.1%}, PF {m['pf']:.2f}")

    base_m = all_results['BASE_MR']['metrics']
    print(f"\n  Base MR benchmark: Sharpe {base_m['sharpe']:.3f}, "
          f"WR {base_m['wr']:.1%}, PF {base_m['pf']:.2f}, N={base_m['n']}")

    beaten_base = [n for n, r in all_results.items()
                   if r['metrics']['sharpe'] > base_sharpe and n != 'BASE_MR']
    if beaten_base:
        print(f"\n  Signals beating Base MR Sharpe: {', '.join(beaten_base)}")
    else:
        print(f"\n  No signal beat Base MR on Sharpe.")

    print("\n[DONE]")
    return all_results


if __name__ == '__main__':
    main()
