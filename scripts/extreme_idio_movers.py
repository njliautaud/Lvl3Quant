#!/usr/bin/env python3
"""
Extreme Idiosyncratic Movers — Bidirectional Strategy Backtest
===============================================================
Core insight: Buying stocks with extreme idiosyncratic moves (vs sector)
works in BOTH directions (drop AND pop). The real edge is magnitude of
the idiosyncratic move, not direction.

Signal: |stock_5d_return - sector_5d_return| > threshold

Variants:
  A: Pure Idiosyncratic Magnitude (|rel_ret| > 5%, 10d hold)
  B: Magnitude + Volume Confirmation (vol > 1.5x 20d avg in past 5d)
  C: Magnitude + Trend Filter (stock > 50-SMA)
  D: 15-Day Hold (same as A but 15d hold)
  E: Adaptive Direction (DOWN=15d hold, UP=5d hold)
  F: Combined Signal Strength (top 3 by magnitude per scan day)

5-Gate Validation + Sub-Period Sharpe (2022, 2023, 2024, 2025, 2026-H1)
"""

import json
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
from pathlib import Path

warnings.filterwarnings('ignore')

# ── Universe: 100 large-cap S&P 500 stocks ──────────────────────────────
TICKERS = [
    'AAPL','MSFT','AMZN','GOOGL','META','NVDA','TSLA','BRK-B','UNH','JNJ',
    'JPM','V','PG','XOM','HD','MA','CVX','MRK','ABBV','LLY',
    'PEP','KO','COST','AVGO','TMO','MCD','WMT','CSCO','ACN','ABT',
    'DHR','CRM','NKE','ADBE','TXN','NEE','PM','UNP','RTX','HON',
    'LOW','INTC','UPS','QCOM','BA','AMGN','CAT','IBM','GE','SBUX',
    'INTU','ISRG','BLK','PLD','MDLZ','ADP','GILD','ADI','SYK','MMC',
    'DE','LMT','TJX','CB','REGN','MO','CI','SO','DUK','CL',
    'CME','ICE','PGR','SHW','ZTS','BSX','VRTX','FISV','APD','MCK',
    'EL','AON','HUM','EMR','ECL','SLB','ORLY','AIG','WM','PSA',
    'SPG','NSC','F','GM','USB','TFC','PNC','MS','GS','SCHW',
]

SECTOR_MAP = {
    'AAPL':'XLK','MSFT':'XLK','AMZN':'XLY','GOOGL':'XLC','META':'XLC',
    'NVDA':'XLK','TSLA':'XLY','BRK-B':'XLF','UNH':'XLV','JNJ':'XLV',
    'JPM':'XLF','V':'XLK','PG':'XLP','XOM':'XLE','HD':'XLY',
    'MA':'XLK','CVX':'XLE','MRK':'XLV','ABBV':'XLV','LLY':'XLV',
    'PEP':'XLP','KO':'XLP','COST':'XLP','AVGO':'XLK','TMO':'XLV',
    'MCD':'XLY','WMT':'XLP','CSCO':'XLK','ACN':'XLK','ABT':'XLV',
    'DHR':'XLV','CRM':'XLK','NKE':'XLY','ADBE':'XLK','TXN':'XLK',
    'NEE':'XLU','PM':'XLP','UNP':'XLI','RTX':'XLI','HON':'XLI',
    'LOW':'XLY','INTC':'XLK','UPS':'XLI','QCOM':'XLK','BA':'XLI',
    'AMGN':'XLV','CAT':'XLI','IBM':'XLK','GE':'XLI','SBUX':'XLY',
    'INTU':'XLK','ISRG':'XLV','BLK':'XLF','PLD':'XLRE','MDLZ':'XLP',
    'ADP':'XLK','GILD':'XLV','ADI':'XLK','SYK':'XLV','MMC':'XLF',
    'DE':'XLI','LMT':'XLI','TJX':'XLY','CB':'XLF','REGN':'XLV',
    'MO':'XLP','CI':'XLV','SO':'XLU','DUK':'XLU','CL':'XLP',
    'CME':'XLF','ICE':'XLF','PGR':'XLF','SHW':'XLB','ZTS':'XLV',
    'BSX':'XLV','VRTX':'XLV','FISV':'XLK','APD':'XLB','MCK':'XLV',
    'EL':'XLP','AON':'XLF','HUM':'XLV','EMR':'XLI','ECL':'XLB',
    'SLB':'XLE','ORLY':'XLY','AIG':'XLF','WM':'XLI','PSA':'XLRE',
    'SPG':'XLRE','NSC':'XLI','F':'XLY','GM':'XLY','USB':'XLF',
    'TFC':'XLF','PNC':'XLF','MS':'XLF','GS':'XLF','SCHW':'XLF',
}

SECTOR_ETFS = list(set(SECTOR_MAP.values()))
SLIPPAGE_PCT = 0.0002  # 0.02% each way
MAX_POSITIONS = 5
STARTING_CAPITAL = 645.0
IDIO_THRESHOLD = 0.05  # 5%

OOT_START = pd.Timestamp('2022-01-01')
OOT_END = pd.Timestamp('2026-07-25')

SUB_PERIODS = {
    '2022': (pd.Timestamp('2022-01-01'), pd.Timestamp('2022-12-31')),
    '2023': (pd.Timestamp('2023-01-01'), pd.Timestamp('2023-12-31')),
    '2024': (pd.Timestamp('2024-01-01'), pd.Timestamp('2024-12-31')),
    '2025': (pd.Timestamp('2025-01-01'), pd.Timestamp('2025-12-31')),
    '2026-H1': (pd.Timestamp('2026-01-01'), pd.Timestamp('2026-07-25')),
}


# ── Data Download ─────────────────────────────────────────────────────────
def download_data():
    all_tickers = list(set(TICKERS + ['SPY'] + SECTOR_ETFS))
    print(f"Downloading data for {len(all_tickers)} tickers...")
    data = yf.download(all_tickers, start='2021-01-01', end='2026-07-29',
                       group_by='ticker', auto_adjust=True, threads=True)

    closes = pd.DataFrame()
    volumes = pd.DataFrame()

    for t in all_tickers:
        try:
            if len(all_tickers) > 1:
                if t in data.columns.get_level_values(0):
                    closes[t] = data[t]['Close']
                    volumes[t] = data[t]['Volume']
            else:
                closes[t] = data['Close']
                volumes[t] = data['Volume']
        except Exception:
            pass

    closes = closes.dropna(how='all')
    volumes = volumes.dropna(how='all')
    print(f"Data: {closes.index[0].date()} to {closes.index[-1].date()}, {len(closes.columns)} tickers")
    return closes, volumes


# ── Pre-compute indicators ───────────────────────────────────────────────
def precompute(closes, volumes):
    """Pre-compute all indicators needed across variants."""
    ret_5d = closes.pct_change(5)
    sma_50 = closes.rolling(50).mean()
    vol_avg_20 = volumes.rolling(20).mean()
    spy_sma200 = closes['SPY'].rolling(200).mean()
    return ret_5d, sma_50, vol_avg_20, spy_sma200


# ── Signal Generators ────────────────────────────────────────────────────
def _compute_idio_signals(closes, ret_5d):
    """Core: compute idiosyncratic relative returns for all stocks on all dates."""
    stock_tickers = [t for t in TICKERS if t in closes.columns]
    records = []

    for ticker in stock_tickers:
        sector_etf = SECTOR_MAP.get(ticker)
        if not sector_etf or sector_etf not in closes.columns:
            continue
        stock_r = ret_5d[ticker]
        sector_r = ret_5d[sector_etf]
        rel_ret = stock_r - sector_r
        abs_rel = rel_ret.abs()

        # Find dates where |rel_ret| > threshold
        mask = abs_rel > IDIO_THRESHOLD
        for date in rel_ret[mask].index:
            if date < OOT_START or date > OOT_END:
                continue
            rr = rel_ret.at[date]
            sr = stock_r.at[date]
            sec_r = sector_r.at[date]
            if pd.isna(rr) or pd.isna(sr):
                continue
            records.append({
                'date': date,
                'ticker': ticker,
                'stock_ret_5d': sr,
                'sector_ret_5d': sec_r,
                'rel_ret': rr,
                'abs_rel_ret': abs(rr),
                'direction': 'UP' if rr > 0 else 'DOWN',
            })

    return pd.DataFrame(records)


def generate_signals_A(closes, volumes, ret_5d, sma_50, vol_avg_20):
    """Variant A: Pure Idiosyncratic Magnitude. |rel_ret| > 5%, 10d hold."""
    signals = _compute_idio_signals(closes, ret_5d)
    return signals, 10


def generate_signals_B(closes, volumes, ret_5d, sma_50, vol_avg_20):
    """Variant B: Magnitude + Volume Confirmation.
    Require volume > 1.5x 20-day avg on at least 1 of past 5 days."""
    signals = _compute_idio_signals(closes, ret_5d)
    if signals.empty:
        return signals, 10

    keep = []
    dates_index = closes.index
    for _, sig in signals.iterrows():
        ticker = sig['ticker']
        date = sig['date']
        try:
            idx = dates_index.get_loc(date)
        except KeyError:
            continue
        # Check past 5 days for volume spike
        start_idx = max(0, idx - 4)
        vol_confirmed = False
        for i in range(start_idx, idx + 1):
            d = dates_index[i]
            try:
                v = volumes.at[d, ticker]
                va = vol_avg_20.at[d, ticker]
                if pd.notna(v) and pd.notna(va) and va > 0 and v > 1.5 * va:
                    vol_confirmed = True
                    break
            except Exception:
                continue
        if vol_confirmed:
            keep.append(sig)

    return pd.DataFrame(keep) if keep else pd.DataFrame(), 10


def generate_signals_C(closes, volumes, ret_5d, sma_50, vol_avg_20):
    """Variant C: Magnitude + Trend Filter. Stock must be above 50-SMA."""
    signals = _compute_idio_signals(closes, ret_5d)
    if signals.empty:
        return signals, 10

    keep = []
    for _, sig in signals.iterrows():
        ticker = sig['ticker']
        date = sig['date']
        try:
            price = closes.at[date, ticker]
            sma = sma_50.at[date, ticker]
        except Exception:
            continue
        if pd.notna(price) and pd.notna(sma) and price > sma:
            keep.append(sig)

    return pd.DataFrame(keep) if keep else pd.DataFrame(), 10


def generate_signals_D(closes, volumes, ret_5d, sma_50, vol_avg_20):
    """Variant D: Same as A but 15-day hold."""
    signals = _compute_idio_signals(closes, ret_5d)
    return signals, 15


def generate_signals_E(closes, volumes, ret_5d, sma_50, vol_avg_20):
    """Variant E: Adaptive Direction.
    DOWN moves -> 15d hold (mean reversion needs time).
    UP moves -> 5d hold (momentum fades faster).
    Returns signals with per-signal hold days."""
    signals = _compute_idio_signals(closes, ret_5d)
    if signals.empty:
        return signals, -1  # -1 = adaptive
    signals['hold_days'] = signals['direction'].map({'DOWN': 15, 'UP': 5})
    return signals, -1  # adaptive


def generate_signals_F(closes, volumes, ret_5d, sma_50, vol_avg_20):
    """Variant F: Top 3 signals per scan day by magnitude."""
    signals = _compute_idio_signals(closes, ret_5d)
    if signals.empty:
        return signals, 10

    # Rank by abs_rel_ret within each date, keep top 3
    signals = signals.sort_values(['date', 'abs_rel_ret'], ascending=[True, False])
    filtered = signals.groupby('date').head(3).reset_index(drop=True)
    return filtered, 10


# ── Backtester ────────────────────────────────────────────────────────────
def backtest(signals_df, hold_days, closes, spy_sma200, adaptive=False):
    """
    Run backtest with position limits, regime hedging, and capital tracking.
    adaptive=True means hold_days comes from signals_df['hold_days'] column.
    """
    if signals_df.empty:
        return pd.DataFrame(), pd.Series(dtype=float)

    spy_close = closes['SPY']
    signals_df = signals_df.sort_values('date').reset_index(drop=True)

    # Filter to OOT
    signals_df = signals_df[
        (signals_df['date'] >= OOT_START) & (signals_df['date'] <= OOT_END)
    ].copy()

    if signals_df.empty:
        return pd.DataFrame(), pd.Series(dtype=float)

    trades = []
    active_positions = []  # (exit_date, ticker)
    dates = closes.index

    for _, sig in signals_df.iterrows():
        entry_date = sig['date']
        ticker = sig['ticker']

        # Determine hold period
        if adaptive and 'hold_days' in sig.index:
            hd = int(sig['hold_days'])
        else:
            hd = hold_days

        # Clean active positions
        active_positions = [(ed, t) for ed, t in active_positions if ed > entry_date]
        if len(active_positions) >= MAX_POSITIONS:
            continue
        if ticker in [t for _, t in active_positions]:
            continue

        try:
            entry_idx = dates.get_loc(entry_date)
        except Exception:
            continue

        exit_idx = min(entry_idx + hd, len(dates) - 1)
        if exit_idx <= entry_idx:
            continue
        exit_date = dates[exit_idx]

        try:
            entry_price = closes.at[entry_date, ticker]
            exit_price = closes.at[exit_date, ticker]
        except Exception:
            continue

        if pd.isna(entry_price) or pd.isna(exit_price) or entry_price <= 0:
            continue

        # Regime hedge
        try:
            spy_val = spy_close.at[entry_date]
            sma_val = spy_sma200.at[entry_date]
            regime_bear = pd.notna(sma_val) and spy_val < sma_val
        except Exception:
            regime_bear = False

        size_mult = 0.5 if regime_bear else 1.0

        raw_ret = (exit_price / entry_price) - 1
        net_ret = raw_ret - 2 * SLIPPAGE_PCT
        weighted_ret = net_ret * size_mult

        trades.append({
            'entry_date': entry_date,
            'exit_date': exit_date,
            'ticker': ticker,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'raw_ret': raw_ret,
            'net_ret': net_ret,
            'weighted_ret': weighted_ret,
            'size_mult': size_mult,
            'regime': 'bear' if regime_bear else 'bull',
            'direction': sig.get('direction', 'UNKNOWN'),
            'abs_rel_ret': sig.get('abs_rel_ret', 0),
            'hold_days': hd,
        })

        active_positions.append((exit_date, ticker))

    trades_df = pd.DataFrame(trades)

    if trades_df.empty:
        return trades_df, pd.Series(dtype=float)

    # Build daily equity curve
    all_dates = closes.loc[OOT_START:OOT_END].index
    daily_rets = pd.Series(0.0, index=all_dates)

    for _, trade in trades_df.iterrows():
        t_dates = closes.loc[trade['entry_date']:trade['exit_date']].index
        if len(t_dates) < 2:
            continue
        ticker = trade['ticker']
        size = trade['size_mult'] / MAX_POSITIONS
        for i in range(1, len(t_dates)):
            d = t_dates[i]
            try:
                p0 = closes.at[t_dates[i-1], ticker]
                p1 = closes.at[d, ticker]
                if pd.notna(p0) and pd.notna(p1) and p0 > 0:
                    daily_rets.at[d] += ((p1/p0) - 1) * size
            except Exception:
                pass

    equity = STARTING_CAPITAL * (1 + daily_rets).cumprod()
    return trades_df, equity


# ── Validation ───────────────────────────────────────────────────────────
def compute_sharpe(trades_df):
    if trades_df.empty or len(trades_df) < 2:
        return 0.0
    rets = trades_df['weighted_ret'].values
    if rets.std() == 0:
        return 0.0
    span_years = max(
        (trades_df['exit_date'].max() - trades_df['entry_date'].min()).days / 365.25, 0.5
    )
    trades_per_year = len(trades_df) / span_years
    return (rets.mean() / rets.std()) * np.sqrt(trades_per_year)


def compute_sortino(trades_df):
    if trades_df.empty or len(trades_df) < 2:
        return 0.0
    rets = trades_df['weighted_ret'].values
    downside = rets[rets < 0]
    if len(downside) < 2:
        return 10.0
    down_std = downside.std()
    if down_std == 0:
        return 0.0
    span_years = max(
        (trades_df['exit_date'].max() - trades_df['entry_date'].min()).days / 365.25, 0.5
    )
    trades_per_year = len(trades_df) / span_years
    return (rets.mean() / down_std) * np.sqrt(trades_per_year)


def compute_max_dd(equity):
    if equity.empty:
        return -1.0
    peak = equity.expanding().max()
    dd = (equity - peak) / peak
    return dd.min()


def permutation_test(trades_df, n_perms=100):
    if trades_df.empty or len(trades_df) < 5:
        return 1.0
    rets = trades_df['weighted_ret'].values
    observed_mean = rets.mean()
    rng = np.random.RandomState(42)
    count_above = 0
    for _ in range(n_perms):
        signs = rng.choice([-1, 1], size=len(rets))
        if (rets * signs).mean() >= observed_mean:
            count_above += 1
    return count_above / n_perms


def regime_gap(trades_df):
    bull = trades_df[trades_df['regime'] == 'bull']
    bear = trades_df[trades_df['regime'] == 'bear']
    if bull.empty or bear.empty:
        return 0.0
    s_bull = bull['weighted_ret'].mean() / max(bull['weighted_ret'].std(), 1e-8)
    s_bear = bear['weighted_ret'].mean() / max(bear['weighted_ret'].std(), 1e-8)
    denom = max(abs(s_bull), abs(s_bear), 1e-8)
    return abs(s_bull - s_bear) / denom


def sub_period_sharpe(trades_df):
    """Compute Sharpe for each sub-period."""
    results = {}
    for period_name, (start, end) in SUB_PERIODS.items():
        sub = trades_df[
            (trades_df['entry_date'] >= start) & (trades_df['entry_date'] <= end)
        ]
        if len(sub) < 3:
            results[period_name] = {'sharpe': None, 'n_trades': len(sub), 'note': 'too few trades'}
        else:
            results[period_name] = {
                'sharpe': round(compute_sharpe(sub), 3),
                'n_trades': int(len(sub)),
                'win_rate_pct': round((sub['weighted_ret'] > 0).mean() * 100, 1),
                'avg_ret_pct': round(sub['weighted_ret'].mean() * 100, 3),
            }
    return results


def validate_5gate(trades_df, equity, variant_name):
    n_trades = len(trades_df)
    if n_trades == 0:
        return {
            'variant': variant_name,
            'n_trades': 0,
            'sharpe': 0, 'sortino': 0, 'perm_p': 1.0,
            'regime_gap': 1.0, 'max_dd_pct': -100.0,
            'win_rate_pct': 0, 'avg_ret_pct': 0, 'profit_factor': 0,
            'gates_passed': 0, 'all_gates_pass': False,
            'sub_period_sharpe': {k: {'sharpe': None, 'n_trades': 0} for k in SUB_PERIODS},
            '2026_h1_ok': False,
        }

    sharpe = compute_sharpe(trades_df)
    sortino = compute_sortino(trades_df)
    perm_p = permutation_test(trades_df)
    rg = regime_gap(trades_df)
    mdd = compute_max_dd(equity) if not equity.empty else -1.0

    wr = (trades_df['weighted_ret'] > 0).mean()
    avg_ret = trades_df['weighted_ret'].mean()
    wins = trades_df[trades_df['weighted_ret'] > 0]['weighted_ret'].sum()
    losses = abs(trades_df[trades_df['weighted_ret'] < 0]['weighted_ret'].sum())
    pf = wins / max(losses, 1e-8)

    # Direction breakdown
    up_trades = trades_df[trades_df['direction'] == 'UP']
    down_trades = trades_df[trades_df['direction'] == 'DOWN']

    # Sub-period analysis
    sp = sub_period_sharpe(trades_df)
    h1_2026 = sp.get('2026-H1', {})
    h1_sharpe = h1_2026.get('sharpe', None)
    h1_ok = h1_sharpe is not None and h1_sharpe > -0.5

    # 5 gates
    g1 = sharpe > 0.5
    g2 = perm_p < 0.05
    g3 = rg < 0.5
    g4 = mdd > -0.50
    g5 = n_trades >= 20

    # Final equity
    final_equity = float(equity.iloc[-1]) if not equity.empty else STARTING_CAPITAL

    return {
        'variant': variant_name,
        'n_trades': int(n_trades),
        'sharpe': round(sharpe, 3),
        'sortino': round(sortino, 3),
        'perm_p': round(perm_p, 3),
        'regime_gap': round(rg, 3),
        'max_dd_pct': round(mdd * 100, 2),
        'win_rate_pct': round(wr * 100, 1),
        'avg_ret_pct': round(avg_ret * 100, 3),
        'profit_factor': round(pf, 2),
        'final_equity': round(final_equity, 2),
        'total_return_pct': round((final_equity / STARTING_CAPITAL - 1) * 100, 2),
        'n_up_trades': int(len(up_trades)),
        'n_down_trades': int(len(down_trades)),
        'up_wr_pct': round((up_trades['weighted_ret'] > 0).mean() * 100, 1) if len(up_trades) > 0 else 0,
        'down_wr_pct': round((down_trades['weighted_ret'] > 0).mean() * 100, 1) if len(down_trades) > 0 else 0,
        'up_avg_ret_pct': round(up_trades['weighted_ret'].mean() * 100, 3) if len(up_trades) > 0 else 0,
        'down_avg_ret_pct': round(down_trades['weighted_ret'].mean() * 100, 3) if len(down_trades) > 0 else 0,
        'gate_1_sharpe_gt_0.5': bool(g1),
        'gate_2_perm_p_lt_0.05': bool(g2),
        'gate_3_regime_gap_lt_0.5': bool(g3),
        'gate_4_maxdd_gt_neg50': bool(g4),
        'gate_5_trades_gte_20': bool(g5),
        'gates_passed': int(sum([g1, g2, g3, g4, g5])),
        'all_gates_pass': bool(all([g1, g2, g3, g4, g5])),
        'sub_period_sharpe': sp,
        '2026_h1_ok': bool(h1_ok),
    }


# ── Main ──────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("EXTREME IDIOSYNCRATIC MOVERS — BIDIRECTIONAL STRATEGY")
    print(f"Signal: |stock_5d_ret - sector_5d_ret| > {IDIO_THRESHOLD*100:.0f}%")
    print(f"OOT: {OOT_START.date()} to {OOT_END.date()}")
    print(f"Capital: ${STARTING_CAPITAL}, Max positions: {MAX_POSITIONS}")
    print("=" * 70)

    closes, volumes = download_data()
    ret_5d, sma_50, vol_avg_20, spy_sma200 = precompute(closes, volumes)

    generators = {
        'A': ('Pure Idiosyncratic Magnitude (10d hold)', generate_signals_A, False),
        'B': ('Magnitude + Volume Confirm (10d hold)', generate_signals_B, False),
        'C': ('Magnitude + Trend Filter >50-SMA (10d hold)', generate_signals_C, False),
        'D': ('15-Day Hold', generate_signals_D, False),
        'E': ('Adaptive Direction (DOWN=15d, UP=5d)', generate_signals_E, True),
        'F': ('Top 3 by Magnitude per Day (10d hold)', generate_signals_F, False),
    }

    all_results = {}

    for key, (desc, gen_func, is_adaptive) in generators.items():
        print(f"\n{'─' * 60}")
        print(f"Variant {key}: {desc}")
        print(f"{'─' * 60}")

        signals_df, default_hold = gen_func(closes, volumes, ret_5d, sma_50, vol_avg_20)
        print(f"  Signals (OOT): {len(signals_df)}")

        if signals_df.empty:
            print("  NO SIGNALS — skipping")
            all_results[key] = validate_5gate(pd.DataFrame(), pd.Series(dtype=float), f"Variant_{key}")
            continue

        # Direction breakdown
        if 'direction' in signals_df.columns:
            n_up = (signals_df['direction'] == 'UP').sum()
            n_down = (signals_df['direction'] == 'DOWN').sum()
            print(f"  UP signals: {n_up}, DOWN signals: {n_down}")

        trades_df, equity = backtest(
            signals_df, default_hold, closes, spy_sma200,
            adaptive=is_adaptive
        )

        result = validate_5gate(trades_df, equity, f"Variant_{key}")
        all_results[key] = result

        print(f"  Trades: {result['n_trades']} (UP: {result['n_up_trades']}, DOWN: {result['n_down_trades']})")
        print(f"  Sharpe: {result['sharpe']}")
        print(f"  Sortino: {result['sortino']}")
        print(f"  Win Rate: {result['win_rate_pct']}% (UP: {result['up_wr_pct']}%, DOWN: {result['down_wr_pct']}%)")
        print(f"  Avg Return: {result['avg_ret_pct']}%")
        print(f"  Profit Factor: {result['profit_factor']}")
        print(f"  Max DD: {result['max_dd_pct']}%")
        print(f"  Final Equity: ${result['final_equity']} (total return: {result['total_return_pct']}%)")
        print(f"  Perm Test p: {result['perm_p']}")
        print(f"  Regime Gap: {result['regime_gap']}")
        print(f"  Gates: {result['gates_passed']}/5 {'PASS' if result['all_gates_pass'] else 'FAIL'}")
        print(f"  2026-H1 OK: {'YES' if result['2026_h1_ok'] else 'NO'}")

        # Sub-period breakdown
        print(f"  Sub-Period Sharpe:")
        for period, data in result['sub_period_sharpe'].items():
            s = data.get('sharpe', 'N/A')
            n = data.get('n_trades', 0)
            wr = data.get('win_rate_pct', 'N/A')
            print(f"    {period}: Sharpe={s}, N={n}, WR={wr}%")

    # ── Summary ───────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    print(f"{'Var':<4} {'Sharpe':>7} {'Sortino':>8} {'WR%':>6} {'PF':>6} "
          f"{'MaxDD%':>7} {'N':>5} {'Gates':>6} {'2026H1':>8} {'Final$':>8}")
    print("-" * 75)

    for key in ['A', 'B', 'C', 'D', 'E', 'F']:
        if key in all_results:
            r = all_results[key]
            status = "PASS" if r['all_gates_pass'] else f"{r['gates_passed']}/5"
            h1 = r['sub_period_sharpe'].get('2026-H1', {}).get('sharpe', 'N/A')
            h1_str = f"{h1}" if h1 is not None else "N/A"
            print(f"  {key:<3} {r['sharpe']:>7.3f} {r['sortino']:>8.3f} {r['win_rate_pct']:>5.1f}% "
                  f"{r['profit_factor']:>6.2f} {r['max_dd_pct']:>6.1f}% {r['n_trades']:>5d} "
                  f"{status:>6} {h1_str:>8} ${r['final_equity']:>7.0f}")

    # Direction analysis
    print(f"\nDIRECTION ANALYSIS (Variant A):")
    a = all_results.get('A', {})
    if a.get('n_trades', 0) > 0:
        print(f"  UP trades: {a['n_up_trades']}, WR={a['up_wr_pct']}%, AvgRet={a['up_avg_ret_pct']}%")
        print(f"  DOWN trades: {a['n_down_trades']}, WR={a['down_wr_pct']}%, AvgRet={a['down_avg_ret_pct']}%")

    # Best variant
    best_key = max(all_results.keys(),
                   key=lambda k: (all_results[k]['gates_passed'], all_results[k]['sharpe']))
    best = all_results[best_key]
    print(f"\nBest: Variant {best_key} ({best['gates_passed']}/5 gates, Sharpe={best['sharpe']})")

    # ── Save ─────────────────────────────────────────────────────────
    output = {
        'strategy': 'Extreme Idiosyncratic Movers — Bidirectional',
        'hypothesis': 'Stocks with extreme idiosyncratic moves vs sector (either direction) tend to produce profitable follow-through',
        'signal': f'|stock_5d_return - sector_5d_return| > {IDIO_THRESHOLD*100:.0f}%',
        'oot_period': f'{OOT_START.date()} to {OOT_END.date()}',
        'universe': 'S&P 500 (100 large-cap)',
        'starting_capital': STARTING_CAPITAL,
        'cost_model': '0.02% slippage each way, no commissions',
        'max_positions': MAX_POSITIONS,
        'regime_hedge': 'Half-size when SPY < 200-SMA',
        'run_timestamp': datetime.now().isoformat(),
        'variant_results': {},
    }

    for key, r in all_results.items():
        # Clean numpy types for JSON
        clean = {}
        for k, v in r.items():
            if isinstance(v, (np.integer,)):
                clean[k] = int(v)
            elif isinstance(v, (np.floating,)):
                clean[k] = float(v)
            elif isinstance(v, (np.bool_,)):
                clean[k] = bool(v)
            elif isinstance(v, dict):
                clean_sub = {}
                for sk, sv in v.items():
                    if isinstance(sv, dict):
                        clean_sub[sk] = {
                            kk: int(vv) if isinstance(vv, np.integer)
                            else float(vv) if isinstance(vv, np.floating)
                            else bool(vv) if isinstance(vv, np.bool_)
                            else vv
                            for kk, vv in sv.items()
                        }
                    else:
                        clean_sub[sk] = sv
                clean[k] = clean_sub
            else:
                clean[k] = v
        output['variant_results'][key] = clean

    out_path = Path('/home/jupiter/Lvl3Quant/data/extreme_idio_movers_results.json')
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nResults saved to {out_path}")
    print("Done.")


if __name__ == '__main__':
    main()
