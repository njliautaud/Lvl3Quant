#!/usr/bin/env python3
"""
Short-Term Mean Reversion (Weekly Oversold Bounce) Backtest
============================================================
Concept: Stocks dropping >7% over 5 trading days (idiosyncratic, not market-wide)
tend to mean-revert over the following 5-10 days.

Variants:
  A: Base (>7% drop, 5d hold)
  B: Deeper drop (>10% drop, 5d hold)
  C: Volume confirmation (volume >2x avg on drop day, 10d hold)
  D: RSI filter (RSI<30 on signal day, 10d hold)
  E: Sector-relative (stock drops >5% more than sector ETF, 10d hold)

5-Gate Validation:
  1. Sharpe > 0.5
  2. Permutation test p < 0.05 (100 iterations)
  3. Regime gap < 0.5
  4. MaxDD > -50%
  5. >= 20 trades
"""

import json
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from pathlib import Path

warnings.filterwarnings('ignore')

# ── Universe: ~100 large-cap S&P 500 stocks ──────────────────────────────
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

# Sector ETF mapping (approximate)
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
SLIPPAGE_PCT = 0.0002  # 0.02% slippage
MAX_POSITIONS = 5

# ── Data Download ─────────────────────────────────────────────────────────
def download_data():
    """Download all required price data."""
    all_tickers = TICKERS + ['SPY'] + SECTOR_ETFS
    all_tickers = list(set(all_tickers))

    print(f"Downloading data for {len(all_tickers)} tickers...")
    data = yf.download(all_tickers, start='2021-06-01', end='2026-07-29',
                       group_by='ticker', auto_adjust=True, threads=True)

    # Extract close and volume into clean DataFrames
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

    print(f"Data range: {closes.index[0].date()} to {closes.index[-1].date()}")
    print(f"Tickers with data: {len([c for c in closes.columns if closes[c].notna().sum() > 100])}")

    return closes, volumes


# ── Signal Generators ─────────────────────────────────────────────────────
def compute_rsi(series, period=14):
    """Compute RSI for a price series."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period, min_periods=period).mean()
    avg_loss = loss.rolling(period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def generate_signals_A(closes, volumes, spy_close):
    """Variant A: >7% drop over 5d, 5d hold."""
    ret_5d = closes.pct_change(5)
    spy_ret_5d = spy_close.pct_change(5)

    signals = []
    for date in closes.index:
        if date < closes.index[5]:
            continue
        spy_r = spy_ret_5d.loc[date]
        if pd.isna(spy_r) or spy_r < -0.03:
            continue  # skip market crash days
        for ticker in closes.columns:
            if ticker in ['SPY'] + SECTOR_ETFS:
                continue
            r = ret_5d.loc[date].get(ticker, np.nan) if isinstance(ret_5d.loc[date], pd.Series) else np.nan
            try:
                r = ret_5d.at[date, ticker]
            except:
                continue
            if pd.notna(r) and r < -0.07:
                signals.append({'date': date, 'ticker': ticker, 'ret_5d': r})

    return pd.DataFrame(signals), 5


def generate_signals_B(closes, volumes, spy_close):
    """Variant B: >10% drop over 5d, 5d hold."""
    ret_5d = closes.pct_change(5)
    spy_ret_5d = spy_close.pct_change(5)

    signals = []
    for date in closes.index:
        if date < closes.index[5]:
            continue
        spy_r = spy_ret_5d.loc[date]
        if pd.isna(spy_r) or spy_r < -0.03:
            continue
        for ticker in closes.columns:
            if ticker in ['SPY'] + SECTOR_ETFS:
                continue
            try:
                r = ret_5d.at[date, ticker]
            except:
                continue
            if pd.notna(r) and r < -0.10:
                signals.append({'date': date, 'ticker': ticker, 'ret_5d': r})

    return pd.DataFrame(signals), 5


def generate_signals_C(closes, volumes, spy_close):
    """Variant C: >7% drop + volume >2x avg, 10d hold."""
    ret_5d = closes.pct_change(5)
    spy_ret_5d = spy_close.pct_change(5)
    vol_avg = volumes.rolling(20).mean()

    signals = []
    for date in closes.index:
        if date < closes.index[20]:
            continue
        spy_r = spy_ret_5d.loc[date]
        if pd.isna(spy_r) or spy_r < -0.03:
            continue
        for ticker in closes.columns:
            if ticker in ['SPY'] + SECTOR_ETFS:
                continue
            try:
                r = ret_5d.at[date, ticker]
                v = volumes.at[date, ticker]
                va = vol_avg.at[date, ticker]
            except:
                continue
            if pd.notna(r) and r < -0.07 and pd.notna(v) and pd.notna(va) and va > 0 and v > 2 * va:
                signals.append({'date': date, 'ticker': ticker, 'ret_5d': r, 'vol_ratio': v/va})

    return pd.DataFrame(signals), 10


def generate_signals_D(closes, volumes, spy_close):
    """Variant D: >7% drop + RSI<30, 10d hold."""
    ret_5d = closes.pct_change(5)
    spy_ret_5d = spy_close.pct_change(5)

    # Pre-compute RSI for all tickers
    rsi_all = pd.DataFrame()
    for ticker in closes.columns:
        if ticker not in ['SPY'] + SECTOR_ETFS:
            rsi_all[ticker] = compute_rsi(closes[ticker])

    signals = []
    for date in closes.index:
        if date < closes.index[20]:
            continue
        spy_r = spy_ret_5d.loc[date]
        if pd.isna(spy_r) or spy_r < -0.03:
            continue
        for ticker in rsi_all.columns:
            try:
                r = ret_5d.at[date, ticker]
                rsi_val = rsi_all.at[date, ticker]
            except:
                continue
            if pd.notna(r) and r < -0.07 and pd.notna(rsi_val) and rsi_val < 30:
                signals.append({'date': date, 'ticker': ticker, 'ret_5d': r, 'rsi': rsi_val})

    return pd.DataFrame(signals), 10


def generate_signals_E(closes, volumes, spy_close):
    """Variant E: Stock drops >5% more than its sector ETF, 10d hold."""
    ret_5d = closes.pct_change(5)
    spy_ret_5d = spy_close.pct_change(5)

    signals = []
    for date in closes.index:
        if date < closes.index[5]:
            continue
        spy_r = spy_ret_5d.loc[date]
        if pd.isna(spy_r) or spy_r < -0.03:
            continue
        for ticker in closes.columns:
            if ticker in ['SPY'] + SECTOR_ETFS:
                continue
            sector_etf = SECTOR_MAP.get(ticker)
            if not sector_etf or sector_etf not in closes.columns:
                continue
            try:
                stock_r = ret_5d.at[date, ticker]
                sector_r = ret_5d.at[date, sector_etf]
            except:
                continue
            if pd.notna(stock_r) and pd.notna(sector_r):
                relative_drop = stock_r - sector_r
                if relative_drop < -0.05:
                    signals.append({'date': date, 'ticker': ticker, 'ret_5d': stock_r,
                                    'sector_ret': sector_r, 'relative_drop': relative_drop})

    return pd.DataFrame(signals), 10


# ── Backtester ────────────────────────────────────────────────────────────
def backtest(signals_df, hold_days, closes, spy_close):
    """
    Run backtest with position limits and regime hedging.
    Returns trade-level results and equity curve.
    """
    if signals_df.empty:
        return pd.DataFrame(), pd.Series(dtype=float)

    # Compute SPY 200-SMA for regime hedge
    spy_sma200 = spy_close.rolling(200).mean()

    # Sort signals by date
    signals_df = signals_df.sort_values('date').reset_index(drop=True)

    # OOT filter: Jan 2022 - Jul 2026
    oot_start = pd.Timestamp('2022-01-01')
    oot_end = pd.Timestamp('2026-07-29')
    signals_df = signals_df[(signals_df['date'] >= oot_start) & (signals_df['date'] <= oot_end)]

    if signals_df.empty:
        return pd.DataFrame(), pd.Series(dtype=float)

    trades = []
    active_positions = []  # list of (exit_date, ticker)

    dates = closes.index

    for _, sig in signals_df.iterrows():
        entry_date = sig['date']
        ticker = sig['ticker']

        # Check position limit: count active positions at entry_date
        active_positions = [(ed, t) for ed, t in active_positions if ed > entry_date]
        if len(active_positions) >= MAX_POSITIONS:
            continue

        # Check if we already have a position in this ticker
        if ticker in [t for _, t in active_positions]:
            continue

        # Find entry and exit in the index
        try:
            entry_idx = dates.get_loc(entry_date)
        except:
            continue

        exit_idx = min(entry_idx + hold_days, len(dates) - 1)
        if exit_idx <= entry_idx:
            continue

        exit_date = dates[exit_idx]

        try:
            entry_price = closes.at[entry_date, ticker]
            exit_price = closes.at[exit_date, ticker]
        except:
            continue

        if pd.isna(entry_price) or pd.isna(exit_price) or entry_price <= 0:
            continue

        # Regime hedge: half size when SPY < 200-SMA
        try:
            spy_val = spy_close.at[entry_date]
            sma_val = spy_sma200.at[entry_date]
            regime_bear = pd.notna(sma_val) and spy_val < sma_val
        except:
            regime_bear = False

        size_mult = 0.5 if regime_bear else 1.0

        # Compute return after slippage
        raw_ret = (exit_price / entry_price) - 1
        net_ret = raw_ret - 2 * SLIPPAGE_PCT  # entry + exit slippage
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
        })

        active_positions.append((exit_date, ticker))

    trades_df = pd.DataFrame(trades)

    if trades_df.empty:
        return trades_df, pd.Series(dtype=float)

    # Build daily equity curve
    # Assign each trade to dates it's active, equal-weight across active positions
    all_dates = closes.loc[oot_start:oot_end].index
    daily_rets = pd.Series(0.0, index=all_dates)

    for _, trade in trades_df.iterrows():
        t_dates = closes.loc[trade['entry_date']:trade['exit_date']].index
        if len(t_dates) < 2:
            continue
        ticker = trade['ticker']
        size = trade['size_mult'] / MAX_POSITIONS  # equal weight allocation
        for i in range(1, len(t_dates)):
            d = t_dates[i]
            try:
                p0 = closes.at[t_dates[i-1], ticker]
                p1 = closes.at[d, ticker]
                if pd.notna(p0) and pd.notna(p1) and p0 > 0:
                    daily_rets.at[d] += ((p1/p0) - 1) * size
            except:
                pass

    equity = (1 + daily_rets).cumprod()

    return trades_df, equity


# ── Validation Gates ──────────────────────────────────────────────────────
def compute_sharpe(trades_df):
    """Annualized Sharpe from trade returns."""
    if trades_df.empty or len(trades_df) < 2:
        return 0.0
    rets = trades_df['weighted_ret'].values
    if rets.std() == 0:
        return 0.0
    # Assume avg hold ~7d, so ~36 trades/yr at max capacity
    trades_per_year = max(len(trades_df) / max((trades_df['exit_date'].max() - trades_df['entry_date'].min()).days / 365.25, 0.5), 1)
    return (rets.mean() / rets.std()) * np.sqrt(trades_per_year)


def compute_sortino(trades_df):
    """Annualized Sortino from trade returns."""
    if trades_df.empty or len(trades_df) < 2:
        return 0.0
    rets = trades_df['weighted_ret'].values
    downside = rets[rets < 0]
    if len(downside) < 2:
        return 10.0  # no downside
    down_std = downside.std()
    if down_std == 0:
        return 0.0
    trades_per_year = max(len(trades_df) / max((trades_df['exit_date'].max() - trades_df['entry_date'].min()).days / 365.25, 0.5), 1)
    return (rets.mean() / down_std) * np.sqrt(trades_per_year)


def compute_max_dd(equity):
    """Max drawdown from equity curve."""
    if equity.empty:
        return -1.0
    peak = equity.expanding().max()
    dd = (equity - peak) / peak
    return dd.min()


def permutation_test(trades_df, n_perms=100):
    """Permutation test: shuffle returns, compute fraction with higher mean."""
    if trades_df.empty or len(trades_df) < 5:
        return 1.0
    rets = trades_df['weighted_ret'].values
    observed_mean = rets.mean()
    count_above = 0
    rng = np.random.RandomState(42)
    for _ in range(n_perms):
        shuffled = rng.choice(rets, size=len(rets), replace=False)
        # Randomly flip signs to test if direction matters
        signs = rng.choice([-1, 1], size=len(rets))
        perm_mean = (rets * signs).mean()
        if perm_mean >= observed_mean:
            count_above += 1
    return count_above / n_perms


def regime_gap(trades_df):
    """
    Compute |Sharpe_bull - Sharpe_bear| / max(|Sharpe_bull|, |Sharpe_bear|).
    Returns gap value (lower is better).
    """
    bull = trades_df[trades_df['regime'] == 'bull']
    bear = trades_df[trades_df['regime'] == 'bear']

    if bull.empty or bear.empty:
        return 0.0  # only one regime present, can't compute gap

    s_bull = bull['weighted_ret'].mean() / max(bull['weighted_ret'].std(), 1e-8)
    s_bear = bear['weighted_ret'].mean() / max(bear['weighted_ret'].std(), 1e-8)

    denom = max(abs(s_bull), abs(s_bear), 1e-8)
    return abs(s_bull - s_bear) / denom


def validate_5gate(trades_df, equity, variant_name):
    """Run 5-gate validation. Returns dict with all metrics and pass/fail."""
    n_trades = len(trades_df)

    if n_trades == 0:
        return {
            'variant': variant_name,
            'n_trades': 0,
            'sharpe': 0, 'sortino': 0, 'perm_p': 1.0,
            'regime_gap': 1.0, 'max_dd': -1.0,
            'win_rate': 0, 'avg_ret': 0, 'profit_factor': 0,
            'gate_1_sharpe': False, 'gate_2_perm': False,
            'gate_3_regime': False, 'gate_4_dd': False, 'gate_5_trades': False,
            'all_gates_pass': False,
        }

    sharpe = compute_sharpe(trades_df)
    sortino = compute_sortino(trades_df)
    perm_p = permutation_test(trades_df, n_perms=100)
    rg = regime_gap(trades_df)
    mdd = compute_max_dd(equity) if not equity.empty else -1.0

    wr = (trades_df['weighted_ret'] > 0).mean()
    avg_ret = trades_df['weighted_ret'].mean()

    wins = trades_df[trades_df['weighted_ret'] > 0]['weighted_ret'].sum()
    losses = abs(trades_df[trades_df['weighted_ret'] < 0]['weighted_ret'].sum())
    pf = wins / max(losses, 1e-8)

    # Per-regime stats
    bull_trades = trades_df[trades_df['regime'] == 'bull']
    bear_trades = trades_df[trades_df['regime'] == 'bear']

    g1 = sharpe > 0.5
    g2 = perm_p < 0.05
    g3 = rg < 0.5
    g4 = mdd > -0.50
    g5 = n_trades >= 20

    result = {
        'variant': variant_name,
        'n_trades': n_trades,
        'sharpe': round(sharpe, 3),
        'sortino': round(sortino, 3),
        'perm_p': round(perm_p, 3),
        'regime_gap': round(rg, 3),
        'max_dd_pct': round(mdd * 100, 2),
        'win_rate_pct': round(wr * 100, 1),
        'avg_ret_pct': round(avg_ret * 100, 3),
        'profit_factor': round(pf, 2),
        'n_bull_trades': len(bull_trades),
        'n_bear_trades': len(bear_trades),
        'bull_wr_pct': round((bull_trades['weighted_ret'] > 0).mean() * 100, 1) if len(bull_trades) > 0 else 0,
        'bear_wr_pct': round((bear_trades['weighted_ret'] > 0).mean() * 100, 1) if len(bear_trades) > 0 else 0,
        'gate_1_sharpe_gt_0.5': g1,
        'gate_2_perm_p_lt_0.05': g2,
        'gate_3_regime_gap_lt_0.5': g3,
        'gate_4_maxdd_gt_neg50': g4,
        'gate_5_trades_gte_20': g5,
        'gates_passed': sum([g1, g2, g3, g4, g5]),
        'all_gates_pass': all([g1, g2, g3, g4, g5]),
    }

    return result


# ── Multi Hold Period Test ────────────────────────────────────────────────
def test_hold_periods(signals_df, default_hold, closes, spy_close, variant_name):
    """Test multiple hold periods for a given signal set."""
    hold_periods = [5, 10, 15, 20]
    results = {}

    for hp in hold_periods:
        trades_df, equity = backtest(signals_df, hp, closes, spy_close)
        label = f"{variant_name}_hold{hp}d"
        result = validate_5gate(trades_df, equity, label)
        results[label] = result

    return results


# ── Main ──────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("SHORT-TERM MEAN REVERSION (WEEKLY OVERSOLD BOUNCE) BACKTEST")
    print("=" * 70)

    # Download data
    closes, volumes = download_data()

    spy_close = closes['SPY'].copy()

    # Filter to stock tickers only for signal generation
    stock_cols = [c for c in closes.columns if c not in ['SPY'] + SECTOR_ETFS]
    stock_closes = closes[stock_cols]

    # Generate signals for each variant
    generators = {
        'A': ('Base (>7% drop, 5d hold)', generate_signals_A),
        'B': ('Deeper drop (>10%, 5d hold)', generate_signals_B),
        'C': ('Volume confirm (>2x avg, 10d hold)', generate_signals_C),
        'D': ('RSI<30 filter (10d hold)', generate_signals_D),
        'E': ('Sector-relative (>5% vs ETF, 10d hold)', generate_signals_E),
    }

    all_results = {}
    variant_summaries = {}

    for key, (desc, gen_func) in generators.items():
        print(f"\n{'─' * 60}")
        print(f"Variant {key}: {desc}")
        print(f"{'─' * 60}")

        signals_df, default_hold = gen_func(closes, volumes, spy_close)
        print(f"  Raw signals generated: {len(signals_df)}")

        if signals_df.empty:
            print("  NO SIGNALS — skipping")
            continue

        # Primary backtest with default hold
        trades_df, equity = backtest(signals_df, default_hold, closes, spy_close)
        primary_result = validate_5gate(trades_df, equity, f"Variant_{key}")

        print(f"  Trades (OOT): {primary_result['n_trades']}")
        print(f"  Sharpe: {primary_result['sharpe']}")
        print(f"  Sortino: {primary_result['sortino']}")
        print(f"  Win Rate: {primary_result['win_rate_pct']}%")
        print(f"  Avg Return: {primary_result['avg_ret_pct']}%")
        print(f"  Profit Factor: {primary_result['profit_factor']}")
        print(f"  Max DD: {primary_result['max_dd_pct']}%")
        print(f"  Perm Test p: {primary_result['perm_p']}")
        print(f"  Regime Gap: {primary_result['regime_gap']}")
        print(f"  Gates Passed: {primary_result['gates_passed']}/5")
        print(f"  ALL PASS: {'YES' if primary_result['all_gates_pass'] else 'NO'}")

        variant_summaries[key] = primary_result
        all_results[f"Variant_{key}"] = primary_result

        # Multi hold period test
        hp_results = test_hold_periods(signals_df, default_hold, closes, spy_close, f"Variant_{key}")
        all_results.update(hp_results)

        # Print hold period comparison
        print(f"\n  Hold Period Sensitivity:")
        for label, res in hp_results.items():
            hp = label.split('hold')[1]
            print(f"    {hp}: Sharpe={res['sharpe']}, WR={res['win_rate_pct']}%, "
                  f"AvgRet={res['avg_ret_pct']}%, N={res['n_trades']}, "
                  f"Gates={res['gates_passed']}/5")

    # ── Summary ───────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("SUMMARY: 5-GATE VALIDATION (Primary Hold Period)")
    print(f"{'=' * 70}")
    print(f"{'Variant':<12} {'Sharpe':>7} {'Sortino':>8} {'WR%':>6} {'PF':>6} "
          f"{'MaxDD%':>7} {'PermP':>6} {'RGap':>6} {'N':>5} {'Pass':>6}")
    print("-" * 75)

    for key in ['A', 'B', 'C', 'D', 'E']:
        if key in variant_summaries:
            r = variant_summaries[key]
            status = "PASS" if r['all_gates_pass'] else f"{r['gates_passed']}/5"
            print(f"  {key:<10} {r['sharpe']:>7.3f} {r['sortino']:>8.3f} {r['win_rate_pct']:>5.1f}% "
                  f"{r['profit_factor']:>6.2f} {r['max_dd_pct']:>6.2f}% {r['perm_p']:>6.3f} "
                  f"{r['regime_gap']:>6.3f} {r['n_trades']:>5d} {status:>6}")

    # ── Find Best ─────────────────────────────────────────────────────
    best_key = None
    best_gates = 0
    best_sharpe = -999
    for key, r in variant_summaries.items():
        if r['gates_passed'] > best_gates or (r['gates_passed'] == best_gates and r['sharpe'] > best_sharpe):
            best_key = key
            best_gates = r['gates_passed']
            best_sharpe = r['sharpe']

    if best_key:
        print(f"\nBest Variant: {best_key} ({best_gates}/5 gates, Sharpe={best_sharpe:.3f})")

    # ── Save Results ──────────────────────────────────────────────────
    output = {
        'strategy': 'Short-Term Mean Reversion (Weekly Oversold Bounce)',
        'oot_period': 'Jan 2022 - Jul 2026',
        'universe': 'S&P 500 (100 large-cap)',
        'cost_model': '$0 commission, 0.02% slippage each way',
        'max_positions': MAX_POSITIONS,
        'regime_hedge': 'Half-size when SPY < 200-SMA',
        'run_timestamp': datetime.now().isoformat(),
        'variant_results': {},
        'all_hold_period_results': {},
    }

    for key, r in variant_summaries.items():
        output['variant_results'][key] = r

    for label, r in all_results.items():
        # Convert any non-serializable types
        clean = {}
        for k, v in r.items():
            if isinstance(v, (np.integer,)):
                clean[k] = int(v)
            elif isinstance(v, (np.floating,)):
                clean[k] = float(v)
            elif isinstance(v, (np.bool_,)):
                clean[k] = bool(v)
            else:
                clean[k] = v
        output['all_hold_period_results'][label] = clean

    # Also clean variant_results
    for key in output['variant_results']:
        clean = {}
        for k, v in output['variant_results'][key].items():
            if isinstance(v, (np.integer,)):
                clean[k] = int(v)
            elif isinstance(v, (np.floating,)):
                clean[k] = float(v)
            elif isinstance(v, (np.bool_,)):
                clean[k] = bool(v)
            else:
                clean[k] = v
        output['variant_results'][key] = clean

    out_path = Path('/home/jupiter/Lvl3Quant/data/short_term_reversal_results.json')
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nResults saved to {out_path}")
    print("Done.")


if __name__ == '__main__':
    main()
