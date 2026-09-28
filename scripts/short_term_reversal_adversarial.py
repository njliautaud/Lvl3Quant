#!/usr/bin/env python3
"""
Adversarial Framework: Short-Term Reversal Variant E
=====================================================
5 tests to stress-test whether Variant E's edge is real:
  1. Inverse Direction — buy risers instead of fallers
  2. Random Timing — 100 iterations with random entry dates
  3. Sub-Period Stability — 4 equal sub-periods, all must be positive Sharpe
  4. Top-Trade Removal — remove 5 best trades, recompute Sharpe
  5. Parameter Sensitivity — grid over drop threshold x hold period

Uses same universe, sector mapping, data, and cost model as the original backtest.
"""

import json
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
from pathlib import Path

warnings.filterwarnings('ignore')

# ── Universe (same as original) ──────────────────────────────────────────
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
SLIPPAGE_PCT = 0.0002
MAX_POSITIONS = 5
OOT_START = pd.Timestamp('2022-01-01')
OOT_END = pd.Timestamp('2026-07-29')


# ── Data Download ────────────────────────────────────────────────────────
def download_data():
    all_tickers = list(set(TICKERS + ['SPY'] + SECTOR_ETFS))
    print(f"Downloading data for {len(all_tickers)} tickers...")
    data = yf.download(all_tickers, start='2021-06-01', end='2026-07-29',
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
    print(f"Data: {closes.index[0].date()} to {closes.index[-1].date()}, {len(closes)} days")
    return closes, volumes


# ── Signal Generator (parameterized) ─────────────────────────────────────
def generate_signals_sector_relative(closes, spy_close, drop_threshold=-0.05, direction='drop'):
    """
    Generate sector-relative signals.
    direction='drop': buy stocks that DROP more than threshold vs sector (mean reversion)
    direction='rise': buy stocks that RISE more than |threshold| vs sector (momentum/inverse)
    """
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
                relative_move = stock_r - sector_r
                if direction == 'drop' and relative_move < drop_threshold:
                    signals.append({'date': date, 'ticker': ticker, 'ret_5d': stock_r,
                                    'sector_ret': sector_r, 'relative_drop': relative_move})
                elif direction == 'rise' and relative_move > abs(drop_threshold):
                    signals.append({'date': date, 'ticker': ticker, 'ret_5d': stock_r,
                                    'sector_ret': sector_r, 'relative_drop': relative_move})

    return pd.DataFrame(signals)


# ── Backtester (same logic as original) ──────────────────────────────────
def backtest(signals_df, hold_days, closes, spy_close):
    if signals_df.empty:
        return pd.DataFrame(), pd.Series(dtype=float)

    spy_sma200 = spy_close.rolling(200).mean()
    signals_df = signals_df.sort_values('date').reset_index(drop=True)
    signals_df = signals_df[(signals_df['date'] >= OOT_START) & (signals_df['date'] <= OOT_END)]

    if signals_df.empty:
        return pd.DataFrame(), pd.Series(dtype=float)

    trades = []
    active_positions = []
    dates = closes.index

    for _, sig in signals_df.iterrows():
        entry_date = sig['date']
        ticker = sig['ticker']
        active_positions = [(ed, t) for ed, t in active_positions if ed > entry_date]
        if len(active_positions) >= MAX_POSITIONS:
            continue
        if ticker in [t for _, t in active_positions]:
            continue
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

        try:
            spy_val = spy_close.at[entry_date]
            sma_val = spy_sma200.at[entry_date]
            regime_bear = pd.notna(sma_val) and spy_val < sma_val
        except:
            regime_bear = False

        size_mult = 0.5 if regime_bear else 1.0
        raw_ret = (exit_price / entry_price) - 1
        net_ret = raw_ret - 2 * SLIPPAGE_PCT
        weighted_ret = net_ret * size_mult

        trades.append({
            'entry_date': entry_date, 'exit_date': exit_date, 'ticker': ticker,
            'entry_price': entry_price, 'exit_price': exit_price,
            'raw_ret': raw_ret, 'net_ret': net_ret, 'weighted_ret': weighted_ret,
            'size_mult': size_mult, 'regime': 'bear' if regime_bear else 'bull',
        })
        active_positions.append((exit_date, ticker))

    trades_df = pd.DataFrame(trades)
    if trades_df.empty:
        return trades_df, pd.Series(dtype=float)

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
            except:
                pass
    equity = (1 + daily_rets).cumprod()
    return trades_df, equity


# ── Metrics ──────────────────────────────────────────────────────────────
def compute_sharpe(trades_df):
    if trades_df.empty or len(trades_df) < 2:
        return 0.0
    rets = trades_df['weighted_ret'].values
    if rets.std() == 0:
        return 0.0
    trades_per_year = max(len(trades_df) / max((trades_df['exit_date'].max() - trades_df['entry_date'].min()).days / 365.25, 0.5), 1)
    return (rets.mean() / rets.std()) * np.sqrt(trades_per_year)


def compute_max_dd(equity):
    if equity.empty:
        return -1.0
    peak = equity.expanding().max()
    dd = (equity - peak) / peak
    return dd.min()


# ══════════════════════════════════════════════════════════════════════════
# TEST 1: INVERSE DIRECTION
# ══════════════════════════════════════════════════════════════════════════
def test_inverse_direction(closes, spy_close):
    print("\n" + "=" * 60)
    print("TEST 1: INVERSE DIRECTION (buy risers instead of fallers)")
    print("=" * 60)

    orig_signals = generate_signals_sector_relative(closes, spy_close, drop_threshold=-0.05, direction='drop')
    orig_trades, orig_eq = backtest(orig_signals, 10, closes, spy_close)
    orig_sharpe = compute_sharpe(orig_trades)

    inv_signals = generate_signals_sector_relative(closes, spy_close, drop_threshold=-0.05, direction='rise')
    inv_trades, inv_eq = backtest(inv_signals, 10, closes, spy_close)
    inv_sharpe = compute_sharpe(inv_trades)

    passes = inv_sharpe < orig_sharpe * 0.5

    print(f"  Original Sharpe: {orig_sharpe:.3f} ({len(orig_trades)} trades)")
    print(f"  Inverse Sharpe:  {inv_sharpe:.3f} ({len(inv_trades)} trades)")
    print(f"  Ratio:           {inv_sharpe/max(orig_sharpe,0.001):.3f}")
    print(f"  RESULT:          {'PASS' if passes else 'FAIL'}")

    return {
        'test': 'inverse_direction',
        'original_sharpe': round(orig_sharpe, 3),
        'inverse_sharpe': round(inv_sharpe, 3),
        'original_trades': len(orig_trades),
        'inverse_trades': len(inv_trades),
        'ratio': round(inv_sharpe / max(orig_sharpe, 0.001), 3),
        'threshold': 'inverse < 50% of original',
        'pass': passes,
    }


# ══════════════════════════════════════════════════════════════════════════
# TEST 2: RANDOM TIMING
# ══════════════════════════════════════════════════════════════════════════
def test_random_timing(closes, spy_close):
    print("\n" + "=" * 60)
    print("TEST 2: RANDOM TIMING (random entry dates, 100 iterations)")
    print("=" * 60)

    orig_signals = generate_signals_sector_relative(closes, spy_close, drop_threshold=-0.05, direction='drop')
    orig_trades, _ = backtest(orig_signals, 10, closes, spy_close)
    orig_sharpe = compute_sharpe(orig_trades)
    target_n = len(orig_trades)

    oot_dates = closes.loc[OOT_START:OOT_END].index
    stock_tickers = [t for t in closes.columns if t not in ['SPY'] + SECTOR_ETFS
                     and t in SECTOR_MAP and closes[t].notna().sum() > 100]

    rng = np.random.RandomState(42)
    random_sharpes = []

    for iteration in range(100):
        random_signals = []
        n_attempts = 0
        while len(random_signals) < target_n and n_attempts < target_n * 10:
            n_attempts += 1
            rand_date = rng.choice(oot_dates[10:-15])
            rand_ticker = rng.choice(stock_tickers)
            random_signals.append({
                'date': rand_date, 'ticker': rand_ticker,
                'ret_5d': 0.0, 'sector_ret': 0.0, 'relative_drop': -0.06,
            })

        rand_df = pd.DataFrame(random_signals)
        rand_trades, _ = backtest(rand_df, 10, closes, spy_close)
        rand_sharpe = compute_sharpe(rand_trades)
        random_sharpes.append(rand_sharpe)

    random_sharpes = np.array(random_sharpes)
    mean_random = random_sharpes.mean()
    pct_above_half = (random_sharpes > orig_sharpe * 0.5).mean()

    passes = mean_random < orig_sharpe * 0.5

    print(f"  Original Sharpe:     {orig_sharpe:.3f}")
    print(f"  Mean Random Sharpe:  {mean_random:.3f}")
    print(f"  Std Random Sharpe:   {random_sharpes.std():.3f}")
    print(f"  Max Random Sharpe:   {random_sharpes.max():.3f}")
    print(f"  % Random > 50% orig: {pct_above_half*100:.1f}%")
    print(f"  Threshold:           mean random < {orig_sharpe*0.5:.3f}")
    print(f"  RESULT:              {'PASS' if passes else 'FAIL'}")

    return {
        'test': 'random_timing',
        'original_sharpe': round(orig_sharpe, 3),
        'mean_random_sharpe': round(mean_random, 3),
        'std_random_sharpe': round(random_sharpes.std(), 3),
        'max_random_sharpe': round(random_sharpes.max(), 3),
        'min_random_sharpe': round(random_sharpes.min(), 3),
        'pct_random_above_half_orig': round(pct_above_half * 100, 1),
        'threshold': f'mean random < {orig_sharpe*0.5:.3f}',
        'pass': passes,
    }


# ══════════════════════════════════════════════════════════════════════════
# TEST 3: SUB-PERIOD STABILITY
# ══════════════════════════════════════════════════════════════════════════
def test_subperiod_stability(closes, spy_close):
    print("\n" + "=" * 60)
    print("TEST 3: SUB-PERIOD STABILITY (4 equal sub-periods)")
    print("=" * 60)

    orig_signals = generate_signals_sector_relative(closes, spy_close, drop_threshold=-0.05, direction='drop')

    oot_days = (OOT_END - OOT_START).days
    period_len = oot_days // 4
    sub_periods = []
    for i in range(4):
        start = OOT_START + pd.Timedelta(days=i * period_len)
        end = OOT_START + pd.Timedelta(days=(i + 1) * period_len) if i < 3 else OOT_END
        sub_periods.append((start, end))

    sub_results = []
    all_positive = True
    any_below_neg05 = False

    for i, (sp_start, sp_end) in enumerate(sub_periods):
        mask = (orig_signals['date'] >= sp_start) & (orig_signals['date'] <= sp_end)
        sub_signals = orig_signals[mask].copy()

        if sub_signals.empty:
            sharpe_val = 0.0
            n_trades = 0
        else:
            sub_trades, sub_eq = backtest(sub_signals, 10, closes, spy_close)
            sharpe_val = compute_sharpe(sub_trades)
            n_trades = len(sub_trades)

        sub_results.append({
            'period': f"{sp_start.date()} to {sp_end.date()}",
            'sharpe': round(sharpe_val, 3),
            'n_trades': n_trades,
        })

        if sharpe_val <= 0:
            all_positive = False
        if sharpe_val < -0.5:
            any_below_neg05 = True

        print(f"  Period {i+1} ({sp_start.date()} to {sp_end.date()}): "
              f"Sharpe={sharpe_val:.3f}, N={n_trades}")

    passes = all_positive and not any_below_neg05

    print(f"  All positive: {all_positive}")
    print(f"  Any < -0.5:   {any_below_neg05}")
    print(f"  RESULT:       {'PASS' if passes else 'FAIL'}")

    return {
        'test': 'subperiod_stability',
        'sub_periods': sub_results,
        'all_positive': all_positive,
        'any_below_neg_0.5': any_below_neg05,
        'pass': passes,
    }


# ══════════════════════════════════════════════════════════════════════════
# TEST 4: TOP-TRADE REMOVAL
# ══════════════════════════════════════════════════════════════════════════
def test_top_trade_removal(closes, spy_close):
    print("\n" + "=" * 60)
    print("TEST 4: TOP-TRADE REMOVAL (remove 5 best trades)")
    print("=" * 60)

    orig_signals = generate_signals_sector_relative(closes, spy_close, drop_threshold=-0.05, direction='drop')
    orig_trades, _ = backtest(orig_signals, 10, closes, spy_close)
    orig_sharpe = compute_sharpe(orig_trades)

    if orig_trades.empty:
        print("  No trades to analyze")
        return {'test': 'top_trade_removal', 'pass': False}

    sorted_trades = orig_trades.sort_values('weighted_ret', ascending=False)
    top5 = sorted_trades.head(5)
    remaining = sorted_trades.iloc[5:].copy()

    remaining_sharpe = compute_sharpe(remaining)

    top5_total_ret = top5['weighted_ret'].sum()
    all_total_ret = orig_trades['weighted_ret'].sum()

    passes = remaining_sharpe > 0.5

    print(f"  Original Sharpe:       {orig_sharpe:.3f} ({len(orig_trades)} trades)")
    print(f"  After removing top 5:  {remaining_sharpe:.3f} ({len(remaining)} trades)")
    print(f"  Top 5 contribution:    {top5_total_ret:.4f} ({top5_total_ret/max(all_total_ret,0.0001)*100:.1f}% of total P&L)")
    print(f"  Top 5 trades:")
    for _, t in top5.iterrows():
        print(f"    {t['ticker']} {t['entry_date'].date()}: {t['weighted_ret']*100:.2f}%")
    print(f"  Threshold:             remaining Sharpe > 0.5")
    print(f"  RESULT:                {'PASS' if passes else 'FAIL'}")

    return {
        'test': 'top_trade_removal',
        'original_sharpe': round(orig_sharpe, 3),
        'remaining_sharpe': round(remaining_sharpe, 3),
        'original_trades': len(orig_trades),
        'remaining_trades': len(remaining),
        'top5_pct_of_pnl': round(top5_total_ret / max(all_total_ret, 0.0001) * 100, 1),
        'threshold': 'remaining Sharpe > 0.5',
        'pass': passes,
    }


# ══════════════════════════════════════════════════════════════════════════
# TEST 5: PARAMETER SENSITIVITY
# ══════════════════════════════════════════════════════════════════════════
def test_parameter_sensitivity(closes, spy_close):
    print("\n" + "=" * 60)
    print("TEST 5: PARAMETER SENSITIVITY (threshold x hold grid)")
    print("=" * 60)

    drop_thresholds = [-0.03, -0.04, -0.05, -0.06, -0.07, -0.08]
    hold_periods = [5, 8, 10, 12, 15]

    grid_results = []
    total_combos = len(drop_thresholds) * len(hold_periods)
    passing_combos = 0

    print(f"  Testing {total_combos} combinations...")
    print(f"  {'Threshold':>10} {'Hold':>6} {'Sharpe':>8} {'N':>6} {'Pass':>6}")
    print(f"  {'-'*40}")

    for thresh in drop_thresholds:
        signals = generate_signals_sector_relative(closes, spy_close, drop_threshold=thresh, direction='drop')
        for hold in hold_periods:
            trades, _ = backtest(signals, hold, closes, spy_close)
            sharpe = compute_sharpe(trades)
            n_trades = len(trades)
            combo_pass = sharpe > 0.5

            if combo_pass:
                passing_combos += 1

            grid_results.append({
                'threshold': thresh,
                'hold_days': hold,
                'sharpe': round(sharpe, 3),
                'n_trades': n_trades,
                'pass': combo_pass,
            })

            marker = 'OK' if combo_pass else '--'
            print(f"  {thresh*100:>9.0f}% {hold:>5}d {sharpe:>8.3f} {n_trades:>6} {marker:>6}")

    pct_passing = passing_combos / total_combos
    passes = pct_passing >= 0.60

    print(f"\n  Passing combos: {passing_combos}/{total_combos} ({pct_passing*100:.1f}%)")
    print(f"  Threshold:      >= 60%")
    print(f"  RESULT:         {'PASS' if passes else 'FAIL'}")

    return {
        'test': 'parameter_sensitivity',
        'grid': grid_results,
        'passing_combos': passing_combos,
        'total_combos': total_combos,
        'pct_passing': round(pct_passing * 100, 1),
        'threshold': '>= 60% of combos with Sharpe > 0.5',
        'pass': passes,
    }


# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════
def main():
    print("=" * 70)
    print("ADVERSARIAL FRAMEWORK: SHORT-TERM REVERSAL VARIANT E")
    print("=" * 70)

    closes, volumes = download_data()
    spy_close = closes['SPY'].copy()

    results = {}
    results['test_1_inverse'] = test_inverse_direction(closes, spy_close)
    results['test_2_random_timing'] = test_random_timing(closes, spy_close)
    results['test_3_subperiod'] = test_subperiod_stability(closes, spy_close)
    results['test_4_top_trade_removal'] = test_top_trade_removal(closes, spy_close)
    results['test_5_param_sensitivity'] = test_parameter_sensitivity(closes, spy_close)

    # ── Final Verdict ────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("FINAL VERDICT")
    print("=" * 70)

    tests = [
        ('1. Inverse Direction', results['test_1_inverse']['pass']),
        ('2. Random Timing', results['test_2_random_timing']['pass']),
        ('3. Sub-Period Stability', results['test_3_subperiod']['pass']),
        ('4. Top-Trade Removal', results['test_4_top_trade_removal']['pass']),
        ('5. Parameter Sensitivity', results['test_5_param_sensitivity']['pass']),
    ]

    pass_count = sum(1 for _, p in tests if p)

    for name, passed in tests:
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")

    print(f"\n  SCORE: {pass_count}/5")
    if pass_count == 5:
        print("  VERDICT: STRONG PASS -- strategy survives all adversarial tests")
    elif pass_count >= 4:
        print("  VERDICT: CONDITIONAL PASS -- strategy is likely real with minor concerns")
    elif pass_count >= 3:
        print("  VERDICT: WEAK -- strategy has significant vulnerabilities")
    else:
        print("  VERDICT: FAIL -- strategy is likely curve-fitted or spurious")

    # ── Save results ─────────────────────────────────────────────────────
    def clean_for_json(obj):
        if isinstance(obj, dict):
            return {k: clean_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [clean_for_json(v) for v in obj]
        elif isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, (np.bool_,)):
            return bool(obj)
        elif isinstance(obj, pd.Timestamp):
            return str(obj.date())
        else:
            return obj

    output = {
        'strategy': 'Short-Term Reversal Variant E (Sector-Relative)',
        'description': 'Buy S&P 500 stocks dropping >5% vs sector ETF over 5d, hold 10d, half-size when SPY < 200-SMA',
        'run_timestamp': datetime.now().isoformat(),
        'pass_count': pass_count,
        'total_tests': 5,
        'verdict': 'STRONG PASS' if pass_count == 5 else
                   'CONDITIONAL PASS' if pass_count >= 4 else
                   'WEAK' if pass_count >= 3 else 'FAIL',
        'tests': {},
    }

    for key, val in results.items():
        output['tests'][key] = clean_for_json(val)

    out_path = Path('/home/jupiter/Lvl3Quant/data/short_term_reversal_adversarial_results.json')
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nResults saved to {out_path}")


if __name__ == '__main__':
    main()
