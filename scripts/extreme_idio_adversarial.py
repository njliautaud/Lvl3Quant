#!/usr/bin/env python3
"""
Adversarial Framework for Extreme Idiosyncratic Movers — Variants C and F
==========================================================================
5 tests per variant:
  1. Inverse Direction (calm stocks)
  2. Random Timing (100 iterations)
  3. Sub-Period Stability (4 equal sub-periods)
  4. Top-Trade Removal (remove top 5)
  5. Parameter Sensitivity (threshold x hold grid)
"""

import json
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
from pathlib import Path

warnings.filterwarnings('ignore')

# ── Universe + Config (from base script) ──────────────────────────────────
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
STARTING_CAPITAL = 645.0
IDIO_THRESHOLD = 0.05

OOT_START = pd.Timestamp('2022-01-01')
OOT_END = pd.Timestamp('2026-07-25')


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


def precompute(closes, volumes):
    ret_5d = closes.pct_change(5)
    sma_50 = closes.rolling(50).mean()
    vol_avg_20 = volumes.rolling(20).mean()
    spy_sma200 = closes['SPY'].rolling(200).mean()
    return ret_5d, sma_50, vol_avg_20, spy_sma200


# ── Core idiosyncratic signal computation ─────────────────────────────────
def compute_idio_signals(closes, ret_5d, threshold=IDIO_THRESHOLD):
    """Compute idiosyncratic relative returns for all stocks."""
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
        mask = abs_rel > threshold
        for date in rel_ret[mask].index:
            if date < OOT_START or date > OOT_END:
                continue
            rr = rel_ret.at[date]
            sr = stock_r.at[date]
            sec_r = sector_r.at[date]
            if pd.isna(rr) or pd.isna(sr):
                continue
            records.append({
                'date': date, 'ticker': ticker,
                'stock_ret_5d': sr, 'sector_ret_5d': sec_r,
                'rel_ret': rr, 'abs_rel_ret': abs(rr),
                'direction': 'UP' if rr > 0 else 'DOWN',
            })
    return pd.DataFrame(records)


def compute_calm_signals(closes, ret_5d, calm_threshold=0.01):
    """INVERSE: stocks with MINIMAL idiosyncratic move (|rel_ret| < calm_threshold)."""
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
        mask = abs_rel < calm_threshold
        for date in rel_ret[mask].index:
            if date < OOT_START or date > OOT_END:
                continue
            rr = rel_ret.at[date]
            sr = stock_r.at[date]
            if pd.isna(rr) or pd.isna(sr):
                continue
            records.append({
                'date': date, 'ticker': ticker,
                'stock_ret_5d': sr, 'sector_ret_5d': ret_5d[sector_etf].at[date],
                'rel_ret': rr, 'abs_rel_ret': abs(rr),
                'direction': 'UP' if rr > 0 else 'DOWN',
            })
    return pd.DataFrame(records)


# ── Signal generators for C and F ─────────────────────────────────────────
def generate_signals_C(closes, ret_5d, sma_50, threshold=IDIO_THRESHOLD):
    """Variant C: Magnitude + Trend Filter (stock > 50-SMA)."""
    signals = compute_idio_signals(closes, ret_5d, threshold)
    if signals.empty:
        return signals
    keep = []
    for _, sig in signals.iterrows():
        ticker, date = sig['ticker'], sig['date']
        try:
            price = closes.at[date, ticker]
            sma = sma_50.at[date, ticker]
        except Exception:
            continue
        if pd.notna(price) and pd.notna(sma) and price > sma:
            keep.append(sig)
    return pd.DataFrame(keep) if keep else pd.DataFrame()


def generate_signals_F(closes, ret_5d, threshold=IDIO_THRESHOLD):
    """Variant F: Top 3 by magnitude per scan day."""
    signals = compute_idio_signals(closes, ret_5d, threshold)
    if signals.empty:
        return signals
    signals = signals.sort_values(['date', 'abs_rel_ret'], ascending=[True, False])
    return signals.groupby('date').head(3).reset_index(drop=True)


def generate_calm_C(closes, ret_5d, sma_50, target_n):
    """Inverse of C: calm stocks above 50-SMA, sample to match target_n."""
    signals = compute_calm_signals(closes, ret_5d)
    if signals.empty:
        return signals
    keep = []
    for _, sig in signals.iterrows():
        ticker, date = sig['ticker'], sig['date']
        try:
            price = closes.at[date, ticker]
            sma = sma_50.at[date, ticker]
        except Exception:
            continue
        if pd.notna(price) and pd.notna(sma) and price > sma:
            keep.append(sig)
    df = pd.DataFrame(keep) if keep else pd.DataFrame()
    if len(df) > target_n:
        df = df.sample(n=target_n, random_state=42).sort_values('date').reset_index(drop=True)
    return df


def generate_calm_F(closes, ret_5d, target_n):
    """Inverse of F: calm stocks, bottom 3 by magnitude per day, sample to match."""
    signals = compute_calm_signals(closes, ret_5d)
    if signals.empty:
        return signals
    signals = signals.sort_values(['date', 'abs_rel_ret'], ascending=[True, True])
    filtered = signals.groupby('date').head(3).reset_index(drop=True)
    if len(filtered) > target_n:
        filtered = filtered.sample(n=target_n, random_state=42).sort_values('date').reset_index(drop=True)
    return filtered


# ── Backtester ────────────────────────────────────────────────────────────
def backtest(signals_df, hold_days, closes, spy_sma200):
    """Run backtest with position limits, regime hedging."""
    if signals_df.empty:
        return pd.DataFrame(), pd.Series(dtype=float)

    spy_close = closes['SPY']
    signals_df = signals_df.sort_values('date').reset_index(drop=True)
    signals_df = signals_df[
        (signals_df['date'] >= OOT_START) & (signals_df['date'] <= OOT_END)
    ].copy()

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
        except Exception:
            continue
        exit_idx = min(entry_idx + hold_days, len(dates) - 1)
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
            'entry_date': entry_date, 'exit_date': exit_date,
            'ticker': ticker, 'entry_price': entry_price,
            'exit_price': exit_price, 'raw_ret': raw_ret,
            'net_ret': net_ret, 'weighted_ret': weighted_ret,
            'size_mult': size_mult,
            'regime': 'bear' if regime_bear else 'bull',
            'direction': sig.get('direction', 'UNKNOWN'),
            'abs_rel_ret': sig.get('abs_rel_ret', 0),
            'hold_days': hold_days,
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
            except Exception:
                pass
    equity = STARTING_CAPITAL * (1 + daily_rets).cumprod()
    return trades_df, equity


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


# ── TEST 1: INVERSE DIRECTION ────────────────────────────────────────────
def test_inverse(variant, closes, ret_5d, sma_50, spy_sma200, orig_sharpe, orig_n):
    """Buy calm stocks instead. If Sharpe similar, signal is fake."""
    print(f"\n  TEST 1: INVERSE DIRECTION (Variant {variant})")
    if variant == 'C':
        calm_sigs = generate_calm_C(closes, ret_5d, sma_50, orig_n)
    else:
        calm_sigs = generate_calm_F(closes, ret_5d, orig_n)

    if calm_sigs.empty:
        print(f"    No calm signals found")
        return {'pass': True, 'calm_sharpe': 0.0, 'orig_sharpe': orig_sharpe,
                'reason': 'No calm signals — original wins by default'}

    trades_df, equity = backtest(calm_sigs, 10, closes, spy_sma200)
    calm_sharpe = compute_sharpe(trades_df)

    # PASS if calm Sharpe is meaningfully lower than original
    passes = calm_sharpe < orig_sharpe * 0.5
    print(f"    Calm Sharpe: {calm_sharpe:.3f} vs Original: {orig_sharpe:.3f}")
    print(f"    Calm trades: {len(trades_df)}, threshold: {orig_sharpe*0.5:.3f}")
    print(f"    {'PASS' if passes else 'FAIL'}: calm stocks {'do NOT' if passes else 'DO'} match original edge")

    return {
        'pass': bool(passes),
        'calm_sharpe': round(calm_sharpe, 3),
        'calm_n_trades': int(len(trades_df)),
        'orig_sharpe': round(orig_sharpe, 3),
        'threshold': round(orig_sharpe * 0.5, 3),
    }


# ── TEST 2: RANDOM TIMING ────────────────────────────────────────────────
def test_random_timing(variant, closes, ret_5d, sma_50, spy_sma200, orig_sharpe, orig_n):
    """Random entry dates, same stock universe, 100 iterations."""
    print(f"\n  TEST 2: RANDOM TIMING (Variant {variant})")

    stock_tickers = [t for t in TICKERS if t in closes.columns]
    oot_dates = closes.loc[OOT_START:OOT_END].index
    rng = np.random.RandomState(42)

    random_sharpes = []
    for iteration in range(100):
        # Sample ~orig_n random (date, ticker) pairs
        n_signals = orig_n
        rand_dates = rng.choice(oot_dates, size=n_signals, replace=True)
        rand_tickers = rng.choice(stock_tickers, size=n_signals, replace=True)

        records = []
        for d, t in zip(rand_dates, rand_tickers):
            records.append({
                'date': d, 'ticker': t,
                'stock_ret_5d': 0, 'sector_ret_5d': 0,
                'rel_ret': 0, 'abs_rel_ret': 0.06,
                'direction': 'UP',
            })
        rand_signals = pd.DataFrame(records)

        # For variant C, apply trend filter
        if variant == 'C':
            keep = []
            for _, sig in rand_signals.iterrows():
                ticker, date = sig['ticker'], sig['date']
                try:
                    price = closes.at[date, ticker]
                    sma = sma_50.at[date, ticker]
                    if pd.notna(price) and pd.notna(sma) and price > sma:
                        keep.append(sig)
                except Exception:
                    continue
            rand_signals = pd.DataFrame(keep) if keep else pd.DataFrame()

        # For variant F, keep top 3 per day
        if variant == 'F' and not rand_signals.empty:
            rand_signals = rand_signals.sort_values('date')
            rand_signals = rand_signals.groupby('date').head(3).reset_index(drop=True)

        if rand_signals.empty:
            random_sharpes.append(0.0)
            continue

        trades_df, equity = backtest(rand_signals, 10, closes, spy_sma200)
        random_sharpes.append(compute_sharpe(trades_df))

    mean_random = np.mean(random_sharpes)
    std_random = np.std(random_sharpes)
    pct_above_half = np.mean([s > orig_sharpe * 0.5 for s in random_sharpes]) * 100

    # PASS if mean random Sharpe < 0.5 * original
    passes = mean_random < orig_sharpe * 0.5
    print(f"    Mean random Sharpe: {mean_random:.3f} +/- {std_random:.3f}")
    print(f"    Original Sharpe: {orig_sharpe:.3f}, threshold: {orig_sharpe*0.5:.3f}")
    print(f"    % random > 50% of original: {pct_above_half:.1f}%")
    print(f"    {'PASS' if passes else 'FAIL'}: timing {'matters' if passes else 'does NOT matter'}")

    return {
        'pass': bool(passes),
        'mean_random_sharpe': round(mean_random, 3),
        'std_random_sharpe': round(std_random, 3),
        'orig_sharpe': round(orig_sharpe, 3),
        'pct_above_half_orig': round(pct_above_half, 1),
        'max_random_sharpe': round(max(random_sharpes), 3),
        'min_random_sharpe': round(min(random_sharpes), 3),
    }


# ── TEST 3: SUB-PERIOD STABILITY ─────────────────────────────────────────
def test_subperiod_stability(variant, trades_df, orig_sharpe):
    """Split OOT into 4 equal sub-periods. All must have Sharpe > -0.5."""
    print(f"\n  TEST 3: SUB-PERIOD STABILITY (Variant {variant})")

    if trades_df.empty:
        return {'pass': False, 'reason': 'No trades', 'sub_periods': {}}

    # Split into 4 equal time periods
    min_date = trades_df['entry_date'].min()
    max_date = trades_df['entry_date'].max()
    total_days = (max_date - min_date).days
    period_days = total_days / 4

    sub_results = {}
    n_positive = 0
    all_above_neg05 = True

    for i in range(4):
        start = min_date + pd.Timedelta(days=int(i * period_days))
        end = min_date + pd.Timedelta(days=int((i + 1) * period_days))
        sub = trades_df[
            (trades_df['entry_date'] >= start) & (trades_df['entry_date'] < end)
        ]
        sharpe = compute_sharpe(sub)
        n = len(sub)
        wr = (sub['weighted_ret'] > 0).mean() * 100 if n > 0 else 0

        sub_results[f'Q{i+1}'] = {
            'start': str(start.date()),
            'end': str(end.date()),
            'sharpe': round(sharpe, 3),
            'n_trades': int(n),
            'win_rate_pct': round(wr, 1),
        }

        if sharpe > 0:
            n_positive += 1
        if sharpe <= -0.5:
            all_above_neg05 = False

        print(f"    Q{i+1} ({start.date()} to {end.date()}): Sharpe={sharpe:.3f}, N={n}, WR={wr:.1f}%")

    # PASS if all > -0.5
    passes = all_above_neg05
    print(f"    Positive sub-periods: {n_positive}/4")
    print(f"    All > -0.5: {all_above_neg05}")
    print(f"    {'PASS' if passes else 'FAIL'}")

    return {
        'pass': bool(passes),
        'n_positive_subperiods': int(n_positive),
        'all_above_neg05': bool(all_above_neg05),
        'sub_periods': sub_results,
    }


# ── TEST 4: TOP-TRADE REMOVAL ────────────────────────────────────────────
def test_top_trade_removal(variant, trades_df, orig_sharpe):
    """Remove top 5 most profitable trades. Sharpe must stay > 0.5."""
    print(f"\n  TEST 4: TOP-TRADE REMOVAL (Variant {variant})")

    if trades_df.empty or len(trades_df) < 10:
        return {'pass': False, 'reason': 'Too few trades'}

    sorted_trades = trades_df.sort_values('weighted_ret', ascending=False)
    top_5 = sorted_trades.head(5)
    remaining = sorted_trades.iloc[5:].copy()

    orig = orig_sharpe
    new_sharpe = compute_sharpe(remaining)
    drop_pct = (1 - new_sharpe / max(orig, 0.001)) * 100

    print(f"    Top 5 trades removed:")
    for _, t in top_5.iterrows():
        print(f"      {t['ticker']} {t['entry_date'].date()}: {t['weighted_ret']*100:.2f}%")
    print(f"    Original Sharpe: {orig:.3f}")
    print(f"    After removal: {new_sharpe:.3f} (drop: {drop_pct:.1f}%)")

    passes = new_sharpe > 0.5
    print(f"    {'PASS' if passes else 'FAIL'}: Sharpe {'stays' if passes else 'drops'} above 0.5")

    return {
        'pass': bool(passes),
        'orig_sharpe': round(orig, 3),
        'sharpe_after_removal': round(new_sharpe, 3),
        'drop_pct': round(drop_pct, 1),
        'top_5_returns_pct': [round(r * 100, 2) for r in top_5['weighted_ret'].values],
    }


# ── TEST 5: PARAMETER SENSITIVITY ────────────────────────────────────────
def test_param_sensitivity(variant, closes, ret_5d, sma_50, spy_sma200):
    """Test threshold x hold grid. At least 60% must have Sharpe > 0.5."""
    print(f"\n  TEST 5: PARAMETER SENSITIVITY (Variant {variant})")

    thresholds = [0.03, 0.04, 0.05, 0.06, 0.07, 0.08]
    hold_periods = [5, 8, 10, 12, 15]

    results_grid = {}
    total_combos = 0
    combos_above_05 = 0

    for thresh in thresholds:
        for hold in hold_periods:
            total_combos += 1
            if variant == 'C':
                signals = generate_signals_C(closes, ret_5d, sma_50, thresh)
            else:
                signals = generate_signals_F(closes, ret_5d, thresh)

            if signals.empty:
                key = f"t{int(thresh*100)}h{hold}"
                results_grid[key] = {'sharpe': 0.0, 'n_trades': 0, 'threshold': thresh, 'hold': hold}
                continue

            trades_df, equity = backtest(signals, hold, closes, spy_sma200)
            sharpe = compute_sharpe(trades_df)
            n = len(trades_df)

            key = f"t{int(thresh*100)}h{hold}"
            results_grid[key] = {
                'sharpe': round(sharpe, 3),
                'n_trades': int(n),
                'threshold': thresh,
                'hold': hold,
            }
            if sharpe > 0.5:
                combos_above_05 += 1

    pct_above = combos_above_05 / max(total_combos, 1) * 100
    passes = pct_above >= 60

    print(f"    Grid: {len(thresholds)} thresholds x {len(hold_periods)} holds = {total_combos} combos")
    print(f"    Combos with Sharpe > 0.5: {combos_above_05}/{total_combos} ({pct_above:.1f}%)")

    # Print grid
    print(f"\n    {'Thresh':>7}", end='')
    for h in hold_periods:
        print(f"  {h}d", end='')
    print()
    for thresh in thresholds:
        print(f"    {thresh*100:5.0f}%  ", end='')
        for hold in hold_periods:
            key = f"t{int(thresh*100)}h{hold}"
            s = results_grid[key]['sharpe']
            marker = '*' if s > 0.5 else ' '
            print(f"{s:5.2f}{marker}", end='')
        print()

    print(f"\n    {'PASS' if passes else 'FAIL'}: {pct_above:.1f}% >= 60% threshold")

    return {
        'pass': bool(passes),
        'pct_above_05': round(pct_above, 1),
        'combos_above_05': int(combos_above_05),
        'total_combos': int(total_combos),
        'grid': results_grid,
    }


# ── Main ──────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("ADVERSARIAL FRAMEWORK: Extreme Idiosyncratic Movers C & F")
    print("=" * 70)

    closes, volumes = download_data()
    ret_5d, sma_50, vol_avg_20, spy_sma200 = precompute(closes, volumes)

    all_results = {}

    for variant in ['C', 'F']:
        print(f"\n{'#' * 70}")
        print(f"# VARIANT {variant}")
        print(f"{'#' * 70}")

        # Generate original signals and backtest
        if variant == 'C':
            orig_signals = generate_signals_C(closes, ret_5d, sma_50)
        else:
            orig_signals = generate_signals_F(closes, ret_5d)

        orig_trades, orig_equity = backtest(orig_signals, 10, closes, spy_sma200)
        orig_sharpe = compute_sharpe(orig_trades)
        orig_n = len(orig_trades)

        print(f"\n  BASELINE: Sharpe={orig_sharpe:.3f}, N={orig_n}")

        # Run 5 tests
        t1 = test_inverse(variant, closes, ret_5d, sma_50, spy_sma200, orig_sharpe, orig_n)
        t2 = test_random_timing(variant, closes, ret_5d, sma_50, spy_sma200, orig_sharpe, orig_n)
        t3 = test_subperiod_stability(variant, orig_trades, orig_sharpe)
        t4 = test_top_trade_removal(variant, orig_trades, orig_sharpe)
        t5 = test_param_sensitivity(variant, closes, ret_5d, sma_50, spy_sma200)

        tests = {
            '1_inverse_direction': t1,
            '2_random_timing': t2,
            '3_subperiod_stability': t3,
            '4_top_trade_removal': t4,
            '5_parameter_sensitivity': t5,
        }

        n_pass = sum(1 for t in tests.values() if t.get('pass', False))
        print(f"\n  {'=' * 50}")
        print(f"  VARIANT {variant} SCORE: {n_pass}/5")
        for name, result in tests.items():
            status = 'PASS' if result.get('pass') else 'FAIL'
            print(f"    {name}: {status}")
        print(f"  {'=' * 50}")

        all_results[variant] = {
            'baseline_sharpe': round(orig_sharpe, 3),
            'baseline_n_trades': int(orig_n),
            'tests': tests,
            'score': f'{n_pass}/5',
            'n_pass': int(n_pass),
        }

    # ── Summary ───────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("FINAL SUMMARY")
    print(f"{'=' * 70}")
    for variant in ['C', 'F']:
        r = all_results[variant]
        print(f"\n  Variant {variant}: {r['score']} (Sharpe={r['baseline_sharpe']}, N={r['baseline_n_trades']})")
        for name, test in r['tests'].items():
            status = 'PASS' if test.get('pass') else 'FAIL'
            print(f"    {name}: {status}")

    # ── Save ──────────────────────────────────────────────────────────
    output = {
        'framework': 'Adversarial 5-Test Framework',
        'strategy': 'Extreme Idiosyncratic Movers',
        'variants_tested': ['C', 'F'],
        'run_timestamp': datetime.now().isoformat(),
        'tests_description': {
            '1_inverse_direction': 'Buy calm stocks (|rel_ret| < 1%). PASS if calm Sharpe < 50% of original.',
            '2_random_timing': 'Random entry dates, 100 iterations. PASS if mean random Sharpe < 50% of original.',
            '3_subperiod_stability': 'Split OOT into 4 equal periods. PASS if all have Sharpe > -0.5.',
            '4_top_trade_removal': 'Remove top 5 most profitable trades. PASS if Sharpe stays > 0.5.',
            '5_parameter_sensitivity': 'Threshold (3-8%) x Hold (5-15d) grid. PASS if >= 60% of combos have Sharpe > 0.5.',
        },
        'results': {},
    }

    # Clean numpy types
    def clean_val(v):
        if isinstance(v, (np.integer,)):
            return int(v)
        elif isinstance(v, (np.floating,)):
            return float(v)
        elif isinstance(v, (np.bool_,)):
            return bool(v)
        elif isinstance(v, dict):
            return {k: clean_val(vv) for k, vv in v.items()}
        elif isinstance(v, list):
            return [clean_val(x) for x in v]
        return v

    for variant in ['C', 'F']:
        output['results'][variant] = clean_val(all_results[variant])

    out_path = Path('/home/jupiter/Lvl3Quant/data/extreme_idio_adversarial_results.json')
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nResults saved to {out_path}")
    print("Done.")


if __name__ == '__main__':
    main()
