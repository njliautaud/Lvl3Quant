#!/usr/bin/env python3
"""
Combined Timing + Stock-Pick Backtest (Vectorized)
====================================================
Signal Aggregator A (market timing) x Extreme Idiosyncratic Movers C (stock selection)
6 variants: A-F, plus SPY/QQQ benchmarks.

OOT: 2022-01-01 to 2026-07-28 | Starting capital: $645 | Slippage: 0.02%
"""

import json
import warnings
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

try:
    import yfinance as yf
except ImportError:
    print("ERROR: yfinance not installed. pip install yfinance")
    sys.exit(1)

# ── Universe (from extreme_idio_paper.py) ──
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

SECTOR_ETFS = sorted(set(SECTOR_MAP.values()))
VOLUME_SECTOR_ETFS = ['XLK', 'XLC', 'XLY', 'XLE', 'XLF', 'XLV']

# ── Parameters ──
START_DATE = '2021-01-01'
OOT_START = '2022-01-01'
OOT_END = '2026-07-28'
STARTING_CAPITAL = 645.0
SLIPPAGE_BPS = 0.0002
POS_SIZE_FULL = 130
POS_SIZE_HALF = 65
MAX_POSITIONS = 5
HOLD_DAYS = 10
MOVE_THRESHOLD = 0.05
LOOKBACK = 5
SMA_50_PERIOD = 50
SMA_200_PERIOD = 200
PERM_ITERATIONS = 500


def download_all_data():
    """Download all needed price/volume data."""
    all_syms = sorted(set(TICKERS + SECTOR_ETFS + ['SPY', 'QQQ', 'RSP']))
    # Download VIX separately (different ticker format)
    print(f"Downloading {len(all_syms)} equity symbols...")
    raw = yf.download(all_syms, start=START_DATE, end=OOT_END,
                       auto_adjust=True, progress=False)

    if isinstance(raw.columns, pd.MultiIndex):
        close = raw['Close'].copy()
        volume = raw['Volume'].copy()
    else:
        close = raw.copy()
        volume = raw.copy()

    # Download VIX
    print("Downloading VIX...")
    vix_raw = yf.download('^VIX', start=START_DATE, end=OOT_END,
                            auto_adjust=True, progress=False)
    if not vix_raw.empty:
        if isinstance(vix_raw.columns, pd.MultiIndex):
            close['VIX'] = vix_raw['Close']['^VIX']
        else:
            close['VIX'] = vix_raw['Close']

    # Remove tickers that failed to download
    valid_tickers = [t for t in TICKERS if t in close.columns]
    print(f"Got {len(close.columns)} symbols, {len(close)} trading days, {len(valid_tickers)}/100 stock tickers")
    return close, volume, valid_tickers


def precompute_all_signals(close, volume, valid_tickers, oot_dates):
    """Vectorized precompute of ALL signals for ALL dates."""
    t0 = time.time()

    # ── Signal Aggregator (vectorized) ──
    spy = close['SPY']
    spy_sma200 = spy.rolling(SMA_200_PERIOD).mean()

    vix = close.get('VIX', pd.Series(15.0, index=close.index))
    qqq = close['QQQ']
    rsp = close.get('RSP')

    # Signal 1: SPY > 200-SMA
    sig1 = (spy > spy_sma200).astype(int)

    # Signal 2: VIX calm
    vix_lt20 = (vix < 20)
    vix_10d_max = vix.rolling(10).max()
    vix_5d_ago = vix.shift(5)
    vix_declining_from_spike = (vix_10d_max > 25) & (vix < vix_5d_ago)
    sig2 = (vix_lt20 | vix_declining_from_spike).astype(int)

    # Signal 3: QQQ 20d return > 0
    qqq_ret20 = qqq / qqq.shift(20) - 1
    sig3 = (qqq_ret20 > 0).astype(int)

    # Signal 4: Any sector ETF with 5+ consecutive days above 20d avg volume
    sig4 = pd.Series(0, index=close.index)
    for etf in VOLUME_SECTOR_ETFS:
        if etf not in volume.columns:
            continue
        v = volume[etf]
        v_avg20 = v.rolling(20).mean()
        above_avg = (v > v_avg20).astype(int)
        # 5 consecutive days above
        consec5 = (above_avg.rolling(5).sum() >= 5).astype(int)
        sig4 = sig4 | consec5
    sig4 = sig4.astype(int)

    # Signal 5: Breadth
    sig5 = pd.Series(0, index=close.index)
    if rsp is not None:
        spy_ret20 = spy / spy.shift(20) - 1
        rsp_ret20 = rsp / rsp.shift(20) - 1
        sig5 = ((spy_ret20 > 0) & (rsp_ret20 > 0.5 * spy_ret20)).astype(int)

    # Total score
    total_score = sig1 + sig2 + sig3 + sig4 + sig5
    total_score = total_score.fillna(0).astype(int)

    # Build score dict for OOT dates
    scores = {}
    for dt in oot_dates:
        if dt in total_score.index:
            scores[dt] = int(total_score.loc[dt])
        else:
            scores[dt] = 0

    print(f"  Signal aggregator computed in {time.time()-t0:.1f}s")

    # ── Idiosyncratic signals (vectorized) ──
    t1 = time.time()

    # Precompute 5d returns and 50-SMA for all tickers and sector ETFs
    stock_ret5 = {}
    stock_sma50 = {}
    sector_ret5 = {}
    sector_sma50_above = {}

    for etf in SECTOR_ETFS:
        if etf in close.columns:
            sector_ret5[etf] = close[etf] / close[etf].shift(LOOKBACK) - 1
            sector_sma50_above[etf] = (close[etf] > close[etf].rolling(SMA_50_PERIOD).mean())

    for ticker in valid_tickers:
        if ticker in close.columns:
            stock_ret5[ticker] = close[ticker] / close[ticker].shift(LOOKBACK) - 1
            stock_sma50[ticker] = close[ticker].rolling(SMA_50_PERIOD).mean()

    # For each OOT date, find idio signals
    # Store as dict: date -> list of (ticker, price, abs_rel_ret, sector_etf)
    idio_signals = {}
    for dt in oot_dates:
        if dt not in close.index:
            idio_signals[dt] = []
            continue

        sigs = []
        for ticker in valid_tickers:
            if ticker not in stock_ret5:
                continue
            sector_etf = SECTOR_MAP.get(ticker)
            if not sector_etf or sector_etf not in sector_ret5:
                continue

            sr = stock_ret5[ticker].get(dt, np.nan)
            er = sector_ret5[sector_etf].get(dt, np.nan)
            sma50v = stock_sma50[ticker].get(dt, np.nan)
            price = close[ticker].get(dt, np.nan)

            if pd.isna(sr) or pd.isna(er) or pd.isna(sma50v) or pd.isna(price):
                continue

            if price <= sma50v:
                continue  # trend filter

            rel_ret = sr - er
            if abs(rel_ret) > MOVE_THRESHOLD:
                sigs.append((ticker, float(price), float(abs(rel_ret)), sector_etf))

        # Sort by magnitude descending
        sigs.sort(key=lambda x: -x[2])
        idio_signals[dt] = sigs

    print(f"  Idio signals computed in {time.time()-t1:.1f}s")

    # ── Bull/bear regime per date ──
    bull_regime = {}
    for dt in oot_dates:
        if dt in spy.index and dt in spy_sma200.index:
            bull_regime[dt] = bool(spy.loc[dt] > spy_sma200.loc[dt])
        else:
            bull_regime[dt] = True

    return scores, idio_signals, bull_regime, sector_sma50_above


def run_variant(close, oot_dates, scores, idio_signals, bull_regime, sector_sma50_above, variant_name):
    """Run a single variant backtest using precomputed signals."""
    capital = STARTING_CAPITAL
    positions = []  # (ticker, entry_price, entry_day_idx, shares)
    trades = []
    equity_curve = []
    daily_returns = []

    for i, dt in enumerate(oot_dates):
        bull = bull_regime.get(dt, True)
        score = scores.get(dt, 0)

        # ── Close expired positions ──
        still_open = []
        for (tk, ep, ei, sh) in positions:
            days_held = i - ei
            if days_held >= HOLD_DAYS:
                exit_p = close[tk].get(dt, ep) if tk in close.columns else ep
                if pd.isna(exit_p):
                    exit_p = ep
                exit_p *= (1 - SLIPPAGE_BPS)
                pnl = (exit_p - ep) * sh
                capital += sh * exit_p
                trades.append({
                    'pnl': pnl,
                    'regime': 'bull' if bull else 'bear',
                })
            else:
                still_open.append((tk, ep, ei, sh))
        positions = still_open

        # ── New entries ──
        open_slots = MAX_POSITIONS - len(positions)
        if open_slots <= 0:
            pass
        else:
            held = {p[0] for p in positions}
            sigs = idio_signals.get(dt, [])

            entries_to_make = []

            if variant_name == 'A':
                if score >= 3:
                    for (tk, pr, ar, se) in sigs:
                        if tk not in held and len(entries_to_make) < open_slots:
                            size = POS_SIZE_FULL if bull else POS_SIZE_HALF
                            entries_to_make.append((tk, pr, size))
                            held.add(tk)

            elif variant_name == 'B':
                if score >= 3:
                    for (tk, pr, ar, se) in sigs:
                        if tk not in held and len(entries_to_make) < open_slots:
                            if score >= 4:
                                size = POS_SIZE_FULL if bull else POS_SIZE_HALF
                            else:
                                size = POS_SIZE_HALF if bull else max(1, POS_SIZE_HALF // 2)
                            entries_to_make.append((tk, pr, size))
                            held.add(tk)

            elif variant_name == 'C':
                for (tk, pr, ar, se) in sigs:
                    if tk not in held and len(entries_to_make) < open_slots:
                        base = POS_SIZE_FULL if bull else POS_SIZE_HALF
                        if score >= 4:
                            size = base * 2
                        elif score <= 2:
                            size = max(1, base // 2)
                        else:
                            size = base
                        entries_to_make.append((tk, pr, size))
                        held.add(tk)

            elif variant_name == 'D':
                if score >= 3:
                    if sigs:
                        for (tk, pr, ar, se) in sigs:
                            if tk not in held and len(entries_to_make) < open_slots:
                                size = POS_SIZE_FULL if bull else POS_SIZE_HALF
                                entries_to_make.append((tk, pr, size))
                                held.add(tk)
                    elif 'QQQ' not in held and open_slots > 0:
                        qqq_p = close['QQQ'].get(dt, np.nan)
                        if not pd.isna(qqq_p):
                            size = POS_SIZE_FULL if bull else POS_SIZE_HALF
                            entries_to_make.append(('QQQ', float(qqq_p), size))

            elif variant_name == 'E':
                if score >= 4:
                    for (tk, pr, ar, se) in sigs:
                        if tk not in held and len(entries_to_make) < open_slots:
                            size = POS_SIZE_FULL if bull else POS_SIZE_HALF
                            entries_to_make.append((tk, pr, size))
                            held.add(tk)

            elif variant_name == 'F':
                if score >= 3:
                    # Only sectors above 50-SMA
                    for (tk, pr, ar, se) in sigs:
                        if tk not in held and len(entries_to_make) < open_slots:
                            if se in sector_sma50_above and dt in sector_sma50_above[se].index:
                                if sector_sma50_above[se].loc[dt]:
                                    entries_to_make.append((tk, pr, POS_SIZE_FULL if bull else POS_SIZE_HALF))
                                    held.add(tk)

            # Execute entries
            for (tk, pr, size) in entries_to_make:
                entry_p = pr * (1 + SLIPPAGE_BPS)
                shares = max(1, int(size / entry_p))
                cost = shares * entry_p
                if cost > capital:
                    continue
                capital -= cost
                positions.append((tk, entry_p, i, shares))

        # ── NAV ──
        pos_val = 0
        for (tk, ep, ei, sh) in positions:
            p = close[tk].get(dt, ep) if tk in close.columns else ep
            if pd.isna(p):
                p = ep
            pos_val += p * sh
        nav = capital + pos_val
        equity_curve.append(nav)

        if len(equity_curve) >= 2:
            daily_returns.append(equity_curve[-1] / equity_curve[-2] - 1)
        else:
            daily_returns.append(0.0)

    # Close remaining
    last_dt = oot_dates[-1]
    for (tk, ep, ei, sh) in positions:
        exit_p = close[tk].get(last_dt, ep) if tk in close.columns else ep
        if pd.isna(exit_p):
            exit_p = ep
        exit_p *= (1 - SLIPPAGE_BPS)
        pnl = (exit_p - ep) * sh
        capital += sh * exit_p
        trades.append({'pnl': pnl, 'regime': 'n/a'})

    return trades, np.array(daily_returns), np.array(equity_curve)


def compute_metrics(trades, daily_returns, equity_curve, close, oot_dates):
    """Compute all required metrics."""
    dr = daily_returns
    n_trades = len(trades)
    wins = sum(1 for t in trades if t['pnl'] > 0)
    wr = (wins / n_trades * 100) if n_trades else 0
    gross_win = sum(t['pnl'] for t in trades if t['pnl'] > 0)
    gross_loss = abs(sum(t['pnl'] for t in trades if t['pnl'] < 0))
    pf = (gross_win / gross_loss) if gross_loss > 0 else float('inf')

    sharpe = (np.mean(dr) / np.std(dr) * np.sqrt(252)) if len(dr) > 1 and np.std(dr) > 0 else 0.0
    downside = dr[dr < 0]
    sortino = (np.mean(dr) / np.std(downside) * np.sqrt(252)) if len(downside) > 1 and np.std(downside) > 0 else sharpe

    peak = np.maximum.accumulate(equity_curve)
    dd = (equity_curve - peak) / peak
    mdd = float(np.min(dd)) if len(dd) > 0 else 0.0

    # Bull/Bear split
    spy = close['SPY']
    spy_sma200 = spy.rolling(SMA_200_PERIOD).mean()
    bull_rets, bear_rets = [], []
    for idx, dt in enumerate(oot_dates):
        if idx >= len(dr):
            break
        if dt in spy.index and dt in spy_sma200.index:
            if not pd.isna(spy_sma200.loc[dt]):
                if spy.loc[dt] > spy_sma200.loc[dt]:
                    bull_rets.append(dr[idx])
                else:
                    bear_rets.append(dr[idx])

    bull_rets = np.array(bull_rets)
    bear_rets = np.array(bear_rets)
    bull_sharpe = (np.mean(bull_rets) / np.std(bull_rets) * np.sqrt(252)) if len(bull_rets) > 10 and np.std(bull_rets) > 0 else 0.0
    bear_sharpe = (np.mean(bear_rets) / np.std(bear_rets) * np.sqrt(252)) if len(bear_rets) > 10 and np.std(bear_rets) > 0 else 0.0
    regime_gap = abs(bull_sharpe - bear_sharpe) / max(abs(bull_sharpe), abs(bear_sharpe), 0.001)

    return {
        'sharpe': round(float(sharpe), 3),
        'sortino': round(float(sortino), 3),
        'pf': round(float(pf), 3),
        'wr': round(float(wr), 1),
        'trades': n_trades,
        'mdd': round(float(mdd * 100), 1),
        'bull_sharpe': round(float(bull_sharpe), 3),
        'bear_sharpe': round(float(bear_sharpe), 3),
        'regime_gap': round(float(regime_gap), 3),
        'final_nav': round(float(equity_curve[-1]), 2) if len(equity_curve) else STARTING_CAPITAL,
        'total_return_pct': round(float((equity_curve[-1] / STARTING_CAPITAL - 1) * 100), 1) if len(equity_curve) else 0,
    }


def permutation_test(close, oot_dates, idio_signals, bull_regime, sector_sma50_above,
                     real_scores, real_sharpe, variant_name, n_iter=PERM_ITERATIONS):
    """Shuffle signal aggregator scores across dates, measure significance."""
    print(f"  Permutation test ({n_iter} iters)...")
    score_values = np.array([real_scores[dt] for dt in oot_dates])
    better = 0

    for it in range(n_iter):
        shuffled = np.random.permutation(score_values)
        shuffled_scores = dict(zip(oot_dates, shuffled.astype(int)))

        _, dr, ec = run_variant(close, oot_dates, shuffled_scores, idio_signals,
                                bull_regime, sector_sma50_above, variant_name)
        if len(dr) > 1 and np.std(dr) > 0:
            perm_sharpe = np.mean(dr) / np.std(dr) * np.sqrt(252)
        else:
            perm_sharpe = 0.0

        if perm_sharpe >= real_sharpe:
            better += 1

        if (it + 1) % 100 == 0:
            print(f"    {it+1}/{n_iter} (p={better/(it+1):.3f})")

    return round(better / n_iter, 4)


def buy_and_hold_benchmark(close, oot_dates, ticker):
    """Simple buy-and-hold benchmark."""
    prices = close[ticker].reindex(oot_dates).ffill()
    start_p = prices.iloc[0]
    ec = (prices / start_p * STARTING_CAPITAL).values
    dr = np.diff(ec) / ec[:-1]
    dr = np.concatenate([[0], dr])

    sharpe = (np.mean(dr) / np.std(dr) * np.sqrt(252)) if np.std(dr) > 0 else 0.0
    down = dr[dr < 0]
    sortino = (np.mean(dr) / np.std(down) * np.sqrt(252)) if len(down) > 1 and np.std(down) > 0 else sharpe
    peak = np.maximum.accumulate(ec)
    dd = (ec - peak) / peak
    mdd = float(np.min(dd))

    return {
        'sharpe': round(float(sharpe), 3),
        'sortino': round(float(sortino), 3),
        'mdd': round(float(mdd * 100), 1),
        'final_nav': round(float(ec[-1]), 2),
        'total_return_pct': round(float((ec[-1] / STARTING_CAPITAL - 1) * 100), 1),
    }


def validate_5gate(metrics, perm_p):
    gates = {
        'sharpe_gt_05': metrics['sharpe'] > 0.5,
        'perm_p_lt_005': perm_p < 0.05,
        'regime_gap_lt_05': metrics['regime_gap'] < 0.5,
        'mdd_gt_neg50': metrics['mdd'] > -50,
        'trades_gte_20': metrics['trades'] >= 20,
    }
    return gates, sum(gates.values())


def main():
    t_start = time.time()
    print("=" * 70)
    print("COMBINED TIMING + STOCK-PICK BACKTEST (Vectorized)")
    print(f"OOT: {OOT_START} to {OOT_END} | Capital: ${STARTING_CAPITAL}")
    print("=" * 70)

    close, volume, valid_tickers = download_all_data()

    # OOT dates
    oot_mask = (close.index >= OOT_START) & (close.index <= OOT_END)
    oot_dates = close.index[oot_mask].tolist()
    print(f"OOT: {len(oot_dates)} trading days ({oot_dates[0].date()} to {oot_dates[-1].date()})")

    # Precompute everything
    print("\nPrecomputing signals...")
    scores, idio_signals, bull_regime, sector_sma50_above = precompute_all_signals(
        close, volume, valid_tickers, oot_dates
    )

    score_vals = [scores[dt] for dt in oot_dates]
    score_s = pd.Series(score_vals)
    print(f"  Score distribution: {dict(score_s.value_counts().sort_index())}")
    print(f"  Risk-on (>=3): {(score_s >= 3).mean()*100:.1f}%")

    # Count total idio signal days
    sig_days = sum(1 for dt in oot_dates if len(idio_signals.get(dt, [])) > 0)
    total_sigs = sum(len(idio_signals.get(dt, [])) for dt in oot_dates)
    print(f"  Idio signal days: {sig_days}/{len(oot_dates)}, total signals: {total_sigs}")

    # Run variants
    variants = ['A', 'B', 'C', 'D', 'E', 'F']
    results = {}

    for v in variants:
        print(f"\n{'='*50}")
        print(f"Variant {v}...")
        trades, dr, ec = run_variant(close, oot_dates, scores, idio_signals,
                                      bull_regime, sector_sma50_above, v)
        metrics = compute_metrics(trades, dr, ec, close, oot_dates)
        print(f"  Sharpe={metrics['sharpe']}, Sortino={metrics['sortino']}, "
              f"PF={metrics['pf']}, WR={metrics['wr']}%, "
              f"Trades={metrics['trades']}, MDD={metrics['mdd']}%")
        print(f"  Bull={metrics['bull_sharpe']}, Bear={metrics['bear_sharpe']}, "
              f"Gap={metrics['regime_gap']}, Return={metrics['total_return_pct']}%")

        perm_p = permutation_test(close, oot_dates, idio_signals, bull_regime,
                                   sector_sma50_above, scores, metrics['sharpe'], v)
        metrics['perm_p'] = perm_p

        gates, passed = validate_5gate(metrics, perm_p)
        metrics['gates'] = gates
        metrics['gates_passed'] = f"{passed}/5"
        print(f"  Perm p={perm_p}, Gates={passed}/5: {gates}")

        results[f'variant_{v}'] = metrics

    # Benchmarks
    print(f"\n{'='*50}")
    print("Benchmarks...")
    for bm in ['SPY', 'QQQ']:
        bm_res = buy_and_hold_benchmark(close, oot_dates, bm)
        print(f"  {bm}: Sharpe={bm_res['sharpe']}, MDD={bm_res['mdd']}%, Return={bm_res['total_return_pct']}%")
        results[f'benchmark_{bm}'] = bm_res

    # Save
    out_path = Path('/home/jupiter/Lvl3Quant/data/combined_timing_stockpick_results.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved to {out_path}")

    # Summary table
    print("\n" + "=" * 90)
    print("FINAL SUMMARY")
    print("=" * 90)
    hdr = f"{'Name':<10} {'Sharpe':>7} {'Sortino':>8} {'PF':>7} {'WR%':>6} {'Trades':>7} {'MDD%':>7} {'Ret%':>8} {'Perm-p':>7} {'Gates':>6}"
    print(hdr)
    print("-" * 90)
    for key in sorted(results.keys()):
        r = results[key]
        name = key.replace('variant_', 'V-').replace('benchmark_', 'BM:')
        pf_str = f"{r['pf']:>7.2f}" if 'pf' in r else "   n/a"
        wr_str = f"{r['wr']:>6.1f}" if 'wr' in r else "  n/a"
        tr_str = f"{r['trades']:>7}" if 'trades' in r else "   n/a"
        pp = r.get('perm_p', 'n/a')
        gt = r.get('gates_passed', 'n/a')
        print(f"{name:<10} {r['sharpe']:>7.3f} {r['sortino']:>8.3f} {pf_str} {wr_str} "
              f"{tr_str} {r['mdd']:>7.1f} {r['total_return_pct']:>8.1f} {str(pp):>7} {str(gt):>6}")

    print(f"\nTotal runtime: {time.time()-t_start:.0f}s")


if __name__ == '__main__':
    main()
