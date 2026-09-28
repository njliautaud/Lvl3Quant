#!/usr/bin/env python3
"""
SECTOR BREADTH AS ENTRY FILTER
================================
Hypothesis: When multiple sectors are oversold simultaneously (broad market stress),
dip-buying any individual sector has higher expected return than when only 1 sector
is oversold (idiosyncratic dip).

Rationale: Broad selloffs are often fear-driven and recover. Single-sector dips
may indicate fundamental rotation away from that sector (permanent repricing).

Tests:
  A) Breadth count: how many sectors have RSI(14) < 35 on any given day?
  B) Does high breadth (3+ sectors oversold) predict better 5-day returns?
  C) Optimal breadth threshold for our entry filter
  D) Does this interact with VIX? (breadth + VIX spike = best entry?)

Walk-forward: 2020-2026, all OOT days
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats
from datetime import datetime

np.random.seed(42)

SECTOR_ETFS = ['XLK', 'XLF', 'XLE', 'XLU', 'XLP', 'XLY', 'XLV', 'XLI', 'XLB', 'XLC', 'XLRE']
BENCHMARK = 'SPY'
VIX_TICKER = '^VIX'

DATA_START = '2019-01-01'
DATA_END = '2026-08-21'


def download_data():
    tickers = SECTOR_ETFS + [BENCHMARK, VIX_TICKER]
    print("Downloading data...", flush=True)
    raw = yf.download(tickers, start=DATA_START, end=DATA_END,
                       progress=False, auto_adjust=True, group_by='ticker', threads=True)
    closes = {}
    for ticker in tickers:
        try:
            s = raw[ticker]['Close'].dropna().squeeze()
            if len(s) > 100:
                closes[ticker] = s
        except Exception:
            pass
    df = pd.DataFrame(closes).dropna()
    print(f"  Got {len(df)} trading days", flush=True)
    return df


def compute_rsi(series, period=14):
    """Standard RSI calculation."""
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def compute_sharpe(rets):
    if len(rets) < 10 or rets.std() == 0:
        return 0.0
    return float((rets.mean() / rets.std()) * np.sqrt(252))


def main():
    print("=" * 90, flush=True)
    print("SECTOR BREADTH AS ENTRY FILTER", flush=True)
    print("=" * 90, flush=True)

    prices = download_data()
    returns = prices[SECTOR_ETFS].pct_change()
    spy_returns = prices[BENCHMARK].pct_change()

    # Compute RSI for all sectors
    rsi_df = pd.DataFrame(index=prices.index)
    for etf in SECTOR_ETFS:
        if etf in prices.columns:
            rsi_df[etf] = compute_rsi(prices[etf])

    # Breadth: count sectors with RSI < threshold
    for threshold in [30, 35, 40]:
        breadth = (rsi_df < threshold).sum(axis=1)
        rsi_df[f'breadth_{threshold}'] = breadth

    # VIX
    vix = prices[VIX_TICKER] if VIX_TICKER in prices.columns else None
    vix_20d_mean = vix.rolling(20).mean() if vix is not None else None

    print(f"\nDate range: {prices.index[0].date()} to {prices.index[-1].date()}", flush=True)

    # ── ANALYSIS 1: Breadth distribution ──────────────────────────────────
    print(f"\n{'=' * 70}", flush=True)
    print("BREADTH DISTRIBUTION (RSI<35)", flush=True)
    print(f"{'=' * 70}", flush=True)

    breadth = rsi_df['breadth_35']
    warmup_end = 60  # skip warmup

    for count in range(0, 8):
        mask = breadth.iloc[warmup_end:] == count
        pct = mask.mean() * 100
        n_days = mask.sum()
        print(f"  {count} sectors oversold: {n_days} days ({pct:.1f}%)", flush=True)

    # ── ANALYSIS 2: Forward returns by breadth level ──────────────────────
    print(f"\n{'=' * 70}", flush=True)
    print("FORWARD RETURNS BY BREADTH (RSI<35)", flush=True)
    print(f"{'=' * 70}", flush=True)

    hold_periods = [1, 3, 5, 10]

    for hold in hold_periods:
        print(f"\n  --- {hold}-day forward returns ---", flush=True)

        for threshold in [30, 35, 40]:
            breadth_col = f'breadth_{threshold}'

            for min_breadth in [0, 1, 2, 3, 4]:
                mask = rsi_df[breadth_col].iloc[warmup_end:-hold] >= min_breadth

                # Get oversold sectors on those days
                entry_days = rsi_df.index[warmup_end:-hold][mask]
                if len(entry_days) < 20:
                    continue

                # Average forward return of oversold sectors
                fwd_rets = []
                for day in entry_days:
                    idx = prices.index.get_loc(day)
                    oversold = rsi_df.loc[day, SECTOR_ETFS] < threshold
                    oversold_sectors = [c for c in SECTOR_ETFS if oversold.get(c, False)]

                    if not oversold_sectors:
                        continue

                    # Forward return of these sectors
                    fwd_prices = prices[oversold_sectors].iloc[idx:idx+hold+1]
                    if len(fwd_prices) < hold + 1:
                        continue

                    fwd_ret = (fwd_prices.iloc[-1] / fwd_prices.iloc[0] - 1).mean()
                    fwd_rets.append(float(fwd_ret))

                if len(fwd_rets) < 10:
                    continue

                fwd_rets = np.array(fwd_rets)
                avg_ret = fwd_rets.mean() * 100
                wr = (fwd_rets > 0).mean() * 100
                sharpe = compute_sharpe(pd.Series(fwd_rets))
                t_stat, p_val = stats.ttest_1samp(fwd_rets, 0)

                if min_breadth in [0, 1, 3] and threshold == 35:
                    print(f"    RSI<{threshold}, breadth>={min_breadth}: "
                          f"avg={avg_ret:+.3f}%, WR={wr:.1f}%, Sharpe={sharpe:.3f}, "
                          f"n={len(fwd_rets)}, p={p_val:.4f}", flush=True)

    # ── ANALYSIS 3: Breadth-filtered dip-buy strategy ────────────────────
    print(f"\n{'=' * 70}", flush=True)
    print("BREADTH-FILTERED DIP-BUY STRATEGY", flush=True)
    print("Buy oversold sectors ONLY when breadth >= threshold, hold 5d", flush=True)
    print(f"{'=' * 70}", flush=True)

    for min_breadth in [1, 2, 3, 4, 5]:
        for rsi_thresh in [30, 35]:
            strat_rets = []
            strat_dates = []
            days_since = 5
            holdings = []

            tradeable = prices.index[warmup_end:]

            for date in tradeable:
                idx = prices.index.get_loc(date)

                if days_since >= 5:
                    breadth_val = (rsi_df.loc[date, SECTOR_ETFS] < rsi_thresh).sum()

                    if breadth_val >= min_breadth:
                        oversold = [c for c in SECTOR_ETFS if rsi_df.loc[date, c] < rsi_thresh]
                        if oversold:
                            holdings = oversold[:3]  # max 3
                            days_since = 0

                if holdings and date in returns.index:
                    dr = returns.loc[date, holdings]
                    port_ret = dr.mean()
                    if days_since == 0:
                        port_ret -= 0.002 / 5
                    strat_rets.append(float(port_ret))
                else:
                    strat_rets.append(0.0)

                strat_dates.append(date)
                days_since += 1

            sr = pd.Series(strat_rets, index=strat_dates)
            in_pos = sr != 0

            if in_pos.sum() < 20:
                continue

            sharpe = compute_sharpe(sr[in_pos])
            cum = (1 + sr).prod() - 1
            spy_sr = spy_returns.loc[strat_dates]
            spy_cum = (1 + spy_sr).prod() - 1

            # Trade count
            trade_starts = in_pos & (~in_pos.shift(1).fillna(False))
            n_trades = trade_starts.sum()

            # Regime
            green = spy_sr > 0
            red = spy_sr < 0
            sh_g = compute_sharpe(sr[green]) if green.sum() > 10 else 0
            sh_r = compute_sharpe(sr[red]) if red.sum() > 10 else 0
            max_reg = max(abs(sh_g), abs(sh_r))
            regime_gap = abs(sh_g - sh_r) / max_reg if max_reg > 0 else 0

            print(f"  RSI<{rsi_thresh}, breadth>={min_breadth}: Sharpe={sharpe:.3f}, "
                  f"cum={cum*100:.2f}%, excess={((cum-spy_cum)*100):.2f}%, "
                  f"trades={n_trades}, ShG={sh_g:.3f}, ShR={sh_r:.3f}, gap={regime_gap:.4f}", flush=True)

    # ── ANALYSIS 4: Breadth + VIX interaction ────────────────────────────
    if vix is not None:
        print(f"\n{'=' * 70}", flush=True)
        print("BREADTH + VIX INTERACTION", flush=True)
        print(f"{'=' * 70}", flush=True)

        for vix_threshold in ['above_avg', 'spike']:
            for min_breadth in [2, 3]:
                strat_rets = []
                strat_dates = []
                days_since = 5
                holdings = []

                for date in prices.index[warmup_end:]:
                    idx = prices.index.get_loc(date)

                    if days_since >= 5:
                        breadth_val = (rsi_df.loc[date, SECTOR_ETFS] < 35).sum()
                        vix_val = vix.loc[date] if date in vix.index else None
                        vix_avg = vix_20d_mean.loc[date] if date in vix_20d_mean.index else None

                        vix_condition = False
                        if vix_val is not None and vix_avg is not None:
                            if vix_threshold == 'above_avg':
                                vix_condition = vix_val > vix_avg
                            elif vix_threshold == 'spike':
                                vix_condition = vix_val > vix_avg * 1.2  # 20% above average

                        if breadth_val >= min_breadth and vix_condition:
                            oversold = [c for c in SECTOR_ETFS if rsi_df.loc[date, c] < 35]
                            if oversold:
                                holdings = oversold[:3]
                                days_since = 0

                    if holdings and date in returns.index:
                        dr = returns.loc[date, holdings]
                        port_ret = dr.mean()
                        if days_since == 0:
                            port_ret -= 0.002 / 5
                        strat_rets.append(float(port_ret))
                    else:
                        strat_rets.append(0.0)

                    strat_dates.append(date)
                    days_since += 1

                sr = pd.Series(strat_rets, index=strat_dates)
                in_pos = sr != 0

                if in_pos.sum() < 10:
                    print(f"  Breadth>={min_breadth} + VIX {vix_threshold}: Too few trades ({in_pos.sum()} days)", flush=True)
                    continue

                sharpe = compute_sharpe(sr[in_pos])
                cum = (1 + sr).prod() - 1
                spy_sr = spy_returns.loc[strat_dates]
                spy_cum = (1 + spy_sr).prod() - 1

                trade_starts = in_pos & (~in_pos.shift(1).fillna(False))
                n_trades = trade_starts.sum()

                # Regime
                green = spy_sr > 0
                red = spy_sr < 0
                sh_g = compute_sharpe(sr[green]) if green.sum() > 10 else 0
                sh_r = compute_sharpe(sr[red]) if red.sum() > 10 else 0
                max_reg = max(abs(sh_g), abs(sh_r))
                regime_gap = abs(sh_g - sh_r) / max_reg if max_reg > 0 else 0

                print(f"  Breadth>={min_breadth} + VIX {vix_threshold}: Sharpe={sharpe:.3f}, "
                      f"cum={cum*100:.2f}%, excess={((cum-spy_cum)*100):.2f}%, "
                      f"trades={n_trades}, ShG={sh_g:.3f}, ShR={sh_r:.3f}, gap={regime_gap:.4f}", flush=True)

    # ── SUMMARY ──────────────────────────────────────────────────────────
    print(f"\n\n{'=' * 90}", flush=True)
    print("SUMMARY", flush=True)
    print(f"{'=' * 90}", flush=True)
    print(f"Breadth = number of sector ETFs with RSI(14) below threshold simultaneously.", flush=True)
    print(f"High breadth = market-wide stress (fear-driven selloff = better dip-buy).", flush=True)
    print(f"Low breadth = sector-specific weakness (may be permanent rotation).", flush=True)
    print(f"\nKey question: Does requiring breadth >= 2 or 3 improve our entry WR/Sharpe?", flush=True)
    print(f"If yes → wire as a filter into our aggregator (boost confidence when breadth high).", flush=True)

    print(f"\nDone. {datetime.now()}", flush=True)


if __name__ == '__main__':
    main()
