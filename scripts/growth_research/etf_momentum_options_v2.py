#!/usr/bin/env python3
"""
ETF Momentum Options v2 — Six Strategy Research Backtest
=========================================================
Tests six distinct directional options strategies on sector ETFs
where momentum/rotation signals fire. Unlike dip-buying (slow 3-5%
recoveries = bad for options), sector momentum produces fast 5-15% moves
that can work with properly structured options.

Strategies tested:
  A) Momentum Breakout Calls     — SMA breakout + rising relative strength
  B) Mean Reversion Puts         — Overbought sectors when VIX<20 (complacency)
  C) Rotation Momentum Calls     — Top-ranked sector by 14d momentum, 10d hold
  D) Contrarian Sector Bounce    — Sector-specific weakness, mean reversion calls
  E) VIX Spike Sector Puts       — Buy puts on defensive laggard after VIX spike
  F) Sector Pair Trade           — Long calls top + long puts bottom, sector-neutral

ETF Universe: XLK, XLF, XLE, XLV, XLY, XLI, XLP, XLU, XLRE, XLB, XLC
Period: 2020-01-01 to 2026-07-01
Capital: $100 max premium per trade, max 2 concurrent
Commission: $0.65 per contract per leg
Pricing: Black-Scholes, VIX as ATM IV proxy, 4.5% risk-free rate

Validation: 5-gate
  1. Regime gap |Sharpe_bull - Sharpe_bear| / max < 0.50
  2. Permutation p < 0.05 (1000 permutations)
  3. 4/4 sub-periods positive
  4. MDD > -50%
  5. N >= 20 trades
"""

import sys
import os
import json
import warnings
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from scipy.stats import norm
import itertools

warnings.filterwarnings('ignore')
sys.path.insert(0, '/home/jupiter/Lvl3Quant')

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

# ─── CONSTANTS ───────────────────────────────────────────────────────────────
SECTOR_ETFS = ['XLK', 'XLF', 'XLE', 'XLV', 'XLY', 'XLI', 'XLP', 'XLU', 'XLRE', 'XLB', 'XLC']
ALL_TICKERS = SECTOR_ETFS + ['SPY', '^VIX']

RISK_FREE = 0.045
COMMISSION_PER_LEG = 0.65   # per contract
MAX_PREMIUM = 100.0         # max dollars spent on premium per trade
MAX_CONCURRENT = 2

START_DATE = '2020-01-01'
END_DATE   = '2026-07-01'
STARTING_CAPITAL = 1000.0   # notional tracker


# ─── BLACK-SCHOLES ────────────────────────────────────────────────────────────
def bs_call(S, K, T, r, sigma):
    if T <= 1e-6 or sigma <= 1e-6:
        return max(S - K, 0.0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

def bs_put(S, K, T, r, sigma):
    if T <= 1e-6 or sigma <= 1e-6:
        return max(K - S, 0.0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


# ─── DATA LOADING ─────────────────────────────────────────────────────────────
def load_data():
    """Load price data from cache or download via yfinance."""
    cache = '/home/jupiter/Lvl3Quant/data/etf_momentum_options_v2_cache.parquet'

    if os.path.exists(cache):
        print(f"Loading cached data from {cache}")
        return pd.read_parquet(cache)

    print("Downloading data via yfinance...")
    try:
        import yfinance as yf
    except ImportError:
        raise ImportError("pip install yfinance")

    frames = []
    for ticker in ALL_TICKERS:
        try:
            raw = yf.download(ticker, start=START_DATE, end=END_DATE,
                              progress=False, auto_adjust=True)
            if len(raw) < 100:
                print(f"  {ticker}: too few rows ({len(raw)}), skipping")
                continue
            df = raw[['Close', 'Volume']].copy()
            df.columns = ['close', 'volume']
            df.index = pd.to_datetime(df.index)
            df.index.name = 'date'
            df['ticker'] = ticker
            df = df.reset_index()
            frames.append(df)
            print(f"  {ticker}: {len(df)} rows")
        except Exception as e:
            print(f"  {ticker}: error {e}")

    if not frames:
        raise RuntimeError("No data downloaded")

    data = pd.concat(frames, ignore_index=True)
    data['date'] = pd.to_datetime(data['date'])
    data = data.sort_values(['ticker', 'date'])
    data.to_parquet(cache)
    print(f"Saved cache: {cache}")
    return data


def build_pivot(data, tickers):
    """Build wide close-price pivot."""
    sub = data[data['ticker'].isin(tickers)].copy()
    piv = sub.pivot_table(index='date', columns='ticker', values='close')
    piv = piv.sort_index()
    return piv


# ─── HELPER METRICS ───────────────────────────────────────────────────────────
def calc_metrics(trades_df, equity_curve, starting=STARTING_CAPITAL):
    """Compute performance metrics for a strategy."""
    if trades_df is None or len(trades_df) == 0:
        return None

    pnls = trades_df['pnl'].values
    eq = np.array(equity_curve)

    n = len(pnls)
    wr = np.mean(pnls > 0)
    gp = pnls[pnls > 0].sum() if (pnls > 0).any() else 0.0
    gl = abs(pnls[pnls < 0].sum()) if (pnls < 0).any() else 1e-9
    pf = gp / gl if gl > 1e-9 else float('inf')

    if len(eq) >= 2:
        period_rets = np.diff(eq) / np.where(eq[:-1] > 0, eq[:-1], 1.0)
        sharpe = period_rets.mean() / (period_rets.std() + 1e-9) * np.sqrt(252 / 14)
        neg = period_rets[period_rets < 0]
        sortino = period_rets.mean() / (neg.std() + 1e-9) * np.sqrt(252 / 14) if len(neg) > 1 else np.nan
    else:
        sharpe = np.nan
        sortino = np.nan

    peak = np.maximum.accumulate(eq)
    dd = (eq - peak) / np.where(peak > 0, peak, 1.0)
    mdd = dd.min()

    years = (trades_df['exit_date'].max() - trades_df['entry_date'].min()).days / 365.25
    final = eq[-1]
    cagr = (final / starting) ** (1 / max(years, 0.1)) - 1 if final > 0 and starting > 0 else -1.0

    return dict(n=n, wr=wr, pf=pf, sharpe=sharpe, sortino=sortino,
                mdd=mdd, cagr=cagr, final_equity=final,
                total_pnl=pnls.sum())


def permutation_test(trades_df, n_perms=1000):
    """Shuffle pnl signs to get null distribution of total pnl."""
    if trades_df is None or len(trades_df) < 5:
        return np.nan
    actual = trades_df['pnl'].sum()
    pnl_abs = trades_df['pnl'].abs().values
    count = 0
    for _ in range(n_perms):
        signs = np.random.choice([-1, 1], size=len(pnl_abs))
        if (pnl_abs * signs).sum() >= actual:
            count += 1
    return count / n_perms


def subperiod_check(trades_df):
    """Check 4/4 sub-periods are positive."""
    if trades_df is None or len(trades_df) == 0:
        return False, []
    start = trades_df['entry_date'].min()
    end   = trades_df['exit_date'].max()
    total_days = (end - start).days
    chunk = total_days / 4
    results = []
    for i in range(4):
        s = start + timedelta(days=i * chunk)
        e = start + timedelta(days=(i + 1) * chunk)
        sub = trades_df[(trades_df['entry_date'] >= s) & (trades_df['entry_date'] < e)]
        results.append(sub['pnl'].sum() if len(sub) > 0 else 0.0)
    passed = sum(r > 0 for r in results)
    return passed >= 4, results


def regime_gap_check(trades_df, spy_returns):
    """Compute Sharpe in bull vs bear regimes. Return gap."""
    if trades_df is None or len(trades_df) < 5:
        return np.nan, np.nan, np.nan

    def daily_sharpe(pnl_series):
        if len(pnl_series) < 3:
            return np.nan
        m = pnl_series.mean()
        s = pnl_series.std() + 1e-9
        return m / s * np.sqrt(252)

    bull_pnls = []
    bear_pnls = []
    for _, row in trades_df.iterrows():
        d = row['entry_date']
        # Look up SPY return in the 20 days before entry
        window = spy_returns.loc[:d].tail(20)
        spy_ret_20d = window.sum() if len(window) > 0 else 0.0
        if spy_ret_20d >= 0:
            bull_pnls.append(row['pnl'])
        else:
            bear_pnls.append(row['pnl'])

    sh_bull = daily_sharpe(np.array(bull_pnls)) if bull_pnls else np.nan
    sh_bear = daily_sharpe(np.array(bear_pnls)) if bear_pnls else np.nan

    if np.isnan(sh_bull) or np.isnan(sh_bear):
        return sh_bull, sh_bear, np.nan

    denom = max(abs(sh_bull), abs(sh_bear), 1e-9)
    gap = abs(sh_bull - sh_bear) / denom
    return sh_bull, sh_bear, gap


def validate(strategy_name, trades_df, equity_curve, spy_returns):
    """5-gate validation. Returns dict."""
    gates = {}

    # Gate 5: N >= 20
    n = len(trades_df) if trades_df is not None else 0
    gates['N>=20'] = (n >= 20, n)

    # Gate 4: MDD > -50%
    eq = np.array(equity_curve)
    if len(eq) >= 2:
        peak = np.maximum.accumulate(eq)
        dd = (eq - peak) / np.where(peak > 0, peak, 1.0)
        mdd = dd.min()
    else:
        mdd = 0.0
    gates['MDD>-50%'] = (mdd > -0.50, mdd)

    # Gate 3: 4/4 sub-periods positive
    ok4, sub_pnls = subperiod_check(trades_df)
    gates['4/4_subperiods'] = (ok4, sub_pnls)

    # Gate 2: Permutation p < 0.05
    p_val = permutation_test(trades_df, n_perms=1000)
    gates['perm_p<0.05'] = (p_val < 0.05, p_val)

    # Gate 1: Regime gap < 0.50
    sh_bull, sh_bear, gap = regime_gap_check(trades_df, spy_returns)
    gates['regime_gap<0.50'] = (gap < 0.50 if not np.isnan(gap) else False,
                                 dict(sh_bull=sh_bull, sh_bear=sh_bear, gap=gap))

    passed = sum(1 for k, (ok, _) in gates.items() if ok)
    return dict(strategy=strategy_name, gates=gates, gates_passed=passed, gates_total=5)


# ─── STRATEGY A: MOMENTUM BREAKOUT CALLS ─────────────────────────────────────
def strategy_a(price_piv, vix_series):
    """
    When sector ETF crosses above 20d SMA after being below for 5+ days
    AND 14d relative strength vs SPY is improving → buy ATM call 30 DTE.
    Exit: +50% premium OR 14d max hold OR -30% stop.
    """
    trades = []
    equity = [STARTING_CAPITAL]
    equity_dates = [price_piv.index[30]]

    spy = price_piv['SPY'] if 'SPY' in price_piv.columns else None
    open_trades = {}  # ticker -> trade_dict

    dates = price_piv.index.tolist()

    for i, d in enumerate(dates):
        if i < 30:
            continue

        vix = float(vix_series.loc[d]) if d in vix_series.index else 20.0
        sigma = max(vix / 100.0, 0.10)

        # Close expired trades
        to_close = []
        pnl_today = 0.0
        for ticker, tr in open_trades.items():
            days_held = (d - tr['entry_date']).days
            S = float(price_piv.loc[d, ticker]) if d in price_piv.index and ticker in price_piv.columns else tr['S']
            T_rem = max(0, (tr['dte'] - days_held)) / 365.0

            # Current option value
            cur_val = bs_call(S, tr['K'], T_rem, RISK_FREE, tr['sigma'])
            prem_change_pct = (cur_val - tr['entry_price']) / (tr['entry_price'] + 1e-9)

            exit_reason = None
            if prem_change_pct >= 0.50:
                exit_reason = '+50%'
            elif prem_change_pct <= -0.30:
                exit_reason = '-30% stop'
            elif days_held >= 14:
                exit_reason = '14d max'

            if exit_reason:
                # Exit premium: intrinsic or BS
                exit_prem = max(cur_val, S - tr['K'])
                exit_prem = min(exit_prem, tr['entry_price'] * 2.0)  # cap at 2x
                n_c = tr['n_contracts']
                trade_pnl = (exit_prem - tr['entry_price']) * 100 * n_c - COMMISSION_PER_LEG * 2 * n_c
                trade_pnl = max(trade_pnl, -tr['cost'])  # can't lose more than paid
                pnl_today += trade_pnl
                tr['pnl'] = trade_pnl
                tr['exit_date'] = d
                tr['exit_reason'] = exit_reason
                tr['exit_prem'] = exit_prem
                trades.append(tr)
                to_close.append(ticker)

        for ticker in to_close:
            del open_trades[ticker]

        # Entry signals
        if len(open_trades) >= MAX_CONCURRENT:
            equity.append(equity[-1] + pnl_today)
            equity_dates.append(d)
            continue

        for ticker in SECTOR_ETFS:
            if ticker not in price_piv.columns:
                continue
            if ticker in open_trades:
                continue
            if len(open_trades) >= MAX_CONCURRENT:
                break

            hist = price_piv[ticker].iloc[max(0, i - 30):i + 1].dropna()
            if len(hist) < 22:
                continue

            sma20 = hist.iloc[-21:].mean()
            S = float(hist.iloc[-1])

            # Was below SMA for 5+ consecutive days?
            recent5 = hist.iloc[-6:-1]
            sma20_recent = [hist.iloc[max(0, j - 20):j].mean() for j in range(len(hist) - 5, len(hist))]
            below_5 = all(recent5.values[k] < sma20_recent[k] for k in range(min(5, len(recent5))))

            # Now above SMA (breakout)
            above_now = S > sma20

            if not (below_5 and above_now):
                continue

            # Relative strength improving vs SPY
            if spy is not None and i >= 14:
                etf_ret_14d = float(price_piv[ticker].iloc[i] / price_piv[ticker].iloc[i - 14] - 1)
                spy_ret_14d = float(spy.iloc[i] / spy.iloc[i - 14] - 1)
                rs_improving = etf_ret_14d > spy_ret_14d
            else:
                rs_improving = True

            if not rs_improving:
                continue

            # BUY ATM call
            K = S
            T = 30 / 365.0
            entry_prem = bs_call(S, K, T, RISK_FREE, sigma)
            if entry_prem <= 0.01:
                continue

            cost_per_contract = entry_prem * 100
            n_contracts = max(1, int(MAX_PREMIUM / cost_per_contract))
            total_cost = cost_per_contract * n_contracts + COMMISSION_PER_LEG * n_contracts

            if total_cost > equity[-1] * 0.40:
                n_contracts = max(1, int(equity[-1] * 0.40 / cost_per_contract))
                total_cost = cost_per_contract * n_contracts + COMMISSION_PER_LEG * n_contracts

            open_trades[ticker] = dict(
                strategy='A_Breakout_Call',
                ticker=ticker,
                entry_date=d,
                exit_date=None,
                S=S, K=K, dte=30, sigma=sigma,
                entry_price=entry_prem,
                n_contracts=n_contracts,
                cost=total_cost,
                pnl=0.0,
                direction='call',
            )

        equity.append(max(0, equity[-1] + pnl_today))
        equity_dates.append(d)

    # Close any still-open trades at last date
    for ticker, tr in open_trades.items():
        d = dates[-1]
        S = float(price_piv.loc[d, ticker]) if d in price_piv.index and ticker in price_piv.columns else tr['S']
        exit_prem = max(S - tr['K'], 0.0)
        n_c = tr['n_contracts']
        trade_pnl = (exit_prem - tr['entry_price']) * 100 * n_c - COMMISSION_PER_LEG * 2 * n_c
        trade_pnl = max(trade_pnl, -tr['cost'])
        tr['pnl'] = trade_pnl
        tr['exit_date'] = d
        tr['exit_reason'] = 'end_of_backtest'
        trades.append(tr)

    return pd.DataFrame(trades) if trades else None, equity, equity_dates


# ─── STRATEGY B: MEAN REVERSION PUTS ON WEAK SECTORS ─────────────────────────
def strategy_b(price_piv, vix_series):
    """
    ETF >2 std devs above 50d SMA AND VIX < 20 (complacency)
    → buy ATM put 30 DTE. Exit +50% or 14d or -30%.
    """
    trades = []
    equity = [STARTING_CAPITAL]
    equity_dates = [price_piv.index[60]]
    open_trades = {}
    dates = price_piv.index.tolist()

    for i, d in enumerate(dates):
        if i < 60:
            continue

        vix = float(vix_series.loc[d]) if d in vix_series.index else 20.0
        sigma = max(vix / 100.0, 0.10)

        # Close expired
        pnl_today = 0.0
        to_close = []
        for ticker, tr in open_trades.items():
            days_held = (d - tr['entry_date']).days
            S = float(price_piv.loc[d, ticker]) if d in price_piv.index and ticker in price_piv.columns else tr['S']
            T_rem = max(0, (tr['dte'] - days_held)) / 365.0
            cur_val = bs_put(S, tr['K'], T_rem, RISK_FREE, tr['sigma'])
            prem_chg = (cur_val - tr['entry_price']) / (tr['entry_price'] + 1e-9)

            exit_reason = None
            if prem_chg >= 0.50:
                exit_reason = '+50%'
            elif prem_chg <= -0.30:
                exit_reason = '-30% stop'
            elif days_held >= 14:
                exit_reason = '14d max'

            if exit_reason:
                exit_prem = max(cur_val, tr['K'] - S)
                n_c = tr['n_contracts']
                pnl = (exit_prem - tr['entry_price']) * 100 * n_c - COMMISSION_PER_LEG * 2 * n_c
                pnl = max(pnl, -tr['cost'])
                pnl_today += pnl
                tr['pnl'] = pnl
                tr['exit_date'] = d
                tr['exit_reason'] = exit_reason
                trades.append(tr)
                to_close.append(ticker)

        for t in to_close:
            del open_trades[t]

        if len(open_trades) >= MAX_CONCURRENT:
            equity.append(equity[-1] + pnl_today)
            equity_dates.append(d)
            continue

        # VIX must be below 20 (complacency)
        if vix >= 20:
            equity.append(equity[-1] + pnl_today)
            equity_dates.append(d)
            continue

        for ticker in SECTOR_ETFS:
            if ticker not in price_piv.columns:
                continue
            if ticker in open_trades or len(open_trades) >= MAX_CONCURRENT:
                break

            hist = price_piv[ticker].iloc[max(0, i - 60):i + 1].dropna()
            if len(hist) < 52:
                continue

            sma50 = hist.iloc[-51:].mean()
            std50 = hist.iloc[-51:].std()
            S = float(hist.iloc[-1])

            z = (S - sma50) / (std50 + 1e-9)
            if z < 2.0:
                continue

            # ATM put
            K = S
            T = 30 / 365.0
            entry_prem = bs_put(S, K, T, RISK_FREE, sigma)
            if entry_prem <= 0.01:
                continue

            cost_per_contract = entry_prem * 100
            n_contracts = max(1, int(MAX_PREMIUM / cost_per_contract))
            total_cost = cost_per_contract * n_contracts + COMMISSION_PER_LEG * n_contracts

            if total_cost > equity[-1] * 0.40:
                n_contracts = max(1, int(equity[-1] * 0.40 / cost_per_contract))
                total_cost = cost_per_contract * n_contracts + COMMISSION_PER_LEG * n_contracts

            open_trades[ticker] = dict(
                strategy='B_MeanRev_Put',
                ticker=ticker,
                entry_date=d,
                exit_date=None,
                S=S, K=K, dte=30, sigma=sigma,
                entry_price=entry_prem,
                n_contracts=n_contracts,
                cost=total_cost,
                pnl=0.0,
                direction='put',
                z_score=z,
            )

        equity.append(max(0, equity[-1] + pnl_today))
        equity_dates.append(d)

    # Close remaining
    for ticker, tr in open_trades.items():
        d = dates[-1]
        S = float(price_piv.loc[d, ticker]) if d in price_piv.index and ticker in price_piv.columns else tr['S']
        exit_prem = max(tr['K'] - S, 0.0)
        n_c = tr['n_contracts']
        pnl = (exit_prem - tr['entry_price']) * 100 * n_c - COMMISSION_PER_LEG * 2 * n_c
        pnl = max(pnl, -tr['cost'])
        tr['pnl'] = pnl
        tr['exit_date'] = d
        tr['exit_reason'] = 'end'
        trades.append(tr)

    return pd.DataFrame(trades) if trades else None, equity, equity_dates


# ─── STRATEGY C: ROTATION MOMENTUM CALLS ─────────────────────────────────────
def strategy_c(price_piv, vix_series):
    """
    Rank all sector ETFs by 14d momentum. Buy ATM call on #1 ranked.
    Hold 10 days. Rebalance when rank changes (or 10d max).
    """
    trades = []
    equity = [STARTING_CAPITAL]
    equity_dates = [price_piv.index[20]]
    open_trade = None  # single trade at a time
    dates = price_piv.index.tolist()

    for i, d in enumerate(dates):
        if i < 20:
            continue

        vix = float(vix_series.loc[d]) if d in vix_series.index else 20.0
        sigma = max(vix / 100.0, 0.10)

        pnl_today = 0.0

        # Check if open trade should close
        if open_trade is not None:
            days_held = (d - open_trade['entry_date']).days
            ticker = open_trade['ticker']
            S = float(price_piv.loc[d, ticker]) if d in price_piv.index and ticker in price_piv.columns else open_trade['S']
            T_rem = max(0, (open_trade['dte'] - days_held)) / 365.0
            cur_val = bs_call(S, open_trade['K'], T_rem, RISK_FREE, open_trade['sigma'])

            # Rank current leader
            moms = {}
            for t in SECTOR_ETFS:
                if t in price_piv.columns and i >= 14:
                    try:
                        ret = float(price_piv[t].iloc[i] / price_piv[t].iloc[i - 14] - 1)
                        moms[t] = ret
                    except Exception:
                        pass
            top_now = max(moms, key=moms.get) if moms else None

            should_exit = days_held >= 10 or (top_now and top_now != ticker)
            if should_exit:
                exit_prem = max(cur_val, S - open_trade['K'])
                n_c = open_trade['n_contracts']
                pnl = (exit_prem - open_trade['entry_price']) * 100 * n_c - COMMISSION_PER_LEG * 2 * n_c
                pnl = max(pnl, -open_trade['cost'])
                pnl_today += pnl
                open_trade['pnl'] = pnl
                open_trade['exit_date'] = d
                open_trade['exit_reason'] = '10d_max' if days_held >= 10 else 'rank_change'
                trades.append(open_trade)
                open_trade = None

        # Open new trade if none active
        if open_trade is None:
            moms = {}
            for t in SECTOR_ETFS:
                if t in price_piv.columns and i >= 14:
                    try:
                        ret = float(price_piv[t].iloc[i] / price_piv[t].iloc[i - 14] - 1)
                        moms[t] = ret
                    except Exception:
                        pass

            if moms:
                top_ticker = max(moms, key=moms.get)
                S = float(price_piv.loc[d, top_ticker]) if top_ticker in price_piv.columns else None
                if S and S > 0:
                    K = S
                    T = 30 / 365.0
                    entry_prem = bs_call(S, K, T, RISK_FREE, sigma)
                    if entry_prem > 0.01:
                        cost_per_contract = entry_prem * 100
                        n_contracts = max(1, int(MAX_PREMIUM / cost_per_contract))
                        total_cost = cost_per_contract * n_contracts + COMMISSION_PER_LEG * n_contracts

                        if total_cost <= equity[-1] * 0.50:
                            open_trade = dict(
                                strategy='C_Rotation_Calls',
                                ticker=top_ticker,
                                entry_date=d,
                                exit_date=None,
                                S=S, K=K, dte=30, sigma=sigma,
                                entry_price=entry_prem,
                                n_contracts=n_contracts,
                                cost=total_cost,
                                pnl=0.0,
                                direction='call',
                                mom_14d=moms[top_ticker],
                            )

        equity.append(max(0, equity[-1] + pnl_today))
        equity_dates.append(d)

    # Close remaining
    if open_trade is not None:
        d = dates[-1]
        ticker = open_trade['ticker']
        S = float(price_piv.loc[d, ticker]) if d in price_piv.index and ticker in price_piv.columns else open_trade['S']
        exit_prem = max(S - open_trade['K'], 0.0)
        n_c = open_trade['n_contracts']
        pnl = (exit_prem - open_trade['entry_price']) * 100 * n_c - COMMISSION_PER_LEG * 2 * n_c
        pnl = max(pnl, -open_trade['cost'])
        open_trade['pnl'] = pnl
        open_trade['exit_date'] = d
        open_trade['exit_reason'] = 'end'
        trades.append(open_trade)

    return pd.DataFrame(trades) if trades else None, equity, equity_dates


# ─── STRATEGY D: CONTRARIAN SECTOR BOUNCE CALLS ──────────────────────────────
def strategy_d(price_piv, vix_series):
    """
    Sector drops >5% in 5 days while SPY drops <2% → sector-specific weakness.
    Buy ATM call 14 DTE expecting mean reversion.
    Exit +40% or 7d or -30%.
    """
    trades = []
    equity = [STARTING_CAPITAL]
    equity_dates = [price_piv.index[10]]
    open_trades = {}
    dates = price_piv.index.tolist()
    spy = price_piv['SPY'] if 'SPY' in price_piv.columns else None

    for i, d in enumerate(dates):
        if i < 10:
            continue

        vix = float(vix_series.loc[d]) if d in vix_series.index else 20.0
        sigma = max(vix / 100.0, 0.10)

        pnl_today = 0.0
        to_close = []
        for ticker, tr in open_trades.items():
            days_held = (d - tr['entry_date']).days
            S = float(price_piv.loc[d, ticker]) if d in price_piv.index and ticker in price_piv.columns else tr['S']
            T_rem = max(0, (tr['dte'] - days_held)) / 365.0
            cur_val = bs_call(S, tr['K'], T_rem, RISK_FREE, tr['sigma'])
            prem_chg = (cur_val - tr['entry_price']) / (tr['entry_price'] + 1e-9)

            exit_reason = None
            if prem_chg >= 0.40:
                exit_reason = '+40%'
            elif prem_chg <= -0.30:
                exit_reason = '-30% stop'
            elif days_held >= 7:
                exit_reason = '7d max'

            if exit_reason:
                exit_prem = max(cur_val, S - tr['K'])
                n_c = tr['n_contracts']
                pnl = (exit_prem - tr['entry_price']) * 100 * n_c - COMMISSION_PER_LEG * 2 * n_c
                pnl = max(pnl, -tr['cost'])
                pnl_today += pnl
                tr['pnl'] = pnl
                tr['exit_date'] = d
                tr['exit_reason'] = exit_reason
                trades.append(tr)
                to_close.append(ticker)

        for t in to_close:
            del open_trades[t]

        if len(open_trades) >= MAX_CONCURRENT:
            equity.append(equity[-1] + pnl_today)
            equity_dates.append(d)
            continue

        # SPY 5d return
        spy_ret_5d = 0.0
        if spy is not None and i >= 5:
            spy_ret_5d = float(spy.iloc[i] / spy.iloc[i - 5] - 1)

        if spy_ret_5d < -0.02:
            # Market-wide drop — not sector-specific
            equity.append(equity[-1] + pnl_today)
            equity_dates.append(d)
            continue

        for ticker in SECTOR_ETFS:
            if ticker not in price_piv.columns:
                continue
            if ticker in open_trades or len(open_trades) >= MAX_CONCURRENT:
                break

            if i < 5:
                continue

            etf_ret_5d = float(price_piv[ticker].iloc[i] / price_piv[ticker].iloc[i - 5] - 1)
            if etf_ret_5d >= -0.05:
                continue  # Not weak enough

            S = float(price_piv.loc[d, ticker])
            K = S
            T = 14 / 365.0
            entry_prem = bs_call(S, K, T, RISK_FREE, sigma)
            if entry_prem <= 0.01:
                continue

            cost_per_contract = entry_prem * 100
            if cost_per_contract <= 0 or np.isnan(cost_per_contract):
                continue
            n_contracts = max(1, int(MAX_PREMIUM / cost_per_contract))
            total_cost = cost_per_contract * n_contracts + COMMISSION_PER_LEG * n_contracts

            if total_cost > equity[-1] * 0.40:
                n_contracts = max(1, int(equity[-1] * 0.40 / cost_per_contract))
                total_cost = cost_per_contract * n_contracts + COMMISSION_PER_LEG * n_contracts

            open_trades[ticker] = dict(
                strategy='D_Contrarian_Call',
                ticker=ticker,
                entry_date=d,
                exit_date=None,
                S=S, K=K, dte=14, sigma=sigma,
                entry_price=entry_prem,
                n_contracts=n_contracts,
                cost=total_cost,
                pnl=0.0,
                direction='call',
                etf_ret_5d=etf_ret_5d,
                spy_ret_5d=spy_ret_5d,
            )

        equity.append(max(0, equity[-1] + pnl_today))
        equity_dates.append(d)

    for ticker, tr in open_trades.items():
        d = dates[-1]
        S = float(price_piv.loc[d, ticker]) if d in price_piv.index and ticker in price_piv.columns else tr['S']
        exit_prem = max(S - tr['K'], 0.0)
        n_c = tr['n_contracts']
        pnl = (exit_prem - tr['entry_price']) * 100 * n_c - COMMISSION_PER_LEG * 2 * n_c
        pnl = max(pnl, -tr['cost'])
        tr['pnl'] = pnl
        tr['exit_date'] = d
        tr['exit_reason'] = 'end'
        trades.append(tr)

    return pd.DataFrame(trades) if trades else None, equity, equity_dates


# ─── STRATEGY E: VIX SPIKE SECTOR PUTS ───────────────────────────────────────
def strategy_e(price_piv, vix_series):
    """
    VIX spikes >30% in 5 days → fear event. Buy puts on the sector
    that dropped LEAST (defensive laggard expected to catch down).
    14 DTE. Exit +50% or 7d or -30%.
    """
    trades = []
    equity = [STARTING_CAPITAL]
    equity_dates = [price_piv.index[10]]
    open_trades = {}
    dates = price_piv.index.tolist()
    last_vix_signal_i = -20  # cooldown

    for i, d in enumerate(dates):
        if i < 10:
            continue

        vix = float(vix_series.loc[d]) if d in vix_series.index else 20.0
        vix_5d_ago_idx = max(0, i - 5)
        d5 = dates[vix_5d_ago_idx]
        vix_5d_ago = float(vix_series.loc[d5]) if d5 in vix_series.index else vix
        vix_spike = (vix - vix_5d_ago) / (vix_5d_ago + 1e-9)

        sigma = max(vix / 100.0, 0.10)

        pnl_today = 0.0
        to_close = []
        for ticker, tr in open_trades.items():
            days_held = (d - tr['entry_date']).days
            S = float(price_piv.loc[d, ticker]) if d in price_piv.index and ticker in price_piv.columns else tr['S']
            T_rem = max(0, (tr['dte'] - days_held)) / 365.0
            cur_val = bs_put(S, tr['K'], T_rem, RISK_FREE, tr['sigma'])
            prem_chg = (cur_val - tr['entry_price']) / (tr['entry_price'] + 1e-9)

            exit_reason = None
            if prem_chg >= 0.50:
                exit_reason = '+50%'
            elif prem_chg <= -0.30:
                exit_reason = '-30% stop'
            elif days_held >= 7:
                exit_reason = '7d max'

            if exit_reason:
                exit_prem = max(cur_val, tr['K'] - S)
                n_c = tr['n_contracts']
                pnl = (exit_prem - tr['entry_price']) * 100 * n_c - COMMISSION_PER_LEG * 2 * n_c
                pnl = max(pnl, -tr['cost'])
                pnl_today += pnl
                tr['pnl'] = pnl
                tr['exit_date'] = d
                tr['exit_reason'] = exit_reason
                trades.append(tr)
                to_close.append(ticker)

        for t in to_close:
            del open_trades[t]

        # Signal: VIX spike >30% in 5 days
        if vix_spike >= 0.30 and (i - last_vix_signal_i) >= 15 and len(open_trades) < MAX_CONCURRENT:
            # Find sector that dropped least (will catch down)
            rets_5d = {}
            for ticker in SECTOR_ETFS:
                if ticker in price_piv.columns and i >= 5:
                    r = float(price_piv[ticker].iloc[i] / price_piv[ticker].iloc[i - 5] - 1)
                    rets_5d[ticker] = r

            if rets_5d:
                # "dropped least" = highest return over 5d (most defensive)
                defensive = max(rets_5d, key=rets_5d.get)
                S = float(price_piv.loc[d, defensive]) if defensive in price_piv.columns else None

                if S and S > 0 and defensive not in open_trades:
                    K = S
                    T = 14 / 365.0
                    entry_prem = bs_put(S, K, T, RISK_FREE, sigma)
                    if entry_prem > 0.01:
                        cost_per_contract = entry_prem * 100
                        n_contracts = max(1, int(MAX_PREMIUM / cost_per_contract))
                        total_cost = cost_per_contract * n_contracts + COMMISSION_PER_LEG * n_contracts

                        if total_cost <= equity[-1] * 0.40:
                            open_trades[defensive] = dict(
                                strategy='E_VIX_Spike_Put',
                                ticker=defensive,
                                entry_date=d,
                                exit_date=None,
                                S=S, K=K, dte=14, sigma=sigma,
                                entry_price=entry_prem,
                                n_contracts=n_contracts,
                                cost=total_cost,
                                pnl=0.0,
                                direction='put',
                                vix_spike_pct=vix_spike,
                                ret_5d=rets_5d[defensive],
                            )
                            last_vix_signal_i = i

        equity.append(max(0, equity[-1] + pnl_today))
        equity_dates.append(d)

    for ticker, tr in open_trades.items():
        d = dates[-1]
        S = float(price_piv.loc[d, ticker]) if d in price_piv.index and ticker in price_piv.columns else tr['S']
        exit_prem = max(tr['K'] - S, 0.0)
        n_c = tr['n_contracts']
        pnl = (exit_prem - tr['entry_price']) * 100 * n_c - COMMISSION_PER_LEG * 2 * n_c
        pnl = max(pnl, -tr['cost'])
        tr['pnl'] = pnl
        tr['exit_date'] = d
        tr['exit_reason'] = 'end'
        trades.append(tr)

    return pd.DataFrame(trades) if trades else None, equity, equity_dates


# ─── STRATEGY F: SECTOR PAIR TRADE ────────────────────────────────────────────
def strategy_f(price_piv, vix_series):
    """
    Long calls on strongest sector + long puts on weakest sector.
    30 DTE. Exit when relative performance reverses or 14d max hold.
    """
    trades = []
    equity = [STARTING_CAPITAL]
    equity_dates = [price_piv.index[30]]
    open_pairs = []  # list of (call_trade, put_trade)
    dates = price_piv.index.tolist()

    rebal_every = 14  # check ranking every 2 weeks
    last_rebal_i = 30

    for i, d in enumerate(dates):
        if i < 30:
            continue

        vix = float(vix_series.loc[d]) if d in vix_series.index else 20.0
        sigma = max(vix / 100.0, 0.10)

        pnl_today = 0.0

        # Close pairs that expired or reversed
        still_open = []
        for pair in open_pairs:
            call_tr, put_tr = pair
            days_held = (d - call_tr['entry_date']).days

            call_ticker = call_tr['ticker']
            put_ticker = put_tr['ticker']

            call_S = float(price_piv.loc[d, call_ticker]) if call_ticker in price_piv.columns else call_tr['S']
            put_S = float(price_piv.loc[d, put_ticker]) if put_ticker in price_piv.columns else put_tr['S']

            # Check for relative performance reversal
            call_ret = (call_S - call_tr['S']) / call_tr['S']
            put_ret = (put_S - put_tr['S']) / put_tr['S']
            reversal = (put_ret > call_ret + 0.02)  # weaker sector outperforming

            should_exit = days_held >= 14 or reversal

            if should_exit:
                T_rem_c = max(0, (30 - days_held)) / 365.0
                T_rem_p = T_rem_c

                cv = bs_call(call_S, call_tr['K'], T_rem_c, RISK_FREE, call_tr['sigma'])
                pv = bs_put(put_S, put_tr['K'], T_rem_p, RISK_FREE, put_tr['sigma'])

                c_pnl = (max(cv, call_S - call_tr['K']) - call_tr['entry_price']) * 100 * call_tr['n_contracts'] - COMMISSION_PER_LEG * 2 * call_tr['n_contracts']
                p_pnl = (max(pv, put_tr['K'] - put_S) - put_tr['entry_price']) * 100 * put_tr['n_contracts'] - COMMISSION_PER_LEG * 2 * put_tr['n_contracts']

                c_pnl = max(c_pnl, -call_tr['cost'])
                p_pnl = max(p_pnl, -put_tr['cost'])

                total_pnl = c_pnl + p_pnl
                pnl_today += total_pnl

                call_tr['pnl'] = c_pnl
                call_tr['exit_date'] = d
                call_tr['exit_reason'] = 'reversal' if reversal else '14d_max'
                put_tr['pnl'] = p_pnl
                put_tr['exit_date'] = d
                put_tr['exit_reason'] = call_tr['exit_reason']

                trades.append(call_tr)
                trades.append(put_tr)
            else:
                still_open.append(pair)

        open_pairs = still_open

        # New pair entry every rebal_every days (if no active pair)
        if len(open_pairs) == 0 and (i - last_rebal_i) >= rebal_every:
            moms = {}
            for t in SECTOR_ETFS:
                if t in price_piv.columns and i >= 20:
                    r = float(price_piv[t].iloc[i] / price_piv[t].iloc[i - 20] - 1)
                    moms[t] = r

            if len(moms) >= 2:
                strongest = max(moms, key=moms.get)
                weakest = min(moms, key=moms.get)

                if strongest != weakest:
                    S_c = float(price_piv.loc[d, strongest]) if strongest in price_piv.columns else None
                    S_p = float(price_piv.loc[d, weakest]) if weakest in price_piv.columns else None

                    if S_c and S_p and S_c > 0 and S_p > 0:
                        T = 30 / 365.0
                        ep_c = bs_call(S_c, S_c, T, RISK_FREE, sigma)
                        ep_p = bs_put(S_p, S_p, T, RISK_FREE, sigma)

                        if ep_c > 0.01 and ep_p > 0.01:
                            budget_each = MAX_PREMIUM  # $100 per leg
                            nc = max(1, int(budget_each / (ep_c * 100)))
                            np_ = max(1, int(budget_each / (ep_p * 100)))
                            cost_c = ep_c * 100 * nc + COMMISSION_PER_LEG * nc
                            cost_p = ep_p * 100 * np_ + COMMISSION_PER_LEG * np_

                            if (cost_c + cost_p) <= equity[-1] * 0.60:
                                call_tr = dict(
                                    strategy='F_Pair_Call',
                                    ticker=strongest,
                                    entry_date=d, exit_date=None,
                                    S=S_c, K=S_c, dte=30, sigma=sigma,
                                    entry_price=ep_c,
                                    n_contracts=nc, cost=cost_c,
                                    pnl=0.0, direction='call',
                                    mom_rank='strongest',
                                )
                                put_tr = dict(
                                    strategy='F_Pair_Put',
                                    ticker=weakest,
                                    entry_date=d, exit_date=None,
                                    S=S_p, K=S_p, dte=30, sigma=sigma,
                                    entry_price=ep_p,
                                    n_contracts=np_, cost=cost_p,
                                    pnl=0.0, direction='put',
                                    mom_rank='weakest',
                                )
                                open_pairs.append((call_tr, put_tr))
                                last_rebal_i = i

        equity.append(max(0, equity[-1] + pnl_today))
        equity_dates.append(d)

    # Close remaining
    for pair in open_pairs:
        call_tr, put_tr = pair
        d = dates[-1]
        call_S = float(price_piv.loc[d, call_tr['ticker']]) if call_tr['ticker'] in price_piv.columns else call_tr['S']
        put_S = float(price_piv.loc[d, put_tr['ticker']]) if put_tr['ticker'] in price_piv.columns else put_tr['S']

        cv = max(call_S - call_tr['K'], 0)
        pv = max(put_tr['K'] - put_S, 0)

        c_pnl = (cv - call_tr['entry_price']) * 100 * call_tr['n_contracts'] - COMMISSION_PER_LEG * 2 * call_tr['n_contracts']
        p_pnl = (pv - put_tr['entry_price']) * 100 * put_tr['n_contracts'] - COMMISSION_PER_LEG * 2 * put_tr['n_contracts']

        c_pnl = max(c_pnl, -call_tr['cost'])
        p_pnl = max(p_pnl, -put_tr['cost'])

        call_tr['pnl'] = c_pnl
        call_tr['exit_date'] = d
        call_tr['exit_reason'] = 'end'
        put_tr['pnl'] = p_pnl
        put_tr['exit_date'] = d
        put_tr['exit_reason'] = 'end'
        trades.append(call_tr)
        trades.append(put_tr)

    return pd.DataFrame(trades) if trades else None, equity, equity_dates


# ─── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("ETF MOMENTUM OPTIONS v2 — Six Strategy Research Backtest")
    print("=" * 70)

    # Load data
    data = load_data()

    # Build price pivot
    print("\nBuilding price pivot...")
    all_tickers_in_data = data['ticker'].unique().tolist()
    available = [t for t in ALL_TICKERS if t in all_tickers_in_data]
    price_piv = build_pivot(data, available)

    # VIX series
    vix_col = '^VIX' if '^VIX' in price_piv.columns else 'VIX'
    if vix_col in price_piv.columns:
        vix_series = price_piv[vix_col].fillna(method='ffill').fillna(20.0)
    else:
        # Fallback: flat VIX at 20
        print("WARNING: VIX data not found, using flat 20.0")
        vix_series = pd.Series(20.0, index=price_piv.index)

    # SPY returns for regime classification
    if 'SPY' in price_piv.columns:
        spy_rets = price_piv['SPY'].pct_change().fillna(0)
    else:
        spy_rets = pd.Series(0.0, index=price_piv.index)

    print(f"Date range: {price_piv.index[0].date()} to {price_piv.index[-1].date()}")
    print(f"Available tickers: {[c for c in price_piv.columns]}")

    # ─── RUN STRATEGIES ──────────────────────────────────────────────────────
    strategy_fns = [
        ('A — Momentum Breakout Calls', strategy_a),
        ('B — Mean Reversion Puts',     strategy_b),
        ('C — Rotation Momentum Calls', strategy_c),
        ('D — Contrarian Bounce Calls', strategy_d),
        ('E — VIX Spike Sector Puts',   strategy_e),
        ('F — Sector Pair Trade',       strategy_f),
    ]

    all_results = {}
    mlflow_run_id = None

    if MLFLOW_AVAILABLE:
        try:
            mlflow.set_tracking_uri("http://localhost:5000")
            mlflow.set_experiment("etf_momentum_options_v2")
            run = mlflow.start_run(run_name="all_six_strategies")
            mlflow_run_id = run.info.run_id
            print(f"\nMLflow run: {mlflow_run_id}")
        except Exception as e:
            print(f"MLflow unavailable: {e}")
            MLFLOW_AVAILABLE_local = False

    print()
    for name, fn in strategy_fns:
        print(f"Running {name}...")
        try:
            trades_df, equity, eq_dates = fn(price_piv, vix_series)

            if trades_df is not None:
                trades_df['entry_date'] = pd.to_datetime(trades_df['entry_date'])
                trades_df['exit_date'] = pd.to_datetime(trades_df['exit_date'])

            metrics = calc_metrics(trades_df, equity) if trades_df is not None else None
            val = validate(name, trades_df, equity, spy_rets) if trades_df is not None else None

            all_results[name] = dict(
                trades=trades_df,
                equity=equity,
                eq_dates=eq_dates,
                metrics=metrics,
                validation=val,
            )

            if metrics:
                print(f"  N={metrics['n']:4d}  WR={metrics['wr']:.1%}  PF={metrics['pf']:.2f}"
                      f"  Sharpe={metrics['sharpe']:.2f}  MDD={metrics['mdd']:.1%}"
                      f"  CAGR={metrics['cagr']:.1%}  Gates={val['gates_passed']}/5")
            else:
                print(f"  No trades generated")

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            all_results[name] = dict(trades=None, equity=[STARTING_CAPITAL], metrics=None, validation=None)

    # ─── VALIDATION DETAIL ───────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("VALIDATION DETAIL (5-gate)")
    print("=" * 70)

    for name, res in all_results.items():
        val = res.get('validation')
        metrics = res.get('metrics')
        print(f"\n{name}")
        if val is None:
            print("  No validation (no trades)")
            continue
        for gate, (ok, detail) in val['gates'].items():
            status = "PASS" if ok else "FAIL"
            print(f"  [{status}] {gate}: {detail}")
        print(f"  OVERALL: {val['gates_passed']}/5 gates passed")

    # ─── SUMMARY TABLE ────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)
    print(f"{'Strategy':<35} {'N':>5} {'WR':>6} {'PF':>5} {'Sharpe':>7} {'Sortino':>8} {'MDD':>7} {'CAGR':>7} {'Gates':>6}")
    print("-" * 90)

    for name, res in all_results.items():
        m = res.get('metrics')
        v = res.get('validation')
        if m is None:
            print(f"{name:<35} {'—':>5} {'—':>6} {'—':>5} {'—':>7} {'—':>8} {'—':>7} {'—':>7} {'—':>6}")
        else:
            gates_str = f"{v['gates_passed']}/5" if v else "—"
            sortino = f"{m['sortino']:.2f}" if not np.isnan(m['sortino']) else "N/A"
            print(f"{name:<35} {m['n']:>5} {m['wr']:>6.1%} {m['pf']:>5.2f}"
                  f" {m['sharpe']:>7.2f} {sortino:>8} {m['mdd']:>7.1%}"
                  f" {m['cagr']:>7.1%} {gates_str:>6}")

    # ─── BEST STRATEGIES ─────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("WINNERS (3+ gates passed AND N >= 20)")
    print("=" * 70)
    winners = []
    for name, res in all_results.items():
        m = res.get('metrics')
        v = res.get('validation')
        if m and v and v['gates_passed'] >= 3 and m['n'] >= 20:
            winners.append((name, m, v))

    if winners:
        for name, m, v in sorted(winners, key=lambda x: x[1]['sharpe'], reverse=True):
            print(f"\n  {name}")
            print(f"    Sharpe {m['sharpe']:.2f} | Sortino {m['sortino']:.2f} | WR {m['wr']:.1%} | PF {m['pf']:.2f}")
            print(f"    CAGR {m['cagr']:.1%} | MDD {m['mdd']:.1%} | N={m['n']} | Gates {v['gates_passed']}/5")
    else:
        print("  None — all strategies failed 3+ gates.")

    # ─── MLFLOW LOGGING ───────────────────────────────────────────────────────
    if MLFLOW_AVAILABLE and mlflow_run_id:
        try:
            for name, res in all_results.items():
                m = res.get('metrics')
                v = res.get('validation')
                prefix = name.replace(' ', '_').replace('—', '').strip()[:30]
                if m:
                    mlflow.log_metric(f"{prefix}_sharpe", round(m['sharpe'], 4))
                    mlflow.log_metric(f"{prefix}_wr", round(m['wr'], 4))
                    mlflow.log_metric(f"{prefix}_pf", round(min(m['pf'], 99.0), 4))
                    mlflow.log_metric(f"{prefix}_mdd", round(m['mdd'], 4))
                    mlflow.log_metric(f"{prefix}_cagr", round(m['cagr'], 4))
                    mlflow.log_metric(f"{prefix}_n", m['n'])
                if v:
                    mlflow.log_metric(f"{prefix}_gates", v['gates_passed'])
            mlflow.end_run()
            print(f"\nMLflow results logged: run {mlflow_run_id}")
        except Exception as e:
            print(f"MLflow log error: {e}")

    # ─── SAVE RESULTS JSON ────────────────────────────────────────────────────
    out_path = '/home/jupiter/Lvl3Quant/scripts/growth_research/logs/etf_momentum_options_v2_results.json'
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    save_data = {}
    for name, res in all_results.items():
        m = res.get('metrics')
        v = res.get('validation')
        n_trades = len(res['trades']) if res['trades'] is not None else 0

        save_data[name] = {
            'n_trades': n_trades,
            'metrics': {k: (float(v_) if not np.isnan(v_) else None) for k, v_ in m.items()
                        if isinstance(v_, (int, float, np.floating))} if m else None,
            'gates_passed': v['gates_passed'] if v else None,
        }

    with open(out_path, 'w') as f:
        json.dump(save_data, f, indent=2)
    print(f"\nResults saved to {out_path}")

    print("\n" + "=" * 70)
    print("BACKTEST COMPLETE")
    print("=" * 70)

    return all_results


if __name__ == '__main__':
    np.random.seed(42)
    main()
