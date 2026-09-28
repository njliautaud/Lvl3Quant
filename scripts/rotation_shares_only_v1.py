#!/usr/bin/env python3
"""
Rotation Shares-Only Backtest v1
================================
Re-tests the SAME 8 rotation/money-flow signals that failed with options,
but using SHARES instead. This isolates whether signals predict direction
vs options costs killing them.

Strategies A-H on sector ETFs (XLK, XLF, XLE, XLV, XLI, XLC, XLY, XLP, XLU, XLB, XLRE)
OOT: 2022-01-01 to 2026-07-30, Starting capital $645, $0 commission (Robinhood)
"""

import json
import sys
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

# ── Config ──────────────────────────────────────────────────────────────────
SECTORS = ["XLK", "XLF", "XLE", "XLV", "XLI", "XLC", "XLY", "XLP", "XLU", "XLB", "XLRE"]
SPY = "SPY"
START = "2021-06-01"  # extra lookback for indicators
OOT_START = "2022-01-01"
OOT_END = "2026-07-30"
STARTING_CAPITAL = 645.0
TRADE_SIZE = 200.0  # per-trade allocation (used by strategy A; others use full capital or $200)
N_PERMUTATIONS = 1000
SHARPE_GATE = 0.5
REGIME_GAP_GATE = 0.5
MDD_GATE = -0.50
MIN_TRADES = 20
PERM_P_GATE = 0.05

RESULTS_PATH = Path("/home/jupiter/Lvl3Quant/data/rotation_shares_only_v1_results.json")

# ── Data Download ───────────────────────────────────────────────────────────
print("Downloading data...")
tickers = SECTORS + [SPY]
data = {}
for t in tickers:
    df = yf.download(t, start=START, end=OOT_END, progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    data[t] = df
    print(f"  {t}: {len(df)} bars")

# Align dates
common_idx = data[SPY].index
for t in SECTORS:
    common_idx = common_idx.intersection(data[t].index)

for t in tickers:
    data[t] = data[t].loc[common_idx].copy()

spy = data[SPY]
spy_200sma = spy["Close"].rolling(200).mean()

print(f"Common dates: {len(common_idx)}, OOT starts: {OOT_START}")


# ── Utility Functions ───────────────────────────────────────────────────────
def compute_mfi(high, low, close, volume, period=14):
    """Money Flow Index"""
    tp = (high + low + close) / 3
    mf = tp * volume
    delta = tp.diff()
    pos_mf = pd.Series(np.where(delta > 0, mf, 0), index=close.index)
    neg_mf = pd.Series(np.where(delta < 0, mf, 0), index=close.index)
    pos_sum = pos_mf.rolling(period).sum()
    neg_sum = neg_mf.rolling(period).sum()
    mfr = pos_sum / neg_sum.replace(0, np.nan)
    mfi = 100 - (100 / (1 + mfr))
    return mfi


def compute_obv(close, volume):
    """On-Balance Volume"""
    sign = np.sign(close.diff()).fillna(0)
    return (sign * volume).cumsum()


def sharpe(returns, ann=252):
    if len(returns) < 2 or returns.std() == 0:
        return 0.0
    return float(returns.mean() / returns.std() * np.sqrt(ann))


def sortino(returns, ann=252):
    if len(returns) < 2:
        return 0.0
    downside = returns[returns < 0]
    if len(downside) == 0 or downside.std() == 0:
        return float("inf") if returns.mean() > 0 else 0.0
    return float(returns.mean() / downside.std() * np.sqrt(ann))


def max_drawdown(equity_curve):
    peak = equity_curve.cummax()
    dd = (equity_curve - peak) / peak
    return float(dd.min())


def profit_factor(trade_pnls):
    gross_profit = trade_pnls[trade_pnls > 0].sum()
    gross_loss = abs(trade_pnls[trade_pnls < 0].sum())
    if gross_loss == 0:
        return float("inf") if gross_profit > 0 else 0.0
    return float(gross_profit / gross_loss)


def win_rate(trade_pnls):
    if len(trade_pnls) == 0:
        return 0.0
    return float((trade_pnls > 0).sum() / len(trade_pnls))


def payoff_ratio(trade_pnls):
    wins = trade_pnls[trade_pnls > 0]
    losses = trade_pnls[trade_pnls < 0]
    if len(wins) == 0 or len(losses) == 0:
        return 0.0
    return float(wins.mean() / abs(losses.mean()))


def regime_sharpes(daily_returns, spy_close, spy_200sma_series):
    """Split returns into bull/bear based on SPY vs 200-SMA"""
    bull_mask = spy_close > spy_200sma_series
    bear_mask = ~bull_mask
    # Align
    common = daily_returns.index.intersection(bull_mask.index)
    dr = daily_returns.loc[common]
    bm = bull_mask.loc[common]
    berm = bear_mask.loc[common]

    bull_ret = dr[bm]
    bear_ret = dr[berm]
    s_bull = sharpe(bull_ret) if len(bull_ret) > 10 else 0.0
    s_bear = sharpe(bear_ret) if len(bear_ret) > 10 else 0.0
    return s_bull, s_bear


def regime_gap(s_bull, s_bear):
    denom = max(abs(s_bull), abs(s_bear))
    if denom == 0:
        return 0.0
    return abs(s_bull - s_bear) / denom


def permutation_test(daily_returns, trade_log, strategy_fn, n_perms=1000):
    """
    Shuffle sector selection in trade_log and recompute Sharpe.
    Returns p-value = fraction of permuted Sharpes >= actual Sharpe.
    """
    actual_sharpe = sharpe(daily_returns)
    count_better = 0
    for _ in range(n_perms):
        # Shuffle which sector was selected on each trade date
        shuffled_log = trade_log.copy()
        shuffled_log["sector"] = np.random.choice(SECTORS, size=len(shuffled_log))
        # Recompute PnL with shuffled sectors
        perm_pnls = []
        for _, row in shuffled_log.iterrows():
            entry_date = row["entry_date"]
            exit_date = row["exit_date"]
            sector = row["sector"]
            alloc = row.get("allocation", TRADE_SIZE)
            if sector not in data:
                perm_pnls.append(0.0)
                continue
            sdf = data[sector]
            if entry_date not in sdf.index or exit_date not in sdf.index:
                perm_pnls.append(0.0)
                continue
            entry_price = sdf.loc[entry_date, "Close"]
            exit_price = sdf.loc[exit_date, "Close"]
            if pd.isna(entry_price) or pd.isna(exit_price) or entry_price == 0:
                perm_pnls.append(0.0)
                continue
            shares = int(alloc / entry_price)
            if shares == 0:
                perm_pnls.append(0.0)
                continue
            direction = row.get("direction", 1)
            pnl = direction * shares * (exit_price - entry_price)
            perm_pnls.append(pnl)

        # Build daily returns from permuted trades
        perm_equity = build_equity_from_trades(shuffled_log, perm_pnls)
        if len(perm_equity) > 1:
            perm_ret = perm_equity.pct_change().dropna()
            perm_s = sharpe(perm_ret)
        else:
            perm_s = 0.0
        if perm_s >= actual_sharpe:
            count_better += 1

    return count_better / n_perms


def build_equity_from_trades(trade_log, pnls):
    """Build a daily equity curve from trade log and PnLs"""
    equity = STARTING_CAPITAL
    records = []
    for i, (_, row) in enumerate(trade_log.iterrows()):
        entry_date = row["entry_date"]
        exit_date = row["exit_date"]
        equity += pnls[i]
        records.append({"date": exit_date, "equity": equity})
    if not records:
        return pd.Series([STARTING_CAPITAL])
    df = pd.DataFrame(records).drop_duplicates("date", keep="last")
    return pd.Series(df["equity"].values, index=pd.to_datetime(df["date"]))


# ── Strategy Implementations ────────────────────────────────────────────────

def strategy_a_volume_momentum():
    """A) Volume-Confirmed Momentum: Weekly rebalance, 20d return * volume ratio, top sector, hold 1 week"""
    oot_dates = common_idx[common_idx >= OOT_START]
    trades = []

    # Weekly rebalance: every 5 trading days
    i = 0
    while i < len(oot_dates) - 5:
        date = oot_dates[i]
        # Rank sectors by 20d return * (volume / 20d avg volume)
        scores = {}
        for s in SECTORS:
            df = data[s]
            loc = df.index.get_loc(date)
            if loc < 20:
                continue
            ret_20d = df["Close"].iloc[loc] / df["Close"].iloc[loc - 20] - 1
            vol_now = df["Volume"].iloc[loc]
            vol_avg = df["Volume"].iloc[loc - 20:loc].mean()
            if vol_avg == 0:
                continue
            scores[s] = ret_20d * (vol_now / vol_avg)

        if not scores:
            i += 5
            continue

        top_sector = max(scores, key=scores.get)
        entry_date = date
        exit_idx = min(i + 5, len(oot_dates) - 1)
        exit_date = oot_dates[exit_idx]

        entry_price = data[top_sector].loc[entry_date, "Close"]
        exit_price = data[top_sector].loc[exit_date, "Close"]
        shares = int(TRADE_SIZE / entry_price)
        if shares == 0:
            i += 5
            continue
        pnl = shares * (exit_price - entry_price)

        trades.append({
            "sector": top_sector,
            "entry_date": entry_date,
            "exit_date": exit_date,
            "entry_price": float(entry_price),
            "exit_price": float(exit_price),
            "shares": shares,
            "pnl": float(pnl),
            "direction": 1,
            "allocation": TRADE_SIZE,
        })
        i += 5

    return trades


def strategy_b_accumulation():
    """B) Accumulation Detector: OBV rising 10d, price flat/falling (<1%), buy, hold until +2% or 20d max"""
    oot_dates = common_idx[common_idx >= OOT_START]
    trades = []
    in_trade = {}  # sector -> trade info

    for date in oot_dates:
        # Check exits
        for s in list(in_trade.keys()):
            t = in_trade[s]
            current_price = data[s].loc[date, "Close"]
            ret = current_price / t["entry_price"] - 1
            days_held = (date - t["entry_date"]).days
            if ret > 0.02 or days_held >= 20:
                pnl = t["shares"] * (current_price - t["entry_price"])
                trades.append({
                    "sector": s,
                    "entry_date": t["entry_date"],
                    "exit_date": date,
                    "entry_price": float(t["entry_price"]),
                    "exit_price": float(current_price),
                    "shares": t["shares"],
                    "pnl": float(pnl),
                    "direction": 1,
                    "allocation": TRADE_SIZE,
                })
                del in_trade[s]

        # Check entries
        for s in SECTORS:
            if s in in_trade:
                continue
            df = data[s]
            loc = df.index.get_loc(date)
            if loc < 20:
                continue

            obv = compute_obv(df["Close"].iloc[:loc + 1], df["Volume"].iloc[:loc + 1])
            obv_10d_ago = obv.iloc[-11] if len(obv) > 10 else obv.iloc[0]
            obv_now = obv.iloc[-1]
            obv_rising = obv_now > obv_10d_ago

            price_change = df["Close"].iloc[loc] / df["Close"].iloc[loc - 10] - 1
            price_flat_or_down = price_change < 0.01

            if obv_rising and price_flat_or_down:
                entry_price = df["Close"].iloc[loc]
                shares = int(TRADE_SIZE / entry_price)
                if shares == 0:
                    continue
                in_trade[s] = {
                    "entry_date": date,
                    "entry_price": float(entry_price),
                    "shares": shares,
                }

    # Close any remaining
    last_date = oot_dates[-1]
    for s in list(in_trade.keys()):
        t = in_trade[s]
        current_price = data[s].loc[last_date, "Close"]
        pnl = t["shares"] * (current_price - t["entry_price"])
        trades.append({
            "sector": s,
            "entry_date": t["entry_date"],
            "exit_date": last_date,
            "entry_price": float(t["entry_price"]),
            "exit_price": float(current_price),
            "shares": t["shares"],
            "pnl": float(pnl),
            "direction": 1,
            "allocation": TRADE_SIZE,
        })

    return trades


def strategy_c_mfi_divergence():
    """C) Money Flow Divergence: MFI crosses above 20 = buy, crosses below 80 = sell/short. Hold 20 days."""
    oot_dates = common_idx[common_idx >= OOT_START]
    trades = []

    for s in SECTORS:
        df = data[s]
        mfi = compute_mfi(df["High"], df["Low"], df["Close"], df["Volume"], 14)
        mfi_prev = mfi.shift(1)

        for date in oot_dates:
            if date not in mfi.index:
                continue
            loc = df.index.get_loc(date)
            if loc < 20:
                continue

            m_now = mfi.iloc[loc]
            m_prev = mfi_prev.iloc[loc]

            if pd.isna(m_now) or pd.isna(m_prev):
                continue

            direction = 0
            # Buy signal: MFI crosses above 20
            if m_prev <= 20 and m_now > 20:
                direction = 1
            # Sell/short signal: MFI crosses below 80
            elif m_prev >= 80 and m_now < 80:
                direction = -1

            if direction == 0:
                continue

            entry_price = df["Close"].iloc[loc]
            exit_loc = min(loc + 20, len(df) - 1)
            exit_date = df.index[exit_loc]
            exit_price = df["Close"].iloc[exit_loc]

            shares = int(TRADE_SIZE / entry_price)
            if shares == 0:
                continue
            pnl = direction * shares * (exit_price - entry_price)

            trades.append({
                "sector": s,
                "entry_date": date,
                "exit_date": exit_date,
                "entry_price": float(entry_price),
                "exit_price": float(exit_price),
                "shares": shares,
                "pnl": float(pnl),
                "direction": direction,
                "allocation": TRADE_SIZE,
            })

    return trades


def strategy_d_pair_rotation():
    """D) Sector Pair Rotation: cyclical vs defensive ratio vs 20d MA. Buy top of winning group, hold 2 weeks."""
    oot_dates = common_idx[common_idx >= OOT_START]
    cyclicals = ["XLY", "XLI", "XLF"]
    defensives = ["XLU", "XLP", "XLV"]
    trades = []

    # Compute ratio
    cyc_avg = sum(data[s]["Close"] for s in cyclicals) / len(cyclicals)
    def_avg = sum(data[s]["Close"] for s in defensives) / len(defensives)
    ratio = cyc_avg / def_avg
    ratio_ma = ratio.rolling(20).mean()

    i = 0
    while i < len(oot_dates) - 10:
        date = oot_dates[i]
        if date not in ratio.index:
            i += 10
            continue

        r_now = ratio.loc[date]
        r_ma = ratio_ma.loc[date]
        if pd.isna(r_now) or pd.isna(r_ma):
            i += 10
            continue

        if r_now > r_ma:
            # Buy top cyclical
            group = cyclicals
        else:
            # Buy top defensive
            group = defensives

        # Pick strongest in group (20d return)
        best_sector = None
        best_ret = -np.inf
        for s in group:
            df = data[s]
            loc = df.index.get_loc(date)
            if loc < 20:
                continue
            ret = df["Close"].iloc[loc] / df["Close"].iloc[loc - 20] - 1
            if ret > best_ret:
                best_ret = ret
                best_sector = s

        if best_sector is None:
            i += 10
            continue

        entry_price = data[best_sector].loc[date, "Close"]
        exit_idx = min(i + 10, len(oot_dates) - 1)
        exit_date = oot_dates[exit_idx]
        exit_price = data[best_sector].loc[exit_date, "Close"]

        shares = int(TRADE_SIZE / entry_price)
        if shares == 0:
            i += 10
            continue
        pnl = shares * (exit_price - entry_price)

        trades.append({
            "sector": best_sector,
            "entry_date": date,
            "exit_date": exit_date,
            "entry_price": float(entry_price),
            "exit_price": float(exit_price),
            "shares": shares,
            "pnl": float(pnl),
            "direction": 1,
            "allocation": TRADE_SIZE,
        })
        i += 10

    return trades


def strategy_e_dispersion_momentum():
    """E) Dispersion + Momentum: Trade only when sector dispersion > 75th pctile. Buy strongest, sell weakest. Hold 1 week."""
    oot_dates = common_idx[common_idx >= OOT_START]
    trades = []

    # Pre-compute 20d returns for all sectors
    sector_rets = {}
    for s in SECTORS:
        sector_rets[s] = data[s]["Close"].pct_change(20)

    i = 0
    while i < len(oot_dates) - 5:
        date = oot_dates[i]
        loc = data[SPY].index.get_loc(date)

        if loc < 252:
            i += 5
            continue

        # Current dispersion
        rets_today = []
        for s in SECTORS:
            r = sector_rets[s].iloc[loc]
            if not pd.isna(r):
                rets_today.append(r)

        if len(rets_today) < 5:
            i += 5
            continue

        disp_now = np.std(rets_today)

        # Trailing 252 days of dispersion
        trailing_disps = []
        for j in range(max(0, loc - 252), loc):
            day_rets = []
            for s in SECTORS:
                r = sector_rets[s].iloc[j]
                if not pd.isna(r):
                    day_rets.append(r)
            if len(day_rets) >= 5:
                trailing_disps.append(np.std(day_rets))

        if not trailing_disps:
            i += 5
            continue

        pctile_75 = np.percentile(trailing_disps, 75)

        if disp_now <= pctile_75:
            i += 5
            continue

        # Buy strongest, sell weakest
        ret_map = {s: sector_rets[s].iloc[loc] for s in SECTORS if not pd.isna(sector_rets[s].iloc[loc])}
        if len(ret_map) < 2:
            i += 5
            continue

        strongest = max(ret_map, key=ret_map.get)
        weakest = min(ret_map, key=ret_map.get)

        exit_idx = min(i + 5, len(oot_dates) - 1)
        exit_date = oot_dates[exit_idx]

        # Long strongest
        entry_price_long = data[strongest].loc[date, "Close"]
        exit_price_long = data[strongest].loc[exit_date, "Close"]
        shares_long = int((TRADE_SIZE / 2) / entry_price_long)

        # Short weakest
        entry_price_short = data[weakest].loc[date, "Close"]
        exit_price_short = data[weakest].loc[exit_date, "Close"]
        shares_short = int((TRADE_SIZE / 2) / entry_price_short)

        if shares_long > 0:
            pnl_long = shares_long * (exit_price_long - entry_price_long)
            trades.append({
                "sector": strongest,
                "entry_date": date,
                "exit_date": exit_date,
                "entry_price": float(entry_price_long),
                "exit_price": float(exit_price_long),
                "shares": shares_long,
                "pnl": float(pnl_long),
                "direction": 1,
                "allocation": TRADE_SIZE / 2,
            })

        if shares_short > 0:
            pnl_short = shares_short * (entry_price_short - exit_price_short)
            trades.append({
                "sector": weakest,
                "entry_date": date,
                "exit_date": exit_date,
                "entry_price": float(entry_price_short),
                "exit_price": float(exit_price_short),
                "shares": shares_short,
                "pnl": float(pnl_short),
                "direction": -1,
                "allocation": TRADE_SIZE / 2,
            })

        i += 5

    return trades


def strategy_f_confluence():
    """F) Multi-Signal Confluence: Score 0-100 (momentum rank + volume rank + OBV trend + MFI zone + RS vs SPY). Buy >80. Hold 2 weeks."""
    oot_dates = common_idx[common_idx >= OOT_START]
    trades = []

    i = 0
    while i < len(oot_dates) - 10:
        date = oot_dates[i]
        loc = data[SPY].index.get_loc(date)
        if loc < 30:
            i += 10
            continue

        scores = {}
        for s in SECTORS:
            df = data[s]
            sloc = df.index.get_loc(date)
            if sloc < 30:
                continue

            score = 0

            # 1. Momentum rank (20d return) -> 0-20 based on rank among sectors
            # We'll compute all then rank
            pass

        # Compute all components
        mom_rets = {}
        vol_ratios = {}
        obv_trends = {}
        mfi_vals = {}
        rs_vals = {}

        spy_ret_20 = spy["Close"].iloc[loc] / spy["Close"].iloc[loc - 20] - 1

        for s in SECTORS:
            df = data[s]
            sloc = df.index.get_loc(date)
            if sloc < 30:
                continue

            # Momentum: 20d return
            mom_rets[s] = df["Close"].iloc[sloc] / df["Close"].iloc[sloc - 20] - 1

            # Volume ratio
            vol_now = df["Volume"].iloc[sloc]
            vol_avg = df["Volume"].iloc[sloc - 20:sloc].mean()
            vol_ratios[s] = vol_now / vol_avg if vol_avg > 0 else 1.0

            # OBV trend (rising over 10 days = 1, else 0)
            obv = compute_obv(df["Close"].iloc[:sloc + 1], df["Volume"].iloc[:sloc + 1])
            obv_trends[s] = 1 if obv.iloc[-1] > obv.iloc[-11] else 0

            # MFI
            mfi = compute_mfi(df["High"].iloc[:sloc + 1], df["Low"].iloc[:sloc + 1],
                            df["Close"].iloc[:sloc + 1], df["Volume"].iloc[:sloc + 1], 14)
            mfi_vals[s] = mfi.iloc[-1] if not pd.isna(mfi.iloc[-1]) else 50

            # Relative strength vs SPY
            rs_vals[s] = mom_rets[s] - spy_ret_20

        if len(mom_rets) < 3:
            i += 10
            continue

        # Rank and score
        sectors_available = list(mom_rets.keys())
        n = len(sectors_available)

        mom_ranked = sorted(sectors_available, key=lambda x: mom_rets[x])
        vol_ranked = sorted(sectors_available, key=lambda x: vol_ratios[x])
        rs_ranked = sorted(sectors_available, key=lambda x: rs_vals[x])

        for s in sectors_available:
            score = 0
            # Momentum rank: 0-20
            score += (mom_ranked.index(s) / (n - 1)) * 20
            # Volume rank: 0-20
            score += (vol_ranked.index(s) / (n - 1)) * 20
            # OBV trend: 0 or 20
            score += obv_trends[s] * 20
            # MFI zone: oversold (<30) = 20, neutral = 10, overbought (>70) = 0
            mfi_v = mfi_vals[s]
            if mfi_v < 30:
                score += 20
            elif mfi_v < 70:
                score += 10
            # RS vs SPY rank: 0-20
            score += (rs_ranked.index(s) / (n - 1)) * 20

            scores[s] = score

        # Buy sectors with score > 80
        for s, sc in scores.items():
            if sc > 80:
                entry_price = data[s].loc[date, "Close"]
                exit_idx = min(i + 10, len(oot_dates) - 1)
                exit_date = oot_dates[exit_idx]
                exit_price = data[s].loc[exit_date, "Close"]

                shares = int(TRADE_SIZE / entry_price)
                if shares == 0:
                    continue
                pnl = shares * (exit_price - entry_price)

                trades.append({
                    "sector": s,
                    "entry_date": date,
                    "exit_date": exit_date,
                    "entry_price": float(entry_price),
                    "exit_price": float(exit_price),
                    "shares": shares,
                    "pnl": float(pnl),
                    "direction": 1,
                    "allocation": TRADE_SIZE,
                })

        i += 10

    return trades


def strategy_g_cta_momentum():
    """G) CTA Momentum: price > 40d EMA AND volume > 1.5x 20d avg -> buy. Hold until price < 20d EMA."""
    oot_dates = common_idx[common_idx >= OOT_START]
    trades = []
    in_trade = {}

    for date in oot_dates:
        # Check exits first
        for s in list(in_trade.keys()):
            df = data[s]
            loc = df.index.get_loc(date)
            ema_20 = df["Close"].iloc[:loc + 1].ewm(span=20).mean().iloc[-1]
            if df["Close"].iloc[loc] < ema_20:
                exit_price = df["Close"].iloc[loc]
                t = in_trade[s]
                pnl = t["shares"] * (exit_price - t["entry_price"])
                trades.append({
                    "sector": s,
                    "entry_date": t["entry_date"],
                    "exit_date": date,
                    "entry_price": float(t["entry_price"]),
                    "exit_price": float(exit_price),
                    "shares": t["shares"],
                    "pnl": float(pnl),
                    "direction": 1,
                    "allocation": TRADE_SIZE,
                })
                del in_trade[s]

        # Check entries
        for s in SECTORS:
            if s in in_trade:
                continue
            df = data[s]
            loc = df.index.get_loc(date)
            if loc < 40:
                continue

            ema_40 = df["Close"].iloc[:loc + 1].ewm(span=40).mean().iloc[-1]
            price = df["Close"].iloc[loc]
            vol = df["Volume"].iloc[loc]
            vol_avg_20 = df["Volume"].iloc[loc - 20:loc].mean()

            if price > ema_40 and vol_avg_20 > 0 and vol > 1.5 * vol_avg_20:
                shares = int(TRADE_SIZE / price)
                if shares == 0:
                    continue
                in_trade[s] = {
                    "entry_date": date,
                    "entry_price": float(price),
                    "shares": shares,
                }

    # Close remaining
    last_date = oot_dates[-1]
    for s in list(in_trade.keys()):
        t = in_trade[s]
        exit_price = data[s].loc[last_date, "Close"]
        pnl = t["shares"] * (exit_price - t["entry_price"])
        trades.append({
            "sector": s,
            "entry_date": t["entry_date"],
            "exit_date": last_date,
            "entry_price": float(t["entry_price"]),
            "exit_price": float(exit_price),
            "shares": t["shares"],
            "pnl": float(pnl),
            "direction": 1,
            "allocation": TRADE_SIZE,
        })

    return trades


def strategy_h_volume_breakout():
    """H) Volume Surge Breakout: volume > 2x 30d avg AND close in top 25% of daily range -> buy. Hold 5 days."""
    oot_dates = common_idx[common_idx >= OOT_START]
    trades = []

    for s in SECTORS:
        df = data[s]
        for date in oot_dates:
            loc = df.index.get_loc(date)
            if loc < 30:
                continue

            vol = df["Volume"].iloc[loc]
            vol_avg_30 = df["Volume"].iloc[loc - 30:loc].mean()
            if vol_avg_30 == 0:
                continue

            if vol <= 2 * vol_avg_30:
                continue

            # Close in top 25% of daily range
            high = df["High"].iloc[loc]
            low = df["Low"].iloc[loc]
            close = df["Close"].iloc[loc]

            if high == low:
                continue

            pct_range = (close - low) / (high - low)
            if pct_range < 0.75:
                continue

            # Buy and hold 5 days
            entry_price = close
            exit_loc = min(loc + 5, len(df) - 1)
            exit_date = df.index[exit_loc]
            exit_price = df["Close"].iloc[exit_loc]

            shares = int(TRADE_SIZE / entry_price)
            if shares == 0:
                continue
            pnl = shares * (exit_price - entry_price)

            trades.append({
                "sector": s,
                "entry_date": date,
                "exit_date": exit_date,
                "entry_price": float(entry_price),
                "exit_price": float(exit_price),
                "shares": shares,
                "pnl": float(pnl),
                "direction": 1,
                "allocation": TRADE_SIZE,
            })

    return trades


# ── Run All Strategies ──────────────────────────────────────────────────────

STRATEGIES = {
    "A_Volume_Momentum": strategy_a_volume_momentum,
    "B_Accumulation_Detector": strategy_b_accumulation,
    "C_MFI_Divergence": strategy_c_mfi_divergence,
    "D_Sector_Pair_Rotation": strategy_d_pair_rotation,
    "E_Dispersion_Momentum": strategy_e_dispersion_momentum,
    "F_Multi_Confluence": strategy_f_confluence,
    "G_CTA_Momentum": strategy_g_cta_momentum,
    "H_Volume_Breakout": strategy_h_volume_breakout,
}

results = {}

for name, fn in STRATEGIES.items():
    print(f"\n{'='*60}")
    print(f"Running Strategy {name}...")
    print(f"{'='*60}")

    trades = fn()
    n_trades = len(trades)
    print(f"  Trades: {n_trades}")

    if n_trades == 0:
        results[name] = {
            "n_trades": 0,
            "total_return_pct": 0.0,
            "sharpe": 0.0,
            "sortino": 0.0,
            "max_drawdown": 0.0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "payoff_ratio": 0.0,
            "monthly_trade_freq": 0.0,
            "perm_p_value": 1.0,
            "regime_sharpe_bull": 0.0,
            "regime_sharpe_bear": 0.0,
            "regime_gap": 0.0,
            "gates_passed": 0,
            "gates_total": 5,
            "gate_details": {},
            "verdict": "FAIL (no trades)",
        }
        continue

    trade_df = pd.DataFrame(trades)
    trade_df["entry_date"] = pd.to_datetime(trade_df["entry_date"])
    trade_df["exit_date"] = pd.to_datetime(trade_df["exit_date"])
    trade_pnls = trade_df["pnl"].values

    # Build equity curve
    equity_list = [STARTING_CAPITAL]
    for pnl in trade_pnls:
        equity_list.append(equity_list[-1] + pnl)

    # Build daily equity for Sharpe/Sortino
    # Map trades to daily returns
    daily_pnl = {}
    for _, row in trade_df.iterrows():
        ed = row["exit_date"]
        if ed not in daily_pnl:
            daily_pnl[ed] = 0.0
        daily_pnl[ed] += row["pnl"]

    # Create daily equity series
    oot_dates = common_idx[common_idx >= OOT_START]
    equity_ts = pd.Series(index=oot_dates, dtype=float)
    running_eq = STARTING_CAPITAL
    for d in oot_dates:
        if d in daily_pnl:
            running_eq += daily_pnl[d]
        equity_ts[d] = running_eq

    daily_returns = equity_ts.pct_change().dropna()
    daily_returns = daily_returns.replace([np.inf, -np.inf], 0).fillna(0)

    # Metrics
    total_ret = (equity_ts.iloc[-1] / STARTING_CAPITAL - 1) * 100
    s = sharpe(daily_returns)
    so = sortino(daily_returns)
    mdd = max_drawdown(equity_ts)
    wr = win_rate(trade_pnls)
    pf = profit_factor(trade_pnls)
    pr = payoff_ratio(trade_pnls)

    # Monthly trade frequency
    oot_months = (pd.to_datetime(OOT_END) - pd.to_datetime(OOT_START)).days / 30.44
    monthly_freq = n_trades / oot_months if oot_months > 0 else 0

    # Regime analysis
    s_bull, s_bear = regime_sharpes(daily_returns, spy["Close"], spy_200sma)
    rg = regime_gap(s_bull, s_bear)

    # Permutation test
    print(f"  Running permutation test ({N_PERMUTATIONS} iterations)...")
    perm_p = permutation_test(daily_returns, trade_df, fn, N_PERMUTATIONS)

    # Gate checks
    gates = {
        "sharpe_gt_0.5": s > SHARPE_GATE,
        "perm_p_lt_0.05": perm_p < PERM_P_GATE,
        "regime_gap_lt_0.5": rg < REGIME_GAP_GATE,
        "mdd_gt_neg50pct": mdd > MDD_GATE,
        "min_20_trades": n_trades >= MIN_TRADES,
    }
    gates_passed = sum(gates.values())

    verdict = "PASS" if all(gates.values()) else "FAIL"
    fail_reasons = [k for k, v in gates.items() if not v]
    if fail_reasons:
        verdict += f" ({', '.join(fail_reasons)})"

    print(f"  Return: {total_ret:.1f}%  Sharpe: {s:.2f}  Sortino: {so:.2f}")
    print(f"  MDD: {mdd:.1%}  WR: {wr:.1%}  PF: {pf:.2f}  Payoff: {pr:.2f}")
    print(f"  Regime Bull Sharpe: {s_bull:.2f}  Bear: {s_bear:.2f}  Gap: {rg:.2f}")
    print(f"  Perm p-value: {perm_p:.3f}")
    print(f"  Gates: {gates_passed}/5  Verdict: {verdict}")

    results[name] = {
        "n_trades": n_trades,
        "total_return_pct": round(total_ret, 2),
        "sharpe": round(s, 3),
        "sortino": round(so, 3),
        "max_drawdown": round(mdd, 4),
        "win_rate": round(wr, 4),
        "profit_factor": round(pf, 3),
        "payoff_ratio": round(pr, 3),
        "monthly_trade_freq": round(monthly_freq, 1),
        "perm_p_value": round(perm_p, 4),
        "regime_sharpe_bull": round(s_bull, 3),
        "regime_sharpe_bear": round(s_bear, 3),
        "regime_gap": round(rg, 3),
        "gates_passed": gates_passed,
        "gates_total": 5,
        "gate_details": gates,
        "verdict": verdict,
    }

# ── Summary ─────────────────────────────────────────────────────────────────
print(f"\n{'='*80}")
print("ROTATION SHARES-ONLY BACKTEST SUMMARY")
print(f"{'='*80}")
print(f"OOT Period: {OOT_START} to {OOT_END}")
print(f"Starting Capital: ${STARTING_CAPITAL}")
print(f"Commission: $0 (Robinhood shares)")
print(f"{'='*80}")

header = f"{'Strategy':<30} {'Trades':>6} {'Return%':>8} {'Sharpe':>7} {'Sortino':>8} {'MDD':>7} {'WR':>6} {'PF':>6} {'Perm_p':>7} {'Verdict':>10}"
print(header)
print("-" * len(header))

any_pass = False
for name, r in results.items():
    v = "PASS" if r["gates_passed"] == 5 else "FAIL"
    if v == "PASS":
        any_pass = True
    print(f"{name:<30} {r['n_trades']:>6} {r['total_return_pct']:>7.1f}% {r['sharpe']:>7.3f} {r['sortino']:>8.3f} {r['max_drawdown']:>6.1%} {r['win_rate']:>5.1%} {r['profit_factor']:>6.2f} {r['perm_p_value']:>7.3f} {v:>10}")

print(f"\n{'='*80}")
if any_pass:
    print("CONCLUSION: Some signals show directional edge even without options.")
    print("OPTIONS COSTS were the primary drag on those strategies.")
else:
    print("CONCLUSION: ALL 8 signals FAIL the 5-gate validation even with SHARES.")
    print("The rotation thesis has NO directional edge. Options costs were NOT the issue.")
    print("These signals do not predict sector direction. KILL the rotation research line.")
print(f"{'='*80}")

# Save results
output = {
    "metadata": {
        "script": "rotation_shares_only_v1.py",
        "run_time": datetime.now().isoformat(),
        "oot_start": OOT_START,
        "oot_end": OOT_END,
        "starting_capital": STARTING_CAPITAL,
        "commission": 0.0,
        "instrument": "shares (not options)",
        "sectors": SECTORS,
        "n_permutations": N_PERMUTATIONS,
        "gates": {
            "sharpe": f"> {SHARPE_GATE}",
            "perm_p": f"< {PERM_P_GATE}",
            "regime_gap": f"< {REGIME_GAP_GATE}",
            "max_drawdown": f"> {MDD_GATE}",
            "min_trades": f">= {MIN_TRADES}",
        },
    },
    "results": results,
    "summary": {
        "any_signal_passed": any_pass,
        "conclusion": "Some signals have directional edge" if any_pass else "No directional edge found - rotation thesis is dead",
        "strategies_passed": [k for k, v in results.items() if v["gates_passed"] == 5],
        "strategies_failed": [k for k, v in results.items() if v["gates_passed"] < 5],
    },
}

RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(RESULTS_PATH, "w") as f:
    json.dump(output, f, indent=2, default=str)

print(f"\nResults saved to {RESULTS_PATH}")
