#!/usr/bin/env python3
"""
Multi-Period Momentum Backtest with Crash Protection
=====================================================
6 variants combining momentum across multiple timeframes (1m, 3m, 6m, 12m).
Walk-forward OOT: Jan 2022 - Jul 2026.
5-gate validation: Sharpe>0.5, perm p<0.05, regime gap<0.5, MaxDD>-50%, >=20 trades.

Academic basis: Jegadeesh & Titman (1993), Asness (2014).
"""

import json
import warnings
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats

warnings.filterwarnings("ignore")

# ── Configuration ──────────────────────────────────────────────────────────────
ACCOUNT_SIZE = 645.0
SLIPPAGE_PCT = 0.0002  # 0.02%
COMMISSION = 0.0

START_DATE = "2020-06-01"  # buffer for 12-month lookback + 200 SMA
OOT_START = "2022-01-01"
OOT_END = "2026-07-28"

INSTRUMENTS = ["SPY", "QQQ", "IWM"]
SECTOR_ETFS = ["XLK", "XLF", "XLE", "XLV", "XLY"]
VIX = "^VIX"
SPY = "SPY"

N_PERM = 1000
RANDOM_SEED = 42

OUTPUT_PATH = Path("/home/jupiter/Lvl3Quant/data/multi_period_momentum_results.json")


# ── Data Download ──────────────────────────────────────────────────────────────
def download_data():
    """Download all required price data."""
    all_tickers = list(set(INSTRUMENTS + SECTOR_ETFS + [SPY]))
    print(f"Downloading {len(all_tickers)} tickers + VIX...")

    data = yf.download(all_tickers, start=START_DATE, end=OOT_END, auto_adjust=True, progress=False)
    close = data["Close"] if "Close" in data.columns.get_level_values(0) else data

    vix_data = yf.download(VIX, start=START_DATE, end=OOT_END, auto_adjust=True, progress=False)
    vix_close = vix_data["Close"].squeeze() if isinstance(vix_data["Close"], pd.DataFrame) else vix_data["Close"]

    close = close.ffill().dropna(how="all")
    vix_close = vix_close.ffill()

    print(f"  Data range: {close.index[0].date()} to {close.index[-1].date()}")
    print(f"  Tickers with data: {close.shape[1]}")
    return close, vix_close


# ── Helpers ────────────────────────────────────────────────────────────────────
def monthly_rebalance_dates(idx, start, end):
    """Get last trading day of each month in range."""
    mask = (idx >= pd.Timestamp(start)) & (idx <= pd.Timestamp(end))
    filtered = idx[mask]
    monthly = filtered.to_series().groupby([filtered.year, filtered.month]).last()
    return monthly.values


def apply_slippage(price, direction="buy"):
    if direction == "buy":
        return price * (1 + SLIPPAGE_PCT)
    return price * (1 - SLIPPAGE_PCT)


def get_return(prices_df, ticker, date, lookback_days):
    """Compute return over lookback_days trading days ending at date."""
    loc = prices_df.index.get_loc(date)
    if loc < lookback_days:
        return np.nan
    cur = prices_df[ticker].iloc[loc]
    prev = prices_df[ticker].iloc[loc - lookback_days]
    if pd.notna(cur) and pd.notna(prev) and prev > 0:
        return (cur / prev) - 1.0
    return np.nan


def get_return_skip_recent(prices_df, ticker, date, lookback_days, skip_days=21):
    """12-1 momentum: lookback_days return, skipping most recent skip_days."""
    loc = prices_df.index.get_loc(date)
    if loc < lookback_days:
        return np.nan
    cur = prices_df[ticker].iloc[loc - skip_days]
    prev = prices_df[ticker].iloc[loc - lookback_days]
    if pd.notna(cur) and pd.notna(prev) and prev > 0:
        return (cur / prev) - 1.0
    return np.nan


def max_drawdown_recent(prices_df, ticker, date, lookback_days=20):
    """Max drawdown over past lookback_days."""
    loc = prices_df.index.get_loc(date)
    if loc < lookback_days:
        return 0.0
    segment = prices_df[ticker].iloc[max(0, loc - lookback_days):loc + 1]
    if len(segment) < 2:
        return 0.0
    peak = segment.expanding().max()
    dd = (segment - peak) / peak
    return dd.min()


def spy_regime(prices_df, date):
    """Bull if SPY > 200-SMA, else Bear."""
    loc = prices_df.index.get_loc(date)
    if loc < 200:
        return "bull"
    sma200 = prices_df[SPY].iloc[max(0, loc - 199):loc + 1].mean()
    return "bull" if prices_df[SPY].iloc[loc] > sma200 else "bear"


def get_vix(vix_series, date):
    """Get VIX value for a date."""
    if date in vix_series.index:
        val = vix_series.loc[date]
        return float(val.iloc[-1]) if isinstance(val, pd.Series) else float(val)
    prior = vix_series.index[vix_series.index <= date]
    if len(prior) == 0:
        return 18.0
    val = vix_series.loc[prior[-1]]
    return float(val.iloc[-1]) if isinstance(val, pd.Series) else float(val)


# ── Strategy Engines ───────────────────────────────────────────────────────────

def run_variant_A(prices_df, vix_series, account=ACCOUNT_SIZE):
    """
    Classic 12-1 Momentum: Buy QQQ when 12-month return (skipping recent month) > 0.
    Cash otherwise. Monthly rebalance.
    """
    reb_dates = monthly_rebalance_dates(prices_df.index, OOT_START, OOT_END)
    oot_mask = (prices_df.index >= pd.Timestamp(OOT_START)) & (prices_df.index <= pd.Timestamp(OOT_END))
    oot_dates = prices_df.index[oot_mask]

    equity = account
    shares = 0.0
    entry_price = 0.0
    in_market = False
    equity_curve = []
    trades = []
    regime_returns = {"bull": [], "bear": []}
    prev_equity = equity

    for date in oot_dates:
        day_str = str(date.date())
        regime = spy_regime(prices_df, date)

        if date in reb_dates:
            mom_12_1 = get_return_skip_recent(prices_df, "QQQ", date, 252, 21)

            if pd.notna(mom_12_1) and mom_12_1 > 0 and not in_market:
                buy_price = apply_slippage(prices_df["QQQ"].iloc[prices_df.index.get_loc(date)], "buy")
                shares = equity / buy_price
                entry_price = buy_price
                in_market = True
                trades.append({"date": day_str, "ticker": "QQQ", "side": "buy",
                               "price": round(buy_price, 2), "reason": "12-1 momentum > 0"})

            elif pd.notna(mom_12_1) and mom_12_1 <= 0 and in_market:
                sell_price = apply_slippage(prices_df["QQQ"].iloc[prices_df.index.get_loc(date)], "sell")
                pnl = (sell_price - entry_price) * shares
                equity += pnl
                in_market = False
                shares = 0.0
                trades.append({"date": day_str, "ticker": "QQQ", "side": "sell",
                               "price": round(sell_price, 2), "pnl": round(pnl, 2),
                               "reason": "12-1 momentum <= 0"})

        # Mark to market
        if in_market:
            cur_p = prices_df["QQQ"].iloc[prices_df.index.get_loc(date)]
            port_val = equity + (cur_p - entry_price) * shares
        else:
            port_val = equity

        daily_ret = (port_val - prev_equity) / prev_equity if prev_equity > 0 else 0
        regime_returns[regime].append(daily_ret)
        equity_curve.append({"date": day_str, "equity": round(port_val, 2), "regime": regime})
        prev_equity = port_val

    # Final liquidation
    if in_market:
        last_p = prices_df["QQQ"].iloc[prices_df.index.get_loc(oot_dates[-1])]
        sell_price = apply_slippage(last_p, "sell")
        pnl = (sell_price - entry_price) * shares
        equity += pnl
        trades.append({"date": str(oot_dates[-1].date()), "ticker": "QQQ", "side": "sell",
                       "price": round(sell_price, 2), "pnl": round(pnl, 2), "reason": "final liquidation"})
        port_val = equity

    return equity_curve, trades, regime_returns


def run_variant_B(prices_df, vix_series, account=ACCOUNT_SIZE):
    """
    Multi-Timeframe Composite: Score = avg(sign(1m), sign(3m), sign(6m), sign(12m)).
    Buy QQQ when score > 0. Cash when <= 0. Monthly rebalance.
    """
    reb_dates = monthly_rebalance_dates(prices_df.index, OOT_START, OOT_END)
    oot_mask = (prices_df.index >= pd.Timestamp(OOT_START)) & (prices_df.index <= pd.Timestamp(OOT_END))
    oot_dates = prices_df.index[oot_mask]

    equity = account
    shares = 0.0
    entry_price = 0.0
    in_market = False
    equity_curve = []
    trades = []
    regime_returns = {"bull": [], "bear": []}
    prev_equity = equity

    for date in oot_dates:
        day_str = str(date.date())
        regime = spy_regime(prices_df, date)

        if date in reb_dates:
            ret_1m = get_return(prices_df, "QQQ", date, 21)
            ret_3m = get_return(prices_df, "QQQ", date, 63)
            ret_6m = get_return(prices_df, "QQQ", date, 126)
            ret_12m = get_return(prices_df, "QQQ", date, 252)

            rets = [r for r in [ret_1m, ret_3m, ret_6m, ret_12m] if pd.notna(r)]
            if len(rets) >= 2:
                score = np.mean([np.sign(r) for r in rets])

                if score > 0 and not in_market:
                    buy_price = apply_slippage(prices_df["QQQ"].iloc[prices_df.index.get_loc(date)], "buy")
                    shares = equity / buy_price
                    entry_price = buy_price
                    in_market = True
                    trades.append({"date": day_str, "ticker": "QQQ", "side": "buy",
                                   "price": round(buy_price, 2),
                                   "reason": f"composite score {score:.2f} > 0"})

                elif score <= 0 and in_market:
                    sell_price = apply_slippage(prices_df["QQQ"].iloc[prices_df.index.get_loc(date)], "sell")
                    pnl = (sell_price - entry_price) * shares
                    equity += pnl
                    in_market = False
                    shares = 0.0
                    trades.append({"date": day_str, "ticker": "QQQ", "side": "sell",
                                   "price": round(sell_price, 2), "pnl": round(pnl, 2),
                                   "reason": f"composite score {score:.2f} <= 0"})

        if in_market:
            cur_p = prices_df["QQQ"].iloc[prices_df.index.get_loc(date)]
            port_val = equity + (cur_p - entry_price) * shares
        else:
            port_val = equity

        daily_ret = (port_val - prev_equity) / prev_equity if prev_equity > 0 else 0
        regime_returns[regime].append(daily_ret)
        equity_curve.append({"date": day_str, "equity": round(port_val, 2), "regime": regime})
        prev_equity = port_val

    if in_market:
        last_p = prices_df["QQQ"].iloc[prices_df.index.get_loc(oot_dates[-1])]
        sell_price = apply_slippage(last_p, "sell")
        pnl = (sell_price - entry_price) * shares
        equity += pnl
        trades.append({"date": str(oot_dates[-1].date()), "ticker": "QQQ", "side": "sell",
                       "price": round(sell_price, 2), "pnl": round(pnl, 2), "reason": "final liquidation"})
        port_val = equity

    return equity_curve, trades, regime_returns


def run_variant_C(prices_df, vix_series, account=ACCOUNT_SIZE):
    """
    Momentum + Crash Filter: Same as B, but 20-day max drawdown < -5% triggers CASH for 1 month.
    """
    reb_dates = monthly_rebalance_dates(prices_df.index, OOT_START, OOT_END)
    oot_mask = (prices_df.index >= pd.Timestamp(OOT_START)) & (prices_df.index <= pd.Timestamp(OOT_END))
    oot_dates = prices_df.index[oot_mask]

    equity = account
    shares = 0.0
    entry_price = 0.0
    in_market = False
    cash_until = None  # date until which we stay in cash
    equity_curve = []
    trades = []
    regime_returns = {"bull": [], "bear": []}
    prev_equity = equity

    for date in oot_dates:
        day_str = str(date.date())
        regime = spy_regime(prices_df, date)

        # Check crash filter
        dd_20 = max_drawdown_recent(prices_df, "QQQ", date, 20)
        if dd_20 < -0.05 and in_market:
            sell_price = apply_slippage(prices_df["QQQ"].iloc[prices_df.index.get_loc(date)], "sell")
            pnl = (sell_price - entry_price) * shares
            equity += pnl
            in_market = False
            shares = 0.0
            cash_until = date + pd.DateOffset(months=1)
            trades.append({"date": day_str, "ticker": "QQQ", "side": "sell",
                           "price": round(sell_price, 2), "pnl": round(pnl, 2),
                           "reason": f"crash filter (DD {dd_20:.1%})"})

        # Check if cash lockout expired
        in_cash_lockout = cash_until is not None and date < cash_until

        if date in reb_dates and not in_cash_lockout:
            ret_1m = get_return(prices_df, "QQQ", date, 21)
            ret_3m = get_return(prices_df, "QQQ", date, 63)
            ret_6m = get_return(prices_df, "QQQ", date, 126)
            ret_12m = get_return(prices_df, "QQQ", date, 252)

            rets = [r for r in [ret_1m, ret_3m, ret_6m, ret_12m] if pd.notna(r)]
            if len(rets) >= 2:
                score = np.mean([np.sign(r) for r in rets])

                if score > 0 and not in_market:
                    buy_price = apply_slippage(prices_df["QQQ"].iloc[prices_df.index.get_loc(date)], "buy")
                    shares = equity / buy_price
                    entry_price = buy_price
                    in_market = True
                    trades.append({"date": day_str, "ticker": "QQQ", "side": "buy",
                                   "price": round(buy_price, 2),
                                   "reason": f"composite + crash clear, score {score:.2f}"})

                elif score <= 0 and in_market:
                    sell_price = apply_slippage(prices_df["QQQ"].iloc[prices_df.index.get_loc(date)], "sell")
                    pnl = (sell_price - entry_price) * shares
                    equity += pnl
                    in_market = False
                    shares = 0.0
                    trades.append({"date": day_str, "ticker": "QQQ", "side": "sell",
                                   "price": round(sell_price, 2), "pnl": round(pnl, 2),
                                   "reason": f"composite score {score:.2f} <= 0"})

        if in_market:
            cur_p = prices_df["QQQ"].iloc[prices_df.index.get_loc(date)]
            port_val = equity + (cur_p - entry_price) * shares
        else:
            port_val = equity

        daily_ret = (port_val - prev_equity) / prev_equity if prev_equity > 0 else 0
        regime_returns[regime].append(daily_ret)
        equity_curve.append({"date": day_str, "equity": round(port_val, 2), "regime": regime})
        prev_equity = port_val

    if in_market:
        last_p = prices_df["QQQ"].iloc[prices_df.index.get_loc(oot_dates[-1])]
        sell_price = apply_slippage(last_p, "sell")
        pnl = (sell_price - entry_price) * shares
        equity += pnl
        trades.append({"date": str(oot_dates[-1].date()), "ticker": "QQQ", "side": "sell",
                       "price": round(sell_price, 2), "pnl": round(pnl, 2), "reason": "final liquidation"})
        port_val = equity

    return equity_curve, trades, regime_returns


def run_variant_D(prices_df, vix_series, account=ACCOUNT_SIZE):
    """
    Relative Momentum: Each month, buy strongest of QQQ/IWM/SPY by 6-month return.
    If ALL negative, go to cash.
    """
    reb_dates = monthly_rebalance_dates(prices_df.index, OOT_START, OOT_END)
    oot_mask = (prices_df.index >= pd.Timestamp(OOT_START)) & (prices_df.index <= pd.Timestamp(OOT_END))
    oot_dates = prices_df.index[oot_mask]

    equity = account
    shares = 0.0
    entry_price = 0.0
    holding = None  # ticker currently held
    equity_curve = []
    trades = []
    regime_returns = {"bull": [], "bear": []}
    prev_equity = equity

    for date in oot_dates:
        day_str = str(date.date())
        regime = spy_regime(prices_df, date)

        if date in reb_dates:
            rets = {}
            for t in INSTRUMENTS:
                r = get_return(prices_df, t, date, 126)
                if pd.notna(r):
                    rets[t] = r

            if len(rets) == 0:
                best = None
            else:
                best_ticker = max(rets, key=rets.get)
                best = best_ticker if rets[best_ticker] > 0 else None

            # Sell current if switching or going to cash
            if holding is not None and (best != holding):
                sell_price = apply_slippage(prices_df[holding].iloc[prices_df.index.get_loc(date)], "sell")
                pnl = (sell_price - entry_price) * shares
                equity += pnl
                trades.append({"date": day_str, "ticker": holding, "side": "sell",
                               "price": round(sell_price, 2), "pnl": round(pnl, 2),
                               "reason": f"switch from {holding} to {best or 'cash'}"})
                holding = None
                shares = 0.0

            # Buy best if not already holding
            if best is not None and holding is None:
                buy_price = apply_slippage(prices_df[best].iloc[prices_df.index.get_loc(date)], "buy")
                shares = equity / buy_price
                entry_price = buy_price
                holding = best
                trades.append({"date": day_str, "ticker": best, "side": "buy",
                               "price": round(buy_price, 2),
                               "reason": f"strongest 6m momentum ({rets[best]:.1%})"})

        if holding is not None:
            cur_p = prices_df[holding].iloc[prices_df.index.get_loc(date)]
            port_val = equity + (cur_p - entry_price) * shares
        else:
            port_val = equity

        daily_ret = (port_val - prev_equity) / prev_equity if prev_equity > 0 else 0
        regime_returns[regime].append(daily_ret)
        equity_curve.append({"date": day_str, "equity": round(port_val, 2), "regime": regime})
        prev_equity = port_val

    if holding is not None:
        last_p = prices_df[holding].iloc[prices_df.index.get_loc(oot_dates[-1])]
        sell_price = apply_slippage(last_p, "sell")
        pnl = (sell_price - entry_price) * shares
        equity += pnl
        trades.append({"date": str(oot_dates[-1].date()), "ticker": holding, "side": "sell",
                       "price": round(sell_price, 2), "pnl": round(pnl, 2), "reason": "final liquidation"})

    return equity_curve, trades, regime_returns


def run_variant_E(prices_df, vix_series, account=ACCOUNT_SIZE):
    """
    Sector Momentum + Crash: Buy top-2 sector ETFs by 3-month momentum.
    If SPY 1-month return < -5%, go to cash for 1 month.
    """
    reb_dates = monthly_rebalance_dates(prices_df.index, OOT_START, OOT_END)
    oot_mask = (prices_df.index >= pd.Timestamp(OOT_START)) & (prices_df.index <= pd.Timestamp(OOT_END))
    oot_dates = prices_df.index[oot_mask]

    equity = account
    positions = {}  # ticker -> {shares, entry_price}
    cash_until = None
    equity_curve = []
    trades = []
    regime_returns = {"bull": [], "bear": []}
    prev_equity = equity

    for date in oot_dates:
        day_str = str(date.date())
        regime = spy_regime(prices_df, date)

        # Check crash filter: SPY 1-month return < -5%
        spy_1m = get_return(prices_df, SPY, date, 21)
        if pd.notna(spy_1m) and spy_1m < -0.05 and len(positions) > 0:
            for t, pos in list(positions.items()):
                sell_price = apply_slippage(prices_df[t].iloc[prices_df.index.get_loc(date)], "sell")
                pnl = (sell_price - pos["entry_price"]) * pos["shares"]
                equity += pnl
                trades.append({"date": day_str, "ticker": t, "side": "sell",
                               "price": round(sell_price, 2), "pnl": round(pnl, 2),
                               "reason": f"SPY crash filter ({spy_1m:.1%})"})
            positions = {}
            cash_until = date + pd.DateOffset(months=1)

        in_cash_lockout = cash_until is not None and date < cash_until

        if date in reb_dates and not in_cash_lockout:
            # Rank sectors by 3-month momentum
            sector_mom = {}
            for t in SECTOR_ETFS:
                r = get_return(prices_df, t, date, 63)
                if pd.notna(r):
                    sector_mom[t] = r

            if len(sector_mom) >= 2:
                ranked = sorted(sector_mom.items(), key=lambda x: x[1], reverse=True)
                new_picks = [t for t, _ in ranked[:2]]

                # Sell positions not in new picks
                for t in list(positions.keys()):
                    if t not in new_picks:
                        sell_price = apply_slippage(prices_df[t].iloc[prices_df.index.get_loc(date)], "sell")
                        pnl = (sell_price - positions[t]["entry_price"]) * positions[t]["shares"]
                        equity += pnl
                        trades.append({"date": day_str, "ticker": t, "side": "sell",
                                       "price": round(sell_price, 2), "pnl": round(pnl, 2),
                                       "reason": "sector rebalance"})
                        del positions[t]

                # Compute portfolio value for sizing
                port_val = equity
                for t, pos in positions.items():
                    cur_p = prices_df[t].iloc[prices_df.index.get_loc(date)]
                    if pd.notna(cur_p):
                        port_val += (cur_p - pos["entry_price"]) * pos["shares"]

                # Buy new picks
                for t in new_picks:
                    if t in positions:
                        continue
                    buy_price = apply_slippage(prices_df[t].iloc[prices_df.index.get_loc(date)], "buy")
                    if pd.isna(buy_price) or buy_price <= 0:
                        continue
                    alloc = port_val * 0.5  # equal weight top 2
                    shares_buy = alloc / buy_price
                    positions[t] = {"shares": shares_buy, "entry_price": buy_price}
                    trades.append({"date": day_str, "ticker": t, "side": "buy",
                                   "price": round(buy_price, 2),
                                   "reason": f"top-2 sector mom ({sector_mom[t]:.1%})"})

        # Mark to market
        port_val = equity
        for t, pos in positions.items():
            cur_p = prices_df[t].iloc[prices_df.index.get_loc(date)]
            if pd.notna(cur_p):
                port_val += (cur_p - pos["entry_price"]) * pos["shares"]

        daily_ret = (port_val - prev_equity) / prev_equity if prev_equity > 0 else 0
        regime_returns[regime].append(daily_ret)
        equity_curve.append({"date": day_str, "equity": round(port_val, 2), "regime": regime})
        prev_equity = port_val

    # Final liquidation
    for t, pos in positions.items():
        last_p = prices_df[t].iloc[prices_df.index.get_loc(oot_dates[-1])]
        sell_price = apply_slippage(last_p, "sell")
        pnl = (sell_price - pos["entry_price"]) * pos["shares"]
        equity += pnl
        trades.append({"date": str(oot_dates[-1].date()), "ticker": t, "side": "sell",
                       "price": round(sell_price, 2), "pnl": round(pnl, 2), "reason": "final liquidation"})

    return equity_curve, trades, regime_returns


def run_variant_F(prices_df, vix_series, account=ACCOUNT_SIZE):
    """
    Adaptive Period: In low-VIX (<18) use 1m momentum. In high-VIX (>22) use 12m.
    In between, use 3m. Buy QQQ when chosen-period return > 0. Cash otherwise.
    """
    reb_dates = monthly_rebalance_dates(prices_df.index, OOT_START, OOT_END)
    oot_mask = (prices_df.index >= pd.Timestamp(OOT_START)) & (prices_df.index <= pd.Timestamp(OOT_END))
    oot_dates = prices_df.index[oot_mask]

    equity = account
    shares = 0.0
    entry_price = 0.0
    in_market = False
    equity_curve = []
    trades = []
    regime_returns = {"bull": [], "bear": []}
    prev_equity = equity

    for date in oot_dates:
        day_str = str(date.date())
        regime = spy_regime(prices_df, date)

        if date in reb_dates:
            vix_val = get_vix(vix_series, date)

            if vix_val < 18:
                lookback = 21   # 1 month
                period_label = "1m (low VIX)"
            elif vix_val > 22:
                lookback = 252  # 12 months
                period_label = "12m (high VIX)"
            else:
                lookback = 63   # 3 months
                period_label = "3m (mid VIX)"

            ret = get_return(prices_df, "QQQ", date, lookback)

            if pd.notna(ret) and ret > 0 and not in_market:
                buy_price = apply_slippage(prices_df["QQQ"].iloc[prices_df.index.get_loc(date)], "buy")
                shares = equity / buy_price
                entry_price = buy_price
                in_market = True
                trades.append({"date": day_str, "ticker": "QQQ", "side": "buy",
                               "price": round(buy_price, 2),
                               "reason": f"adaptive {period_label}, ret={ret:.1%}"})

            elif (pd.notna(ret) and ret <= 0 and in_market) or (pd.isna(ret) and in_market):
                sell_price = apply_slippage(prices_df["QQQ"].iloc[prices_df.index.get_loc(date)], "sell")
                pnl = (sell_price - entry_price) * shares
                equity += pnl
                in_market = False
                shares = 0.0
                trades.append({"date": day_str, "ticker": "QQQ", "side": "sell",
                               "price": round(sell_price, 2), "pnl": round(pnl, 2),
                               "reason": f"adaptive {period_label}, ret<=0"})

        if in_market:
            cur_p = prices_df["QQQ"].iloc[prices_df.index.get_loc(date)]
            port_val = equity + (cur_p - entry_price) * shares
        else:
            port_val = equity

        daily_ret = (port_val - prev_equity) / prev_equity if prev_equity > 0 else 0
        regime_returns[regime].append(daily_ret)
        equity_curve.append({"date": day_str, "equity": round(port_val, 2), "regime": regime})
        prev_equity = port_val

    if in_market:
        last_p = prices_df["QQQ"].iloc[prices_df.index.get_loc(oot_dates[-1])]
        sell_price = apply_slippage(last_p, "sell")
        pnl = (sell_price - entry_price) * shares
        equity += pnl
        trades.append({"date": str(oot_dates[-1].date()), "ticker": "QQQ", "side": "sell",
                       "price": round(sell_price, 2), "pnl": round(pnl, 2), "reason": "final liquidation"})
        port_val = equity

    return equity_curve, trades, regime_returns


# ── Validation ─────────────────────────────────────────────────────────────────

def compute_metrics(equity_curve, trades, regime_returns):
    """Compute all performance metrics from equity curve."""
    eqs = [e["equity"] for e in equity_curve]
    if len(eqs) < 2:
        return None

    eq_series = pd.Series(eqs)
    daily_rets = eq_series.pct_change().dropna()

    if len(daily_rets) == 0 or daily_rets.std() == 0:
        return None

    # Core metrics
    total_ret = (eqs[-1] / eqs[0]) - 1.0
    years = len(daily_rets) / 252.0
    ann_ret = (1 + total_ret) ** (1 / years) - 1.0 if years > 0 else 0
    ann_vol = daily_rets.std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0

    # Sortino
    downside = daily_rets[daily_rets < 0]
    downside_vol = downside.std() * np.sqrt(252) if len(downside) > 0 else 1e-6
    sortino = ann_ret / downside_vol if downside_vol > 0 else 0

    # Max drawdown
    peak = eq_series.expanding().max()
    dd = (eq_series - peak) / peak
    max_dd = dd.min()

    # Trade count
    n_trades = len([t for t in trades if t["side"] == "sell" and "pnl" in t])
    winning = [t for t in trades if t["side"] == "sell" and "pnl" in t and t["pnl"] > 0]
    losing = [t for t in trades if t["side"] == "sell" and "pnl" in t and t["pnl"] <= 0]
    win_rate = len(winning) / n_trades if n_trades > 0 else 0

    # Profit factor
    gross_win = sum(t["pnl"] for t in winning) if winning else 0
    gross_loss = abs(sum(t["pnl"] for t in losing)) if losing else 1e-6
    profit_factor = gross_win / gross_loss if gross_loss > 0 else 0

    # Regime analysis
    bull_rets = regime_returns.get("bull", [])
    bear_rets = regime_returns.get("bear", [])
    bull_sharpe = (np.mean(bull_rets) / np.std(bull_rets) * np.sqrt(252)) if len(bull_rets) > 10 and np.std(bull_rets) > 0 else 0
    bear_sharpe = (np.mean(bear_rets) / np.std(bear_rets) * np.sqrt(252)) if len(bear_rets) > 10 and np.std(bear_rets) > 0 else 0

    regime_gap = abs(bull_sharpe - bear_sharpe) / max(abs(bull_sharpe), abs(bear_sharpe), 1e-6)

    return {
        "total_return_pct": round(total_ret * 100, 2),
        "ann_return_pct": round(ann_ret * 100, 2),
        "ann_vol_pct": round(ann_vol * 100, 2),
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "max_dd_pct": round(max_dd * 100, 2),
        "n_trades": n_trades,
        "win_rate": round(win_rate, 3),
        "profit_factor": round(profit_factor, 3),
        "final_equity": round(eqs[-1], 2),
        "bull_sharpe": round(bull_sharpe, 3),
        "bear_sharpe": round(bear_sharpe, 3),
        "regime_gap": round(regime_gap, 3),
        "bull_days": len(bull_rets),
        "bear_days": len(bear_rets),
    }


def permutation_test(equity_curve, n_perm=N_PERM, seed=RANDOM_SEED):
    """Shuffle entry months to test if signal is real."""
    eqs = [e["equity"] for e in equity_curve]
    eq_series = pd.Series(eqs)
    daily_rets = eq_series.pct_change().dropna().values

    if len(daily_rets) < 20 or np.std(daily_rets) == 0:
        return 1.0

    actual_sharpe = np.mean(daily_rets) / np.std(daily_rets) * np.sqrt(252)

    rng = np.random.RandomState(seed)
    count_better = 0
    for _ in range(n_perm):
        shuffled = rng.permutation(daily_rets)
        s = np.mean(shuffled) / np.std(shuffled) * np.sqrt(252)
        if s >= actual_sharpe:
            count_better += 1

    return (count_better + 1) / (n_perm + 1)


def validate_5gate(metrics, perm_p):
    """Apply 5-gate validation."""
    if metrics is None:
        return {"pass": False, "reason": "no metrics"}

    gates = {
        "sharpe_gt_0.5": metrics["sharpe"] > 0.5,
        "perm_p_lt_0.05": perm_p < 0.05,
        "regime_gap_lt_0.5": metrics["regime_gap"] < 0.5,
        "max_dd_gt_neg50": metrics["max_dd_pct"] > -50,
        "min_20_trades": metrics["n_trades"] >= 20,
    }
    all_pass = all(gates.values())
    return {"pass": all_pass, "gates": gates}


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("MULTI-PERIOD MOMENTUM BACKTEST WITH CRASH PROTECTION")
    print("=" * 70)
    print(f"Account: ${ACCOUNT_SIZE}  |  OOT: {OOT_START} to {OOT_END}")
    print(f"Slippage: {SLIPPAGE_PCT*100:.2f}%  |  Commission: ${COMMISSION}")
    print()

    prices_df, vix_series = download_data()

    variant_runners = {
        "A_classic_12_1": ("A) Classic 12-1 Momentum", run_variant_A),
        "B_multi_timeframe": ("B) Multi-Timeframe Composite", run_variant_B),
        "C_momentum_crash": ("C) Momentum + Crash Filter", run_variant_C),
        "D_relative_momentum": ("D) Relative Momentum", run_variant_D),
        "E_sector_crash": ("E) Sector Momentum + Crash", run_variant_E),
        "F_adaptive_period": ("F) Adaptive Period (VIX)", run_variant_F),
    }

    results = {}

    for key, (label, runner) in variant_runners.items():
        print(f"\n{'─' * 60}")
        print(f"Running {label}...")

        equity_curve, trades, regime_returns = runner(prices_df, vix_series)
        metrics = compute_metrics(equity_curve, trades, regime_returns)

        if metrics is None:
            print(f"  SKIP: no valid metrics")
            results[key] = {"label": label, "metrics": None, "validation": {"pass": False}}
            continue

        print(f"  Sharpe: {metrics['sharpe']:.3f}  |  Sortino: {metrics['sortino']:.3f}")
        print(f"  Return: {metrics['total_return_pct']:.1f}%  |  MaxDD: {metrics['max_dd_pct']:.1f}%")
        print(f"  Trades: {metrics['n_trades']}  |  WR: {metrics['win_rate']:.1%}  |  PF: {metrics['profit_factor']:.2f}")
        print(f"  Final equity: ${metrics['final_equity']:.2f}")
        print(f"  Bull Sharpe: {metrics['bull_sharpe']:.3f}  |  Bear Sharpe: {metrics['bear_sharpe']:.3f}  |  Gap: {metrics['regime_gap']:.3f}")

        print(f"  Running permutation test ({N_PERM} iterations)...")
        perm_p = permutation_test(equity_curve)
        print(f"  Permutation p-value: {perm_p:.4f}")

        validation = validate_5gate(metrics, perm_p)
        status = "PASS" if validation["pass"] else "FAIL"
        print(f"  5-GATE VALIDATION: {status}")
        if not validation["pass"]:
            failed = [g for g, v in validation.get("gates", {}).items() if not v]
            print(f"    Failed gates: {', '.join(failed)}")

        results[key] = {
            "label": label,
            "metrics": metrics,
            "perm_p": round(perm_p, 4),
            "validation": validation,
            "equity_curve_summary": {
                "start": equity_curve[0] if equity_curve else None,
                "end": equity_curve[-1] if equity_curve else None,
                "n_days": len(equity_curve),
            },
            "trade_count": len(trades),
            "sample_trades": trades[:5] + trades[-5:] if len(trades) > 10 else trades,
        }

    # ── Summary ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SUMMARY — MULTI-PERIOD MOMENTUM BACKTEST")
    print("=" * 70)
    print(f"{'Variant':<35} {'Sharpe':>7} {'Sortino':>8} {'Return':>8} {'MaxDD':>7} {'Trades':>7} {'WR':>6} {'PF':>6} {'Perm-p':>7} {'5-Gate':>7}")
    print("-" * 110)

    for key, res in results.items():
        m = res.get("metrics")
        if m is None:
            print(f"{res['label']:<35} {'N/A':>7}")
            continue
        p = res.get("perm_p", 1.0)
        v = "PASS" if res["validation"]["pass"] else "FAIL"
        print(f"{res['label']:<35} {m['sharpe']:>7.3f} {m['sortino']:>8.3f} {m['total_return_pct']:>7.1f}% {m['max_dd_pct']:>6.1f}% {m['n_trades']:>7} {m['win_rate']:>5.1%} {m['profit_factor']:>6.2f} {p:>7.4f} {v:>7}")

    # Best variant
    passing = {k: v for k, v in results.items() if v.get("validation", {}).get("pass", False)}
    if passing:
        best_key = max(passing, key=lambda k: passing[k]["metrics"]["sharpe"])
        print(f"\nBEST PASSING VARIANT: {results[best_key]['label']} (Sharpe {results[best_key]['metrics']['sharpe']:.3f})")
    else:
        print("\nNO VARIANT PASSED ALL 5 GATES.")
        # Show best Sharpe anyway
        with_metrics = {k: v for k, v in results.items() if v.get("metrics") is not None}
        if with_metrics:
            best_key = max(with_metrics, key=lambda k: with_metrics[k]["metrics"]["sharpe"])
            print(f"Best Sharpe (non-passing): {results[best_key]['label']} ({results[best_key]['metrics']['sharpe']:.3f})")

    # Save results
    output = {
        "strategy": "Multi-Period Momentum with Crash Protection",
        "account_size": ACCOUNT_SIZE,
        "oot_period": f"{OOT_START} to {OOT_END}",
        "slippage_pct": SLIPPAGE_PCT,
        "commission": COMMISSION,
        "n_permutations": N_PERM,
        "generated_at": dt.datetime.now().isoformat(),
        "variants": results,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
