#!/usr/bin/env python3
"""
Crypto-Equity Lead-Lag Backtest
===============================
Tests 6 variants of crypto signals leading equity moves.
OOT: Jan 2022 – Jul 2026. Capital: $645, $0 commission (Robinhood).
Regime: SPY 200-SMA bull/bear.
5-gate validation + 1000-shuffle permutation test.
"""

import json
import math
import warnings
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats
from scipy.stats import norm

warnings.filterwarnings("ignore")

# ─── CONFIG ───────────────────────────────────────────────────────────────────
INITIAL_CAPITAL = 645.0
OOT_START = "2022-01-01"
OOT_END = "2026-07-28"
DATA_START = "2020-01-01"  # extra history for SMA/vol lookback
PERM_ITERS = 1000
OUTPUT_PATH = Path("/home/jupiter/Lvl3Quant/data/crypto_equity_leadlag_results.json")

TICKERS = ["BTC-USD", "ETH-USD", "COIN", "MARA", "RIOT", "MSTR", "QQQ", "SPY", "SHY"]


# ─── DATA ─────────────────────────────────────────────────────────────────────
def fetch_data():
    """Fetch daily OHLCV for all tickers via yfinance."""
    print("Fetching data...")
    frames = {}
    for t in TICKERS:
        try:
            df = yf.download(t, start=DATA_START, end=OOT_END, progress=False, auto_adjust=True)
            if df is not None and len(df) > 50:
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                df.index = pd.to_datetime(df.index).tz_localize(None)
                frames[t] = df
                print(f"  {t}: {len(df)} rows, {df.index[0].date()} -> {df.index[-1].date()}")
            else:
                print(f"  {t}: insufficient data")
        except Exception as e:
            print(f"  {t}: FAILED - {e}")
    return frames


# ─── BLACK-SCHOLES FOR VARIANT C ─────────────────────────────────────────────
def bs_call_price(S, K, T, r, sigma):
    """Black-Scholes call price."""
    if T <= 0 or sigma <= 0:
        return max(S - K, 0)
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)


# ─── BACKTEST ENGINE ────────────────────────────────────────────────────────
class SimpleBacktest:
    """Long-only equity backtest with fixed hold periods. $0 commission."""

    def __init__(self, prices: pd.Series, signals: pd.DataFrame, hold_days: int,
                 capital: float = INITIAL_CAPITAL):
        self.prices = prices.dropna()
        self.signals = signals
        self.hold_days = hold_days
        self.capital = capital

    def run(self):
        """Run backtest, return trade list and daily equity series."""
        trades = []
        equity = self.capital
        active_exit = None

        sig_dates = self.signals[self.signals["signal"] == 1].index.sort_values()
        price_dates = self.prices.index

        for sig_date in sig_dates:
            if active_exit is not None and sig_date < active_exit:
                continue
            future = price_dates[price_dates > sig_date]
            if len(future) < self.hold_days + 1:
                continue

            entry_date = future[0]
            exit_idx = min(self.hold_days, len(future) - 1)
            exit_date = future[exit_idx]

            entry_price = float(self.prices.loc[entry_date])
            exit_price = float(self.prices.loc[exit_date])

            shares = equity / entry_price
            pnl = shares * (exit_price - entry_price)
            ret = (exit_price / entry_price) - 1.0

            equity += pnl
            active_exit = exit_date

            trades.append({
                "entry_date": str(entry_date.date()),
                "exit_date": str(exit_date.date()),
                "entry_price": round(entry_price, 4),
                "exit_price": round(exit_price, 4),
                "return": round(float(ret), 6),
                "pnl": round(float(pnl), 2),
                "equity": round(float(equity), 2),
            })

        return trades


class RotationBacktest:
    """Rotation between two instruments based on signal. $0 commission."""

    def __init__(self, risk_on_prices: pd.Series, risk_off_prices: pd.Series,
                 signals: pd.DataFrame, hold_days: int, capital: float = INITIAL_CAPITAL):
        self.risk_on = risk_on_prices.dropna()
        self.risk_off = risk_off_prices.dropna()
        self.signals = signals
        self.hold_days = hold_days
        self.capital = capital

    def run(self):
        trades = []
        equity = self.capital
        active_exit = None

        all_sig = self.signals.dropna(subset=["signal"])
        # Only trade when signal changes or new entry after exit
        sig_dates = all_sig.index.sort_values()

        for sig_date in sig_dates:
            if active_exit is not None and sig_date < active_exit:
                continue

            sig_val = int(all_sig.loc[sig_date, "signal"])
            if sig_val == 0:
                continue  # cash

            prices = self.risk_on if sig_val == 1 else self.risk_off
            price_dates = prices.index
            future = price_dates[price_dates > sig_date]
            if len(future) < self.hold_days + 1:
                continue

            entry_date = future[0]
            exit_date = future[min(self.hold_days, len(future) - 1)]

            entry_price = float(prices.loc[entry_date])
            exit_price = float(prices.loc[exit_date])

            shares = equity / entry_price
            pnl = shares * (exit_price - entry_price)
            ret = (exit_price / entry_price) - 1.0

            equity += pnl
            active_exit = exit_date

            instrument = "risk_on" if sig_val == 1 else "risk_off"
            trades.append({
                "entry_date": str(entry_date.date()),
                "exit_date": str(exit_date.date()),
                "entry_price": round(entry_price, 4),
                "exit_price": round(exit_price, 4),
                "return": round(float(ret), 6),
                "pnl": round(float(pnl), 2),
                "equity": round(float(equity), 2),
                "instrument": instrument,
            })

        return trades


# ─── SIGNAL GENERATORS ───────────────────────────────────────────────────────

def variant_a(data: dict) -> tuple:
    """BTC Momentum -> COIN: BTC 5d ret > 5% -> buy COIN next day, hold 5d. BTC 5d ret < -5% -> avoid/sell."""
    btc = data["BTC-USD"]["Close"]
    coin = data["COIN"]["Close"]

    btc_mom = btc.pct_change(5)
    signals = pd.DataFrame(index=btc_mom.index)
    signals["signal"] = 0
    signals.loc[btc_mom > 0.05, "signal"] = 1
    # When btc_mom < -0.05, signal stays 0 (avoid)

    signals = signals.loc[OOT_START:OOT_END]
    coin_oot = coin.loc[DATA_START:OOT_END]

    bt = SimpleBacktest(coin_oot, signals, hold_days=5)
    return bt.run(), "A_BTC_Momentum_COIN", "COIN"


def variant_b(data: dict) -> tuple:
    """Weekend Crypto -> Monday Equity: BTC Fri-to-Sun > +2% -> buy QQQ Mon open, hold Mon-Fri. BTC < -2% -> cash."""
    btc = data["BTC-USD"]["Close"]
    qqq = data["QQQ"]["Close"]

    btc_dow = btc.index.dayofweek  # 0=Mon ... 4=Fri, 5=Sat, 6=Sun

    # BTC trades weekends, so we have Sat/Sun data
    fridays = btc[btc_dow == 4]
    sundays = btc[btc_dow == 6]

    signals = pd.DataFrame(index=qqq.index)
    signals["signal"] = 0

    for fri_date, fri_close in fridays.items():
        # Find closest Sunday after this Friday
        sun_after = sundays.index[sundays.index > fri_date]
        if len(sun_after) == 0:
            continue
        sun_date = sun_after[0]
        # Only if it's the very next Sunday (2 days later)
        if (sun_date - fri_date).days > 4:
            continue

        sun_close = float(sundays.loc[sun_date])
        weekend_ret = (sun_close / float(fri_close)) - 1.0

        # Find next Monday in QQQ
        mon_candidates = qqq.index[qqq.index > sun_date]
        if len(mon_candidates) == 0:
            continue
        next_trading_day = mon_candidates[0]

        if weekend_ret > 0.02:
            signals.loc[next_trading_day, "signal"] = 1
        # < -2% stays 0 (cash)

    signals = signals.loc[OOT_START:OOT_END]
    qqq_oot = qqq.loc[DATA_START:OOT_END]

    bt = SimpleBacktest(qqq_oot, signals, hold_days=5)
    return bt.run(), "B_Weekend_Crypto_Monday_QQQ", "QQQ"


def variant_c(data: dict) -> tuple:
    """Crypto Crash Bounce: BTC drops >15% in 7d -> buy COIN ATM call (30 DTE, 60% IV, BS pricing). Hold to expiry or 50% gain."""
    btc = data["BTC-USD"]["Close"]
    coin = data["COIN"]["Close"]

    btc_ret7 = btc.pct_change(7)
    crash_dates = btc_ret7[btc_ret7 < -0.15].index
    crash_dates = crash_dates[(crash_dates >= OOT_START) & (crash_dates <= OOT_END)]

    trades = []
    equity = INITIAL_CAPITAL
    active_expiry = None
    rfr = 0.04  # risk-free rate approx

    coin_prices = coin.loc[DATA_START:OOT_END].dropna()
    price_dates = coin_prices.index

    for crash_date in crash_dates:
        if active_expiry is not None and crash_date < active_expiry:
            continue

        # Entry: next trading day for COIN
        future = price_dates[price_dates > crash_date]
        if len(future) < 22:  # need ~30 cal days of data
            continue

        entry_date = future[0]
        S0 = float(coin_prices.loc[entry_date])
        K = S0  # ATM
        T0 = 30 / 365.0
        iv = 0.60

        call_price_entry = bs_call_price(S0, K, T0, rfr, iv)
        if call_price_entry < 0.01:
            continue

        # Number of contracts we can buy (each contract = premium)
        n_contracts = equity / call_price_entry

        # Walk forward up to 30 calendar days (~22 trading days)
        exited = False
        for i in range(1, min(22, len(future))):
            check_date = future[i]
            St = float(coin_prices.loc[check_date])
            days_elapsed = (check_date - entry_date).days
            T_remaining = max((30 - days_elapsed) / 365.0, 0.001)

            call_price_now = bs_call_price(St, K, T_remaining, rfr, iv)
            option_ret = (call_price_now / call_price_entry) - 1.0

            # Exit on 50% gain or expiry
            if option_ret >= 0.50 or days_elapsed >= 30:
                pnl = n_contracts * (call_price_now - call_price_entry)
                equity += pnl
                active_expiry = check_date

                trades.append({
                    "entry_date": str(entry_date.date()),
                    "exit_date": str(check_date.date()),
                    "entry_price": round(call_price_entry, 4),
                    "exit_price": round(call_price_now, 4),
                    "return": round(float(option_ret), 6),
                    "pnl": round(float(pnl), 2),
                    "equity": round(float(equity), 2),
                    "underlying_entry": round(S0, 2),
                    "underlying_exit": round(St, 2),
                    "exit_reason": "50pct_gain" if option_ret >= 0.50 else "expiry",
                })
                exited = True
                break

        if not exited:
            # Fallback: expiry at last available price
            last_date = future[min(21, len(future) - 1)]
            St = float(coin_prices.loc[last_date])
            T_remaining = 0.001
            call_price_now = max(St - K, 0)  # intrinsic at expiry
            option_ret = (call_price_now / call_price_entry) - 1.0
            pnl = n_contracts * (call_price_now - call_price_entry)
            equity += pnl
            active_expiry = last_date

            trades.append({
                "entry_date": str(entry_date.date()),
                "exit_date": str(last_date.date()),
                "entry_price": round(call_price_entry, 4),
                "exit_price": round(call_price_now, 4),
                "return": round(float(option_ret), 6),
                "pnl": round(float(pnl), 2),
                "equity": round(float(equity), 2),
                "underlying_entry": round(S0, 2),
                "underlying_exit": round(St, 2),
                "exit_reason": "expiry",
            })

    return trades, "C_Crypto_Crash_Bounce_Call", "COIN_call"


def variant_d(data: dict) -> tuple:
    """ETH/BTC Ratio Signal: ETH outperforms BTC over 10d -> risk-on (buy QQQ). BTC outperforms -> risk-off (buy SHY). Otherwise cash."""
    btc = data["BTC-USD"]["Close"]
    eth = data["ETH-USD"]["Close"]
    qqq = data["QQQ"]["Close"]
    shy = data["SHY"]["Close"]

    common = btc.index.intersection(eth.index)
    ratio = eth.loc[common] / btc.loc[common]
    ratio_change = ratio.pct_change(10)

    signals = pd.DataFrame(index=ratio_change.index)
    signals["signal"] = 0  # cash
    signals.loc[ratio_change > 0, "signal"] = 1   # ETH outperforming -> risk-on QQQ
    signals.loc[ratio_change < 0, "signal"] = -1  # BTC outperforming -> risk-off SHY

    signals = signals.loc[OOT_START:OOT_END]
    qqq_oot = qqq.loc[DATA_START:OOT_END]
    shy_oot = shy.loc[DATA_START:OOT_END]

    # Custom rotation backtest
    trades = []
    equity = INITIAL_CAPITAL
    active_exit = None

    sig_dates = signals.index.sort_values()
    for sig_date in sig_dates:
        if active_exit is not None and sig_date < active_exit:
            continue

        sig_val = signals.loc[sig_date, "signal"]
        if sig_val == 0:
            continue

        prices = qqq_oot if sig_val == 1 else shy_oot
        instrument_name = "QQQ" if sig_val == 1 else "SHY"
        price_dates = prices.index
        future = price_dates[price_dates > sig_date]
        if len(future) < 11:
            continue

        entry_date = future[0]
        exit_date = future[min(10, len(future) - 1)]

        entry_price = float(prices.loc[entry_date])
        exit_price = float(prices.loc[exit_date])

        shares = equity / entry_price
        pnl = shares * (exit_price - entry_price)
        ret = (exit_price / entry_price) - 1.0
        equity += pnl
        active_exit = exit_date

        trades.append({
            "entry_date": str(entry_date.date()),
            "exit_date": str(exit_date.date()),
            "entry_price": round(entry_price, 4),
            "exit_price": round(exit_price, 4),
            "return": round(float(ret), 6),
            "pnl": round(float(pnl), 2),
            "equity": round(float(equity), 2),
            "instrument": instrument_name,
        })

    return trades, "D_ETH_BTC_Ratio_QQQ_SHY", "QQQ/SHY"


def variant_e(data: dict) -> tuple:
    """Crypto Vol Spillover: BTC 10d realized vol > 80th pctile -> defensive (SHY). Vol < 20th pctile -> aggressive (QQQ). Otherwise cash."""
    btc = data["BTC-USD"]["Close"]
    qqq = data["QQQ"]["Close"]
    shy = data["SHY"]["Close"]

    btc_ret = btc.pct_change()
    btc_vol = btc_ret.rolling(10).std() * np.sqrt(252)

    # Rolling 252-day percentile (no lookahead)
    vol_pctile = btc_vol.rolling(252, min_periods=60).apply(
        lambda x: stats.percentileofscore(x[:-1], x.iloc[-1]) if len(x) > 1 else 50,
        raw=False
    )

    signals = pd.DataFrame(index=vol_pctile.index)
    signals["signal"] = 0  # cash
    signals.loc[vol_pctile < 20, "signal"] = 1   # low vol -> QQQ
    signals.loc[vol_pctile > 80, "signal"] = -1  # high vol -> SHY

    signals = signals.loc[OOT_START:OOT_END]
    qqq_oot = qqq.loc[DATA_START:OOT_END]
    shy_oot = shy.loc[DATA_START:OOT_END]

    trades = []
    equity = INITIAL_CAPITAL
    active_exit = None

    for sig_date in signals.index.sort_values():
        if active_exit is not None and sig_date < active_exit:
            continue

        sig_val = signals.loc[sig_date, "signal"]
        if sig_val == 0:
            continue

        prices = qqq_oot if sig_val == 1 else shy_oot
        instrument_name = "QQQ" if sig_val == 1 else "SHY"
        price_dates = prices.index
        future = price_dates[price_dates > sig_date]
        if len(future) < 11:
            continue

        entry_date = future[0]
        exit_date = future[min(10, len(future) - 1)]

        entry_price = float(prices.loc[entry_date])
        exit_price = float(prices.loc[exit_date])

        shares = equity / entry_price
        pnl = shares * (exit_price - entry_price)
        ret = (exit_price / entry_price) - 1.0
        equity += pnl
        active_exit = exit_date

        trades.append({
            "entry_date": str(entry_date.date()),
            "exit_date": str(exit_date.date()),
            "entry_price": round(entry_price, 4),
            "exit_price": round(exit_price, 4),
            "return": round(float(ret), 6),
            "pnl": round(float(pnl), 2),
            "equity": round(float(equity), 2),
            "instrument": instrument_name,
        })

    return trades, "E_Crypto_Vol_Spillover", "QQQ/SHY"


def variant_f(data: dict, sig_a_df=None, sig_b_df=None, sig_d_df=None) -> tuple:
    """Multi-Signal Composite: Combine A+B+D scores (-3 to +3). Score>=2 -> COIN, score==1 -> QQQ, score<=0 -> cash."""
    btc = data["BTC-USD"]["Close"]
    eth = data["ETH-USD"]["Close"]
    coin = data["COIN"]["Close"]
    qqq = data["QQQ"]["Close"]

    # Reconstruct signal A: BTC 5d momentum > 5%
    btc_mom5 = btc.pct_change(5)
    sig_a = pd.Series(0, index=btc_mom5.index)
    sig_a[btc_mom5 > 0.05] = 1
    sig_a[btc_mom5 < -0.05] = -1

    # Reconstruct signal B: weekend crypto
    btc_dow = btc.index.dayofweek
    fridays = btc[btc_dow == 4]
    sundays = btc[btc_dow == 6]
    sig_b = pd.Series(0, index=btc.index)
    for fri_date, fri_close in fridays.items():
        sun_after = sundays.index[sundays.index > fri_date]
        if len(sun_after) == 0:
            continue
        sun_date = sun_after[0]
        if (sun_date - fri_date).days > 4:
            continue
        sun_close = float(sundays.loc[sun_date])
        weekend_ret = (sun_close / float(fri_close)) - 1.0
        # Assign to next Monday-ish
        next_days = btc.index[btc.index > sun_date]
        if len(next_days) > 0:
            if weekend_ret > 0.02:
                sig_b.loc[next_days[0]] = 1
            elif weekend_ret < -0.02:
                sig_b.loc[next_days[0]] = -1

    # Reconstruct signal D: ETH/BTC ratio 10d change
    common = btc.index.intersection(eth.index)
    ratio = eth.loc[common] / btc.loc[common]
    ratio_change = ratio.pct_change(10)
    sig_d = pd.Series(0, index=ratio_change.index)
    sig_d[ratio_change > 0] = 1
    sig_d[ratio_change < 0] = -1

    # Combine: align all to common dates
    combined = pd.DataFrame({
        "a": sig_a, "b": sig_b, "d": sig_d
    }).fillna(0)
    combined["score"] = combined["a"] + combined["b"] + combined["d"]

    combined = combined.loc[OOT_START:OOT_END]
    coin_oot = coin.loc[DATA_START:OOT_END]
    qqq_oot = qqq.loc[DATA_START:OOT_END]

    trades = []
    equity = INITIAL_CAPITAL
    active_exit = None

    for sig_date in combined.index.sort_values():
        if active_exit is not None and sig_date < active_exit:
            continue

        score = combined.loc[sig_date, "score"]
        if score >= 2:
            prices = coin_oot
            instrument_name = "COIN"
            hold = 5
        elif score == 1:
            prices = qqq_oot
            instrument_name = "QQQ"
            hold = 5
        else:
            continue  # cash

        price_dates = prices.index
        future = price_dates[price_dates > sig_date]
        if len(future) < hold + 1:
            continue

        entry_date = future[0]
        exit_date = future[min(hold, len(future) - 1)]

        entry_price = float(prices.loc[entry_date])
        exit_price = float(prices.loc[exit_date])

        shares = equity / entry_price
        pnl = shares * (exit_price - entry_price)
        ret = (exit_price / entry_price) - 1.0
        equity += pnl
        active_exit = exit_date

        trades.append({
            "entry_date": str(entry_date.date()),
            "exit_date": str(exit_date.date()),
            "entry_price": round(entry_price, 4),
            "exit_price": round(exit_price, 4),
            "return": round(float(ret), 6),
            "pnl": round(float(pnl), 2),
            "equity": round(float(equity), 2),
            "instrument": instrument_name,
            "score": int(score),
        })

    return trades, "F_Multi_Signal_Composite", "COIN/QQQ"


# ─── METRICS & VALIDATION ────────────────────────────────────────────────────
def compute_metrics(trades: list, capital: float = INITIAL_CAPITAL) -> dict:
    """Compute performance metrics from trade list."""
    if len(trades) == 0:
        return {"n_trades": 0, "sharpe": 0, "sortino": 0, "pf": 0, "wr": 0,
                "total_return_pct": 0, "cagr_pct": 0, "max_dd_pct": 0,
                "final_equity": capital}

    returns = np.array([t["return"] for t in trades])
    pnls = np.array([t["pnl"] for t in trades])
    equities = np.array([capital] + [t["equity"] for t in trades])

    n = len(returns)
    mean_ret = np.mean(returns)
    std_ret = np.std(returns, ddof=1) if n > 1 else 1e-9

    # OOT span in years
    first_entry = pd.Timestamp(trades[0]["entry_date"])
    last_exit = pd.Timestamp(trades[-1]["exit_date"])
    years = max((last_exit - first_entry).days / 365.25, 0.5)

    # Annualized Sharpe
    trades_per_year = n / years
    sharpe = (mean_ret / max(std_ret, 1e-9)) * np.sqrt(max(trades_per_year, 1))

    # Sortino
    downside = returns[returns < 0]
    downside_std = np.std(downside, ddof=1) if len(downside) > 1 else 1e-9
    sortino = (mean_ret / max(downside_std, 1e-9)) * np.sqrt(max(trades_per_year, 1))

    # Profit Factor
    gross_profit = float(pnls[pnls > 0].sum()) if (pnls > 0).any() else 0
    gross_loss = float(abs(pnls[pnls < 0].sum())) if (pnls < 0).any() else 1e-9
    pf = gross_profit / max(gross_loss, 1e-9)

    # Win Rate
    wr = float((returns > 0).sum() / n)

    # Max Drawdown
    peak = np.maximum.accumulate(equities)
    dd = (equities - peak) / np.where(peak > 0, peak, 1)
    max_dd = float(dd.min())

    # Total return and CAGR
    total_ret = (equities[-1] / capital - 1) * 100
    cagr = ((equities[-1] / capital) ** (1 / years) - 1) * 100

    return {
        "n_trades": int(n),
        "total_return_pct": round(float(total_ret), 2),
        "cagr_pct": round(float(cagr), 2),
        "sharpe": round(float(sharpe), 3),
        "sortino": round(float(sortino), 3),
        "pf": round(float(pf), 3),
        "wr": round(float(wr), 4),
        "max_dd_pct": round(float(max_dd * 100), 2),
        "final_equity": round(float(equities[-1]), 2),
        "mean_return_per_trade_pct": round(float(mean_ret * 100), 4),
        "avg_pnl": round(float(np.mean(pnls)), 2),
    }


def regime_analysis(trades: list, spy_data: pd.Series) -> dict:
    """Split trades by bull/bear regime (SPY vs 200-SMA)."""
    spy_sma200 = spy_data.rolling(200).mean()

    bull_returns, bear_returns = [], []

    for t in trades:
        entry = pd.Timestamp(t["entry_date"])
        valid = spy_sma200.index[spy_sma200.index <= entry]
        if len(valid) == 0:
            continue
        closest = valid[-1]
        if pd.isna(spy_sma200.loc[closest]):
            continue

        if spy_data.loc[closest] > spy_sma200.loc[closest]:
            bull_returns.append(t["return"])
        else:
            bear_returns.append(t["return"])

    bull_sharpe = _quick_sharpe(bull_returns)
    bear_sharpe = _quick_sharpe(bear_returns)

    max_abs = max(abs(bull_sharpe), abs(bear_sharpe), 1e-9)
    regime_gap = abs(bull_sharpe - bear_sharpe) / max_abs

    return {
        "bull_trades": len(bull_returns),
        "bear_trades": len(bear_returns),
        "bull_sharpe": round(bull_sharpe, 3),
        "bear_sharpe": round(bear_sharpe, 3),
        "regime_gap": round(regime_gap, 3),
        "bull_wr": round(sum(1 for r in bull_returns if r > 0) / max(len(bull_returns), 1), 4),
        "bear_wr": round(sum(1 for r in bear_returns if r > 0) / max(len(bear_returns), 1), 4),
    }


def _quick_sharpe(returns_list):
    if len(returns_list) < 2:
        return 0.0
    arr = np.array(returns_list)
    std = np.std(arr, ddof=1)
    if std < 1e-9:
        return 0.0
    first = pd.Timestamp(OOT_START)
    last = pd.Timestamp(OOT_END)
    years = max((last - first).days / 365.25, 0.5)
    tpy = max(len(arr) / years, 1)
    return float(np.mean(arr) / std * np.sqrt(tpy))


def permutation_test(trades: list, n_iter: int = PERM_ITERS) -> float:
    """Shuffle signal-return mapping; perm_p = fraction of shuffles beating actual Sharpe."""
    if len(trades) < 5:
        return 1.0

    returns = np.array([t["return"] for t in trades])
    actual_sharpe = np.mean(returns) / max(np.std(returns, ddof=1), 1e-9)

    count_better = 0
    for _ in range(n_iter):
        perm = np.random.permutation(returns)
        perm_sharpe = np.mean(perm) / max(np.std(perm, ddof=1), 1e-9)
        if perm_sharpe >= actual_sharpe:
            count_better += 1

    return count_better / n_iter


def validate_5gate(metrics: dict, regime: dict, perm_p: float) -> dict:
    """5-gate validation."""
    gates = {
        "sharpe_gt_0.5": metrics["sharpe"] > 0.5,
        "perm_p_lt_0.05": perm_p < 0.05,
        "regime_gap_lt_0.5": regime["regime_gap"] < 0.5,
        "max_dd_gt_neg50": metrics["max_dd_pct"] > -50,
        "min_20_trades": metrics["n_trades"] >= 20,
    }
    gates["all_pass"] = all(v for k, v in gates.items())
    gates["gates_passed"] = sum(v for k, v in gates.items() if k not in ("all_pass", "gates_passed"))
    return gates


# ─── MAIN ─────────────────────────────────────────────────────────────────────
def main():
    np.random.seed(42)

    data = fetch_data()

    required = ["BTC-USD", "ETH-USD", "SPY", "QQQ", "COIN"]
    for r in required:
        if r not in data:
            print(f"FATAL: Missing required ticker {r}")
            return

    spy_close = data["SPY"]["Close"]

    variant_funcs = [
        ("A", variant_a),
        ("B", variant_b),
        ("C", variant_c),
        ("D", variant_d),
        ("E", variant_e),
        ("F", variant_f),
    ]

    results = {}

    for label, vfunc in variant_funcs:
        try:
            result = vfunc(data)
            trades, name, instrument = result
        except Exception as e:
            print(f"  Variant {label}: ERROR - {e}")
            import traceback
            traceback.print_exc()
            continue

        metrics = compute_metrics(trades)
        regime = regime_analysis(trades, spy_close)

        print(f"\n{'='*60}")
        print(f"Variant {name} (trading {instrument})")
        print(f"  Trades: {metrics['n_trades']}, Sharpe: {metrics['sharpe']}, "
              f"Sortino: {metrics['sortino']}, PF: {metrics['pf']}, WR: {metrics['wr']:.1%}")
        print(f"  Total Return: {metrics['total_return_pct']:.1f}%, CAGR: {metrics['cagr_pct']:.1f}%, "
              f"MaxDD: {metrics['max_dd_pct']:.1f}%, Final Equity: ${metrics['final_equity']:.2f}")
        print(f"  Regime: Bull {regime['bull_trades']}t (Sharpe {regime['bull_sharpe']}), "
              f"Bear {regime['bear_trades']}t (Sharpe {regime['bear_sharpe']}), Gap: {regime['regime_gap']}")

        print(f"  Running permutation test ({PERM_ITERS} shuffles)...")
        perm_p = permutation_test(trades)
        print(f"  Permutation p-value: {perm_p:.4f}")

        gates = validate_5gate(metrics, regime, perm_p)
        passed = gates["gates_passed"]
        print(f"  5-Gate: {passed}/5 passed | ALL PASS: {gates['all_pass']}")
        for g, v in gates.items():
            if g not in ("all_pass", "gates_passed"):
                print(f"    {'PASS' if v else 'FAIL'}: {g}")

        results[name] = {
            "variant": label,
            "instrument": instrument,
            "description": vfunc.__doc__.strip().split("\n")[0] if vfunc.__doc__ else name,
            "metrics": metrics,
            "regime": regime,
            "perm_p": round(perm_p, 4),
            "gates": gates,
            "sample_trades": trades[:5] if trades else [],
            "n_trades_total": len(trades),
        }

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    passing = [k for k, v in results.items() if v["gates"]["all_pass"]]
    failing = [k for k, v in results.items() if not v["gates"]["all_pass"]]
    print(f"  Passing 5-gate: {len(passing)} -- {passing}")
    print(f"  Failing 5-gate: {len(failing)} -- {failing}")

    best = None
    if results:
        best = max(results.items(), key=lambda x: x[1]["metrics"]["sharpe"])
        print(f"  Best Sharpe: {best[0]} = {best[1]['metrics']['sharpe']}")

    output = {
        "metadata": {
            "strategy": "Crypto-Equity Lead-Lag",
            "concept": "Bitcoin and Ethereum often lead equity market moves, especially for crypto-adjacent stocks. Weekend crypto moves can predict Monday equity direction.",
            "oot_period": f"{OOT_START} to {OOT_END}",
            "initial_capital": INITIAL_CAPITAL,
            "commission": 0.0,
            "regime_def": "SPY vs 200-SMA",
            "perm_iterations": PERM_ITERS,
            "run_timestamp": str(dt.datetime.now()),
        },
        "variants": results,
        "summary": {
            "total_variants": len(results),
            "passing_5gate": len(passing),
            "passing_names": passing,
            "failing_names": failing,
            "best_sharpe_variant": best[0] if best else None,
            "best_sharpe_value": best[1]["metrics"]["sharpe"] if best else None,
        }
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
