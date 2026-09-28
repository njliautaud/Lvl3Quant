#!/usr/bin/env python3
"""
wheel_mean_reversion_overlay.py — Mean-Reversion Entry Timing for Wheel CSP

Hypothesis: Selling CSPs after short-term dips captures higher premium (stock is
temporarily depressed) and benefits from mean-reversion bounce (less likely to
breach the put strike).

Signals:
  1. RSI(5) < 30            — short-term oversold
  2. Price < BB_lower(20,2)  — below lower Bollinger Band
  3. 5d_ret < -2σ(60d)      — recent drawdown exceeds 2 standard deviations

Combo: at least 2 of 3 must fire to open a new CSP.

Backtest: walk-forward sliding window (60-day train / 1-day OOS).
All metrics are OOS only. Permutation test shuffles signal dates.

Same V5 config as baseline — only TIMING of entry changes.
"""
from __future__ import annotations

import json
import math
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path("/home/jupiter/Lvl3Quant")
CACHE = ROOT / "wheel_strategy_v1" / "data" / "cache"
OUT_DIR = ROOT / "output" / "mean_reversion_overlay"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── V5 Config (identical to portfolio backtest — only timing changes) ──
START_DATE = pd.Timestamp("2019-01-01")
STARTING_CAPITAL = 100_000.0
MAX_PER_NAME_PCT = 0.10
MAX_ACTIVE_POSITIONS = 10
TRADING_DAYS = 252
RISK_FREE = 0.04

PUT_DELTA = 0.25
CALL_DELTA = 0.30
DTE_MIN = 10
DTE_MAX = 18
DTE_TARGET = 14
PROFIT_TAKE = 0.50
VIX_MAX = 35.0

COST_PER_CONTRACT = 0.65
SLIPPAGE_FRAC = 0.025
SLIPPAGE_MIN = 0.03

BASKET = [
    'WYNN', 'EXC', 'XOM', 'TXN', 'EA', 'IBM', 'DLR', 'AEP', 'GILD', 'CAT',
    'VZ', 'CL', 'TMUS', 'IRM', 'CVS', 'AXP', 'DUK', 'ABT', 'CVX', 'PG',
    'HON', 'CSCO', 'HD', 'VLO', 'SBUX', 'GS', 'TGT', 'SPG', 'UNP', 'V', 'LIN',
]

# ── Mean-reversion signal parameters ──
RSI_PERIOD = 5
RSI_THRESHOLD = 30
BB_PERIOD = 20
BB_STD = 2.0
RET_LOOKBACK = 5
RET_VOL_LOOKBACK = 60
RET_THRESHOLD_SIGMA = 2.0
COMBO_MIN_SIGNALS = 2

# ── Walk-forward params ──
WF_TRAIN_DAYS = 60  # Not used for signal fitting (signals are fixed), but for
                     # defining the minimum warmup before OOS starts
WF_WARMUP = max(BB_PERIOD, RET_VOL_LOOKBACK) + 10  # need enough history for indicators

# ── Permutation test ──
N_PERMUTATIONS = 1000


# ── BS pricing (copied from portfolio backtest for self-containment) ──
def _Phi(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def bs_price(S, K, T, sigma, r=RISK_FREE, kind="put"):
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return max(K - S, 0.0) if kind == "put" else max(S - K, 0.0)
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    if kind == "put":
        return K * math.exp(-r * T) * _Phi(-d2) - S * _Phi(-d1)
    return S * _Phi(d1) - K * math.exp(-r * T) * _Phi(d2)

def strike_from_delta(S, T, sigma, target_delta, kind="put", r=RISK_FREE):
    if T <= 0 or sigma <= 0:
        return S
    p = (1.0 - abs(target_delta)) if kind == "put" else abs(target_delta)
    p = min(max(p, 1e-9), 1 - 1e-9)
    a = [-39.69683028665376, 220.9460984245205, -275.9285104469687,
         138.3577518672690, -30.66479806614716, 2.506628277459239]
    b = [-54.47609879822406, 161.5858368580409, -155.6989798598866,
         66.80131188771972, -13.28068155288572]
    c = [-0.007784894002430293, -0.3223964580411365, -2.400758277161838,
         -2.549732539343734, 4.374664141464968, 2.938163982698783]
    d_ = [0.007784695709041462, 0.3224671290700398, 2.445134137142996,
          3.754408661907416]
    pl, pu = 0.02425, 1 - 0.02425
    if p < pl:
        q = math.sqrt(-2 * math.log(p))
        z = (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d_[0]*q+d_[1])*q+d_[2])*q+d_[3])*q+1)
    elif p <= pu:
        q = p - 0.5
        rr = q*q
        z = (((((a[0]*rr+a[1])*rr+a[2])*rr+a[3])*rr+a[4])*rr+a[5])*q / (((((b[0]*rr+b[1])*rr+b[2])*rr+b[3])*rr+b[4])*rr+1)
    else:
        q = math.sqrt(-2 * math.log(1-p))
        z = -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d_[0]*q+d_[1])*q+d_[2])*q+d_[3])*q+1)
    d1 = z
    K = S * math.exp((r + 0.5 * sigma**2) * T - d1 * sigma * math.sqrt(T))
    return max(0.01, round(K, 2))

def slippage(premium):
    return max(SLIPPAGE_MIN, SLIPPAGE_FRAC * premium) if premium and premium > 0 else 0.0

def find_expiry(open_date):
    best, best_dist = None, 10000
    for d_off in range(DTE_MIN, DTE_MAX + 1):
        cand = open_date + pd.Timedelta(days=d_off)
        shift = (4 - cand.weekday()) % 7
        cand_fri = cand + pd.Timedelta(days=shift)
        dte = (cand_fri - open_date).days
        if dte < DTE_MIN or dte > DTE_MAX:
            continue
        dist = abs(dte - DTE_TARGET)
        if dist < best_dist:
            best, best_dist = cand_fri, dist
    return best


# ── Position tracking ──
class TickerPosition:
    __slots__ = ['ticker', 'side', 'strike', 'expiry', 'open_date',
                 'open_price', 'contracts', 'share_basis']

    def __init__(self, ticker, side, strike, expiry, open_date, open_price,
                 contracts, share_basis=0.0):
        self.ticker = ticker
        self.side = side
        self.strike = strike
        self.expiry = expiry
        self.open_date = open_date
        self.open_price = open_price
        self.contracts = contracts
        self.share_basis = share_basis


# ── Mean-reversion signal computation ──
def compute_signals(prices_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute mean-reversion signals for each ticker-date.
    Returns a DataFrame with columns: ticker, date, rsi_signal, bb_signal,
    ret_signal, combo_signal, n_signals_active.
    """
    results = []
    for ticker, grp in prices_df.groupby("ticker"):
        grp = grp.sort_values("date").copy()
        close = grp["close"].values
        dates = grp["date"].values
        n = len(close)

        # 1. RSI(5)
        rsi = np.full(n, np.nan)
        if n > RSI_PERIOD:
            delta = np.diff(close, prepend=close[0])
            gain = np.where(delta > 0, delta, 0.0)
            loss = np.where(delta < 0, -delta, 0.0)
            # Use exponential moving average for RSI
            avg_gain = np.full(n, np.nan)
            avg_loss = np.full(n, np.nan)
            avg_gain[RSI_PERIOD] = np.mean(gain[1:RSI_PERIOD + 1])
            avg_loss[RSI_PERIOD] = np.mean(loss[1:RSI_PERIOD + 1])
            for i in range(RSI_PERIOD + 1, n):
                avg_gain[i] = (avg_gain[i-1] * (RSI_PERIOD - 1) + gain[i]) / RSI_PERIOD
                avg_loss[i] = (avg_loss[i-1] * (RSI_PERIOD - 1) + loss[i]) / RSI_PERIOD
            with np.errstate(divide='ignore', invalid='ignore'):
                rs = avg_gain / np.where(avg_loss > 0, avg_loss, 1e-10)
                rsi = 100.0 - 100.0 / (1.0 + rs)

        rsi_signal = (rsi < RSI_THRESHOLD).astype(float)
        rsi_signal[np.isnan(rsi)] = 0.0

        # 2. Bollinger Band (20d, 2σ)
        bb_signal = np.zeros(n)
        if n > BB_PERIOD:
            sma = pd.Series(close).rolling(BB_PERIOD).mean().values
            std = pd.Series(close).rolling(BB_PERIOD).std().values
            bb_lower = sma - BB_STD * std
            bb_signal = np.where(
                (~np.isnan(bb_lower)) & (close < bb_lower), 1.0, 0.0
            )

        # 3. 5d return < -2σ of trailing 60d returns
        ret_signal = np.zeros(n)
        if n > RET_VOL_LOOKBACK:
            log_ret = np.log(close[1:] / close[:-1])
            log_ret = np.insert(log_ret, 0, 0.0)
            for i in range(RET_VOL_LOOKBACK, n):
                ret_5d = np.sum(log_ret[i - RET_LOOKBACK + 1: i + 1])
                trailing_rets = log_ret[i - RET_VOL_LOOKBACK + 1: i + 1]
                # 5d rolling returns within 60d window
                if len(trailing_rets) >= 20:
                    mu = np.mean(trailing_rets) * RET_LOOKBACK
                    sd = np.std(trailing_rets, ddof=1) * np.sqrt(RET_LOOKBACK)
                    if sd > 0 and ret_5d < mu - RET_THRESHOLD_SIGMA * sd:
                        ret_signal[i] = 1.0

        # Combo: at least 2 of 3
        n_active = rsi_signal + bb_signal + ret_signal
        combo = (n_active >= COMBO_MIN_SIGNALS).astype(float)

        for i in range(n):
            results.append({
                "ticker": ticker,
                "date": dates[i],
                "rsi_signal": rsi_signal[i],
                "bb_signal": bb_signal[i],
                "ret_signal": ret_signal[i],
                "n_signals_active": n_active[i],
                "combo_signal": combo[i],
            })

    return pd.DataFrame(results)


def load_data():
    """Load price data for basket + SPY (for regime gate) + VIX."""
    print("[load] Loading prices ...")
    p1 = pd.read_parquet(CACHE / "prices.parquet")[["ticker", "date", "close"]].copy()
    p1["date"] = pd.to_datetime(p1["date"], utc=False)
    if p1["date"].dt.tz is not None:
        p1["date"] = p1["date"].dt.tz_localize(None)

    p2 = pd.read_parquet(CACHE / "prices_expanded.parquet")
    p2 = p2.rename(columns={"Close": "close"})[["ticker", "date", "close"]].copy()
    p2["date"] = pd.to_datetime(p2["date"], utc=False)
    if p2["date"].dt.tz is not None:
        p2["date"] = p2["date"].dt.tz_localize(None)

    prices = pd.concat([p1, p2], ignore_index=True)
    prices = prices.sort_values(["ticker", "date"]).reset_index(drop=True)
    prices = prices.dropna(subset=["close"])
    prices = prices[prices["close"] > 0]
    prices = prices[prices["date"] >= START_DATE].copy()
    # Deduplicate
    prices = prices.drop_duplicates(subset=["ticker", "date"], keep="last")

    # Realized vol
    prices["log_ret"] = prices.groupby("ticker")["close"].transform(
        lambda x: np.log(x / x.shift(1))
    )
    prices["sigma"] = prices.groupby("ticker")["log_ret"].transform(
        lambda x: x.rolling(20, min_periods=15).std() * np.sqrt(252)
    )
    prices["sigma"] = prices["sigma"].clip(lower=0.05, upper=2.0)
    prices = prices.dropna(subset=["sigma"]).reset_index(drop=True)

    # VIX
    macro = pd.read_parquet(CACHE / "macro.parquet")[["date", "vix"]].copy()
    macro["date"] = pd.to_datetime(macro["date"], utc=False)
    if macro["date"].dt.tz is not None:
        macro["date"] = macro["date"].dt.tz_localize(None)
    prices = prices.merge(macro, on="date", how="left")
    prices["vix"] = prices["vix"].ffill().fillna(20.0)

    # SPY for regime gate
    spy = prices[prices["ticker"] == "SPY"][["date", "close"]].copy()
    if spy.empty:
        import yfinance as yf
        spy_data = yf.Ticker("SPY").history(period="max", interval="1d")
        spy = pd.DataFrame({"date": spy_data.index.tz_localize(None),
                            "close": spy_data["Close"].values})
    spy = spy.sort_values("date")
    spy["spy_sma50"] = spy["close"].rolling(50).mean()
    spy["bear"] = spy["close"] < spy["spy_sma50"]
    spy_regime = spy[["date", "bear"]].dropna()

    # All tickers needed (basket + SPY for signals)
    needed = set(BASKET) | {"SPY"}
    prices_all = prices[prices["ticker"].isin(needed)].copy()
    prices_basket = prices[prices["ticker"].isin(set(BASKET))].copy()

    all_dates = sorted(prices_basket["date"].unique())

    return prices_all, prices_basket, spy_regime, all_dates


def run_backtest(prices_basket, spy_regime, all_dates, price_lookup, bear_lookup,
                 signal_lookup=None, label="baseline"):
    """
    Run the wheel portfolio backtest.

    If signal_lookup is None: baseline (open CSPs anytime V5 rules allow).
    If signal_lookup is a dict {(ticker, date): bool}: only open CSPs when
    signal_lookup[(ticker, date)] is True.
    """
    cash = STARTING_CAPITAL
    positions: dict[str, TickerPosition] = {}
    equity_series = []
    trade_log = []
    daily_pnl = []
    n_trades = 0
    wins = 0
    total_premium = 0.0
    prev_equity = STARTING_CAPITAL
    entries_attempted = 0
    entries_blocked_by_signal = 0

    for di, date in enumerate(all_dates):
        is_bear = bear_lookup.get(date, False)

        # ── Process existing positions ──
        tickers_to_remove = []
        for ticker, pos in list(positions.items()):
            data = price_lookup.get((ticker, date))
            if data is None:
                continue
            S, sigma, vix = data
            T = max((pos.expiry - date).days, 0) / 365.0

            if pos.side == "short_put":
                opt = bs_price(S, pos.strike, T, sigma, kind="put")
                pf = (pos.open_price - opt) / pos.open_price if pos.open_price > 0 else 0
                is_expiry = date >= pos.expiry

                if is_bear and not is_expiry:
                    slip_cost = slippage(opt)
                    cost = (opt + slip_cost) * 100 * pos.contracts + COST_PER_CONTRACT * pos.contracts
                    realized = pos.open_price * 100 * pos.contracts - cost
                    cash += realized
                    n_trades += 1
                    if realized > 0: wins += 1
                    trade_log.append({"date": str(date.date()), "ticker": ticker,
                                     "action": "bear_close", "pnl": realized})
                    tickers_to_remove.append(ticker)
                    continue

                if pf >= PROFIT_TAKE and not is_expiry:
                    slip_cost = slippage(opt)
                    cost = (opt + slip_cost) * 100 * pos.contracts + COST_PER_CONTRACT * pos.contracts
                    realized = pos.open_price * 100 * pos.contracts - cost
                    cash += realized
                    n_trades += 1
                    if realized > 0: wins += 1
                    trade_log.append({"date": str(date.date()), "ticker": ticker,
                                     "action": "profit_take", "pnl": realized})
                    tickers_to_remove.append(ticker)
                    continue

                if is_expiry:
                    if S < pos.strike:
                        cost = pos.strike * 100 * pos.contracts
                        cash -= cost
                        basis = pos.strike - pos.open_price
                        n_trades += 1
                        wins += 1
                        positions[ticker] = TickerPosition(
                            ticker=ticker, side="long_shares", strike=basis,
                            expiry=date, open_date=date, open_price=basis,
                            contracts=pos.contracts, share_basis=basis,
                        )
                        trade_log.append({"date": str(date.date()), "ticker": ticker,
                                         "action": "assigned", "pnl": 0})
                    else:
                        realized = pos.open_price * 100 * pos.contracts - COST_PER_CONTRACT * pos.contracts
                        n_trades += 1
                        if realized > 0: wins += 1
                        trade_log.append({"date": str(date.date()), "ticker": ticker,
                                         "action": "expired_otm", "pnl": realized})
                        tickers_to_remove.append(ticker)
                    continue

            elif pos.side == "short_call":
                opt = bs_price(S, pos.strike, T, sigma, kind="call")
                pf = (pos.open_price - opt) / pos.open_price if pos.open_price > 0 else 0
                is_expiry = date >= pos.expiry

                if pf >= PROFIT_TAKE and not is_expiry:
                    slip_cost = slippage(opt)
                    cost = (opt + slip_cost) * 100 * pos.contracts + COST_PER_CONTRACT * pos.contracts
                    realized = pos.open_price * 100 * pos.contracts - cost
                    cash += realized
                    n_trades += 1
                    if realized > 0: wins += 1
                    positions[ticker] = TickerPosition(
                        ticker=ticker, side="long_shares", strike=pos.share_basis,
                        expiry=date, open_date=date, open_price=pos.share_basis,
                        contracts=pos.contracts, share_basis=pos.share_basis,
                    )
                    trade_log.append({"date": str(date.date()), "ticker": ticker,
                                     "action": "close_cc_profit", "pnl": realized})
                    continue

                if is_expiry:
                    if S > pos.strike:
                        proceeds = pos.strike * 100 * pos.contracts
                        premium = pos.open_price * 100 * pos.contracts - COST_PER_CONTRACT * pos.contracts
                        share_pnl = (pos.strike - pos.share_basis) * 100 * pos.contracts
                        cash += proceeds + premium
                        n_trades += 1
                        if (premium + share_pnl) > 0: wins += 1
                        trade_log.append({"date": str(date.date()), "ticker": ticker,
                                         "action": "called_away", "pnl": premium + share_pnl})
                        tickers_to_remove.append(ticker)
                    else:
                        realized = pos.open_price * 100 * pos.contracts - COST_PER_CONTRACT * pos.contracts
                        cash += realized
                        n_trades += 1
                        if realized > 0: wins += 1
                        positions[ticker] = TickerPosition(
                            ticker=ticker, side="long_shares", strike=pos.share_basis,
                            expiry=date, open_date=date, open_price=pos.share_basis,
                            contracts=pos.contracts, share_basis=pos.share_basis,
                        )
                        trade_log.append({"date": str(date.date()), "ticker": ticker,
                                         "action": "cc_expired", "pnl": realized})
                    continue

        for t in tickers_to_remove:
            if t in positions:
                del positions[t]

        # ── Open new positions / sell CCs ──
        nav = cash
        for ticker, pos in positions.items():
            data = price_lookup.get((ticker, date))
            if data is None:
                continue
            S = data[0]
            if pos.side in ("long_shares", "short_call"):
                nav += S * 100 * pos.contracts

        max_per_name = nav * MAX_PER_NAME_PCT
        n_csp = sum(1 for p in positions.values() if p.side == "short_put")

        for ticker in BASKET:
            data = price_lookup.get((ticker, date))
            if data is None:
                continue
            S, sigma, vix = data

            if ticker not in positions:
                if n_csp >= MAX_ACTIVE_POSITIONS:
                    continue
                if is_bear:
                    continue
                if vix > VIX_MAX:
                    continue

                entries_attempted += 1

                # ── MEAN REVERSION GATE ──
                if signal_lookup is not None:
                    if not signal_lookup.get((ticker, date), False):
                        entries_blocked_by_signal += 1
                        continue

                expiry = find_expiry(date)
                if expiry is None:
                    continue
                T = (expiry - date).days / 365.0
                K = strike_from_delta(S, T, sigma, PUT_DELTA, kind="put")
                prem = bs_price(S, K, T, sigma, kind="put")
                slip_cost = slippage(prem)
                net_prem = prem - slip_cost

                collateral = K * 100
                if collateral > max_per_name or collateral > cash * 0.50:
                    continue
                if net_prem < 0.05:
                    continue

                max_contracts = max(1, int(max_per_name / collateral))
                contracts = min(max_contracts, 1)
                credit = net_prem * 100 * contracts - COST_PER_CONTRACT * contracts
                cash += credit
                total_premium += net_prem * 100 * contracts
                positions[ticker] = TickerPosition(
                    ticker=ticker, side="short_put", strike=K, expiry=expiry,
                    open_date=date, open_price=net_prem, contracts=contracts,
                )
                n_csp += 1

            elif positions[ticker].side == "long_shares":
                pos = positions[ticker]
                expiry = find_expiry(date)
                if expiry is None:
                    continue
                T = (expiry - date).days / 365.0
                K = strike_from_delta(S, T, sigma, CALL_DELTA, kind="call")
                prem = bs_price(S, K, T, sigma, kind="call")
                slip_cost = slippage(prem)
                net_prem = prem - slip_cost
                if net_prem < 0.05:
                    continue

                credit = net_prem * 100 * pos.contracts - COST_PER_CONTRACT * pos.contracts
                cash += credit
                total_premium += net_prem * 100 * pos.contracts
                positions[ticker] = TickerPosition(
                    ticker=ticker, side="short_call", strike=K, expiry=expiry,
                    open_date=date, open_price=net_prem, contracts=pos.contracts,
                    share_basis=pos.share_basis,
                )

        # ── Portfolio equity ──
        equity = cash
        for ticker, pos in positions.items():
            data = price_lookup.get((ticker, date))
            if data is None:
                continue
            S, sigma, _ = data
            T = max((pos.expiry - date).days, 0) / 365.0
            if pos.side == "short_put":
                opt = bs_price(S, pos.strike, T, sigma, kind="put")
                equity += (pos.open_price - opt) * 100 * pos.contracts
            elif pos.side == "long_shares":
                equity += (S - pos.share_basis) * 100 * pos.contracts
            elif pos.side == "short_call":
                opt = bs_price(S, pos.strike, T, sigma, kind="call")
                equity += (S - pos.share_basis) * 100 * pos.contracts
                equity += (pos.open_price - opt) * 100 * pos.contracts

        equity_series.append({"date": date, "equity": equity, "is_bear": is_bear})
        daily_ret = (equity - prev_equity) / prev_equity if prev_equity > 0 else 0
        daily_pnl.append(daily_ret)
        prev_equity = equity

    return {
        "equity_series": equity_series,
        "daily_returns": np.array(daily_pnl[1:]),  # skip first day
        "n_trades": n_trades,
        "wins": wins,
        "total_premium": total_premium,
        "entries_attempted": entries_attempted,
        "entries_blocked": entries_blocked_by_signal,
        "trade_log": trade_log,
    }


def compute_metrics(daily_rets, equity_series, label=""):
    """Compute risk-adjusted metrics from daily returns."""
    rets = daily_rets[np.isfinite(daily_rets)]
    if len(rets) < 20:
        return {"label": label, "error": "too few returns"}

    eq_df = pd.DataFrame(equity_series)
    years = (eq_df["date"].iloc[-1] - eq_df["date"].iloc[0]).days / 365.25
    total_ret = eq_df["equity"].iloc[-1] / eq_df["equity"].iloc[0] - 1.0
    ann_ret = (1 + total_ret) ** (1.0 / max(years, 0.01)) - 1.0

    mu = np.mean(rets)
    sd = np.std(rets, ddof=1)
    sharpe = (mu / sd) * np.sqrt(TRADING_DAYS) if sd > 0 else np.nan

    down_rets = rets[rets < 0]
    downside = np.std(down_rets, ddof=1) if len(down_rets) > 1 else np.nan
    sortino = (mu / downside) * np.sqrt(TRADING_DAYS) if downside and downside > 0 else np.nan

    peak = np.maximum.accumulate(eq_df["equity"].values)
    dd = eq_df["equity"].values / peak - 1.0
    max_dd = float(np.min(dd))
    calmar = ann_ret / abs(max_dd) if max_dd != 0 else np.nan

    wr = float(np.mean(rets > 0))

    # Regime stratification
    bull_mask = ~eq_df["is_bear"].values[1:]
    bear_mask = eq_df["is_bear"].values[1:]
    # Align with rets (which skips first day)
    if len(bull_mask) > len(rets):
        bull_mask = bull_mask[:len(rets)]
        bear_mask = bear_mask[:len(rets)]

    bull_rets = rets[bull_mask] if bull_mask.any() else np.array([])
    bear_rets = rets[bear_mask] if bear_mask.any() else np.array([])

    bull_sharpe = (np.mean(bull_rets) / np.std(bull_rets, ddof=1) * np.sqrt(252)) if len(bull_rets) > 20 else np.nan
    bear_sharpe = (np.mean(bear_rets) / np.std(bear_rets, ddof=1) * np.sqrt(252)) if len(bear_rets) > 20 else np.nan

    if np.isfinite(bull_sharpe) and np.isfinite(bear_sharpe) and max(abs(bull_sharpe), abs(bear_sharpe)) > 0:
        regime_gap = abs(bull_sharpe - bear_sharpe) / max(abs(bull_sharpe), abs(bear_sharpe))
    else:
        regime_gap = np.nan

    return {
        "label": label,
        "years": years,
        "total_return_pct": total_ret * 100,
        "ann_return_pct": ann_ret * 100,
        "sharpe": float(sharpe),
        "sortino": float(sortino) if np.isfinite(sortino) else None,
        "max_dd_pct": max_dd * 100,
        "calmar": float(calmar) if np.isfinite(calmar) else None,
        "win_rate": wr,
        "bull_sharpe": float(bull_sharpe) if np.isfinite(bull_sharpe) else None,
        "bear_sharpe": float(bear_sharpe) if np.isfinite(bear_sharpe) else None,
        "regime_gap": float(regime_gap) if np.isfinite(regime_gap) else None,
        "ending_equity": float(eq_df["equity"].iloc[-1]),
    }


def run_permutation_test(prices_basket, spy_regime, all_dates, price_lookup,
                         bear_lookup, signal_df, mr_sharpe, n_perms=N_PERMUTATIONS):
    """
    Shuffle signal dates within each ticker (preserve signal frequency, break timing).
    Count how often random timing achieves >= observed Sharpe.
    """
    print(f"\n[permutation] Running {n_perms} permutations ...")
    rng = np.random.default_rng(42)
    better_count = 0
    perm_sharpes = []

    # Get all signal dates per ticker for fast shuffling
    combo_dates_by_ticker = {}
    for ticker in BASKET:
        tk_signals = signal_df[signal_df["ticker"] == ticker]
        active_dates = tk_signals[tk_signals["combo_signal"] > 0]["date"].values
        all_tk_dates = tk_signals["date"].values
        combo_dates_by_ticker[ticker] = (all_tk_dates, len(active_dates))

    for perm_i in range(n_perms):
        if perm_i % 100 == 0:
            print(f"  permutation {perm_i}/{n_perms} ...")

        # Create shuffled signal lookup
        shuffled_lookup = {}
        for ticker in BASKET:
            all_tk_dates, n_active = combo_dates_by_ticker[ticker]
            if len(all_tk_dates) == 0 or n_active == 0:
                continue
            # Randomly pick n_active dates from all dates
            chosen_idx = rng.choice(len(all_tk_dates), size=min(n_active, len(all_tk_dates)),
                                    replace=False)
            for idx in chosen_idx:
                shuffled_lookup[(ticker, all_tk_dates[idx])] = True

        result = run_backtest(prices_basket, spy_regime, all_dates, price_lookup,
                             bear_lookup, signal_lookup=shuffled_lookup,
                             label=f"perm_{perm_i}")
        rets = result["daily_returns"]
        rets = rets[np.isfinite(rets)]
        if len(rets) > 20:
            mu = np.mean(rets)
            sd = np.std(rets, ddof=1)
            perm_sharpe = (mu / sd) * np.sqrt(TRADING_DAYS) if sd > 0 else 0
        else:
            perm_sharpe = 0
        perm_sharpes.append(perm_sharpe)
        if perm_sharpe >= mr_sharpe:
            better_count += 1

    p_value = better_count / n_perms
    return p_value, perm_sharpes


def main():
    t0 = time.time()

    # ── Load data ──
    prices_all, prices_basket, spy_regime, all_dates = load_data()

    # ── Compute signals ──
    print("[signals] Computing mean-reversion indicators ...")
    signal_df = compute_signals(prices_all[prices_all["ticker"].isin(set(BASKET))])

    # Signal statistics
    total_obs = len(signal_df)
    rsi_fire = signal_df["rsi_signal"].sum()
    bb_fire = signal_df["bb_signal"].sum()
    ret_fire = signal_df["ret_signal"].sum()
    combo_fire = signal_df["combo_signal"].sum()

    print(f"\n  Signal frequencies (total obs: {total_obs:,}):")
    print(f"    RSI(5)<30:           {rsi_fire:,.0f} ({rsi_fire/total_obs*100:.1f}%)")
    print(f"    Price < BB_lower:    {bb_fire:,.0f} ({bb_fire/total_obs*100:.1f}%)")
    print(f"    5d ret < -2σ(60d):   {ret_fire:,.0f} ({ret_fire/total_obs*100:.1f}%)")
    print(f"    Combo (≥2 of 3):     {combo_fire:,.0f} ({combo_fire/total_obs*100:.1f}%)")

    # Build lookups
    price_lookup = {}
    for _, row in prices_basket.iterrows():
        price_lookup[(row["ticker"], row["date"])] = (row["close"], row["sigma"], row["vix"])

    bear_lookup = {}
    for _, row in spy_regime.iterrows():
        bear_lookup[row["date"]] = row["bear"]

    # Signal lookup: (ticker, date) -> True when combo fires
    signal_lookup = {}
    for _, row in signal_df.iterrows():
        if row["combo_signal"] > 0:
            signal_lookup[(row["ticker"], row["date"])] = True

    # ── OOS walk-forward split ──
    # Use first WF_WARMUP days for indicator warmup, rest is OOS
    oos_start_idx = WF_WARMUP
    if oos_start_idx >= len(all_dates):
        print("ERROR: Not enough data for warmup period")
        return
    oos_dates = all_dates[oos_start_idx:]
    print(f"\n[walkforward] Warmup: {WF_WARMUP} days, OOS: {len(oos_dates)} days")
    print(f"  OOS period: {oos_dates[0].date()} to {oos_dates[-1].date()}")

    # ── Run baseline (no signal filter) ──
    print("\n[baseline] Running baseline backtest (no MR filter) ...")
    baseline = run_backtest(prices_basket, spy_regime, oos_dates, price_lookup,
                           bear_lookup, signal_lookup=None, label="baseline")
    baseline_metrics = compute_metrics(baseline["daily_returns"], baseline["equity_series"],
                                       label="Baseline (V5)")

    # ── Run MR-timed (combo signal required) ──
    print("[mr_timed] Running MR-timed backtest (combo ≥2/3 required) ...")
    mr_timed = run_backtest(prices_basket, spy_regime, oos_dates, price_lookup,
                           bear_lookup, signal_lookup=signal_lookup, label="mr_timed")
    mr_metrics = compute_metrics(mr_timed["daily_returns"], mr_timed["equity_series"],
                                label="MR-Timed")

    # ── Individual signals ──
    print("[individual] Running individual signal backtests ...")
    individual_results = {}
    for sig_name, sig_col in [("RSI_only", "rsi_signal"),
                               ("BB_only", "bb_signal"),
                               ("RetDip_only", "ret_signal")]:
        sig_lk = {}
        for _, row in signal_df.iterrows():
            if row[sig_col] > 0:
                sig_lk[(row["ticker"], row["date"])] = True
        result = run_backtest(prices_basket, spy_regime, oos_dates, price_lookup,
                             bear_lookup, signal_lookup=sig_lk, label=sig_name)
        metrics = compute_metrics(result["daily_returns"], result["equity_series"],
                                 label=sig_name)
        individual_results[sig_name] = {
            "metrics": metrics,
            "n_trades": result["n_trades"],
            "entries_attempted": result["entries_attempted"],
            "entries_blocked": result["entries_blocked"],
        }

    # ── Permutation test ──
    mr_sharpe = mr_metrics["sharpe"]
    p_value, perm_sharpes = run_permutation_test(
        prices_basket, spy_regime, oos_dates, price_lookup, bear_lookup,
        signal_df, mr_sharpe, n_perms=N_PERMUTATIONS
    )

    elapsed = time.time() - t0

    # ── Print results ──
    print(f"\n{'='*75}")
    print(f"MEAN-REVERSION ENTRY TIMING OVERLAY — OOS RESULTS")
    print(f"{'='*75}")

    print(f"\nOOS Period: {oos_dates[0].date()} to {oos_dates[-1].date()} "
          f"({baseline_metrics.get('years', 0):.1f} years)")

    print(f"\n{'─'*75}")
    print(f"{'Metric':<25} {'Baseline':>12} {'MR-Timed':>12} {'Delta':>12}")
    print(f"{'─'*75}")

    for key, fmt in [("sharpe", ".2f"), ("sortino", ".2f"), ("ann_return_pct", ".1f"),
                      ("max_dd_pct", ".1f"), ("win_rate", ".3f"),
                      ("bull_sharpe", ".2f"), ("bear_sharpe", ".2f"),
                      ("regime_gap", ".3f")]:
        bv = baseline_metrics.get(key)
        mv = mr_metrics.get(key)
        bv_s = f"{bv:{fmt}}" if bv is not None else "N/A"
        mv_s = f"{mv:{fmt}}" if mv is not None else "N/A"
        if bv is not None and mv is not None:
            delta = mv - bv
            delta_s = f"{delta:+{fmt}}"
        else:
            delta_s = "N/A"
        print(f"  {key:<23} {bv_s:>12} {mv_s:>12} {delta_s:>12}")

    print(f"\n  {'Trades':<23} {baseline['n_trades']:>12} {mr_timed['n_trades']:>12}")
    print(f"  {'Entries attempted':<23} {baseline['entries_attempted']:>12} {mr_timed['entries_attempted']:>12}")
    print(f"  {'Entries blocked':<23} {baseline['entries_blocked']:>12} {mr_timed['entries_blocked']:>12}")

    print(f"\n{'─'*75}")
    print(f"INDIVIDUAL SIGNAL BACKTESTS:")
    print(f"{'─'*75}")
    print(f"  {'Signal':<20} {'Sharpe':>8} {'Sortino':>8} {'CAGR%':>8} {'MaxDD%':>8} {'WR':>8} {'Trades':>8}")
    for sig_name, sig_data in individual_results.items():
        m = sig_data["metrics"]
        print(f"  {sig_name:<20} {m.get('sharpe', 0):>8.2f} "
              f"{(m.get('sortino') or 0):>8.2f} "
              f"{m.get('ann_return_pct', 0):>8.1f} "
              f"{m.get('max_dd_pct', 0):>8.1f} "
              f"{m.get('win_rate', 0):>8.3f} "
              f"{sig_data['n_trades']:>8}")

    print(f"\n{'─'*75}")
    print(f"PERMUTATION TEST (n={N_PERMUTATIONS}):")
    print(f"{'─'*75}")
    print(f"  MR-Timed Sharpe:     {mr_sharpe:.3f}")
    print(f"  Perm mean Sharpe:    {np.mean(perm_sharpes):.3f}")
    print(f"  Perm std Sharpe:     {np.std(perm_sharpes):.3f}")
    print(f"  Perm p5/p50/p95:     {np.percentile(perm_sharpes, 5):.3f} / "
          f"{np.percentile(perm_sharpes, 50):.3f} / {np.percentile(perm_sharpes, 95):.3f}")
    print(f"  p-value:             {p_value:.4f} ({'SIGNIFICANT' if p_value < 0.05 else 'NOT significant'})")

    # Regime gap check
    regime_gap = mr_metrics.get("regime_gap")
    regime_pass = regime_gap is not None and regime_gap < 0.50
    print(f"\n  Regime gap (MR):     {regime_gap:.3f} ({'PASS' if regime_pass else 'FAIL'} < 0.50)")

    # Overall assessment
    print(f"\n{'='*75}")
    improvement = (mr_metrics.get("sharpe", 0) > baseline_metrics.get("sharpe", 0)) and p_value < 0.05
    if improvement:
        print(f"VERDICT: MR timing IMPROVES Sharpe by "
              f"{mr_metrics['sharpe'] - baseline_metrics['sharpe']:+.2f} (p={p_value:.4f})")
    else:
        if mr_metrics.get("sharpe", 0) <= baseline_metrics.get("sharpe", 0):
            print(f"VERDICT: MR timing does NOT improve Sharpe "
                  f"(baseline {baseline_metrics.get('sharpe', 0):.2f} vs MR {mr_metrics.get('sharpe', 0):.2f})")
        else:
            print(f"VERDICT: MR timing improves Sharpe numerically "
                  f"({baseline_metrics.get('sharpe', 0):.2f} -> {mr_metrics.get('sharpe', 0):.2f}) "
                  f"but NOT statistically significant (p={p_value:.4f})")
    print(f"{'='*75}")
    print(f"\nRuntime: {elapsed:.1f}s")

    # ── Save results ──
    summary = {
        "config": {
            "basket": BASKET,
            "put_delta": PUT_DELTA,
            "call_delta": CALL_DELTA,
            "dte_target": DTE_TARGET,
            "profit_take": PROFIT_TAKE,
            "vix_max": VIX_MAX,
            "signals": {
                "rsi_period": RSI_PERIOD,
                "rsi_threshold": RSI_THRESHOLD,
                "bb_period": BB_PERIOD,
                "bb_std": BB_STD,
                "ret_lookback": RET_LOOKBACK,
                "ret_vol_lookback": RET_VOL_LOOKBACK,
                "ret_threshold_sigma": RET_THRESHOLD_SIGMA,
                "combo_min_signals": COMBO_MIN_SIGNALS,
            },
        },
        "oos_period": f"{oos_dates[0].date()} to {oos_dates[-1].date()}",
        "baseline": baseline_metrics,
        "mr_timed": mr_metrics,
        "mr_timed_trade_stats": {
            "n_trades": mr_timed["n_trades"],
            "entries_attempted": mr_timed["entries_attempted"],
            "entries_blocked": mr_timed["entries_blocked"],
        },
        "individual_signals": {k: v["metrics"] for k, v in individual_results.items()},
        "permutation_test": {
            "n_permutations": N_PERMUTATIONS,
            "mr_sharpe": float(mr_sharpe),
            "perm_mean_sharpe": float(np.mean(perm_sharpes)),
            "perm_std_sharpe": float(np.std(perm_sharpes)),
            "p_value": float(p_value),
            "significant_at_5pct": p_value < 0.05,
        },
        "verdict": "IMPROVES" if improvement else "NO_IMPROVEMENT",
    }

    with open(OUT_DIR / "mr_overlay_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    # Save equity curves
    pd.DataFrame(baseline["equity_series"]).to_csv(OUT_DIR / "baseline_equity.csv", index=False)
    pd.DataFrame(mr_timed["equity_series"]).to_csv(OUT_DIR / "mr_timed_equity.csv", index=False)

    # Save signal data for analysis
    signal_df.to_parquet(OUT_DIR / "signal_data.parquet", index=False)

    # Save trade logs
    pd.DataFrame(baseline["trade_log"]).to_csv(OUT_DIR / "baseline_trades.csv", index=False)
    pd.DataFrame(mr_timed["trade_log"]).to_csv(OUT_DIR / "mr_timed_trades.csv", index=False)

    # Save permutation distribution
    np.save(OUT_DIR / "perm_sharpes.npy", np.array(perm_sharpes))

    print(f"\n[saved] All results saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
