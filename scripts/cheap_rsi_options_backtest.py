#!/usr/bin/env python3
"""
Cheap Stock RSI Options Backtest
================================
Applies the proven Adaptive RSI E (Vol-Regime Bucketed) strategy to cheap growth
stocks, using OPTIONS instead of shares for leverage.

Strategy: When RSI signal fires on a cheap stock (<$50), buy call options.
  - Low vol (<20%): RSI<15, hold 15d
  - Med vol (20-40%): RSI<20, hold 10d
  - High vol (>40%): RSI<30, hold 5d
  - Entry requires price > 200-SMA
  - Exit when RSI > 50 or max hold reached

Variants:
  A) ATM Calls 2-week DTE
  B) 5% OTM Calls (cheaper, more leverage)
  C) Deep ITM Calls (delta ~0.8, most like shares)
  D) Call Spreads (ATM buy, 10% OTM sell)
  E) Shares-only baseline (same RSI, same universe)
  F) Combined: shares on >$30 stocks, options on <$30

5-Gate: Sharpe>0.5, Perm p<0.05, Regime gap<0.5, MaxDD>-50%, >=20 trades.
OOT: Jan 2022 - Jul 2026. $669 account.
"""

import json
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
from pathlib import Path
from scipy.stats import norm

warnings.filterwarnings("ignore")
np.random.seed(42)

# ── Configuration ─────────────────────────────────────────────────────────
CAPITAL = 669.0
START = "2020-01-01"
END = "2026-07-30"
OOT_START = "2022-01-01"
N_PERM = 1000

# Cheap growth stock universe (stock price <$50 as of mid-2024 reference)
TICKERS = ["SOFI", "SNAP", "PLTR", "RBLX", "RIVN", "PINS", "LYFT", "NU"]

# Options cost parameters
OPTION_SPREAD_COST_PCT = 0.05   # 5% of premium each way (10% RT)
OPTION_COMMISSION = 0.65         # per contract per leg
MAX_POSITION_SIZE = 200.0        # max $200 per trade
STOP_LOSS_PCT = 0.50             # 50% loss on premium = stop out

ALL_TICKERS = sorted(set(TICKERS + ["SPY", "^VIX"]))

# ── Data Download ─────────────────────────────────────────────────────────
print("Downloading price data ...")
raw = yf.download(ALL_TICKERS, start=START, end=END,
                  group_by="ticker", auto_adjust=True, progress=False)


def get_close(ticker):
    try:
        if len(ALL_TICKERS) == 1:
            s = raw["Close"].dropna()
        else:
            s = raw[ticker]["Close"].dropna()
        if isinstance(s, pd.DataFrame):
            s = s.iloc[:, 0]
        return s
    except Exception:
        return pd.Series(dtype=float)


closes = {t: get_close(t) for t in ALL_TICKERS}
spy_close = closes.get("SPY", pd.Series(dtype=float))

loaded = sum(1 for t in TICKERS if len(closes.get(t, [])) > 252)
print(f"  Tickers with sufficient data: {loaded}/{len(TICKERS)}")
print(f"  SPY rows: {len(spy_close)}")


# ── Indicator Helpers ─────────────────────────────────────────────────────
def calc_rsi(series, period):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def sma(series, period):
    return series.rolling(period).mean()


def realized_vol(series, window=21):
    log_ret = np.log(series / series.shift(1))
    return log_ret.rolling(window).std() * np.sqrt(252)


# ── Black-Scholes Option Pricing ──────────────────────────────────────────
def bs_call_price(S, K, T, r, sigma):
    """Black-Scholes call price. T in years, sigma annualized."""
    if T <= 0 or sigma <= 0 or S <= 0:
        return max(S - K, 0.0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def bs_call_delta(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0:
        return 1.0 if S > K else 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return float(norm.cdf(d1))


# ── Pre-compute indicators ───────────────────────────────────────────────
indicators = {}
for t in TICKERS:
    c = closes.get(t, pd.Series(dtype=float))
    if len(c) < 300:
        print(f"  Skipping {t}: only {len(c)} bars")
        continue
    ind = pd.DataFrame(index=c.index)
    ind["close"] = c
    ind["sma200"] = sma(c, 200)
    ind["rsi5"] = calc_rsi(c, 5)
    ind["above_200sma"] = c > ind["sma200"]
    ind["vol21"] = realized_vol(c, 21)
    ind["vol_median"] = ind["vol21"].rolling(252, min_periods=63).median()
    indicators[t] = ind.dropna(subset=["sma200", "vol_median"])

spy_sma200 = sma(spy_close, 200)
spy_regime = (spy_close > spy_sma200).reindex(spy_close.index).fillna(False)

print(f"  Tickers with indicators: {len(indicators)}")


# ── Signal Generator: Adaptive RSI E (Vol-Regime Bucketed) ────────────────
def gen_signals(tickers):
    """
    Vol-Regime Bucketed RSI signals.
    Returns: list of (date, ticker, entry_price, vol_regime_tag, exit_rsi_thresh, max_hold)
    """
    signals = []
    for t in tickers:
        ind = indicators.get(t)
        if ind is None:
            continue
        for i in range(len(ind)):
            dt = ind.index[i]
            if str(dt.date()) < OOT_START:
                continue
            if not ind["above_200sma"].iloc[i]:
                continue
            vol = ind["vol21"].iloc[i]
            if pd.isna(vol):
                continue

            if vol < 0.20:
                entry_thresh, max_hold = 15.0, 15
            elif vol < 0.40:
                entry_thresh, max_hold = 20.0, 10
            else:
                entry_thresh, max_hold = 30.0, 5

            rsi_val = ind["rsi5"].iloc[i]
            if rsi_val < entry_thresh:
                signals.append((dt, t, float(ind["close"].iloc[i]),
                                f"vol_{vol:.0%}", 50, max_hold, float(vol)))
    return sorted(signals, key=lambda x: x[0])


# ── Backtest Engine: Shares ──────────────────────────────────────────────
def run_backtest_shares(signals, capital=CAPITAL, max_pos_size=MAX_POSITION_SIZE):
    """Shares-only backtest. Robinhood: $0 commission, 0.02% slippage."""
    if not signals:
        return np.array([]), [], []

    SLIPPAGE = 0.0002
    equity = capital
    equity_curve = [(OOT_START, capital)]
    realized = []

    # No concurrent position overlap (simplify: sequential trades)
    last_exit = "2000-01-01"

    for sig in signals:
        date, ticker, entry_price, vol_tag, exit_rsi_thresh, max_hold, vol = sig
        date_str = str(date.date()) if hasattr(date, 'date') else str(date)
        if date_str < OOT_START or date_str < last_exit:
            continue

        ind = indicators.get(ticker)
        if ind is None or date not in ind.index:
            continue
        loc = ind.index.get_loc(date)

        # Find exit
        exit_idx = None
        for i in range(loc + 1, min(loc + max_hold + 1, len(ind))):
            if ind["rsi5"].iloc[i] > exit_rsi_thresh:
                exit_idx = i
                break
        if exit_idx is None:
            exit_idx = min(loc + max_hold, len(ind) - 1)
        if exit_idx <= loc:
            exit_idx = min(loc + 1, len(ind) - 1)

        exit_date_str = str(ind.index[exit_idx].date())

        actual_entry = entry_price * (1 + SLIPPAGE)
        exit_price = float(ind["close"].iloc[exit_idx]) * (1 - SLIPPAGE)

        # Position size: min(max_pos_size, equity)
        alloc = min(max_pos_size, equity)
        shares = int(alloc / actual_entry)
        if shares < 1:
            if actual_entry <= equity:
                shares = 1
            else:
                continue

        pnl = shares * (exit_price - actual_entry)
        ret = pnl / (shares * actual_entry)
        hold_days = (ind.index[exit_idx] - ind.index[loc]).days
        regime = "Bull" if spy_regime.get(ind.index[loc], False) else "Bear"

        trade = {
            "ticker": ticker, "entry_date": date_str, "exit_date": exit_date_str,
            "entry_price": round(actual_entry, 4), "exit_price": round(exit_price, 4),
            "shares": shares, "pnl": round(pnl, 2), "return": round(ret, 6),
            "hold_days": hold_days, "regime": regime, "type": "shares",
        }

        equity += pnl
        equity_curve.append((exit_date_str, round(equity, 2)))
        realized.append(trade)
        last_exit = exit_date_str

    returns = np.array([t["return"] for t in realized])
    return returns, equity_curve, realized


# ── Backtest Engine: Options ─────────────────────────────────────────────
def run_backtest_options(signals, capital=CAPITAL, option_type="atm",
                         max_pos_size=MAX_POSITION_SIZE):
    """
    Options backtest using Black-Scholes pricing.

    option_type: "atm", "otm5", "ditm", "spread"
      - atm: ATM call, strike = stock price
      - otm5: 5% OTM call, strike = 1.05 * stock price
      - ditm: Deep ITM call, strike = 0.90 * stock price (delta ~0.8)
      - spread: Buy ATM call, sell 10% OTM call
    """
    if not signals:
        return np.array([]), [], []

    RISK_FREE = 0.05  # ~5% during most of OOT period
    equity = capital
    equity_curve = [(OOT_START, capital)]
    realized = []
    last_exit = "2000-01-01"

    for sig in signals:
        date, ticker, entry_price, vol_tag, exit_rsi_thresh, max_hold, vol = sig
        date_str = str(date.date()) if hasattr(date, 'date') else str(date)
        if date_str < OOT_START or date_str < last_exit:
            continue

        ind = indicators.get(ticker)
        if ind is None or date not in ind.index:
            continue
        loc = ind.index.get_loc(date)

        S = entry_price
        sigma = max(vol, 0.15)  # floor vol at 15%

        # DTE: use ~14 calendar days for 2-week options
        DTE_days = 14
        T_entry = DTE_days / 365.0

        # Strike selection
        if option_type == "atm":
            K = round(S, 0)  # ATM
        elif option_type == "otm5":
            K = round(S * 1.05, 0)  # 5% OTM
        elif option_type == "ditm":
            K = round(S * 0.90, 0)  # Deep ITM
        elif option_type == "spread":
            K_long = round(S, 0)
            K_short = round(S * 1.10, 0)
        else:
            K = round(S, 0)

        # Price the option at entry
        if option_type == "spread":
            long_premium = bs_call_price(S, K_long, T_entry, RISK_FREE, sigma)
            short_premium = bs_call_price(S, K_short, T_entry, RISK_FREE, sigma)
            net_premium = long_premium - short_premium
            if net_premium <= 0.05:
                continue
        else:
            premium = bs_call_price(S, K, T_entry, RISK_FREE, sigma)
            if premium <= 0.05:
                continue

        # Cost per contract (100 shares per contract)
        if option_type == "spread":
            cost_per_contract = net_premium * 100
        else:
            cost_per_contract = premium * 100

        # Add bid-ask spread cost (5% of premium on entry)
        entry_cost_per_contract = cost_per_contract * (1 + OPTION_SPREAD_COST_PCT)
        entry_cost_per_contract += OPTION_COMMISSION  # $0.65 per leg

        if option_type == "spread":
            entry_cost_per_contract += OPTION_COMMISSION  # second leg

        # How many contracts can we buy?
        alloc = min(max_pos_size, equity)
        n_contracts = max(1, int(alloc / entry_cost_per_contract))
        # Cap at 2 contracts
        n_contracts = min(n_contracts, 2)

        total_entry_cost = n_contracts * entry_cost_per_contract
        if total_entry_cost > equity:
            if entry_cost_per_contract <= equity:
                n_contracts = 1
                total_entry_cost = entry_cost_per_contract
            else:
                continue  # can't afford even 1 contract

        # Find exit point
        exit_idx = None
        stop_triggered = False
        for i in range(loc + 1, min(loc + max_hold + 1, len(ind))):
            stock_price_now = float(ind["close"].iloc[i])
            days_elapsed = (ind.index[i] - ind.index[loc]).days
            T_remaining = max((DTE_days - days_elapsed) / 365.0, 0.001)

            # Price option at this point
            if option_type == "spread":
                long_val = bs_call_price(stock_price_now, K_long, T_remaining, RISK_FREE, sigma)
                short_val = bs_call_price(stock_price_now, K_short, T_remaining, RISK_FREE, sigma)
                current_val_per_share = long_val - short_val
            else:
                current_val_per_share = bs_call_price(stock_price_now, K, T_remaining, RISK_FREE, sigma)

            current_val_per_contract = current_val_per_share * 100

            # Check stop loss (50% of premium lost)
            if option_type == "spread":
                orig_premium_per_contract = net_premium * 100
            else:
                orig_premium_per_contract = premium * 100

            if current_val_per_contract < orig_premium_per_contract * (1 - STOP_LOSS_PCT):
                exit_idx = i
                stop_triggered = True
                break

            # RSI exit
            if ind["rsi5"].iloc[i] > exit_rsi_thresh:
                exit_idx = i
                break

        if exit_idx is None:
            exit_idx = min(loc + max_hold, len(ind) - 1)
        if exit_idx <= loc:
            exit_idx = min(loc + 1, len(ind) - 1)

        exit_date_str = str(ind.index[exit_idx].date())

        # Price option at exit
        stock_price_exit = float(ind["close"].iloc[exit_idx])
        days_elapsed = (ind.index[exit_idx] - ind.index[loc]).days
        T_remaining = max((DTE_days - days_elapsed) / 365.0, 0.001)

        if option_type == "spread":
            long_exit = bs_call_price(stock_price_exit, K_long, T_remaining, RISK_FREE, sigma)
            short_exit = bs_call_price(stock_price_exit, K_short, T_remaining, RISK_FREE, sigma)
            exit_val_per_share = long_exit - short_exit
        else:
            exit_val_per_share = bs_call_price(stock_price_exit, K, T_remaining, RISK_FREE, sigma)

        exit_val_per_contract = exit_val_per_share * 100

        # Subtract exit spread cost and commission
        exit_proceeds_per_contract = exit_val_per_contract * (1 - OPTION_SPREAD_COST_PCT)
        exit_proceeds_per_contract -= OPTION_COMMISSION
        if option_type == "spread":
            exit_proceeds_per_contract -= OPTION_COMMISSION

        total_exit_proceeds = n_contracts * max(exit_proceeds_per_contract, 0)
        pnl = total_exit_proceeds - total_entry_cost
        ret = pnl / total_entry_cost if total_entry_cost > 0 else 0.0
        hold_days = days_elapsed
        regime = "Bull" if spy_regime.get(ind.index[loc], False) else "Bear"

        # Effective leverage
        stock_ret = (stock_price_exit - S) / S
        leverage = ret / stock_ret if abs(stock_ret) > 0.001 else 0.0

        trade = {
            "ticker": ticker, "entry_date": date_str, "exit_date": exit_date_str,
            "stock_entry": round(S, 4), "stock_exit": round(stock_price_exit, 4),
            "stock_return": round(stock_ret, 6),
            "option_type": option_type,
            "strike": round(K, 2) if option_type != "spread" else f"{K_long}/{K_short}",
            "entry_premium": round(cost_per_contract, 2),
            "exit_premium": round(exit_val_per_contract, 2),
            "contracts": n_contracts,
            "total_cost": round(total_entry_cost, 2),
            "total_proceeds": round(total_exit_proceeds, 2),
            "pnl": round(pnl, 2), "return": round(ret, 6),
            "leverage": round(leverage, 2),
            "hold_days": hold_days, "regime": regime,
            "stop_triggered": stop_triggered,
            "type": "option",
        }

        equity += pnl
        if equity < 10:  # Account blown
            equity_curve.append((exit_date_str, round(equity, 2)))
            realized.append(trade)
            break

        equity_curve.append((exit_date_str, round(equity, 2)))
        realized.append(trade)
        last_exit = exit_date_str

    returns = np.array([t["return"] for t in realized])
    return returns, equity_curve, realized


# ── Combined Backtest: Shares >$30, Options <$30 ─────────────────────────
def run_backtest_combined(signals, capital=CAPITAL, max_pos_size=MAX_POSITION_SIZE):
    """Use shares for stocks >$30, ATM options for stocks <$30."""
    if not signals:
        return np.array([]), [], []

    SLIPPAGE = 0.0002
    RISK_FREE = 0.05
    DTE_days = 14
    equity = capital
    equity_curve = [(OOT_START, capital)]
    realized = []
    last_exit = "2000-01-01"

    for sig in signals:
        date, ticker, entry_price, vol_tag, exit_rsi_thresh, max_hold, vol = sig
        date_str = str(date.date()) if hasattr(date, 'date') else str(date)
        if date_str < OOT_START or date_str < last_exit:
            continue

        ind = indicators.get(ticker)
        if ind is None or date not in ind.index:
            continue
        loc = ind.index.get_loc(date)

        use_options = entry_price < 30.0
        S = entry_price

        if use_options:
            sigma = max(vol, 0.15)
            T_entry = DTE_days / 365.0
            K = round(S, 0)
            premium = bs_call_price(S, K, T_entry, RISK_FREE, sigma)
            if premium <= 0.05:
                continue

            cost_per_contract = premium * 100
            entry_cost = cost_per_contract * (1 + OPTION_SPREAD_COST_PCT) + OPTION_COMMISSION
            alloc = min(max_pos_size, equity)
            n_contracts = min(2, max(1, int(alloc / entry_cost)))
            total_entry_cost = n_contracts * entry_cost
            if total_entry_cost > equity:
                if entry_cost <= equity:
                    n_contracts = 1
                    total_entry_cost = entry_cost
                else:
                    continue

            # Find exit
            exit_idx = None
            stop_triggered = False
            for i in range(loc + 1, min(loc + max_hold + 1, len(ind))):
                sp = float(ind["close"].iloc[i])
                de = (ind.index[i] - ind.index[loc]).days
                T_r = max((DTE_days - de) / 365.0, 0.001)
                cv = bs_call_price(sp, K, T_r, RISK_FREE, sigma)
                if cv * 100 < premium * 100 * (1 - STOP_LOSS_PCT):
                    exit_idx = i
                    stop_triggered = True
                    break
                if ind["rsi5"].iloc[i] > exit_rsi_thresh:
                    exit_idx = i
                    break
            if exit_idx is None:
                exit_idx = min(loc + max_hold, len(ind) - 1)
            if exit_idx <= loc:
                exit_idx = min(loc + 1, len(ind) - 1)

            exit_date_str = str(ind.index[exit_idx].date())
            sp_exit = float(ind["close"].iloc[exit_idx])
            de = (ind.index[exit_idx] - ind.index[loc]).days
            T_r = max((DTE_days - de) / 365.0, 0.001)
            exit_val = bs_call_price(sp_exit, K, T_r, RISK_FREE, sigma) * 100
            exit_proceeds = n_contracts * max(exit_val * (1 - OPTION_SPREAD_COST_PCT) - OPTION_COMMISSION, 0)
            pnl = exit_proceeds - total_entry_cost
            ret = pnl / total_entry_cost if total_entry_cost > 0 else 0
            hold_days = de
            ttype = "option"

        else:
            # Shares
            actual_entry = S * (1 + SLIPPAGE)
            exit_idx = None
            for i in range(loc + 1, min(loc + max_hold + 1, len(ind))):
                if ind["rsi5"].iloc[i] > exit_rsi_thresh:
                    exit_idx = i
                    break
            if exit_idx is None:
                exit_idx = min(loc + max_hold, len(ind) - 1)
            if exit_idx <= loc:
                exit_idx = min(loc + 1, len(ind) - 1)

            exit_date_str = str(ind.index[exit_idx].date())
            exit_price = float(ind["close"].iloc[exit_idx]) * (1 - SLIPPAGE)
            alloc = min(max_pos_size, equity)
            shares = max(1, int(alloc / actual_entry))
            pnl = shares * (exit_price - actual_entry)
            ret = pnl / (shares * actual_entry)
            hold_days = (ind.index[exit_idx] - ind.index[loc]).days
            stop_triggered = False
            ttype = "shares"

        regime = "Bull" if spy_regime.get(ind.index[loc], False) else "Bear"
        trade = {
            "ticker": ticker, "entry_date": date_str, "exit_date": exit_date_str,
            "stock_price": round(S, 4), "pnl": round(pnl, 2), "return": round(ret, 6),
            "hold_days": hold_days, "regime": regime, "type": ttype,
        }

        equity += pnl
        if equity < 10:
            equity_curve.append((exit_date_str, round(equity, 2)))
            realized.append(trade)
            break
        equity_curve.append((exit_date_str, round(equity, 2)))
        realized.append(trade)
        last_exit = exit_date_str

    returns = np.array([t["return"] for t in realized])
    return returns, equity_curve, realized


# ── Metrics ───────────────────────────────────────────────────────────────
def calc_metrics(returns, equity_curve, realized, capital=CAPITAL):
    if len(returns) < 2:
        return {
            "n_trades": len(returns), "sharpe": 0.0, "sortino": 0.0,
            "profit_factor": 0.0, "win_rate": 0.0, "max_dd_pct": 0.0,
            "total_return_pct": 0.0, "cagr_pct": 0.0, "final_equity": capital,
            "avg_hold_days": 0.0, "bull_sharpe": 0.0, "bear_sharpe": 0.0,
            "regime_gap": 0.0, "bull_trades": 0, "bear_trades": 0,
            "avg_leverage": 0.0, "stops_triggered": 0,
        }

    n = len(returns)
    wins = int((returns > 0).sum())
    wr = wins / n

    avg_hold = np.mean([t["hold_days"] for t in realized]) if realized else 8.0
    trades_per_year = max(1, 252 / max(avg_hold, 1))
    ann_factor = np.sqrt(trades_per_year)

    mean_r = returns.mean()
    std_r = returns.std() if returns.std() > 0 else 1e-9
    sharpe = (mean_r / std_r) * ann_factor

    downside = returns[returns < 0]
    down_std = downside.std() if len(downside) > 0 and downside.std() > 0 else 1e-9
    sortino = (mean_r / down_std) * ann_factor

    gross_profit = returns[returns > 0].sum() if (returns > 0).any() else 0
    gross_loss = abs(returns[returns < 0].sum()) if (returns < 0).any() else 1e-9
    profit_factor = gross_profit / gross_loss

    eq_vals = [e[1] for e in equity_curve]
    peak = eq_vals[0]
    max_dd = 0
    for v in eq_vals:
        if v > peak:
            peak = v
        dd = (v - peak) / peak
        if dd < max_dd:
            max_dd = dd

    final_eq = eq_vals[-1]
    total_return_pct = ((final_eq - capital) / capital) * 100

    if realized:
        first_date = pd.Timestamp(realized[0]["entry_date"])
        last_date = pd.Timestamp(realized[-1]["exit_date"])
        years = max((last_date - first_date).days / 365.25, 0.5)
        cagr = ((final_eq / capital) ** (1 / years) - 1) * 100
    else:
        cagr = 0.0

    bull_rets = np.array([t["return"] for t in realized if t["regime"] == "Bull"])
    bear_rets = np.array([t["return"] for t in realized if t["regime"] == "Bear"])

    def regime_sharpe(rets):
        if len(rets) < 2:
            return 0.0
        s = rets.std()
        if s < 1e-12:
            return 0.0
        return float((rets.mean() / s) * ann_factor)

    bull_s = regime_sharpe(bull_rets)
    bear_s = regime_sharpe(bear_rets)
    denom = max(abs(bull_s), abs(bear_s), 1e-9)
    regime_gap = abs(bull_s - bear_s) / denom

    avg_lev = np.mean([t.get("leverage", 1.0) for t in realized])
    stops = sum(1 for t in realized if t.get("stop_triggered", False))

    return {
        "n_trades": int(n),
        "win_rate": round(float(wr), 4),
        "sharpe": round(float(sharpe), 3),
        "sortino": round(float(sortino), 3),
        "profit_factor": round(float(profit_factor), 3),
        "max_dd_pct": round(float(max_dd * 100), 2),
        "total_return_pct": round(float(total_return_pct), 2),
        "cagr_pct": round(float(cagr), 2),
        "final_equity": round(float(final_eq), 2),
        "avg_hold_days": round(float(avg_hold), 1),
        "bull_trades": int(len(bull_rets)),
        "bear_trades": int(len(bear_rets)),
        "bull_sharpe": round(float(bull_s), 3),
        "bear_sharpe": round(float(bear_s), 3),
        "regime_gap": round(float(regime_gap), 3),
        "avg_leverage": round(float(avg_lev), 2),
        "stops_triggered": int(stops),
    }


# ── Permutation Test ──────────────────────────────────────────────────────
def permutation_test(returns, signals, realized, metrics, run_fn, run_kwargs=None):
    """Shuffle entry timing 1000 times, compute p-value."""
    if len(returns) < 3 or len(signals) < 3:
        return 1.0

    real_sharpe = metrics["sharpe"]

    oot_dates_per_ticker = {}
    for t in set(s[1] for s in signals):
        ind = indicators.get(t)
        if ind is not None:
            valid = ind.index[ind.index >= OOT_START]
            if len(valid) > 15:
                oot_dates_per_ticker[t] = valid[:-10]

    signal_tickers = [s[1] for s in signals
                      if str(s[0].date()) >= OOT_START and s[1] in oot_dates_per_ticker]

    if len(signal_tickers) < 3:
        return 1.0

    perm_sharpes = np.zeros(N_PERM)
    for i in range(N_PERM):
        shuffled = []
        for ticker in signal_tickers:
            valid_dates = oot_dates_per_ticker[ticker]
            rand_date = valid_dates[np.random.randint(0, len(valid_dates))]
            ind = indicators[ticker]
            ep = ind.loc[rand_date, "close"]
            if isinstance(ep, pd.Series):
                ep = ep.iloc[0]
            vol_val = ind.loc[rand_date, "vol21"]
            if isinstance(vol_val, pd.Series):
                vol_val = vol_val.iloc[0]
            if pd.isna(vol_val):
                vol_val = 0.30
            shuffled.append((rand_date, ticker, float(ep), "perm", 50, 10, float(vol_val)))

        perm_ret, _, _ = run_fn(shuffled, **(run_kwargs or {}))
        if len(perm_ret) >= 2:
            avg_h = 8.0
            af = np.sqrt(252 / avg_h)
            s = perm_ret.std()
            perm_sharpes[i] = (perm_ret.mean() / s * af) if s > 1e-12 else 0.0
        else:
            perm_sharpes[i] = 0.0

    p_value = float(np.mean(perm_sharpes >= real_sharpe))
    return round(p_value, 4)


# ── 5-Gate Validation ─────────────────────────────────────────────────────
def five_gate_check(metrics, p_value):
    gates = {
        "G1_sharpe_gt_0.5": {"value": metrics["sharpe"], "passed": metrics["sharpe"] > 0.5},
        "G2_perm_p_lt_0.05": {"value": p_value, "passed": p_value < 0.05},
        "G3_regime_gap_lt_0.5": {"value": metrics["regime_gap"], "passed": metrics["regime_gap"] < 0.5},
        "G4_maxdd_gt_neg50": {"value": metrics["max_dd_pct"], "passed": metrics["max_dd_pct"] > -50.0},
        "G5_trades_gte_20": {"value": metrics["n_trades"], "passed": metrics["n_trades"] >= 20},
    }
    n_passed = sum(1 for g in gates.values() if g["passed"])
    verdict = ("PASS" if n_passed == 5 else
               "STRONG" if n_passed == 4 else
               "MARGINAL" if n_passed == 3 else "FAIL")
    return {"gates": gates, "passed": n_passed, "total": 5, "verdict": verdict}


# ══════════════════════════════════════════════════════════════════════════
#  RUN ALL VARIANTS
# ══════════════════════════════════════════════════════════════════════════

signals = gen_signals(TICKERS)
print(f"\nTotal RSI signals generated: {len(signals)}")
print(f"Signals by ticker:")
from collections import Counter
ticker_counts = Counter(s[1] for s in signals)
for t, c in sorted(ticker_counts.items()):
    print(f"  {t}: {c}")

results = {}

variant_configs = {
    "A_atm_calls": {
        "name": "A) ATM Calls (2-week DTE)",
        "run_fn": run_backtest_options,
        "kwargs": {"option_type": "atm"},
    },
    "B_otm5_calls": {
        "name": "B) 5% OTM Calls",
        "run_fn": run_backtest_options,
        "kwargs": {"option_type": "otm5"},
    },
    "C_ditm_calls": {
        "name": "C) Deep ITM Calls (delta ~0.8)",
        "run_fn": run_backtest_options,
        "kwargs": {"option_type": "ditm"},
    },
    "D_call_spreads": {
        "name": "D) Call Spreads (ATM/+10% OTM)",
        "run_fn": run_backtest_options,
        "kwargs": {"option_type": "spread"},
    },
    "E_shares_baseline": {
        "name": "E) Shares-Only Baseline",
        "run_fn": run_backtest_shares,
        "kwargs": {},
    },
    "F_combined": {
        "name": "F) Combined (options <$30, shares >$30)",
        "run_fn": run_backtest_combined,
        "kwargs": {},
    },
}

for key, cfg in variant_configs.items():
    print(f"\n{'='*70}")
    print(f"  {cfg['name']}")
    print(f"{'='*70}")

    returns, eq_curve, trades = cfg["run_fn"](signals, **cfg["kwargs"])
    metrics = calc_metrics(returns, eq_curve, trades)

    print(f"  Trades: {metrics['n_trades']}, Sharpe: {metrics['sharpe']}, "
          f"Sortino: {metrics['sortino']}, WR: {metrics['win_rate']:.1%}")
    print(f"  PF: {metrics['profit_factor']}, MaxDD: {metrics['max_dd_pct']:.1f}%, "
          f"Total Return: {metrics['total_return_pct']:.1f}%, CAGR: {metrics['cagr_pct']:.1f}%")
    print(f"  Final Equity: ${metrics['final_equity']:.2f} (from $669)")
    if metrics.get('avg_leverage', 0) > 0:
        print(f"  Avg Leverage: {metrics['avg_leverage']:.1f}x, "
              f"Stops Triggered: {metrics['stops_triggered']}")
    print(f"  Bull: {metrics['bull_trades']} trades (Sharpe {metrics['bull_sharpe']}), "
          f"Bear: {metrics['bear_trades']} trades (Sharpe {metrics['bear_sharpe']}), "
          f"Regime Gap: {metrics['regime_gap']:.3f}")

    print(f"  Running permutation test ({N_PERM} shuffles) ...")
    p_val = permutation_test(returns, signals, trades, metrics,
                              cfg["run_fn"], cfg["kwargs"])
    print(f"  Perm p-value: {p_val}")

    gate_result = five_gate_check(metrics, p_val)
    print(f"  5-Gate: {gate_result['passed']}/5 — {gate_result['verdict']}")
    for gname, ginfo in gate_result["gates"].items():
        status = "PASS" if ginfo["passed"] else "FAIL"
        print(f"    {gname}: {ginfo['value']} [{status}]")

    # Show some sample trades
    if trades:
        print(f"\n  Sample trades (first 5):")
        for t in trades[:5]:
            print(f"    {t['ticker']} {t['entry_date']} -> {t.get('exit_date','?')} | "
                  f"PnL ${t['pnl']:+.2f} ({t['return']:+.1%}) | "
                  f"{t.get('type','?')} | {t['regime']}")

    results[key] = {
        "name": cfg["name"],
        "metrics": metrics,
        "p_value": p_val,
        "five_gate": gate_result,
        "n_signals": len(signals),
        "sample_trades": trades[:10] if trades else [],
    }


# ── Save Results ──────────────────────────────────────────────────────────
output_path = Path("/home/jupiter/Lvl3Quant/data/cheap_rsi_options_results.json")
with open(output_path, "w") as f:
    json.dump(results, f, indent=2, default=str)
print(f"\nResults saved to {output_path}")


# ── Summary Table ─────────────────────────────────────────────────────────
print("\n" + "=" * 110)
print("  CHEAP STOCK RSI OPTIONS — SUMMARY")
print("  Universe: SOFI, SNAP, PLTR, RBLX, RIVN, PINS, LYFT, NU")
print("  Strategy: Adaptive RSI E (Vol-Regime Bucketed) | OOT: Jan 2022 - Jul 2026 | $669 account")
print("=" * 110)
print(f"{'Variant':<40} {'Trades':>6} {'Sharpe':>7} {'Sortino':>8} {'WR':>6} {'PF':>6} "
      f"{'MaxDD':>7} {'Return':>8} {'Final$':>8} {'p-val':>6} {'Gate':>5}")
print("-" * 110)

for key in ["A_atm_calls", "B_otm5_calls", "C_ditm_calls", "D_call_spreads",
            "E_shares_baseline", "F_combined"]:
    r = results[key]
    m = r["metrics"]
    g = r["five_gate"]
    print(f"{r['name']:<40} {m['n_trades']:>6} {m['sharpe']:>7.2f} {m['sortino']:>8.2f} "
          f"{m['win_rate']:>5.0%} {m['profit_factor']:>6.2f} {m['max_dd_pct']:>6.1f}% "
          f"{m['total_return_pct']:>7.1f}% ${m['final_equity']:>7.0f} {r['p_value']:>6.3f} {g['verdict']:>5}")

print("-" * 110)

# Best variant
all_keys = list(results.keys())
passing = [k for k in all_keys if results[k]["five_gate"]["passed"] == 5]
if passing:
    best = max(passing, key=lambda k: results[k]["metrics"]["total_return_pct"])
    print(f"\nBEST (all 5 gates): {results[best]['name']}")
    m = results[best]["metrics"]
    print(f"  Sharpe {m['sharpe']}, Sortino {m['sortino']}, WR {m['win_rate']:.0%}, "
          f"PF {m['profit_factor']}, MaxDD {m['max_dd_pct']:.1f}%")
    print(f"  $669 -> ${m['final_equity']:.0f} ({m['total_return_pct']:.1f}% return, "
          f"CAGR {m['cagr_pct']:.1f}%)")
else:
    strong = [k for k in all_keys if results[k]["five_gate"]["passed"] >= 4]
    if strong:
        best = max(strong, key=lambda k: results[k]["metrics"]["total_return_pct"])
        print(f"\nBEST (4+/5 gates): {results[best]['name']}")
    else:
        best = max(all_keys, key=lambda k: results[k]["five_gate"]["passed"])
        print(f"\nBEST ({results[best]['five_gate']['passed']}/5 gates): {results[best]['name']}")
    m = results[best]["metrics"]
    print(f"  Sharpe {m['sharpe']}, Sortino {m['sortino']}, WR {m['win_rate']:.0%}, "
          f"PF {m['profit_factor']}")
    print(f"  $669 -> ${m['final_equity']:.0f} ({m['total_return_pct']:.1f}%)")

# Options vs shares comparison
print("\n  OPTIONS vs SHARES COMPARISON:")
shares_ret = results["E_shares_baseline"]["metrics"]["total_return_pct"]
for key in ["A_atm_calls", "B_otm5_calls", "C_ditm_calls", "D_call_spreads", "F_combined"]:
    opt_ret = results[key]["metrics"]["total_return_pct"]
    mult = opt_ret / shares_ret if shares_ret != 0 else 0
    print(f"    {results[key]['name']:<40} {opt_ret:>+8.1f}% vs shares {shares_ret:>+.1f}% "
          f"({mult:.1f}x leverage)")

print("\nDone.")
