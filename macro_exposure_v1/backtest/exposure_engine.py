"""
exposure_engine.py — Daily backtest engine.

Inputs
------
allocation : pd.Series of {-1, -0.5, 0, +0.5, +1, +1.5}  (per date)
basket_w   : dict like {"SPY": 0.5, "QQQ": 0.3, "IWM": 0.2}  sums to 1
prices_w   : wide close panel — index date, cols include SPY/QQQ/IWM
rebalance_dates: pd.DatetimeIndex of dates we actually trade on

Costs
-----
Commission: $0   (Alpaca / Robinhood / IBKR Lite on SPY/QQQ/IWM — per HC #541 R1)
Slippage:   1 bp per leg of notional traded on rebalance days.
Borrow fee: not modeled in v1 — TODO add ~25-40 bps annualized when going live.

Outputs
-------
dict with:
  equity_curve  : pd.Series of $-equity over time, indexed by date
  daily_ret     : pd.Series of daily strategy returns (after costs)
  position_curve: pd.Series of net target exposure (daily)
  metrics       : dict with cagr, sortino, sharpe, max_dd_pct, worst_month_pct,
                  turnover_per_year, avg_leverage, pct_long, pct_flat, pct_short
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional
import numpy as np
import pandas as pd

SLIPPAGE_BPS_PER_LEG = 1.0  # 1 bp slippage per leg of notional traded


def _basket_returns(prices: pd.DataFrame, basket_w: Dict[str, float]) -> pd.Series:
    """Daily simple returns of the weighted ETF basket (rebalanced daily for valuation)."""
    rets = []
    weights = []
    for t, w in basket_w.items():
        if t in prices.columns and w > 0:
            r = prices[t].pct_change()
            rets.append(r * w)
            weights.append(w)
    if not rets:
        return pd.Series(0.0, index=prices.index)
    total_w = sum(weights) or 1.0
    br = sum(rets) / total_w
    return br.fillna(0.0)


def _drawdown(equity: pd.Series) -> pd.Series:
    peak = equity.cummax()
    return equity / peak - 1.0


@dataclass
class BacktestResult:
    equity_curve: pd.Series
    daily_ret: pd.Series
    position_curve: pd.Series
    metrics: Dict[str, float]


def run_exposure_backtest(
    allocation: pd.Series,
    basket_w: Dict[str, float],
    prices: pd.DataFrame,
    rebalance_dates: Optional[pd.DatetimeIndex] = None,
    starting_cash: float = 100_000.0,
    allow_short: bool = True,
    max_leverage: float = 1.5,
) -> BacktestResult:
    """
    Run the daily backtest.

    The allocation series is target exposure in units of NAV. We hold that target
    constant until the next rebalance date. Slippage is charged on the |delta|
    between successive held targets when we actually trade.
    """
    if not allow_short:
        allocation = allocation.clip(lower=0)
    allocation = allocation.clip(lower=-max_leverage, upper=max_leverage)

    # Normalize basket weights
    bw = {k: max(0.0, float(v)) for k, v in basket_w.items()}
    tot = sum(bw.values())
    if tot <= 0:
        bw = {"SPY": 1.0}
    else:
        bw = {k: v / tot for k, v in bw.items()}

    br = _basket_returns(prices, bw)
    idx = br.index

    allocation = allocation.reindex(idx).ffill().fillna(0.0)

    # Held position = allocation as of previous close (we execute on signal, then earn next-day return)
    if rebalance_dates is None:
        rebalance_dates = idx
    rebal_mask = pd.Series(False, index=idx)
    rebal_mask.loc[rebal_mask.index.isin(pd.DatetimeIndex(rebalance_dates))] = True

    held = pd.Series(np.nan, index=idx)
    last = 0.0
    for d, want_rebal in rebal_mask.items():
        if want_rebal:
            last = float(allocation.loc[d])
        held.loc[d] = last
    held = held.ffill().fillna(0.0)

    # Gross daily return: held_position(t-1) * basket_return(t)
    pos_lag = held.shift(1).fillna(0.0)
    gross_ret = pos_lag * br

    # Slippage: charge 1bp * |delta_position| on rebalance days
    pos_delta = held.diff().fillna(held).abs()
    slip_cost = pos_delta * (SLIPPAGE_BPS_PER_LEG / 1e4)

    net_ret = gross_ret - slip_cost

    equity = (1.0 + net_ret).cumprod() * starting_cash

    # ------ Metrics ------
    days = len(net_ret)
    yrs = max(days / 252.0, 1e-6)
    final = float(equity.iloc[-1]) if len(equity) else starting_cash
    cagr = (final / starting_cash) ** (1.0 / yrs) - 1.0
    ann_factor = 252.0
    mu = net_ret.mean() * ann_factor
    sd = net_ret.std() * np.sqrt(ann_factor)
    sharpe = mu / sd if sd > 1e-12 else 0.0
    downside = net_ret[net_ret < 0]
    dd_sd = downside.std() * np.sqrt(ann_factor) if len(downside) else 0.0
    sortino = mu / dd_sd if dd_sd > 1e-12 else 0.0
    dd = _drawdown(equity)
    max_dd = float(dd.min()) if len(dd) else 0.0
    monthly = (1.0 + net_ret).resample("ME").prod() - 1.0
    worst_month = float(monthly.min()) if len(monthly) else 0.0

    # turnover: sum of |position changes| per year
    turnover_per_year = float(pos_delta.sum() / yrs) if yrs > 0 else 0.0
    avg_leverage = float(held.abs().mean())
    pct_long = float((held > 0).mean())
    pct_flat = float((held == 0).mean())
    pct_short = float((held < 0).mean())

    metrics = dict(
        cagr=cagr,
        sortino=sortino,
        sharpe=sharpe,
        max_dd_pct=abs(max_dd) * 100.0,
        worst_month_pct=worst_month * 100.0,
        turnover_per_year=turnover_per_year,
        avg_leverage=avg_leverage,
        pct_long=pct_long,
        pct_flat=pct_flat,
        pct_short=pct_short,
        final_equity=final,
        years=yrs,
    )

    return BacktestResult(
        equity_curve=equity,
        daily_ret=net_ret,
        position_curve=held,
        metrics=metrics,
    )
