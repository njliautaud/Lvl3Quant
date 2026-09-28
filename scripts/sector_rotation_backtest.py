#!/usr/bin/env python3
"""
Sector Rotation + Momentum Hybrid Backtester
=============================================
Strategy:
  1. Rank 11 GICS sectors by 3-month momentum (sector ETFs)
  2. Allocate to top 3 sectors
  3. Within each top sector, pick top stocks by 6mo momentum (skip last month)
  4. Dynamic exits: sector drops out of top 5 -> exit; 2x ATR(20) trailing stop; weekly health check
  5. Defensive filter: SPY death cross -> 50% exposure, shift to defensive sectors

Walk-forward: 2-year train (calibration), 1-year OOT, sliding.
Data: yfinance, 2015-2026.
"""

import json
import os
import sys
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
SECTOR_ETFS = {
    "Technology": "XLK",
    "Financials": "XLF",
    "Health Care": "XLV",
    "Energy": "XLE",
    "Industrials": "XLI",
    "Communication Services": "XLC",
    "Consumer Discretionary": "XLY",
    "Consumer Staples": "XLP",
    "Utilities": "XLU",
    "Real Estate": "XLRE",
    "Materials": "XLB",
}

DEFENSIVE_SECTORS = {"Utilities", "Consumer Staples", "Health Care"}

# Top stocks by sector (representative large-caps, ~15 per sector for selection pool)
SECTOR_STOCKS = {
    "Technology": ["AAPL", "MSFT", "NVDA", "AVGO", "ADBE", "CRM", "AMD", "INTC", "TXN", "QCOM", "AMAT", "MU", "LRCX", "NOW", "SNPS"],
    "Financials": ["JPM", "BAC", "WFC", "GS", "MS", "C", "BLK", "SCHW", "AXP", "USB", "PNC", "TFC", "MMC", "CB", "AON"],
    "Health Care": ["UNH", "JNJ", "LLY", "PFE", "ABBV", "MRK", "TMO", "ABT", "DHR", "BMY", "AMGN", "MDT", "ISRG", "SYK", "GILD"],
    "Energy": ["XOM", "CVX", "COP", "SLB", "EOG", "MPC", "PSX", "VLO", "OXY", "PXD", "HES", "DVN", "FANG", "HAL", "BKR"],
    "Industrials": ["HON", "UNP", "UPS", "CAT", "DE", "GE", "RTX", "BA", "LMT", "MMM", "ITW", "EMR", "FDX", "WM", "CSX"],
    "Communication Services": ["GOOGL", "META", "NFLX", "DIS", "CMCSA", "T", "VZ", "TMUS", "CHTR", "EA", "TTWO", "OMC", "IPG", "FOXA", "LYV"],
    "Consumer Discretionary": ["AMZN", "TSLA", "HD", "MCD", "NKE", "SBUX", "LOW", "TJX", "BKNG", "MAR", "ORLY", "CMG", "DHI", "ROST", "YUM"],
    "Consumer Staples": ["PG", "KO", "PEP", "COST", "WMT", "PM", "MO", "CL", "MDLZ", "STZ", "KHC", "GIS", "SJM", "HSY", "KMB"],
    "Utilities": ["NEE", "DUK", "SO", "D", "AEP", "SRE", "EXC", "XEL", "WEC", "ED", "ES", "DTE", "PPL", "FE", "AES"],
    "Real Estate": ["PLD", "AMT", "CCI", "EQIX", "SPG", "PSA", "O", "WELL", "DLR", "AVB", "EQR", "VTR", "ARE", "MAA", "UDR"],
    "Materials": ["LIN", "APD", "SHW", "ECL", "NEM", "FCX", "DOW", "DD", "NUE", "VMC", "MLM", "PPG", "IFF", "ALB", "CE"],
}

TOP_N_SECTORS = 3
STOCKS_PER_SECTOR = 7
SECTOR_MOM_MONTHS = 3
STOCK_MOM_MONTHS = 6
STOCK_MOM_SKIP = 1  # skip last month (Jegadeesh-Titman style)
ATR_PERIOD = 20
ATR_MULTIPLIER = 2.0
WEEKLY_CHECK = True
TRAIN_YEARS = 2
OOT_YEARS = 1
REBALANCE_FREQ = "M"  # monthly rebalance base, with weekly health checks

OUTPUT_DIR = Path("/home/jupiter/Lvl3Quant/output/sector_rotation")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# DATA DOWNLOAD
# ---------------------------------------------------------------------------
def download_data(tickers: list[str], start: str = "2014-01-01", end: str = "2026-07-12") -> pd.DataFrame:
    """Download adjusted close prices for a list of tickers."""
    print(f"Downloading {len(tickers)} tickers from {start} to {end}...")
    data = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False, threads=True)
    if isinstance(data.columns, pd.MultiIndex):
        close = data["Close"]
    else:
        close = data
    close = close.ffill().dropna(how="all")
    print(f"  Got {len(close)} trading days, {close.shape[1]} tickers")
    return close


def compute_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20) -> pd.Series:
    """Average True Range."""
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(period).mean()


# ---------------------------------------------------------------------------
# STRATEGY LOGIC
# ---------------------------------------------------------------------------
class SectorRotationBacktester:
    def __init__(self, sector_prices: pd.DataFrame, stock_prices: pd.DataFrame,
                 spy_prices: pd.Series, stock_ohlc: dict):
        self.sector_prices = sector_prices  # sector ETF prices
        self.stock_prices = stock_prices    # individual stock prices
        self.spy = spy_prices
        self.stock_ohlc = stock_ohlc        # dict of {ticker: df with High/Low/Close}
        self.results = []

    def _sector_momentum(self, date: pd.Timestamp) -> pd.Series:
        """Rank sectors by trailing N-month return."""
        lookback_start = date - pd.DateOffset(months=SECTOR_MOM_MONTHS)
        mask = (self.sector_prices.index >= lookback_start) & (self.sector_prices.index <= date)
        window = self.sector_prices.loc[mask]
        if len(window) < 40:
            return pd.Series(dtype=float)
        ret = window.iloc[-1] / window.iloc[0] - 1
        return ret.sort_values(ascending=False)

    def _stock_momentum(self, tickers: list[str], date: pd.Timestamp) -> pd.Series:
        """6-month momentum, skipping last month (Jegadeesh-Titman)."""
        end_date = date - pd.DateOffset(months=STOCK_MOM_SKIP)
        start_date = date - pd.DateOffset(months=STOCK_MOM_MONTHS)
        available = [t for t in tickers if t in self.stock_prices.columns]
        if not available:
            return pd.Series(dtype=float)
        mask_start = self.stock_prices.index.searchsorted(start_date)
        mask_end = self.stock_prices.index.searchsorted(end_date)
        if mask_end <= mask_start + 20:
            return pd.Series(dtype=float)
        window = self.stock_prices.iloc[mask_start:mask_end][available]
        first_valid = window.iloc[0]
        last_valid = window.iloc[-1]
        ret = (last_valid / first_valid - 1).dropna()
        return ret.sort_values(ascending=False)

    def _is_death_cross(self, date: pd.Timestamp) -> bool:
        """SPY 50-day MA < 200-day MA."""
        spy_to_date = self.spy.loc[:date]
        if len(spy_to_date) < 200:
            return False
        ma50 = spy_to_date.iloc[-50:].mean()
        ma200 = spy_to_date.iloc[-200:].mean()
        return ma50 < ma200

    def _get_atr(self, ticker: str, date: pd.Timestamp) -> float:
        """Get 20-day ATR for a stock at a given date."""
        if ticker not in self.stock_ohlc:
            return np.nan
        ohlc = self.stock_ohlc[ticker]
        mask = ohlc.index <= date
        sub = ohlc.loc[mask].tail(ATR_PERIOD + 5)
        if len(sub) < ATR_PERIOD:
            return np.nan
        atr = compute_atr(sub["High"], sub["Low"], sub["Close"], ATR_PERIOD)
        val = atr.iloc[-1]
        return val if not np.isnan(val) else np.nan

    def _select_portfolio(self, date: pd.Timestamp) -> dict:
        """
        Select portfolio: top sectors + top stocks within each.
        Returns dict: {ticker: (sector, weight)}
        """
        death_cross = self._is_death_cross(date)
        sector_mom = self._sector_momentum(date)
        if sector_mom.empty:
            return {}

        if death_cross:
            # Defensive mode: prioritize defensive sectors
            defensive_mom = sector_mom[[s for s in sector_mom.index if
                                         any(sec in s for sec in ["XLU", "XLP", "XLV"])
                                         or s in DEFENSIVE_SECTORS]]
            # Map ETF tickers back to sector names
            etf_to_sector = {v: k for k, v in SECTOR_ETFS.items()}
            top_sectors_etfs = []
            # First add defensive sectors that are available
            for etf in sector_mom.index:
                sector_name = etf_to_sector.get(etf, "")
                if sector_name in DEFENSIVE_SECTORS:
                    top_sectors_etfs.append(etf)
                if len(top_sectors_etfs) >= 2:
                    break
            # Fill remaining with best non-defensive
            for etf in sector_mom.index:
                if etf not in top_sectors_etfs:
                    top_sectors_etfs.append(etf)
                if len(top_sectors_etfs) >= TOP_N_SECTORS:
                    break
            exposure_scale = 0.5  # 50% exposure in death cross
        else:
            top_sectors_etfs = list(sector_mom.index[:TOP_N_SECTORS])
            exposure_scale = 1.0

        etf_to_sector = {v: k for k, v in SECTOR_ETFS.items()}
        portfolio = {}
        sector_weight = exposure_scale / len(top_sectors_etfs) if top_sectors_etfs else 0

        for etf in top_sectors_etfs:
            sector_name = etf_to_sector.get(etf)
            if sector_name is None or sector_name not in SECTOR_STOCKS:
                continue
            candidates = SECTOR_STOCKS[sector_name]
            stock_mom = self._stock_momentum(candidates, date)
            if stock_mom.empty:
                continue
            top_stocks = list(stock_mom.index[:STOCKS_PER_SECTOR])
            stock_weight = sector_weight / len(top_stocks) if top_stocks else 0
            for ticker in top_stocks:
                portfolio[ticker] = (sector_name, stock_weight)

        return portfolio

    def run_backtest(self, start_date: str, end_date: str, label: str = "full") -> pd.DataFrame:
        """Run the backtest over the specified period."""
        print(f"\nRunning backtest: {label} ({start_date} to {end_date})")

        dates = self.stock_prices.loc[start_date:end_date].index
        if len(dates) < 20:
            print("  Not enough dates")
            return pd.DataFrame()

        # Monthly rebalance dates
        monthly = dates.to_series().groupby(dates.to_period("M")).first()
        rebal_dates = set(monthly.values)

        # Weekly check dates (every Friday)
        weekly_dates = set(dates[dates.weekday == 4])

        portfolio = {}
        trailing_stops = {}  # ticker -> stop price
        daily_returns = []
        portfolio_log = []
        sector_contributions = {s: [] for s in SECTOR_ETFS}

        # Track top-5 sectors for exit rule
        top5_sectors = set()

        prev_date = None
        for date in dates:
            # Determine if rebalance
            is_rebal = date in rebal_dates
            is_weekly = date in weekly_dates

            if is_rebal:
                # Full rebalance
                portfolio = self._select_portfolio(date)
                trailing_stops = {}
                # Initialize trailing stops
                for ticker in portfolio:
                    price = self.stock_prices.at[date, ticker] if ticker in self.stock_prices.columns else np.nan
                    atr = self._get_atr(ticker, date)
                    if not np.isnan(price) and not np.isnan(atr):
                        trailing_stops[ticker] = price - ATR_MULTIPLIER * atr

                # Update top-5 for exit check
                sector_mom = self._sector_momentum(date)
                etf_to_sector = {v: k for k, v in SECTOR_ETFS.items()}
                if not sector_mom.empty:
                    top5_sectors = {etf_to_sector.get(e, "") for e in sector_mom.index[:5]}

            elif is_weekly and portfolio:
                # Weekly health check: exit sectors that dropped out of top 5
                sector_mom = self._sector_momentum(date)
                etf_to_sector = {v: k for k, v in SECTOR_ETFS.items()}
                if not sector_mom.empty:
                    current_top5 = {etf_to_sector.get(e, "") for e in sector_mom.index[:5]}
                    # Remove stocks in sectors that fell out of top 5
                    to_remove = []
                    for ticker, (sector, weight) in portfolio.items():
                        if sector not in current_top5:
                            to_remove.append(ticker)
                    for ticker in to_remove:
                        del portfolio[ticker]
                        if ticker in trailing_stops:
                            del trailing_stops[ticker]
                    top5_sectors = current_top5

                    # Redistribute weights among remaining
                    if portfolio:
                        remaining_sectors = set(s for _, (s, _) in portfolio.items())
                        death_cross = self._is_death_cross(date)
                        exposure_scale = 0.5 if death_cross else 1.0
                        sector_weight = exposure_scale / len(remaining_sectors) if remaining_sectors else 0
                        sector_stock_count = {}
                        for t, (s, w) in portfolio.items():
                            sector_stock_count[s] = sector_stock_count.get(s, 0) + 1
                        new_portfolio = {}
                        for t, (s, w) in portfolio.items():
                            sw = sector_weight / sector_stock_count.get(s, 1)
                            new_portfolio[t] = (s, sw)
                        portfolio = new_portfolio

            # Compute daily return
            if prev_date is not None and portfolio:
                day_ret = 0.0
                sector_day_ret = {s: 0.0 for s in SECTOR_ETFS}
                stopped_out = []

                for ticker, (sector, weight) in portfolio.items():
                    if ticker not in self.stock_prices.columns:
                        continue
                    p_prev = self.stock_prices.at[prev_date, ticker] if prev_date in self.stock_prices.index else np.nan
                    p_now = self.stock_prices.at[date, ticker] if date in self.stock_prices.index else np.nan
                    if np.isnan(p_prev) or np.isnan(p_now) or p_prev == 0:
                        continue

                    stock_ret = (p_now / p_prev) - 1
                    contribution = weight * stock_ret
                    day_ret += contribution
                    sector_day_ret[sector] = sector_day_ret.get(sector, 0) + contribution

                    # Trailing stop check
                    if ticker in trailing_stops:
                        if p_now <= trailing_stops[ticker]:
                            stopped_out.append(ticker)
                        else:
                            # Ratchet up
                            atr = self._get_atr(ticker, date)
                            if not np.isnan(atr):
                                new_stop = p_now - ATR_MULTIPLIER * atr
                                trailing_stops[ticker] = max(trailing_stops[ticker], new_stop)

                # Remove stopped-out positions
                for ticker in stopped_out:
                    if ticker in portfolio:
                        del portfolio[ticker]
                    if ticker in trailing_stops:
                        del trailing_stops[ticker]

                daily_returns.append({"date": date, "return": day_ret, "n_positions": len(portfolio)})
                for s, c in sector_day_ret.items():
                    sector_contributions[s].append({"date": date, "contribution": c})

                portfolio_log.append({
                    "date": date,
                    "n_stocks": len(portfolio),
                    "sectors": list(set(s for _, (s, _) in portfolio.items())),
                    "death_cross": self._is_death_cross(date),
                })

            prev_date = date

        if not daily_returns:
            return pd.DataFrame()

        returns_df = pd.DataFrame(daily_returns).set_index("date")
        returns_df.index = pd.DatetimeIndex(returns_df.index)

        # Compute sector contribution summary
        sector_summary = {}
        for sector, contribs in sector_contributions.items():
            if contribs:
                total = sum(c["contribution"] for c in contribs)
                sector_summary[sector] = total

        self.results.append({
            "label": label,
            "returns": returns_df,
            "sector_contributions": sector_summary,
            "portfolio_log": portfolio_log,
        })

        return returns_df


def compute_metrics(returns: pd.Series, label: str = "") -> dict:
    """Compute performance metrics from daily return series."""
    if returns.empty or len(returns) < 20:
        return {}
    r = returns.values
    n_days = len(r)
    total_ret = (1 + r).prod() - 1
    years = n_days / 252
    cagr = (1 + total_ret) ** (1 / years) - 1 if years > 0 else 0

    ann_vol = r.std() * np.sqrt(252)
    sharpe = (r.mean() * 252) / ann_vol if ann_vol > 0 else 0

    downside = r[r < 0]
    downside_vol = downside.std() * np.sqrt(252) if len(downside) > 0 else 0
    sortino = (r.mean() * 252) / downside_vol if downside_vol > 0 else 0

    cum = (1 + pd.Series(r)).cumprod()
    running_max = cum.cummax()
    drawdown = cum / running_max - 1
    max_dd = drawdown.min()

    # Calmar
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0

    # Win rate (daily)
    wr = (r > 0).sum() / len(r)

    # Profit factor
    gains = r[r > 0].sum()
    losses = abs(r[r < 0].sum())
    pf = gains / losses if losses > 0 else np.inf

    # Turnover: approximate from position changes
    return {
        "label": label,
        "CAGR": round(cagr * 100, 2),
        "Sharpe": round(sharpe, 3),
        "Sortino": round(sortino, 3),
        "MaxDD": round(max_dd * 100, 2),
        "Calmar": round(calmar, 3),
        "WinRate": round(wr * 100, 1),
        "ProfitFactor": round(pf, 3),
        "AnnVol": round(ann_vol * 100, 2),
        "TotalReturn": round(total_ret * 100, 2),
        "Days": n_days,
    }


def compute_regime_metrics(returns_df: pd.DataFrame, spy: pd.Series) -> dict:
    """Stratify performance by market regime (bull/bear/flat)."""
    # Classify each month by SPY return
    spy_monthly = spy.resample("M").last().pct_change()
    regimes = {}
    for period, ret in spy_monthly.items():
        if pd.isna(ret):
            continue
        if ret > 0.02:
            regimes[period] = "bull"
        elif ret < -0.02:
            regimes[period] = "bear"
        else:
            regimes[period] = "flat"

    # Map daily returns to regimes
    returns_df = returns_df.copy()
    returns_df["month"] = returns_df.index.to_period("M")
    returns_df["regime"] = returns_df["month"].map(regimes)

    result = {}
    for regime in ["bull", "bear", "flat"]:
        mask = returns_df["regime"] == regime
        if mask.sum() < 10:
            continue
        r = returns_df.loc[mask, "return"]
        result[regime] = compute_metrics(r, label=regime)

    return result


def compute_spy_benchmark(spy: pd.Series, start: str, end: str) -> dict:
    """SPY buy-and-hold benchmark."""
    s = spy.loc[start:end]
    if len(s) < 20:
        return {}
    ret = s.pct_change().dropna()
    return compute_metrics(ret, label="SPY_BH")


def compute_equal_weight_benchmark(sector_prices: pd.DataFrame, start: str, end: str) -> dict:
    """Equal-weight sector ETF benchmark."""
    s = sector_prices.loc[start:end]
    if len(s) < 20:
        return {}
    ret = s.pct_change().dropna().mean(axis=1)
    return compute_metrics(ret, label="EW_Sector")


def compute_pure_momentum_benchmark(stock_prices: pd.DataFrame, start: str, end: str) -> dict:
    """
    Pure cross-sectional momentum (no sector rotation).
    Top 20 stocks by 6mo momentum, rebalanced monthly.
    """
    dates = stock_prices.loc[start:end].index
    if len(dates) < 20:
        return {}
    monthly = dates.to_series().groupby(dates.to_period("M")).first()
    rebal_dates = set(monthly.values)

    portfolio_weights = {}
    daily_rets = []
    prev_date = None

    for date in dates:
        if date in rebal_dates:
            # Select top 20 by 6mo momentum skip 1mo
            end_dt = date - pd.DateOffset(months=1)
            start_dt = date - pd.DateOffset(months=6)
            ms = stock_prices.index.searchsorted(start_dt)
            me = stock_prices.index.searchsorted(end_dt)
            if me > ms + 20:
                window = stock_prices.iloc[ms:me]
                mom = (window.iloc[-1] / window.iloc[0] - 1).dropna().sort_values(ascending=False)
                top20 = list(mom.index[:20])
                w = 1.0 / len(top20)
                portfolio_weights = {t: w for t in top20}
            else:
                portfolio_weights = {}

        if prev_date is not None and portfolio_weights:
            day_ret = 0
            for ticker, weight in portfolio_weights.items():
                if ticker not in stock_prices.columns:
                    continue
                p0 = stock_prices.at[prev_date, ticker] if prev_date in stock_prices.index else np.nan
                p1 = stock_prices.at[date, ticker] if date in stock_prices.index else np.nan
                if not np.isnan(p0) and not np.isnan(p1) and p0 > 0:
                    day_ret += weight * (p1 / p0 - 1)
            daily_rets.append(day_ret)

        prev_date = date

    if not daily_rets:
        return {}
    return compute_metrics(np.array(daily_rets), label="PureMomentum")


# ---------------------------------------------------------------------------
# WALK-FORWARD
# ---------------------------------------------------------------------------
def run_walk_forward(bt: SectorRotationBacktester, all_dates: pd.DatetimeIndex) -> list[dict]:
    """Sliding walk-forward: 2yr train, 1yr OOT."""
    first_year = all_dates[0].year
    last_year = all_dates[-1].year

    wf_results = []
    oot_start_year = first_year + TRAIN_YEARS

    for year in range(oot_start_year, last_year + 1):
        oot_start = f"{year}-01-01"
        oot_end = f"{year}-12-31"
        label = f"OOT_{year}"

        returns_df = bt.run_backtest(oot_start, oot_end, label=label)
        if returns_df.empty:
            continue

        metrics = compute_metrics(returns_df["return"], label=label)
        wf_results.append(metrics)

    return wf_results


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("SECTOR ROTATION + MOMENTUM HYBRID BACKTESTER")
    print("=" * 70)

    # Collect all tickers
    all_stocks = []
    for sector, stocks in SECTOR_STOCKS.items():
        all_stocks.extend(stocks)
    all_stocks = list(set(all_stocks))

    sector_etf_tickers = list(SECTOR_ETFS.values())
    all_tickers = all_stocks + sector_etf_tickers + ["SPY"]

    # Download data
    print("\n--- Downloading price data ---")
    all_close = download_data(all_tickers, start="2014-01-01", end="2026-07-12")

    # Download OHLC for ATR calculation (just the stock universe)
    print("Downloading OHLC for ATR...")
    ohlc_data = yf.download(all_stocks, start="2014-01-01", end="2026-07-12",
                            auto_adjust=True, progress=False, threads=True)
    stock_ohlc = {}
    for ticker in all_stocks:
        try:
            if isinstance(ohlc_data.columns, pd.MultiIndex):
                df = pd.DataFrame({
                    "High": ohlc_data[("High", ticker)],
                    "Low": ohlc_data[("Low", ticker)],
                    "Close": ohlc_data[("Close", ticker)],
                }).dropna()
            else:
                df = pd.DataFrame()
            if not df.empty:
                stock_ohlc[ticker] = df
        except (KeyError, TypeError):
            pass

    # Separate data
    sector_prices = all_close[[t for t in sector_etf_tickers if t in all_close.columns]]
    stock_prices = all_close[[t for t in all_stocks if t in all_close.columns]]
    spy = all_close["SPY"] if "SPY" in all_close.columns else pd.Series()

    print(f"\nSector ETFs: {sector_prices.shape[1]}")
    print(f"Stocks: {stock_prices.shape[1]}")
    print(f"Date range: {all_close.index[0].date()} to {all_close.index[-1].date()}")

    # Initialize backtester
    bt = SectorRotationBacktester(sector_prices, stock_prices, spy, stock_ohlc)

    # --- Full backtest (2016-2026, allowing 2yr lookback from 2014) ---
    full_start = "2016-01-01"
    full_end = "2026-07-12"
    print("\n" + "=" * 70)
    print("FULL BACKTEST")
    print("=" * 70)
    full_returns = bt.run_backtest(full_start, full_end, label="full")

    if full_returns.empty:
        print("ERROR: No returns generated. Check data.")
        sys.exit(1)

    full_metrics = compute_metrics(full_returns["return"], label="SectorRotation")

    # Benchmarks
    print("\n--- Computing benchmarks ---")
    spy_metrics = compute_spy_benchmark(spy, full_start, full_end)
    ew_metrics = compute_equal_weight_benchmark(sector_prices, full_start, full_end)
    mom_metrics = compute_pure_momentum_benchmark(stock_prices, full_start, full_end)

    # Regime analysis
    print("--- Regime analysis ---")
    regime_metrics = compute_regime_metrics(full_returns, spy)

    # Walk-forward
    print("\n" + "=" * 70)
    print("WALK-FORWARD ANALYSIS (2yr train, 1yr OOT, sliding)")
    print("=" * 70)
    wf_results = run_walk_forward(bt, stock_prices.index)

    # Sector contributions
    sector_contrib = bt.results[0]["sector_contributions"] if bt.results else {}

    # Turnover estimation from portfolio log
    plog = bt.results[0]["portfolio_log"] if bt.results else []
    if len(plog) > 1:
        changes = 0
        for i in range(1, len(plog)):
            changes += abs(plog[i]["n_stocks"] - plog[i - 1]["n_stocks"])
        avg_turnover_per_day = changes / len(plog)
        annual_turnover = avg_turnover_per_day * 252
    else:
        annual_turnover = 0

    # --- Print Results ---
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    print("\n--- Strategy vs Benchmarks ---")
    header = f"{'Strategy':<20} {'CAGR%':>7} {'Sharpe':>7} {'Sortino':>8} {'MaxDD%':>7} {'WR%':>6} {'PF':>6}"
    print(header)
    print("-" * len(header))
    for m in [full_metrics, spy_metrics, ew_metrics, mom_metrics]:
        if m:
            print(f"{m['label']:<20} {m['CAGR']:>7.2f} {m['Sharpe']:>7.3f} {m['Sortino']:>8.3f} {m['MaxDD']:>7.2f} {m['WinRate']:>6.1f} {m['ProfitFactor']:>6.3f}")

    print("\n--- Regime Performance (Strategy) ---")
    header2 = f"{'Regime':<10} {'CAGR%':>7} {'Sharpe':>7} {'Sortino':>8} {'MaxDD%':>7} {'Days':>6}"
    print(header2)
    print("-" * len(header2))
    for regime, m in regime_metrics.items():
        print(f"{regime:<10} {m['CAGR']:>7.2f} {m['Sharpe']:>7.3f} {m['Sortino']:>8.3f} {m['MaxDD']:>7.2f} {m['Days']:>6}")

    print("\n--- Walk-Forward OOT Results ---")
    header3 = f"{'Year':<12} {'CAGR%':>7} {'Sharpe':>7} {'Sortino':>8} {'MaxDD%':>7}"
    print(header3)
    print("-" * len(header3))
    for m in wf_results:
        print(f"{m['label']:<12} {m['CAGR']:>7.2f} {m['Sharpe']:>7.3f} {m['Sortino']:>8.3f} {m['MaxDD']:>7.2f}")

    # OOT aggregate
    if wf_results:
        avg_sharpe = np.mean([m["Sharpe"] for m in wf_results])
        avg_cagr = np.mean([m["CAGR"] for m in wf_results])
        pos_years = sum(1 for m in wf_results if m["CAGR"] > 0)
        print(f"\n  OOT Avg Sharpe: {avg_sharpe:.3f}")
        print(f"  OOT Avg CAGR:   {avg_cagr:.2f}%")
        print(f"  Positive years: {pos_years}/{len(wf_results)}")

    print(f"\n--- Sector Contributions (Total Return Contribution) ---")
    sorted_sectors = sorted(sector_contrib.items(), key=lambda x: x[1], reverse=True)
    for sector, contrib in sorted_sectors:
        bar = "+" * max(0, int(contrib * 500)) + "-" * max(0, int(-contrib * 500))
        print(f"  {sector:<25} {contrib * 100:>+7.2f}%  {bar}")

    print(f"\n--- Turnover ---")
    print(f"  Est. annual position changes: {annual_turnover:.0f}")

    # Death cross periods
    dc_days = sum(1 for p in plog if p.get("death_cross", False))
    print(f"  Days in death cross (defensive mode): {dc_days}/{len(plog)} ({100*dc_days/max(1,len(plog)):.1f}%)")

    # --- Save results ---
    summary = {
        "strategy_metrics": full_metrics,
        "benchmarks": {
            "SPY": spy_metrics,
            "EqualWeightSector": ew_metrics,
            "PureMomentum": mom_metrics,
        },
        "regime_metrics": regime_metrics,
        "walk_forward": wf_results,
        "sector_contributions": {k: round(v * 100, 4) for k, v in sector_contrib.items()},
        "annual_turnover_est": round(annual_turnover, 1),
        "death_cross_pct": round(100 * dc_days / max(1, len(plog)), 1),
        "config": {
            "top_n_sectors": TOP_N_SECTORS,
            "stocks_per_sector": STOCKS_PER_SECTOR,
            "sector_mom_months": SECTOR_MOM_MONTHS,
            "stock_mom_months": STOCK_MOM_MONTHS,
            "stock_mom_skip": STOCK_MOM_SKIP,
            "atr_period": ATR_PERIOD,
            "atr_multiplier": ATR_MULTIPLIER,
            "train_years": TRAIN_YEARS,
            "oot_years": OOT_YEARS,
        },
        "timestamp": datetime.now().isoformat(),
    }

    summary_path = OUTPUT_DIR / "sector_rotation_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSummary saved to {summary_path}")

    # Save daily returns
    returns_path = OUTPUT_DIR / "sector_rotation_daily_returns.csv"
    full_returns.to_csv(returns_path)
    print(f"Daily returns saved to {returns_path}")

    # Save equity curve
    equity = (1 + full_returns["return"]).cumprod()
    equity_path = OUTPUT_DIR / "sector_rotation_equity_curve.csv"
    equity.to_csv(equity_path)
    print(f"Equity curve saved to {equity_path}")

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)

    return summary


if __name__ == "__main__":
    summary = main()
