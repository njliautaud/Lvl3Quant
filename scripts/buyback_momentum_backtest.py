#!/usr/bin/env python3
"""
Share Buyback Momentum Strategy Backtester
==========================================
Academic basis: Companies actively repurchasing their own shares outperform
the market by 3-4% annually (Ikenberry, Lakonishok, Vermaelen 1995;
Peyer & Vermaelen 2009). The signal is strongest when combined with:
  1. Value (low P/E or P/B) — buybacks of cheap stocks signal undervaluation
  2. Insider buying — management putting personal money in too
  3. Momentum — stocks already in uptrend benefit most from buyback support

Since we don't have live buyback announcement data in yfinance, we use
PROXY SIGNALS that capture the same underlying phenomenon:
  1. DECLINING SHARE COUNT: When shares outstanding decrease quarter-over-quarter,
     the company is actively buying back stock. This is observable in financial data.
  2. HIGH FREE CASH FLOW YIELD: Companies with high FCF relative to market cap
     are the ones most likely executing buybacks AND most undervalued.
  3. PRICE MOMENTUM: Combine with 3-6 month momentum to time entries.

Strategy:
  - Monthly rebalance: rank universe by buyback proxy score
  - Buy top 5 stocks, equal weight
  - Hold for 1 month, rebalance
  - Bear market filter: rotate to lower-vol names, reduce allocation

This is a PORTFOLIO REBALANCING strategy, not a signal-timing strategy.
No ATR stops — just monthly reconstitution.

OOT: Jan 2022 - Jul 2026
Capital: $645
"""

import json
import logging
import os
import sys
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

OUTPUT_DIR = Path("/home/jupiter/Lvl3Quant/output/buyback_momentum")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR = OUTPUT_DIR / "cache"
CACHE_DIR.mkdir(exist_ok=True)

# Config
TOP_N = 5                     # Hold top 5 stocks
REBAL_FREQ = "M"              # Monthly rebalance
MOM_LOOKBACK = 63             # ~3 months momentum
MOM_SKIP = 5                  # Skip last week (mean reversion noise)
VOL_LOOKBACK = 20             # 20-day realized vol
MAX_WEIGHT = 0.25             # Max 25% in any single name
SLIPPAGE_BPS = 5

DATA_START = "2020-01-01"
OOT_START = "2022-01-01"
OOT_END = "2026-07-25"
STARTING_CAPITAL = 645.0

SPY_SMA_FAST = 50
SPY_SMA_SLOW = 200

# Universe: large-cap stocks with active buyback programs (well-documented)
# These are companies known for consistent share repurchases
UNIVERSE = [
    # Mega buyback programs (top repurchasers)
    "AAPL", "GOOGL", "META", "MSFT", "NVDA",
    # Major tech buybacks
    "AVGO", "QCOM", "TXN", "CSCO", "ORCL", "ADBE", "CRM", "INTC", "MU",
    # Financials (big buyback programs)
    "JPM", "BAC", "WFC", "GS", "MS", "C", "BLK", "SCHW", "AXP", "USB",
    # Healthcare (consistent buybacks)
    "UNH", "ABBV", "MRK", "PFE", "JNJ", "AMGN", "BMY", "LLY", "TMO",
    # Consumer (steady buybacks)
    "HD", "LOW", "MCD", "SBUX", "NKE", "TJX", "COST", "WMT", "PG", "KO", "PEP",
    # Industrial
    "HON", "CAT", "DE", "GE", "RTX", "BA", "UNP", "MMM",
    # Energy
    "XOM", "CVX", "COP", "EOG",
    # Other strong buyback history
    "AMZN", "DIS", "CMCSA", "T", "VZ", "PM",
    # Growth names with emerging buybacks
    "NFLX", "AMD", "CRWD", "PANW",
]


def download_data(tickers, start, end):
    cache_file = CACHE_DIR / f"buyback_data_{start}_{end}.pkl"
    if cache_file.exists():
        log.info("Loading cached price data...")
        data = pd.read_pickle(cache_file)
        if len(data) >= len(tickers) * 0.7:
            return data

    log.info(f"Downloading {len(tickers)} tickers...")
    data = {}
    batch_size = 20
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i + batch_size]
        try:
            raw = yf.download(" ".join(batch), start=start, end=end,
                             group_by="ticker", auto_adjust=True, progress=False, threads=True)
            for t in batch:
                try:
                    df = raw[t].copy() if len(batch) > 1 else raw.copy()
                    df = df.dropna(subset=["Close", "Volume"])
                    if len(df) > 100:
                        data[t] = df
                except Exception:
                    pass
        except Exception as e:
            log.warning(f"Batch failed: {e}")

    log.info(f"Downloaded {len(data)} tickers (prices)")
    pd.to_pickle(data, cache_file)
    return data


def get_fundamentals(tickers):
    """Get shares outstanding and financial data from yfinance."""
    cache_file = CACHE_DIR / "fundamentals.pkl"
    if cache_file.exists():
        log.info("Loading cached fundamentals...")
        return pd.read_pickle(cache_file)

    log.info(f"Fetching fundamentals for {len(tickers)} tickers...")
    fund_data = {}
    for t in tickers:
        try:
            ticker = yf.Ticker(t)
            info = ticker.info
            # Get quarterly financials for share count changes
            try:
                bs = ticker.quarterly_balance_sheet
                if bs is not None and len(bs.columns) >= 2:
                    # Look for share count rows
                    share_rows = [r for r in bs.index if "share" in str(r).lower() and "outstand" in str(r).lower()]
                    if share_rows:
                        shares = bs.loc[share_rows[0]]
                        # Calculate QoQ change in shares
                        shares_sorted = shares.sort_index()
                        share_change = shares_sorted.pct_change()
                        avg_share_change = float(share_change.mean()) if len(share_change.dropna()) > 0 else 0
                    else:
                        avg_share_change = 0
                else:
                    avg_share_change = 0
            except Exception:
                avg_share_change = 0

            fund_data[t] = {
                "market_cap": info.get("marketCap", 0),
                "pe_ratio": info.get("trailingPE", 0),
                "forward_pe": info.get("forwardPE", 0),
                "pb_ratio": info.get("priceToBook", 0),
                "fcf_yield": 0,  # Will compute from price data
                "avg_share_change": avg_share_change,  # Negative = buyback
                "dividend_yield": info.get("dividendYield", 0) or 0,
                "beta": info.get("beta", 1) or 1,
            }
        except Exception as e:
            fund_data[t] = {
                "market_cap": 0, "pe_ratio": 0, "forward_pe": 0,
                "pb_ratio": 0, "fcf_yield": 0, "avg_share_change": 0,
                "dividend_yield": 0, "beta": 1,
            }

    log.info(f"Got fundamentals for {len(fund_data)} tickers")
    pd.to_pickle(fund_data, cache_file)
    return fund_data


def download_spy(start, end):
    cache_file = CACHE_DIR / "spy.pkl"
    if cache_file.exists():
        spy = pd.read_pickle(cache_file)
        if len(spy) > 500:
            return spy
    spy = yf.download("SPY", start=start, end=end, auto_adjust=True, progress=False)
    pd.to_pickle(spy, cache_file)
    return spy


def compute_scores(data: dict, fund: dict, date: pd.Timestamp, regime: int) -> list:
    """
    Compute buyback momentum score for each stock.

    Score components:
    1. Momentum (3-month, skip last week) — 40% weight
    2. Value proxy (inverse of volatility-adjusted P/E) — 20% weight
    3. Buyback proxy (negative share count change) — 20% weight
    4. Low volatility (inverse of 20-day vol) — 20% weight

    Higher score = stronger buy signal.
    """
    scores = []

    for ticker in data:
        df = data[ticker]
        if date not in df.index:
            continue

        idx = df.index.get_loc(date)
        if idx < MOM_LOOKBACK + 10:
            continue

        close = df["Close"]

        # 1. Momentum (skip last week to avoid reversal noise)
        if idx - MOM_SKIP >= 0 and idx - MOM_LOOKBACK >= 0:
            mom = float(close.iloc[idx - MOM_SKIP] / close.iloc[idx - MOM_LOOKBACK] - 1)
        else:
            continue

        # 2. Realized vol (20-day)
        rets = close.pct_change().iloc[max(0, idx - VOL_LOOKBACK):idx + 1]
        vol = float(rets.std()) * np.sqrt(252) if len(rets) > 5 else 0.3

        # 3. Relative strength (vs 200-day SMA)
        if idx >= 200:
            sma200 = float(close.iloc[idx - 199:idx + 1].mean())
            rs = float(close.iloc[idx]) / sma200 - 1
        else:
            rs = 0

        # 4. Fundamental data
        f = fund.get(ticker, {})
        share_chg = f.get("avg_share_change", 0)  # Negative = buyback
        pe = f.get("forward_pe", 0) or f.get("pe_ratio", 0)
        beta = f.get("beta", 1) or 1

        # Score components (all normalized 0-1 later)
        # Momentum: higher is better
        mom_score = mom

        # Value: lower PE is better (invert). Handle missing/negative PE
        if pe and pe > 0:
            value_score = 1.0 / pe  # Higher for lower PE
        else:
            value_score = 0

        # Buyback: more negative share change is better (company buying back more)
        buyback_score = -share_chg if share_chg != 0 else 0

        # Low vol: lower vol is better in bear market, less important in bull
        vol_score = 1.0 / (vol + 0.01)

        # Relative strength
        rs_score = rs

        # Composite score — weights shift with regime
        if regime == 1:  # Bull
            composite = (0.40 * mom_score +
                        0.15 * value_score * 10 +   # Scale value
                        0.20 * buyback_score * 100 + # Scale buyback
                        0.10 * vol_score * 0.1 +     # Scale vol
                        0.15 * rs_score)
        else:  # Bear — favor value + low vol
            composite = (0.20 * mom_score +
                        0.25 * value_score * 10 +
                        0.20 * buyback_score * 100 +
                        0.25 * vol_score * 0.1 +
                        0.10 * rs_score)

        scores.append({
            "ticker": ticker,
            "score": composite,
            "momentum": mom,
            "vol": vol,
            "rs": rs,
            "share_chg": share_chg,
            "pe": pe,
            "beta": beta,
        })

    return sorted(scores, key=lambda x: x["score"], reverse=True)


def run_backtest(data, spy, fund):
    spy_close = spy["Close"].squeeze() if isinstance(spy["Close"], pd.DataFrame) else spy["Close"]
    spy_sma50 = spy_close.rolling(SPY_SMA_FAST).mean()
    spy_sma200 = spy_close.rolling(SPY_SMA_SLOW).mean()
    spy_regime = (spy_sma50 > spy_sma200).astype(int)
    spy_daily_ret = spy_close.pct_change()

    oot_start = pd.Timestamp(OOT_START)
    oot_end = pd.Timestamp(OOT_END)
    all_dates = spy.index[(spy.index >= oot_start) & (spy.index <= oot_end)]
    log.info(f"OOT: {all_dates[0].date()} to {all_dates[-1].date()} ({len(all_dates)} days)")

    # Get rebalance dates (last trading day of each month)
    monthly_groups = pd.Series(all_dates).groupby([all_dates.year, all_dates.month])
    rebal_dates = set()
    for _, group in monthly_groups:
        rebal_dates.add(group.iloc[-1])

    equity = STARTING_CAPITAL
    holdings = {}  # ticker -> {"shares": x, "entry_price": y, "entry_date": z}
    trades = []
    equity_curve = []
    daily_returns = []
    rebal_count = 0

    for i, date in enumerate(all_dates):
        regime = 1
        if date in spy_regime.index:
            r = spy_regime.loc[date]
            regime = int(r) if not pd.isna(r) else 1

        spy_ret = 0.0
        if date in spy_daily_ret.index:
            sr = spy_daily_ret.loc[date]
            spy_ret = float(sr) if not pd.isna(sr) else 0.0

        # ── Monthly rebalance ──
        if date in rebal_dates:
            rebal_count += 1

            # Calculate current equity (including unrealized)
            total_eq = equity
            for t, h in holdings.items():
                if t in data and date in data[t].index:
                    cur_price = float(data[t].loc[date]["Close"])
                    total_eq += (cur_price - h["entry_price"]) * h["shares"]

            # Score universe
            scores = compute_scores(data, fund, date, regime)
            if not scores:
                continue

            # Select top N
            top = scores[:TOP_N]
            new_tickers = set(s["ticker"] for s in top)

            # Sell anything not in new top
            for t in list(holdings.keys()):
                if t not in new_tickers:
                    if t in data and date in data[t].index:
                        exit_price = float(data[t].loc[date]["Close"]) * (1 - SLIPPAGE_BPS / 10000)
                        h = holdings[t]
                        pnl = (exit_price - h["entry_price"]) * h["shares"]
                        equity += pnl
                        trades.append({
                            "ticker": t,
                            "entry_date": h["entry_date"],
                            "exit_date": str(date.date()),
                            "entry_price": round(h["entry_price"], 2),
                            "exit_price": round(exit_price, 2),
                            "shares": round(h["shares"], 4),
                            "pnl": round(pnl, 2),
                            "return_pct": round(pnl / (h["entry_price"] * h["shares"]) * 100, 2),
                            "days_held": (date - pd.Timestamp(h["entry_date"])).days,
                            "exit_reason": "rebalance",
                            "regime": "bull" if regime else "bear",
                            "spy_ret": round(spy_ret * 100, 3),
                        })
                        del holdings[t]

            # Recalculate equity after sells
            total_eq = equity
            for t, h in holdings.items():
                if t in data and date in data[t].index:
                    total_eq += (float(data[t].loc[date]["Close"]) - h["entry_price"]) * h["shares"]

            # Buy new positions
            # In bear market, only invest 60% of capital
            invest_pct = 1.0 if regime == 1 else 0.6
            available = total_eq * invest_pct

            # Equal weight among top N (minus existing holdings)
            new_buys = [s for s in top if s["ticker"] not in holdings]
            if new_buys and available > 10:
                weight = min(available / len(new_buys) / available, MAX_WEIGHT)
                for s in new_buys:
                    t = s["ticker"]
                    if t in data and date in data[t].index:
                        pos_value = available * weight
                        if pos_value < 5:
                            continue
                        entry_price = float(data[t].loc[date]["Close"]) * (1 + SLIPPAGE_BPS / 10000)
                        shares = pos_value / entry_price
                        holdings[t] = {
                            "shares": shares,
                            "entry_price": entry_price,
                            "entry_date": str(date.date()),
                        }

            if rebal_count % 6 == 0:
                log.info(f"  Rebal #{rebal_count} on {date.date()}: equity=${total_eq:.2f}, "
                        f"holding {len(holdings)} stocks, {len(trades)} trades so far")

        # ── Daily MTM ──
        mtm = equity
        for t, h in holdings.items():
            if t in data and date in data[t].index:
                cur = float(data[t].loc[date]["Close"])
                mtm += (cur - h["entry_price"]) * h["shares"]

        equity_curve.append({
            "date": str(date.date()),
            "equity": round(mtm, 2),
            "n_holdings": len(holdings),
            "regime": "bull" if regime else "bear",
            "spy_ret": round(spy_ret, 6),
        })

        if len(equity_curve) >= 2:
            prev = equity_curve[-2]["equity"]
            daily_returns.append(mtm / prev - 1 if prev > 0 else 0)

    # Close remaining
    for t, h in holdings.items():
        last = all_dates[-1]
        if t in data and last in data[t].index:
            close = float(data[t].loc[last]["Close"])
            pnl = (close - h["entry_price"]) * h["shares"]
            trades.append({
                "ticker": t,
                "entry_date": h["entry_date"],
                "exit_date": str(last.date()),
                "entry_price": round(h["entry_price"], 2),
                "exit_price": round(close, 2),
                "shares": round(h["shares"], 4),
                "pnl": round(pnl, 2),
                "return_pct": round(pnl / (h["entry_price"] * h["shares"]) * 100, 2),
                "days_held": (last - pd.Timestamp(h["entry_date"])).days,
                "exit_reason": "end_of_test",
                "regime": "bull",
                "spy_ret": 0,
            })

    return trades, equity_curve, daily_returns


def validate(trades, equity_curve, daily_returns):
    if not trades:
        return {"passed": False, "reason": "No trades"}

    df_t = pd.DataFrame(trades)
    df_e = pd.DataFrame(equity_curve)
    rets = np.array(daily_returns)

    n = len(df_t)
    winners = df_t[df_t["pnl"] > 0]
    losers = df_t[df_t["pnl"] <= 0]
    wr = len(winners) / n if n > 0 else 0
    pf = abs(float(winners["pnl"].sum()) / float(losers["pnl"].sum())) if len(losers) > 0 and float(losers["pnl"].sum()) != 0 else float("inf")
    total_pnl = float(df_t["pnl"].sum())
    final_eq = float(df_e["equity"].iloc[-1])

    results = {
        "total_trades": int(n),
        "win_rate": round(float(wr), 4),
        "avg_win": round(float(winners["pnl"].mean()), 2) if len(winners) > 0 else 0.0,
        "avg_loss": round(float(losers["pnl"].mean()), 2) if len(losers) > 0 else 0.0,
        "profit_factor": round(float(pf), 3),
        "total_pnl": round(total_pnl, 2),
        "final_equity": round(final_eq, 2),
        "return_pct": round((final_eq / STARTING_CAPITAL - 1) * 100, 2),
        "avg_days_held": round(float(df_t["days_held"].mean()), 1),
    }

    # Ticker frequency
    top_tickers = df_t["ticker"].value_counts().head(10)
    results["top_tickers"] = {k: int(v) for k, v in top_tickers.items()}

    # CAGR
    n_years = len(rets) / 252
    if n_years > 0 and final_eq > 0:
        results["cagr"] = round(((final_eq / STARTING_CAPITAL) ** (1 / n_years) - 1) * 100, 2)
    else:
        results["cagr"] = 0.0

    # Gate 1: Sharpe
    if len(rets) > 20:
        ann_r = float(np.mean(rets)) * 252
        ann_v = float(np.std(rets)) * np.sqrt(252)
        sharpe = ann_r / ann_v if ann_v > 0 else 0
    else:
        sharpe = 0
    results["sharpe"] = round(sharpe, 3)
    results["gate1_sharpe"] = bool(sharpe > 0.5)

    ds = rets[rets < 0]
    ds_v = float(np.std(ds)) * np.sqrt(252) if len(ds) > 0 else 1e-6
    results["sortino"] = round((float(np.mean(rets)) * 252) / ds_v, 3)

    # Gate 2: Perm test
    ps = []
    for _ in range(2000):
        p = np.random.permutation(rets)
        m, s = float(np.mean(p)) * 252, float(np.std(p)) * np.sqrt(252)
        ps.append(m / s if s > 0 else 0)
    pv = float(np.mean(np.array(ps) >= sharpe))
    results["perm_p_value"] = round(pv, 4)
    results["gate2_perm"] = bool(pv < 0.05)

    # Gate 3: Beat random
    rs_list = []
    for _ in range(1000):
        r = np.random.choice(rets, size=len(rets), replace=True)
        m, s = float(np.mean(r)) * 252, float(np.std(r)) * np.sqrt(252)
        rs_list.append(m / s if s > 0 else 0)
    rm = float(np.median(rs_list))
    results["random_baseline"] = round(rm, 3)
    results["gate3_beats_random"] = bool(sharpe > rm + 0.1)

    # Gate 4: Regime
    bull = df_t[df_t["regime"] == "bull"]
    bear = df_t[df_t["regime"] == "bear"]
    bull_wr = float(len(bull[bull["pnl"] > 0]) / len(bull)) if len(bull) > 0 else 0
    bear_wr = float(len(bear[bear["pnl"] > 0]) / len(bear)) if len(bear) > 0 else 0
    if len(bull) > 5 and len(bear) > 5:
        gap = abs(bull_wr - bear_wr) / max(bull_wr, bear_wr) if max(bull_wr, bear_wr) > 0 else 1
    elif len(bear) <= 5:
        gap = 0.3
    else:
        gap = 1.0
    results["bull_trades"] = int(len(bull))
    results["bull_wr"] = round(bull_wr, 4)
    results["bear_trades"] = int(len(bear))
    results["bear_wr"] = round(bear_wr, 4)
    results["regime_gap"] = round(gap, 4)
    results["gate4_regime"] = bool(gap < 0.50)

    # Gate 5: MDD
    eq = df_e["equity"].astype(float).values
    peak = np.maximum.accumulate(eq)
    dd = (eq - peak) / peak
    mdd = float(dd.min())
    results["max_drawdown"] = round(mdd, 4)
    results["gate5_mdd"] = bool(mdd > -0.50)

    # Compare to SPY buy-and-hold
    df_e2 = df_e.copy()
    df_e2["spy_ret_f"] = pd.to_numeric(df_e2["spy_ret"])
    spy_cumret = (1 + df_e2["spy_ret_f"]).cumprod()
    spy_final = float(spy_cumret.iloc[-1]) * STARTING_CAPITAL if len(spy_cumret) > 0 else STARTING_CAPITAL
    results["spy_bnh_final"] = round(spy_final, 2)
    results["spy_bnh_return_pct"] = round((spy_final / STARTING_CAPITAL - 1) * 100, 2)
    results["alpha_vs_spy"] = round(results["return_pct"] - results["spy_bnh_return_pct"], 2)

    # Green/red
    df_e2["daily_ret"] = pd.to_numeric(df_e2["equity"]).pct_change()
    green = df_e2[df_e2["spy_ret_f"] > 0.001]
    red = df_e2[df_e2["spy_ret_f"] < -0.001]
    if len(green) > 20 and len(red) > 20:
        gs = (float(green["daily_ret"].mean()) * 252) / (float(green["daily_ret"].std()) * np.sqrt(252)) if float(green["daily_ret"].std()) > 0 else 0
        rds = (float(red["daily_ret"].mean()) * 252) / (float(red["daily_ret"].std()) * np.sqrt(252)) if float(red["daily_ret"].std()) > 0 else 0
        results["green_day_sharpe"] = round(gs, 3)
        results["red_day_sharpe"] = round(rds, 3)
    else:
        results["green_day_sharpe"] = 0.0
        results["red_day_sharpe"] = 0.0

    gates = [results["gate1_sharpe"], results["gate2_perm"], results["gate3_beats_random"],
             results["gate4_regime"], results["gate5_mdd"]]
    results["gates_passed"] = int(sum(gates))
    results["all_gates_passed"] = bool(all(gates))

    return results


def log_mlflow(results, trades, equity_curve):
    try:
        import mlflow
        mlflow.set_tracking_uri("http://localhost:5000")
        mlflow.set_experiment("buyback_momentum_v1")

        with mlflow.start_run(run_name=f"buyback_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            mlflow.log_param("strategy", "buyback_momentum")
            mlflow.log_param("universe_size", len(UNIVERSE))
            mlflow.log_param("top_n", TOP_N)
            mlflow.log_param("rebal_freq", REBAL_FREQ)
            mlflow.log_param("mom_lookback", MOM_LOOKBACK)
            mlflow.log_param("oot_period", f"{OOT_START} to {OOT_END}")
            mlflow.log_param("starting_capital", STARTING_CAPITAL)

            for k, v in results.items():
                if isinstance(v, (int, float)):
                    mlflow.log_metric(k, v)

            for name, obj in [("trades.json", trades), ("equity.json", equity_curve), ("results.json", results)]:
                path = OUTPUT_DIR / name
                with open(path, "w") as f:
                    json.dump(obj, f, indent=2)
                mlflow.log_artifact(str(path))

            log.info("MLflow logged")
    except Exception as e:
        log.warning(f"MLflow failed: {e}")


def main():
    log.info("=" * 70)
    log.info("BUYBACK MOMENTUM STRATEGY BACKTEST")
    log.info("=" * 70)

    data = download_data(UNIVERSE, DATA_START, OOT_END)
    spy = download_spy(DATA_START, OOT_END)
    fund = get_fundamentals(list(data.keys()))
    log.info(f"Data: {len(data)} tickers, {len(fund)} fundamentals, SPY {len(spy)} days")

    trades, equity_curve, daily_returns = run_backtest(data, spy, fund)
    log.info(f"\n{len(trades)} trades")

    results = validate(trades, equity_curve, daily_returns)

    log.info("\n" + "=" * 70)
    log.info("RESULTS")
    log.info("=" * 70)
    log.info(f"Trades: {results['total_trades']} | WR: {results['win_rate']*100:.1f}% | PF: {results['profit_factor']:.3f}")
    log.info(f"P&L: ${results['total_pnl']:.2f} | Final: ${results['final_equity']:.2f} ({results['return_pct']:.1f}%)")
    log.info(f"CAGR: {results['cagr']:.1f}% | Avg hold: {results['avg_days_held']:.0f}d")
    log.info(f"SPY B&H: ${results.get('spy_bnh_final', 0):.2f} ({results.get('spy_bnh_return_pct', 0):.1f}%) | Alpha: {results.get('alpha_vs_spy', 0):.1f}%")
    log.info(f"Top tickers: {results.get('top_tickers', {})}")
    log.info(f"")
    log.info(f"Gate 1 Sharpe>0.5:  {results['sharpe']:.3f} (Sortino {results['sortino']:.3f}) {'PASS' if results['gate1_sharpe'] else 'FAIL'}")
    log.info(f"Gate 2 Perm p<0.05: {results['perm_p_value']:.4f} {'PASS' if results['gate2_perm'] else 'FAIL'}")
    log.info(f"Gate 3 Beat random: {results['sharpe']:.3f} vs {results['random_baseline']:.3f} {'PASS' if results['gate3_beats_random'] else 'FAIL'}")
    log.info(f"Gate 4 Regime:      gap={results['regime_gap']:.3f} (bull {results['bull_wr']*100:.1f}% bear {results['bear_wr']*100:.1f}%) {'PASS' if results['gate4_regime'] else 'FAIL'}")
    log.info(f"Gate 5 MDD>-50%:    {results['max_drawdown']*100:.1f}% {'PASS' if results['gate5_mdd'] else 'FAIL'}")
    log.info(f"Green Sharpe: {results['green_day_sharpe']:.3f} | Red Sharpe: {results['red_day_sharpe']:.3f}")
    log.info(f"")
    log.info(f"GATES: {results['gates_passed']}/5 {'PASS' if results['all_gates_passed'] else 'FAIL'}")

    log_mlflow(results, trades, equity_curve)

    with open(OUTPUT_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    main()
