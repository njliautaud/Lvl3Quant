#!/usr/bin/env python3
"""
REIT / Real Estate Momentum Backtest
=====================================
6 variants testing REIT strategies for QQQ-uncorrelated alpha.
Walk-forward SLIDING window, OOT 2022-01-01 to 2026-07-29.
Starting capital $645, Robinhood ($0 commission), 0.02% slippage.
"""

import json
import warnings
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

# ─── CONFIG ───────────────────────────────────────────────────────────────────
START_CAPITAL = 645.0
SLIPPAGE_PCT = 0.0002  # 0.02%
COMMISSION = 0.0
OOT_START = "2022-01-01"
OOT_END = "2026-07-29"
DATA_START = "2020-01-01"  # extra lookback for indicators
PERM_ITERS = 1000

TICKERS = ["VNQ", "IYR", "XLRE", "RWR", "REM", "VNQI", "TLT", "SPY", "QQQ"]

RESULTS_PATH = Path("/home/jupiter/Lvl3Quant/data/reit_momentum_results.json")

np.random.seed(42)


# ─── DATA ─────────────────────────────────────────────────────────────────────
def download_data():
    """Download all required tickers."""
    print("Downloading data...")
    data = {}
    for t in TICKERS:
        try:
            df = yf.download(t, start=DATA_START, end="2026-07-30", progress=False, auto_adjust=True)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(1)
            df.index = pd.to_datetime(df.index).tz_localize(None)
            data[t] = df
            print(f"  {t}: {len(df)} rows ({df.index[0].date()} to {df.index[-1].date()})")
        except Exception as e:
            print(f"  {t}: FAILED - {e}")
    return data


def get_close(data, ticker):
    return data[ticker]["Close"].copy()


def get_vix(data):
    """Download VIX separately."""
    print("Downloading VIX...")
    try:
        vix = yf.download("^VIX", start=DATA_START, end="2026-07-30", progress=False, auto_adjust=True)
        if isinstance(vix.columns, pd.MultiIndex):
            vix.columns = vix.columns.droplevel(1)
        vix.index = pd.to_datetime(vix.index).tz_localize(None)
        print(f"  VIX: {len(vix)} rows")
        return vix["Close"]
    except Exception as e:
        print(f"  VIX FAILED: {e}")
        return None


# ─── BACKTEST ENGINE ──────────────────────────────────────────────────────────
class BacktestEngine:
    """Simple long-only backtest engine with slippage."""

    def __init__(self, capital=START_CAPITAL):
        self.initial_capital = capital
        self.capital = capital
        self.positions = {}  # ticker -> {shares, entry_price, entry_date}
        self.trades = []
        self.equity_curve = []

    def buy(self, ticker, price, date, fraction=1.0):
        """Buy with slippage."""
        exec_price = price * (1 + SLIPPAGE_PCT)
        invest = self.capital * fraction
        if invest < 1:
            return
        shares = invest / exec_price
        self.capital -= invest
        self.positions[ticker] = {
            "shares": shares,
            "entry_price": exec_price,
            "entry_date": date,
        }

    def sell(self, ticker, price, date):
        """Sell with slippage."""
        if ticker not in self.positions:
            return
        pos = self.positions[ticker]
        exec_price = price * (1 - SLIPPAGE_PCT)
        proceeds = pos["shares"] * exec_price
        pnl = proceeds - pos["shares"] * pos["entry_price"]
        self.capital += proceeds
        self.trades.append({
            "ticker": ticker,
            "entry_date": str(pos["entry_date"]),
            "exit_date": str(date),
            "entry_price": pos["entry_price"],
            "exit_price": exec_price,
            "pnl": pnl,
            "return_pct": pnl / (pos["shares"] * pos["entry_price"]) * 100,
        })
        del self.positions[ticker]

    def sell_all(self, prices, date):
        """Sell all positions."""
        for ticker in list(self.positions.keys()):
            if ticker in prices:
                self.sell(ticker, prices[ticker], date)

    def mark_to_market(self, prices, date):
        """Record equity curve point."""
        mtm = self.capital
        for ticker, pos in self.positions.items():
            if ticker in prices:
                mtm += pos["shares"] * prices[ticker]
        self.equity_curve.append({"date": str(date), "equity": mtm})
        return mtm

    def get_equity_series(self):
        """Return equity as pandas Series."""
        ec = pd.DataFrame(self.equity_curve)
        ec["date"] = pd.to_datetime(ec["date"])
        return ec.set_index("date")["equity"]


# ─── METRICS ──────────────────────────────────────────────────────────────────
def compute_metrics(engine, qqq_returns=None):
    """Compute all required metrics."""
    equity = engine.get_equity_series()
    if len(equity) < 2:
        return None

    daily_ret = equity.pct_change().dropna()
    total_ret = (equity.iloc[-1] / equity.iloc[0] - 1) * 100

    # Annualized
    n_years = len(daily_ret) / 252
    ann_ret = (1 + total_ret / 100) ** (1 / max(n_years, 0.01)) - 1
    ann_vol = daily_ret.std() * np.sqrt(252)

    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
    downside = daily_ret[daily_ret < 0].std() * np.sqrt(252)
    sortino = ann_ret / downside if downside > 0 else 0

    # Max drawdown
    cummax = equity.cummax()
    drawdown = (equity - cummax) / cummax
    max_dd = drawdown.min() * 100

    # Trade stats
    trades = engine.trades
    n_trades = len(trades)
    if n_trades > 0:
        wins = [t for t in trades if t["pnl"] > 0]
        losses = [t for t in trades if t["pnl"] <= 0]
        win_rate = len(wins) / n_trades * 100
        gross_profit = sum(t["pnl"] for t in wins) if wins else 0
        gross_loss = abs(sum(t["pnl"] for t in losses)) if losses else 0.001
        profit_factor = gross_profit / gross_loss
    else:
        win_rate = 0
        profit_factor = 0

    # QQQ correlation
    qqq_corr = 0.0
    if qqq_returns is not None and len(daily_ret) > 10:
        aligned = pd.concat([daily_ret, qqq_returns], axis=1, join="inner")
        aligned.columns = ["strat", "qqq"]
        aligned = aligned.dropna()
        if len(aligned) > 10:
            qqq_corr = float(aligned["strat"].corr(aligned["qqq"]))

    return {
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "total_return_pct": round(total_ret, 2),
        "max_drawdown_pct": round(max_dd, 2),
        "profit_factor": round(profit_factor, 3),
        "win_rate_pct": round(win_rate, 1),
        "n_trades": n_trades,
        "final_equity": round(equity.iloc[-1], 2),
        "ann_return_pct": round(ann_ret * 100, 2),
        "ann_volatility_pct": round(ann_vol * 100, 2),
        "qqq_correlation": round(qqq_corr, 3),
        "daily_returns": daily_ret,
        "equity_series": equity,
    }


def permutation_test(engine_trades, n_iter=PERM_ITERS):
    """Permutation test: shuffle trade PnLs, compute fraction with Sharpe >= actual."""
    if len(engine_trades) < 5:
        return 1.0

    pnls = np.array([t["pnl"] for t in engine_trades])
    actual_sharpe = pnls.mean() / (pnls.std() + 1e-10)

    count = 0
    for _ in range(n_iter):
        shuffled = np.random.choice(pnls, size=len(pnls), replace=True)
        # Randomly flip signs to destroy temporal structure
        signs = np.random.choice([-1, 1], size=len(pnls))
        shuffled = pnls * signs
        s = shuffled.mean() / (shuffled.std() + 1e-10)
        if s >= actual_sharpe:
            count += 1

    return round(count / n_iter, 4)


def regime_split(metrics, spy_close, equity_series):
    """Split performance by bull/bear regime (SPY vs 200-SMA)."""
    sma200 = spy_close.rolling(200).mean()
    bull_mask = spy_close > sma200
    bear_mask = spy_close <= sma200

    daily_ret = equity_series.pct_change().dropna()

    results = {}
    for regime_name, mask in [("bull", bull_mask), ("bear", bear_mask)]:
        aligned = pd.concat([daily_ret, mask], axis=1, join="inner")
        aligned.columns = ["ret", "in_regime"]
        regime_rets = aligned[aligned["in_regime"]]["ret"]

        if len(regime_rets) > 10:
            ann_vol = regime_rets.std() * np.sqrt(252)
            ann_ret = regime_rets.mean() * 252
            sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
            results[regime_name] = {
                "sharpe": round(sharpe, 3),
                "n_days": int(len(regime_rets)),
                "mean_daily_ret_pct": round(regime_rets.mean() * 100, 4),
            }
        else:
            results[regime_name] = {"sharpe": 0, "n_days": 0, "mean_daily_ret_pct": 0}

    return results


def check_gates(metrics, perm_p, regime_data, n_trades):
    """Check all validation gates."""
    gates = {}
    gates["sharpe_gt_0.5"] = metrics["sharpe"] > 0.5
    gates["perm_p_lt_0.05"] = perm_p < 0.05

    bull_s = regime_data.get("bull", {}).get("sharpe", 0)
    bear_s = regime_data.get("bear", {}).get("sharpe", 0)
    max_s = max(abs(bull_s), abs(bear_s), 0.001)
    regime_gap = abs(bull_s - bear_s) / max_s
    gates["regime_gap_lt_0.5"] = regime_gap < 0.5

    gates["mdd_gt_neg50"] = metrics["max_drawdown_pct"] > -50
    gates["trades_gte_20"] = n_trades >= 20

    return gates


# ─── STRATEGY IMPLEMENTATIONS ────────────────────────────────────────────────

def run_variant_a(data, vix_data):
    """A: REIT Rate Sensitivity — Long VNQ when TLT trending up."""
    print("\n=== Variant A: REIT Rate Sensitivity ===")
    engine = BacktestEngine()

    vnq = get_close(data, "VNQ")
    tlt = get_close(data, "TLT")

    # TLT 20d return
    tlt_ret20 = tlt.pct_change(20)

    oot_dates = vnq.loc[OOT_START:OOT_END].index
    hold_counter = 0
    in_position = False

    for i, date in enumerate(oot_dates):
        if date not in tlt_ret20.index:
            continue

        price_dict = {}
        for t in ["VNQ", "TLT"]:
            if date in get_close(data, t).index:
                price_dict[t] = get_close(data, t).loc[date]

        engine.mark_to_market(price_dict, date)

        if in_position:
            hold_counter += 1
            if hold_counter >= 10:
                # Reassess
                signal = tlt_ret20.loc[date] if date in tlt_ret20.index else 0
                if signal <= 0:
                    engine.sell("VNQ", vnq.loc[date], date)
                    in_position = False
                    hold_counter = 0
                else:
                    hold_counter = 0  # reset, keep holding
        else:
            signal = tlt_ret20.loc[date] if date in tlt_ret20.index else 0
            if signal > 0 and date in vnq.index:
                engine.buy("VNQ", vnq.loc[date], date)
                in_position = True
                hold_counter = 0

    # Close any open position
    last_date = oot_dates[-1]
    if in_position and last_date in vnq.index:
        engine.sell("VNQ", vnq.loc[last_date], last_date)

    return engine


def run_variant_b(data, vix_data):
    """B: REIT Sector Rotation — Monthly rank by 1m momentum, long top 2."""
    print("\n=== Variant B: REIT Sector Rotation ===")
    engine = BacktestEngine()

    tickers = ["VNQ", "XLRE", "RWR", "REM"]
    closes = {t: get_close(data, t) for t in tickers}
    mom1m = {t: closes[t].pct_change(21) for t in tickers}

    # Get OOT trading dates from VNQ
    oot_dates = closes["VNQ"].loc[OOT_START:OOT_END].index
    last_rebal = None

    for date in oot_dates:
        price_dict = {}
        for t in tickers:
            if date in closes[t].index:
                price_dict[t] = closes[t].loc[date]

        engine.mark_to_market(price_dict, date)

        # Rebalance monthly (every ~21 trading days)
        if last_rebal is None or (date - last_rebal).days >= 28:
            # Rank by 1m momentum
            scores = {}
            for t in tickers:
                if date in mom1m[t].index and not np.isnan(mom1m[t].loc[date]):
                    scores[t] = mom1m[t].loc[date]

            if len(scores) >= 2:
                ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
                top2 = [r[0] for r in ranked[:2]]

                # Sell anything not in top 2
                for t in list(engine.positions.keys()):
                    if t not in top2 and t in price_dict:
                        engine.sell(t, price_dict[t], date)

                # Buy top 2 (equal weight)
                to_buy = [t for t in top2 if t not in engine.positions]
                if to_buy:
                    frac = 1.0 / len(to_buy) if len(engine.positions) == 0 else 0.5
                    for t in to_buy:
                        if t in price_dict:
                            engine.buy(t, price_dict[t], date, fraction=min(frac, 1.0))

                last_rebal = date

    # Close all
    last_date = oot_dates[-1]
    price_dict = {t: closes[t].loc[last_date] for t in tickers if last_date in closes[t].index}
    engine.sell_all(price_dict, last_date)

    return engine


def run_variant_c(data, vix_data):
    """C: REIT-Tech Divergence — Long VNQ when VNQ/QQQ ratio drops >5% below 60d MA."""
    print("\n=== Variant C: REIT-Tech Divergence ===")
    engine = BacktestEngine()

    vnq = get_close(data, "VNQ")
    qqq = get_close(data, "QQQ")

    ratio = vnq / qqq
    ratio_ma60 = ratio.rolling(60).mean()

    oot_dates = vnq.loc[OOT_START:OOT_END].index
    hold_counter = 0
    in_position = False

    for date in oot_dates:
        if date not in ratio.index or date not in ratio_ma60.index:
            continue

        price_dict = {"VNQ": vnq.loc[date]} if date in vnq.index else {}
        engine.mark_to_market(price_dict, date)

        if in_position:
            hold_counter += 1
            if hold_counter >= 20:
                engine.sell("VNQ", vnq.loc[date], date)
                in_position = False
                hold_counter = 0
        else:
            r = ratio.loc[date]
            ma = ratio_ma60.loc[date]
            if not np.isnan(ma) and ma > 0:
                deviation = (r - ma) / ma
                if deviation < -0.05:
                    engine.buy("VNQ", vnq.loc[date], date)
                    in_position = True
                    hold_counter = 0

    last_date = oot_dates[-1]
    if in_position and last_date in vnq.index:
        engine.sell("VNQ", vnq.loc[last_date], last_date)

    return engine


def run_variant_d(data, vix_data):
    """D: International REIT Diversification — VNQI vs VNQ relative momentum."""
    print("\n=== Variant D: International REIT Diversification ===")
    engine = BacktestEngine()

    vnq = get_close(data, "VNQ")
    vnqi = get_close(data, "VNQI")

    vnq_ret20 = vnq.pct_change(20)
    vnqi_ret20 = vnqi.pct_change(20)

    oot_dates = vnq.loc[OOT_START:OOT_END].index
    last_rebal = None
    current_holding = None

    for date in oot_dates:
        price_dict = {}
        for t in ["VNQ", "VNQI"]:
            if date in get_close(data, t).index:
                price_dict[t] = get_close(data, t).loc[date]

        engine.mark_to_market(price_dict, date)

        if last_rebal is None or (date - last_rebal).days >= 15:
            vnqi_r = vnqi_ret20.loc[date] if date in vnqi_ret20.index and not np.isnan(vnqi_ret20.loc[date]) else -999
            vnq_r = vnq_ret20.loc[date] if date in vnq_ret20.index and not np.isnan(vnq_ret20.loc[date]) else -999

            target = "VNQI" if vnqi_r > vnq_r else "VNQ"

            if current_holding != target:
                # Sell current
                if current_holding and current_holding in price_dict:
                    engine.sell(current_holding, price_dict[current_holding], date)
                # Buy target
                if target in price_dict:
                    engine.buy(target, price_dict[target], date)
                    current_holding = target

            last_rebal = date

    last_date = oot_dates[-1]
    if current_holding:
        price_dict = {}
        for t in ["VNQ", "VNQI"]:
            if last_date in get_close(data, t).index:
                price_dict[t] = get_close(data, t).loc[last_date]
        engine.sell_all(price_dict, last_date)

    return engine


def run_variant_e(data, vix_data):
    """E: Mortgage REIT Yield Play — Long REM when VIX < 20 AND TLT trending up."""
    print("\n=== Variant E: Mortgage REIT Yield Play ===")
    engine = BacktestEngine()

    rem = get_close(data, "REM")
    tlt = get_close(data, "TLT")
    tlt_ret20 = tlt.pct_change(20)

    oot_dates = rem.loc[OOT_START:OOT_END].index
    hold_counter = 0
    in_position = False

    for date in oot_dates:
        price_dict = {"REM": rem.loc[date]} if date in rem.index else {}
        engine.mark_to_market(price_dict, date)

        vix_val = vix_data.loc[date] if (vix_data is not None and date in vix_data.index) else 25
        tlt_sig = tlt_ret20.loc[date] if date in tlt_ret20.index else 0

        if in_position:
            hold_counter += 1
            if hold_counter >= 10:
                # Reassess
                if vix_val >= 20 or tlt_sig <= 0:
                    engine.sell("REM", rem.loc[date], date)
                    in_position = False
                    hold_counter = 0
                else:
                    hold_counter = 0  # reset, keep holding
        else:
            if vix_val < 20 and tlt_sig > 0:
                engine.buy("REM", rem.loc[date], date)
                in_position = True
                hold_counter = 0

    last_date = oot_dates[-1]
    if in_position and last_date in rem.index:
        engine.sell("REM", rem.loc[last_date], last_date)

    return engine


def run_variant_f(data, vix_data):
    """F: REIT Momentum + VIX Filter — VNQ momentum score + VIX < 25."""
    print("\n=== Variant F: REIT Momentum + VIX Filter ===")
    engine = BacktestEngine()

    vnq = get_close(data, "VNQ")
    mom1m = vnq.pct_change(21)
    mom3m = vnq.pct_change(63)

    oot_dates = vnq.loc[OOT_START:OOT_END].index
    hold_counter = 0
    in_position = False

    for date in oot_dates:
        price_dict = {"VNQ": vnq.loc[date]} if date in vnq.index else {}
        engine.mark_to_market(price_dict, date)

        vix_val = vix_data.loc[date] if (vix_data is not None and date in vix_data.index) else 30

        m1 = mom1m.loc[date] if date in mom1m.index and not np.isnan(mom1m.loc[date]) else 0
        m3 = mom3m.loc[date] if date in mom3m.index and not np.isnan(mom3m.loc[date]) else 0
        score = m1 + m3  # simple additive

        if in_position:
            hold_counter += 1
            if hold_counter >= 15:
                if score <= 0 or vix_val >= 25:
                    engine.sell("VNQ", vnq.loc[date], date)
                    in_position = False
                    hold_counter = 0
                else:
                    hold_counter = 0
        else:
            if score > 0 and vix_val < 25:
                engine.buy("VNQ", vnq.loc[date], date)
                in_position = True
                hold_counter = 0

    last_date = oot_dates[-1]
    if in_position and last_date in vnq.index:
        engine.sell("VNQ", vnq.loc[last_date], last_date)

    return engine


# ─── MAIN ─────────────────────────────────────────────────────────────────────
def main():
    data = download_data()
    vix_data = get_vix(data)

    qqq_close = get_close(data, "QQQ")
    qqq_returns = qqq_close.pct_change().dropna()
    spy_close = get_close(data, "SPY")

    variants = {
        "A": ("REIT Rate Sensitivity (TLT signal)", run_variant_a),
        "B": ("REIT Sector Rotation (top 2 momentum)", run_variant_b),
        "C": ("REIT-Tech Divergence (VNQ/QQQ mean reversion)", run_variant_c),
        "D": ("International REIT Diversification (VNQI vs VNQ)", run_variant_d),
        "E": ("Mortgage REIT Yield Play (REM + VIX/TLT)", run_variant_e),
        "F": ("REIT Momentum + VIX Filter", run_variant_f),
    }

    results = {
        "meta": {
            "strategy_category": "REIT / Real Estate Momentum",
            "oot_period": f"{OOT_START} to {OOT_END}",
            "starting_capital": START_CAPITAL,
            "slippage_pct": SLIPPAGE_PCT * 100,
            "commission": COMMISSION,
            "data_source": "yfinance",
            "run_timestamp": str(dt.datetime.now()),
            "n_variants": len(variants),
        },
        "strategies": {},
    }

    for key, (name, func) in variants.items():
        print(f"\n{'='*60}")
        print(f"Running Variant {key}: {name}")
        print(f"{'='*60}")

        try:
            engine = func(data, vix_data)
            metrics = compute_metrics(engine, qqq_returns)

            if metrics is None:
                results["strategies"][key] = {
                    "name": name,
                    "VALIDATED": False,
                    "error": "Insufficient data for metrics",
                }
                continue

            # Permutation test
            perm_p = permutation_test(engine.trades)

            # Regime split
            regime = regime_split(metrics, spy_close, metrics["equity_series"])

            # Gate checks
            gates = check_gates(metrics, perm_p, regime, metrics["n_trades"])
            gates_passed = all(gates.values())

            # Clean metrics for JSON (remove pandas objects)
            clean_metrics = {k: v for k, v in metrics.items()
                            if k not in ("daily_returns", "equity_series")}

            results["strategies"][key] = {
                "name": name,
                "metrics": clean_metrics,
                "n_trades": metrics["n_trades"],
                "qqq_correlation": metrics["qqq_correlation"],
                "perm_p_value": perm_p,
                "regime": regime,
                "gates": gates,
                "gates_passed": sum(gates.values()),
                "gates_total": len(gates),
                "VALIDATED": gates_passed,
            }

            status = "PASS" if gates_passed else "FAIL"
            print(f"\n  Variant {key} [{status}]: Sharpe={metrics['sharpe']}, "
                  f"Return={metrics['total_return_pct']}%, "
                  f"MDD={metrics['max_drawdown_pct']}%, "
                  f"QQQ_corr={metrics['qqq_correlation']}, "
                  f"Perm_p={perm_p}, Trades={metrics['n_trades']}")
            print(f"  Gates: {gates}")

        except Exception as e:
            import traceback
            results["strategies"][key] = {
                "name": name,
                "VALIDATED": False,
                "error": str(e),
                "traceback": traceback.format_exc(),
            }
            print(f"  Variant {key} ERROR: {e}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    validated = sum(1 for s in results["strategies"].values() if s.get("VALIDATED"))
    print(f"Validated: {validated}/{len(variants)}")
    for key, strat in results["strategies"].items():
        m = strat.get("metrics", {})
        status = "PASS" if strat.get("VALIDATED") else "FAIL"
        print(f"  {key}: [{status}] Sharpe={m.get('sharpe','N/A')}, "
              f"QQQ_corr={strat.get('qqq_correlation','N/A')}, "
              f"Return={m.get('total_return_pct','N/A')}%, "
              f"Gates={strat.get('gates_passed',0)}/{strat.get('gates_total',5)}")

    # Save
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {RESULTS_PATH}")

    return results


if __name__ == "__main__":
    main()
