#!/usr/bin/env python3
"""
Mean Reversion Sector Rotation
===============================

Our production strategy ranks by MOMENTUM (buy winners, sell losers).
This tests the OPPOSITE thesis: buy the worst-performing sectors (reversal).

Academic evidence suggests:
- Short-term (1-4 weeks): reversal dominates
- Medium-term (3-12 months): momentum dominates
- Long-term (3-5 years): reversal dominates

Since our DTE=28 (monthly) aligns with short-term reversal horizon, this might work.

Variants:
  A: Momentum baseline (V9.1 production — buy winners)
  B: Pure reversal (buy losers, sell winners)
  C: Short-term reversal + medium momentum (buy 1w losers with 3m uptrend)
  D: Relative strength reversal (buy sectors that underperformed most vs SPY last week)
  E: Sector RSI reversal (buy sectors with RSI<30, sell RSI>70)
  F: Momentum-reversal regime switch (reversal when VIX>25, momentum when VIX<25)

Output: output/growth_research/v1_mean_reversion_sectors_v1/
MLflow experiment: v1_mean_reversion_sectors_v1

KB target: Does mean reversion outperform momentum for DTE=28 sector spreads?
"""

import json
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

_builtin_print = print
def fprint(*args, **kwargs):
    _builtin_print(*args, **kwargs)
    sys.stdout.flush()

sys.path.insert(0, "/home/jupiter/Lvl3Quant")
from research.tools.options_pricer import (
    price_bull_call_spread, price_bear_put_spread, COMMISSION_RT_SPREAD,
)
from research.tools.adversarial_validator import validate_trades

BASE = Path("/home/jupiter/Lvl3Quant")
CHAINS_DIR = BASE / "wheel_strategy_v1" / "data" / "cache" / "options_real" / "chains"
OUTPUT_DIR = BASE / "output" / "growth_research" / "v1_mean_reversion_sectors_v1"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SECTORS = ["XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLB", "XLU", "XLRE", "XLC"]
EXTRA_TICKERS = ["SPY", "^VIX", "^VIX3M", "TLT", "SHY", "HYG", "GLD"]
CAP = 645.0
TOP_K = 3
DTE = 28
REBAL_INTERVAL = 10

COST_WIDTH_MAX = 0.50
DTE_TOLERANCE = 5
STRIKE_TOLERANCE = 0.03

REGIME_FILE = BASE / "output" / "regime_detector_v1" / "regime_predictions_v1.npz"
MLFLOW_URI = "http://jupiter:5000"
EXPERIMENT_NAME = "v1_mean_reversion_sectors_v1"
REGIME_BULL_THRESHOLD = 0.4

MLFLOW_OK = False
try:
    import urllib.request
    urllib.request.urlopen(MLFLOW_URI, timeout=3)
    import mlflow
    mlflow.set_tracking_uri(MLFLOW_URI)
    MLFLOW_OK = True
    fprint(f"MLflow connected: {MLFLOW_URI}")
except Exception:
    fprint("MLflow unavailable")


def load_all_chains():
    chains = {}
    for tk in SECTORS:
        path = CHAINS_DIR / f"{tk}.parquet"
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        df["date"] = pd.to_datetime(df["date"])
        df["expiration"] = pd.to_datetime(df["expiration"])
        for c in ["strike", "bid", "ask", "mid", "vol", "delta"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        df["dte"] = (df["expiration"] - df["date"]).dt.days
        chains[tk] = df
        fprint(f"  {tk}: {len(df):,} rows")
    return chains


def find_chain_spread_price(chain_df, trade_date, direction, K1, K2, dte_target):
    if chain_df is None:
        return None
    day_mask = chain_df["date"] == pd.Timestamp(trade_date)
    chain_day = chain_df[day_mask]
    if chain_day.empty:
        nearby = chain_df[(chain_df["date"] >= pd.Timestamp(trade_date) - pd.Timedelta(days=2)) &
                          (chain_df["date"] <= pd.Timestamp(trade_date) + pd.Timedelta(days=2))]
        if nearby.empty:
            return None
        nearest_date = min(nearby["date"].unique(),
                          key=lambda x: abs((x - pd.Timestamp(trade_date)).days))
        chain_day = chain_df[chain_df["date"] == nearest_date]
    exps = chain_day[["expiration", "dte"]].drop_duplicates()
    exps["dte_dist"] = (exps["dte"] - dte_target).abs()
    valid_exps = exps[exps["dte_dist"] <= DTE_TOLERANCE]
    if valid_exps.empty:
        return None
    best_exp_row = valid_exps.loc[valid_exps["dte_dist"].idxmin()]
    chain_exp = chain_day[chain_day["expiration"] == best_exp_row["expiration"]]
    opt_type = "c" if direction == "bull" else "p"
    near_target = K1 if direction == "bull" else K2
    far_target = K2 if direction == "bull" else K1
    near_opts = chain_exp[chain_exp["type"] == opt_type].copy()
    if near_opts.empty:
        return None
    near_opts["dist"] = (near_opts["strike"] - near_target).abs()
    near_leg = near_opts.sort_values("dist").iloc[0]
    if near_leg["dist"] / max(near_target, 1) > STRIKE_TOLERANCE:
        return None
    far_opts = chain_exp[chain_exp["type"] == opt_type].copy()
    far_opts["dist"] = (far_opts["strike"] - far_target).abs()
    far_leg = far_opts.sort_values("dist").iloc[0]
    if far_leg["dist"] / max(far_target, 1) > STRIKE_TOLERANCE:
        return None
    near_mid = float(near_leg["mid"]) if not pd.isna(near_leg["mid"]) else \
        (float(near_leg["bid"]) + float(near_leg["ask"])) / 2
    far_mid = float(far_leg["mid"]) if not pd.isna(far_leg["mid"]) else \
        (float(far_leg["bid"]) + float(far_leg["ask"])) / 2
    spread_cost_mid = abs(near_mid - far_mid)
    return {
        "found": True, "spread_cost_mid": spread_cost_mid,
        "near_strike": float(near_leg["strike"]), "far_strike": float(far_leg["strike"]),
    }


def download_data():
    import yfinance as yf
    all_tickers = SECTORS + EXTRA_TICKERS
    fprint(f"Downloading {len(all_tickers)} tickers...")
    raw = yf.download(all_tickers, start="2008-01-01", progress=False, auto_adjust=True)
    mi = isinstance(raw.columns, pd.MultiIndex)
    close = raw["Close"] if mi else raw[["Close"]]
    high = raw["High"] if mi else raw[["High"]]
    low = raw["Low"] if mi else raw[["Low"]]
    if isinstance(close.columns, pd.MultiIndex):
        close.columns = close.columns.get_level_values(-1)
        high.columns = high.columns.get_level_values(-1)
        low.columns = low.columns.get_level_values(-1)
    close = close.ffill(); high = high.ffill(); low = low.ffill()
    rename_map = {"^VIX": "VIX", "^VIX3M": "VIX3M"}
    close = close.rename(columns=rename_map)
    high = high.rename(columns=rename_map)
    low = low.rename(columns=rename_map)
    fprint(f"Data: {len(close)} days, {close.index[0].date()} to {close.index[-1].date()}")
    return close, high, low


def load_regime_predictions():
    if not REGIME_FILE.exists():
        return None
    data = np.load(REGIME_FILE, allow_pickle=True)
    dates = pd.to_datetime(data["dates"])
    scores = data["regime_scores"]
    regime_series = pd.Series(scores, index=dates, name="regime_score")
    regime_series = regime_series[~regime_series.index.duplicated(keep="last")]
    return regime_series


def get_regime_score_at(regime_series, dt):
    if regime_series is None:
        return 0.5
    if dt in regime_series.index:
        return float(regime_series.loc[dt])
    nearest = regime_series.index[regime_series.index.get_indexer([dt], method="ffill")]
    if len(nearest) > 0:
        return float(regime_series.loc[nearest[0]])
    return 0.5


def compute_atr_series(high, low, close, period=14):
    atr_dict = {}
    for tk in SECTORS:
        if tk in high.columns and tk in low.columns and tk in close.columns:
            h = high[tk].dropna(); l = low[tk].dropna(); c = close[tk].dropna()
            common = h.index.intersection(l.index).intersection(c.index)
            if len(common) > period:
                tr1 = h.loc[common] - l.loc[common]
                tr2 = (h.loc[common] - c.loc[common].shift(1)).abs()
                tr3 = (l.loc[common] - c.loc[common].shift(1)).abs()
                tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
                atr_dict[tk] = tr.ewm(alpha=1/period, min_periods=period).mean()
    return atr_dict


def compute_sector_rankings(close, dt, mode="momentum"):
    """Compute sector rankings based on different strategies.

    Returns dict of {ticker: score} where HIGHER score = buy bull, LOWER = buy bear.
    """
    idx = close.index.get_indexer([dt], method="ffill")[0]
    if idx < 260:
        return {}

    sector_cols = [c for c in SECTORS if c in close.columns]
    scores = {}

    for tk in sector_cols:
        px = close[tk].iloc[:idx + 1].dropna()
        if len(px) < 260:
            continue

        if mode == "momentum":
            # Standard momentum: buy winners (high ret = high score)
            ret_5d = float(px.iloc[-1] / px.iloc[-5] - 1) if len(px) > 5 else 0
            ret_21d = float(px.iloc[-1] / px.iloc[-21] - 1) if len(px) > 21 else 0
            ret_63d = float(px.iloc[-1] / px.iloc[-63] - 1) if len(px) > 63 else 0
            scores[tk] = ret_5d * 0.2 + ret_21d * 0.3 + ret_63d * 0.5

        elif mode == "reversal":
            # Pure reversal: buy losers (low ret = high score, INVERTED)
            ret_5d = float(px.iloc[-1] / px.iloc[-5] - 1) if len(px) > 5 else 0
            ret_10d = float(px.iloc[-1] / px.iloc[-10] - 1) if len(px) > 10 else 0
            scores[tk] = -(ret_5d * 0.6 + ret_10d * 0.4)  # Negate = buy losers

        elif mode == "reversal_with_trend":
            # Short-term reversal but only for sectors with medium-term uptrend
            ret_5d = float(px.iloc[-1] / px.iloc[-5] - 1) if len(px) > 5 else 0
            ret_63d = float(px.iloc[-1] / px.iloc[-63] - 1) if len(px) > 63 else 0
            # Buy if: dropped recently BUT still in uptrend (ret_63d > 0)
            if ret_63d > 0:
                scores[tk] = -ret_5d  # Buy the dip in uptrending sectors
            else:
                scores[tk] = ret_5d  # In downtrend, buy momentum (don't catch falling knife)

        elif mode == "relative_strength_reversal":
            # Buy sectors that underperformed SPY most recently
            spy = close["SPY"].iloc[:idx + 1].dropna()
            if len(spy) < 10:
                continue
            spy_ret_5d = float(spy.iloc[-1] / spy.iloc[-5] - 1) if len(spy) > 5 else 0
            sec_ret_5d = float(px.iloc[-1] / px.iloc[-5] - 1) if len(px) > 5 else 0
            relative = sec_ret_5d - spy_ret_5d
            scores[tk] = -relative  # Buy underperformers

        elif mode == "rsi_reversal":
            # RSI-based: buy oversold sectors, sell overbought
            rets = px.pct_change().dropna()
            if len(rets) < 14:
                continue
            recent = rets.iloc[-14:]
            gains = recent[recent > 0].sum()
            losses = abs(recent[recent < 0].sum())
            if losses == 0:
                rsi = 100
            else:
                rs = gains / losses
                rsi = 100 - (100 / (1 + rs))
            # Invert: low RSI = high score (buy oversold)
            scores[tk] = -rsi + 50  # Center at 0, oversold positive

        elif mode == "regime_switch":
            # Use VIX to decide: reversal when stressed, momentum when calm
            vix = close["VIX"] if "VIX" in close.columns else None
            cv = float(vix.iloc[idx]) if vix is not None and not pd.isna(vix.iloc[idx]) else 20
            ret_5d = float(px.iloc[-1] / px.iloc[-5] - 1) if len(px) > 5 else 0
            ret_63d = float(px.iloc[-1] / px.iloc[-63] - 1) if len(px) > 63 else 0
            if cv > 25:
                # High VIX = reversal (buy recent losers)
                scores[tk] = -ret_5d
            else:
                # Low VIX = momentum (buy recent winners)
                scores[tk] = ret_5d * 0.3 + ret_63d * 0.7

    return scores


def compute_strikes(S, direction):
    if direction == "bull":
        K1 = round(S * 1.02, 2)
        pct_w = K1 * 0.03
        w = max(3.0, pct_w)
        K2 = round(K1 + w, 2)
    else:
        K2 = round(S * 0.98, 2)
        pct_w = K2 * 0.03
        w = max(3.0, pct_w)
        K1 = round(K2 - w, 2)
    if K2 <= K1:
        K2 = K1 + 0.50
    return K1, K2


def execute_trade(tk, dt, direction, close, atr_dict, vix_val, equity, chains, max_pos):
    if tk not in close.columns or tk not in atr_dict:
        return None
    S = float(close[tk].loc[dt])
    di = close.index.get_loc(dt)
    ei = min(di + DTE, len(close) - 1)
    if ei <= di:
        return None
    av = float(atr_dict[tk].loc[dt]) if dt in atr_dict[tk].index and not pd.isna(atr_dict[tk].loc[dt]) else S * 0.015
    K1, K2 = compute_strikes(S, direction)
    used_real = False
    entry_cost_ps = None
    chain_df = chains.get(tk)
    if chain_df is not None:
        result = find_chain_spread_price(chain_df, dt, direction, K1, K2, DTE)
        if result and result["found"]:
            entry_cost_ps = result["spread_cost_mid"]
            used_real = True
            if direction == "bull":
                K1 = result["near_strike"]
                K2 = result["far_strike"]
            else:
                K1 = result["far_strike"]
                K2 = result["near_strike"]
    if entry_cost_ps is None:
        try:
            if direction == "bull":
                entry_cost_ps, _ = price_bull_call_spread(S=S, K1=K1, K2=K2, dte=DTE, atr=av, vix=vix_val)
            else:
                entry_cost_ps, _ = price_bear_put_spread(S=S, K1=K1, K2=K2, dte=DTE, atr=av, vix=vix_val)
        except Exception:
            return None
    if entry_cost_ps is None or entry_cost_ps <= 0:
        return None
    spread_width = abs(K2 - K1)
    cwr = entry_cost_ps / max(spread_width, 0.01)
    if cwr > COST_WIDTH_MAX:
        return None
    total_cost = entry_cost_ps * 100 + COMMISSION_RT_SPREAD
    if total_cost <= 0 or total_cost > max_pos or total_cost > equity * 0.40:
        return None
    Se = float(close[tk].iloc[ei])
    if direction == "bull":
        intrinsic = max(Se - K1, 0.0) - max(Se - K2, 0.0)
    else:
        intrinsic = max(K2 - Se, 0.0) - max(K1 - Se, 0.0)
    pnl = (intrinsic - entry_cost_ps) * 100 - COMMISSION_RT_SPREAD
    return {
        "pnl": round(pnl, 2), "entry_cost_ps": round(entry_cost_ps, 4),
        "total_cost": round(total_cost, 2), "used_real_pricing": used_real,
        "K1": K1, "K2": K2, "S_entry": round(S, 2), "S_exit": round(Se, 2),
        "intrinsic": round(intrinsic, 4), "spread_width": round(spread_width, 2),
        "cost_width_ratio": round(cwr, 3),
    }


def simulate_variant(close, atr_dict, chains, ranking_mode, regime_series):
    """Simulate with different ranking approaches."""
    spy = close["SPY"]
    vix = close["VIX"] if "VIX" in close.columns else None

    equity = CAP
    trades = []
    real_count = 0
    bs_count = 0

    # Build rebalance dates
    all_fridays = pd.DatetimeIndex(
        close.index.to_series().resample("W-FRI").last().dropna().values
    )
    # Biweekly
    n = REBAL_INTERVAL // 5
    rebal_dates = all_fridays[::n]

    # Skip first year for warmup
    rebal_dates = rebal_dates[rebal_dates >= close.index[260]]

    for dt in rebal_dates:
        if dt not in spy.index:
            continue

        # Regime filter
        rscore = get_regime_score_at(regime_series, dt)
        if rscore <= REGIME_BULL_THRESHOLD:
            continue

        cv = float(vix.loc[dt]) if vix is not None and dt in vix.index else 20.0

        scores = compute_sector_rankings(close, dt, mode=ranking_mode)
        if not scores or len(scores) < 5:
            continue

        trade_mode = "pairs" if cv < 20.0 else "bull_only"
        ranked_desc = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        ranked_asc = sorted(scores.items(), key=lambda x: x[1])
        bull_picks = [t for t, _ in ranked_desc[:TOP_K]]
        bear_picks = [t for t, _ in ranked_asc[:TOP_K]] if trade_mode == "pairs" else []

        max_pos = min(100, equity / 6) if trade_mode == "pairs" else min(200, equity / 3)
        if max_pos < 30:
            continue

        for direction, picks in [("bull", bull_picks), ("bear", bear_picks)]:
            for tk in picks:
                result = execute_trade(
                    tk, dt, direction, close, atr_dict, cv, equity, chains, max_pos
                )
                if result is not None:
                    equity += result["pnl"]
                    real_count += 1 if result["used_real_pricing"] else 0
                    bs_count += 0 if result["used_real_pricing"] else 1
                    di = close.index.get_loc(dt)
                    ei = min(di + DTE, len(close) - 1)
                    sv = float(spy.loc[dt])
                    se = float(spy.iloc[ei]) if ei < len(spy) else sv
                    trades.append({
                        **result, "entry_date": str(dt.date()),
                        "exit_date": str(close.index[ei].date()),
                        "ticker": tk, "regime": "bull" if se >= sv else "bear",
                        "direction": direction, "vix": round(cv, 1),
                        "win": result["pnl"] > 0, "trade_mode": trade_mode,
                    })

    return trades, equity, real_count, bs_count


def monte_carlo_ci(trades, n_bootstrap=1000, seed=42):
    if len(trades) < 10:
        return None
    pnls = np.array([t["pnl"] for t in trades])
    rng = np.random.RandomState(seed)
    sharpes = [float(np.mean(rng.choice(pnls, len(pnls), True)) /
               (np.std(rng.choice(pnls, len(pnls), True)) + 1e-10) * np.sqrt(52))
               for _ in range(n_bootstrap)]
    return {
        "mean": float(np.mean(sharpes)),
        "ci_95_low": float(np.percentile(sharpes, 2.5)),
        "ci_95_high": float(np.percentile(sharpes, 97.5)),
    }


# Permutation test for alpha
def permutation_test(trades, n_perms=1000, seed=42):
    """Test if ranking adds value vs random sector selection."""
    if len(trades) < 10:
        return None
    actual_pnl = sum(t["pnl"] for t in trades)
    rng = np.random.RandomState(seed)
    random_pnls = []
    pnl_arr = np.array([t["pnl"] for t in trades])
    for _ in range(n_perms):
        shuffled = rng.permutation(pnl_arr)
        random_pnls.append(float(shuffled.sum()))
    p_value = float(np.mean([rp >= actual_pnl for rp in random_pnls]))
    return {"actual_pnl": actual_pnl, "random_mean": np.mean(random_pnls),
            "p_value": p_value, "lift": actual_pnl / max(np.mean(random_pnls), 1)}


VARIANTS = {
    "A_momentum": {"mode": "momentum", "desc": "Momentum (V9.1 production baseline — buy winners)"},
    "B_reversal": {"mode": "reversal", "desc": "Pure reversal (buy 1w losers)"},
    "C_reversal_trend": {"mode": "reversal_with_trend", "desc": "Reversal with trend filter (buy dips in uptrends)"},
    "D_relative_reversal": {"mode": "relative_strength_reversal", "desc": "Relative strength reversal (buy SPY underperformers)"},
    "E_rsi_reversal": {"mode": "rsi_reversal", "desc": "RSI reversal (buy oversold RSI<30)"},
    "F_regime_switch": {"mode": "regime_switch", "desc": "Regime switch (reversal when VIX>25, momentum when calm)"},
}


def main():
    t0 = datetime.now()
    fprint("=" * 100)
    fprint(f"MEAN REVERSION SECTOR ROTATION v1 — {t0.strftime('%Y-%m-%d %H:%M:%S')}")
    fprint("=" * 100)
    fprint(f"Testing reversal vs momentum for DTE=28 sector spreads")
    fprint(f"Config: DTE={DTE}, biweekly rebal, adaptive width, cost/width<{COST_WIDTH_MAX}")
    fprint(f"Capital: ${CAP:.0f}")
    fprint(f"\n{len(VARIANTS)} variants:")
    for vn, vc in VARIANTS.items():
        fprint(f"  {vn}: {vc['desc']}")
    fprint()

    fprint("Loading chains...")
    chains = load_all_chains()
    close, high, low = download_data()
    regime_series = load_regime_predictions()
    atr_dict = compute_atr_series(high, low, close)

    spy_close = close["SPY"]
    all_results = {}

    for vname, vcfg in VARIANTS.items():
        fprint(f"\n{'~' * 90}")
        fprint(f"VARIANT {vname}: {vcfg['desc']}")
        fprint(f"{'~' * 90}")

        trades, final_eq, real_count, bs_count = simulate_variant(
            close, atr_dict, chains, vcfg["mode"], regime_series
        )

        if not trades or len(trades) < 5:
            fprint(f"  Only {len(trades) if trades else 0} trades, skipping")
            all_results[vname] = {"n_trades": len(trades) if trades else 0, "sharpe": 0}
            continue

        total = real_count + bs_count
        fprint(f"\n  Trades: {len(trades)} | Real: {real_count} ({real_count/max(total,1)*100:.0f}%) | "
               f"Final: ${final_eq:,.0f}")

        result = validate_trades(trades, initial_capital=CAP, spy_prices=spy_close, strategy_name=vname)
        result.print_summary()

        for side in ["bull", "bear"]:
            st = [t for t in trades if t["direction"] == side]
            if st:
                pnls = [t["pnl"] for t in st]
                wr = sum(1 for p in pnls if p > 0) / len(pnls)
                fprint(f"  {side}: {len(st)} trades, WR {wr:.1%}, PnL ${sum(pnls):,.0f}")

        real_t = [t for t in trades if t.get("used_real_pricing")]
        real_sharpe = None
        if real_t:
            rp = [t["pnl"] for t in real_t]
            real_sharpe = np.mean(rp) / (np.std(rp) + 1e-10) * np.sqrt(52)
            fprint(f"  REAL-ONLY: {len(real_t)} trades, Sharpe {real_sharpe:.2f}, "
                   f"WR {sum(1 for p in rp if p > 0)/len(rp):.1%}")

        # Year-by-year
        yearly = {}
        for t in trades:
            yr = t["entry_date"][:4]
            yearly.setdefault(yr, []).append(t["pnl"])
        fprint(f"  Year-by-year:")
        for yr in sorted(yearly.keys()):
            yp = yearly[yr]
            fprint(f"    {yr}: {len(yp)} trades, PnL ${sum(yp):,.0f}, "
                   f"WR {sum(1 for p in yp if p > 0)/len(yp):.1%}")

        pnls = [t["pnl"] for t in trades]
        rd = result.to_dict()
        mc = monte_carlo_ci(trades)
        all_results[vname] = {
            **rd, "final_equity": round(final_eq, 2),
            "monte_carlo": mc,
            "real_pct": round(real_count / max(total, 1) * 100, 1),
            "real_sharpe": round(real_sharpe, 2) if real_sharpe else None,
            "avg_pnl_per_trade": round(np.mean(pnls), 2),
        }

    # ── COMPARISON ──
    fprint(f"\n{'=' * 140}")
    fprint("COMPARISON TABLE — MOMENTUM vs REVERSAL")
    fprint(f"{'=' * 140}")
    fprint(f"{'Variant':<35} {'Trades':>7} {'Sharpe':>8} {'Sortino':>8} {'PF':>6} {'WR':>6} "
           f"{'MDD':>8} {'Final$':>10} {'RealSh':>8} {'AvgPnL':>8}")
    fprint("-" * 140)

    baseline_sharpe = None
    for vn in VARIANTS:
        if vn not in all_results:
            continue
        r = all_results[vn]
        if r.get("n_trades", 0) < 5:
            fprint(f"  {vn:<33} {r.get('n_trades', 0):>7} — insufficient trades —")
            continue
        sharpe = r.get("sharpe", 0)
        if baseline_sharpe is None:
            baseline_sharpe = sharpe
        delta = ((sharpe / baseline_sharpe) - 1) * 100 if baseline_sharpe and baseline_sharpe != 0 else 0
        fprint(f"  {vn:<33} {r.get('n_trades', 0):>7} {sharpe:>8.2f} {r.get('sortino', 0):>8.2f} "
               f"{r.get('profit_factor', 0):>6.2f} {r.get('win_rate', 0):>5.1%} "
               f"{r.get('max_drawdown', 0):>7.1%} ${r['final_equity']:>9,.0f} "
               f"{r.get('real_sharpe', 'N/A'):>8} ${r['avg_pnl_per_trade']:>7.2f} "
               f"({'baseline' if delta == 0 else f'{delta:+.0f}%'})")

    # MC CI
    fprint(f"\n{'=' * 100}")
    fprint("MONTE CARLO 95% CI")
    fprint(f"{'=' * 100}")
    for vn in VARIANTS:
        if vn not in all_results or all_results[vn].get("monte_carlo") is None:
            continue
        mc = all_results[vn]["monte_carlo"]
        fprint(f"  {vn}: Sharpe [{mc['ci_95_low']:.2f}, {mc['ci_95_high']:.2f}], mean {mc['mean']:.2f}")

    # Pick overlap analysis
    fprint(f"\n{'=' * 100}")
    fprint("PICK OVERLAP: MOMENTUM vs REVERSAL")
    fprint(f"{'=' * 100}")
    # Compare which tickers momentum and reversal pick on same dates
    mom_picks = {}
    rev_picks = {}
    for t in all_results.get("A_momentum", {}).get("trades", []):
        pass  # Would need to track picks per date

    # Conclusion
    fprint(f"\n{'=' * 100}")
    fprint("CONCLUSION")
    fprint(f"{'=' * 100}")
    best_vn = max(all_results, key=lambda k: all_results[k].get("sharpe", 0)) if all_results else None
    if best_vn:
        best = all_results[best_vn]
        fprint(f"  BEST: {best_vn} — Sharpe {best.get('sharpe', 0):.2f}")
        if best_vn == "A_momentum":
            fprint(f"  Momentum BEATS all reversal variants")
        elif "reversal" in best_vn.lower() or "rsi" in best_vn.lower():
            fprint(f"  REVERSAL beats momentum! This is a new finding.")
        elif "regime" in best_vn.lower():
            fprint(f"  Regime switching beats both pure approaches")

    # Save
    with open(OUTPUT_DIR / "results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    elapsed = (datetime.now() - t0).total_seconds()
    fprint(f"\nCompleted in {elapsed:.1f}s ({elapsed/60:.1f} min)")

    # MLflow
    if MLFLOW_OK:
        try:
            mlflow.set_experiment(EXPERIMENT_NAME)
            with mlflow.start_run(run_name="mean_reversion_sectors_v1"):
                mlflow.log_param("dte", DTE)
                mlflow.log_param("rebal_interval", REBAL_INTERVAL)
                mlflow.log_param("n_variants", len(VARIANTS))
                for vn, vr in all_results.items():
                    mlflow.log_metric(f"{vn}_sharpe", vr.get("sharpe", 0))
                    mlflow.log_metric(f"{vn}_trades", vr.get("n_trades", 0))
                    if vr.get("real_sharpe"):
                        mlflow.log_metric(f"{vn}_real_sharpe", vr["real_sharpe"])
                mlflow.log_artifact(str(OUTPUT_DIR / "results.json"))
            fprint("MLflow logged")
        except Exception as e:
            fprint(f"MLflow error: {e}")


if __name__ == "__main__":
    main()
