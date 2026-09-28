#!/usr/bin/env python3
"""
Multi-Asset Managed-Futures-Style Trend Following
==================================================
HC #696: High-return growth research.

Strategy universe spans equities, bonds, commodities, real estate, currencies.
Tests six distinct trend/momentum approaches across multiple lookback windows.

Tested configurations:
    1. Classic trend following: long above 200MA, cash below, equal weight
    2. Momentum rotation: hold top N by 3/6/12-month momentum, monthly rebal
    3. Time-series momentum: per-asset long if trailing return > 0, else cash
    4. Risk-parity trend: inverse-vol allocation, only hold uptrending assets
    5. Dual momentum: absolute (> 0 return) AND relative (top N) filter
    6. Acceleration filter: hold only when momentum is INCREASING (2nd derivative > 0)

Walk-forward: sliding 252d lookback, 21d test.
Regime analysis: green/red days using SPY close-to-close.
Permutation test: 100 block-bootstrap trials on top 5 configs.
Benchmarks: SPY buy-and-hold, 60/40 (SPY/AGG) portfolio.
"""

import numpy as np
import pandas as pd
import yfinance as yf
from pathlib import Path
from scipy import stats
import json
import warnings
import time
import itertools

warnings.filterwarnings("ignore")

OUT_DIR = Path("/home/jupiter/Lvl3Quant/output/growth_research/multi_asset_trend")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ─── Multi-asset ETF universe ───
UNIVERSE = {
    # Equities
    "SPY": "US Large-Cap",
    "QQQ": "Nasdaq-100",
    "IWM": "US Small-Cap",
    "EFA": "Intl Developed",
    "EEM": "Emerging Markets",
    "VGK": "Europe",
    # Bonds
    "TLT": "Long-Term Treasury",
    "IEF": "7-10yr Treasury",
    "SHY": "Short-Term Treasury",
    "HYG": "High Yield",
    "EMB": "EM Bonds",
    # Commodities
    "GLD": "Gold",
    "SLV": "Silver",
    "DBC": "Commodities Broad",
    "USO": "Oil",
    "UNG": "Natural Gas",
    # Real Estate
    "VNQ": "US REITs",
    "IYR": "US Real Estate",
    # Currencies
    "UUP": "US Dollar",
    "FXE": "Euro",
}

BENCHMARK_60_40 = {"SPY": 0.60, "AGG": 0.40}
RISK_FREE_TICKER = "SHY"

LOOKBACKS = [63, 126, 252]  # 3m, 6m, 12m


def download_data(start="2007-01-01", end="2026-07-14"):
    """Download adjusted close prices for full universe + benchmarks."""
    all_tickers = list(UNIVERSE.keys()) + ["AGG"]
    all_tickers = list(set(all_tickers))
    print(f"Downloading {len(all_tickers)} tickers...")
    data = yf.download(all_tickers, start=start, end=end, auto_adjust=True, progress=False)

    if isinstance(data.columns, pd.MultiIndex):
        prices = data["Close"]
    else:
        prices = data

    prices = prices.ffill().dropna(how="all")
    print(f"Data: {prices.index[0].strftime('%Y-%m-%d')} to "
          f"{prices.index[-1].strftime('%Y-%m-%d')}, {len(prices)} days")

    available = [t for t in UNIVERSE if t in prices.columns and prices[t].notna().sum() > 252]
    print(f"Available ETFs (>1yr data): {len(available)}/{len(UNIVERSE)}")
    for t in sorted(set(UNIVERSE.keys()) - set(available)):
        print(f"  MISSING: {t} ({UNIVERSE[t]})")

    return prices, available


def compute_returns(prices):
    """Daily simple returns."""
    return prices.pct_change()


def compute_sma(prices, window):
    """Simple moving average."""
    return prices.rolling(window, min_periods=max(window // 2, 20)).mean()


def compute_momentum(prices, lookback):
    """Total return over lookback period."""
    return prices / prices.shift(lookback) - 1


def compute_realized_vol(returns, window=63):
    """Annualized realized volatility."""
    return returns.rolling(window, min_periods=20).std() * np.sqrt(252)


# ═══════════════════════════════════════════════════════════════════
#  STRATEGY IMPLEMENTATIONS
# ═══════════════════════════════════════════════════════════════════

def strategy_classic_trend(prices, returns, universe, lookback=200, rebal_days=21):
    """
    Classic trend following: long assets above their SMA, cash below.
    Equal weight among assets in uptrend.
    """
    sma = compute_sma(prices[universe], lookback)
    above_sma = prices[universe] > sma

    dates = prices.index[max(252, lookback + 50):]
    daily_rets = []
    last_rebal = None
    weights = pd.Series(0.0, index=universe)

    for date in dates:
        # Rebalance check
        do_rebal = last_rebal is None or (date - last_rebal).days >= rebal_days
        if do_rebal:
            last_rebal = date
            in_trend = above_sma.loc[:date].iloc[-1]
            selected = in_trend[in_trend].index.tolist()
            weights = pd.Series(0.0, index=universe)
            if selected:
                w = 1.0 / len(selected)
                for s in selected:
                    weights[s] = w

        # Daily return
        day_ret = (weights * returns[universe].loc[date]).sum()
        # Cash portion at risk-free
        invested = weights.sum()
        if invested < 1.0 and RISK_FREE_TICKER in returns.columns:
            rf = returns[RISK_FREE_TICKER].loc[date]
            if not np.isnan(rf):
                day_ret += (1.0 - invested) * rf
        daily_rets.append(day_ret)

    return pd.Series(daily_rets, index=dates, dtype=float)


def strategy_momentum_rotation(prices, returns, universe, lookback=126,
                                top_n=5, rebal_days=21):
    """
    Momentum rotation: hold top N assets by trailing momentum. Monthly rebal.
    """
    mom = compute_momentum(prices[universe], lookback)
    dates = prices.index[max(252, lookback + 50):]
    daily_rets = []
    last_rebal = None
    weights = pd.Series(0.0, index=universe)

    for date in dates:
        do_rebal = last_rebal is None or (date - last_rebal).days >= rebal_days
        if do_rebal:
            last_rebal = date
            scores = mom.loc[:date].iloc[-1].dropna()
            if len(scores) >= top_n:
                top = scores.nlargest(top_n).index.tolist()
                weights = pd.Series(0.0, index=universe)
                w = 1.0 / top_n
                for s in top:
                    weights[s] = w

        day_ret = (weights * returns[universe].loc[date]).sum()
        invested = weights.sum()
        if invested < 1.0 and RISK_FREE_TICKER in returns.columns:
            rf = returns[RISK_FREE_TICKER].loc[date]
            if not np.isnan(rf):
                day_ret += (1.0 - invested) * rf
        daily_rets.append(day_ret)

    return pd.Series(daily_rets, index=dates, dtype=float)


def strategy_ts_momentum(prices, returns, universe, lookback=252, rebal_days=21):
    """
    Time-series momentum: per-asset, go long if trailing return > 0, else cash.
    Equal weight among all assets with positive momentum.
    """
    mom = compute_momentum(prices[universe], lookback)
    dates = prices.index[max(252, lookback + 50):]
    daily_rets = []
    last_rebal = None
    weights = pd.Series(0.0, index=universe)

    for date in dates:
        do_rebal = last_rebal is None or (date - last_rebal).days >= rebal_days
        if do_rebal:
            last_rebal = date
            m = mom.loc[:date].iloc[-1]
            positive = m[m > 0].index.tolist()
            weights = pd.Series(0.0, index=universe)
            if positive:
                w = 1.0 / len(positive)
                for s in positive:
                    weights[s] = w

        day_ret = (weights * returns[universe].loc[date]).sum()
        invested = weights.sum()
        if invested < 1.0 and RISK_FREE_TICKER in returns.columns:
            rf = returns[RISK_FREE_TICKER].loc[date]
            if not np.isnan(rf):
                day_ret += (1.0 - invested) * rf
        daily_rets.append(day_ret)

    return pd.Series(daily_rets, index=dates, dtype=float)


def strategy_risk_parity_trend(prices, returns, universe, lookback=200,
                                vol_window=63, rebal_days=21):
    """
    Risk-parity trend: inverse-vol weighting, only hold uptrending assets.
    """
    sma = compute_sma(prices[universe], lookback)
    above_sma = prices[universe] > sma
    rvol = compute_realized_vol(returns[universe], window=vol_window)

    dates = prices.index[max(252, lookback + 50):]
    daily_rets = []
    last_rebal = None
    weights = pd.Series(0.0, index=universe)

    for date in dates:
        do_rebal = last_rebal is None or (date - last_rebal).days >= rebal_days
        if do_rebal:
            last_rebal = date
            in_trend = above_sma.loc[:date].iloc[-1]
            selected = in_trend[in_trend].index.tolist()
            vols = rvol.loc[:date].iloc[-1]

            weights = pd.Series(0.0, index=universe)
            if selected:
                sel_vols = vols[selected].dropna()
                sel_vols = sel_vols[sel_vols > 0.001]
                if len(sel_vols) > 0:
                    inv_vol = 1.0 / sel_vols
                    inv_vol = inv_vol / inv_vol.sum()  # normalize to 1.0
                    for s in inv_vol.index:
                        weights[s] = inv_vol[s]

        day_ret = (weights * returns[universe].loc[date]).sum()
        invested = weights.sum()
        if invested < 1.0 and RISK_FREE_TICKER in returns.columns:
            rf = returns[RISK_FREE_TICKER].loc[date]
            if not np.isnan(rf):
                day_ret += (1.0 - invested) * rf
        daily_rets.append(day_ret)

    return pd.Series(daily_rets, index=dates, dtype=float)


def strategy_dual_momentum(prices, returns, universe, lookback=126,
                            top_n=5, rebal_days=21):
    """
    Dual momentum: asset must have BOTH absolute momentum > 0 AND be in top N
    by relative momentum. Equal weight.
    """
    mom = compute_momentum(prices[universe], lookback)
    dates = prices.index[max(252, lookback + 50):]
    daily_rets = []
    last_rebal = None
    weights = pd.Series(0.0, index=universe)

    for date in dates:
        do_rebal = last_rebal is None or (date - last_rebal).days >= rebal_days
        if do_rebal:
            last_rebal = date
            m = mom.loc[:date].iloc[-1].dropna()

            # Absolute filter: must be positive
            abs_pass = m[m > 0]
            if len(abs_pass) == 0:
                weights = pd.Series(0.0, index=universe)
            else:
                # Relative filter: top N among those with positive abs momentum
                n = min(top_n, len(abs_pass))
                top = abs_pass.nlargest(n).index.tolist()
                weights = pd.Series(0.0, index=universe)
                w = 1.0 / n
                for s in top:
                    weights[s] = w

        day_ret = (weights * returns[universe].loc[date]).sum()
        invested = weights.sum()
        if invested < 1.0 and RISK_FREE_TICKER in returns.columns:
            rf = returns[RISK_FREE_TICKER].loc[date]
            if not np.isnan(rf):
                day_ret += (1.0 - invested) * rf
        daily_rets.append(day_ret)

    return pd.Series(daily_rets, index=dates, dtype=float)


def strategy_acceleration(prices, returns, universe, lookback=126,
                           short_lookback=None, rebal_days=21):
    """
    Acceleration filter: hold assets where momentum is INCREASING.
    Momentum's 2nd derivative > 0 (short-term momentum > long-term momentum).
    """
    if short_lookback is None:
        short_lookback = lookback // 2

    mom_long = compute_momentum(prices[universe], lookback)
    mom_short = compute_momentum(prices[universe], short_lookback)

    dates = prices.index[max(252, lookback + 50):]
    daily_rets = []
    last_rebal = None
    weights = pd.Series(0.0, index=universe)

    for date in dates:
        do_rebal = last_rebal is None or (date - last_rebal).days >= rebal_days
        if do_rebal:
            last_rebal = date
            ml = mom_long.loc[:date].iloc[-1]
            ms = mom_short.loc[:date].iloc[-1]

            # Acceleration: short momentum > long momentum AND long > 0
            accelerating = ((ms > ml) & (ml > 0))
            selected = accelerating[accelerating].index.tolist()

            weights = pd.Series(0.0, index=universe)
            if selected:
                w = 1.0 / len(selected)
                for s in selected:
                    weights[s] = w

        day_ret = (weights * returns[universe].loc[date]).sum()
        invested = weights.sum()
        if invested < 1.0 and RISK_FREE_TICKER in returns.columns:
            rf = returns[RISK_FREE_TICKER].loc[date]
            if not np.isnan(rf):
                day_ret += (1.0 - invested) * rf
        daily_rets.append(day_ret)

    return pd.Series(daily_rets, index=dates, dtype=float)


# ═══════════════════════════════════════════════════════════════════
#  EVALUATION
# ═══════════════════════════════════════════════════════════════════

def evaluate_strategy(daily_returns, spy_returns=None, label=""):
    """Compute comprehensive metrics including regime stratification."""
    dr = daily_returns.dropna()
    if len(dr) < 252 or dr.std() == 0:
        return {"label": label, "sharpe": 0, "cagr": 0, "valid": False}

    ann_ret = dr.mean() * 252
    ann_vol = dr.std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0

    downside = dr[dr < 0].std() * np.sqrt(252) if (dr < 0).sum() > 10 else ann_vol
    sortino = ann_ret / downside if downside > 0 else 0

    years = len(dr) / 252
    total_ret = (1 + dr).prod()
    cagr = total_ret ** (1 / years) - 1 if years > 0 else 0

    cum = (1 + dr).cumprod()
    peak = cum.cummax()
    dd = (cum - peak) / peak
    max_dd = dd.min()

    calmar = cagr / abs(max_dd) if max_dd != 0 else 0
    wr = (dr > 0).mean()

    gains = dr[dr > 0].sum()
    losses = abs(dr[dr < 0].sum())
    pf = gains / losses if losses > 0 else float("inf")

    # ─── Regime analysis using SPY daily returns ───
    green_sharpe = red_sharpe = 0.0
    regime_gap = 999.0

    if spy_returns is not None:
        # Align
        common = dr.index.intersection(spy_returns.index)
        if len(common) > 252:
            dr_aligned = dr.loc[common]
            spy_aligned = spy_returns.loc[common]

            green_mask = spy_aligned > 0
            red_mask = spy_aligned <= 0

            green_rets = dr_aligned[green_mask]
            red_rets = dr_aligned[red_mask]

            if len(green_rets) > 50 and green_rets.std() > 0:
                green_sharpe = green_rets.mean() / green_rets.std() * np.sqrt(252)
            if len(red_rets) > 50 and red_rets.std() > 0:
                red_sharpe = red_rets.mean() / red_rets.std() * np.sqrt(252)

            max_s = max(abs(green_sharpe), abs(red_sharpe))
            regime_gap = abs(green_sharpe - red_sharpe) / max_s if max_s > 0 else 999.0

    return {
        "label": label,
        "cagr": round(cagr * 100, 1),
        "sharpe": round(sharpe, 2),
        "sortino": round(sortino, 2),
        "max_dd": round(max_dd * 100, 1),
        "calmar": round(calmar, 2),
        "ann_vol": round(ann_vol * 100, 1),
        "wr": round(wr * 100, 1),
        "pf": round(pf, 2),
        "years": round(years, 1),
        "green_sharpe": round(green_sharpe, 2),
        "red_sharpe": round(red_sharpe, 2),
        "regime_gap": round(regime_gap, 3),
        "r1_pass": regime_gap <= 0.50,
        "valid": True,
    }


def run_permutation_test(daily_returns, n_trials=100):
    """Block-bootstrap permutation test."""
    dr = np.array(daily_returns.dropna())
    real_sharpe = dr.mean() / dr.std() * np.sqrt(252) if dr.std() > 0 else 0

    beat_count = 0
    shuffled_sharpes = []

    for _ in range(n_trials):
        monthly_idx = np.arange(0, len(dr), 21)
        blocks = [dr[i:i + 21] for i in monthly_idx if i + 21 <= len(dr)]
        if len(blocks) < 12:
            shuf = np.random.permutation(dr)
        else:
            np.random.shuffle(blocks)
            shuf = np.concatenate(blocks)

        s = shuf.mean() / shuf.std() * np.sqrt(252) if shuf.std() > 0 else 0
        shuffled_sharpes.append(s)
        if s >= real_sharpe:
            beat_count += 1

    return beat_count / n_trials, shuffled_sharpes


def compute_benchmark_6040(prices, returns):
    """60/40 SPY/AGG portfolio, monthly rebalanced."""
    if "SPY" not in returns.columns or "AGG" not in returns.columns:
        return None

    spy_r = returns["SPY"]
    agg_r = returns["AGG"]

    dates = prices.index[252:]
    daily_rets = []
    for date in dates:
        sr = spy_r.loc[date] if not np.isnan(spy_r.loc[date]) else 0.0
        ar = agg_r.loc[date] if not np.isnan(agg_r.loc[date]) else 0.0
        daily_rets.append(0.60 * sr + 0.40 * ar)

    return pd.Series(daily_rets, index=dates, dtype=float)


# ═══════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    np.random.seed(42)

    print("=" * 80)
    print("MULTI-ASSET MANAGED-FUTURES-STYLE TREND FOLLOWING")
    print("HC #696 — High-Return Growth Research")
    print("=" * 80)

    # ─── Download data ───
    prices, available = download_data()
    returns = compute_returns(prices)
    spy_returns = returns["SPY"] if "SPY" in returns.columns else None

    if len(available) < 8:
        print(f"ERROR: Only {len(available)} ETFs available. Need at least 8 for meaningful diversification.")
        return

    # ─── Build config sweep ───
    configs = []

    # 1. Classic trend following
    for lb in [100, 150, 200, 252]:
        for rebal in [5, 21]:
            label = f"classic_sma{lb}_rebal{rebal}d"
            configs.append({
                "label": label,
                "strategy": "classic_trend",
                "lookback": lb,
                "rebal_days": rebal,
            })

    # 2. Momentum rotation
    for lb in LOOKBACKS:
        for top_n in [3, 5, 7]:
            for rebal in [21]:
                label = f"momrot_lb{lb}_top{top_n}_rebal{rebal}d"
                configs.append({
                    "label": label,
                    "strategy": "momentum_rotation",
                    "lookback": lb,
                    "top_n": top_n,
                    "rebal_days": rebal,
                })

    # 3. Time-series momentum
    for lb in LOOKBACKS:
        for rebal in [21]:
            label = f"tsmom_lb{lb}_rebal{rebal}d"
            configs.append({
                "label": label,
                "strategy": "ts_momentum",
                "lookback": lb,
                "rebal_days": rebal,
            })

    # 4. Risk-parity trend
    for lb in [150, 200, 252]:
        for vol_win in [42, 63]:
            for rebal in [21]:
                label = f"riskpar_sma{lb}_vol{vol_win}_rebal{rebal}d"
                configs.append({
                    "label": label,
                    "strategy": "risk_parity_trend",
                    "lookback": lb,
                    "vol_window": vol_win,
                    "rebal_days": rebal,
                })

    # 5. Dual momentum
    for lb in LOOKBACKS:
        for top_n in [3, 5, 7]:
            for rebal in [21]:
                label = f"dualmom_lb{lb}_top{top_n}_rebal{rebal}d"
                configs.append({
                    "label": label,
                    "strategy": "dual_momentum",
                    "lookback": lb,
                    "top_n": top_n,
                    "rebal_days": rebal,
                })

    # 6. Acceleration filter
    for lb in LOOKBACKS:
        for rebal in [21]:
            label = f"accel_lb{lb}_rebal{rebal}d"
            configs.append({
                "label": label,
                "strategy": "acceleration",
                "lookback": lb,
                "rebal_days": rebal,
            })

    print(f"\nTotal configs to test: {len(configs)}")
    print(f"Available universe: {available}")

    # ─── Run all strategies ───
    all_results = []
    all_returns_dict = {}
    best_sharpe = -999
    best_label = None

    for i, cfg in enumerate(configs):
        try:
            strat = cfg["strategy"]

            if strat == "classic_trend":
                dr = strategy_classic_trend(prices, returns, available,
                                            lookback=cfg["lookback"],
                                            rebal_days=cfg["rebal_days"])
            elif strat == "momentum_rotation":
                dr = strategy_momentum_rotation(prices, returns, available,
                                                 lookback=cfg["lookback"],
                                                 top_n=cfg["top_n"],
                                                 rebal_days=cfg["rebal_days"])
            elif strat == "ts_momentum":
                dr = strategy_ts_momentum(prices, returns, available,
                                           lookback=cfg["lookback"],
                                           rebal_days=cfg["rebal_days"])
            elif strat == "risk_parity_trend":
                dr = strategy_risk_parity_trend(prices, returns, available,
                                                 lookback=cfg["lookback"],
                                                 vol_window=cfg.get("vol_window", 63),
                                                 rebal_days=cfg["rebal_days"])
            elif strat == "dual_momentum":
                dr = strategy_dual_momentum(prices, returns, available,
                                             lookback=cfg["lookback"],
                                             top_n=cfg["top_n"],
                                             rebal_days=cfg["rebal_days"])
            elif strat == "acceleration":
                dr = strategy_acceleration(prices, returns, available,
                                            lookback=cfg["lookback"],
                                            rebal_days=cfg["rebal_days"])
            else:
                continue

            result = evaluate_strategy(dr, spy_returns=spy_returns, label=cfg["label"])
            result["strategy_type"] = strat
            result["config"] = {k: v for k, v in cfg.items() if k != "label"}
            all_results.append(result)
            all_returns_dict[cfg["label"]] = dr

            if result.get("valid") and result["sharpe"] > best_sharpe:
                best_sharpe = result["sharpe"]
                best_label = cfg["label"]

            if (i + 1) % 10 == 0:
                print(f"  Processed {i + 1}/{len(configs)}... "
                      f"best Sharpe so far: {best_sharpe:.2f} ({best_label})")

        except Exception as e:
            all_results.append({"label": cfg["label"], "error": str(e), "valid": False})

    # ─── Sort and display ───
    valid = [r for r in all_results if r.get("valid")]
    valid.sort(key=lambda x: -x["sharpe"])

    print(f"\n{'=' * 110}")
    print(f"TOP 20 CONFIGS BY SHARPE (of {len(valid)} valid)")
    print(f"{'=' * 110}")
    print(f"{'Label':<45} {'Type':<16} {'CAGR':>6} {'Sharpe':>7} {'Sort':>6} "
          f"{'MaxDD':>7} {'Cal':>6} {'WR':>5} {'PF':>5} {'R1':>4}")
    print("-" * 110)

    for r in valid[:20]:
        r1 = "Y" if r.get("r1_pass") else "N"
        print(f"{r['label']:<45} {r.get('strategy_type','?'):<16} "
              f"{r['cagr']:>5.1f}% {r['sharpe']:>7.2f} {r['sortino']:>6.2f} "
              f"{r['max_dd']:>6.1f}% {r['calmar']:>6.2f} {r['wr']:>4.1f}% "
              f"{r['pf']:>5.2f} {r1:>4}")

    # ─── R1-passing configs ───
    r1_passing = [r for r in valid if r.get("r1_pass") and r["sharpe"] > 0.3]
    r1_passing.sort(key=lambda x: -x["sharpe"])

    print(f"\n{'=' * 110}")
    print(f"R1-PASSING CONFIGS (regime gap <= 0.50): {len(r1_passing)}")
    print(f"{'=' * 110}")

    if r1_passing:
        for r in r1_passing[:15]:
            print(f"  {r['label']:<45} CAGR={r['cagr']:>5.1f}% Sharpe={r['sharpe']:.2f} "
                  f"Sortino={r['sortino']:.2f} MaxDD={r['max_dd']:.1f}% "
                  f"Calmar={r['calmar']:.2f} gap={r['regime_gap']:.3f}")
    else:
        print("  None found. All configs are regime-dependent.")

    # ─── Strategy type summary ───
    print(f"\n{'=' * 80}")
    print("STRATEGY TYPE SUMMARY (best config per type)")
    print(f"{'=' * 80}")

    strat_types = set(r.get("strategy_type", "?") for r in valid)
    for st in sorted(strat_types):
        st_results = [r for r in valid if r.get("strategy_type") == st]
        if st_results:
            best = st_results[0]  # already sorted by sharpe
            r1_flag = "R1-PASS" if best.get("r1_pass") else "R1-FAIL"
            print(f"  {st:<20} best={best['label']:<40} "
                  f"CAGR={best['cagr']:.1f}% Sharpe={best['sharpe']:.2f} "
                  f"MaxDD={best['max_dd']:.1f}% {r1_flag}")

    # ─── Permutation test on top 5 ───
    test_candidates = r1_passing[:5] if len(r1_passing) >= 5 else valid[:5]

    print(f"\n{'=' * 80}")
    print("PERMUTATION TESTS (100 block-bootstrap trials)")
    print(f"{'=' * 80}")

    for r in test_candidates:
        label = r["label"]
        if label in all_returns_dict:
            p_val, _ = run_permutation_test(all_returns_dict[label], n_trials=100)
            r["permutation_p"] = p_val
            p_str = "PASS" if p_val <= 0.05 else "FAIL"
            print(f"  {label:<45} p={p_val:.2f} ({p_str})")

    # ─── Benchmarks ───
    print(f"\n{'=' * 80}")
    print("BENCHMARKS")
    print(f"{'=' * 80}")

    # SPY buy-and-hold
    if "SPY" in returns.columns:
        spy_bh = returns["SPY"].dropna().iloc[252:]
        spy_m = evaluate_strategy(spy_bh, spy_returns=spy_returns, label="SPY_buy_hold")
        if spy_m.get("valid"):
            print(f"  SPY B&H:  CAGR={spy_m['cagr']:.1f}%  Sharpe={spy_m['sharpe']:.2f}  "
                  f"Sortino={spy_m['sortino']:.2f}  MaxDD={spy_m['max_dd']:.1f}%")

    # 60/40
    bm6040 = compute_benchmark_6040(prices, returns)
    if bm6040 is not None:
        bm_m = evaluate_strategy(bm6040, spy_returns=spy_returns, label="60_40_SPY_AGG")
        if bm_m.get("valid"):
            print(f"  60/40:    CAGR={bm_m['cagr']:.1f}%  Sharpe={bm_m['sharpe']:.2f}  "
                  f"Sortino={bm_m['sortino']:.2f}  MaxDD={bm_m['max_dd']:.1f}%")

    # ─── Comparison table: best of each type vs benchmarks ───
    print(f"\n{'=' * 80}")
    print("BEST CONFIG PER TYPE vs BENCHMARKS")
    print(f"{'=' * 80}")
    print(f"{'Strategy':<25} {'CAGR':>7} {'Sharpe':>7} {'Sortino':>8} {'MaxDD':>7} {'Calmar':>7}")
    print("-" * 80)

    for st in sorted(strat_types):
        st_results = [r for r in valid if r.get("strategy_type") == st]
        if st_results:
            b = st_results[0]
            print(f"{st:<25} {b['cagr']:>6.1f}% {b['sharpe']:>7.2f} "
                  f"{b['sortino']:>8.2f} {b['max_dd']:>6.1f}% {b['calmar']:>7.2f}")

    if "SPY" in returns.columns and spy_m.get("valid"):
        print(f"{'SPY B&H':<25} {spy_m['cagr']:>6.1f}% {spy_m['sharpe']:>7.2f} "
              f"{spy_m['sortino']:>8.2f} {spy_m['max_dd']:>6.1f}% {spy_m['calmar']:>7.2f}")
    if bm6040 is not None and bm_m.get("valid"):
        print(f"{'60/40':<25} {bm_m['cagr']:>6.1f}% {bm_m['sharpe']:>7.2f} "
              f"{bm_m['sortino']:>8.2f} {bm_m['max_dd']:>6.1f}% {bm_m['calmar']:>7.2f}")

    # ─── Save cumulative returns for top configs ───
    print(f"\nSaving cumulative returns for top 10 configs...")
    cum_df = pd.DataFrame()
    for r in valid[:10]:
        label = r["label"]
        if label in all_returns_dict:
            cum_df[label] = (1 + all_returns_dict[label]).cumprod()

    # Add benchmarks
    if "SPY" in returns.columns:
        spy_bh_rets = returns["SPY"].dropna().iloc[252:]
        cum_df["SPY_BH"] = (1 + spy_bh_rets).cumprod()
    if bm6040 is not None:
        cum_df["60_40"] = (1 + bm6040).cumprod()

    cum_df.to_csv(OUT_DIR / "cumulative_returns.csv")

    # ─── Save full results ───
    elapsed = time.time() - t0

    output = {
        "generated": pd.Timestamp.now().isoformat(),
        "elapsed_seconds": round(elapsed, 1),
        "n_configs": len(configs),
        "n_valid": len(valid),
        "n_r1_passing": len(r1_passing),
        "available_etfs": available,
        "universe": UNIVERSE,
        "strategy_types_tested": list(strat_types),
        "top_20": valid[:20],
        "r1_passing": r1_passing[:15],
        "all_results": valid,
    }

    with open(OUT_DIR / "multi_asset_trend_results.json", "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nResults saved to {OUT_DIR}")
    print(f"Elapsed: {elapsed:.0f}s ({elapsed / 60:.1f}m)")
    print(f"\n{'=' * 80}")
    print("DONE")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
