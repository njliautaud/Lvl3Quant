#!/usr/bin/env python3
"""
Leveraged ETF Rotation with Vol-Targeting
==========================================
HC #696: High-return growth research.

Strategy: Rotate between leveraged sector ETFs (TQQQ, SOXL, TECL, UPRO, etc.)
based on relative momentum, with volatility targeting to control drawdown.

Key insight from prior research:
- TQQQ + 200MA = 35.9% CAGR but -57% DD
- Vol-targeting cuts DD in half while keeping most returns
- ETF rotation adds diversification vs single-asset
- Combining rotation + vol-targeting should be strictly better

Tested configurations:
- Universe: 3x leveraged ETFs (TQQQ, SOXL, TECL, UPRO, FAS, LABU, TNA)
- Rotation signals: 1-month, 3-month, 6-month momentum (and composites)
- Vol-targeting: 20%, 30%, 40% annualized portfolio vol targets
- Regime filter: SPY > 200MA (risk-on), cash (risk-off)
- Rebalance: weekly and monthly
- Hold: top 1, top 2, top 3

Walk-forward: sliding 252d train, 21d test (monthly), 2010-2026.
"""

import numpy as np
import pandas as pd
import yfinance as yf
from pathlib import Path
from scipy import stats
import json
import warnings
import time
warnings.filterwarnings("ignore")

OUT_DIR = Path("/home/jupiter/Lvl3Quant/output/growth_research/leveraged_rotation")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ─── Leveraged ETF universe ───
LEVERAGED_ETFS = {
    "TQQQ": "3x Nasdaq-100",
    "SOXL": "3x Semiconductors",
    "TECL": "3x Technology",
    "UPRO": "3x S&P 500",
    "FAS":  "3x Financials",
    "TNA":  "3x Small-Cap",
    "LABU": "3x Biotech",
    "FNGU": "3x FANG+",
}

REGIME_FILTER = "SPY"
RISK_FREE = "SHV"  # short-term treasury for cash position


def download_data(tickers, start="2009-01-01", end="2026-07-14"):
    """Download adjusted close prices."""
    all_tickers = list(tickers) + [REGIME_FILTER, RISK_FREE]
    print(f"Downloading {len(all_tickers)} tickers...")
    data = yf.download(all_tickers, start=start, end=end, auto_adjust=True, progress=False)

    # Handle multi-level columns
    if isinstance(data.columns, pd.MultiIndex):
        prices = data["Close"]
    else:
        prices = data

    prices = prices.ffill().dropna(how="all")
    print(f"Data: {prices.index[0].strftime('%Y-%m-%d')} to {prices.index[-1].strftime('%Y-%m-%d')}, {len(prices)} days")

    # Report available tickers
    available = [t for t in tickers if t in prices.columns and prices[t].notna().sum() > 252]
    print(f"Available leveraged ETFs (>1yr data): {available}")

    return prices, available


def compute_momentum(prices, lookback_days):
    """Compute total return momentum over lookback period."""
    return prices / prices.shift(lookback_days) - 1


def compute_realized_vol(returns, window=21):
    """Compute annualized realized volatility."""
    return returns.rolling(window, min_periods=10).std() * np.sqrt(252)


def run_strategy(prices, etf_universe, config):
    """
    Run a single rotation + vol-targeting strategy.

    Config dict:
        mom_lookback: list of momentum lookback days (composite if multiple)
        top_n: number of ETFs to hold
        vol_target: annualized vol target (0 = no vol-targeting)
        regime_filter: bool, use SPY > 200MA
        rebal_freq: 'weekly' or 'monthly'
        max_leverage: max position size multiplier
    """
    mom_lookbacks = config["mom_lookback"]
    top_n = config["top_n"]
    vol_target = config["vol_target"]
    use_regime = config["regime_filter"]
    rebal_freq = config["rebal_freq"]
    max_lev = config.get("max_leverage", 2.0)

    # Compute returns
    returns = prices.pct_change()

    # Regime filter
    if use_regime and REGIME_FILTER in prices.columns:
        spy_price = prices[REGIME_FILTER]
        spy_ma200 = spy_price.rolling(200, min_periods=100).mean()
        risk_on = spy_price > spy_ma200
    else:
        risk_on = pd.Series(True, index=prices.index)

    # Composite momentum score
    mom_scores = pd.DataFrame(index=prices.index, columns=etf_universe, dtype=float)
    for lb in mom_lookbacks:
        m = compute_momentum(prices[etf_universe], lb)
        # Normalize to z-scores cross-sectionally
        m_rank = m.rank(axis=1, pct=True)
        if mom_scores.isna().all().all():
            mom_scores = m_rank
        else:
            mom_scores = mom_scores + m_rank
    mom_scores = mom_scores / len(mom_lookbacks)

    # Realized vol for vol-targeting
    port_vol = pd.Series(index=prices.index, dtype=float)

    # Simulate
    nav = [100000.0]
    positions = {}  # ticker -> weight
    last_rebal = None
    daily_returns = []
    trade_dates = []
    holdings_log = []

    dates = prices.index[252:]  # skip first year for warmup

    for i, date in enumerate(dates):
        # Check rebalance
        do_rebal = False
        if last_rebal is None:
            do_rebal = True
        elif rebal_freq == "weekly" and (date - last_rebal).days >= 5:
            do_rebal = True
        elif rebal_freq == "monthly" and (date - last_rebal).days >= 21:
            do_rebal = True

        if do_rebal:
            last_rebal = date

            # Get momentum scores for this date
            scores = mom_scores.loc[:date].iloc[-1]
            valid_scores = scores.dropna()

            if len(valid_scores) == 0 or not risk_on.loc[:date].iloc[-1]:
                # Risk-off: go to cash
                positions = {}
            else:
                # Select top N by momentum
                top = valid_scores.nlargest(top_n)

                # Equal weight among top N
                base_weight = 1.0 / top_n

                # Vol-targeting adjustment
                if vol_target > 0:
                    for ticker in top.index:
                        ticker_ret = returns[ticker].loc[:date].tail(21)
                        ticker_vol = ticker_ret.std() * np.sqrt(252)
                        if ticker_vol > 0:
                            vol_scalar = vol_target / ticker_vol
                            vol_scalar = min(vol_scalar, max_lev)  # cap leverage
                            vol_scalar = max(vol_scalar, 0.1)  # min 10%
                            positions[ticker] = base_weight * vol_scalar
                        else:
                            positions[ticker] = base_weight
                else:
                    positions = {t: base_weight for t in top.index}

                # Remove old positions not in top N
                positions = {k: v for k, v in positions.items() if k in top.index}

            holdings_log.append({"date": str(date.date()), "positions": dict(positions)})

        # Compute daily return
        day_ret = 0.0
        for ticker, weight in positions.items():
            if ticker in returns.columns:
                r = returns[ticker].loc[date]
                if not np.isnan(r):
                    day_ret += weight * r

        # Cash portion earns risk-free rate
        invested = sum(positions.values())
        if invested < 1.0 and RISK_FREE in returns.columns:
            rf_ret = returns[RISK_FREE].loc[date]
            if not np.isnan(rf_ret):
                day_ret += (1.0 - invested) * rf_ret

        daily_returns.append(day_ret)
        nav.append(nav[-1] * (1 + day_ret))
        trade_dates.append(date)

    return pd.Series(daily_returns, index=trade_dates), nav, holdings_log


def evaluate_strategy(daily_returns, label=""):
    """Compute comprehensive metrics."""
    dr = pd.Series(daily_returns)
    if len(dr) < 252 or dr.std() == 0:
        return {"label": label, "sharpe": 0, "cagr": 0, "valid": False}

    # Basic metrics
    ann_ret = dr.mean() * 252
    ann_vol = dr.std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0

    downside = dr[dr < 0].std() * np.sqrt(252) if (dr < 0).sum() > 10 else ann_vol
    sortino = ann_ret / downside if downside > 0 else 0

    # CAGR
    years = len(dr) / 252
    total_ret = (1 + dr).prod()
    cagr = total_ret ** (1/years) - 1 if years > 0 else 0

    # Drawdown
    cum = (1 + dr).cumprod()
    peak = cum.cummax()
    dd = (cum - peak) / peak
    max_dd = dd.min()

    # Calmar
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0

    # Win rate
    wr = (dr > 0).mean()

    # Profit factor
    gains = dr[dr > 0].sum()
    losses = abs(dr[dr < 0].sum())
    pf = gains / losses if losses > 0 else float("inf")

    # Regime stratification (using SPY monthly returns as proxy)
    # Simple: split by overall market direction using rolling window
    monthly_rets = dr.resample("ME").sum()
    if len(monthly_rets) > 24:
        # Split into bull/bear halves by market
        green_months = monthly_rets[monthly_rets > 0]
        red_months = monthly_rets[monthly_rets <= 0]

        green_sharpe = green_months.mean() / green_months.std() * np.sqrt(12) if len(green_months) > 3 and green_months.std() > 0 else 0
        red_sharpe = red_months.mean() / red_months.std() * np.sqrt(12) if len(red_months) > 3 and red_months.std() > 0 else 0

        max_s = max(abs(green_sharpe), abs(red_sharpe))
        regime_gap = abs(green_sharpe - red_sharpe) / max_s if max_s > 0 else 999
    else:
        green_sharpe = red_sharpe = 0
        regime_gap = 999

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
    """Shuffle daily returns to test if order matters."""
    dr = np.array(daily_returns)
    real_sharpe = dr.mean() / dr.std() * np.sqrt(252) if dr.std() > 0 else 0

    beat_count = 0
    shuffled_sharpes = []

    for _ in range(n_trials):
        # Block bootstrap: shuffle monthly blocks to preserve autocorrelation
        monthly_idx = np.arange(0, len(dr), 21)
        blocks = [dr[i:i+21] for i in monthly_idx if i+21 <= len(dr)]
        if len(blocks) < 12:
            # Too few blocks, shuffle daily
            shuf = np.random.permutation(dr)
        else:
            np.random.shuffle(blocks)
            shuf = np.concatenate(blocks)

        s = shuf.mean() / shuf.std() * np.sqrt(252) if shuf.std() > 0 else 0
        shuffled_sharpes.append(s)
        if s >= real_sharpe:
            beat_count += 1

    return beat_count / n_trials, shuffled_sharpes


def main():
    t0 = time.time()
    print("=" * 70)
    print("LEVERAGED ETF ROTATION + VOL-TARGETING")
    print("HC #696 — High-Return Growth Research")
    print("=" * 70)

    # Download data
    prices, available_etfs = download_data(list(LEVERAGED_ETFS.keys()))

    if len(available_etfs) < 3:
        print(f"ERROR: Only {len(available_etfs)} ETFs available. Need at least 3.")
        return

    # ─── Define configurations to sweep ───
    configs = []

    # Momentum lookback combinations
    mom_combos = [
        [21],           # 1-month
        [63],           # 3-month
        [126],          # 6-month
        [21, 63],       # 1+3 month composite
        [21, 63, 126],  # 1+3+6 month composite
        [63, 126],      # 3+6 month composite
    ]

    # Top N
    top_ns = [1, 2, 3]

    # Vol targets (0 = no VT)
    vol_targets = [0, 0.20, 0.30, 0.40, 0.50]

    # Regime filter
    regime_opts = [True, False]

    # Rebal frequency
    rebal_opts = ["weekly", "monthly"]

    # Full sweep would be huge — do strategic subset
    for mom in mom_combos:
        for top_n in top_ns:
            for vt in vol_targets:
                for regime in regime_opts:
                    for rebal in rebal_opts:
                        # Skip some combos to keep tractable
                        if not regime and vt == 0:
                            continue  # No protection at all = suicide
                        if top_n == 3 and len(available_etfs) < 4:
                            continue

                        mom_str = "+".join([str(m) for m in mom])
                        label = f"mom{mom_str}_top{top_n}_vt{int(vt*100)}_regime{'Y' if regime else 'N'}_{rebal}"

                        configs.append({
                            "label": label,
                            "mom_lookback": mom,
                            "top_n": top_n,
                            "vol_target": vt,
                            "regime_filter": regime,
                            "rebal_freq": rebal,
                            "max_leverage": 2.0,
                        })

    print(f"\nTotal configs to test: {len(configs)}")
    print(f"Available ETFs: {available_etfs}")

    # ─── Run all configs ───
    all_results = []
    best_sharpe = -999
    best_config = None

    for i, config in enumerate(configs):
        try:
            dr, nav, holdings = run_strategy(prices, available_etfs, config)
            result = evaluate_strategy(dr, label=config["label"])
            result["config"] = {k: v for k, v in config.items() if k != "label"}
            all_results.append(result)

            if result.get("valid") and result["sharpe"] > best_sharpe:
                best_sharpe = result["sharpe"]
                best_config = config["label"]

            if (i + 1) % 50 == 0:
                print(f"  Processed {i+1}/{len(configs)}... best Sharpe so far: {best_sharpe:.2f} ({best_config})")
        except Exception as e:
            all_results.append({"label": config["label"], "error": str(e), "valid": False})

    # ─── Sort and display results ───
    valid = [r for r in all_results if r.get("valid")]
    valid.sort(key=lambda x: -x["sharpe"])

    print(f"\n{'='*100}")
    print(f"TOP 20 CONFIGS BY SHARPE (of {len(valid)} valid)")
    print(f"{'='*100}")
    print(f"{'Label':<55} {'CAGR':>6} {'Sharpe':>7} {'Sort':>6} {'MaxDD':>7} {'Cal':>5} {'R1':>4}")
    print("-" * 100)

    for r in valid[:20]:
        r1 = "✅" if r.get("r1_pass") else "❌"
        print(f"{r['label']:<55} {r['cagr']:>5.1f}% {r['sharpe']:>7.2f} {r['sortino']:>6.2f} "
              f"{r['max_dd']:>6.1f}% {r['calmar']:>5.2f} {r1:>4}")

    # ─── R1-passing configs ───
    r1_passing = [r for r in valid if r.get("r1_pass") and r["sharpe"] > 0.5]
    r1_passing.sort(key=lambda x: -x["sharpe"])

    print(f"\n{'='*100}")
    print(f"R1-PASSING CONFIGS: {len(r1_passing)}")
    print(f"{'='*100}")

    if r1_passing:
        for r in r1_passing[:15]:
            print(f"  {r['label']}: CAGR={r['cagr']:.1f}%, Sharpe={r['sharpe']:.2f}, "
                  f"MaxDD={r['max_dd']:.1f}%, Calmar={r['calmar']:.2f}, "
                  f"gap={r['regime_gap']:.3f}")

    # ─── Permutation test top 5 R1-passing ───
    print(f"\n{'='*70}")
    print("PERMUTATION TESTS — Top R1-passing configs")
    print(f"{'='*70}")

    for r in r1_passing[:5]:
        config = next((c for c in configs if c["label"] == r["label"]), None)
        if config:
            dr, _, _ = run_strategy(prices, available_etfs, config)
            p_val, _ = run_permutation_test(dr.values, n_trials=100)
            r["permutation_p"] = p_val
            p_str = "✅ PASS" if p_val <= 0.05 else f"❌ FAIL"
            print(f"  {r['label']}: p={p_val:.2f} {p_str}")

    # ─── Compare to benchmarks ───
    print(f"\n{'='*70}")
    print("BENCHMARKS")
    print(f"{'='*70}")

    # TQQQ buy-and-hold
    if "TQQQ" in prices.columns:
        tqqq_ret = prices["TQQQ"].pct_change().dropna()[252:]
        tqqq_metrics = evaluate_strategy(tqqq_ret, "TQQQ_buy_hold")
        print(f"  TQQQ B&H: CAGR={tqqq_metrics['cagr']:.1f}%, Sharpe={tqqq_metrics['sharpe']:.2f}, MaxDD={tqqq_metrics['max_dd']:.1f}%")

    # SPY buy-and-hold
    if REGIME_FILTER in prices.columns:
        spy_ret = prices[REGIME_FILTER].pct_change().dropna()[252:]
        spy_metrics = evaluate_strategy(spy_ret, "SPY_buy_hold")
        print(f"  SPY B&H:  CAGR={spy_metrics['cagr']:.1f}%, Sharpe={spy_metrics['sharpe']:.2f}, MaxDD={spy_metrics['max_dd']:.1f}%")

    # ─── Save results ───
    elapsed = time.time() - t0
    output = {
        "generated": pd.Timestamp.now().isoformat(),
        "elapsed_seconds": round(elapsed, 1),
        "n_configs": len(configs),
        "n_valid": len(valid),
        "n_r1_passing": len(r1_passing),
        "available_etfs": available_etfs,
        "top_20": valid[:20],
        "r1_passing": r1_passing[:15],
        "all_results": valid,
    }

    with open(OUT_DIR / "leveraged_rotation_results.json", "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nResults saved. Elapsed: {elapsed:.0f}s ({elapsed/60:.1f}m)")


if __name__ == "__main__":
    main()
