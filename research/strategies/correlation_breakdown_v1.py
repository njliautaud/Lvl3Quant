#!/usr/bin/env python3
"""
Correlation Breakdown Opportunities v1 — Decorrelation as Contrarian Signal
============================================================================

OBSERVATION: Stocks normally move with their sector. When a stock's rolling
correlation to its sector ETF suddenly drops (decorrelation event), it often
means stock-specific news has driven a divergence. If the stock dropped during
decorrelation (negative idiosyncratic move), the historical pattern is mean
reversion back toward sector behavior.

HYPOTHESIS: When a stock's 20d rolling correlation to its sector drops below
the 5th/10th percentile of its own history, AND the stock has underperformed
its sector during that window, buying the stock produces asymmetric upside
as correlation reverts to normal.

KEY META-FINDING ALIGNMENT:
  - Contrarian: buying stocks that have diverged negatively from their sector
  - Vol compression compatible: decorrelation often happens during low-vol periods
  - "Buy fear, not greed": sector divergence = stock-specific fear

SIGNAL VARIANTS (36 total):
  - Correlation threshold: 5th, 10th, 15th percentile of own history
  - Underperformance requirement: stock < sector by 3%, 5%, 8% over lookback
  - Hold periods: 5d, 10d, 21d
  - With/without vol compression filter

VALIDATION (HC #428 + HC #432):
  - Regime analysis: SPY close vs 200 SMA -> GREEN/RED
  - Regime gap test: |Sharpe_green - Sharpe_red| / max(|Sharpe_green|, |Sharpe_red|) <= 0.50
  - Permutation test: 1000 shuffles, p < 0.05
  - Year consistency: >=60% of years profitable

Author: Claude Opus 4.6 / Teleclaude Research
Date: 2026-07-22
"""

import datetime as dt
import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# MLflow
try:
    import mlflow
    mlflow.set_tracking_uri("http://jupiter:5000")
    mlflow.set_experiment("correlation_breakdown_v1")
    HAS_MLFLOW = True
except Exception:
    HAS_MLFLOW = False
    print("MLflow not available, skipping tracking")

# Constants
COST_BPS_RT = 10  # 5 bps each way
NUM_PERMUTATIONS = 1000
REGIME_GAP_THRESHOLD = 0.50
PERM_P_THRESHOLD = 0.05
MIN_TRADES = 30
CORR_WINDOW = 20  # rolling correlation window
CORR_HISTORY = 252  # percentile ranking lookback
DATA_YEARS = 13

OUTPUT_DIR = Path("/home/nick/Lvl3Quant/output/correlation_breakdown_v1")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Sector ETF mapping
SECTOR_MAP = {
    'XLK': ['AAPL','MSFT','NVDA','AVGO','CSCO','ADBE','ORCL','ACN','CRM','AMD','INTC','TXN','QCOM',
            'ANET','INTU','NOW','ADI','KLAC','CDNS','SNPS','MCHP','FTNT','PANW','CRWD','WDAY',
            'TEAM','HUBS','VEEV','ANSS','NET','DDOG','MDB','SNOW','PLTR','ZS','OKTA','DOCU',
            'SQ','COIN','HOOD','NFLX','EA','TTWO','RBLX','SPOT','ZM'],
    'XLF': ['JPM','V','MA','BAC','WFC','GS','MS','SCHW','BLK','AXP','C','BK','SPGI',
            'ICE','CME','MCO','MSCI','AJG','AON','MMC','CB','TRV','ALL','MET','PRU','AFL',
            'PGR','CINF','GL','WRB','L','BEN','TROW','IVZ','AMP','NTRS','STT','FITB',
            'HBAN','CFG','KEY','RF','CMA','ZION','MTB','DFS','COF','SYF','ALLY'],
    'XLV': ['UNH','JNJ','LLY','ABBV','MRK','TMO','ABT','PFE','DHR','AMGN','BMY','SYK',
            'ISRG','MDT','REGN','VRTX','BSX','EW','BDX','CI','HUM','CNC','HCA','DXCM',
            'IDXX','IQV','ZTS','HOLX','WAT','A','MTD','BIIB','GILD','MRNA','ILMN',
            'ALGN','BAX','LH','DGX','DVA','UHS','THC','MOH'],
    'XLE': ['XOM','CVX','COP','EOG','SLB','MPC','PSX','OXY','DVN','HAL','BKR','FANG',
            'HES','MRO','APA','CTRA','EQT','AR','RRC','NOV','FTI','CHK'],
    'XLI': ['CAT','GE','HON','UNP','RTX','DE','BA','MMM','LMT','GD','NOC','LHX',
            'ETN','EMR','ITW','ROK','PH','IR','DOV','FDX','UPS','CSX','NSC',
            'WM','RSG','CARR','OTIS','TT','WAB','GWW','FAST','SWK','SNA','ROP',
            'AME','TDY','GNRC','XYL','NDSN','AOS','IEX'],
    'XLY': ['AMZN','TSLA','HD','MCD','NKE','LOW','TJX','SBUX','BKNG','CMG','ROST','ORLY',
            'AZO','AAP','GPC','LKQ','POOL','DHI','LEN','NVR','PHM','BBY','DG','DLTR',
            'F','GM','MAR','HLT','RCL','CCL','DAL','UAL','LUV','AAL','WYNN','MGM',
            'CZR','NCLH','UBER','LYFT','DASH','ABNB','ETSY','W','CHWY','DIS'],
    'XLP': ['PG','PEP','KO','COST','WMT','PM','MO','MDLZ','CL','KMB','GIS','SJM',
            'HRL','CPB','CAG','MKC','HSY','KDP','CHD','CLX','KR','SFM','TAP','STZ',
            'SAM','K','MNST','TSN','TGT'],
    'XLU': ['NEE','SO','DUK','D','SRE','AEP','XEL','WEC','ED','EXC','ES','AES',
            'PPL','CMS','CNP','NI','EVRG','ATO','PNW','NRG','AWK','ETR','FE','PEG'],
    'XLB': ['LIN','APD','SHW','ECL','DD','PPG','NUE','CF','VMC','MLM','FCX','DOW',
            'CTVA','IFF','CE','EMN','SEE','SON','PKG','IP','WRK','AVY','FMC','ALB'],
    'XLRE': ['PLD','AMT','CCI','EQIX','SPG','PSA','O','DLR','WELL','AVB','EQR',
             'VTR','ARE','MAA','UDR','CPT','REG','KIM','FRT','HST','PEAK','BXP'],
    'XLC': ['META','GOOGL','GOOG','CMCSA','T','VZ','TMUS','CHTR','DIS','NFLX',
            'EA','TTWO','ATVI','PARA','WBD','FOX','NWSA','LYV','SNAP','PINS'],
}

# Build reverse map: ticker -> sector ETF
TICKER_TO_SECTOR = {}
for etf, tickers in SECTOR_MAP.items():
    for t in tickers:
        TICKER_TO_SECTOR[t] = etf

ALL_SECTOR_ETFS = list(SECTOR_MAP.keys())
ALL_STOCK_TICKERS = sorted(set(t for ticks in SECTOR_MAP.values() for t in ticks))

# Signal variants
VARIANTS = []
for corr_pctile in [5, 10, 15]:
    for underperf in [0.03, 0.05, 0.08]:
        for hold in [5, 10, 21]:
            for vol_filter in [False, True]:
                vf_str = "_VC" if vol_filter else ""
                name = f"CORR{corr_pctile}_UP{int(underperf*100)}_H{hold}{vf_str}"
                VARIANTS.append((name, corr_pctile, underperf, hold, vol_filter))

print(f"Total variants: {len(VARIANTS)}")
print(f"Total stocks: {len(ALL_STOCK_TICKERS)}")
print(f"Sectors: {len(ALL_SECTOR_ETFS)}")


# ─────────────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def compute_rsi(prices, period=5):
    delta = prices.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def compute_vol_percentile(prices, vol_window=20, hist_window=252):
    log_ret = np.log(prices / prices.shift(1))
    realized_vol = log_ret.rolling(vol_window).std() * np.sqrt(252)
    def _pctile(s):
        if len(s) < hist_window:
            return np.nan
        current = s.iloc[-1]
        history = s.iloc[:-1]
        if np.isnan(current) or history.isna().all():
            return np.nan
        return (history < current).sum() / len(history) * 100
    return realized_vol.rolling(hist_window + 1).apply(_pctile, raw=False)


def annualized_sharpe(returns, trades_per_year=50):
    if len(returns) < 2 or np.std(returns) == 0:
        return 0.0
    return (np.mean(returns) / np.std(returns)) * np.sqrt(trades_per_year)


def sortino_ratio(returns, trades_per_year=50):
    if len(returns) < 2:
        return 0.0
    downside = returns[returns < 0]
    if len(downside) == 0 or np.std(downside) == 0:
        return float('inf') if np.mean(returns) > 0 else 0.0
    return (np.mean(returns) / np.std(downside)) * np.sqrt(trades_per_year)


def profit_factor(returns):
    gains = returns[returns > 0].sum()
    losses = abs(returns[returns < 0].sum())
    if losses == 0:
        return float("inf") if gains > 0 else 0.0
    return gains / losses


def permutation_test(returns, n_perms=1000):
    actual_sharpe = annualized_sharpe(returns)
    count_ge = 0
    abs_returns = np.abs(returns)
    for _ in range(n_perms):
        signs = np.random.choice([-1, 1], size=len(returns))
        shuffled = abs_returns * signs
        shuf_sharpe = annualized_sharpe(shuffled)
        if shuf_sharpe >= actual_sharpe:
            count_ge += 1
    return count_ge / n_perms


def regime_gap(sharpe_green, sharpe_red):
    denom = max(abs(sharpe_green), abs(sharpe_red))
    if denom == 0:
        return 0.0
    return abs(sharpe_green - sharpe_red) / denom


def compute_regime(spy_close):
    sma200 = spy_close.rolling(200).mean()
    regime = pd.Series("RED", index=spy_close.index)
    regime[spy_close > sma200] = "GREEN"
    regime[sma200.isna()] = np.nan
    return regime


# ─────────────────────────────────────────────────────────────────────────────
# DATA DOWNLOAD
# ─────────────────────────────────────────────────────────────────────────────

def download_data():
    import yfinance as yf

    end = dt.datetime.now()
    start = end - dt.timedelta(days=DATA_YEARS * 365)

    all_tickers = sorted(set(ALL_STOCK_TICKERS + ALL_SECTOR_ETFS + ["SPY"]))
    print(f"[INFO] Downloading {len(all_tickers)} tickers, {start.date()} to {end.date()}")
    t0 = time.time()

    stock_data = {}
    batch_size = 50
    for i in range(0, len(all_tickers), batch_size):
        batch = all_tickers[i:i+batch_size]
        try:
            data = yf.download(" ".join(batch), start=start, end=end,
                             group_by="ticker", progress=False, threads=True)
            if len(batch) == 1:
                ticker = batch[0]
                if not data.empty:
                    stock_data[ticker] = data
            else:
                for ticker in batch:
                    try:
                        if ticker in data.columns.get_level_values(0):
                            df = data[ticker].dropna(how="all")
                            if len(df) > 252:
                                stock_data[ticker] = df
                    except Exception:
                        pass
        except Exception as e:
            print(f"[WARN] Batch {i//batch_size+1} failed: {e}")

        if (i // batch_size + 1) % 5 == 0:
            print(f"[INFO] Downloaded {min(i+batch_size, len(all_tickers))}/{len(all_tickers)}")

    elapsed = time.time() - t0
    print(f"[INFO] Downloaded {len(stock_data)} tickers in {elapsed:.0f}s")
    return stock_data


# ─────────────────────────────────────────────────────────────────────────────
# SIGNAL COMPUTATION
# ─────────────────────────────────────────────────────────────────────────────

def compute_signals(stock_data):
    """Compute correlation breakdown signals for all stocks."""
    spy_close = stock_data.get("SPY")
    if spy_close is None:
        raise RuntimeError("No SPY data")
    spy_close = spy_close["Close"].squeeze()
    spy_regime = compute_regime(spy_close)

    # Build sector ETF close series
    sector_closes = {}
    for etf in ALL_SECTOR_ETFS:
        if etf in stock_data:
            sector_closes[etf] = stock_data[etf]["Close"].squeeze()

    all_signals = []
    processed = 0

    for ticker in ALL_STOCK_TICKERS:
        if ticker not in stock_data or ticker not in TICKER_TO_SECTOR:
            continue
        sector_etf = TICKER_TO_SECTOR[ticker]
        if sector_etf not in sector_closes:
            continue

        try:
            close = stock_data[ticker]["Close"].squeeze()
            if isinstance(close, pd.DataFrame):
                close = close.iloc[:, 0]
            if len(close) < 500:
                continue

            sector_close = sector_closes[sector_etf]

            # Align dates
            common_idx = close.index.intersection(sector_close.index)
            if len(common_idx) < 500:
                continue
            close = close.loc[common_idx]
            sector_close = sector_close.loc[common_idx]

            # Daily returns
            stock_ret = close.pct_change()
            sector_ret = sector_close.pct_change()

            # Rolling correlation
            rolling_corr = stock_ret.rolling(CORR_WINDOW).corr(sector_ret)

            # Correlation percentile (rank current vs own history)
            def _corr_pctile(s):
                if len(s) < CORR_HISTORY:
                    return np.nan
                current = s.iloc[-1]
                history = s.iloc[:-1]
                if np.isnan(current) or history.isna().all():
                    return np.nan
                return (history < current).sum() / len(history) * 100
            corr_pctile = rolling_corr.rolling(CORR_HISTORY + 1).apply(_corr_pctile, raw=False)

            # Stock vs sector performance over lookback
            stock_perf_20d = close / close.shift(CORR_WINDOW) - 1
            sector_perf_20d = sector_close / sector_close.shift(CORR_WINDOW) - 1
            underperformance = stock_perf_20d - sector_perf_20d

            # Vol percentile
            vol_pctile = compute_vol_percentile(close)

            # Forward returns
            fwd_5d = close.shift(-5) / close - 1
            fwd_10d = close.shift(-10) / close - 1
            fwd_21d = close.shift(-21) / close - 1

            sig = pd.DataFrame({
                "ticker": ticker,
                "sector": sector_etf,
                "close": close,
                "corr_pctile": corr_pctile,
                "underperf_20d": underperformance,
                "vol_pctile": vol_pctile,
                "fwd_5d": fwd_5d,
                "fwd_10d": fwd_10d,
                "fwd_21d": fwd_21d,
            }, index=close.index)

            sig["regime"] = sig.index.map(spy_regime)
            sig["year"] = sig.index.year
            sig = sig.dropna(subset=["corr_pctile", "underperf_20d"])
            all_signals.append(sig)
            processed += 1

        except Exception:
            pass

    print(f"[INFO] Computed signals for {processed} stocks")
    if not all_signals:
        raise RuntimeError("No signals generated")
    return pd.concat(all_signals, axis=0)


# ─────────────────────────────────────────────────────────────────────────────
# VARIANT EVALUATION
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_variant(name, corr_pctile_thresh, underperf_thresh, hold_days,
                     vol_filter, signals_df):
    """Evaluate a single variant."""
    # Decorrelation condition: correlation drops below Nth percentile of own history
    mask = signals_df["corr_pctile"] < corr_pctile_thresh

    # Stock underperformed sector by at least X%
    mask &= signals_df["underperf_20d"] < -underperf_thresh

    # Optional vol compression filter
    if vol_filter:
        mask &= signals_df["vol_pctile"] < 15

    trades = signals_df[mask].copy()

    fwd_col = f"fwd_{hold_days}d"
    if fwd_col not in trades.columns:
        return None
    trades = trades.dropna(subset=[fwd_col])

    if len(trades) < MIN_TRADES:
        return None

    # Apply costs
    cost_pct = COST_BPS_RT / 10000
    returns = trades[fwd_col].values - cost_pct
    n_trades = len(returns)

    # Core metrics
    sharpe = annualized_sharpe(returns)
    sort_r = sortino_ratio(returns)
    pf = profit_factor(returns)
    wr = (returns > 0).mean() * 100
    mean_ret = np.mean(returns) * 100
    median_ret = np.median(returns) * 100

    # Regime analysis
    green_mask = trades["regime"] == "GREEN"
    red_mask = trades["regime"] == "RED"
    green_returns = returns[green_mask.values]
    red_returns = returns[red_mask.values]

    sharpe_green = annualized_sharpe(green_returns) if len(green_returns) >= 10 else 0
    sharpe_red = annualized_sharpe(red_returns) if len(red_returns) >= 10 else 0
    gap = regime_gap(sharpe_green, sharpe_red)

    # Year consistency
    trades["_ret"] = returns
    yearly = trades.groupby("year")["_ret"].mean()
    pct_years_profitable = (yearly > 0).mean() * 100
    n_years = len(yearly)

    return {
        "name": name,
        "n_trades": n_trades,
        "sharpe": round(sharpe, 3),
        "sortino": round(sort_r, 3),
        "profit_factor": round(pf, 2),
        "win_rate": round(wr, 1),
        "mean_ret_pct": round(mean_ret, 3),
        "median_ret_pct": round(median_ret, 3),
        "sharpe_green": round(sharpe_green, 3),
        "sharpe_red": round(sharpe_red, 3),
        "regime_gap": round(gap, 3),
        "pct_years_profitable": round(pct_years_profitable, 1),
        "n_years": n_years,
        "green_trades": int(green_mask.sum()),
        "red_trades": int(red_mask.sum()),
    }


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    t_start = time.time()
    print("=" * 80)
    print(f"CORRELATION BREAKDOWN v1 — {len(ALL_STOCK_TICKERS)} stocks × {DATA_YEARS} years")
    print(f"Variants: {len(VARIANTS)}")
    print("=" * 80)

    # Start MLflow run
    run = None
    if HAS_MLFLOW:
        run = mlflow.start_run(run_name=f"correlation_breakdown_v1_{dt.datetime.now().strftime('%Y%m%d_%H%M')}")

    # Download data
    stock_data = download_data()
    print(f"\n[INFO] {len(stock_data)} tickers downloaded")

    # Compute signals
    signals_df = compute_signals(stock_data)
    print(f"[INFO] {len(signals_df)} total signal-days")

    # Evaluate all variants
    results = []
    for i, (name, corr_pctile, underperf, hold, vol_filt) in enumerate(VARIANTS):
        result = evaluate_variant(name, corr_pctile, underperf, hold, vol_filt, signals_df)
        if result:
            results.append(result)
            status = "PASS" if result["regime_gap"] <= REGIME_GAP_THRESHOLD and result["sharpe"] > 0 else "skip"
            print(f"  [{i+1}/{len(VARIANTS)}] {name}: Sharpe {result['sharpe']}, WR {result['win_rate']}%, "
                  f"RG {result['regime_gap']}, trades {result['n_trades']} [{status}]")
        else:
            print(f"  [{i+1}/{len(VARIANTS)}] {name}: insufficient trades")

    if not results:
        print("\n[ERROR] No variants had sufficient trades")
        if run:
            mlflow.log_param("status", "no_valid_variants")
            mlflow.end_run()
        return

    # Sort by Sharpe
    results.sort(key=lambda x: x["sharpe"], reverse=True)

    # Identify candidates that pass initial screen
    candidates = [r for r in results if r["sharpe"] > 0 and r["regime_gap"] <= REGIME_GAP_THRESHOLD]
    print(f"\n{'='*80}")
    print(f"INITIAL SCREEN: {len(candidates)} of {len(results)} variants pass (Sharpe>0 + regime gap<=0.50)")

    # Permutation test on top candidates (up to 20)
    top_candidates = sorted(candidates, key=lambda x: x["sharpe"], reverse=True)[:20]
    print(f"\nRunning {NUM_PERMUTATIONS}-shuffle permutation test on top {len(top_candidates)} candidates...")

    final_pass = []
    for r in top_candidates:
        name = r["name"]
        # Re-extract the variant params
        for vname, cp, up, hd, vf in VARIANTS:
            if vname == name:
                corr_pctile, underperf, hold, vol_filt = cp, up, hd, vf
                break

        mask = signals_df["corr_pctile"] < corr_pctile
        mask &= signals_df["underperf_20d"] < -underperf
        if vol_filt:
            mask &= signals_df["vol_pctile"] < 15

        trades = signals_df[mask].copy()
        fwd_col = f"fwd_{hold}d"
        trades = trades.dropna(subset=[fwd_col])
        returns = trades[fwd_col].values - COST_BPS_RT / 10000

        perm_p = permutation_test(returns, NUM_PERMUTATIONS)
        r["perm_p"] = round(perm_p, 3)

        gate1 = perm_p < PERM_P_THRESHOLD
        gate2 = r["regime_gap"] <= REGIME_GAP_THRESHOLD
        gate3 = r["pct_years_profitable"] >= 60

        r["gate_perm"] = gate1
        r["gate_regime"] = gate2
        r["gate_years"] = gate3
        r["gates_passed"] = sum([gate1, gate2, gate3])

        status = "PASS ALL" if all([gate1, gate2, gate3]) else f"{r['gates_passed']}/3"
        print(f"  {name}: perm p={perm_p:.3f} {'PASS' if gate1 else 'FAIL'}, "
              f"RG={r['regime_gap']:.3f} {'PASS' if gate2 else 'FAIL'}, "
              f"years={r['pct_years_profitable']:.0f}% {'PASS' if gate3 else 'FAIL'} "
              f"→ [{status}]")

        if all([gate1, gate2, gate3]):
            final_pass.append(r)

    # Summary
    print(f"\n{'='*80}")
    print(f"FINAL: {len(final_pass)} of {len(VARIANTS)} variants PASS ALL 3 GATES")
    print("="*80)

    if final_pass:
        print("\nPASSING VARIANTS:")
        for r in sorted(final_pass, key=lambda x: x["sharpe"], reverse=True):
            print(f"  {r['name']}: Sharpe {r['sharpe']}, Sortino {r['sortino']}, "
                  f"WR {r['win_rate']}%, PF {r['profit_factor']}, trades {r['n_trades']}, "
                  f"regime_gap {r['regime_gap']}, perm_p {r['perm_p']}, "
                  f"years_prof {r['pct_years_profitable']}%")

    # Save results
    all_results = {"variants": results, "final_pass": final_pass,
                   "meta": {"total_variants": len(VARIANTS), "total_stocks": len(ALL_STOCK_TICKERS),
                            "data_years": DATA_YEARS, "n_signal_days": len(signals_df),
                            "runtime_s": round(time.time() - t_start, 1)}}
    with open(OUTPUT_DIR / "results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    # Log to MLflow
    if HAS_MLFLOW and run:
        mlflow.log_param("n_stocks", len(ALL_STOCK_TICKERS))
        mlflow.log_param("n_variants", len(VARIANTS))
        mlflow.log_param("data_years", DATA_YEARS)
        mlflow.log_param("n_signal_days", len(signals_df))
        mlflow.log_metric("n_pass_all_gates", len(final_pass))
        mlflow.log_metric("n_pass_initial", len(candidates))
        if final_pass:
            best = max(final_pass, key=lambda x: x["sharpe"])
            mlflow.log_metric("best_sharpe", best["sharpe"])
            mlflow.log_metric("best_regime_gap", best["regime_gap"])
            mlflow.log_metric("best_win_rate", best["win_rate"])
            mlflow.log_metric("best_n_trades", best["n_trades"])
            mlflow.log_param("best_variant", best["name"])
        mlflow.log_artifact(str(OUTPUT_DIR / "results.json"))
        mlflow.end_run()

    elapsed = time.time() - t_start
    print(f"\nTotal runtime: {elapsed/60:.1f} minutes")


if __name__ == "__main__":
    main()
