"""
Stock Rotation Comparison Backtest
===================================
Universe A: 11 Sector SPDR ETFs (K=3)
Universe B: 50 Mega-Cap Stocks (K=10)
Universe C: Hybrid — Top 3 sectors + Top 5 stocks within those sectors

Walk-forward sliding window, no lookahead bias.
"""

import numpy as np
import pandas as pd
import yfinance as yf
from pathlib import Path
from datetime import datetime
import warnings
import json

warnings.filterwarnings("ignore")

# ============================================================
# CONFIGURATION
# ============================================================

OUTPUT_DIR = Path("/home/jupiter/Lvl3Quant/output/stock_rotation_comparison")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

START_DATE = "2015-01-01"  # Extra buffer for feature computation
END_DATE = "2026-07-10"
LOOKBACK = 378  # Trading days for rolling features
HOLD_PERIOD = 21  # Rebalance every 21 trading days
MAX_CONSECUTIVE_HOLDS = 3  # Anti-concentration decay

# Universe A: Sector ETFs
SECTOR_ETFS = ["XLK", "XLV", "XLE", "XLF", "XLI", "XLP", "XLU", "XLB", "XLRE", "XLC", "XLY"]
K_SECTORS = 3
COST_BPS_ETF = 5  # 5 bps per trade

# Universe B: Mega-cap stocks
MEGA_CAPS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "JPM", "JNJ", "UNH",
    "V", "PG", "HD", "MA", "DIS", "NFLX", "ADBE", "CRM", "COST", "PEP",
    "KO", "MRK", "ABT", "TMO", "ACN", "MCD", "LIN", "TXN", "NEE", "UPS",
    "LOW", "INTU", "AMD", "ISRG", "MDLZ", "ADP", "GILD", "VRTX", "REGN", "LRCX",
    "SNPS", "KLAC", "CDNS", "AMAT", "ADI", "MRVL", "PANW", "CRWD", "NOW", "ABNB"
]
K_STOCKS = 10
COST_BPS_STOCK = 10  # 10 bps per trade

# Sector mapping for Universe C (hybrid)
STOCK_SECTOR_MAP = {
    "XLK": ["AAPL", "MSFT", "NVDA", "ADBE", "CRM", "INTU", "AMD", "TXN", "LRCX", "SNPS", "KLAC", "CDNS", "AMAT", "ADI", "MRVL", "NOW", "ACN"],
    "XLV": ["JNJ", "UNH", "MRK", "ABT", "TMO", "ISRG", "GILD", "VRTX", "REGN"],
    "XLE": [],
    "XLF": ["JPM", "V", "MA"],
    "XLI": ["UPS"],
    "XLP": ["PG", "COST", "PEP", "KO", "MDLZ"],
    "XLU": ["NEE"],
    "XLB": ["LIN"],
    "XLRE": [],
    "XLC": ["GOOGL", "META", "DIS", "NFLX"],
    "XLY": ["AMZN", "TSLA", "HD", "MCD", "LOW", "ABNB", "ADP"],
}
# Cybersecurity / misc that don't fit neatly
STOCK_SECTOR_MAP["XLK"] += ["PANW", "CRWD"]


# ============================================================
# DATA DOWNLOAD
# ============================================================

def download_data(tickers, start, end):
    """Download adjusted close prices from yfinance."""
    print(f"  Downloading {len(tickers)} tickers...")
    data = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    if isinstance(data.columns, pd.MultiIndex):
        prices = data["Close"]
    else:
        prices = data[["Close"]]
        prices.columns = tickers
    # Drop tickers with insufficient data (< 80% of max length)
    min_len = int(0.8 * len(prices))
    valid = prices.columns[prices.notna().sum() >= min_len]
    prices = prices[valid].ffill().dropna()
    print(f"  Got {len(valid)} valid tickers, {len(prices)} trading days")
    return prices


# ============================================================
# FEATURE COMPUTATION (NO LOOKAHEAD)
# ============================================================

def compute_features(prices, lookback=LOOKBACK):
    """
    Compute rotation features for each asset using only past data.
    Returns a dict of DataFrames: {feature_name: DataFrame(date x asset)}
    """
    returns = prices.pct_change()

    # 20d momentum (cumulative return over last 20 days)
    mom_20 = prices.pct_change(20)
    # 60d momentum
    mom_60 = prices.pct_change(60)
    # Momentum acceleration: 10d change in 20d momentum
    mom_accel = mom_20 - mom_20.shift(10)
    # Relative strength vs universe mean
    universe_mean_ret = returns.rolling(20).mean().mean(axis=1)
    asset_ret_20 = returns.rolling(20).mean()
    rel_strength = asset_ret_20.sub(universe_mean_ret, axis=0)
    # RS rank change (rank today vs rank 10 days ago)
    rs_rank = rel_strength.rank(axis=1, pct=True)
    rs_rank_change = rs_rank - rs_rank.shift(10)

    return {
        "mom_20": mom_20,
        "mom_60": mom_60,
        "mom_accel": mom_accel,
        "rel_strength": rel_strength,
        "rs_rank_change": rs_rank_change,
    }


def compute_scores(features, date_idx, assets, hold_counts):
    """
    Compute composite score for each asset at a given date.
    Applies anti-concentration decay.
    """
    weights = {
        "mom_20": 0.20,
        "mom_60": 0.15,
        "mom_accel": 0.30,
        "rel_strength": 0.20,
        "rs_rank_change": 0.15,
    }

    scores = pd.Series(0.0, index=assets)
    for feat_name, w in weights.items():
        feat_df = features[feat_name]
        if date_idx in feat_df.index:
            row = feat_df.loc[date_idx, assets]
            # Rank-normalize to [0, 1]
            ranked = row.rank(pct=True)
            scores += w * ranked.fillna(0.5)

    # Anti-concentration decay: penalize assets held too many consecutive periods
    for asset in assets:
        consecutive = hold_counts.get(asset, 0)
        if consecutive >= MAX_CONSECUTIVE_HOLDS:
            decay = 0.85 ** (consecutive - MAX_CONSECUTIVE_HOLDS + 1)
            scores[asset] *= decay

    return scores


# ============================================================
# BACKTEST ENGINE
# ============================================================

def run_backtest(prices, K, cost_bps, universe_name=""):
    """
    Walk-forward rotation backtest with sliding window.
    """
    returns = prices.pct_change().iloc[1:]
    features = compute_features(prices)
    assets = list(prices.columns)

    # Start after enough data for features (max lookback = 378 days, features need ~60d)
    start_idx = max(LOOKBACK, 80)
    dates = prices.index[start_idx:]

    # Rebalance dates
    rebal_dates = dates[::HOLD_PERIOD]

    portfolio_returns = []
    held_assets_history = []
    hold_counts = {a: 0 for a in assets}
    current_holdings = set()
    turnover_list = []
    hit_rates = []

    for i, rebal_date in enumerate(rebal_dates[:-1]):
        # Compute scores using only data up to rebal_date
        scores = compute_scores(features, rebal_date, assets, hold_counts)

        # Select top K
        top_k = scores.nlargest(K).index.tolist()

        # Compute turnover
        prev_holdings = current_holdings
        new_holdings = set(top_k)
        if prev_holdings:
            turnover = len(new_holdings - prev_holdings) / K
        else:
            turnover = 1.0
        turnover_list.append(turnover)

        # Update hold counts
        for a in assets:
            if a in new_holdings and a in prev_holdings:
                hold_counts[a] += 1
            elif a in new_holdings:
                hold_counts[a] = 1
            else:
                hold_counts[a] = 0

        current_holdings = new_holdings
        held_assets_history.append((rebal_date, top_k))

        # Compute returns for the hold period
        next_rebal = rebal_dates[i + 1] if i + 1 < len(rebal_dates) else dates[-1]
        period_mask = (returns.index > rebal_date) & (returns.index <= next_rebal)
        period_returns = returns.loc[period_mask, top_k]

        if len(period_returns) == 0:
            continue

        # Equal weight portfolio return
        daily_port_ret = period_returns.mean(axis=1)

        # Apply transaction costs on rebalance day
        cost = cost_bps / 10000.0 * turnover
        daily_port_ret.iloc[0] -= cost

        portfolio_returns.append(daily_port_ret)

        # Hit rate: % of held assets that beat universe median over hold period
        period_all = returns.loc[period_mask]
        if len(period_all) > 0:
            held_cum = period_all[top_k].sum()
            median_cum = period_all.sum().median()
            hit_rate = (held_cum > median_cum).mean()
            hit_rates.append(hit_rate)

    # Combine all portfolio returns
    port_ret = pd.concat(portfolio_returns)
    port_ret = port_ret[~port_ret.index.duplicated(keep='first')]
    port_ret = port_ret.sort_index()

    return port_ret, turnover_list, hit_rates


def run_hybrid_backtest(sector_prices, stock_prices):
    """
    Universe C: Top 3 sectors + Top 5 stocks within those sectors.
    """
    sector_returns = sector_prices.pct_change().iloc[1:]
    stock_returns = stock_prices.pct_change().iloc[1:]
    sector_features = compute_features(sector_prices)
    stock_features = compute_features(stock_prices)

    sector_assets = list(sector_prices.columns)
    stock_assets = list(stock_prices.columns)

    start_idx = max(LOOKBACK, 80)
    dates = sector_prices.index[start_idx:]
    rebal_dates = dates[::HOLD_PERIOD]

    portfolio_returns = []
    hold_counts_sector = {a: 0 for a in sector_assets}
    hold_counts_stock = {a: 0 for a in stock_assets}
    current_sectors = set()
    current_stocks = set()
    turnover_list = []
    hit_rates = []

    for i, rebal_date in enumerate(rebal_dates[:-1]):
        if rebal_date not in sector_prices.index or rebal_date not in stock_prices.index:
            continue

        # Step 1: Pick top 3 sectors
        sector_scores = compute_scores(sector_features, rebal_date, sector_assets, hold_counts_sector)
        top_sectors = sector_scores.nlargest(3).index.tolist()

        # Step 2: Within those sectors, pick top 5 stocks
        candidate_stocks = []
        for sec in top_sectors:
            sec_stocks = [s for s in STOCK_SECTOR_MAP.get(sec, []) if s in stock_assets]
            candidate_stocks.extend(sec_stocks)

        if len(candidate_stocks) < 5:
            # Fall back to all stocks if sectors don't have enough
            candidate_stocks = stock_assets

        stock_scores = compute_scores(stock_features, rebal_date, candidate_stocks, hold_counts_stock)
        top_stocks = stock_scores.nlargest(min(5, len(candidate_stocks))).index.tolist()

        # Turnover
        prev_sectors = current_sectors
        prev_stocks = current_stocks
        new_sectors = set(top_sectors)
        new_stocks = set(top_stocks)

        total_positions = 3 + len(top_stocks)
        if prev_sectors or prev_stocks:
            changes = len(new_sectors - prev_sectors) + len(new_stocks - prev_stocks)
            turnover = changes / total_positions
        else:
            turnover = 1.0
        turnover_list.append(turnover)

        # Update hold counts
        for a in sector_assets:
            if a in new_sectors and a in prev_sectors:
                hold_counts_sector[a] += 1
            elif a in new_sectors:
                hold_counts_sector[a] = 1
            else:
                hold_counts_sector[a] = 0
        for a in stock_assets:
            if a in new_stocks and a in prev_stocks:
                hold_counts_stock[a] += 1
            elif a in new_stocks:
                hold_counts_stock[a] = 1
            else:
                hold_counts_stock[a] = 0

        current_sectors = new_sectors
        current_stocks = new_stocks

        # Compute hold period returns
        next_rebal = rebal_dates[i + 1] if i + 1 < len(rebal_dates) else dates[-1]
        period_mask_sec = (sector_returns.index > rebal_date) & (sector_returns.index <= next_rebal)
        period_mask_stk = (stock_returns.index > rebal_date) & (stock_returns.index <= next_rebal)

        period_sec = sector_returns.loc[period_mask_sec, top_sectors] if any(period_mask_sec) else pd.DataFrame()
        period_stk = stock_returns.loc[period_mask_stk, top_stocks] if any(period_mask_stk) else pd.DataFrame()

        if len(period_sec) == 0 and len(period_stk) == 0:
            continue

        # Weight: 40% to sectors, 60% to stocks (stocks provide alpha, sectors provide stability)
        sector_weight = 0.40
        stock_weight = 0.60

        common_dates = period_sec.index.intersection(period_stk.index) if len(period_sec) > 0 and len(period_stk) > 0 else pd.Index([])

        if len(common_dates) > 0:
            sec_ret = period_sec.loc[common_dates].mean(axis=1) * sector_weight
            stk_ret = period_stk.loc[common_dates].mean(axis=1) * stock_weight
            daily_port_ret = sec_ret + stk_ret
        elif len(period_sec) > 0:
            daily_port_ret = period_sec.mean(axis=1)
        else:
            daily_port_ret = period_stk.mean(axis=1)

        # Transaction costs (blended)
        cost = (COST_BPS_ETF * sector_weight + COST_BPS_STOCK * stock_weight) / 10000.0 * turnover
        if len(daily_port_ret) > 0:
            daily_port_ret.iloc[0] -= cost

        portfolio_returns.append(daily_port_ret)

        # Hit rate
        if len(common_dates) > 0:
            all_held = top_sectors + top_stocks
            all_returns_period = pd.concat([
                sector_returns.loc[common_dates],
                stock_returns.loc[common_dates]
            ], axis=1)
            valid_held = [h for h in all_held if h in all_returns_period.columns]
            if valid_held:
                held_cum = all_returns_period[valid_held].sum()
                median_cum = all_returns_period.sum().median()
                hit_rate = (held_cum > median_cum).mean()
                hit_rates.append(hit_rate)

    port_ret = pd.concat(portfolio_returns) if portfolio_returns else pd.Series(dtype=float)
    port_ret = port_ret[~port_ret.index.duplicated(keep='first')]
    port_ret = port_ret.sort_index()

    return port_ret, turnover_list, hit_rates


# ============================================================
# METRICS
# ============================================================

def compute_metrics(port_ret, spy_ret, name=""):
    """Compute performance metrics."""
    # Align with SPY
    common = port_ret.index.intersection(spy_ret.index)
    port_aligned = port_ret.loc[common]
    spy_aligned = spy_ret.loc[common]

    ann_factor = 252
    mean_ret = port_aligned.mean() * ann_factor
    std_ret = port_aligned.std() * np.sqrt(ann_factor)
    sharpe = mean_ret / std_ret if std_ret > 0 else 0

    downside = port_aligned[port_aligned < 0].std() * np.sqrt(ann_factor)
    sortino = mean_ret / downside if downside > 0 else 0

    # CAGR
    total_days = (common[-1] - common[0]).days
    total_return = (1 + port_aligned).prod()
    cagr = total_return ** (365.25 / total_days) - 1 if total_days > 0 else 0

    # Max Drawdown
    cum_ret = (1 + port_aligned).cumprod()
    rolling_max = cum_ret.cummax()
    drawdown = (cum_ret - rolling_max) / rolling_max
    max_dd = drawdown.min()

    # Calmar
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0

    # Correlation to SPY
    corr_spy = port_aligned.corr(spy_aligned)

    return {
        "Universe": name,
        "Sharpe": round(sharpe, 3),
        "Sortino": round(sortino, 3),
        "CAGR": f"{cagr*100:.2f}%",
        "MaxDD": f"{max_dd*100:.2f}%",
        "Calmar": round(calmar, 3),
        "Corr_SPY": round(corr_spy, 3),
    }


# ============================================================
# PERMUTATION TEST
# ============================================================

def permutation_test(prices, K, cost_bps, actual_sharpe, n_iter=1000):
    """
    Shuffle scores randomly and compare Sharpe distribution to actual.
    """
    returns = prices.pct_change().iloc[1:]
    assets = list(prices.columns)

    start_idx = max(LOOKBACK, 80)
    dates = prices.index[start_idx:]
    rebal_dates = dates[::HOLD_PERIOD]

    perm_sharpes = []

    for iteration in range(n_iter):
        portfolio_returns = []
        for i, rebal_date in enumerate(rebal_dates[:-1]):
            # Random selection instead of score-based
            chosen = np.random.choice(assets, size=min(K, len(assets)), replace=False)

            next_rebal = rebal_dates[i + 1] if i + 1 < len(rebal_dates) else dates[-1]
            period_mask = (returns.index > rebal_date) & (returns.index <= next_rebal)
            period_returns = returns.loc[period_mask, chosen]

            if len(period_returns) == 0:
                continue

            daily_port_ret = period_returns.mean(axis=1)
            # Apply average cost
            daily_port_ret.iloc[0] -= cost_bps / 10000.0 * 0.5  # Assume ~50% turnover on average
            portfolio_returns.append(daily_port_ret)

        if portfolio_returns:
            port_ret = pd.concat(portfolio_returns)
            port_ret = port_ret[~port_ret.index.duplicated(keep='first')]
            ann_ret = port_ret.mean() * 252
            ann_std = port_ret.std() * np.sqrt(252)
            perm_sharpe = ann_ret / ann_std if ann_std > 0 else 0
            perm_sharpes.append(perm_sharpe)

    perm_sharpes = np.array(perm_sharpes)
    p_value = (perm_sharpes >= actual_sharpe).mean()

    return {
        "actual_sharpe": actual_sharpe,
        "perm_mean": round(np.mean(perm_sharpes), 3),
        "perm_std": round(np.std(perm_sharpes), 3),
        "perm_p95": round(np.percentile(perm_sharpes, 95), 3),
        "p_value": round(p_value, 4),
        "significant": p_value < 0.05,
    }


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 70)
    print("STOCK ROTATION COMPARISON BACKTEST")
    print("=" * 70)

    # Download data
    all_tickers = list(set(SECTOR_ETFS + MEGA_CAPS + ["SPY"]))
    print(f"\nDownloading {len(all_tickers)} tickers from {START_DATE} to {END_DATE}...")
    all_prices = download_data(all_tickers, START_DATE, END_DATE)

    # Separate universes
    available_etfs = [t for t in SECTOR_ETFS if t in all_prices.columns]
    available_stocks = [t for t in MEGA_CAPS if t in all_prices.columns]
    print(f"\n  Available ETFs: {len(available_etfs)}/{len(SECTOR_ETFS)}")
    print(f"  Available Stocks: {len(available_stocks)}/{len(MEGA_CAPS)}")

    sector_prices = all_prices[available_etfs]
    stock_prices = all_prices[available_stocks]
    spy_prices = all_prices["SPY"] if "SPY" in all_prices.columns else None
    spy_ret = spy_prices.pct_change().dropna() if spy_prices is not None else pd.Series(dtype=float)

    # ---- Universe A: Sector ETFs ----
    print("\n" + "-" * 50)
    print("Running Universe A: 11 Sector ETFs (K=3)...")
    port_ret_a, turnover_a, hits_a = run_backtest(sector_prices, K_SECTORS, COST_BPS_ETF, "Sectors")
    metrics_a = compute_metrics(port_ret_a, spy_ret, "A: Sector ETFs (K=3)")
    metrics_a["Avg_Turnover"] = f"{np.mean(turnover_a)*100:.1f}%"
    metrics_a["Hit_Rate"] = f"{np.mean(hits_a)*100:.1f}%"
    print(f"  Sharpe: {metrics_a['Sharpe']}, CAGR: {metrics_a['CAGR']}")

    # ---- Universe B: Mega-Cap Stocks ----
    print("\n" + "-" * 50)
    print("Running Universe B: 50 Mega-Cap Stocks (K=10)...")
    port_ret_b, turnover_b, hits_b = run_backtest(stock_prices, K_STOCKS, COST_BPS_STOCK, "Stocks")
    metrics_b = compute_metrics(port_ret_b, spy_ret, "B: Mega-Cap Stocks (K=10)")
    metrics_b["Avg_Turnover"] = f"{np.mean(turnover_b)*100:.1f}%"
    metrics_b["Hit_Rate"] = f"{np.mean(hits_b)*100:.1f}%"
    print(f"  Sharpe: {metrics_b['Sharpe']}, CAGR: {metrics_b['CAGR']}")

    # ---- Universe C: Hybrid ----
    print("\n" + "-" * 50)
    print("Running Universe C: Hybrid (Top 3 sectors + Top 5 stocks within)...")
    port_ret_c, turnover_c, hits_c = run_hybrid_backtest(sector_prices, stock_prices)
    metrics_c = compute_metrics(port_ret_c, spy_ret, "C: Hybrid (3 sectors + 5 stocks)")
    metrics_c["Avg_Turnover"] = f"{np.mean(turnover_c)*100:.1f}%"
    metrics_c["Hit_Rate"] = f"{np.mean(hits_c)*100:.1f}%"
    print(f"  Sharpe: {metrics_c['Sharpe']}, CAGR: {metrics_c['CAGR']}")

    # ---- SPY Buy-and-Hold Benchmark ----
    spy_metrics = compute_metrics(spy_ret, spy_ret, "Benchmark: SPY B&H")
    spy_metrics["Avg_Turnover"] = "0.0%"
    spy_metrics["Hit_Rate"] = "N/A"

    # ---- COMPARISON TABLE ----
    print("\n" + "=" * 70)
    print("COMPARISON TABLE")
    print("=" * 70)

    results_df = pd.DataFrame([metrics_a, metrics_b, metrics_c, spy_metrics])
    cols = ["Universe", "Sharpe", "Sortino", "CAGR", "MaxDD", "Calmar", "Corr_SPY", "Avg_Turnover", "Hit_Rate"]
    results_df = results_df[cols]
    print(results_df.to_string(index=False))

    # ---- PERMUTATION TEST on best-performing universe ----
    all_results = [
        ("A", metrics_a, sector_prices, K_SECTORS, COST_BPS_ETF, port_ret_a),
        ("B", metrics_b, stock_prices, K_STOCKS, COST_BPS_STOCK, port_ret_b),
    ]

    # Find best Sharpe
    best = max(all_results, key=lambda x: x[1]["Sharpe"])
    best_name, best_metrics, best_prices, best_k, best_cost, best_port_ret = best

    print(f"\n{'=' * 70}")
    print(f"PERMUTATION TEST: Universe {best_name} (best Sharpe = {best_metrics['Sharpe']})")
    print(f"Running 1000 random-selection iterations...")
    print(f"{'=' * 70}")

    perm_results = permutation_test(best_prices, best_k, best_cost, best_metrics["Sharpe"], n_iter=1000)
    print(f"\n  Actual Sharpe:      {perm_results['actual_sharpe']:.3f}")
    print(f"  Random Mean Sharpe: {perm_results['perm_mean']:.3f}")
    print(f"  Random Std:         {perm_results['perm_std']:.3f}")
    print(f"  Random 95th pctile: {perm_results['perm_p95']:.3f}")
    print(f"  P-value:            {perm_results['p_value']:.4f}")
    print(f"  Significant (p<0.05): {'YES' if perm_results['significant'] else 'NO'}")

    # ---- SAVE RESULTS ----
    output = {
        "run_date": datetime.now().isoformat(),
        "config": {
            "start_date": START_DATE,
            "end_date": END_DATE,
            "lookback": LOOKBACK,
            "hold_period": HOLD_PERIOD,
            "max_consecutive_holds": MAX_CONSECUTIVE_HOLDS,
        },
        "metrics": {
            "universe_a": metrics_a,
            "universe_b": metrics_b,
            "universe_c": metrics_c,
            "spy_benchmark": spy_metrics,
        },
        "permutation_test": perm_results,
        "available_tickers": {
            "etfs": available_etfs,
            "stocks": available_stocks,
        }
    }

    with open(OUTPUT_DIR / "comparison_results.json", "w") as f:
        json.dump(output, f, indent=2, default=str)

    # Save equity curves
    equity_curves = pd.DataFrame({
        "Sectors": (1 + port_ret_a).cumprod(),
        "MegaCaps": (1 + port_ret_b).cumprod(),
        "Hybrid": (1 + port_ret_c).cumprod(),
        "SPY": (1 + spy_ret).cumprod(),
    })
    equity_curves.to_parquet(OUTPUT_DIR / "equity_curves.parquet")

    print(f"\nResults saved to {OUTPUT_DIR}")
    print("Done.")


if __name__ == "__main__":
    main()
