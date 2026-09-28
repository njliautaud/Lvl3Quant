"""
Megacap Tech Rotation — LGBM Ranker with Walk-Forward OOS Validation
=====================================================================
HC #668: Build LGBM-based cross-sectional ranker to replace simple momentum.

Features:
  - Multi-horizon momentum (5d, 20d, 60d)
  - RSI(14), realized vol (20d, 60d), vol ratio (20/60)
  - VIX interaction, VIX term structure
  - Theme features (AI/semis inflows from theme_features.parquet)
  - Relative strength vs SPY
  - Fundamental PIT features (earnings beat, rev growth, FCF yield, etc.)

Target: next 21-day forward return rank (cross-sectional percentile among universe)

Walk-forward: 252d train, 21d OOS, sliding window
Regime split: green/red days based on SPY close-to-close
"""
from __future__ import annotations
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import lightgbm as lgb
from pathlib import Path

ROOT = Path("/home/jupiter/Lvl3Quant")

# ── Universe ─────────────────────────────────────────────────────────
UNIVERSE = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA",
    "AVGO", "AMD", "CRM", "NFLX", "ADBE", "ORCL", "INTC", "QCOM"
]
BENCH = "SPY"
K = 6               # pick top-K stocks
HOLD_DAYS = 5        # weekly rebalance (5 trading days)
TRAIN_DAYS = 252     # 1-year training window
OOS_DAYS = 21        # 1-month OOS step
COST_BPS = 10        # 5bps commission + 5bps slippage per side = 10bps round-trip

# ── Load & merge data ───────────────────────────────────────────────
def load_data():
    print("Loading data...")
    px = pd.read_parquet(ROOT / "wheel_strategy_v1/data/cache/prices_v2.parquet")
    px["date"] = pd.to_datetime(px["date"])

    # Filter to universe + SPY
    tickers_needed = UNIVERSE + [BENCH]
    px = px[px["ticker"].isin(tickers_needed)].copy()

    # Theme features
    tf = pd.read_parquet(ROOT / "data/feature_store/v2/theme_features.parquet")
    tf["date"] = pd.to_datetime(tf["date"])
    tf = tf[tf["ticker"].isin(UNIVERSE)].copy()

    # Macro
    mc = pd.read_parquet(ROOT / "wheel_strategy_v1/data/cache/macro.parquet")
    mc["date"] = pd.to_datetime(mc["date"])

    # Fundamentals PIT
    fp = pd.read_parquet(ROOT / "data/feature_store/v2/fund_pit_features.parquet")
    fp["date"] = pd.to_datetime(fp["date"])
    fp = fp[fp["ticker"].isin(UNIVERSE)].copy()

    return px, tf, mc, fp


def build_features(px, tf, mc, fp):
    """Build feature panel: one row per (ticker, date)."""
    print("Building features...")

    # SPY prices for relative strength and regime
    spy = px[px["ticker"] == BENCH][["date", "close"]].rename(columns={"close": "spy_close"})
    spy = spy.sort_values("date").drop_duplicates("date")
    spy["spy_ret_5d"] = spy["spy_close"].pct_change(5)
    spy["spy_ret_20d"] = spy["spy_close"].pct_change(20)
    spy["spy_ret_60d"] = spy["spy_close"].pct_change(60)
    spy["spy_ma50"] = spy["spy_close"].rolling(50, min_periods=30).mean()
    spy["spy_above_ma50"] = (spy["spy_close"] > spy["spy_ma50"]).astype(float)
    # SPY daily return for regime classification
    spy["spy_daily_ret"] = spy["spy_close"].pct_change()

    # Build per-ticker features
    rows = []
    for ticker in UNIVERSE:
        s = px[px["ticker"] == ticker][["date", "close", "high", "low", "volume",
                                         "ret", "rv_20", "rv_60"]].copy()
        s = s.sort_values("date").drop_duplicates("date").reset_index(drop=True)

        # Momentum features
        s["ret_5d"] = s["close"].pct_change(5)
        s["ret_10d"] = s["close"].pct_change(10)
        s["ret_20d"] = s["close"].pct_change(20)
        s["ret_60d"] = s["close"].pct_change(60)

        # RSI(14)
        delta = s["close"].diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        s["rsi_14"] = 100 - (100 / (1 + rs))

        # Vol ratio
        s["vol_ratio_20_60"] = s["rv_20"] / s["rv_60"].replace(0, np.nan)

        # Volume momentum
        s["vol_sma_20"] = s["volume"].rolling(20).mean()
        s["vol_ratio_vs_avg"] = s["volume"] / s["vol_sma_20"].replace(0, np.nan)

        # 52-week high distance
        s["high_52w"] = s["high"].rolling(252, min_periods=60).max()
        s["dist_52w_high"] = s["close"] / s["high_52w"] - 1.0

        # Merge SPY for relative strength
        s = s.merge(spy[["date", "spy_ret_5d", "spy_ret_20d", "spy_ret_60d",
                          "spy_above_ma50", "spy_daily_ret"]], on="date", how="left")
        s["rel_str_5d"] = s["ret_5d"] - s["spy_ret_5d"]
        s["rel_str_20d"] = s["ret_20d"] - s["spy_ret_20d"]
        s["rel_str_60d"] = s["ret_60d"] - s["spy_ret_60d"]

        s["ticker"] = ticker
        rows.append(s)

    panel = pd.concat(rows, ignore_index=True)

    # Merge macro
    panel = panel.merge(mc[["date", "vix", "vix3m", "vix_ts"]], on="date", how="left")

    # VIX interaction features
    panel["vix_x_mom20"] = panel["vix"] * panel["ret_20d"]
    panel["vix_x_vol20"] = panel["vix"] * panel["rv_20"]

    # Merge theme features (AI/semis are the key ones for megacap tech)
    theme_cols = [c for c in tf.columns if c.startswith("theme_") and c != "theme_primary"]
    tf_merge = tf[["ticker", "date"] + theme_cols].copy()
    panel = panel.merge(tf_merge, on=["ticker", "date"], how="left")

    # Merge fundamental PIT features
    fund_cols = [c for c in fp.columns if c.startswith("fund_pit_")]
    fp_merge = fp[["ticker", "date"] + fund_cols].copy()
    panel = panel.merge(fp_merge, on=["ticker", "date"], how="left")

    # Forward-fill fundamental features (they update quarterly)
    panel = panel.sort_values(["ticker", "date"]).reset_index(drop=True)
    for c in fund_cols:
        panel[c] = panel.groupby("ticker")[c].ffill()

    # Cross-sectional rank features (within each date)
    for feat in ["ret_20d", "ret_60d", "rv_20", "rsi_14"]:
        panel[f"xs_rank_{feat}"] = panel.groupby("date")[feat].rank(pct=True)

    # Forward return (target)
    panel["fwd_ret_21d"] = panel.groupby("ticker")["close"].shift(-HOLD_DAYS * 4) / panel["close"] - 1.0
    # Actually use proper 21-day forward return
    panel["fwd_ret_21d"] = panel.groupby("ticker").apply(
        lambda g: g["close"].shift(-21) / g["close"] - 1.0
    ).reset_index(level=0, drop=True)

    # Cross-sectional rank of forward return (target for ranking)
    panel["fwd_rank"] = panel.groupby("date")["fwd_ret_21d"].rank(pct=True)

    print(f"Panel shape: {panel.shape}, date range: {panel.date.min()} to {panel.date.max()}")
    return panel


# ── Feature list ─────────────────────────────────────────────────────
def get_feature_cols(panel):
    """Return list of feature columns (excluding target, identifiers)."""
    exclude = {"ticker", "date", "close", "open", "high", "low", "volume",
               "ret", "fwd_ret_21d", "fwd_rank", "spy_close", "spy_ma50",
               "spy_daily_ret", "vol_sma_20", "high_52w",
               "spy_ret_5d", "spy_ret_20d", "spy_ret_60d"}
    return [c for c in panel.columns if c not in exclude and panel[c].dtype in [np.float64, np.float32, float, np.int64]]


# ── Walk-forward backtest ────────────────────────────────────────────
def walk_forward_backtest(panel):
    """Walk-forward LGBM ranker with 252d train, 21d OOS, sliding."""
    print("\nRunning walk-forward backtest...")

    feature_cols = get_feature_cols(panel)
    print(f"Using {len(feature_cols)} features: {feature_cols[:10]}...")

    dates = sorted(panel["date"].unique())
    # Need at least TRAIN_DAYS + OOS_DAYS of data
    min_start_idx = TRAIN_DAYS + 60  # buffer for feature computation

    oos_results = []
    rebal_dates = []
    n_folds = 0

    # Step through OOS windows
    oos_start_idx = min_start_idx
    while oos_start_idx + OOS_DAYS < len(dates):
        oos_end_idx = min(oos_start_idx + OOS_DAYS, len(dates))
        train_start_idx = max(0, oos_start_idx - TRAIN_DAYS)

        train_start = dates[train_start_idx]
        train_end = dates[oos_start_idx - 1]
        oos_start = dates[oos_start_idx]
        oos_end = dates[oos_end_idx - 1]

        # Train data
        train_mask = (panel["date"] >= train_start) & (panel["date"] <= train_end)
        train_data = panel[train_mask].dropna(subset=["fwd_rank"])

        # OOS data (don't need fwd_rank for prediction, but need it for eval)
        oos_mask = (panel["date"] >= oos_start) & (panel["date"] <= oos_end)
        oos_data = panel[oos_mask].copy()

        if len(train_data) < 100 or len(oos_data) < 10:
            oos_start_idx += OOS_DAYS
            continue

        X_train = train_data[feature_cols].values
        y_train = train_data["fwd_rank"].values

        X_oos = oos_data[feature_cols].values

        # Train LGBM
        dtrain = lgb.Dataset(X_train, label=y_train, feature_name=feature_cols, free_raw_data=False)

        params = {
            "objective": "regression",
            "metric": "rmse",
            "num_leaves": 31,
            "learning_rate": 0.05,
            "feature_fraction": 0.7,
            "bagging_fraction": 0.7,
            "bagging_freq": 5,
            "min_child_samples": 20,
            "lambda_l1": 0.1,
            "lambda_l2": 1.0,
            "verbose": -1,
            "seed": 42,
        }

        model = lgb.train(params, dtrain, num_boost_round=200)

        # Score OOS
        oos_data["lgbm_score"] = model.predict(X_oos)

        # Save feature importance for last fold
        if oos_start_idx + 2 * OOS_DAYS >= len(dates):
            importance = pd.DataFrame({
                "feature": feature_cols,
                "importance": model.feature_importance(importance_type="gain")
            }).sort_values("importance", ascending=False)

        oos_results.append(oos_data)
        n_folds += 1
        oos_start_idx += OOS_DAYS

    print(f"Completed {n_folds} walk-forward folds")

    oos_all = pd.concat(oos_results, ignore_index=True)
    return oos_all, importance


def simulate_portfolio(oos_all, strategy_name="LGBM", rank_col="lgbm_score"):
    """Simulate weekly rebalanced portfolio picking top-K by rank_col.

    Rebalances every HOLD_DAYS trading days. Equal-weight top-K.
    Returns daily portfolio returns series.
    """
    dates = sorted(oos_all["date"].unique())
    nav = 1.0
    daily_returns = []
    daily_dates = []
    holdings = []
    current_positions = {}  # ticker -> weight
    last_rebal_idx = -HOLD_DAYS  # force rebalance on first day
    trade_log = []

    for i, d in enumerate(dates):
        day_data = oos_all[oos_all["date"] == d].copy()
        if len(day_data) < 5:  # need enough tickers
            continue

        # Check if rebalance day
        if i - last_rebal_idx >= HOLD_DAYS:
            # Pick top K by model score
            top_k = day_data.nlargest(K, rank_col)["ticker"].tolist()

            # Calculate turnover cost
            old_set = set(current_positions.keys())
            new_set = set(top_k)
            turnover = len(old_set.symmetric_difference(new_set)) / max(len(old_set | new_set), 1)

            current_positions = {t: 1.0 / K for t in top_k}
            last_rebal_idx = i
            trade_log.append({"date": d, "picks": top_k, "turnover": turnover})

        # Daily return of portfolio
        if current_positions:
            port_ret = 0.0
            for t, w in current_positions.items():
                t_data = day_data[day_data["ticker"] == t]
                if not t_data.empty:
                    port_ret += w * float(t_data.iloc[0]["ret"])
            # Deduct turnover cost on rebalance days
            if i == last_rebal_idx and trade_log:
                cost = trade_log[-1]["turnover"] * (COST_BPS / 10000)
                port_ret -= cost
        else:
            port_ret = 0.0

        daily_returns.append(port_ret)
        daily_dates.append(d)
        holdings.append(list(current_positions.keys()))

    result = pd.DataFrame({
        "date": daily_dates,
        "ret": daily_returns,
        "holdings": holdings
    })
    result["cum_ret"] = (1 + result["ret"]).cumprod()
    result["strategy"] = strategy_name
    return result, trade_log


def simulate_momentum_baseline(panel):
    """Simple 60-day momentum baseline — pick top-K by ret_60d."""
    dates = sorted(panel["date"].unique())
    # Use same OOS period as LGBM
    min_date = panel[panel["ret_60d"].notna()]["date"].min()
    panel_clean = panel[(panel["date"] >= min_date) & panel["ticker"].isin(UNIVERSE)].copy()
    return simulate_portfolio(panel_clean, strategy_name="Momentum60d", rank_col="ret_60d")


def simulate_spy_baseline(panel):
    """SPY buy-and-hold baseline."""
    spy_data = panel[panel["ticker"] == BENCH][["date", "ret"]].copy() if BENCH in panel["ticker"].values else None
    # We'll handle this separately
    pass


def compute_metrics(daily_rets: pd.Series, ann_factor=252):
    """Compute Sharpe, Sortino, CAGR, MaxDD."""
    if len(daily_rets) < 10:
        return {}
    mean_ret = daily_rets.mean() * ann_factor
    std_ret = daily_rets.std() * np.sqrt(ann_factor)
    sharpe = mean_ret / std_ret if std_ret > 0 else 0

    downside = daily_rets[daily_rets < 0].std() * np.sqrt(ann_factor)
    sortino = mean_ret / downside if downside > 0 else 0

    cum = (1 + daily_rets).cumprod()
    total_return = cum.iloc[-1] / cum.iloc[0] - 1
    n_years = len(daily_rets) / ann_factor
    cagr = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0

    running_max = cum.cummax()
    drawdown = cum / running_max - 1
    max_dd = drawdown.min()

    calmar = cagr / abs(max_dd) if max_dd != 0 else 0

    return {
        "Sharpe": round(sharpe, 2),
        "Sortino": round(sortino, 2),
        "CAGR": f"{cagr*100:.1f}%",
        "MaxDD": f"{max_dd*100:.1f}%",
        "Calmar": round(calmar, 2),
        "WinRate": f"{(daily_rets > 0).mean()*100:.1f}%",
        "N_days": len(daily_rets),
    }


def regime_analysis(result_df, spy_daily):
    """Split results by green/red SPY days."""
    # Merge SPY daily return
    merged = result_df.merge(spy_daily[["date", "spy_daily_ret"]], on="date", how="left")

    green = merged[merged["spy_daily_ret"] > 0]["ret"]
    red = merged[merged["spy_daily_ret"] <= 0]["ret"]

    green_sharpe = (green.mean() * 252) / (green.std() * np.sqrt(252)) if green.std() > 0 else 0
    red_sharpe = (red.mean() * 252) / (red.std() * np.sqrt(252)) if red.std() > 0 else 0

    regime_gap = abs(green_sharpe - red_sharpe) / max(abs(green_sharpe), abs(red_sharpe), 0.01)

    return {
        "green_days": len(green),
        "red_days": len(red),
        "green_sharpe": round(green_sharpe, 2),
        "red_sharpe": round(red_sharpe, 2),
        "regime_gap": round(regime_gap, 2),
        "regime_gate_pass": regime_gap <= 0.50,
        "green_mean_bps": round(green.mean() * 10000, 1),
        "red_mean_bps": round(red.mean() * 10000, 1),
    }


def main():
    px, tf, mc, fp = load_data()
    panel = build_features(px, tf, mc, fp)

    # Get SPY daily returns for regime analysis
    spy_daily = panel[["date", "spy_daily_ret"]].drop_duplicates("date").dropna()

    # Run LGBM walk-forward
    oos_all, feat_importance = walk_forward_backtest(panel)

    # Align OOS dates
    oos_dates = sorted(oos_all["date"].unique())
    print(f"\nOOS period: {oos_dates[0]} to {oos_dates[-1]} ({len(oos_dates)} days)")

    # ── LGBM strategy ────────────────────────────────────────────────
    lgbm_result, lgbm_trades = simulate_portfolio(oos_all, "LGBM_Ranker", "lgbm_score")

    # ── Momentum baseline ────────────────────────────────────────────
    # Use same OOS date range as LGBM
    panel_oos = panel[(panel["date"] >= oos_dates[0]) & (panel["date"] <= oos_dates[-1]) &
                      panel["ticker"].isin(UNIVERSE)].copy()
    mom_result, _ = simulate_portfolio(panel_oos, "Momentum_60d", "ret_60d")

    # ── SPY buy-and-hold baseline ────────────────────────────────────
    spy_px = px[px["ticker"] == BENCH][["date", "close"]].sort_values("date").drop_duplicates("date")
    spy_px = spy_px[(spy_px["date"] >= oos_dates[0]) & (spy_px["date"] <= oos_dates[-1])].copy()
    spy_px["ret"] = spy_px["close"].pct_change().fillna(0)
    spy_px["cum_ret"] = (1 + spy_px["ret"]).cumprod()
    spy_px["strategy"] = "SPY_BH"

    # ── Metrics ──────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("MEGACAP TECH ROTATION — LGBM RANKER vs BASELINES")
    print("="*70)
    print(f"OOS Period: {oos_dates[0].date()} to {oos_dates[-1].date()} ({len(oos_dates)} trading days)")
    print(f"Universe: {len(UNIVERSE)} stocks, K={K}, rebal every {HOLD_DAYS}d")
    print(f"Walk-forward: {TRAIN_DAYS}d train, {OOS_DAYS}d OOS, sliding")
    print(f"Cost assumption: {COST_BPS}bps round-trip")
    print()

    for name, df in [("LGBM Ranker", lgbm_result), ("Momentum 60d", mom_result)]:
        metrics = compute_metrics(df["ret"])
        regime = regime_analysis(df, spy_daily)
        print(f"── {name} ──")
        print(f"  Sharpe={metrics['Sharpe']}  Sortino={metrics['Sortino']}  "
              f"CAGR={metrics['CAGR']}  MaxDD={metrics['MaxDD']}  "
              f"Calmar={metrics['Calmar']}  WR={metrics['WinRate']}")
        print(f"  Regime: Green Sharpe={regime['green_sharpe']}  Red Sharpe={regime['red_sharpe']}  "
              f"Gap={regime['regime_gap']}  {'PASS' if regime['regime_gate_pass'] else 'FAIL'}")
        print(f"  Green avg={regime['green_mean_bps']}bps  Red avg={regime['red_mean_bps']}bps")
        print()

    spy_metrics = compute_metrics(spy_px["ret"])
    print(f"── SPY Buy & Hold ──")
    print(f"  Sharpe={spy_metrics['Sharpe']}  Sortino={spy_metrics['Sortino']}  "
          f"CAGR={spy_metrics['CAGR']}  MaxDD={spy_metrics['MaxDD']}  "
          f"Calmar={spy_metrics['Calmar']}  WR={spy_metrics['WinRate']}")
    print()

    # ── Feature Importance ───────────────────────────────────────────
    print("── Top 20 Features (LGBM gain) ──")
    print(feat_importance.head(20).to_string(index=False))
    print()

    # ── LGBM vs Momentum improvement ────────────────────────────────
    lgbm_m = compute_metrics(lgbm_result["ret"])
    mom_m = compute_metrics(mom_result["ret"])
    print("── LGBM vs Momentum Improvement ──")
    print(f"  Sharpe: {lgbm_m['Sharpe']} vs {mom_m['Sharpe']} (delta={lgbm_m['Sharpe'] - mom_m['Sharpe']:+.2f})")
    print(f"  Sortino: {lgbm_m['Sortino']} vs {mom_m['Sortino']} (delta={lgbm_m['Sortino'] - mom_m['Sortino']:+.2f})")
    print()

    # ── Final cumulative returns ─────────────────────────────────────
    print("── Final Cumulative Returns ──")
    print(f"  LGBM Ranker:  {lgbm_result['cum_ret'].iloc[-1]:.3f}x")
    print(f"  Momentum 60d: {mom_result['cum_ret'].iloc[-1]:.3f}x")
    print(f"  SPY B&H:      {spy_px['cum_ret'].iloc[-1]:.3f}x")
    print()

    # ── Yearly breakdown ─────────────────────────────────────────────
    print("── Yearly OOS Returns ──")
    lgbm_result["year"] = lgbm_result["date"].dt.year
    for year in sorted(lgbm_result["year"].unique()):
        yr = lgbm_result[lgbm_result["year"] == year]
        yr_m = compute_metrics(yr["ret"])
        print(f"  {year}: Sharpe={yr_m['Sharpe']}  CAGR={yr_m['CAGR']}  MaxDD={yr_m['MaxDD']}  WR={yr_m['WinRate']}")

    # Save results
    out_dir = ROOT / "research" / "findings"
    out_dir.mkdir(parents=True, exist_ok=True)
    lgbm_result.to_parquet(out_dir / "megacap_lgbm_oos_results.parquet", index=False)
    feat_importance.to_csv(out_dir / "megacap_lgbm_feature_importance.csv", index=False)
    print(f"\nResults saved to {out_dir}")


if __name__ == "__main__":
    main()
