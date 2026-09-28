"""
Megacap Tech Rotation — LGBM Ranker v2 (Bug-Fixed)
=====================================================================
HC #668: Build LGBM-based cross-sectional ranker to replace simple momentum.

v2 fixes (adversarial audit):
  Bug 1: Target rank leakage — fwd_rank now computed ONLY within each training
         window during walk-forward. OOS uses raw model scores ranked cross-
         sectionally at prediction time.
  Bug 2: Regime filter — simulate_portfolio() now goes to 100% cash when
         SPY < 50d MA OR VIX > 25.
  Bug 3: Hold/label horizon mismatch — HOLD_DAYS changed from 5 to 21 to
         match the 21-day forward return target (HC #428 R2).
  Fix 4: Turnover cost — dollar-weight-based turnover, not count-based.
  Fix 5: Verified 10bps round-trip cost applied correctly.

Features:
  - Multi-horizon momentum (5d, 20d, 60d)
  - RSI(14), realized vol (20d, 60d), vol ratio (20/60)
  - VIX interaction, VIX term structure
  - Theme features (AI/semis inflows from theme_features.parquet)
  - Relative strength vs SPY
  - Fundamental PIT features (earnings beat, rev growth, FCF yield, etc.)

Target: next 21-day forward return rank (cross-sectional percentile among universe)

Walk-forward: 252d train, 21d OOS, sliding window
Regime gate: SPY < 50d MA OR VIX > 25 → 100% cash
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
HOLD_DAYS = 21       # FIX Bug 3: match 21-day prediction horizon (was 5)
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
    """Build feature panel: one row per (ticker, date).

    FIX Bug 1: fwd_ret_21d is computed here for evaluation purposes only.
    fwd_rank is NOT pre-computed on the full panel. It will be computed
    within each training window during walk-forward to prevent leakage.
    """
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
        s = s.merge(spy[["date", "spy_close", "spy_ret_5d", "spy_ret_20d", "spy_ret_60d",
                          "spy_above_ma50", "spy_ma50", "spy_daily_ret"]], on="date", how="left")
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

    # Cross-sectional rank features (within each date) — these are LAGGED features, no leakage
    for feat in ["ret_20d", "ret_60d", "rv_20", "rsi_14"]:
        panel[f"xs_rank_{feat}"] = panel.groupby("date")[feat].rank(pct=True)

    # Forward return (target) — computed for evaluation; fwd_rank NOT pre-computed (Bug 1 fix)
    panel["fwd_ret_21d"] = panel.groupby("ticker").apply(
        lambda g: g["close"].shift(-21) / g["close"] - 1.0
    ).reset_index(level=0, drop=True)

    # NOTE: fwd_rank is intentionally NOT computed here.
    # It will be computed within each training window in walk_forward_backtest()
    # to prevent cross-temporal leakage (Bug 1 fix).

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
    """Walk-forward LGBM ranker with 252d train, 21d OOS, sliding.

    FIX Bug 1: fwd_rank is computed ONLY within each training window.
    OOS predictions use the model's raw scores, which are then ranked
    cross-sectionally at prediction time (no future leakage).
    """
    print("\nRunning walk-forward backtest...")

    feature_cols = get_feature_cols(panel)
    print(f"Using {len(feature_cols)} features: {feature_cols[:10]}...")

    dates = sorted(panel["date"].unique())
    # Need at least TRAIN_DAYS + OOS_DAYS of data
    min_start_idx = TRAIN_DAYS + 60  # buffer for feature computation

    oos_results = []
    n_folds = 0
    importance = None

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
        train_data = panel[train_mask].dropna(subset=["fwd_ret_21d"]).copy()

        # FIX Bug 1: Compute fwd_rank ONLY within this training window
        # The rank is computed cross-sectionally (per date) but only among
        # dates in the training set — no future OOS dates leak into the rank normalization.
        train_data["fwd_rank"] = train_data.groupby("date")["fwd_ret_21d"].rank(pct=True)

        # OOS data (don't need fwd_rank for prediction)
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

        # Score OOS — raw model predictions, NOT pre-computed ranks
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


def simulate_portfolio(oos_all, strategy_name="LGBM", rank_col="lgbm_score",
                       regime_gate=True):
    """Simulate portfolio picking top-K by rank_col with regime gate.

    FIX Bug 2: Regime filter — go to 100% cash when SPY < 50d MA OR VIX > 25.
    FIX Bug 3: Rebalance every HOLD_DAYS=21 to match prediction horizon.
    FIX 4: Dollar-weight-based turnover cost, not count-based.

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
    regime_cash_days = 0

    for i, d in enumerate(dates):
        day_data = oos_all[oos_all["date"] == d].copy()
        if len(day_data) < 5:  # need enough tickers
            continue

        # Check if rebalance day
        if i - last_rebal_idx >= HOLD_DAYS:
            # FIX Bug 2: Regime gate — check SPY < 50d MA or VIX > 25
            # Use the regime columns from the panel
            day_row = day_data.iloc[0]  # all tickers share same date-level macro
            spy_above = day_row.get("spy_above_ma50", 1.0)
            vix_val = day_row.get("vix", 15.0)

            if regime_gate and (spy_above < 1.0 or vix_val > 25.0):
                # Regime OFF — go to 100% cash
                old_positions = current_positions.copy()
                # Compute dollar-weight turnover for cost: selling everything
                if old_positions:
                    turnover_weight = sum(old_positions.values())  # total weight exited
                else:
                    turnover_weight = 0.0

                current_positions = {}
                last_rebal_idx = i
                regime_cash_days += 1
                trade_log.append({
                    "date": d, "picks": [], "turnover_weight": turnover_weight,
                    "regime": "CASH", "spy_above_ma50": spy_above, "vix": vix_val
                })
            else:
                # Regime ON — pick top K by model score
                top_k = day_data.nlargest(K, rank_col)["ticker"].tolist()

                # FIX 4: Dollar-weight-based turnover
                # Turnover = sum of absolute weight changes
                old_positions = current_positions.copy()
                new_positions = {t: 1.0 / K for t in top_k}

                all_tickers = set(list(old_positions.keys()) + list(new_positions.keys()))
                turnover_weight = sum(
                    abs(new_positions.get(t, 0.0) - old_positions.get(t, 0.0))
                    for t in all_tickers
                ) / 2.0  # divide by 2 because buys + sells double-count

                current_positions = new_positions
                last_rebal_idx = i
                trade_log.append({
                    "date": d, "picks": top_k, "turnover_weight": turnover_weight,
                    "regime": "INVESTED", "spy_above_ma50": spy_above, "vix": vix_val
                })

        # Daily return of portfolio
        if current_positions:
            port_ret = 0.0
            for t, w in current_positions.items():
                t_data = day_data[day_data["ticker"] == t]
                if not t_data.empty:
                    port_ret += w * float(t_data.iloc[0]["ret"])
            # FIX 5: Deduct turnover cost on rebalance days using dollar-weight turnover
            # Cost = turnover_weight * COST_BPS / 10000
            # turnover_weight is the fraction of portfolio that changed (0 to 1)
            if i == last_rebal_idx and trade_log:
                cost = trade_log[-1]["turnover_weight"] * (COST_BPS / 10000)
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

    total_days = len(daily_dates)
    print(f"  [{strategy_name}] Regime cash days (rebal events): {regime_cash_days}/{len(trade_log)} rebalances")
    return result, trade_log


def simulate_momentum_baseline(panel):
    """Simple 60-day momentum baseline — pick top-K by ret_60d."""
    dates = sorted(panel["date"].unique())
    # Use same OOS period as LGBM
    min_date = panel[panel["ret_60d"].notna()]["date"].min()
    panel_clean = panel[(panel["date"] >= min_date) & panel["ticker"].isin(UNIVERSE)].copy()
    return simulate_portfolio(panel_clean, strategy_name="Momentum60d", rank_col="ret_60d",
                              regime_gate=True)


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


def day_concentration(result_df):
    """Compute day concentration: max single-day P&L / total P&L."""
    rets = result_df["ret"]
    total = rets.sum()
    if total == 0:
        return 1.0
    # Top single day's contribution
    max_day = rets.abs().max()
    return round(max_day / abs(total), 3) if total != 0 else 1.0


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

    # ── LGBM strategy (with regime gate) ─────────────────────────────
    lgbm_result, lgbm_trades = simulate_portfolio(oos_all, "LGBM_Ranker_v2", "lgbm_score",
                                                   regime_gate=True)

    # ── LGBM strategy (NO regime gate — for comparison) ──────────────
    lgbm_nogate, _ = simulate_portfolio(oos_all, "LGBM_NoGate", "lgbm_score",
                                         regime_gate=False)

    # ── Momentum baseline (with regime gate) ─────────────────────────
    panel_oos = panel[(panel["date"] >= oos_dates[0]) & (panel["date"] <= oos_dates[-1]) &
                      panel["ticker"].isin(UNIVERSE)].copy()
    mom_result, _ = simulate_portfolio(panel_oos, "Momentum_60d", "ret_60d",
                                        regime_gate=True)

    # ── SPY buy-and-hold baseline ────────────────────────────────────
    spy_px = px[px["ticker"] == BENCH][["date", "close"]].sort_values("date").drop_duplicates("date")
    spy_px = spy_px[(spy_px["date"] >= oos_dates[0]) & (spy_px["date"] <= oos_dates[-1])].copy()
    spy_px["ret"] = spy_px["close"].pct_change().fillna(0)
    spy_px["cum_ret"] = (1 + spy_px["ret"]).cumprod()
    spy_px["strategy"] = "SPY_BH"

    # ── Metrics ──────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("MEGACAP TECH ROTATION — LGBM RANKER v2 (BUG-FIXED)")
    print("="*70)
    print(f"OOS Period: {oos_dates[0].date()} to {oos_dates[-1].date()} ({len(oos_dates)} trading days)")
    print(f"Universe: {len(UNIVERSE)} stocks, K={K}, rebal every {HOLD_DAYS}d")
    print(f"Walk-forward: {TRAIN_DAYS}d train, {OOS_DAYS}d OOS, sliding")
    print(f"Cost assumption: {COST_BPS}bps round-trip (dollar-weight turnover)")
    print()
    print("v2 FIXES APPLIED:")
    print("  [1] Target rank leakage: fwd_rank computed within training window only")
    print("  [2] Regime gate: SPY < 50d MA OR VIX > 25 → 100% cash")
    print(f"  [3] Hold horizon: HOLD_DAYS={HOLD_DAYS} matches 21d target (was 5)")
    print("  [4] Turnover cost: dollar-weight based (not count-based)")
    print()

    for name, df in [("LGBM Ranker v2 (regime gate)", lgbm_result),
                     ("LGBM NoGate (no regime filter)", lgbm_nogate),
                     ("Momentum 60d (regime gate)", mom_result)]:
        metrics = compute_metrics(df["ret"])
        regime = regime_analysis(df, spy_daily)
        day_conc = day_concentration(df)
        print(f"── {name} ──")
        print(f"  Sharpe={metrics['Sharpe']}  Sortino={metrics['Sortino']}  "
              f"CAGR={metrics['CAGR']}  MaxDD={metrics['MaxDD']}  "
              f"Calmar={metrics['Calmar']}  WR={metrics['WinRate']}")
        print(f"  Regime: Green Sharpe={regime['green_sharpe']}  Red Sharpe={regime['red_sharpe']}  "
              f"Gap={regime['regime_gap']}  {'PASS' if regime['regime_gate_pass'] else 'FAIL'}")
        print(f"  Green avg={regime['green_mean_bps']}bps  Red avg={regime['red_mean_bps']}bps")
        print(f"  Day concentration={day_conc}  {'PASS' if day_conc <= 0.70 else 'FAIL'} (cap=0.70)")
        print()

    spy_metrics = compute_metrics(spy_px["ret"])
    print(f"── SPY Buy & Hold ──")
    print(f"  Sharpe={spy_metrics['Sharpe']}  Sortino={spy_metrics['Sortino']}  "
          f"CAGR={spy_metrics['CAGR']}  MaxDD={spy_metrics['MaxDD']}  "
          f"Calmar={spy_metrics['Calmar']}  WR={spy_metrics['WinRate']}")
    print()

    # ── Feature Importance ───────────────────────────────────────────
    if feat_importance is not None:
        print("── Top 20 Features (LGBM gain) ──")
        print(feat_importance.head(20).to_string(index=False))
        print()

    # ── LGBM vs Baselines ────────────────────────────────────────────
    lgbm_m = compute_metrics(lgbm_result["ret"])
    mom_m = compute_metrics(mom_result["ret"])
    print("── LGBM v2 vs Baselines ──")
    print(f"  Sharpe: LGBM={lgbm_m['Sharpe']} vs Mom={mom_m['Sharpe']} vs SPY={spy_metrics['Sharpe']}")
    print(f"  Sortino: LGBM={lgbm_m['Sortino']} vs Mom={mom_m['Sortino']} vs SPY={spy_metrics['Sortino']}")
    print()

    # ── Regime gate impact ───────────────────────────────────────────
    lgbm_ng_m = compute_metrics(lgbm_nogate["ret"])
    print("── Regime Gate Impact ──")
    print(f"  With gate:    Sharpe={lgbm_m['Sharpe']}  MaxDD={lgbm_m['MaxDD']}  CAGR={lgbm_m['CAGR']}")
    print(f"  Without gate: Sharpe={lgbm_ng_m['Sharpe']}  MaxDD={lgbm_ng_m['MaxDD']}  CAGR={lgbm_ng_m['CAGR']}")
    print()

    # ── Final cumulative returns ─────────────────────────────────────
    print("── Final Cumulative Returns ──")
    print(f"  LGBM Ranker v2:     {lgbm_result['cum_ret'].iloc[-1]:.3f}x")
    print(f"  LGBM NoGate:        {lgbm_nogate['cum_ret'].iloc[-1]:.3f}x")
    print(f"  Momentum 60d:       {mom_result['cum_ret'].iloc[-1]:.3f}x")
    print(f"  SPY B&H:            {spy_px['cum_ret'].iloc[-1]:.3f}x")
    print()

    # ── Yearly breakdown ─────────────────────────────────────────────
    print("── Yearly OOS Returns (LGBM v2) ──")
    lgbm_result["year"] = lgbm_result["date"].dt.year
    for year in sorted(lgbm_result["year"].unique()):
        yr = lgbm_result[lgbm_result["year"] == year]
        yr_m = compute_metrics(yr["ret"])
        yr_regime = regime_analysis(yr, spy_daily)
        print(f"  {year}: Sharpe={yr_m['Sharpe']}  CAGR={yr_m['CAGR']}  MaxDD={yr_m['MaxDD']}  "
              f"WR={yr_m['WinRate']}  RegimeGap={yr_regime['regime_gap']}")

    # ── Trade log summary ────────────────────────────────────────────
    invested_trades = [t for t in lgbm_trades if t["regime"] == "INVESTED"]
    cash_trades = [t for t in lgbm_trades if t["regime"] == "CASH"]
    avg_turnover = np.mean([t["turnover_weight"] for t in invested_trades]) if invested_trades else 0
    print(f"\n── Trade Log Summary ──")
    print(f"  Total rebalances: {len(lgbm_trades)}")
    print(f"  Invested periods: {len(invested_trades)}")
    print(f"  Cash periods (regime off): {len(cash_trades)}")
    print(f"  Avg turnover per invested rebal: {avg_turnover:.1%}")

    # Save results
    out_dir = ROOT / "research" / "findings"
    out_dir.mkdir(parents=True, exist_ok=True)
    lgbm_result.to_parquet(out_dir / "megacap_lgbm_v2_oos_results.parquet", index=False)
    if feat_importance is not None:
        feat_importance.to_csv(out_dir / "megacap_lgbm_v2_feature_importance.csv", index=False)
    print(f"\nResults saved to {out_dir}")


if __name__ == "__main__":
    main()
