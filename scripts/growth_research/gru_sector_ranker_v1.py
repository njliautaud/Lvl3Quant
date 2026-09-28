"""
GRU Sector Timing Model v1
===========================
Tests GRU neural network for sector ranking (14-day forward returns).
Potential replacement for LGBM ranker in V8 production strategy.

Variants:
  A: GRU ranker (12-week lookback)
  B: GRU ranker (24-week lookback)
  C: LGBM baseline (same data, for comparison)

Output: C:\\Users\\claude\\Lvl3Quant\\output\\growth_research\\gru_sector_ranker_v1\\
"""

import os
import sys
import json
import warnings
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# --- Platform-aware paths ---
if sys.platform == "win32":
    sys.path.insert(0, r"C:\Users\claude\Lvl3Quant")
    OUTPUT_DIR = Path(r"C:\Users\claude\Lvl3Quant\output\growth_research\gru_sector_ranker_v1")
else:
    sys.path.insert(0, "/home/jupiter/Lvl3Quant")
    OUTPUT_DIR = Path("/home/jupiter/Lvl3Quant/output/growth_research/gru_sector_ranker_v1")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --- Config ---
SECTORS = ["XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLB", "XLU", "XLRE", "XLC"]
BENCHMARK = "SPY"
VIX_TICKER = "^VIX"
DTE = 14
FORWARD_HORIZON = 14  # calendar days for forward return
TRAIN_WINDOW = 500    # trading days for walk-forward train
TOP_K = 3
CAP = 645
SPREAD_PCT = 3.0
COMMISSION = 2.60
ENTRY_HAIRCUT = 0.15
VIX_FILTER = 30.0

# GRU config
GRU_HIDDEN = 64
GRU_LAYERS = 2
GRU_DROPOUT = 0.2
GRU_EPOCHS = 30
GRU_BATCH = 32
GRU_LR = 1e-3
LOOKBACKS = {"A": 12, "B": 24}  # weeks

# Feature list (17 V6 features)
FEATURE_NAMES = [
    "ret_5d", "ret_10d", "ret_21d", "ret_63d", "ret_126d", "ret_252d",
    "sharpe_63d", "pct_52w_high", "mom_accel", "pct_pos_months_12m",
    "sortino_63d", "calmar_1y", "up_capture", "trend_r2_63d",
    "trend_slope_63d", "sector_spy_beta_63d", "cross_sector_dispersion"
]
N_FEATURES = len(FEATURE_NAMES)


def download_data():
    """Download sector ETF + SPY + VIX data via yfinance."""
    import yfinance as yf
    tickers = SECTORS + [BENCHMARK, VIX_TICKER]
    print(f"Downloading {len(tickers)} tickers...")
    data = yf.download(tickers, start="2015-01-01", auto_adjust=True, progress=False)
    close = data["Close"].copy()
    close.columns = [str(c) for c in close.columns]
    # Rename VIX column
    if "^VIX" in close.columns:
        close = close.rename(columns={"^VIX": "VIX"})
    close = close.dropna(how="all")
    print(f"Downloaded data: {close.shape[0]} days, {close.shape[1]} tickers")
    return close


def compute_features(close: pd.DataFrame) -> pd.DataFrame:
    """Compute 17 V6 features for each sector at weekly frequency."""
    # Resample to weekly (Friday close)
    weekly = close.resample("W-FRI").last().dropna(how="all")
    spy = weekly[BENCHMARK]
    vix = weekly["VIX"] if "VIX" in weekly.columns else None

    records = []
    for sector in SECTORS:
        px = weekly[sector].dropna()
        if len(px) < 260:
            continue

        ret = px.pct_change()
        spy_ret = spy.reindex(px.index).pct_change()

        for i in range(260, len(px)):
            date = px.index[i]
            window = px.iloc[max(0, i-260):i+1]
            ret_window = ret.iloc[max(0, i-260):i+1]
            spy_ret_window = spy_ret.iloc[max(0, i-260):i+1]

            row = {"date": date, "sector": sector}

            # Returns
            row["ret_5d"] = px.iloc[i] / px.iloc[i-1] - 1 if i >= 1 else 0
            row["ret_10d"] = px.iloc[i] / px.iloc[i-2] - 1 if i >= 2 else 0
            row["ret_21d"] = px.iloc[i] / px.iloc[i-4] - 1 if i >= 4 else 0
            row["ret_63d"] = px.iloc[i] / px.iloc[i-13] - 1 if i >= 13 else 0
            row["ret_126d"] = px.iloc[i] / px.iloc[i-26] - 1 if i >= 26 else 0
            row["ret_252d"] = px.iloc[i] / px.iloc[i-52] - 1 if i >= 52 else 0

            # Sharpe 63d (13 weeks)
            r13 = ret_window.iloc[-13:]
            row["sharpe_63d"] = r13.mean() / (r13.std() + 1e-10) * np.sqrt(52)

            # Pct of 52-week high
            high_52w = window.max()
            row["pct_52w_high"] = px.iloc[i] / high_52w if high_52w > 0 else 0

            # Momentum acceleration
            mom_short = px.iloc[i] / px.iloc[i-4] - 1 if i >= 4 else 0
            mom_long = px.iloc[i-4] / px.iloc[i-8] - 1 if i >= 8 else 0
            row["mom_accel"] = mom_short - mom_long

            # Pct positive months (12m)
            monthly_rets = ret_window.iloc[-52:].resample("ME").sum()
            row["pct_pos_months_12m"] = (monthly_rets > 0).mean() if len(monthly_rets) > 0 else 0.5

            # Sortino 63d
            neg_ret = r13[r13 < 0]
            downside_std = neg_ret.std() if len(neg_ret) > 2 else r13.std()
            row["sortino_63d"] = r13.mean() / (downside_std + 1e-10) * np.sqrt(52)

            # Calmar 1y
            peak = window.cummax()
            dd = (window - peak) / peak
            max_dd = dd.min()
            ann_ret = row["ret_252d"]
            row["calmar_1y"] = ann_ret / (abs(max_dd) + 1e-10) if max_dd < -0.001 else ann_ret * 10

            # Up capture
            up_mask = spy_ret_window > 0
            up_spy = spy_ret_window[up_mask].iloc[-13:]
            up_sector = ret_window.reindex(up_spy.index)
            row["up_capture"] = up_sector.mean() / (up_spy.mean() + 1e-10) if len(up_spy) > 2 else 1.0

            # Trend R2 and slope (63d = 13 weeks)
            y = np.log(window.iloc[-13:].values + 1e-10)
            x = np.arange(len(y))
            if len(y) >= 5:
                coeffs = np.polyfit(x, y, 1)
                y_pred = np.polyval(coeffs, x)
                ss_res = np.sum((y - y_pred) ** 2)
                ss_tot = np.sum((y - y.mean()) ** 2)
                row["trend_r2_63d"] = 1 - ss_res / (ss_tot + 1e-10)
                row["trend_slope_63d"] = coeffs[0]
            else:
                row["trend_r2_63d"] = 0
                row["trend_slope_63d"] = 0

            # Beta to SPY (63d)
            s_ret = ret_window.iloc[-13:].values
            b_ret = spy_ret_window.reindex(ret_window.index).iloc[-13:].values
            valid = ~(np.isnan(s_ret) | np.isnan(b_ret))
            if valid.sum() > 5:
                cov = np.cov(s_ret[valid], b_ret[valid])
                row["sector_spy_beta_63d"] = cov[0, 1] / (cov[1, 1] + 1e-10)
            else:
                row["sector_spy_beta_63d"] = 1.0

            # Cross-sector dispersion (current week)
            all_sector_rets = []
            for s in SECTORS:
                if s in weekly.columns and date in weekly.index:
                    sr = weekly[s].pct_change()
                    if date in sr.index and not np.isnan(sr.loc[date]):
                        all_sector_rets.append(sr.loc[date])
            row["cross_sector_dispersion"] = np.std(all_sector_rets) if len(all_sector_rets) > 3 else 0

            # Forward return (target) — 2-week forward
            if i + 2 < len(px):
                row["fwd_ret_14d"] = px.iloc[i+2] / px.iloc[i] - 1
            else:
                row["fwd_ret_14d"] = np.nan

            # VIX
            if vix is not None and date in vix.index:
                row["vix"] = vix.loc[date]
            else:
                row["vix"] = 20.0

            records.append(row)

    df = pd.DataFrame(records)
    df = df.sort_values(["date", "sector"]).reset_index(drop=True)
    print(f"Computed features: {df.shape[0]} rows, {df.date.nunique()} weeks, {df.sector.nunique()} sectors")
    return df


def build_sequences(df: pd.DataFrame, lookback: int):
    """
    Build (X, y) for GRU: X shape = (n_samples, lookback, n_features), y = percentile rank.
    Each sample is one (sector, date) with `lookback` weeks of history.
    """
    dates = sorted(df["date"].unique())
    sectors = sorted(df["sector"].unique())

    # Pivot features
    feature_dfs = {}
    for feat in FEATURE_NAMES:
        pivot = df.pivot(index="date", columns="sector", values=feat)
        feature_dfs[feat] = pivot

    # Forward return pivot for targets
    fwd_pivot = df.pivot(index="date", columns="sector", values="fwd_ret_14d")
    vix_series = df.groupby("date")["vix"].first()

    X_list, y_list, meta_list = [], [], []

    for t in range(lookback, len(dates)):
        date = dates[t]
        if date not in fwd_pivot.index:
            continue

        fwd_rets = fwd_pivot.loc[date].dropna()
        if len(fwd_rets) < 5:
            continue

        # VIX filter
        vix_val = vix_series.get(date, 20.0)
        if vix_val > VIX_FILTER:
            continue

        # Percentile ranks
        ranks = fwd_rets.rank(pct=True)

        hist_dates = dates[t - lookback:t]

        for sector in fwd_rets.index:
            seq = np.zeros((lookback, N_FEATURES))
            valid = True
            for wi, hd in enumerate(hist_dates):
                for fi, feat in enumerate(FEATURE_NAMES):
                    if hd in feature_dfs[feat].index and sector in feature_dfs[feat].columns:
                        val = feature_dfs[feat].loc[hd, sector]
                        if np.isnan(val):
                            valid = False
                            break
                        seq[wi, fi] = val
                    else:
                        valid = False
                        break
                if not valid:
                    break

            if not valid:
                continue

            X_list.append(seq)
            y_list.append(ranks[sector])
            meta_list.append({"date": date, "sector": sector, "fwd_ret": fwd_rets[sector], "vix": vix_val})

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    meta = pd.DataFrame(meta_list)

    print(f"  Built sequences: X={X.shape}, y={y.shape}, dates={meta['date'].nunique()}")
    return X, y, meta


def normalize_sequences(X_train, X_test):
    """Z-score normalize per feature across training set."""
    n_train = X_train.shape[0]
    # Reshape to (n * lookback, features)
    flat = X_train.reshape(-1, X_train.shape[-1])
    mu = flat.mean(axis=0)
    sigma = flat.std(axis=0) + 1e-8
    X_train_norm = (X_train - mu) / sigma
    X_test_norm = (X_test - mu) / sigma
    return X_train_norm, X_test_norm, mu, sigma


# ============================================================
# GRU Model
# ============================================================
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class GRURanker(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        # x: (batch, seq_len, features)
        out, _ = self.gru(x)
        last = out[:, -1, :]  # last hidden state
        return self.head(last).squeeze(-1)


def listmle_loss(y_pred, y_true):
    """
    ListMLE loss for learning to rank.
    y_pred: predicted scores (batch,)
    y_true: true relevance/rank scores (batch,)
    """
    # Sort by true relevance (descending)
    _, indices = y_true.sort(descending=True)
    y_pred_sorted = y_pred[indices]
    # ListMLE: sum of log-softmax from top down
    n = len(y_pred_sorted)
    loss = torch.tensor(0.0, device=y_pred.device)
    for i in range(n - 1):
        loss -= torch.log_softmax(y_pred_sorted[i:], dim=0)[0]
    return loss / max(n - 1, 1)


def train_gru_variant(X, y, meta, lookback_name, device):
    """Walk-forward training of GRU ranker."""
    dates = sorted(meta["date"].unique())
    n_dates = len(dates)

    # Weekly walk-forward: ~500 trading days = ~100 weeks for train
    train_weeks = 100
    results = []

    print(f"\n{'='*60}")
    print(f"Variant {lookback_name}: GRU Ranker")
    print(f"  Dates: {n_dates}, Train window: {train_weeks} weeks")
    print(f"  Device: {device}")
    print(f"{'='*60}")

    for test_idx in range(train_weeks, n_dates):
        test_date = dates[test_idx]
        train_dates = set(dates[max(0, test_idx - train_weeks):test_idx])

        train_mask = meta["date"].isin(train_dates).values
        test_mask = (meta["date"] == test_date).values

        if train_mask.sum() < 50 or test_mask.sum() < 3:
            continue

        X_tr, y_tr = X[train_mask], y[train_mask]
        X_te, y_te = X[test_mask], y[test_mask]
        meta_te = meta[test_mask].copy()

        # Normalize
        X_tr_n, X_te_n, _, _ = normalize_sequences(X_tr, X_te)

        # Convert to tensors
        X_tr_t = torch.FloatTensor(X_tr_n).to(device)
        y_tr_t = torch.FloatTensor(y_tr).to(device)
        X_te_t = torch.FloatTensor(X_te_n).to(device)

        # Train
        model = GRURanker(N_FEATURES, GRU_HIDDEN, GRU_LAYERS, GRU_DROPOUT).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=GRU_LR)
        mse_loss = nn.MSELoss()

        dataset = TensorDataset(X_tr_t, y_tr_t)
        loader = DataLoader(dataset, batch_size=GRU_BATCH, shuffle=True, drop_last=False)

        model.train()
        for epoch in range(GRU_EPOCHS):
            for xb, yb in loader:
                optimizer.zero_grad()
                pred = model(xb)
                loss = mse_loss(pred, yb)
                # Add listwise ranking loss on small batches
                if len(pred) >= 3:
                    loss = loss + 0.1 * listmle_loss(pred, yb)
                loss.backward()
                optimizer.step()

        # Predict
        model.eval()
        with torch.no_grad():
            preds = model(X_te_t).cpu().numpy()

        meta_te = meta_te.copy()
        meta_te["pred_score"] = preds
        meta_te["pred_rank"] = meta_te["pred_score"].rank(ascending=False)

        results.append(meta_te)

        if (test_idx - train_weeks) % 50 == 0:
            print(f"  Fold {test_idx - train_weeks}/{n_dates - train_weeks} done ({test_date.strftime('%Y-%m-%d')})")

    if not results:
        print("  No results!")
        return pd.DataFrame()

    all_results = pd.concat(results, ignore_index=True)
    print(f"  Total predictions: {len(all_results)}, dates: {all_results['date'].nunique()}")
    return all_results


def train_lgbm_baseline(df: pd.DataFrame):
    """Walk-forward LGBM ranker for comparison."""
    try:
        import lightgbm as lgb
    except ImportError:
        print("  LightGBM not installed, skipping baseline C")
        return pd.DataFrame()

    dates = sorted(df["date"].unique())
    n_dates = len(dates)
    train_weeks = 100

    # Filter to rows with valid forward return and VIX < 30
    df_valid = df.dropna(subset=["fwd_ret_14d"]).copy()
    df_valid = df_valid[df_valid["vix"] <= VIX_FILTER].copy()

    # Compute percentile rank target per date
    df_valid["target_rank"] = df_valid.groupby("date")["fwd_ret_14d"].rank(pct=True)

    results = []
    print(f"\n{'='*60}")
    print(f"Variant C: LGBM Baseline Ranker")
    print(f"{'='*60}")

    for test_idx in range(train_weeks, n_dates):
        test_date = dates[test_idx]
        train_dates = dates[max(0, test_idx - train_weeks):test_idx]

        train_data = df_valid[df_valid["date"].isin(train_dates)]
        test_data = df_valid[df_valid["date"] == test_date]

        if len(train_data) < 50 or len(test_data) < 3:
            continue

        X_tr = train_data[FEATURE_NAMES].values
        y_tr = train_data["target_rank"].values
        X_te = test_data[FEATURE_NAMES].values

        # Handle NaN
        X_tr = np.nan_to_num(X_tr, 0)
        X_te = np.nan_to_num(X_te, 0)

        model = lgb.LGBMRegressor(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_samples=10,
            verbosity=-1,
        )
        model.fit(X_tr, y_tr)
        preds = model.predict(X_te)

        meta_te = test_data[["date", "sector", "fwd_ret_14d", "vix"]].copy()
        meta_te["pred_score"] = preds
        meta_te["pred_rank"] = meta_te["pred_score"].rank(ascending=False)
        meta_te["fwd_ret"] = meta_te["fwd_ret_14d"]
        results.append(meta_te)

        if (test_idx - train_weeks) % 50 == 0:
            print(f"  Fold {test_idx - train_weeks}/{n_dates - train_weeks} done ({test_date.strftime('%Y-%m-%d')})")

    if not results:
        return pd.DataFrame()

    all_results = pd.concat(results, ignore_index=True)
    print(f"  Total predictions: {len(all_results)}, dates: {all_results['date'].nunique()}")
    return all_results


def evaluate_ranker(results: pd.DataFrame, variant_name: str):
    """Evaluate ranking quality and simulated portfolio performance."""
    from scipy.stats import spearmanr

    if results.empty:
        return {"variant": variant_name, "error": "no results"}

    dates = sorted(results["date"].unique())

    # Per-date Spearman correlation
    spearmans = []
    ndcg3_list = []
    top3_rets = []
    bottom3_rets = []

    for date in dates:
        day = results[results["date"] == date]
        if len(day) < 5:
            continue

        # Spearman
        rho, _ = spearmanr(day["pred_score"], day["fwd_ret"])
        if not np.isnan(rho):
            spearmans.append(rho)

        # NDCG@3
        sorted_by_pred = day.sort_values("pred_score", ascending=False)
        ideal = day.sort_values("fwd_ret", ascending=False)
        dcg3 = sum([(sorted_by_pred.iloc[i]["fwd_ret"] if i < len(sorted_by_pred) else 0) / np.log2(i + 2)
                     for i in range(min(3, len(sorted_by_pred)))])
        idcg3 = sum([(ideal.iloc[i]["fwd_ret"] if i < len(ideal) else 0) / np.log2(i + 2)
                      for i in range(min(3, len(ideal)))])
        ndcg = dcg3 / (idcg3 + 1e-10) if idcg3 > 0 else 0
        ndcg3_list.append(ndcg)

        # Top-3 and bottom-3 average return
        top3 = sorted_by_pred.head(TOP_K)["fwd_ret"].mean()
        bot3 = sorted_by_pred.tail(TOP_K)["fwd_ret"].mean()
        top3_rets.append(top3)
        bottom3_rets.append(bot3)

    # Portfolio simulation (buy top-K sectors every rebalance)
    portfolio_rets = np.array(top3_rets)
    # Apply costs: entry haircut + commission
    n_trades = len(portfolio_rets)
    cost_per_trade = COMMISSION * TOP_K * 2 / (CAP * 100)  # rough pct cost
    net_rets = portfolio_rets - cost_per_trade

    cum_ret = (1 + net_rets).cumprod()
    total_ret = cum_ret[-1] - 1 if len(cum_ret) > 0 else 0

    # Annualize (biweekly rebalance ~ 26 per year)
    n_periods = len(net_rets)
    ann_factor = 26  # ~26 biweekly periods per year
    ann_ret = (1 + total_ret) ** (ann_factor / max(n_periods, 1)) - 1

    # Risk metrics
    ret_std = net_rets.std() * np.sqrt(ann_factor)
    sharpe = ann_ret / (ret_std + 1e-10)

    neg_rets = net_rets[net_rets < 0]
    downside_std = neg_rets.std() * np.sqrt(ann_factor) if len(neg_rets) > 2 else ret_std
    sortino = ann_ret / (downside_std + 1e-10)

    # Max drawdown
    running_max = np.maximum.accumulate(cum_ret)
    dd = (cum_ret - running_max) / running_max
    max_dd = dd.min()

    # Win rate
    win_rate = (net_rets > 0).mean()

    # Profit factor
    gross_profit = net_rets[net_rets > 0].sum()
    gross_loss = abs(net_rets[net_rets < 0].sum())
    pf = gross_profit / (gross_loss + 1e-10)

    # Long-short spread
    ls_spread = np.mean(top3_rets) - np.mean(bottom3_rets)

    metrics = {
        "variant": variant_name,
        "n_periods": n_periods,
        "n_years": n_periods / ann_factor,
        "spearman_mean": np.mean(spearmans),
        "spearman_median": np.median(spearmans),
        "ndcg3_mean": np.mean(ndcg3_list),
        "top3_avg_ret": np.mean(top3_rets),
        "bot3_avg_ret": np.mean(bottom3_rets),
        "ls_spread": ls_spread,
        "ann_return": ann_ret,
        "ann_vol": ret_std,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_dd": max_dd,
        "win_rate": win_rate,
        "profit_factor": pf,
        "total_return": total_ret,
    }
    return metrics


def print_comparison(all_metrics):
    """Print comparison table."""
    print(f"\n{'='*80}")
    print("GRU SECTOR RANKER v1 — FULL COMPARISON")
    print(f"{'='*80}")

    header = f"{'Metric':<25} "
    for m in all_metrics:
        header += f"{'[' + m['variant'] + ']':>15} "
    print(header)
    print("-" * 80)

    keys = [
        ("n_periods", "Rebalance Periods", "{:.0f}"),
        ("n_years", "Years", "{:.1f}"),
        ("spearman_mean", "Spearman (mean)", "{:.4f}"),
        ("spearman_median", "Spearman (median)", "{:.4f}"),
        ("ndcg3_mean", "NDCG@3 (mean)", "{:.4f}"),
        ("top3_avg_ret", "Top-3 Avg Ret", "{:.4f}"),
        ("bot3_avg_ret", "Bot-3 Avg Ret", "{:.4f}"),
        ("ls_spread", "L/S Spread", "{:.4f}"),
        ("ann_return", "Ann. Return", "{:.2%}"),
        ("ann_vol", "Ann. Volatility", "{:.2%}"),
        ("sharpe", "Sharpe Ratio", "{:.3f}"),
        ("sortino", "Sortino Ratio", "{:.3f}"),
        ("max_dd", "Max Drawdown", "{:.2%}"),
        ("win_rate", "Win Rate", "{:.2%}"),
        ("profit_factor", "Profit Factor", "{:.2f}"),
        ("total_return", "Total Return", "{:.2%}"),
    ]

    for key, label, fmt in keys:
        row = f"{label:<25} "
        for m in all_metrics:
            val = m.get(key, "N/A")
            if isinstance(val, (int, float)):
                row += f"{fmt.format(val):>15} "
            else:
                row += f"{'N/A':>15} "
        print(row)

    # Winner
    valid = [m for m in all_metrics if "error" not in m]
    if valid:
        best = max(valid, key=lambda x: x.get("sharpe", -999))
        print(f"\n>>> BEST VARIANT BY SHARPE: {best['variant']} (Sharpe={best['sharpe']:.3f})")
        best_spearman = max(valid, key=lambda x: x.get("spearman_mean", -999))
        print(f">>> BEST VARIANT BY RANKING: {best_spearman['variant']} (Spearman={best_spearman['spearman_mean']:.4f})")


def main():
    print("=" * 60)
    print("GRU Sector Timing Model v1")
    print(f"Started: {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # Check GPU
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("WARNING: No GPU detected, running on CPU")

    # Step 1: Download data
    print("\n--- Step 1: Download Data ---")
    close = download_data()

    # Step 2: Compute features
    print("\n--- Step 2: Compute Features ---")
    df = compute_features(close)
    feat_path = OUTPUT_DIR / "features.parquet"
    df.to_parquet(feat_path, index=False)
    print(f"  Saved features to {feat_path}")

    # Step 3: Build sequences and train GRU variants
    all_metrics = []

    for variant_name, lookback in LOOKBACKS.items():
        print(f"\n--- Step 3{variant_name}: Build Sequences (lookback={lookback} weeks) ---")
        X, y, meta = build_sequences(df, lookback)

        if len(X) == 0:
            print(f"  Skipping variant {variant_name}: no data")
            all_metrics.append({"variant": f"{variant_name}_GRU_{lookback}w", "error": "no data"})
            continue

        print(f"\n--- Step 4{variant_name}: Train GRU Variant {variant_name} ---")
        results = train_gru_variant(X, y, meta, f"{variant_name}_GRU_{lookback}w", device)

        if not results.empty:
            # Save predictions
            pred_path = OUTPUT_DIR / f"predictions_{variant_name}.parquet"
            results.to_parquet(pred_path, index=False)

            metrics = evaluate_ranker(results, f"{variant_name}_GRU_{lookback}w")
            all_metrics.append(metrics)
        else:
            all_metrics.append({"variant": f"{variant_name}_GRU_{lookback}w", "error": "no results"})

    # Step 5: LGBM baseline
    print("\n--- Step 5: LGBM Baseline ---")
    lgbm_results = train_lgbm_baseline(df)
    if not lgbm_results.empty:
        pred_path = OUTPUT_DIR / "predictions_C_lgbm.parquet"
        lgbm_results.to_parquet(pred_path, index=False)
        metrics_c = evaluate_ranker(lgbm_results, "C_LGBM")
        all_metrics.append(metrics_c)
    else:
        all_metrics.append({"variant": "C_LGBM", "error": "no results"})

    # Step 6: Print comparison
    print_comparison(all_metrics)

    # Save results
    results_path = OUTPUT_DIR / "comparison_results.json"
    # Convert numpy types for JSON serialization
    serializable = []
    for m in all_metrics:
        sm = {}
        for k, v in m.items():
            if isinstance(v, (np.floating, np.integer)):
                sm[k] = float(v)
            else:
                sm[k] = v
        serializable.append(sm)

    with open(results_path, "w") as f:
        json.dump(serializable, f, indent=2, default=str)
    print(f"\nResults saved to {results_path}")
    print(f"Finished: {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
