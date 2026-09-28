"""
Post-processing for pairs_trading_v1 — regime analysis, permutation test, final summary.
Reads from output/pairs_trading_v1/ CSVs. Logs to MLflow run.
"""
import sys
import warnings
import logging
import numpy as np
import pandas as pd
import yfinance as yf
import mlflow

warnings.filterwarnings("ignore")

OUTPUT_DIR = "/home/nick/Lvl3Quant/output/pairs_trading_v1"
MLFLOW_URI = "http://jupiter:5000"
EXPERIMENT_NAME = "pairs_trading_v1"
N_PERMS = 100

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# ─── Load results ─────────────────────────────────────────────────────────────
wf_df = pd.read_csv(f"{OUTPUT_DIR}/walkforward_results.csv")
full_returns = pd.read_csv(f"{OUTPUT_DIR}/portfolio_returns.csv", index_col=0, parse_dates=True).squeeze()
log.info(f"Loaded {len(wf_df)} windows, {len(full_returns)} return days")

# ─── Overall metrics ─────────────────────────────────────────────────────────
r = full_returns.values
ann = 252
sharpe = (r.mean() / r.std()) * np.sqrt(ann)
down_std = r[r < 0].std() if (r < 0).any() else 1e-10
sortino = (r.mean() / down_std) * np.sqrt(ann)
total_ret = (1 + r).prod() - 1
cum = (1 + r).cumprod()
running_max = np.maximum.accumulate(cum)
max_dd = ((cum - running_max) / running_max).min()
wr = (r > 0).mean()
gross_profit = r[r > 0].sum() if (r > 0).any() else 0
gross_loss = abs(r[r < 0].sum()) if (r < 0).any() else 1e-10
pf = gross_profit / gross_loss

log.info("\n" + "=" * 60)
log.info("OVERALL METRICS")
log.info("=" * 60)
log.info(f"  Sharpe:         {sharpe:.3f}")
log.info(f"  Sortino:        {sortino:.3f}")
log.info(f"  Win Rate:       {wr:.1%}")
log.info(f"  Profit Factor:  {pf:.3f}")
log.info(f"  Max Drawdown:   {max_dd:.1%}")
log.info(f"  Total Return:   {total_ret:.1%}")
log.info(f"  N days:         {len(r)}")

# ─── SPY for regime ───────────────────────────────────────────────────────────
log.info("\nDownloading SPY for regime classification ...")
spy_raw = yf.download("SPY", start="2014-01-01", end="2026-07-01", auto_adjust=True, progress=False)
if isinstance(spy_raw.columns, pd.MultiIndex):
    spy_raw.columns = spy_raw.columns.get_level_values(0)
spy_prices = spy_raw["Close"].squeeze()
spy_returns = spy_prices.pct_change().dropna()

# ─── Regime classification ────────────────────────────────────────────────────
aligned = spy_returns.reindex(full_returns.index).fillna(0).squeeze()
# Ensure unique index
aligned = aligned[~aligned.index.duplicated(keep='first')]
full_returns_clean = full_returns[~full_returns.index.duplicated(keep='first')]

# classify
regimes = pd.Series("flat", index=aligned.index, dtype=str)
for i, (date, ret) in enumerate(aligned.items()):
    if ret > 0.003:
        regimes.iloc[i] = "green"
    elif ret < -0.003:
        regimes.iloc[i] = "red"

green_r = full_returns_clean[regimes == "green"].values
red_r   = full_returns_clean[regimes == "red"].values
flat_r  = full_returns_clean[regimes == "flat"].values

def ann_sharpe(r):
    if len(r) < 5 or r.std() < 1e-10:
        return 0.0
    return (r.mean() / r.std()) * np.sqrt(252)

sg = ann_sharpe(green_r)
sr = ann_sharpe(red_r)
sf = ann_sharpe(flat_r)
denom = max(abs(sg), abs(sr), 1e-10)
gap = abs(sg - sr) / denom
regime_pass = gap <= 0.50

log.info("\n" + "=" * 60)
log.info("REGIME ANALYSIS (HC #428 R1)")
log.info("=" * 60)
log.info(f"  Green days ({(regimes=='green').sum()}): Sharpe={sg:.3f}")
log.info(f"  Red days   ({(regimes=='red').sum()}): Sharpe={sr:.3f}")
log.info(f"  Flat days  ({(regimes=='flat').sum()}): Sharpe={sf:.3f}")
log.info(f"  Regime gap: {gap:.3f} → {'PASS' if regime_pass else 'FAIL (>0.50)'}")

# ─── Permutation test ─────────────────────────────────────────────────────────
log.info(f"\nRunning {N_PERMS} permutation iterations ...")
# Use per-window sharpe as the statistic to shuffle
window_sharpes = wf_df["sharpe"].values
obs_mean_sharpe = window_sharpes.mean()
rng = np.random.default_rng(42)
perm_means = []
for _ in range(N_PERMS):
    perm_means.append(rng.permutation(window_sharpes).mean())
perm_pval = np.mean(np.array(perm_means) >= obs_mean_sharpe)

log.info(f"  Observed mean window Sharpe: {obs_mean_sharpe:.3f}")
log.info(f"  Permuted mean Sharpe (mean of dist): {np.mean(perm_means):.3f}")
log.info(f"  Permutation p-value: {perm_pval:.3f}")
log.info(f"  (p < 0.05 = pair selection adds real value)")

# ─── Window distribution ──────────────────────────────────────────────────────
log.info("\nPer-window Sharpe distribution:")
log.info(f"  mean={wf_df['sharpe'].mean():.3f}  median={wf_df['sharpe'].median():.3f}  "
         f"std={wf_df['sharpe'].std():.3f}  min={wf_df['sharpe'].min():.3f}  max={wf_df['sharpe'].max():.3f}")

# Best/worst windows
best = wf_df.nlargest(5, "sharpe")[["window","test_start","test_end","n_pairs","sharpe","wr","pf"]]
worst = wf_df.nsmallest(5, "sharpe")[["window","test_start","test_end","n_pairs","sharpe","wr","pf"]]
log.info("\nTop 5 windows:")
log.info(best.to_string(index=False))
log.info("\nBottom 5 windows:")
log.info(worst.to_string(index=False))

# ─── MLflow: start new run for post-processing summary ───────────────────────
mlflow.set_tracking_uri(MLFLOW_URI)
mlflow.set_experiment(EXPERIMENT_NAME)
with mlflow.start_run(run_name="pairs_v1_postprocess"):
    mlflow.log_metrics({
        "sharpe":           sharpe,
        "sortino":          sortino,
        "win_rate":         wr,
        "profit_factor":    pf,
        "max_drawdown":     max_dd,
        "total_return":     total_ret,
        "regime_gap":       gap,
        "sharpe_green":     sg,
        "sharpe_red":       sr,
        "sharpe_flat":      sf,
        "perm_pval":        perm_pval,
        "n_wf_windows":     float(len(wf_df)),
        "total_pairs_found": float(wf_df["n_pairs"].sum()),
        "mean_window_sharpe": float(wf_df["sharpe"].mean()),
    })

# ─── FINAL SUMMARY ───────────────────────────────────────────────────────────
log.info("\n" + "=" * 70)
log.info("FINAL SUMMARY — PAIRS TRADING V1")
log.info("=" * 70)
log.info(f"  Walk-forward windows: {len(wf_df)}")
log.info(f"  Total pairs used:     {wf_df['n_pairs'].sum()} (avg {wf_df['n_pairs'].mean():.0f}/window)")
log.info(f"  Cointegration found:  {wf_df['n_pairs'].mean():.0f} pairs/window out of 547 same-sector checked")
log.info(f"")
log.info(f"  SHARPE:               {sharpe:.3f}")
log.info(f"  SORTINO:              {sortino:.3f}")
log.info(f"  WIN RATE:             {wr:.1%}")
log.info(f"  PROFIT FACTOR:        {pf:.3f}")
log.info(f"  MAX DRAWDOWN:         {max_dd:.1%}")
log.info(f"  TOTAL RETURN:         {total_ret:.1%}")
log.info(f"")
log.info(f"  REGIME GAP:           {gap:.3f} ({'PASS' if regime_pass else 'FAIL'})")
log.info(f"  Sharpe(green):        {sg:.3f}")
log.info(f"  Sharpe(red):          {sr:.3f}")
log.info(f"")
log.info(f"  PERMUTATION p-val:    {perm_pval:.3f}")
log.info(f"  Interpretation:       {'Pair selection IS statistically significant' if perm_pval < 0.05 else 'Pair selection NOT significant vs random (p>0.05)'}")
log.info("=" * 70)
