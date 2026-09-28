#!/usr/bin/env bash
# HC #413 — smoke test on v3.3 NPZ. Idempotent (seed=42).
# Designed to swap to v3.4.2 by changing only --npz when 60d NPZ lands.
set -euo pipefail

LVL3="/home/jupiter/Lvl3Quant"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NPZ="${NPZ:-$LVL3/output/cnn_mamba_v3_3_uncertainty_weighted/fold_00_predictions.npz}"
MFE_CFG="${MFE_CFG:-$LVL3/output/hc411_regime_agnostic_20260517_215211/mfe_at_confidence_matrix.csv}"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT="${OUT:-$LVL3/output/hc413_scalping_smoke_${STAMP}}"
MODEL_FILTER="${MODEL_FILTER:-v3.3}"   # change to v3.4.2 once NPZ swap in
ORDER_TYPE="${ORDER_TYPE:-passive_at_touch}"

mkdir -p "$OUT"

echo "[smoke] NPZ:        $NPZ"
echo "[smoke] MFE config: $MFE_CFG"
echo "[smoke] OUT:        $OUT"
echo "[smoke] model:      $MODEL_FILTER  order_type: $ORDER_TYPE"

python3 "$HERE/backtester.py" \
    --npz "$NPZ" \
    --mfe-config "$MFE_CFG" \
    --output-dir "$OUT" \
    --confidence-tier all \
    --horizon all \
    --side both \
    --model "$MODEL_FILTER" \
    --order-type "$ORDER_TYPE" \
    --cancel-eval-window 40 \
    --seed 42

echo ""
echo "[smoke] CSV head:"
head -2 "$OUT/scalping_backtest_results.csv" | cut -c -240
echo "..."
echo "[smoke] Cells passing HC #408 honesty AND net>0 (sorted by Sharpe desc):"
python3 - <<PY
import pandas as pd
df = pd.read_csv("$OUT/scalping_backtest_results.csv")
pf = df[(df.pass_hc408_honesty==True) & (df.realized_net_per_fill>0)].sort_values("sharpe_sqrtN", ascending=False)
if pf.empty:
    print("  (none)")
else:
    cols = ["cell_id","n_fills","realized_net_per_fill","sharpe_sqrtN","pf","wr","day_conc","ci_low_95_net"]
    print(pf[cols].head(10).to_string(index=False))
print()
print("Top 3 cells by Sharpe√N (any pass status):")
cols = ["cell_id","pass_hc408_honesty","n_fills","realized_net_per_fill","sharpe_sqrtN","pf","wr","day_conc","ci_low_95_net"]
print(df.sort_values("sharpe_sqrtN", ascending=False)[cols].head(3).to_string(index=False))
PY

echo ""
echo "[smoke] Verdict at: $OUT/verdict.md"
echo "[smoke] DONE"
