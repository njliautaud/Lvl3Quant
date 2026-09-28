"""
Sector picker v5 — re-fits the v4 per-sector ridge pipeline on master_panel_v2.

What v2 panel adds vs the "with_text" panel v4 ran on:
  - fp_*  : 19 fundamentals features (revenue/margins/ROE/growth/etc., PIT)
  - sf_*  : 10 sector ETF flow features (DV Z-score, RS vs SPY, macro corrs)
  - ins_* : 8 full-universe Form 4 insider columns (replaces v1 smoke set)

Methodology is unchanged from v4 (ridge over 36mo train / 12mo OOT walk-forward,
21d hold long-short top3/bot3, 5bps turnover, pooled-OOT metrics) — only the
feature shelf changes. This isolates the question "does the richer panel
unstick the dead sectors and push pooled Calmar above 1.0?"

Output: research/findings/sector_picker_v5_master_v2/{report.json,report.md,per_day_pnl.parquet}
"""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path("/home/jupiter/Lvl3Quant")
sys.path.insert(0, str(ROOT / "strategy/macro_picker"))

import sector_picker_v4 as v4  # type: ignore

# ---- Patch the module globals BEFORE calling main() ----
v4.PANEL = ROOT / "data/feature_store/master_panel/master_panel_v2.parquet"
v4.OUT_DIR = ROOT / "research/findings/sector_picker_v5_master_v2"
v4.OUT_DIR.mkdir(parents=True, exist_ok=True)

# fp_* — direct from fundamentals_pit (already z-scored cross-sectionally per date
# inside v4._xs_zscore; raw values OK as input).
FUND_PIT = [
    "fp_revenue_ttm", "fp_fcf_ttm", "fp_ni_ttm", "fp_eps_ttm", "fp_ebitda_ttm",
    "fp_gross_margin", "fp_net_margin", "fp_ebitda_margin",
    "fp_roe", "fp_debt_to_equity", "fp_current_ratio",
    "fp_market_cap_pit", "fp_fcf_yield",
    "fp_rev_yoy_growth", "fp_ni_yoy_growth", "fp_eps_yoy_growth",
    "fp_fcf_yoy_growth", "fp_margin_trend_4q", "fp_beat_rate_4q",
]

# sf_* — per-ETF flow/regime features (broadcast across tickers in a sector)
SECTOR_FLOWS = [
    "sf_dv_z_20d", "sf_dv_chg_60d",
    "sf_rs_60d_spy", "sf_rs_252d_spy",
    "sf_px_200dma", "sf_vol_252d",
    "sf_corr_tlt_60d", "sf_corr_hyg_60d", "sf_corr_uup_60d", "sf_corr_gld_60d",
]

# Updated ins_* names (Form 4 v2 schema). The fallback inside v4.load_panel
# aliases ins_net_share_change <- ins_net_insider_usd so that column survives;
# we just add the new columns to the pool.
INSIDER_V2 = [
    "ins_net_insider_usd", "ins_gross_buy_usd", "ins_gross_sell_usd",
    "ins_n_buys", "ins_n_sells", "ins_n_buyers", "ins_n_sellers",
    "ins_mean_price",
]

EXTRA = FUND_PIT + SECTOR_FLOWS + INSIDER_V2
# de-dupe (some ins_* already in v4 base pool)
v4.FEATURE_POOL = list(dict.fromkeys(v4.FEATURE_POOL_BASE + v4.TEXT_FEATURES + EXTRA))

print(f"[v5] PANEL = {v4.PANEL.name}")
print(f"[v5] OUT_DIR = {v4.OUT_DIR}")
print(f"[v5] feature pool: {len(v4.FEATURE_POOL)} features "
      f"(base {len(v4.FEATURE_POOL_BASE)} + text {len(v4.TEXT_FEATURES)} + new {len(EXTRA)})")
print()

if __name__ == "__main__":
    v4.main()
