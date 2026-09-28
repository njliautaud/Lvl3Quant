"""
chromosome.py — GA gene definitions and decode.

Gene order (all real-valued, decoded into typed config):
  0  put_delta_target            [0.10, 0.40]
  1  call_delta_target           [0.10, 0.40]
  2  dte_min                     [7,    21]   int
  3  dte_max                     [21,   45]   int  (clamped to >= dte_min)
  4  profit_take_pct             [0.25, 0.75]
  5  roll_dte_trigger            [3,    14]   int
  6  max_concurrent_names        [5,    30]   int
  7  sector_cap_pct              [0.15, 0.40]
  8  vix_max_gate                [15,   50]
  9  naaim_min_gate              [-100, +50]
 10  fund_score_floor            [0,    100]

HC #544 R6 — LEVERAGE / CASH-VS-LEVERAGE GENES (added 2026-06-05):
 11  leverage_cap                [1.00, 2.00]   max effective gross exposure (1.0 = unlevered, 2.0 = full margin)
 12  cash_floor_pct              [0.00, 0.50]   minimum % of NAV held in cash always
 13  lever_vix_below             [10,   30]     VIX threshold below which we may use leverage
 14  delever_dd_above_pct        [3,    15]     intraperiod drawdown % above which forced deleverage kicks in
 15  lever_spy_above_dma         [0,    1]      binary: require SPY>200DMA to lever (>= 0.5 = ON)
"""
from __future__ import annotations
from dataclasses import dataclass

GENE_BOUNDS = [
    (0.10, 0.40),     # 0  put_delta_target
    (0.10, 0.40),     # 1  call_delta_target
    (7,    21),       # 2  dte_min
    (21,   45),       # 3  dte_max
    (0.25, 0.75),     # 4  profit_take_pct
    (3,    14),       # 5  roll_dte_trigger
    (5,    30),       # 6  max_concurrent_names
    (0.15, 0.40),     # 7  sector_cap_pct
    (15.0, 50.0),     # 8  vix_max_gate
    (-100.0, 50.0),   # 9  naaim_min_gate
    (0.0,  100.0),    # 10 fund_score_floor
    # HC #544 R6 — leverage / cash-vs-leverage
    (1.00, 2.00),     # 11 leverage_cap
    (0.00, 0.50),     # 12 cash_floor_pct
    (10.0, 30.0),     # 13 lever_vix_below
    (3.0,  15.0),     # 14 delever_dd_above_pct
    (0.0,  1.0),      # 15 lever_spy_above_dma (binary)
]
GENE_NAMES = [
    "put_delta_target","call_delta_target","dte_min","dte_max",
    "profit_take_pct","roll_dte_trigger","max_concurrent_names",
    "sector_cap_pct","vix_max_gate","naaim_min_gate","fund_score_floor",
    # HC #544 R6
    "leverage_cap","cash_floor_pct","lever_vix_below",
    "delever_dd_above_pct","lever_spy_above_dma",
]
N_GENES = len(GENE_BOUNDS)


def decode(genes) -> dict:
    g = list(genes)
    # Backfill defaults if a legacy 11-gene chromosome is passed in
    while len(g) < N_GENES:
        # Use lower bound as default — equivalent to leverage_cap=1.0 (unlevered)
        # and cash_floor=0 — i.e. original pre-HC#544 behaviour.
        g.append(GENE_BOUNDS[len(g)][0])

    cfg = {
        "put_delta_target":     float(_clip(g[0],  *GENE_BOUNDS[0])),
        "call_delta_target":    float(_clip(g[1],  *GENE_BOUNDS[1])),
        "dte_min":              int(round(_clip(g[2],  *GENE_BOUNDS[2]))),
        "dte_max":              int(round(_clip(g[3],  *GENE_BOUNDS[3]))),
        "profit_take_pct":      float(_clip(g[4],  *GENE_BOUNDS[4])),
        "roll_dte_trigger":     int(round(_clip(g[5],  *GENE_BOUNDS[5]))),
        "max_concurrent_names": int(round(_clip(g[6],  *GENE_BOUNDS[6]))),
        "sector_cap_pct":       float(_clip(g[7],  *GENE_BOUNDS[7])),
        "vix_max_gate":         float(_clip(g[8],  *GENE_BOUNDS[8])),
        "naaim_min_gate":       float(_clip(g[9],  *GENE_BOUNDS[9])),
        "fund_score_floor":     float(_clip(g[10], *GENE_BOUNDS[10])),
        # HC #544 R6 — leverage / cash-vs-leverage
        "leverage_cap":         float(_clip(g[11], *GENE_BOUNDS[11])),
        "cash_floor_pct":       float(_clip(g[12], *GENE_BOUNDS[12])),
        "lever_vix_below":      float(_clip(g[13], *GENE_BOUNDS[13])),
        "delever_dd_above_pct": float(_clip(g[14], *GENE_BOUNDS[14])),
        "lever_spy_above_dma":  bool(_clip(g[15], *GENE_BOUNDS[15]) >= 0.5),
    }
    if cfg["dte_max"] < cfg["dte_min"] + 5:
        cfg["dte_max"] = cfg["dte_min"] + 5
    return cfg


def regime_leverage_decision(
    cfg: dict,
    vix: float,
    spy_above_200dma: bool,
    current_drawdown_pct: float,
) -> float:
    """
    HC #544 R6 — Regime-conditional leverage decision ("raise cash vs raise leverage").

    Returns the GROSS-EXPOSURE TARGET in [cash_floor_complement, leverage_cap].

    Rules (composed):
      1. If current_drawdown_pct >= delever_dd_above_pct  → force 1.0 - cash_floor_pct (raise cash).
      2. Else if VIX > vix_max_gate                       → 1.0 (no leverage, no forced cash).
      3. Else if VIX < lever_vix_below
            AND (NOT lever_spy_above_dma OR spy_above_200dma) → leverage_cap (lever up).
      4. Else                                             → 1.0 (neutral, fully invested, unlevered).

    The cash floor is ALWAYS respected — gross exposure ≤ 1.0 - cash_floor_pct
    is never less than what the GA wants because raising cash IS lowering exposure.
    """
    base_unlevered_target = 1.0 - cfg["cash_floor_pct"]

    if current_drawdown_pct >= cfg["delever_dd_above_pct"]:
        # Risk-off: raise cash, exposure goes to the cash-floor-complement
        return base_unlevered_target * 0.5  # halve exposure in stress

    if vix > cfg["vix_max_gate"]:
        return base_unlevered_target

    lever_ok_on_trend = (not cfg["lever_spy_above_dma"]) or spy_above_200dma
    if vix < cfg["lever_vix_below"] and lever_ok_on_trend:
        return cfg["leverage_cap"]

    return base_unlevered_target


def _clip(x, lo, hi):
    return max(lo, min(hi, x))


if __name__ == "__main__":
    # Self-test
    sample_genes = [0.25, 0.20, 14, 35, 0.50, 7, 15, 0.25, 30.0, -20.0, 50.0,
                    1.5, 0.20, 18.0, 8.0, 1.0]
    cfg = decode(sample_genes)
    print("Decoded config:")
    for k, v in cfg.items():
        print(f"  {k:24s} = {v}")
    print("\nRegime decisions (gross-exposure target):")
    for vix, dd, spy in [(12, 0, True), (12, 0, False), (35, 0, True),
                        (15, 10, True), (20, 0, True), (25, 4, True)]:
        tgt = regime_leverage_decision(cfg, vix, spy, dd)
        print(f"  VIX={vix:5.1f}  DD={dd:4.1f}%  SPY>200d={spy!s:5s}  → target_exposure={tgt:.2f}")
