"""
slider.py — Continuous Aggressiveness Slider (1-10) for Wheel Strategy
======================================================================

Maps a single integer (1-10) to a fully interpolated wheel configuration.
The user just says "set it to 7" and gets the right delta, DTE, universe,
and risk controls automatically.

Slider levels:
    1   Ultra-Conservative  — Far OTM on megacaps, ~0.3-0.5%/wk target
    2   Conservative        — Safe blue-chips, ~0.5-0.8%/wk
    3   Moderate-Safe       — Broad large-cap, ~0.8-1.0%/wk
    4   Moderate            — Balanced risk/return, ~1.0-1.5%/wk
    5   Moderate-Growth     — Wider universe, ~1.5-2.0%/wk
    6   Growth              — Higher delta + IV names, ~2.0-2.5%/wk
    7   Aggressive-Growth   — Near-ATM on high-beta, ~2.5-3.5%/wk
    8   Aggressive          — ATM high-beta, weeklies, ~3.5-4.5%/wk
    9   Turbo               — Tight strikes, frequent assignment, ~4.5-5.5%/wk
   10   Max Aggression      — Maximum premium, highest risk, ~5.5%+/wk

Each level interpolates these parameters:
  - put_delta_target (how close to ATM)
  - call_delta_target (how close to ATM on CCs)
  - DTE range (monthly → weekly)
  - IV rank floor (how juicy the premium must be)
  - VIX gate (when to stop trading)
  - fund_score_floor (quality floor)
  - max concurrent positions
  - sector cap
  - profit take %
  - roll DTE trigger
  - universe composition (blue chips → high beta)

Author: Claude (2026-07-14, per user request for slider control)
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import List, Dict, Callable, Optional
import math

# Import engine config
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backtest.wheel_engine import WheelConfig


# ── Anchor points for interpolation ──
# Each key maps to (level_1_value, level_10_value) — linear interpolation between
PARAM_ANCHORS = {
    "put_delta_target":    (0.12, 0.45),     # 12-delta → 45-delta
    "call_delta_target":   (0.12, 0.40),     # conservative CC → aggressive CC
    "dte_min":             (30, 3),           # monthly → ~3 day
    "dte_max":             (50, 10),          # monthly → weekly
    "profit_take_pct":     (0.40, 0.65),      # quick exits → hold for more
    "roll_dte_trigger":    (12, 1),           # roll early → let expire (assignment)
    "iv_rank_floor":       (0.35, 0.15),      # only sell rich → sell anything
    "vix_max_gate":        (25.0, 45.0),      # strict → loose VIX gate
    "naaim_min_gate":      (-20.0, -80.0),    # needs bullish → trades in fear
    "fund_score_floor":    (60.0, 15.0),      # blue chip only → almost anything
    "max_concurrent":      (8, 35),           # few positions → many
    "sector_cap_pct":      (0.25, 0.40),      # diversified → concentrated OK
    "margin_cap":          (0.15, 0.50),      # conservative leverage → aggressive
    "per_name_pct":        (0.04, 0.015),     # big per-name → more spread out
}

# Universe blend: what % of the portfolio comes from high-beta names
# Level 1 = 0% high-beta (all blue chips), Level 10 = 80% high-beta
HIGHBETA_BLEND = {1: 0.0, 2: 0.0, 3: 0.05, 4: 0.10, 5: 0.20,
                  6: 0.35, 7: 0.50, 8: 0.65, 9: 0.75, 10: 0.80}

# Weekly return targets (for reporting/display only — NOT used as gates)
WEEKLY_TARGETS = {
    1: "0.3-0.5%", 2: "0.5-0.8%", 3: "0.8-1.0%", 4: "1.0-1.5%", 5: "1.5-2.0%",
    6: "2.0-2.5%", 7: "2.5-3.5%", 8: "3.5-4.5%", 9: "4.5-5.5%", 10: "5.5%+",
}

LEVEL_NAMES = {
    1: "Ultra-Conservative", 2: "Conservative", 3: "Moderate-Safe",
    4: "Moderate", 5: "Moderate-Growth", 6: "Growth",
    7: "Aggressive-Growth", 8: "Aggressive", 9: "Turbo", 10: "Max Aggression",
}


# ── Blue-chip and high-beta ticker sets ──
BLUE_CHIP_TICKERS = {
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "JPM", "BAC", "WFC",
    "JNJ", "PG", "KO", "PEP", "WMT", "COST", "HD", "MCD", "MRK",
    "ABBV", "LLY", "UNH", "CVX", "XOM", "VZ", "DIS", "NKE", "MA", "V",
    "BRK-B", "CRM", "ORCL", "CSCO", "INTC", "IBM", "CAT", "GE", "HON",
    "LOW", "TGT", "SBUX", "MMM", "AXP",
}

MID_BETA_TICKERS = {
    "NVDA", "AMD", "NFLX", "PYPL", "SQ", "UBER", "ABNB", "CRM",
    "NOW", "ADBE", "INTU", "ISRG", "PANW", "CRWD", "ZS", "SNPS",
    "CDNS", "KLAC", "LRCX", "AMAT", "MU", "QCOM", "AVGO", "TXN",
    "DE", "CMG", "LULU", "DASH", "ARM", "MELI",
}

HIGH_BETA_TICKERS = {
    "TSLA", "PLTR", "COIN", "MSTR", "SHOP", "SNAP", "ROKU",
    "RIOT", "MARA", "SOFI", "AFRM", "RBLX", "NET", "DDOG", "SNOW",
    "MDB", "OKTA", "U", "PATH", "DKNG", "CHWY", "HOOD", "UPST",
    "SMCI", "IONQ", "RIVN", "LCID", "NIO", "XPEV",
}


def _lerp(level: int, lo: float, hi: float) -> float:
    """Linear interpolation from level 1 (lo) to level 10 (hi)."""
    t = (level - 1) / 9.0
    return lo + t * (hi - lo)


def _lerp_int(level: int, lo: float, hi: float) -> int:
    return int(round(_lerp(level, lo, hi)))


@dataclass
class SliderConfig:
    """Complete wheel configuration generated from a single slider level."""
    level: int
    level_name: str
    weekly_target: str

    # Wheel engine params
    put_delta_target: float
    call_delta_target: float
    dte_min: int
    dte_max: int
    profit_take_pct: float
    roll_dte_trigger: int
    iv_rank_floor: float
    vix_max_gate: float
    naaim_min_gate: float
    fund_score_floor: float
    max_concurrent: int
    sector_cap_pct: float
    margin_cap: float
    per_name_pct: float

    # Universe composition
    blue_chip_pct: float     # % of universe that's blue chip
    mid_beta_pct: float      # % that's mid-beta
    high_beta_pct: float     # % that's high-beta

    def to_wheel_config(self) -> WheelConfig:
        """Convert to the engine's WheelConfig dataclass."""
        return WheelConfig(
            put_delta_target=self.put_delta_target,
            call_delta_target=self.call_delta_target,
            dte_min=self.dte_min,
            dte_max=self.dte_max,
            profit_take_pct=self.profit_take_pct,
            roll_dte_trigger=self.roll_dte_trigger,
            max_concurrent_names=self.max_concurrent,
            sector_cap_pct=self.sector_cap_pct,
            vix_max_gate=self.vix_max_gate,
            naaim_min_gate=self.naaim_min_gate,
            fund_score_floor=self.fund_score_floor,
        )

    def get_universe(self, available_tickers: set = None) -> List[str]:
        """Return the ticker universe for this slider level.

        If available_tickers is provided, only return tickers present in it.
        Otherwise return the full theoretical universe for this level.
        """
        # Build universe by blending the three pools
        hb = HIGHBETA_BLEND.get(self.level, 0.0)

        # Level 1-3: blue chips dominate
        # Level 4-6: mix in mid-beta
        # Level 7-10: heavy high-beta
        if self.level <= 3:
            pool = list(BLUE_CHIP_TICKERS)
            # sprinkle in some mid-beta at level 3
            if self.level >= 3:
                pool.extend(list(MID_BETA_TICKERS)[:10])
        elif self.level <= 6:
            pool = list(BLUE_CHIP_TICKERS) + list(MID_BETA_TICKERS)
            if self.level >= 5:
                pool.extend(list(HIGH_BETA_TICKERS)[:10])
        else:
            pool = list(BLUE_CHIP_TICKERS) + list(MID_BETA_TICKERS) + list(HIGH_BETA_TICKERS)

        if available_tickers:
            pool = [t for t in pool if t in available_tickers]

        return sorted(set(pool))

    def summary(self) -> str:
        """Human-readable summary of this slider level."""
        lines = [
            f"{'='*60}",
            f"  SLIDER LEVEL {self.level}: {self.level_name}",
            f"  Weekly Return Target: {self.weekly_target}",
            f"{'='*60}",
            f"  Put Delta:     {self.put_delta_target:.2f} ({int(self.put_delta_target*100)}-delta)",
            f"  Call Delta:    {self.call_delta_target:.2f} ({int(self.call_delta_target*100)}-delta)",
            f"  DTE Range:     {self.dte_min}-{self.dte_max} days",
            f"  Profit Take:   {self.profit_take_pct:.0%}",
            f"  Roll Trigger:  DTE ≤ {self.roll_dte_trigger}",
            f"  IV Rank Floor: {self.iv_rank_floor:.0%}",
            f"  VIX Gate:      ≤ {self.vix_max_gate:.0f}",
            f"  Quality Floor: {self.fund_score_floor:.0f}",
            f"  Max Positions: {self.max_concurrent}",
            f"  Sector Cap:    {self.sector_cap_pct:.0%}",
            f"  Margin Cap:    {self.margin_cap:.0%}",
            f"  Per-Name:      {self.per_name_pct:.1%}",
            f"  Universe:      ~{len(self.get_universe())} tickers",
            f"  High-Beta Mix: {HIGHBETA_BLEND.get(self.level, 0):.0%}",
        ]
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return asdict(self)


def slider(level: int) -> SliderConfig:
    """
    Generate a complete wheel configuration from a slider level (1-10).

    This is the main entry point. Just call:
        cfg = slider(7)
    """
    if level < 1 or level > 10:
        raise ValueError(f"Slider level must be 1-10, got {level}")

    hb = HIGHBETA_BLEND.get(level, 0.0)

    return SliderConfig(
        level=level,
        level_name=LEVEL_NAMES[level],
        weekly_target=WEEKLY_TARGETS[level],

        put_delta_target=round(_lerp(level, *PARAM_ANCHORS["put_delta_target"]), 3),
        call_delta_target=round(_lerp(level, *PARAM_ANCHORS["call_delta_target"]), 3),
        dte_min=_lerp_int(level, *PARAM_ANCHORS["dte_min"]),
        dte_max=_lerp_int(level, *PARAM_ANCHORS["dte_max"]),
        profit_take_pct=round(_lerp(level, *PARAM_ANCHORS["profit_take_pct"]), 3),
        roll_dte_trigger=_lerp_int(level, *PARAM_ANCHORS["roll_dte_trigger"]),
        iv_rank_floor=round(_lerp(level, *PARAM_ANCHORS["iv_rank_floor"]), 3),
        vix_max_gate=round(_lerp(level, *PARAM_ANCHORS["vix_max_gate"]), 1),
        naaim_min_gate=round(_lerp(level, *PARAM_ANCHORS["naaim_min_gate"]), 1),
        fund_score_floor=round(_lerp(level, *PARAM_ANCHORS["fund_score_floor"]), 1),
        max_concurrent=_lerp_int(level, *PARAM_ANCHORS["max_concurrent"]),
        sector_cap_pct=round(_lerp(level, *PARAM_ANCHORS["sector_cap_pct"]), 3),
        margin_cap=round(_lerp(level, *PARAM_ANCHORS["margin_cap"]), 3),
        per_name_pct=round(_lerp(level, *PARAM_ANCHORS["per_name_pct"]), 4),

        blue_chip_pct=round(1.0 - hb, 2),
        mid_beta_pct=round(min(hb, 0.30), 2),
        high_beta_pct=round(max(0, hb - 0.30), 2),
    )


def slider_ladder() -> List[SliderConfig]:
    """Return all 10 slider configs for comparison/backtesting."""
    return [slider(i) for i in range(1, 11)]


# ── CLI preview ──
if __name__ == "__main__":
    import sys as _sys
    if len(_sys.argv) > 1:
        lvl = int(_sys.argv[1])
        cfg = slider(lvl)
        print(cfg.summary())
        print(f"\nTicker universe ({len(cfg.get_universe())} names):")
        print(", ".join(cfg.get_universe()))
    else:
        print("WHEEL STRATEGY — AGGRESSIVENESS SLIDER (1-10)")
        print("=" * 60)
        for cfg in slider_ladder():
            print(f"\n  Level {cfg.level:2d} | {cfg.level_name:20s} | "
                  f"Delta {cfg.put_delta_target:.2f} | "
                  f"DTE {cfg.dte_min:2d}-{cfg.dte_max:2d} | "
                  f"Target {cfg.weekly_target:>8s}/wk | "
                  f"~{len(cfg.get_universe()):3d} tickers")
        print("\nUsage: python slider.py <level>  (e.g. python slider.py 7)")
