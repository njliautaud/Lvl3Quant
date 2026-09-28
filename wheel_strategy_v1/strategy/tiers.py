"""
tiers.py — Five explicit wheel-strategy tiers per HC #556 R1.

Each tier is a calibrated WheelConfig + a universe filter recipe, targeting
progressively higher net-premium yield and risk profile:

    Tier 1  CONSERVATIVE   ~ 8-12%  annualized  (blue chips, 15-20Δ, IV-rank > 30)
    Tier 2  BALANCED       ~12-18%  annualized  (broad, 20-25Δ, IV-rank > 25)
    Tier 3  INCOME         ~18-25%  annualized  (higher IV, 25-30Δ, IV-rank > 40)
    Tier 4  AGGRESSIVE     ~25-40%  annualized  (high β + high IV, 30-40Δ, IV-rank > 50)
    Tier 5  TURBO          ~40%+    annualized  (weeklies on liquid high-IV, 35-45Δ, IV-rank > 60)

Each TierSpec ships:
  - wheel_cfg : a WheelConfig (gene-equivalent dict) consumed by backtest.wheel_engine.run_wheel
  - universe_filter : callable (universe_df, fundamentals_df, iv_features_df, prices_df) -> List[ticker]
  - iv_rank_floor : minimum IV rank required at entry (gate inside run_wheel candidate selection)
  - regime_sensitivity : 'low' | 'med' | 'high' — how much HC #555 regime overlay scales down size
  - target_annual_yield_pct : numeric goal used in reporting (NOT used to gate the GA)

These tiers are deployable presets — they ARE the strategy ladder the user dials.
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Callable, Optional, List
import pandas as pd

# Import the engine's config dataclass so a TierSpec converts cleanly.
from backtest.wheel_engine import WheelConfig


# ---------- universe filter helpers ----------

# Curated "blue chip" anchors — large-cap, low-β, dividend-payers, deep liquidity.
BLUE_CHIP_TICKERS = {
    "AAPL", "MSFT", "GOOGL", "GOOG", "AMZN", "META", "JPM", "BAC", "WFC",
    "JNJ", "PG", "KO", "PEP", "WMT", "COST", "HD", "MCD", "MRK", "PFE",
    "ABBV", "LLY", "UNH", "CVX", "XOM", "VZ", "T", "DIS", "NKE", "MA", "V",
    "SPY", "QQQ", "IWM", "DIA", "VTI", "VOO", "IVV",
}

# High-β / high-IV names typically suitable for Aggressive / Turbo tiers.
HIGH_VOL_TICKERS = {
    "TSLA", "NVDA", "AMD", "PLTR", "COIN", "MSTR", "SHOP", "SNAP", "ROKU",
    "RIOT", "MARA", "SOFI", "AFRM", "RBLX", "NET", "DDOG", "SNOW", "CRWD",
    "PANW", "ZS", "MDB", "OKTA", "U", "PATH", "DKNG", "CHWY",
}


def _has_iv_history(iv_df: pd.DataFrame, ticker: str, min_obs: int = 250) -> bool:
    """Tickers must have at least ~1 year of IV history to be considered."""
    if iv_df is None or iv_df.empty:
        return False
    s = iv_df.loc[iv_df["ticker"] == ticker, "sigma"]
    return s.notna().sum() >= min_obs


def _ticker_avg_iv(iv_df: pd.DataFrame, ticker: str) -> float:
    if iv_df is None or iv_df.empty:
        return float("nan")
    s = iv_df.loc[iv_df["ticker"] == ticker, "sigma"]
    s = s.dropna()
    return float(s.mean()) if len(s) else float("nan")


# ---------- universe filter functions ----------

def filter_conservative(universe: pd.DataFrame,
                        fundamentals: pd.DataFrame,
                        iv_features: pd.DataFrame,
                        prices: pd.DataFrame) -> List[str]:
    """Blue chips with high fund_score, low avg IV, dividend payers preferred."""
    pool = set(universe["ticker"]) & BLUE_CHIP_TICKERS
    fs = fundamentals.set_index("ticker")
    keep = []
    for tk in pool:
        if tk not in fs.index:
            continue
        if not _has_iv_history(iv_features, tk):
            continue
        if fs.at[tk, "fund_score"] < 55.0:
            continue
        keep.append(tk)
    return sorted(keep)


def filter_balanced(universe: pd.DataFrame,
                    fundamentals: pd.DataFrame,
                    iv_features: pd.DataFrame,
                    prices: pd.DataFrame) -> List[str]:
    """Broad large-cap universe with decent fund_score."""
    fs = fundamentals.set_index("ticker")
    keep = []
    for tk in universe["ticker"]:
        if tk not in fs.index:
            continue
        if not _has_iv_history(iv_features, tk):
            continue
        if fs.at[tk, "fund_score"] < 45.0:
            continue
        keep.append(tk)
    return sorted(keep)


def filter_income(universe: pd.DataFrame,
                  fundamentals: pd.DataFrame,
                  iv_features: pd.DataFrame,
                  prices: pd.DataFrame) -> List[str]:
    """Higher-IV names — top half of universe by avg sigma."""
    fs = fundamentals.set_index("ticker")
    cand = []
    for tk in universe["ticker"]:
        if tk not in fs.index:
            continue
        if not _has_iv_history(iv_features, tk):
            continue
        if fs.at[tk, "fund_score"] < 40.0:
            continue
        avg_iv = _ticker_avg_iv(iv_features, tk)
        if avg_iv != avg_iv:  # NaN
            continue
        cand.append((tk, avg_iv))
    cand.sort(key=lambda r: r[1], reverse=True)
    cut = max(20, len(cand) // 2)
    return sorted([tk for tk, _ in cand[:cut]])


def filter_aggressive(universe: pd.DataFrame,
                      fundamentals: pd.DataFrame,
                      iv_features: pd.DataFrame,
                      prices: pd.DataFrame) -> List[str]:
    """High-β + high-IV names. Top quintile by avg sigma + HIGH_VOL anchors."""
    fs = fundamentals.set_index("ticker")
    cand = []
    for tk in universe["ticker"]:
        if tk not in fs.index:
            continue
        if not _has_iv_history(iv_features, tk):
            continue
        if fs.at[tk, "fund_score"] < 30.0:
            continue
        avg_iv = _ticker_avg_iv(iv_features, tk)
        if avg_iv != avg_iv:
            continue
        cand.append((tk, avg_iv))
    cand.sort(key=lambda r: r[1], reverse=True)
    cut = max(15, len(cand) // 5)
    top = {tk for tk, _ in cand[:cut]}
    top |= (HIGH_VOL_TICKERS & set(universe["ticker"]))
    return sorted(top)


def filter_turbo(universe: pd.DataFrame,
                 fundamentals: pd.DataFrame,
                 iv_features: pd.DataFrame,
                 prices: pd.DataFrame) -> List[str]:
    """Liquid high-IV names — Turbo. Hand-anchored to HIGH_VOL_TICKERS + top decile by IV."""
    fs = fundamentals.set_index("ticker")
    cand = []
    for tk in universe["ticker"]:
        if tk not in fs.index:
            continue
        if not _has_iv_history(iv_features, tk):
            continue
        avg_iv = _ticker_avg_iv(iv_features, tk)
        if avg_iv != avg_iv:
            continue
        cand.append((tk, avg_iv))
    cand.sort(key=lambda r: r[1], reverse=True)
    cut = max(10, len(cand) // 10)
    top = {tk for tk, _ in cand[:cut]}
    top |= (HIGH_VOL_TICKERS & set(universe["ticker"]))
    return sorted(top)


# ---------- tier specs ----------

@dataclass
class TierSpec:
    name: str
    target_annual_yield_pct: float        # narrative target, NOT enforced
    description: str
    wheel_cfg: WheelConfig
    universe_filter: Callable
    iv_rank_floor: float                  # 0..1
    regime_sensitivity: str               # 'low' | 'med' | 'high'
    capital_share_default: float          # suggested portfolio weight if user runs the ladder together

    def to_dict(self) -> dict:
        d = asdict(self)
        # callables are not serialisable
        d["universe_filter"] = self.universe_filter.__name__
        d["wheel_cfg"] = asdict(self.wheel_cfg)
        return d


def conservative_tier() -> TierSpec:
    cfg = WheelConfig(
        put_delta_target=0.17,
        call_delta_target=0.17,
        dte_min=30, dte_max=45,
        profit_take_pct=0.50,
        roll_dte_trigger=10,
        max_concurrent_names=12,
        sector_cap_pct=0.30,
        vix_max_gate=28.0,
        naaim_min_gate=-30.0,
        fund_score_floor=55.0,
    )
    return TierSpec(
        name="Tier1_Conservative",
        target_annual_yield_pct=10.0,
        description="Blue chips, 17Δ puts, 30-45 DTE, IV-rank > 30. Capital-preservation first.",
        wheel_cfg=cfg,
        universe_filter=filter_conservative,
        iv_rank_floor=0.30,
        regime_sensitivity="low",
        capital_share_default=0.35,
    )


def balanced_tier() -> TierSpec:
    cfg = WheelConfig(
        put_delta_target=0.22,
        call_delta_target=0.22,
        dte_min=30, dte_max=45,
        profit_take_pct=0.50,
        roll_dte_trigger=10,
        max_concurrent_names=18,
        sector_cap_pct=0.25,
        vix_max_gate=32.0,
        naaim_min_gate=-50.0,
        fund_score_floor=45.0,
    )
    return TierSpec(
        name="Tier2_Balanced",
        target_annual_yield_pct=15.0,
        description="Broad large-cap, 22Δ puts, 30-45 DTE, IV-rank > 25. Balanced yield/risk.",
        wheel_cfg=cfg,
        universe_filter=filter_balanced,
        iv_rank_floor=0.25,
        regime_sensitivity="med",
        capital_share_default=0.30,
    )


def income_tier() -> TierSpec:
    cfg = WheelConfig(
        put_delta_target=0.27,
        call_delta_target=0.25,
        dte_min=21, dte_max=35,
        profit_take_pct=0.50,
        roll_dte_trigger=7,
        max_concurrent_names=20,
        sector_cap_pct=0.25,
        vix_max_gate=35.0,
        naaim_min_gate=-60.0,
        fund_score_floor=40.0,
    )
    return TierSpec(
        name="Tier3_Income",
        target_annual_yield_pct=22.0,
        description="Higher-IV half of universe, 27Δ puts, 21-35 DTE, IV-rank > 40. Income focus.",
        wheel_cfg=cfg,
        universe_filter=filter_income,
        iv_rank_floor=0.40,
        regime_sensitivity="med",
        capital_share_default=0.20,
    )


def aggressive_tier() -> TierSpec:
    cfg = WheelConfig(
        put_delta_target=0.34,
        call_delta_target=0.30,
        dte_min=14, dte_max=28,
        profit_take_pct=0.50,
        roll_dte_trigger=5,
        max_concurrent_names=15,
        sector_cap_pct=0.30,
        vix_max_gate=38.0,
        naaim_min_gate=-70.0,
        fund_score_floor=30.0,
    )
    return TierSpec(
        name="Tier4_Aggressive",
        target_annual_yield_pct=32.0,
        description="High-β + high-IV, 34Δ puts, 14-28 DTE, IV-rank > 50. Aggressive premium.",
        wheel_cfg=cfg,
        universe_filter=filter_aggressive,
        iv_rank_floor=0.50,
        regime_sensitivity="high",
        capital_share_default=0.10,
    )


def turbo_tier() -> TierSpec:
    cfg = WheelConfig(
        put_delta_target=0.40,
        call_delta_target=0.35,
        dte_min=7, dte_max=14,
        profit_take_pct=0.50,
        roll_dte_trigger=3,
        max_concurrent_names=10,
        sector_cap_pct=0.35,
        vix_max_gate=30.0,        # tighter — Turbo gates harder on stress
        naaim_min_gate=-30.0,     # tighter — needs risk-on tape
        fund_score_floor=25.0,
    )
    return TierSpec(
        name="Tier5_Turbo",
        target_annual_yield_pct=45.0,
        description="Weeklies on liquid high-IV, 40Δ puts, 7-14 DTE, IV-rank > 60. Turbo, regime-gated.",
        wheel_cfg=cfg,
        universe_filter=filter_turbo,
        iv_rank_floor=0.60,
        regime_sensitivity="high",
        capital_share_default=0.05,
    )


# Canonical ordered ladder
def all_tiers() -> List[TierSpec]:
    return [
        conservative_tier(),
        balanced_tier(),
        income_tier(),
        aggressive_tier(),
        turbo_tier(),
    ]


# ---------- assignment-allowed variants ----------
#
# The default ladder above uses roll_dte_trigger 3..10 — meaning positions
# usually close at DTE = trigger rather than going to expiry. In practice
# this is "CSP scalping" — high WR, low assignment rate. To get TRUE WHEEL
# behaviour (sell CSP, take assignment, sell CC, repeat), we need to allow
# expiry. The variants below set roll_dte_trigger = 1 and profit_take_pct
# higher (0.65) so we hold longer for premium and accept assignment.
#
# Use --full-wheel on tier_runner to run these instead.

def _full_wheel(spec: TierSpec) -> TierSpec:
    """Convert a scalp-mode TierSpec into an assignment-allowed full wheel."""
    cfg = spec.wheel_cfg
    new_cfg = WheelConfig(
        put_delta_target=cfg.put_delta_target,
        call_delta_target=cfg.call_delta_target,
        dte_min=cfg.dte_min, dte_max=cfg.dte_max,
        profit_take_pct=0.65,
        roll_dte_trigger=1,
        max_concurrent_names=cfg.max_concurrent_names,
        sector_cap_pct=cfg.sector_cap_pct,
        vix_max_gate=cfg.vix_max_gate,
        naaim_min_gate=cfg.naaim_min_gate,
        fund_score_floor=cfg.fund_score_floor,
        r=cfg.r,
    )
    return TierSpec(
        name=spec.name + "_FW",
        target_annual_yield_pct=spec.target_annual_yield_pct,
        description=spec.description + " [FULL WHEEL: profit-take 0.65, roll DTE 1, allows assignment]",
        wheel_cfg=new_cfg,
        universe_filter=spec.universe_filter,
        iv_rank_floor=spec.iv_rank_floor,
        regime_sensitivity=spec.regime_sensitivity,
        capital_share_default=spec.capital_share_default,
    )


def all_tiers_full_wheel() -> List[TierSpec]:
    return [_full_wheel(t) for t in all_tiers()]


if __name__ == "__main__":
    import json
    for t in all_tiers():
        print(f"\n=== {t.name} (target {t.target_annual_yield_pct:.1f}% annualized) ===")
        print(f"  desc: {t.description}")
        print(f"  iv_rank_floor: {t.iv_rank_floor}")
        print(f"  regime_sensitivity: {t.regime_sensitivity}")
        print(f"  capital_share_default: {t.capital_share_default}")
        print(f"  wheel_cfg:")
        for k, v in asdict(t.wheel_cfg).items():
            print(f"    {k:24s} = {v}")
