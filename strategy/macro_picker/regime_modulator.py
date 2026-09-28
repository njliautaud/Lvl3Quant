"""
Regime modulator (HC #561 R2).

Macro lenses (VIX, VIX3M, NAAIM, DXY, yield curve) are STATE VARIABLES.
They classify the current regime and emit a continuous `risk_dial` ∈ [0, 1]
that multiplies downstream position size. They are NOT per-name ranking
features — strict separation enforced here by exposing only `RegimeState`.

Inputs (read by helpers):
    wheel_strategy_v1/data/cache/macro.parquet         (vix, vix3m, dxy, naaim)
    wheel_strategy_v1/data/cache/macro_extra.parquet   (yc_2s10s)
    wheel_strategy_v1/data/cache/regime_overlay.parquet (optional pass-through)

Gates:
    vix_high            : VIX > 25  (severe > 35)
    vix_term_inverted   : VIX > VIX3M
    naaim_low           : NAAIM exposure < 30 (severe < 0)
    dxy_strong          : DXY 60d return z-score > +5% z (i.e. z > 1 over 252d)
    yc_inverted         : T10Y2Y < 0

Severity → dial:
    0 gates → risk_on_strong  → 1.00
    1 gate  → risk_on         → 0.85
    2 gates → neutral         → 0.65
    3 gates → risk_off        → 0.40
    4-5     → risk_off_severe → 0.15
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional
import numpy as np
import pandas as pd

ROOT = Path("/home/jupiter/Lvl3Quant")
CACHE = ROOT / "wheel_strategy_v1/data/cache"

_DIAL_BY_SEVERITY = {0: 1.00, 1: 0.85, 2: 0.65, 3: 0.40, 4: 0.15, 5: 0.15}
_STATE_BY_SEVERITY = {
    0: "risk_on_strong",
    1: "risk_on",
    2: "neutral",
    3: "risk_off",
    4: "risk_off_severe",
    5: "risk_off_severe",
}


@dataclass
class RegimeState:
    state: str
    risk_dial: float
    contributing_gates: Dict[str, bool] = field(default_factory=dict)
    severity: int = 0
    severe_flags: Dict[str, bool] = field(default_factory=dict)


def _load_macro_combined() -> pd.DataFrame:
    """Join macro + macro_extra + regime_overlay on date, sorted, ffilled."""
    m = pd.read_parquet(CACHE / "macro.parquet")
    me = pd.read_parquet(CACHE / "macro_extra.parquet")
    out = m.merge(me, on="date", how="outer")
    try:
        ro = pd.read_parquet(CACHE / "regime_overlay.parquet")
        out = out.merge(ro, on="date", how="left")
    except Exception:
        pass
    out = out.sort_values("date").reset_index(drop=True)
    # PIT-safe forward fill of slow series (NAAIM weekly, macro_extra monthly)
    for c in out.columns:
        if c == "date":
            continue
        out[c] = out[c].ffill()
    # 60d DXY return + 252d z
    if "dxy" in out.columns:
        out["dxy_ret_60d"] = out["dxy"].pct_change(60)
        m_ = out["dxy_ret_60d"].rolling(252, min_periods=60).mean()
        s_ = out["dxy_ret_60d"].rolling(252, min_periods=60).std()
        out["dxy_ret_60d_z"] = (out["dxy_ret_60d"] - m_) / s_
    return out


def _gates_from_row(row: pd.Series) -> tuple[Dict[str, bool], Dict[str, bool]]:
    vix = row.get("vix", np.nan)
    vix3m = row.get("vix3m", np.nan)
    naaim = row.get("naaim", np.nan)
    dxy_z = row.get("dxy_ret_60d_z", np.nan)
    yc = row.get("yc_2s10s", np.nan)

    vix_high = bool(np.isfinite(vix) and vix > 25)
    vix_severe = bool(np.isfinite(vix) and vix > 35)
    vix_term_inverted = bool(np.isfinite(vix) and np.isfinite(vix3m) and vix > vix3m)
    naaim_low = bool(np.isfinite(naaim) and naaim < 30)
    naaim_severe = bool(np.isfinite(naaim) and naaim < 0)
    dxy_strong = bool(np.isfinite(dxy_z) and dxy_z > 1.0)
    yc_inverted = bool(np.isfinite(yc) and yc < 0)

    gates = {
        "vix_high": vix_high,
        "vix_term_inverted": vix_term_inverted,
        "naaim_low": naaim_low,
        "dxy_strong": dxy_strong,
        "yc_inverted": yc_inverted,
    }
    severe = {
        "vix_severe": vix_severe,
        "naaim_severe": naaim_severe,
    }
    return gates, severe


def classify_regime(date, macro_df: Optional[pd.DataFrame] = None) -> RegimeState:
    """Classify regime for a given date. macro_df may be the combined frame
    from _load_macro_combined() or any frame with the required columns."""
    if macro_df is None:
        macro_df = _load_macro_combined()
    date = pd.Timestamp(date)
    df = macro_df.sort_values("date")
    # PIT: use most recent row with date <= target date
    sub = df[df["date"] <= date]
    if sub.empty:
        return RegimeState(state="neutral", risk_dial=0.65, contributing_gates={}, severity=0)
    row = sub.iloc[-1]
    gates, severe = _gates_from_row(row)
    severity = int(sum(gates.values()))
    # bump severity if a severe flag is on
    if severe.get("vix_severe") or severe.get("naaim_severe"):
        severity = min(severity + 1, 5)
    state = _STATE_BY_SEVERITY[severity]
    dial = _DIAL_BY_SEVERITY[severity]
    return RegimeState(
        state=state,
        risk_dial=dial,
        contributing_gates=gates,
        severity=severity,
        severe_flags=severe,
    )


def regime_series(macro_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Vectorized: classify every date in macro_df. Returns DataFrame keyed
    by date with columns: state, risk_dial, severity, gate_*, severe_*."""
    if macro_df is None:
        macro_df = _load_macro_combined()
    df = macro_df.sort_values("date").reset_index(drop=True)
    states, dials, sevs = [], [], []
    gate_cols = ["vix_high", "vix_term_inverted", "naaim_low", "dxy_strong", "yc_inverted"]
    sev_cols = ["vix_severe", "naaim_severe"]
    gate_arrs = {c: [] for c in gate_cols}
    sev_arrs = {c: [] for c in sev_cols}
    for _, row in df.iterrows():
        gates, severe = _gates_from_row(row)
        severity = int(sum(gates.values()))
        if severe.get("vix_severe") or severe.get("naaim_severe"):
            severity = min(severity + 1, 5)
        states.append(_STATE_BY_SEVERITY[severity])
        dials.append(_DIAL_BY_SEVERITY[severity])
        sevs.append(severity)
        for c in gate_cols:
            gate_arrs[c].append(gates[c])
        for c in sev_cols:
            sev_arrs[c].append(severe[c])
    out = pd.DataFrame({"date": df["date"], "state": states, "risk_dial": dials, "severity": sevs})
    for c in gate_cols:
        out[f"gate_{c}"] = gate_arrs[c]
    for c in sev_cols:
        out[f"severe_{c}"] = sev_arrs[c]
    return out


if __name__ == "__main__":
    macro = _load_macro_combined()
    print("macro combined shape:", macro.shape)
    test_dates = ["2024-06-14", "2020-03-16", "2023-09-05"]
    for d in test_dates:
        rs = classify_regime(d, macro)
        print(f"\n=== {d} ===")
        print(f"  state       : {rs.state}")
        print(f"  risk_dial   : {rs.risk_dial:.2f}")
        print(f"  severity    : {rs.severity}")
        print(f"  gates       : {rs.contributing_gates}")
        print(f"  severe      : {rs.severe_flags}")
    rs_series = regime_series(macro)
    print(f"\nregime_series shape: {rs_series.shape}")
    print("state distribution:")
    print(rs_series["state"].value_counts())
