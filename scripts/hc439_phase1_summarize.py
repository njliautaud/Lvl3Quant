#!/usr/bin/env python3
"""HC #439 Phase 1 summary builder — reads the CSV tables from
output/hc439_deep_mfe_mae/phase1/ and produces summary.md plus a compact
Discord-friendly text. No tabulate dependency."""
from pathlib import Path
import pandas as pd

LVL3 = Path("/home/jupiter/Lvl3Quant")
DIR = LVL3 / "output/hc439_deep_mfe_mae/phase1"

PASSIVE = 0.376
MARKET = 1.376


def fmt_table(df: pd.DataFrame, cols: list[str], floatfmt: str = ".3f") -> str:
    """Format a dataframe as a fixed-width text table (no tabulate)."""
    if len(df) == 0:
        return "(no rows)"
    # Compute column widths
    rows = [cols]
    for _, r in df.iterrows():
        row = []
        for c in cols:
            v = r[c]
            if isinstance(v, float):
                row.append(format(v, floatfmt))
            else:
                row.append(str(v))
        rows.append(row)
    widths = [max(len(r[i]) for r in rows) for i in range(len(cols))]
    out = []
    sep = " | "
    out.append(sep.join(rows[0][i].ljust(widths[i]) for i in range(len(cols))))
    out.append("-+-".join("-" * widths[i] for i in range(len(cols))))
    for r in rows[1:]:
        out.append(sep.join(r[i].ljust(widths[i]) for i in range(len(cols))))
    return "\n".join(out)


def top_by_passive(df: pd.DataFrame, min_n: int, k: int) -> pd.DataFrame:
    d = df[df["n_signals"] >= min_n].copy()
    d = d.sort_values("passive_net", ascending=False).head(k)
    return d


def main():
    overall = pd.read_csv(DIR / "overall_table.csv")
    by_vol = pd.read_csv(DIR / "by_vol_table.csv")
    by_evt = pd.read_csv(DIR / "by_evt_table.csv")
    by_buyaggr = pd.read_csv(DIR / "by_buyaggr_table.csv")
    by_spread = pd.read_csv(DIR / "by_spread_table.csv")
    by_tod = pd.read_csv(DIR / "by_tod_table.csv")
    by_dow = pd.read_csv(DIR / "by_dow_table.csv")

    lines = []
    lines.append("# HC #439 Phase 1 — MFE/MAE deep analysis")
    lines.append("")
    lines.append(f"Cost regime: passive_net = mean_label - {PASSIVE} ticks")
    lines.append(f"             market_net  = mean_label - {MARKET} ticks")
    lines.append("All values in ticks (1 tick = 0.25 ES points = $12.50).")
    lines.append("")

    cols_overall = ["side", "horizon", "band", "n_signals", "label_mean",
                    "passive_net", "market_net", "wr", "hit_1tk", "hit_2tk",
                    "mfe_mean", "mae_mean"]
    cols_slice = ["tag", "side", "horizon", "band", "n_signals", "label_mean",
                  "passive_net", "market_net", "wr", "hit_1tk", "mfe_mean",
                  "mae_mean"]

    # 1. OVERALL — top 20
    lines.append("## OVERALL — top 20 (side, horizon, band) by passive-net EV "
                 "(n>=500)")
    lines.append("")
    lines.append("```")
    lines.append(fmt_table(top_by_passive(overall, 500, 20), cols_overall))
    lines.append("```")
    lines.append("")

    # 2. Filter slice winners
    slice_pairs = [
        (by_vol,     "Volatility (filter_vol_500ev_tk) terciles"),
        (by_evt,     "Event-rate (filter_evt_per_sec_30s) terciles"),
        (by_buyaggr, "Buy-aggression (filter_buy_aggr_50) terciles"),
        (by_spread,  "Spread-proxy terciles"),
        (by_tod,     "Time-of-day buckets"),
        (by_dow,     "Day-of-week"),
    ]
    for t, label in slice_pairs:
        lines.append(f"## Top 15 — {label} (n>=200)")
        lines.append("")
        lines.append("```")
        lines.append(fmt_table(top_by_passive(t, 200, 15), cols_slice))
        lines.append("```")
        lines.append("")

    # 3. Cross-check: best market-tradable slice (market_net positive)
    all_slices = pd.concat([
        overall.assign(source="all"),
        by_vol.assign(source="vol"),
        by_evt.assign(source="evt"),
        by_buyaggr.assign(source="buyaggr"),
        by_spread.assign(source="spread"),
        by_tod.assign(source="tod"),
        by_dow.assign(source="dow"),
    ], ignore_index=True)
    market_positive = all_slices[(all_slices["market_net"] > 0)
                                 & (all_slices["n_signals"] >= 200)].copy()
    market_positive = market_positive.sort_values("market_net",
                                                  ascending=False).head(25)
    lines.append("## ALL SLICES with positive MARKET-order EV (n>=200)")
    lines.append("")
    lines.append("```")
    lines.append(fmt_table(
        market_positive[["source", "tag", "side", "horizon", "band",
                         "n_signals", "label_mean", "passive_net",
                         "market_net", "wr"]],
        ["source", "tag", "side", "horizon", "band", "n_signals", "label_mean",
         "passive_net", "market_net", "wr"]))
    lines.append("```")
    lines.append("")

    out = DIR / "summary.md"
    out.write_text("\n".join(lines))
    print(f"Wrote {out}")

    # Print Discord-friendly TL;DR
    print("\n=== TL;DR (top overall by passive_net, n>=500) ===")
    top = top_by_passive(overall, 500, 10)
    print(fmt_table(top, cols_overall))

    print("\n=== Best MARKET-tradable slices (market_net > 0, n>=200) ===")
    if len(market_positive) > 0:
        print(fmt_table(
            market_positive.head(10),
            ["source", "tag", "side", "horizon", "band", "n_signals",
             "label_mean", "passive_net", "market_net", "wr"]))
    else:
        print("(none — no slice has positive expected value with market "
              "orders)")


if __name__ == "__main__":
    main()
