"""
render_report.py — Render per-sector GA formula JSON outputs into a clean markdown.

Per HC #560 R4: report the STUDY findings (which features matter, regime
behavior), not just strategy P&L.

Usage:
    cd /home/jupiter/Lvl3Quant/wheel_strategy_v1
    python3 -m strategy.macro_picker.render_report --tag v1
    python3 -m strategy.macro_picker.render_report --tag v2
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from collections import Counter

import pandas as pd

ROOT = Path("/home/jupiter/Lvl3Quant/wheel_strategy_v1")
FORMULAS = ROOT / "strategy" / "macro_picker" / "formulas"
FINDINGS = Path("/home/jupiter/Lvl3Quant/research/findings")
FINDINGS.mkdir(parents=True, exist_ok=True)


def fmt_pct(x):
    if x is None: return "—"
    return f"{x*100:+.1f}%"

def fmt_num(x, digits=2):
    if x is None: return "—"
    return f"{x:+.{digits}f}"

def render(tag: str) -> str:
    files = sorted(glob.glob(str(FORMULAS / f"formula_{tag}_*.json")))
    if not files:
        return f"No formulas found for tag={tag}"
    rows = []
    feature_counter = Counter()
    feature_signed = []
    for fp in files:
        with open(fp) as f:
            art = json.load(f)
        sector = art["sector"]
        ft = art.get("fitness_train") or art.get("fitness") or {}
        fo = art.get("fitness_oot") or {}
        rows.append({
            "sector": sector,
            "expr": art["expression"],
            "n_tickers": art["trained_on"]["n_tickers"],
            "train_sharpe": ft.get("sharpe"),
            "train_calmar": ft.get("calmar"),
            "train_cagr": ft.get("cagr"),
            "train_maxdd": ft.get("maxdd"),
            "train_green_S": ft.get("green_sharpe"),
            "train_red_S": ft.get("red_sharpe"),
            "oot_sharpe": fo.get("sharpe"),
            "oot_calmar": fo.get("calmar"),
            "oot_cagr": fo.get("cagr"),
            "oot_maxdd": fo.get("maxdd"),
            "oot_green_S": fo.get("green_sharpe"),
            "oot_red_S": fo.get("red_sharpe"),
            "deployable": art.get("deployable", False),
        })
        # Tally features used (abs weight > 0.05)
        feats = art["features"]
        weights = art["weights"]
        for f_, w_ in zip(feats, weights):
            if abs(w_) >= 0.05:
                feature_counter[f_] += 1
                feature_signed.append((f_, w_, sector))

    md = [
        f"# Per-Sector GA Formula Report — tag={tag}",
        f"_Generated {pd.Timestamp.now().isoformat()}_",
        "",
        "## Headline (HC #559 verdict)",
        "",
    ]
    deployable = [r for r in rows if r["deployable"]]
    pass_train = [r for r in rows if r["train_calmar"] is not None and r["train_calmar"] >= 1.0]
    pass_oot = [r for r in rows if r["oot_calmar"] is not None and r["oot_calmar"] >= 1.0]
    md.append(f"- Sectors fit: **{len(rows)}**")
    md.append(f"- Pass Calmar≥1.0 in-sample: **{len(pass_train)}** ({', '.join(r['sector'] for r in pass_train) or '—'})")
    md.append(f"- Pass Calmar≥1.0 OOT     : **{len(pass_oot)}** ({', '.join(r['sector'] for r in pass_oot) or '—'})")
    md.append(f"- Deployable (Calmar≥1 OOT + regime-stable OOT + Sharpe>0.7): **{len(deployable)}** ({', '.join(r['sector'] for r in deployable) or '—'})")
    md.append("")
    md.append("## Per-sector summary")
    md.append("")
    md.append("| Sector | N | Train Sh / Calmar / CAGR / DD | OOT Sh / Calmar / CAGR / DD | Regime (G/R) OOT | Verdict |")
    md.append("|---|---:|---|---|---|---|")
    for r in rows:
        verdict = "✅ DEPLOY" if r["deployable"] else ("⚠ IS-only" if (r["train_calmar"] or 0) >= 1.0 and (r["oot_calmar"] is None) else "❌")
        ts = f"{fmt_num(r['train_sharpe'])} / {fmt_num(r['train_calmar'])} / {fmt_pct(r['train_cagr'])} / {fmt_pct(r['train_maxdd'])}"
        os_ = f"{fmt_num(r['oot_sharpe'])} / {fmt_num(r['oot_calmar'])} / {fmt_pct(r['oot_cagr'])} / {fmt_pct(r['oot_maxdd'])}"
        rg = f"{fmt_num(r['oot_green_S'])} / {fmt_num(r['oot_red_S'])}" if r["oot_green_S"] is not None else "—"
        md.append(f"| {r['sector']} | {r['n_tickers']} | {ts} | {os_} | {rg} | {verdict} |")

    md.append("")
    md.append("## Feature consensus (which signals matter across sectors)")
    md.append("")
    md.append("| Feature | # sectors using | avg |w| | sign consensus |")
    md.append("|---|---:|---:|---|")
    by_feat = {}
    for f_, w_, s_ in feature_signed:
        by_feat.setdefault(f_, []).append(w_)
    for f_ in sorted(feature_counter, key=lambda x: -feature_counter[x]):
        ws = by_feat[f_]
        avg_abs = sum(abs(w) for w in ws) / len(ws)
        pos = sum(1 for w in ws if w > 0)
        neg = sum(1 for w in ws if w < 0)
        sign = "+" if pos > neg else ("-" if neg > pos else "±")
        md.append(f"| {f_} | {feature_counter[f_]} | {avg_abs:.2f} | {sign} ({pos}+/{neg}-) |")
    md.append("")

    md.append("## Formulas (top-weighted terms shown)")
    md.append("")
    for r in rows:
        verdict = "DEPLOY" if r["deployable"] else "IS-only" if (r["train_calmar"] or 0) >= 1.0 else "FAIL"
        md.append(f"### {r['sector']} [{verdict}]")
        md.append(f"`{r['expr']}`")
        md.append("")

    out_md = FINDINGS / f"ga_per_sector_formula_{tag}.md"
    with open(out_md, "w") as f:
        f.write("\n".join(md))
    return str(out_md)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="v1")
    args = ap.parse_args()
    path = render(args.tag)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
