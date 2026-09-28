"""
HC #358 part (g) — Cross-model STRONGEST-VERDICT using ADVANCED methods.

Methods:
  1. Per-band cumulative-edge (Lorenz-like) curves.
  2. First-order stochastic dominance (FOSD) on per-trade net-ticks.
  3. Edge-decay area-under-curve (AUC over hold-time).
  4. Head-importance ablation (uses Δ-Sharpe from meta_mlp_results.json).
  5. Realized PnL under each model's optimal strategy (from optuna_best.json).

Output:
  output/v_test_bench_20260514/CROSS_MODEL_VERDICT.md
  output/v_test_bench_20260514/CROSS_MODEL_DOMINANCE.png

Per HC #353 — if no clear dominance OR if newer models trail v2 at matched
bands, flag the failure mode explicitly. Do NOT declare a winner without naming
the dominance method that decided it.

Per HC #350 — final line:
  "STRONGEST MODEL FOR EXECUTION: [model] because [method-cited 1-sentence reason]"
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path("/home/jupiter/Lvl3Quant")
sys.path.insert(0, str(REPO))

from scripts.v3_3_research.full_market_replay import (  # noqa: E402
    TradeConfig,
    full_market_replay,
)


def load_per_trade_ledgers(bench_root: Path, models: list[str],
                            labels_dir: Path) -> dict:
    """Re-run full_market_replay with each model's recommended config to get
    the per-trade dataframes (for FOSD + Lorenz curves + bottom-line PnL)."""
    out = {}
    for m in models:
        mdir = bench_root / m
        cfg_path = mdir / "recommended_config.json"
        if not cfg_path.exists():
            print(f"[xmodel] skip {m}: no recommended_config.json")
            continue
        cfg_json = json.loads(cfg_path.read_text())
        # Find NPZ path used for that model
        npz_glob = list((mdir / "_tmp_adapter").glob("*.npz")) if m == "v2" else []
        if m == "v2" and npz_glob:
            npz_path = npz_glob[0]
            dates = ["20260224"]
        elif m == "v3_2":
            npz_path = REPO / "output" / "v3_2_deep_sim_20260512" / "fold_00_oot_predictions.npz"
            d = np.load(npz_path, allow_pickle=True)
            dates = [str(x) for x in d["oot_dates"]]
        elif m == "v3_3":
            npz_path = REPO / "output" / "v3_3_oot_20260223" / "predictions.npz"
            if not npz_path.exists():
                print(f"[xmodel] v3_3 NPZ missing — skip")
                continue
            dates = ["20260223"]
        else:
            continue

        # Reconstruct TradeConfig from recommended_config.json
        order = "passive_at_touch"
        if cfg_json["passive_offset_ticks"] == 1:
            order = "passive_at_touch_plus_1"
        elif cfg_json["passive_offset_ticks"] == 2:
            order = "passive_at_touch_plus_2"
        elif cfg_json["passive_offset_ticks"] == -1:
            order = "ioc_market"
        cfg = TradeConfig(
            side=cfg_json["side_bias"],
            horizon=cfg_json["cnn_horizon"],
            confidence_threshold=cfg_json["cnn_percentile_tail"],
            order_type=order,
            cancel_eval_window=cfg_json["cancel_eval_window"],
            hold_seconds=cfg_json["max_hold_seconds"],
        )
        try:
            led = full_market_replay(npz_path, labels_dir, cfg, dates=dates)
            out[m] = {"config": cfg, "ledger": led, "cfg_json": cfg_json}
        except Exception as e:
            print(f"[xmodel] {m} replay failed: {e}")
    return out


def cumulative_edge_curves(ledgers: dict, out_path: Path) -> dict:
    """Sort filled trades by confidence rank, plot cumulative net-ticks."""
    fig, ax = plt.subplots(figsize=(10, 6))
    res = {}
    for m, info in ledgers.items():
        df = info["ledger"].per_trade_df
        filled = df[df["filled"]].copy()
        if filled.empty:
            continue
        # Sort by |prediction| descending (most-confident first)
        filled["abs_pred"] = filled["prediction"].abs()
        filled = filled.sort_values("abs_pred", ascending=False)
        cum = filled["net_ticks"].cumsum().values
        x = np.arange(1, len(cum) + 1)
        ax.plot(x, cum, label=f"{m} (n_filled={len(filled)})", linewidth=2)
        res[m] = {"n_filled": int(len(filled)), "cum_ticks_final": float(cum[-1]) if len(cum) else 0.0}
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("# of fills (sorted by |pred| desc)")
    ax.set_ylabel("Cumulative net ticks")
    ax.set_title("Lorenz-like cumulative-edge curves (HC #358g.1)")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=110)
    plt.close()
    return res


def fosd_test(ledgers: dict) -> dict:
    """First-order stochastic dominance: does ECDF_A lie below ECDF_B everywhere?
    If yes, A FOSD B → A's distribution shifts mass to higher net-ticks."""
    res = {}
    arrs = {}
    for m, info in ledgers.items():
        df = info["ledger"].per_trade_df
        filled = df[df["filled"]]
        if filled.empty:
            continue
        arrs[m] = filled["net_ticks"].dropna().values

    keys = list(arrs.keys())
    for i, ma in enumerate(keys):
        for mb in keys[i + 1:]:
            a, b = arrs[ma], arrs[mb]
            if a.size < 30 or b.size < 30:
                res[f"{ma}_vs_{mb}"] = {"verdict": "insufficient_data"}
                continue
            # Build common grid
            grid = np.linspace(min(a.min(), b.min()), max(a.max(), b.max()), 200)
            ecdf_a = np.array([(a <= g).mean() for g in grid])
            ecdf_b = np.array([(b <= g).mean() for g in grid])
            a_dom_b = bool((ecdf_a <= ecdf_b + 1e-9).all() and (ecdf_a < ecdf_b - 1e-3).any())
            b_dom_a = bool((ecdf_b <= ecdf_a + 1e-9).all() and (ecdf_b < ecdf_a - 1e-3).any())
            if a_dom_b:
                verdict = f"{ma} FOSD {mb}"
            elif b_dom_a:
                verdict = f"{mb} FOSD {ma}"
            else:
                verdict = "no_dominance (curves cross)"
            res[f"{ma}_vs_{mb}"] = {
                "verdict": verdict,
                "mean_a": float(a.mean()), "mean_b": float(b.mean()),
                "median_a": float(np.median(a)), "median_b": float(np.median(b)),
            }
    return res


def edge_decay_auc(bench_root: Path, models: list[str]) -> dict:
    """Integrate net-ticks-per-trade across hold-time buckets → AUC.
    Higher AUC = more persistent edge."""
    res = {}
    for m in models:
        f = bench_root / m / "edge_decay.csv"
        if not f.exists():
            continue
        df = pd.read_csv(f)
        if df.empty:
            continue
        # Use P99/passive_at_touch slice, sum across horizons & sides
        sub = df[(df["band"] == "P99") & (df["err"].fillna("") == "")]
        if sub.empty:
            continue
        # AUC = trapezoid net_ticks over hold_sec, per (head, side); then sum
        aucs = []
        for (head, side), g in sub.groupby(["head", "side"]):
            g = g.sort_values("hold_sec")
            x = g["hold_sec"].values
            y = g["net_ticks_per_trade"].values
            if x.size < 3:
                continue
            mask = np.isfinite(y)
            if mask.sum() < 3:
                continue
            auc = float(np.trapz(y[mask], x[mask]))
            aucs.append({"head": head, "side": side, "auc": auc})
        res[m] = {
            "aucs": aucs,
            "total_auc": sum(a["auc"] for a in aucs),
            "best_head_side_auc": max(aucs, key=lambda a: a["auc"]) if aucs else None,
        }
    return res


def collect_head_importance(bench_root: Path, models: list[str]) -> dict:
    res = {}
    for m in models:
        f = bench_root / m / "meta_mlp_results.json"
        if not f.exists():
            continue
        j = json.loads(f.read_text())
        abl = j.get("ablation_delta_sharpe", {})
        if not abl:
            continue
        ranked = sorted(abl.items(), key=lambda kv: -(kv[1].get("delta_sharpe") or 0))
        res[m] = {"top3": ranked[:3], "auc": j.get("auc_mean")}
    return res


def realized_pnl_summary(ledgers: dict, dollars_per_tick: float = 12.50) -> dict:
    res = {}
    for m, info in ledgers.items():
        led = info["ledger"]
        res[m] = {
            "n_attempted": led.n_attempted,
            "n_filled": led.n_filled,
            "fill_rate": led.fill_rate,
            "sharpe": led.sharpe,
            "sortino": led.sortino,
            "profit_factor": led.profit_factor,
            "win_rate_pct": led.win_rate,
            "pnl_ticks_total": led.pnl_ticks_total,
            "pnl_dollars_total": led.pnl_ticks_total * dollars_per_tick,
            "pnl_ticks_per_trade": led.pnl_ticks_per_trade,
            "max_drawdown_ticks": led.max_drawdown_ticks,
            "adverse_sel_ticks_avg": led.adverse_selection_cost_ticks_avg,
            "config": info["cfg_json"]["razer_cli_flags"],
        }
    return res


def pick_winner(cum_res: dict, fosd: dict, auc_res: dict,
                pnl_res: dict) -> tuple[str, str]:
    """Apply the dominance hierarchy:
      1. FOSD wins outright if one model dominates all others.
      2. Else: model with highest realized $ PnL AND positive Sharpe.
      3. Else: highest edge-decay AUC.
      4. Else: explicit non-dominance with failure-mode call (HC #353).
    """
    models = list(pnl_res.keys())
    if not models:
        return ("none", "no models had usable ledgers")

    # 1. FOSD
    fosd_winners = {}
    for k, v in fosd.items():
        if "FOSD" in v.get("verdict", ""):
            winner = v["verdict"].split(" FOSD ")[0]
            fosd_winners[winner] = fosd_winners.get(winner, 0) + 1
    if fosd_winners:
        best = max(fosd_winners.items(), key=lambda kv: kv[1])
        if best[1] == len(models) - 1:  # dominates all others
            return (best[0], f"{best[0]} first-order stochastically dominates all other models on per-trade net-ticks distribution")

    # 2. Realized $ PnL with positive Sharpe
    profitable = {m: v for m, v in pnl_res.items() if v["sharpe"] > 0 and v["pnl_ticks_total"] > 0}
    if profitable:
        best = max(profitable.items(), key=lambda kv: kv[1]["pnl_dollars_total"])
        return (best[0], f"{best[0]} produces the highest realized $ PnL ({best[1]['pnl_dollars_total']:.0f}) under its optimal strategy with positive Sharpe={best[1]['sharpe']:.2f}")

    # 3. Edge-decay AUC
    if auc_res:
        best = max(auc_res.items(), key=lambda kv: kv[1]["total_auc"])
        if best[1]["total_auc"] > 0:
            return (best[0], f"{best[0]} has the highest edge-decay AUC ({best[1]['total_auc']:.2f}) — edge persists longest across hold-time buckets")

    # 4. No model wins
    return ("INCONCLUSIVE", "no model achieves FOSD, positive realized PnL, or positive edge-decay AUC — flagging architectural failure (HC #353)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench-root", required=True)
    ap.add_argument("--labels-dir", required=True)
    ap.add_argument("--models", default="v2,v3_2,v3_3")
    args = ap.parse_args()

    bench_root = Path(args.bench_root)
    models_req = args.models.split(",")
    models = [m for m in models_req if (bench_root / m / "recommended_config.json").exists()]
    print(f"[xmodel] models with completed bench: {models}")
    if not models:
        print("[xmodel] nothing to compare; exiting.")
        return

    ledgers = load_per_trade_ledgers(bench_root, models, Path(args.labels_dir))
    cum_res = cumulative_edge_curves(ledgers, bench_root / "CROSS_MODEL_DOMINANCE.png")
    fosd = fosd_test(ledgers)
    auc_res = edge_decay_auc(bench_root, models)
    head_imp = collect_head_importance(bench_root, models)
    pnl_res = realized_pnl_summary(ledgers)

    winner, reason = pick_winner(cum_res, fosd, auc_res, pnl_res)

    lines = [
        "# CROSS-MODEL VERDICT (HC #358g / HC #350)",
        f"Models compared: {', '.join(models)}",
        "",
        "## 1. Cumulative-edge final PnL (Lorenz-like)",
    ]
    for m, v in cum_res.items():
        lines.append(f"- {m}: n_filled={v['n_filled']}, cum_ticks_final={v['cum_ticks_final']:.2f}")

    lines += ["", "## 2. FOSD pairwise tests"]
    for k, v in fosd.items():
        lines.append(f"- {k}: **{v.get('verdict')}** | mean_a={v.get('mean_a', float('nan')):.3f} mean_b={v.get('mean_b', float('nan')):.3f}")

    lines += ["", "## 3. Edge-decay AUC (P99 / passive)"]
    for m, v in auc_res.items():
        bh = v.get("best_head_side_auc")
        bh_str = f"head={bh['head']} side={bh['side']} auc={bh['auc']:.3f}" if bh else "n/a"
        lines.append(f"- {m}: total_auc={v['total_auc']:.3f} | best: {bh_str}")

    lines += ["", "## 4. Head-importance (Δ-Sharpe ablation, top 3 per model)"]
    for m, v in head_imp.items():
        lines.append(f"- {m}: meta-MLP AUC={v['auc']:.4f}")
        for col, info in v["top3"]:
            lines.append(f"   - {col}: ΔSharpe={info.get('delta_sharpe', float('nan')):.3f}")

    lines += ["", "## 5. Realized PnL under each model's optimal strategy"]
    for m, v in pnl_res.items():
        lines.append(f"### {m}")
        lines.append(f"- Sharpe: {v['sharpe']:.2f} | Sortino: {v['sortino']:.2f} | PF: {v['profit_factor']:.2f} | WR: {v['win_rate_pct']:.2f}%")
        lines.append(f"- fills/attempts: {v['n_filled']}/{v['n_attempted']} ({v['fill_rate']*100:.1f}%)")
        lines.append(f"- PnL: {v['pnl_ticks_total']:.2f} ticks  ≈  ${v['pnl_dollars_total']:.0f}")
        lines.append(f"- Max DD: {v['max_drawdown_ticks']:.2f} ticks | adverse-sel: {v['adverse_sel_ticks_avg']:.3f} ticks")
        lines.append(f"- Config: `{' '.join(v['config'])}`")

    lines += ["", "---", "", f"## STRONGEST MODEL FOR EXECUTION: **{winner}**",
              f"because {reason}.", ""]

    (bench_root / "CROSS_MODEL_VERDICT.md").write_text("\n".join(lines))
    print(f"[xmodel] wrote {bench_root / 'CROSS_MODEL_VERDICT.md'}")
    print(f"[xmodel] STRONGEST MODEL FOR EXECUTION: {winner} — {reason}")


if __name__ == "__main__":
    main()
