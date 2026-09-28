#!/usr/bin/env python3
"""HC #441 FULL VERDICT — adverse-selection, per-day P&L, MFE/MAE,
exit-reason breakdown, regime stratification, and plots for the champion
config (SL=0.50, TP=3.00, H=1.5s) on the FULL cached OOT set.

Also resolves the UPSIDE variant (TP=8.0) and CONSERV (TP=2.5) for
comparison. Adds per-fill MFE/MAE captured WITHIN the hold window so we
can quantify adverse selection.

Inputs:
  - Per-day fill caches built by hc437_pathB_exit_sweep.py
Outputs:
  - output/hc441_full_verdict/per_fill_<variant>.csv
  - output/hc441_full_verdict/per_day_<variant>.csv
  - output/hc441_full_verdict/plots/*.png
  - output/hc441_full_verdict/verdict.md
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime as dt

LVL3 = Path("/home/jupiter/Lvl3Quant")
sys.path.insert(0, str(LVL3 / "scripts/v3_4_research"))
import hc437_pathB_exit_sweep as hc437  # noqa: E402

TICK_RAW = 25_000_000
TICK_USD = 12.50
COMMISSION_TICKS = 0.376

CACHE_DIR = LVL3 / "output/hc437_pathB_exit_sweep/fill_cache"
OUT_DIR = LVL3 / "output/hc441_full_verdict"
PLOTS_DIR = OUT_DIR / "plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

CONFIGS = [
    {"tag": "PRIMARY",  "SL": 0.50, "TP": 3.0, "H": 1.5},
    {"tag": "CONSERV",  "SL": 0.50, "TP": 2.5, "H": 1.5},
    {"tag": "UPSIDE",   "SL": 0.50, "TP": 8.0, "H": 1.5},
]


def load_cache(date_str: str) -> dict:
    p = CACHE_DIR / f"{date_str}_cache.npz"
    raw = np.load(p, allow_pickle=True)
    fills = list(raw['fills'])
    lens = raw['traj_lens']
    flat = raw['traj_flat']
    trajs = []
    cursor = 0
    for L in lens:
        trajs.append(flat[cursor:cursor + L])
        cursor += L
    return {'fills': fills, 'traj': trajs, 'eod_ts': int(raw['eod_ts'])}


def resolve_with_mfe_mae(cache: dict, TP: float, SL: float, H: float):
    """Same as hc437.resolve_cell but also records MFE / MAE within hold
    window, time-to-TP / time-to-SL, and time-to-MFE."""
    rows = []
    hold_ns = int(H * 1e9)
    tp_raw = int(round(TP * TICK_RAW))
    sl_raw = int(round(SL * TICK_RAW))
    for fill, traj in zip(cache['fills'], cache['traj']):
        direction = fill['direction']
        entry = fill['entry_price_raw']
        sig_ts = fill['sig_ts_ns']
        ent_ts = fill['entry_ts_ns']

        if direction == 'long':
            tp_price = entry + tp_raw
            sl_price = entry - sl_raw
        else:
            tp_price = entry - tp_raw
            sl_price = entry + sl_raw

        if traj.size == 0:
            exit_pr = entry; exit_reason = 'no_trades'; hold_actual = hold_ns
            mfe_tk = 0.0; mae_tk = 0.0; t_mfe = 0; t_tp = -1; t_sl = -1
        else:
            offs = traj[:, 0]; prs = traj[:, 1]
            within = offs <= hold_ns
            if not within.any():
                exit_pr = entry; exit_reason = 'no_trades'; hold_actual = hold_ns
                mfe_tk = 0.0; mae_tk = 0.0; t_mfe = 0; t_tp = -1; t_sl = -1
            else:
                offs_h = offs[within]; prs_h = prs[within]
                if direction == 'long':
                    tp_hit = prs_h >= tp_price
                    sl_hit = prs_h <= sl_price
                    max_fav = prs_h.max() - entry
                    max_adv = entry - prs_h.min()
                    idx_mfe = int(np.argmax(prs_h))
                else:
                    tp_hit = prs_h <= tp_price
                    sl_hit = prs_h >= sl_price
                    max_fav = entry - prs_h.min()
                    max_adv = prs_h.max() - entry
                    idx_mfe = int(np.argmin(prs_h))
                mfe_tk = max_fav / TICK_RAW
                mae_tk = max_adv / TICK_RAW
                t_mfe = int(offs_h[idx_mfe])

                first_tp = int(np.argmax(tp_hit)) if tp_hit.any() else -1
                first_sl = int(np.argmax(sl_hit)) if sl_hit.any() else -1
                t_tp = int(offs_h[first_tp]) if first_tp >= 0 else -1
                t_sl = int(offs_h[first_sl]) if first_sl >= 0 else -1

                if first_tp >= 0 and (first_sl < 0 or first_tp <= first_sl):
                    exit_pr = int(tp_price); exit_reason = 'tp'
                    hold_actual = int(offs_h[first_tp])
                elif first_sl >= 0:
                    exit_pr = int(sl_price); exit_reason = 'sl'
                    hold_actual = int(offs_h[first_sl])
                else:
                    exit_pr = int(prs_h[-1]); exit_reason = 'max_hold'
                    hold_actual = int(offs_h[-1])

        gross_raw = (exit_pr - entry) if direction == 'long' else (entry - exit_pr)
        gross_tk = gross_raw / TICK_RAW
        net_tk = gross_tk - COMMISSION_TICKS

        rows.append({
            'date': fill['date'], 'sig_ts_ns': sig_ts, 'entry_ts_ns': ent_ts,
            'direction': direction,
            'entry_price_raw': entry, 'exit_price_raw': exit_pr,
            'pred_strength': fill['pred_strength'],
            'queue_ahead': fill['queue_ahead'],
            'queue_wait_ns': fill['queue_wait_ns'],
            'slippage_ticks': fill['slippage_ticks'],
            'exit_reason': exit_reason,
            'hold_ns_actual': hold_actual,
            'gross_ticks': gross_tk, 'net_ticks': net_tk,
            'net_dollars': net_tk * TICK_USD,
            'mfe_ticks': mfe_tk, 'mae_ticks': mae_tk,
            't_to_mfe_ns': t_mfe,
            't_to_tp_ns': t_tp, 't_to_sl_ns': t_sl,
        })
    return rows


def metrics(df: pd.DataFrame) -> dict:
    n = len(df)
    if n == 0:
        return {}
    net = df['net_ticks']
    mu, sd = float(net.mean()), float(net.std())
    sh = mu / sd * np.sqrt(n) if sd > 0 else float('nan')
    gw = float(net[net > 0].sum())
    gl = float(-net[net <= 0].sum())
    pf = gw / gl if gl > 0 else float('inf')
    wr = float((net > 0).mean()) * 100
    per_day = df.groupby('date')['net_ticks'].sum()
    n_days = int(per_day.shape[0])
    pos = int((per_day > 0).sum())
    return dict(n=n, net_tk=mu, net_std=sd, PF=pf, WR=wr, Sh=sh,
                n_days=n_days, pos_days=pos,
                total_net_tk=float(net.sum()),
                total_net_usd=float(net.sum() * TICK_USD))


def stratified_metrics(df: pd.DataFrame):
    """Per-day classification: green / red based on first-vs-last fill mid
    (proxy for daily direction; if positive we call it 'green', negative
    'red'). Then per-regime Sharpe."""
    daily_close = (df.sort_values('entry_ts_ns')
                     .groupby('date')['entry_price_raw']
                     .agg(['first', 'last']))
    daily_close['regime'] = np.where(daily_close['last'] > daily_close['first'],
                                     'green', 'red')
    df = df.merge(daily_close[['regime']], left_on='date', right_index=True)
    out = {}
    for reg, sub in df.groupby('regime'):
        out[reg] = metrics(sub)
    return df, out


def adverse_selection(df: pd.DataFrame) -> dict:
    """Adverse selection breakdown.

    For SHORT trades the model expects price to fall. Adverse-selection signals:
      A) Winners with high MAE — we suffered a big adverse excursion before
         the move materialized. mean MAE on TP-winners.
      B) Losers with high MFE — price moved in our favor first, then
         reverted to SL. mean MFE on SL-losers.
      C) Time-to-MFE on winners vs losers — winners hit max-favorable late,
         losers hit max-favorable early then reverted.
    """
    out = {}
    for reason in ('tp', 'sl', 'max_hold', 'no_trades'):
        sub = df[df['exit_reason'] == reason]
        out[f'{reason}_n'] = len(sub)
        if len(sub):
            out[f'{reason}_mfe_mean'] = float(sub['mfe_ticks'].mean())
            out[f'{reason}_mae_mean'] = float(sub['mae_ticks'].mean())
            out[f'{reason}_net_mean'] = float(sub['net_ticks'].mean())
            out[f'{reason}_t_mfe_ms'] = float(sub['t_to_mfe_ns'].mean() / 1e6)
            out[f'{reason}_hold_ms']  = float(sub['hold_ns_actual'].mean() / 1e6)
    return out


def make_plots(df_primary: pd.DataFrame, df_conserv: pd.DataFrame,
               df_upside: pd.DataFrame):
    """Build the diagnostic plots."""
    # PLOT 1: Cumulative P&L equity curves (3 variants)
    fig, ax = plt.subplots(figsize=(12, 5))
    for tag, df, color in [("PRIMARY (TP=3.0)", df_primary, "tab:blue"),
                            ("CONSERV (TP=2.5)", df_conserv, "tab:gray"),
                            ("UPSIDE (TP=8.0)",  df_upside,  "tab:orange")]:
        s = df.sort_values('entry_ts_ns')['net_ticks'].cumsum().values * TICK_USD
        ax.plot(range(len(s)), s, label=tag, color=color, lw=1.5)
    ax.axhline(0, color='k', lw=0.5)
    ax.set_xlabel("Fill # (chronological across 36-day OOT)")
    ax.set_ylabel("Cumulative net P&L ($/contract)")
    ax.set_title("HC #441 Champion config — cumulative equity curve, full OOT")
    ax.legend(loc='upper left')
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "01_equity_curves.png", dpi=110)
    plt.close(fig)

    # PLOT 2: Per-day net P&L bar chart (PRIMARY)
    per_day = df_primary.groupby('date')['net_ticks'].sum().reset_index()
    per_day['net_usd'] = per_day['net_ticks'] * TICK_USD
    per_day = per_day.sort_values('date')
    fig, ax = plt.subplots(figsize=(13, 5))
    colors = ['tab:green' if x > 0 else 'tab:red' for x in per_day['net_usd']]
    ax.bar(range(len(per_day)), per_day['net_usd'], color=colors)
    ax.axhline(0, color='k', lw=0.5)
    ax.set_xticks(range(len(per_day)))
    ax.set_xticklabels(per_day['date'], rotation=70, fontsize=7)
    ax.set_ylabel("Net P&L per day ($/contract)")
    ax.set_title(f"PRIMARY config per-day P&L  "
                 f"({(per_day['net_usd'] > 0).sum()}/{len(per_day)} green days)")
    ax.grid(alpha=0.3, axis='y')
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "02_per_day_pnl.png", dpi=110)
    plt.close(fig)

    # PLOT 3: Exit reason breakdown (3 variants stacked bar)
    fig, ax = plt.subplots(figsize=(9, 5))
    reasons = ['tp', 'sl', 'max_hold', 'no_trades']
    counts = {}
    for tag, df in [("PRIMARY", df_primary), ("CONSERV", df_conserv),
                    ("UPSIDE", df_upside)]:
        counts[tag] = [int((df['exit_reason'] == r).sum()) for r in reasons]
    x = np.arange(3)
    width = 0.2
    for i, r in enumerate(reasons):
        ax.bar(x + i*width,
               [counts['PRIMARY'][i], counts['CONSERV'][i], counts['UPSIDE'][i]],
               width, label=r)
    ax.set_xticks(x + width*1.5)
    ax.set_xticklabels(["PRIMARY", "CONSERV", "UPSIDE"])
    ax.set_ylabel("Number of fills")
    ax.set_title("Exit reason breakdown by variant")
    ax.legend()
    ax.grid(alpha=0.3, axis='y')
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "03_exit_reasons.png", dpi=110)
    plt.close(fig)

    # PLOT 4: MFE vs MAE density by exit reason (PRIMARY)
    # Split into 2 panels so TP-wins and SL-losses don't visually overlap.
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, reason, title in [
        (axes[0], 'tp', "TP-winners (n=%d)" % (df_primary['exit_reason'] == 'tp').sum()),
        (axes[1], 'sl', "SL-losers (n=%d)"  % (df_primary['exit_reason'] == 'sl').sum()),
    ]:
        sub = df_primary[df_primary['exit_reason'] == reason]
        # Add tiny jitter so quantized points don't stack on a single pixel
        rng = np.random.default_rng(0)
        x = sub['mae_ticks'].values + rng.normal(0, 0.08, len(sub))
        y = sub['mfe_ticks'].values + rng.normal(0, 0.15, len(sub))
        hb = ax.hexbin(x, y, gridsize=40, cmap='viridis', mincnt=1,
                       extent=(0, 40, 0, 60))
        cb = fig.colorbar(hb, ax=ax, label='count')
        ax.axhline(3.0, color='lime', ls='--', alpha=0.7, lw=1.3, label='TP=3.0')
        ax.axvline(0.5, color='red',  ls='--', alpha=0.7, lw=1.3, label='SL=0.5')
        ax.set_xlabel("MAE within 1.5s hold (ticks adverse)")
        ax.set_ylabel("MFE within 1.5s hold (ticks favorable)")
        ax.set_title(f"PRIMARY — {title} — MFE vs MAE density")
        ax.set_xlim(0, 40)
        ax.set_ylim(0, 60)
        ax.legend(loc='upper right')
        ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "04_mfe_mae_scatter.png", dpi=110)
    plt.close(fig)

    # PLOT 5: Net-ticks histogram (PRIMARY)
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.hist(df_primary['net_ticks'], bins=80, color='tab:blue',
            edgecolor='k', alpha=0.7)
    mu = df_primary['net_ticks'].mean()
    ax.axvline(mu, color='r', ls='--', label=f'mean = {mu:+.2f} tk')
    ax.axvline(0, color='k', lw=0.5)
    ax.set_xlabel("Net ticks per fill (after commission)")
    ax.set_ylabel("Frequency")
    ax.set_title(f"PRIMARY net-ticks distribution (n={len(df_primary)})")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "05_net_ticks_histogram.png", dpi=110)
    plt.close(fig)

    # PLOT 6: Time to TP / SL / MFE distributions (PRIMARY winners vs losers)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    tp_w = df_primary[df_primary['exit_reason'] == 'tp']
    sl_l = df_primary[df_primary['exit_reason'] == 'sl']
    bins = np.linspace(0, 1500, 60)
    axes[0].hist(tp_w['t_to_tp_ns'] / 1e6, bins=bins, color='tab:green',
                 alpha=0.6, label=f'TP winners (n={len(tp_w)})')
    axes[0].hist(sl_l['t_to_sl_ns'] / 1e6, bins=bins, color='tab:red',
                 alpha=0.6, label=f'SL losers (n={len(sl_l)})')
    axes[0].set_xlabel("Time to exit (ms)")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Time-to-exit distribution")
    axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].hist(df_primary['t_to_mfe_ns'] / 1e6, bins=bins, color='tab:blue',
                 alpha=0.7)
    axes[1].set_xlabel("Time to max-favorable excursion (ms)")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Time to MFE within 1.5s hold")
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "06_time_to_exit.png", dpi=110)
    plt.close(fig)

    # PLOT 7: Win-rate by time-of-day
    fig, ax = plt.subplots(figsize=(11, 4.5))
    df_p = df_primary.copy()
    df_p['ts'] = pd.to_datetime(df_p['sig_ts_ns'], unit='ns', utc=True) \
                   .dt.tz_convert('US/Eastern')
    df_p['hour'] = df_p['ts'].dt.hour
    df_p['win'] = (df_p['net_ticks'] > 0).astype(int)
    by_hour = df_p.groupby('hour').agg(n=('win', 'count'),
                                        wr=('win', 'mean'),
                                        net=('net_ticks', 'mean')).reset_index()
    ax.bar(by_hour['hour'], by_hour['net'], color='tab:blue', alpha=0.7)
    ax2 = ax.twinx()
    ax2.plot(by_hour['hour'], by_hour['wr'], color='tab:red', marker='o',
             label='Win rate')
    ax2.set_ylim(0, 1)
    ax2.set_ylabel("Win rate", color='tab:red')
    ax.set_xlabel("Hour of day (ET)")
    ax.set_ylabel("Mean net ticks", color='tab:blue')
    ax.set_title("PRIMARY — mean net ticks & win rate by hour-of-day")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "07_by_hour.png", dpi=110)
    plt.close(fig)

    # PLOT 8: Pred-strength vs net ticks
    fig, ax = plt.subplots(figsize=(9, 5))
    df_p = df_primary.copy()
    df_p['ps_bucket'] = pd.qcut(df_p['pred_strength'], 10, labels=False)
    by_ps = df_p.groupby('ps_bucket').agg(n=('net_ticks', 'count'),
                                            net=('net_ticks', 'mean'),
                                            ps=('pred_strength', 'mean')).reset_index()
    ax.bar(by_ps['ps_bucket'], by_ps['net'], color='teal')
    ax.axhline(0, color='k', lw=0.5)
    ax.set_xlabel("Confidence decile (low → high)")
    ax.set_ylabel("Mean net ticks")
    ax.set_title("PRIMARY — net edge by confidence decile within top-0.5% band")
    ax.grid(alpha=0.3, axis='y')
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "08_by_confidence.png", dpi=110)
    plt.close(fig)


def main():
    dates = sorted([p.stem.replace('_cache', '')
                    for p in CACHE_DIR.glob("*_cache.npz")])
    print(f"Resolving champion on {len(dates)} cached dates: "
          f"{dates[0]}..{dates[-1]}")

    dfs = {}
    for cfg in CONFIGS:
        rows = []
        for d in dates:
            cache = load_cache(d)
            rows.extend(resolve_with_mfe_mae(cache, cfg['TP'], cfg['SL'], cfg['H']))
        df = pd.DataFrame(rows)
        dfs[cfg['tag']] = df
        df.to_csv(OUT_DIR / f"per_fill_{cfg['tag']}.csv", index=False)
        m = metrics(df)
        adv = adverse_selection(df)
        print(f"\n=== {cfg['tag']} (SL={cfg['SL']} TP={cfg['TP']} H={cfg['H']}s) ===")
        print(f"  n={m['n']}  net={m['net_tk']:+.4f} tk  std={m['net_std']:.3f}  "
              f"PF={m['PF']:.3f}  WR={m['WR']:.1f}%  Sh√N={m['Sh']:.2f}")
        print(f"  days={m['n_days']}  positive_days={m['pos_days']}/{m['n_days']}")
        print(f"  total net = {m['total_net_tk']:+.1f} ticks "
              f"(${m['total_net_usd']:+.0f} per contract)")
        print(f"  ADVERSE SELECTION:")
        for r in ('tp', 'sl', 'max_hold', 'no_trades'):
            if f'{r}_n' in adv and adv[f'{r}_n'] > 0:
                print(f"    {r:9s} n={adv[f'{r}_n']:>4d}  "
                      f"mfe={adv[f'{r}_mfe_mean']:5.2f}  "
                      f"mae={adv[f'{r}_mae_mean']:5.2f}  "
                      f"net={adv[f'{r}_net_mean']:+5.2f}  "
                      f"t→exit={adv[f'{r}_hold_ms']:.0f}ms")

        # Per-day CSV
        per_day = df.groupby('date').agg(
            n_fills=('net_ticks', 'count'),
            net_tk=('net_ticks', 'sum'),
            net_avg=('net_ticks', 'mean'),
            wr=('net_ticks', lambda x: (x > 0).mean() * 100),
        ).reset_index()
        per_day['net_usd'] = per_day['net_tk'] * TICK_USD
        per_day.to_csv(OUT_DIR / f"per_day_{cfg['tag']}.csv", index=False)

    # Regime stratification on PRIMARY
    df_strat, regs = stratified_metrics(dfs['PRIMARY'])
    print("\n=== PRIMARY regime stratification ===")
    for reg, m in regs.items():
        print(f"  {reg:5s} n={m['n']:>4d}  net={m['net_tk']:+.4f}  "
              f"PF={m['PF']:.2f}  Sh={m['Sh']:.2f}  "
              f"days={m['n_days']} pos={m['pos_days']}")
    if 'green' in regs and 'red' in regs:
        sgn = max(abs(regs['green']['Sh']), abs(regs['red']['Sh']))
        if sgn > 0:
            delta = abs(regs['green']['Sh'] - regs['red']['Sh']) / sgn
            print(f"  reg_delta_norm = {delta:.3f}  "
                  f"(HC #428 R1 threshold ≤ 0.50)")

    # Train/test holdout (21/15 split)
    train_dates = set(dates[:21])
    print(f"\n=== PRIMARY 21/15 holdout ===")
    for label, ds in [("TRAIN", train_dates),
                       ("TEST",  set(dates) - train_dates)]:
        sub = dfs['PRIMARY'][dfs['PRIMARY']['date'].isin(ds)]
        m = metrics(sub)
        print(f"  {label}: n={m['n']}  net={m['net_tk']:+.4f}  "
              f"PF={m['PF']:.2f}  Sh={m['Sh']:.2f}  "
              f"pos_days={m['pos_days']}/{m['n_days']}")

    # Plots
    print("\nBuilding plots...")
    make_plots(dfs['PRIMARY'], dfs['CONSERV'], dfs['UPSIDE'])
    plots = sorted(PLOTS_DIR.glob("*.png"))
    print(f"  Wrote {len(plots)} plots to {PLOTS_DIR}")


if __name__ == "__main__":
    main()
