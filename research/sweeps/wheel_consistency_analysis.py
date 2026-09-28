"""
Megacap Wheel v2 — Consistency & Sustainability Analysis
$100K base, 1x leverage
"""

import pandas as pd
import numpy as np

BASE = 100_000.0

# ── helpers ──────────────────────────────────────────────────────────────────

def load_equity(path):
    df = pd.read_parquet(path)
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    # normalize to $100K start
    scale = BASE / df['equity'].iloc[0]
    df['equity_norm'] = df['equity'] * scale
    df['daily_pnl'] = df['equity_norm'].diff().fillna(0)
    return df


def streak(series_bool):
    """Return (max_true_streak, max_false_streak) for a boolean series."""
    max_pos = max_neg = cur_pos = cur_neg = 0
    for v in series_bool:
        if v:
            cur_pos += 1
            cur_neg = 0
        else:
            cur_neg += 1
            cur_pos = 0
        max_pos = max(max_pos, cur_pos)
        max_neg = max(max_neg, cur_neg)
    return max_pos, max_neg


def analyze(df, label):
    print(f"\n{'='*70}")
    print(f"  MEGACAP WHEEL v2  —  {label}")
    print(f"{'='*70}")

    eq = df['equity_norm']
    start_date = eq.index[0].strftime('%Y-%m-%d')
    end_date   = eq.index[-1].strftime('%Y-%m-%d')
    total_return = (eq.iloc[-1] / eq.iloc[0] - 1) * 100
    n_years = (eq.index[-1] - eq.index[0]).days / 365.25
    cagr = ((eq.iloc[-1] / eq.iloc[0]) ** (1/n_years) - 1) * 100

    print(f"\n  Period: {start_date}  →  {end_date}  ({n_years:.1f} years)")
    print(f"  Total return: {total_return:+.1f}%   CAGR: {cagr:+.1f}%")
    print(f"  Final equity (normalized): ${eq.iloc[-1]:,.0f}")

    # ── WEEKLY ──────────────────────────────────────────────────────────────
    weekly = df['equity_norm'].resample('W').last().dropna()
    weekly_pnl = weekly.diff().dropna()          # $ change each week
    weekly_pct = weekly.pct_change().dropna() * 100

    pct_pos_w  = (weekly_pnl > 0).mean() * 100
    avg_w      = weekly_pnl.mean()
    med_w      = weekly_pnl.median()
    worst_w    = weekly_pnl.min()
    best_w     = weekly_pnl.max()
    std_w      = weekly_pnl.std()
    p25_w      = weekly_pnl.quantile(0.25)
    win_streak_w, lose_streak_w = streak(weekly_pnl > 0)

    worst_w_date = weekly_pnl.idxmin().strftime('%Y-%m-%d')
    best_w_date  = weekly_pnl.idxmax().strftime('%Y-%m-%d')

    print(f"""
  ┌─ WEEKLY ANALYSIS ({len(weekly_pnl)} weeks) ─────────────────────────────────┐
  │  Profitable weeks          : {pct_pos_w:5.1f}%
  │  Average weekly income     : ${avg_w:+8,.0f}
  │  Median weekly income      : ${med_w:+8,.0f}
  │  Worst week  ({worst_w_date})  : ${worst_w:+8,.0f}
  │  Best week   ({best_w_date})   : ${best_w:+8,.0f}
  │  Std dev (weekly)          : ${std_w:8,.0f}
  │  25th-percentile week      : ${p25_w:+8,.0f}
  │  Longest WINNING streak    : {win_streak_w} consecutive weeks
  │  Longest LOSING  streak    : {lose_streak_w} consecutive weeks
  └──────────────────────────────────────────────────────────────────────""")

    # ── MONTHLY ─────────────────────────────────────────────────────────────
    monthly = df['equity_norm'].resample('ME').last().dropna()
    monthly_pnl = monthly.diff().dropna()
    monthly_pct = monthly.pct_change().dropna() * 100

    pct_pos_m  = (monthly_pnl > 0).mean() * 100
    avg_m      = monthly_pnl.mean()
    med_m      = monthly_pnl.median()
    worst_m    = monthly_pnl.min()
    best_m     = monthly_pnl.max()
    std_m      = monthly_pnl.std()
    p25_m      = monthly_pnl.quantile(0.25)
    win_streak_m, lose_streak_m = streak(monthly_pnl > 0)

    worst_m_date = monthly_pnl.idxmin().strftime('%Y-%m')
    best_m_date  = monthly_pnl.idxmax().strftime('%Y-%m')

    print(f"""
  ┌─ MONTHLY ANALYSIS ({len(monthly_pnl)} months) ──────────────────────────────┐
  │  Profitable months         : {pct_pos_m:5.1f}%
  │  Average monthly income    : ${avg_m:+8,.0f}
  │  Median monthly income     : ${med_m:+8,.0f}
  │  Worst month  ({worst_m_date})     : ${worst_m:+8,.0f}
  │  Best month   ({best_m_date})      : ${best_m:+8,.0f}
  │  Std dev (monthly)         : ${std_m:8,.0f}
  │  25th-percentile month     : ${p25_m:+8,.0f}
  │  Longest WINNING streak    : {win_streak_m} consecutive months
  │  Longest LOSING  streak    : {lose_streak_m} consecutive months
  └──────────────────────────────────────────────────────────────────────""")

    # ── QUARTERLY ───────────────────────────────────────────────────────────
    quarterly = df['equity_norm'].resample('QE').last().dropna()
    quarterly_pnl = quarterly.diff().dropna()
    quarterly_pct = quarterly.pct_change().dropna() * 100

    pct_pos_q      = (quarterly_pnl > 0).mean() * 100
    neg_quarters   = quarterly_pnl[quarterly_pnl <= 0]
    worst_q        = quarterly_pnl.min()
    best_q         = quarterly_pnl.max()
    win_streak_q, lose_streak_q = streak(quarterly_pnl > 0)

    print(f"""
  ┌─ QUARTERLY ANALYSIS ({len(quarterly_pnl)} quarters) ────────────────────────┐
  │  Profitable quarters       : {pct_pos_q:5.1f}%
  │  Quarters that LOST money  : {len(neg_quarters)} of {len(quarterly_pnl)}""")
    if len(neg_quarters) > 0:
        for dt, val in neg_quarters.items():
            print(f"  │    → Q ending {dt.strftime('%Y-%m')}: ${val:+,.0f}")
    print(f"""  │  Best quarter              : ${best_q:+8,.0f}
  │  Worst quarter             : ${worst_q:+8,.0f}
  │  Longest WINNING streak    : {win_streak_q} consecutive quarters
  │  Longest LOSING  streak    : {lose_streak_q} consecutive quarters
  └──────────────────────────────────────────────────────────────────────""")

    # ── SUSTAINABILITY SCORE ─────────────────────────────────────────────────
    pct_1000 = (monthly_pnl >= 1000).mean() * 100
    pct_500  = (monthly_pnl >= 500).mean()  * 100
    pct_250w = (weekly_pnl  >= 250).mean()  * 100

    # cash buffer recommendation = longest losing streak + 1 safety month
    recommended_buffer = lose_streak_m + 1

    print(f"""
  ┌─ SUSTAINABILITY SCORE ───────────────────────────────────────────────┐
  │  Months ≥ $1,000 income    : {pct_1000:5.1f}%  (travel budget test)
  │  Months ≥ $500  income     : {pct_500:5.1f}%
  │  Weeks  ≥ $250  income     : {pct_250w:5.1f}%
  │  Recommended cash buffer   : {recommended_buffer} months
  │  (longest losing streak = {lose_streak_m} mo + 1 safety month)
  └──────────────────────────────────────────────────────────────────────""")

    return {
        'label': label,
        'pct_pos_w': pct_pos_w,
        'avg_w': avg_w,
        'med_w': med_w,
        'worst_w': worst_w,
        'lose_streak_w': lose_streak_w,
        'pct_pos_m': pct_pos_m,
        'avg_m': avg_m,
        'med_m': med_m,
        'worst_m': worst_m,
        'pct_pos_q': pct_pos_q,
        'n_neg_q': len(neg_quarters),
        'pct_1000': pct_1000,
        'pct_500': pct_500,
        'pct_250w': pct_250w,
        'lose_streak_m': lose_streak_m,
        'recommended_buffer': recommended_buffer,
    }


# ── MAIN ─────────────────────────────────────────────────────────────────────

base_path = '/home/jupiter/Lvl3Quant/output/megacap_wheel_v2/'

results = []
for level, label in [
    ('level_1', 'Level 1 — Conservative'),
    ('level_3', 'Level 3 — Moderate (BEST RISK-ADJUSTED)'),
    ('level_5', 'Level 5 — Aggressive'),
]:
    path = f'{base_path}equity_{level}.parquet'
    try:
        df = load_equity(path)
        r = analyze(df, label)
        results.append(r)
    except Exception as e:
        print(f"\n[WARN] Could not load {path}: {e}")

# ── CROSS-LEVEL COMPARISON ────────────────────────────────────────────────────
print(f"\n{'='*70}")
print("  CROSS-LEVEL COMPARISON SUMMARY")
print(f"{'='*70}")
print(f"  {'Metric':<35} {'Conservative':>13} {'Moderate':>13} {'Aggressive':>13}")
print(f"  {'-'*74}")

metrics = [
    ('% profitable weeks',          'pct_pos_w',      '{:.1f}%'),
    ('Avg weekly income',           'avg_w',          '${:,.0f}'),
    ('Median weekly income',        'med_w',          '${:,.0f}'),
    ('Worst week ever',             'worst_w',        '${:,.0f}'),
    ('Max consec losing weeks',     'lose_streak_w',  '{}'),
    ('% profitable months',         'pct_pos_m',      '{:.1f}%'),
    ('Avg monthly income',          'avg_m',          '${:,.0f}'),
    ('Median monthly income',       'med_m',          '${:,.0f}'),
    ('Worst month ever',            'worst_m',        '${:,.0f}'),
    ('% profitable quarters',       'pct_pos_q',      '{:.1f}%'),
    ('Negative quarters',           'n_neg_q',        '{}'),
    ('Months ≥ $1K (travel test)',  'pct_1000',       '{:.1f}%'),
    ('Months ≥ $500',               'pct_500',        '{:.1f}%'),
    ('Weeks ≥ $250',                'pct_250w',       '{:.1f}%'),
    ('Max consec losing months',    'lose_streak_m',  '{}'),
    ('Recommended cash buffer',     'recommended_buffer', '{} months'),
]

for name, key, fmt in metrics:
    vals = []
    for r in results:
        try:
            vals.append(fmt.format(r[key]))
        except:
            vals.append('N/A')
    while len(vals) < 3:
        vals.append('N/A')
    print(f"  {name:<35} {vals[0]:>13} {vals[1]:>13} {vals[2]:>13}")

print(f"\n{'='*70}")
print("  VERDICT")
print(f"{'='*70}")

# Auto-generate verdict based on Level 3
if results:
    r3 = next((r for r in results if 'Moderate' in r['label']), results[0])
    feast_famine = "FEAST-OR-FAMINE" if r3['pct_pos_m'] < 65 or r3['lose_streak_m'] >= 3 else "REASONABLY CONSISTENT"
    print(f"""
  Strategy classification : {feast_famine}
  Win rate (monthly)      : {r3['pct_pos_m']:.1f}% profitable months
  Typical monthly payout  : ${r3['med_m']:,.0f} (median, $100K base)
  Worst-case month        : ${r3['worst_m']:,.0f}
  Longest pain streak     : {r3['lose_streak_m']} consecutive losing months
  Travel budget ($1K/mo)  : {r3['pct_1000']:.1f}% of months delivered this
  Recommended buffer      : {r3['recommended_buffer']} months of living expenses
""")
