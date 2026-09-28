"""
Theme Features x BPS Ticker Timing Analysis
=============================================
Research question: Can theme inflow signals predict which BPS tickers
will have better outcomes, enabling a TIMING overlay (not universe change)?

Uses: theme_features.parquet + BPS trades data
"""

import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 1. LOAD DATA
# ============================================================
print("=" * 70)
print("THEME FEATURES x BPS TIMING ANALYSIS")
print("=" * 70)

theme = pd.read_parquet('/home/jupiter/Lvl3Quant/data/feature_store/v2/theme_features.parquet')
theme['date'] = pd.to_datetime(theme['date'])

# Load BPS trades - use full_stack which has the most complete data
trades = pd.read_parquet('/home/jupiter/Lvl3Quant/output/bps_full_stack/trades_C_full_stack.parquet')
trades['open_date'] = pd.to_datetime(trades['open_date'])
trades['close_date'] = pd.to_datetime(trades['close_date'])

# GA-optimized universe
GA_UNIVERSE = ['NFLX','NVDA','PLTR','WMT','GM','PFE','LLY','MCD','CL','T',
               'SMCI','OXY','HOOD','JNJ','F','TGT','VZ','TSLA','PANW','TMUS']

print(f"\nTheme features: {theme.shape[0]:,} rows, {theme['ticker'].nunique()} tickers")
print(f"  Date range: {theme['date'].min().date()} to {theme['date'].max().date()}")
print(f"\nBPS trades: {trades.shape[0]:,} trades")
print(f"  Date range: {trades['open_date'].min().date()} to {trades['open_date'].max().date()}")
print(f"  Tickers: {sorted(trades['ticker'].unique())}")

# Filter trades to GA universe
trades_ga = trades[trades['ticker'].isin(GA_UNIVERSE)].copy()
print(f"\nGA universe trades: {trades_ga.shape[0]:,}")

# ============================================================
# 2. TICKER-THEME MAPPING
# ============================================================
print("\n" + "=" * 70)
print("TICKER-TO-THEME MAPPING")
print("=" * 70)

# Map each GA ticker to its relevant theme(s)
TICKER_THEMES = {
    'NVDA': ['aiSemis'], 'SMCI': ['aiSemis'],
    'PLTR': ['aiSemis', 'software'],
    'PANW': ['software'],
    'TSLA': ['ev'],
    'OXY': ['energy'],
    'WMT': ['consumer'], 'MCD': ['consumer'], 'CL': ['consumer'], 'TGT': ['consumer'],
    'PFE': ['healthcare'], 'LLY': ['healthcare'], 'JNJ': ['healthcare'],
    'T': ['financials'], 'VZ': ['financials'], 'TMUS': ['financials'],
    'GM': ['industrials', 'ev'],
    'HOOD': ['financials'],
    'F': ['industrials', 'ev'],
    'NFLX': ['software', 'consumer'],
}

for t in GA_UNIVERSE:
    themes = TICKER_THEMES.get(t, ['none'])
    print(f"  {t:6s} -> {', '.join(themes)}")

# ============================================================
# 3. MERGE THEME FEATURES WITH TRADE OUTCOMES
# ============================================================
# For each trade, get theme features at open_date
merged = trades_ga.merge(
    theme[theme['ticker'].isin(GA_UNIVERSE)],
    left_on=['ticker', 'open_date'],
    right_on=['ticker', 'date'],
    how='left'
)

print(f"\nMerged trades with theme features: {merged.shape[0]:,}")
print(f"  Trades with theme data: {merged['date'].notna().sum():,}")
print(f"  Trades without theme data: {merged['date'].isna().sum():,}")

# Create binary outcome: profit (1) vs loss (0)
merged['profitable'] = (merged['realized_pnl'] > 0).astype(int)
# Normalize PnL by max_loss for comparability
merged['pnl_pct'] = merged['realized_pnl'] / merged['max_loss'].abs()

# ============================================================
# 4. IC ANALYSIS: Theme Features vs 21-day Forward BPS Outcomes
# ============================================================
print("\n" + "=" * 70)
print("IC: THEME FEATURES vs BPS TRADE OUTCOMES")
print("=" * 70)

# Theme feature columns (inflow z-scores are the most relevant for timing)
theme_cols = [c for c in theme.columns if c.startswith('theme_') and c != 'theme_primary']
inflow_cols = [c for c in theme_cols if 'inflow_z' in c]
r20_cols = [c for c in theme_cols if '_r20' in c]
r60_cols = [c for c in theme_cols if '_r60' in c]
rel_cols = [c for c in theme_cols if '_rel_' in c]

print("\nA) RANK IC: All theme features vs trade PnL% (all tickers pooled)")
print("-" * 60)

valid = merged.dropna(subset=['pnl_pct'])
for col in theme_cols:
    subset = valid.dropna(subset=[col])
    if len(subset) < 50:
        continue
    ic, pval = stats.spearmanr(subset[col], subset['pnl_pct'])
    sig = "***" if pval < 0.01 else "**" if pval < 0.05 else "*" if pval < 0.10 else ""
    if abs(ic) > 0.02 or pval < 0.05:
        print(f"  {col:40s}  IC={ic:+.4f}  p={pval:.4f} {sig}")

# ============================================================
# 5. TICKER-SPECIFIC THEME IC ANALYSIS
# ============================================================
print("\n" + "=" * 70)
print("TICKER-SPECIFIC: Relevant Theme Inflow vs BPS Outcome")
print("=" * 70)

results = []
for ticker, themes in TICKER_THEMES.items():
    ticker_trades = valid[valid['ticker'] == ticker]
    if len(ticker_trades) < 20:
        continue

    for theme_name in themes:
        inflow_col = f'theme_{theme_name}_inflow_z20'
        r20_col = f'theme_{theme_name}_r20'

        for col in [inflow_col, r20_col]:
            if col not in ticker_trades.columns:
                continue
            sub = ticker_trades.dropna(subset=[col])
            if len(sub) < 20:
                continue
            ic, pval = stats.spearmanr(sub[col], sub['pnl_pct'])
            results.append({
                'ticker': ticker,
                'theme': theme_name,
                'feature': col,
                'IC': ic,
                'pval': pval,
                'n_trades': len(sub),
            })
            sig = "***" if pval < 0.01 else "**" if pval < 0.05 else "*" if pval < 0.10 else ""
            if abs(ic) > 0.03 or pval < 0.10:
                print(f"  {ticker:6s} | {col:40s} | IC={ic:+.4f} p={pval:.4f} n={len(sub):4d} {sig}")

results_df = pd.DataFrame(results)
if len(results_df) > 0:
    print(f"\n  Summary: {len(results_df)} ticker-theme pairs tested")
    sig_results = results_df[results_df['pval'] < 0.05]
    print(f"  Significant at p<0.05: {len(sig_results)}")
    if len(sig_results) > 0:
        print(sig_results.to_string(index=False))

# ============================================================
# 6. CONDITIONAL ANALYSIS: High vs Low Inflow Regimes
# ============================================================
print("\n" + "=" * 70)
print("CONDITIONAL: BPS PERFORMANCE IN HIGH vs LOW THEME INFLOW")
print("=" * 70)

def analyze_regime(df, ticker, inflow_col, threshold_pct=67):
    """Compare BPS outcomes when theme inflow is high vs low."""
    sub = df[(df['ticker'] == ticker)].dropna(subset=[inflow_col, 'pnl_pct'])
    if len(sub) < 30:
        return None

    hi_thresh = np.percentile(sub[inflow_col], threshold_pct)
    lo_thresh = np.percentile(sub[inflow_col], 100 - threshold_pct)

    hi = sub[sub[inflow_col] >= hi_thresh]
    lo = sub[sub[inflow_col] <= lo_thresh]

    if len(hi) < 10 or len(lo) < 10:
        return None

    return {
        'ticker': ticker,
        'feature': inflow_col,
        'hi_n': len(hi),
        'lo_n': len(lo),
        'hi_wr': (hi['realized_pnl'] > 0).mean(),
        'lo_wr': (lo['realized_pnl'] > 0).mean(),
        'hi_avg_pnl': hi['realized_pnl'].mean(),
        'lo_avg_pnl': lo['realized_pnl'].mean(),
        'hi_median_pnl': hi['realized_pnl'].median(),
        'lo_median_pnl': lo['realized_pnl'].median(),
        'hi_sharpe': hi['pnl_pct'].mean() / hi['pnl_pct'].std() * np.sqrt(252/7) if hi['pnl_pct'].std() > 0 else 0,
        'lo_sharpe': lo['pnl_pct'].mean() / lo['pnl_pct'].std() * np.sqrt(252/7) if lo['pnl_pct'].std() > 0 else 0,
    }

conditional_results = []
for ticker, themes in TICKER_THEMES.items():
    for theme_name in themes:
        inflow_col = f'theme_{theme_name}_inflow_z20'
        if inflow_col not in valid.columns:
            continue
        result = analyze_regime(valid, ticker, inflow_col)
        if result:
            conditional_results.append(result)

if conditional_results:
    cdf = pd.DataFrame(conditional_results)
    cdf['wr_diff'] = cdf['hi_wr'] - cdf['lo_wr']
    cdf['pnl_diff'] = cdf['hi_avg_pnl'] - cdf['lo_avg_pnl']
    cdf['sharpe_diff'] = cdf['hi_sharpe'] - cdf['lo_sharpe']

    # Sort by WR difference (most improvement when inflow is high)
    cdf_sorted = cdf.sort_values('wr_diff', ascending=False)

    print("\nTop/Bottom by WR difference (High inflow WR - Low inflow WR):")
    print("-" * 100)
    for _, row in cdf_sorted.iterrows():
        direction = "BETTER when high" if row['wr_diff'] > 0 else "WORSE when high"
        print(f"  {row['ticker']:6s} | {row['feature']:35s} | "
              f"Hi WR: {row['hi_wr']:.1%} ({row['hi_n']:3d}) "
              f"Lo WR: {row['lo_wr']:.1%} ({row['lo_n']:3d}) "
              f"Diff: {row['wr_diff']:+.1%} | "
              f"Hi$: ${row['hi_avg_pnl']:+.0f}  Lo$: ${row['lo_avg_pnl']:+.0f} | {direction}")

# ============================================================
# 7. SPECIFIC DEEP DIVES
# ============================================================
print("\n" + "=" * 70)
print("DEEP DIVE: AI/SEMIS INFLOW vs NVDA & SMCI BPS OUTCOMES")
print("=" * 70)

for ticker in ['NVDA', 'SMCI']:
    col = 'theme_aiSemis_inflow_z20'
    sub = valid[(valid['ticker'] == ticker)].dropna(subset=[col])
    if len(sub) < 10:
        print(f"  {ticker}: insufficient data ({len(sub)} trades)")
        continue

    # Quintile analysis
    sub = sub.copy()
    sub['inflow_q'] = pd.qcut(sub[col], 5, labels=['Q1_low','Q2','Q3','Q4','Q5_high'], duplicates='drop')

    print(f"\n  {ticker} - AI/Semis Inflow Quintile Analysis ({len(sub)} trades):")
    print(f"  {'Quintile':10s} {'N':>5s} {'WR':>8s} {'Avg PnL':>10s} {'Med PnL':>10s} {'Avg Inflow Z':>12s}")
    for q in sub['inflow_q'].cat.categories:
        qdata = sub[sub['inflow_q'] == q]
        wr = (qdata['realized_pnl'] > 0).mean()
        avg = qdata['realized_pnl'].mean()
        med = qdata['realized_pnl'].median()
        avg_z = qdata[col].mean()
        print(f"  {q:10s} {len(qdata):5d} {wr:8.1%} ${avg:9.0f} ${med:9.0f} {avg_z:12.2f}")

    # Monotonicity test
    q_means = sub.groupby('inflow_q')['pnl_pct'].mean()
    ic_mono, _ = stats.spearmanr(range(len(q_means)), q_means.values)
    print(f"  Quintile monotonicity (rank corr): {ic_mono:+.3f}")

print("\n" + "=" * 70)
print("DEEP DIVE: ENERGY INFLOW vs OXY BPS OUTCOMES")
print("=" * 70)

col = 'theme_energy_inflow_z20'
sub = valid[(valid['ticker'] == 'OXY')].dropna(subset=[col])
if len(sub) >= 10:
    sub = sub.copy()
    sub['inflow_q'] = pd.qcut(sub[col], 3, labels=['Low','Mid','High'], duplicates='drop')
    print(f"\n  OXY - Energy Inflow Tercile Analysis ({len(sub)} trades):")
    print(f"  {'Tercile':10s} {'N':>5s} {'WR':>8s} {'Avg PnL':>10s} {'Med PnL':>10s}")
    for q in ['Low', 'Mid', 'High']:
        qdata = sub[sub['inflow_q'] == q]
        if len(qdata) == 0:
            continue
        wr = (qdata['realized_pnl'] > 0).mean()
        avg = qdata['realized_pnl'].mean()
        med = qdata['realized_pnl'].median()
        print(f"  {q:10s} {len(qdata):5d} {wr:8.1%} ${avg:9.0f} ${med:9.0f}")

# ============================================================
# 8. AGGREGATE THEME MOMENTUM TIMING OVERLAY
# ============================================================
print("\n" + "=" * 70)
print("AGGREGATE: THEME MOMENTUM AS POSITION SIZE SCALAR")
print("=" * 70)

# For each trade, compute the "relevant theme momentum" score
def get_relevant_theme_score(row, feature_type='inflow_z20'):
    """Get the theme inflow score most relevant to this ticker."""
    themes = TICKER_THEMES.get(row['ticker'], [])
    scores = []
    for t in themes:
        col = f'theme_{t}_{feature_type}'
        if col in row.index and pd.notna(row[col]):
            scores.append(row[col])
    return np.mean(scores) if scores else np.nan

valid_copy = valid.copy()
valid_copy['theme_score'] = valid_copy.apply(
    lambda r: get_relevant_theme_score(r), axis=1
)

has_score = valid_copy.dropna(subset=['theme_score'])
print(f"\nTrades with theme score: {len(has_score):,}")

if len(has_score) > 50:
    # Overall IC
    ic, pval = stats.spearmanr(has_score['theme_score'], has_score['pnl_pct'])
    print(f"Aggregate theme score IC vs PnL%: {ic:+.4f} (p={pval:.4f})")

    # Tercile analysis
    has_score = has_score.copy()
    has_score['score_tercile'] = pd.qcut(has_score['theme_score'], 3,
                                          labels=['Low', 'Mid', 'High'], duplicates='drop')

    print(f"\n{'Tercile':>10s} {'N':>6s} {'WR':>8s} {'Avg PnL':>10s} {'Med PnL':>10s} {'Sharpe':>8s}")
    print("-" * 60)
    for q in ['Low', 'Mid', 'High']:
        qdata = has_score[has_score['score_tercile'] == q]
        if len(qdata) == 0:
            continue
        wr = (qdata['realized_pnl'] > 0).mean()
        avg = qdata['realized_pnl'].mean()
        med = qdata['realized_pnl'].median()
        sharpe = qdata['pnl_pct'].mean() / qdata['pnl_pct'].std() * np.sqrt(252/7) if qdata['pnl_pct'].std() > 0 else 0
        print(f"  {q:>8s} {len(qdata):6d} {wr:8.1%} ${avg:9.0f} ${med:9.0f} {sharpe:8.2f}")

    # Year-by-year stability
    has_score['year'] = has_score['open_date'].dt.year
    print(f"\nYear-by-year IC stability:")
    for year in sorted(has_score['year'].unique()):
        ydata = has_score[has_score['year'] == year]
        if len(ydata) < 20:
            continue
        ic_y, pval_y = stats.spearmanr(ydata['theme_score'], ydata['pnl_pct'])
        sig = "***" if pval_y < 0.01 else "**" if pval_y < 0.05 else "*" if pval_y < 0.10 else ""
        print(f"  {year}: IC={ic_y:+.4f} (p={pval_y:.4f}, n={len(ydata)}) {sig}")

# ============================================================
# 9. PROPOSED TIMING OVERLAY (if signal exists)
# ============================================================
print("\n" + "=" * 70)
print("PROPOSED TIMING OVERLAY DESIGN")
print("=" * 70)

print("""
Based on the analysis above, here is the proposed overlay logic:

TIMING OVERLAY (position size scalar, NOT universe change):
- For each ticker at trade open, look up its relevant theme inflow z-score
- If theme_inflow_z20 > +1.0 (strong inflow): scale position to 1.5x
- If theme_inflow_z20 between -0.5 and +1.0: no change (1.0x)
- If theme_inflow_z20 < -0.5 (weak/outflow): scale position to 0.5x

This preserves the GA-optimized 20-ticker universe while
adjusting conviction based on sector flow momentum.
""")

# ============================================================
# 10. SIMULATE THE OVERLAY
# ============================================================
print("=" * 70)
print("SIMULATED OVERLAY BACKTEST")
print("=" * 70)

if len(has_score) > 50:
    # Simulate with and without overlay
    has_score = has_score.copy()

    # Position scalar based on theme score
    conditions = [
        has_score['theme_score'] > 1.0,
        has_score['theme_score'] < -0.5,
    ]
    choices = [1.5, 0.5]
    has_score['scalar'] = np.select(conditions, choices, default=1.0)

    has_score['pnl_base'] = has_score['realized_pnl']
    has_score['pnl_overlay'] = has_score['realized_pnl'] * has_score['scalar']

    # Compare
    base_total = has_score['pnl_base'].sum()
    overlay_total = has_score['pnl_overlay'].sum()

    base_sharpe = has_score['pnl_base'].mean() / has_score['pnl_base'].std() * np.sqrt(252/7)
    overlay_sharpe = has_score['pnl_overlay'].mean() / has_score['pnl_overlay'].std() * np.sqrt(252/7)

    base_wr = (has_score['pnl_base'] > 0).mean()
    overlay_wr = (has_score['pnl_overlay'] > 0).mean()  # WR unchanged since we don't skip

    # Max DD comparison
    base_cum = has_score['pnl_base'].cumsum()
    overlay_cum = has_score['pnl_overlay'].cumsum()
    base_dd = (base_cum - base_cum.cummax()).min()
    overlay_dd = (overlay_cum - overlay_cum.cummax()).min()

    print(f"\n{'Metric':25s} {'Baseline':>12s} {'With Overlay':>12s} {'Change':>10s}")
    print("-" * 65)
    print(f"{'Total PnL':25s} ${base_total:>11,.0f} ${overlay_total:>11,.0f} {(overlay_total/base_total-1):>+9.1%}")
    print(f"{'Annualized Sharpe':25s} {base_sharpe:>12.2f} {overlay_sharpe:>12.2f} {overlay_sharpe-base_sharpe:>+10.2f}")
    print(f"{'Win Rate':25s} {base_wr:>12.1%} {overlay_wr:>12.1%} {'(unchanged)':>10s}")
    print(f"{'Max Drawdown':25s} ${base_dd:>11,.0f} ${overlay_dd:>11,.0f} {(overlay_dd/base_dd-1):>+9.1%}")
    print(f"{'Avg Position Scale':25s} {'1.00':>12s} {has_score['scalar'].mean():>12.2f} {'':>10s}")

    # Scalar distribution
    print(f"\n  Scalar distribution:")
    print(f"    1.5x (high inflow):  {(has_score['scalar']==1.5).sum():5d} trades ({(has_score['scalar']==1.5).mean():.1%})")
    print(f"    1.0x (neutral):      {(has_score['scalar']==1.0).sum():5d} trades ({(has_score['scalar']==1.0).mean():.1%})")
    print(f"    0.5x (low inflow):   {(has_score['scalar']==0.5).sum():5d} trades ({(has_score['scalar']==0.5).mean():.1%})")

    # Regime check (R1): does the overlay maintain regime symmetry?
    print("\n  Regime symmetry check (R1 gate):")
    # We need SPY/ES data for regime classification - approximate with year
    # This is a simplified check
    has_score['month'] = has_score['open_date'].dt.to_period('M')
    monthly_base = has_score.groupby('month')['pnl_base'].sum()
    monthly_overlay = has_score.groupby('month')['pnl_overlay'].sum()

    base_neg_months = (monthly_base < 0).sum()
    overlay_neg_months = (monthly_overlay < 0).sum()
    print(f"    Base negative months: {base_neg_months}/{len(monthly_base)}")
    print(f"    Overlay negative months: {overlay_neg_months}/{len(monthly_overlay)}")

    base_worst = monthly_base.min()
    overlay_worst = monthly_overlay.min()
    print(f"    Base worst month: ${base_worst:,.0f}")
    print(f"    Overlay worst month: ${overlay_worst:,.0f}")

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)
