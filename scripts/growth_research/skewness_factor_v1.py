#!/usr/bin/env python3
"""
Skewness Factor Strategy v1
============================
OBSERVATION (from cross_sectional_anomaly_scanner, HC #735):
  Positive-skew stocks outperform negative-skew stocks by ~10% annually.
  This CONTRADICTS academic orthodoxy (Brunnermeier, Barberis, etc.) which predicts
  investors overpay for lottery-like (positive-skew) payoffs, so negative-skew stocks
  should earn a premium.

HYPOTHESIS:
  Rolling 63d return skewness captures a persistent cross-sectional factor.
  Stocks with positive skewness (big right-tail moves) may be experiencing
  fundamental momentum / breakout tendencies that drive future returns.

STRATEGY:
  - Universe: S&P 500 (~400-500 stocks via yfinance)
  - Compute rolling 63d return skewness monthly
  - Long top quintile (most positive skew), short bottom quintile (most negative skew)
  - Equal-weight, monthly rebalance
  - 10bps round-trip transaction cost

VALIDATION (HC #428):
  R1: Regime-agnostic — bull/bear/flat, regime gap < 0.50
  R2: Permutation test — 200 shuffles, real Sharpe must beat p95 of null
  Sub-period stability: 3 equal periods, all profitable
  Sector neutrality: does edge survive within-sector ranking?

OUTPUT: /home/jupiter/Lvl3Quant/output/skewness_factor_v1/
"""

import os, sys, json, warnings, time
import numpy as np
import pandas as pd
from datetime import datetime
from scipy import stats as scipy_stats
warnings.filterwarnings('ignore')

sys.path.insert(0, '/home/jupiter/Lvl3Quant')

# ─── Config ───
START_DATE = '2014-01-01'  # Extra year for skewness warmup
END_DATE = '2026-07-01'
SKEW_LOOKBACK = 63  # Quarterly rolling skewness
REBALANCE_FREQ = 'M'  # Monthly rebalance
QUINTILE_FRAC = 0.20  # Top/bottom 20%
COST_BPS = 10  # 10bps round-trip
N_PERMS = 200
OUTPUT_DIR = '/home/jupiter/Lvl3Quant/output/skewness_factor_v1'
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"{'='*70}")
print(f"SKEWNESS FACTOR STRATEGY v1")
print(f"Observation-first research (HC #735)")
print(f"{'='*70}")

# ═══════════════════════════════════════════════════════════════════════
# PHASE 0: DATA ACQUISITION
# ═══════════════════════════════════════════════════════════════════════

def get_sp500_tickers():
    """Get S&P 500 tickers from Wikipedia or fallback list."""
    import urllib.request

    # Try Wikipedia with proper user agent
    try:
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        html = urllib.request.urlopen(req, timeout=15).read().decode()
        tables = pd.read_html(html)
        df = tables[0]
        tickers = df['Symbol'].tolist()
        tickers = [t.replace('.', '-') for t in tickers]
        sectors = dict(zip(
            [t.replace('.', '-') for t in df['Symbol'].tolist()],
            df['GICS Sector'].tolist()
        ))
        print(f"  Loaded {len(tickers)} S&P 500 tickers from Wikipedia")
        return tickers, sectors
    except Exception as e:
        print(f"  Wikipedia fetch failed: {e}")

    # Fallback: large-cap tickers covering all sectors (~450 stocks)
    print("  Using hardcoded large-cap fallback list (~450 stocks)")
    fallback = [
        # Technology
        'AAPL','MSFT','GOOGL','GOOG','META','NVDA','AVGO','ADBE','CRM','CSCO',
        'ACN','ORCL','TXN','QCOM','AMD','INTC','INTU','IBM','NOW','AMAT',
        'ADI','MU','LRCX','KLAC','SNPS','CDNS','MCHP','TEL','FTNT','PANW',
        'NXPI','MPWR','ON','KEYS','SWKS','AKAM','FFIV','JNPR','HPQ','HPE',
        'CTSH','IT','EPAM','GEN','LDOS','PAYC','TRMB','TYL','ZBRA','CDW',
        # Healthcare
        'UNH','JNJ','LLY','PFE','ABBV','MRK','TMO','ABT','DHR','BMY',
        'AMGN','MDT','ISRG','ELV','CI','SYK','GILD','VRTX','REGN','BSX',
        'ZTS','BDX','HCA','IDXX','IQV','EW','MTD','A','DXCM','BAX',
        'HOLX','RMD','ALGN','TECH','PODD','INCY','MRNA','BIIB','MOH','CNC',
        'HUM','ILMN','COO','XRAY','DGX','LH','STE','PKI','WAT','BIO',
        # Financials
        'BRK-B','JPM','V','MA','BAC','WFC','GS','MS','SPGI','BLK',
        'AXP','C','SCHW','CB','PGR','MMC','AON','ICE','CME','MCO',
        'MET','AIG','PRU','TRV','AFL','ALL','AJG','MSCI','FIS','FISV',
        'COF','USB','PNC','TFC','FITB','MTB','HBAN','RF','CFG','KEY',
        'DFS','SYF','CINF','NDAQ','BEN','IVZ','TROW','WRB','GL','L',
        # Consumer Discretionary
        'AMZN','TSLA','HD','MCD','NKE','LOW','SBUX','TJX','BKNG','CMG',
        'MAR','HLT','ORLY','AZO','ROST','DHI','LEN','PHM','NVR','GPC',
        'EBAY','ETSY','APTV','GM','F','YUM','DPZ','DARDEN','POOL','BBY',
        'KMX','DRI','LKQ','GRMN','MGM','WYNN','CZR','LVS','RCL','CCL',
        'NCLH','ULTA','TPR','RL','PVH','HAS','DECK','LULU','EXPE','ABNB',
        # Consumer Staples
        'PG','KO','PEP','COST','WMT','PM','MO','MDLZ','CL','EL',
        'STZ','KMB','GIS','SJM','K','HSY','CPB','MKC','HRL','TSN',
        'CAG','CLX','CHD','WBA','KR','SYY','ADM','BG','TAP','SAM',
        'MNST','KDP','CAGR','LAMB','USFD',
        # Industrials
        'CAT','UNP','UPS','HON','RTX','BA','DE','LMT','GE','MMM',
        'GD','NOC','ITW','EMR','ETN','PH','ROK','CMI','PCAR','FDX',
        'CSX','NSC','WM','RSG','VRSK','FAST','CTAS','OTIS','CARR','IR',
        'DOV','AME','SWK','GNRC','XYL','TT','A','PWR','WAB','J',
        'MAS','ALLE','AOS','NDSN','RHI','TXT','LDOS','LHX','HII','CW',
        # Energy
        'XOM','CVX','COP','EOG','SLB','MPC','PSX','VLO','OXY','PXD',
        'DVN','HES','HAL','FANG','BKR','CTRA','MRO','APA','TRGP','OKE',
        'WMB','KMI','ET','LNG','DINO',
        # Materials
        'LIN','APD','SHW','ECL','FCX','NUE','NEM','DOW','DD','PPG',
        'VMC','MLM','CF','MOS','FMC','ALB','CE','IFF','EMN','BALL',
        'PKG','IP','SEE','AVY','RPM','AMCR',
        # Utilities
        'NEE','SO','DUK','D','SRE','AEP','EXC','XEL','WEC','ES',
        'ED','PEG','EIX','AWK','DTE','AEE','CMS','CNP','ATO','EVRG',
        'NI','PPL','FE','LNT','PNW','NRG',
        # Real Estate
        'PLD','AMT','CCI','EQIX','PSA','SPG','O','WELL','DLR','VICI',
        'ARE','AVB','EQR','MAA','UDR','ESS','CPT','REG','KIM','FRT',
        'BXP','SLG','VNO','HST','PEAK','INVH',
        # Communication
        'DIS','CMCSA','NFLX','T','VZ','CHTR','TMUS','EA','TTWO','ATVI',
        'WBD','PARA','FOX','FOXA','NWS','NWSA','IPG','OMC','MTCH','ZG',
    ]
    # Sector mapping for fallback
    sector_assignments = {}
    sector_names = [
        ('Technology', fallback[:50]),
        ('Health Care', fallback[50:100]),
        ('Financials', fallback[100:150]),
        ('Consumer Discretionary', fallback[150:200]),
        ('Consumer Staples', fallback[200:235]),
        ('Industrials', fallback[235:285]),
        ('Energy', fallback[285:310]),
        ('Materials', fallback[310:336]),
        ('Utilities', fallback[336:362]),
        ('Real Estate', fallback[362:388]),
        ('Communication Services', fallback[388:]),
    ]
    for sec_name, sec_tickers in sector_names:
        for t in sec_tickers:
            sector_assignments[t] = sec_name

    return fallback, sector_assignments


def download_price_data(tickers, start, end):
    """Download daily prices for all tickers via yfinance."""
    import yfinance as yf

    print(f"\n  Downloading {len(tickers)} tickers from {start} to {end}...")
    # Download in batches to avoid timeouts
    batch_size = 50
    all_data = {}
    failed = []

    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i+batch_size]
        batch_str = ' '.join(batch)
        try:
            data = yf.download(batch_str, start=start, end=end,
                             auto_adjust=True, progress=False, threads=True)
            if 'Close' in data.columns.get_level_values(0) if isinstance(data.columns, pd.MultiIndex) else 'Close' in data.columns:
                if isinstance(data.columns, pd.MultiIndex):
                    closes = data['Close']
                else:
                    # Single ticker
                    closes = data[['Close']]
                    closes.columns = batch
                for col in closes.columns:
                    if closes[col].notna().sum() > 252:  # At least 1 year of data
                        all_data[col] = closes[col]
        except Exception as e:
            failed.extend(batch)

        if (i // batch_size) % 5 == 0:
            print(f"    Batch {i//batch_size + 1}/{(len(tickers)-1)//batch_size + 1}: "
                  f"{len(all_data)} tickers loaded")

    prices = pd.DataFrame(all_data)
    print(f"  Downloaded {prices.shape[1]} tickers with {prices.shape[0]} trading days")
    print(f"  Date range: {prices.index[0].strftime('%Y-%m-%d')} to {prices.index[-1].strftime('%Y-%m-%d')}")
    if failed:
        print(f"  Failed tickers: {len(failed)}")
    return prices


# ─── Load or cache data ───
CACHE_FILE = os.path.join(OUTPUT_DIR, '_price_cache.parquet')

if os.path.exists(CACHE_FILE):
    print("\n  Loading cached price data...")
    prices = pd.read_parquet(CACHE_FILE)
    tickers_list, sector_map = get_sp500_tickers()
    if sector_map is None:
        sector_map = {}
    print(f"  Loaded {prices.shape[1]} tickers, {prices.shape[0]} days")
else:
    tickers_list, sector_map = get_sp500_tickers()
    if tickers_list is None:
        print("FATAL: Cannot get S&P 500 tickers")
        sys.exit(1)
    if sector_map is None:
        sector_map = {}
    prices = download_price_data(tickers_list, START_DATE, END_DATE)
    prices.to_parquet(CACHE_FILE)
    print(f"  Cached to {CACHE_FILE}")

# Compute daily returns
returns = prices.pct_change().dropna(how='all')
print(f"  Returns matrix: {returns.shape[0]} days x {returns.shape[1]} stocks")

# ═══════════════════════════════════════════════════════════════════════
# PHASE 1: DEEP OBSERVATION — Skewness characteristics
# ═══════════════════════════════════════════════════════════════════════

print(f"\n{'='*70}")
print(f"PHASE 1: DEEP OBSERVATION")
print(f"{'='*70}")

# 1a. Compute rolling skewness for every stock
print("\n  Computing rolling 63d skewness...")
rolling_skew = returns.rolling(SKEW_LOOKBACK, min_periods=50).skew()

# Drop warmup period
rolling_skew = rolling_skew.loc['2015-01-01':]
returns_trimmed = returns.loc['2015-01-01':]

print(f"  Skewness matrix: {rolling_skew.shape[0]} days x {rolling_skew.shape[1]} stocks")

# 1b. Distribution of skewness across stocks
latest_skew = rolling_skew.iloc[-1].dropna()
print(f"\n  SKEWNESS DISTRIBUTION (latest cross-section, {len(latest_skew)} stocks):")
print(f"    Mean:   {latest_skew.mean():.3f}")
print(f"    Median: {latest_skew.median():.3f}")
print(f"    Std:    {latest_skew.std():.3f}")
print(f"    Min:    {latest_skew.min():.3f}")
print(f"    Max:    {latest_skew.max():.3f}")
print(f"    % positive: {(latest_skew > 0).mean()*100:.1f}%")

# 1c. Persistence of skewness rankings
print("\n  SKEWNESS PERSISTENCE (month-to-month rank autocorrelation):")
# Sample at monthly frequency
monthly_dates = rolling_skew.resample('M').last().index
monthly_skew = rolling_skew.loc[rolling_skew.index.isin(monthly_dates) |
                                 rolling_skew.index.to_series().dt.is_month_end]
# Use actual month-end resampling
monthly_skew = rolling_skew.resample('M').last()

rank_autocorrs = []
for i in range(1, len(monthly_skew)):
    prev = monthly_skew.iloc[i-1].dropna()
    curr = monthly_skew.iloc[i].dropna()
    common = prev.index.intersection(curr.index)
    if len(common) > 50:
        r_prev = prev[common].rank()
        r_curr = curr[common].rank()
        ac = r_prev.corr(r_curr)
        rank_autocorrs.append(ac)

rank_ac_mean = np.mean(rank_autocorrs)
rank_ac_std = np.std(rank_autocorrs)
print(f"    Mean rank autocorrelation:  {rank_ac_mean:.3f} +/- {rank_ac_std:.3f}")
print(f"    Interpretation: {'PERSISTENT' if rank_ac_mean > 0.5 else 'MODERATE' if rank_ac_mean > 0.3 else 'WEAK'}")
print(f"    (1.0 = perfectly persistent, 0.0 = random)")

# 1d. Sector patterns
if sector_map:
    print("\n  SECTOR PATTERNS (median skewness by sector, latest):")
    sector_skew = {}
    for ticker in latest_skew.index:
        sec = sector_map.get(ticker, 'Unknown')
        if sec not in sector_skew:
            sector_skew[sec] = []
        sector_skew[sec].append(latest_skew[ticker])

    for sec in sorted(sector_skew.keys()):
        vals = sector_skew[sec]
        print(f"    {sec:35s}: median={np.median(vals):.3f}, n={len(vals)}")

# 1e. Time-series of cross-sectional mean skewness
cs_mean_skew = rolling_skew.mean(axis=1).dropna()
cs_std_skew = rolling_skew.std(axis=1).dropna()
print(f"\n  TIME-SERIES OF CROSS-SECTIONAL MEAN SKEWNESS:")
print(f"    Overall mean: {cs_mean_skew.mean():.3f}")
print(f"    Overall std:  {cs_mean_skew.std():.3f}")
print(f"    Min period:   {cs_mean_skew.idxmin().strftime('%Y-%m-%d')} ({cs_mean_skew.min():.3f})")
print(f"    Max period:   {cs_mean_skew.idxmax().strftime('%Y-%m-%d')} ({cs_mean_skew.max():.3f})")


# ═══════════════════════════════════════════════════════════════════════
# PHASE 2: STRATEGY — Monthly long-short skewness portfolio
# ═══════════════════════════════════════════════════════════════════════

print(f"\n{'='*70}")
print(f"PHASE 2: STRATEGY CONSTRUCTION")
print(f"{'='*70}")


def build_skewness_strategy(skew_df, ret_df, quintile=0.20, cost_bps=10,
                             sector_neutral=False, sector_mapping=None):
    """
    Build monthly long-short portfolio based on trailing skewness.

    Long top quintile (most positive skew), short bottom quintile.
    Equal-weight within quintile, monthly rebalance.

    Returns: DataFrame with strategy returns and metadata.
    """
    cost_frac = cost_bps / 10000.0

    # Get monthly rebalance dates
    monthly_idx = skew_df.resample('M').last().index
    # Use last trading day of each month
    rebal_dates = []
    for m in monthly_idx:
        mask = skew_df.index <= m
        if mask.any():
            rebal_dates.append(skew_df.index[mask][-1])
    rebal_dates = sorted(set(rebal_dates))

    strategy_returns = []
    long_returns_list = []
    short_returns_list = []
    turnover_list = []
    n_long_list = []
    n_short_list = []

    prev_long = set()
    prev_short = set()

    for i in range(len(rebal_dates) - 1):
        rebal_date = rebal_dates[i]
        next_rebal = rebal_dates[i+1]

        # Get skewness on rebalance date
        skew_vals = skew_df.loc[rebal_date].dropna()
        if len(skew_vals) < 50:
            continue

        if sector_neutral and sector_mapping:
            # Within-sector ranking
            long_tickers = []
            short_tickers = []
            for sec in set(sector_mapping.values()):
                sec_tickers = [t for t in skew_vals.index if sector_mapping.get(t) == sec]
                if len(sec_tickers) < 10:
                    continue
                sec_skew = skew_vals[sec_tickers].sort_values()
                n_q = max(1, int(len(sec_skew) * quintile))
                short_tickers.extend(sec_skew.index[:n_q].tolist())
                long_tickers.extend(sec_skew.index[-n_q:].tolist())
        else:
            # Cross-sectional ranking
            sorted_skew = skew_vals.sort_values()
            n_q = max(1, int(len(sorted_skew) * quintile))
            short_tickers = sorted_skew.index[:n_q].tolist()  # Bottom quintile (most negative skew)
            long_tickers = sorted_skew.index[-n_q:].tolist()   # Top quintile (most positive skew)

        # Get daily returns for the holding period
        hold_mask = (ret_df.index > rebal_date) & (ret_df.index <= next_rebal)
        hold_rets = ret_df.loc[hold_mask]

        if len(hold_rets) == 0:
            continue

        # Compute turnover
        new_long = set(long_tickers)
        new_short = set(short_tickers)
        if prev_long:
            long_turnover = 1.0 - len(new_long & prev_long) / max(len(new_long), 1)
            short_turnover = 1.0 - len(new_short & prev_short) / max(len(new_short), 1)
        else:
            long_turnover = 1.0
            short_turnover = 1.0
        avg_turnover = (long_turnover + short_turnover) / 2
        prev_long = new_long
        prev_short = new_short

        # Equal-weight daily returns
        for date, row in hold_rets.iterrows():
            long_avail = [t for t in long_tickers if pd.notna(row.get(t, np.nan))]
            short_avail = [t for t in short_tickers if pd.notna(row.get(t, np.nan))]

            if len(long_avail) < 5 or len(short_avail) < 5:
                continue

            long_ret = row[long_avail].mean()
            short_ret = row[short_avail].mean()
            ls_ret = long_ret - short_ret  # Long-short

            # Apply transaction cost (prorated daily, applied at rebalance)
            # Turnover cost applied on rebalance day only
            daily_cost = 0
            if date == hold_rets.index[0]:
                daily_cost = avg_turnover * cost_frac * 2  # Both sides

            strategy_returns.append({
                'date': date,
                'ls_return': ls_ret - daily_cost,
                'long_return': long_ret,
                'short_return': short_ret,
                'turnover': avg_turnover if date == hold_rets.index[0] else 0,
                'n_long': len(long_avail),
                'n_short': len(short_avail),
            })

    df = pd.DataFrame(strategy_returns).set_index('date')
    return df


# ─── Run base strategy ───
print("\n  Building base long-short skewness strategy...")
strat = build_skewness_strategy(rolling_skew, returns_trimmed,
                                 quintile=QUINTILE_FRAC, cost_bps=COST_BPS)
print(f"  Strategy period: {strat.index[0].strftime('%Y-%m-%d')} to {strat.index[-1].strftime('%Y-%m-%d')}")
print(f"  Total trading days: {len(strat)}")
print(f"  Avg stocks long: {strat['n_long'].mean():.0f}, short: {strat['n_short'].mean():.0f}")

# ─── Compute strategy metrics ───
def compute_metrics(returns_series, label="Strategy"):
    """Compute key performance metrics."""
    r = returns_series.dropna()
    if len(r) < 20:
        return {'label': label, 'sharpe': 0, 'sortino': 0, 'cagr': 0, 'pf': 0,
                'wr': 0, 'max_dd': 0, 'n_days': len(r)}

    ann_factor = 252
    mu = r.mean() * ann_factor
    sigma = r.std() * np.sqrt(ann_factor)
    sharpe = mu / sigma if sigma > 0 else 0

    downside = r[r < 0].std() * np.sqrt(ann_factor)
    sortino = mu / downside if downside > 0 else 0

    # CAGR
    cum = (1 + r).cumprod()
    years = len(r) / 252
    cagr = (cum.iloc[-1] ** (1/years) - 1) if years > 0 and cum.iloc[-1] > 0 else 0

    # Profit factor
    gross_profit = r[r > 0].sum()
    gross_loss = abs(r[r < 0].sum())
    pf = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    # Win rate (monthly)
    monthly = r.resample('M').sum()
    wr = (monthly > 0).mean()

    # Max drawdown
    cum_max = cum.cummax()
    dd = (cum - cum_max) / cum_max
    max_dd = dd.min()

    return {
        'label': label,
        'sharpe': sharpe,
        'sortino': sortino,
        'cagr': cagr * 100,
        'pf': pf,
        'wr': wr * 100,
        'max_dd': max_dd * 100,
        'n_days': len(r),
        'ann_return': mu * 100,
        'ann_vol': sigma * 100,
    }


base_metrics = compute_metrics(strat['ls_return'], "Skewness L/S (base)")
print(f"\n  BASE STRATEGY METRICS:")
print(f"    Sharpe:      {base_metrics['sharpe']:.3f}")
print(f"    Sortino:     {base_metrics['sortino']:.3f}")
print(f"    CAGR:        {base_metrics['cagr']:.2f}%")
print(f"    Ann Return:  {base_metrics['ann_return']:.2f}%")
print(f"    Ann Vol:     {base_metrics['ann_vol']:.2f}%")
print(f"    Profit Factor: {base_metrics['pf']:.2f}")
print(f"    Monthly WR:  {base_metrics['wr']:.1f}%")
print(f"    Max DD:      {base_metrics['max_dd']:.2f}%")

# Long-only and short-only legs
long_metrics = compute_metrics(strat['long_return'], "Long leg only")
short_metrics = compute_metrics(-strat['short_return'], "Short leg (inverted)")
print(f"\n  LONG LEG (top quintile pos-skew):")
print(f"    Sharpe: {long_metrics['sharpe']:.3f}, CAGR: {long_metrics['cagr']:.2f}%")
print(f"  SHORT LEG (bottom quintile neg-skew, inverted for comparison):")
print(f"    Sharpe: {short_metrics['sharpe']:.3f}, CAGR: {short_metrics['cagr']:.2f}%")


# ═══════════════════════════════════════════════════════════════════════
# PHASE 3: VALIDATION
# ═══════════════════════════════════════════════════════════════════════

print(f"\n{'='*70}")
print(f"PHASE 3: VALIDATION (HC #428)")
print(f"{'='*70}")

# ─── 3a. Regime analysis ───
print("\n  3a. REGIME ANALYSIS (SPY > 200 SMA = bull, < = bear):")

# Get SPY data for regime classification
try:
    import yfinance as yf
    spy = yf.download('SPY', start=START_DATE, end=END_DATE, auto_adjust=True, progress=False)
    spy_close = spy['Close'].squeeze()
    spy_sma200 = spy_close.rolling(200).mean()
    spy_regime = pd.Series(index=spy_close.index, dtype=str)
    spy_regime[spy_close > spy_sma200] = 'bull'
    spy_regime[spy_close <= spy_sma200] = 'bear'
    spy_regime = spy_regime.dropna()
except:
    # Fallback: use market return
    spy_close = returns_trimmed.mean(axis=1).cumsum()
    spy_sma200 = spy_close.rolling(200).mean()
    spy_regime = pd.Series('bull', index=spy_close.index)
    spy_regime[spy_close <= spy_sma200] = 'bear'

# Align regime with strategy
strat_regime = spy_regime.reindex(strat.index).fillna(method='ffill')

bull_mask = strat_regime == 'bull'
bear_mask = strat_regime == 'bear'

bull_metrics = compute_metrics(strat.loc[bull_mask, 'ls_return'], "Bull regime")
bear_metrics = compute_metrics(strat.loc[bear_mask, 'ls_return'], "Bear regime")

print(f"    Bull days: {bull_mask.sum()}, Bear days: {bear_mask.sum()}")
print(f"    Bull Sharpe: {bull_metrics['sharpe']:.3f}, Bear Sharpe: {bear_metrics['sharpe']:.3f}")
print(f"    Bull CAGR:   {bull_metrics['cagr']:.2f}%, Bear CAGR: {bear_metrics['cagr']:.2f}%")

# Regime gap check (HC #428 R1)
max_sharpe = max(abs(bull_metrics['sharpe']), abs(bear_metrics['sharpe']))
if max_sharpe > 0:
    regime_gap = abs(bull_metrics['sharpe'] - bear_metrics['sharpe']) / max_sharpe
else:
    regime_gap = 0
regime_pass = regime_gap < 0.50
print(f"    Regime gap: {regime_gap:.3f} {'PASS' if regime_pass else 'FAIL'} (threshold: 0.50)")

# ─── 3b. Permutation test ───
print(f"\n  3b. PERMUTATION TEST ({N_PERMS} shuffles):")
print(f"      (Shuffling skewness rankings to create null distribution)")

real_sharpe = base_metrics['sharpe']
null_sharpes = []

t0 = time.time()
for p in range(N_PERMS):
    # Shuffle the skewness rankings at each rebalance date
    shuffled_skew = rolling_skew.copy()
    for col_idx in range(0, len(shuffled_skew), 21):  # Shuffle monthly
        if col_idx < len(shuffled_skew):
            row = shuffled_skew.iloc[min(col_idx, len(shuffled_skew)-1)]
            valid = row.dropna()
            if len(valid) > 0:
                shuffled_vals = valid.values.copy()
                np.random.shuffle(shuffled_vals)
                shuffled_skew.iloc[min(col_idx, len(shuffled_skew)-1),
                                   shuffled_skew.columns.get_indexer(valid.index)] = shuffled_vals

    perm_strat = build_skewness_strategy(shuffled_skew, returns_trimmed,
                                          quintile=QUINTILE_FRAC, cost_bps=COST_BPS)
    if len(perm_strat) > 20:
        perm_metrics = compute_metrics(perm_strat['ls_return'], f"Perm {p}")
        null_sharpes.append(perm_metrics['sharpe'])

    if (p+1) % 50 == 0:
        elapsed = time.time() - t0
        print(f"      Completed {p+1}/{N_PERMS} permutations ({elapsed:.0f}s)")

null_sharpes = np.array(null_sharpes)
perm_pval = (null_sharpes >= real_sharpe).mean()
perm_p95 = np.percentile(null_sharpes, 95)
perm_pass = real_sharpe > perm_p95
print(f"    Real Sharpe:     {real_sharpe:.3f}")
print(f"    Null mean:       {null_sharpes.mean():.3f} +/- {null_sharpes.std():.3f}")
print(f"    Null 95th pctl:  {perm_p95:.3f}")
print(f"    p-value:         {perm_pval:.3f}")
print(f"    Result:          {'PASS' if perm_pass else 'FAIL'} (real > null p95)")

# ─── 3c. Sub-period stability ───
print(f"\n  3c. SUB-PERIOD STABILITY (3 equal periods):")
n = len(strat)
period_size = n // 3
periods = [
    strat.iloc[:period_size],
    strat.iloc[period_size:2*period_size],
    strat.iloc[2*period_size:]
]
sub_pass = True
for i, period in enumerate(periods):
    m = compute_metrics(period['ls_return'], f"Period {i+1}")
    profitable = m['cagr'] > 0
    if not profitable:
        sub_pass = False
    print(f"    Period {i+1} ({period.index[0].strftime('%Y-%m')}-{period.index[-1].strftime('%Y-%m')}): "
          f"Sharpe={m['sharpe']:.3f}, CAGR={m['cagr']:.2f}%, "
          f"{'PROFITABLE' if profitable else 'LOSS'}")
print(f"    Result: {'PASS' if sub_pass else 'FAIL'} (all periods profitable)")

# ─── 3d. Sector neutrality ───
print(f"\n  3d. SECTOR NEUTRALITY TEST:")
if sector_map and len(sector_map) > 0:
    sn_strat = build_skewness_strategy(rolling_skew, returns_trimmed,
                                        quintile=QUINTILE_FRAC, cost_bps=COST_BPS,
                                        sector_neutral=True, sector_mapping=sector_map)
    sn_metrics = compute_metrics(sn_strat['ls_return'], "Sector-neutral")
    print(f"    Sector-neutral Sharpe: {sn_metrics['sharpe']:.3f} (vs base {base_metrics['sharpe']:.3f})")
    print(f"    Sector-neutral CAGR:   {sn_metrics['cagr']:.2f}% (vs base {base_metrics['cagr']:.2f}%)")
    sn_survives = sn_metrics['sharpe'] > 0
    print(f"    Edge survives within-sector: {'YES' if sn_survives else 'NO'}")
else:
    print(f"    Sector map not available, skipping.")
    sn_metrics = None
    sn_survives = None

# ─── 3e. Quintile spread analysis ───
print(f"\n  3e. QUINTILE SPREAD (monotonicity check):")
n_quintiles = 5
monthly_skew_signal = rolling_skew.resample('M').last()
monthly_fwd_ret = returns_trimmed.resample('M').sum().shift(-1)  # Next month's return

quintile_rets = {q: [] for q in range(1, n_quintiles+1)}
for date in monthly_skew_signal.index:
    if date not in monthly_fwd_ret.index:
        continue
    skew_vals = monthly_skew_signal.loc[date].dropna()
    fwd_vals = monthly_fwd_ret.loc[date]
    common = skew_vals.index.intersection(fwd_vals.dropna().index)
    if len(common) < 50:
        continue

    ranked = skew_vals[common].rank(pct=True)
    for q in range(1, n_quintiles+1):
        lo = (q-1) / n_quintiles
        hi = q / n_quintiles
        mask = (ranked > lo) & (ranked <= hi) if q > 1 else (ranked <= hi)
        if mask.sum() > 0:
            quintile_rets[q].append(fwd_vals[common[mask]].mean())

print(f"    Quintile  |  Ann Return  |  Sharpe-like")
print(f"    ----------|--------------|-------------")
for q in range(1, n_quintiles+1):
    rets = np.array(quintile_rets[q])
    ann_ret = rets.mean() * 12 * 100 if len(rets) > 0 else 0
    sharpe_like = (rets.mean() / rets.std() * np.sqrt(12)) if len(rets) > 1 and rets.std() > 0 else 0
    label = "SHORT" if q == 1 else "LONG" if q == n_quintiles else ""
    print(f"    Q{q} {label:5s}  |  {ann_ret:+.2f}%     |  {sharpe_like:.3f}")

# Check monotonicity
q_means = [np.mean(quintile_rets[q]) for q in range(1, n_quintiles+1)]
monotonic = all(q_means[i] <= q_means[i+1] for i in range(len(q_means)-1))
print(f"    Monotonic spread: {'YES' if monotonic else 'NO'}")


# ═══════════════════════════════════════════════════════════════════════
# PHASE 4: PLOTS & OUTPUT
# ═══════════════════════════════════════════════════════════════════════

print(f"\n{'='*70}")
print(f"PHASE 4: OUTPUT")
print(f"{'='*70}")

# ─── Equity curve ───
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

fig, axes = plt.subplots(3, 1, figsize=(14, 12))

# Panel 1: Cumulative return
cum_ret = (1 + strat['ls_return']).cumprod()
axes[0].plot(cum_ret.index, cum_ret.values, 'b-', linewidth=1.5, label='L/S Skewness Factor')
axes[0].axhline(y=1, color='gray', linestyle='--', alpha=0.5)
axes[0].set_title(f"Skewness Factor Strategy — Sharpe={base_metrics['sharpe']:.2f}, "
                  f"CAGR={base_metrics['cagr']:.1f}%", fontsize=14, fontweight='bold')
axes[0].set_ylabel('Cumulative Return')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Panel 2: Drawdown
dd = (cum_ret - cum_ret.cummax()) / cum_ret.cummax()
axes[1].fill_between(dd.index, dd.values, 0, color='red', alpha=0.3)
axes[1].set_title('Drawdown')
axes[1].set_ylabel('Drawdown %')
axes[1].grid(True, alpha=0.3)

# Panel 3: Rolling 12m Sharpe
rolling_sharpe = strat['ls_return'].rolling(252).mean() / strat['ls_return'].rolling(252).std() * np.sqrt(252)
axes[2].plot(rolling_sharpe.index, rolling_sharpe.values, 'g-', linewidth=1)
axes[2].axhline(y=0, color='red', linestyle='--', alpha=0.5)
axes[2].set_title('Rolling 12-Month Sharpe Ratio')
axes[2].set_ylabel('Sharpe')
axes[2].grid(True, alpha=0.3)

for ax in axes:
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

plt.tight_layout()
eq_path = os.path.join(OUTPUT_DIR, 'equity_curve.png')
plt.savefig(eq_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved equity curve: {eq_path}")

# ─── Regime analysis plot ───
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Panel 1: Bull vs Bear cumulative
bull_cum = (1 + strat.loc[bull_mask, 'ls_return']).cumprod()
bear_cum = (1 + strat.loc[bear_mask, 'ls_return']).cumprod()
axes[0,0].plot(bull_cum.index, bull_cum.values, 'g-', label=f"Bull (Sharpe={bull_metrics['sharpe']:.2f})")
axes[0,0].plot(bear_cum.index, bear_cum.values, 'r-', label=f"Bear (Sharpe={bear_metrics['sharpe']:.2f})")
axes[0,0].set_title('Regime-Conditional Performance')
axes[0,0].legend()
axes[0,0].grid(True, alpha=0.3)

# Panel 2: Permutation test histogram
axes[0,1].hist(null_sharpes, bins=30, color='gray', alpha=0.7, label='Null distribution')
axes[0,1].axvline(x=real_sharpe, color='red', linewidth=2, label=f'Real Sharpe={real_sharpe:.3f}')
axes[0,1].axvline(x=perm_p95, color='orange', linewidth=1.5, linestyle='--', label=f'95th pctl={perm_p95:.3f}')
axes[0,1].set_title(f'Permutation Test (p={perm_pval:.3f})')
axes[0,1].legend()
axes[0,1].grid(True, alpha=0.3)

# Panel 3: Quintile spread
q_ann_rets = [np.mean(quintile_rets[q]) * 12 * 100 for q in range(1, n_quintiles+1)]
colors = ['red' if r < 0 else 'green' for r in q_ann_rets]
axes[1,0].bar(range(1, n_quintiles+1), q_ann_rets, color=colors, alpha=0.7)
axes[1,0].set_xlabel('Quintile (1=most neg skew, 5=most pos skew)')
axes[1,0].set_ylabel('Ann Return (%)')
axes[1,0].set_title('Return by Skewness Quintile')
axes[1,0].grid(True, alpha=0.3)

# Panel 4: Skewness persistence
axes[1,1].hist(rank_autocorrs, bins=30, color='steelblue', alpha=0.7)
axes[1,1].axvline(x=rank_ac_mean, color='red', linewidth=2, label=f'Mean={rank_ac_mean:.3f}')
axes[1,1].set_title('Month-to-Month Skewness Rank Autocorrelation')
axes[1,1].set_xlabel('Rank Correlation')
axes[1,1].legend()
axes[1,1].grid(True, alpha=0.3)

plt.suptitle('Skewness Factor — Validation (HC #428)', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
regime_path = os.path.join(OUTPUT_DIR, 'regime_analysis.png')
plt.savefig(regime_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved regime analysis: {regime_path}")

# ─── Summary JSON ───
summary = {
    'strategy': 'Skewness Factor L/S v1',
    'observation': 'Positive-skew stocks outperform negative-skew stocks (~10% annually)',
    'universe': f'S&P 500 ({returns_trimmed.shape[1]} stocks)',
    'period': f"{strat.index[0].strftime('%Y-%m-%d')} to {strat.index[-1].strftime('%Y-%m-%d')}",
    'lookback': f'{SKEW_LOOKBACK}d rolling skewness',
    'rebalance': 'Monthly',
    'cost': f'{COST_BPS}bps round-trip',
    'base_metrics': {k: round(v, 4) if isinstance(v, float) else v
                     for k, v in base_metrics.items()},
    'observation_stats': {
        'skewness_persistence_rank_ac': round(rank_ac_mean, 3),
        'cross_section_mean_skew': round(cs_mean_skew.mean(), 3),
        'pct_stocks_positive_skew': round((latest_skew > 0).mean() * 100, 1),
    },
    'validation': {
        'regime': {
            'bull_sharpe': round(bull_metrics['sharpe'], 3),
            'bear_sharpe': round(bear_metrics['sharpe'], 3),
            'regime_gap': round(regime_gap, 3),
            'pass': regime_pass,
        },
        'permutation': {
            'real_sharpe': round(real_sharpe, 3),
            'null_mean': round(null_sharpes.mean(), 3),
            'null_std': round(null_sharpes.std(), 3),
            'null_p95': round(perm_p95, 3),
            'p_value': round(perm_pval, 3),
            'pass': perm_pass,
        },
        'sub_period_stable': sub_pass,
        'quintile_monotonic': monotonic,
        'sector_neutral': {
            'sharpe': round(sn_metrics['sharpe'], 3) if sn_metrics else None,
            'survives': sn_survives,
        },
    },
    'verdict': 'TBD',
    'timestamp': datetime.now().isoformat(),
}

# Determine overall verdict
all_checks = [regime_pass, perm_pass, sub_pass]
if sn_survives is not None:
    all_checks.append(sn_survives)

if all(all_checks):
    verdict = 'PASS — All validation checks passed. Skewness factor shows robust edge.'
elif sum(all_checks) >= len(all_checks) - 1:
    verdict = 'CONDITIONAL PASS — Most checks passed, review failing test.'
else:
    verdict = 'FAIL — Multiple validation checks failed.'

summary['verdict'] = verdict

summary_path = os.path.join(OUTPUT_DIR, 'summary.json')
with open(summary_path, 'w') as f:
    json.dump(summary, f, indent=2, default=str)
print(f"  Saved summary: {summary_path}")

# ═══════════════════════════════════════════════════════════════════════
# FINAL VERDICT
# ═══════════════════════════════════════════════════════════════════════

print(f"\n{'='*70}")
print(f"FINAL VERDICT")
print(f"{'='*70}")
print(f"\n  Strategy: Skewness Factor Long/Short (monthly, S&P 500)")
print(f"  Sharpe:   {base_metrics['sharpe']:.3f}")
print(f"  Sortino:  {base_metrics['sortino']:.3f}")
print(f"  CAGR:     {base_metrics['cagr']:.2f}%")
print(f"  Max DD:   {base_metrics['max_dd']:.2f}%")
print(f"  PF:       {base_metrics['pf']:.2f}")
print(f"  Monthly WR: {base_metrics['wr']:.1f}%")
print(f"\n  Regime gap:     {regime_gap:.3f} {'PASS' if regime_pass else 'FAIL'}")
print(f"  Permutation:    p={perm_pval:.3f} {'PASS' if perm_pass else 'FAIL'}")
print(f"  Sub-period:     {'PASS' if sub_pass else 'FAIL'}")
print(f"  Quintile mono:  {'YES' if monotonic else 'NO'}")
if sn_survives is not None:
    print(f"  Sector neutral: {'YES' if sn_survives else 'NO'}")
print(f"\n  >>> {verdict}")
print(f"\n{'='*70}")
print(f"Done. Output saved to {OUTPUT_DIR}")
