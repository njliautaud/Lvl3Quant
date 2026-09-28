#!/usr/bin/env python3
"""
Weekly Stock Momentum Scanner
Runs Sunday nights. Produces actionable report of top momentum stocks
for the growth book, combining sector momentum + dual momentum + LEAPS screening.

Output: /home/jupiter/Lvl3Quant/output/growth_research/weekly_scans/scan_YYYY-MM-DD.txt
"""

import os, sys, json, warnings, time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import yfinance as yf

warnings.filterwarnings('ignore')

OUTPUT_DIR = '/home/jupiter/Lvl3Quant/output/growth_research/weekly_scans'
os.makedirs(OUTPUT_DIR, exist_ok=True)

###############################################################################
# CONSTANTS
###############################################################################

SECTOR_ETFS = {
    'XLK': 'Technology',
    'XLF': 'Financials',
    'XLE': 'Energy',
    'XLV': 'Health Care',
    'XLI': 'Industrials',
    'XLY': 'Consumer Discretionary',
    'XLC': 'Communication Services',
    'XLP': 'Consumer Staples',
    'XLRE': 'Real Estate',
    'XLB': 'Materials',
    'XLU': 'Utilities',
}

# S&P 500 stocks mapped to GICS sectors
SECTOR_STOCKS = {
    'Technology': ['AAPL','MSFT','NVDA','AVGO','CSCO','ACN','TXN','QCOM','INTC','ADI',
                   'AMAT','LRCX','KLAC','MCHP','CDNS','SNPS','FTNT','PANW','CRM','NOW',
                   'ADBE','ORCL','INTU','PLTR','ANET','MPWR','ON','NXPI','GEN','KEYS'],
    'Financials': ['JPM','V','MA','BAC','WFC','GS','MS','BLK','SCHW','CB',
                   'AXP','CME','ICE','PNC','USB','TFC','MMC','AON','AJG','MSCI',
                   'MET','PRU','AFL','AIG','ALL','TRV','PGR','SPGI','MCO','FI'],
    'Energy': ['XOM','CVX','COP','SLB','EOG','MPC','PSX','VLO','OXY',
               'HES','DVN','FANG','HAL','BKR','WMB','OKE','KMI','TRGP','CTRA'],
    'Health Care': ['UNH','JNJ','LLY','ABBV','MRK','TMO','ABT','DHR','PFE','AMGN',
                    'ISRG','SYK','VRTX','REGN','GILD','CI','ELV','HUM','ZTS','BDX',
                    'BSX','MDT','DXCM','A','IQV','EW','IDXX','MTD','HOLX','ALGN'],
    'Industrials': ['CAT','DE','UNP','HON','RTX','BA','GE','LMT','NOC','GD',
                    'ITW','EMR','FDX','UPS','WM','RSG','CSX','NSC','PCAR','TDG',
                    'ROK','SWK','IR','PH','CTAS','FAST','ODFL','J','VRSK','CPRT'],
    'Consumer Discretionary': ['AMZN','TSLA','HD','MCD','LOW','NKE','SBUX','TJX','BKNG','ORLY',
                                'AZO','ROST','CMG','DHI','LEN','PHM','GM','F','YUM','DPZ',
                                'POOL','GRMN','ULTA','EBAY','ETSY','BBY','KMX','GPC','LKQ','APTV'],
    'Communication Services': ['META','GOOGL','GOOG','NFLX','DIS','CMCSA','T','VZ','TMUS','CHTR',
                                'EA','TTWO','MTCH','LYV','PARA','WBD','FOX','IPG','OMC'],
    'Consumer Staples': ['PG','PEP','KO','COST','WMT','PM','MO','CL','MDLZ','KDP',
                          'GIS','SJM','HSY','K','CAG','STZ','TAP','KR','SYY',
                          'ADM','TSN','HRL','MKC','CHD','CLX','EL','KMB','MNST','WBA'],
    'Real Estate': ['PLD','AMT','CCI','EQIX','SPG','PSA','WELL','O','DLR','VICI',
                     'ARE','AVB','EQR','ESS','MAA','UDR','CPT','SUI','ELS','REG'],
    'Materials': ['LIN','APD','SHW','FCX','NEM','NUE','DOW','DD','ECL','PPG',
                   'VMC','MLM','CTVA','ALB','CE','EMN','IFF','FMC','CF','MOS'],
    'Utilities': ['NEE','SO','DUK','D','AEP','SRE','EXC','XEL','WEC',
                   'ED','AEE','CMS','DTE','FE','ETR','PEG','PPL','AWK','ATO'],
}

# S&P 400 mid-cap additions (top names per sector)
MIDCAP_STOCKS = {
    'Technology': ['SMCI','CRDO','VRT','CIEN','SYNA','CAVM','MANH','PCTY','CYBR','TENB'],
    'Financials': ['RGA','FNF','WBS','CFR','SNV','EWBC','FHN','IBKR','SEIC','CBSH'],
    'Energy': ['SM','RRC','AR','CNX','MTDR','NOV','CHX','DINO','HLX','PTEN'],
    'Health Care': ['UTHR','MEDP','TECH','NBIX','RARE','EXAS','NVCR','ITCI','AZTA','CRL'],
    'Industrials': ['AXON','TTEK','RBC','GGG','AIT','SSD','KNX','SAIA','XPO','WERN'],
    'Consumer Discretionary': ['TOL','TPX','DECK','BOOT','WING','TXRH','CAVA','SHAK','DKS','ANF'],
    'Communication Services': ['YELP','CARS','ZD','CARG','WLY'],
    'Consumer Staples': ['BRBR','FLO','CASY','SPTN','USFD'],
    'Real Estate': ['REXR','KRG','BRX','NNN','STAG','IRT','IIPR','NSA','CUZ','HPP'],
    'Materials': ['ATI','CRS','BERY','OLN','TROX','WLK','HUN','KWR','IOSP','GCP'],
    'Utilities': ['NRG','VST','MGEE','OGE','AVA','BKH','NWE','POR','SWX','SR'],
}

# Merge mid-caps into sector stocks
for sector, stocks in MIDCAP_STOCKS.items():
    if sector in SECTOR_STOCKS:
        SECTOR_STOCKS[sector].extend(stocks)

# Build flat lists
ALL_STOCKS = []
STOCK_TO_SECTOR = {}
for sector, stocks in SECTOR_STOCKS.items():
    for s in stocks:
        if s not in STOCK_TO_SECTOR:
            ALL_STOCKS.append(s)
            STOCK_TO_SECTOR[s] = sector

# Momentum weights
MOM_WEIGHTS = {
    '6m': 0.40,
    '3m': 0.30,
    '1m': 0.20,
    '1w': 0.10,
}

###############################################################################
# DATA DOWNLOAD
###############################################################################

def download_batch(tickers, period='1y'):
    """Download price data with batching to avoid API limits."""
    all_data = {}
    batch_size = 50
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i+batch_size]
        ticker_str = ' '.join(batch)
        try:
            data = yf.download(ticker_str, period=period, progress=False, auto_adjust=True, threads=True)
            if data.empty:
                continue
            # Check if multi-level columns (multiple tickers)
            if isinstance(data.columns, pd.MultiIndex):
                tickers_in_data = data.columns.get_level_values(1).unique()
                for t in batch:
                    if t in tickers_in_data:
                        df = data.xs(t, level=1, axis=1)
                        if not df.empty and df['Close'].notna().sum() > 20:
                            all_data[t] = df
            else:
                # Single ticker - flat columns
                if data['Close'].notna().sum() > 20:
                    all_data[batch[0]] = data
        except Exception as e:
            print(f"  Warning: batch {i}-{i+batch_size} failed: {e}")
        if i + batch_size < len(tickers):
            time.sleep(0.5)
    return all_data


def compute_rsi(prices, period=14):
    """Compute RSI(14)."""
    delta = prices.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.iloc[-1] if len(rsi) > 0 else np.nan


def compute_sma(prices, period=200):
    """Compute SMA."""
    if len(prices) < period:
        return np.nan
    return prices.rolling(window=period).mean().iloc[-1]


###############################################################################
# SECTOR RANKING
###############################################################################

def rank_sectors():
    """Rank 11 GICS sectors by 6-month momentum using sector ETFs."""
    print("=" * 70)
    print("SECTOR MOMENTUM RANKING (6-month returns)")
    print("=" * 70)

    etf_tickers = list(SECTOR_ETFS.keys())
    data = download_batch(etf_tickers, period='1y')

    sector_returns = {}
    for etf, sector in SECTOR_ETFS.items():
        if etf in data:
            closes = data[etf]['Close'].dropna()
            if len(closes) >= 126:  # ~6 months
                ret_6m = (closes.iloc[-1] / closes.iloc[-126]) - 1
                ret_3m = (closes.iloc[-1] / closes.iloc[-63]) - 1 if len(closes) >= 63 else np.nan
                ret_1m = (closes.iloc[-1] / closes.iloc[-21]) - 1 if len(closes) >= 21 else np.nan
                sector_returns[sector] = {
                    'etf': etf,
                    '6m_return': ret_6m,
                    '3m_return': ret_3m,
                    '1m_return': ret_1m,
                    'price': closes.iloc[-1],
                }
            else:
                print(f"  Warning: {etf} ({sector}) has insufficient data ({len(closes)} days)")
        else:
            print(f"  Warning: {etf} ({sector}) download failed")

    # Sort by 6-month momentum
    ranked = sorted(sector_returns.items(), key=lambda x: x[1]['6m_return'], reverse=True)

    print(f"\n{'Rank':<6}{'Sector':<28}{'ETF':<6}{'6M Return':>10}{'3M Return':>10}{'1M Return':>10}")
    print("-" * 70)
    for i, (sector, info) in enumerate(ranked, 1):
        print(f"{i:<6}{sector:<28}{info['etf']:<6}{info['6m_return']:>9.1%}{info['3m_return']:>10.1%}{info['1m_return']:>10.1%}")

    top_3_sectors = [s for s, _ in ranked[:3]]
    print(f"\nTop 3 sectors: {', '.join(top_3_sectors)}")

    return ranked, top_3_sectors


###############################################################################
# MARKET REGIME
###############################################################################

def check_market_regime():
    """Check if SPY is above 200-day SMA (safe to be long)."""
    print("\n" + "=" * 70)
    print("MARKET REGIME CHECK")
    print("=" * 70)

    spy_data = download_batch(['SPY'], period='2y')
    if 'SPY' not in spy_data:
        print("  ERROR: Could not download SPY data")
        return None, None

    closes = spy_data['SPY']['Close'].dropna()
    # Handle case where Close is multi-level (single ticker download)
    if isinstance(closes, pd.DataFrame):
        closes = closes.iloc[:, 0]
    current_price = float(closes.iloc[-1])
    sma_200 = float(compute_sma(closes, 200))
    sma_50 = float(compute_sma(closes, 50))

    pct_above_200 = ((current_price / sma_200) - 1) * 100
    regime = "BULLISH" if current_price > sma_200 else "BEARISH"

    print(f"  SPY Price:    ${current_price:.2f}")
    print(f"  200-day SMA:  ${sma_200:.2f} ({pct_above_200:+.1f}% {'above' if pct_above_200 > 0 else 'below'})")
    print(f"  50-day SMA:   ${sma_50:.2f}")
    print(f"  50 vs 200:    {'Golden Cross' if sma_50 > sma_200 else 'Death Cross'}")
    print(f"  Regime:       {'SAFE TO BE LONG' if regime == 'BULLISH' else 'CAUTION - RISK-OFF'}")

    regime_info = {
        'spy_price': float(current_price),
        'sma_200': float(sma_200),
        'sma_50': float(sma_50),
        'pct_above_200': float(pct_above_200),
        'regime': regime,
        'golden_cross': bool(sma_50 > sma_200),
    }
    return regime, regime_info


###############################################################################
# EARNINGS CALENDAR CHECK
###############################################################################

def check_earnings_soon(ticker, days=10):
    """Check if a stock has earnings within the next N days. Returns True if earnings are near."""
    try:
        tk = yf.Ticker(ticker)
        cal = tk.calendar
        if cal is not None and not cal.empty:
            # calendar can be a DataFrame or dict
            if isinstance(cal, pd.DataFrame):
                if 'Earnings Date' in cal.index:
                    ed = cal.loc['Earnings Date']
                    for d in ed:
                        if isinstance(d, (datetime, pd.Timestamp)):
                            if 0 <= (d - datetime.now()).days <= days:
                                return True
            elif isinstance(cal, dict):
                if 'Earnings Date' in cal:
                    dates = cal['Earnings Date']
                    if not isinstance(dates, list):
                        dates = [dates]
                    for d in dates:
                        if isinstance(d, (datetime, pd.Timestamp)):
                            if 0 <= (d - datetime.now()).days <= days:
                                return True
    except Exception:
        pass
    return False


###############################################################################
# STOCK SCORING & FILTERING
###############################################################################

def score_and_filter_stocks(top_sectors, all_stock_data):
    """Score stocks in top sectors by composite momentum, apply filters."""
    print("\n" + "=" * 70)
    print("STOCK SCREENING & RANKING")
    print("=" * 70)

    candidates = []
    filtered_reasons = {'no_data': 0, 'below_200sma': 0, 'low_volume': 0,
                        'overbought': 0, 'earnings_soon': 0, 'insufficient_history': 0}

    # Get stocks in top 3 sectors only
    target_stocks = []
    for sector in top_sectors:
        if sector in SECTOR_STOCKS:
            target_stocks.extend([(s, sector) for s in SECTOR_STOCKS[sector]])

    print(f"  Screening {len(target_stocks)} stocks in top 3 sectors...")

    for ticker, sector in target_stocks:
        if ticker not in all_stock_data:
            filtered_reasons['no_data'] += 1
            continue

        df = all_stock_data[ticker]
        closes = df['Close'].dropna()
        volumes = df['Volume'].dropna()

        if len(closes) < 200:
            filtered_reasons['insufficient_history'] += 1
            continue

        current_price = closes.iloc[-1]

        # Filter 1: Above 200-day SMA (dual momentum / absolute momentum)
        sma_200 = compute_sma(closes, 200)
        if pd.isna(sma_200) or current_price < sma_200:
            filtered_reasons['below_200sma'] += 1
            continue

        # Filter 2: Average daily dollar volume > $5M
        avg_dollar_vol = (closes.tail(20) * volumes.tail(20)).mean()
        if avg_dollar_vol < 5_000_000:
            filtered_reasons['low_volume'] += 1
            continue

        # Filter 3: RSI(14) < 80 (not overbought)
        rsi = compute_rsi(closes, 14)
        if pd.isna(rsi) or rsi >= 80:
            filtered_reasons['overbought'] += 1
            continue

        # Compute momentum scores
        mom_6m = (closes.iloc[-1] / closes.iloc[-126]) - 1 if len(closes) >= 126 else np.nan
        mom_3m = (closes.iloc[-1] / closes.iloc[-63]) - 1 if len(closes) >= 63 else np.nan
        mom_1m = (closes.iloc[-1] / closes.iloc[-21]) - 1 if len(closes) >= 21 else np.nan
        mom_1w = (closes.iloc[-1] / closes.iloc[-5]) - 1 if len(closes) >= 5 else np.nan

        if any(pd.isna(x) for x in [mom_6m, mom_3m, mom_1m, mom_1w]):
            filtered_reasons['insufficient_history'] += 1
            continue

        # Composite score
        composite = (MOM_WEIGHTS['6m'] * mom_6m +
                     MOM_WEIGHTS['3m'] * mom_3m +
                     MOM_WEIGHTS['1m'] * mom_1m +
                     MOM_WEIGHTS['1w'] * mom_1w)

        # 52-week high distance
        high_52w = closes.tail(252).max() if len(closes) >= 252 else closes.max()
        dist_from_high = (current_price / high_52w - 1) * 100

        # SMA distance
        dist_from_200 = (current_price / sma_200 - 1) * 100

        candidates.append({
            'ticker': ticker,
            'sector': sector,
            'price': float(current_price),
            'composite_score': float(composite),
            'mom_6m': float(mom_6m),
            'mom_3m': float(mom_3m),
            'mom_1m': float(mom_1m),
            'mom_1w': float(mom_1w),
            'rsi': float(rsi),
            'avg_dollar_vol_m': float(avg_dollar_vol / 1e6),
            'dist_from_52w_high': float(dist_from_high),
            'dist_from_200sma': float(dist_from_200),
        })

    # Sort by composite momentum
    candidates.sort(key=lambda x: x['composite_score'], reverse=True)

    print(f"\n  Filter results:")
    print(f"    Passed all filters: {len(candidates)}")
    for reason, count in filtered_reasons.items():
        if count > 0:
            print(f"    Filtered ({reason}): {count}")

    # Check earnings for top candidates (slow API call, limit to top 25)
    print(f"\n  Checking earnings dates for top {min(25, len(candidates))} candidates...")
    final_candidates = []
    for c in candidates[:25]:
        if check_earnings_soon(c['ticker'], days=10):
            filtered_reasons['earnings_soon'] += 1
            print(f"    {c['ticker']}: SKIPPED (earnings within 10 days)")
        else:
            final_candidates.append(c)
        if len(final_candidates) >= 15:
            break

    # Add remaining if we don't have 15 yet
    if len(final_candidates) < 15:
        for c in candidates[25:]:
            if len(final_candidates) >= 15:
                break
            final_candidates.append(c)

    return final_candidates[:15]


###############################################################################
# LEAPS SCREENING
###############################################################################

def screen_leaps(top_stocks, max_cost=500):
    """For top 5 stocks, check if affordable LEAPS exist (<$500 for Agentic account)."""
    print("\n" + "=" * 70)
    print(f"LEAPS SCREENING (max cost: ${max_cost})")
    print("=" * 70)

    leaps_candidates = []

    for stock in top_stocks[:5]:
        ticker = stock['ticker']
        price = stock['price']
        try:
            tk = yf.Ticker(ticker)
            expirations = tk.options

            if not expirations:
                print(f"  {ticker}: No options data available")
                continue

            # Find LEAPS (expiration > 1 year out)
            one_year_out = datetime.now() + timedelta(days=365)
            leaps_exps = [e for e in expirations
                          if datetime.strptime(e, '%Y-%m-%d') > one_year_out]

            if not leaps_exps:
                print(f"  {ticker} (${price:.0f}): No LEAPS expirations found")
                continue

            # Use the earliest LEAPS expiration
            exp = leaps_exps[0]
            exp_date = datetime.strptime(exp, '%Y-%m-%d')
            dte = (exp_date - datetime.now()).days

            chain = tk.option_chain(exp)
            calls = chain.calls

            if calls.empty:
                print(f"  {ticker}: Empty options chain for {exp}")
                continue

            # Find deep ITM calls (delta ~0.80) - strike around 70-80% of current price
            # and ATM/slightly OTM calls
            target_strike_deep = price * 0.75  # deep ITM for LEAPS
            target_strike_atm = price * 0.95   # slightly ITM

            best_leap = None
            for _, row in calls.iterrows():
                strike = row['strike']
                ask = row.get('ask', row.get('lastPrice', np.nan))
                bid = row.get('bid', 0)
                mid_price = (ask + bid) / 2 if bid > 0 else ask
                premium = mid_price * 100  # per contract

                if pd.isna(premium) or premium <= 0:
                    continue

                # Check affordability
                if premium <= max_cost:
                    leverage = price / mid_price if mid_price > 0 else 0
                    intrinsic = max(0, price - strike)
                    extrinsic = mid_price - intrinsic

                    if best_leap is None or strike < best_leap['strike']:
                        best_leap = {
                            'ticker': ticker,
                            'expiration': exp,
                            'dte': dte,
                            'strike': float(strike),
                            'stock_price': float(price),
                            'premium': float(premium),
                            'mid_price': float(mid_price),
                            'intrinsic': float(intrinsic),
                            'extrinsic': float(extrinsic),
                            'leverage_ratio': float(leverage),
                            'moneyness': float(strike / price),
                        }

            if best_leap:
                leaps_candidates.append(best_leap)
                print(f"  {ticker} (${price:.0f}): {exp} ${best_leap['strike']:.0f}C "
                      f"@ ${best_leap['mid_price']:.2f} (${best_leap['premium']:.0f}/contract) "
                      f"| Leverage: {best_leap['leverage_ratio']:.1f}x | DTE: {dte}")
            else:
                print(f"  {ticker} (${price:.0f}): No LEAPS under ${max_cost} for {exp}")

        except Exception as e:
            print(f"  {ticker}: LEAPS check failed - {e}")

    return leaps_candidates


###############################################################################
# REPORT GENERATION
###############################################################################

def generate_report(sector_ranking, top_stocks, regime_info, leaps, scan_date):
    """Generate clean text report."""
    lines = []
    lines.append("=" * 75)
    lines.append(f"  WEEKLY MOMENTUM SCANNER — {scan_date}")
    lines.append("=" * 75)

    # Market regime
    lines.append("")
    lines.append("MARKET REGIME")
    lines.append("-" * 40)
    if regime_info:
        lines.append(f"  SPY: ${regime_info['spy_price']:.2f}  |  200 SMA: ${regime_info['sma_200']:.2f}  "
                      f"|  {regime_info['pct_above_200']:+.1f}%")
        lines.append(f"  Signal: {'SAFE TO BE LONG' if regime_info['regime'] == 'BULLISH' else 'RISK-OFF / CAUTION'}")
        lines.append(f"  50/200 Cross: {'Golden Cross (bullish)' if regime_info['golden_cross'] else 'Death Cross (bearish)'}")
    else:
        lines.append("  Could not determine market regime")

    # Sector rankings
    lines.append("")
    lines.append("SECTOR RANKINGS (by 6-month momentum)")
    lines.append("-" * 70)
    lines.append(f"  {'Rank':<5}{'Sector':<28}{'ETF':<6}{'6M':>8}{'3M':>8}{'1M':>8}")
    lines.append("  " + "-" * 63)
    for i, (sector, info) in enumerate(sector_ranking, 1):
        marker = " ***" if i <= 3 else ""
        lines.append(f"  {i:<5}{sector:<28}{info['etf']:<6}"
                      f"{info['6m_return']:>7.1%}{info['3m_return']:>8.1%}{info['1m_return']:>8.1%}{marker}")

    # Top stocks
    lines.append("")
    lines.append("TOP 15 MOMENTUM STOCKS (from top 3 sectors)")
    lines.append("-" * 95)
    lines.append(f"  {'#':<4}{'Ticker':<8}{'Sector':<24}{'Price':>8}{'Score':>8}"
                 f"{'6M':>8}{'3M':>8}{'1M':>8}{'RSI':>6}{'vs52wH':>8}{'$Vol(M)':>9}")
    lines.append("  " + "-" * 91)
    for i, s in enumerate(top_stocks, 1):
        lines.append(f"  {i:<4}{s['ticker']:<8}{s['sector']:<24}${s['price']:>6.0f}"
                     f"{s['composite_score']:>7.1%}{s['mom_6m']:>7.1%}{s['mom_3m']:>7.1%}"
                     f"{s['mom_1m']:>7.1%}{s['rsi']:>6.0f}{s['dist_from_52w_high']:>7.1f}%"
                     f"{s['avg_dollar_vol_m']:>8.0f}")

    # LEAPS
    lines.append("")
    lines.append("LEAPS CANDIDATES (top 5 stocks, max $500/contract)")
    lines.append("-" * 80)
    if leaps:
        lines.append(f"  {'Ticker':<8}{'Expiry':<12}{'Strike':>8}{'Premium':>10}{'Leverage':>10}{'DTE':>6}{'Moneyness':>11}")
        lines.append("  " + "-" * 63)
        for l in leaps:
            lines.append(f"  {l['ticker']:<8}{l['expiration']:<12}${l['strike']:>6.0f}"
                         f"  ${l['premium']:>6.0f}{l['leverage_ratio']:>9.1f}x{l['dte']:>6}"
                         f"{l['moneyness']:>10.0%}")
    else:
        lines.append("  No affordable LEAPS found for top 5 stocks.")

    # Action items
    lines.append("")
    lines.append("ACTIONABLE SUMMARY")
    lines.append("-" * 40)
    if regime_info and regime_info['regime'] == 'BULLISH':
        lines.append("  Market regime: BULLISH — green light for momentum longs")
    else:
        lines.append("  Market regime: BEARISH — reduce exposure, tighten stops")

    if top_stocks:
        lines.append(f"  Top pick: {top_stocks[0]['ticker']} ({top_stocks[0]['sector']}) "
                     f"— composite momentum {top_stocks[0]['composite_score']:.1%}")
        if len(top_stocks) >= 3:
            top3 = ', '.join(s['ticker'] for s in top_stocks[:3])
            lines.append(f"  Top 3: {top3}")

    if leaps:
        best_leap = min(leaps, key=lambda x: x['premium'])
        lines.append(f"  Best LEAPS value: {best_leap['ticker']} {best_leap['expiration']} "
                     f"${best_leap['strike']:.0f}C @ ${best_leap['premium']:.0f} "
                     f"({best_leap['leverage_ratio']:.1f}x leverage)")

    lines.append("")
    lines.append(f"  Scan generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 75)

    return '\n'.join(lines)


###############################################################################
# MAIN
###############################################################################

def main():
    scan_date = datetime.now().strftime('%Y-%m-%d')
    print(f"\nWeekly Momentum Scanner — {scan_date}")
    print(f"Downloading data for {len(ALL_STOCKS)} stocks + {len(SECTOR_ETFS)} sector ETFs...\n")

    # Step 1: Rank sectors
    sector_ranking, top_3_sectors = rank_sectors()

    # Step 2: Market regime
    regime, regime_info = check_market_regime()

    # Step 3: Download all stock data for top sectors
    target_stocks = []
    for sector in top_3_sectors:
        if sector in SECTOR_STOCKS:
            target_stocks.extend(SECTOR_STOCKS[sector])
    target_stocks = list(set(target_stocks))

    print(f"\nDownloading data for {len(target_stocks)} stocks in top 3 sectors...")
    all_stock_data = download_batch(target_stocks, period='2y')
    print(f"  Downloaded: {len(all_stock_data)} stocks")

    # Step 4: Score and filter
    top_stocks = score_and_filter_stocks(top_3_sectors, all_stock_data)

    # Step 5: LEAPS screening
    leaps = screen_leaps(top_stocks, max_cost=500)

    # Step 6: Generate report
    report = generate_report(sector_ranking, top_stocks, regime_info, leaps, scan_date)

    # Save report
    report_file = os.path.join(OUTPUT_DIR, f'scan_{scan_date}.txt')
    with open(report_file, 'w') as f:
        f.write(report)

    # Save JSON for programmatic access
    json_file = os.path.join(OUTPUT_DIR, f'scan_{scan_date}.json')
    with open(json_file, 'w') as f:
        json.dump({
            'scan_date': scan_date,
            'regime': regime_info,
            'sector_ranking': [(s, info) for s, info in sector_ranking],
            'top_sectors': top_3_sectors,
            'top_stocks': top_stocks,
            'leaps_candidates': leaps,
        }, f, indent=2, default=str)

    # Print report to stdout
    print("\n")
    print(report)
    print(f"\nSaved to: {report_file}")
    print(f"JSON:     {json_file}")

    return top_stocks, leaps


if __name__ == '__main__':
    main()
