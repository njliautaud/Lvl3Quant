#!/usr/bin/env python3
"""
Build Expanded Wheel Universe — HC #660
========================================

Expands from 70 large-cap names to ~200+ names covering:
  - Higher beta (1.3+) names for richer premiums
  - Biotech/pharma (high IV, great for wheel)
  - Semiconductors (cyclical, high beta)
  - Energy (commodity-driven vol)
  - REITs (income + vol)
  - Mid-cap growth (SMID with liquid options)
  - Small-cap with liquid options
  - Commodities/materials
  - Utilities (low beta anchor)

Data source: yfinance (free) for prices + info.
Modeled IV via Black-Scholes + realized vol (HC #556).

Author: Claude (HC #660 wheel expansion)
"""

import pandas as pd
import numpy as np
import yfinance as yf
from pathlib import Path
import time
import logging
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(
    format='%(asctime)s [UNIV-EXP] %(levelname)s %(message)s',
    level=logging.INFO,
)
log = logging.getLogger('UNIV-EXP')

ROOT = Path(__file__).resolve().parent.parent.parent
CACHE_DIR = Path(__file__).resolve().parent / "cache"
OUTPUT_DIR = ROOT / "output" / "wheel_expanded_universe"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# EXPANDED UNIVERSE — ~200 names across all sectors + beta ranges
# =============================================================================

EXPANSION_TICKERS = {
    # ── HIGH BETA TECH (1.3+) ──
    'MSTR': ('Technology', 'Bitcoin proxy, extreme vol'),
    'ROKU': ('Technology', 'Streaming, high beta'),
    'SNAP': ('Communication Services', 'Social media, high vol'),
    'SQ': ('Financial Services', 'Fintech, high beta'),
    'UPST': ('Financial Services', 'AI lending, extreme vol'),
    'RBLX': ('Technology', 'Gaming/metaverse'),
    'U': ('Technology', 'Unity, gaming engine'),
    'AFRM': ('Financial Services', 'BNPL, high beta'),
    'SOFI': ('Financial Services', 'Fintech'),
    'RIVN': ('Consumer Cyclical', 'EV startup, high vol'),
    'LCID': ('Consumer Cyclical', 'EV startup'),
    'PATH': ('Technology', 'RPA/AI automation'),
    'BILL': ('Technology', 'Fintech B2B'),
    'ZS': ('Technology', 'Cybersecurity'),
    'NET': ('Technology', 'Cloud/CDN'),
    'SNOW': ('Technology', 'Cloud data'),
    'MDB': ('Technology', 'Database'),
    'PINS': ('Communication Services', 'Social'),
    'TTWO': ('Communication Services', 'Gaming'),
    'EA': ('Communication Services', 'Gaming'),

    # ── BIOTECH / PHARMA (HIGH IV, IDEAL FOR WHEEL) ──
    'BIIB': ('Healthcare', 'Large-cap biotech'),
    'REGN': ('Healthcare', 'Large-cap biotech'),
    'GILD': ('Healthcare', 'Large-cap pharma'),
    'VRTX': ('Healthcare', 'Biotech, consistent'),
    'AMGN': ('Healthcare', 'Large-cap biotech'),
    'BMY': ('Healthcare', 'Large-cap pharma'),
    'ZTS': ('Healthcare', 'Animal health'),
    'DXCM': ('Healthcare', 'Med devices, high beta'),
    'ISRG': ('Healthcare', 'Robotic surgery'),
    'HIMS': ('Healthcare', 'Telehealth, high vol'),
    'CRSP': ('Healthcare', 'Gene editing, extreme vol'),
    'ARKG_proxy': None,  # Skip ETFs, track individual names
    'ILMN': ('Healthcare', 'Genomics'),
    'EXAS': ('Healthcare', 'Diagnostics, high beta'),
    'SGEN': None,  # Acquired
    'MDGL': ('Healthcare', 'Biotech, high vol'),

    # ── SEMICONDUCTORS (CYCLICAL, HIGH BETA) ──
    'AVGO': ('Technology', 'Broadcom, semi leader'),
    'MU': ('Technology', 'Memory, cyclical'),
    'MRVL': ('Technology', 'Networking chips'),
    'ON': ('Technology', 'Power semis'),
    'LRCX': ('Technology', 'Semi equipment'),
    'KLAC': ('Technology', 'Semi equipment'),
    'AMAT': ('Technology', 'Semi equipment'),
    'ADI': ('Technology', 'Analog semis'),
    'TXN': ('Technology', 'Analog semis'),
    'QCOM': ('Technology', 'Mobile chips'),
    'TSM': ('Technology', 'Foundry'),
    'ASML': ('Technology', 'Litho equipment'),
    'MCHP': ('Technology', 'Microcontrollers'),
    'SWKS': ('Technology', 'RF semis'),
    'WOLF': ('Technology', 'SiC, high vol'),

    # ── ENERGY (COMMODITY-DRIVEN VOL) ──
    'HAL': ('Energy', 'Oilfield services'),
    'DVN': ('Energy', 'E&P, high beta'),
    'FANG': ('Energy', 'Permian E&P'),
    'MPC': ('Energy', 'Refining'),
    'PSX': ('Energy', 'Refining'),
    'VLO': ('Energy', 'Refining'),
    'EOG': ('Energy', 'E&P'),
    'PXD': None,  # Acquired by XOM
    'COP': ('Energy', 'E&P large-cap'),
    'ET': ('Energy', 'Midstream MLP'),
    'ENPH': ('Energy', 'Solar, extreme vol'),
    'SEDG': ('Energy', 'Solar, extreme vol'),
    'FSLR': ('Energy', 'Solar'),
    'NEE': ('Utilities', 'Renewables utility'),

    # ── REITs (INCOME + VOL) ──
    'O': ('Real Estate', 'Realty Income, monthly div'),
    'AMT': ('Real Estate', 'Tower REIT'),
    'PLD': ('Real Estate', 'Industrial REIT'),
    'SPG': ('Real Estate', 'Mall REIT'),
    'EQIX': ('Real Estate', 'Data center REIT'),
    'DLR': ('Real Estate', 'Data center REIT'),
    'VNO': ('Real Estate', 'Office REIT, high vol'),
    'MPW': ('Real Estate', 'Healthcare REIT, high vol'),
    'IRM': ('Real Estate', 'Storage/data'),
    'VICI': ('Real Estate', 'Casino REIT'),

    # ── MATERIALS / COMMODITIES ──
    'FCX': ('Basic Materials', 'Copper, cyclical'),
    'NEM': ('Basic Materials', 'Gold mining'),
    'GOLD': ('Basic Materials', 'Gold mining'),
    'CLF': ('Basic Materials', 'Steel, cyclical'),
    'AA': ('Basic Materials', 'Aluminum'),
    'X': ('Basic Materials', 'Steel, high vol'),
    'MP': ('Basic Materials', 'Rare earth, high vol'),

    # ── INDUSTRIALS / DEFENSE ──
    'LMT': ('Industrials', 'Defense'),
    'NOC': ('Industrials', 'Defense'),
    'GD': ('Industrials', 'Defense'),
    'HON': ('Industrials', 'Diversified industrial'),
    'UPS': ('Industrials', 'Logistics'),
    'FDX': ('Industrials', 'Logistics'),
    'CARR': ('Industrials', 'HVAC'),
    'DAL': ('Industrials', 'Airline'),
    'UAL': ('Industrials', 'Airline'),
    'AAL': ('Industrials', 'Airline, high beta'),
    'LUV': ('Industrials', 'Airline'),

    # ── CONSUMER / RETAIL / RESTAURANTS ──
    'LULU': ('Consumer Cyclical', 'Athleisure'),
    'NKE': None,  # Already in universe
    'CMG': ('Consumer Cyclical', 'Fast casual'),
    'DASH': ('Consumer Cyclical', 'Delivery, high vol'),
    'ETSY': ('Consumer Cyclical', 'E-commerce, high vol'),
    'BABA': ('Consumer Cyclical', 'China e-commerce'),
    'JD': ('Consumer Cyclical', 'China e-commerce'),
    'W': ('Consumer Cyclical', 'E-commerce, high vol'),
    'CHWY': ('Consumer Cyclical', 'Pet e-commerce'),
    'DECK': ('Consumer Cyclical', 'Footwear'),

    # ── UTILITIES (LOW BETA ANCHOR, STEADY PREMIUMS) ──
    'SO': ('Utilities', 'Southern Company'),
    'DUK': ('Utilities', 'Duke Energy'),
    'AEP': ('Utilities', 'American Electric'),
    'XEL': ('Utilities', 'Xcel'),
    'ED': ('Utilities', 'ConEd'),
    'EXC': ('Utilities', 'Exelon'),

    # ── CRYPTO-ADJACENT (EXTREME VOL = RICH PREMIUMS) ──
    'MARA': ('Financial Services', 'Bitcoin mining'),
    'RIOT': ('Financial Services', 'Bitcoin mining'),
    'HUT': ('Financial Services', 'Bitcoin mining'),
    'BITF': ('Financial Services', 'Bitcoin mining'),
    'CLSK': ('Financial Services', 'Bitcoin mining'),

    # ── MID-CAP GROWTH (LIQUID OPTIONS, HIGH BETA) ──
    'CELH': ('Consumer Defensive', 'Energy drinks, high growth'),
    'DUOL': ('Technology', 'EdTech'),
    'MNDY': ('Technology', 'Work management'),
    'GLOB': ('Technology', 'IT services'),
    'APP': ('Technology', 'Ad tech'),
    'TOST': ('Technology', 'Restaurant tech'),
    'BROS': ('Consumer Cyclical', 'Coffee chain'),
    'DKNG': ('Consumer Cyclical', 'Sports betting'),
    'PENN': ('Consumer Cyclical', 'Gaming/betting'),
    'WYNN': ('Consumer Cyclical', 'Casino'),
    'LVS': ('Consumer Cyclical', 'Casino'),
    'MGM': ('Consumer Cyclical', 'Casino'),

    # ── LARGE-CAP FILLS (ensure good coverage) ──
    'IBM': ('Technology', 'Legacy tech'),
    'CSCO': ('Technology', 'Networking'),
    'WBA': ('Healthcare', 'Pharmacy'),
    'CVS': ('Healthcare', 'Pharmacy/health'),
    'MDT': ('Healthcare', 'Med devices'),
    'ABT': ('Healthcare', 'Med devices'),
    'TMO': ('Healthcare', 'Life sciences'),
    'DHR': ('Healthcare', 'Life sciences'),
    'UNP': ('Industrials', 'Railroad'),
    'MMM': ('Industrials', '3M'),
    'DOW': ('Basic Materials', 'Chemicals'),
    'LIN': ('Basic Materials', 'Industrial gas'),
    'APD': ('Basic Materials', 'Industrial gas'),
}

# Filter out None (acquired/skip) and existing universe
EXPANSION_TICKERS = {k: v for k, v in EXPANSION_TICKERS.items() if v is not None}


def fetch_ticker_data(ticker, start='2015-01-01'):
    """Fetch price history + info for a ticker."""
    try:
        t = yf.Ticker(ticker)
        hist = t.history(start=start, auto_adjust=True)
        if hist.empty or len(hist) < 252:  # Need at least 1 year
            return None, None

        info = t.info or {}
        beta = info.get('beta', np.nan)
        market_cap = info.get('marketCap', 0)
        avg_volume = info.get('averageVolume', 0)

        # Need liquid options: avg volume > 500K shares/day
        if avg_volume < 500_000:
            return None, None

        return hist, {
            'ticker': ticker,
            'sector': EXPANSION_TICKERS[ticker][0],
            'description': EXPANSION_TICKERS[ticker][1],
            'beta': beta,
            'market_cap': market_cap,
            'avg_volume': avg_volume,
        }
    except Exception as e:
        log.warning(f"Failed {ticker}: {e}")
        return None, None


def compute_rv_features(prices_df):
    """Compute realized vol features from price history."""
    prices_df = prices_df.copy()
    prices_df['ret'] = prices_df['Close'].pct_change()
    prices_df['log_ret'] = np.log(prices_df['Close'] / prices_df['Close'].shift(1))

    for w in [20, 60, 252]:
        prices_df[f'rv_{w}'] = prices_df['log_ret'].rolling(w).std() * np.sqrt(252)

    return prices_df


def main():
    log.info("=" * 60)
    log.info("WHEEL UNIVERSE EXPANSION — HC #660")
    log.info(f"Candidates: {len(EXPANSION_TICKERS)} new tickers")
    log.info("=" * 60)

    # Load existing universe
    existing = pd.read_parquet(CACHE_DIR / 'universe.parquet')
    existing_tickers = set(existing['ticker'].tolist())
    log.info(f"Existing universe: {len(existing_tickers)} names")

    # Filter out already-existing tickers
    new_tickers = {k: v for k, v in EXPANSION_TICKERS.items()
                   if k not in existing_tickers}
    log.info(f"New tickers to fetch: {len(new_tickers)}")

    # Fetch data for all new tickers
    all_prices = []
    all_info = []
    failed = []

    for i, (ticker, (sector, desc)) in enumerate(new_tickers.items()):
        log.info(f"[{i+1}/{len(new_tickers)}] Fetching {ticker} ({sector})...")
        hist, info = fetch_ticker_data(ticker)

        if hist is not None:
            hist_processed = compute_rv_features(hist)
            hist_processed['ticker'] = ticker
            hist_processed['date'] = hist_processed.index
            all_prices.append(hist_processed)
            all_info.append(info)
            log.info(f"  ✓ {ticker}: {len(hist)} days, beta={info.get('beta','?')}, "
                     f"avg_vol={info.get('avg_volume',0):,.0f}")
        else:
            failed.append(ticker)
            log.warning(f"  ✗ {ticker}: skipped (insufficient data or liquidity)")

        # Rate limit
        if (i + 1) % 20 == 0:
            log.info(f"  -- Rate limit pause ({i+1}/{len(new_tickers)}) --")
            time.sleep(2)

    log.info(f"\nFetch complete: {len(all_info)} succeeded, {len(failed)} failed")
    if failed:
        log.info(f"Failed: {failed}")

    # Build expanded universe DataFrame
    new_universe_df = pd.DataFrame(all_info)
    log.info(f"\nNew tickers by sector:")
    log.info(new_universe_df['sector'].value_counts().to_string())

    # Combine with existing
    existing_info = existing.copy()
    if 'description' not in existing_info.columns:
        existing_info['description'] = 'Original universe'
    if 'beta' not in existing_info.columns:
        existing_info['beta'] = np.nan
    if 'market_cap' not in existing_info.columns:
        existing_info['market_cap'] = 0
    if 'avg_volume' not in existing_info.columns:
        existing_info['avg_volume'] = 0

    # Align columns
    cols = ['ticker', 'sector', 'description', 'beta', 'market_cap', 'avg_volume']
    for c in cols:
        if c not in existing_info.columns:
            existing_info[c] = ''
        if c not in new_universe_df.columns:
            new_universe_df[c] = ''

    combined = pd.concat([existing_info[cols], new_universe_df[cols]], ignore_index=True)
    combined = combined.drop_duplicates(subset='ticker', keep='first')
    combined = combined.sort_values('ticker').reset_index(drop=True)

    log.info(f"\nCombined universe: {len(combined)} names")
    log.info(f"Sectors: {combined['sector'].value_counts().to_dict()}")

    # Save expanded universe
    combined.to_parquet(CACHE_DIR / 'universe_expanded.parquet', index=False)
    log.info(f"Saved universe_expanded.parquet")

    # Save prices for new tickers
    if all_prices:
        prices_combined = pd.concat(all_prices, ignore_index=True)
        prices_combined.to_parquet(CACHE_DIR / 'prices_expanded.parquet', index=False)
        log.info(f"Saved prices_expanded.parquet: {len(prices_combined)} rows")

    # Beta analysis
    if not new_universe_df.empty:
        beta_df = new_universe_df.dropna(subset=['beta'])
        if not beta_df.empty:
            log.info(f"\nBeta distribution of new names:")
            log.info(f"  Low beta (<0.8):    {len(beta_df[beta_df.beta < 0.8])}")
            log.info(f"  Mid beta (0.8-1.2): {len(beta_df[(beta_df.beta >= 0.8) & (beta_df.beta <= 1.2)])}")
            log.info(f"  High beta (1.2-2):  {len(beta_df[(beta_df.beta > 1.2) & (beta_df.beta <= 2)])}")
            log.info(f"  Extreme beta (>2):  {len(beta_df[beta_df.beta > 2])}")
            log.info(f"  Mean beta: {beta_df.beta.mean():.2f}")
            log.info(f"  Median beta: {beta_df.beta.median():.2f}")

    # Summary
    summary = {
        'total_universe': len(combined),
        'new_added': len(all_info),
        'failed': failed,
        'sectors': combined['sector'].value_counts().to_dict(),
        'beta_stats': {
            'mean': float(combined['beta'].mean()) if 'beta' in combined.columns else None,
            'median': float(combined['beta'].median()) if 'beta' in combined.columns else None,
        }
    }

    import json
    with open(OUTPUT_DIR / 'expansion_summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    log.info(f"\n{'='*60}")
    log.info(f"UNIVERSE EXPANSION COMPLETE")
    log.info(f"Total: {len(combined)} names ({len(all_info)} new + {len(existing_tickers)} existing)")
    log.info(f"{'='*60}")

    return combined


if __name__ == '__main__':
    main()
