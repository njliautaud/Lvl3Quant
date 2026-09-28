#!/usr/bin/env python3
"""
BPS Fill Quality Analysis
=========================
Analyzes real bid-ask spreads from yfinance option chains to determine
which tickers produce viable bull put spreads at real market costs.

Commission: $0.65/contract/leg = $1.30 per spread
A spread is viable only if net_credit > $1.30

Categories:
  GREEN:  net_credit > $1.00 (viable after commissions)
  YELLOW: net_credit $0.01 - $0.99 (marginal)
  RED:    net_credit <= $0 (untradeable)

Usage: python scripts/bps_fill_quality_analysis.py
"""

import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timedelta
from collections import defaultdict

import numpy as np
import pandas as pd

# ── Config ──────────────────────────────────────────────────────────────
COMMISSION_PER_SPREAD = 1.30  # $0.65/contract/leg x 2 legs
GREEN_THRESHOLD = 1.00        # net credit > $1.00 = viable
OUTPUT_DIR = "/home/jupiter/Lvl3Quant/output/bps_fill_quality_analysis"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Top 60 most liquid option tickers (by volume)
TOP_TICKERS = [
    # Mega-cap tech
    "AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "TSLA", "AVGO",
    # Large-cap tech
    "CRM", "ADBE", "ORCL", "AMD", "INTC", "CSCO", "NFLX", "UBER",
    # Finance
    "JPM", "BAC", "GS", "MS", "AXP", "V", "MA", "BRK-B",
    # Consumer/retail
    "HD", "WMT", "COST", "TGT", "MCD", "SBUX", "NKE", "LOW",
    # Energy
    "XOM", "CVX", "COP", "SLB", "OXY",
    # Healthcare
    "UNH", "JNJ", "LLY", "ABBV", "PFE", "MRK", "AMGN", "BMY",
    # Industrial/defense
    "BA", "CAT", "GE", "RTX", "LMT", "DE",
    # Other high-vol
    "DDOG", "CRWD", "SNOW", "SHOP", "SQ", "COIN", "MARA", "RIOT",
    "ABNB", "DASH", "BABA", "FSLR",
]


def parse_pm2_logs():
    """Parse existing PM2 logs for REAL CHAIN data."""
    print("=" * 70)
    print("PART 1: Parsing existing PM2 logs for real chain data")
    print("=" * 70)

    try:
        result = subprocess.run(
            ["pm2", "logs", "wheel-bps-conservative", "--lines", "500", "--nostream"],
            capture_output=True, text=True, timeout=15
        )
        raw = result.stdout + result.stderr
    except Exception as e:
        print(f"Could not read PM2 logs: {e}")
        return []

    # Parse REAL CHAIN lines
    # Format: REAL CHAIN TICKER: short K bid=X ask=Y BA=Z%, long K bid=X ask=Y BA=Z%, net_credit_real=X net_credit_bs=Y
    pattern = re.compile(
        r'REAL CHAIN (\w[\w-]*): '
        r'short ([\d.]+) bid=([\d.]+) ask=([\d.]+) BA=(\d+)%, '
        r'long ([\d.]+) bid=([\d.nan]+) ask=([\d.]+) BA=(\d+)%, '
        r'net_credit_real=([-\d.]+) net_credit_bs=([\d.]+)'
    )

    records = []
    seen = set()  # deduplicate by ticker (take latest)

    lines = raw.split('\n')
    # Process in reverse to get latest data first
    for line in reversed(lines):
        m = pattern.search(line)
        if m:
            ticker = m.group(1)
            if ticker in seen:
                continue
            seen.add(ticker)

            records.append({
                'ticker': ticker,
                'short_strike': float(m.group(2)),
                'short_bid': float(m.group(3)),
                'short_ask': float(m.group(4)),
                'short_ba_pct': int(m.group(5)),
                'long_strike': float(m.group(6)),
                'long_bid': float(m.group(7).replace('nan', '0')),
                'long_ask': float(m.group(8)),
                'long_ba_pct': int(m.group(9)),
                'net_credit_real': float(m.group(10)),
                'net_credit_bs': float(m.group(11)),
            })

    # Also find BS_FALLBACK and SKIP tickers
    fallback_pattern = re.compile(r'BS_FALLBACK: (\w+)')
    skip_pattern = re.compile(r'SKIP (\w+): real net credit')
    earnings_pattern = re.compile(r'EARNINGS SKIP: (\w+)')

    fallbacks = set()
    skips = set()
    earnings_skips = set()
    for line in lines:
        m = fallback_pattern.search(line)
        if m:
            fallbacks.add(m.group(1))
        m = skip_pattern.search(line)
        if m:
            skips.add(m.group(1))
        m = earnings_pattern.search(line)
        if m:
            earnings_skips.add(m.group(1))

    print(f"Parsed {len(records)} unique tickers with real chain data")
    print(f"BS Fallbacks (no viable chain): {sorted(fallbacks)}")
    print(f"Skipped (credit too low): {sorted(skips)}")
    print(f"Earnings skips: {sorted(earnings_skips)}")

    return records


def pull_fresh_chains():
    """Pull fresh option chain data for top liquid tickers."""
    print("\n" + "=" * 70)
    print("PART 2: Pulling fresh option chain data from yfinance")
    print("=" * 70)

    try:
        import yfinance as yf
    except ImportError:
        print("yfinance not installed, skipping fresh chain pull")
        return []

    records = []
    target_dte = 7  # ~7 DTE
    target_delta_short = 0.30
    target_delta_long = 0.15

    today = datetime.now()

    for i, ticker in enumerate(TOP_TICKERS):
        try:
            tk = yf.Ticker(ticker)

            # Get available expiration dates
            exps = tk.options
            if not exps:
                print(f"  {ticker}: no options available")
                continue

            # Find closest to 7 DTE
            best_exp = None
            best_dte_diff = 999
            for exp_str in exps:
                exp_dt = datetime.strptime(exp_str, "%Y-%m-%d")
                dte = (exp_dt - today).days
                if 3 <= dte <= 14 and abs(dte - target_dte) < best_dte_diff:
                    best_dte_diff = abs(dte - target_dte)
                    best_exp = exp_str

            if not best_exp:
                # Try wider range
                for exp_str in exps:
                    exp_dt = datetime.strptime(exp_str, "%Y-%m-%d")
                    dte = (exp_dt - today).days
                    if 1 <= dte <= 21 and abs(dte - target_dte) < best_dte_diff:
                        best_dte_diff = abs(dte - target_dte)
                        best_exp = exp_str

            if not best_exp:
                print(f"  {ticker}: no suitable expiration")
                continue

            chain = tk.option_chain(best_exp)
            puts = chain.puts

            if puts.empty:
                print(f"  {ticker}: empty puts chain")
                continue

            # Get current price
            info = tk.fast_info
            price = getattr(info, 'last_price', None) or getattr(info, 'previous_close', None)
            if not price or price <= 0:
                hist = tk.history(period="1d")
                if not hist.empty:
                    price = hist['Close'].iloc[-1]
                else:
                    print(f"  {ticker}: no price data")
                    continue

            exp_dt = datetime.strptime(best_exp, "%Y-%m-%d")
            actual_dte = (exp_dt - today).days

            # Find ~30 delta put (short leg) - roughly 5-8% OTM
            otm_30d = price * 0.94  # ~6% OTM for 30-delta at 7 DTE
            otm_15d = price * 0.88  # ~12% OTM for 15-delta at 7 DTE

            # Filter to OTM puts only
            otm_puts = puts[puts['strike'] < price].copy()
            if otm_puts.empty:
                print(f"  {ticker}: no OTM puts")
                continue

            # Find closest to target strikes
            otm_puts['dist_30d'] = abs(otm_puts['strike'] - otm_30d)
            otm_puts['dist_15d'] = abs(otm_puts['strike'] - otm_15d)

            short_row = otm_puts.loc[otm_puts['dist_30d'].idxmin()]
            long_row = otm_puts.loc[otm_puts['dist_15d'].idxmin()]

            # Make sure long strike < short strike
            if long_row['strike'] >= short_row['strike']:
                # Find the next lower strike
                lower = otm_puts[otm_puts['strike'] < short_row['strike']]
                if lower.empty:
                    print(f"  {ticker}: can't form spread")
                    continue
                long_row = lower.iloc[-1]  # highest strike below short

            short_bid = short_row.get('bid', 0) or 0
            short_ask = short_row.get('ask', 0) or 0
            long_bid = long_row.get('bid', 0) or 0
            long_ask = long_row.get('ask', 0) or 0

            short_vol = int(short_row.get('volume', 0) or 0)
            long_vol = int(long_row.get('volume', 0) or 0)
            short_oi = int(short_row.get('openInterest', 0) or 0)
            long_oi = int(long_row.get('openInterest', 0) or 0)
            short_iv = float(short_row.get('impliedVolatility', 0) or 0)

            # Net credit = short_bid - long_ask (worst case: sell at bid, buy at ask)
            net_credit = short_bid - long_ask

            # Bid-ask spread %
            short_mid = (short_bid + short_ask) / 2 if (short_bid + short_ask) > 0 else 0
            short_ba_pct = ((short_ask - short_bid) / short_mid * 100) if short_mid > 0 else 999

            long_mid = (long_bid + long_ask) / 2 if (long_bid + long_ask) > 0 else 0
            long_ba_pct = ((long_ask - long_bid) / long_mid * 100) if long_mid > 0 else 999

            spread_width = short_row['strike'] - long_row['strike']

            # After commission profit
            net_after_commission = (net_credit - COMMISSION_PER_SPREAD / 100) * 100  # per contract

            rec = {
                'ticker': ticker,
                'price': round(price, 2),
                'expiry': best_exp,
                'dte': actual_dte,
                'short_strike': float(short_row['strike']),
                'short_bid': round(short_bid, 2),
                'short_ask': round(short_ask, 2),
                'short_ba_pct': round(short_ba_pct, 1),
                'short_volume': short_vol,
                'short_oi': short_oi,
                'short_iv': round(short_iv * 100, 1),
                'long_strike': float(long_row['strike']),
                'long_bid': round(long_bid, 2),
                'long_ask': round(long_ask, 2),
                'long_ba_pct': round(long_ba_pct, 1),
                'long_volume': int(long_row.get('volume', 0) or 0),
                'long_oi': int(long_row.get('openInterest', 0) or 0),
                'spread_width': round(spread_width, 2),
                'net_credit': round(net_credit, 2),
                'net_credit_mid': round((short_mid - long_mid), 2),
                'market_cap_B': round(price * 1e-9, 1),  # rough placeholder
            }

            # Category
            if net_credit > GREEN_THRESHOLD:
                rec['category'] = 'GREEN'
            elif net_credit > 0:
                rec['category'] = 'YELLOW'
            else:
                rec['category'] = 'RED'

            records.append(rec)

            status = f"  {ticker:6s}: short {short_row['strike']:.0f} bid={short_bid:.2f} ask={short_ask:.2f} BA={short_ba_pct:.0f}% | " \
                     f"long {long_row['strike']:.0f} bid={long_bid:.2f} ask={long_ask:.2f} | " \
                     f"net_credit=${net_credit:.2f} [{rec['category']}] " \
                     f"vol={short_vol} oi={short_oi}"
            print(status)

            time.sleep(0.5)  # Rate limit

        except Exception as e:
            print(f"  {ticker}: ERROR - {e}")
            time.sleep(0.5)
            continue

    return records


def analyze_and_report(pm2_records, fresh_records):
    """Generate the full analysis report."""
    print("\n" + "=" * 70)
    print("PART 3: Analysis & Categorization")
    print("=" * 70)

    # ── Analyze PM2 log data ──────────────────────────────────────
    if pm2_records:
        print("\n--- PM2 Log Data (from paper engine) ---")
        df_pm2 = pd.DataFrame(pm2_records)
        df_pm2['category'] = pd.cut(
            df_pm2['net_credit_real'],
            bins=[-999, 0, GREEN_THRESHOLD, 999],
            labels=['RED', 'YELLOW', 'GREEN']
        )

        green = df_pm2[df_pm2['category'] == 'GREEN']
        yellow = df_pm2[df_pm2['category'] == 'YELLOW']
        red = df_pm2[df_pm2['category'] == 'RED']

        print(f"\nTotal scanned: {len(df_pm2)}")
        print(f"  GREEN  (net > $1.00): {len(green):3d} ({len(green)/len(df_pm2)*100:.0f}%)")
        print(f"  YELLOW ($0.01-0.99): {len(yellow):3d} ({len(yellow)/len(df_pm2)*100:.0f}%)")
        print(f"  RED    (net <= $0):   {len(red):3d} ({len(red)/len(df_pm2)*100:.0f}%)")

        print(f"\nGREEN tickers (viable with real bid-ask):")
        for _, r in green.sort_values('net_credit_real', ascending=False).iterrows():
            bs_slip = r['net_credit_bs'] - r['net_credit_real']
            print(f"  {r['ticker']:6s}: real=${r['net_credit_real']:.2f}  BS=${r['net_credit_bs']:.2f}  "
                  f"slip=${bs_slip:.2f} ({bs_slip/r['net_credit_bs']*100:.0f}%)  "
                  f"short_ba={r['short_ba_pct']}%")

        print(f"\nRED tickers (untradeable):")
        for _, r in red.sort_values('net_credit_real').iterrows():
            print(f"  {r['ticker']:6s}: real=${r['net_credit_real']:.2f}  BS=${r['net_credit_bs']:.2f}  "
                  f"short_ba={r['short_ba_pct']}%")

        # BS vs Real credit comparison
        print(f"\n--- BS Model vs Real Market Comparison ---")
        df_pm2['bs_slip'] = df_pm2['net_credit_bs'] - df_pm2['net_credit_real']
        df_pm2['bs_slip_pct'] = (df_pm2['bs_slip'] / df_pm2['net_credit_bs'] * 100).clip(-500, 500)

        print(f"Mean BS credit:   ${df_pm2['net_credit_bs'].mean():.2f}")
        print(f"Mean Real credit: ${df_pm2['net_credit_real'].mean():.2f}")
        print(f"Mean slippage:    ${df_pm2['bs_slip'].mean():.2f} ({df_pm2['bs_slip_pct'].mean():.0f}%)")
        print(f"Median slippage:  ${df_pm2['bs_slip'].median():.2f} ({df_pm2['bs_slip_pct'].median():.0f}%)")

        # How many BS-viable become real-unviable?
        bs_viable = df_pm2[df_pm2['net_credit_bs'] > GREEN_THRESHOLD]
        real_viable_from_bs = bs_viable[bs_viable['net_credit_real'] > GREEN_THRESHOLD]
        print(f"\nBS says viable: {len(bs_viable)}")
        print(f"Actually viable: {len(real_viable_from_bs)} ({len(real_viable_from_bs)/max(1,len(bs_viable))*100:.0f}%)")
        print(f"BS FALSE POSITIVES: {len(bs_viable) - len(real_viable_from_bs)}")

    # ── Analyze fresh chain data ──────────────────────────────────
    if fresh_records:
        print("\n\n--- Fresh Chain Data Analysis ---")
        df = pd.DataFrame(fresh_records)

        green = df[df['category'] == 'GREEN']
        yellow = df[df['category'] == 'YELLOW']
        red = df[df['category'] == 'RED']

        print(f"\nTotal scanned: {len(df)}")
        print(f"  GREEN  (net > $1.00): {len(green):3d} ({len(green)/len(df)*100:.0f}%)")
        print(f"  YELLOW ($0.01-0.99): {len(yellow):3d} ({len(yellow)/len(df)*100:.0f}%)")
        print(f"  RED    (net <= $0):   {len(red):3d} ({len(red)/len(df)*100:.0f}%)")

        # Sort by net credit
        print(f"\n{'='*90}")
        print(f"{'Ticker':>7} {'Price':>8} {'ShortK':>7} {'ShBid':>6} {'ShAsk':>6} {'ShBA%':>6} "
              f"{'LongK':>7} {'LgAsk':>6} {'NetCr':>7} {'ShVol':>6} {'ShOI':>6} {'IV%':>5} {'Cat':>7}")
        print(f"{'='*90}")

        for _, r in df.sort_values('net_credit', ascending=False).iterrows():
            color = {'GREEN': '***', 'YELLOW': ' . ', 'RED': ' X '}[r['category']]
            print(f"{r['ticker']:>7} {r['price']:>8.2f} {r['short_strike']:>7.0f} "
                  f"{r['short_bid']:>6.2f} {r['short_ask']:>6.2f} {r['short_ba_pct']:>5.0f}% "
                  f"{r['long_strike']:>7.0f} {r['long_ask']:>6.2f} {r['net_credit']:>7.2f} "
                  f"{r['short_volume']:>6d} {r['short_oi']:>6d} {r['short_iv']:>4.0f}% {color}")

        # ── What makes GREEN tickers different? ──
        print(f"\n\n{'='*70}")
        print("PART 4: What makes GREEN tickers viable?")
        print(f"{'='*70}")

        if len(green) > 0 and len(df) > 3:
            for col in ['short_ba_pct', 'short_volume', 'short_oi', 'short_iv', 'price']:
                g_med = green[col].median()
                all_med = df[col].median()
                non_g = df[df['category'] != 'GREEN'][col].median()
                print(f"  {col:>15}: GREEN median={g_med:>8.1f}  |  non-GREEN median={non_g:>8.1f}  |  all={all_med:>8.1f}")

        # ── Volume/OI threshold analysis ──
        print(f"\n\n{'='*70}")
        print("PART 5: Minimum volume/OI threshold for viability")
        print(f"{'='*70}")

        if len(df) > 5:
            for threshold_col in ['short_volume', 'short_oi']:
                print(f"\n  --- {threshold_col} threshold analysis ---")
                thresholds = [0, 10, 50, 100, 250, 500, 1000, 2500, 5000]
                for t in thresholds:
                    subset = df[df[threshold_col] >= t]
                    if len(subset) == 0:
                        continue
                    n_green = len(subset[subset['category'] == 'GREEN'])
                    green_rate = n_green / len(subset) * 100
                    avg_credit = subset['net_credit'].mean()
                    print(f"    {threshold_col} >= {t:>5d}: {len(subset):>3d} tickers, "
                          f"{n_green:>3d} GREEN ({green_rate:>4.0f}%), avg credit=${avg_credit:.2f}")

            # BA% threshold
            print(f"\n  --- short_ba_pct threshold analysis ---")
            for t in [5, 10, 15, 20, 25, 30, 50]:
                subset = df[df['short_ba_pct'] <= t]
                if len(subset) == 0:
                    continue
                n_green = len(subset[subset['category'] == 'GREEN'])
                green_rate = n_green / len(subset) * 100
                avg_credit = subset['net_credit'].mean()
                print(f"    short_ba_pct <= {t:>3d}%: {len(subset):>3d} tickers, "
                      f"{n_green:>3d} GREEN ({green_rate:>4.0f}%), avg credit=${avg_credit:.2f}")

        # ── Commission analysis ──
        print(f"\n\n{'='*70}")
        print("PART 6: After-commission profitability")
        print(f"{'='*70}")

        df['net_after_comm'] = df['net_credit'] - COMMISSION_PER_SPREAD / 100
        df['profitable'] = df['net_after_comm'] > 0

        profitable = df[df['profitable']]
        print(f"\nCommission per spread: ${COMMISSION_PER_SPREAD:.2f} = ${COMMISSION_PER_SPREAD/100:.4f}/share")
        print(f"Profitable after commission: {len(profitable)}/{len(df)} ({len(profitable)/len(df)*100:.0f}%)")

        if len(profitable) > 0:
            print(f"\nProfitable tickers (net credit after $1.30 commission):")
            for _, r in profitable.sort_values('net_after_comm', ascending=False).iterrows():
                profit_per_contract = r['net_after_comm'] * 100
                print(f"  {r['ticker']:>7}: credit=${r['net_credit']:.2f} - comm=$0.013 = "
                      f"net=${r['net_after_comm']:.2f}/share (${profit_per_contract:.0f}/contract)")

        # ── Diversification with smaller universe ──
        print(f"\n\n{'='*70}")
        print("PART 7: Diversification with GREEN-only universe")
        print(f"{'='*70}")

        n_green = len(green)
        print(f"GREEN universe size: {n_green} tickers")
        print(f"Full universe size:  {len(df)} tickers")

        if n_green >= 5:
            print(f"\nWith {n_green} GREEN tickers:")
            print(f"  - Max position 2% of NAV each -> max {min(n_green, 50)} concurrent spreads")
            print(f"  - Adequate diversification: {'YES' if n_green >= 15 else 'MARGINAL' if n_green >= 8 else 'NO'}")

            avg_green_credit = green['net_credit'].mean()
            avg_green_width = green['spread_width'].mean() if 'spread_width' in green.columns else 15

            # Expected return per spread (credit / max_loss)
            max_loss = avg_green_width - avg_green_credit
            ror = avg_green_credit / max_loss * 100 if max_loss > 0 else 0

            print(f"  - Avg GREEN credit: ${avg_green_credit:.2f}")
            print(f"  - Avg spread width: ${avg_green_width:.2f}")
            print(f"  - Return on risk:   {ror:.1f}%")
            print(f"  - After commission: ${avg_green_credit - COMMISSION_PER_SPREAD/100:.2f}/share")
        else:
            print(f"  WARNING: Only {n_green} GREEN tickers - insufficient diversification!")

        # Save results
        df.to_csv(os.path.join(OUTPUT_DIR, "fresh_chain_analysis.csv"), index=False)

        summary = {
            'timestamp': datetime.now().isoformat(),
            'total_scanned': len(df),
            'green_count': len(green),
            'yellow_count': len(yellow),
            'red_count': len(red),
            'green_tickers': sorted(green['ticker'].tolist()),
            'red_tickers': sorted(red['ticker'].tolist()),
            'green_pct': round(len(green)/len(df)*100, 1),
            'avg_green_credit': round(green['net_credit'].mean(), 2) if len(green) > 0 else 0,
            'avg_red_credit': round(red['net_credit'].mean(), 2) if len(red) > 0 else 0,
            'commission_per_spread': COMMISSION_PER_SPREAD,
            'profitable_after_commission': len(profitable),
        }

        with open(os.path.join(OUTPUT_DIR, "summary.json"), 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"\nResults saved to {OUTPUT_DIR}/")

    # ── Combined conclusions ──
    print(f"\n\n{'='*70}")
    print("CONCLUSIONS")
    print(f"{'='*70}")

    if pm2_records:
        df_pm2 = pd.DataFrame(pm2_records)
        n_total = len(df_pm2)
        n_green = len(df_pm2[df_pm2['net_credit_real'] > GREEN_THRESHOLD])
        n_red = len(df_pm2[df_pm2['net_credit_real'] <= 0])

        print(f"\nFrom paper engine logs ({n_total} tickers scanned):")
        print(f"  {n_green} tickers ({n_green/n_total*100:.0f}%) have viable real spreads (credit > $1.00)")
        print(f"  {n_red} tickers ({n_red/n_total*100:.0f}%) are completely untradeable (credit <= $0)")

        # BS overestimation
        mean_bs = df_pm2['net_credit_bs'].mean()
        mean_real = df_pm2['net_credit_real'].mean()
        print(f"\n  BS model systematically OVERESTIMATES credit:")
        print(f"    BS avg:   ${mean_bs:.2f}")
        print(f"    Real avg: ${mean_real:.2f}")
        print(f"    Overestimation: {(mean_bs - mean_real)/mean_bs*100:.0f}%")

    if fresh_records:
        df = pd.DataFrame(fresh_records)
        n_green = len(df[df['category'] == 'GREEN'])
        print(f"\nFrom fresh chain pull ({len(df)} liquid tickers):")
        print(f"  {n_green} tickers have real net credit > $1.00")
        profitable = df[df['net_credit'] > COMMISSION_PER_SPREAD / 100]
        print(f"  {len(profitable)} tickers profitable after ${COMMISSION_PER_SPREAD:.2f} commission")


if __name__ == "__main__":
    pm2_records = parse_pm2_logs()
    fresh_records = pull_fresh_chains()
    analyze_and_report(pm2_records, fresh_records)
