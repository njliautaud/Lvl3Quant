#!/usr/bin/env python3
"""
Options Chain Data Collector — Daily SPX/SPY Options for Premium Selling
========================================================================
Collects real option chain snapshots for calibrating our ML premium selling strategy.
Currently that strategy shows Sharpe 10+ (unrealistic from simplified premium model).
Real data will give honest calibration.

Collects: SPY 0-45 DTE puts and calls, with Greeks, IV, bid/ask.
Stores daily snapshots for building historical option pricing dataset.
"""
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import os, json, warnings
warnings.filterwarnings('ignore')

OUTPUT = '/home/jupiter/Lvl3Quant/data/options_chains'
os.makedirs(OUTPUT, exist_ok=True)

print("=" * 70)
print("OPTIONS CHAIN DATA COLLECTOR")
print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
print("=" * 70)

# Tickers to collect
TICKERS = ['SPY', 'QQQ', 'IWM']
MAX_DTE = 45

for ticker_sym in TICKERS:
    print(f"\n--- {ticker_sym} ---")
    try:
        ticker = yf.Ticker(ticker_sym)
        spot = ticker.info.get('regularMarketPrice') or ticker.info.get('previousClose', 0)

        # Get available expiration dates
        exps = ticker.options
        if not exps:
            print(f"  No options data available for {ticker_sym}")
            continue

        today = datetime.now().date()
        valid_exps = []
        for exp in exps:
            exp_date = datetime.strptime(exp, '%Y-%m-%d').date()
            dte = (exp_date - today).days
            if 0 <= dte <= MAX_DTE:
                valid_exps.append((exp, dte))

        print(f"  Spot: ${spot:.2f}")
        print(f"  Expirations within {MAX_DTE} DTE: {len(valid_exps)}")

        all_chains = []

        for exp, dte in valid_exps:
            try:
                chain = ticker.option_chain(exp)

                # Process puts
                puts = chain.puts.copy()
                puts['type'] = 'put'
                puts['expiration'] = exp
                puts['dte'] = dte
                puts['spot'] = spot
                puts['moneyness'] = puts['strike'] / spot

                # Process calls
                calls = chain.calls.copy()
                calls['type'] = 'call'
                calls['expiration'] = exp
                calls['dte'] = dte
                calls['spot'] = spot
                calls['moneyness'] = calls['strike'] / spot

                # Filter to reasonable strikes (0.8x to 1.2x spot)
                puts = puts[(puts['moneyness'] >= 0.8) & (puts['moneyness'] <= 1.05)]
                calls = calls[(calls['moneyness'] >= 0.95) & (calls['moneyness'] <= 1.2)]

                all_chains.append(puts)
                all_chains.append(calls)
            except Exception as e:
                continue

        if all_chains:
            combined = pd.concat(all_chains, ignore_index=True)

            # Key columns
            keep_cols = ['type', 'expiration', 'dte', 'spot', 'strike', 'moneyness',
                        'lastPrice', 'bid', 'ask', 'volume', 'openInterest',
                        'impliedVolatility']
            keep_cols = [c for c in keep_cols if c in combined.columns]
            combined = combined[keep_cols]

            # Save
            date_str = today.strftime('%Y%m%d')
            outfile = f'{OUTPUT}/{ticker_sym}_{date_str}.parquet'
            combined.to_parquet(outfile)
            print(f"  Saved {len(combined)} options contracts")

            # Summary stats
            puts_data = combined[combined['type'] == 'put']
            if len(puts_data) > 0:
                otm_puts = puts_data[puts_data['moneyness'] < 0.98]
                if len(otm_puts) > 0:
                    avg_iv = otm_puts['impliedVolatility'].mean()
                    avg_spread = ((otm_puts['ask'] - otm_puts['bid']) / otm_puts['bid']).mean()
                    print(f"  OTM Puts: avg IV={avg_iv:.1%}, avg spread={avg_spread:.1%}")

            # Credit spread pricing (sell -1 delta put, buy further OTM)
            # Look at 30-DTE puts
            dte30 = puts_data[(puts_data['dte'] >= 25) & (puts_data['dte'] <= 35)]
            if len(dte30) > 3:
                # 5-wide put spread at ~0.95 moneyness
                short_strike = dte30.iloc[(dte30['moneyness'] - 0.95).abs().argsort()[:1]]
                long_strike = dte30.iloc[(dte30['moneyness'] - 0.90).abs().argsort()[:1]]

                if len(short_strike) > 0 and len(long_strike) > 0:
                    credit = short_strike['bid'].values[0] - long_strike['ask'].values[0]
                    width = short_strike['strike'].values[0] - long_strike['strike'].values[0]
                    if width > 0 and credit > 0:
                        roi = credit / (width - credit)
                        print(f"  Sample 30DTE put spread: ${credit:.2f} credit on ${width:.0f} wide = {roi:.1%} ROI")
        else:
            print(f"  No valid chain data collected")

    except Exception as e:
        print(f"  Error: {e}")

# Check how many days we have collected so far
existing = [f for f in os.listdir(OUTPUT) if f.endswith('.parquet')]
print(f"\n{'=' * 70}")
print(f"TOTAL COLLECTED: {len(existing)} snapshots in database")
if existing:
    dates = sorted(set(f.split('_')[1].split('.')[0] for f in existing))
    print(f"  Date range: {dates[0]} to {dates[-1]}")
    print(f"  Unique dates: {len(dates)}")
print(f"{'=' * 70}")
print("\nDONE — Run daily to build historical option pricing dataset")
