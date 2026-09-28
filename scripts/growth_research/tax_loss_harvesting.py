#!/usr/bin/env python3
"""
Tax-Loss Harvesting Analysis for Vol-Adjusted System
=====================================================
Quantifies the tax benefit of systematic TLH within the vol-adjusted UPRO system.

Questions:
1. How many TLH opportunities does the vol-adjusted system generate per year?
2. What's the annual tax savings at various tax rates?
3. Does TLH with substitute ETFs (SPXL for UPRO) maintain similar returns?
4. What's the optimal TLH threshold (harvest at -5%? -10%? any loss?)
5. How does wash sale rule (30-day) affect TLH frequency?

Key tax concepts:
- Short-term capital losses offset short-term gains (taxed at ordinary income rate)
- Long-term losses offset long-term gains (taxed at 15-20%)
- Net losses up to $3K/year offset ordinary income
- Excess losses carry forward indefinitely
- Wash sale rule: can't repurchase "substantially identical" within 30 days

For UPRO: substitute with SPXL (Direxion 3x S&P 500) during wash sale period.
For SPY: substitute with VOO or IVV.
"""

import numpy as np
import pandas as pd
import yfinance as yf
import json
import os
import warnings
warnings.filterwarnings('ignore')

OUTPUT_DIR = '/home/jupiter/Lvl3Quant/output/growth_research/tax_analysis'
os.makedirs(OUTPUT_DIR, exist_ok=True)

INITIAL = 500
WEEKLY_DCA = 100

# Tax rates
TAX_RATES = {
    'no_income_tax': {'st': 0.10, 'lt': 0.0},   # 10% bracket, no state
    'middle_bracket': {'st': 0.22, 'lt': 0.15},  # 22% bracket
    'high_bracket': {'st': 0.37, 'lt': 0.20},    # 37% bracket
    'california_high': {'st': 0.50, 'lt': 0.33},  # 37% + 13.3% state
}


def download_data():
    tickers = ['SPY', 'UPRO', 'SPXL', 'GLD', 'TLT', 'VOO', 'IVV']
    data = yf.download(tickers, start='2012-01-01', period='max',
                       auto_adjust=True, threads=True, progress=False)
    if isinstance(data.columns, pd.MultiIndex):
        closes = data['Close']
    else:
        closes = data
    if hasattr(closes.columns, 'droplevel'):
        try:
            closes.columns = closes.columns.droplevel(1)
        except:
            pass
    closes = closes.dropna(how='all').dropna(subset=['SPY', 'UPRO'])
    print(f"  Data: {len(closes)} days")
    return closes


def simulate_with_tax_lots(closes, tlh_threshold=-0.05, use_substitutes=True):
    """
    Simulate vol-adjusted system tracking individual tax lots for TLH.

    Each DCA purchase creates a new tax lot with its own cost basis.
    When a regime switch occurs, we sell ALL lots. TLH opportunities
    arise when we sell lots at a loss.

    tlh_threshold: minimum loss % to trigger proactive TLH harvest
                   (sell and rebuy substitute even without regime change)
    """
    spy = closes['SPY']
    spy_ret = spy.pct_change()
    vol_21d = spy_ret.rolling(21).std() * np.sqrt(252)
    spy_sma50 = spy.rolling(50).mean()
    returns = closes.pct_change().fillna(0)

    warmup = 63

    # Tax lots: list of {ticker, shares, cost_basis, purchase_date, current_value}
    tax_lots = []
    cash = float(INITIAL)
    total_contributed = float(INITIAL)
    last_week = None
    last_regime = None

    # TLH tracking
    harvested_losses_st = []  # short-term losses harvested
    harvested_losses_lt = []  # long-term losses harvested
    realized_gains_st = []
    realized_gains_lt = []
    tlh_events = []
    wash_sale_blocked = {}  # ticker -> unblock_date

    daily_values = []
    annual_summary = {}

    def get_regime(i):
        vol = vol_21d.iloc[i] if not np.isnan(vol_21d.iloc[i]) else 0.15
        protection = spy.iloc[i] > spy_sma50.iloc[i] if not np.isnan(spy_sma50.iloc[i]) else True

        if not protection:
            return 'CASH', {'SPY': 1.0}
        elif vol < 0.20:
            return 'UPRO', {'UPRO': 1.0}
        elif vol < 0.30:
            return 'SPY', {'SPY': 1.0}
        else:
            return 'SAFE', {'GLD': 0.5, 'TLT': 0.5}

    def sell_all_lots(date, reason="regime_switch"):
        """Sell all tax lots, track gains/losses."""
        nonlocal cash
        for lot in tax_lots:
            ticker = lot['ticker']
            if ticker in closes.columns:
                current_price = closes[ticker].loc[date] if date in closes.index else lot['current_value'] / lot['shares']
                if np.isnan(current_price):
                    current_price = lot['current_value'] / lot['shares']
                proceeds = lot['shares'] * current_price
            else:
                proceeds = lot['current_value']

            gain = proceeds - lot['cost_basis']
            hold_days = (date - lot['purchase_date']).days

            if hold_days > 365:
                if gain < 0:
                    harvested_losses_lt.append({'date': date, 'amount': gain, 'ticker': ticker})
                else:
                    realized_gains_lt.append({'date': date, 'amount': gain, 'ticker': ticker})
            else:
                if gain < 0:
                    harvested_losses_st.append({'date': date, 'amount': gain, 'ticker': ticker})
                else:
                    realized_gains_st.append({'date': date, 'amount': gain, 'ticker': ticker})

            cash += proceeds

        result = list(tax_lots)
        tax_lots.clear()
        return result

    def buy_target(date, target_alloc):
        """Buy target allocation, creating new tax lots."""
        nonlocal cash
        if cash <= 0:
            return

        for ticker, weight in target_alloc.items():
            if ticker not in closes.columns:
                continue
            price = closes[ticker].loc[date] if date in closes.index else np.nan
            if np.isnan(price):
                continue

            amount = cash * weight
            shares = amount / price

            tax_lots.append({
                'ticker': ticker,
                'shares': shares,
                'cost_basis': amount,
                'purchase_date': date,
                'current_value': amount,
            })

        cash = 0

    for i in range(warmup, len(closes)):
        date = closes.index[i]
        year = date.year

        # Weekly DCA
        week_key = (date.year, date.isocalendar()[1])
        if week_key != last_week:
            cash += WEEKLY_DCA
            total_contributed += WEEKLY_DCA
            last_week = week_key

        # Update lot values
        for lot in tax_lots:
            ticker = lot['ticker']
            if ticker in returns.columns:
                r = returns.loc[date, ticker]
                if not np.isnan(r):
                    lot['current_value'] *= (1 + r)

        # Get current regime
        regime, target = get_regime(i)

        # Regime switch → sell everything and rebuy
        if regime != last_regime:
            sell_all_lots(date, "regime_switch")
            buy_target(date, target)
            last_regime = regime
        else:
            # Proactive TLH: check if any lots have unrealized losses > threshold
            if tlh_threshold is not None:
                lots_to_harvest = []
                for lot_idx, lot in enumerate(tax_lots):
                    unrealized_pct = (lot['current_value'] - lot['cost_basis']) / lot['cost_basis']
                    if unrealized_pct < tlh_threshold:
                        # Check wash sale rule
                        ticker = lot['ticker']
                        if ticker in wash_sale_blocked and date < wash_sale_blocked[ticker]:
                            continue
                        lots_to_harvest.append(lot_idx)

                if lots_to_harvest:
                    # Harvest these specific lots
                    for idx in sorted(lots_to_harvest, reverse=True):
                        lot = tax_lots[idx]
                        ticker = lot['ticker']
                        gain = lot['current_value'] - lot['cost_basis']
                        hold_days = (date - lot['purchase_date']).days

                        if hold_days > 365:
                            harvested_losses_lt.append({'date': date, 'amount': gain, 'ticker': ticker})
                        else:
                            harvested_losses_st.append({'date': date, 'amount': gain, 'ticker': ticker})

                        # Sell and immediately buy substitute
                        proceeds = lot['current_value']
                        cash += proceeds
                        tax_lots.pop(idx)

                        # Buy substitute (SPXL for UPRO, VOO for SPY)
                        if use_substitutes:
                            sub = 'SPXL' if ticker == 'UPRO' else ('VOO' if ticker == 'SPY' else ticker)
                            if sub in closes.columns:
                                sub_price = closes[sub].loc[date] if date in closes.index else np.nan
                                if not np.isnan(sub_price):
                                    shares = proceeds / sub_price
                                    tax_lots.append({
                                        'ticker': sub,
                                        'shares': shares,
                                        'cost_basis': proceeds,
                                        'purchase_date': date,
                                        'current_value': proceeds,
                                    })
                                    cash -= proceeds
                                    wash_sale_blocked[ticker] = date + pd.Timedelta(days=31)

                        tlh_events.append({
                            'date': date,
                            'ticker': ticker,
                            'loss': gain,
                            'hold_days': hold_days,
                        })

            # Deploy DCA cash into current holdings
            if cash > 50 and tax_lots:
                buy_target(date, target)

        portfolio_val = cash + sum(lot['current_value'] for lot in tax_lots)
        daily_values.append(portfolio_val)

        # Annual summary
        if year not in annual_summary:
            annual_summary[year] = {
                'st_losses': 0, 'lt_losses': 0,
                'st_gains': 0, 'lt_gains': 0,
                'tlh_events': 0,
            }

        # Update annual totals
        for loss in harvested_losses_st:
            if loss['date'] == date:
                annual_summary[year]['st_losses'] += abs(loss['amount'])
                annual_summary[year]['tlh_events'] += 1
        for loss in harvested_losses_lt:
            if loss['date'] == date:
                annual_summary[year]['lt_losses'] += abs(loss['amount'])
                annual_summary[year]['tlh_events'] += 1
        for gain in realized_gains_st:
            if gain['date'] == date:
                annual_summary[year]['st_gains'] += gain['amount']
        for gain in realized_gains_lt:
            if gain['date'] == date:
                annual_summary[year]['lt_gains'] += gain['amount']

    portfolio = pd.Series(daily_values, index=closes.index[warmup:])

    total_st_losses = sum(abs(l['amount']) for l in harvested_losses_st)
    total_lt_losses = sum(abs(l['amount']) for l in harvested_losses_lt)
    total_st_gains = sum(g['amount'] for g in realized_gains_st)
    total_lt_gains = sum(g['amount'] for g in realized_gains_lt)

    return {
        'portfolio': portfolio,
        'total_contributed': total_contributed,
        'st_losses': total_st_losses,
        'lt_losses': total_lt_losses,
        'st_gains': total_st_gains,
        'lt_gains': total_lt_gains,
        'tlh_events': len(tlh_events),
        'annual_summary': annual_summary,
    }


def compute_tax_savings(result, tax_rate):
    """Compute tax savings from TLH at a given tax rate."""
    st_rate = tax_rate['st']
    lt_rate = tax_rate['lt']

    # Net ST: losses offset gains first
    net_st = result['st_gains'] - result['st_losses']
    net_lt = result['lt_gains'] - result['lt_losses']

    # If net ST is negative, it offsets LT gains
    if net_st < 0:
        lt_offset = min(abs(net_st), result['lt_gains'])
        remaining_loss = abs(net_st) - lt_offset
        savings = lt_offset * lt_rate + min(remaining_loss, 3000) * st_rate  # $3K ordinary income offset
        # Carry forward
        carry_forward = max(0, remaining_loss - 3000)
    else:
        savings = 0
        carry_forward = 0

    # Direct savings from harvested losses
    st_savings = result['st_losses'] * st_rate
    lt_savings = result['lt_losses'] * lt_rate

    return {
        'gross_st_savings': st_savings,
        'gross_lt_savings': lt_savings,
        'total_gross_savings': st_savings + lt_savings,
        'carry_forward': carry_forward,
    }


def main():
    print("="*70)
    print("TAX-LOSS HARVESTING ANALYSIS — VOL-ADJUSTED SYSTEM")
    print("="*70)

    closes = download_data()

    # Check substitute availability
    print(f"\n  SPXL available: {'SPXL' in closes.columns}")
    print(f"  VOO available: {'VOO' in closes.columns}")

    # --- Test different TLH thresholds ---
    thresholds = {
        'No TLH (baseline)': None,
        'Any loss (0%)': -0.001,
        'Loss > 3%': -0.03,
        'Loss > 5%': -0.05,
        'Loss > 10%': -0.10,
        'Loss > 20%': -0.20,
    }

    results = {}
    print(f"\n  Testing {len(thresholds)} TLH thresholds...")

    for name, threshold in thresholds.items():
        print(f"\n  --- {name} ---")
        result = simulate_with_tax_lots(closes, tlh_threshold=threshold, use_substitutes=True)
        portfolio = result['portfolio']
        r = portfolio.pct_change().dropna()
        years = len(r) / 252
        final = portfolio.iloc[-1]
        ann_ret = (1 + r).prod() ** (252 / len(r)) - 1
        ann_vol = r.std() * np.sqrt(252)
        sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
        peak = portfolio.expanding().max()
        max_dd = ((portfolio - peak) / peak).min()

        print(f"    Final value: ${final:,.0f}")
        print(f"    Sharpe: {sharpe:.3f}, MaxDD: {max_dd*100:.1f}%")
        print(f"    TLH events: {result['tlh_events']}")
        print(f"    ST losses harvested: ${result['st_losses']:,.0f}")
        print(f"    LT losses harvested: ${result['lt_losses']:,.0f}")
        print(f"    ST gains realized: ${result['st_gains']:,.0f}")
        print(f"    LT gains realized: ${result['lt_gains']:,.0f}")

        results[name] = {
            'final_value': float(final),
            'sharpe': float(sharpe),
            'max_dd': float(max_dd * 100),
            'tlh_events': result['tlh_events'],
            'st_losses': float(result['st_losses']),
            'lt_losses': float(result['lt_losses']),
            'st_gains': float(result['st_gains']),
            'lt_gains': float(result['lt_gains']),
            'total_losses': float(result['st_losses'] + result['lt_losses']),
            'total_gains': float(result['st_gains'] + result['lt_gains']),
            'years': float(years),
            'annual_summary': {str(k): v for k, v in result['annual_summary'].items()},
        }

        # Tax savings at each rate
        for rate_name, rate in TAX_RATES.items():
            savings = compute_tax_savings(result, rate)
            results[name][f'savings_{rate_name}'] = savings['total_gross_savings']
            annual_savings = savings['total_gross_savings'] / years if years > 0 else 0
            if threshold is not None:  # Skip printing for baseline
                print(f"    Tax savings ({rate_name}): ${savings['total_gross_savings']:,.0f} "
                      f"(${annual_savings:,.0f}/yr)")

    # --- Summary table ---
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)

    baseline_val = results.get('No TLH (baseline)', {}).get('final_value', 0)

    print(f"\n  {'Threshold':<22s} {'Final $':>10s} {'Sharpe':>7s} {'Events':>7s} "
          f"{'Losses':>10s} {'Gains':>10s} {'Net':>10s}")
    print("  " + "-"*78)
    for name, m in results.items():
        net = m['total_gains'] - m['total_losses']
        print(f"  {name:<22s} ${m['final_value']:>9,.0f} {m['sharpe']:>7.3f} {m['tlh_events']:>7d} "
              f"${m['total_losses']:>9,.0f} ${m['total_gains']:>9,.0f} ${net:>9,.0f}")

    # --- Tax savings comparison ---
    print(f"\n  ANNUAL TAX SAVINGS BY BRACKET:")
    print(f"  {'Threshold':<22s}", end="")
    for rate_name in TAX_RATES:
        print(f" {rate_name:>16s}", end="")
    print()
    print("  " + "-"*86)

    for name, m in results.items():
        if m.get('years', 0) == 0:
            continue
        print(f"  {name:<22s}", end="")
        for rate_name in TAX_RATES:
            annual = m.get(f'savings_{rate_name}', 0) / m['years']
            print(f"     ${annual:>9,.0f}", end="")
        print()

    # --- Year-by-year for best threshold ---
    best_threshold = 'Loss > 5%'
    if best_threshold in results:
        print(f"\n  YEAR-BY-YEAR BREAKDOWN ({best_threshold}):")
        annual = results[best_threshold].get('annual_summary', {})
        print(f"  {'Year':<6s} {'ST Loss':>10s} {'LT Loss':>10s} {'ST Gain':>10s} {'LT Gain':>10s} {'Events':>7s}")
        print("  " + "-"*55)
        for year in sorted(annual.keys()):
            a = annual[year]
            print(f"  {year:<6s} ${a['st_losses']:>9,.0f} ${a['lt_losses']:>9,.0f} "
                  f"${a['st_gains']:>9,.0f} ${a['lt_gains']:>9,.0f} {a['tlh_events']:>7d}")

    # --- UPRO vs SPXL tracking difference ---
    print(f"\n  SUBSTITUTE ETF TRACKING:")
    if 'UPRO' in closes.columns and 'SPXL' in closes.columns:
        overlap = closes[['UPRO', 'SPXL']].dropna()
        if len(overlap) > 252:
            tracking_diff = overlap['UPRO'].pct_change() - overlap['SPXL'].pct_change()
            print(f"    UPRO vs SPXL daily tracking difference:")
            print(f"      Mean: {tracking_diff.mean()*100:.4f}%/day ({tracking_diff.mean()*252*100:.2f}%/yr)")
            print(f"      Std: {tracking_diff.std()*100:.4f}%/day")
            print(f"      Correlation: {overlap['UPRO'].pct_change().corr(overlap['SPXL'].pct_change()):.6f}")

    # --- Key findings ---
    print("\n" + "="*70)
    print("KEY FINDINGS")
    print("="*70)

    best = max(results.items(), key=lambda x: x[1].get('savings_middle_bracket', 0) / max(x[1].get('years', 1), 1))
    print(f"\n  Best TLH threshold: {best[0]}")
    print(f"    Annual tax savings (22% bracket): ${best[1].get('savings_middle_bracket', 0) / best[1].get('years', 1):,.0f}/yr")
    print(f"    Annual tax savings (37% bracket): ${best[1].get('savings_high_bracket', 0) / best[1].get('years', 1):,.0f}/yr")
    print(f"    TLH events per year: {best[1]['tlh_events'] / best[1].get('years', 1):.1f}")

    # Impact on portfolio returns
    if baseline_val > 0:
        val_diff = best[1]['final_value'] - baseline_val
        print(f"    Portfolio impact: ${val_diff:+,.0f} ({val_diff/baseline_val*100:+.1f}%)")

    # Save
    output = {
        'run_date': pd.Timestamp.now().isoformat(),
        'results': {k: {kk: vv for kk, vv in v.items() if kk != 'annual_summary'}
                    for k, v in results.items()},
    }
    with open(os.path.join(OUTPUT_DIR, 'tlh_results.json'), 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Results saved.")

    print("\n" + "="*70)
    print("DONE")
    print("="*70)


if __name__ == '__main__':
    main()
