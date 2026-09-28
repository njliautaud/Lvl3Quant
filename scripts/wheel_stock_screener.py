#!/usr/bin/env python3
"""
Wheel Strategy Stock Screener
Finds quality stocks you'd WANT to own if assigned, ranked by wheel suitability.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
import traceback
from scipy.stats import norm

warnings.filterwarnings('ignore')

# ── Universe ──────────────────────────────────────────────────────────────────
UNIVERSE = sorted(set([
    # Large-cap ETFs
    'SPY', 'QQQ', 'IWM', 'DIA',
    # Sector ETFs
    'XLK', 'XLF', 'XLV', 'XLE', 'XLY', 'XLP', 'XLI', 'XLU', 'XLRE', 'XLC',
    # Dividend aristocrats / blue chips
    'KO', 'JNJ', 'PG', 'PEP', 'MCD', 'HD', 'ABBV', 'WMT', 'COST', 'MRK', 'PFE', 'LLY', 'UNH',
    # Quality tech
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA',
    # Financials
    'JPM', 'BAC', 'GS', 'V', 'MA',
    # Healthcare (some overlap, set dedupes)
    'UNH', 'JNJ', 'ABBV', 'MRK', 'PFE', 'LLY',
    # Industrials
    'CAT', 'HON', 'GE', 'MMM', 'UPS',
    # Consumer
    'WMT', 'COST', 'TGT', 'NKE', 'SBUX',
]))

# ── Own-it assessments ────────────────────────────────────────────────────────
OWN_IT = {
    'AAPL': 'Fortress balance sheet, massive buybacks, sticky ecosystem',
    'ABBV': 'Strong pharma pipeline, 50+ yr dividend growth streak',
    'AMZN': 'Cloud + e-commerce duopoly, unmatched logistics moat',
    'BAC': 'Largest US consumer bank, benefits from higher rates',
    'CAT': 'Global infrastructure leader, pricing power in all cycles',
    'COST': 'Membership model = recurring revenue, cult-like customer loyalty',
    'DIA': 'Dow 30 basket: blue-chip diversification in one ticker',
    'GE': 'Aerospace + energy spinoff play, leaner and focused',
    'GOOGL': 'Search monopoly + YouTube + Cloud, massive cash generation',
    'GS': 'Premier investment bank, trading + advisory powerhouse',
    'HD': 'Housing repair/remodel is non-cyclical, massive scale advantage',
    'HON': 'Diversified industrial with aerospace, automation, materials',
    'IWM': 'Small-cap exposure for contrarian wheel plays',
    'JNJ': 'Healthcare conglomerate, 60+ yr dividend increases',
    'JPM': 'Best-run bank in America, fortress balance sheet',
    'KO': 'Global brand moat, 60+ yr dividend king, recession-proof',
    'LLY': 'GLP-1 drug franchise (Mounjaro/Zepbound) is generational',
    'MA': 'Duopoly on global payments, asset-light, massive margins',
    'MCD': 'Real estate empire disguised as fast food, global moat',
    'META': 'Social media monopoly, AI pivot, massive cash flow',
    'MMM': '100+ yr industrial conglomerate, restructuring upside',
    'MRK': 'Keytruda oncology franchise, strong pipeline',
    'MSFT': 'Cloud + AI leader, enterprise lock-in, dividend grower',
    'NVDA': 'AI compute monopoly, datacenter GPU demand secular',
    'NKE': 'Global brand dominance, DTC pivot, athletic wear leader',
    'PEP': 'Snacks + beverages diversification, consistent dividend grower',
    'PFE': 'Deep value pharma, rebuilding pipeline post-COVID',
    'PG': 'Consumer staples king, pricing power, 65+ yr dividend growth',
    'QQQ': 'Nasdaq 100 = concentrated quality tech exposure',
    'SBUX': 'Global coffee monopoly, 35k+ stores, loyalty program moat',
    'SPY': 'S&P 500 = the market, ultimate diversification for wheel',
    'TGT': 'Discount retail with style, suburban stronghold',
    'UNH': 'Healthcare + insurance giant, aging population tailwind',
    'UPS': 'Logistics duopoly with FedEx, e-commerce backbone',
    'V': 'Global payments network, toll-booth model, 60%+ margins',
    'WMT': 'Largest retailer on Earth, e-commerce growth, dividend aristocrat',
    'XLC': 'Communication services sector: META + GOOGL heavy',
    'XLE': 'Energy sector ETF, good for high-IV wheel premium',
    'XLF': 'Financial sector basket, rate-sensitive premium plays',
    'XLI': 'Industrial sector: infrastructure + defense exposure',
    'XLK': 'Tech sector ETF: AAPL + MSFT heavy, quality growth',
    'XLP': 'Consumer staples: recession-proof defensive basket',
    'XLRE': 'Real estate sector: rate-sensitive but high-yield',
    'XLU': 'Utilities sector: bond proxy, stable dividends',
    'XLV': 'Healthcare sector: aging demographics tailwind',
    'XLY': 'Consumer discretionary: AMZN + TSLA heavy',
}


def black_scholes_put(S, K, T, r, sigma):
    """Black-Scholes put price."""
    if T <= 0 or sigma <= 0:
        return max(K - S, 0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def bs_put_greeks(S, K, T, r, sigma):
    """Return dict of put greeks."""
    if T <= 0 or sigma <= 0:
        return {'delta': -1.0 if S < K else 0.0, 'gamma': 0, 'theta': 0, 'vega': 0, 'price': max(K-S, 0)}
    sqrt_T = np.sqrt(T)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    nd1 = norm.pdf(d1)
    price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    delta = norm.cdf(d1) - 1  # put delta is negative
    gamma = nd1 / (S * sigma * sqrt_T)
    theta = (-(S * nd1 * sigma) / (2 * sqrt_T) + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365
    vega = S * nd1 * sqrt_T / 100  # per 1% vol move
    return {'delta': delta, 'gamma': gamma, 'theta': theta, 'vega': vega, 'price': price}


def find_strike_for_delta(S, T, r, sigma, target_delta=-0.20, precision=0.25):
    """Find the put strike closest to target delta, snapped to $0.50 or $1 increments."""
    # Binary search for strike
    lo, hi = S * 0.70, S * 1.0
    for _ in range(100):
        mid = (lo + hi) / 2
        g = bs_put_greeks(S, mid, T, r, sigma)
        if g['delta'] < target_delta:  # too deep ITM
            hi = mid
        else:
            lo = mid
    raw_strike = (lo + hi) / 2
    # Snap to nearest standard increment
    if S > 100:
        increment = 1.0
    elif S > 20:
        increment = 0.50
    else:
        increment = 0.50
    strike = round(raw_strike / increment) * increment
    return strike


def get_atm_iv(ticker_obj, price):
    """Try to get ATM implied vol from options chain. Return IV or None."""
    try:
        expirations = ticker_obj.options
        if not expirations:
            return None
        # Find expiration closest to 30 days
        today = datetime.now()
        target = today + timedelta(days=30)
        best_exp = None
        best_diff = 999
        for exp in expirations:
            exp_date = datetime.strptime(exp, '%Y-%m-%d')
            diff = abs((exp_date - target).days)
            if diff < best_diff:
                best_diff = diff
                best_exp = exp
        if best_exp is None or best_diff > 45:
            return None
        chain = ticker_obj.option_chain(best_exp)
        puts = chain.puts
        if puts.empty:
            return None
        # Find ATM put (closest strike to current price)
        puts = puts.copy()
        puts['dist'] = abs(puts['strike'] - price)
        atm = puts.loc[puts['dist'].idxmin()]
        iv = atm.get('impliedVolatility', None)
        if iv is not None and iv > 0:
            return float(iv)
        return None
    except Exception:
        return None


def score_quality(info):
    """Score stock quality 0-100 based on fundamentals."""
    score = 50  # base score

    # Market cap: large cap preferred
    mcap = info.get('marketCap', 0) or 0
    if mcap > 200e9:
        score += 15
    elif mcap > 50e9:
        score += 10
    elif mcap > 10e9:
        score += 5

    # Profit margins
    margins = info.get('profitMargins', None)
    if margins is not None:
        if margins > 0.25:
            score += 12
        elif margins > 0.15:
            score += 8
        elif margins > 0.05:
            score += 4
        elif margins < 0:
            score -= 10

    # ROE
    roe = info.get('returnOnEquity', None)
    if roe is not None:
        if roe > 0.25:
            score += 10
        elif roe > 0.15:
            score += 7
        elif roe > 0.08:
            score += 3
        elif roe < 0:
            score -= 5

    # PE ratio: reasonable valuation
    pe = info.get('trailingPE', None) or info.get('forwardPE', None)
    if pe is not None:
        if 8 < pe < 20:
            score += 8  # value
        elif 20 <= pe < 30:
            score += 5  # reasonable
        elif 30 <= pe < 50:
            score += 2  # growth premium
        elif pe > 80:
            score -= 5  # expensive

    # Debt to equity: lower is better
    dte = info.get('debtToEquity', None)
    if dte is not None:
        if dte < 50:
            score += 5
        elif dte < 100:
            score += 2
        elif dte > 200:
            score -= 5

    return min(max(score, 0), 100)


def score_premium(iv):
    """Score premium yield potential 0-100. Sweet spot 20-40% IV."""
    if iv is None:
        return 30  # neutral if unknown
    iv_pct = iv * 100
    if 25 <= iv_pct <= 40:
        return 90  # sweet spot
    elif 20 <= iv_pct < 25:
        return 75
    elif 40 < iv_pct <= 55:
        return 70  # high but manageable
    elif 15 <= iv_pct < 20:
        return 55
    elif iv_pct > 55:
        return 50  # too volatile, risky assignment
    elif iv_pct < 15:
        return 30  # barely worth the wheel
    return 30


def score_price(price):
    """Score price suitability 0-100 for wheel strategy."""
    if price is None or price <= 0:
        return 0
    if 50 <= price <= 150:
        return 95  # ideal
    elif 30 <= price < 50 or 150 < price <= 250:
        return 80
    elif 20 <= price < 30 or 250 < price <= 350:
        return 60
    elif 350 < price <= 500:
        return 40
    elif price > 500:
        return 20  # too much capital
    elif price < 20:
        return 35  # too cheap, probably not quality
    return 30


def score_dividend(div_yield):
    """Score dividend 0-100."""
    if div_yield is None or div_yield <= 0:
        return 20  # no dividend isn't disqualifying for wheel
    dy = div_yield * 100
    if 2.0 <= dy <= 4.0:
        return 90  # sweet spot
    elif 1.0 <= dy < 2.0:
        return 70
    elif 4.0 < dy <= 6.0:
        return 75
    elif dy > 6.0:
        return 50  # yield trap risk
    elif 0 < dy < 1.0:
        return 45
    return 20


def fmt_mcap(mcap):
    if mcap is None:
        return 'N/A'
    if mcap >= 1e12:
        return f'${mcap/1e12:.1f}T'
    elif mcap >= 1e9:
        return f'${mcap/1e9:.0f}B'
    elif mcap >= 1e6:
        return f'${mcap/1e6:.0f}M'
    return f'${mcap:,.0f}'


def main():
    print(f"Wheel Stock Screener - {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"Universe: {len(UNIVERSE)} tickers\n")

    results = []
    errors = []

    for i, ticker in enumerate(UNIVERSE):
        print(f"  [{i+1}/{len(UNIVERSE)}] {ticker}...", end=' ', flush=True)
        try:
            t = yf.Ticker(ticker)
            info = t.info or {}

            price = info.get('currentPrice') or info.get('regularMarketPrice') or info.get('previousClose')
            if price is None or price <= 0:
                print("NO PRICE")
                errors.append((ticker, "No price data"))
                continue

            mcap = info.get('marketCap', 0) or 0
            pe = info.get('trailingPE', None)
            fwd_pe = info.get('forwardPE', None)
            div_yield = info.get('dividendYield', None)
            margins = info.get('profitMargins', None)
            roe = info.get('returnOnEquity', None)
            dte = info.get('debtToEquity', None)

            # Get IV from options
            iv = get_atm_iv(t, price)

            # Calculate scores
            q_score = score_quality(info)
            p_score = score_premium(iv)
            px_score = score_price(price)
            d_score = score_dividend(div_yield)

            # Weighted composite: Quality 40%, Premium 25%, Price 20%, Dividend 15%
            wheel_score = (q_score * 0.40 + p_score * 0.25 + px_score * 0.20 + d_score * 0.15)

            # Estimate monthly premium at 0.20 delta CSP (30 DTE)
            est_iv = iv if iv else 0.20
            T = 30 / 365
            r = 0.05  # risk-free rate
            strike_020 = find_strike_for_delta(price, T, r, est_iv, target_delta=-0.20)
            premium_per_share = black_scholes_put(price, strike_020, T, r, est_iv)
            collateral = strike_020 * 100
            monthly_yield = (premium_per_share / strike_020) * 100 if strike_020 > 0 else 0
            annual_yield = monthly_yield * 12

            own_it = OWN_IT.get(ticker, 'Solid company in a quality universe')

            results.append({
                'Ticker': ticker,
                'Price': price,
                'MarketCap': mcap,
                'MarketCapFmt': fmt_mcap(mcap),
                'PE': pe,
                'FwdPE': fwd_pe,
                'DivYield': div_yield,
                'Margins': margins,
                'ROE': roe,
                'DebtEquity': dte,
                'IV': iv,
                'IV_used': est_iv,
                'QualityScore': q_score,
                'PremiumScore': p_score,
                'PriceScore': px_score,
                'DividendScore': d_score,
                'WheelScore': wheel_score,
                'Strike020': strike_020,
                'PremiumPerShare': premium_per_share,
                'Collateral': collateral,
                'MonthlyYield': monthly_yield,
                'AnnualYield': annual_yield,
                'OwnIt': own_it,
            })
            print(f"${price:.2f}  IV={est_iv*100:.1f}%  Score={wheel_score:.1f}")

        except Exception as e:
            print(f"ERROR: {e}")
            errors.append((ticker, str(e)))

    if not results:
        print("\nNo results! Check network/yfinance.")
        return

    # Sort by wheel score descending
    df = pd.DataFrame(results).sort_values('WheelScore', ascending=False).reset_index(drop=True)

    # ── Print Top 30 Table ────────────────────────────────────────────────────
    top30 = df.head(30)

    print("\n" + "="*120)
    print(f"  TOP 30 WHEEL STRATEGY CANDIDATES  ({datetime.now().strftime('%Y-%m-%d')})")
    print("="*120)
    print(f"{'#':>3} {'Ticker':<6} {'Price':>8} {'MktCap':>8} {'PE':>7} {'DivYld':>7} {'IV':>6} {'Score':>6} {'Mo.Yld':>7} {'Collat':>9} {'Own It?'}")
    print("-"*120)

    for idx, row in top30.iterrows():
        rank = idx + 1
        pe_str = f"{row['PE']:.1f}" if row['PE'] else 'N/A'
        div_str = f"{row['DivYield']*100:.2f}%" if row['DivYield'] else 'N/A'
        iv_str = f"{row['IV']*100:.1f}%" if row['IV'] else 'est'
        print(f"{rank:>3} {row['Ticker']:<6} ${row['Price']:>7.2f} {row['MarketCapFmt']:>8} {pe_str:>7} {div_str:>7} "
              f"{iv_str:>6} {row['WheelScore']:>5.1f} {row['MonthlyYield']:>6.2f}% ${row['Collateral']:>8,.0f}  {row['OwnIt'][:60]}")

    # ── Top 10 Detailed Greeks ────────────────────────────────────────────────
    top10 = df.head(10)
    r = 0.05
    T = 30 / 365

    print("\n\n" + "="*120)
    print("  TOP 10 DETAILED: 30-DTE 0.20-DELTA CASH-SECURED PUT")
    print("="*120)

    greeks_data = []
    for idx, row in top10.iterrows():
        S = row['Price']
        sigma = row['IV_used']
        strike = row['Strike020']
        g = bs_put_greeks(S, strike, T, r, sigma)

        premium_total = g['price'] * 100
        eff_buy = strike - g['price']
        ann_yield = (g['price'] / strike) * 12 * 100

        greeks_data.append({
            'Ticker': row['Ticker'],
            'Price': S,
            'Strike': strike,
            'Premium': g['price'],
            'PremiumTotal': premium_total,
            'Delta': g['delta'],
            'Theta': g['theta'],
            'Gamma': g['gamma'],
            'Vega': g['vega'],
            'MonthlyIncome': premium_total,
            'AnnYield': ann_yield,
            'EffBuyPrice': eff_buy,
            'Collateral': strike * 100,
            'IV': row['IV_used'],
            'WheelScore': row['WheelScore'],
        })

    print(f"\n{'Ticker':<6} {'Price':>8} {'Strike':>8} {'Prem/sh':>8} {'Delta':>7} {'Theta/d':>8} {'Gamma':>8} {'Vega':>7} "
          f"{'Mo.Inc':>8} {'Ann.Yld':>8} {'Eff.Buy':>8}")
    print("-"*110)

    for g in greeks_data:
        print(f"{g['Ticker']:<6} ${g['Price']:>7.2f} ${g['Strike']:>7.2f} ${g['Premium']:>7.2f} "
              f"{g['Delta']:>7.3f} ${g['Theta']:>7.2f} {g['Gamma']:>8.5f} ${g['Vega']:>6.2f} "
              f"${g['MonthlyIncome']:>7.0f} {g['AnnYield']:>7.1f}% ${g['EffBuyPrice']:>7.2f}")

    # ── Save to markdown ──────────────────────────────────────────────────────
    out_path = '/home/jupiter/Lvl3Quant/research/findings/wheel_stock_screener_results.md'

    lines = []
    lines.append(f"# Wheel Strategy Stock Screener Results")
    lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M ET')}")
    lines.append(f"**Universe:** {len(UNIVERSE)} tickers | **Scored:** {len(df)} | **Failed:** {len(errors)}")
    lines.append(f"**Scoring:** Quality 40% + Premium Yield 25% + Price Suitability 20% + Dividend 15%")
    lines.append("")
    lines.append("## Top 30 Wheel Candidates")
    lines.append("")
    lines.append("| # | Ticker | Price | Mkt Cap | PE | Div Yield | IV (30d) | Wheel Score | Mo. Yield | Collateral | Why Own It |")
    lines.append("|---|--------|-------|---------|-----|-----------|----------|-------------|-----------|------------|------------|")

    for idx, row in top30.iterrows():
        rank = idx + 1
        pe_str = f"{row['PE']:.1f}" if row['PE'] else 'N/A'
        div_str = f"{row['DivYield']*100:.2f}%" if row['DivYield'] else '-'
        iv_str = f"{row['IV']*100:.1f}%" if row['IV'] else 'est'
        lines.append(f"| {rank} | **{row['Ticker']}** | ${row['Price']:.2f} | {row['MarketCapFmt']} | {pe_str} | {div_str} | "
                      f"{iv_str} | **{row['WheelScore']:.1f}** | {row['MonthlyYield']:.2f}% | ${row['Collateral']:,.0f} | {row['OwnIt'][:80]} |")

    lines.append("")
    lines.append("## Top 10 Detailed: 30-DTE 0.20-Delta Cash-Secured Put")
    lines.append("")
    lines.append("| Ticker | Price | Strike | Prem/Share | Delta | Theta/Day | Gamma | Vega | Mo. Income | Ann. Yield | Eff. Buy Price |")
    lines.append("|--------|-------|--------|------------|-------|-----------|-------|------|------------|------------|----------------|")

    for g in greeks_data:
        lines.append(f"| **{g['Ticker']}** | ${g['Price']:.2f} | ${g['Strike']:.2f} | ${g['Premium']:.2f} | "
                      f"{g['Delta']:.3f} | ${g['Theta']:.2f} | {g['Gamma']:.5f} | ${g['Vega']:.2f} | "
                      f"${g['MonthlyIncome']:.0f} | {g['AnnYield']:.1f}% | ${g['EffBuyPrice']:.2f} |")

    lines.append("")
    lines.append("## Scoring Methodology")
    lines.append("")
    lines.append("- **Quality (40%)**: Market cap, profit margins, ROE, PE valuation, debt levels")
    lines.append("- **Premium Yield (25%)**: ATM implied volatility. Sweet spot 20-40% IV")
    lines.append("- **Price Suitability (20%)**: $50-$150 ideal for wheel (affordable collateral)")
    lines.append("- **Dividend (15%)**: Extra income if assigned. 2-4% yield is sweet spot")
    lines.append("")
    lines.append("## Key Takeaways")
    lines.append("")

    # Generate takeaways
    best = greeks_data[0]
    highest_yield = max(greeks_data, key=lambda x: x['AnnYield'])
    cheapest = min(greeks_data, key=lambda x: x['Collateral'])

    lines.append(f"- **Best overall**: {best['Ticker']} (Score {best['WheelScore']:.1f}) - quality + premium balance")
    lines.append(f"- **Highest yield**: {highest_yield['Ticker']} at {highest_yield['AnnYield']:.1f}% annualized")
    lines.append(f"- **Most capital-efficient**: {cheapest['Ticker']} at ${cheapest['Collateral']:,.0f} collateral per contract")
    lines.append(f"- All top 10 are stocks you'd genuinely want to own if assigned")

    if errors:
        lines.append("")
        lines.append(f"## Errors ({len(errors)} tickers)")
        lines.append("")
        for ticker, err in errors:
            lines.append(f"- {ticker}: {err[:80]}")

    with open(out_path, 'w') as f:
        f.write('\n'.join(lines))

    print(f"\n\nResults saved to: {out_path}")

    # Also print a quick summary
    print("\n" + "="*60)
    print("QUICK SUMMARY")
    print("="*60)
    print(f"Best overall:       {best['Ticker']} (Score {best['WheelScore']:.1f})")
    print(f"Highest yield:      {highest_yield['Ticker']} ({highest_yield['AnnYield']:.1f}% ann.)")
    print(f"Capital efficient:  {cheapest['Ticker']} (${cheapest['Collateral']:,.0f}/contract)")
    print(f"Top 5: {', '.join(df.head(5)['Ticker'].tolist())}")

    if errors:
        print(f"\nFailed tickers ({len(errors)}): {', '.join(t for t,_ in errors)}")


if __name__ == '__main__':
    main()
