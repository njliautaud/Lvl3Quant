#!/usr/bin/env python3
"""
Covered Call Income Strategy with VIX Timing
=============================================
Core thesis: Systematic covered call writing on SPY generates reliable income.
Enhance with VIX timing:
- Low VIX (<15): Write aggressive calls (closer to ATM = more premium)
- Normal VIX (15-25): Standard 30-delta OTM calls
- High VIX (>25): Either skip writing (let upside run post-spike) or write far OTM

Also test BuyWrite index (BXM) replication and enhancements.

Income source: Option premium decay (theta).
$100K fixed capital (HC #713). No DCA.
Realistic premium estimates using VIX as implied vol proxy.
"""
import sys
sys.path.insert(0, '/home/jupiter/Lvl3Quant/scripts/growth_research/autonomous')
from research_template import *
from scipy.stats import norm


def bs_call_price(S, K, T, r, sigma):
    """Black-Scholes call price"""
    if T <= 0 or sigma <= 0:
        return max(S - K, 0)
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)


def bs_delta(S, K, T, r, sigma):
    """BS delta"""
    if T <= 0 or sigma <= 0:
        return 1.0 if S > K else 0.0
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    return norm.cdf(d1)


def find_strike_for_delta(S, T, r, sigma, target_delta=0.30):
    """Find strike that gives target delta (OTM call)"""
    # Binary search for strike
    low, high = S, S * 1.5
    for _ in range(50):
        mid = (low + high) / 2
        d = bs_delta(S, mid, T, r, sigma)
        if d > target_delta:
            low = mid
        else:
            high = mid
    return mid


def main():
    print("=" * 70)
    print("COVERED CALL INCOME + VIX TIMING")
    print("=" * 70)

    # Download data
    tickers = ['SPY', 'TLT', 'GLD']
    prices = download_etfs(tickers, start='2007-01-01')
    vix = download_vix(start='2007-01-01')
    prices['VIX'] = vix
    prices = prices.dropna()
    print(f"Data: {len(prices)} rows, {prices.index[0].date()} → {prices.index[-1].date()}")

    spy = prices['SPY']
    vix_series = prices['VIX']
    spy_ret = spy.pct_change()

    start = 252  # Need 1yr warmup
    r = 0.02  # Risk-free rate approximation

    configs = {}

    # === BASELINE: BUY AND HOLD SPY ===
    spy_eq = spy.iloc[start:] / spy.iloc[start] * INITIAL_CAPITAL
    configs['SPY B&H'] = spy_eq

    # === STRATEGY 1: STANDARD COVERED CALL (BXM-like) ===
    # Write 30-day, 2% OTM calls monthly
    bxm_eq = pd.Series(INITIAL_CAPITAL, index=prices.index)
    call_premium = 0.0
    call_strike = 0.0
    call_expiry_idx = 0
    shares = INITIAL_CAPITAL / spy.iloc[start]

    for i in range(start, len(prices)):
        # Collect underlying return
        stock_ret = spy_ret.iloc[i]
        bxm_eq.iloc[i] = bxm_eq.iloc[i-1] * (1 + stock_ret)

        # Monthly roll (every 21 trading days)
        if (i - start) % 21 == 0:
            S = spy.iloc[i]
            v = vix_series.iloc[i] / 100  # VIX is in % points
            T = 21/252  # 1 month to expiry

            # 2% OTM call
            K = S * 1.02
            premium = bs_call_price(S, K, T, r, v)
            # Realistic: collect 80% of theoretical (bid-ask, execution)
            premium_pct = (premium / S) * 0.80
            call_strike = K
            call_expiry_idx = i + 21

            # Add premium as income
            bxm_eq.iloc[i] *= (1 + premium_pct)

        # At expiry, if SPY > strike, cap the gain
        if i == call_expiry_idx and call_strike > 0:
            S = spy.iloc[i]
            if S > call_strike:
                # Called away — lose upside above strike
                excess_return = (S - call_strike) / call_strike
                # We lose the excess (it was sold via the call)
                bxm_eq.iloc[i] *= (1 - excess_return * 0.5)  # Approximate capping

    configs['Standard CC (BXM)'] = bxm_eq.iloc[start:]

    # === STRATEGY 2: VIX-TIMED COVERED CALL ===
    # Low VIX: aggressive (closer ATM, more premium)
    # High VIX: skip writing (let recovery run), or very far OTM
    vix_cc_eq = pd.Series(INITIAL_CAPITAL, index=prices.index)

    for i in range(start, len(prices)):
        stock_ret = spy_ret.iloc[i]
        vix_cc_eq.iloc[i] = vix_cc_eq.iloc[i-1] * (1 + stock_ret)

        if (i - start) % 21 == 0:
            S = spy.iloc[i]
            v_raw = vix_series.iloc[i]
            v = v_raw / 100
            T = 21/252

            if v_raw < 15:
                # Low VIX: write closer to money (more premium, but low chance of being hit)
                K = S * 1.01  # 1% OTM
                premium = bs_call_price(S, K, T, r, v)
                premium_pct = (premium / S) * 0.80
            elif v_raw < 25:
                # Normal: standard 2% OTM
                K = S * 1.02
                premium = bs_call_price(S, K, T, r, v)
                premium_pct = (premium / S) * 0.80
            elif v_raw < 35:
                # Elevated: write far OTM (5%) for juicy premium with safety
                K = S * 1.05
                premium = bs_call_price(S, K, T, r, v)
                premium_pct = (premium / S) * 0.80
            else:
                # VIX > 35: DON'T WRITE — let recovery upside run uncapped
                premium_pct = 0
                K = S * 2.0  # Effectively no cap

            call_strike = K
            call_expiry_idx = i + 21
            vix_cc_eq.iloc[i] *= (1 + premium_pct)

        if i == call_expiry_idx and call_strike > 0:
            S = spy.iloc[i]
            if S > call_strike:
                excess_return = (S - call_strike) / call_strike
                vix_cc_eq.iloc[i] *= (1 - excess_return * 0.5)

    configs['VIX-Timed CC'] = vix_cc_eq.iloc[start:]

    # === STRATEGY 3: PUT WRITING + VIX TIMING (cash-secured puts) ===
    # Sell OTM puts: collect premium, buy SPY if it drops
    # Better in high VIX (rich premium) but risk increases
    put_write_eq = pd.Series(INITIAL_CAPITAL, index=prices.index)
    position_in_spy = False

    for i in range(start, len(prices)):
        if position_in_spy:
            # Holding SPY from put assignment
            put_write_eq.iloc[i] = put_write_eq.iloc[i-1] * (1 + spy_ret.iloc[i])
            # Exit if SPY recovers above entry (simplified)
            if spy.iloc[i] > spy.iloc[i-21:i].mean() * 1.02:
                position_in_spy = False
        else:
            put_write_eq.iloc[i] = put_write_eq.iloc[i-1]

        if (i - start) % 21 == 0 and not position_in_spy:
            S = spy.iloc[i]
            v_raw = vix_series.iloc[i]
            v = v_raw / 100
            T = 21/252

            if v_raw > 20:
                # Rich premium: sell 3% OTM put
                K = S * 0.97
            else:
                # Low premium: sell 2% OTM put
                K = S * 0.98

            # Put premium (put-call parity approximation)
            call_price = bs_call_price(S, K, T, r, v)
            put_price = call_price - S + K * np.exp(-r*T)
            put_price = max(put_price, 0)
            premium_pct = (put_price / S) * 0.75  # Conservative fill

            put_write_eq.iloc[i] *= (1 + premium_pct)

            # Check if put was assigned (SPY dropped below strike during month)
            if i + 21 < len(prices):
                future_low = spy.iloc[i:i+21].min()
                if future_low < K:
                    position_in_spy = True

    configs['Put Writing'] = put_write_eq.iloc[start:]

    # === STRATEGY 4: WHEEL STRATEGY (puts → assignment → covered calls → called away → repeat) ===
    wheel_eq = pd.Series(INITIAL_CAPITAL, index=prices.index)
    wheel_state = 'cash'  # 'cash' = selling puts, 'stock' = selling calls
    assignment_price = 0

    for i in range(start, len(prices)):
        if wheel_state == 'stock':
            wheel_eq.iloc[i] = wheel_eq.iloc[i-1] * (1 + spy_ret.iloc[i])
        else:
            wheel_eq.iloc[i] = wheel_eq.iloc[i-1]

        if (i - start) % 21 == 0:
            S = spy.iloc[i]
            v_raw = vix_series.iloc[i]
            v = v_raw / 100
            T = 21/252

            if wheel_state == 'cash':
                # Sell put
                K = S * 0.97
                call_price = bs_call_price(S, K, T, r, v)
                put_price = max(call_price - S + K * np.exp(-r*T), 0)
                premium_pct = (put_price / S) * 0.75
                wheel_eq.iloc[i] *= (1 + premium_pct)

                # Check assignment
                if i + 21 < len(prices) and spy.iloc[i:min(i+21, len(prices))].min() < K:
                    wheel_state = 'stock'
                    assignment_price = K
            else:
                # Sell call
                K = max(S * 1.02, assignment_price)  # At least breakeven
                premium = bs_call_price(S, K, T, r, v)
                premium_pct = (premium / S) * 0.80
                wheel_eq.iloc[i] *= (1 + premium_pct)

                # Check if called away
                if i + 21 < len(prices) and spy.iloc[i:min(i+21, len(prices))].max() > K:
                    wheel_state = 'cash'

    configs['Wheel Strategy'] = wheel_eq.iloc[start:]

    # === STRATEGY 5: INCOME COMBO (CC + VIX spike buying) ===
    # Normal: covered calls for income
    # VIX > 30: deploy capital into SVIX/VIX puts for spike bounce income
    combo_eq = pd.Series(INITIAL_CAPITAL, index=prices.index)
    vix_pct = vix_series.pct_change()

    for i in range(start, len(prices)):
        v_raw = vix_series.iloc[i]

        if v_raw > 30:
            # Spike mode: capture VIX mean-reversion (income from spike buying)
            vix_ret = -0.5 * vix_pct.iloc[i]  # Inverse VIX exposure
            vix_ret = max(vix_ret, -0.05)  # Floor losses
            combo_eq.iloc[i] = combo_eq.iloc[i-1] * (1 + vix_ret * 0.5)  # Half position
            # Other half earns stock return
            combo_eq.iloc[i] *= (1 + spy_ret.iloc[i] * 0.5)
        else:
            # Normal: stock + covered call premium
            combo_eq.iloc[i] = combo_eq.iloc[i-1] * (1 + spy_ret.iloc[i])

            if (i - start) % 21 == 0:
                S = spy.iloc[i]
                v = v_raw / 100
                T = 21/252
                K = S * 1.02
                premium = bs_call_price(S, K, T, r, v)
                premium_pct = (premium / S) * 0.80
                combo_eq.iloc[i] *= (1 + premium_pct)

    configs['CC + Spike Buy'] = combo_eq.iloc[start:]

    # === RESULTS ===
    print(f"\n{'Strategy':<22s} {'Sharpe':>7s} {'Sortino':>8s} {'CAGR':>7s} {'MaxDD':>7s} {'Calmar':>7s}")
    print(f"{'-'*22} {'-'*7} {'-'*8} {'-'*7} {'-'*7} {'-'*7}")

    best_name, best_sharpe = None, -999
    for name, eq in configs.items():
        m = compute_metrics(eq, name)
        print(f"{m['name']:<22s} {m['sharpe']:>7.3f} {m['sortino']:>8.3f} "
              f"{m['cagr']:>6.1%} {m['max_dd']:>6.1%} {m.get('calmar', 0):>7.3f}")
        if name != 'SPY B&H' and m['sharpe'] > best_sharpe:
            best_sharpe = m['sharpe']
            best_name = name

    print(f"\nBest: {best_name}")

    # Income metrics for all strategies
    print(f"\n{'Strategy':<22s} {'Monthly $/100K':>15s}")
    print(f"{'-'*22} {'-'*15}")
    for name, eq in configs.items():
        if name == 'SPY B&H':
            continue
        total_ret = eq.iloc[-1] / eq.iloc[0] - 1
        years = len(eq) / 252
        monthly = (eq.iloc[-1] - eq.iloc[0]) / (years * 12)
        print(f"{name:<22s} ${monthly:>12,.0f}")

    # === ADVERSARIAL ===
    best_eq = configs[best_name]
    metrics = compute_metrics(best_eq, best_name)

    adv = full_adversarial(best_eq, spy_eq)
    print(f"\nAdversarial ({best_name}):")
    print(f"  Perm: p={adv['permutation']['p_value']:.3f} {'PASS' if adv['permutation']['pass'] else 'FAIL'}")
    print(f"  SubP: CV={adv['subperiod']['cv']:.3f} {'PASS' if adv['subperiod']['pass'] else 'FAIL'}")
    print(f"  R1: gap={adv['regime']['gap']:.3f} {'PASS' if adv['regime']['pass'] else 'FAIL'}")
    print(f"  Gates: {adv['gates_passed']}/3")

    # Income summary
    total_ret = best_eq.iloc[-1] / best_eq.iloc[0] - 1
    years = len(best_eq) / 252
    monthly_income = (best_eq.iloc[-1] - best_eq.iloc[0]) / (years * 12)

    emit_result(
        name=f"Covered Call Income ({best_name})",
        description="Systematic covered call/put writing with VIX timing for income generation",
        metrics=metrics,
        adversarial=adv,
        extra={
            'monthly_income_100k': round(float(monthly_income), 0),
            'income_source': 'option theta decay',
            'frequency': 'monthly rolls',
        }
    )


if __name__ == '__main__':
    main()
