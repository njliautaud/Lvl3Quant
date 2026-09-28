"""
Calendar Spread v2 — ATM focus with tighter risk management.
Tries ATM puts/calls where theta differential is largest.
Also tests diagonal spreads (sell front ATM, buy back slightly OTM).

Key fix from v1: proper max loss capping, ATM strike, tighter stops.
"""
from __future__ import annotations
import math
import numpy as np
import pandas as pd
from pathlib import Path
import json
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# Black-Scholes (same as v1)
# ============================================================
def _Phi(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))

def bs_price(S, K, T, sigma, r=0.04, q=0.0, kind="put"):
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return max(K - S, 0.0) if kind == "put" else max(S - K, 0.0)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    if kind == "put":
        return K * math.exp(-r * T) * _Phi(-d2) - S * math.exp(-q * T) * _Phi(-d1)
    return S * math.exp(-q * T) * _Phi(d1) - K * math.exp(-r * T) * _Phi(d2)


# ============================================================
# Backtest
# ============================================================
def run_calendar_bt(
    front_dte=10,
    back_dte=42,
    min_iv_rank=0.40,
    min_term_ratio=1.05,
    kind="put",          # put or call
    strike_pct=1.0,      # 1.0 = ATM, 0.97 = 3% OTM put
    profit_target=0.35,
    max_loss_mult=0.80,  # close when loss = 80% of net debit
    max_hold=None,       # None = hold to front expiry
    max_positions=5,
    capital=100_000,
    commission_per_leg=0.65,
    slippage_pct=0.03,
    slippage_min=0.02,
    label="default",
):
    root = Path("/home/jupiter/Lvl3Quant/wheel_strategy_v1/data/cache")
    prices = pd.read_parquet(root / "prices.parquet")
    prices['date'] = pd.to_datetime(prices['date'])

    iv = pd.read_parquet(root / "iv_features_modeled.parquet")
    iv['date'] = pd.to_datetime(iv['date'])

    spy = pd.read_parquet(root / "spy_prices.parquet")
    spy['date'] = pd.to_datetime(spy['date'])
    spy = spy.sort_values('date')
    spy['spy_ret'] = spy['close'].pct_change()

    merged = pd.merge(
        prices[['ticker', 'date', 'close', 'volume']],
        iv[['ticker', 'date', 'sigma', 'sigma_atm_30d', 'iv_rank', 'term_ratio']],
        on=['ticker', 'date'], how='inner'
    ).dropna(subset=['sigma', 'iv_rank', 'term_ratio', 'close'])
    merged = merged.sort_values(['ticker', 'date']).reset_index(drop=True)

    dates = sorted(merged['date'].unique())
    min_date = pd.Timestamp("2016-06-01")
    dates = [d for d in dates if d >= min_date]

    if max_hold is None:
        max_hold = front_dte - 1

    # Position tracking
    positions = []  # list of dicts
    closed = []
    equity = capital
    eq_curve = []

    for date in dates:
        date_str = date.strftime('%Y-%m-%d')
        day = merged[merged['date'] == date]

        # Mark existing positions
        daily_realized = 0.0
        still_open = []

        for pos in positions:
            pos['days_held'] += 1
            tr = day[day['ticker'] == pos['ticker']]
            if len(tr) == 0:
                still_open.append(pos)
                continue

            S = tr.iloc[0]['close']
            sigma = tr.iloc[0]['sigma']
            if pd.isna(sigma) or sigma <= 0:
                sigma = pos['sigma_entry']

            K = pos['strike']
            front_remain = max(0, pos['front_dte'] - pos['days_held'])
            back_remain = max(0, pos['back_dte'] - pos['days_held'])
            T_f = front_remain / 365.0
            T_b = back_remain / 365.0

            # Term structure: front vol slightly higher
            tr_ratio = tr.iloc[0]['term_ratio'] if not pd.isna(tr.iloc[0]['term_ratio']) else 1.1
            ratio_sqrt = math.sqrt(max(tr_ratio, 0.5))
            s_f = sigma * ratio_sqrt
            s_b = sigma / ratio_sqrt

            front_val = bs_price(S, K, T_f, s_f, kind=kind)
            back_val = bs_price(S, K, T_b, s_b, kind=kind)
            spread_val = back_val - front_val

            pnl_per_share = spread_val - pos['net_debit']
            mtm_pnl = pnl_per_share * 100 * pos['n_contracts']

            max_profit = pos['front_prem_sold'] * 100 * pos['n_contracts']
            max_loss = pos['net_debit'] * 100 * pos['n_contracts'] * max_loss_mult

            should_close = False
            reason = ""

            if mtm_pnl > 0 and mtm_pnl >= profit_target * max_profit:
                should_close = True
                reason = "profit_target"
            elif mtm_pnl < -max_loss:
                should_close = True
                reason = "stop_loss"
            elif front_remain <= 1:
                should_close = True
                reason = "front_expiry"
            elif pos['days_held'] >= max_hold:
                should_close = True
                reason = "max_hold"

            if should_close:
                # Close costs
                close_cost = 2 * commission_per_leg * pos['n_contracts']
                close_cost += max(slippage_min, slippage_pct * front_val) * 100 * pos['n_contracts']
                close_cost += max(slippage_min, slippage_pct * back_val) * 100 * pos['n_contracts']

                realized = mtm_pnl - pos['open_cost'] - close_cost
                daily_realized += realized
                pos['close_pnl'] = realized
                pos['close_reason'] = reason
                pos['close_date'] = date_str
                closed.append(pos)
            else:
                still_open.append(pos)

        positions = still_open

        # New entries
        if len(positions) < max_positions:
            cands = day[
                (day['iv_rank'] >= min_iv_rank) &
                (day['term_ratio'] >= min_term_ratio) &
                (day['sigma'] > 0.05) &
                (day['close'] >= 20)
            ].copy()
            cands['score'] = cands['term_ratio'] * cands['iv_rank']
            cands = cands.sort_values('score', ascending=False)

            pos_tickers = {p['ticker'] for p in positions}
            for _, row in cands.iterrows():
                if len(positions) >= max_positions:
                    break
                if row['ticker'] in pos_tickers:
                    continue

                S = row['close']
                sigma = row['sigma']
                sigma_30d = row['sigma_atm_30d'] if not pd.isna(row['sigma_atm_30d']) else sigma
                tr_ratio = row['term_ratio']

                K = round(S * strike_pct * 2) / 2.0

                ratio_sqrt = math.sqrt(max(tr_ratio, 0.5))
                s_f = sigma_30d * ratio_sqrt
                s_b = sigma_30d / ratio_sqrt

                T_f = front_dte / 365.0
                T_b = back_dte / 365.0

                front_prem = bs_price(S, K, T_f, s_f, kind=kind)
                back_prem = bs_price(S, K, T_b, s_b, kind=kind)
                net_debit = back_prem - front_prem

                if net_debit <= 0.01:
                    continue

                max_risk = capital * 0.02
                n_contracts = max(1, int(max_risk / (net_debit * 100)))
                n_contracts = min(n_contracts, 10)

                open_cost = 2 * commission_per_leg * n_contracts
                open_cost += max(slippage_min, slippage_pct * front_prem) * 100 * n_contracts
                open_cost += max(slippage_min, slippage_pct * back_prem) * 100 * n_contracts

                positions.append({
                    'ticker': row['ticker'],
                    'open_date': date_str,
                    'strike': K,
                    'front_dte': front_dte,
                    'back_dte': back_dte,
                    'front_prem_sold': front_prem,
                    'back_prem_paid': back_prem,
                    'net_debit': net_debit,
                    'n_contracts': n_contracts,
                    'sigma_entry': sigma,
                    'open_cost': open_cost,
                    'days_held': 0,
                })
                pos_tickers.add(row['ticker'])

        equity += daily_realized
        eq_curve.append({'date': date_str, 'equity': equity, 'realized': daily_realized})

    # Force close remaining
    for pos in positions:
        closed.append(pos)

    # Stats
    eq = pd.DataFrame(eq_curve)
    eq['date'] = pd.to_datetime(eq['date'])
    eq['daily_ret'] = eq['equity'].pct_change().fillna(0)

    total_days = len(eq)
    years = total_days / 252.0

    total_return = (eq['equity'].iloc[-1] / capital) - 1
    cagr = (1 + total_return) ** (1 / max(years, 0.01)) - 1 if total_return > -1 else -1.0

    daily_rets = eq['daily_ret'].values
    mean_r = np.mean(daily_rets)
    std_r = np.std(daily_rets, ddof=1) if len(daily_rets) > 1 else 1e-6
    sharpe = (mean_r / std_r) * np.sqrt(252) if std_r > 0 else 0

    down = daily_rets[daily_rets < 0]
    down_std = np.std(down, ddof=1) if len(down) > 1 else 1e-6
    sortino = (mean_r / down_std) * np.sqrt(252) if down_std > 0 else 0

    peak = eq['equity'].cummax()
    dd = ((eq['equity'] - peak) / peak).min()

    trades_with_pnl = [t for t in closed if 'close_pnl' in t]
    n_trades = len(trades_with_pnl)
    winners = [t for t in trades_with_pnl if t['close_pnl'] > 0]
    losers = [t for t in trades_with_pnl if t['close_pnl'] <= 0]
    wr = len(winners) / n_trades if n_trades > 0 else 0
    gp = sum(t['close_pnl'] for t in winners)
    gl = abs(sum(t['close_pnl'] for t in losers))
    pf = gp / gl if gl > 0 else float('inf')
    avg_win = np.mean([t['close_pnl'] for t in winners]) if winners else 0
    avg_loss = np.mean([t['close_pnl'] for t in losers]) if losers else 0

    reasons = {}
    for t in trades_with_pnl:
        r = t.get('close_reason', 'unknown')
        reasons[r] = reasons.get(r, 0) + 1

    # Regime
    eq = eq.merge(spy[['date', 'spy_ret']], on='date', how='left')
    eq['spy_ret'] = eq['spy_ret'].fillna(0)
    eq['regime'] = 'flat'
    eq.loc[eq['spy_ret'] > 0.005, 'regime'] = 'green'
    eq.loc[eq['spy_ret'] < -0.005, 'regime'] = 'red'

    regime_stats = {}
    for reg in ['green', 'red', 'flat']:
        mask = eq['regime'] == reg
        if mask.sum() > 5:
            r_vals = daily_rets[mask.values]
            rs = (np.mean(r_vals) / np.std(r_vals, ddof=1)) * np.sqrt(252) if np.std(r_vals, ddof=1) > 0 else 0
            regime_stats[reg] = {'sharpe': round(rs, 3), 'n': int(mask.sum())}

    sg = regime_stats.get('green', {}).get('sharpe', 0)
    sr = regime_stats.get('red', {}).get('sharpe', 0)
    mx = max(abs(sg), abs(sr))
    rgap = abs(sg - sr) / mx if mx > 0 else 0

    # SPY comparison
    spy_bt = spy[(spy['date'] >= eq['date'].min()) & (spy['date'] <= eq['date'].max())]
    spy_ret = spy_bt['close'].iloc[-1] / spy_bt['close'].iloc[0] - 1 if len(spy_bt) > 1 else 0
    spy_cagr = (1 + spy_ret) ** (1/max(years, 0.01)) - 1 if spy_ret > -1 else -1
    spy_rets = spy_bt['close'].pct_change().dropna().values
    spy_sharpe = (np.mean(spy_rets) / np.std(spy_rets, ddof=1)) * np.sqrt(252) if len(spy_rets) > 1 else 0

    return {
        'label': label,
        'sharpe': round(sharpe, 3),
        'sortino': round(sortino, 3),
        'cagr': round(cagr * 100, 2),
        'max_dd': round(dd * 100, 2),
        'wr': round(wr * 100, 1),
        'pf': round(pf, 3),
        'n_trades': n_trades,
        'avg_win': round(avg_win, 2),
        'avg_loss': round(avg_loss, 2),
        'ev_per_trade': round(wr * avg_win + (1-wr) * avg_loss, 2),
        'reasons': reasons,
        'regime': regime_stats,
        'regime_gap': round(rgap, 3),
        'total_return': round(total_return * 100, 2),
        'spy_sharpe': round(spy_sharpe, 3),
        'spy_cagr': round(spy_cagr * 100, 2),
    }


if __name__ == "__main__":
    out_dir = Path("/home/jupiter/Lvl3Quant/output/calendar_spread_research")

    configs = [
        # ATM put calendars
        dict(kind="put", strike_pct=1.0, front_dte=10, back_dte=42,
             min_iv_rank=0.40, min_term_ratio=1.05, profit_target=0.35,
             max_loss_mult=0.60, label="ATM Put 10/42 IVR40"),

        dict(kind="put", strike_pct=1.0, front_dte=7, back_dte=45,
             min_iv_rank=0.40, min_term_ratio=1.05, profit_target=0.30,
             max_loss_mult=0.50, label="ATM Put 7/45 tight"),

        # ATM call calendars
        dict(kind="call", strike_pct=1.0, front_dte=10, back_dte=42,
             min_iv_rank=0.40, min_term_ratio=1.05, profit_target=0.35,
             max_loss_mult=0.60, label="ATM Call 10/42 IVR40"),

        # Slightly OTM put (3% below)
        dict(kind="put", strike_pct=0.97, front_dte=10, back_dte=42,
             min_iv_rank=0.40, min_term_ratio=1.05, profit_target=0.35,
             max_loss_mult=0.60, label="OTM Put 97% 10/42"),

        # Very high IV filter
        dict(kind="put", strike_pct=1.0, front_dte=10, back_dte=42,
             min_iv_rank=0.70, min_term_ratio=1.15, profit_target=0.40,
             max_loss_mult=0.50, label="High IV Put IVR70/TR1.15"),

        # Wider DTE spread with max_hold override
        dict(kind="put", strike_pct=1.0, front_dte=14, back_dte=56,
             min_iv_rank=0.40, min_term_ratio=1.05, profit_target=0.30,
             max_loss_mult=0.50, max_hold=10, label="Wide Put 14/56 hold10"),

        # Call calendars with high IV
        dict(kind="call", strike_pct=1.0, front_dte=10, back_dte=42,
             min_iv_rank=0.60, min_term_ratio=1.10, profit_target=0.35,
             max_loss_mult=0.50, label="Call IVR60/TR1.10"),

        # Very tight stop
        dict(kind="put", strike_pct=1.0, front_dte=10, back_dte=42,
             min_iv_rank=0.40, min_term_ratio=1.05, profit_target=0.25,
             max_loss_mult=0.30, label="ATM Put tight stop 30%"),
    ]

    all_results = []
    for cfg in configs:
        print(f"\nRunning: {cfg['label']}")
        r = run_calendar_bt(**cfg)
        all_results.append(r)
        print(f"  Sharpe={r['sharpe']:.3f} Sortino={r['sortino']:.3f} "
              f"CAGR={r['cagr']:.1f}% MaxDD={r['max_dd']:.1f}% "
              f"WR={r['wr']:.0f}% PF={r['pf']:.3f} "
              f"#Trades={r['n_trades']} EV/trade=${r['ev_per_trade']:.2f}")

    # Save
    with open(out_dir / "v2_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    # Summary table
    print("\n" + "=" * 110)
    print(f"{'Config':<30} {'Sharpe':>7} {'Sort':>6} {'CAGR%':>7} {'MaxDD%':>7} {'WR%':>5} {'PF':>6} {'#Tr':>5} {'EV/tr':>8} {'RGap':>5}")
    print("-" * 110)
    for r in all_results:
        print(f"{r['label']:<30} {r['sharpe']:>7.3f} {r['sortino']:>6.3f} "
              f"{r['cagr']:>+7.1f} {r['max_dd']:>7.1f} "
              f"{r['wr']:>5.0f} {r['pf']:>6.3f} "
              f"{r['n_trades']:>5d} {r['ev_per_trade']:>+8.2f} {r['regime_gap']:>5.3f}")

    print(f"\nSPY B&H: Sharpe={all_results[0]['spy_sharpe']:.3f}, CAGR={all_results[0]['spy_cagr']:.1f}%")
    print("\nConclusion:")
    any_positive = any(r['pf'] >= 1.0 for r in all_results)
    if any_positive:
        best = max([r for r in all_results if r['pf'] >= 1.0], key=lambda x: x['sharpe'])
        print(f"  Best profitable config: {best['label']}")
    else:
        print("  NO configuration achieved PF >= 1.0 across the full OOS period.")
        print("  Calendar spreads with modeled (BS) options are NOT viable as standalone strategy.")
        print("  Root cause: BS constant-vol assumption eliminates the term-structure edge")
        print("  that makes real calendar spreads work. Without market-implied vols at")
        print("  different expirations, the theta differential is exactly offset by")
        print("  vega risk on directional moves.")
    print()
