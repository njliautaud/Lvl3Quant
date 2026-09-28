#!/usr/bin/env python3
"""
IV vs Realized Volatility Spread Trading Backtest
==================================================
Walk-forward OOT: Jan 2022 - Jul 2026
Account: $645 Robinhood
6 Variants: A-F

5-Gate Validation:
  1. Sharpe > 0.5
  2. Permutation p-value < 0.05
  3. Regime Sharpe gap < 0.5
  4. Max DD > -50%
  5. >= 20 trades
"""

import json
import warnings
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats

warnings.filterwarnings("ignore")
np.random.seed(42)

# ── Constants ──────────────────────────────────────────────────────────────────
ACCOUNT = 645.0
MAX_TRADE_SIZE = 200.0
COMMISSION_PER_CONTRACT = 0.65
BA_HAIRCUT = 0.10  # 10% of premium on entry AND exit
START_DATE = "2020-01-01"  # extra lookback for indicators
OOT_START = "2022-01-01"
OOT_END = "2026-07-28"
N_PERMUTATIONS = 1000

GROWTH_TICKERS = ["TSLA", "NVDA", "AMD", "NFLX", "COIN"]
ALL_TICKERS = ["SPY", "QQQ", "^VIX"] + GROWTH_TICKERS

# ── Data Download ──────────────────────────────────────────────────────────────
print("Downloading data...")
data = {}
for t in ALL_TICKERS:
    try:
        df = yf.download(t, start=START_DATE, end=OOT_END, progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        if len(df) > 50:
            data[t] = df
            print(f"  {t}: {len(df)} rows")
        else:
            print(f"  {t}: insufficient data ({len(df)} rows), skipping")
    except Exception as e:
        print(f"  {t}: download failed ({e})")

# ── Feature Engineering ────────────────────────────────────────────────────────
def compute_features(price_df, vix_df=None):
    """Compute RV, IV proxy, and regime features."""
    df = price_df.copy()
    close = df["Close"]
    log_ret = np.log(close / close.shift(1))

    df["rv_5"] = log_ret.rolling(5).std() * np.sqrt(252)
    df["rv_10"] = log_ret.rolling(10).std() * np.sqrt(252)
    df["rv_20"] = log_ret.rolling(20).std() * np.sqrt(252)

    if vix_df is not None:
        vix_close = vix_df["Close"]
        df["iv_proxy"] = vix_close.reindex(df.index, method="ffill") / 100.0
    else:
        df["iv_proxy"] = df["rv_20"] * 1.15

    df["iv_rv_spread"] = df["iv_proxy"] - df["rv_20"]
    df["iv_rv_ratio"] = df["iv_proxy"] / df["rv_20"].replace(0, np.nan)

    df["sma_200"] = close.rolling(200).mean()
    df["bull"] = (close > df["sma_200"]).astype(int)

    df["vol_compress"] = (df["rv_5"] - df["rv_10"]) / df["rv_10"].replace(0, np.nan)

    df["spread_pctile"] = df["iv_rv_spread"].rolling(60).apply(
        lambda x: stats.percentileofscore(x.dropna(), x.iloc[-1]) / 100.0
        if len(x.dropna()) > 10 else np.nan
    )

    # RSI 14
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss_s = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss_s.replace(0, np.nan)
    df["rsi"] = 100 - (100 / (1 + rs))

    df["daily_ret"] = log_ret
    df["close"] = close
    return df


print("Computing features...")
features = {}
vix_df = data.get("^VIX")
spy_df = data.get("SPY")

if spy_df is not None:
    features["SPY"] = compute_features(spy_df, vix_df)
if "QQQ" in data:
    features["QQQ"] = compute_features(data["QQQ"], vix_df)
for t in GROWTH_TICKERS:
    if t in data:
        features[t] = compute_features(data[t])

spy_regime = features["SPY"]["bull"] if "SPY" in features else pd.Series(1, index=pd.date_range(OOT_START, OOT_END, freq="B"))


# ── P&L Models (corrected) ────────────────────────────────────────────────────
# All P&L models take premium_pct (option premium as fraction of stock price)
# and return P&L as a fraction of the capital_at_risk (the dollar amount risked).

def iron_condor_pnl_model(price_entry, price_exit, rv_during, premium_pct, wing_width_pct, dte_days):
    """
    Iron condor: collect premium_pct, risk wing_width_pct - premium_pct.
    Win if price stays within wings. Model using actual price move.
    Returns P&L as fraction of max_risk.
    """
    actual_move_pct = abs(price_exit - price_entry) / price_entry
    half_wing = wing_width_pct / 2  # distance from ATM to wing

    if actual_move_pct < half_wing * 0.7:
        # Comfortably within wings, keep ~85% of premium (theta)
        return premium_pct * 0.85
    elif actual_move_pct < half_wing:
        # Close to wing, keep ~40%
        return premium_pct * 0.40
    elif actual_move_pct < half_wing + wing_width_pct * 0.3:
        # Partially breached
        return -premium_pct * 0.5
    else:
        # Fully breached, lose width - premium
        return -(wing_width_pct - premium_pct)


def straddle_pnl_model(price_entry, price_exit, premium_pct, dte_days):
    """
    Straddle buyer: pay premium_pct, profit if move > breakeven.

    Key reality check: ATM straddle premium IS the market's estimate of the expected
    move. On average, the stock moves LESS than the straddle costs (variance risk premium).
    The straddle buyer needs the stock to move MORE than what the market already prices in.

    For a 30-DTE ATM straddle:
    - Premium ≈ stock_price × IV × sqrt(DTE/365) × 0.8 (for both legs combined)
    - For growth stocks with 50% IV: premium ≈ 0.50 × sqrt(30/365) × 0.8 ≈ 11.5% of stock
    - For SPY with 15% IV: premium ≈ 0.15 × sqrt(30/365) × 0.8 ≈ 3.4% of stock

    The breakeven is approximately the premium paid, and on average stocks move LESS
    than their implied vol (that's the variance risk premium = why vol selling works).

    Returns P&L as fraction of stock price.
    """
    actual_move_pct = abs(price_exit - price_entry) / price_entry

    # Theta decay: critical for straddles held to near-expiry.
    # Theta accelerates: sqrt model. After dte_days of 30, ~100% of extrinsic is gone.
    theta_fraction = min(1.0, np.sqrt(dte_days / 30.0))
    # Straddle loses ~60-80% of extrinsic to theta if held to expiry
    theta_loss = premium_pct * theta_fraction * 0.65

    # At exit, remaining value = intrinsic + remaining extrinsic
    remaining_extrinsic = premium_pct * (1 - theta_fraction) * 0.3

    # Intrinsic at exit: only the ITM leg has value
    intrinsic = actual_move_pct

    # Total value at exit
    exit_value = intrinsic + remaining_extrinsic

    # P&L = exit value - entry cost (premium)
    # Also subtract BA haircut on exit (entry haircut handled by caller)
    net = exit_value - premium_pct - premium_pct * BA_HAIRCUT  # exit BA haircut

    return net


def put_sell_pnl_model(price_entry, price_exit, premium_pct, strike_offset_pct=0.03):
    """
    Short put: collect premium_pct, risk strike_offset_pct below current.
    Win if stock stays above strike. Returns P&L as fraction of stock price.
    """
    price_change_pct = (price_exit - price_entry) / price_entry
    strike_pct = -strike_offset_pct  # e.g., 3% OTM put

    if price_change_pct > strike_pct:
        # Put expires worthless or near worthless
        if price_change_pct > 0:
            return premium_pct * 0.90  # keep almost all
        else:
            # Stock dropped but above strike
            intrusion = abs(price_change_pct) / abs(strike_pct)
            return premium_pct * (0.90 - intrusion * 0.5)
    else:
        # Below strike, losing
        below_strike = abs(price_change_pct) - abs(strike_pct)
        loss = below_strike + premium_pct * 0.1  # lose intrinsic minus premium
        return premium_pct - loss  # can be very negative


def apply_costs(gross_dollar_pnl, n_legs=1):
    """Apply commissions. BA haircut is already in premium pricing."""
    commission = COMMISSION_PER_CONTRACT * 2 * n_legs  # entry + exit
    return gross_dollar_pnl - commission


def cap_position(dollar_amount):
    """Cap position size to MAX_TRADE_SIZE."""
    return min(dollar_amount, MAX_TRADE_SIZE)


def option_dollar_pnl(pnl_pct, ref_pct, capital, n_legs=1):
    """
    Convert option P&L from percentage-of-stock-price to dollars.
    pnl_pct: P&L as fraction of stock price (from model)
    ref_pct: the premium or risk fraction that corresponds to our capital outlay
    capital: actual dollars at risk / deployed
    n_legs: number of option legs for commission calc

    The idea: if we risked 'capital' dollars which corresponds to 'ref_pct' of notional,
    then pnl_pct of notional = pnl_pct/ref_pct * capital in dollar terms.
    But we also cap max profit at 2x capital and max loss at -capital.
    """
    if ref_pct == 0 or np.isnan(ref_pct):
        return 0.0
    raw = (pnl_pct / ref_pct) * capital
    # BA haircut
    raw *= (1 - BA_HAIRCUT)
    # Commission
    raw = apply_costs(raw, n_legs)
    # Cap: can't make more than 2x capital, can't lose more than capital
    raw = max(raw, -capital)
    raw = min(raw, capital * 2.0)
    return raw


# ── Strategy Implementations ──────────────────────────────────────────────────
class Trade:
    def __init__(self, date, ticker, direction, capital_risked, dte, strategy):
        self.entry_date = date
        self.ticker = ticker
        self.direction = direction
        self.capital_risked = capital_risked
        self.dte = dte
        self.strategy = strategy
        self.exit_date = None
        self.pnl = 0.0


def strategy_A_spy_vol_selling(feat_spy):
    """
    Variant A: SPY Vol Selling
    VIX/RV20 > 1.3 → sell iron condors. VIX/RV20 < 0.9 → buy straddles.
    """
    trades = []
    df = feat_spy.loc[OOT_START:OOT_END].dropna(subset=["iv_rv_ratio", "rv_20", "close"])

    i = 0
    dates = df.index.tolist()
    while i < len(dates) - 22:
        d = dates[i]
        row = df.loc[d]
        ratio = row["iv_rv_ratio"]
        price = row["close"]

        exit_idx = min(i + 22, len(dates) - 1)
        exit_d = dates[exit_idx]
        exit_price = df.loc[exit_d, "close"]

        if ratio > 1.3:
            premium_pct = 0.01
            wing_pct = 0.05
            pnl_pct = iron_condor_pnl_model(price, exit_price, row["rv_20"],
                                              premium_pct, wing_pct, 22)
            capital = cap_position(wing_pct * price * 100)
            dollar_pnl = option_dollar_pnl(pnl_pct, wing_pct, capital, n_legs=4)

            t = Trade(d, "SPY", "sell", capital, 30, "A_IC_sell")
            t.exit_date = exit_d
            t.pnl = dollar_pnl
            trades.append(t)
            i += 22

        elif ratio < 0.9:
            # Straddle cost from IV: IV * sqrt(DTE/365) * 0.8
            iv = row.get("iv_proxy", 0.15)
            premium_pct = iv * np.sqrt(22 / 365) * 0.80
            premium_pct = max(premium_pct, 0.02)
            pnl_pct = straddle_pnl_model(price, exit_price, premium_pct, 22)
            capital = cap_position(premium_pct * price * 100)
            dollar_pnl = option_dollar_pnl(pnl_pct, premium_pct, capital, n_legs=2)

            t = Trade(d, "SPY", "buy", capital, 30, "A_straddle_buy")
            t.exit_date = exit_d
            t.pnl = dollar_pnl
            trades.append(t)
            i += 22
        else:
            i += 1

    return trades


def strategy_B_growth_vol_selling(features_dict):
    """
    Variant B: Growth Stock Vol Selling
    Sell puts when stock is oversold (RSI < 40) AND vol is elevated.
    For growth stocks with IV proxy = 1.15*RV, ratio > 1.1 is "rich" (above median).
    """
    trades = []

    for ticker in GROWTH_TICKERS:
        if ticker not in features_dict:
            continue
        df = features_dict[ticker].loc[OOT_START:OOT_END].copy()
        df = df.dropna(subset=["iv_rv_ratio", "rv_20", "close", "rsi"])

        i = 0
        dates = df.index.tolist()
        while i < len(dates) - 22:
            d = dates[i]
            row = df.loc[d]

            # Sell put when: RSI < 40 (oversold) AND spread_pctile > 0.6 (vol relatively rich)
            spread_pctile = row.get("spread_pctile", 0.5)
            if pd.isna(spread_pctile):
                spread_pctile = 0.5

            if row["rsi"] < 40 and spread_pctile > 0.55:
                price = row["close"]
                premium_pct = 0.025  # 2.5% OTM put premium
                strike_offset = 0.05  # 5% OTM

                exit_idx = min(i + 22, len(dates) - 1)
                exit_d = dates[exit_idx]
                exit_price = df.loc[exit_d, "close"]

                pnl_pct = put_sell_pnl_model(price, exit_price, premium_pct, strike_offset)
                capital = cap_position(strike_offset * price * 100)
                dollar_pnl = option_dollar_pnl(pnl_pct, strike_offset, capital, n_legs=1)

                t = Trade(d, ticker, "sell", capital, 30, "B_put_sell")
                t.exit_date = exit_d
                t.pnl = dollar_pnl
                trades.append(t)
                i += 22
            else:
                i += 1

    return trades


def strategy_C_vol_compression(features_dict):
    """
    Variant C: Vol Compression Play
    5d RV < 10d RV by >25% → buy straddles expecting breakout.
    """
    trades = []

    for ticker in GROWTH_TICKERS:
        if ticker not in features_dict:
            continue
        df = features_dict[ticker].loc[OOT_START:OOT_END].copy()
        df = df.dropna(subset=["vol_compress", "rv_20", "close"])

        i = 0
        dates = df.index.tolist()
        while i < len(dates) - 15:
            d = dates[i]
            row = df.loc[d]

            if row["vol_compress"] < -0.25:
                price = row["close"]
                # Straddle cost derived from IV proxy: IV * sqrt(DTE/365) * 0.8 (both legs)
                # Even though vol is "compressed", options are priced off IV not RV
                iv = row["iv_proxy"]
                premium_pct = iv * np.sqrt(15 / 365) * 0.80
                premium_pct = max(premium_pct, 0.02)  # floor at 2%

                exit_idx = min(i + 15, len(dates) - 1)
                exit_d = dates[exit_idx]
                exit_price = df.loc[exit_d, "close"]

                pnl_pct = straddle_pnl_model(price, exit_price, premium_pct, 15)
                capital = cap_position(premium_pct * price * 100)
                dollar_pnl = option_dollar_pnl(pnl_pct, premium_pct, capital, n_legs=2)

                t = Trade(d, ticker, "buy", capital, 15, "C_vol_compress")
                t.exit_date = exit_d
                t.pnl = dollar_pnl
                trades.append(t)
                i += 15
            else:
                i += 1

    return trades


def strategy_D_post_vix_spike(feat_spy, vix_df):
    """
    Variant D: Post-Vol-Spike Selling
    VIX > 22 and declining (3d downtrend) → sell SPY puts.
    """
    trades = []
    if vix_df is None:
        return trades

    df = feat_spy.loc[OOT_START:OOT_END].copy()
    vix_close = vix_df["Close"]
    df["vix"] = vix_close.reindex(df.index, method="ffill")
    df["vix_3d_chg"] = df["vix"].diff(3)
    df = df.dropna(subset=["vix", "vix_3d_chg", "close", "rv_20"])

    i = 0
    dates = df.index.tolist()
    while i < len(dates) - 22:
        d = dates[i]
        row = df.loc[d]

        if row["vix"] > 22 and row["vix_3d_chg"] < -1.0:
            price = row["close"]
            # Elevated vol = rich premium (~2.5%)
            premium_pct = 0.025
            strike_offset = 0.04

            exit_idx = min(i + 22, len(dates) - 1)
            exit_d = dates[exit_idx]
            exit_price = df.loc[exit_d, "close"]

            pnl_pct = put_sell_pnl_model(price, exit_price, premium_pct, strike_offset)
            capital = cap_position(strike_offset * price * 100)
            dollar_pnl = option_dollar_pnl(pnl_pct, strike_offset, capital, n_legs=1)

            t = Trade(d, "SPY", "sell", capital, 30, "D_post_spike")
            t.exit_date = exit_d
            t.pnl = dollar_pnl
            trades.append(t)
            i += 22
        else:
            i += 1

    return trades


def strategy_E_earnings_vol(features_dict):
    """
    Variant E: Earnings Vol Spread
    Use actual pre/post-earnings vol ratio from realized moves.
    If 10d pre-earnings RV > 1.5x 60d RV → vol is elevated, sell strangles.
    If 10d pre-earnings RV < 1.1x 60d RV → vol is compressed, buy straddles.
    """
    trades = []
    earnings_months = [1, 4, 7, 10]

    for ticker in GROWTH_TICKERS:
        if ticker not in features_dict:
            continue
        df = features_dict[ticker].loc[OOT_START:OOT_END].copy()
        df["rv_60"] = df["daily_ret"].rolling(60).std() * np.sqrt(252)
        df = df.dropna(subset=["rv_20", "rv_60", "close"])

        for year in range(2022, 2027):
            for month in earnings_months:
                target = pd.Timestamp(year, month, 20)
                if target > pd.Timestamp(OOT_END):
                    continue

                mask = (df.index >= target - pd.Timedelta(days=10)) & \
                       (df.index <= target + pd.Timedelta(days=5))
                candidates = df.loc[mask]
                if len(candidates) < 5:
                    continue

                # "pre-earnings" = 5 days before target
                pre_d = candidates.index[0]
                post_idx = min(5, len(candidates) - 1)
                post_d = candidates.index[post_idx]

                pre_row = df.loc[pre_d]
                post_row = df.loc[post_d]
                price = pre_row["close"]

                if pre_row["rv_60"] == 0 or np.isnan(pre_row["rv_60"]):
                    continue

                vol_ratio = pre_row["rv_20"] / pre_row["rv_60"]

                if vol_ratio > 1.4:
                    # Rich pre-earnings vol → sell strangle
                    premium_pct = 0.04  # 4% for earnings strangle
                    actual_move = abs(post_row["close"] - price) / price

                    # Strangle seller wins if move < premium
                    if actual_move < premium_pct * 0.7:
                        pnl_frac = premium_pct * 0.75
                    elif actual_move < premium_pct:
                        pnl_frac = premium_pct * 0.25
                    elif actual_move < premium_pct * 2:
                        pnl_frac = -(actual_move - premium_pct) * 0.8
                    else:
                        pnl_frac = -(actual_move - premium_pct)

                    capital = cap_position(premium_pct * price * 100)
                    dollar_pnl = option_dollar_pnl(pnl_frac, premium_pct, capital, n_legs=2)

                    t = Trade(pre_d, ticker, "sell", capital, 5, "E_earnings_sell")
                    t.exit_date = post_d
                    t.pnl = dollar_pnl
                    trades.append(t)

                elif vol_ratio < 1.1:
                    # Cheap pre-earnings vol → buy straddle
                    premium_pct = 0.03
                    actual_move = abs(post_row["close"] - price) / price

                    pnl_frac = actual_move - premium_pct  # intrinsic - cost

                    capital = cap_position(premium_pct * price * 100)
                    dollar_pnl = option_dollar_pnl(pnl_frac, premium_pct, capital, n_legs=2)

                    t = Trade(pre_d, ticker, "buy", capital, 5, "E_earnings_buy")
                    t.exit_date = post_d
                    t.pnl = dollar_pnl
                    trades.append(t)

    return trades


def strategy_F_regime_adjusted(feat_spy, features_dict, spy_regime):
    """
    Variant F: Regime-Adjusted
    Bull (SPY > 200-SMA): sell premium when spread is rich
    Bear (SPY < 200-SMA): buy premium when spread is cheap
    """
    trades = []

    # SPY trades
    df = feat_spy.loc[OOT_START:OOT_END].copy()
    df = df.dropna(subset=["iv_rv_ratio", "rv_20", "close", "bull", "spread_pctile"])

    i = 0
    dates = df.index.tolist()
    while i < len(dates) - 22:
        d = dates[i]
        row = df.loc[d]
        price = row["close"]

        exit_idx = min(i + 22, len(dates) - 1)
        exit_d = dates[exit_idx]
        exit_price = df.loc[exit_d, "close"]

        if row["bull"] == 1 and row["spread_pctile"] > 0.75:
            # Bull + rich vol → sell IC
            premium_pct = 0.008
            wing_pct = 0.04
            pnl_pct = iron_condor_pnl_model(price, exit_price, row["rv_20"],
                                              premium_pct, wing_pct, 22)
            capital = cap_position(wing_pct * price * 100)
            dollar_pnl = option_dollar_pnl(pnl_pct, wing_pct, capital, n_legs=4)

            t = Trade(d, "SPY", "sell", capital, 30, "F_bull_sell")
            t.exit_date = exit_d
            t.pnl = dollar_pnl
            trades.append(t)
            i += 22

        elif row["bull"] == 0 and row["spread_pctile"] < 0.30:
            # Bear + cheap vol → buy straddle
            iv = row.get("iv_proxy", 0.20)
            premium_pct = iv * np.sqrt(22 / 365) * 0.80
            premium_pct = max(premium_pct, 0.02)
            pnl_pct = straddle_pnl_model(price, exit_price, premium_pct, 22)
            capital = cap_position(premium_pct * price * 100)
            dollar_pnl = option_dollar_pnl(pnl_pct, premium_pct, capital, n_legs=2)

            t = Trade(d, "SPY", "buy", capital, 30, "F_bear_buy")
            t.exit_date = exit_d
            t.pnl = dollar_pnl
            trades.append(t)
            i += 22
        else:
            i += 1

    # Growth stock trades (regime-gated)
    for ticker in GROWTH_TICKERS:
        if ticker not in features_dict:
            continue
        gdf = features_dict[ticker].loc[OOT_START:OOT_END].copy()
        gdf = gdf.dropna(subset=["iv_rv_ratio", "rv_20", "close"])
        gdf["spy_bull"] = spy_regime.reindex(gdf.index, method="ffill")
        gdf = gdf.dropna(subset=["spy_bull"])

        i = 0
        gdates = gdf.index.tolist()
        while i < len(gdates) - 22:
            d = gdates[i]
            row = gdf.loc[d]
            price = row["close"]

            exit_idx = min(i + 22, len(gdates) - 1)
            exit_d = gdates[exit_idx]
            exit_price = gdf.loc[exit_d, "close"]

            sp = row.get("spread_pctile", 0.5)
            if pd.isna(sp):
                sp = 0.5

            if row["spy_bull"] == 1 and sp > 0.65:
                premium_pct = 0.02
                strike_offset = 0.05
                pnl_pct = put_sell_pnl_model(price, exit_price, premium_pct, strike_offset)
                capital = cap_position(strike_offset * price * 100)
                dollar_pnl = option_dollar_pnl(pnl_pct, strike_offset, capital, n_legs=1)

                t = Trade(d, ticker, "sell", capital, 30, "F_growth_bull_sell")
                t.exit_date = exit_d
                t.pnl = dollar_pnl
                trades.append(t)
                i += 22
            else:
                i += 1

    return trades


# ── Evaluation ─────────────────────────────────────────────────────────────────
def evaluate_strategy(name, trades, spy_regime):
    """Run 5-gate validation on a list of trades."""
    if len(trades) == 0:
        print(f"  No trades generated")
        return {
            "n_trades": 0, "gates_passed": 0, "gates_total": 5,
            "verdict": "FAIL_NO_TRADES"
        }

    pnls = np.array([t.pnl for t in trades])

    # Cap individual trade P&L to realistic bounds (can't lose more than risked)
    for idx, t in enumerate(trades):
        pnls[idx] = max(pnls[idx], -t.capital_risked)

    cumulative = np.cumsum(pnls)
    equity = ACCOUNT + cumulative

    n_trades = len(trades)
    total_pnl = pnls.sum()
    win_rate = (pnls > 0).sum() / n_trades
    avg_win = pnls[pnls > 0].mean() if (pnls > 0).any() else 0
    avg_loss = pnls[pnls < 0].mean() if (pnls < 0).any() else 0
    pf_num = pnls[pnls > 0].sum()
    pf_den = abs(pnls[pnls < 0].sum()) if (pnls < 0).any() else 1
    profit_factor = pf_num / pf_den if pf_den > 0 else (99.9 if pf_num > 0 else 0)

    # Max drawdown (on equity curve)
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / peak
    max_dd = dd.min()

    # Annualized Sharpe
    years = (pd.Timestamp(OOT_END) - pd.Timestamp(OOT_START)).days / 365.25
    tpy = n_trades / years if years > 0 else n_trades

    sharpe = (pnls.mean() / pnls.std() * np.sqrt(tpy)) if pnls.std() > 0 else 0

    downside = pnls[pnls < 0]
    sortino = (pnls.mean() / downside.std() * np.sqrt(tpy)) if len(downside) > 1 and downside.std() > 0 else (99.9 if pnls.mean() > 0 else 0)

    # ── GATE 1: Sharpe > 0.5
    g1 = sharpe > 0.5

    # ── GATE 2: Permutation test p < 0.05
    observed = pnls.mean()
    n_better = sum(1 for _ in range(N_PERMUTATIONS)
                   if (pnls * np.random.choice([-1, 1], size=len(pnls))).mean() >= observed)
    perm_p = n_better / N_PERMUTATIONS
    g2 = perm_p < 0.05

    # ── GATE 3: Regime gap < 0.5
    bull_pnls, bear_pnls = [], []
    for t in trades:
        d = t.entry_date
        try:
            rv = spy_regime.reindex([d], method="ffill")
            if len(rv) > 0 and rv.iloc[0] == 1:
                bull_pnls.append(t.pnl)
            else:
                bear_pnls.append(t.pnl)
        except:
            bull_pnls.append(t.pnl)

    bull_arr = np.array(bull_pnls) if bull_pnls else np.array([0.0])
    bear_arr = np.array(bear_pnls) if bear_pnls else np.array([0.0])

    bull_sr = (bull_arr.mean() / bull_arr.std() * np.sqrt(tpy)) if len(bull_arr) > 2 and bull_arr.std() > 0 else 0
    bear_sr = (bear_arr.mean() / bear_arr.std() * np.sqrt(tpy)) if len(bear_arr) > 2 and bear_arr.std() > 0 else 0

    max_sr = max(abs(bull_sr), abs(bear_sr), 0.001)
    regime_gap = abs(bull_sr - bear_sr) / max_sr
    g3 = regime_gap < 0.5

    # ── GATE 4: MaxDD > -50%
    g4 = max_dd > -0.50

    # ── GATE 5: >= 20 trades
    g5 = n_trades >= 20

    gates_passed = sum([g1, g2, g3, g4, g5])
    verdict = "PASS" if gates_passed == 5 else "FAIL"

    # Ticker breakdown
    ticker_bd = {}
    for t in trades:
        tk = t.ticker
        if tk not in ticker_bd:
            ticker_bd[tk] = {"n": 0, "pnl": 0.0, "wins": 0}
        ticker_bd[tk]["n"] += 1
        ticker_bd[tk]["pnl"] += t.pnl
        ticker_bd[tk]["wins"] += 1 if t.pnl > 0 else 0
    for tk in ticker_bd:
        ticker_bd[tk]["wr"] = round(ticker_bd[tk]["wins"] / ticker_bd[tk]["n"], 3)
        ticker_bd[tk]["pnl"] = round(ticker_bd[tk]["pnl"], 2)

    # Print
    print(f"  Trades: {n_trades} | WR: {win_rate:.1%} | PF: {profit_factor:.2f}")
    print(f"  Total P&L: ${total_pnl:.2f} ({total_pnl/ACCOUNT*100:.1f}%) | Final: ${equity[-1]:.2f}")
    print(f"  Sharpe: {sharpe:.3f} | Sortino: {min(sortino,99.9):.3f} | MaxDD: {max_dd*100:.1f}%")
    print(f"  Regime: Bull SR={bull_sr:.2f}, Bear SR={bear_sr:.2f}, Gap={regime_gap:.3f}")
    print(f"  Perm p={perm_p:.4f}")
    gate_strs = ['PASS' if g else 'FAIL' for g in [g1, g2, g3, g4, g5]]
    print(f"  Gates: {'|'.join(gate_strs)} -> {verdict} ({gates_passed}/5)")
    parts = [f"{k}({v['n']}t ${v['pnl']})" for k, v in ticker_bd.items()]
    if parts:
        print(f"  Tickers: {', '.join(parts)}")

    result = {
        "n_trades": n_trades,
        "total_pnl": round(total_pnl, 2),
        "total_return_pct": round(total_pnl / ACCOUNT * 100, 2),
        "win_rate": round(win_rate, 4),
        "avg_win": round(avg_win, 2),
        "avg_loss": round(avg_loss, 2),
        "profit_factor": round(min(profit_factor, 99.9), 2),
        "sharpe": round(sharpe, 3),
        "sortino": round(min(sortino, 99.9), 3),
        "max_dd_pct": round(max_dd * 100, 2),
        "final_equity": round(equity[-1], 2),
        "years": round(years, 2),
        "trades_per_year": round(tpy, 1),
        "gates": {
            "G1_sharpe_gt_0.5": g1,
            "G2_perm_p_lt_0.05": g2,
            "G3_regime_gap_lt_0.5": g3,
            "G4_maxdd_gt_neg50": g4,
            "G5_min_20_trades": g5,
        },
        "gates_passed": gates_passed,
        "gates_total": 5,
        "verdict": verdict,
        "perm_p_value": round(perm_p, 4),
        "regime_sharpe_bull": round(bull_sr, 3),
        "regime_sharpe_bear": round(bear_sr, 3),
        "regime_gap": round(regime_gap, 3),
        "ticker_breakdown": ticker_bd,
        "sample_trades": [
            {
                "date": str(t.entry_date.date()) if hasattr(t.entry_date, 'date') else str(t.entry_date),
                "exit": str(t.exit_date.date()) if hasattr(t.exit_date, 'date') else str(t.exit_date),
                "ticker": t.ticker,
                "dir": t.direction,
                "pnl": round(t.pnl, 2),
                "strategy": t.strategy,
            }
            for t in trades[:10]
        ],
    }
    return result


# ── Run All Strategies ─────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("RUNNING STRATEGIES")
print("=" * 70)

all_results = {}

strat_runners = {
    "A_SPY_Vol_Selling": lambda: strategy_A_spy_vol_selling(features["SPY"]) if "SPY" in features else [],
    "B_Growth_Vol_Selling": lambda: strategy_B_growth_vol_selling(features),
    "C_Vol_Compression": lambda: strategy_C_vol_compression(features),
    "D_Post_VIX_Spike": lambda: strategy_D_post_vix_spike(features["SPY"], vix_df) if "SPY" in features else [],
    "E_Earnings_Vol": lambda: strategy_E_earnings_vol(features),
    "F_Regime_Adjusted": lambda: strategy_F_regime_adjusted(features["SPY"], features, spy_regime) if "SPY" in features else [],
}

for name, runner in strat_runners.items():
    print(f"\n--- {name} ---")
    trades = runner()
    all_results[name] = evaluate_strategy(name, trades, spy_regime)


# ── Combined Portfolio ─────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("COMBINED PORTFOLIO")
print("=" * 70)

passing = [n for n, r in all_results.items() if r.get("verdict") == "PASS"]
four_plus = [n for n, r in all_results.items() if r.get("gates_passed", 0) >= 4]
three_plus = [n for n, r in all_results.items() if r.get("gates_passed", 0) >= 3]

print(f"5/5 gates: {passing if passing else 'NONE'}")
print(f"4+ gates:  {four_plus if four_plus else 'NONE'}")
print(f"3+ gates:  {three_plus if three_plus else 'NONE'}")

best = max(all_results.items(), key=lambda x: x[1].get("sharpe", -99))
print(f"Best Sharpe: {best[0]} ({best[1].get('sharpe', 'N/A')})")

# ── Save Results ───────────────────────────────────────────────────────────────
output = {
    "metadata": {
        "run_date": str(dt.datetime.now()),
        "account_size": ACCOUNT,
        "oot_period": f"{OOT_START} to {OOT_END}",
        "n_permutations": N_PERMUTATIONS,
        "cost_assumptions": {
            "commission_per_contract": COMMISSION_PER_CONTRACT,
            "ba_haircut_pct": BA_HAIRCUT * 100,
            "max_trade_size": MAX_TRADE_SIZE,
        },
    },
    "strategies": all_results,
    "summary": {
        "all_5_gates_pass": passing,
        "4_plus_gates": four_plus,
        "3_plus_gates": three_plus,
        "best_sharpe_strategy": best[0],
        "best_sharpe_value": best[1].get("sharpe", None),
    },
}

output_path = Path("/home/jupiter/Lvl3Quant/data/iv_rv_spread_results.json")
with open(output_path, "w") as f:
    json.dump(output, f, indent=2, default=str)

print(f"\nResults saved to {output_path}")

print("\n" + "=" * 70)
print("FINAL VERDICT TABLE")
print("=" * 70)
header = f"{'Strategy':<25} {'Trades':>6} {'Sharpe':>7} {'WR':>6} {'MaxDD':>7} {'P&L':>9} {'Gates':>5} {'Verdict':>7}"
print(header)
print("-" * len(header))
for name, r in all_results.items():
    nt = r.get('n_trades', 0)
    sh = r.get('sharpe', 0)
    wr = r.get('win_rate', 0)
    mdd = r.get('max_dd_pct', 0)
    pnl = r.get('total_pnl', 0)
    gp = r.get('gates_passed', 0)
    v = r.get('verdict', 'FAIL')
    print(f"{name:<25} {nt:>6} {sh:>7.3f} {wr:>5.1%} {mdd:>6.1f}% ${pnl:>8.2f} {gp:>2}/5 {'PASS' if v=='PASS' else 'FAIL':>7}")
print("=" * 70)
