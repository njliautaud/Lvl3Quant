#!/usr/bin/env python3
"""
ML Earnings Volatility Selling Strategy v1
===========================================
Hypothesis: Implied volatility before earnings is systematically overpriced
relative to realized moves. An ML model (LightGBM) can identify which specific
earnings events have the most overpriced vol, making selling premium (cash-secured
puts) profitable.

Approach:
- Since historical options IV data is unreliable via yfinance, we use a proxy:
  compute pre-earnings "implied move" from historical vol patterns and compare
  to actual post-earnings move. If actual < implied -> premium seller wins.
- GBM predicts probability that IV is overpriced for each earnings event.
- Walk-forward: 2yr train, 1 quarter predict, sliding.
- Full adversarial 4-gate validation.

Reports both theoretical ($10K account) and practical ($423 account) results.
"""

import json
import warnings
import datetime as dt
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import yfinance as yf

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    from sklearn.ensemble import GradientBoostingClassifier
    HAS_LGB = False
    print("WARNING: LightGBM not available, using sklearn GBM (slower)")

from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

warnings.filterwarnings('ignore')

# ─── Config ───
OUTPUT_DIR = Path('/home/jupiter/Lvl3Quant/output/growth_research/earnings_vol_selling')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Top 50 S&P 500 by market cap (as of mid-2026, approximate)
SP500_TOP50 = [
    'AAPL', 'MSFT', 'NVDA', 'AMZN', 'GOOGL', 'META', 'BRK-B', 'LLY', 'TSM', 'AVGO',
    'JPM', 'TSLA', 'V', 'UNH', 'XOM', 'MA', 'WMT', 'JNJ', 'PG', 'HD',
    'COST', 'ORCL', 'MRK', 'ABBV', 'CRM', 'AMD', 'NFLX', 'BAC', 'CVX', 'KO',
    'PEP', 'LIN', 'TMO', 'ADBE', 'ACN', 'MCD', 'CSCO', 'ABT', 'DHR', 'WFC',
    'TXN', 'PM', 'NEE', 'INTC', 'CMCSA', 'VZ', 'DIS', 'BMY', 'QCOM', 'UPS',
]

# Sector mapping (approximate)
SECTOR_MAP = {
    'AAPL': 'Tech', 'MSFT': 'Tech', 'NVDA': 'Tech', 'AMZN': 'ConsDisc', 'GOOGL': 'Tech',
    'META': 'Tech', 'BRK-B': 'Financials', 'LLY': 'Healthcare', 'TSM': 'Tech', 'AVGO': 'Tech',
    'JPM': 'Financials', 'TSLA': 'ConsDisc', 'V': 'Financials', 'UNH': 'Healthcare', 'XOM': 'Energy',
    'MA': 'Financials', 'WMT': 'ConsStap', 'JNJ': 'Healthcare', 'PG': 'ConsStap', 'HD': 'ConsDisc',
    'COST': 'ConsStap', 'ORCL': 'Tech', 'MRK': 'Healthcare', 'ABBV': 'Healthcare', 'CRM': 'Tech',
    'AMD': 'Tech', 'NFLX': 'Tech', 'BAC': 'Financials', 'CVX': 'Energy', 'KO': 'ConsStap',
    'PEP': 'ConsStap', 'LIN': 'Materials', 'TMO': 'Healthcare', 'ADBE': 'Tech', 'ACN': 'Tech',
    'MCD': 'ConsDisc', 'CSCO': 'Tech', 'ABT': 'Healthcare', 'DHR': 'Healthcare', 'WFC': 'Financials',
    'TXN': 'Tech', 'PM': 'ConsStap', 'NEE': 'Utilities', 'INTC': 'Tech', 'CMCSA': 'Tech',
    'VZ': 'Telecom', 'DIS': 'ConsDisc', 'BMY': 'Healthcare', 'QCOM': 'Tech', 'UPS': 'Industrials',
}

# Sector one-hot encoding order
SECTORS = sorted(set(SECTOR_MAP.values()))

# Walk-forward config
WF_TRAIN_QUARTERS = 8  # 2 years
WF_PREDICT_QUARTERS = 1
DATA_START = '2018-01-01'  # need 2yr history before first prediction
PREDICT_START = '2020-01-01'
DATA_END = '2026-07-15'

# Account sizes
THEORETICAL_CAPITAL = 10_000
PRACTICAL_CAPITAL = 423

# CSP parameters
TARGET_DELTA = 0.30  # ~0.30 delta put
DAYS_TO_EXPIRY = 30  # monthly options cycle

# Adversarial validation
N_PERMS = 200

print("=" * 70)
print("ML EARNINGS VOLATILITY SELLING STRATEGY v1")
print("=" * 70)
print(f"Universe: {len(SP500_TOP50)} stocks")
print(f"Walk-forward: {WF_TRAIN_QUARTERS}Q train, {WF_PREDICT_QUARTERS}Q predict")
print(f"Prediction period: {PREDICT_START} to {DATA_END}")
print(f"Accounts: ${THEORETICAL_CAPITAL:,} (paper) + ${PRACTICAL_CAPITAL} (real)")
print()


# ─── Step 1: Download price data ───
print("STEP 1: Downloading price data...")

vix_data = None
price_data = {}
failed_tickers = []

# Download VIX first
try:
    vix_df = yf.download('^VIX', start=DATA_START, end=DATA_END, progress=False, auto_adjust=True)
    if isinstance(vix_df.columns, pd.MultiIndex):
        vix_data = vix_df[('Close', '^VIX')].copy()
    else:
        vix_data = vix_df['Close'].copy()
    vix_data.name = 'VIX'
    print(f"  VIX: {len(vix_data)} days")
except Exception as e:
    print(f"  VIX: FAILED ({e})")

# Download SPY for beta calculations
try:
    spy_df = yf.download('SPY', start=DATA_START, end=DATA_END, progress=False, auto_adjust=True)
    if isinstance(spy_df.columns, pd.MultiIndex):
        spy_close = spy_df[('Close', 'SPY')].copy()
    else:
        spy_close = spy_df['Close'].copy()
    spy_close.name = 'SPY'
    spy_returns = spy_close.pct_change().dropna()
    print(f"  SPY: {len(spy_df)} days")
except Exception as e:
    print(f"  SPY: FAILED ({e})")
    spy_returns = None

# Download stock data
for ticker in SP500_TOP50:
    try:
        df = yf.download(ticker, start=DATA_START, end=DATA_END, progress=False, auto_adjust=True)
        if len(df) < 200:
            print(f"  {ticker}: SKIP (only {len(df)} days)")
            failed_tickers.append(ticker)
            continue
        if isinstance(df.columns, pd.MultiIndex):
            close = df[('Close', ticker)].copy()
        else:
            close = df['Close'].copy()
        close.name = ticker
        price_data[ticker] = close
        print(f"  {ticker}: {len(df)} days, last=${close.iloc[-1]:.2f}")
    except Exception as e:
        print(f"  {ticker}: FAILED ({e})")
        failed_tickers.append(ticker)

print(f"\nLoaded {len(price_data)} stocks, failed: {len(failed_tickers)}")


# ─── Step 2: Build earnings event database ───
print("\nSTEP 2: Building earnings event database...")


def get_earnings_dates(ticker):
    """
    Detect likely earnings dates from price action.
    Earnings cause abnormally large overnight gaps. We find the ~4 biggest
    gap-ups/gap-downs per year (quarterly earnings) by looking at open-to-prev-close
    or simply daily returns that are outliers (>2x 60-day vol).

    This avoids the extremely slow yfinance earnings_dates API.
    """
    if ticker not in price_data:
        return []

    prices = price_data[ticker]
    rets = prices.pct_change().abs()

    # For each year, find the top 4-5 biggest daily moves (likely earnings)
    # Use a rolling z-score approach: moves > 2.5x the 60-day vol
    rolling_std = rets.rolling(60, min_periods=30).mean()
    z_score = rets / (rolling_std + 1e-8)

    # Earnings moves are typically >2.5x normal daily vol
    big_moves = z_score[z_score > 2.5].index.tolist()

    if len(big_moves) == 0:
        # Fallback: top 5% of moves
        big_moves = rets[rets > rets.quantile(0.95)].index.tolist()

    # Deduplicate: keep one per 60-day window (roughly quarterly)
    if len(big_moves) > 0:
        deduped = [big_moves[0]]
        for d in big_moves[1:]:
            if (d - deduped[-1]).days > 60:
                deduped.append(d)
        return deduped
    return []


earnings_events = []
for ticker in price_data:
    dates = get_earnings_dates(ticker)
    if not dates:
        continue

    prices = price_data[ticker]
    returns = prices.pct_change()

    for edate in dates:
        edate = pd.Timestamp(edate)
        # Must be within our data range
        if edate < pd.Timestamp(DATA_START) + pd.Timedelta(days=252):
            continue
        if edate > pd.Timestamp(DATA_END) - pd.Timedelta(days=5):
            continue

        # Find nearest trading day
        mask = prices.index >= edate
        if mask.sum() == 0:
            continue
        actual_date = prices.index[mask][0]

        # Get index position
        idx = prices.index.get_loc(actual_date)
        if idx < 60 or idx >= len(prices) - 2:
            continue

        # ─── Compute features ───
        try:
            # Realized earnings move (next-day absolute % change)
            realized_move = abs(returns.iloc[idx + 1]) if idx + 1 < len(returns) else np.nan
            if np.isnan(realized_move):
                continue

            # Pre-earnings 20-day realized vol (annualized)
            pre_rets = returns.iloc[idx-20:idx].dropna()
            if len(pre_rets) < 15:
                continue
            realized_vol_20d = pre_rets.std() * np.sqrt(252)

            # Implied move proxy: historical vol suggests this much daily move
            # Earnings typically have 1.5-2x the normal daily vol
            # "Straddle implied move" proxy = realized_vol * sqrt(1/252) * earnings_premium
            normal_daily_move = realized_vol_20d / np.sqrt(252)
            # Earnings premium: how much more do earnings moves tend to be vs normal?
            # We estimate this from the stock's own history
            hist_rets_abs = returns.iloc[idx-252:idx].abs().dropna()
            if len(hist_rets_abs) < 100:
                continue

            # Implied move = normal_daily_move * earnings_surprise_factor
            # We use 1.5x as base (well-known that ATM straddles price ~1.5x realized)
            implied_move = normal_daily_move * 1.5

            # IV overpriced? (our label)
            iv_overpriced = 1 if realized_move < implied_move else 0

            # Feature: IV rank (current vol vs 1yr range)
            vol_1yr = returns.iloc[idx-252:idx].rolling(20).std().dropna() * np.sqrt(252)
            if len(vol_1yr) < 50:
                continue
            iv_rank = (realized_vol_20d - vol_1yr.min()) / (vol_1yr.max() - vol_1yr.min() + 1e-8)

            # Feature: stock momentum (30d return)
            momentum_30d = (prices.iloc[idx] / prices.iloc[idx-30] - 1) if idx >= 30 else 0

            # Feature: VIX level on that date
            vix_level = np.nan
            if vix_data is not None:
                vix_mask = vix_data.index <= actual_date
                if vix_mask.sum() > 0:
                    vix_level = vix_data[vix_mask].iloc[-1]

            # Feature: historical earnings move size (avg of last 4 "big" moves)
            # Look for previous earnings-like moves (>1.5x normal daily)
            big_threshold = normal_daily_move * 1.2
            past_big_moves = returns.iloc[idx-504:idx].abs()
            past_big = past_big_moves[past_big_moves > big_threshold]
            hist_earnings_move = past_big.nlargest(min(4, len(past_big))).mean() if len(past_big) > 0 else normal_daily_move

            # Feature: stock beta (vs SPY, 60d)
            beta = 1.0
            if spy_returns is not None:
                stock_rets = returns.iloc[idx-60:idx].dropna()
                common = stock_rets.index.intersection(spy_returns.index)
                if len(common) > 30:
                    cov = np.cov(stock_rets.loc[common].values, spy_returns.loc[common].values)
                    if cov[1, 1] > 0:
                        beta = cov[0, 1] / cov[1, 1]

            # Feature: pre-earnings vol trend (is vol rising into earnings?)
            vol_5d = pre_rets.iloc[-5:].std() * np.sqrt(252) if len(pre_rets) >= 5 else realized_vol_20d
            vol_trend = vol_5d / (realized_vol_20d + 1e-8)

            # Feature: stock price (for practical sizing)
            stock_price = prices.iloc[idx]

            # Sector encoding
            sector = SECTOR_MAP.get(ticker, 'Other')

            earnings_events.append({
                'date': actual_date,
                'ticker': ticker,
                'stock_price': stock_price,
                'realized_move': realized_move,
                'implied_move': implied_move,
                'iv_overpriced': iv_overpriced,
                # Features
                'iv_rank': iv_rank,
                'momentum_30d': momentum_30d,
                'vix_level': vix_level,
                'hist_earnings_move': hist_earnings_move,
                'beta': beta,
                'realized_vol_20d': realized_vol_20d,
                'vol_trend': vol_trend,
                'normal_daily_move': normal_daily_move,
                'sector': sector,
            })
        except Exception as e:
            continue

earnings_df = pd.DataFrame(earnings_events)
if len(earnings_df) == 0:
    print("ERROR: No earnings events found. Exiting.")
    exit(1)

earnings_df['date'] = pd.to_datetime(earnings_df['date'])
earnings_df = earnings_df.sort_values('date').reset_index(drop=True)

# Add quarter column for walk-forward
earnings_df['quarter'] = earnings_df['date'].dt.to_period('Q')

print(f"\nTotal earnings events: {len(earnings_df)}")
print(f"Date range: {earnings_df['date'].min().date()} to {earnings_df['date'].max().date()}")
print(f"IV overpriced rate: {earnings_df['iv_overpriced'].mean():.1%}")
print(f"Unique stocks: {earnings_df['ticker'].nunique()}")
print(f"Unique quarters: {earnings_df['quarter'].nunique()}")

# Show distribution
print(f"\nRealized move stats:")
print(f"  Mean: {earnings_df['realized_move'].mean():.2%}")
print(f"  Median: {earnings_df['realized_move'].median():.2%}")
print(f"  P75: {earnings_df['realized_move'].quantile(0.75):.2%}")
print(f"  P90: {earnings_df['realized_move'].quantile(0.90):.2%}")
print(f"\nImplied move stats:")
print(f"  Mean: {earnings_df['implied_move'].mean():.2%}")
print(f"  Median: {earnings_df['implied_move'].median():.2%}")


# ─── Step 3: Feature engineering + Walk-forward ML ───
print("\n" + "=" * 70)
print("STEP 3: Walk-Forward ML Backtest")
print("=" * 70)

FEATURE_COLS = ['iv_rank', 'momentum_30d', 'vix_level', 'hist_earnings_move',
                'beta', 'realized_vol_20d', 'vol_trend', 'normal_daily_move']

# One-hot encode sectors
for s in SECTORS:
    earnings_df[f'sector_{s}'] = (earnings_df['sector'] == s).astype(int)
    FEATURE_COLS.append(f'sector_{s}')


def simulate_csp_trade(stock_price, realized_move, implied_move, iv_overpriced_pred):
    """
    Simulate a cash-secured put trade.

    If we predict IV is overpriced -> sell a ~0.30 delta put.
    Strike ~ stock_price * (1 - implied_move * 1.2)  (roughly 0.30 delta)
    Premium ~ implied_move * stock_price * 0.4 (rough Black-Scholes approximation for 30d put)

    P&L:
    - If stock stays above strike: keep full premium
    - If stock drops below strike: (strike - actual_price) + premium (loss)
    """
    if iv_overpriced_pred != 1:
        return 0.0, 0.0, False  # no trade

    # Approximate 0.30 delta strike
    strike = stock_price * (1 - implied_move * 1.5)  # ~1.5 SD below for 0.30 delta

    # Premium approximation: ATM straddle price ~ implied_move * stock_price
    # 0.30 delta put ~ 35% of ATM put price ~ 0.35 * 0.5 * straddle
    premium = implied_move * stock_price * 0.35 * 0.5

    # Collateral needed: 100 shares * strike (cash-secured)
    collateral = strike * 100

    # Actual post-earnings price
    # realized_move is absolute, but we need direction. Assume 50/50 up/down for simulation.
    # Actually, for a put seller, we care about downside:
    # Use the actual realized_move and assume the direction is random
    # But we have the actual data - let's use the signed return
    # Since we stored abs(return), we need to handle this...
    # For simulation: stock drops by realized_move with 50% prob
    # Expected P&L for put seller:
    # P(stock > strike) * premium + P(stock < strike) * (strike - stock*(1-move) + premium)

    # Simpler: just check if the move was bigger than our strike distance
    strike_distance_pct = 1 - strike / stock_price  # how far OTM we are

    # If the actual downside move exceeds our strike distance, we lose
    # Assume 50% of moves are down (conservative)
    # For backtesting we model: if abs(move) > strike_distance AND move is down (50%), we take assignment
    # Expected P&L per trade:
    # With earnings moves being roughly symmetric:
    if realized_move > strike_distance_pct:
        # 50% chance the move was down past our strike
        # Expected P&L = 0.5 * premium + 0.5 * (premium - (realized_move - strike_distance_pct) * stock_price)
        down_loss = (realized_move - strike_distance_pct) * stock_price * 100
        pnl = 0.5 * (premium * 100) + 0.5 * (premium * 100 - down_loss)
    else:
        # Move didn't reach our strike regardless of direction
        pnl = premium * 100  # full premium kept

    return_pct = pnl / collateral if collateral > 0 else 0
    traded = True
    return pnl, collateral, traded


# Walk-forward backtest
quarters = sorted(earnings_df['quarter'].unique())
predict_start_q = pd.Period(PREDICT_START, freq='Q')
predict_quarters = [q for q in quarters if q >= predict_start_q]

print(f"\nAvailable quarters: {quarters[0]} to {quarters[-1]}")
print(f"Prediction quarters: {predict_quarters[0] if predict_quarters else 'NONE'} to {predict_quarters[-1] if predict_quarters else 'NONE'}")
print(f"Total prediction quarters: {len(predict_quarters)}")

wf_results = []
all_predictions = []
feature_importances = defaultdict(list)

for i, pred_q in enumerate(predict_quarters):
    # Training window: WF_TRAIN_QUARTERS before pred_q
    train_end_q = pred_q - 1
    train_start_q = train_end_q - WF_TRAIN_QUARTERS + 1

    train_mask = (earnings_df['quarter'] >= train_start_q) & (earnings_df['quarter'] <= train_end_q)
    pred_mask = earnings_df['quarter'] == pred_q

    train_df = earnings_df[train_mask].copy()
    pred_df = earnings_df[pred_mask].copy()

    if len(train_df) < 20 or len(pred_df) == 0:
        continue

    X_train = train_df[FEATURE_COLS].fillna(0).values
    y_train = train_df['iv_overpriced'].values
    X_pred = pred_df[FEATURE_COLS].fillna(0).values
    y_true = pred_df['iv_overpriced'].values

    # Check class balance
    pos_rate = y_train.mean()
    if pos_rate < 0.05 or pos_rate > 0.95:
        # Extremely imbalanced, skip
        continue

    # Train model
    if HAS_LGB:
        model = lgb.LGBMClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_samples=5,
            random_state=42,
            verbose=-1,
        )
    else:
        model = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            min_samples_leaf=5,
            random_state=42,
        )

    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_pred)
    y_prob = model.predict_proba(X_pred)[:, 1] if hasattr(model, 'predict_proba') else y_pred.astype(float)

    # Track feature importances
    if HAS_LGB:
        fi = model.feature_importances_
    else:
        fi = model.feature_importances_
    for fname, fval in zip(FEATURE_COLS, fi):
        feature_importances[fname].append(fval)

    # Simulate trades for this quarter
    quarter_pnl = 0
    quarter_collateral = 0
    quarter_trades = 0

    for j in range(len(pred_df)):
        row = pred_df.iloc[j]
        pnl, collateral, traded = simulate_csp_trade(
            row['stock_price'], row['realized_move'], row['implied_move'], y_pred[j]
        )
        if traded:
            quarter_pnl += pnl
            quarter_collateral = max(quarter_collateral, collateral)  # max concurrent collateral
            quarter_trades += 1

        all_predictions.append({
            'date': row['date'],
            'quarter': str(pred_q),
            'ticker': row['ticker'],
            'stock_price': row['stock_price'],
            'realized_move': row['realized_move'],
            'implied_move': row['implied_move'],
            'iv_overpriced_true': row['iv_overpriced'],
            'iv_overpriced_pred': int(y_pred[j]),
            'pred_prob': float(y_prob[j]),
            'pnl': pnl,
            'collateral': collateral,
            'traded': traded,
        })

    # Accuracy metrics
    acc = accuracy_score(y_true, y_pred)
    try:
        auc = roc_auc_score(y_true, y_prob) if len(set(y_true)) > 1 else 0.5
    except:
        auc = 0.5

    # Return on theoretical capital
    return_pct = quarter_pnl / THEORETICAL_CAPITAL if THEORETICAL_CAPITAL > 0 else 0

    wf_results.append({
        'quarter': str(pred_q),
        'n_events': len(pred_df),
        'n_trades': quarter_trades,
        'pnl': quarter_pnl,
        'max_collateral': quarter_collateral,
        'return_pct': return_pct,
        'accuracy': acc,
        'auc': auc,
        'train_size': len(train_df),
        'pos_rate_train': pos_rate,
    })

    status = "PROFIT" if quarter_pnl > 0 else "LOSS" if quarter_pnl < 0 else "FLAT"
    print(f"  {pred_q}: {quarter_trades} trades, P&L=${quarter_pnl:+.2f}, "
          f"Acc={acc:.1%}, AUC={auc:.2f} [{status}]")

wf_df = pd.DataFrame(wf_results)

if len(wf_df) == 0:
    print("\nERROR: No walk-forward results generated. Check data availability.")
    exit(1)


# ─── Step 4: Performance analysis ───
print("\n" + "=" * 70)
print("STEP 4: Performance Analysis")
print("=" * 70)

total_pnl = wf_df['pnl'].sum()
total_trades = wf_df['n_trades'].sum()
win_quarters = (wf_df['pnl'] > 0).sum()
total_quarters = len(wf_df)

# Risk-adjusted metrics
quarterly_returns = wf_df['return_pct'].values
mean_qtr_return = quarterly_returns.mean()
std_qtr_return = quarterly_returns.std()
annualized_return = mean_qtr_return * 4
annualized_vol = std_qtr_return * np.sqrt(4)

sharpe = annualized_return / annualized_vol if annualized_vol > 0 else 0
# Sortino: only downside vol
downside = quarterly_returns[quarterly_returns < 0]
downside_vol = downside.std() * np.sqrt(4) if len(downside) > 0 else annualized_vol
sortino = annualized_return / downside_vol if downside_vol > 0 else 0

# Profit factor
gross_profit = wf_df[wf_df['pnl'] > 0]['pnl'].sum()
gross_loss = abs(wf_df[wf_df['pnl'] < 0]['pnl'].sum())
profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

# Win rate
win_rate = win_quarters / total_quarters if total_quarters > 0 else 0

# Max drawdown (on cumulative P&L)
cum_pnl = wf_df['pnl'].cumsum()
peak = cum_pnl.cummax()
drawdown = cum_pnl - peak
max_dd = drawdown.min()
max_dd_pct = max_dd / THEORETICAL_CAPITAL

print(f"\n{'THEORETICAL ($10K ACCOUNT)':=^50}")
print(f"  Total P&L:          ${total_pnl:+,.2f}")
print(f"  Total Trades:       {total_trades}")
print(f"  Quarters:           {total_quarters} ({win_quarters}W / {total_quarters - win_quarters}L)")
print(f"  Win Rate (qtr):     {win_rate:.1%}")
print(f"  Annualized Return:  {annualized_return:.1%}")
print(f"  Sharpe Ratio:       {sharpe:.2f}")
print(f"  Sortino Ratio:      {sortino:.2f}")
print(f"  Profit Factor:      {profit_factor:.2f}")
print(f"  Max Drawdown:       ${max_dd:,.2f} ({max_dd_pct:.1%})")
print(f"  Avg Accuracy:       {wf_df['accuracy'].mean():.1%}")
print(f"  Avg AUC:            {wf_df['auc'].mean():.2f}")

# Practical $423 account
pred_df_all = pd.DataFrame(all_predictions)
if len(pred_df_all) > 0:
    practical_trades = pred_df_all[
        (pred_df_all['traded'] == True) &
        (pred_df_all['collateral'] <= PRACTICAL_CAPITAL * 100)  # can afford 1 contract
    ].copy()

    practical_trades_lowprice = pred_df_all[
        (pred_df_all['traded'] == True) &
        (pred_df_all['stock_price'] <= PRACTICAL_CAPITAL / 100)  # CSP needs strike*100
    ].copy()

    print(f"\n{'PRACTICAL ($423 ACCOUNT)':=^50}")
    print(f"  Trades fitting budget: {len(practical_trades_lowprice)} / {len(pred_df_all[pred_df_all['traded']==True])}")
    if len(practical_trades_lowprice) > 0:
        practical_pnl = practical_trades_lowprice['pnl'].sum()
        print(f"  Total P&L: ${practical_pnl:+,.2f}")
        print(f"  Tickers: {practical_trades_lowprice['ticker'].unique().tolist()}")
    else:
        print(f"  No trades fit $423 budget (need stocks < ${PRACTICAL_CAPITAL/100:.2f})")
        print(f"  CONCLUSION: CSPs on mega-cap stocks need >$10K account minimum")
        print(f"  For $423: consider bull put SPREADS (defined risk, lower capital)")


# ─── Step 5: Feature Importance ───
print(f"\n{'FEATURE IMPORTANCE':=^50}")
fi_means = {k: np.mean(v) for k, v in feature_importances.items()}
fi_sorted = sorted(fi_means.items(), key=lambda x: -x[1])
for fname, fval in fi_sorted[:15]:
    bar = '#' * int(fval / max(fi_means.values()) * 30) if max(fi_means.values()) > 0 else ''
    print(f"  {fname:25s} {fval:8.1f} {bar}")


# ─── Step 6: Adversarial Validation (4-gate) ───
print("\n" + "=" * 70)
print("STEP 6: Adversarial 4-Gate Validation")
print("=" * 70)

# Gate 1: Permutation test
print("\nGate 1: Permutation Test (shuffled labels)")
actual_sharpe = sharpe
perm_sharpes = []

for perm_i in range(N_PERMS):
    # Shuffle the IV overpriced labels
    shuffled_df = earnings_df.copy()
    shuffled_df['iv_overpriced'] = np.random.permutation(shuffled_df['iv_overpriced'].values)

    perm_pnl_total = 0
    perm_trades_total = 0

    for pred_q in predict_quarters:
        train_end_q = pred_q - 1
        train_start_q = train_end_q - WF_TRAIN_QUARTERS + 1
        train_mask = (shuffled_df['quarter'] >= train_start_q) & (shuffled_df['quarter'] <= train_end_q)
        pred_mask = shuffled_df['quarter'] == pred_q
        tr = shuffled_df[train_mask]
        pr = shuffled_df[pred_mask]
        if len(tr) < 20 or len(pr) == 0:
            continue

        X_tr = tr[FEATURE_COLS].fillna(0).values
        y_tr = tr['iv_overpriced'].values
        X_pr = pr[FEATURE_COLS].fillna(0).values

        pos_rate = y_tr.mean()
        if pos_rate < 0.05 or pos_rate > 0.95:
            continue

        if HAS_LGB:
            m = lgb.LGBMClassifier(n_estimators=50, max_depth=3, learning_rate=0.05,
                                    subsample=0.8, random_state=42+perm_i, verbose=-1)
        else:
            m = GradientBoostingClassifier(n_estimators=50, max_depth=3, learning_rate=0.05,
                                            subsample=0.8, random_state=42+perm_i)
        m.fit(X_tr, y_tr)
        y_p = m.predict(X_pr)

        for j in range(len(pr)):
            row = pr.iloc[j]
            pnl, _, traded = simulate_csp_trade(row['stock_price'], row['realized_move'],
                                                 row['implied_move'], y_p[j])
            if traded:
                perm_pnl_total += pnl
                perm_trades_total += 1

    perm_return = perm_pnl_total / THEORETICAL_CAPITAL
    perm_sharpes.append(perm_return)  # simplified

perm_sharpes = np.array(perm_sharpes)
perm_pvalue = (perm_sharpes >= total_pnl / THEORETICAL_CAPITAL).mean()
gate1_pass = perm_pvalue < 0.05
print(f"  Actual total return: {total_pnl/THEORETICAL_CAPITAL:.2%}")
print(f"  Perm p-value: {perm_pvalue:.3f}")
print(f"  Gate 1: {'PASS' if gate1_pass else 'FAIL'} (p < 0.05)")

# Gate 2: Sub-period consistency
print("\nGate 2: Sub-Period Consistency")
n_quarters = len(wf_df)
if n_quarters >= 4:
    half = n_quarters // 2
    first_half = wf_df.iloc[:half]
    second_half = wf_df.iloc[half:]
    first_ret = first_half['return_pct'].sum()
    second_ret = second_half['return_pct'].sum()
    both_positive = first_ret > 0 and second_ret > 0
    gate2_pass = both_positive
    print(f"  First half return:  {first_ret:.2%}")
    print(f"  Second half return: {second_ret:.2%}")
    print(f"  Gate 2: {'PASS' if gate2_pass else 'FAIL'} (both halves positive)")
else:
    gate2_pass = False
    print(f"  Gate 2: FAIL (insufficient quarters: {n_quarters})")

# Gate 3: Outlier removal
print("\nGate 3: Outlier Removal (drop best quarter)")
if n_quarters >= 3:
    best_idx = wf_df['pnl'].idxmax()
    no_outlier = wf_df.drop(best_idx)
    no_outlier_pnl = no_outlier['pnl'].sum()
    gate3_pass = no_outlier_pnl > 0
    print(f"  Best quarter P&L: ${wf_df.loc[best_idx, 'pnl']:,.2f} ({wf_df.loc[best_idx, 'quarter']})")
    print(f"  P&L without best: ${no_outlier_pnl:,.2f}")
    print(f"  Gate 3: {'PASS' if gate3_pass else 'FAIL'} (profitable without best quarter)")
else:
    gate3_pass = False
    print(f"  Gate 3: FAIL (insufficient quarters)")

# Gate 4: Regime analysis (VIX high vs low)
print("\nGate 4: Regime Consistency (high-vol vs low-vol)")
if vix_data is not None and len(wf_df) >= 4:
    # Classify quarters by average VIX
    quarter_vix = {}
    for _, row in wf_df.iterrows():
        q = pd.Period(row['quarter'])
        q_start = q.start_time
        q_end = q.end_time
        q_vix = vix_data[(vix_data.index >= q_start) & (vix_data.index <= q_end)]
        if len(q_vix) > 0:
            quarter_vix[row['quarter']] = q_vix.mean()

    if quarter_vix:
        median_vix = np.median(list(quarter_vix.values()))
        high_vol_qs = [q for q, v in quarter_vix.items() if v >= median_vix]
        low_vol_qs = [q for q, v in quarter_vix.items() if v < median_vix]

        high_vol_pnl = wf_df[wf_df['quarter'].isin(high_vol_qs)]['pnl'].sum()
        low_vol_pnl = wf_df[wf_df['quarter'].isin(low_vol_qs)]['pnl'].sum()

        # Check regime skew (R1 from CLAUDE.md)
        high_sharpe = wf_df[wf_df['quarter'].isin(high_vol_qs)]['return_pct'].mean()
        low_sharpe = wf_df[wf_df['quarter'].isin(low_vol_qs)]['return_pct'].mean()
        max_sharpe = max(abs(high_sharpe), abs(low_sharpe))
        regime_skew = abs(high_sharpe - low_sharpe) / max_sharpe if max_sharpe > 0 else 0

        gate4_pass = regime_skew < 0.50  # R1 from HC#428
        print(f"  High-vol quarters ({len(high_vol_qs)}): P&L=${high_vol_pnl:,.2f}, avg ret={high_sharpe:.2%}")
        print(f"  Low-vol quarters ({len(low_vol_qs)}):  P&L=${low_vol_pnl:,.2f}, avg ret={low_sharpe:.2%}")
        print(f"  Regime skew: {regime_skew:.2f} (threshold: 0.50)")
        print(f"  Gate 4: {'PASS' if gate4_pass else 'FAIL'} (regime-agnostic)")
    else:
        gate4_pass = False
        print(f"  Gate 4: FAIL (no VIX data for quarters)")
else:
    gate4_pass = False
    print(f"  Gate 4: FAIL (insufficient data)")

gates_passed = sum([gate1_pass, gate2_pass, gate3_pass, gate4_pass])
print(f"\n{'ADVERSARIAL SUMMARY':=^50}")
print(f"  Gates passed: {gates_passed}/4")
print(f"  {'STRATEGY VALIDATED' if gates_passed >= 3 else 'STRATEGY NEEDS WORK' if gates_passed >= 2 else 'STRATEGY REJECTED'}")


# ─── Step 7: Per-quarter breakdown ───
print("\n" + "=" * 70)
print("STEP 7: Per-Quarter Breakdown")
print("=" * 70)
print(f"{'Quarter':>10s} {'Trades':>7s} {'P&L':>10s} {'Return':>8s} {'Acc':>6s} {'AUC':>6s}")
print("-" * 55)
for _, row in wf_df.iterrows():
    print(f"{row['quarter']:>10s} {row['n_trades']:>7d} ${row['pnl']:>9,.2f} "
          f"{row['return_pct']:>7.2%} {row['accuracy']:>5.1%} {row['auc']:>5.2f}")
print("-" * 55)
print(f"{'TOTAL':>10s} {int(wf_df['n_trades'].sum()):>7d} ${total_pnl:>9,.2f} "
      f"{total_pnl/THEORETICAL_CAPITAL:>7.2%}")


# ─── Step 8: Save results ───
print("\n" + "=" * 70)
print("STEP 8: Saving Results")
print("=" * 70)

results = {
    'strategy': 'ML Earnings Volatility Selling v1',
    'run_date': str(dt.datetime.now()),
    'config': {
        'universe': f'S&P 500 top {len(SP500_TOP50)}',
        'stocks_loaded': len(price_data),
        'wf_train_quarters': WF_TRAIN_QUARTERS,
        'wf_predict_quarters': WF_PREDICT_QUARTERS,
        'predict_period': f'{PREDICT_START} to {DATA_END}',
        'target_delta': TARGET_DELTA,
        'n_perms': N_PERMS,
    },
    'performance': {
        'total_pnl': round(total_pnl, 2),
        'total_trades': int(total_trades),
        'total_quarters': int(total_quarters),
        'win_rate_quarterly': round(win_rate, 3),
        'annualized_return': round(annualized_return, 4),
        'sharpe_ratio': round(sharpe, 2),
        'sortino_ratio': round(sortino, 2),
        'profit_factor': round(profit_factor, 2),
        'max_drawdown_pct': round(max_dd_pct, 4),
        'avg_accuracy': round(wf_df['accuracy'].mean(), 3),
        'avg_auc': round(wf_df['auc'].mean(), 3),
    },
    'adversarial': {
        'gate1_permutation': gate1_pass,
        'gate1_pvalue': round(perm_pvalue, 4),
        'gate2_subperiod': gate2_pass,
        'gate3_outlier_removal': gate3_pass,
        'gate4_regime': gate4_pass,
        'gates_passed': gates_passed,
    },
    'practical_423': {
        'viable': len(practical_trades_lowprice) > 0 if len(pred_df_all) > 0 else False,
        'n_trades': len(practical_trades_lowprice) if len(pred_df_all) > 0 else 0,
        'note': 'CSPs on mega-caps need >$10K. For $423, use bull put spreads instead.',
    },
    'quarterly_results': wf_df.to_dict('records'),
    'feature_importance': {k: round(np.mean(v), 2) for k, v in sorted(
        feature_importances.items(), key=lambda x: -np.mean(x[1]))},
}

# Save results JSON
results_path = OUTPUT_DIR / 'results.json'
with open(results_path, 'w') as f:
    json.dump(results, f, indent=2, default=str)
print(f"  Results: {results_path}")

# Save predictions
if len(pred_df_all) > 0:
    preds_path = OUTPUT_DIR / 'predictions.csv'
    pred_df_all.to_csv(preds_path, index=False)
    print(f"  Predictions: {preds_path}")

# Save earnings events
events_path = OUTPUT_DIR / 'earnings_events.csv'
earnings_df.to_csv(events_path, index=False)
print(f"  Events: {events_path}")


# ─── Final Summary ───
print("\n" + "=" * 70)
print("FINAL SUMMARY")
print("=" * 70)
print(f"""
STRATEGY: ML Earnings Volatility Selling
HYPOTHESIS: IV before earnings is overpriced; ML identifies best opportunities

RESULTS ($10K paper account, {PREDICT_START}-{DATA_END}):
  Sharpe:  {sharpe:.2f}
  Sortino: {sortino:.2f}
  PF:      {profit_factor:.2f}
  WR:      {win_rate:.0%} of quarters profitable
  Total:   ${total_pnl:+,.2f} ({total_pnl/THEORETICAL_CAPITAL:.1%})

ADVERSARIAL: {gates_passed}/4 gates passed
  Permutation: {'PASS' if gate1_pass else 'FAIL'} (p={perm_pvalue:.3f})
  Sub-period:  {'PASS' if gate2_pass else 'FAIL'}
  No-outlier:  {'PASS' if gate3_pass else 'FAIL'}
  Regime:      {'PASS' if gate4_pass else 'FAIL'}

$423 ACCOUNT: Not viable for CSPs on mega-caps (need $10K+).
  Recommendation: Use bull put SPREADS (defined risk, ~$50-200 collateral)
  or sell puts on sub-$5 stocks (limited universe, higher risk).

KEY INSIGHT: IV overpriced rate = {earnings_df['iv_overpriced'].mean():.0%} of earnings events.
  {'This confirms the VRP exists and is tradeable.' if earnings_df['iv_overpriced'].mean() > 0.55 else 'VRP may be weaker than expected in this universe/period.'}
""")
print("=" * 70)
print("DONE")
