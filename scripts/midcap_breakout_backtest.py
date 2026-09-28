#!/usr/bin/env python3
"""
Mid-Cap Breakout / Mean-Reversion Hybrid Backtester
====================================================
Universe : S&P 400 MidCap (tickers from Wikipedia)
Period   : 2016-01-01  →  2026-01-01
Walk-fwd : 2-yr train  →  1-yr OOT (sliding)

Entry signals
  A) 52-wk high breakout on volume > 1.5× 20d avg
  B) Pullback 5-10 % from 52-wk high, 50d MA > 200d MA

Exit rules
  - Trailing stop  : 2.5 × ATR(20)
  - Death cross    : 50d MA < 200d MA
  - Max hold       : 126 trading days (~6 months)
  - Volume dry-up  : 20d avg vol < 50 % of entry-day volume

Position sizing : equal-weight, max 20 positions, 2 % risk per pos via ATR
Risk controls   : max 5 per sector, max 30 % portfolio in any sector

Benchmark       : MDY (S&P 400 ETF)
"""

import os, sys, json, time, warnings, logging
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)

try:
    import yfinance as yf
except ImportError:
    sys.exit("pip install yfinance")

# ── paths ────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
OUT  = ROOT / "output" / "midcap_breakout"
OUT.mkdir(parents=True, exist_ok=True)
CACHE_DIR = OUT / "cache"
CACHE_DIR.mkdir(exist_ok=True)

# ── constants ────────────────────────────────────────────────────────
START       = "2016-01-01"
END         = "2026-01-01"
TRAIN_DAYS  = 504          # ~2 yr
OOT_DAYS    = 252          # ~1 yr
MAX_POS     = 20
RISK_PCT    = 0.02         # 2 % risk per position
ATR_MULT    = 2.5          # trailing-stop multiplier
ATR_LEN     = 20
VOL_MULT    = 1.5          # breakout volume threshold
PB_LO       = 0.05         # pullback lower bound
PB_HI       = 0.10         # pullback upper bound
MAX_HOLD    = 126          # trading days
VOL_DRYUP   = 0.50         # volume dry-up threshold
MAX_SECTOR_POS = 5
MAX_SECTOR_PCT = 0.30
COMMISSION_BPS = 5         # 5 bps round-trip for equities

# ── 1. Universe: S&P 400 tickers from Wikipedia ─────────────────────

def fetch_sp400_tickers() -> pd.DataFrame:
    """Scrape S&P 400 MidCap constituents from Wikipedia."""
    cache = CACHE_DIR / "sp400_tickers.csv"
    if cache.exists():
        return pd.read_csv(cache)

    log.info("Fetching S&P 400 tickers from Wikipedia …")
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_400_companies"
    try:
        tables = pd.read_html(url)
        df = tables[0]
        # Normalise column names
        df.columns = [c.strip() for c in df.columns]
        # Find ticker & sector columns
        ticker_col = [c for c in df.columns if "symbol" in c.lower() or "ticker" in c.lower()][0]
        sector_col = [c for c in df.columns if "sector" in c.lower() or "gics" in c.lower()][0]
        out = df[[ticker_col, sector_col]].copy()
        out.columns = ["ticker", "sector"]
        out["ticker"] = out["ticker"].str.replace(".", "-", regex=False)  # BRK.B → BRK-B
        out.to_csv(cache, index=False)
        log.info(f"  → {len(out)} tickers cached")
        return out
    except Exception as e:
        log.warning(f"Wikipedia scrape failed ({e}); using fallback MDY holdings approach")
        # Fallback: just use a representative set
        return _fallback_tickers()


def _fallback_tickers():
    """Expanded fallback: ~200 representative S&P 400 mid-caps across sectors."""
    tickers_sectors = {
        # Industrials
        "SAIA": "Industrials", "FIX": "Industrials", "MTZ": "Industrials",
        "GXO": "Industrials", "LSTR": "Industrials", "CW": "Industrials",
        "ESAB": "Industrials", "RBC": "Industrials", "WMS": "Industrials",
        "SITE": "Industrials", "AIT": "Industrials", "GATX": "Industrials",
        "KNF": "Industrials", "AAON": "Industrials", "TREX": "Industrials",
        "EXPO": "Industrials", "BWXT": "Industrials", "MMS": "Industrials",
        "NDSN": "Industrials", "SSD": "Industrials", "TTC": "Industrials",
        "RRX": "Industrials", "ROAD": "Industrials", "APOG": "Industrials",
        "BMI": "Industrials", "SPSC": "Industrials",
        # Consumer Discretionary
        "DECK": "Consumer Discretionary", "BURL": "Consumer Discretionary",
        "TOL": "Consumer Discretionary", "TXRH": "Consumer Discretionary",
        "CROX": "Consumer Discretionary", "WSM": "Consumer Discretionary",
        "WYNN": "Consumer Discretionary", "PENN": "Consumer Discretionary",
        "MGM": "Consumer Discretionary", "LKQ": "Consumer Discretionary",
        "GPC": "Consumer Discretionary", "ETSY": "Consumer Discretionary",
        "GNTX": "Consumer Discretionary", "BC": "Consumer Discretionary",
        "CBT": "Consumer Discretionary", "IPAR": "Consumer Discretionary",
        "SHOO": "Consumer Discretionary", "BOOT": "Consumer Discretionary",
        "PLNT": "Consumer Discretionary", "CABO": "Consumer Discretionary",
        "DDS": "Consumer Discretionary",
        # Technology
        "MANH": "Information Technology", "PCTY": "Information Technology",
        "LSCC": "Information Technology", "RMBS": "Information Technology",
        "EXLS": "Information Technology", "CACI": "Information Technology",
        "SMAR": "Information Technology", "POWI": "Information Technology",
        "CGNX": "Information Technology", "NOVT": "Information Technology",
        "WK": "Information Technology", "ASGN": "Information Technology",
        "VRNS": "Information Technology", "MTSI": "Information Technology",
        "CALX": "Information Technology", "IDCC": "Information Technology",
        "CIEN": "Information Technology", "SMTC": "Information Technology",
        "LFUS": "Information Technology", "CVLT": "Information Technology",
        "NSIT": "Information Technology", "TTMI": "Information Technology",
        "DIOD": "Information Technology",
        # Financials
        "FNF": "Financials", "ALLY": "Financials", "EWBC": "Financials",
        "FHN": "Financials", "HLI": "Financials", "EVR": "Financials",
        "RNR": "Financials", "SEIC": "Financials", "OZK": "Financials",
        "WBS": "Financials", "UBSI": "Financials", "PRI": "Financials",
        "CATY": "Financials", "PNFP": "Financials", "FNB": "Financials",
        "IBOC": "Financials", "PPBI": "Financials", "HWC": "Financials",
        "PIPR": "Financials", "SFBS": "Financials", "GBCI": "Financials",
        "SBCF": "Financials", "TCBI": "Financials",
        # Healthcare
        "MEDP": "Health Care", "EHC": "Health Care", "OMCL": "Health Care",
        "ENSG": "Health Care", "LNTH": "Health Care", "ITCI": "Health Care",
        "PRCT": "Health Care", "HALO": "Health Care", "RARE": "Health Care",
        "ARVN": "Health Care", "PGNY": "Health Care", "NEO": "Health Care",
        "MASI": "Health Care", "CAKE": "Health Care", "AMED": "Health Care",
        "CORT": "Health Care", "SUPN": "Health Care", "TNDM": "Health Care",
        "ACHC": "Health Care", "AMN": "Health Care",
        # Energy
        "WFRD": "Energy", "DTM": "Energy", "SM": "Energy",
        "CIVI": "Energy", "MTDR": "Energy", "PTEN": "Energy",
        "HP": "Energy", "RRC": "Energy", "ARCH": "Energy",
        "CNX": "Energy", "GPOR": "Energy", "NOG": "Energy",
        "MGY": "Energy", "TRGP": "Energy",
        # Real Estate
        "GLPI": "Real Estate", "REXR": "Real Estate", "NNN": "Real Estate",
        "OHI": "Real Estate", "KRG": "Real Estate", "STAG": "Real Estate",
        "NSA": "Real Estate", "BRX": "Real Estate", "IIPR": "Real Estate",
        "PEB": "Real Estate", "RHP": "Real Estate", "CTRE": "Real Estate",
        "PECO": "Real Estate",
        # Consumer Staples
        "BRBR": "Consumer Staples", "ELF": "Consumer Staples",
        "CASY": "Consumer Staples", "PPC": "Consumer Staples",
        "INGR": "Consumer Staples", "FLO": "Consumer Staples",
        "FIZZ": "Consumer Staples", "SMPL": "Consumer Staples",
        "SPB": "Consumer Staples",
        # Materials
        "ATR": "Materials", "AVNT": "Materials", "CBT": "Materials",
        "UFPI": "Materials", "OLN": "Materials", "SLVM": "Materials",
        "HAYW": "Materials", "ATKR": "Materials", "BCPC": "Materials",
        # Utilities
        "OGE": "Utilities", "PNR": "Utilities", "NJR": "Utilities",
        "MDU": "Utilities", "SWX": "Utilities", "AVA": "Utilities",
        "NWE": "Utilities", "OTTR": "Utilities", "BKH": "Utilities",
        # Communication Services
        "CABO": "Communication Services", "LBRDA": "Communication Services",
        "IART": "Communication Services", "ZD": "Communication Services",
    }
    tickers = list(tickers_sectors.keys())
    sectors = list(tickers_sectors.values())
    df = pd.DataFrame({"ticker": tickers, "sector": sectors})
    return df


# ── 2. Data download ────────────────────────────────────────────────

def download_data(tickers: list, start: str, end: str) -> dict:
    """Download OHLCV for all tickers; cache as parquet."""
    cache = CACHE_DIR / "ohlcv_all.parquet"
    if cache.exists():
        log.info("Loading cached OHLCV data …")
        big = pd.read_parquet(cache)
        return {t: big[big["ticker"] == t].drop(columns="ticker").copy()
                for t in big["ticker"].unique()}

    log.info(f"Downloading {len(tickers)} tickers from yfinance …")
    data = {}
    batch_sz = 50
    for i in range(0, len(tickers), batch_sz):
        batch = tickers[i:i+batch_sz]
        log.info(f"  batch {i//batch_sz + 1}/{(len(tickers)-1)//batch_sz + 1}  ({len(batch)} tickers)")
        try:
            raw = yf.download(batch, start=start, end=end,
                              group_by="ticker", auto_adjust=True,
                              threads=True, progress=False)
            if len(batch) == 1:
                t = batch[0]
                if not raw.empty:
                    df = raw.copy()
                    df.columns = [c.lower() if isinstance(c, str) else c for c in df.columns]
                    # Flatten MultiIndex columns if present
                    if isinstance(df.columns, pd.MultiIndex):
                        df.columns = [c[0].lower() for c in df.columns]
                    data[t] = df
            else:
                for t in batch:
                    try:
                        if isinstance(raw.columns, pd.MultiIndex):
                            sub = raw[t].dropna(how="all")
                        else:
                            sub = raw.dropna(how="all")
                        if len(sub) < 252:
                            continue
                        sub.columns = [c.lower() if isinstance(c, str) else c for c in sub.columns]
                        if isinstance(sub.columns, pd.MultiIndex):
                            sub.columns = [c[0].lower() for c in sub.columns]
                        data[t] = sub.copy()
                    except Exception:
                        pass
        except Exception as e:
            log.warning(f"  batch download failed: {e}")
        time.sleep(0.5)

    # Cache
    frames = []
    for t, df in data.items():
        tmp = df.copy()
        tmp["ticker"] = t
        frames.append(tmp)
    if frames:
        big = pd.concat(frames)
        big.to_parquet(cache)
    log.info(f"  → {len(data)} tickers with data")
    return data


def download_benchmark(start: str, end: str) -> pd.Series:
    """Download MDY total return."""
    cache = CACHE_DIR / "mdy.parquet"
    if cache.exists():
        return pd.read_parquet(cache)["close"]
    raw = yf.download("MDY", start=start, end=end, auto_adjust=True, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = [c[0].lower() for c in raw.columns]
    else:
        raw.columns = [c.lower() for c in raw.columns]
    raw[["close"]].to_parquet(cache)
    return raw["close"]


# ── 3. Feature computation ──────────────────────────────────────────

def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add rolling features needed for signals and exits."""
    df = df.copy()
    c, h, l, v = df["close"], df["high"], df["low"], df["volume"]

    df["ma50"]  = c.rolling(50).mean()
    df["ma200"] = c.rolling(200).mean()
    df["hi252"] = h.rolling(252).max()            # 52-wk high
    df["vol20"] = v.rolling(20).mean()             # 20d avg volume

    # ATR(20)
    tr = pd.concat([
        h - l,
        (h - c.shift(1)).abs(),
        (l - c.shift(1)).abs(),
    ], axis=1).max(axis=1)
    df["atr20"] = tr.rolling(ATR_LEN).mean()

    # Drawdown from 52-wk high (for pullback signal)
    df["dd_from_hi"] = (df["hi252"] - c) / df["hi252"]

    return df


# ── 4. Signal generation ────────────────────────────────────────────

def check_breakout(row) -> bool:
    """New 52-wk high on elevated volume."""
    if pd.isna(row.get("hi252")) or pd.isna(row.get("vol20")):
        return False
    return (row["high"] >= row["hi252"]) and (row["volume"] > VOL_MULT * row["vol20"])


def check_pullback(row) -> bool:
    """5-10 % pullback from 52-wk high, trend intact."""
    if pd.isna(row.get("dd_from_hi")) or pd.isna(row.get("ma50")) or pd.isna(row.get("ma200")):
        return False
    return (PB_LO <= row["dd_from_hi"] <= PB_HI) and (row["ma50"] > row["ma200"])


# ── 5. Backtester engine ────────────────────────────────────────────

class Position:
    __slots__ = ("ticker", "entry_price", "entry_date", "entry_idx",
                 "entry_volume", "shares", "entry_type", "sector",
                 "trail_stop", "atr_at_entry", "max_price")

    def __init__(self, ticker, price, date, idx, volume, shares,
                 entry_type, sector, atr):
        self.ticker = ticker
        self.entry_price = price
        self.entry_date = date
        self.entry_idx = idx
        self.entry_volume = volume
        self.shares = shares
        self.entry_type = entry_type      # "breakout" or "pullback"
        self.sector = sector
        self.atr_at_entry = atr
        self.trail_stop = price - ATR_MULT * atr
        self.max_price = price


class Backtester:
    def __init__(self, data: dict, ticker_sectors: dict, benchmark: pd.Series):
        self.data = data                  # ticker → DataFrame (with features)
        self.sectors = ticker_sectors     # ticker → sector string
        self.benchmark = benchmark
        self.positions: list[Position] = []
        self.trades: list[dict] = []
        self.equity_curve: list[tuple] = []  # (date, nav)
        self.initial_capital = 1_000_000
        self.cash = self.initial_capital

    # ── helpers ──────────────────────────────────────────────────
    def _sector_counts(self) -> dict:
        counts = defaultdict(int)
        for p in self.positions:
            counts[p.sector] += 1
        return counts

    def _sector_exposure(self, nav: float) -> dict:
        exp = defaultdict(float)
        for p in self.positions:
            row = self._latest_row(p.ticker, p.entry_date)
            if row is not None:
                exp[p.sector] += p.shares * row["close"]
        return {s: v / nav for s, v in exp.items()}

    def _latest_row(self, ticker, date):
        df = self.data.get(ticker)
        if df is None:
            return None
        mask = df.index <= date
        if mask.sum() == 0:
            return None
        return df.loc[mask].iloc[-1]

    def _get_row(self, ticker, date):
        df = self.data.get(ticker)
        if df is None or date not in df.index:
            return None
        return df.loc[date]

    def _position_value(self, date):
        val = 0.0
        for p in self.positions:
            row = self._get_row(p.ticker, date)
            if row is not None:
                val += p.shares * row["close"]
            else:
                # Use last known price
                df = self.data.get(p.ticker)
                if df is not None:
                    prior = df.loc[df.index <= date]
                    if len(prior) > 0:
                        val += p.shares * prior.iloc[-1]["close"]
        return val

    # ── main loop ────────────────────────────────────────────────
    def run(self, oot_start, oot_end):
        """Run backtest over a specific OOT window."""
        # Build common date index
        all_dates = set()
        for df in self.data.values():
            all_dates.update(df.index)
        dates = sorted([d for d in all_dates if oot_start <= d <= oot_end])
        if not dates:
            return

        for date in dates:
            nav = self.cash + self._position_value(date)

            # ── check exits ──────────────────────────────────────
            to_close = []
            for i, pos in enumerate(self.positions):
                row = self._get_row(pos.ticker, date)
                if row is None:
                    continue
                df_ticker = self.data[pos.ticker]
                idx_loc = df_ticker.index.get_loc(date)

                exit_reason = None

                # 1. trailing stop
                pos.max_price = max(pos.max_price, row["high"])
                new_trail = pos.max_price - ATR_MULT * pos.atr_at_entry
                pos.trail_stop = max(pos.trail_stop, new_trail)
                if row["low"] <= pos.trail_stop:
                    exit_reason = "trailing_stop"

                # 2. death cross: 50d MA < 200d MA
                if exit_reason is None and not pd.isna(row.get("ma50")) and not pd.isna(row.get("ma200")):
                    if row["ma50"] < row["ma200"]:
                        exit_reason = "death_cross"

                # 3. max hold
                if exit_reason is None:
                    hold_days = idx_loc - df_ticker.index.get_loc(pos.entry_date) \
                        if pos.entry_date in df_ticker.index else MAX_HOLD + 1
                    if hold_days >= MAX_HOLD:
                        exit_reason = "max_hold"

                # 4. volume dry-up
                if exit_reason is None and not pd.isna(row.get("vol20")):
                    if row["vol20"] < VOL_DRYUP * pos.entry_volume:
                        exit_reason = "volume_dryup"

                if exit_reason:
                    exit_price = min(row["open"], pos.trail_stop) if exit_reason == "trailing_stop" else row["open"]
                    exit_price = max(exit_price, row["low"])  # can't exit below day low
                    to_close.append((i, exit_price, exit_reason, date))

            # close positions (reverse order to preserve indices)
            for i, exit_px, reason, dt in sorted(to_close, key=lambda x: -x[0]):
                pos = self.positions.pop(i)
                proceeds = pos.shares * exit_px
                commission = proceeds * COMMISSION_BPS / 10_000
                self.cash += proceeds - commission
                ret = (exit_px - pos.entry_price) / pos.entry_price
                self.trades.append({
                    "ticker": pos.ticker,
                    "entry_type": pos.entry_type,
                    "entry_date": str(pos.entry_date.date()) if hasattr(pos.entry_date, 'date') else str(pos.entry_date),
                    "exit_date": str(dt.date()) if hasattr(dt, 'date') else str(dt),
                    "entry_price": round(pos.entry_price, 2),
                    "exit_price": round(exit_px, 2),
                    "return_pct": round(ret * 100, 2),
                    "exit_reason": reason,
                    "sector": pos.sector,
                })

            # ── check entries ────────────────────────────────────
            if len(self.positions) < MAX_POS:
                nav = self.cash + self._position_value(date)
                sector_counts = self._sector_counts()
                sector_exp = self._sector_exposure(nav) if nav > 0 else {}

                candidates = []
                for ticker, df in self.data.items():
                    if date not in df.index:
                        continue
                    # Skip if already holding
                    if any(p.ticker == ticker for p in self.positions):
                        continue
                    row = df.loc[date]
                    if pd.isna(row.get("atr20")) or row["atr20"] <= 0:
                        continue

                    sig = None
                    if check_breakout(row):
                        sig = "breakout"
                    elif check_pullback(row):
                        sig = "pullback"

                    if sig:
                        sector = self.sectors.get(ticker, "Unknown")
                        # sector constraints
                        if sector_counts.get(sector, 0) >= MAX_SECTOR_POS:
                            continue
                        if sector_exp.get(sector, 0) >= MAX_SECTOR_PCT:
                            continue
                        candidates.append((ticker, row, sig, sector))

                # Sort by volume ratio (strongest breakouts first)
                candidates.sort(
                    key=lambda x: x[1]["volume"] / max(x[1]["vol20"], 1) if not pd.isna(x[1]["vol20"]) else 0,
                    reverse=True,
                )

                for ticker, row, sig, sector in candidates:
                    if len(self.positions) >= MAX_POS:
                        break
                    # Position sizing: risk 2% of NAV, stop = 2.5 ATR away
                    risk_dollars = nav * RISK_PCT
                    stop_dist = ATR_MULT * row["atr20"]
                    if stop_dist <= 0:
                        continue
                    shares = int(risk_dollars / stop_dist)
                    if shares <= 0:
                        continue
                    cost = shares * row["close"]
                    commission = cost * COMMISSION_BPS / 10_000
                    if cost + commission > self.cash:
                        continue

                    self.cash -= (cost + commission)
                    pos = Position(
                        ticker=ticker, price=row["close"], date=date,
                        idx=0, volume=row["vol20"] if not pd.isna(row["vol20"]) else row["volume"],
                        shares=shares, entry_type=sig, sector=sector,
                        atr=row["atr20"],
                    )
                    self.positions.append(pos)

            # ── record equity ────────────────────────────────────
            nav = self.cash + self._position_value(date)
            self.equity_curve.append((date, nav))

        # Force-close remaining positions at last date
        if dates:
            last_date = dates[-1]
            for pos in list(self.positions):
                row = self._get_row(pos.ticker, last_date)
                if row is not None:
                    exit_px = row["close"]
                else:
                    exit_px = pos.entry_price
                proceeds = pos.shares * exit_px
                commission = proceeds * COMMISSION_BPS / 10_000
                self.cash += proceeds - commission
                ret = (exit_px - pos.entry_price) / pos.entry_price
                self.trades.append({
                    "ticker": pos.ticker,
                    "entry_type": pos.entry_type,
                    "entry_date": str(pos.entry_date.date()) if hasattr(pos.entry_date, 'date') else str(pos.entry_date),
                    "exit_date": str(last_date.date()) if hasattr(last_date, 'date') else str(last_date),
                    "entry_price": round(pos.entry_price, 2),
                    "exit_price": round(exit_px, 2),
                    "return_pct": round(ret * 100, 2),
                    "exit_reason": "force_close_eow",
                    "sector": pos.sector,
                })
            self.positions.clear()

    def reset(self):
        self.positions.clear()
        self.trades.clear()
        self.equity_curve.clear()
        self.cash = self.initial_capital


# ── 6. Metrics ───────────────────────────────────────────────────────

def calc_metrics(equity_curve: list, trades: list, benchmark: pd.Series,
                 oot_start, oot_end) -> dict:
    if not equity_curve:
        return {}

    eq = pd.DataFrame(equity_curve, columns=["date", "nav"]).set_index("date")
    eq = eq[~eq.index.duplicated(keep="last")]
    rets = eq["nav"].pct_change().dropna()

    if len(rets) < 2:
        return {}

    # Benchmark returns over same period
    bm = benchmark.loc[(benchmark.index >= oot_start) & (benchmark.index <= oot_end)]
    bm_rets = bm.pct_change().dropna()

    trading_days = 252
    total_ret = eq["nav"].iloc[-1] / eq["nav"].iloc[0] - 1
    years = len(rets) / trading_days
    cagr = (1 + total_ret) ** (1 / max(years, 0.01)) - 1

    sharpe = rets.mean() / rets.std() * np.sqrt(trading_days) if rets.std() > 0 else 0
    downside = rets[rets < 0].std()
    sortino = rets.mean() / downside * np.sqrt(trading_days) if downside > 0 else 0

    # Max drawdown
    cummax = eq["nav"].cummax()
    dd = (eq["nav"] - cummax) / cummax
    max_dd = dd.min()

    # Trade stats
    trade_rets = [t["return_pct"] for t in trades]
    winners = [r for r in trade_rets if r > 0]
    losers = [r for r in trade_rets if r <= 0]
    wr = len(winners) / len(trade_rets) if trade_rets else 0
    avg_win = np.mean(winners) if winners else 0
    avg_loss = abs(np.mean(losers)) if losers else 1
    pf = (sum(winners) / abs(sum(losers))) if losers and sum(losers) != 0 else float("inf")

    # Holding period
    hold_days = []
    for t in trades:
        try:
            d1 = pd.Timestamp(t["entry_date"])
            d2 = pd.Timestamp(t["exit_date"])
            hold_days.append((d2 - d1).days)
        except Exception:
            pass

    # Benchmark
    bm_total = bm.iloc[-1] / bm.iloc[0] - 1 if len(bm) > 1 else 0
    bm_cagr = (1 + bm_total) ** (1 / max(years, 0.01)) - 1
    bm_sharpe = bm_rets.mean() / bm_rets.std() * np.sqrt(trading_days) if len(bm_rets) > 1 and bm_rets.std() > 0 else 0

    # Entry type breakdown
    breakout_trades = [t for t in trades if t["entry_type"] == "breakout"]
    pullback_trades = [t for t in trades if t["entry_type"] == "pullback"]

    def _sub_stats(tlist):
        if not tlist:
            return {"count": 0, "wr": 0, "avg_ret": 0, "pf": 0}
        rs = [t["return_pct"] for t in tlist]
        w = [r for r in rs if r > 0]
        l = [r for r in rs if r <= 0]
        return {
            "count": len(tlist),
            "wr": round(len(w) / len(rs), 3),
            "avg_ret": round(np.mean(rs), 2),
            "pf": round(sum(w) / abs(sum(l)), 2) if l and sum(l) != 0 else float("inf"),
        }

    # Exit reason breakdown
    exit_reasons = defaultdict(int)
    for t in trades:
        exit_reasons[t["exit_reason"]] += 1

    return {
        "total_return_pct": round(total_ret * 100, 2),
        "cagr_pct": round(cagr * 100, 2),
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "max_drawdown_pct": round(max_dd * 100, 2),
        "win_rate": round(wr, 3),
        "profit_factor": round(pf, 2),
        "avg_win_pct": round(avg_win, 2),
        "avg_loss_pct": round(avg_loss, 2),
        "total_trades": len(trades),
        "avg_hold_days": round(np.mean(hold_days), 1) if hold_days else 0,
        "median_hold_days": round(np.median(hold_days), 1) if hold_days else 0,
        "breakout_stats": _sub_stats(breakout_trades),
        "pullback_stats": _sub_stats(pullback_trades),
        "exit_reasons": dict(exit_reasons),
        "benchmark_total_ret_pct": round(bm_total * 100, 2),
        "benchmark_cagr_pct": round(bm_cagr * 100, 2),
        "benchmark_sharpe": round(bm_sharpe, 3),
        "excess_return_pct": round((total_ret - bm_total) * 100, 2),
    }


# ── 7. Regime analysis ──────────────────────────────────────────────

def regime_analysis(equity_curve: list, benchmark: pd.Series) -> dict:
    """Classify periods by benchmark regime and compute per-regime metrics."""
    if not equity_curve:
        return {}
    eq = pd.DataFrame(equity_curve, columns=["date", "nav"]).set_index("date")
    eq = eq[~eq.index.duplicated(keep="last")]

    bm_aligned = benchmark.reindex(eq.index).ffill()
    bm_ret_60d = bm_aligned.pct_change(60)

    regimes = {}
    for label, cond in [
        ("bull", bm_ret_60d > 0.05),
        ("bear", bm_ret_60d < -0.05),
        ("flat", (bm_ret_60d >= -0.05) & (bm_ret_60d <= 0.05)),
    ]:
        mask = cond.reindex(eq.index).fillna(False)
        sub_rets = eq["nav"].pct_change().loc[mask].dropna()
        if len(sub_rets) < 5:
            regimes[label] = {"days": int(mask.sum()), "sharpe": 0, "total_ret_pct": 0}
            continue
        sharpe = sub_rets.mean() / sub_rets.std() * np.sqrt(252) if sub_rets.std() > 0 else 0
        total = (1 + sub_rets).prod() - 1
        regimes[label] = {
            "days": int(mask.sum()),
            "sharpe": round(sharpe, 3),
            "total_ret_pct": round(total * 100, 2),
        }
    return regimes


# ── 8. Walk-forward orchestration ───────────────────────────────────

def walk_forward_backtest(data: dict, sectors: dict, benchmark: pd.Series):
    """Sliding 2-yr train → 1-yr OOT walk-forward."""
    # Get common date range
    all_dates = set()
    for df in data.values():
        all_dates.update(df.index)
    all_dates = sorted(all_dates)

    if len(all_dates) < TRAIN_DAYS + OOT_DAYS:
        log.error("Not enough data for walk-forward")
        return None

    # Define OOT windows
    windows = []
    start_idx = TRAIN_DAYS
    while start_idx + OOT_DAYS <= len(all_dates):
        oot_start = all_dates[start_idx]
        oot_end = all_dates[min(start_idx + OOT_DAYS - 1, len(all_dates) - 1)]
        train_start = all_dates[start_idx - TRAIN_DAYS]
        windows.append({
            "train_start": train_start,
            "oot_start": oot_start,
            "oot_end": oot_end,
        })
        start_idx += OOT_DAYS  # slide by 1 year

    log.info(f"Walk-forward: {len(windows)} OOT windows")

    all_trades = []
    all_equity = []
    per_window_metrics = []

    for wi, w in enumerate(windows):
        log.info(f"  Window {wi+1}/{len(windows)}: OOT {w['oot_start'].date()} → {w['oot_end'].date()}")

        bt = Backtester(data, sectors, benchmark)
        bt.run(w["oot_start"], w["oot_end"])

        metrics = calc_metrics(bt.equity_curve, bt.trades, benchmark,
                               w["oot_start"], w["oot_end"])
        metrics["window"] = f"{w['oot_start'].date()} → {w['oot_end'].date()}"
        per_window_metrics.append(metrics)

        all_trades.extend(bt.trades)

        # Chain equity curves (scale to continuation)
        if all_equity:
            last_nav = all_equity[-1][1]
            if bt.equity_curve:
                scale = last_nav / bt.equity_curve[0][1]
                for d, v in bt.equity_curve:
                    all_equity.append((d, v * scale))
        else:
            all_equity.extend(bt.equity_curve)

    # Overall metrics
    overall = calc_metrics(all_equity, all_trades, benchmark,
                           all_dates[TRAIN_DAYS], all_dates[-1])
    overall["n_oot_windows"] = len(windows)

    # Regime analysis
    regimes = regime_analysis(all_equity, benchmark)

    return {
        "overall": overall,
        "per_window": per_window_metrics,
        "regimes": regimes,
        "trades": all_trades,
        "equity_curve": [(str(d), v) for d, v in all_equity],
    }


# ── 9. Main ─────────────────────────────────────────────────────────

def main():
    log.info("=" * 60)
    log.info("Mid-Cap Breakout / Mean-Reversion Hybrid Backtester")
    log.info("=" * 60)

    # 1. Universe
    sp400 = fetch_sp400_tickers()
    tickers = sp400["ticker"].tolist()
    sectors = dict(zip(sp400["ticker"], sp400["sector"]))
    log.info(f"Universe: {len(tickers)} tickers")

    # 2. Data
    data = download_data(tickers, START, END)
    if not data:
        log.error("No data downloaded!")
        return

    # Filter tickers that have enough history
    good_data = {}
    for t, df in data.items():
        if len(df) >= 252:  # at least 1 year
            good_data[t] = compute_features(df)
    data = good_data
    log.info(f"Tickers with sufficient data: {len(data)}")

    # 3. Benchmark
    benchmark = download_benchmark(START, END)
    log.info(f"Benchmark (MDY): {len(benchmark)} days")

    # 4. Walk-forward backtest
    results = walk_forward_backtest(data, sectors, benchmark)
    if results is None:
        log.error("Backtest failed!")
        return

    # 5. Print summary
    ov = results["overall"]
    print("\n" + "=" * 60)
    print("OVERALL RESULTS (Walk-Forward OOT)")
    print("=" * 60)
    print(f"  CAGR           : {ov.get('cagr_pct', 0):>8.2f} %")
    print(f"  Total Return   : {ov.get('total_return_pct', 0):>8.2f} %")
    print(f"  Sharpe         : {ov.get('sharpe', 0):>8.3f}")
    print(f"  Sortino        : {ov.get('sortino', 0):>8.3f}")
    print(f"  Max Drawdown   : {ov.get('max_drawdown_pct', 0):>8.2f} %")
    print(f"  Win Rate       : {ov.get('win_rate', 0):>8.1%}")
    print(f"  Profit Factor  : {ov.get('profit_factor', 0):>8.2f}")
    print(f"  Avg Win        : {ov.get('avg_win_pct', 0):>8.2f} %")
    print(f"  Avg Loss       : {ov.get('avg_loss_pct', 0):>8.2f} %")
    print(f"  Total Trades   : {ov.get('total_trades', 0):>8d}")
    print(f"  Avg Hold (days): {ov.get('avg_hold_days', 0):>8.1f}")
    print(f"  OOT Windows    : {ov.get('n_oot_windows', 0):>8d}")

    print(f"\n  MDY CAGR       : {ov.get('benchmark_cagr_pct', 0):>8.2f} %")
    print(f"  MDY Sharpe     : {ov.get('benchmark_sharpe', 0):>8.3f}")
    print(f"  Excess Return  : {ov.get('excess_return_pct', 0):>8.2f} %")

    print("\n── Entry Type Comparison ──")
    for etype in ["breakout_stats", "pullback_stats"]:
        s = ov.get(etype, {})
        label = etype.replace("_stats", "").upper()
        print(f"  {label:12s}:  N={s.get('count',0):4d}  WR={s.get('wr',0):.1%}  "
              f"AvgRet={s.get('avg_ret',0):+.2f}%  PF={s.get('pf',0):.2f}")

    print("\n── Exit Reasons ──")
    for reason, count in sorted(ov.get("exit_reasons", {}).items(), key=lambda x: -x[1]):
        print(f"  {reason:20s}: {count:5d}")

    print("\n── Regime Analysis ──")
    for regime, stats in results.get("regimes", {}).items():
        print(f"  {regime:5s}: {stats['days']:4d} days  Sharpe={stats['sharpe']:+.3f}  "
              f"TotRet={stats['total_ret_pct']:+.2f}%")

    print("\n── Per-Window OOT Results ──")
    for wm in results.get("per_window", []):
        w = wm.get("window", "?")
        print(f"  {w}  Sharpe={wm.get('sharpe',0):+.3f}  "
              f"Ret={wm.get('total_return_pct',0):+.2f}%  "
              f"Trades={wm.get('total_trades',0)}  WR={wm.get('win_rate',0):.1%}")

    # 6. Save results
    # Remove trades from summary (save separately)
    trades = results.pop("trades", [])
    equity = results.pop("equity_curve", [])

    summary_path = OUT / "backtest_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    log.info(f"Summary saved → {summary_path}")

    trades_path = OUT / "trades.json"
    with open(trades_path, "w") as f:
        json.dump(trades, f, indent=2, default=str)
    log.info(f"Trades saved → {trades_path}")

    eq_path = OUT / "equity_curve.csv"
    pd.DataFrame(equity, columns=["date", "nav"]).to_csv(eq_path, index=False)
    log.info(f"Equity curve saved → {eq_path}")

    print(f"\nAll outputs saved to {OUT}/")
    print("Done.")


if __name__ == "__main__":
    main()
