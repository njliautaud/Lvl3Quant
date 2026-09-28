#!/usr/bin/env python3
"""
Stock Universe Definitions
===========================
Hardcoded fallback universes when Wikipedia scraping fails.
Updated periodically. Last update: 2026-07-13.
"""

import pandas as pd


def get_sp500_tickers():
    """Fetch S&P 500 tickers. Try Wikipedia first, fall back to hardcoded."""
    try:
        tables = pd.read_html(
            "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
            attrs={"id": "constituents"},
        )
        df = tables[0]
        tickers = df["Symbol"].str.replace(".", "-", regex=False).tolist()
        if len(tickers) > 400:
            return tickers
    except Exception:
        pass

    # Fallback: top ~200 S&P 500 by market cap (covers ~85% of index weight)
    return _SP500_FALLBACK


def get_nasdaq100_tickers():
    """Fetch NASDAQ-100. Try Wikipedia first, fall back to hardcoded."""
    try:
        tables = pd.read_html("https://en.wikipedia.org/wiki/Nasdaq-100#Components")
        for t in tables:
            for col in ["Ticker", "Symbol"]:
                if col in t.columns:
                    tickers = t[col].str.replace(".", "-", regex=False).tolist()
                    if len(tickers) > 80:
                        return tickers
    except Exception:
        pass

    return _NDX100_FALLBACK


def build_universe():
    """Build deduplicated universe of S&P 500 + NASDAQ-100."""
    sp500 = get_sp500_tickers()
    ndx100 = get_nasdaq100_tickers()
    universe = sorted(set(sp500 + ndx100))
    print(f"[INFO] Universe: {len(sp500)} S&P 500 + {len(ndx100)} NASDAQ-100 = {len(universe)} unique tickers")
    return universe


# ── Hardcoded Fallbacks (top by market cap) ──────────────────────────────────

_NDX100_FALLBACK = [
    "AAPL", "ABNB", "ADBE", "ADI", "ADP", "ADSK", "AEP", "AMAT", "AMD",
    "AMGN", "AMZN", "ANSS", "APP", "ARM", "ASML", "AVGO", "AZN",
    "BIIB", "BKNG", "BKR", "CCEP", "CDNS", "CDW", "CEG", "CHTR",
    "CMCSA", "COST", "CPRT", "CRWD", "CSCO", "CSGP", "CTAS", "CTSH",
    "DASH", "DDOG", "DLTR", "DXCM", "EA", "EXC", "FANG", "FAST",
    "FTNT", "GEHC", "GFS", "GILD", "GOOG", "GOOGL", "HON", "IDXX",
    "ILMN", "INTC", "INTU", "ISRG", "KDP", "KHC", "KLAC", "LIN",
    "LRCX", "LULU", "MAR", "MCHP", "MDB", "MDLZ", "MELI", "META",
    "MNST", "MRVL", "MSFT", "MU", "NFLX", "NVDA", "NXPI", "ODFL",
    "ON", "ORLY", "PANW", "PAYX", "PCAR", "PDD", "PEP", "PLTR",
    "PYPL", "QCOM", "REGN", "ROP", "ROST", "SBUX", "SMCI", "SNPS",
    "TEAM", "TMUS", "TSLA", "TTD", "TTWO", "TXN", "VRSK", "VRTX",
    "WBD", "WDAY", "XEL", "ZS",
]

_SP500_FALLBACK = [
    # Technology
    "AAPL", "MSFT", "NVDA", "AVGO", "META", "GOOGL", "GOOG", "CRM",
    "AMD", "ADBE", "ACN", "CSCO", "ORCL", "INTC", "IBM", "TXN",
    "QCOM", "INTU", "AMAT", "NOW", "MU", "LRCX", "ADI", "SNPS",
    "CDNS", "KLAC", "FTNT", "PANW", "CRWD", "MRVL", "NXPI",
    # Comm Services
    "AMZN", "TSLA", "NFLX", "DIS", "CMCSA", "TMUS", "VZ", "T",
    # Healthcare
    "UNH", "JNJ", "LLY", "ABBV", "MRK", "PFE", "TMO", "ABT",
    "DHR", "BMY", "AMGN", "GILD", "ISRG", "VRTX", "REGN", "MDT",
    "BSX", "SYK", "CI", "ELV", "HCA", "ZTS", "DXCM", "IDXX",
    # Financials
    "BRK-B", "JPM", "V", "MA", "BAC", "WFC", "GS", "MS", "SPGI",
    "BLK", "AXP", "C", "SCHW", "CB", "MMC", "PGR", "AON", "ICE",
    "CME", "MCO", "USB", "TFC", "PNC", "MET", "AIG", "AFL",
    # Consumer
    "WMT", "PG", "KO", "PEP", "COST", "MCD", "NKE", "SBUX",
    "TGT", "LOW", "HD", "TJX", "ROST", "DG", "DLTR", "ORLY",
    "AZO", "EL", "CL", "KMB", "GIS", "SJM", "K", "CPB", "MKC",
    # Industrials
    "GE", "CAT", "HON", "UNP", "UPS", "RTX", "BA", "LMT", "DE",
    "MMM", "GD", "NOC", "ITW", "EMR", "ETN", "PH", "ROK", "CMI",
    "FDX", "CSX", "NSC", "WM", "RSG", "FAST", "PCAR", "ODFL",
    # Energy
    "XOM", "CVX", "COP", "SLB", "EOG", "MPC", "PSX", "VLO",
    "OXY", "DVN", "HES", "FANG", "HAL", "BKR",
    # Utilities
    "NEE", "DUK", "SO", "D", "AEP", "SRE", "EXC", "XEL",
    "WEC", "ES", "ED", "AEE", "CMS", "DTE", "FE", "PPL",
    # Real Estate
    "PLD", "AMT", "CCI", "EQIX", "SPG", "PSA", "O", "WELL",
    "DLR", "AVB", "EQR", "VTR", "ARE", "MAA", "UDR",
    # Materials
    "LIN", "APD", "SHW", "ECL", "FCX", "NEM", "NUE", "DOW",
    "DD", "PPG", "VMC", "MLM", "CF", "MOS", "ALB",
    # Misc large caps
    "BKNG", "ABNB", "UBER", "COIN", "SQ", "SHOP", "NET", "SNOW",
    "DDOG", "MDB", "PLTR", "SOFI", "RIVN", "LCID", "NIO", "LI",
    "GM", "F", "PYPL",
]
