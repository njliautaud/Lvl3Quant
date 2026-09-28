"""Sub-industry taxonomy for fine-grained sector rotation (HC pending — user 2026-06-08 ~19:38 ET).

User verbatim: "sector rotation even within sectors... for example space physical ai semis
all are within tech but there are tons of sub indistries within subindistries... all
categorized for sector rotation"

Implementation strategy:
1. Start with a curated mapping of tickers -> sub_industry.
2. Picker pivots from GICS-sector buckets to these sub-industry buckets.
3. Same panel/WF/books/costs as sector_picker_v6/v7 — just finer granularity.

This is V1 — hand-curated for the names we actually have in the master_panel.
Future: pull from a paid taxonomy source (e.g., Refinitiv, MSCI) instead.
"""

from __future__ import annotations
from typing import Dict, List


# ---------------------------------------------------------------------------
# Sub-industry buckets within Information Technology / Communication Services
# ---------------------------------------------------------------------------

TECH_SEMICONDUCTORS = [
    # AI/datacenter chip leaders
    "NVDA", "AMD", "INTC", "QCOM", "AVGO", "MRVL",
    # Memory + IDM
    "MU", "WDC", "STX",
    # Equipment
    "AMAT", "LRCX", "KLAC", "ASML", "TER",
    # Specialty / RF / power
    "TXN", "ADI", "ON", "MCHP", "MPWR", "QRVO", "SWKS",
    # Foundry
    "TSM", "UMC", "GFS",
]

TECH_SEMI_AI_FOUNDRY = ["TSM", "ASML"]  # pure-play AI infrastructure leaders

TECH_AI_SOFTWARE = [
    # AI native or LLM-first
    "PLTR", "AI", "SOUN", "BBAI",
    # AI-leveraging incumbents
    "MSFT", "GOOG", "GOOGL", "META", "AMZN", "ORCL", "CRM",
    # AI infra software
    "SNOW", "DDOG", "MDB", "NET",
]

TECH_CLOUD_SAAS = [
    "MSFT", "GOOGL", "AMZN", "CRM", "NOW", "ADBE", "INTU", "WDAY",
    "TEAM", "ZS", "OKTA", "DDOG", "MDB", "NET", "SNOW", "ZM",
]

TECH_CYBERSECURITY = ["CRWD", "PANW", "ZS", "OKTA", "FTNT", "NET", "S", "RBRK"]

TECH_FINTECH = ["V", "MA", "PYPL", "SQ", "AFRM", "SOFI", "COIN", "HOOD", "SEZL"]

TECH_PHYSICAL_AI_ROBOTICS = [
    # Industrial robotics + autonomous
    "ABB", "IRBT", "TER",
    # Self-driving / autonomous platforms
    "TSLA", "MBLY", "INTC",
    # Robotics-adjacent (vision, lidar, sensors)
    "OUST", "INVZ", "LAZR",
    # Drones / UAS
    "AVAV", "RDW", "RCAT",
    # Automation / control
    "ROK", "EMR",
]

TECH_QUANTUM = ["IONQ", "RGTI", "QBTS", "QUBT", "ARQQ"]

# ---------------------------------------------------------------------------
# Aerospace / Space / Defense Tech (some live under Industrials in GICS but
# user thinking calls them "tech sub-industries" — we keep them grouped here)
# ---------------------------------------------------------------------------

SPACE_PURE_PLAY = [
    "RKLB",   # Rocket Lab
    "RDW",    # Redwire (space infra)
    "ASTS",   # AST SpaceMobile
    "PL",     # Planet Labs
    "BKSY",   # BlackSky
    "SPIR",   # Spire Global
    # NOTE: SpaceX IPO ticker (SPCX) not yet listed; leave for post-IPO add
    "MNTS",   # Momentus
    "LUNR",   # Intuitive Machines
    "AMPX",   # Amprius (space-power adjacent)
]

DEFENSE_TECH = [
    "AVAV",   # AeroVironment (drones)
    "KTOS",   # Kratos
    "LDOS",   # Leidos
    "RCAT",   # Red Cat (drones)
    "PLTR",   # defense software
    "LMT", "RTX", "NOC", "GD", "HII",
]

AEROSPACE_INDUSTRIAL = ["BA", "TXT", "HXL", "TDG", "HEI", "TDY"]

# ---------------------------------------------------------------------------
# Energy sub-industries
# ---------------------------------------------------------------------------

ENERGY_OIL_MAJORS = ["XOM", "CVX", "SHEL", "BP", "TTE", "COP", "EOG", "PXD"]
ENERGY_OIL_SERVICES = ["SLB", "HAL", "BKR", "NOV", "FTI"]
ENERGY_NATGAS = ["LNG", "EQT", "AR", "RRC", "CTRA", "EQNR"]
ENERGY_NUCLEAR = ["CCJ", "URA", "URNM", "OKLO", "SMR", "NNE", "BWXT"]
ENERGY_SOLAR = ["FSLR", "ENPH", "SEDG", "RUN", "JKS", "CSIQ"]
ENERGY_HYDROGEN = ["PLUG", "BE", "BLDP"]
ENERGY_POWER_IPP = ["CEG", "VST", "TLN", "NRG", "ETR", "EXC"]

# ---------------------------------------------------------------------------
# Healthcare sub-industries
# ---------------------------------------------------------------------------

HC_BIG_PHARMA = ["JNJ", "PFE", "MRK", "LLY", "ABBV", "BMY", "AZN", "NVS", "ROG.SW"]
HC_BIOTECH_LARGE = ["AMGN", "VRTX", "REGN", "GILD", "BIIB"]
HC_BIOTECH_SMID = ["MRNA", "BNTX", "INCY", "MRTX", "SRPT"]
HC_MEDDEV = ["MDT", "SYK", "ISRG", "BSX", "ABT", "EW", "ZBH"]
HC_INSURANCE = ["UNH", "ELV", "CI", "HUM", "CVS"]
HC_GLP1 = ["LLY", "NVO"]   # diabetes / obesity drug leaders

# ---------------------------------------------------------------------------
# Financials sub-industries
# ---------------------------------------------------------------------------

FIN_BIG_BANKS = ["JPM", "BAC", "WFC", "C", "USB", "PNC"]
FIN_REGIONAL_BANKS = ["TFC", "FITB", "RF", "KEY", "ZION", "CFG"]
FIN_BROKERAGE = ["SCHW", "IBKR", "HOOD"]
FIN_ASSET_MGMT = ["BLK", "BX", "KKR", "APO", "ARES", "BAM"]
FIN_INSURANCE = ["BRK-B", "AIG", "CB", "TRV", "ALL", "PGR"]
FIN_PAYMENTS = ["V", "MA", "PYPL", "FIS", "FISV", "GPN"]
FIN_CRYPTO = ["COIN", "MARA", "RIOT", "MSTR", "HUT", "CLSK"]

# ---------------------------------------------------------------------------
# Consumer sub-industries
# ---------------------------------------------------------------------------

CONS_EV_AUTO = ["TSLA", "RIVN", "LCID", "F", "GM", "STLA", "TM"]
CONS_LUX = ["LVMUY", "BIRK", "RL", "TPR", "CPRI"]
CONS_RESTAURANTS = ["MCD", "SBUX", "CMG", "QSR", "YUM"]
CONS_E_COMMERCE = ["AMZN", "BABA", "SHOP", "MELI", "SE"]

# ---------------------------------------------------------------------------
# REITs
# ---------------------------------------------------------------------------

REITS_DATA_CENTER = ["EQIX", "DLR"]
REITS_INDUSTRIAL = ["PLD", "STAG", "REXR"]
REITS_TOWER = ["AMT", "CCI", "SBAC"]
REITS_RESIDENTIAL = ["AVB", "EQR", "ESS", "MAA"]

# ---------------------------------------------------------------------------
# Materials sub-industries
# ---------------------------------------------------------------------------

MAT_LITHIUM = ["ALB", "SQM", "LAC", "PLL"]
MAT_RARE_EARTH = ["MP", "TMC", "USAR", "LYSCY"]
MAT_COPPER = ["FCX", "SCCO", "TECK", "WPM"]
MAT_GOLD = ["NEM", "GOLD", "AEM", "KGC", "FNV", "WPM"]
MAT_URANIUM_MINERS = ["CCJ", "DNN", "NXE", "UEC", "URG"]

# ---------------------------------------------------------------------------
# Public bucket map (TICKER -> list of sub_industry tags it belongs to).
# A ticker can belong to MULTIPLE sub-industries (e.g. CCJ in nuclear & uranium).
# ---------------------------------------------------------------------------

def build_ticker_to_subindustries() -> Dict[str, List[str]]:
    bucket_pairs = [
        ("semiconductors", TECH_SEMICONDUCTORS),
        ("semi_ai_foundry", TECH_SEMI_AI_FOUNDRY),
        ("ai_software", TECH_AI_SOFTWARE),
        ("cloud_saas", TECH_CLOUD_SAAS),
        ("cybersecurity", TECH_CYBERSECURITY),
        ("fintech", TECH_FINTECH),
        ("physical_ai_robotics", TECH_PHYSICAL_AI_ROBOTICS),
        ("quantum", TECH_QUANTUM),
        ("space", SPACE_PURE_PLAY),
        ("defense_tech", DEFENSE_TECH),
        ("aerospace_industrial", AEROSPACE_INDUSTRIAL),
        ("oil_majors", ENERGY_OIL_MAJORS),
        ("oil_services", ENERGY_OIL_SERVICES),
        ("natgas", ENERGY_NATGAS),
        ("nuclear", ENERGY_NUCLEAR),
        ("solar", ENERGY_SOLAR),
        ("hydrogen", ENERGY_HYDROGEN),
        ("power_ipp", ENERGY_POWER_IPP),
        ("big_pharma", HC_BIG_PHARMA),
        ("biotech_large", HC_BIOTECH_LARGE),
        ("biotech_smid", HC_BIOTECH_SMID),
        ("meddev", HC_MEDDEV),
        ("health_insurance", HC_INSURANCE),
        ("glp1", HC_GLP1),
        ("big_banks", FIN_BIG_BANKS),
        ("regional_banks", FIN_REGIONAL_BANKS),
        ("brokerage", FIN_BROKERAGE),
        ("asset_mgmt", FIN_ASSET_MGMT),
        ("p_c_insurance", FIN_INSURANCE),
        ("payments", FIN_PAYMENTS),
        ("crypto_proxy", FIN_CRYPTO),
        ("ev_auto", CONS_EV_AUTO),
        ("luxury", CONS_LUX),
        ("restaurants", CONS_RESTAURANTS),
        ("e_commerce", CONS_E_COMMERCE),
        ("reits_datacenter", REITS_DATA_CENTER),
        ("reits_industrial", REITS_INDUSTRIAL),
        ("reits_tower", REITS_TOWER),
        ("reits_residential", REITS_RESIDENTIAL),
        ("lithium", MAT_LITHIUM),
        ("rare_earth", MAT_RARE_EARTH),
        ("copper", MAT_COPPER),
        ("gold", MAT_GOLD),
        ("uranium_miners", MAT_URANIUM_MINERS),
    ]
    out: Dict[str, List[str]] = {}
    for tag, tickers in bucket_pairs:
        for t in tickers:
            out.setdefault(t, []).append(tag)
    return out


def get_subindustries_for(ticker: str) -> List[str]:
    return TICKER_TO_SUBINDUSTRIES.get(ticker.upper(), [])


def get_tickers_for(subindustry: str) -> List[str]:
    return [t for t, tags in TICKER_TO_SUBINDUSTRIES.items() if subindustry in tags]


def list_subindustries() -> List[str]:
    seen = set()
    out: List[str] = []
    for tags in TICKER_TO_SUBINDUSTRIES.values():
        for t in tags:
            if t not in seen:
                seen.add(t)
                out.append(t)
    return sorted(out)


TICKER_TO_SUBINDUSTRIES: Dict[str, List[str]] = build_ticker_to_subindustries()


if __name__ == "__main__":
    print(f"Sub-industries defined: {len(list_subindustries())}")
    print(f"Tickers mapped: {len(TICKER_TO_SUBINDUSTRIES)}")
    print()
    print("Sample buckets and counts:")
    for sub in list_subindustries():
        tks = get_tickers_for(sub)
        print(f"  {sub:25s} {len(tks):3d}  {tks[:6]}{'…' if len(tks)>6 else ''}")
