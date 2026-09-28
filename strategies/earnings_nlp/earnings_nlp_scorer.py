#!/usr/bin/env python3
"""
Earnings NLP Scorer
====================
Rule-based NLP analysis of earnings transcripts/filings.
Uses Loughran-McDonald financial sentiment dictionary approach
(no LLM API calls — pure keyword/pattern analysis).

Produces structured scores:
  - Management tone (-1 to +1)
  - Guidance direction (raised/maintained/lowered)
  - Forward-looking confidence (-1 to +1)
  - Quantitative strength (-1 to +1)
  - Composite signal (-1 to +1)

Usage:
    python earnings_nlp_scorer.py                         # Score all transcripts
    python earnings_nlp_scorer.py --ticker AAPL           # Score single ticker
    python earnings_nlp_scorer.py --file path/to/file.json  # Score specific file
"""

import argparse
import json
import logging
import os
import re
from datetime import datetime
from pathlib import Path

# Paths
BASE_DIR = Path("/home/jupiter/Lvl3Quant")
TRANSCRIPTS_DIR = BASE_DIR / "data" / "earnings_transcripts"
SCORES_DIR = BASE_DIR / "data" / "earnings_transcripts" / "scores"
LOG_DIR = BASE_DIR / "logs" / "earnings_nlp"

SCORES_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "nlp_scorer.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

# ============================================================================
# Loughran-McDonald inspired financial sentiment dictionaries
# These are curated for earnings context specifically
# ============================================================================

POSITIVE_WORDS = {
    # Strong positive
    "record", "exceeded", "surpassed", "outperformed", "outstanding", "exceptional",
    "remarkable", "unprecedented", "transformative", "breakthrough", "milestone",
    # Growth
    "growth", "grew", "growing", "accelerated", "accelerating", "momentum",
    "expansion", "expanded", "expanding", "increased", "increasing",
    # Financial strength
    "profitable", "profitability", "strong", "strength", "robust", "solid",
    "healthy", "resilient", "durable", "sustainable", "efficient", "efficiency",
    # Guidance positive
    "raised", "raising", "upgraded", "upside", "above", "higher",
    "beat", "beats", "beating", "topped", "topping",
    # Strategic
    "innovative", "innovation", "opportunity", "opportunities", "optimistic",
    "confident", "confidence", "encouraged", "encouraging", "pleased",
    "proud", "excited", "thrilled", "delighted", "impressed",
    # Market position
    "leader", "leading", "dominant", "share gains", "market share",
    "competitive advantage", "best-in-class",
    # Shareholder
    "dividend", "buyback", "repurchase", "returned", "returning",
}

NEGATIVE_WORDS = {
    # Weakness
    "declined", "declining", "decline", "decreased", "decreasing", "decrease",
    "weakened", "weakening", "weakness", "deteriorated", "deteriorating",
    "contraction", "contracted", "contracting", "slowed", "slowing", "slowdown",
    # Risk/uncertainty
    "challenging", "challenges", "headwinds", "headwind", "uncertain",
    "uncertainty", "volatile", "volatility", "risk", "risks", "difficult",
    "difficulty", "pressure", "pressured", "pressures", "concerned", "concern",
    "cautious", "caution", "cautionary",
    # Financial distress
    "loss", "losses", "impairment", "impaired", "write-down", "writedown",
    "restructuring", "layoff", "layoffs", "reduction", "downsizing",
    "underperformed", "underperformance", "missed", "miss", "below",
    "shortfall", "fell short", "disappointing", "disappointed", "disappointment",
    # Guidance negative
    "lowered", "lowering", "reduced", "reducing", "cut", "cutting",
    "downgraded", "downside", "revised down", "lower than expected",
    # Macro concerns
    "recession", "recessionary", "inflation", "inflationary",
    "geopolitical", "tariff", "tariffs", "supply chain disruption",
}

GUIDANCE_RAISED_PATTERNS = [
    r"(?i)rais\w+\s+(?:our\s+)?(?:full[- ]year|annual|quarterly|q[1-4])?\s*(?:guidance|outlook|forecast|expectations?)",
    r"(?i)(?:guidance|outlook|forecast)\s+(?:was\s+)?(?:raised|increased|upgraded|improved|revised\s+(?:up|upward|higher))",
    r"(?i)increas\w+\s+(?:our\s+)?(?:guidance|outlook|forecast|expectations?)",
    r"(?i)(?:now\s+)?expect(?:s|ing)?\s+(?:revenue|earnings|eps|income)\s+(?:to\s+be\s+)?(?:above|higher|greater)",
    r"(?i)upward\s+revision",
]

GUIDANCE_LOWERED_PATTERNS = [
    r"(?i)lower\w+\s+(?:our\s+)?(?:full[- ]year|annual|quarterly|q[1-4])?\s*(?:guidance|outlook|forecast|expectations?)",
    r"(?i)(?:guidance|outlook|forecast)\s+(?:was\s+)?(?:lowered|decreased|cut|reduced|revised\s+(?:down|downward|lower))",
    r"(?i)reduc\w+\s+(?:our\s+)?(?:guidance|outlook|forecast|expectations?)",
    r"(?i)(?:now\s+)?expect(?:s|ing)?\s+(?:revenue|earnings|eps|income)\s+(?:to\s+be\s+)?(?:below|lower|less)",
    r"(?i)downward\s+revision",
]

GUIDANCE_MAINTAINED_PATTERNS = [
    r"(?i)(?:reaffirm|reiterat|maintain|confirm)\w*\s+(?:our\s+)?(?:guidance|outlook|forecast|expectations?)",
    r"(?i)(?:guidance|outlook|forecast)\s+(?:remains?|unchanged|intact|on\s+track)",
]

FORWARD_LOOKING_POSITIVE = [
    r"(?i)(?:well[- ])?position\w+\s+(?:for|to)\s+(?:growth|success|deliver|capitalize)",
    r"(?i)(?:confident|optimistic|excited)\s+(?:about|in|for)\s+(?:the\s+)?(?:future|remainder|second half|next)",
    r"(?i)(?:pipeline|backlog)\s+(?:remains?\s+)?(?:strong|robust|healthy|record)",
    r"(?i)(?:expect|anticipate|project)\s+(?:continued|strong|further|sustained)\s+(?:growth|momentum|improvement)",
    r"(?i)long[- ]term\s+(?:growth|value|opportunity|potential)",
    r"(?i)(?:secular|structural)\s+(?:growth|tailwind|trend)",
]

FORWARD_LOOKING_NEGATIVE = [
    r"(?i)(?:expect|anticipate|project)\s+(?:headwinds?|challenges?|pressure|softness|weakness)",
    r"(?i)(?:uncertain|unclear|murky)\s+(?:outlook|environment|macro|landscape)",
    r"(?i)(?:near[- ]term|short[- ]term)\s+(?:headwinds?|challenges?|pressure|uncertainty)",
    r"(?i)(?:visibility|demand)\s+(?:remains?\s+)?(?:limited|low|poor|unclear)",
]

# Quantitative patterns - detect percentage changes mentioned
PCT_INCREASE_PATTERN = r"(?i)(?:increased?|grew|up|rose|gained|improved)\s+(?:by\s+)?(\d+(?:\.\d+)?)\s*(?:%|percent|basis\s+points)"
PCT_DECREASE_PATTERN = r"(?i)(?:decreased?|declined?|down|fell|dropped|lost)\s+(?:by\s+)?(\d+(?:\.\d+)?)\s*(?:%|percent|basis\s+points)"
RECORD_PATTERN = r"(?i)record\s+(?:revenue|earnings|income|profit|margin|cash\s+flow|quarter|year|results)"
BEAT_PATTERN = r"(?i)(?:beat|exceeded|surpassed|topped)\s+(?:consensus|estimates?|expectations?|street|analysts?)"
MISS_PATTERN = r"(?i)(?:missed|fell\s+short\s+of|below|under)\s+(?:consensus|estimates?|expectations?|street|analysts?)"


def tokenize(text: str) -> list:
    """Simple word tokenization."""
    return re.findall(r'\b[a-zA-Z]+(?:-[a-zA-Z]+)*\b', text.lower())


def score_management_tone(text: str) -> dict:
    """Score management tone based on positive/negative word ratios."""
    words = tokenize(text)
    total_words = len(words) if words else 1

    pos_count = sum(1 for w in words if w in POSITIVE_WORDS)
    neg_count = sum(1 for w in words if w in NEGATIVE_WORDS)

    # Also check multi-word phrases
    text_lower = text.lower()
    for phrase in POSITIVE_WORDS:
        if " " in phrase and phrase in text_lower:
            pos_count += text_lower.count(phrase)
    for phrase in NEGATIVE_WORDS:
        if " " in phrase and phrase in text_lower:
            neg_count += text_lower.count(phrase)

    total_sentiment = pos_count + neg_count
    if total_sentiment == 0:
        score = 0.0
    else:
        # Score from -1 to +1
        score = (pos_count - neg_count) / total_sentiment

    # Normalize intensity by density (sentiment words per 1000 words)
    density = (total_sentiment / total_words) * 1000
    # Higher density = more confident score; low density = dampen toward 0
    confidence_factor = min(density / 20.0, 1.0)  # Saturates at ~20 sentiment words per 1000
    adjusted_score = score * confidence_factor

    return {
        "score": round(adjusted_score, 4),
        "raw_score": round(score, 4),
        "positive_count": pos_count,
        "negative_count": neg_count,
        "density_per_1k": round(density, 2),
        "confidence_factor": round(confidence_factor, 4),
    }


def score_guidance_direction(text: str) -> dict:
    """Detect guidance direction changes."""
    raised_matches = sum(len(re.findall(p, text)) for p in GUIDANCE_RAISED_PATTERNS)
    lowered_matches = sum(len(re.findall(p, text)) for p in GUIDANCE_LOWERED_PATTERNS)
    maintained_matches = sum(len(re.findall(p, text)) for p in GUIDANCE_MAINTAINED_PATTERNS)

    total = raised_matches + lowered_matches + maintained_matches
    if total == 0:
        direction = "not_mentioned"
        score = 0.0
    elif raised_matches > lowered_matches and raised_matches > maintained_matches:
        direction = "raised"
        score = min(raised_matches / max(total, 1), 1.0)
    elif lowered_matches > raised_matches and lowered_matches > maintained_matches:
        direction = "lowered"
        score = -min(lowered_matches / max(total, 1), 1.0)
    else:
        direction = "maintained"
        score = 0.1  # Slight positive — maintaining guidance is mildly bullish

    return {
        "direction": direction,
        "score": round(score, 4),
        "raised_mentions": raised_matches,
        "lowered_mentions": lowered_matches,
        "maintained_mentions": maintained_matches,
    }


def score_surprise_indicators(text: str) -> dict:
    """Detect beat/miss indicators."""
    beat_matches = len(re.findall(BEAT_PATTERN, text))
    miss_matches = len(re.findall(MISS_PATTERN, text))
    record_matches = len(re.findall(RECORD_PATTERN, text))

    total = beat_matches + miss_matches + record_matches
    if total == 0:
        indicator = "neutral"
        score = 0.0
    elif (beat_matches + record_matches) > miss_matches:
        indicator = "beat"
        score = min((beat_matches + record_matches - miss_matches) / max(total, 1), 1.0)
    else:
        indicator = "miss"
        score = -min((miss_matches - beat_matches - record_matches) / max(total, 1), 1.0)

    return {
        "indicator": indicator,
        "score": round(score, 4),
        "beat_mentions": beat_matches,
        "miss_mentions": miss_matches,
        "record_mentions": record_matches,
    }


def score_forward_looking(text: str) -> dict:
    """Score forward-looking language confidence."""
    pos_matches = sum(len(re.findall(p, text)) for p in FORWARD_LOOKING_POSITIVE)
    neg_matches = sum(len(re.findall(p, text)) for p in FORWARD_LOOKING_NEGATIVE)

    total = pos_matches + neg_matches
    if total == 0:
        score = 0.0
    else:
        score = (pos_matches - neg_matches) / total

    return {
        "score": round(score, 4),
        "positive_forward": pos_matches,
        "negative_forward": neg_matches,
    }


def score_quantitative_strength(text: str) -> dict:
    """Analyze quantitative mentions (percentage increases/decreases)."""
    increases = re.findall(PCT_INCREASE_PATTERN, text)
    decreases = re.findall(PCT_DECREASE_PATTERN, text)

    inc_values = [float(v) for v in increases]
    dec_values = [float(v) for v in decreases]

    avg_increase = sum(inc_values) / len(inc_values) if inc_values else 0
    avg_decrease = sum(dec_values) / len(dec_values) if dec_values else 0

    # Score based on balance and magnitude of quantitative mentions
    n_inc = len(inc_values)
    n_dec = len(dec_values)
    total = n_inc + n_dec

    if total == 0:
        score = 0.0
    else:
        # Direction score
        direction = (n_inc - n_dec) / total
        # Magnitude factor: bigger increases = stronger signal
        magnitude = 0.0
        if n_inc > 0 and n_dec > 0:
            magnitude = (avg_increase - avg_decrease) / max(avg_increase, avg_decrease, 1)
        elif n_inc > 0:
            magnitude = min(avg_increase / 30.0, 1.0)  # 30% increase = max
        elif n_dec > 0:
            magnitude = -min(avg_decrease / 30.0, 1.0)
        score = 0.6 * direction + 0.4 * magnitude

    return {
        "score": round(max(-1, min(1, score)), 4),
        "increases_mentioned": n_inc,
        "decreases_mentioned": n_dec,
        "avg_increase_pct": round(avg_increase, 2),
        "avg_decrease_pct": round(avg_decrease, 2),
    }


def compute_composite_score(scores: dict) -> float:
    """Compute weighted composite score from all sub-scores."""
    weights = {
        "management_tone": 0.25,
        "guidance_direction": 0.30,
        "surprise_indicators": 0.20,
        "forward_looking": 0.15,
        "quantitative_strength": 0.10,
    }

    composite = 0.0
    for key, weight in weights.items():
        sub_score = scores.get(key, {}).get("score", 0.0)
        composite += sub_score * weight

    return round(max(-1, min(1, composite)), 4)


def score_transcript(transcript_data: dict) -> dict:
    """Score a full transcript/filing."""
    ticker = transcript_data.get("ticker", "UNKNOWN")
    filing_date = transcript_data.get("filing_date", "")

    # Get text content
    content = transcript_data.get("content", {})
    raw_text = content.get("raw_text", "")

    if not raw_text or len(raw_text) < 100:
        log.warning(f"{ticker}: Insufficient text content ({len(raw_text)} chars)")
        return {
            "ticker": ticker,
            "filing_date": filing_date,
            "scored_at": datetime.now().isoformat(),
            "status": "insufficient_data",
            "text_length": len(raw_text),
            "composite_score": 0.0,
            "scores": {},
        }

    # Run all scorers
    scores = {
        "management_tone": score_management_tone(raw_text),
        "guidance_direction": score_guidance_direction(raw_text),
        "surprise_indicators": score_surprise_indicators(raw_text),
        "forward_looking": score_forward_looking(raw_text),
        "quantitative_strength": score_quantitative_strength(raw_text),
    }

    composite = compute_composite_score(scores)

    result = {
        "ticker": ticker,
        "filing_date": filing_date,
        "filing_type": transcript_data.get("filing_type", ""),
        "scored_at": datetime.now().isoformat(),
        "status": "scored",
        "text_length": len(raw_text),
        "composite_score": composite,
        "composite_label": "bullish" if composite > 0.15 else ("bearish" if composite < -0.15 else "neutral"),
        "scores": scores,
    }

    log.info(
        f"{ticker}: Composite={composite:.3f} ({result['composite_label']}) | "
        f"Tone={scores['management_tone']['score']:.3f} | "
        f"Guidance={scores['guidance_direction']['direction']} | "
        f"Surprise={scores['surprise_indicators']['indicator']}"
    )

    return result


def score_all_transcripts(ticker_filter: str = None) -> list:
    """Score all available transcripts."""
    results = []

    # Find all transcript files
    transcript_files = sorted(TRANSCRIPTS_DIR.glob("*.json"))
    transcript_files = [f for f in transcript_files if not f.name.startswith("_")]

    if ticker_filter:
        transcript_files = [f for f in transcript_files if f.name.startswith(ticker_filter)]

    log.info(f"Found {len(transcript_files)} transcript files to score")

    for filepath in transcript_files:
        try:
            with open(filepath) as f:
                data = json.load(f)

            score_result = score_transcript(data)
            results.append(score_result)

            # Save individual score
            score_filename = f"{score_result['ticker']}_{score_result['filing_date']}_score.json"
            with open(SCORES_DIR / score_filename, "w") as f:
                json.dump(score_result, f, indent=2)

        except Exception as e:
            log.error(f"Error scoring {filepath.name}: {e}")

    # Save summary of all scores
    summary = {
        "scored_at": datetime.now().isoformat(),
        "total_scored": len(results),
        "scores": sorted(results, key=lambda x: x.get("composite_score", 0), reverse=True),
    }
    with open(SCORES_DIR / "_scores_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    return results


def main():
    parser = argparse.ArgumentParser(description="Score earnings transcripts with NLP")
    parser.add_argument("--ticker", type=str, help="Score only this ticker")
    parser.add_argument("--file", type=str, help="Score a specific transcript file")
    args = parser.parse_args()

    log.info("=" * 60)
    log.info("Earnings NLP Scorer Starting")

    if args.file:
        with open(args.file) as f:
            data = json.load(f)
        result = score_transcript(data)
        print(json.dumps(result, indent=2))
        return [result]

    results = score_all_transcripts(ticker_filter=args.ticker)

    if results:
        log.info("=" * 60)
        log.info("Score Summary:")
        for r in sorted(results, key=lambda x: x.get("composite_score", 0), reverse=True):
            log.info(
                f"  {r['ticker']:6s} | {r['composite_score']:+.3f} ({r.get('composite_label', 'n/a'):8s}) | "
                f"Filed: {r.get('filing_date', 'n/a')}"
            )
    else:
        log.info("No transcripts found to score. Run earnings_transcript_fetcher.py first.")

    return results


if __name__ == "__main__":
    main()
