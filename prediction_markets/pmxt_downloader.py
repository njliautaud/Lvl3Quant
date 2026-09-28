#!/usr/bin/env python3
"""
pmxt.dev Data Downloader — Downloads hourly Parquet snapshots of prediction market data.

The pmxt.dev archive provides free hourly snapshots of Polymarket (and sometimes Kalshi)
orderbook and trade data in Parquet format.

Archive URL structure:
    https://archive.pmxt.dev/dumps/polymarket_orderbook_{YYYY-MM-DD}T{HH}.parquet

Data goes back to approximately February 2026.

Usage:
    python pmxt_downloader.py                         # Download last 30 days
    python pmxt_downloader.py --days 7                # Download last 7 days
    python pmxt_downloader.py --extract-spx           # Extract SPX bracket markets
    python pmxt_downloader.py --list-only             # Just list available files
    python pmxt_downloader.py --hours 9 16            # Only market hours (9-16 UTC)

Output:
    data/pmxt/polymarket/YYYY-MM-DD/  — Raw Parquet files by date
    data/pmxt/extracted/spx_brackets.parquet — Extracted SPX bracket data
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

try:
    import pyarrow.parquet as pq
    HAS_PYARROW = True
except ImportError:
    HAS_PYARROW = False

logger = logging.getLogger("pmxt_downloader")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)

BASE_URL = "https://archive.pmxt.dev"
DUMPS_PATH = "/dumps"
API_DOWNLOAD_PATH = "/api/download"
DATA_DIR = Path(__file__).parent / "data" / "pmxt"

# Rate limit config
RATE_LIMIT_INITIAL_WAIT = 30   # seconds to wait on first 429
RATE_LIMIT_MAX_WAIT = 300      # max backoff wait
RATE_LIMIT_BACKOFF = 2.0       # exponential backoff multiplier

# Keywords to identify SPX bracket / S&P 500 related markets in Polymarket data
SPX_KEYWORDS = [
    "s&p", "s&p 500", "sp500", "sp 500", "spx", "spy",
    ".inx", "^gspc", "snp500",
]


def list_available_files(max_pages: int = 7) -> list[dict]:
    """Scrape the pmxt.dev archive listing pages for available files.

    The site is a Next.js app. File links are at /dumps/filename.parquet
    but actual downloads go through /api/download/filename.parquet.

    Returns list of dicts: {filename, url, date_str, hour, date_utc}
    """
    import re
    all_files = []
    seen = set()

    for page in range(1, max_pages + 1):
        try:
            url = f"{BASE_URL}/data/" if page == 1 else f"{BASE_URL}/data/?page={page}"
            resp = requests.get(url, timeout=30)
            if resp.status_code != 200:
                logger.warning(f"Page {page} returned {resp.status_code}")
                break

            # Find href links to parquet files: /dumps/polymarket_orderbook_...parquet
            pattern = r'href="(/dumps/[^"]*\.parquet)"'
            matches = re.findall(pattern, resp.text)

            if not matches:
                break

            for path in matches:
                filename = path.split("/")[-1]
                if filename in seen:
                    continue
                seen.add(filename)

                date_match = re.search(r'(\d{4}-\d{2}-\d{2})T(\d{1,2})', filename)
                if date_match:
                    date_str = date_match.group(1)
                    hour = int(date_match.group(2))
                    # Use api/download for actual file download
                    download_url = f"{BASE_URL}{API_DOWNLOAD_PATH}/{filename}"
                    all_files.append({
                        "filename": filename,
                        "url": download_url,
                        "date_str": date_str,
                        "hour": hour,
                        "date_utc": f"{date_str}T{hour:02d}:00:00Z",
                    })

            time.sleep(1)  # Be polite between pages

        except Exception as e:
            logger.error(f"Failed to list files from page {page}: {e}")
            break

    logger.info(f"Found {len(all_files)} files across {min(page, max_pages)} pages")
    return all_files


def generate_expected_urls(days: int = 30, hours: tuple = None) -> list[dict]:
    """Generate expected download URLs for the last N days.

    Since we know the naming pattern, we can construct URLs directly
    without scraping the listing page.

    Args:
        days: Number of days to look back
        hours: Tuple of (start_hour, end_hour) in UTC to filter. None = all hours.

    Returns:
        List of dicts with url, filename, date_str, hour
    """
    now = datetime.now(timezone.utc)
    urls = []

    for day_offset in range(days):
        date = now - timedelta(days=day_offset)
        date_str = date.strftime("%Y-%m-%d")

        start_h = hours[0] if hours else 0
        end_h = hours[1] if hours else 23

        for hour in range(start_h, end_h + 1):
            # Don't generate future hours for today
            if day_offset == 0 and hour > now.hour:
                continue

            filename = f"polymarket_orderbook_{date_str}T{hour:02d}.parquet"
            url = f"{BASE_URL}{API_DOWNLOAD_PATH}/{filename}"
            urls.append({
                "filename": filename,
                "url": url,
                "date_str": date_str,
                "hour": hour,
                "date_utc": f"{date_str}T{hour:02d}:00:00Z",
            })

    return urls


def download_file(url: str, dest_path: Path, retries: int = 5,
                  timeout: int = 300) -> bool:
    """Download a single file with retry, rate limit backoff, and validation.

    The pmxt.dev site serves files via /api/download/ which has rate limiting.
    Returns True if downloaded successfully.
    """
    if dest_path.exists():
        # Check if file is a valid Parquet (starts with PAR1 magic bytes)
        try:
            with open(dest_path, "rb") as f:
                magic = f.read(4)
            if magic == b"PAR1" and dest_path.stat().st_size > 1024:
                logger.debug(f"Already downloaded (valid Parquet): {dest_path.name}")
                return True
            else:
                # Invalid file (HTML or truncated), remove and retry
                dest_path.unlink()
        except Exception:
            dest_path.unlink(missing_ok=True)

    dest_path.parent.mkdir(parents=True, exist_ok=True)

    backoff_wait = RATE_LIMIT_INITIAL_WAIT

    for attempt in range(retries):
        try:
            resp = requests.get(url, timeout=timeout, stream=True)

            if resp.status_code == 404:
                logger.debug(f"Not found (404): {url.split('/')[-1]}")
                return False

            if resp.status_code == 429:
                retry_after = int(resp.headers.get("Retry-After", backoff_wait))
                wait = max(retry_after, backoff_wait)
                logger.warning(
                    f"Rate limited (429) on {dest_path.name}. "
                    f"Waiting {wait}s (attempt {attempt+1}/{retries})..."
                )
                time.sleep(wait)
                backoff_wait = min(backoff_wait * RATE_LIMIT_BACKOFF, RATE_LIMIT_MAX_WAIT)
                continue

            if resp.status_code != 200:
                logger.warning(f"HTTP {resp.status_code} for {url.split('/')[-1]}")
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)
                continue

            # Check content type — the site returns HTML for non-file routes
            content_type = resp.headers.get("content-type", "")
            if "text/html" in content_type:
                logger.warning(
                    f"Got HTML instead of Parquet for {dest_path.name}. "
                    f"The download URL may be incorrect or the site requires JS."
                )
                return False

            # Stream download
            total = int(resp.headers.get("content-length", 0))
            downloaded = 0
            with open(dest_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192 * 16):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)

            # Validate: check Parquet magic bytes
            with open(dest_path, "rb") as f:
                magic = f.read(4)
            if magic != b"PAR1":
                logger.warning(
                    f"Downloaded file is not valid Parquet: {dest_path.name} "
                    f"(magic={magic.hex()}, size={downloaded})"
                )
                dest_path.unlink(missing_ok=True)
                if attempt < retries - 1:
                    time.sleep(5)
                continue

            if total > 0 and downloaded < total * 0.95:
                logger.warning(f"Incomplete download: {downloaded}/{total} bytes")
                dest_path.unlink(missing_ok=True)
                continue

            logger.info(f"Downloaded: {dest_path.name} ({downloaded / 1e6:.1f} MB)")
            return True

        except requests.exceptions.Timeout:
            logger.warning(f"Timeout on attempt {attempt+1}/{retries}: {url.split('/')[-1]}")
            if attempt < retries - 1:
                time.sleep(5)
        except requests.exceptions.RequestException as e:
            logger.warning(f"Error on attempt {attempt+1}/{retries}: {e}")
            if attempt < retries - 1:
                time.sleep(2 ** attempt)

    return False


def download_range(
    days: int = 30,
    hours: tuple = None,
    dest_dir: Path = None,
    max_concurrent: int = 1,
    skip_existing: bool = True,
) -> dict:
    """Download Parquet snapshots for the given date range.

    Args:
        days: Number of days to download
        hours: (start_hour, end_hour) in UTC. None = all hours.
        dest_dir: Destination directory. Default: data/pmxt/polymarket/
        max_concurrent: Max concurrent downloads (1 = sequential)
        skip_existing: Skip files that already exist

    Returns:
        Summary dict with counts
    """
    if dest_dir is None:
        dest_dir = DATA_DIR / "polymarket"

    urls = generate_expected_urls(days=days, hours=hours)
    logger.info(f"Generated {len(urls)} expected URLs for {days} days")

    results = {"downloaded": 0, "skipped": 0, "failed": 0, "not_found": 0}

    for i, entry in enumerate(urls):
        date_dir = dest_dir / entry["date_str"]
        dest_path = date_dir / entry["filename"]

        if skip_existing and dest_path.exists() and dest_path.stat().st_size > 1024:
            results["skipped"] += 1
            continue

        success = download_file(entry["url"], dest_path)
        if success:
            results["downloaded"] += 1
        else:
            results["not_found"] += 1

        # Progress logging every 20 files
        if (i + 1) % 20 == 0:
            logger.info(
                f"Progress: {i+1}/{len(urls)} — "
                f"downloaded={results['downloaded']}, "
                f"skipped={results['skipped']}, "
                f"not_found={results['not_found']}"
            )

        # Rate limiting: pmxt.dev has aggressive rate limits
        if success:
            time.sleep(5)  # 5s between successful downloads to avoid 429
        else:
            time.sleep(2)

    logger.info(f"Download complete: {results}")
    return results


def read_parquet_file(file_path: Path) -> "pd.DataFrame | None":
    """Read a single Parquet file into a DataFrame.

    Falls back to JSON metadata if pyarrow is not available.
    """
    if not HAS_PANDAS:
        logger.error("pandas is required. pip install pandas")
        return None

    if not HAS_PYARROW:
        logger.error("pyarrow is required. pip install pyarrow")
        return None

    try:
        df = pd.read_parquet(file_path)
        return df
    except Exception as e:
        logger.error(f"Failed to read {file_path}: {e}")
        return None


def inspect_parquet_schema(file_path: Path) -> dict:
    """Inspect a Parquet file's schema and basic stats without loading full data."""
    if not HAS_PYARROW:
        return {"error": "pyarrow not installed"}

    try:
        pf = pq.ParquetFile(file_path)
        schema = pf.schema_arrow
        metadata = pf.metadata

        return {
            "file": str(file_path),
            "num_rows": metadata.num_rows,
            "num_columns": metadata.num_columns,
            "num_row_groups": metadata.num_row_groups,
            "columns": [
                {"name": schema.field(i).name, "type": str(schema.field(i).type)}
                for i in range(len(schema))
            ],
            "size_bytes": file_path.stat().st_size,
        }
    except Exception as e:
        return {"error": str(e), "file": str(file_path)}


def extract_spx_brackets(
    source_dir: Path = None,
    output_path: Path = None,
    keywords: list = None,
) -> "pd.DataFrame | None":
    """Extract SPX bracket markets from downloaded Parquet files.

    Scans all downloaded Parquet files and filters for SPX-related markets.

    Args:
        source_dir: Directory containing downloaded Parquet files
        output_path: Where to save extracted data
        keywords: List of keywords to match (default: SPX_KEYWORDS)

    Returns:
        DataFrame of SPX bracket data, or None if no data found
    """
    if not HAS_PANDAS or not HAS_PYARROW:
        logger.error("pandas and pyarrow required for extraction")
        return None

    if source_dir is None:
        source_dir = DATA_DIR / "polymarket"
    if output_path is None:
        output_path = DATA_DIR / "extracted" / "spx_brackets.parquet"
    if keywords is None:
        keywords = SPX_KEYWORDS

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Find all parquet files
    parquet_files = sorted(source_dir.rglob("*.parquet"))
    if not parquet_files:
        logger.warning(f"No Parquet files found in {source_dir}")
        return None

    logger.info(f"Scanning {len(parquet_files)} Parquet files for SPX brackets...")

    all_spx = []
    columns_seen = set()

    for i, pf_path in enumerate(parquet_files):
        try:
            df = pd.read_parquet(pf_path)
            columns_seen.update(df.columns.tolist())

            # Try different column names that might contain market description
            text_cols = [c for c in df.columns if any(
                k in c.lower() for k in ["question", "title", "name", "description",
                                          "market", "slug", "ticker", "symbol"]
            )]

            if not text_cols:
                # If we don't know the columns yet, log schema for first file
                if i == 0:
                    logger.info(f"Columns in first file: {df.columns.tolist()[:20]}")
                continue

            # Search across all text columns
            mask = pd.Series(False, index=df.index)
            for col in text_cols:
                if df[col].dtype == object:
                    col_lower = df[col].str.lower().fillna("")
                    for kw in keywords:
                        mask |= col_lower.str.contains(kw, na=False)

            spx_rows = df[mask]
            if len(spx_rows) > 0:
                # Add source file metadata
                spx_rows = spx_rows.copy()
                spx_rows["_source_file"] = pf_path.name
                all_spx.append(spx_rows)
                logger.info(f"  {pf_path.name}: {len(spx_rows)} SPX rows")

        except Exception as e:
            logger.warning(f"Error reading {pf_path.name}: {e}")

        if (i + 1) % 10 == 0:
            logger.info(f"  Scanned {i+1}/{len(parquet_files)} files, "
                        f"found {sum(len(d) for d in all_spx)} SPX rows so far")

    if not all_spx:
        logger.warning("No SPX bracket data found in any files")
        logger.info(f"Columns seen across files: {sorted(columns_seen)[:30]}")
        # Save column info for debugging
        meta_path = output_path.parent / "column_schema.json"
        with open(meta_path, "w") as f:
            json.dump({"columns_seen": sorted(columns_seen)}, f, indent=2)
        logger.info(f"Saved column schema to {meta_path}")
        return None

    result = pd.concat(all_spx, ignore_index=True)
    logger.info(f"Total SPX bracket rows: {len(result)}")

    # Save
    try:
        result.to_parquet(output_path, index=False)
        logger.info(f"Saved SPX brackets to {output_path}")
    except Exception:
        # Fallback to JSON
        json_path = output_path.with_suffix(".json")
        result.to_json(json_path, orient="records", lines=True)
        logger.info(f"Saved SPX brackets to {json_path} (JSON fallback)")

    return result


def get_download_summary(dest_dir: Path = None) -> dict:
    """Summarize what's been downloaded."""
    if dest_dir is None:
        dest_dir = DATA_DIR / "polymarket"

    if not dest_dir.exists():
        return {"status": "no downloads yet", "path": str(dest_dir)}

    files = list(dest_dir.rglob("*.parquet"))
    total_size = sum(f.stat().st_size for f in files)

    # Group by date
    dates = set()
    for f in files:
        parts = f.name.split("_")
        for p in parts:
            if "-" in p and len(p) >= 10:
                dates.add(p[:10])

    return {
        "path": str(dest_dir),
        "total_files": len(files),
        "total_size_gb": round(total_size / 1e9, 2),
        "date_range": sorted(dates) if dates else [],
        "oldest": min(dates) if dates else None,
        "newest": max(dates) if dates else None,
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Download prediction market data from pmxt.dev archive",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--days", type=int, default=30,
                        help="Number of days to download")
    parser.add_argument("--hours", type=int, nargs=2, metavar=("START", "END"),
                        help="UTC hour range to download (e.g., 9 16 for market hours)")
    parser.add_argument("--dest", type=str, default=None,
                        help="Destination directory")
    parser.add_argument("--extract-spx", action="store_true",
                        help="Extract SPX bracket markets after download")
    parser.add_argument("--extract-only", action="store_true",
                        help="Skip download, only extract from existing files")
    parser.add_argument("--inspect", type=str, default=None,
                        help="Inspect a single Parquet file (show schema)")
    parser.add_argument("--list-only", action="store_true",
                        help="Only list available files, don't download")
    parser.add_argument("--summary", action="store_true",
                        help="Show download summary")
    parser.add_argument("--sample", type=int, default=0,
                        help="Download only N sample files (for testing)")

    args = parser.parse_args()

    # Summary mode
    if args.summary:
        summary = get_download_summary()
        print(json.dumps(summary, indent=2))
        return

    # Inspect mode
    if args.inspect:
        schema = inspect_parquet_schema(Path(args.inspect))
        print(json.dumps(schema, indent=2, default=str))
        return

    # List mode
    if args.list_only:
        urls = generate_expected_urls(days=args.days, hours=tuple(args.hours) if args.hours else None)
        print(f"Expected {len(urls)} files for {args.days} days:")
        for u in urls[:20]:
            print(f"  {u['filename']}")
        if len(urls) > 20:
            print(f"  ... and {len(urls) - 20} more")
        return

    # Extract-only mode
    if args.extract_only:
        dest = Path(args.dest) if args.dest else None
        df = extract_spx_brackets(source_dir=dest)
        if df is not None:
            print(f"\nExtracted {len(df)} SPX bracket rows")
            print(f"Columns: {df.columns.tolist()[:15]}")
            if len(df) > 0:
                print(f"\nSample row:\n{df.iloc[0].to_dict()}")
        return

    # Download mode
    dest = Path(args.dest) if args.dest else DATA_DIR / "polymarket"
    hours = tuple(args.hours) if args.hours else None
    days = args.days

    if args.sample > 0:
        # Download just a few files for testing
        urls = generate_expected_urls(days=min(days, 2), hours=hours)
        urls = urls[:args.sample]
        logger.info(f"Sample mode: downloading {len(urls)} files")
        results = {"downloaded": 0, "failed": 0}
        for entry in urls:
            date_dir = dest / entry["date_str"]
            dest_path = date_dir / entry["filename"]
            success = download_file(entry["url"], dest_path)
            if success:
                results["downloaded"] += 1
                # Inspect first successful download
                schema = inspect_parquet_schema(dest_path)
                print(f"\nSchema of {entry['filename']}:")
                print(json.dumps(schema, indent=2, default=str))
            else:
                results["failed"] += 1
        print(f"\nSample results: {results}")
        return

    # Full download
    logger.info(f"Downloading {days} days of data to {dest}")
    results = download_range(days=days, hours=hours, dest_dir=dest)
    print(f"\nDownload results: {json.dumps(results, indent=2)}")

    # Optionally extract SPX data
    if args.extract_spx:
        print("\nExtracting SPX bracket data...")
        df = extract_spx_brackets(source_dir=dest)
        if df is not None:
            print(f"Extracted {len(df)} SPX bracket rows")

    # Final summary
    summary = get_download_summary(dest)
    print(f"\nSummary: {json.dumps(summary, indent=2)}")


if __name__ == "__main__":
    main()
