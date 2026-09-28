#!/usr/bin/env python3
"""BTC funding-premium monitor (zero-cost follow-up to crypto_funding_v1 closure).

crypto_funding_v1 (RUN_HISTORY 2026-06-12) found BTC delta-neutral carry passes
every house gate but the premium compressed below T-bills (0.9% ann 2026 YTD).
Closure condition: revisit ONLY if BTC funding sustainably exceeds T-bill + 5%
on a sustained basis. This script checks that condition monthly.

Data: data.binance.vision public archive (NOT geo-blocked from Jupiter, unlike
the REST API). Pulls the last full month of BTCUSDT funding rates.
Threshold: 9.0% annualized (proxy for T-bill ~4% + 5%).
On trigger: injects an alert into the orchestrator session via autonomy_inject.sh.
Always appends one summary line to logs/btc_funding_monitor.log.
"""
import csv
import datetime as dt
import io
import subprocess
import sys
import urllib.request
import zipfile

THRESHOLD_ANN = 0.09  # T-bill (~4%) + 5%
LOG = "/home/jupiter/Lvl3Quant/logs/btc_funding_monitor.log"
INJECT = "/home/jupiter/Lvl3Quant/scripts/autonomy_inject.sh"


def last_full_month():
    today = dt.date.today().replace(day=1)
    prev = today - dt.timedelta(days=1)
    return prev.strftime("%Y-%m")


def fetch_month(ym: str):
    url = (f"https://data.binance.vision/data/futures/um/monthly/fundingRate/"
           f"BTCUSDT/BTCUSDT-fundingRate-{ym}.zip")
    with urllib.request.urlopen(url, timeout=60) as r:
        zf = zipfile.ZipFile(io.BytesIO(r.read()))
    name = zf.namelist()[0]
    rows = list(csv.reader(io.TextIOWrapper(zf.open(name))))
    # header optional; fundingRate is 3rd col (calc_time, funding_interval_hours?, ...)
    rates = []
    for row in rows:
        try:
            rates.append(float(row[-1]))
        except (ValueError, IndexError):
            continue  # header or malformed
    return rates


def main():
    ym = last_full_month()
    try:
        rates = fetch_month(ym)
    except Exception as e:  # noqa: BLE001
        line = f"{dt.datetime.now().isoformat()} ERROR fetching {ym}: {e}\n"
        open(LOG, "a").write(line)
        sys.exit(0)  # silent on fetch errors; next month retries
    if not rates:
        open(LOG, "a").write(f"{dt.datetime.now().isoformat()} {ym}: no rows\n")
        sys.exit(0)
    # funding is per-interval (8h => 3/day => 1095/yr)
    intervals_per_year = 365 * 24 / 8
    ann = sum(rates) / len(rates) * intervals_per_year
    line = (f"{dt.datetime.now().isoformat()} {ym}: n={len(rates)} "
            f"ann_funding={ann:.4f} threshold={THRESHOLD_ANN}\n")
    open(LOG, "a").write(line)
    if ann > THRESHOLD_ANN:
        msg = (f"BTC_FUNDING_ALERT: annualized BTC perp funding for {ym} = "
               f"{ann:.1%} > {THRESHOLD_ANN:.0%} revisit threshold from "
               f"crypto_funding_v1 closure (RUN_HISTORY 2026-06-12). Premium may "
               f"be back. Re-evaluate BTC delta-neutral carry INCLUDING US-venue "
               f"accessibility before any re-run; check 3-month sustainability "
               f"(closure requires sustained, not one month).")
        subprocess.run([INJECT, msg], check=False)


if __name__ == "__main__":
    main()
