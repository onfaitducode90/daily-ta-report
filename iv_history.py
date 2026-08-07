#!/usr/bin/env python3
"""
Real ATM-IV history log (G27, 2nd Opus audit).

fetch_real_iv_context (daily_ta_report.py) reads real ATM IV from
whatever local ThinkorSwim chain snapshot happens to be available, but
has never had a real IV-RANK to offer -- calc_iv_rank only ever built a
percentile from realized vol (HV), because no history of actual IV
readings was being kept anywhere. This module closes that gap: every
report run that resolves a real_iv_context appends that day's ATM IV to
a small local CSV log, idempotent per (ticker, date), so that once enough
daily snapshots accumulate a GENUINE IV-rank percentile can be computed
from real implied volatility instead of a realized-vol proxy.

This does not retire the HV-based fallback -- with only a handful of
snapshots logged so far, there isn't remotely enough history to trust a
percentile from real_iv_rank() yet (see MIN_SAMPLES_FOR_RANK), and this
module says so plainly rather than reporting a number built on too little
data. Once FULL_LOOKBACK samples exist for a ticker, real_iv_rank()
becomes a real ~1-year IV-rank read; until then it returns None and
callers must keep using the existing proxy, exactly the same pattern
prediction_log.py/calibrate.py already use for pattern-confidence
calibration.

Writing here must never break report generation: every write is wrapped
and failures are printed as warnings, not raised (same rule as
prediction_log.py).
"""

import csv
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(SCRIPT_DIR, "logs")
IV_HISTORY_PATH = os.path.join(LOG_DIR, "iv_history.csv")

# A percentile from under ~30 points is mostly noise; a genuine "IV rank"
# in the usual sense (a full trailing year) needs ~252 trading days of
# snapshots. Since these snapshots are downloaded manually, reaching 252
# takes roughly a year of daily discipline -- that's the honest cost of
# a REAL IV rank, not a shortcut around it.
MIN_SAMPLES_FOR_RANK = 30
FULL_LOOKBACK = 252

_FIELDNAMES = ["ticker", "date", "iv", "expiration", "dte", "snapshot_date"]


def _read_all():
    if not os.path.exists(IV_HISTORY_PATH):
        return []
    with open(IV_HISTORY_PATH, newline="") as f:
        return list(csv.DictReader(f))


def _write_all(rows):
    """Write via a temp file + atomic os.replace, not a direct truncating
    write -- a 3rd audit verified that writing straight to IV_HISTORY_PATH
    loses the ENTIRE log (not just one row) if interrupted between the
    truncate and the writerows call (5 seeded rows -> 0 after a simulated
    crash mid-write). os.replace is atomic on both POSIX and Windows, so a
    reader always sees either the old complete file or the new complete
    file, never a half-written one."""
    tmp_path = IV_HISTORY_PATH + ".tmp"
    with open(tmp_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp_path, IV_HISTORY_PATH)


def log_iv(ticker, report_date, iv, expiration, dte, snapshot_date):
    """Idempotently record today's real ATM IV reading for this ticker --
    re-running the same ticker/date (e.g. morning then evening the same
    day) overwrites rather than duplicates, same convention as
    prediction_log.log_run. Never raises."""
    try:
        os.makedirs(LOG_DIR, exist_ok=True)
        rows = _read_all()
        key = (ticker, str(report_date))
        rows = [r for r in rows if (r["ticker"], r["date"]) != key]
        rows.append({
            "ticker": ticker, "date": str(report_date), "iv": f"{iv:.6f}",
            "expiration": str(expiration), "dte": dte, "snapshot_date": str(snapshot_date),
        })
        rows.sort(key=lambda r: (r["ticker"], r["date"]))
        _write_all(rows)
    except Exception as e:
        print(f"WARNING: IV history log write failed: {e}")


def real_iv_rank(ticker, current_iv, lookback=FULL_LOOKBACK):
    """Percentile rank of `current_iv` among this ticker's last `lookback`
    logged real-IV readings (current_iv itself included, matching
    calc_iv_rank's convention). Returns (percentile, n_samples) --
    percentile is None if fewer than MIN_SAMPLES_FOR_RANK readings exist,
    so callers never present a rank built on too little history as if it
    were a real one.

    Wrapped end-to-end and skips unparseable rows rather than raising --
    this is called from analyze_ticker on every report run (unlike
    log_iv, it wasn't wrapped before this fix), and the module's own rule
    is that nothing here may break report generation. A single blank/
    corrupt `iv` cell in the CSV (e.g. from a partially-written row) must
    not take the whole report down with it."""
    try:
        rows = [r for r in _read_all() if r.get("ticker") == ticker]
        rows.sort(key=lambda r: r["date"])
        rows = rows[-lookback:]
        values = []
        for r in rows:
            try:
                values.append(float(r["iv"]))
            except (TypeError, ValueError, KeyError):
                continue
        n = len(values)
        if n < MIN_SAMPLES_FOR_RANK:
            return None, n
        from scipy import stats as scipy_stats
        return float(scipy_stats.percentileofscore(values, current_iv)), n
    except Exception as e:
        print(f"WARNING: IV history rank lookup failed: {e}")
        return None, 0
