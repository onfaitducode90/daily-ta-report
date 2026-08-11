#!/usr/bin/env python3
"""
Local last-known-good cache for the yfinance-backed data daily_ta_report.py
depends on (daily OHLCV, VIX, earnings/dividend dates).

daily_ta_report.py has no offline story today: every fetch_* function wraps
its yfinance call in try/except and returns None on ANY failure, including
"no internet." With no local persistence anywhere in that path, a report
run with no connection produces a full report shell where every ticker
section is just "No data available -- skipping", which reads exactly like
a legitimate "no signal today" verdict instead of an outage. This module
closes that gap by remembering the last successful read for each piece of
data, so an offline run can still produce a real, honest report -- clearly
labeled as built from cached data -- instead of an empty one.

This module only stores/retrieves; it never calls yfinance itself and has
no knowledge of network state. Callers (daily_ta_report.py) attempt the
live fetch first, write through to this cache on success, and fall back to
load_*() on failure. That keeps this module trivially testable and avoids
a circular import with the fetch_* functions.

Same rules as iv_history.py: writes are atomic (temp file + os.replace) and
every read/write is wrapped so a cache problem can never break report
generation -- a failure here degrades to "no cached data", never a raised
exception.
"""

import json
import os

import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(SCRIPT_DIR, "logs", "cache")

_HISTORY_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]


def _history_path(ticker):
    return os.path.join(CACHE_DIR, f"history_{ticker}.csv")


def _vix_path():
    return os.path.join(CACHE_DIR, "vix.json")


def _calendar_path():
    return os.path.join(CACHE_DIR, "calendar.json")


def _atomic_write_text(path, text):
    os.makedirs(CACHE_DIR, exist_ok=True)
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8", newline="") as f:
        f.write(text)
    os.replace(tmp_path, path)


# ---------------------------------------------------------------------------
# Daily OHLCV history
# ---------------------------------------------------------------------------

def save_history(ticker, df):
    """Overwrite this ticker's cached history with the freshly-fetched
    DataFrame (same shape fetch_history returns). Always stores the whole
    window rather than merging -- a fresh live fetch is a strict superset
    of anything a merge could add, so overwrite is simplest and can't
    accumulate stale rows a since-adjusted split/dividend would leave
    behind under a merge."""
    if df is None or df.empty:
        return
    try:
        out = df[_HISTORY_COLUMNS].copy()
        # yfinance's daily index is tz-aware (America/New_York) and, across
        # a 300-day window, spans a DST transition -- mixed UTC offsets in
        # the written strings make pandas' read_csv date-inference give up
        # and leave the index as plain strings (silently, no exception) on
        # read-back. These are daily bars, so only the calendar date -- the
        # part that's actually unambiguous here -- needs to round-trip;
        # drop the tz/time component before writing rather than relying on
        # generic datetime parsing to reconstruct it correctly.
        idx = pd.DatetimeIndex(out.index)
        if idx.tz is not None:
            idx = idx.tz_localize(None)
        out.index = idx.strftime("%Y-%m-%d")
        out.index.name = "Date"
        _atomic_write_text(_history_path(ticker), out.to_csv())
    except Exception as e:
        print(f"WARNING: Failed to cache history for {ticker}: {e}")


def load_history(ticker):
    """Returns (df, as_of_date) from the last successful cache write for
    this ticker, or (None, None) if no cache exists yet -- e.g. this
    machine has never had a successful online run for this ticker."""
    path = _history_path(ticker)
    if not os.path.exists(path):
        return None, None
    try:
        df = pd.read_csv(path, index_col=0)
        df.index = pd.to_datetime(df.index, format="%Y-%m-%d")
        df = df[_HISTORY_COLUMNS].dropna(subset=["Close"])
        if df.empty:
            return None, None
        return df, df.index[-1].date()
    except Exception as e:
        print(f"WARNING: Failed to read cached history for {ticker}: {e}")
        return None, None


# ---------------------------------------------------------------------------
# VIX
# ---------------------------------------------------------------------------

def save_vix(value, timestamp):
    try:
        _atomic_write_text(_vix_path(), json.dumps({
            "value": value, "timestamp": timestamp.isoformat(),
        }))
    except Exception as e:
        print(f"WARNING: Failed to cache VIX: {e}")


def load_vix():
    """Returns (value, timestamp) or (None, None). Timestamp is a
    pandas.Timestamp so callers can run the same staleness math they'd run
    on a live read."""
    path = _vix_path()
    if not os.path.exists(path):
        return None, None
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return float(data["value"]), pd.Timestamp(data["timestamp"])
    except Exception as e:
        print(f"WARNING: Failed to read cached VIX: {e}")
        return None, None


# ---------------------------------------------------------------------------
# Earnings / dividend calendar
# ---------------------------------------------------------------------------

def _load_calendar_all():
    path = _calendar_path()
    if not os.path.exists(path):
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"WARNING: Failed to read cached calendar data: {e}")
        return {}


def save_calendar(ticker, earnings_date, dividend_date):
    """earnings_date/dividend_date are `date` objects or None. Stored even
    when None so a confirmed "nothing scheduled" from a live read isn't
    indistinguishable from "never fetched" -- but load_calendar still
    treats a cached date that has since passed as expired (see below),
    since a NEGATIVE result doesn't go stale the way a positive one does."""
    try:
        all_data = _load_calendar_all()
        all_data[ticker] = {
            "earnings": earnings_date.isoformat() if earnings_date else None,
            "dividend": dividend_date.isoformat() if dividend_date else None,
        }
        _atomic_write_text(_calendar_path(), json.dumps(all_data, indent=2, sort_keys=True))
    except Exception as e:
        print(f"WARNING: Failed to cache calendar data for {ticker}: {e}")


def load_calendar(ticker, report_date):
    """Returns (earnings_date, dividend_date), each a `date` or None. A
    cached date that's already on or before report_date is treated as
    expired (None) rather than presented as "next" -- the same guard
    fetch_next_dividend_date already applies to a live read (2nd audit:
    yfinance has been observed to hand back a past date as if it were
    upcoming), extended here so a stale cache can't reintroduce the exact
    bug that guard exists to prevent."""
    all_data = _load_calendar_all()
    entry = all_data.get(ticker)
    if not entry:
        return None, None

    def _parse(key):
        raw = entry.get(key)
        if not raw:
            return None
        try:
            d = pd.Timestamp(raw).date()
        except Exception:
            return None
        return d if d > report_date else None

    return _parse("earnings"), _parse("dividend")
