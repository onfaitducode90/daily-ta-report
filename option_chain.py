#!/usr/bin/env python3
"""
Parser for daily ThinkorSwim option-chain CSV exports (partial F29 --
real strikes/premiums/IV/greeks/open-interest from point-in-time daily
snapshots the user downloads manually, not a live feed).

Expected layout on disk:
    <base_dir>/<TICKER>/<YYYY-MM-DD>-StockAndOptionQuoteFor<TICKER>.csv

File format (a ToS "matrix" export): an UNDERLYING quote block, an
UNDERLYING EXTRA INFO block, then one section per expiration date -- a
header line like "7 AUG 26  (1)  100 (Weeklys)" (date, days-to-expiration,
contract multiplier, optional cycle label), followed by a column-header
row, then one row per strike with calls on the left and puts on the right
of a shared Exp/Strike pair of columns:

    Impl Vol,Delta,Volume,Open.Int,BID,BX,ASK,AX,Exp,Strike,
    BID,BX,ASK,AX,Impl Vol,Delta,Volume,Open.Int

Cells can be "<empty>" (no data) or "--" (IV not computable, e.g. a
worthless deep-ITM/OTM contract) -- both parse to None. Thousands
separators appear as quoted CSV fields ("10,624"), which the csv module
already un-quotes; this module additionally strips the embedded comma
before converting to int.

This module only PARSES the files -- it does not change how
daily_ta_report.py computes IV rank or names option structures. That's a
separate integration decision once the parser itself is verified against
real exports (see the module docstring's note in daily_ta_report.py's
Trade Idea section for why that integration hasn't been wired in yet).
"""

import csv
import os
import re
from dataclasses import dataclass, field
from datetime import date, datetime

BASE_DIR = r"C:\Users\Steph\Documents\code to review by Claude\spread_tool\optionchain"

_SECTION_HEADER_RE = re.compile(
    r"^(\d{1,2} [A-Z]{3} \d{2})\s+\((\d+)\)\s+(\d+)(?:\s+\((\w+)\))?\s*$")
_FIRST_LINE_RE = re.compile(
    r"^Stock quote and option quote for (\S+) on (\d{1,2}/\d{1,2}/\d{2}) (\d{1,2}:\d{2}:\d{2})")
_EXPECTED_COLUMNS = ["Impl Vol", "Delta", "Volume", "Open.Int", "BID", "BX", "ASK", "AX",
                     "Exp", "Strike", "BID", "BX", "ASK", "AX", "Impl Vol", "Delta",
                     "Volume", "Open.Int"]


@dataclass
class OptionQuote:
    expiration: date
    days_to_expiration: int
    strike: float
    call_bid: float = None
    call_ask: float = None
    call_iv: float = None       # fraction, e.g. 8.3831 for "838.31%"
    call_delta: float = None
    call_volume: int = None
    call_open_interest: int = None
    put_bid: float = None
    put_ask: float = None
    put_iv: float = None
    put_delta: float = None
    put_volume: int = None
    put_open_interest: int = None

    @property
    def call_mid(self):
        if self.call_bid is not None and self.call_ask is not None:
            return (self.call_bid + self.call_ask) / 2
        return None

    @property
    def put_mid(self):
        if self.put_bid is not None and self.put_ask is not None:
            return (self.put_bid + self.put_ask) / 2
        return None


@dataclass
class OptionChain:
    ticker: str
    snapshot_time: datetime
    underlying_last: float
    underlying_bid: float
    underlying_ask: float
    week52_high: float = None
    week52_low: float = None
    beta: float = None
    quotes: list = field(default_factory=list)  # list[OptionQuote]

    def expirations(self):
        """Sorted unique expiration dates in this chain."""
        return sorted({q.expiration for q in self.quotes})

    def for_expiration(self, expiration):
        return [q for q in self.quotes if q.expiration == expiration]

    def nearest_expiration(self, min_days=0):
        """Nearest expiration at least `min_days` out (e.g. to skip 0-DTE
        when you want at least a week of theta) -- None if nothing qualifies."""
        candidates = [e for e in self.expirations()
                      if (e - self.snapshot_time.date()).days >= min_days]
        return min(candidates, default=None)

    def atm_quote(self, expiration):
        """The quote whose strike is closest to the underlying's last price,
        for a given expiration -- None if that expiration isn't in the chain."""
        rows = self.for_expiration(expiration)
        if not rows:
            return None
        return min(rows, key=lambda q: abs(q.strike - self.underlying_last))

    def atm_iv(self, expiration):
        """Average of ATM call and put IV for the given expiration (the
        two should be close absent major skew) -- a real, market-priced IV
        reading, unlike the realized-volatility proxy daily_ta_report.py
        falls back to when no chain is available. None if unavailable."""
        q = self.atm_quote(expiration)
        if q is None:
            return None
        ivs = [v for v in (q.call_iv, q.put_iv) if v is not None]
        return sum(ivs) / len(ivs) if ivs else None


def _parse_month_date(text, snapshot_year_hint):
    """'7 AUG 26' -> date(2026, 8, 7)."""
    parts = text.split()
    day = int(parts[0])
    month = datetime.strptime(parts[1], "%b").month
    year = 2000 + int(parts[2])
    return date(year, month, day)


def _clean_cell(raw):
    raw = raw.strip()
    if raw in ("", "<empty>", "--"):
        return None
    return raw


def _to_float(raw):
    raw = _clean_cell(raw)
    if raw is None:
        return None
    raw = raw.replace(",", "").replace("%", "")
    try:
        return float(raw)
    except ValueError:
        return None


def _to_int(raw):
    raw = _clean_cell(raw)
    if raw is None:
        return None
    raw = raw.replace(",", "")
    try:
        return int(raw)
    except ValueError:
        return None


def _to_pct_fraction(raw):
    """'838.31%' -> 8.3831 (a fraction, matching how the rest of this
    codebase expresses HV/IV -- e.g. calc_hv returns a fraction, not a
    0-100 percent)."""
    raw = _clean_cell(raw)
    if raw is None:
        return None
    raw = raw.replace(",", "").replace("%", "")
    try:
        return float(raw) / 100
    except ValueError:
        return None


def parse_option_chain_csv(path):
    """Parse one ThinkorSwim option-chain CSV export into an OptionChain.
    Raises ValueError if the file doesn't match the expected format at all
    (e.g. empty file, unrecognized first line) -- malformed individual rows
    within an otherwise-valid file are skipped, not fatal, since a single
    corrupted strike shouldn't discard an entire day's chain."""
    with open(path, encoding="utf-8-sig", newline="") as f:
        raw_lines = f.read().splitlines()
    if not raw_lines:
        raise ValueError(f"{path}: empty file")

    m = _FIRST_LINE_RE.match(raw_lines[0])
    if not m:
        raise ValueError(f"{path}: unrecognized first line: {raw_lines[0]!r}")
    ticker, date_str, time_str = m.groups()
    month, day, year2 = (int(x) for x in date_str.split("/"))
    snapshot_time = datetime.strptime(f"20{year2:02d}-{month:02d}-{day:02d} {time_str}",
                                       "%Y-%m-%d %H:%M:%S")

    rows = list(csv.reader(raw_lines))

    underlying_last = underlying_bid = underlying_ask = None
    week52_high = week52_low = beta = None
    quotes = []

    i = 0
    n = len(rows)
    while i < n:
        line = raw_lines[i] if i < len(raw_lines) else ""

        if line.strip() == "UNDERLYING" and i + 2 < n:
            header, data = rows[i + 1], rows[i + 2]
            values = dict(zip(header, data))
            underlying_last = _to_float(values.get("LAST"))
            underlying_bid = _to_float(values.get("BID"))
            underlying_ask = _to_float(values.get("ASK"))
            i += 3
            continue

        if line.strip() == "UNDERLYING EXTRA INFO" and i + 2 < n:
            header, data = rows[i + 1], rows[i + 2]
            values = dict(zip(header, data))
            week52_high = _to_float(values.get("52High"))
            week52_low = _to_float(values.get("52Low"))
            beta = _to_float(values.get("Beta"))
            i += 3
            continue

        section_match = _SECTION_HEADER_RE.match(line.strip())
        if section_match:
            exp_text, dte_text, _mult, _cycle = section_match.groups()
            expiration = _parse_month_date(exp_text, snapshot_time.year)
            dte = int(dte_text)
            i += 1
            # The header row has 2 leading blank columns (",,Impl Vol,...")
            # before the columns this parser actually keys off of.
            actual_cols = [c.strip() for c in rows[i][2:2 + len(_EXPECTED_COLUMNS)]] if i < n else []
            if actual_cols != _EXPECTED_COLUMNS:
                # Not the column header we expect right after a section
                # title -- don't guess column positions against a layout
                # that doesn't match what this parser was built for.
                raise ValueError(f"{path}: unexpected columns after {exp_text!r} "
                                  f"section header: {actual_cols!r}")
            i += 1
            while i < n and raw_lines[i].strip() != "" and not _SECTION_HEADER_RE.match(raw_lines[i].strip()):
                cells = rows[i]
                if len(cells) >= 20:
                    quotes.append(OptionQuote(
                        expiration=expiration, days_to_expiration=dte,
                        strike=_to_float(cells[11]),
                        call_iv=_to_pct_fraction(cells[2]), call_delta=_to_float(cells[3]),
                        call_volume=_to_int(cells[4]), call_open_interest=_to_int(cells[5]),
                        call_bid=_to_float(cells[6]), call_ask=_to_float(cells[8]),
                        put_bid=_to_float(cells[12]), put_ask=_to_float(cells[14]),
                        put_iv=_to_pct_fraction(cells[16]), put_delta=_to_float(cells[17]),
                        put_volume=_to_int(cells[18]), put_open_interest=_to_int(cells[19]),
                    ))
                i += 1
            continue

        i += 1

    if underlying_last is None:
        raise ValueError(f"{path}: could not find an UNDERLYING quote block")

    return OptionChain(
        ticker=ticker, snapshot_time=snapshot_time,
        underlying_last=underlying_last, underlying_bid=underlying_bid, underlying_ask=underlying_ask,
        week52_high=week52_high, week52_low=week52_low, beta=beta, quotes=quotes,
    )


def available_dates(ticker, base_dir=BASE_DIR):
    """Sorted list of dates with a downloaded chain for this ticker."""
    ticker_dir = os.path.join(base_dir, ticker)
    if not os.path.isdir(ticker_dir):
        return []
    out = []
    for fname in os.listdir(ticker_dir):
        m = re.match(r"(\d{4}-\d{2}-\d{2})-StockAndOptionQuoteFor" + re.escape(ticker) + r"\.csv$", fname)
        if m:
            out.append(date.fromisoformat(m.group(1)))
    return sorted(out)


def load_chain(ticker, for_date=None, base_dir=BASE_DIR):
    """Load the chain for `for_date` (a date or ISO string), or the most
    recent available if for_date is None. Returns None if nothing is
    available -- callers must treat a missing local snapshot the same way
    they'd treat any other unavailable data source, not as an error."""
    dates = available_dates(ticker, base_dir)
    if not dates:
        return None
    if for_date is None:
        target = dates[-1]
    else:
        target = date.fromisoformat(for_date) if isinstance(for_date, str) else for_date
        if target not in dates:
            return None
    path = os.path.join(base_dir, ticker, f"{target.isoformat()}-StockAndOptionQuoteFor{ticker}.csv")
    return parse_option_chain_csv(path)
