#!/usr/bin/env python3
"""
Structured logging of every pattern detection, confluence verdict, and
trade-idea stance this tool produces, keyed by ticker/date/mode.

This exists for one reason: right now there is no way to know whether any
of this tool's output has predictive value, because nothing about a past
run is retained anywhere queryable. Every day without this log is a day of
unrecoverable evidence. Once enough runs are logged, an outcome scorer can
join these predictions against what price actually did afterward (F35) and
a calibration curve can check whether stated pattern "confidence" bears any
relationship to hit rate (F36) -- neither is possible without this first.

Writing here must never break report generation, so every write is wrapped
and failures are printed as warnings, not raised.
"""

import json
import os
import sqlite3
import subprocess
from datetime import datetime, timezone

import chart_patterns

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(SCRIPT_DIR, "logs")
DB_PATH = os.path.join(LOG_DIR, "prediction_log.db")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    ticker TEXT NOT NULL,
    report_date TEXT NOT NULL,
    mode TEXT NOT NULL,
    spot REAL,
    logged_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS patterns (
    run_id TEXT NOT NULL,
    ticker TEXT NOT NULL,
    report_date TEXT NOT NULL,
    pattern_name TEXT NOT NULL,
    category TEXT,
    bias TEXT,
    confidence REAL,
    quality_tier TEXT,
    status TEXT,
    formed_date TEXT,
    price_target TEXT,
    spot REAL,
    logged_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS confluence (
    run_id TEXT PRIMARY KEY,
    ticker TEXT NOT NULL,
    report_date TEXT NOT NULL,
    net REAL,
    net_label TEXT,
    bull_weight REAL,
    bear_weight REAL,
    total_weight REAL,
    sufficient_evidence INTEGER,
    signal_vector TEXT,
    spot REAL,
    logged_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS trade_ideas (
    run_id TEXT PRIMARY KEY,
    ticker TEXT NOT NULL,
    report_date TEXT NOT NULL,
    directional_bias TEXT,
    net REAL,
    iv_rank REAL,
    hv30 REAL,
    adx REAL,
    trending INTEGER,
    earnings_date TEXT,
    days_to_earnings INTEGER,
    reference_level REAL,
    reference_stop_distance REAL,
    reference_target TEXT,
    spot REAL,
    logged_at TEXT NOT NULL
);
"""

# Tables that get a code_version column via migration below (every table
# with something calibrate.py/score_outcomes.py might one day need to
# segment by version).
_VERSIONED_TABLES = ("runs", "patterns", "confluence", "trade_ideas")

_code_version_cache = None


def get_code_version():
    """Short git SHA of the currently checked-out commit, or 'unknown' if
    git isn't available/this isn't a repo. Cached for the process's
    lifetime -- a 3rd audit found the DB already contained rows from at
    least 3 materially different confluence implementations (different
    STRONG_CONFLUENCE_WEIGHT logic) with no way to tell them apart:
    same run_id, same total_weight, contradictory net_label. Tagging
    every row lets calibrate.py filter to a single, consistent version
    instead of unknowingly calibrating a mixture. Uncommitted local
    changes aren't reflected in this SHA -- it identifies the commit, not
    the exact bytes on disk -- which is an accepted gap, not a promise
    this makes about mid-edit runs."""
    global _code_version_cache
    if _code_version_cache is not None:
        return _code_version_cache
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=SCRIPT_DIR, capture_output=True, text=True, timeout=5)
        _code_version_cache = result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        _code_version_cache = "unknown"
    return _code_version_cache


def _migrate_schema(conn):
    """Add code_version to any table that predates this column -- ALTER
    TABLE ADD COLUMN is the only schema change sqlite supports without a
    full table rebuild, and it's safe to re-run (skipped once the column
    exists). Existing rows get NULL, which calibrate.py treats as
    'unknown version' and excludes, rather than guessing which version
    wrote them."""
    for table in _VERSIONED_TABLES:
        cols = [row[1] for row in conn.execute(f"PRAGMA table_info({table})")]
        if "code_version" not in cols:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN code_version TEXT")


def _connect():
    os.makedirs(LOG_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.executescript(_SCHEMA)
    _migrate_schema(conn)
    return conn


def log_run(ticker, report_date, mode, spot, pattern_matches, confluence, trade_idea):
    """Idempotently log one ticker's run (re-running the same ticker/date/mode
    overwrites rather than duplicates). `confluence` and `trade_idea` are
    plain dicts assembled by the caller; `pattern_matches` is the list of
    PatternMatch from chart_patterns.detect_all."""
    run_id = f"{ticker}_{report_date}_{mode}"
    now = datetime.now(timezone.utc).isoformat()
    code_version = get_code_version()

    try:
        conn = _connect()
        with conn:
            conn.execute(
                "INSERT OR REPLACE INTO runs "
                "(run_id, ticker, report_date, mode, spot, logged_at, code_version) "
                "VALUES (?,?,?,?,?,?,?)",
                (run_id, ticker, str(report_date), mode, spot, now, code_version))

            conn.execute("DELETE FROM patterns WHERE run_id = ?", (run_id,))
            for pm in pattern_matches:
                conn.execute(
                    "INSERT INTO patterns "
                    "(run_id, ticker, report_date, pattern_name, category, bias, confidence, "
                    " quality_tier, status, formed_date, price_target, spot, logged_at, code_version) "
                    "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (run_id, ticker, str(report_date), pm.name, pm.category, pm.bias,
                     pm.confidence, chart_patterns.confidence_tier(pm.confidence), pm.status,
                     pm.formed_date, pm.price_target, spot, now, code_version))

            conn.execute(
                "INSERT OR REPLACE INTO confluence "
                "(run_id, ticker, report_date, net, net_label, bull_weight, bear_weight, "
                " total_weight, sufficient_evidence, signal_vector, spot, logged_at, code_version) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (run_id, ticker, str(report_date), confluence["net"], confluence["net_label"],
                 confluence["bull_weight"], confluence["bear_weight"], confluence["total_weight"],
                 int(confluence["sufficient_evidence"]), json.dumps(confluence["signals"]),
                 spot, now, code_version))

            conn.execute(
                "INSERT OR REPLACE INTO trade_ideas "
                "(run_id, ticker, report_date, directional_bias, net, iv_rank, hv30, adx, trending, "
                " earnings_date, days_to_earnings, reference_level, reference_stop_distance, "
                " reference_target, spot, logged_at, code_version) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (run_id, ticker, str(report_date), trade_idea["directional_bias"], trade_idea["net"],
                 trade_idea["iv_rank"], trade_idea["hv30"], trade_idea["adx"],
                 None if trade_idea["trending"] is None else int(trade_idea["trending"]),
                 str(trade_idea["earnings_date"]) if trade_idea["earnings_date"] else None,
                 trade_idea["days_to_earnings"], trade_idea["reference_level"],
                 trade_idea["reference_stop_distance"], trade_idea["reference_target"],
                 spot, now, code_version))
        conn.close()
    except Exception as e:
        print(f"WARNING: prediction log write failed: {e}")
