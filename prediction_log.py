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


def _connect():
    os.makedirs(LOG_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.executescript(_SCHEMA)
    return conn


def log_run(ticker, report_date, mode, spot, pattern_matches, confluence, trade_idea):
    """Idempotently log one ticker's run (re-running the same ticker/date/mode
    overwrites rather than duplicates). `confluence` and `trade_idea` are
    plain dicts assembled by the caller; `pattern_matches` is the list of
    PatternMatch from chart_patterns.detect_all."""
    run_id = f"{ticker}_{report_date}_{mode}"
    now = datetime.now(timezone.utc).isoformat()

    try:
        conn = _connect()
        with conn:
            conn.execute(
                "INSERT OR REPLACE INTO runs VALUES (?,?,?,?,?,?)",
                (run_id, ticker, str(report_date), mode, spot, now))

            conn.execute("DELETE FROM patterns WHERE run_id = ?", (run_id,))
            for pm in pattern_matches:
                conn.execute(
                    "INSERT INTO patterns VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (run_id, ticker, str(report_date), pm.name, pm.category, pm.bias,
                     pm.confidence, chart_patterns.confidence_tier(pm.confidence), pm.status,
                     pm.formed_date, pm.price_target, spot, now))

            conn.execute(
                "INSERT OR REPLACE INTO confluence VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (run_id, ticker, str(report_date), confluence["net"], confluence["net_label"],
                 confluence["bull_weight"], confluence["bear_weight"], confluence["total_weight"],
                 int(confluence["sufficient_evidence"]), json.dumps(confluence["signals"]),
                 spot, now))

            conn.execute(
                "INSERT OR REPLACE INTO trade_ideas VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (run_id, ticker, str(report_date), trade_idea["directional_bias"], trade_idea["net"],
                 trade_idea["iv_rank"], trade_idea["hv30"], trade_idea["adx"],
                 None if trade_idea["trending"] is None else int(trade_idea["trending"]),
                 str(trade_idea["earnings_date"]) if trade_idea["earnings_date"] else None,
                 trade_idea["days_to_earnings"], trade_idea["reference_level"],
                 trade_idea["reference_stop_distance"], trade_idea["reference_target"],
                 spot, now))
        conn.close()
    except Exception as e:
        print(f"WARNING: prediction log write failed: {e}")
