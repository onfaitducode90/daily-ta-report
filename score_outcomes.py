#!/usr/bin/env python3
"""
Outcome scorer (F35) -- meant to run nightly (or whenever), after
daily_ta_report.py has been logging via prediction_log.py for a while.

For every logged pattern, confluence verdict, and trade-idea stance that is
now old enough to check, this looks up what price actually did afterward at
+1/+5/+10/+20 trading bars and records:
  - direction_hit: did price move the way the call's bias said it would?
  - target_hit: did price reach the stated price target within the horizon?
  - (trade ideas only) stop_hit / which_first: for calls with both a
    reference target and a reference (stop-ish) level, which was touched
    first -- scanning day-by-day High/Low, not just the horizon's close, so
    a target/stop touched mid-window still counts.

Known limitation, stated plainly: this uses daily OHLC, so when a target and
a stop are hit on the SAME bar there is no way to tell which happened first
intraday -- that case is recorded as "both_same_bar", not guessed at.

Nothing here is scoreable until enough calendar time has passed for that
many trading bars to exist after the call was logged -- a call logged today
has zero scoreable horizons until tomorrow at the earliest. That's expected,
not a bug, and this script reports it explicitly rather than going quiet.
"""

import sqlite3
from datetime import date, datetime, timezone

import numpy as np

import daily_ta_report as report
import prediction_log

HORIZONS = (1, 5, 10, 20)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS pattern_outcomes (
    run_id TEXT NOT NULL,
    pattern_name TEXT NOT NULL,
    ticker TEXT NOT NULL,
    report_date TEXT NOT NULL,
    horizon_bars INTEGER NOT NULL,
    eval_date TEXT,
    spot_at_call REAL,
    spot_at_eval REAL,
    forward_return_pct REAL,
    bias TEXT,
    quality_tier TEXT,
    direction_hit INTEGER,
    target_hit INTEGER,
    scored_at TEXT NOT NULL,
    PRIMARY KEY (run_id, pattern_name, horizon_bars)
);

CREATE TABLE IF NOT EXISTS confluence_outcomes (
    run_id TEXT NOT NULL,
    ticker TEXT NOT NULL,
    report_date TEXT NOT NULL,
    horizon_bars INTEGER NOT NULL,
    eval_date TEXT,
    spot_at_call REAL,
    spot_at_eval REAL,
    forward_return_pct REAL,
    net_label TEXT,
    direction_hit INTEGER,
    scored_at TEXT NOT NULL,
    PRIMARY KEY (run_id, horizon_bars)
);

CREATE TABLE IF NOT EXISTS trade_idea_outcomes (
    run_id TEXT NOT NULL,
    ticker TEXT NOT NULL,
    report_date TEXT NOT NULL,
    horizon_bars INTEGER NOT NULL,
    eval_date TEXT,
    spot_at_call REAL,
    spot_at_eval REAL,
    forward_return_pct REAL,
    directional_bias TEXT,
    direction_hit INTEGER,
    target_hit INTEGER,
    stop_hit INTEGER,
    which_first TEXT,
    scored_at TEXT NOT NULL,
    PRIMARY KEY (run_id, horizon_bars)
);
"""


def _connect():
    conn = sqlite3.connect(prediction_log.DB_PATH)
    conn.executescript(_SCHEMA)
    return conn


def _parse_direction(label):
    """'Bullish confluence'/'Bullish lean' -> +1, 'Bearish ...' -> -1,
    anything else ('Conflicting signals', 'No clear directional edge',
    'Insufficient evidence', 'Neutral') -> None (no directional call made,
    so there is nothing to score as right or wrong)."""
    if not label:
        return None
    if label.startswith("Bullish"):
        return 1
    if label.startswith("Bearish"):
        return -1
    return None


def _parse_price(text):
    if not text or text == "N/A":
        return None
    try:
        return float(str(text).replace("$", "").replace(",", ""))
    except ValueError:
        return None


def _bar_position_on_or_before(df, report_date):
    mask = df.index.date <= report_date
    if not mask.any():
        return None
    return int(np.where(mask)[0][-1])


def _target_hit_in_window(window, direction, target):
    if target is None or window.empty:
        return None
    if direction > 0:
        return bool((window["High"] >= target).any())
    return bool((window["Low"] <= target).any())


def _race_target_stop(window, direction, target, stop):
    """Scan bar-by-bar; return (target_hit, stop_hit, which_first)."""
    if target is None or stop is None or window.empty:
        return None, None, None
    for _, bar in window.iterrows():
        hit_target = bar["High"] >= target if direction > 0 else bar["Low"] <= target
        hit_stop = bar["Low"] <= stop if direction > 0 else bar["High"] >= stop
        if hit_target and hit_stop:
            return True, True, "both_same_bar"
        if hit_target:
            return True, False, "target"
        if hit_stop:
            return False, True, "stop"
    return False, False, "neither"


def score_ticker(conn, ticker, df):
    now = datetime.now(timezone.utc).isoformat()
    scored, pending = 0, 0

    # --- patterns ---
    rows = conn.execute(
        "SELECT run_id, pattern_name, report_date, bias, quality_tier, price_target "
        "FROM patterns WHERE ticker = ?", (ticker,)).fetchall()
    for run_id, pattern_name, report_date_str, bias, tier, target_text in rows:
        direction = _parse_direction(bias)
        if direction is None:
            continue
        report_date = date.fromisoformat(report_date_str)
        pos = _bar_position_on_or_before(df, report_date)
        if pos is None:
            continue
        spot_at_call = float(df["Close"].iloc[pos])
        target = _parse_price(target_text)
        for h in HORIZONS:
            eval_pos = pos + h
            if eval_pos >= len(df):
                pending += 1
                continue
            spot_at_eval = float(df["Close"].iloc[eval_pos])
            fwd_ret = (spot_at_eval - spot_at_call) / spot_at_call
            direction_hit = int(np.sign(fwd_ret) == direction)
            window = df.iloc[pos + 1: eval_pos + 1]
            target_hit = _target_hit_in_window(window, direction, target)
            conn.execute(
                "INSERT OR REPLACE INTO pattern_outcomes VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (run_id, pattern_name, ticker, report_date_str, h, str(df.index[eval_pos].date()),
                 spot_at_call, spot_at_eval, fwd_ret * 100, bias, tier, direction_hit,
                 None if target_hit is None else int(target_hit), now))
            scored += 1

    # --- confluence ---
    rows = conn.execute(
        "SELECT run_id, report_date, net_label FROM confluence WHERE ticker = ?", (ticker,)).fetchall()
    for run_id, report_date_str, net_label in rows:
        direction = _parse_direction(net_label)
        if direction is None:
            continue
        report_date = date.fromisoformat(report_date_str)
        pos = _bar_position_on_or_before(df, report_date)
        if pos is None:
            continue
        spot_at_call = float(df["Close"].iloc[pos])
        for h in HORIZONS:
            eval_pos = pos + h
            if eval_pos >= len(df):
                pending += 1
                continue
            spot_at_eval = float(df["Close"].iloc[eval_pos])
            fwd_ret = (spot_at_eval - spot_at_call) / spot_at_call
            direction_hit = int(np.sign(fwd_ret) == direction)
            conn.execute(
                "INSERT OR REPLACE INTO confluence_outcomes VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (run_id, ticker, report_date_str, h, str(df.index[eval_pos].date()),
                 spot_at_call, spot_at_eval, fwd_ret * 100, net_label, direction_hit, now))
            scored += 1

    # --- trade ideas ---
    rows = conn.execute(
        "SELECT run_id, report_date, directional_bias, reference_level, reference_target "
        "FROM trade_ideas WHERE ticker = ?", (ticker,)).fetchall()
    for run_id, report_date_str, bias_label, ref_level, ref_target_text in rows:
        direction = _parse_direction(bias_label)
        if direction is None:
            continue
        report_date = date.fromisoformat(report_date_str)
        pos = _bar_position_on_or_before(df, report_date)
        if pos is None:
            continue
        spot_at_call = float(df["Close"].iloc[pos])
        target = _parse_price(ref_target_text)
        for h in HORIZONS:
            eval_pos = pos + h
            if eval_pos >= len(df):
                pending += 1
                continue
            spot_at_eval = float(df["Close"].iloc[eval_pos])
            fwd_ret = (spot_at_eval - spot_at_call) / spot_at_call
            direction_hit = int(np.sign(fwd_ret) == direction)
            window = df.iloc[pos + 1: eval_pos + 1]
            target_hit, stop_hit, which_first = _race_target_stop(window, direction, target, ref_level)
            conn.execute(
                "INSERT OR REPLACE INTO trade_idea_outcomes VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (run_id, ticker, report_date_str, h, str(df.index[eval_pos].date()),
                 spot_at_call, spot_at_eval, fwd_ret * 100, bias_label, direction_hit,
                 None if target_hit is None else int(target_hit),
                 None if stop_hit is None else int(stop_hit), which_first, now))
            scored += 1

    return scored, pending


def print_summary(conn):
    print("\n=== Outcome summary ===")
    for label, table in (("Patterns", "pattern_outcomes"),
                          ("Confluence calls", "confluence_outcomes"),
                          ("Trade ideas", "trade_idea_outcomes")):
        rows = conn.execute(
            f"SELECT horizon_bars, count(*), avg(direction_hit) FROM {table} "
            f"GROUP BY horizon_bars ORDER BY horizon_bars").fetchall()
        if not rows:
            print(f"{label}: nothing scoreable yet.")
            continue
        print(f"{label}:")
        for horizon, n, hit_rate in rows:
            print(f"  +{horizon} bars: n={n}, direction hit rate={hit_rate * 100:.1f}%")
    print("\nNote: with only a handful of logged days this is not yet a meaningful "
          "sample -- see F36/F38 for what it takes to draw a real conclusion "
          "(hundreds of independent calls, walk-forward validation, a baseline "
          "comparison). This is a running tally, not a verdict.")


def main():
    conn = _connect()
    tickers = [r[0] for r in conn.execute("SELECT DISTINCT ticker FROM runs").fetchall()]
    if not tickers:
        print("No logged runs yet -- run daily_ta_report.py first (and again on "
              "later dates) before there's anything to score.")
        return

    total_scored, total_pending = 0, 0
    for ticker in tickers:
        df = report.fetch_history(ticker)
        if df is None:
            print(f"WARNING: could not fetch history for {ticker}, skipping.")
            continue
        scored, pending = score_ticker(conn, ticker, df)
        conn.commit()
        total_scored += scored
        total_pending += pending
        print(f"{ticker}: {scored} horizon-outcomes scored, {pending} still pending "
              f"(not enough forward trading days have passed yet).")

    print(f"\nTotal: {total_scored} scored, {total_pending} pending across {len(tickers)} ticker(s).")
    print_summary(conn)
    conn.close()


if __name__ == "__main__":
    main()
