#!/usr/bin/env python3
"""
Local queue + state machine that gets every generated report into Drive
eventually, without ever losing or duplicating one.

daily_ta_report.py always writes its report to a local .txt file first --
that's the durable source of truth and is untouched by any of this. This
module is purely additive: enqueue() records that a report exists and
needs syncing, flush_queue() attempts to upload whatever's outstanding.
Both are wrapped end-to-end and never raise -- report generation (and,
for flush_queue, the Task Scheduler job that calls this standalone) must
never fail because Drive is unreachable or misconfigured.

Two independent triggers call flush_queue():
  1. daily_ta_report.py, once, right after it saves a new report.
  2. A separate Task Scheduler job running `python report_sync.py --flush`
     on a timer, so a report generated while offline doesn't sit PENDING
     until the next scheduled report run hours later.
Because of that, flush_queue() claims a row (UPDATE ... WHERE status IN
(...)) immediately before uploading it rather than batching the read and
the status change, so two near-simultaneous flushes can't both attempt
the same row.

Status lifecycle: PENDING -> UPLOADING -> UPLOADED, or -> FAILED (which
retries, back through UPLOADING, up to MAX_RETRIES times). A row stuck at
UPLOADING past STALE_UPLOADING_MINUTES (the process died mid-upload) is
reclaimed as FAILED so it isn't lost. data_source ("live" or "cached")
records whether the report itself was built from live or cached market
data -- see market_data_cache.py -- kept separate from `status` (the sync
state) since a report can be, for example, both "generated from cached
data" and "successfully uploaded" at once.

Dedup: drive_client.upload_or_update() finds an existing Drive file by
name before creating one, so retrying an upload -- whether because this
row failed and is being retried, or because two flush triggers raced --
can never leave two copies of the same report in Drive.
"""

import argparse
import os
import sqlite3
from datetime import datetime, timedelta, timezone

import drive_client

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(SCRIPT_DIR, "logs")
DB_PATH = os.path.join(LOG_DIR, "report_sync.db")

MAX_RETRIES = 5
STALE_UPLOADING_MINUTES = 10

_SCHEMA = """
CREATE TABLE IF NOT EXISTS report_queue (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    filepath TEXT NOT NULL UNIQUE,
    filename TEXT NOT NULL,
    report_date TEXT NOT NULL,
    mode TEXT NOT NULL,
    data_source TEXT NOT NULL,
    status TEXT NOT NULL,
    drive_file_id TEXT,
    retry_count INTEGER NOT NULL DEFAULT 0,
    last_error TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
"""


def _now_iso():
    return datetime.now(timezone.utc).isoformat()


def _connect():
    os.makedirs(LOG_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.execute(_SCHEMA)
    return conn


def enqueue(filepath, report_date, mode, data_source):
    """Records a freshly-written report as needing sync. Re-enqueuing the
    same filepath (a same-day/same-mode re-run overwriting its own file)
    resets it to PENDING with a fresh retry count -- the on-disk content
    just changed, so whatever sync state applied to the old content no
    longer means anything."""
    try:
        now = _now_iso()
        with _connect() as conn:
            conn.execute("""
                INSERT INTO report_queue
                    (filepath, filename, report_date, mode, data_source, status,
                     retry_count, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, 'PENDING', 0, ?, ?)
                ON CONFLICT(filepath) DO UPDATE SET
                    data_source = excluded.data_source,
                    status = 'PENDING',
                    retry_count = 0,
                    last_error = NULL,
                    updated_at = excluded.updated_at
            """, (filepath, os.path.basename(filepath), str(report_date), mode,
                  data_source, now, now))
    except Exception as e:
        print(f"WARNING: Failed to enqueue report for sync: {e}")


def _reclaim_stale_uploads(conn):
    cutoff = (datetime.now(timezone.utc) - timedelta(minutes=STALE_UPLOADING_MINUTES)).isoformat()
    conn.execute("""
        UPDATE report_queue SET status = 'FAILED',
            last_error = 'Reclaimed: stuck in UPLOADING past staleness window (likely a crash mid-upload)',
            updated_at = ?
        WHERE status = 'UPLOADING' AND updated_at < ?
    """, (_now_iso(), cutoff))


def flush_queue(max_retries=MAX_RETRIES):
    """Attempts to upload every outstanding (PENDING or previously-FAILED,
    under the retry cap) report. Returns a dict of counts for the caller
    to log. Never raises -- a Drive/auth failure degrades to "nothing
    uploaded this attempt, try again next flush", not a crash, whether
    the caller is daily_ta_report.py finishing a report or the standalone
    scheduled job."""
    result = {"uploaded": 0, "failed": 0, "skipped_no_connection": 0}
    try:
        with _connect() as conn:
            _reclaim_stale_uploads(conn)
            conn.commit()
            rows = conn.execute("""
                SELECT id, filepath, filename FROM report_queue
                WHERE status IN ('PENDING', 'FAILED') AND retry_count < ?
                ORDER BY created_at
            """, (max_retries,)).fetchall()

            if not rows:
                return result

            try:
                session = drive_client.get_session()
            except drive_client.DriveError as e:
                print(f"NOTE: Report sync skipped this attempt -- Drive unreachable ({e}). "
                      "Reports remain queued locally and will retry next time.")
                result["skipped_no_connection"] = len(rows)
                return result

            for row_id, filepath, filename in rows:
                claimed = conn.execute("""
                    UPDATE report_queue SET status = 'UPLOADING', updated_at = ?
                    WHERE id = ? AND status IN ('PENDING', 'FAILED')
                """, (_now_iso(), row_id))
                conn.commit()
                if claimed.rowcount == 0:
                    continue  # another flush_queue() call already claimed this row

                try:
                    if not os.path.exists(filepath):
                        raise FileNotFoundError(f"local report file missing: {filepath}")
                    with open(filepath, "rb") as f:
                        content = f.read()
                    file_id = drive_client.upload_or_update(session, filename, content, "text/plain")
                    conn.execute("""
                        UPDATE report_queue SET status = 'UPLOADED', drive_file_id = ?,
                            last_error = NULL, updated_at = ?
                        WHERE id = ?
                    """, (file_id, _now_iso(), row_id))
                    result["uploaded"] += 1
                except Exception as e:
                    conn.execute("""
                        UPDATE report_queue SET status = 'FAILED', retry_count = retry_count + 1,
                            last_error = ?, updated_at = ?
                        WHERE id = ?
                    """, (str(e)[:500], _now_iso(), row_id))
                    result["failed"] += 1
                conn.commit()
    except Exception as e:
        print(f"WARNING: Report sync flush failed unexpectedly: {e}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flush", action="store_true",
                         help="Attempt to upload every queued report (for the Task Scheduler job).")
    args = parser.parse_args()
    if args.flush:
        summary = flush_queue()
        print(f"Sync flush: {summary['uploaded']} uploaded, {summary['failed']} failed, "
              f"{summary['skipped_no_connection']} skipped (no connection).")
    else:
        parser.print_help()
