#!/usr/bin/env python3
"""
Syncs the local persistent-state files (prediction/calibration history,
market-data last-known-good cache, fill log, Drive-sync queue) to/from a
dedicated Drive folder.

This exists because report generation moved to a GitHub Actions runner,
which starts from a clean checkout on every run -- nothing under logs/
survives between runs on disk the way it does on a long-lived machine.
Without this, prediction_log.db's calibration history and
market_data_cache's offline fallback data would silently reset to empty
every single run. restore_state() runs before daily_ta_report.py,
persist_state() after.

Deliberately a flat one-folder mirror of logs/ (relative paths flattened
into Drive filenames via "__", reversed on restore) rather than mirroring
the directory structure in Drive -- simpler to list/dedupe with the same
find-by-name calls drive_client.py already exposes. Safe as long as no
file under logs/ has "__" in its name, which none of this repo's do.

Both directions are best-effort and never raise past this module -- same
rule as report_sync.py: a state hiccup degrades gracefully (empty/stale
state) rather than blocking report generation, which is the actual job.
"""

import os
import sys

import drive_client

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(SCRIPT_DIR, "logs")
STATE_FOLDER_NAME = "_state"

# drive_token.json is injected fresh from a GitHub secret at the start of
# every run (see the workflow) -- it must never round-trip through the
# state folder as a plain file alongside it. *.log files are per-run
# console output, not state; nothing reads them back.
_EXCLUDED_NAMES = {"drive_token.json"}


def _iter_local_state_files():
    if not os.path.isdir(LOG_DIR):
        return
    for root, _, files in os.walk(LOG_DIR):
        for name in files:
            if name in _EXCLUDED_NAMES or name.endswith(".log"):
                continue
            full = os.path.join(root, name)
            rel = os.path.relpath(full, LOG_DIR)
            yield full, rel


def _drive_name(rel_path):
    return rel_path.replace(os.sep, "__").replace("/", "__")


def _local_path(drive_name):
    return os.path.join(LOG_DIR, drive_name.replace("__", os.sep))


def restore_state():
    """Best-effort pull of every file in the Drive state folder down into
    logs/, overwriting whatever's there. Missing folder/no files/auth
    failure all degrade to "nothing restored" rather than raising -- a
    first-ever run (no state yet) looks identical to a restore failure
    from here, and both should just proceed with an empty logs/."""
    try:
        session = drive_client.get_session()
        folder_id = drive_client.find_or_create_folder(
            session, STATE_FOLDER_NAME, drive_client.DRIVE_FOLDER_ID)
        files = drive_client.list_files(session, folder_id)
        os.makedirs(os.path.join(LOG_DIR, "cache"), exist_ok=True)
        restored = 0
        for f in files:
            local_path = _local_path(f["name"])
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            content = drive_client.download_file(session, f["id"])
            with open(local_path, "wb") as fh:
                fh.write(content)
            restored += 1
        print(f"State restore: {restored} file(s) pulled from Drive.")
    except Exception as e:
        print(f"NOTE: State restore skipped/incomplete ({e}) -- continuing with empty/local state.")


def persist_state():
    """Best-effort push of every file currently under logs/ (minus the
    exclusions above) up to the Drive state folder. Runs even if the
    report step failed (see workflow's `if: always()`), so whatever
    partial state exists locally -- e.g. a cache write that succeeded
    before a later ticker crashed -- isn't lost for the next run."""
    try:
        session = drive_client.get_session()
        folder_id = drive_client.find_or_create_folder(
            session, STATE_FOLDER_NAME, drive_client.DRIVE_FOLDER_ID)
        persisted = 0
        for full, rel in _iter_local_state_files():
            with open(full, "rb") as fh:
                content = fh.read()
            drive_client.upload_or_update(
                session, _drive_name(rel), content, "application/octet-stream", folder_id=folder_id)
            persisted += 1
        print(f"State persist: {persisted} file(s) pushed to Drive.")
    except Exception as e:
        print(f"WARNING: State persist failed ({e}) -- next run may start from stale/empty state.")


if __name__ == "__main__":
    if "--restore" in sys.argv:
        restore_state()
    elif "--persist" in sys.argv:
        persist_state()
    else:
        print("Usage: python state_sync.py --restore | --persist")
        sys.exit(1)
