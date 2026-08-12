#!/usr/bin/env python3
"""
Syncs the daily ThinkorSwim option-chain CSV exports (option_chain.py's
data source) between a dedicated Drive folder and the <TICKER>/<file>.csv
layout option_chain.py expects on disk.

Report generation now runs on GitHub Actions (see state_sync.py's
docstring for why), which starts from a clean checkout every run --
there's no local machine for option_chain.py's hardcoded default path
(BASE_DIR) to find. The user still downloads these CSVs manually from
ThinkorSwim each day -- no automated feed exists -- so this module has
two independent commands:
  --push    Run locally, after exporting the day's chains. Uploads
            whatever's under the local option-chain directory (default:
            option_chain.BASE_DIR, i.e. wherever the user's ToS exports
            already land) up to the flat Drive folder, then prunes old
            files so the folder doesn't grow forever.
  --restore Run by the GitHub Actions workflow before daily_ta_report.py.
            Pulls everything in the Drive folder back down into
            <local_dir>/<TICKER>/<file>.csv, and the workflow points
            TA_OPTION_CHAIN_DIR at that restored directory.

Filenames already encode both the ticker and the date
(<YYYY-MM-DD>-StockAndOptionQuoteFor<TICKER>.csv, matching
option_chain.py's expected format), so the Drive folder itself stays
flat -- no per-ticker subfolder there, even though option_chain.py
expects one locally on both ends. Files that don't match the expected
name are skipped, not treated as an error, so an unrelated file sitting
in the same folder can't break either direction.

Retention: load_chain() with no explicit for_date (the only way
daily_ta_report.py calls it) always picks the single most recent
snapshot per ticker, so there's no live-report reason to keep old ones
around -- OPTIONCHAIN_RETENTION_COUNT keeps a small buffer per ticker
(parsed from the filename, not Drive's modifiedTime, since a batch
upload can give several files the same modified timestamp) rather than
one global cutoff, so an infrequently-traded name's history isn't
crowded out by a heavily-exported one.

Both directions are best-effort and never raise past this module -- same
rule as state_sync.py/report_sync.py: a sync hiccup degrades gracefully
(restore: no option-chain data this run, via option_chain.load_chain()
returning None; push: local files just stay unsynced until next --push)
rather than blocking anything.
"""

import argparse
import os
import re
import sys

import drive_client
import option_chain

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OPTIONCHAIN_FOLDER_NAME = "OptionChain"
DEFAULT_LOCAL_DIR = os.path.join(SCRIPT_DIR, "optionchain")
OPTIONCHAIN_RETENTION_COUNT = 3

_FILENAME_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})-StockAndOptionQuoteFor([A-Za-z0-9]+)\.csv$")


def restore(local_dir=None):
    """Best-effort pull of every recognizably-named file in the Drive
    OptionChain folder down into <local_dir>/<TICKER>/, overwriting
    whatever's there. Missing folder/no files/auth failure all degrade
    to "nothing restored" rather than raising."""
    local_dir = local_dir or DEFAULT_LOCAL_DIR
    try:
        session = drive_client.get_session()
        folder_id = drive_client.find_or_create_folder(
            session, OPTIONCHAIN_FOLDER_NAME, drive_client.DRIVE_FOLDER_ID)
        files = drive_client.list_files(session, folder_id)
        restored = 0
        skipped = 0
        for f in files:
            m = _FILENAME_RE.match(f["name"])
            if not m:
                skipped += 1
                continue
            ticker = m.group(2)
            ticker_dir = os.path.join(local_dir, ticker)
            os.makedirs(ticker_dir, exist_ok=True)
            content = drive_client.download_file(session, f["id"])
            with open(os.path.join(ticker_dir, f["name"]), "wb") as fh:
                fh.write(content)
            restored += 1
        suffix = f", {skipped} skipped (unrecognized name)" if skipped else ""
        print(f"Option chain restore: {restored} file(s) pulled from Drive{suffix}.")
    except Exception as e:
        print(f"NOTE: Option chain restore skipped/incomplete ({e}) -- "
              f"report will run without local option-chain data.")


def _iter_local_chain_files(local_dir):
    for root, _, files in os.walk(local_dir):
        for name in files:
            if _FILENAME_RE.match(name):
                yield os.path.join(root, name), name


def _enforce_retention(session, folder_id, keep=OPTIONCHAIN_RETENTION_COUNT):
    """Deletes all but the `keep` most-recent (by filename date, per
    ticker) files from the Drive OptionChain folder. Never raises: a
    cleanup failure just means old snapshots linger an extra push, not
    something worth failing the sync over."""
    try:
        files = drive_client.list_files(session, folder_id)
        by_ticker = {}
        for f in files:
            m = _FILENAME_RE.match(f["name"])
            if not m:
                continue
            by_ticker.setdefault(m.group(2), []).append((m.group(1), f))
        for ticker, entries in by_ticker.items():
            entries.sort(key=lambda e: e[0], reverse=True)
            for _, f in entries[keep:]:
                drive_client.delete_file(session, f["id"])
                print(f"Retention: deleted old option chain {f['name']}")
    except Exception as e:
        print(f"WARNING: Option chain retention cleanup failed ({e}) -- old snapshots may linger.")


def push(local_dir=None):
    """Best-effort push of every recognizably-named file currently under
    the local option-chain directory (default: wherever option_chain.py
    itself looks, so this stays in sync with the actual local layout by
    construction) up to the Drive OptionChain folder, then prunes old
    snapshots per ticker."""
    local_dir = local_dir or option_chain.BASE_DIR
    try:
        session = drive_client.get_session()
        folder_id = drive_client.find_or_create_folder(
            session, OPTIONCHAIN_FOLDER_NAME, drive_client.DRIVE_FOLDER_ID)
        pushed = 0
        for full, name in _iter_local_chain_files(local_dir):
            with open(full, "rb") as fh:
                content = fh.read()
            drive_client.upload_or_update(session, name, content, "text/csv", folder_id=folder_id)
            pushed += 1
        print(f"Option chain push: {pushed} file(s) pushed to Drive.")
        if pushed > 0:
            _enforce_retention(session, folder_id)
    except Exception as e:
        print(f"WARNING: Option chain push failed ({e}) -- local files remain unsynced.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--restore", action="store_true",
                         help="Pull Drive's OptionChain folder down into local <TICKER>/ layout (used by GitHub Actions).")
    parser.add_argument("--push", action="store_true",
                         help="Upload the local option-chain directory up to Drive (run this yourself after exporting from ToS).")
    args = parser.parse_args()
    if args.restore:
        restore()
    elif args.push:
        push()
    else:
        parser.print_help()
        sys.exit(1)
