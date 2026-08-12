#!/usr/bin/env python3
"""
Pulls the daily ThinkorSwim option-chain CSV exports (option_chain.py's
data source) down from a dedicated Drive folder into the layout
option_chain.py expects on disk.

Report generation now runs on GitHub Actions (see state_sync.py's
docstring for why), which starts from a clean checkout every run --
there's no local machine for option_chain.py's hardcoded default path
(BASE_DIR) to find. The user still downloads these CSVs manually from
ThinkorSwim each day -- no automated feed exists -- but now drops them
into one flat Drive folder instead of relying on the local
spread_tool/optionchain/ layout being reachable; this module fans them
back out into <base_dir>/<TICKER>/<filename>.csv before
daily_ta_report.py runs, and the workflow points TA_OPTION_CHAIN_DIR at
that restored directory.

Filenames already encode both the ticker and the date
(<YYYY-MM-DD>-StockAndOptionQuoteFor<TICKER>.csv, matching
option_chain.py's expected format), so a single flat Drive folder is
enough -- no per-ticker subfolder is needed there even though
option_chain.py expects one locally. Files that don't match the
expected name are skipped, not treated as an error, so an unrelated
file dropped into the same folder can't break the restore.

One-way (Drive -> local) and read-only from this repo's side -- nothing
here ever writes back to Drive; that direction is entirely up to the
user's own upload/export process.

Best-effort, like state_sync.py: a failure here must degrade to "no
option chain data this run" (which daily_ta_report.py already handles
via option_chain.load_chain() returning None for a missing ticker
directory) rather than block report generation.
"""

import os
import re
import sys

import drive_client

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OPTIONCHAIN_FOLDER_NAME = "OptionChain"
DEFAULT_LOCAL_DIR = os.path.join(SCRIPT_DIR, "optionchain")

_FILENAME_RE = re.compile(r"^\d{4}-\d{2}-\d{2}-StockAndOptionQuoteFor([A-Za-z0-9]+)\.csv$")


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
            ticker = m.group(1)
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


if __name__ == "__main__":
    if "--restore" in sys.argv:
        restore()
    else:
        print("Usage: python optionchain_sync.py --restore")
        sys.exit(1)
