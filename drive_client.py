#!/usr/bin/env python3
"""
Thin Google Drive v3 REST wrapper for report_sync.py -- authenticates as
YOU (via OAuth) and finds/creates/updates files in one target folder.

Deliberately raw REST over `requests` (via google-auth's AuthorizedSession,
which handles token refresh automatically) rather than the
google-api-python-client package: that package isn't installed anywhere in
this environment, while google-auth/google-auth-oauthlib and requests all
already are (requests as an existing yfinance dependency), and all this
needs is three endpoints (list/create/update). Adding a whole client
library for three REST calls doesn't match this repo's habit of pinning a
small, deliberate dependency set (see the comment above yfinance in
requirements.txt).

Auth note: this started as a service-account credential, but Google
rejects a service account writing into a personal ("My Drive") folder --
"Service Accounts do not have storage quota" -- service accounts can only
own files in a Shared Drive (Workspace-only) or via domain-wide delegation
(also Workspace-only). Neither applies to a personal Drive, so this uses a
standard OAuth "installed app" flow instead: a ONE-TIME browser consent
(run this file with --setup) mints a refresh token stored locally at
TOKEN_PATH, after which every upload runs as your own account against your
own quota, fully unattended (no repeated browser prompts) as long as that
token file exists.
"""

import argparse
import json
import os

import requests
from google.auth.transport.requests import AuthorizedSession, Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

CLIENT_SECRET_PATH = os.environ.get(
    "TA_GOOGLE_CLIENT_SECRET_PATH",
    os.path.join(SCRIPT_DIR, "google credentials",
                 "client_secret_1059404297679-pb2c36uo036od0gu3v7er9lfppsja7n7.apps.googleusercontent.com.json"))
TOKEN_PATH = os.environ.get("TA_GOOGLE_TOKEN_PATH", os.path.join(SCRIPT_DIR, "logs", "drive_token.json"))
DRIVE_FOLDER_ID = os.environ.get("TA_DRIVE_FOLDER_ID", "1p2kE0ocEB11cGtFNfu8aKw9M7ADwGJZq")

SCOPES = ["https://www.googleapis.com/auth/drive"]
REQUEST_TIMEOUT_SECONDS = 20
_FILES_URL = "https://www.googleapis.com/drive/v3/files"
_UPLOAD_URL = "https://www.googleapis.com/upload/drive/v3/files"


class DriveError(Exception):
    """Raised for any Drive auth/API failure -- callers (report_sync.py)
    catch this rather than letting requests' own exception hierarchy leak
    through, so a change in the HTTP layer doesn't ripple into callers."""


def run_setup(client_secret_path=None, token_path=None):
    """One-time interactive step: opens a browser for you to log into
    Google and grant this script Drive access, then stores the resulting
    refresh token locally. Must be run manually, with a real browser
    available -- this can never be triggered automatically (e.g. from the
    Task Scheduler job), since only a human can complete a login."""
    client_secret_path = client_secret_path or CLIENT_SECRET_PATH
    token_path = token_path or TOKEN_PATH
    if not os.path.exists(client_secret_path):
        raise DriveError(f"OAuth client secret not found: {client_secret_path}")
    flow = InstalledAppFlow.from_client_secrets_file(client_secret_path, SCOPES)
    creds = flow.run_local_server(port=0)
    os.makedirs(os.path.dirname(token_path), exist_ok=True)
    tmp_path = token_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        f.write(creds.to_json())
    os.replace(tmp_path, token_path)
    print(f"Drive access granted -- refresh token saved to {token_path}")


def get_session(token_path=None):
    """Returns an AuthorizedSession (a requests.Session subclass that
    attaches/refreshes the OAuth access token automatically) built from
    the locally stored refresh token. Raises DriveError if run_setup()
    hasn't been completed yet, or if the stored token has been revoked --
    either way, this needs the one-time manual step above, not something
    this module can recover from on its own."""
    path = token_path or TOKEN_PATH
    if not os.path.exists(path):
        raise DriveError(f"No Drive token at {path} -- run `python drive_client.py --setup` "
                          "once, interactively, to authorize this script against your Drive.")
    try:
        creds = Credentials.from_authorized_user_file(path, SCOPES)
        if creds.expired and creds.refresh_token:
            creds.refresh(Request())
        return AuthorizedSession(creds)
    except Exception as e:
        raise DriveError(f"Failed to load/refresh Drive token from {path}: {e}") from e


def _escape(name):
    return name.replace("\\", "\\\\").replace("'", "\\'")


def find_file(session, filename, folder_id=None):
    """Returns the Drive file ID of an existing, non-trashed file with
    this exact name in the target folder, or None if there isn't one.
    This is what makes upload_or_update idempotent: re-syncing the same
    report (a corrected re-run, or a retried upload after a partial
    failure) updates the existing file instead of creating a duplicate."""
    folder_id = folder_id or DRIVE_FOLDER_ID
    query = f"'{_escape(folder_id)}' in parents and name = '{_escape(filename)}' and trashed = false"
    try:
        resp = session.get(_FILES_URL, params={"q": query, "fields": "files(id,name)"},
                            timeout=REQUEST_TIMEOUT_SECONDS)
        resp.raise_for_status()
    except requests.exceptions.RequestException as e:
        raise DriveError(f"Drive lookup failed for {filename}: {e}") from e
    files = resp.json().get("files", [])
    return files[0]["id"] if files else None


def find_folder(session, name, parent_id):
    """Same idea as find_file, but scoped to folders via mimeType -- kept
    separate rather than reusing find_file so a same-named regular file
    can never be mistaken for the folder callers actually want."""
    query = (f"'{_escape(parent_id)}' in parents and name = '{_escape(name)}' "
             f"and mimeType = 'application/vnd.google-apps.folder' and trashed = false")
    try:
        resp = session.get(_FILES_URL, params={"q": query, "fields": "files(id,name)"},
                            timeout=REQUEST_TIMEOUT_SECONDS)
        resp.raise_for_status()
    except requests.exceptions.RequestException as e:
        raise DriveError(f"Drive folder lookup failed for {name}: {e}") from e
    files = resp.json().get("files", [])
    return files[0]["id"] if files else None


def create_folder(session, name, parent_id):
    metadata = {"name": name, "mimeType": "application/vnd.google-apps.folder", "parents": [parent_id]}
    try:
        resp = session.post(_FILES_URL, json=metadata, timeout=REQUEST_TIMEOUT_SECONDS)
        resp.raise_for_status()
        return resp.json()["id"]
    except requests.exceptions.RequestException as e:
        raise DriveError(f"Drive folder creation failed for {name}: {e}") from e


def find_or_create_folder(session, name, parent_id):
    return find_folder(session, name, parent_id) or create_folder(session, name, parent_id)


def list_files(session, folder_id):
    """Non-trashed files directly inside folder_id, as a list of
    {"id", "name"} dicts."""
    query = f"'{_escape(folder_id)}' in parents and trashed = false"
    try:
        resp = session.get(_FILES_URL, params={"q": query, "fields": "files(id,name)"},
                            timeout=REQUEST_TIMEOUT_SECONDS)
        resp.raise_for_status()
    except requests.exceptions.RequestException as e:
        raise DriveError(f"Drive folder listing failed: {e}") from e
    return resp.json().get("files", [])


def download_file(session, file_id):
    try:
        resp = session.get(f"{_FILES_URL}/{file_id}", params={"alt": "media"},
                            timeout=REQUEST_TIMEOUT_SECONDS)
        resp.raise_for_status()
        return resp.content
    except requests.exceptions.RequestException as e:
        raise DriveError(f"Drive download failed for file {file_id}: {e}") from e


def upload_or_update(session, filename, content_bytes, mime_type="text/plain", folder_id=None):
    """Finds an existing file by name and updates its content in place, or
    creates a new one if none exists. Returns the Drive file ID either
    way. This find-then-create/update sequence -- rather than always
    creating -- is the dedup guarantee: interrupting and re-running a sync
    can never leave two copies of the same report in Drive."""
    folder_id = folder_id or DRIVE_FOLDER_ID
    existing_id = find_file(session, filename, folder_id=folder_id)
    try:
        if existing_id:
            resp = session.patch(
                f"{_UPLOAD_URL}/{existing_id}", params={"uploadType": "media"},
                data=content_bytes, headers={"Content-Type": mime_type},
                timeout=REQUEST_TIMEOUT_SECONDS)
            resp.raise_for_status()
            return existing_id

        metadata = {"name": filename, "parents": [folder_id]}
        files = {
            "metadata": ("metadata", json.dumps(metadata), "application/json"),
            "file": (filename, content_bytes, mime_type),
        }
        resp = session.post(_UPLOAD_URL, params={"uploadType": "multipart"},
                             files=files, timeout=REQUEST_TIMEOUT_SECONDS)
        resp.raise_for_status()
        return resp.json()["id"]
    except requests.exceptions.RequestException as e:
        raise DriveError(f"Drive upload failed for {filename}: {e}") from e


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--setup", action="store_true",
                         help="One-time interactive Drive authorization (opens a browser).")
    args = parser.parse_args()
    if args.setup:
        run_setup()
    else:
        parser.print_help()
