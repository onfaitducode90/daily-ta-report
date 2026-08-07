#!/usr/bin/env python3
"""
Fill-quality tracker (G28, 2nd Opus audit).

Every credit structure this tool prints is priced at the chain's MID --
"verified" strikes and credit are real, but the number itself assumes a
fill exactly halfway between bid and ask, which a real order often
doesn't get, especially on the wider/thinner legs the liquidity filters
already flag as marginal. There was no way to find out how optimistic
that mid-based number actually is, because nothing recorded what a real
fill looked like against it.

This is a manual log, by design: fills happen in a real brokerage
account this tool has no access to, so the only honest way to populate
this is for you to log what you actually got filled at, right after you
take a trade based on a report's numbers. One CSV row per fill.

Usage (from the command line, right after placing a trade):
    python fill_tracker.py log NVDA "Bull Put Spread" --mid-credit 0.29 \\
        --actual-credit 0.22 --contracts 2 --notes "wide market at open"

    python fill_tracker.py summary

Or from Python:
    import fill_tracker
    fill_tracker.log_fill("NVDA", "2026-08-07", "Bull Put Spread",
                           mid_credit=0.29, actual_credit=0.22, contracts=2)
"""

import argparse
import csv
import os
from datetime import date, datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(SCRIPT_DIR, "logs")
FILL_LOG_PATH = os.path.join(LOG_DIR, "fill_log.csv")

_FIELDNAMES = ["ticker", "report_date", "structure", "mid_credit", "actual_credit",
               "contracts", "slippage_per_contract", "slippage_pct_of_mid", "notes", "logged_at"]


def _read_all():
    if not os.path.exists(FILL_LOG_PATH):
        return []
    with open(FILL_LOG_PATH, newline="") as f:
        return list(csv.DictReader(f))


def log_fill(ticker, report_date, structure, mid_credit, actual_credit, contracts=1, notes=""):
    """Append one real fill. Not idempotent by design -- multiple real
    fills can legitimately exist for the same ticker/day (different
    structures, or scaling into a position), so unlike prediction_log.py
    this never overwrites, it only appends. `mid_credit`/`actual_credit`
    are both credits received (positive = you collected premium); a
    positive slippage means you collected LESS than the mid-based number
    implied -- the mid-price optimism this is meant to quantify."""
    os.makedirs(LOG_DIR, exist_ok=True)
    # A 3rd audit noted notes was written verbatim -- a note starting with
    # =/+/-/@ is a formula-injection payload if this CSV is later opened
    # in Excel/Sheets. Prefix with a literal quote to neutralize it as a
    # formula without changing what the note says.
    if notes and notes[0] in "=+-@":
        notes = "'" + notes
    slippage = mid_credit - actual_credit
    slippage_pct = (slippage / mid_credit * 100) if mid_credit else None
    row = {
        "ticker": ticker, "report_date": str(report_date), "structure": structure,
        "mid_credit": f"{mid_credit:.4f}", "actual_credit": f"{actual_credit:.4f}",
        "contracts": contracts, "slippage_per_contract": f"{slippage:.4f}",
        "slippage_pct_of_mid": f"{slippage_pct:.1f}" if slippage_pct is not None else "",
        "notes": notes, "logged_at": datetime.now().isoformat(),
    }
    # A 3rd audit found that checking existence alone silently eats the
    # first fill if the file exists but is empty (e.g. an interrupted
    # earlier write, or a manually `touch`-ed file): DictReader would
    # then consume that first real fill AS the header, and
    # summarize_fills would report "No fills logged yet" while the fill
    # sits unrecoverable in the file. Also write the header on a
    # zero-byte file.
    write_header = not os.path.exists(FILL_LOG_PATH) or os.path.getsize(FILL_LOG_PATH) == 0
    with open(FILL_LOG_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_FIELDNAMES)
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    return row


def get_empirical_haircut_fraction(min_fills=20):
    """Mean slippage as a fraction of mid credit, from real logged fills
    (H14, 3rd Opus audit) -- e.g. 0.15 means real fills have averaged 15%
    worse than the mid-based credit a report showed. Returns (fraction,
    n) where fraction is None until `min_fills` real fills are logged, so
    daily_ta_report.py's build_credit_spread can fall back to its own
    modeled haircut (half the combined bid-ask width) until there's
    enough real data to trust instead of that estimate. 20 is a
    judgment call, not a statistically-derived minimum -- real fills
    don't accumulate on any schedule this module controls."""
    rows = _read_all()
    pcts = [float(r["slippage_pct_of_mid"]) for r in rows if r.get("slippage_pct_of_mid")]
    if len(pcts) < min_fills:
        return None, len(pcts)
    return (sum(pcts) / len(pcts)) / 100.0, len(pcts)


def summarize_fills():
    """Print mean/median slippage across every logged fill -- the actual
    answer to "how optimistic is this tool's mid-based 'verified' number",
    once enough fills have been logged to say anything. Returns the parsed
    rows (empty list if nothing logged yet)."""
    rows = _read_all()
    if not rows:
        print("No fills logged yet. Use `python fill_tracker.py log ...` right after "
              "taking a trade based on this tool's numbers.")
        return rows

    slippages = [float(r["slippage_per_contract"]) for r in rows]
    pcts = [float(r["slippage_pct_of_mid"]) for r in rows if r["slippage_pct_of_mid"]]

    print(f"=== Fill quality: {len(rows)} logged fill(s) ===")
    print(f"Mean slippage: ${sum(slippages) / len(slippages):+.3f}/contract "
          f"(positive = filled worse than the mid-based number shown)")
    sorted_slip = sorted(slippages)
    mid = len(sorted_slip) // 2
    # True median, not just the upper-middle element on an even n (a 3rd
    # audit caught sorted(x)[len(x)//2] silently mislabeling that as the
    # median for even-sized samples).
    median_slip = (sorted_slip[mid] if len(sorted_slip) % 2
                   else (sorted_slip[mid - 1] + sorted_slip[mid]) / 2)
    print(f"Median slippage: ${median_slip:+.3f}/contract")
    if pcts:
        print(f"Mean slippage as % of mid credit: {sum(pcts) / len(pcts):+.1f}%")
    worst = max(rows, key=lambda r: float(r["slippage_per_contract"]))
    best = min(rows, key=lambda r: float(r["slippage_per_contract"]))
    print(f"Worst fill: {worst['ticker']} {worst['structure']} on {worst['report_date']} "
          f"-- mid ${worst['mid_credit']}, actual ${worst['actual_credit']}")
    print(f"Best fill:  {best['ticker']} {best['structure']} on {best['report_date']} "
          f"-- mid ${best['mid_credit']}, actual ${best['actual_credit']}")
    if len(rows) < 20:
        print(f"\n(n={len(rows)} is still a small sample -- treat this as a running tally, "
              "not a stable estimate of typical slippage. Keep logging.)")
    return rows


def main():
    parser = argparse.ArgumentParser(description="Log and summarize real fills vs. this tool's mid-based numbers.")
    sub = parser.add_subparsers(dest="command", required=True)

    log_p = sub.add_parser("log", help="Log a real fill")
    log_p.add_argument("ticker")
    log_p.add_argument("structure", help='e.g. "Bull Put Spread", "Iron Condor"')
    log_p.add_argument("--report-date", default=str(date.today()), help="Defaults to today")
    log_p.add_argument("--mid-credit", type=float, required=True, help="The mid-based credit the report showed")
    log_p.add_argument("--actual-credit", type=float, required=True, help="What you actually got filled at")
    log_p.add_argument("--contracts", type=int, default=1)
    log_p.add_argument("--notes", default="")

    sub.add_parser("summary", help="Print fill-quality summary stats")

    args = parser.parse_args()
    if args.command == "log":
        row = log_fill(args.ticker, args.report_date, args.structure, args.mid_credit,
                        args.actual_credit, args.contracts, args.notes)
        print(f"Logged: {row}")
    elif args.command == "summary":
        summarize_fills()


if __name__ == "__main__":
    main()
