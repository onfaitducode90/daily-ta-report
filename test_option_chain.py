#!/usr/bin/env python3
"""
Sanity/regression checks for option_chain.py.

Two layers:
1. Committed fixture CSVs under test_fixtures/optionchain/ZTEST/ -- small,
   synthetic, hand-built with known values, run UNCONDITIONALLY. Before
   this existed, the only coverage of the parser (the highest-stakes new
   code -- the only thing that emits real strikes) was the best-effort
   check below, which silently reports success having exercised zero of
   the parsing code whenever the local data folder isn't present (e.g.
   any fresh checkout of this repo, or CI). A 2nd audit caught that.
2. Whatever real CSV exports are currently available locally under
   option_chain.BASE_DIR -- still best-effort, since that's a local,
   manually-downloaded daily snapshot no fresh checkout would have. But
   if BASE_DIR was explicitly overridden via TA_OPTION_CHAIN_DIR and that
   directory doesn't exist, that's a real misconfiguration and this now
   fails loudly instead of quietly skipping.
"""

import os
import sys

import option_chain as oc

FIXTURE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_fixtures", "optionchain")
TICKERS = ["NVDA", "SPCX", "INTC", "GLD", "SLV", "PLTR"]
RESULTS = []


def check(label, cond):
    RESULTS.append((label, cond))
    print(("PASS" if cond else "FAIL"), "-", label)


def check_chain_structure(ticker, chain):
    """Structural sanity checks that apply to any correctly-parsed chain,
    fixture or real."""
    check(f"{ticker}: chain parses without error", chain is not None)
    if chain is None:
        return

    check(f"{ticker}: underlying_last is a positive number",
          chain.underlying_last is not None and chain.underlying_last > 0)
    check(f"{ticker}: has at least one expiration", len(chain.expirations()) > 0)
    check(f"{ticker}: has at least one quote", len(chain.quotes) > 0)
    check(f"{ticker}: expirations are sorted ascending",
          chain.expirations() == sorted(chain.expirations()))
    check(f"{ticker}: all strikes are positive",
          all(q.strike is not None and q.strike > 0 for q in chain.quotes))

    bad_call = [q for q in chain.quotes
                if q.call_bid is not None and q.call_ask is not None and q.call_bid > q.call_ask]
    check(f"{ticker}: no call has bid > ask ({len(bad_call)} violations)", len(bad_call) == 0)
    bad_put = [q for q in chain.quotes
               if q.put_bid is not None and q.put_ask is not None and q.put_bid > q.put_ask]
    check(f"{ticker}: no put has bid > ask ({len(bad_put)} violations)", len(bad_put) == 0)

    pairs = [(q.expiration, q.strike) for q in chain.quotes]
    check(f"{ticker}: no duplicate (expiration, strike) pairs", len(pairs) == len(set(pairs)))

    near = chain.nearest_expiration(min_days=0)
    check(f"{ticker}: nearest_expiration resolves", near is not None)
    atm = chain.atm_quote(near) if near else None
    check(f"{ticker}: atm_quote resolves for the nearest expiration", atm is not None)


def check_fixture_values():
    """Exact-value checks against the committed ZTEST fixture -- since the
    fixture's contents are known ground truth (I wrote them), this catches
    parsing bugs (wrong column offset, bad cell cleaning, wrong pct/int
    conversion) that structural-only checks can't: a column-shifted parse
    can still produce a structurally "valid" chain with silently wrong
    numbers."""
    dates = oc.available_dates("ZTEST", base_dir=FIXTURE_DIR)
    check("ZTEST fixture: both snapshot dates found", dates == [
        __import__("datetime").date(2026, 1, 15), __import__("datetime").date(2026, 1, 16)])

    chain = oc.load_chain("ZTEST", for_date="2026-01-15", base_dir=FIXTURE_DIR)
    check_chain_structure("ZTEST (2026-01-15)", chain)
    if chain is None:
        return

    check("ZTEST: underlying_last == 100.00", chain.underlying_last == 100.00)
    check("ZTEST: underlying_bid/ask == 99.95/100.05",
          chain.underlying_bid == 99.95 and chain.underlying_ask == 100.05)
    check("ZTEST: 52High/52Low == 120.00/80.00",
          chain.week52_high == 120.00 and chain.week52_low == 80.00)
    check("ZTEST: beta == 1.20", chain.beta == 1.20)
    check("ZTEST: two expirations parsed",
          chain.expirations() == [__import__("datetime").date(2026, 8, 14), __import__("datetime").date(2026, 8, 21)])
    check("ZTEST: 5 strikes per expiration", len(chain.for_expiration(chain.expirations()[0])) == 5)

    exp1 = chain.expirations()[0]
    atm = chain.atm_quote(exp1)
    check("ZTEST: ATM strike (exp1) == 100", atm.strike == 100.0)
    check("ZTEST: ATM call bid/ask == 3.00/3.20", atm.call_bid == 3.00 and atm.call_ask == 3.20)
    check("ZTEST: ATM call delta == 0.50", atm.call_delta == 0.50)
    check("ZTEST: ATM call IV == 0.40 (40.00% -> fraction)", atm.call_iv == 0.40)
    check("ZTEST: ATM call volume/OI == 50/1200", atm.call_volume == 50 and atm.call_open_interest == 1200)
    check("ZTEST: ATM put bid/ask == 2.90/3.10", atm.put_bid == 2.90 and atm.put_ask == 3.10)
    check("ZTEST: ATM put delta == -0.50", atm.put_delta == -0.50)
    check("ZTEST: days_to_expiration == 8", atm.days_to_expiration == 8)

    otm_put = next(q for q in chain.for_expiration(exp1) if q.strike == 90.0)
    check("ZTEST: 90-strike put delta == -0.15 (thousands-free, sign preserved)", otm_put.put_delta == -0.15)

    # Second snapshot date has a different (smaller) chain -- confirms
    # load_chain(for_date=...) actually selects the requested date, not
    # always the most recent.
    chain2 = oc.load_chain("ZTEST", for_date="2026-01-16", base_dir=FIXTURE_DIR)
    check_chain_structure("ZTEST (2026-01-16)", chain2)
    if chain2 is not None:
        check("ZTEST (2026-01-16): underlying_last == 101.00", chain2.underlying_last == 101.00)
        check("ZTEST (2026-01-16): single expiration with 3 strikes",
              len(chain2.expirations()) == 1 and len(chain2.quotes) == 3)

    # load_chain() with no for_date picks the MOST RECENT snapshot.
    chain_latest = oc.load_chain("ZTEST", base_dir=FIXTURE_DIR)
    check("ZTEST: load_chain() with no date picks the latest snapshot (2026-01-16)",
          chain_latest is not None and chain_latest.underlying_last == 101.00)


def check_local_real_data():
    """Best-effort check against whatever real chain snapshots happen to
    be downloaded locally. Distinguishes 'nothing configured, nothing
    there' (expected on a fresh checkout -- skip) from 'explicitly
    configured via TA_OPTION_CHAIN_DIR, but that directory doesn't exist'
    (a real misconfiguration -- fail loudly)."""
    configured = "TA_OPTION_CHAIN_DIR" in os.environ
    if configured and not os.path.isdir(oc.BASE_DIR):
        check(f"TA_OPTION_CHAIN_DIR is set to {oc.BASE_DIR!r} but that directory doesn't exist", False)
        return

    any_data = any(oc.available_dates(tk) for tk in TICKERS)
    if not any_data:
        print(f"No local option-chain data found under {oc.BASE_DIR} -- skipping (not failing; "
              "this is a local, manually-downloaded daily snapshot, not something a fresh "
              "checkout of this repo would have).")
        return

    for ticker in TICKERS:
        dates = oc.available_dates(ticker)
        if not dates:
            continue
        chain = oc.load_chain(ticker)
        check_chain_structure(ticker, chain)


def main():
    check_fixture_values()
    check_local_real_data()

    passed = sum(1 for _, ok in RESULTS if ok)
    total = len(RESULTS)
    print(f"\n{passed}/{total} checks passed")
    if passed != total:
        sys.exit(1)


if __name__ == "__main__":
    main()
