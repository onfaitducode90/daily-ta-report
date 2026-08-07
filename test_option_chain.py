#!/usr/bin/env python3
"""
Sanity/regression checks for option_chain.py, run against whatever real
CSV exports are currently available locally under option_chain.BASE_DIR.
Skips (does not fail) if that folder isn't present or empty -- this is a
local, manually-downloaded daily snapshot, not something a fresh checkout
of this repo would have, so its absence isn't a regression.
"""

import sys

import option_chain as oc

TICKERS = ["NVDA", "SPCX", "INTC", "GLD", "SLV", "PLTR"]
RESULTS = []


def check(label, cond):
    RESULTS.append((label, cond))
    print(("PASS" if cond else "FAIL"), "-", label)


def main():
    any_data = any(oc.available_dates(tk) for tk in TICKERS)
    if not any_data:
        print(f"No local option-chain data found under {oc.BASE_DIR} -- skipping (not failing).")
        return

    for ticker in TICKERS:
        dates = oc.available_dates(ticker)
        if not dates:
            continue
        chain = oc.load_chain(ticker)
        check(f"{ticker}: chain parses without error", chain is not None)
        if chain is None:
            continue

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

    passed = sum(1 for _, ok in RESULTS if ok)
    total = len(RESULTS)
    print(f"\n{passed}/{total} checks passed")
    if passed != total:
        sys.exit(1)


if __name__ == "__main__":
    main()
