#!/usr/bin/env python3
"""
Tests for iv_history.py's IV Percentile and IV Rank formulas.

Added after a user-reported discrepancy between this report's IV number
and ThinkorSwim's IV Rank turned out to be two separate issues: (1) the
report's number was HV-based, not from real IV, and (2) even once real
IV history exists, "IV Percentile" and "IV Rank" are two genuinely
different formulas that different platforms conflate under one name.
This suite locks in both formulas against hand-computed worked examples
so a future refactor can't quietly re-merge them or drift from the
standard definitions.

Runs against a temp CSV (iv_history.IV_HISTORY_PATH is monkeypatched for
the duration of each test), never the real logs/iv_history.csv.
"""
import datetime
import os
import sys
import tempfile

import iv_history as ivh

RESULTS = []


def check(label, cond):
    RESULTS.append((label, cond))
    print(("PASS" if cond else "FAIL"), "-", label)


def with_temp_history(fn):
    """Runs `fn` with iv_history pointed at an empty temp CSV, restoring
    the real path afterward regardless of outcome -- these tests must
    never read or write the real report history."""
    original_path = ivh.IV_HISTORY_PATH
    fd, tmp_path = tempfile.mkstemp(suffix=".csv")
    os.close(fd)
    os.remove(tmp_path)  # log_iv/_read_all must handle a missing file too
    ivh.IV_HISTORY_PATH = tmp_path
    try:
        fn()
    finally:
        ivh.IV_HISTORY_PATH = original_path
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def _seed(ticker, ivs):
    """Logs `ivs` (floats, oldest first) as consecutive daily snapshots
    for `ticker` under unique ascending fake dates."""
    base = datetime.date(2020, 1, 1)
    for i, iv in enumerate(ivs):
        date = (base + datetime.timedelta(days=i)).isoformat()
        ivh.log_iv(ticker, date, iv, "2020-06-19", 30, date)


def test_iv_percentile_worked_example():
    """User's worked example: current IV higher than 210 of the last 252
    days -> 210/252 = 83.3%."""
    def run():
        ivs = [0.10] * 210 + [1.50] * 41 + [0.99]  # 210 below, 41 above, today's own 0.99
        _seed("TEST", ivs)
        pct, n = ivh.real_iv_percentile("TEST", 0.99)
        check(f"IV Percentile worked example: n=252 (got {n})", n == 252)
        check(f"IV Percentile worked example: 83.3% (got {pct})",
              pct is not None and abs(pct - 83.3) < 0.1)
    with_temp_history(run)


def test_iv_rank_worked_example():
    """User's worked example: current=40%, low=20%, high=60% -> 50%."""
    def run():
        ivs = [0.20] + [0.35] * 29 + [0.60]  # low=20%, high=60%, 31 samples
        _seed("TEST", ivs)
        rank, n = ivh.real_iv_rank("TEST", 0.40)
        check(f"IV Rank worked example: n>=30 (got {n})", n >= 30)
        check(f"IV Rank worked example: 50.0% (got {rank})",
              rank is not None and abs(rank - 50.0) < 0.01)
    with_temp_history(run)


def test_insufficient_samples_returns_none():
    def run():
        _seed("TEST", [0.30] * 10)  # well under MIN_SAMPLES_FOR_RANK (30)
        pct, n = ivh.real_iv_percentile("TEST", 0.30)
        rank, n2 = ivh.real_iv_rank("TEST", 0.30)
        check("IV Percentile returns None under min-sample threshold", pct is None)
        check("IV Rank returns None under min-sample threshold", rank is None)
        check(f"n reflects actual logged count (got {n}, {n2})", n == 10 and n2 == 10)
    with_temp_history(run)


def test_iv_rank_handles_flat_history_without_dividing_by_zero():
    def run():
        _seed("TEST", [0.25] * 35)  # every reading identical -> high == low
        rank, n = ivh.real_iv_rank("TEST", 0.25)
        check("IV Rank on flat history returns 0.0, not an exception/NaN", rank == 0.0)
    with_temp_history(run)


def test_metrics_are_not_the_same_formula():
    """Same history, same current IV -- Percentile and Rank must not
    collapse to the same number. A skewed distribution (mostly low
    readings, a few high) makes the two formulas diverge clearly if each
    is implemented correctly and independently."""
    def run():
        ivs = [0.20] * 40 + [0.80] * 5  # skewed low
        _seed("TEST", ivs)
        pct, _ = ivh.real_iv_percentile("TEST", 0.50)
        rank, _ = ivh.real_iv_rank("TEST", 0.50)
        check(f"IV Percentile and IV Rank diverge on skewed history (pct={pct}, rank={rank})",
              pct is not None and rank is not None and abs(pct - rank) > 5)
    with_temp_history(run)


def test_consistent_across_tickers():
    """Same synthetic history under two different ticker symbols must
    produce identical results -- the formula must not special-case any
    particular underlying."""
    def run():
        ivs = [0.20] + [0.35] * 29 + [0.60]
        _seed("AAA", ivs)
        _seed("ZZZ", ivs)
        rank_a, _ = ivh.real_iv_rank("AAA", 0.40)
        rank_z, _ = ivh.real_iv_rank("ZZZ", 0.40)
        pct_a, _ = ivh.real_iv_percentile("AAA", 0.40)
        pct_z, _ = ivh.real_iv_percentile("ZZZ", 0.40)
        check("IV Rank identical across tickers given identical history", rank_a == rank_z)
        check("IV Percentile identical across tickers given identical history", pct_a == pct_z)
    with_temp_history(run)


def main():
    test_iv_percentile_worked_example()
    test_iv_rank_worked_example()
    test_insufficient_samples_returns_none()
    test_iv_rank_handles_flat_history_without_dividing_by_zero()
    test_metrics_are_not_the_same_formula()
    test_consistent_across_tickers()

    passed = sum(1 for _, ok in RESULTS if ok)
    total = len(RESULTS)
    print(f"\n{passed}/{total} checks passed")
    if passed != total:
        print("\nFAILURES:")
        for label, ok in RESULTS:
            if not ok:
                print(f"  - {label}")
        sys.exit(1)


if __name__ == "__main__":
    main()
