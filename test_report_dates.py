#!/usr/bin/env python3
"""
Regression tests for the report's date-windowed reference levels
(Prior day/week/month, Current week) in daily_ta_report.py.

Added after two real bugs found against live data:
  1. "Prior week" used a rolling 5-trading-day window, which included
     today's own bar.
  2. Even after excluding today, a rolling window still leaks OTHER
     current-week bars into "prior week" on any day but Monday (e.g. on
     a Wednesday, the trailing 5 trading days are last week's Thu/Fri
     plus this week's Mon/Tue/Wed).

"Prior week" must be the fully completed CALENDAR week before the
current one -- zero overlap with the current week's bars, regardless of
which weekday the report runs on. These tests exercise analyze_ticker()
end-to-end (not a reimplementation of the date math) so a future
regression in the real code path actually trips this suite.
"""
import sys

import pandas as pd

import daily_ta_report as R

RESULTS = []


def check(label, cond):
    RESULTS.append((label, cond))
    print(("PASS" if cond else "FAIL"), "-", label)


def _extract_line(text, prefix):
    return next((ln for ln in text.split("\n") if ln.startswith(prefix)), None)


def _make_df(end_date, n=120, base=100.0):
    """n trading days ending on end_date (inclusive), flat baseline
    OHLCV the caller then overwrites for specific weeks."""
    dates = pd.bdate_range(end=end_date, periods=n)
    df = pd.DataFrame({
        "Open": base, "High": base + 2, "Low": base - 2, "Close": base,
        "Volume": 1_000_000,
    }, index=dates)
    return df


def test_prior_week_excludes_a_more_extreme_current_week_on_a_midweek_run():
    """The exact scenario the fix targets: report runs on a Wednesday,
    and THIS week's high/low so far are more extreme than last week's
    real high/low. Prior Week High/Low must reflect only last week."""
    today = pd.Timestamp("2026-08-12")  # a Wednesday
    check("Sanity: test date is actually a Wednesday", today.day_name() == "Wednesday")

    df = _make_df(today, n=120, base=100.0)

    last_week_mask = (df.index >= "2026-08-03") & (df.index <= "2026-08-07")
    df.loc[last_week_mask, "High"] = 110.0
    df.loc[last_week_mask, "Low"] = 92.0

    # Current week (Mon 08-10 through today, Wed 08-12) blows past last
    # week's range on both sides -- if this leaks in, the test fails loudly.
    cur_week_mask = (df.index >= "2026-08-10") & (df.index <= "2026-08-12")
    df.loc[cur_week_mask, "High"] = 999.0
    df.loc[cur_week_mask, "Low"] = 1.0

    spy_df = _make_df(today, n=120, base=400.0)
    text, _ = R.analyze_ticker("TEST", df, "intraday", today.date(), spy_df=spy_df,
                                data_source="live", data_as_of=today.date())

    high_line = _extract_line(text, "Prior week high:")
    low_line = _extract_line(text, "Prior week low:")
    check(f"Prior week high line present ({high_line})", high_line is not None)
    check(f"Prior week low line present ({low_line})", low_line is not None)
    check(f"Prior week high reflects last week's true high, 110.00, not this week's 999 ({high_line})",
          high_line is not None and "110.00" in high_line)
    check(f"Prior week low reflects last week's true low, 92.00, not this week's 1 ({low_line})",
          low_line is not None and "92.00" in low_line)
    check("Current week's fabricated 999 high does not appear in the prior week line",
          high_line is not None and "999" not in high_line)
    check("Current week's fabricated 1 low does not appear in the prior week line",
          low_line is not None and low_line.split(":")[1].strip().split(" ")[0] != "$1.00")


def test_prior_week_on_a_monday_still_excludes_todays_own_bar():
    """On a Monday, the trailing-window bug and the calendar-week bug
    would coincide (today IS the only current-week bar so far) -- still
    worth locking in that Monday's own extreme doesn't leak in."""
    today = pd.Timestamp("2026-08-10")  # a Monday
    check("Sanity: test date is actually a Monday", today.day_name() == "Monday")

    df = _make_df(today, n=120, base=100.0)
    last_week_mask = (df.index >= "2026-08-03") & (df.index <= "2026-08-07")
    df.loc[last_week_mask, "High"] = 110.0
    df.loc[last_week_mask, "Low"] = 92.0
    df.loc[df.index == today, "High"] = 999.0
    df.loc[df.index == today, "Low"] = 1.0

    spy_df = _make_df(today, n=120, base=400.0)
    text, _ = R.analyze_ticker("TEST", df, "intraday", today.date(), spy_df=spy_df,
                                data_source="live", data_as_of=today.date())
    high_line = _extract_line(text, "Prior week high:")
    check(f"Monday run: prior week high still reflects last week's 110.00, not today's 999 ({high_line})",
          high_line is not None and "110.00" in high_line)


def test_prior_month_excludes_a_more_extreme_current_month():
    """Same fix, same reason, one level up: prior month must be the fully
    completed calendar month before this one. No "Prior month" report
    line exists (it only feeds the support/resistance model), so this
    checks the fabricated current-month extreme doesn't leak into the
    report ANYWHERE -- a stronger, path-agnostic check than grepping one
    specific line, and robust to however the S/R model clusters things.

    Today is deliberately late in the month (the 26th, a Wednesday) so
    there's a clean stretch of "current month, but not also prior-week/
    prior-day/current-week" trading days (Aug 3-16) to plant the fabricated
    extreme on, without it also being expected to show up in those OTHER
    date-windowed fields for unrelated, legitimate reasons."""
    today = pd.Timestamp("2026-08-26")  # a Wednesday, late in the month
    check("Sanity: test date is actually a Wednesday", today.day_name() == "Wednesday")
    df = _make_df(today, n=150, base=200.0)

    last_month_mask = (df.index >= "2026-07-01") & (df.index <= "2026-07-31")
    df.loc[last_month_mask, "High"] = 210.0
    df.loc[last_month_mask, "Low"] = 192.0

    # Aug 3-16: current month, but outside prior-week (17-23)/prior-day
    # (25)/current-week (24-26) -- isolated enough that only "prior month"
    # logic could possibly pick this up.
    early_aug_mask = (df.index >= "2026-08-03") & (df.index <= "2026-08-16")
    df.loc[early_aug_mask, "High"] = 1999.0
    df.loc[early_aug_mask, "Low"] = 2.0

    spy_df = _make_df(today, n=150, base=400.0)
    text, _ = R.analyze_ticker("TEST", df, "intraday", today.date(), spy_df=spy_df,
                                data_source="live", data_as_of=today.date())

    # Checking for the bare string "1999.00" anywhere would be too broad:
    # 52-week high/low is a DIFFERENT, intentionally-inclusive rolling
    # window (it's supposed to reflect the whole trailing year including
    # recent data), so it correctly picks up the fabricated spike too --
    # that's not a leak, that's 52-week high doing its actual job. The
    # thing that must specifically NOT happen is the "prior-month" label
    # attaching to the fabricated value.
    check('Fabricated high is not labeled "prior-month" evidence anywhere',
          "prior-month @ 1999.00" not in text)
    check('Fabricated low is not labeled "prior-month" evidence anywhere',
          "prior-month @ 2.00" not in text)
    # Positive checks too, not just absence -- prove prior-month evidence
    # actually surfaces with July's true values, not just that it's silent.
    check("prior-month evidence shows July's true high, 210.00", "prior-month @ 210.00" in text)
    check("prior-month evidence shows July's true low, 192.00", "prior-month @ 192.00" in text)


def main():
    test_prior_week_excludes_a_more_extreme_current_week_on_a_midweek_run()
    test_prior_week_on_a_monday_still_excludes_todays_own_bar()
    test_prior_month_excludes_a_more_extreme_current_month()
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
