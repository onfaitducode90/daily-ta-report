#!/usr/bin/env python3
"""
Calibration (F36) -- checks whether stated pattern confidence bears any
relationship to realized outcomes and, once there's enough data, fits an
isotonic (monotonic) regression mapping raw confidence -> a calibrated hit
rate. Meant to run monthly (or whenever), after score_outcomes.py has
populated pattern_outcomes for a while.

This is the actual fix for B4 from the independent audit: the 0-100
confidence numbers are a weighted blend of geometric heuristics with no
empirical backing. If the isotonic fit comes back FLAT, that is the honest
answer -- it means confidence carries no measurable information at the
current sample size, not that something went wrong. This script reports
that plainly rather than force a curve that isn't there, and refuses to
even attempt a fit below a minimum sample size rather than overfitting
noise from a handful of points.

sklearn is only needed here, for fitting. The saved output is just a list
of (x, y) breakpoints, so remap_confidence() -- used at report-generation
time -- only needs numpy, not sklearn as a runtime dependency.
"""

import json
import os
import sqlite3

import numpy as np
from scipy import stats as scipy_stats

import prediction_log

CALIBRATION_DIR = os.path.join(prediction_log.LOG_DIR, "calibration")
# A 2nd audit measured the false-positive rate of the p<0.05 significance
# test under this pipeline's actual sample structure: with samples that
# are really just 3 near-duplicates of each other (morning/intraday/
# evening share the same report_date, spot, and usually the same detected
# patterns), the nominal-5% test fired 25.6% of the time; pooled further
# across the (then-untested) horizon axis, 57.8%. De-duplicating by
# (ticker, report_date, pattern_name) below removes the worst of that
# (the exact-triplicate case), and MIN_SAMPLES_PROVISIONAL/RELIABLE now
# count DISTINCT (ticker, report_date) pairs, not raw rows, so the
# reported "n" can't be inflated by mode alone. What de-duplication does
# NOT fix: forward-return windows from NEARBY dates for the SAME ticker
# still overlap (a +10-bar horizon shares 9 of 10 bars with the call made
# one trading day later), so consecutive-date samples remain positively
# autocorrelated and the significance test below is still anti-conservative
# to some degree. A real fix needs a block-bootstrap or Newey-West p-value;
# short of that, treat "significant" here as "clears a much higher bar than
# it used to," not as a rigorous hypothesis test.
MIN_SAMPLES_PROVISIONAL = 150
MIN_SAMPLES_RELIABLE = 500
HORIZONS = (1, 5, 10, 20)


def _connect():
    return sqlite3.connect(prediction_log.DB_PATH)


def _fetch_confidence_hit_pairs(conn, horizon):
    """One (confidence, direction_hit) pair per DISTINCT (ticker,
    report_date, pattern_name) -- de-duplicated across mode (morning/
    intraday/evening), which otherwise silently triples nearly every
    observation (same day, same spot, usually the same detected pattern).
    Prefers the evening row when more than one mode logged the same
    pattern that day (most complete data for that session); falls back to
    whichever mode is present otherwise."""
    rows = conn.execute(
        "SELECT p.ticker, p.report_date, p.pattern_name, p.confidence, po.direction_hit, r.mode "
        "FROM pattern_outcomes po "
        "JOIN patterns p ON p.run_id = po.run_id AND p.pattern_name = po.pattern_name "
        "JOIN runs r ON r.run_id = po.run_id "
        "WHERE po.horizon_bars = ? AND po.direction_hit IS NOT NULL",
        (horizon,)).fetchall()

    mode_rank = {"evening": 0, "intraday": 1, "morning": 2}
    best = {}
    for ticker, report_date, pattern_name, confidence, direction_hit, mode in rows:
        key = (ticker, report_date, pattern_name)
        rank = mode_rank.get(mode, 99)
        if key not in best or rank < best[key][0]:
            best[key] = (rank, confidence, direction_hit)

    pairs = [(v[1], v[2]) for v in best.values()]
    n_distinct_ticker_dates = len({(k[0], k[1]) for k in best})
    return pairs, n_distinct_ticker_dates


def calibrate_horizon(conn, horizon):
    pairs, n_distinct_ticker_dates = _fetch_confidence_hit_pairs(conn, horizon)
    n = len(pairs)
    result = {
        "horizon": horizon, "n": n, "n_distinct_ticker_dates": n_distinct_ticker_dates,
        "sufficient": n_distinct_ticker_dates >= MIN_SAMPLES_PROVISIONAL,
        "reliable": n_distinct_ticker_dates >= MIN_SAMPLES_RELIABLE,
    }
    if not result["sufficient"]:
        return result

    from sklearn.isotonic import IsotonicRegression

    x = np.array([p[0] for p in pairs], dtype=float)
    y = np.array([p[1] for p in pairs], dtype=float)

    # increasing='auto' lets the fit find a DECREASING relationship too --
    # increasing=True (the previous setting) forces a monotonic-increasing
    # shape no matter what the data says, which would silently flatten a
    # genuinely adverse (higher-stated-confidence, lower-hit-rate)
    # relationship into a flat curve instead of showing it.
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip", increasing="auto")
    iso.fit(x, y)

    breakpoints_x = iso.X_thresholds_.tolist()
    breakpoints_y = iso.y_thresholds_.tolist()

    # A raw threshold on the isotonic curve's spread is a poor test for
    # "is there really a relationship" -- isotonic regression will show SOME
    # apparent curvature from pure sampling noise even at n=200, especially
    # near sparsely-populated tails. A significance test on the correlation
    # is the honest way to say whether the data can distinguish this from
    # noise at the current sample size (see the module-level comment above
    # for why this p-value is still anti-conservative even after
    # de-duplication, and why Holm-Bonferroni is applied across horizons
    # in main() below).
    if len(set(x)) > 1:
        corr, pvalue = scipy_stats.pearsonr(x, y)
        corr, pvalue = float(corr), float(pvalue)
    else:
        corr, pvalue = float("nan"), float("nan")

    result.update({
        "overall_hit_rate": float(y.mean()),
        "correlation": corr,
        "p_value": pvalue,
        # Default; _holm_bonferroni (called once per batch of horizons in
        # main()) overrides this for every result with a testable p-value.
        # Left False here so a horizon with too little confidence variance
        # to test (nan p-value) never reads as "significant" by omission.
        "significant": False,
        "breakpoints_x": breakpoints_x,
        "breakpoints_y": breakpoints_y,
    })
    return result


def _holm_bonferroni(results, alpha=0.05):
    """Adjust significance across the horizons actually tested together in
    one run, in place on `results` (sets result["significant"] on each
    sufficient result). Without this, testing 4 horizons at nominal
    alpha=0.05 gives roughly a 1-(1-0.05)^4 ~= 19% chance one looks
    significant purely by chance, with no correction (the previous
    version marked "significant" straight off the raw per-horizon
    p-value)."""
    testable = [r for r in results if r.get("sufficient") and not np.isnan(r.get("p_value", float("nan")))]
    testable.sort(key=lambda r: r["p_value"])
    m = len(testable)
    for i, r in enumerate(testable):
        threshold = alpha / (m - i)
        r["significant"] = r["p_value"] < threshold
        if not r["significant"]:
            # Holm-Bonferroni stops rejecting once one comparison fails --
            # everything after it in sorted order is also not significant.
            for later in testable[i:]:
                later["significant"] = False
            break


def save_calibration(result):
    os.makedirs(CALIBRATION_DIR, exist_ok=True)
    path = os.path.join(CALIBRATION_DIR, f"pattern_h{result['horizon']}.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    return path


def load_calibration(horizon):
    """Saved calibration dict for this horizon, or None if it hasn't been
    fit yet, or was below the minimum sample size when last fit."""
    path = os.path.join(CALIBRATION_DIR, f"pattern_h{horizon}.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        data = json.load(f)
    if not data.get("sufficient"):
        return None
    return data


def remap_confidence(raw_confidence, horizon=10):
    """Map a raw 0-100 pattern confidence through the saved isotonic
    calibration curve for this horizon. Returns None if no calibration is
    available yet (nothing fit, or too little data at fit time) -- callers
    must handle that by falling back to the uncalibrated quality tier, not
    by guessing a number."""
    data = load_calibration(horizon)
    if data is None:
        return None
    calibrated_pct = float(np.interp(raw_confidence, data["breakpoints_x"], data["breakpoints_y"])) * 100
    return {"calibrated_pct": calibrated_pct, "n": data["n"], "reliable": data["reliable"],
            "significant": data["significant"]}


def main():
    conn = _connect()
    print("=== Pattern confidence calibration ===\n")

    results = [calibrate_horizon(conn, h) for h in HORIZONS]
    # All 4 horizons are tested together in this one run -- correct for
    # that multiple-comparison exposure BEFORE printing/saving anything,
    # not per-horizon in isolation (see calibrate_horizon's docstring:
    # untested, 4 horizons at nominal alpha=0.05 gives ~19% odds one
    # looks significant by chance alone).
    _holm_bonferroni(results)

    any_sufficient = False
    for result in results:
        h = result["horizon"]
        if not result["sufficient"]:
            print(f"+{h} bars: {result['n_distinct_ticker_dates']} distinct ticker-dates "
                  f"({result['n']} pattern observations) -- below the {MIN_SAMPLES_PROVISIONAL} "
                  f"distinct-ticker-date minimum to attempt a fit (need "
                  f"{MIN_SAMPLES_PROVISIONAL - result['n_distinct_ticker_dates']} more).")
            continue
        any_sufficient = True
        path = save_calibration(result)
        reliability = ("RELIABLE" if result["reliable"]
                        else f"PROVISIONAL -- need {MIN_SAMPLES_RELIABLE}+ distinct ticker-dates for real reliability")
        print(f"+{h} bars: {result['n_distinct_ticker_dates']} distinct ticker-dates "
              f"({result['n']} pattern observations) [{reliability}]")
        print(f"  overall hit rate: {result['overall_hit_rate'] * 100:.1f}%")
        print(f"  correlation(confidence, hit): {result['correlation']:.3f} "
              f"(p={result['p_value']:.3f}, Holm-Bonferroni-adjusted across {len(HORIZONS)} horizons)")
        if result["significant"]:
            print(f"  Statistically distinguishable from zero at this sample size -- "
                  f"calibration curve saved to {path}")
        else:
            print(f"  NOT statistically distinguishable from zero at this sample size. "
                  f"Saved to {path} for tracking, but treat it as noise, not signal, "
                  "until the correlation strengthens or n grows -- remap_confidence() "
                  "still flags it as not-significant.")
        print()

    if not any_sufficient:
        print(f"No horizon has {MIN_SAMPLES_PROVISIONAL}+ distinct ticker-dates yet. Keep running "
              "daily_ta_report.py and score_outcomes.py, and re-run this monthly "
              "(per F36's schedule) until enough accumulates.")
    conn.close()


if __name__ == "__main__":
    main()
