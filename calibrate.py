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
MIN_SAMPLES_PROVISIONAL = 30
MIN_SAMPLES_RELIABLE = 500
HORIZONS = (1, 5, 10, 20)


def _connect():
    return sqlite3.connect(prediction_log.DB_PATH)


def _fetch_confidence_hit_pairs(conn, horizon):
    return conn.execute(
        "SELECT p.confidence, po.direction_hit "
        "FROM pattern_outcomes po "
        "JOIN patterns p ON p.run_id = po.run_id AND p.pattern_name = po.pattern_name "
        "WHERE po.horizon_bars = ? AND po.direction_hit IS NOT NULL",
        (horizon,)).fetchall()


def calibrate_horizon(conn, horizon):
    pairs = _fetch_confidence_hit_pairs(conn, horizon)
    n = len(pairs)
    result = {
        "horizon": horizon, "n": n,
        "sufficient": n >= MIN_SAMPLES_PROVISIONAL,
        "reliable": n >= MIN_SAMPLES_RELIABLE,
    }
    if not result["sufficient"]:
        return result

    from sklearn.isotonic import IsotonicRegression

    x = np.array([p[0] for p in pairs], dtype=float)
    y = np.array([p[1] for p in pairs], dtype=float)

    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip", increasing=True)
    iso.fit(x, y)

    breakpoints_x = iso.X_thresholds_.tolist()
    breakpoints_y = iso.y_thresholds_.tolist()

    # A raw threshold on the isotonic curve's spread is a poor test for
    # "is there really a relationship" -- isotonic regression will show SOME
    # apparent curvature from pure sampling noise even at n=200, especially
    # near sparsely-populated tails. A significance test on the correlation
    # is the honest way to say whether the data can distinguish this from
    # noise at the current sample size.
    if len(set(x)) > 1:
        corr, pvalue = scipy_stats.pearsonr(x, y)
        corr, pvalue = float(corr), float(pvalue)
    else:
        corr, pvalue = float("nan"), float("nan")
    significant = (not np.isnan(pvalue)) and pvalue < 0.05

    result.update({
        "overall_hit_rate": float(y.mean()),
        "correlation": corr,
        "p_value": pvalue,
        "significant": significant,
        "breakpoints_x": breakpoints_x,
        "breakpoints_y": breakpoints_y,
    })
    return result


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
    any_sufficient = False
    for h in HORIZONS:
        result = calibrate_horizon(conn, h)
        if not result["sufficient"]:
            print(f"+{h} bars: n={result['n']} scored calls -- below the "
                  f"{MIN_SAMPLES_PROVISIONAL} minimum to attempt a fit "
                  f"(need {MIN_SAMPLES_PROVISIONAL - result['n']} more).")
            continue
        any_sufficient = True
        path = save_calibration(result)
        reliability = ("RELIABLE" if result["reliable"]
                        else f"PROVISIONAL -- need {MIN_SAMPLES_RELIABLE}+ for real reliability")
        print(f"+{h} bars: n={result['n']} [{reliability}]")
        print(f"  overall hit rate: {result['overall_hit_rate'] * 100:.1f}%")
        print(f"  correlation(confidence, hit): {result['correlation']:.3f} "
              f"(p={result['p_value']:.3f})")
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
        print(f"No horizon has {MIN_SAMPLES_PROVISIONAL}+ scored calls yet. Keep running "
              "daily_ta_report.py and score_outcomes.py, and re-run this monthly "
              "(per F36's schedule) until enough accumulates.")
    conn.close()


if __name__ == "__main__":
    main()
