#!/usr/bin/env python3
"""
Regression test suite (F39). Two parts:

  A. Golden-file pattern fixtures -- hand-built synthetic OHLCV for each
     pattern family, with a known expected detection. Catches "I changed a
     threshold and a pattern that used to detect correctly no longer does"
     regressions immediately, rather than relying on someone noticing
     during a live run days later.

  B. Synthetic-noise false-positive thresholds -- the same methodology as
     the independent audit's t2_patterns.py / t3_confluence.py (pure
     random-walk data, which by construction contains no real patterns),
     but as hard PASS/FAIL assertions instead of informational prints.
     Thresholds are set with margin above the last verified numbers (see
     comments at each assertion) so a real regression trips this suite
     before it ships, not after.

Run directly: `python test_regression.py`. Exits non-zero on any failure,
so it can be used as a pre-commit/CI gate.
"""

import sys
from collections import Counter

import numpy as np
import pandas as pd

import chart_patterns as cp
import daily_ta_report as R

RESULTS = []


def check(label, cond):
    RESULTS.append((label, cond))
    print(("PASS" if cond else "FAIL"), "-", label)


def make_df(prices, vols=None, wick=0.6):
    n = len(prices)
    dates = pd.bdate_range("2026-01-01", periods=n)
    vols = vols or [1_000_000] * n
    prices = [float(p) for p in prices]  # force float64 columns so later
    # .iloc float assignments (e.g. a 100.05 close) never hit pandas'
    # LossySetitemError against an int64 column inferred from all-integer input.
    return pd.DataFrame({
        "Open": prices, "High": [p + wick for p in prices],
        "Low": [p - wick for p in prices], "Close": prices, "Volume": vols,
    }, index=dates)


# ===========================================================================
# PART A: Golden-file pattern fixtures
# ===========================================================================

def test_ascending_triangle():
    prices, vols = [], []
    top, low = 110, 95
    for i in range(24):
        frac = i / 23
        cur_low = low + frac * (top - 2 - low)
        prices.append(top - 0.2 if i % 4 == 0 else cur_low if i % 4 == 2 else (top + cur_low) / 2)
        vols.append(int(2_000_000 * (1 - 0.6 * frac)))
    m = cp.classify_trendline_pattern(make_df(prices, vols), window=24)
    check("Ascending Triangle detected", m is not None and m.name == "Ascending Triangle")
    check("Ascending Triangle confidence is a real number", m is not None and not np.isnan(m.confidence))


def test_descending_triangle():
    # Mirror of the ascending triangle: flat bottom (support), falling highs.
    prices, vols = [], []
    bottom, high = 95, 110
    for i in range(24):
        frac = i / 23
        cur_high = high - frac * (high - bottom - 2)
        prices.append(bottom + 0.2 if i % 4 == 0 else cur_high if i % 4 == 2 else (bottom + cur_high) / 2)
        vols.append(int(2_000_000 * (1 - 0.6 * frac)))
    m = cp.classify_trendline_pattern(make_df(prices, vols), window=24)
    check("Descending Triangle detected", m is not None and m.name == "Descending Triangle")


def test_rectangle():
    # Pole up, then a short FLAT (non-drifting) oscillating consolidation.
    # The "+ 0.002*i" is a negligible tiebreaker (nowhere near enough to
    # affect the flat/rising/falling classification) -- a pure integer-
    # sampled sine with period 6 produces exact-value plateaus (sin(60deg)
    # == sin(120deg)), which breaks find_swings' "unique max in window"
    # requirement and silently produces zero pivots.
    baseline = [95.0] * 15
    pole = list(np.linspace(100, 130, 10))
    consolidation = [130 + 1.8 * np.sin(2 * np.pi * i / 6) + 0.002 * i for i in range(18)]
    df = make_df(baseline + pole + consolidation)
    m = cp.classify_trendline_pattern(df, window=20, swing_lookback=2)
    check("Rectangle detected", m is not None and m.name == "Rectangle")


def test_bull_flag():
    baseline = [95.0] * 15
    pole = list(np.linspace(100, 130, 10))
    flag = [130 - i * 0.3 + 1.8 * np.sin(2 * np.pi * i / 6) for i in range(18)]
    df = make_df(baseline + pole + flag)
    m = cp.classify_trendline_pattern(df, window=20, swing_lookback=2)
    check("Bull Flag detected", m is not None and m.name == "Bull Flag")


def test_bear_flag():
    baseline = [130.0] * 15
    pole = list(np.linspace(125, 95, 10))
    flag = [95 + i * 0.3 + 1.8 * np.sin(2 * np.pi * i / 6) for i in range(18)]
    df = make_df(baseline + pole + flag)
    m = cp.classify_trendline_pattern(df, window=20, swing_lookback=2)
    check("Bear Flag detected", m is not None and m.name == "Bear Flag")


def test_double_top():
    prices = [100, 103, 106, 109, 112, 115, 117, 119, 120, 118, 115, 111, 107, 104, 102, 101,
              103, 106, 109, 112, 115, 118, 119.5, 118, 114, 110, 106, 102, 99, 97, 96, 95, 94]
    m = cp.classify_peak_trough_pattern(make_df(prices), lookback_bars=len(prices))
    check("Double Top detected", m is not None and m.name == "Double Top")


def test_double_bottom():
    prices = [100, 103, 106, 109, 112, 115, 117, 119, 120, 118, 115, 111, 107, 104, 102, 101,
              103, 106, 109, 112, 115, 118, 119.5, 118, 114, 110, 106, 102, 99, 97, 96, 95, 94]
    mirrored = [200 - p for p in prices]  # reflect double-top shape into a double-bottom
    m = cp.classify_peak_trough_pattern(make_df(mirrored), lookback_bars=len(mirrored))
    check("Double Bottom detected", m is not None and m.name == "Double Bottom")


def test_head_and_shoulders():
    prices = [100, 102, 104, 106, 104, 102, 100, 98, 96, 98, 100, 103, 106, 110, 114, 118, 122,
              120, 116, 112, 108, 104, 100, 98, 96, 98, 101, 104, 107, 109, 108, 106, 104, 103,
              102, 101, 100, 99, 98, 97]
    m = cp.classify_peak_trough_pattern(make_df(prices), lookback_bars=len(prices))
    check("Head and Shoulders detected", m is not None and m.name == "Head and Shoulders")


def test_rounded_bottom():
    prices = [100 + 0.03 * (i - 20) ** 2 for i in range(40)]
    m = cp.detect_rounded_pattern(make_df(prices), window=40)
    check("Rounded Bottom detected", m is not None and m.name == "Rounded Bottom")


def test_hammer_after_downtrend():
    down = list(np.linspace(120, 100, 25))
    df = make_df(down)
    o, c = df.columns.get_loc("Open"), df.columns.get_loc("Close")
    h, l = df.columns.get_loc("High"), df.columns.get_loc("Low")
    df.iloc[-1, o] = 100.2
    df.iloc[-1, c] = 100.6
    df.iloc[-1, h] = 100.65
    df.iloc[-1, l] = 97.0
    m = cp.detect_pin_bar_patterns(df)
    check("Hammer detected after downtrend", m is not None and m.name == "Hammer")


def test_bearish_engulfing():
    df = make_df([95, 96, 97, 100, 101], wick=0.5)
    # Overwrite last two bars: yday bullish (100->110), today bearish, fully engulfing.
    o, c = df.columns.get_loc("Open"), df.columns.get_loc("Close")
    h, l = df.columns.get_loc("High"), df.columns.get_loc("Low")
    df.iloc[-2, [o, h, l, c]] = [100, 110.5, 99.5, 110]
    df.iloc[-1, [o, h, l, c]] = [112, 112.5, 97.5, 98]
    m = cp.detect_engulfing(df)
    check("Bearish Engulfing detected", m is not None and m.name == "Bearish Engulfing")


def test_doji():
    df = make_df([100, 100, 100, 100, 100])
    o, c = df.columns.get_loc("Open"), df.columns.get_loc("Close")
    h, l = df.columns.get_loc("High"), df.columns.get_loc("Low")
    df.iloc[-1, [o, h, l, c]] = [100.0, 105.0, 95.0, 100.05]
    m = cp.detect_doji(df)
    check("Doji detected", m is not None and m.name == "Doji")


def test_marubozu():
    df = make_df([100, 100, 100, 100, 100])
    o, c = df.columns.get_loc("Open"), df.columns.get_loc("Close")
    h, l = df.columns.get_loc("High"), df.columns.get_loc("Low")
    df.iloc[-1, [o, h, l, c]] = [100.0, 110.1, 99.9, 110.0]
    m = cp.detect_marubozu(df)
    check("Marubozu detected", m is not None and m.name == "Marubozu" and m.bias == "Bullish")


def test_three_white_soldiers():
    df = make_df([100, 99, 101, 100.5, 102, 101.5])
    df.iloc[-3] = [100.0, 103.5, 99.8, 103.2, 1_000_000]
    df.iloc[-2] = [101.5, 106.5, 101.2, 106.2, 1_000_000]
    df.iloc[-1] = [103.5, 109.5, 103.2, 109.2, 1_000_000]
    m = cp.detect_three_soldiers_crows(df)
    check("Three White Soldiers detected", m is not None and m.name == "Three White Soldiers")


def test_three_black_crows():
    df = make_df([100, 101, 99, 100.5, 98, 98.5])
    df.iloc[-3] = [103.5, 103.8, 99.8, 100.0, 1_000_000]
    df.iloc[-2] = [101.2, 101.5, 96.8, 97.5, 1_000_000]
    df.iloc[-1] = [98.5, 98.8, 93.8, 94.5, 1_000_000]
    m = cp.detect_three_soldiers_crows(df)
    check("Three Black Crows detected", m is not None and m.name == "Three Black Crows")


def test_confidence_floor_filters_low_quality():
    matches = [cp.PatternMatch("Test", "Continuation", "Neutral", 10.0, "2026-01-01", "Forming", "N/A", "x")]
    filtered = [m for m in matches if m.confidence >= cp.CONFIDENCE_FLOOR]
    check("Confidence floor filters a low-quality synthetic match", len(filtered) == 0)


# ===========================================================================
# PART B: Synthetic-noise false-positive thresholds
# ===========================================================================
# Same construction as the independent audit's t2_patterns.py/t3_confluence.py:
# pure zero/low-drift geometric Brownian motion, which by construction
# contains no real chart patterns or directional edge. A fixed seed range
# keeps this deterministic across runs. Thresholds carry margin above the
# last verified numbers (see each comment) so normal noise doesn't flake
# the suite -- only a real regression should trip these.

def _make_noise_df(n=300, seed=0, drift=0.0, vol=0.02):
    rng = np.random.default_rng(seed)
    r = rng.normal(drift, vol, n)
    close = 100 * np.exp(np.cumsum(r))
    op = np.r_[close[0], close[:-1]]
    hi = np.maximum(op, close) * (1 + np.abs(rng.normal(0, vol / 2, n)))
    lo = np.minimum(op, close) * (1 - np.abs(rng.normal(0, vol / 2, n)))
    return pd.DataFrame({
        "Open": op, "High": hi, "Low": lo, "Close": close,
        "Volume": rng.lognormal(15, 0.4, n),
    }, index=pd.bdate_range("2024-01-01", periods=n))


def test_noise_false_positive_rate(n_trials=200):
    total_patterns = 0
    confidences = []
    confluence_labels = Counter()
    for s in range(n_trials):
        df = _make_noise_df(seed=s + 1000)
        matches = cp.detect_all(df)
        total_patterns += len(matches)
        confidences.extend(m.confidence for m in matches)
        confluence = R.compute_confluence(df, matches)
        if confluence["sufficient_evidence"] and abs(confluence["net"]) >= 0.75:
            confluence_labels["confluence"] += 1
        elif confluence["sufficient_evidence"] and abs(confluence["net"]) >= 0.15:
            confluence_labels["lean"] += 1
        else:
            confluence_labels["no_call"] += 1

    patterns_per_report = total_patterns / n_trials
    mean_confidence = float(np.mean(confidences)) if confidences else 0.0
    false_confluence_rate = confluence_labels["confluence"] / n_trials

    print(f"  [noise n={n_trials}] patterns/report={patterns_per_report:.2f}, "
          f"mean confidence={mean_confidence:.1f}%, false confluence rate={false_confluence_rate:.1%}")

    # Last verified: 1.90 patterns/report (400 trials, post-F22/F24/F25) -- margin to 2.5.
    check(f"Noise patterns/report stays below 2.5 (got {patterns_per_report:.2f})",
          patterns_per_report < 2.5)
    # Last verified: 70.9% mean confidence on noise -- margin to 80%.
    check(f"Noise mean confidence stays below 80% (got {mean_confidence:.1f}%)",
          mean_confidence < 80.0)
    # Last verified: 38.7% false "confluence" rate on noise (post-F23) -- margin to 50%.
    check(f"Noise false-confluence rate stays below 50% (got {false_confluence_rate:.1%})",
          false_confluence_rate < 0.50)


def main():
    print("=== Part A: golden-file pattern fixtures ===")
    test_ascending_triangle()
    test_descending_triangle()
    test_rectangle()
    test_bull_flag()
    test_bear_flag()
    test_double_top()
    test_double_bottom()
    test_head_and_shoulders()
    test_rounded_bottom()
    test_hammer_after_downtrend()
    test_bearish_engulfing()
    test_doji()
    test_marubozu()
    test_three_white_soldiers()
    test_three_black_crows()
    test_confidence_floor_filters_low_quality()

    print("\n=== Part B: synthetic-noise false-positive thresholds ===")
    test_noise_false_positive_rate()

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
