#!/usr/bin/env python3
"""
Chart pattern detection for the daily technical analysis report.

Covers three families of patterns:
  - Continuation: triangles, wedges, rectangles, channels, flags, pennants
  - Reversal: double/triple top/bottom, head & shoulders, rounded top/bottom,
    broadening formation
  - Candlestick: single/multi-bar Japanese candlestick shapes

Every detector returns zero or more PatternMatch records with a name, bias,
confidence score (0-100), formation date, status, and price objective where
one is meaningful. This module is self-contained (duplicates the small swing
/ATR primitives from daily_ta_report.py rather than importing them, since
daily_ta_report.py imports this module and a two-way import would cycle).
"""

from collections import Counter
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats


@dataclass
class PatternMatch:
    name: str
    category: str        # "Continuation" | "Reversal" | "Candlestick"
    bias: str             # "Bullish" | "Bearish" | "Neutral"
    confidence: float     # 0-100
    formed_date: str      # ISO date of the bar that completes the pattern
    status: str            # "Forming" | "Confirmed" | "Invalidated"
    price_target: str     # formatted price, or "N/A"
    detail: str


CONTINUATION_NAMES = {
    "Ascending Triangle", "Descending Triangle", "Symmetrical Triangle", "Pennant",
    "Rectangle", "Ascending Channel", "Descending Channel", "Horizontal Channel",
    "Bull Flag", "Bear Flag",
}
# Rising/Falling Wedge deliberately excluded: standard TA more commonly
# classifies wedges as reversal patterns (this classifier has no reliable
# way to tell continuation-in-a-trend from reversal-after-a-trend without
# higher-timeframe context it doesn't have access to), so they fall through
# to "Reversal" below.

CONFIDENCE_FLOOR = 40.0


# The 0-100 confidence score is a heuristic blend of fit quality, pivot
# count, touch count, and volume confirmation -- it is NOT a calibrated
# probability. An independent audit (see t2_patterns.py) measured it against
# synthetic no-pattern data: mean confidence on pure noise was 74.6%, and
# patterns with essentially no linear fit (r^2 < 0.35) still averaged 62.2%.
# Correlation with actual 10-bar forward direction was -0.06, statistically
# indistinguishable from zero. Displaying it as a percentage implies a win
# rate that has never been measured, so callers should show this ordinal
# tier instead of the raw number.
def confidence_tier(confidence):
    if confidence >= 75:
        return "Strong"
    if confidence >= 55:
        return "Moderate"
    return "Weak"


def _fmt_price(value):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "N/A"
    return f"${value:.2f}"


# ---------------------------------------------------------------------------
# Shared primitives (duplicated from daily_ta_report.py to avoid a circular
# import — these are small and pure, so keeping this module self-contained
# is simpler than a deferred/indirect import).
# ---------------------------------------------------------------------------

def _screen_outlier_ranges(df, max_range_atr_mult=5.0):
    """Kept identical to daily_ta_report.py's _screen_outlier_ranges -- see
    its docstring for why bars with an extreme range get capped before
    pivot detection."""
    highs = df["High"].values.copy()
    lows = df["Low"].values.copy()
    n = len(df)
    if n < 15:
        return highs, lows
    close = df["Close"].values
    prev_close = np.r_[close[0], close[:-1]]
    tr = np.maximum(highs - lows, np.maximum(np.abs(highs - prev_close), np.abs(lows - prev_close)))
    atr_est = pd.Series(tr).rolling(14, min_periods=5).mean().bfill().values
    bar_range = highs - lows
    safe_atr = np.where(atr_est > 0, atr_est, np.inf)
    outliers = bar_range > max_range_atr_mult * safe_atr
    if outliers.any():
        cap = 2 * atr_est
        highs = np.where(outliers, np.minimum(highs, close + cap), highs)
        lows = np.where(outliers, np.maximum(lows, close - cap), lows)
    return highs, lows


def find_swings(df, lookback=3):
    """Fractal swing highs/lows: a bar's high/low exceeds `lookback` bars on
    each side. Returns (swing_highs, swing_lows) as (pos, price) lists,
    positions relative to `df`."""
    highs, lows = _screen_outlier_ranges(df)
    n = len(df)
    swing_highs = []
    swing_lows = []
    for i in range(lookback, n - lookback):
        window_high = highs[i - lookback:i + lookback + 1]
        if highs[i] == window_high.max() and np.sum(window_high == highs[i]) == 1:
            swing_highs.append((i, highs[i]))
        window_low = lows[i - lookback:i + lookback + 1]
        if lows[i] == window_low.min() and np.sum(window_low == lows[i]) == 1:
            swing_lows.append((i, lows[i]))
    return swing_highs, swing_lows


def _wilders_smooth(series, period):
    """Kept identical to daily_ta_report.py's wilders_smooth -- see its
    docstring for why leading NaN is stripped (not averaged as if valid)
    and interior NaN is zero-filled (not dropped, which would shift every
    later value's place in the recursion)."""
    values = series.values.astype(float)
    n = len(values)
    first_valid = 0
    while first_valid < n and np.isnan(values[first_valid]):
        first_valid += 1
    usable = np.nan_to_num(values[first_valid:], nan=0.0)
    result = np.full(n, np.nan)
    if len(usable) < period:
        return pd.Series(result, index=series.index)
    prev = np.mean(usable[:period])
    result[first_valid + period - 1] = prev
    for i in range(period, len(usable)):
        prev = (prev * (period - 1) + usable[i]) / period
        result[first_valid + i] = prev
    return pd.Series(result, index=series.index)


def calc_atr(df, period=14):
    prev_close = df["Close"].shift(1)
    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - prev_close).abs(),
        (df["Low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    return _wilders_smooth(tr, period)


def _merge_pivots(swing_highs, swing_lows):
    return sorted(
        [(i, p, "H") for i, p in swing_highs] + [(i, p, "L") for i, p in swing_lows],
        key=lambda x: x[0],
    )


def _fit_line(points):
    """points: list of (index, price). Returns (slope, intercept, r2) or None.
    A perfectly flat (or 2-point) set of pivots has an undefined correlation
    coefficient in scipy (0/0) even though it's a perfect fit by construction,
    so those cases are special-cased to r2=1.0 instead of propagating NaN."""
    if len(points) < 2:
        return None
    xs = np.array([p[0] for p in points], dtype=float)
    ys = np.array([p[1] for p in points], dtype=float)
    if np.allclose(ys, ys[0]):
        return 0.0, float(ys[0]), 1.0
    res = scipy_stats.linregress(xs, ys)
    r2 = res.rvalue ** 2
    if np.isnan(r2):
        r2 = 1.0 if len(points) == 2 else 0.0
    return res.slope, res.intercept, r2


# ---------------------------------------------------------------------------
# Engine 1 — trendline patterns (triangles, wedges, rectangles, channels,
# flags, pennants, broadening formation). All are two trendlines fit through
# swing pivots, differing only in slope combination + duration + whether a
# sharp "pole" move precedes them.
# ---------------------------------------------------------------------------

def classify_trendline_pattern(df, window=60, swing_lookback=3, min_pivots_per_line=2):
    window = min(window, len(df))
    min_bars = (swing_lookback * 2 + 1) * 2
    if window < min_bars:
        return None

    sub = df.tail(window)
    current_price = float(df["Close"].iloc[-1])

    # Computed early (moved up from where it used to sit, near the status
    # logic) so it's available to normalize the FLAT/pole thresholds below
    # by volatility rather than by a fixed percentage of price -- a fixed
    # %-of-price-per-bar threshold means the exact same slope counts as
    # "flat" for a calm ticker and "trending" for a volatile one, and vice
    # versa; comparing the slope to the ticker's own ATR is transferable
    # across tickers instead of tuned to whichever ones happened to be
    # tested during development.
    atr_val = float(calc_atr(df, 14).iloc[-1])
    atr_val = atr_val if not np.isnan(atr_val) else 0.0

    swing_highs, swing_lows = find_swings(sub, lookback=swing_lookback)
    # Flags/pennants are inherently brief and structurally can't offer more
    # than 2 pivots per line -- the caller passes min_pivots_per_line=2 for
    # that short pass. For the standard (longer-window) pass, the caller
    # requires 3+ so a 2-point "perfect fit" (r2=1.0 by construction, not by
    # evidence) can't anchor a pattern on its own.
    if len(swing_highs) < min_pivots_per_line or len(swing_lows) < min_pivots_per_line:
        return None

    upper_pts = swing_highs[-4:]
    lower_pts = swing_lows[-4:]
    upper_fit = _fit_line(upper_pts)
    lower_fit = _fit_line(lower_pts)
    if upper_fit is None or lower_fit is None:
        return None
    slope_u, intercept_u, r2_u = upper_fit
    slope_l, intercept_l, r2_l = lower_fit

    # Reject outright rather than just scoring low: a line through noise
    # (r2 well below what a real trend would show) shouldn't be presented
    # as a pattern at any confidence, since the whole shape classification
    # below depends on trusting these two fitted slopes.
    fit_score = (r2_u + r2_l) / 2
    if fit_score < 0.5:
        return None

    # Slope in ATR-per-bar units, not %-of-price-per-bar -- see comment
    # above. FLAT_ATR=0.05 means a line drifting less than 5% of one ATR
    # per bar counts as flat; tuned against the same synthetic
    # triangle/wedge/flag fixtures used elsewhere in this codebase's tests.
    FLAT_ATR = 0.05
    if atr_val > 0:
        norm_u = slope_u / atr_val
        norm_l = slope_l / atr_val
    else:
        norm_u = norm_l = 0.0

    def dirn(s):
        if s > FLAT_ATR:
            return "rising"
        if s < -FLAT_ATR:
            return "falling"
        return "flat"

    dir_u, dir_l = dirn(norm_u), dirn(norm_l)

    n = len(sub)
    last_pos = n - 1
    upper_now = slope_u * last_pos + intercept_u
    lower_now = slope_l * last_pos + intercept_l
    width_now = upper_now - lower_now
    if width_now <= 0:
        return None

    first_pos = min(upper_pts[0][0], lower_pts[0][0])
    upper_first = slope_u * first_pos + intercept_u
    lower_first = slope_l * first_pos + intercept_l
    width_first = upper_first - lower_first
    if width_first <= 0:
        return None

    converging = width_now < width_first * 0.85
    diverging = width_now > width_first * 1.15
    parallel = not converging and not diverging

    # Pole size in ATR units, not a fixed +/-8% of price -- 8% is a huge
    # move for a low-volatility name and unremarkable noise for a high-beta
    # one, so a fixed percentage threshold isn't transferable across
    # tickers. POLE_ATR_MULT=2.0 means the pre-window move must exceed 2
    # ATRs to count as a genuine pole.
    POLE_ATR_MULT = 2.0
    pole_dir = None
    pre_start = len(df) - window - 10
    if pre_start >= 0 and atr_val > 0:
        window_start_price = df["Close"].iloc[len(df) - window]
        pre_price = df["Close"].iloc[pre_start]
        pre_move_atr = (window_start_price - pre_price) / atr_val
        if pre_move_atr > POLE_ATR_MULT:
            pole_dir = "up"
        elif pre_move_atr < -POLE_ATR_MULT:
            pole_dir = "down"

    short = window <= 20
    poled = pole_dir is not None

    name = None
    bias = "Neutral"
    breakout_dir = None

    if dir_u == "flat" and dir_l == "rising" and converging:
        name, bias, breakout_dir = "Ascending Triangle", "Bullish", "up"
    elif dir_u == "falling" and dir_l == "flat" and converging:
        name, bias, breakout_dir = "Descending Triangle", "Bearish", "down"
    elif dir_u == "falling" and dir_l == "rising" and converging:
        if poled and short:
            name = "Pennant"
            bias = "Bullish" if pole_dir == "up" else "Bearish"
            breakout_dir = pole_dir
        else:
            name, bias, breakout_dir = "Symmetrical Triangle", "Neutral", None
    elif dir_u == "rising" and dir_l == "rising" and converging and norm_u < norm_l:
        name, bias, breakout_dir = "Rising Wedge", "Bearish", "down"
    elif dir_u == "falling" and dir_l == "falling" and converging and abs(norm_l) < abs(norm_u):
        name, bias, breakout_dir = "Falling Wedge", "Bullish", "up"
    elif parallel and dir_u == "flat" and dir_l == "flat":
        name = "Rectangle" if (poled and short) else "Horizontal Channel"
        bias, breakout_dir = "Neutral", None
    elif parallel and dir_u == "rising" and dir_l == "rising":
        if poled and short and pole_dir == "down":
            # Down pole + a small upward-drifting (counter-trend) consolidation
            name, bias, breakout_dir = "Bear Flag", "Bearish", "down"
        else:
            name, bias, breakout_dir = "Ascending Channel", "Bullish", "up"
    elif parallel and dir_u == "falling" and dir_l == "falling":
        if poled and short and pole_dir == "up":
            # Up pole + a small downward-drifting (counter-trend) consolidation
            name, bias, breakout_dir = "Bull Flag", "Bullish", "up"
        else:
            name, bias, breakout_dir = "Descending Channel", "Bearish", "down"
    elif diverging:
        name, bias, breakout_dir = "Broadening Formation", "Neutral", None
    else:
        return None

    def count_independent_touches(price_series, slope, intercept, exclude_positions, tol_frac=0.015):
        """Count bars NOT used to fit this line whose actual price still
        lands near it. The previous version checked the same points the
        line was least-squares fitted through -- a point can't fail to be
        near the line that was fitted to minimize its distance from it, so
        that wasn't corroboration, it was measuring the fit a second time
        and calling it "touches". This checks every OTHER bar in the
        window instead, which is real independent evidence."""
        cnt = 0
        for i, actual in enumerate(price_series):
            if i in exclude_positions:
                continue
            fit_val = slope * i + intercept
            if fit_val > 0 and abs(actual - fit_val) / fit_val <= tol_frac:
                cnt += 1
        return cnt

    upper_excl = {i for i, _ in upper_pts}
    lower_excl = {i for i, _ in lower_pts}
    touches = (count_independent_touches(sub["High"].values, slope_u, intercept_u, upper_excl)
               + count_independent_touches(sub["Low"].values, slope_l, intercept_l, lower_excl))
    # Normalized against how many non-fit bars were even available to touch,
    # not a fixed count -- a 20-bar window and a 90-bar window shouldn't be
    # held to the same raw touch count.
    touchable_bars = 2 * max(len(sub) - len(upper_excl) - len(lower_excl), 1)
    touch_score = min(touches / max(touchable_bars * 0.15, 1), 1.0)

    vol_slope = None
    if "Volume" in sub.columns and sub["Volume"].notna().sum() >= 3:
        vol_xs = np.arange(len(sub))
        vol_res = scipy_stats.linregress(vol_xs, sub["Volume"].values)
        vol_slope = vol_res.slope
    contracting_expected = name != "Broadening Formation"
    if vol_slope is not None:
        vol_conf = 1.0 if ((vol_slope < 0) == contracting_expected) else 0.5
    else:
        vol_conf = 0.5

    # A 2-point line fits "perfectly" (r2=1.0) by mathematical construction,
    # not because it reflects a validated trend -- pivot_score explicitly
    # rewards having more than the bare minimum pivots per line.
    pivot_count = len(upper_pts) + len(lower_pts)
    pivot_score = max(min((pivot_count - 4) / 4, 1.0), 0.0)

    # Confidence is MULTIPLICATIVE in fit_score, not additive: an r2=0.5
    # line (the minimum that survives the gate above) can reach at most 50%
    # of whatever the other factors would otherwise give, rather than fit
    # quality being just one of four equally-weighted terms that touch
    # count/volume/pivot count could offset on a genuinely bad fit.
    other_factors = 0.4 * touch_score + 0.3 * vol_conf + 0.3 * pivot_score
    confidence = round(fit_score * other_factors * 100, 1)

    atr_val = float(calc_atr(df, 14).iloc[-1])
    atr_val = atr_val if not np.isnan(atr_val) else 0.0
    buffer = 0.25 * atr_val

    status = "Forming"
    if breakout_dir == "up":
        if current_price > upper_now + buffer:
            status = "Confirmed"
        elif current_price < lower_now - buffer:
            status = "Invalidated"
    elif breakout_dir == "down":
        if current_price < lower_now - buffer:
            status = "Confirmed"
        elif current_price > upper_now + buffer:
            status = "Invalidated"
    else:
        if current_price > upper_now + buffer:
            status, bias, breakout_dir = "Confirmed", "Bullish", "up"
        elif current_price < lower_now - buffer:
            status, bias, breakout_dir = "Confirmed", "Bearish", "down"

    # width_first is a back-extrapolated gap between two FITTED lines, not
    # an observed pattern height -- when either line fits poorly, that
    # extrapolation is arbitrary, and adding it to spot can land a "target"
    # well outside any price the stock has actually traded at. Suppress the
    # target when the fit is too weak to trust the extrapolation, and cap
    # whatever remains at the 52-week range as a sanity bound.
    price_target = "N/A"
    if name != "Broadening Formation" and fit_score >= 0.5:
        raw_target = None
        if breakout_dir == "up":
            raw_target = upper_now + width_first
        elif breakout_dir == "down":
            raw_target = lower_now - width_first
        if raw_target is not None:
            year_window = df.tail(min(len(df), 252))
            year_high, year_low = float(year_window["High"].max()), float(year_window["Low"].min())
            capped_target = min(max(raw_target, year_low), year_high)
            price_target = _fmt_price(capped_target)

    formed_pos = max(upper_pts[-1][0], lower_pts[-1][0])
    formed_date = str(sub.index[formed_pos].date())

    # Show both widths: width_first is the pattern's height at its widest
    # point, which is what the measured-move price_target above is actually
    # derived from -- width_now (current, narrower for converging patterns)
    # is shown too since it's useful context but must not be confused with
    # the figure driving the target.
    detail = (f"Upper trendline {dir_u} (r²={r2_u:.2f}), lower trendline {dir_l} "
              f"(r²={r2_l:.2f}), pattern height {_fmt_price(width_first)} "
              f"(currently {_fmt_price(width_now)} wide)")

    return PatternMatch(
        name=name,
        category="Continuation" if name in CONTINUATION_NAMES else "Reversal",
        bias=bias,
        confidence=confidence,
        formed_date=formed_date,
        status=status,
        price_target=price_target,
        detail=detail,
    )


def classify_trendline_pattern_ensemble(df, windows=(40, 50, 60, 70, 80), swing_lookback=3,
                                         min_pivots_per_line=3, min_agreement=3):
    """Run classify_trendline_pattern across several window sizes and only
    report a pattern if the SAME name+bias shows up in at least
    `min_agreement` of them. Measured directly (see t2_patterns.py, section
    C): adding just ONE extra bar of leading history changed the detected
    pattern's name 18.7% of the time and flipped the directional bias
    13.9% of the time -- "which window did the code happen to use" was
    silently deciding part of the answer. This converts that brittleness
    into a measured stability score instead of hiding it: how many of the
    windows agree becomes part of the confidence, rather than reporting
    whichever single window was asked.

    Not used for flags/pennants/rectangles (see detect_all) -- those are
    inherently brief, so ensembling across 40-80 bar windows would just
    never detect them at all; they get one purpose-built short pass
    instead."""
    results = []
    for w in windows:
        m = classify_trendline_pattern(df, window=w, swing_lookback=swing_lookback,
                                        min_pivots_per_line=min_pivots_per_line)
        if m is not None:
            results.append(m)
    if not results:
        return None

    counts = Counter((m.name, m.bias) for m in results)
    (best_name, best_bias), agreement = counts.most_common(1)[0]
    if agreement < min_agreement:
        return None

    matching = [m for m in results if (m.name, m.bias) == (best_name, best_bias)]
    representative = max(matching, key=lambda m: m.confidence)
    agreement_rate = agreement / len(windows)
    # A pattern that only shows up in the bare minimum of windows shouldn't
    # get full credit for the representative window's own confidence -- the
    # agreement rate scales it down, floored at half so a bare-minimum
    # agreement doesn't zero out an otherwise well-fit pattern.
    blended_confidence = round(representative.confidence * (0.5 + 0.5 * agreement_rate), 1)
    detail = f"{representative.detail}; stable across {agreement}/{len(windows)} lookback windows"

    return PatternMatch(
        name=representative.name,
        category=representative.category,
        bias=representative.bias,
        confidence=blended_confidence,
        formed_date=representative.formed_date,
        status=representative.status,
        price_target=representative.price_target,
        detail=detail,
    )


# ---------------------------------------------------------------------------
# Engine 2 — peak/trough patterns (double/triple top/bottom, head &
# shoulders, inverse head & shoulders). All are sequences of 3 or 5
# alternating swing pivots differing only in the shape of the middle pivot(s).
# ---------------------------------------------------------------------------

def _build_pt_match(name, bias, shape_diff, neckline_diff, extreme_price, neckline,
                     atr_val, formed_pos, sub, current_price, breakout_dir,
                     pullback_depth=None, shoulder_tol=0.05, neckline_tol=0.04):
    shape_score = max(0.0, 1 - shape_diff / shoulder_tol)
    neckline_score = max(0.0, 1 - neckline_diff / neckline_tol)
    if pullback_depth is None:
        pullback_depth = abs(extreme_price - neckline) / max(atr_val, 1e-9)
    depth_score = min(pullback_depth / 3.0, 1.0)
    confidence = round(((shape_score + neckline_score + depth_score) / 3) * 100, 1)

    height = abs(extreme_price - neckline)
    buffer = 0.25 * atr_val
    status = "Forming"
    if breakout_dir == "down":
        if current_price < neckline - buffer:
            status = "Confirmed"
        elif current_price > extreme_price + buffer:
            status = "Invalidated"
        target = neckline - height
    else:
        if current_price > neckline + buffer:
            status = "Confirmed"
        elif current_price < extreme_price - buffer:
            status = "Invalidated"
        target = neckline + height

    formed_date = str(sub.index[formed_pos].date())
    detail = f"Neckline ~{_fmt_price(neckline)}, extreme {_fmt_price(extreme_price)}"

    return PatternMatch(
        name=name, category="Reversal", bias=bias, confidence=confidence,
        formed_date=formed_date, status=status,
        price_target=_fmt_price(target), detail=detail,
    )


def classify_peak_trough_pattern(df, lookback_bars=90, swing_lookback=3,
                                  shoulder_tol=0.05, neckline_tol=0.04):
    sub = df.tail(lookback_bars) if len(df) > lookback_bars else df
    if len(sub) < (swing_lookback * 2 + 1) * 3:
        return None
    swing_highs, swing_lows = find_swings(sub, lookback=swing_lookback)
    pivots = _merge_pivots(swing_highs, swing_lows)
    if len(pivots) < 3:
        return None

    current_price = float(df["Close"].iloc[-1])
    atr_val = float(calc_atr(df, 14).iloc[-1])
    atr_val = atr_val if not np.isnan(atr_val) else 0.0

    for size in (5, 3):
        if len(pivots) < size:
            continue
        window_pivots = pivots[-size:]
        kinds = [k for _, _, k in window_pivots]
        prices = [p for _, p, _ in window_pivots]
        idxs = [i for i, _, _ in window_pivots]
        mid = prices[len(prices) // 2] if size == 5 else None

        if kinds == ["H", "L", "H", "L", "H"] or kinds == ["H", "L", "H"]:
            outer = [prices[0], prices[-1]]
            outer_diff = abs(outer[0] - outer[1]) / max(outer)
            if size == 5:
                # necks are pivots[1] and pivots[3] specifically -- NOT
                # prices[1:-1], which for a 5-element window is [neck1,
                # head, neck2] (3 items). Indexing that slice as [0]/[1]
                # silently compares neck1 against the HEAD instead of
                # neck2, and averaging all 3 for "neckline" skews it
                # toward the head's price. This was a real, previously
                # undetected bug -- caught by test_regression.py -- that
                # made 5-pivot patterns (H&S, Inverse H&S, Triple Top/
                # Bottom) fail to match in almost every real case, since
                # the head is rarely within neckline_tol of either neckline
                # point.
                neck1, neck2 = prices[1], prices[3]
                if mid > outer[0] and mid > outer[1] and outer_diff <= shoulder_tol:
                    neckline = (neck1 + neck2) / 2
                    neckline_diff = abs(neck1 - neck2) / max(neck1, neck2)
                    if neckline_diff <= neckline_tol:
                        return _build_pt_match("Head and Shoulders", "Bearish", outer_diff,
                                                neckline_diff, mid, neckline, atr_val,
                                                idxs[-1], sub, current_price, "down",
                                                shoulder_tol=shoulder_tol, neckline_tol=neckline_tol)
                all_vals = [outer[0], mid, outer[1]]
                spread = (max(all_vals) - min(all_vals)) / max(all_vals)
                if spread <= shoulder_tol:
                    neckline = (neck1 + neck2) / 2
                    neckline_diff = abs(neck1 - neck2) / max(neck1, neck2)
                    if neckline_diff <= neckline_tol:
                        return _build_pt_match("Triple Top", "Bearish", spread, neckline_diff,
                                                max(all_vals), neckline, atr_val,
                                                idxs[-1], sub, current_price, "down",
                                                shoulder_tol=shoulder_tol, neckline_tol=neckline_tol)
            else:
                if outer_diff <= shoulder_tol:
                    neckline = prices[1]
                    pullback_depth = (outer[0] - neckline) / max(atr_val, 1e-9)
                    if pullback_depth >= 1.0:
                        return _build_pt_match("Double Top", "Bearish", outer_diff, 0.0,
                                                max(outer), neckline, atr_val,
                                                idxs[-1], sub, current_price, "down",
                                                pullback_depth=pullback_depth,
                                                shoulder_tol=shoulder_tol, neckline_tol=neckline_tol)

        if kinds == ["L", "H", "L", "H", "L"] or kinds == ["L", "H", "L"]:
            outer = [prices[0], prices[-1]]
            outer_diff = abs(outer[0] - outer[1]) / max(outer)
            if size == 5:
                neck1, neck2 = prices[1], prices[3]
                if mid < outer[0] and mid < outer[1] and outer_diff <= shoulder_tol:
                    neckline = (neck1 + neck2) / 2
                    neckline_diff = abs(neck1 - neck2) / max(neck1, neck2)
                    if neckline_diff <= neckline_tol:
                        return _build_pt_match("Inverse Head and Shoulders", "Bullish", outer_diff,
                                                neckline_diff, mid, neckline, atr_val,
                                                idxs[-1], sub, current_price, "up",
                                                shoulder_tol=shoulder_tol, neckline_tol=neckline_tol)
                all_vals = [outer[0], mid, outer[1]]
                spread = (max(all_vals) - min(all_vals)) / max(all_vals)
                if spread <= shoulder_tol:
                    neckline = (neck1 + neck2) / 2
                    neckline_diff = abs(neck1 - neck2) / max(neck1, neck2)
                    if neckline_diff <= neckline_tol:
                        return _build_pt_match("Triple Bottom", "Bullish", spread, neckline_diff,
                                                min(all_vals), neckline, atr_val,
                                                idxs[-1], sub, current_price, "up",
                                                shoulder_tol=shoulder_tol, neckline_tol=neckline_tol)
            else:
                if outer_diff <= shoulder_tol:
                    neckline = prices[1]
                    pullback_depth = (neckline - outer[0]) / max(atr_val, 1e-9)
                    if pullback_depth >= 1.0:
                        return _build_pt_match("Double Bottom", "Bullish", outer_diff, 0.0,
                                                min(outer), neckline, atr_val,
                                                idxs[-1], sub, current_price, "up",
                                                pullback_depth=pullback_depth,
                                                shoulder_tol=shoulder_tol, neckline_tol=neckline_tol)
    return None


# ---------------------------------------------------------------------------
# Engine 3 — rounded top/bottom (curve fit)
# ---------------------------------------------------------------------------

def detect_rounded_pattern(df, window=40):
    window = min(window, len(df))
    if window < 20:
        return None
    sub = df.tail(window)
    xs = np.arange(window, dtype=float)
    ys = sub["Close"].values.astype(float)
    coeffs = np.polyfit(xs, ys, 2)
    fitted = np.polyval(coeffs, xs)
    ss_res = np.sum((ys - fitted) ** 2)
    ss_tot = np.sum((ys - ys.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    if r2 < 0.5:
        return None

    a, b, _c = coeffs
    if a == 0:
        return None
    vertex_x = -b / (2 * a)
    # A vertex anywhere in [0, window-1] -- including the edges, i.e. the
    # very first bar or the current/incomplete last bar -- lets a smoothly
    # rising or falling trend with no real rounding at all fit a parabola
    # whose vertex just happens to land at one end, and get misclassified
    # as a completed Rounded Top/Bottom. Require the vertex to actually sit
    # in the middle of the window, where a real rounding turn would be.
    if not (window * 0.25 <= vertex_x <= window * 0.75):
        return None

    current_price = float(df["Close"].iloc[-1])
    atr_val = float(calc_atr(df, 14).iloc[-1])
    atr_val = atr_val if not np.isnan(atr_val) else 0.0
    buffer = 0.25 * atr_val
    confidence = round(min(r2, 1.0) * 100, 1)

    if a < 0:
        name, bias = "Rounded Top", "Bearish"
        edge_level = min(ys[0], ys[-1])
        status = "Confirmed" if current_price < edge_level - buffer else "Forming"
    else:
        name, bias = "Rounded Bottom", "Bullish"
        edge_level = max(ys[0], ys[-1])
        status = "Confirmed" if current_price > edge_level + buffer else "Forming"

    formed_date = str(sub.index[-1].date())
    detail = f"Curve fit r²={r2:.2f} over {window} bars"

    return PatternMatch(
        name=name, category="Reversal", bias=bias, confidence=confidence,
        formed_date=formed_date, status=status, price_target="N/A", detail=detail,
    )


# ---------------------------------------------------------------------------
# Engine 4 — candlestick patterns
# ---------------------------------------------------------------------------

def _body(row):
    return abs(row["Close"] - row["Open"])


def _upper_wick(row):
    return row["High"] - max(row["Open"], row["Close"])


def _lower_wick(row):
    return min(row["Open"], row["Close"]) - row["Low"]


def _range(row):
    return row["High"] - row["Low"]


def _is_bullish(row):
    return row["Close"] > row["Open"]


def _avg_body(df, n=10):
    sub = df.tail(n)
    if sub.empty:
        return 1e-9
    avg = (sub["Close"] - sub["Open"]).abs().mean()
    return float(avg) if avg and not np.isnan(avg) and avg > 0 else 1e-9


def _trend_bias(df, lookback=20):
    if len(df) < lookback + 1:
        return "Neutral"
    sub = df["Close"].tail(lookback + 1)
    change = (sub.iloc[-1] - sub.iloc[0]) / sub.iloc[0]
    if change > 0.03:
        return "Uptrend"
    if change < -0.03:
        return "Downtrend"
    return "Neutral"


def _cs_match(name, bias, confidence, formed_date, status, detail):
    return PatternMatch(
        name=name, category="Candlestick", bias=bias,
        confidence=round(min(max(confidence, 0), 100), 1), formed_date=formed_date,
        status=status, price_target="N/A", detail=detail,
    )


def detect_engulfing(df):
    if len(df) < 2:
        return None
    today, yday = df.iloc[-1], df.iloc[-2]
    body_today, body_yday = _body(today), _body(yday)
    if body_yday <= 0:
        return None
    # True containment: today's body must fully contain yesterday's body,
    # not merely overlap it. Checking only open/close against the "wrong"
    # side (e.g. today.Open < yday.Close) degrades to a weak overlap test
    # whenever yesterday's own direction flips the meaning of Open vs Close
    # as the top/bottom of its body.
    today_lo, today_hi = min(today["Open"], today["Close"]), max(today["Open"], today["Close"])
    yday_lo, yday_hi = min(yday["Open"], yday["Close"]), max(yday["Open"], yday["Close"])
    contains = today_lo < yday_lo and today_hi > yday_hi
    ratio = body_today / body_yday
    confidence = 50 + (ratio - 1) * 25
    formed_date = str(df.index[-1].date())
    if contains and today["Close"] > today["Open"] and yday["Close"] < yday["Open"]:
        return _cs_match("Bullish Engulfing", "Bullish", confidence, formed_date, "Forming",
                          f"Body {ratio:.1f}x prior bar")
    if contains and today["Close"] < today["Open"] and yday["Close"] > yday["Open"]:
        return _cs_match("Bearish Engulfing", "Bearish", confidence, formed_date, "Forming",
                          f"Body {ratio:.1f}x prior bar")
    return None


def detect_pin_bar_patterns(df):
    if len(df) < 1:
        return None
    row = df.iloc[-1]
    body = _body(row)
    if body == 0:
        body = 1e-9
    lower, upper = _lower_wick(row), _upper_wick(row)
    trend = _trend_bias(df.iloc[:-1], lookback=20)
    formed_date = str(df.index[-1].date())

    if lower > 2 * body and upper < 0.5 * body:
        ratio = lower / body
        confidence = 40 + (ratio - 2) * 15
        if trend == "Downtrend":
            return _cs_match("Hammer", "Bullish", confidence, formed_date, "Forming",
                              f"Lower wick {ratio:.1f}x body after downtrend")
        if trend == "Uptrend":
            return _cs_match("Hanging Man", "Bearish", confidence, formed_date, "Forming",
                              f"Lower wick {ratio:.1f}x body after uptrend")
        return None

    if upper > 2 * body and lower < 0.5 * body:
        ratio = upper / body
        confidence = 40 + (ratio - 2) * 15
        if trend == "Uptrend":
            return _cs_match("Shooting Star", "Bearish", confidence, formed_date, "Forming",
                              f"Upper wick {ratio:.1f}x body after uptrend")
        if trend == "Downtrend":
            return _cs_match("Inverted Hammer", "Bullish", confidence, formed_date, "Forming",
                              f"Upper wick {ratio:.1f}x body after downtrend")
        return None
    return None


def detect_doji(df):
    if len(df) < 1:
        return None
    row = df.iloc[-1]
    rng = _range(row)
    if rng == 0:
        return None
    body = _body(row)
    if body < 0.10 * rng:
        confidence = 50 + (0.10 - body / rng) * 400
        formed_date = str(df.index[-1].date())
        return _cs_match("Doji", "Neutral", confidence, formed_date, "Forming",
                          f"Body {body / rng * 100:.1f}% of range")
    return None


def detect_marubozu(df):
    if len(df) < 1:
        return None
    row = df.iloc[-1]
    rng = _range(row)
    if rng == 0:
        return None
    body, upper, lower = _body(row), _upper_wick(row), _lower_wick(row)
    if body >= 0.90 * rng and upper <= 0.05 * rng and lower <= 0.05 * rng:
        confidence = 60 + (body / rng - 0.90) * 350
        formed_date = str(df.index[-1].date())
        bias = "Bullish" if _is_bullish(row) else "Bearish"
        return _cs_match("Marubozu", bias, confidence, formed_date, "Confirmed",
                          f"Body {body / rng * 100:.0f}% of range, negligible wicks")
    return None


def detect_spinning_top(df):
    if len(df) < 1:
        return None
    row = df.iloc[-1]
    rng = _range(row)
    if rng == 0:
        return None
    body, upper, lower = _body(row), _upper_wick(row), _lower_wick(row)
    if (body < 0.30 * rng and upper > body and lower > body
            and min(upper, lower) / max(upper, lower, 1e-9) > 0.5):
        formed_date = str(df.index[-1].date())
        return _cs_match("Spinning Top", "Neutral", 55, formed_date, "Forming",
                          "Small body with balanced upper/lower wicks — indecision")
    return None


def detect_piercing_dark_cloud(df):
    if len(df) < 2:
        return None
    prev, cur = df.iloc[-2], df.iloc[-1]
    context = df.iloc[:-2] if len(df) > 12 else df.iloc[:-1]
    avg_body = _avg_body(context, 10)
    prev_body = _body(prev)
    formed_date = str(df.index[-1].date())
    if prev_body < avg_body * 0.8:
        return None
    prev_mid = (prev["Open"] + prev["Close"]) / 2

    if (not _is_bullish(prev) and _is_bullish(cur) and cur["Open"] < prev["Close"]
            and prev_mid < cur["Close"] < prev["Open"]):
        pen = (cur["Close"] - prev_mid) / max(prev_body, 1e-9)
        confidence = 50 + pen * 40
        return _cs_match("Piercing Pattern", "Bullish", confidence, formed_date, "Forming",
                          "Bullish bar closes above prior bearish bar's midpoint")

    if (_is_bullish(prev) and not _is_bullish(cur) and cur["Open"] > prev["Close"]
            and prev["Open"] < cur["Close"] < prev_mid):
        pen = (prev_mid - cur["Close"]) / max(prev_body, 1e-9)
        confidence = 50 + pen * 40
        return _cs_match("Dark Cloud Cover", "Bearish", confidence, formed_date, "Forming",
                          "Bearish bar closes below prior bullish bar's midpoint")
    return None


def detect_harami(df):
    if len(df) < 2:
        return None
    prev, cur = df.iloc[-2], df.iloc[-1]
    prev_lo, prev_hi = sorted([prev["Open"], prev["Close"]])
    cur_lo, cur_hi = sorted([cur["Open"], cur["Close"]])
    prev_body, cur_body = _body(prev), _body(cur)
    if prev_body <= 0:
        return None
    if not (cur_lo > prev_lo and cur_hi < prev_hi):
        return None
    size_ratio = cur_body / prev_body
    if size_ratio > 0.6:
        return None
    confidence = 50 + (0.6 - size_ratio) * 60
    formed_date = str(df.index[-1].date())
    if not _is_bullish(prev):
        return _cs_match("Bullish Harami", "Bullish", confidence, formed_date, "Forming",
                          f"Small body ({size_ratio * 100:.0f}% of prior) inside prior bearish bar")
    return _cs_match("Bearish Harami", "Bearish", confidence, formed_date, "Forming",
                      f"Small body ({size_ratio * 100:.0f}% of prior) inside prior bullish bar")


def detect_tweezers(df):
    if len(df) < 2:
        return None
    prev, cur = df.iloc[-2], df.iloc[-1]
    avg_range = (df["High"] - df["Low"]).tail(10).mean()
    if not avg_range or avg_range <= 0:
        return None
    formed_date = str(df.index[-1].date())
    high_diff = abs(prev["High"] - cur["High"]) / avg_range
    low_diff = abs(prev["Low"] - cur["Low"]) / avg_range

    if high_diff < 0.1 and _is_bullish(prev) and not _is_bullish(cur):
        confidence = 50 + (0.1 - high_diff) * 400
        return _cs_match("Tweezer Top", "Bearish", confidence, formed_date, "Forming",
                          "Matching highs on consecutive bars, reversal down")
    if low_diff < 0.1 and not _is_bullish(prev) and _is_bullish(cur):
        confidence = 50 + (0.1 - low_diff) * 400
        return _cs_match("Tweezer Bottom", "Bullish", confidence, formed_date, "Forming",
                          "Matching lows on consecutive bars, reversal up")
    return None


def detect_star_patterns(df):
    if len(df) < 3:
        return None
    b1, b2, b3 = df.iloc[-3], df.iloc[-2], df.iloc[-1]
    context = df.iloc[:-3] if len(df) > 13 else df
    avg_body = _avg_body(context, 10)
    body1, body2, body3 = _body(b1), _body(b2), _body(b3)
    formed_date = str(df.index[-1].date())

    if not (body1 > avg_body * 1.1 and body2 < avg_body * 0.5 and body3 > avg_body * 1.1):
        return None

    b1_mid = (b1["Open"] + b1["Close"]) / 2

    if not _is_bullish(b1) and _is_bullish(b3) and b3["Close"] > b1_mid:
        pen = (b3["Close"] - b1_mid) / max(body1, 1e-9)
        confidence = 55 + pen * 30
        return _cs_match("Morning Star", "Bullish", confidence, formed_date, "Confirmed",
                          "Long bearish, small indecision bar, long bullish closing above midpoint")

    if _is_bullish(b1) and not _is_bullish(b3) and b3["Close"] < b1_mid:
        pen = (b1_mid - b3["Close"]) / max(body1, 1e-9)
        confidence = 55 + pen * 30
        return _cs_match("Evening Star", "Bearish", confidence, formed_date, "Confirmed",
                          "Long bullish, small indecision bar, long bearish closing below midpoint")
    return None


def detect_three_soldiers_crows(df):
    if len(df) < 3:
        return None
    b1, b2, b3 = df.iloc[-3], df.iloc[-2], df.iloc[-1]
    context = df.iloc[:-3] if len(df) > 13 else df
    avg_body = _avg_body(context, 10)
    formed_date = str(df.index[-1].date())

    if not all(_body(b) > avg_body * 0.8 for b in (b1, b2, b3)):
        return None

    if _is_bullish(b1) and _is_bullish(b2) and _is_bullish(b3):
        rising = b2["Close"] > b1["Close"] and b3["Close"] > b2["Close"]
        opens_inside = b1["Close"] > b2["Open"] > b1["Open"] and b2["Close"] > b3["Open"] > b2["Open"]
        if rising and opens_inside:
            return _cs_match("Three White Soldiers", "Bullish", 80, formed_date, "Confirmed",
                              "Three consecutive long bullish bars, each closing higher")

    if not _is_bullish(b1) and not _is_bullish(b2) and not _is_bullish(b3):
        falling = b2["Close"] < b1["Close"] and b3["Close"] < b2["Close"]
        opens_inside = b1["Close"] < b2["Open"] < b1["Open"] and b2["Close"] < b3["Open"] < b2["Open"]
        if falling and opens_inside:
            return _cs_match("Three Black Crows", "Bearish", 80, formed_date, "Confirmed",
                              "Three consecutive long bearish bars, each closing lower")
    return None


def _detect_candlesticks(df):
    detectors = [
        detect_engulfing, detect_pin_bar_patterns, detect_doji, detect_marubozu,
        detect_spinning_top, detect_piercing_dark_cloud, detect_harami,
        detect_tweezers, detect_star_patterns, detect_three_soldiers_crows,
    ]
    matches = []
    for fn in detectors:
        result = fn(df)
        if result is not None:
            matches.append(result)
    return matches


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def detect_all(df):
    """Run every pattern engine against `df` (daily OHLCV) and return the
    detected patterns above the confidence floor, most confident first."""
    if df is None or len(df) < 20:
        return []

    matches = []

    # Flags/pennants/rectangles are inherently brief (a handful of bars right
    # after a pole), so they need a shorter window + tighter swing_lookback
    # than channels/triangles/wedges to even have enough pivots to fit a
    # line -- ensembling across 40-80 bar windows would never catch them at
    # all, so they get one purpose-built short pass. Everything else uses
    # the multi-window ensemble (classify_trendline_pattern_ensemble) so the
    # result reflects agreement across window choices, not whichever single
    # window happened to be asked.
    short_match = classify_trendline_pattern(df, window=20, swing_lookback=2, min_pivots_per_line=2)
    if short_match and short_match.name in {"Bull Flag", "Bear Flag", "Pennant", "Rectangle"}:
        trend_match = short_match
    else:
        trend_match = classify_trendline_pattern_ensemble(df)
        if trend_match is None:
            trend_match = short_match
    if trend_match:
        matches.append(trend_match)

    pt_match = classify_peak_trough_pattern(df, lookback_bars=90)
    if pt_match:
        matches.append(pt_match)

    rounded_match = detect_rounded_pattern(df, window=40)
    if rounded_match:
        matches.append(rounded_match)

    matches.extend(_detect_candlesticks(df))

    matches = [m for m in matches if m.confidence >= CONFIDENCE_FLOOR]
    matches.sort(key=lambda m: m.confidence, reverse=True)
    return matches
