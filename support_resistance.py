#!/usr/bin/env python3
"""
Multi-source, scored, clustered support/resistance model.

Replaces the previous nearest_support/nearest_resistance in
daily_ta_report.py (max/min of prior-day/5-day/52-week highs-lows,
picked purely by proximity) with a model that combines multiple
independent sources of evidence -- daily AND weekly swing structure,
prior-period reference levels, volume nodes, VWAP, gaps, and detected
chart-pattern targets -- into scored, clustered zones.

Key design choice: this module takes ALREADY-COMPUTED inputs (swing
points, ATR, VWAP values, volume nodes, pattern targets) rather than
importing daily_ta_report.py and recomputing them itself. Two reasons:
  1. daily_ta_report.py imports this module, so the reverse would be a
     circular import.
  2. The daily/weekly swing points passed in are the EXACT SAME ones
     classify_structure() already uses for the Daily/Weekly Trend
     sections -- not a re-detection -- so support/resistance can never
     silently disagree with those sections about where a swing point is.

Scoring philosophy (deliberately simple -- see module-level tunables
below): every candidate level carries a source-type weight, a recency
factor, and (for swing points) a reaction-amplitude filter so
insignificant wiggles never become candidates at all. Clustering merges
candidates within an ATR-scaled distance rather than a fixed price
gap, so the model adapts to each ticker's own volatility instead of
using arbitrary universal thresholds. Strength (historical significance)
and relevance (usefulness given current price) are tracked as two
separate numbers throughout, never collapsed into one.
"""

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------
# Tunables -- deliberately few, and ATR/time-scaled rather than fixed
# price/bar thresholds, so the model doesn't need re-tuning per ticker
# or volatility regime.
# ---------------------------------------------------------------------
CLUSTER_ATR_MULT = 0.5        # merge candidates within this many ATRs of each other into one zone
SWING_PROMINENCE_ATR = 1.0    # a swing point only becomes a candidate if its reaction to the
                               # next opposing swing is at least this many ATRs (point 1: filter noise)
RELEVANCE_DECAY_ATR = 8.0     # e-folding distance (in ATRs) for strength -> relevance decay
MAX_ZONES_PER_SIDE = 3
RECENCY_HALFLIFE_DAILY_BARS = 60   # ~3 trading months
RECENCY_HALFLIFE_WEEKLY_BARS = 12  # ~3 calendar months, in weekly-bar units
MIN_GAP_ATR = 0.5              # minimum gap size (in ATRs) to count as a gap level
GAP_LOOKBACK_BARS = 90
CONFLUENCE_BONUS_PER_SOURCE = 1.0   # bonus per additional INDEPENDENT source type agreeing in a zone

SOURCE_WEIGHTS = {
    "daily swing": 3.0,
    "weekly swing": 4.0,     # higher timeframe = more significant, matching the same
                              # daily < weekly weighting convention compute_confluence uses
    "prior-day": 1.5,
    "prior-week": 1.5,
    "prior-month": 1.5,
    "current-week": 1.0,
    "52-week": 2.5,
    "volume node": 2.0,
    "VWAP": 1.5,
    "anchored VWAP": 2.0,
    "gap": 1.5,
    "pattern target": 2.0,
}


@dataclass
class _Candidate:
    price: float
    source: str
    weight: float
    tests: int = 0
    reaction_atr: float = 0.0


@dataclass
class Zone:
    zone_low: float
    zone_high: float
    price: float                # weighted-average representative price
    level_type: str              # "support" | "resistance"
    strength_score: float        # historical significance -- NOT distance-adjusted
    relevance_score: float       # strength decayed by distance from current price
    distance_from_price: float
    distance_pct: float
    number_of_tests: int
    confluence_factors: list = field(default_factory=list)
    evidence: dict = field(default_factory=dict)


# ---------------------------------------------------------------------
# Candidate generation
# ---------------------------------------------------------------------

def _swing_candidates(swing_highs, swing_lows, source_label, current_price, atr_val,
                       halflife_bars, n_bars):
    """Merges swing highs/lows (each a list of (index_pos, price)) into one
    chronological sequence and keeps a point only if its reaction to the
    NEXT opposing swing point (or, for the most recent swing, to the
    current price) is at least SWING_PROMINENCE_ATR ATRs -- this is both
    the significance filter (point 1) and the reaction-strength evidence
    (point 2) in one pass, since they're the same underlying quantity."""
    points = sorted(
        [(idx, price, "high") for idx, price in swing_highs] +
        [(idx, price, "low") for idx, price in swing_lows],
        key=lambda p: p[0],
    )
    candidates = []
    for i, (idx, price, _kind) in enumerate(points):
        if i + 1 < len(points):
            reaction = abs(points[i + 1][1] - price)
        else:
            reaction = abs(current_price - price)
        reaction_atr = reaction / atr_val if atr_val else 0.0
        if reaction_atr < SWING_PROMINENCE_ATR:
            continue
        # exponential half-life decay: weight halves every `halflife_bars` bars of age
        age = max(n_bars - idx, 0)
        recency = 0.5 ** (age / halflife_bars) if halflife_bars else 1.0
        candidates.append(_Candidate(
            price=float(price), source=source_label,
            weight=SOURCE_WEIGHTS[source_label] * recency,
            tests=1, reaction_atr=reaction_atr,
        ))
    return candidates


def _reference_candidates(reference_levels):
    """One candidate per named reference level (prior-day/week/month,
    current-week, 52-week highs/lows) -- these are single computed
    levels, not repeated touches, so tests=0 and reaction=0 (unknown);
    they still contribute to strength/confluence via their source
    weight."""
    candidates = []
    for label, price in reference_levels.items():
        if price is None:
            continue
        source = label.rsplit(" ", 1)[0]  # "prior-day high" -> "prior-day"
        weight = SOURCE_WEIGHTS.get(source)
        if weight is None:
            continue
        candidates.append(_Candidate(price=float(price), source=source, weight=weight))
    return candidates


def _volume_node_candidates(volume_nodes):
    """volume_nodes: list of (price, relative_weight in [0, 1], where 1.0
    is the single strongest node). Weight scales with relative strength
    so a minor secondary node doesn't count as much as the true POC."""
    return [
        _Candidate(price=float(p), source="volume node",
                   weight=SOURCE_WEIGHTS["volume node"] * max(min(w, 1.0), 0.0))
        for p, w in volume_nodes if p is not None
    ]


def _vwap_candidates(rolling_vwap, session_vwap, anchored_vwap):
    candidates = []
    if rolling_vwap is not None:
        candidates.append(_Candidate(price=float(rolling_vwap), source="VWAP", weight=SOURCE_WEIGHTS["VWAP"]))
    if session_vwap is not None:
        candidates.append(_Candidate(price=float(session_vwap), source="VWAP", weight=SOURCE_WEIGHTS["VWAP"]))
    if anchored_vwap is not None:
        candidates.append(_Candidate(price=float(anchored_vwap), source="anchored VWAP",
                                      weight=SOURCE_WEIGHTS["anchored VWAP"]))
    return candidates


def _pattern_target_candidates(pattern_targets):
    return [
        _Candidate(price=float(p), source="pattern target", weight=SOURCE_WEIGHTS["pattern target"])
        for p, _label in pattern_targets if p is not None
    ]


def _gap_candidates(df, atr_val, lookback_bars=GAP_LOOKBACK_BARS, min_gap_atr=MIN_GAP_ATR):
    """Unfilled gaps in the last `lookback_bars` daily bars: a gap-up
    leaves its lower edge (prior day's high) as potential support below
    current price if price hasn't traded back down through it since; a
    gap-down leaves its upper edge (prior day's low) as potential
    resistance if price hasn't traded back up through it since."""
    if atr_val is None or atr_val <= 0 or len(df) < 2:
        return []
    sub = df.tail(min(lookback_bars, len(df))).copy()
    candidates = []
    highs, lows = sub["High"].values, sub["Low"].values
    for i in range(1, len(sub)):
        prev_high, prev_low = highs[i - 1], lows[i - 1]
        cur_high, cur_low = highs[i], lows[i]
        if cur_low > prev_high and (cur_low - prev_high) >= min_gap_atr * atr_val:
            edge = prev_high
            filled = np.any(lows[i + 1:] <= edge)
            if not filled:
                candidates.append(_Candidate(price=float(edge), source="gap",
                                              weight=SOURCE_WEIGHTS["gap"],
                                              reaction_atr=(cur_low - prev_high) / atr_val))
        elif cur_high < prev_low and (prev_low - cur_high) >= min_gap_atr * atr_val:
            edge = prev_low
            filled = np.any(highs[i + 1:] >= edge)
            if not filled:
                candidates.append(_Candidate(price=float(edge), source="gap",
                                              weight=SOURCE_WEIGHTS["gap"],
                                              reaction_atr=(prev_low - cur_high) / atr_val))
    return candidates


def _anchored_vwap(df, anchor_idx):
    """Cumulative typical-price VWAP from `anchor_idx` (a positional index
    into df) through the last bar. Self-contained -- only needs the raw
    OHLCV data, not anything from daily_ta_report.py."""
    if anchor_idx is None or anchor_idx >= len(df) - 1:
        return None
    sub = df.iloc[anchor_idx:]
    typical = (sub["High"] + sub["Low"] + sub["Close"]) / 3
    vol = sub["Volume"]
    if vol.sum() <= 0:
        return None
    return float((typical * vol).sum() / vol.sum())


def _most_recent_swing_index(daily_swing_highs, daily_swing_lows):
    all_idx = [idx for idx, _ in daily_swing_highs] + [idx for idx, _ in daily_swing_lows]
    return max(all_idx) if all_idx else None


# ---------------------------------------------------------------------
# Clustering + scoring
# ---------------------------------------------------------------------

def _cluster(candidates, cluster_dist):
    """Sort by price, greedily merge a candidate into the running cluster
    if it's within `cluster_dist` of that cluster's FIRST (minimum-price)
    member -- not its most-recently-added one. Comparing to the last-added
    member is single-linkage chaining: a long run of candidates each just
    within cluster_dist of its immediate predecessor can span an
    unbounded total range (verified against real SLV data: chained an 11-
    candidate run into a single "zone" spanning $49.25-$57.52, a 14% wide
    band that isn't a zone in any useful sense). Anchoring to the
    cluster's own minimum instead caps every zone's total width at
    cluster_dist, by construction."""
    if not candidates:
        return []
    ordered = sorted(candidates, key=lambda c: c.price)
    clusters = [[ordered[0]]]
    for c in ordered[1:]:
        if c.price - clusters[-1][0].price <= cluster_dist:
            clusters[-1].append(c)
        else:
            clusters.append([c])
    return clusters


def _score_cluster(members, current_price, atr_val, level_type):
    total_weight = sum(m.weight for m in members)
    distinct_sources = sorted(set(m.source for m in members))
    confluence_bonus = (len(distinct_sources) - 1) * CONFLUENCE_BONUS_PER_SOURCE
    strength = total_weight + confluence_bonus

    price = sum(m.price * m.weight for m in members) / total_weight if total_weight > 0 else \
        sum(m.price for m in members) / len(members)
    distance = abs(price - current_price)
    relevance = strength * math.exp(-distance / (RELEVANCE_DECAY_ATR * atr_val)) if atr_val else strength

    number_of_tests = sum(m.tests for m in members)
    reactions = [m.reaction_atr for m in members if m.reaction_atr > 0]
    avg_reaction_atr = float(np.mean(reactions)) if reactions else 0.0

    evidence = {
        "swing_evidence": [f"{m.source} @ {m.price:.2f} (reaction {m.reaction_atr:.2f} ATR)"
                            for m in members if "swing" in m.source],
        "prior_period_evidence": [f"{m.source} @ {m.price:.2f}" for m in members
                                   if m.source in ("prior-day", "prior-week", "prior-month",
                                                    "current-week", "52-week")],
        "volume_evidence": [f"{m.source} @ {m.price:.2f}" for m in members if m.source == "volume node"],
        "vwap_evidence": [f"{m.source} @ {m.price:.2f}" for m in members if "VWAP" in m.source],
        "gap_evidence": [f"{m.source} @ {m.price:.2f} ({m.reaction_atr:.2f} ATR gap)"
                          for m in members if m.source == "gap"],
        "pattern_evidence": [f"{m.source} @ {m.price:.2f}" for m in members if m.source == "pattern target"],
        "number_of_tests": number_of_tests,
        "reaction_strength_atr": round(avg_reaction_atr, 2),
        "confluence_count": len(distinct_sources),
    }

    return Zone(
        zone_low=min(m.price for m in members), zone_high=max(m.price for m in members),
        price=price, level_type=level_type,
        strength_score=round(strength, 2), relevance_score=round(relevance, 2),
        distance_from_price=round(distance, 4),
        distance_pct=round(distance / current_price * 100, 2) if current_price else 0.0,
        number_of_tests=number_of_tests, confluence_factors=distinct_sources,
        evidence=evidence,
    )


# ---------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------

def compute_levels(df, current_price, atr_val,
                    daily_swing_highs, daily_swing_lows,
                    weekly_swing_highs, weekly_swing_lows,
                    reference_levels, volume_nodes,
                    rolling_vwap, session_vwap,
                    pattern_targets, max_zones=MAX_ZONES_PER_SIDE):
    """Builds, clusters, and scores support/resistance zones from every
    available source. Returns {"support": [Zone, ...], "resistance": [Zone, ...]},
    each list ranked by relevance (ties by strength), longest 3 (S1..S3 /
    R1..R3) by default. Requires atr_val -- callers should skip calling
    this (and print "Insufficient data") when ATR itself isn't available,
    same convention as the rest of the report."""
    n_daily = len(df)
    anchor_idx = _most_recent_swing_index(daily_swing_highs, daily_swing_lows)
    anchored_vwap = _anchored_vwap(df, anchor_idx)

    candidates = []
    candidates += _swing_candidates(daily_swing_highs, daily_swing_lows, "daily swing",
                                     current_price, atr_val, RECENCY_HALFLIFE_DAILY_BARS, n_daily)
    # Weekly swing index positions are into the (much shorter) weekly
    # series, not the daily one -- recency there is relative to however
    # many weekly bars were supplied.
    n_weekly = max([idx for idx, _ in weekly_swing_highs + weekly_swing_lows], default=0) + 1
    candidates += _swing_candidates(weekly_swing_highs, weekly_swing_lows, "weekly swing",
                                     current_price, atr_val, RECENCY_HALFLIFE_WEEKLY_BARS, n_weekly)
    candidates += _reference_candidates(reference_levels)
    candidates += _volume_node_candidates(volume_nodes)
    candidates += _vwap_candidates(rolling_vwap, session_vwap, anchored_vwap)
    candidates += _pattern_target_candidates(pattern_targets)
    candidates += _gap_candidates(df, atr_val)

    below = [c for c in candidates if c.price < current_price]
    above = [c for c in candidates if c.price > current_price]

    cluster_dist = CLUSTER_ATR_MULT * atr_val if atr_val else current_price * 0.005

    support_zones = [_score_cluster(m, current_price, atr_val, "support") for m in _cluster(below, cluster_dist)]
    resistance_zones = [_score_cluster(m, current_price, atr_val, "resistance") for m in _cluster(above, cluster_dist)]

    support_zones.sort(key=lambda z: (-z.relevance_score, -z.strength_score))
    resistance_zones.sort(key=lambda z: (-z.relevance_score, -z.strength_score))

    return {
        "support": support_zones[:max_zones],
        "resistance": resistance_zones[:max_zones],
    }
