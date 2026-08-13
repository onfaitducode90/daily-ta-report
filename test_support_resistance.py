#!/usr/bin/env python3
"""
Tests for support_resistance.py's clustering/scoring model.

Verifies the properties the design is actually supposed to guarantee:
significance filtering (noise swings never become candidates), ATR-based
clustering (nearby levels merge, distant ones don't), confluence scoring
(independent sources agreeing raises strength beyond just summing one
source repeatedly), the strength/relevance distinction (a strong-but-far
zone keeps its strength but drops in relevance), gap fill detection, and
that the module never crashes on ragged/missing inputs.
"""
import sys

import numpy as np
import pandas as pd

import support_resistance as sr

RESULTS = []


def check(label, cond):
    RESULTS.append((label, cond))
    print(("PASS" if cond else "FAIL"), "-", label)


def make_df(n=120, base=100.0, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2025-01-01", periods=n)
    closes = base + np.cumsum(rng.normal(0, 0.5, n))
    highs = closes + rng.uniform(0.2, 0.6, n)
    lows = closes - rng.uniform(0.2, 0.6, n)
    opens = closes + rng.uniform(-0.3, 0.3, n)
    vols = rng.integers(1_000_000, 2_000_000, n)
    return pd.DataFrame({"Open": opens, "High": highs, "Low": lows, "Close": closes,
                          "Volume": vols}, index=dates)


def test_significance_filter_drops_tiny_swings():
    """A swing point whose reaction to the next opposing swing is well
    under SWING_PROMINENCE_ATR should never become a candidate."""
    highs = [(10, 100.5), (30, 100.6)]   # only $0.10 apart
    lows = [(20, 100.0)]
    candidates = sr._swing_candidates(highs, lows, "daily swing", current_price=100.3,
                                       atr_val=2.0, halflife_bars=60, n_bars=40)
    check("Tiny-reaction swings are filtered out as insignificant", len(candidates) == 0)


def test_significance_filter_keeps_real_swings():
    highs = [(10, 110.0)]   # $10 reaction vs the low below, well over 1 ATR of 2.0
    lows = [(20, 100.0)]
    candidates = sr._swing_candidates(highs, lows, "daily swing", current_price=105.0,
                                       atr_val=2.0, halflife_bars=60, n_bars=40)
    check("Significant swings survive the prominence filter", len(candidates) == 2)


def test_clustering_merges_nearby_levels():
    members = [
        sr._Candidate(price=100.0, source="prior-day", weight=1.5),
        sr._Candidate(price=100.3, source="VWAP", weight=1.5),
        sr._Candidate(price=100.4, source="volume node", weight=2.0),
        sr._Candidate(price=110.0, source="52-week", weight=2.5),  # far away, separate cluster
    ]
    clusters = sr._cluster(members, cluster_dist=1.0)
    check("Nearby candidates merge into one cluster", len(clusters) == 2)
    check("The merged cluster has all 3 nearby members", len(clusters[0]) == 3)
    check("The distant candidate stays in its own cluster", len(clusters[1]) == 1)


def test_clustering_does_not_chain_across_a_long_run():
    """Regression for a real bug found against SLV data: a long run of
    candidates each within cluster_dist of its immediate predecessor
    used to chain (single-linkage) into one zone spanning the whole run,
    even though the first and last members were nowhere near each other.
    A zone's total width must never exceed cluster_dist."""
    step = 0.9
    cluster_dist = 1.0
    members = [sr._Candidate(price=100.0 + i * step, source="daily swing", weight=3.0) for i in range(10)]
    # Total span is 100.0 to 108.1 (8.1), each consecutive gap is 0.9 <= cluster_dist.
    clusters = sr._cluster(members, cluster_dist)
    widths = [c[-1].price - c[0].price for c in clusters]
    check(f"No cluster's total width exceeds cluster_dist (got widths {[round(w, 2) for w in widths]})",
          all(w <= cluster_dist + 1e-9 for w in widths))
    check("The long run splits into multiple clusters instead of chaining into one",
          len(clusters) > 1)


def test_confluence_bonus_beats_single_source_repeated():
    """Same total base weight, but one zone has 3 independent sources and
    the other has the same source repeated 3x -- the confluence bonus
    should make the independent-agreement zone score strictly higher."""
    diverse = [
        sr._Candidate(price=100.0, source="prior-day", weight=1.5),
        sr._Candidate(price=100.1, source="VWAP", weight=1.5),
        sr._Candidate(price=100.2, source="volume node", weight=1.5),
    ]
    repeated = [
        sr._Candidate(price=100.0, source="prior-day", weight=1.5),
        sr._Candidate(price=100.1, source="prior-day", weight=1.5),
        sr._Candidate(price=100.2, source="prior-day", weight=1.5),
    ]
    z_diverse = sr._score_cluster(diverse, current_price=100.0, atr_val=2.0, level_type="support")
    z_repeated = sr._score_cluster(repeated, current_price=100.0, atr_val=2.0, level_type="support")
    check(f"Independent-source confluence scores higher than repeated single source "
          f"(diverse={z_diverse.strength_score}, repeated={z_repeated.strength_score})",
          z_diverse.strength_score > z_repeated.strength_score)


def test_strength_vs_relevance_distinction():
    """A historically strong zone far from price keeps a high strength
    score but a much lower relevance score than a weaker, nearer zone."""
    far_strong = [
        sr._Candidate(price=50.0, source="52-week", weight=2.5),
        sr._Candidate(price=50.1, source="weekly swing", weight=4.0),
        sr._Candidate(price=49.9, source="volume node", weight=2.0),
    ]
    near_weak = [sr._Candidate(price=99.0, source="prior-day", weight=1.5)]
    z_far = sr._score_cluster(far_strong, current_price=100.0, atr_val=2.0, level_type="support")
    z_near = sr._score_cluster(near_weak, current_price=100.0, atr_val=2.0, level_type="support")
    check(f"Far zone has higher strength than near zone (far={z_far.strength_score}, near={z_near.strength_score})",
          z_far.strength_score > z_near.strength_score)
    check(f"But near zone has higher relevance despite lower strength "
          f"(far relevance={z_far.relevance_score}, near relevance={z_near.relevance_score})",
          z_near.relevance_score > z_far.relevance_score)


def test_gap_detection_unfilled_vs_filled():
    dates = pd.bdate_range("2025-01-01", periods=6)
    # Bar 2 gaps up hard from bar 1 and price never comes back down through it.
    df = pd.DataFrame({
        "Open":  [100, 100.5, 110, 110.5, 111, 111.5],
        "High":  [100.5, 101, 111, 111.5, 112, 112.5],
        "Low":   [99.5, 100, 109.5, 110, 110.5, 111],
        "Close": [100, 100.5, 110.5, 111, 111.5, 112],
        "Volume": [1_000_000] * 6,
    }, index=dates)
    candidates = sr._gap_candidates(df, atr_val=1.0, lookback_bars=10, min_gap_atr=0.5)
    check("Unfilled gap produces exactly one gap candidate", len(candidates) == 1)
    if candidates:
        check(f"Gap candidate sits at the prior bar's high (got {candidates[0].price})",
              abs(candidates[0].price - 101.0) < 1e-9)

    # Now let price trade back down through the gap -- it should no longer count.
    df2 = df.copy()
    df2.loc[dates[5], "Low"] = 100.8  # dips back below the gap's lower edge (101)
    candidates2 = sr._gap_candidates(df2, atr_val=1.0, lookback_bars=10, min_gap_atr=0.5)
    check("Filled gap no longer produces a candidate", len(candidates2) == 0)


def test_compute_levels_end_to_end_no_crash_and_ranked():
    df = make_df(n=150)
    current_price = float(df["Close"].iloc[-1])
    atr_val = 2.0
    # Synthetic swings, deliberately including a couple of insignificant ones.
    daily_highs = [(20, current_price + 8), (60, current_price + 1.0), (110, current_price + 12)]
    daily_lows = [(35, current_price - 6), (80, current_price - 15)]
    weekly_highs = [(3, current_price + 20)]
    weekly_lows = [(5, current_price - 18)]
    reference_levels = {
        "prior-day high": current_price + 2, "prior-day low": current_price - 2,
        "prior-week high": current_price + 4, "prior-week low": current_price - 4,
        "prior-month high": current_price + 9, "prior-month low": current_price - 9,
        "current-week high": current_price + 1.5, "current-week low": current_price - 1.5,
        "52-week high": current_price + 25, "52-week low": current_price - 25,
    }
    volume_nodes = [(current_price - 3, 1.0), (current_price + 6, 0.4)]

    result = sr.compute_levels(
        df, current_price, atr_val,
        daily_highs, daily_lows, weekly_highs, weekly_lows,
        reference_levels, volume_nodes,
        rolling_vwap=current_price - 1.0, session_vwap=current_price - 0.5,
        pattern_targets=[(current_price + 5, "measured move")],
    )
    check("compute_levels returns support and resistance keys",
          "support" in result and "resistance" in result)
    check("At least one support zone found", len(result["support"]) > 0)
    check("At least one resistance zone found", len(result["resistance"]) > 0)
    check("At most 3 zones per side", len(result["support"]) <= 3 and len(result["resistance"]) <= 3)

    supports = result["support"]
    check("Support zones ranked by relevance descending",
          all(supports[i].relevance_score >= supports[i + 1].relevance_score for i in range(len(supports) - 1)))
    check("All support zone prices are below current price",
          all(z.price < current_price for z in supports))
    check("All resistance zone prices are above current price",
          all(z.price > current_price for z in result["resistance"]))
    check("Every zone carries confluence_factors and evidence",
          all(z.confluence_factors and z.evidence for z in supports + result["resistance"]))


def test_compute_levels_handles_sparse_inputs_without_crashing():
    """No swings, no volume nodes, no VWAP, no patterns, no gaps -- just
    the bare reference levels. Must degrade gracefully, not crash."""
    df = make_df(n=40)
    current_price = float(df["Close"].iloc[-1])
    result = sr.compute_levels(
        df, current_price, atr_val=1.5,
        daily_swing_highs=[], daily_swing_lows=[],
        weekly_swing_highs=[], weekly_swing_lows=[],
        reference_levels={"prior-day high": current_price + 1, "prior-day low": current_price - 1},
        volume_nodes=[], rolling_vwap=None, session_vwap=None, pattern_targets=[],
    )
    check("Sparse input still returns a well-formed result", "support" in result and "resistance" in result)
    check("Sparse input finds the one prior-day support candidate", len(result["support"]) == 1)
    check("Sparse input finds the one prior-day resistance candidate", len(result["resistance"]) == 1)


def test_compute_levels_no_crash_with_no_atr():
    df = make_df(n=40)
    current_price = float(df["Close"].iloc[-1])
    result = sr.compute_levels(
        df, current_price, atr_val=None,
        daily_swing_highs=[], daily_swing_lows=[],
        weekly_swing_highs=[], weekly_swing_lows=[],
        reference_levels={"prior-day high": current_price + 1, "prior-day low": current_price - 1},
        volume_nodes=[], rolling_vwap=None, session_vwap=None, pattern_targets=[],
    )
    check("Missing ATR degrades gracefully instead of raising", "support" in result and "resistance" in result)


def main():
    test_significance_filter_drops_tiny_swings()
    test_significance_filter_keeps_real_swings()
    test_clustering_merges_nearby_levels()
    test_clustering_does_not_chain_across_a_long_run()
    test_confluence_bonus_beats_single_source_repeated()
    test_strength_vs_relevance_distinction()
    test_gap_detection_unfilled_vs_filled()
    test_compute_levels_end_to_end_no_crash_and_ranked()
    test_compute_levels_handles_sparse_inputs_without_crashing()
    test_compute_levels_no_crash_with_no_atr()

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
