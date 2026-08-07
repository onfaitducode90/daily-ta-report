#!/usr/bin/env python3
"""
Noise floor monitor (F40). Complementary to test_regression.py's synthetic
GBM noise test: instead of pure random-walk data, this shuffles each REAL
ticker's own daily bars (destroying temporal order/serial correlation --
no more genuine trends, momentum, or chart-pattern structure -- while
exactly preserving that ticker's actual volatility and return-magnitude
distribution) and compares chart_patterns.detect_all /
daily_ta_report.compute_confluence output on the real (order-intact) price
path against many shuffled (order-destroyed) reconstructions of the SAME
ticker.

If the real path doesn't produce meaningfully more patterns, higher
confidence, or a stronger confluence net than shuffled versions of its own
history, that's direct evidence the detector isn't picking up anything
specific to this ticker's actual price structure -- it would say much the
same thing about a scrambled version of its own past.

Each shuffled bar's Open/High/Low/Close is reconstructed from that day's
own ORIGINAL ratios relative to the prior close (so each day's real shape
-- gap size, wick sizes -- travels with its own return, just reassigned to
a different day), which is what keeps a shuffled bar internally consistent
(High >= max(Open,Close), Low <= min(Open,Close)) automatically.
"""

from datetime import date, datetime

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm

import daily_ta_report as R
import chart_patterns as CP
import option_chain as OC

TICKERS = ["NVDA", "INTC", "SPY", "QQQ"]
N_SHUFFLES = 50
WINDOW = 300  # matches daily_ta_report.HISTORY_DAYS


def fetch_history(ticker, period="5y"):
    df = yf.Ticker(ticker).history(period=period, interval="1d", auto_adjust=False)
    if df is None or df.empty:
        return None
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    df.index = pd.to_datetime(df.index)
    return df.tail(WINDOW)  # keep the real DatetimeIndex -- chart_patterns.py needs .date() on it


def shuffle_bars(df, rng):
    """Reconstruct a synthetic price path with the same day-to-day bar
    SHAPES as the original, in a shuffled order. Bar i's shape is
    expressed as (Open, High, Low, Close, Volume) all as ratios to the
    PRIOR bar's close -- dimensionless, so replaying them against a
    different running close preserves each bar's real internal proportions
    (gap size, wick size, that day's real volume) exactly, just reassigns
    which day's shape lands at which position."""
    o, h, l, c, v = (df[col].values for col in ("Open", "High", "Low", "Close", "Volume"))
    prev_c = np.r_[c[0], c[:-1]]
    o_ratio, h_ratio, l_ratio, c_ratio = o / prev_c, h / prev_c, l / prev_c, c / prev_c

    n = len(df)
    order = rng.permutation(np.arange(1, n))  # shuffle bars 1..n-1 (bar 0 is the fixed start)
    new_o = np.empty(n)
    new_h = np.empty(n)
    new_l = np.empty(n)
    new_c = np.empty(n)
    new_v = np.empty(n)
    new_o[0], new_h[0], new_l[0], new_c[0], new_v[0] = o[0], h[0], l[0], c[0], v[0]

    running_close = c[0]
    for t, src in enumerate(order, start=1):
        new_o[t] = running_close * o_ratio[src]
        new_h[t] = running_close * h_ratio[src]
        new_l[t] = running_close * l_ratio[src]
        new_c[t] = running_close * c_ratio[src]
        new_v[t] = v[src]
        running_close = new_c[t]

    return pd.DataFrame({"Open": new_o, "High": new_h, "Low": new_l, "Close": new_c, "Volume": new_v},
                         index=pd.bdate_range("2020-01-01", periods=n))


def measure(df):
    matches = CP.detect_all(df)
    confluence = R.compute_confluence(df, matches)
    return {
        "n_patterns": len(matches),
        "mean_confidence": float(np.mean([m.confidence for m in matches])) if matches else 0.0,
        "abs_net": abs(confluence["net"]),
        "sufficient_evidence": confluence["sufficient_evidence"],
    }


def run_ticker(ticker, rng):
    df = fetch_history(ticker)
    if df is None or len(df) < WINDOW:
        print(f"{ticker}: insufficient history, skipping")
        return None

    real = measure(df)
    shuffled = [measure(shuffle_bars(df, rng)) for _ in range(N_SHUFFLES)]

    def summarize_field(field):
        real_val = real[field]
        shuf_vals = np.array([s[field] for s in shuffled], dtype=float)
        percentile = float((shuf_vals < real_val).mean() * 100)
        return real_val, shuf_vals.mean(), shuf_vals.std(), percentile

    print(f"\n{ticker} (n={N_SHUFFLES} shuffles of its own {WINDOW}-bar history):")
    for field, label in (("n_patterns", "patterns detected"),
                          ("mean_confidence", "mean confidence"),
                          ("abs_net", "|confluence net|")):
        real_val, shuf_mean, shuf_std, pct = summarize_field(field)
        flag = " <-- real is in the TOP 10% of its own shuffled distribution" if pct >= 90 else ""
        print(f"  {label}: real={real_val:.2f}, shuffled mean={shuf_mean:.2f} (std={shuf_std:.2f}), "
              f"real is at the {pct:.0f}th percentile of shuffled{flag}")

    real_sufficient = real["sufficient_evidence"]
    shuf_sufficient_rate = np.mean([s["sufficient_evidence"] for s in shuffled])
    print(f"  sufficient_evidence: real={real_sufficient}, "
          f"shuffled rate={shuf_sufficient_rate:.1%} of {N_SHUFFLES} reshuffles")

    return real, shuffled


# ---------------------------------------------------------------------------
# G29 (2nd Opus audit): does build_credit_spread's exact-delta-targeted
# strike selection actually do anything a nearby-but-imprecise selection
# wouldn't? Real option chains are too scarce locally (a handful of daily
# snapshots for 6 tickers) to answer this from real data, so this
# generates synthetic chains from Black-Scholes -- not because BS is a
# perfect market model, but because it gives an internally consistent
# price/delta/IV relationship to test the SELECTION LOGIC against,
# independent of whether BS itself is realistic.
# ---------------------------------------------------------------------------

SYNTH_EXP = date(2099, 1, 1)  # placeholder; only days_to_expiration is used downstream


def bs_price_delta(spot, strike, sigma, years, r, option_type):
    """European Black-Scholes price and delta. Falls back to intrinsic
    value / a 0-or-1 delta at expiry (years<=0) or zero vol, where the
    standard d1/d2 formula is undefined."""
    if years <= 0 or sigma <= 0:
        if option_type == "call":
            return max(0.0, spot - strike), (1.0 if spot > strike else 0.0)
        return max(0.0, strike - spot), (-1.0 if spot < strike else 0.0)
    d1 = (np.log(spot / strike) + (r + 0.5 * sigma ** 2) * years) / (sigma * np.sqrt(years))
    d2 = d1 - sigma * np.sqrt(years)
    if option_type == "call":
        price = spot * norm.cdf(d1) - strike * np.exp(-r * years) * norm.cdf(d2)
        delta = float(norm.cdf(d1))
    else:
        price = strike * np.exp(-r * years) * norm.cdf(-d2) - spot * norm.cdf(-d1)
        delta = float(norm.cdf(d1)) - 1.0
    return max(0.0, float(price)), delta


def make_synthetic_chain(spot, sigma, dte_days, rng, r=0.04, n_strikes=41):
    """A synthetic OptionChain (real option_chain.py dataclasses, so
    build_credit_spread runs completely unmodified against it) with
    Black-Scholes prices/deltas, a modeled bid-ask spread that widens
    (relatively) for cheaper/further-OTM contracts, and open interest that
    decays with distance from the money -- so the liquidity/width filters
    in build_credit_spread get genuinely exercised, not just the delta
    targeting."""
    strike_step = max(0.5, round(spot * 0.01, 1))
    years = dte_days / 365.0
    center_strike = round(spot / strike_step) * strike_step

    def spread_for(mid):
        # Tuned to keep a realistic-but-not-dominant rejection rate under
        # build_credit_spread's MAX_BID_ASK_PCT_OF_CREDIT guard on a
        # narrow (strike_step-wide) vertical -- real liquid-name spreads
        # run tighter (2-4% of mid) than a first attempt at 6% did, which
        # rejected most narrow near-money verticals outright.
        half = max(0.01, 0.03 * mid) * float(rng.uniform(0.8, 1.3))
        return max(0.01, mid - half), mid + half

    quotes = []
    for i in range(-(n_strikes // 2), n_strikes // 2 + 1):
        strike = round(center_strike + i * strike_step, 2)
        if strike <= 0:
            continue
        call_mid, call_delta = bs_price_delta(spot, strike, sigma, years, r, "call")
        put_mid, put_delta = bs_price_delta(spot, strike, sigma, years, r, "put")
        call_bid, call_ask = spread_for(call_mid)
        put_bid, put_ask = spread_for(put_mid)
        dist = abs(strike - spot) / spot
        oi_base = 3000 * np.exp(-dist / 0.12)
        call_oi = max(0, int(oi_base * rng.uniform(0.4, 1.6)))
        put_oi = max(0, int(oi_base * rng.uniform(0.4, 1.6)))
        quotes.append(OC.OptionQuote(
            expiration=SYNTH_EXP, days_to_expiration=dte_days, strike=strike,
            call_bid=round(call_bid, 2), call_ask=round(call_ask, 2),
            call_iv=sigma, call_delta=round(call_delta, 4),
            call_volume=int(call_oi * 0.1), call_open_interest=call_oi,
            put_bid=round(put_bid, 2), put_ask=round(put_ask, 2),
            put_iv=sigma, put_delta=round(put_delta, 4),
            put_volume=int(put_oi * 0.1), put_open_interest=put_oi,
        ))
    return OC.OptionChain(
        ticker="SYN", snapshot_time=datetime(2026, 1, 1),
        underlying_last=spot, underlying_bid=spot - 0.01, underlying_ask=spot + 0.01,
        quotes=quotes,
    )


def credit_structure_quality(chain, side, target_delta):
    """Build a credit spread at `target_delta` and return its EV (using
    the tool's own estimate_credit_structure_ev, judged by the structure's
    OWN achieved delta -- the same standard the live report holds every
    printed structure to) and credit/width ratio, or None if no valid
    structure could be built at that target (delta tolerance, liquidity,
    or width guards rejected everything)."""
    spread = R.build_credit_spread(chain, SYNTH_EXP, side, target_short_delta=target_delta)
    if spread is None or spread["max_loss"] <= 0:
        return None
    ev = R.estimate_credit_structure_ev(1 - abs(spread["short_delta"]), spread["credit"],
                                          spread["max_loss"], num_legs=2)
    return {"ev_net": ev["ev_net"], "ratio": spread["credit"] / spread["width"],
            "achieved_delta": abs(spread["short_delta"])}


def run_strike_selection_comparison(n_trials=1000, target_delta=0.25, jitter=0.05, seed=7):
    """Tool's EXACT target (0.25 delta, this codebase's actual default)
    vs. a RANDOM target uniformly drawn from the same +/-jitter neighborhood
    each trial -- both run through the identical build_credit_spread code
    path and identical filters, so this isolates whether hitting the exact
    target delta matters, not whether the algorithm or filters differ."""
    rng = np.random.default_rng(seed)
    tool_results, random_results = [], []
    tool_none, random_none = 0, 0

    for _ in range(n_trials):
        spot = float(rng.uniform(20, 500))
        sigma = float(rng.uniform(0.20, 1.00))
        dte = int(rng.integers(3, 45))
        side = "put" if rng.random() < 0.5 else "call"
        chain = make_synthetic_chain(spot, sigma, dte, rng)

        tool = credit_structure_quality(chain, side, target_delta)
        random_target = float(np.clip(rng.uniform(target_delta - jitter, target_delta + jitter), 0.01, 0.99))
        rand = credit_structure_quality(chain, side, random_target)

        if tool is None:
            tool_none += 1
        else:
            tool_results.append(tool)
        if rand is None:
            random_none += 1
        else:
            random_results.append(rand)

    print(f"\n=== G29: exact-delta-target ({target_delta}) vs. random target in "
          f"[{target_delta - jitter:.2f}, {target_delta + jitter:.2f}] -- n={n_trials} synthetic BS chains ===")
    print(f"  Tool:   {len(tool_results)}/{n_trials} produced a valid structure "
          f"({tool_none} rejected by delta/liquidity/width guards)")
    print(f"  Random: {len(random_results)}/{n_trials} produced a valid structure "
          f"({random_none} rejected)")

    for label, results in (("Tool (exact 0.25 target)", tool_results),
                            ("Random (jittered target)", random_results)):
        if not results:
            continue
        ev = np.array([r["ev_net"] for r in results])
        ratio = np.array([r["ratio"] for r in results])
        delta_err = np.array([abs(r["achieved_delta"] - target_delta) for r in results])
        print(f"  {label}: mean EV/contract=${ev.mean():+.2f} (median ${np.median(ev):+.2f}), "
              f"mean credit/width={ratio.mean():.2f}, mean |delta err|={delta_err.mean():.3f}")

    if tool_results and random_results:
        ev_tool = np.array([r["ev_net"] for r in tool_results])
        ev_rand = np.array([r["ev_net"] for r in random_results])
        print(f"  EV difference (tool - random, on their own respective valid subsets): "
              f"${ev_tool.mean() - ev_rand.mean():+.2f}/contract")
        print("  (Same underlying EV MODEL judges both -- P(win) is derived from whichever delta was "
              "actually picked, so this measures whether targeting precision changes the OUTCOME, not "
              "just whether it changes which strike gets picked.)")
        # Last measured (n=1000): tool $-6.84 vs random $-6.85, a $0.00
        # difference -- exact delta targeting buys essentially nothing on
        # THIS metric. That's not a bug in the targeting logic; it's a
        # property of judging both structures by the same delta-implied
        # P(win) model -- of course a structure looks about as "good" as
        # whatever delta it landed on, regardless of whether that delta
        # was hit precisely or approximately. What precise targeting DOES
        # buy, that this test doesn't measure, is a more PREDICTABLE risk
        # profile (the trader asked for a 25-delta short and reliably gets
        # one, rather than something in a 20-30 delta band) -- consistency
        # of the resulting structure, not a better EV estimate under this
        # tool's own simplified probability model.


def main():
    rng = np.random.default_rng(42)
    print(f"Noise floor monitor -- {len(TICKERS)} tickers x {N_SHUFFLES} shuffles each")
    print("(A real ticker landing consistently in the top percentiles of its own shuffled "
          "distribution would be evidence of real structure; landing near the middle is "
          "evidence the detector can't tell this ticker's real history from a scrambled one.)")
    for ticker in TICKERS:
        run_ticker(ticker, rng)

    run_strike_selection_comparison()


if __name__ == "__main__":
    main()
