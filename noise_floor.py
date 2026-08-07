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

import numpy as np
import pandas as pd
import yfinance as yf

import daily_ta_report as R
import chart_patterns as CP

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


def main():
    rng = np.random.default_rng(42)
    print(f"Noise floor monitor -- {len(TICKERS)} tickers x {N_SHUFFLES} shuffles each")
    print("(A real ticker landing consistently in the top percentiles of its own shuffled "
          "distribution would be evidence of real structure; landing near the middle is "
          "evidence the detector can't tell this ticker's real history from a scrambled one.)")
    for ticker in TICKERS:
        run_ticker(ticker, rng)


if __name__ == "__main__":
    main()
