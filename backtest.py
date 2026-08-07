#!/usr/bin/env python3
"""
Historical backtest (F37/F38) -- walks through past dates using ONLY data
available up to that point (no lookahead) to measure whether this tool's
confluence-directional calls have any historical edge. Uses the exact
production functions (daily_ta_report.compute_confluence,
chart_patterns.detect_all) against a historical slice at each date, so this
tests the real deployed logic, not a hand-reimplementation that could
silently drift out of sync with it. Compares against two baselines any real
trading idea needs to beat: buy-and-hold and a naive 50/200-day SMA
crossover.

Framing, stated plainly: none of this tool's thresholds were ever FIT
against historical price data -- they were tuned against synthetic
no-signal tests (t1_adx.py / t2_patterns.py / t3_confluence.py) and
hand-built pattern fixtures (test_patterns.py), not backtested against any
specific historical period. So there's no "training window" to freeze and
a separate "test window" to hold out in the traditional walk-forward
sense -- the entire historical run below is out-of-sample relative to how
the tool was actually built.

Caveat, stated just as plainly: evaluations spaced EVAL_STEP trading days
apart share overlapping lookback windows and overlapping forward-return
windows, so they are NOT independent observations. The "n" reported here
overstates true statistical power -- this is a first-pass directional
read, not a rigorously power-analyzed backtest (see F38's "walk-forward"
framing in the original audit for what a fuller version would require).
"""

import numpy as np
import pandas as pd
import yfinance as yf

import daily_ta_report as R
import chart_patterns as CP

BACKTEST_TICKERS = ["NVDA", "INTC", "SPY", "QQQ"]
HORIZONS = (1, 5, 10, 20)
EVAL_STEP = 5       # evaluate every 5 trading bars (~weekly)
WARMUP_BARS = 250   # need 200+ bars for the 200-day SMA, plus buffer


def fetch_long_history(ticker, period="10y"):
    df = yf.Ticker(ticker).history(period=period, interval="1d", auto_adjust=False)
    if df is None or df.empty:
        return None
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    df.index = pd.to_datetime(df.index)
    return df


def sma_crossover_signal(df):
    """Baseline: bullish when 50-day SMA > 200-day SMA, bearish when
    below. Well-known and dead simple -- whatever this tool claims as an
    edge needs to beat this, not just beat a coin flip."""
    if len(df) < 200:
        return None
    sma50 = R.sma(df["Close"], 50).iloc[-1]
    sma200 = R.sma(df["Close"], 200).iloc[-1]
    if np.isnan(sma50) or np.isnan(sma200):
        return None
    return 1 if sma50 > sma200 else -1


def forward_returns(df, pos, horizons):
    out = {}
    spot = float(df["Close"].iloc[pos])
    for h in horizons:
        if pos + h < len(df):
            out[h] = (float(df["Close"].iloc[pos + h]) - spot) / spot
    return out


def run_ticker(ticker):
    df = fetch_long_history(ticker)
    if df is None or len(df) < WARMUP_BARS + max(HORIZONS) + 1:
        print(f"{ticker}: insufficient history, skipping")
        return []

    rows = []
    for pos in range(WARMUP_BARS, len(df) - max(HORIZONS), EVAL_STEP):
        slice_df = df.iloc[:pos + 1]  # NO LOOKAHEAD: only bars up to and including pos
        eval_date = df.index[pos].date()

        patterns = CP.detect_all(slice_df)
        confluence = R.compute_confluence(slice_df, patterns)
        net = confluence["net"]
        sufficient = confluence["sufficient_evidence"]
        tool_dir = 1 if (sufficient and net > 0) else -1 if (sufficient and net < 0) else None

        sma_dir = sma_crossover_signal(slice_df)

        fwd = forward_returns(df, pos, HORIZONS)
        for h, ret in fwd.items():
            # Cast to plain Python bool, not left as numpy bool_: a numpy 2.x
            # + pandas quirk means .mean() on an OBJECT-dtype column (forced
            # by mixing None in) of np.True_/np.False_ silently uses boolean
            # OR instead of arithmetic addition when summing -- confirmed via
            # a minimal repro: 3 True + 1 False out of 4 gave a "mean" of
            # 0.25 instead of 0.75. Plain Python bool doesn't have this
            # problem, but every .mean() call site below is ALSO defensively
            # cast to float, since a plain bool column can still end up
            # object-dtype once concatenated with other tickers' None values.
            rows.append({
                "ticker": ticker, "eval_date": eval_date, "horizon": h, "fwd_return": ret,
                "tool_dir": tool_dir, "tool_hit": bool(np.sign(ret) == tool_dir) if tool_dir else None,
                "sma_dir": sma_dir, "sma_hit": bool(np.sign(ret) == sma_dir) if sma_dir else None,
                "bh_hit": bool(ret > 0),
            })
    return rows


def summarize(rows):
    d = pd.DataFrame(rows)
    print(f"\nTotal (ticker, eval_date, horizon) rows: {len(d)}")
    print(f"Tickers: {sorted(d.ticker.unique())}")
    print(f"Date range: {d.eval_date.min()} to {d.eval_date.max()}")
    print(f"Evaluated every {EVAL_STEP} trading bars -- NOT independent observations "
          "(overlapping windows); treat n as an upper bound on statistical power, not the real one.\n")

    for h in HORIZONS:
        sub = d[d.horizon == h]
        print(f"=== +{h} bars ===")

        tool = sub[sub.tool_dir.notna()]
        if len(tool):
            mean_adj = (tool.fwd_return * tool.tool_dir).mean() * 100
            print(f"  Tool directional calls: n={len(tool)} ({len(tool) / len(sub):.1%} of dates), "
                  f"hit rate={tool.tool_hit.astype(float).mean():.1%}, mean fwd return (direction-adjusted)={mean_adj:+.2f}%")
        else:
            print("  Tool directional calls: none cleared the evidence floor at this horizon's dates")

        sma = sub[sub.sma_dir.notna()]
        if len(sma):
            mean_adj = (sma.fwd_return * sma.sma_dir).mean() * 100
            print(f"  50/200 SMA crossover:   n={len(sma)}, hit rate={sma.sma_hit.astype(float).mean():.1%}, "
                  f"mean fwd return (direction-adjusted)={mean_adj:+.2f}%")

        print(f"  Buy-and-hold (all dates): n={len(sub)}, hit rate (positive fwd return)={sub.bh_hit.astype(float).mean():.1%}, "
              f"mean fwd return={sub.fwd_return.mean() * 100:+.2f}%")
        print("  Coin flip (theoretical): hit rate=50.0%, mean fwd return=0.00%")

        if len(tool):
            same = sub.loc[tool.index]
            print(f"  Apples-to-apples -- on the SAME {len(tool)} dates the tool called a direction:")
            print(f"    Buy-and-hold there: hit rate={(same.fwd_return > 0).astype(float).mean():.1%}, "
                  f"mean fwd return={same.fwd_return.mean() * 100:+.2f}%")
            same_sma = same[same.sma_dir.notna()]
            if len(same_sma):
                mean_adj = (same_sma.fwd_return * same_sma.sma_dir).mean() * 100
                print(f"    SMA crossover there (n={len(same_sma)}): hit rate={same_sma.sma_hit.astype(float).mean():.1%}, "
                      f"mean fwd return (direction-adjusted)={mean_adj:+.2f}%")
        print()


def main():
    all_rows = []
    for ticker in BACKTEST_TICKERS:
        print(f"Backtesting {ticker}...")
        rows = run_ticker(ticker)
        all_rows.extend(rows)
        print(f"  {len(rows)} (date, horizon) rows generated")
    if not all_rows:
        print("No rows generated -- nothing to summarize.")
        return
    summarize(all_rows)


if __name__ == "__main__":
    main()
