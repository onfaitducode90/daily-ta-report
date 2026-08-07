#!/usr/bin/env python3
"""
Daily Technical Analysis Report Generator
Generates a morning/evening/intraday technical analysis report for a
defined watchlist using yfinance data only.
"""

import os
import sys
import warnings
from datetime import datetime, timedelta, date, time

import numpy as np
import pandas as pd
from pandas.tseries.holiday import (
    AbstractHolidayCalendar, Holiday, nearest_workday,
    USMartinLutherKingJr, USPresidentsDay, USMemorialDay, USLaborDay,
    USThanksgivingDay, GoodFriday,
)
from scipy import stats as scipy_stats

warnings.filterwarnings("ignore")

try:
    import yfinance as yf
except ImportError:
    print("ERROR: yfinance is required. Install with: pip install yfinance")
    sys.exit(1)

import chart_patterns
import prediction_log
import calibrate
import option_chain
import iv_history

# Horizon (in trading bars) that pattern confidence is calibrated against
# for display purposes -- see calibrate.py. An arbitrary but fixed choice;
# changing it invalidates any saved calibration curves until they're
# refit at the new horizon.
CALIBRATION_HORIZON = 10

try:
    from zoneinfo import ZoneInfo
    ET = ZoneInfo("America/New_York")
except Exception:
    ET = None

WATCHLIST = ["NVDA", "SPCX", "INTC"]
MARKET_CONTEXT_TICKERS = ["SPY", "QQQ"]
ALL_TICKERS = MARKET_CONTEXT_TICKERS + WATCHLIST
HISTORY_DAYS = 300
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Position sizing (F28). Only meaningful for the directional (share-based)
# side of Trade Idea -- an options structure's risk depends on its actual
# premium/margin, not share count, so sizing there is left to you against
# the real chain. Update these if your account size or risk rule changes.
ACCOUNT_SIZE = 230_000
RISK_PCT_PER_TRADE = 0.01
# The 1%-risk-at-1-ATR share count alone has no ceiling: a low-ATR name
# can size up to a large fraction of the account before the stop-distance
# math even notices. A 2nd audit's own live example -- 296 shares of a
# ~$219 stock, $64.6k notional, 28% of a $230k account -- came entirely
# from the risk formula with nothing capping total dollars deployed in
# one name. This caps notional independently of the risk-based count.
MAX_NOTIONAL_PCT_PER_TRADE = 0.20

# A "verified" Jade Lizard (see check_jade_lizard) swaps the put spread's
# long leg for a naked short put -- upside risk becomes fully covered by
# credit, but downside becomes UNDEFINED (assignment exposure = strike x
# 100/contract). A 2nd audit noted this reads as an upgrade while quietly
# handing back the defined-risk-only guarantee a put spread has. Set False
# (the default) to still show the Jade Lizard math as information, but
# never let it read as the tool's actual recommendation; set True to
# allow it if you've decided you're fine holding naked short premium.
ALLOW_NAKED_STRUCTURES = False

# Credit-structure strike selection / staleness / EV gates (2nd Opus audit,
# G1-G4). These don't make the tool a broker-grade pricer -- they exist to
# stop it from presenting a structure as "verified" when the chain is
# stale, the achieved strike misses its delta target, the quote is too
# illiquid to trust the mid, or the tool's own delta-implied numbers say
# the structure is a net loser.
CHAIN_MAX_AGE_DAYS = 3          # reject a snapshot older than this vs. report_date
DELTA_TOLERANCE = 0.05          # reject a strike whose |delta| misses the target by more
MIN_OPEN_INTEREST = 100         # reject a leg with less OI than this
# Gates the COMBINED bid-ask width across BOTH legs, not each leg
# independently -- a per-leg version at 0.5 let each side be up to 50%
# of credit wide, permitting a combined worst-case crossing cost up to
# 2x that (a 3rd audit measured this at 5.6x the EV model's modeled
# commission). half of this combined width is also what the
# haircut_credit fill-quality estimate below is based on.
MAX_COMBINED_BID_ASK_PCT_OF_CREDIT = 0.275
MIN_CREDIT_TO_WIDTH = 1.0 / 3.0  # below this, flag as a thin reward-to-risk trade (common retail rule of thumb)
# 3rd audit: for a normal (concave, saturating) OTM premium curve,
# credit/width is HIGHEST at the narrowest possible width and falls
# monotonically as width grows (verified directly: 0.270 at 1pt of width
# down to 0.181 at 8pt, same short strike, same chain) -- so "search
# wider until the ratio clears MIN_CREDIT_TO_WIDTH" is self-defeating,
# it never succeeds where the narrowest strike didn't (measured: search
# collapsed the credit-structure hit rate on synthetic chains to 2.4%).
# What was actually wrong (round 2's B3) was the WIDTH being spacing-
# arbitrary and asymmetric between a structure's two legs (a live $1 put
# wing beside a $2 call wing), not that it wasn't wide enough. Targeting
# a small, deliberate, volatility-scaled width instead -- applied
# symmetrically to both legs of a combined structure -- fixes that
# without fighting the ratio's own monotonicity. 0.3 matches the ATR
# multiplier calc_volume_poc already uses for intraday bin sizing.
CREDIT_SPREAD_WIDTH_ATR_MULT = 0.3
COMMISSION_PER_CONTRACT_LEG = 0.65  # MODELED flat commission per contract per leg (open+close each count) --
                                     # not your actual broker's schedule, just enough to catch structures whose
                                     # credit doesn't clear typical costs.

# Source: https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm
# (fetched 2026-08-06). Each entry is the SECOND day of the 2-day FOMC
# meeting -- that's the actual decision/press-conference day that moves
# markets, not the first day. This is a market-wide event (not per-ticker),
# so it's checked once for the whole report, not inside analyze_ticker.
# Hardcoded from the official calendar rather than fetched live because
# the Fed publishes the full-year schedule well in advance and it doesn't
# change -- update this list once a year from the source above.
FOMC_DECISION_DATES_2026 = [
    date(2026, 1, 28), date(2026, 3, 18), date(2026, 4, 29), date(2026, 6, 17),
    date(2026, 7, 29), date(2026, 9, 16), date(2026, 10, 28), date(2026, 12, 9),
]


# ---------------------------------------------------------------------------
# Utility / mode helpers
# ---------------------------------------------------------------------------

def get_run_mode():
    """Determine morning/evening/intraday mode based on current ET time,
    and whether "today" is even a trading day (weekday, not a US market
    holiday). Comparing raw wall-clock time against 09:30/16:00 with no
    check on whether the market is actually open meant a Saturday run was
    previously silently labeled "intraday" with no indication the market
    was closed -- callers should use the returned `is_trading_day` flag to
    note that rather than presenting the report as if it reflects today's
    (nonexistent) session."""
    if ET is not None:
        now_et = datetime.now(ET)
    else:
        # Fallback: assume local time is already ET-ish; best effort only.
        now_et = datetime.now()

    market_open = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = now_et.replace(hour=16, minute=0, second=0, microsecond=0)

    if now_et < market_open:
        mode = "morning"
    elif now_et >= market_close:
        mode = "evening"
    else:
        mode = "intraday"

    is_weekday = now_et.weekday() < 5
    is_holiday = False
    if is_weekday:
        is_holiday = len(_US_HOLIDAY_CALENDAR.holidays(
            start=now_et.date(), end=now_et.date())) > 0
    is_trading_day = is_weekday and not is_holiday

    return mode, now_et, is_trading_day


def safe_print(*args, **kwargs):
    print(*args, **kwargs)


# ---------------------------------------------------------------------------
# Data fetching
# ---------------------------------------------------------------------------

def fetch_history(ticker, days=HISTORY_DAYS):
    """Fetch daily OHLCV history for a ticker. Returns DataFrame or None."""
    try:
        period_days = days + 30  # buffer for weekends/holidays
        t = yf.Ticker(ticker)
        df = t.history(period=f"{period_days}d", interval="1d", auto_adjust=False)
        if df is None or df.empty:
            return None
        df = df.tail(days).copy()
        df.index = pd.to_datetime(df.index)
        df = df[["Open", "High", "Low", "Close", "Volume"]]
        df = df.dropna(subset=["Close"])
        if df.empty:
            return None
        return df
    except Exception as e:
        safe_print(f"WARNING: Failed to download history for {ticker}: {e}")
        return None


def fetch_premarket(ticker):
    """Best-effort attempt to fetch pre-market price data. Returns dict or
    None -- and returns None (never a fallback to unfiltered/wrong-timezone
    data) whenever the ET conversion or the pre-market time filter can't be
    trusted. The previous version swallowed a tz_convert failure and ran
    the "<09:30" filter against whatever timezone the raw data happened to
    be in (e.g. UTC, which would select ~20:00-05:30 ET -- overnight bars
    mislabeled as pre-market), and fell all the way through to the FULL
    session if the filter itself raised. A wrong pre-market price silently
    presented as real is worse than no pre-market price at all."""
    if ET is None:
        return None
    try:
        t = yf.Ticker(ticker)
        df = t.history(period="1d", interval="1m", prepost=True)
        if df is None or df.empty:
            return None
        df.index = pd.to_datetime(df.index)
        try:
            df.index = df.index.tz_convert(ET)
        except Exception:
            return None
        premkt = df[df.index.time < datetime.strptime("09:30", "%H:%M").time()]
        if premkt.empty:
            return None
        last = premkt.iloc[-1]
        return {
            "price": float(last["Close"]),
            "volume": float(last.get("Volume", np.nan)),
            "timestamp": premkt.index[-1],
        }
    except Exception:
        return None


def fetch_intraday_bars(ticker, days=6, interval="5m"):
    """Real intraday bars (yfinance 5-minute), for a genuine session VWAP
    and a much finer volume-at-price profile than the daily-bar
    approximations elsewhere in this report use (Opus audit F33 -- the
    5-day "VWAP" and 20-day "POC" built from daily OHLCV are still just
    daily-bar approximations of what are fundamentally intraday concepts).
    Returns None on any failure -- intraday history is only available for
    roughly the last 60 days from yfinance, and this endpoint is flakier
    than the daily one fetch_history uses, so callers must be able to fall
    back to the daily-bar approximation rather than fail the whole report."""
    try:
        t = yf.Ticker(ticker)
        df = t.history(period=f"{days}d", interval=interval, prepost=False)
        if df is None or df.empty:
            return None
        df.index = pd.to_datetime(df.index)
        if ET is not None:
            try:
                df.index = df.index.tz_convert(ET)
            except Exception:
                pass
        df = df[["Open", "High", "Low", "Close", "Volume"]].dropna(subset=["Close"])
        return df if not df.empty else None
    except Exception:
        return None


def calc_session_vwap(intraday_df, session_date=None):
    """Genuine intraday VWAP for ONE trading session: cumulative
    typical-price*volume / cumulative volume across that session's own
    real intraday bars, not a daily-bar approximation. Uses the most
    recent session present in intraday_df unless `session_date` (itself
    present in the data) is given. Returns (vwap, session_date_used,
    is_partial) or (None, None, None).

    is_partial is True when the session's last bar is earlier than the
    normal close (15:55 ET, the start of the last 5-minute bar of a full
    day) -- an intraday run partway through the day would otherwise
    silently return a partial-session VWAP with nothing distinguishing it
    from a completed session's final (different, and no longer moving)
    number."""
    if intraday_df is None or intraday_df.empty:
        return None, None, None
    bar_dates = intraday_df.index.date
    dates_present = sorted(set(bar_dates))
    target = session_date if session_date in dates_present else dates_present[-1]
    session = intraday_df[bar_dates == target]
    if session.empty or session["Volume"].sum() <= 0:
        return None, None, None
    typical = (session["High"] + session["Low"] + session["Close"]) / 3
    vwap = (typical * session["Volume"]).sum() / session["Volume"].sum()
    is_partial = session.index[-1].time() < time(15, 55)
    return float(vwap), target, is_partial


def fetch_vix():
    """Returns (value, timestamp, is_stale) or (None, None, None). ^VIX has
    no pre/post session, so requesting 1-minute bars before 09:30 ET
    silently returns the PREVIOUS session's last minute bar -- indistinguishable
    from a live quote unless the timestamp is checked. is_stale is True
    whenever the returned quote is more than 15 minutes old (e.g. a prior
    close returned outside market hours), so callers can flag it rather
    than present a stale VIX as if it were live -- this can otherwise
    classify the volatility regime off a pre-weekend close after a gap
    event over the weekend."""
    try:
        t = yf.Ticker("^VIX")
        intraday = t.history(period="1d", interval="1m")
        if intraday is not None and not intraday.empty:
            ts = intraday.index[-1]
            value = float(intraday["Close"].iloc[-1])
        else:
            df = t.history(period="10d", interval="1d")
            if df is None or df.empty:
                return None, None, None
            ts = df.index[-1]
            value = float(df["Close"].iloc[-1])

        ts = pd.Timestamp(ts)
        now = datetime.now(ts.tzinfo) if ts.tzinfo is not None else datetime.now()
        is_stale = (now - ts.to_pydatetime()) > timedelta(minutes=15)
        return value, ts, is_stale
    except Exception as e:
        safe_print(f"WARNING: Failed to fetch VIX: {e}")
        return None, None, None


def fetch_next_earnings_date(ticker):
    """Best-effort next earnings date. Returns a date or None -- yfinance's
    earnings calendar is not always populated, so absence isn't treated as
    'no earnings coming', just as 'unknown'."""
    try:
        t = yf.Ticker(ticker)
        cal = t.get_earnings_dates(limit=8)
        if cal is None or cal.empty:
            return None
        now = pd.Timestamp.now(tz=cal.index.tz) if cal.index.tz is not None else pd.Timestamp.now()
        future = cal[cal.index >= now]
        if future.empty:
            return None
        return future.index.min().date()
    except Exception:
        return None


def fetch_next_dividend_date(ticker):
    """Best-effort next ex-dividend date, from yfinance's `calendar` field.
    Returns a date or None. That field has been observed to report the
    MOST RECENT PAST ex-div date rather than a forward projection (e.g. it
    kept returning INTC's 2024-08-06 ex-div long after INTC suspended its
    dividend) -- silently treating a past date as "next" would be wrong,
    not just missing, so a date is only returned if it's actually in the
    future relative to `today`."""
    try:
        t = yf.Ticker(ticker)
        cal = t.calendar
        if not cal:
            return None
        ex_div = cal.get("Ex-Dividend Date")
        if ex_div is None:
            return None
        if isinstance(ex_div, datetime):
            ex_div = ex_div.date()
        if ex_div <= date.today():
            return None
        return ex_div
    except Exception:
        return None


def fetch_real_iv_context(ticker, hv30_val, report_date, min_days=5, max_age_days=CHAIN_MAX_AGE_DAYS):
    """Best-effort REAL implied-volatility read from a locally downloaded
    ThinkorSwim option-chain snapshot (option_chain.py), for tickers where
    one exists. Returns None if no chain is available -- callers must fall
    back to the HV-percentile proxy, never fabricate a number.

    Prefers comparing real IV directly against realized vol (HV30) -- the
    classic variance-risk-premium framing (premium selling is favorable
    when IV exceeds what has actually been realized, not just when IV is
    "high" in some absolute sense) -- over trying to build an IV-rank
    percentile, since only a handful of daily snapshots exist locally
    (nowhere near the ~252 trading days a real percentile needs).

    `report_date` anchors every staleness/DTE check to the report's own
    date -- not wall-clock "today", and NOT the chain snapshot's own date,
    which this function used to silently trust. A 2nd audit caught that: a
    week-old snapshot could hand back an expiration that had already
    passed relative to today, with dte computed as if the snapshot were
    current. Now this returns None outright once the snapshot is older
    than `max_age_days`, and again if the nearest qualifying expiration
    turns out to have already passed report_date."""
    try:
        chain = option_chain.load_chain(ticker)
    except Exception:
        return None
    if chain is None:
        return None
    snapshot_date = chain.snapshot_time.date()
    if (report_date - snapshot_date).days > max_age_days:
        return None
    expiration = chain.nearest_expiration(min_days=min_days)
    if expiration is None or expiration <= report_date:
        return None
    iv = chain.atm_iv(expiration)
    if iv is None:
        return None
    dte = (expiration - report_date).days
    # A RATIO (iv_ratio), not the raw point spread (iv_minus_hv, kept for
    # display/backward compat) -- a 2nd audit noted the point spread isn't
    # scale-invariant: 5 points on a 15%-IV name is a 33% premium, 5
    # points on a 95%-IV name (SPCX) is barely 5%. iv_ratio makes "rich"
    # mean the same thing regardless of the name's baseline vol level.
    iv_ratio = (iv / hv30_val) if hv30_val is not None and hv30_val > 0 else None
    return {
        "chain": chain, "expiration": expiration, "dte": dte, "iv": iv,
        "snapshot_date": snapshot_date,
        "iv_minus_hv": (iv - hv30_val) if hv30_val is not None else None,
        "iv_ratio": iv_ratio,
    }


def build_credit_spread(chain, expiration, side, target_short_delta=0.25, target_width=None):
    """Pick real strikes for a put or call credit spread from a loaded
    option chain: the short leg is whichever strike's |delta| is closest
    to target_short_delta.

    The long leg is chosen by TARGET WIDTH, not by blindly taking the
    adjacent strike: among every further-OTM, liquidity-passing strike,
    this picks the one whose width is CLOSEST to `target_width` (pass the
    ticker's own ATR for a volatility-scaled, comparable-across-tickers
    width; None falls back to the narrowest liquidity-passing strike).

    An earlier version of this fix tried "the narrowest width whose
    credit/width clears a 1/3 ratio floor" -- which turned out to be
    self-defeating: for a normal (concave, saturating) OTM premium curve,
    credit/width is HIGHEST at the narrowest possible width and decreases
    monotonically as width grows (verified directly against a synthetic
    chain: ratio fell from 0.270 at 1 point of width to 0.181 at 8).
    Widening a spread never improves its ratio for a well-behaved chain,
    so gating the width SEARCH on that ratio just rejected almost
    everything (hit rate on synthetic chains: 2.4%). Targeting a
    consistent WIDTH instead -- not a ratio -- is what actually fixes
    the two real problems: a 2nd audit's asymmetric-wings finding (a $1
    put wing beside a $2 call wing, purely from where strike spacing
    happened to change) and a 3rd audit's 100%-firing thin-credit caveat
    (which fired every time because "the narrowest available strike" is
    definitionally the same shape every time -- an ATR-scaled target
    gives the resulting ratio real variance to actually be informative
    about, instead of being deterministic).

    Returns None (never a fabricated/guessed strike, and never a
    structure this tool can't stand behind) if:
      - delta data isn't populated for this expiration,
      - the achieved short delta misses target_short_delta by more than
        DELTA_TOLERANCE (a sparse delta column, common in the wings of a
        ToS export, can otherwise hand back a strike nowhere near what was
        asked for with no indication anything was off-target),
      - no further-OTM strike on the chain passes the liquidity checks
        below (with target_width given, this considers every candidate
        rather than stopping at the first; with target_width=None it
        still just needs the nearest one to pass),
      - the resulting credit would be >= the width (a stale/crossed/wide
        quote can otherwise produce a zero-or-negative max_loss, which
        crashed the report the first time a bad quote hit it),
      - either leg's own bid-ask is too wide relative to the credit to
        trust its mid, or
      - either leg's open interest is too thin to trust the market is
        real (this and the bid-ask check use data option_chain.py already
        parses -- greeks, OI, both bid/ask legs -- that nothing was
        checking before)."""
    quotes = sorted(chain.for_expiration(expiration), key=lambda q: q.strike)
    if side == "put":
        delta_attr, mid_attr, iv_attr = "put_delta", "put_mid", "put_iv"
        bid_attr, ask_attr, oi_attr = "put_bid", "put_ask", "put_open_interest"
    elif side == "call":
        delta_attr, mid_attr, iv_attr = "call_delta", "call_mid", "call_iv"
        bid_attr, ask_attr, oi_attr = "call_bid", "call_ask", "call_open_interest"
    else:
        raise ValueError(f"side must be 'put' or 'call', got {side!r}")

    def leg_bid_ask(leg):
        bid, ask = getattr(leg, bid_attr), getattr(leg, ask_attr)
        if bid is None or ask is None:
            return None
        return bid, ask

    def leg_oi_ok(leg):
        oi = getattr(leg, oi_attr)
        return oi is not None and oi >= MIN_OPEN_INTEREST

    candidates = [q for q in quotes if getattr(q, delta_attr) is not None]
    if not candidates:
        return None
    short_q = min(candidates, key=lambda q: abs(abs(getattr(q, delta_attr)) - target_short_delta))
    achieved_delta = abs(getattr(short_q, delta_attr))
    if abs(achieved_delta - target_short_delta) > DELTA_TOLERANCE:
        return None
    short_ba = leg_bid_ask(short_q)
    if short_ba is None or not leg_oi_ok(short_q):
        return None
    short_bid, short_ask = short_ba

    idx = quotes.index(short_q)
    # Further OTM means lower strikes for a put, higher strikes for a
    # call -- iterate nearest-to-short-strike first in both cases.
    further_strikes = list(reversed(quotes[:idx])) if side == "put" else quotes[idx + 1:]

    short_mid = getattr(short_q, mid_attr)
    if short_mid is None:
        return None

    best = None
    for long_q in further_strikes:
        long_mid = getattr(long_q, mid_attr)
        if long_mid is None:
            continue
        credit = short_mid - long_mid
        width = abs(short_q.strike - long_q.strike)
        if credit <= 0 or width <= 0 or credit >= width:
            continue
        long_ba = leg_bid_ask(long_q)
        if long_ba is None or not leg_oi_ok(long_q):
            continue
        long_bid, long_ask = long_ba
        # Gate on the COMBINED bid-ask width across BOTH legs as a share
        # of credit, not each leg independently -- a 3rd audit found the
        # per-leg version let each leg independently be up to
        # MAX_BID_ASK_PCT_OF_CREDIT wide, permitting a combined worst-case
        # crossing cost up to 2x that share of credit (e.g. up to $14.50
        # of unmodeled slippage per contract against a $2.60 modeled
        # commission -- 5.6x the cost the EV model actually accounted
        # for). combined_width is also what the fill-quality haircut
        # below is based on.
        combined_width = (short_ask - short_bid) + (long_ask - long_bid)
        if combined_width > MAX_COMBINED_BID_ASK_PCT_OF_CREDIT * credit:
            continue
        candidate = (long_q, credit, width, combined_width)
        if target_width is None:
            # No target -- the nearest liquidity-passing strike IS the
            # answer (it's also the highest-ratio one, since ratio only
            # decreases from here as width grows).
            best = candidate
            break
        if best is None or abs(width - target_width) < abs(best[2] - target_width):
            best = candidate
        elif width > target_width and abs(width - target_width) > abs(best[2] - target_width):
            # Strikes get monotonically farther from target_width past
            # this point (we're iterating nearest-to-short-strike first,
            # so width is increasing) -- once a step moves further away
            # than the current best, every later one will too.
            break

    if best is None:
        return None
    long_q, credit, width, combined_width = best
    # "Fill at mid" is optimistic -- a real entry crosses part of the
    # spread on at least one leg. haircut_credit assumes you give up HALF
    # of the combined bid-ask width (i.e. filling roughly at the midpoint
    # between mid and the worse side on each leg, not at the natural mid
    # on both) -- used for EV purposes alongside the honestly-labeled
    # mid-based `credit` shown in the report text.
    haircut_credit = credit - 0.5 * combined_width
    return {
        "short_strike": short_q.strike, "long_strike": long_q.strike,
        "credit": credit, "width": width, "max_loss": width - credit,
        "haircut_credit": haircut_credit, "haircut_max_loss": width - haircut_credit,
        "short_delta": getattr(short_q, delta_attr),
        # Real IV at the ACTUAL strike being sold, not the ATM figure the
        # rich/cheap gate above uses -- a 2nd audit noted ATM IV can be far
        # from OTM IV on a skewed name, so "rich ATM" says little about
        # whether this specific strike is rich. None if this expiration's
        # export didn't have it (e.g. "--"/"<empty>" on a thin strike).
        "short_iv": getattr(short_q, iv_attr),
    }


def check_jade_lizard(chain, expiration, put_spread, target_short_call_delta=0.25, target_width=None):
    """Given an already-selected bull put spread, check whether swapping
    its long put for a naked short put and adding a short call spread on
    top prices out as a true Jade Lizard: upside-riskless requires the
    total credit collected (naked put + call spread) to be at least the
    call spread's own width. Returns None if a call spread can't be built
    from the chain; otherwise returns the numbers either way so the caller
    can report an honest pass/fail rather than only a pass."""
    quotes = sorted(chain.for_expiration(expiration), key=lambda q: q.strike)
    short_put_q = next((q for q in quotes if q.strike == put_spread["short_strike"]), None)
    if short_put_q is None or short_put_q.put_mid is None:
        return None
    call_spread = build_credit_spread(chain, expiration, "call", target_short_call_delta, target_width)
    if call_spread is None:
        return None
    total_credit = short_put_q.put_mid + call_spread["credit"]
    return {
        "call_short_strike": call_spread["short_strike"],
        "call_long_strike": call_spread["long_strike"],
        "call_width": call_spread["width"],
        "total_credit": total_credit,
        "verified": total_credit >= call_spread["width"],
    }


def bs_true_p_itm(delta, iv, dte_days, side):
    """Black-Scholes risk-neutral P(finishes in the money) for one leg,
    recovered from its own quoted delta/IV/DTE -- NOT the same number as
    |delta|, and a 3rd audit's own EV-model review caught that
    |delta| is biased in OPPOSITE directions for calls vs. puts, not
    uniformly as that audit's summary implied:

    delta = N(d1); true P(ITM) = N(d2) for a call, N(-d2) for a put;
    d1 - d2 = IV*sqrt(T) > 0 always. Since N is increasing:
      calls: N(d1) > N(d2)   -> |delta| OVERSTATES true P(ITM)
                                 -> 1-|delta| UNDERSTATES true P(win)
      puts:  N(-d1) < N(-d2) -> |delta| UNDERSTATES true P(ITM)
                                 -> 1-|delta| OVERSTATES true P(win)
    Verified numerically at sigma=0.30/DTE=7 and sigma=0.60/DTE=21 against
    a strike solved to match the target delta exactly, both directions
    confirmed. This matters here because the Bull Put Spread and the
    Jade Lizard's naked put -- this tool's two most common credit
    structures -- are put-side, where the naive formula is too
    OPTIMISTIC (understates loss probability), not too conservative.

    Recovers d1 from delta via the inverse normal CDF, then converts to
    d2 using the leg's own IV and time to expiration -- no spot/strike
    needed. Falls back to |delta| (the old proxy) if IV or DTE is
    missing, since d1 can't be recovered without them."""
    if delta is None:
        return None
    if iv is None or iv <= 0 or dte_days is None or dte_days <= 0:
        return abs(delta)
    years = dte_days / 365.0
    # Clip to keep ppf() finite for a delta that (due to quote noise) sits
    # at exactly 0 or 1.
    if side == "call":
        d1 = scipy_stats.norm.ppf(min(max(delta, 1e-6), 1 - 1e-6))
        d2 = d1 - iv * np.sqrt(years)
        p_itm = scipy_stats.norm.cdf(d2)
    else:
        d1 = scipy_stats.norm.ppf(min(max(delta + 1, 1e-6), 1 - 1e-6))
        d2 = d1 - iv * np.sqrt(years)
        p_itm = scipy_stats.norm.cdf(-d2)
    return float(np.clip(p_itm, 0.0, 1.0))


def estimate_credit_structure_ev(p_win, credit, max_loss, num_legs):
    """Rough per-contract expected value from a structure's win
    probability (see bs_true_p_itm for how that's derived from the
    structure's own delta/IV/DTE -- not a real probability model, since
    this still ignores gamma/time-decay path and the loss distribution
    conditional on breach, but a materially more accurate P(win) than the
    1-|delta| proxy this replaced). The point isn't precision, it's
    refusing to print a structure that is a loser by ITS OWN inputs
    before you've even paid a real bid-ask spread to get in.

    Callers should pass `credit`/`max_loss` from build_credit_spread's
    `haircut_credit`/`haircut_max_loss` (fill assumed halfway between mid
    and the worse side on each leg), not the raw mid-based `credit`/
    `max_loss` also on that dict -- a 3rd audit found the gate permitted
    up to 5.6x more unmodeled bid-ask slippage than the commission term
    below actually models; the haircut folds a real fill-quality estimate
    into EV instead of assuming a fill exactly at mid.

    `num_legs` is the total option legs (2 for a vertical, 4 for an iron
    condor) -- commission is modeled as open+close on every leg, so
    round-trip cost = num_legs * 2 * COMMISSION_PER_CONTRACT_LEG."""
    p_loss = 1 - p_win
    ev_gross = (p_win * credit - p_loss * max_loss) * 100
    commission = num_legs * 2 * COMMISSION_PER_CONTRACT_LEG
    ev_net = ev_gross - commission
    return {"p_win": p_win, "p_loss": p_loss, "ev_gross": ev_gross,
            "commission": commission, "ev_net": ev_net}


def credit_structure_caveats(ev, credit, width):
    """Plain-language flags for a credit structure that is technically
    "verified" (real strikes, positive credit) but still a bad trade by
    the tool's own numbers -- never hides the structure, just refuses to
    let a real chain snapshot read as an implicit recommendation."""
    caveats = []
    if ev is not None and ev["ev_net"] <= 0:
        caveats.append(f"NEGATIVE EXPECTED VALUE by this tool's own Black-Scholes-implied odds: "
                        f"P(win) ~{ev['p_win']:.0%}, EV ${ev['ev_gross']:+.0f}/contract gross "
                        "(credit already haircut for estimated bid-ask fill quality, not the mid), "
                        f"${ev['ev_net']:+.0f}/contract after ~${ev['commission']:.0f} modeled "
                        "round-trip commissions. Do not treat this as a recommendation.")
    ratio = credit / width if width else 0
    if ratio < MIN_CREDIT_TO_WIDTH:
        caveats.append(f"Thin reward-to-risk: credit is only {ratio:.0%} of width "
                        f"(below the {MIN_CREDIT_TO_WIDTH:.0%} rule-of-thumb minimum) -- "
                        "risking a lot to make a little.")
    return caveats


# ---------------------------------------------------------------------------
# Swing structure
# ---------------------------------------------------------------------------

def _screen_outlier_ranges(df, max_range_atr_mult=5.0):
    """Cap High/Low for bars whose range is an extreme multiple of a local
    ATR proxy, for pivot-detection purposes only. A single bad print (a
    known data-quality issue -- Yahoo/yfinance occasionally returns a
    spurious spike) can otherwise create a phantom swing pivot that anchors
    a trendline for the rest of the lookback window, with nothing else in
    the pipeline positioned to catch it. Returns plain numpy arrays, not a
    modified DataFrame -- the original df is untouched everywhere else
    (wick analysis, ATR itself, price displays all still see the real
    print)."""
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
    """Identify swing highs/lows: bar's high/low is higher/lower than
    `lookback` bars before and after it."""
    highs, lows = _screen_outlier_ranges(df)
    n = len(df)
    swing_highs = []  # (index_pos, price)
    swing_lows = []

    for i in range(lookback, n - lookback):
        window_high = highs[i - lookback:i + lookback + 1]
        if highs[i] == window_high.max() and np.sum(window_high == highs[i]) == 1:
            swing_highs.append((i, highs[i]))
        window_low = lows[i - lookback:i + lookback + 1]
        if lows[i] == window_low.min() and np.sum(window_low == lows[i]) == 1:
            swing_lows.append((i, lows[i]))

    return swing_highs, swing_lows


def classify_structure(df, lookback_bars=60, swing_lookback=3):
    """Classify trend structure using swing highs/lows over the last N bars."""
    sub = df.tail(lookback_bars) if len(df) > lookback_bars else df
    if len(sub) < (swing_lookback * 2 + 1) * 3:
        # not much data, still try
        pass

    swing_highs, swing_lows = find_swings(sub, lookback=swing_lookback)

    if len(swing_highs) < 2 or len(swing_lows) < 2:
        return "RANGE", "Insufficient swing points for clear structure", swing_highs, swing_lows

    last_highs = [p for _, p in swing_highs[-3:]]
    last_lows = [p for _, p in swing_lows[-3:]]

    highs_rising = all(last_highs[i] < last_highs[i + 1] for i in range(len(last_highs) - 1))
    highs_falling = all(last_highs[i] > last_highs[i + 1] for i in range(len(last_highs) - 1))
    lows_rising = all(last_lows[i] < last_lows[i + 1] for i in range(len(last_lows) - 1))
    lows_falling = all(last_lows[i] > last_lows[i + 1] for i in range(len(last_lows) - 1))

    if highs_rising and lows_rising:
        return "UPTREND", "HH+HL structure confirmed", swing_highs, swing_lows
    elif highs_falling and lows_falling:
        return "DOWNTREND", "LH+LL structure confirmed", swing_highs, swing_lows
    else:
        return "RANGE", "Mixed swing structure", swing_highs, swing_lows


def resample_weekly(df):
    """Resample daily OHLCV bars to weekly (Friday-close) bars, for a
    genuine higher-timeframe read. What this tool previously called "HTF
    Trend" was classify_structure run over the last 60 DAILY bars -- fewer
    bars of the same timeframe, not a higher one (Opus audit F31). The most
    recent row may be a still-forming partial week."""
    weekly = df.resample("W-FRI").agg({
        "Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum",
    }).dropna(subset=["Close"])
    return weekly


# ---------------------------------------------------------------------------
# Indicators
# ---------------------------------------------------------------------------

def ema(series, span):
    return series.ewm(span=span, adjust=False).mean()


def sma(series, window):
    return series.rolling(window=window).mean()


def wilders_smooth(series, period):
    """Wilder's smoothing: first value = simple average of the first
    `period` values AFTER any leading NaN run is stripped (a raw
    np.nanmean(values[:period]) would silently average only period-1 real
    values whenever the input starts with NaN -- e.g. RSI's gain/loss series
    from .diff() -- and anchor the seed one bar too early). Interior NaN
    (rare; e.g. a momentary 0/0 in a DI ratio) is treated as 0 rather than
    dropped, since dropping it would silently shift every subsequent value's
    place in the recursion instead of just filling the gap."""
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


def true_range(df):
    prev_close = df["Close"].shift(1)
    tr1 = df["High"] - df["Low"]
    tr2 = (df["High"] - prev_close).abs()
    tr3 = (df["Low"] - prev_close).abs()
    return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)


def calc_atr(df, period=14):
    tr = true_range(df)
    return wilders_smooth(tr, period)


def calc_rsi(close, period=14):
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = wilders_smooth(gain, period)
    avg_loss = wilders_smooth(loss, period)
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi[(avg_loss == 0) & (avg_gain > 0)] = 100
    rsi[(avg_loss == 0) & (avg_gain == 0)] = 50
    return rsi


def calc_macd(close, fast=12, slow=26, signal=9):
    ema_fast = ema(close, fast)
    ema_slow = ema(close, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def calc_adx(df, period=14):
    high = df["High"]
    low = df["Low"]

    prev_high = high.shift(1)
    prev_low = low.shift(1)

    up_move = high - prev_high
    down_move = prev_low - low

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    plus_dm = pd.Series(plus_dm, index=df.index)
    minus_dm = pd.Series(minus_dm, index=df.index)

    tr = true_range(df)

    smoothed_tr = wilders_smooth(tr, period)
    smoothed_plus_dm = wilders_smooth(plus_dm, period)
    smoothed_minus_dm = wilders_smooth(minus_dm, period)

    plus_di = 100 * smoothed_plus_dm / smoothed_tr.replace(0, np.nan)
    minus_di = 100 * smoothed_minus_dm / smoothed_tr.replace(0, np.nan)

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    # dx is NaN for its own ~2*period leading bars (it's derived from the
    # Wilder-smoothed +DM/-DM/TR, which are themselves NaN until their seed
    # position). fillna(0) used to convert that entire lead-in into
    # fabricated zero DX readings and seed ADX's own smoothing from them --
    # wilders_smooth now strips a genuine leading NaN run itself, so ADX
    # seeds from the first real DX values instead of manufactured zeros.
    adx = wilders_smooth(dx, period)

    return adx, plus_di, minus_di


def calc_bollinger(close, period=20, num_std=2.0):
    mid = sma(close, period)
    std = close.rolling(window=period).std(ddof=0)
    upper = mid + num_std * std
    lower = mid - num_std * std
    return upper, mid, lower


def calc_keltner(df, period=20, atr_period=20, mult=1.5):
    mid = ema(df["Close"], period)
    tr = true_range(df)
    atr = wilders_smooth(tr, atr_period)
    upper = mid + mult * atr
    lower = mid - mult * atr
    return upper, mid, lower


def calc_ttm_squeeze(df):
    """Returns dict with squeeze bool series, consecutive count, fired flag,
    and momentum histogram (linreg of delta)."""
    bb_upper, bb_mid, bb_lower = calc_bollinger(df["Close"], 20, 2.0)
    kc_upper, kc_mid, kc_lower = calc_keltner(df, 20, 20, 1.5)

    squeeze_on = (bb_lower > kc_lower) & (bb_upper < kc_upper)

    # consecutive squeeze-on counter
    consec = np.zeros(len(df), dtype=int)
    count = 0
    for i in range(len(df)):
        if bool(squeeze_on.iloc[i]) if not pd.isna(squeeze_on.iloc[i]) else False:
            count += 1
        else:
            count = 0
        consec[i] = count

    highest_high = df["High"].rolling(window=20).max()
    lowest_low = df["Low"].rolling(window=20).min()
    ema20 = ema(df["Close"], 20)
    delta = df["Close"] - ((highest_high + lowest_low) / 2 + ema20) / 2

    momentum = pd.Series(np.full(len(df), np.nan), index=df.index)
    window = 20
    delta_vals = delta.values
    for i in range(window - 1, len(df)):
        seg = delta_vals[i - window + 1:i + 1]
        if np.any(np.isnan(seg)):
            continue
        x = np.arange(window)
        try:
            coeffs = np.polyfit(x, seg, 1)
            fitted = coeffs[0] * (window - 1) + coeffs[1]
            momentum.iloc[i] = fitted
        except Exception:
            pass

    return {
        "squeeze_on": squeeze_on,
        "consec": pd.Series(consec, index=df.index),
        "momentum": momentum,
        "bb_upper": bb_upper, "bb_mid": bb_mid, "bb_lower": bb_lower,
        "kc_upper": kc_upper, "kc_mid": kc_mid, "kc_lower": kc_lower,
    }


def calc_rolling_vwap(df, window=5):
    sub = df.tail(window)
    typical_price = (sub["High"] + sub["Low"] + sub["Close"]) / 3
    vwap = (typical_price * sub["Volume"]).sum() / sub["Volume"].sum()
    return float(vwap)


def calc_volume_poc(df, window=20, atr_val=None, bins=None):
    """Volume point-of-control. Each bar's volume is distributed
    PROPORTIONALLY across every bin its High-Low range overlaps (assuming
    volume is spread uniformly across the bar's range -- still an
    approximation with daily bars, since we don't have intraday
    volume-at-price, but a real distribution rather than the previous
    behavior of dumping 100% of a bar's volume into whichever single bin
    happened to contain its typical price).

    Bin count is sized so each bin is roughly 0.3 ATR wide when `atr_val` is
    given -- finer bins just chase daily noise, coarser ones wash out any
    real concentration. Falls back to a fixed 20 bins if no ATR is
    available. An explicit `bins` always overrides both."""
    sub = df.tail(window)
    if sub.empty:
        return None
    price_min = sub["Low"].min()
    price_max = sub["High"].max()
    price_range = price_max - price_min
    if price_range <= 0:
        return float(price_max)

    if bins is None:
        if atr_val is not None and atr_val > 0:
            bin_width = 0.3 * atr_val
            bins = max(int(round(price_range / bin_width)), 5)
        else:
            bins = 20

    bin_edges = np.linspace(price_min, price_max, bins + 1)
    bin_volumes = np.zeros(bins)

    for _, row in sub.iterrows():
        low, high, vol = row["Low"], row["High"], row["Volume"]
        if high <= low:
            idx = min(max(int(np.searchsorted(bin_edges, low, side="right")) - 1, 0), bins - 1)
            bin_volumes[idx] += vol
            continue
        overlap_lo = np.maximum(bin_edges[:-1], low)
        overlap_hi = np.minimum(bin_edges[1:], high)
        overlap = np.clip(overlap_hi - overlap_lo, 0, None)
        bin_volumes += (overlap / (high - low)) * vol

    max_bin = int(np.argmax(bin_volumes))
    poc = (bin_edges[max_bin] + bin_edges[max_bin + 1]) / 2
    return float(poc)


def calc_intraday_poc_bins(intraday_df, min_bins=20, max_bins=40):
    """Bin count for calc_volume_poc when called on real intraday bars
    (not daily ones) -- neither of the two things tried before this
    worked: a fixed bins=100 was 0.0146 daily-ATR per bin (3.5x less
    stable under bootstrap resampling than ATR-sized bins, i.e. overfit
    to individual 5-minute prints), and sizing off the DAILY ATR (the
    0.3*atr_val default in calc_volume_poc) gave only ~5 bins across a
    multi-session intraday range -- too coarse to be a profile at all.
    Targets a bin width of roughly one typical 5-minute bar's own range
    -- a natural intraday scale that needs no daily-vs-intraday
    conversion factor -- clamped to [min_bins, max_bins]."""
    price_range = float(intraday_df["High"].max() - intraday_df["Low"].min())
    if price_range <= 0:
        return min_bins
    median_bar_range = float((intraday_df["High"] - intraday_df["Low"]).median())
    if median_bar_range <= 0:
        return min_bins
    return int(min(max(round(price_range / median_bar_range), min_bins), max_bins))


def calc_beta_correlation(ticker_df, spy_df, window=126, recent_window=63):
    """Beta and correlation of this ticker's daily returns against SPY's,
    via OLS (scipy.stats.linregress, which also hands back the slope's
    standard error for free) over the trailing `window` trading days both
    series actually have in common. Distinguishes an idiosyncratic,
    name-specific move from a market-wide one riding on the same tape
    (Opus audit F32).

    A 2nd audit simulated this at a KNOWN true beta and found the
    original 63-day window's beta estimate for a high-idiosyncratic-vol
    name (sigma 5.5%/day, e.g. SPCX) had a standard deviation of 0.81
    around a true beta of 1.6 -- a 95% range of roughly [0.12, 3.07],
    i.e. close to uninformative -- while the SAME window's correlation
    averaged only 0.24, because high idiosyncratic vol mechanically
    suppresses correlation without reducing market exposure at all.
    Doubling the window to 126 days cut that beta sd roughly in half in
    the same simulation. window=60 is this function's hard floor (below
    that, returns None rather than print a number the audit's own
    simulation showed was closer to noise than signal); `recent_window`
    is a secondary, faster-moving read of the same relationship, kept
    separate so a regime change shows up without discarding the more
    stable primary estimate.

    Returns None if either series is missing or there isn't enough
    overlapping history for the primary window."""
    if ticker_df is None or spy_df is None:
        return None
    t_ret = ticker_df["Close"].pct_change().dropna()
    s_ret = spy_df["Close"].pct_change().dropna()
    t_ret.index = t_ret.index.date
    s_ret.index = s_ret.index.date
    joined = pd.concat([t_ret, s_ret], axis=1, join="inner").tail(window)
    joined.columns = ["ticker", "spy"]
    joined = joined.dropna()
    if len(joined) < 60:
        return None

    reg = scipy_stats.linregress(joined["spy"].values, joined["ticker"].values)
    if np.isnan(reg.slope) or np.isnan(reg.rvalue):
        return None

    result = {
        "beta": float(reg.slope), "beta_stderr": float(reg.stderr),
        "correlation": float(reg.rvalue), "n": len(joined),
    }

    recent = joined.tail(recent_window)
    if len(recent) >= 20:
        reg_r = scipy_stats.linregress(recent["spy"].values, recent["ticker"].values)
        if not (np.isnan(reg_r.slope) or np.isnan(reg_r.rvalue)):
            result["recent_beta"] = float(reg_r.slope)
            result["recent_correlation"] = float(reg_r.rvalue)
            result["recent_n"] = len(recent)

    return result


def calc_hv(close, window=30, annualize=252):
    log_returns = np.log(close / close.shift(1))
    recent = log_returns.tail(window).dropna()
    if len(recent) < 2:
        return None
    hv = recent.std(ddof=1) * np.sqrt(annualize)
    return float(hv)


def calc_hv_series(close, window=30, annualize=252):
    log_returns = np.log(close / close.shift(1))
    hv_series = log_returns.rolling(window=window).std(ddof=1) * np.sqrt(annualize)
    return hv_series


def calc_iv_rank(close, window=30, lookback=252):
    hv_series = calc_hv_series(close, window=window).dropna()
    if len(hv_series) < 2:
        return None
    hv_lookback = hv_series.tail(lookback)
    current_hv = hv_lookback.iloc[-1]
    rank = scipy_stats.percentileofscore(hv_lookback.values, current_hv)
    return float(rank)


class NYSEHolidayCalendar(AbstractHolidayCalendar):
    """NYSE market holidays. USFederalHolidayCalendar (used here
    previously) is a FEDERAL GOVERNMENT holiday list, not a market one --
    it wrongly included Columbus Day and Veterans Day (NYSE is open on
    both) and omitted Good Friday (NYSE is closed, no federal holiday
    covers it -- e.g. 2026-04-03). A 2nd audit caught this still being
    wrong despite the caveat in the docstring below acknowledging it.
    Reuses pandas' own rule objects for every date this calendar shares
    with the federal one, so those stay in sync automatically; only the
    NYSE-specific differences are hand-written. Still not exhaustive --
    one-off closures (e.g. a national day of mourning) aren't captured --
    but no longer wrong on two recurring, predictable dates every year."""
    rules = [
        Holiday("New Year's Day", month=1, day=1, observance=nearest_workday),
        USMartinLutherKingJr,
        USPresidentsDay,
        GoodFriday,
        USMemorialDay,
        Holiday("Juneteenth National Independence Day", month=6, day=19,
                start_date="2021-06-18", observance=nearest_workday),
        Holiday("Independence Day", month=7, day=4, observance=nearest_workday),
        USLaborDay,
        USThanksgivingDay,
        Holiday("Christmas Day", month=12, day=25, observance=nearest_workday),
    ]


_US_HOLIDAY_CALENDAR = NYSEHolidayCalendar()


def days_until_next_friday(from_date):
    """TRADING days from `from_date` (exclusive) through the next Friday
    (inclusive). The caller scales volatility by sqrt(this), a trading-day
    convention -- returning calendar days instead overstated the expected
    move by 18% when run on a Friday and 9.5% on a Saturday, since a
    calendar-day count includes weekend days with no price movement to
    scale for. Uses NYSEHolidayCalendar (see above) as the market-holiday
    proxy -- not perfectly NYSE-accurate (one-off closures aren't
    captured), but correct on every recurring NYSE holiday, including the
    two the previous federal-calendar version got wrong."""
    calendar_days_ahead = (4 - from_date.weekday()) % 7  # Friday = weekday 4
    if calendar_days_ahead == 0:
        calendar_days_ahead = 7  # next Friday, not today
    next_friday = from_date + timedelta(days=calendar_days_ahead)
    bdays = pd.bdate_range(start=from_date + timedelta(days=1), end=next_friday)
    holidays = _US_HOLIDAY_CALENDAR.holidays(start=from_date, end=next_friday)
    trading_days = bdays[~bdays.isin(holidays)]
    return max(len(trading_days), 1)


# ---------------------------------------------------------------------------
# Candle pattern recognition
# ---------------------------------------------------------------------------

def detect_inside_bars(df):
    """Return count of consecutive inside bars ending at the most recent bar."""
    if len(df) < 2:
        return 0
    count = 0
    for i in range(len(df) - 1, 0, -1):
        today = df.iloc[i]
        yday = df.iloc[i - 1]
        if today["High"] < yday["High"] and today["Low"] > yday["Low"]:
            count += 1
        else:
            break
    return count


def wick_analysis(df, n=3):
    if len(df) < n:
        n = len(df)
    if n == 0:
        return None
    sub = df.tail(n)
    upper_wicks = sub["High"] - sub[["Open", "Close"]].max(axis=1)
    lower_wicks = sub[["Open", "Close"]].min(axis=1) - sub["Low"]
    avg_upper = upper_wicks.mean()
    avg_lower = lower_wicks.mean()
    if avg_upper > avg_lower * 1.2:
        return "Upper wicks dominant — selling pressure at highs", avg_upper, avg_lower
    elif avg_lower > avg_upper * 1.2:
        return "Lower wicks dominant — buyers defending lows", avg_upper, avg_lower
    else:
        return "No dominant wick bias", avg_upper, avg_lower


def acceptance_or_rejection(df, key_zone, zone_label):
    if key_zone is None or len(df) < 3:
        return None
    last3 = df["Close"].tail(3)
    last3_highs = df["High"].tail(3)
    last3_lows = df["Low"].tail(3)

    above_count = (last3 > key_zone).sum()
    below_count = (last3 < key_zone).sum()

    wicked_through = ((last3_highs > key_zone) & (last3 < key_zone)).any() or \
                      ((last3_lows < key_zone) & (last3 > key_zone)).any()

    if above_count == 3:
        return f"Acceptance above {zone_label} (${key_zone:.2f}) — 3 consecutive closes above"
    elif below_count == 3:
        return f"Acceptance below {zone_label} (${key_zone:.2f}) — 3 consecutive closes below"
    elif wicked_through:
        return f"Rejection at {zone_label} (${key_zone:.2f}) — wick through, close back"
    else:
        return f"Mixed price action around {zone_label} (${key_zone:.2f})"


def detect_break_and_retest(df, levels):
    """levels: dict label -> price. Look over last 10 bars for a break and
    retest within 1%."""
    if len(df) < 10:
        return None
    sub = df.tail(10)
    current_price = df["Close"].iloc[-1]
    results = []
    for label, level in levels.items():
        if level is None or np.isnan(level):
            continue
        broke_above = (sub["Close"] > level).any() and (sub["Close"].iloc[0] <= level or sub["Open"].iloc[0] <= level)
        broke_below = (sub["Close"] < level).any() and (sub["Close"].iloc[0] >= level or sub["Open"].iloc[0] >= level)
        near = abs(current_price - level) / level <= 0.01
        if near and (broke_above or broke_below):
            results.append(f"Break and retest occurring at ${level:.2f} ({label})")
    return results if results else None


def rsi_divergence(df, rsi_series, lookback_bars=60, swing_lookback=3,
                    min_pivot_gap=5, min_rsi_gap=8.0, max_pivot_age=10):
    """Classical divergence compares RSI at CONFIRMED swing pivots, not
    today's bar against a running N-bar extreme -- the old approach fired
    on any marginal new high/low regardless of whether a real pivot
    structure existed, and mixed an intraday High/Low price series against
    a Close-based RSI reading from a different bar.

    Just switching to swing pivots was NOT enough on its own -- tested
    against synthetic no-pattern (pure random walk) data, two adjacent
    pivots showing SOME divergence by pure chance turned out to be common
    (~22% of days), actually worse than the original bug's ~11%. Two
    additional filters, both empirically tuned against that same synthetic
    no-signal test until the false-fire rate reached the low single digits
    the audit predicted:
      - `min_rsi_gap`: the RSI reading at the two pivots must differ by a
        real amount, not just be marginally different (noise).
      - `max_pivot_age`: the second pivot must be recent (within this many
        bars of the most recent bar) -- a "divergence" anchored on a stale
        pivot from well in the past isn't a live signal for today."""
    sub = df.tail(lookback_bars) if len(df) > lookback_bars else df
    if len(sub) < (swing_lookback * 2 + 1) * 2:
        return None
    swing_highs, swing_lows = find_swings(sub, lookback=swing_lookback)
    sub_rsi = rsi_series.reindex(sub.index)
    last_pos = len(sub) - 1

    findings = []

    if len(swing_highs) >= 2:
        (i1, p1), (i2, p2) = swing_highs[-2], swing_highs[-1]
        if i2 - i1 >= min_pivot_gap and (last_pos - i2) <= max_pivot_age:
            rsi1, rsi2 = sub_rsi.iloc[i1], sub_rsi.iloc[i2]
            if (not (np.isnan(rsi1) or np.isnan(rsi2)) and p2 > p1
                    and (rsi1 - rsi2) >= min_rsi_gap):
                findings.append("Bearish divergence — price higher high, RSI lower high")

    if len(swing_lows) >= 2:
        (i1, p1), (i2, p2) = swing_lows[-2], swing_lows[-1]
        if i2 - i1 >= min_pivot_gap and (last_pos - i2) <= max_pivot_age:
            rsi1, rsi2 = sub_rsi.iloc[i1], sub_rsi.iloc[i2]
            if (not (np.isnan(rsi1) or np.isnan(rsi2)) and p2 < p1
                    and (rsi2 - rsi1) >= min_rsi_gap):
                findings.append("Bullish divergence — price lower low, RSI higher low")

    return findings if findings else None


# ---------------------------------------------------------------------------
# Report building blocks
# ---------------------------------------------------------------------------

def fmt_pct(value):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "N/A"
    return f"{value:+.2f}%"


def fmt_price(value):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "N/A"
    return f"${value:.2f}"


def build_portfolio_section(portfolio_infos):
    """Aggregate directional bias and volatility stance across every
    watchlist ticker that actually had enough data to analyze, and flag
    when the watchlist isn't offering independent ideas -- several
    correlated names all leaning the same direction is one concentrated
    bet wearing multiple tickers' names, not several separate
    opportunities. This can't see actual portfolio-level correlation
    between the watchlist names themselves (that needs real position/
    sector data this tool doesn't have) -- but it does have each name's
    real correlation to SPY (F32, calc_beta_correlation), which is
    appended to the flag below as a partial, honest substitute: it
    can't prove NVDA and SPCX move together, but it can show whether
    each is itself mostly a market bet."""
    lines = []
    usable = [p for p in portfolio_infos if p is not None]
    if not usable:
        lines.append("No tickers had enough data to include in the portfolio view.")
        return lines

    bullish = [p for p in usable if p["direction"] > 0 and p["sufficient_evidence"]]
    bearish = [p for p in usable if p["direction"] < 0 and p["sufficient_evidence"]]
    no_edge = [p for p in usable if not (p["sufficient_evidence"] and p["direction"] != 0)]

    lines.append(f"Watchlist: {len(usable)} ticker(s) analyzed — "
                 f"{len(bullish)} bullish, {len(bearish)} bearish, {len(no_edge)} no clear edge.")

    def spy_corr_bit(group):
        # Driven by BETA (real market exposure), not correlation -- see
        # calc_beta_correlation's docstring: correlation is suppressed by
        # idiosyncratic vol even when beta (actual market exposure) is
        # high, so using correlation here made the same "independent of
        # the market" mistake this fix corrects in the per-ticker line.
        known = [p for p in group if p.get("spy_beta") is not None]
        if not known:
            return ""
        beta_text = ", ".join(f"{p['ticker']} beta {p['spy_beta']:+.2f}" for p in known)
        avg_abs_beta = np.mean([abs(p["spy_beta"]) for p in known])
        market_bit = (" -- high market exposure across these names, not independent ideas" if avg_abs_beta > 1.5
                       else " -- low market exposure, more idiosyncratic than a pure market bet" if avg_abs_beta < 0.5
                       else "")
        return f" vs-SPY beta: {beta_text}{market_bit}."

    flagged = False
    if len(bullish) >= 2:
        names = ", ".join(p["ticker"] for p in bullish)
        lines.append(f"CONCENTRATION: {names} are ALL bullish today. If these names are "
                      "correlated (same sector, same factor exposure), this is one directional "
                      f"bet sized across several tickers, not several independent ideas.{spy_corr_bit(bullish)}")
        flagged = True
    if len(bearish) >= 2:
        names = ", ".join(p["ticker"] for p in bearish)
        lines.append(f"CONCENTRATION: {names} are ALL bearish today. Same caveat as "
                      f"above.{spy_corr_bit(bearish)}")
        flagged = True

    sell_prem = [p["ticker"] for p in usable if p["vol_stance"] == "sell premium"]
    buy_prem = [p["ticker"] for p in usable if p["vol_stance"] == "buy premium"]
    if len(sell_prem) >= 2:
        lines.append(f"VOLATILITY CONCENTRATION: {', '.join(sell_prem)} all show a volatility "
                      "read favoring selling premium. Short-vega positions across correlated "
                      "names add up to one larger short-vol bet, not independent trades.")
        flagged = True
    if len(buy_prem) >= 2:
        lines.append(f"VOLATILITY CONCENTRATION: {', '.join(buy_prem)} all show a volatility "
                      "read favoring buying premium (long vega across these names).")
        flagged = True

    earnings_soon = [p["ticker"] for p in usable if p["earnings_soon"]]
    if earnings_soon:
        lines.append(f"EARNINGS THIS WEEK: {', '.join(earnings_soon)} — the volatility read for "
                      "these names is unreliable heading into the event (see their Trade Idea sections).")
        flagged = True

    if not flagged:
        lines.append("No concentration flags today — directional and volatility reads are "
                      "reasonably spread out across the watchlist.")

    return lines


def parse_price(text):
    """Inverse of fmt_price -- needed to compute risk/reward from a
    PatternMatch's already-formatted price_target string."""
    if not text or text == "N/A":
        return None
    try:
        return float(str(text).replace("$", "").replace(",", ""))
    except ValueError:
        return None


def pivot_age_sessions(df, formed_date_str):
    """How many trading sessions ago a pattern's anchor pivot was, given
    the full history `df` and the pattern's `formed_date` string. Swing
    pivots need lookback bars confirmed on both sides, so the freshest
    possible anchor is always a few sessions old -- that lag is invisible
    unless surfaced explicitly, and "Confirmed" can otherwise read as if it
    means "as of right now" when the anchor may be a week or more stale."""
    try:
        formed_date = pd.Timestamp(formed_date_str).date()
        matches = df.index[df.index.date == formed_date]
        if len(matches) == 0:
            return None
        pos = df.index.get_loc(matches[0])
        return len(df) - 1 - pos
    except Exception:
        return None


def quality_text(confidence):
    """Ordinal quality tier, plus a calibrated historical hit rate when one
    has actually been fit and is statistically distinguishable from noise
    (see calibrate.py) -- never fabricates a number when calibration isn't
    available or isn't significant yet."""
    tier = chart_patterns.confidence_tier(confidence)
    calibration = calibrate.remap_confidence(confidence, horizon=CALIBRATION_HORIZON)
    if calibration is None:
        return tier
    if not calibration["significant"]:
        return f"{tier} (calibration not yet significant at n={calibration['n']})"
    reliability = "" if calibration["reliable"] else f", provisional n={calibration['n']}"
    return f"{tier} (calibrated ~{calibration['calibrated_pct']:.0f}% historical hit rate{reliability})"


def next_fomc_date(today):
    """Next FOMC rate-decision date on/after `today`, from the hardcoded
    FOMC_DECISION_DATES_2026 list. Returns None once past the last date in
    that list (i.e. this needs a yearly refresh from the source cited
    above it) -- never estimates or guesses a date beyond what's verified."""
    upcoming = [d for d in FOMC_DECISION_DATES_2026 if d >= today]
    return min(upcoming) if upcoming else None


def build_market_context_section(vix_value, vix_timestamp, vix_is_stale, spy_trend_line,
                                  qqq_trend_line, report_date=None):
    lines = []
    if vix_value is not None:
        if vix_value < 15:
            regime = "Low volatility — complacent market"
        elif vix_value < 20:
            regime = "Normal volatility"
        elif vix_value < 30:
            regime = "Elevated volatility — premium selling favored"
        else:
            regime = "Crisis volatility — reduce size, be cautious"
        stale_bit = " [STALE — >15 min old, treat as last known, not live]" if vix_is_stale else ""
        ts_bit = f" (as of {vix_timestamp.strftime('%Y-%m-%d %H:%M %Z')})" if vix_timestamp is not None else ""
        lines.append(f"VIX: {vix_value:.1f} — {regime}{stale_bit}{ts_bit}")
    else:
        lines.append("VIX: N/A — data unavailable")
    lines.append(spy_trend_line)
    lines.append(qqq_trend_line)

    if report_date is not None:
        fomc_date = next_fomc_date(report_date)
        if fomc_date is not None:
            days_to_fomc = (fomc_date - report_date).days
            if 0 <= days_to_fomc <= 7:
                lines.append(f"FOMC WARNING: Fed rate decision on {fomc_date} "
                              f"({days_to_fomc} day{'s' if days_to_fomc != 1 else ''} away) -- a "
                              "market-wide event, not ticker-specific. Expect IV to run up into it "
                              "and crush after; every ticker's volatility read below is unreliable "
                              "until it's past.")
            else:
                lines.append(f"Next FOMC decision: {fomc_date}.")
        else:
            lines.append(f"FOMC calendar needs its annual refresh -- no dates on file on/after "
                          f"{report_date} (see FOMC_DECISION_DATES_2026 and the source URL above it).")
    return lines


def analyze_market_ticker(ticker, df):
    """For SPY/QQQ market context — returns a one-line trend summary."""
    if df is None or len(df) < 20:
        return f"{ticker}: Insufficient data for structure analysis"
    trend, detail, _, _ = classify_structure(df, lookback_bars=len(df), swing_lookback=3)
    return f"{ticker}: {trend} — {detail}"


def collapse_family(label, family_signals, weight_cap, coverage_denominator=None):
    """Collapse every signal in a correlated family into ONE vote:
    direction is the family's net lean, weight reflects how lopsided that
    lean is (scaled to weight_cap), and -- when `coverage_denominator` is
    given -- further scaled down when only a fraction of the family's
    possible indicators actually had data (a lean built on 1 of 5 possible
    trend indicators shouldn't carry the same weight as one built on all
    5, even if the 1 available indicator is one-sided)."""
    if not family_signals:
        return None
    bull = sum(w for _, d, w, _ in family_signals if d > 0)
    bear = sum(w for _, d, w, _ in family_signals if d < 0)
    total = bull + bear
    if total == 0:
        return None
    lean = (bull - bear) / total
    direction = 1 if lean > 0 else -1
    weight = abs(lean) * weight_cap
    if coverage_denominator:
        weight *= min(len(family_signals) / coverage_denominator, 1.0)
    agreeing = sorted([s for s in family_signals if s[1] == direction], key=lambda s: -s[2])
    top_notes = "; ".join(n[3] for n in agreeing[:2])
    note = (f"{top_notes} ({len(agreeing)}/{len(family_signals)} {label.lower()} signals agree)"
            if len(family_signals) > 1 else top_notes)
    return (label, direction, weight, note)


FAMILY_WEIGHT_CAP = 4.0
TREND_FAMILY_SIZE = 6  # Swing structure (daily), Weekly structure, 200 SMA, MACD, ADX/DI, Momentum
# G20 (2nd Opus audit): backtest.py was re-run on identical data (NVDA,
# INTC, SPY, QQQ; 2017-08-04 to 2026-07-07; same 1796 eval dates/ticker)
# comparing TREND_FAMILY_SIZE=5 with the weekly-structure signal disabled
# (pre-F31) against 6 with it (current). Result, stated plainly: adding
# the weekly signal cut the directional-call rate from 33.7% to 21.2%
# (more selective, as intended) and improved the +1-bar hit rate (54.5%
# -> 56.8%), but WORSENED hit rate at +5/+10/+20 bars (51.6/56.5/56.0% ->
# 51.1/54.5/53.9%). In BOTH configurations, the tool's own directional
# calls underperformed simple buy-and-hold and 50/200-SMA-crossover
# baselines on the SAME dates at every horizon beyond +1 bar. This is one
# non-independent-observations backtest (see backtest.py's own caveats on
# overlapping windows), not a rigorous conclusion either way -- but it is
# real evidence, where before there was none, and it does NOT show the
# weekly signal earning its added selectivity through better realized
# accuracy. Kept at 6 because the selectivity itself (fewer, more
# conservative calls) is independently defensible and the accuracy
# swing is small relative to this backtest's own noise -- but this
# should be revisited if a properly independent (non-overlapping,
# walk-forward) backtest becomes feasible.
# Unlike the trend family, "how many patterns could possibly exist" isn't a
# fixed number -- but the pattern family had NO coverage scaling at all,
# which a 2nd audit caught: a single Forming candlestick (weight 1.5, the
# only signal in its family) collapses to lean=1.0 -> weight = 1.0 *
# FAMILY_WEIGHT_CAP = 4.0, i.e. 89% of MIN_CONFLUENCE_WEIGHT from one
# forming pattern alone. 3 is chosen from the noise-day measurement in
# that audit (mean ~1.9 patterns/report on pure noise) -- one pattern
# should contribute a THIRD of the cap, not all of it.
PATTERN_FAMILY_COVERAGE = 3
# With only 2 possible families, a single family's vote alone can reach at
# most FAMILY_WEIGHT_CAP. Setting the floor just above that means a
# directional call structurally REQUIRES both families to be present and
# to agree (partially or fully) -- one family voting alone, however
# lopsided, can never clear this floor by construction.
MIN_CONFLUENCE_WEIGHT = 4.5
# With only 2 families, `net` is close to degenerate: a 2nd audit measured
# 68% of directional calls landing at EXACTLY net=+/-1.00 (both families
# simply agreeing), which happens identically whether total_weight is
# barely over MIN_CONFLUENCE_WEIGHT or twice that -- net's magnitude
# stopped carrying strength information the moment sufficient_evidence
# was true. total_weight is the number that actually varies with how much
# evidence exists (max possible with 2 families at FAMILY_WEIGHT_CAP each
# is 8.0), so the "confluence" vs. "lean" strength label -- and the
# reference number shown alongside it -- is now driven by total_weight,
# not net. Measured on pure noise: this drops the "confluence" label's
# share of (already rare, post-G5) directional calls from 78% to 25%.
STRONG_CONFLUENCE_WEIGHT = 6.0


def compute_confluence(df, pattern_matches):
    """Self-contained confluence computation: given a daily OHLCV history
    and the chart patterns already detected for it (chart_patterns.detect_all),
    returns the same signals/net/weights the live report's Plain English
    Summary is built from. Deliberately self-contained -- it recomputes
    trend/SMA/MACD/ADX/momentum internally, redundant with what
    analyze_ticker's other sections already compute for DISPLAY, but these
    are cheap pure functions of df -- so this exact logic can be called
    directly against a historical slice by a backtest, rather than the
    backtest needing to hand-reimplement it and risk silently drifting out
    of sync with whatever the live report actually does.

    Returns a dict: signals, net, bull_weight, bear_weight, total_weight,
    sufficient_evidence, trend, has_min_data."""
    has_min_data = len(df) >= 30
    current_price = float(df["Close"].iloc[-1])

    trend = "RANGE"
    if has_min_data:
        trend, _, _, _ = classify_structure(df, lookback_bars=min(60, len(df)), swing_lookback=3)

    # A genuine higher-timeframe read (F31) -- the "trend" above is 60
    # DAILY bars, still a daily-timeframe read, just a shorter window of it.
    weekly_trend = None
    weekly_df = resample_weekly(df)
    if len(weekly_df) >= 15:
        weekly_trend, _, _, _ = classify_structure(
            weekly_df, lookback_bars=min(60, len(weekly_df)), swing_lookback=2)

    sma200 = sma(df["Close"], 200)

    macd_status = None
    if len(df) >= 35:
        macd_line, signal_line, _ = calc_macd(df["Close"])
        macd_status = "Bullish" if macd_line.iloc[-1] > signal_line.iloc[-1] else "Bearish"

    adx_val = plus_di_val = minus_di_val = None
    if len(df) >= 30:
        adx_series, plus_di_series, minus_di_series = calc_adx(df, 14)
        if not np.isnan(adx_series.iloc[-1]):
            adx_val = float(adx_series.iloc[-1])
            plus_di_val = float(plus_di_series.iloc[-1])
            minus_di_val = float(minus_di_series.iloc[-1])

    mom_dir_positive = mom_dir_negative = mom_dir_rising = mom_dir_falling = False
    if len(df) >= 25:
        squeeze_data = calc_ttm_squeeze(df)
        ms = squeeze_data["momentum"]
        if not ms.dropna().empty:
            mc = ms.iloc[-1]
            mp = ms.iloc[-2] if len(ms) >= 2 else np.nan
            mom_current = float(mc) if not np.isnan(mc) else None
            mom_prev = float(mp) if not np.isnan(mp) else None
            mom_dir_positive = mom_current is not None and mom_current > 0
            mom_dir_negative = mom_current is not None and mom_current < 0
            mom_dir_rising = mom_current is not None and mom_prev is not None and mom_current > mom_prev
            mom_dir_falling = mom_current is not None and mom_prev is not None and mom_current < mom_prev

    trend_indicator_signals = []
    if has_min_data:
        if trend == "UPTREND":
            trend_indicator_signals.append(("Swing structure (daily)", 1, 3.0, "the daily swing structure is an uptrend (HH+HL)"))
        elif trend == "DOWNTREND":
            trend_indicator_signals.append(("Swing structure (daily)", -1, 3.0, "the daily swing structure is a downtrend (LH+LL)"))

    if weekly_trend == "UPTREND":
        trend_indicator_signals.append(("Weekly structure", 1, 3.0, "the WEEKLY structure is an uptrend (HH+HL) -- a real higher timeframe, not just fewer daily bars"))
    elif weekly_trend == "DOWNTREND":
        trend_indicator_signals.append(("Weekly structure", -1, 3.0, "the WEEKLY structure is a downtrend (LH+LL) -- a real higher timeframe, not just fewer daily bars"))

    if len(df) >= 200 and not np.isnan(sma200.iloc[-1]):
        above_200 = current_price > sma200.iloc[-1]
        trend_indicator_signals.append(("200 SMA", 1 if above_200 else -1, 2.0,
                         f"price is {'above' if above_200 else 'below'} the 200-day SMA"))

    if macd_status is not None:
        trend_indicator_signals.append(("MACD", 1 if macd_status == "Bullish" else -1, 2.0,
                         f"MACD is {macd_status.lower()}"))

    if adx_val is not None and plus_di_val is not None:
        di_dir = 1 if plus_di_val > minus_di_val else -1
        di_weight = 2.0 if adx_val > 25 else 1.0
        di_note = (f"ADX {adx_val:.0f} confirms a trending tape with "
                   f"{'+DI' if di_dir > 0 else '-DI'} in control") if adx_val > 25 else \
                  (f"ADX {adx_val:.0f} (ranging) with "
                   f"{'+DI' if di_dir > 0 else '-DI'} slightly ahead")
        trend_indicator_signals.append(("ADX/DI", di_dir, di_weight, di_note))

    if mom_dir_positive:
        mom_weight = 2.0 if mom_dir_rising else 1.0
        mom_note = "momentum is positive and rising" if mom_dir_rising else "momentum is positive but fading"
        trend_indicator_signals.append(("Momentum", 1, mom_weight, mom_note))
    elif mom_dir_negative:
        mom_weight = 2.0 if mom_dir_falling else 1.0
        mom_note = "momentum is negative and falling" if mom_dir_falling else "momentum is negative but improving"
        trend_indicator_signals.append(("Momentum", -1, mom_weight, mom_note))

    pattern_signals = []
    for pm in pattern_matches:
        if pm.bias == "Neutral":
            continue
        base_dir = 1 if pm.bias == "Bullish" else -1
        if pm.status == "Invalidated":
            direction, weight = -base_dir, 2.5
            note = f"{pm.name} failed against its {pm.bias.lower()} bias — often a reversal tell"
        elif pm.status == "Confirmed":
            direction, weight = base_dir, 3.0
            note = f"{pm.name} is confirmed"
        else:
            direction, weight = base_dir, 1.5
            note = f"{pm.name} is forming (not yet confirmed)"
        pattern_signals.append((pm.name, direction, weight, note))

    trend_vote = collapse_family("Trend/Momentum", trend_indicator_signals,
                                  FAMILY_WEIGHT_CAP, coverage_denominator=TREND_FAMILY_SIZE)
    pattern_vote = collapse_family("Pattern geometry", pattern_signals, FAMILY_WEIGHT_CAP,
                                    coverage_denominator=PATTERN_FAMILY_COVERAGE)
    signals = [v for v in (trend_vote, pattern_vote) if v is not None]

    bull_weight = sum(w for _, d, w, _ in signals if d > 0)
    bear_weight = sum(w for _, d, w, _ in signals if d < 0)
    total_weight = bull_weight + bear_weight
    net = (bull_weight - bear_weight) / total_weight if total_weight > 0 else 0.0
    sufficient_evidence = total_weight >= MIN_CONFLUENCE_WEIGHT and len(signals) >= 2

    return {
        "signals": signals, "net": net, "bull_weight": bull_weight, "bear_weight": bear_weight,
        "total_weight": total_weight, "sufficient_evidence": sufficient_evidence,
        "trend": trend, "has_min_data": has_min_data,
    }


def analyze_ticker(ticker, df, mode, report_date, premarket_data=None, spy_df=None):
    """Build the full multi-section report text for a single watchlist
    ticker. Returns (report_text, portfolio_info) -- portfolio_info is a
    dict of the facts the PORTFOLIO VIEW section (built once, after every
    ticker has been analyzed) needs to aggregate across the watchlist, or
    None if this ticker couldn't be analyzed at all."""
    lines = []
    lines.append("=" * 60)
    lines.append(f"{ticker}")
    lines.append("=" * 60)

    if df is None or df.empty:
        lines.append("WARNING: No data available for this ticker — skipping.")
        return "\n".join(lines), None

    min_days_required = 30
    has_min_data = len(df) >= min_days_required

    # Generalized from a ticker == "SPCX" hardcode (SPCX was a recent IPO
    # early in this project with under 60 days of history at the time) --
    # any young ticker in WATCHLIST should get the same heads-up, not just
    # whichever one prompted the original note.
    if len(df) < 60:
        lines.append(f"NOTE: {ticker} has only {len(df)} trading days of history — "
                      f"sections requiring more history will be skipped/limited.")

    if not has_min_data:
        lines.append(f"WARNING: {ticker} has fewer than {min_days_required} trading days "
                      f"({len(df)} available) — skipping indicators that require more data.")

    current_price = float(df["Close"].iloc[-1])
    lines.append(f"Current price: {fmt_price(current_price)}")

    if premarket_data:
        lines.append(f"Pre-market: {fmt_price(premarket_data['price'])} "
                      f"(as of {premarket_data['timestamp']})")
    elif mode == "morning":
        lines.append("Pre-market: Not available")

    # ---------------- SECTION 2: Market Structure ----------------
    lines.append("")
    lines.append("--- MARKET STRUCTURE ---")
    if has_min_data:
        trend, detail, _, _ = classify_structure(
            df, lookback_bars=min(60, len(df)), swing_lookback=3)
        lines.append(f"Daily swing structure (60 bars): {trend} — {detail}")

        weekly_df = resample_weekly(df)
        if len(weekly_df) >= 15:
            weekly_trend_disp, weekly_detail, _, _ = classify_structure(
                weekly_df, lookback_bars=min(60, len(weekly_df)), swing_lookback=2)
            weekly_partial_bit = (" (most recent weekly bar is still forming)"
                                   if weekly_df.index[-1].date() > df.index[-1].date() else "")
            lines.append(f"Weekly Trend (real higher timeframe): {weekly_trend_disp} — "
                          f"{weekly_detail}{weekly_partial_bit}")
        else:
            lines.append("Weekly Trend: Insufficient data (need 15+ weekly bars)")

        if trend == "UPTREND":
            bias = "Long bias"
        elif trend == "DOWNTREND":
            bias = "Short bias"
        else:
            bias = "Neutral"
        lines.append(f"Market bias: {bias}")
    else:
        trend, bias = "RANGE", "Neutral"
        lines.append("Daily swing structure: Insufficient data")
        lines.append("Weekly Trend: Insufficient data")
        lines.append("Market bias: Neutral")

    beta_ctx = calc_beta_correlation(df, spy_df)
    if beta_ctx is not None:
        beta, corr, se = beta_ctx["beta"], beta_ctx["correlation"], beta_ctx["beta_stderr"]
        ci_lo, ci_hi = beta - 1.96 * se, beta + 1.96 * se
        abs_beta, abs_corr = abs(beta), abs(corr)
        # Market-exposure language is driven by BETA, not correlation --
        # a 2nd audit caught the previous correlation-only framing getting
        # this backwards for high-idiosyncratic-vol names: high idiosyncratic
        # vol mechanically suppresses correlation while leaving actual market
        # exposure (beta) untouched, so a name with real leverage to the
        # market but a noisy day-to-day relationship was being called
        # "independent of the market". Correlation now only describes how
        # much to trust the beta NUMBER, not whether market exposure exists.
        if abs_beta > 1.5:
            exposure = ("high market exposure -- today's setup may be more a leveraged "
                        "market bet than a name-specific edge")
        elif abs_beta > 0.5:
            exposure = "roughly market-level exposure"
        else:
            exposure = "low market exposure -- moves here look more idiosyncratic than market-driven"
        if abs_corr > 0.6:
            reliability = "a reasonably reliable beta read"
        elif abs_corr > 0.3:
            reliability = "a noisy beta read -- don't over-read the exact number"
        else:
            reliability = ("a very noisy beta read -- low correlation to SPY means this could be "
                            "mostly idiosyncratic-vol noise rather than a precise market-exposure estimate")
        lines.append(f"vs SPY ({beta_ctx['n']}d): beta {beta:+.2f} (95% CI {ci_lo:+.2f} to {ci_hi:+.2f}), "
                      f"correlation {corr:+.2f} — {ticker} shows {exposure}; correlation suggests {reliability}.")
        if "recent_beta" in beta_ctx:
            lines.append(f"  Recent ({beta_ctx['recent_n']}d, faster-moving) beta: {beta_ctx['recent_beta']:+.2f}, "
                          f"correlation {beta_ctx['recent_correlation']:+.2f} -- a large gap from the figure "
                          "above would suggest a recent shift in how this name trades vs. the market.")
    else:
        lines.append("vs SPY: Insufficient overlapping history (need 60+ days) to compute a beta/correlation read.")

    # ---------------- SECTION 3: Key Technical Zones ----------------
    lines.append("")
    lines.append("--- KEY TECHNICAL ZONES ---")

    prev_day = df.iloc[-2] if len(df) >= 2 else None
    if prev_day is not None:
        pd_high, pd_low, pd_close = float(prev_day["High"]), float(prev_day["Low"]), float(prev_day["Close"])
        lines.append(f"Prior day high: {fmt_price(pd_high)} ({fmt_pct((current_price - pd_high) / pd_high * 100)})")
        lines.append(f"Prior day low: {fmt_price(pd_low)} ({fmt_pct((current_price - pd_low) / pd_low * 100)})")
        lines.append(f"Prior day close: {fmt_price(pd_close)} ({fmt_pct((current_price - pd_close) / pd_close * 100)})")
    else:
        pd_high = pd_low = pd_close = None
        lines.append("Prior day levels: Insufficient data")

    if len(df) >= 5:
        last5 = df.tail(5)
        pw_high, pw_low = float(last5["High"].max()), float(last5["Low"].min())
        lines.append(f"Prior week high: {fmt_price(pw_high)} ({fmt_pct((current_price - pw_high) / pw_high * 100)})")
        lines.append(f"Prior week low: {fmt_price(pw_low)} ({fmt_pct((current_price - pw_low) / pw_low * 100)})")
    else:
        pw_high = pw_low = None
        lines.append("Prior week high/low: Insufficient data")

    # Moving averages
    sma200 = sma(df["Close"], 200)
    sma50 = sma(df["Close"], 50)
    ema21 = ema(df["Close"], 21)
    ema9 = ema(df["Close"], 9)

    if len(df) >= 200 and not np.isnan(sma200.iloc[-1]):
        val = float(sma200.iloc[-1])
        rel = "above" if current_price > val else "below"
        lines.append(f"200-day SMA: {fmt_price(val)} — price {rel} ({fmt_pct((current_price - val) / val * 100)})")
    else:
        lines.append("200-day SMA: Insufficient data")

    if len(df) >= 50 and not np.isnan(sma50.iloc[-1]):
        val = float(sma50.iloc[-1])
        rel = "above" if current_price > val else "below"
        lines.append(f"50-day SMA: {fmt_price(val)} — price {rel} ({fmt_pct((current_price - val) / val * 100)})")
    else:
        lines.append("50-day SMA: Insufficient data")

    if len(df) >= 21 and not np.isnan(ema21.iloc[-1]):
        val = float(ema21.iloc[-1])
        rel = "above" if current_price > val else "below"
        lines.append(f"21-day EMA: {fmt_price(val)} — price {rel}")
    else:
        lines.append("21-day EMA: Insufficient data")

    if len(df) >= 9 and not np.isnan(ema9.iloc[-1]):
        val9 = float(ema9.iloc[-1])
        rel = "above" if current_price > val9 else "below"
        lines.append(f"9-day EMA: {fmt_price(val9)} — price {rel}")
    else:
        val9 = None
        lines.append("9-day EMA: Insufficient data")

    if len(df) >= 21 and not np.isnan(ema21.iloc[-1]) and not np.isnan(ema9.iloc[-1]):
        alignment = "Bullish (9 EMA above 21 EMA)" if ema9.iloc[-1] > ema21.iloc[-1] else "Bearish (9 EMA below 21 EMA)"
        lines.append(f"EMA alignment: {alignment}")

    # Golden / death cross within last 20 days
    if len(df) >= 200:
        diff = sma50 - sma200
        recent_diff = diff.tail(21)
        golden_cross = False
        death_cross = False
        for i in range(1, len(recent_diff)):
            prev_val = recent_diff.iloc[i - 1]
            cur_val = recent_diff.iloc[i]
            if np.isnan(prev_val) or np.isnan(cur_val):
                continue
            if prev_val <= 0 and cur_val > 0:
                golden_cross = True
            if prev_val >= 0 and cur_val < 0:
                death_cross = True
        if golden_cross:
            lines.append("Golden cross: YES — 50 SMA crossed above 200 SMA within last 20 days")
        if death_cross:
            lines.append("Death cross: YES — 50 SMA crossed below 200 SMA within last 20 days")
        if not golden_cross and not death_cross:
            lines.append("Golden/Death cross: None in last 20 days")

    # Real intraday bars (F33), fetched once and reused for both the
    # session VWAP and a finer volume-at-price profile below -- None if
    # unavailable, in which case both fall back to the daily-bar-only
    # approximations that already existed.
    intraday_df = fetch_intraday_bars(ticker)

    # Rolling VWAP
    if len(df) >= 5:
        vwap5 = calc_rolling_vwap(df, window=5)
        lines.append(f"5-day VWAP (daily-bar approx): {fmt_price(vwap5)} — price "
                      f"{'above' if current_price > vwap5 else 'below'} "
                      f"({fmt_pct((current_price - vwap5) / vwap5 * 100)})")
    else:
        lines.append("5-day VWAP: Insufficient data")

    session_vwap, session_vwap_date, session_vwap_partial = calc_session_vwap(intraday_df)
    if session_vwap is not None:
        partial_bit = " [session in progress -- will keep moving until close]" if session_vwap_partial else ""
        lines.append(f"Session VWAP ({session_vwap_date}, real intraday bars){partial_bit}: {fmt_price(session_vwap)} "
                      f"— price {'above' if current_price > session_vwap else 'below'} "
                      f"({fmt_pct((current_price - session_vwap) / session_vwap * 100)})")

    # Volume POC
    if len(df) >= 20:
        poc_atr = calc_atr(df, 14).iloc[-1]
        poc_atr = float(poc_atr) if not np.isnan(poc_atr) else None
        poc = calc_volume_poc(df, window=20, atr_val=poc_atr)
        lines.append(f"Volume POC (20-day, daily-bar approx): {fmt_price(poc)} — highest volume concentration")
    else:
        poc = None
        poc_atr = None
        lines.append("Volume POC: Insufficient data")

    if intraday_df is not None and len(intraday_df) >= 30:
        intraday_bins = calc_intraday_poc_bins(intraday_df)
        intraday_poc = calc_volume_poc(intraday_df, window=len(intraday_df), bins=intraday_bins)
        n_sessions = len(set(intraday_df.index.date))
        lines.append(f"Volume POC ({n_sessions}-session, real intraday bars): {fmt_price(intraday_poc)} "
                      "— highest volume concentration, real volume-at-price rather than a daily-bar approximation")

    # 52-week levels
    if len(df) >= 30:
        window = min(len(df), 252)
        sub52 = df.tail(window)
        high52, low52 = float(sub52["High"].max()), float(sub52["Low"].min())
        lines.append(f"52-week high: {fmt_price(high52)} ({fmt_pct((current_price - high52) / high52 * 100)})")
        lines.append(f"52-week low: {fmt_price(low52)} ({fmt_pct((current_price - low52) / low52 * 100)})")
    else:
        high52 = low52 = None
        lines.append("52-week high/low: Insufficient data")

    # Expected weekly move
    if len(df) >= 30:
        hv30 = calc_hv(df["Close"], window=30)
        if hv30 is not None:
            days_to_friday = days_until_next_friday(report_date)
            expected_move = current_price * (hv30 / np.sqrt(252)) * np.sqrt(days_to_friday)
            pct_move = expected_move / current_price * 100
            lines.append(f"Expected weekly move: +/- {fmt_price(expected_move)} ({pct_move:.2f}%)")
        else:
            lines.append("Expected weekly move: Insufficient data")
    else:
        hv30 = None
        lines.append("Expected weekly move: Insufficient data")

    # ---------------- SECTION 4: Price Action Analysis ----------------
    lines.append("")
    lines.append("--- PRICE ACTION ---")
    recent10 = df.tail(10)

    inside_count = detect_inside_bars(recent10)
    if inside_count > 0:
        lines.append(f"Inside bar(s): {inside_count} consecutive — compression signal")

    wick_result = wick_analysis(df, n=3)
    if wick_result:
        wick_label, _, _ = wick_result
        lines.append(f"Wick analysis: {wick_label}")

    # ---------------- SECTION 4b: Chart Patterns ----------------
    lines.append("")
    lines.append("--- CHART PATTERNS ---")
    pattern_matches = chart_patterns.detect_all(df)
    if pattern_matches:
        lines.append("(Quality is a geometric heuristic -- fit/pivots/volume -- not a "
                      "calibrated probability. Measured against synthetic no-pattern data, "
                      "it shows no relationship to actual outcomes. Treat as a filter for "
                      "which patterns are worth a manual look, not a win-rate estimate.)")
        for pm in pattern_matches:
            age = pivot_age_sessions(df, pm.formed_date)
            age_bit = f" ({age} session{'s' if age != 1 else ''} ago)" if age is not None else ""
            lines.append(f"[{pm.bias}] {pm.name} ({pm.category}) — "
                          f"Quality: {quality_text(pm.confidence)} — Formed: {pm.formed_date}{age_bit} — "
                          f"Status: {pm.status}")
            target_bit = f" | Target: {pm.price_target}" if pm.price_target != "N/A" else ""
            lines.append(f"    {pm.detail}{target_bit}")
    else:
        lines.append("No chart patterns detected.")

    key_zones = {}
    if len(df) >= 200 and not np.isnan(sma200.iloc[-1]):
        key_zones["200 SMA"] = float(sma200.iloc[-1])
    if pw_high is not None:
        key_zones["Prior week high"] = pw_high
    if pw_low is not None:
        key_zones["Prior week low"] = pw_low
    if poc is not None:
        key_zones["Volume POC"] = poc

    if key_zones:
        nearest_label, nearest_price = min(
            key_zones.items(), key=lambda kv: abs(current_price - kv[1]))
        result = acceptance_or_rejection(df, nearest_price, nearest_label)
        if result:
            lines.append(result)

        bnr = detect_break_and_retest(df, key_zones)
        if bnr:
            for r in bnr:
                lines.append(r)

    # ---------------- SECTION 5: Momentum & Entry Triggers ----------------
    lines.append("")
    lines.append("--- MOMENTUM & ENTRY TRIGGERS ---")

    momentum_positive = False
    momentum_rising = False
    squeeze_data = None

    if len(df) >= 25:
        squeeze_data = calc_ttm_squeeze(df)
        squeeze_on_series = squeeze_data["squeeze_on"]
        consec_series = squeeze_data["consec"]
        momentum_series = squeeze_data["momentum"]

        current_squeeze_on = bool(squeeze_on_series.iloc[-1]) if not pd.isna(squeeze_on_series.iloc[-1]) else False
        prev_squeeze_on = bool(squeeze_on_series.iloc[-2]) if len(squeeze_on_series) >= 2 and not pd.isna(squeeze_on_series.iloc[-2]) else False
        prev_consec = int(consec_series.iloc[-2]) if len(consec_series) >= 2 else 0
        current_consec = int(consec_series.iloc[-1])

        fired_today = (not current_squeeze_on) and prev_squeeze_on and prev_consec >= 6

        if current_squeeze_on:
            lines.append(f"SQUEEZE: COMPRESSED ({current_consec} bars) — volatility coiling")
        elif fired_today:
            lines.append("SQUEEZE: FIRED today — breakout alert")
        else:
            bars_since_fire = 0
            for i in range(len(squeeze_on_series) - 1, 0, -1):
                cur = squeeze_on_series.iloc[i]
                prv = squeeze_on_series.iloc[i - 1]
                if pd.isna(cur) or pd.isna(prv):
                    break
                if (not cur) and prv:
                    break
                bars_since_fire += 1
            lines.append(f"SQUEEZE: OFF ({bars_since_fire} bars since fire)")

        if not momentum_series.dropna().empty:
            mom_current = momentum_series.iloc[-1]
            mom_prev = momentum_series.iloc[-2] if len(momentum_series) >= 2 else np.nan
            if not np.isnan(mom_current):
                momentum_positive = mom_current > 0
                if not np.isnan(mom_prev):
                    momentum_rising = mom_current > mom_prev
                    if momentum_positive and momentum_rising:
                        lines.append("MOMENTUM: Positive and rising — bullish pressure building")
                    elif momentum_positive and not momentum_rising:
                        lines.append("MOMENTUM: Positive but falling — momentum fading")
                    elif not momentum_positive and not momentum_rising:
                        lines.append("MOMENTUM: Negative and falling — bearish pressure")
                    else:
                        lines.append("MOMENTUM: Negative but rising — potential bottom")
                else:
                    lines.append("MOMENTUM: Insufficient data for trend comparison")
            else:
                lines.append("MOMENTUM: Insufficient data")
        else:
            lines.append("MOMENTUM: Insufficient data")
    else:
        lines.append("SQUEEZE: Insufficient data")
        lines.append("MOMENTUM: Insufficient data")

    # MACD
    macd_cross_recent = False
    hist_expanding = False
    if len(df) >= 35:
        macd_line, signal_line, hist = calc_macd(df["Close"])
        last3_macd = macd_line.tail(3)
        last3_signal = signal_line.tail(3)
        for i in range(1, len(last3_macd)):
            prev_diff = last3_macd.iloc[i - 1] - last3_signal.iloc[i - 1]
            cur_diff = last3_macd.iloc[i] - last3_signal.iloc[i]
            if prev_diff <= 0 and cur_diff > 0:
                macd_cross_recent = True
        hist_expanding = hist.iloc[-1] > hist.iloc[-2] if len(hist) >= 2 else False
        macd_status = "Bullish" if macd_line.iloc[-1] > signal_line.iloc[-1] else "Bearish"
        lines.append(f"MACD: {macd_status} "
                      f"(line {macd_line.iloc[-1]:.3f} vs signal {signal_line.iloc[-1]:.3f}) — "
                      f"{'bullish cross in last 3 bars' if macd_cross_recent else 'no recent cross'}, "
                      f"histogram {'expanding' if hist_expanding else 'contracting'}")
    else:
        lines.append("MACD: Insufficient data")

    # ADX
    adx_val = None
    plus_di_val = None
    minus_di_val = None
    if len(df) >= 30:
        adx_series, plus_di_series, minus_di_series = calc_adx(df, 14)
        if not np.isnan(adx_series.iloc[-1]):
            adx_val = float(adx_series.iloc[-1])
            plus_di_val = float(plus_di_series.iloc[-1])
            minus_di_val = float(minus_di_series.iloc[-1])
            adx_prev = adx_series.iloc[-2] if len(adx_series) >= 2 and not np.isnan(adx_series.iloc[-2]) else None
            trending = "Trending" if adx_val > 25 else "Ranging"
            rising_str = ""
            if adx_prev is not None:
                rising_str = "rising" if adx_val > adx_prev else "falling"
            di_dir = "+DI above -DI (bullish)" if plus_di_val > minus_di_val else "-DI above +DI (bearish)"
            lines.append(f"ADX: {adx_val:.1f} — {trending}"
                         f"{', ' + rising_str if rising_str else ''}, {di_dir}")
        else:
            lines.append("ADX: Insufficient data")
    else:
        lines.append("ADX: Insufficient data")

    # RSI
    rsi_val = None
    divergence = None
    if len(df) >= 20:
        rsi_series = calc_rsi(df["Close"], 14)
        if not np.isnan(rsi_series.iloc[-1]):
            rsi_val = float(rsi_series.iloc[-1])
            flag = ""
            if rsi_val > 70:
                flag = " — OVERBOUGHT"
            elif rsi_val < 30:
                flag = " — OVERSOLD"
            lines.append(f"RSI(14): {rsi_val:.1f}{flag}")
            divergence = rsi_divergence(df, rsi_series)
            if divergence:
                for d in divergence:
                    lines.append(f"DIVERGENCE: {d}")
        else:
            lines.append("RSI(14): Insufficient data")
    else:
        lines.append("RSI(14): Insufficient data")

    # ---------------- SECTION 6: Volatility Environment ----------------
    lines.append("")
    lines.append("--- VOLATILITY ENVIRONMENT ---")

    atr_val = None
    if len(df) >= 20:
        atr_series = calc_atr(df, 14)
        if not np.isnan(atr_series.iloc[-1]):
            atr_val = float(atr_series.iloc[-1])
            atr_pct = atr_val / current_price * 100
            lines.append(f"ATR(14): {fmt_price(atr_val)} ({atr_pct:.2f}% of price)")
            lines.append(f"Suggested stop for swing trade: {fmt_price(atr_val)} below entry (1 ATR)")
        else:
            lines.append("ATR(14): Insufficient data")
    else:
        lines.append("ATR(14): Insufficient data")

    iv_rank = None
    hv30_val = None
    real_iv_context = None
    if len(df) >= 30:
        hv30_val = calc_hv(df["Close"], window=30)
        if hv30_val is not None:
            lines.append(f"HV30: {hv30_val * 100:.1f}%")

            real_iv_context = fetch_real_iv_context(ticker, hv30_val, report_date)
            if real_iv_context is not None:
                iv = real_iv_context["iv"]
                spread = real_iv_context["iv_minus_hv"]
                ratio = real_iv_context["iv_ratio"]
                ratio_bit = f", IV/HV ratio {ratio:.2f}x" if ratio is not None else ""
                spread_bit = f" ({spread * 100:+.1f} pts)" if spread is not None else ""
                stale_bit = (" [snapshot from a prior day]"
                              if real_iv_context["snapshot_date"] != report_date else "")
                lines.append(f"Real ATM IV ({real_iv_context['expiration']} exp, "
                              f"{real_iv_context['dte']}d): {iv * 100:.1f}%{ratio_bit}{spread_bit}{stale_bit}")

                # G27: log today's real IV so a genuine IV-rank percentile
                # can be built over time (see iv_history.py) -- only once
                # enough daily snapshots exist does real_iv_rank return a
                # number rather than None, so this is silent for now and
                # becomes useful as the log accumulates.
                iv_history.log_iv(ticker, report_date, iv, real_iv_context["expiration"],
                                   real_iv_context["dte"], real_iv_context["snapshot_date"])
                real_rank, real_rank_n = iv_history.real_iv_rank(ticker, iv)
                if real_rank is not None:
                    lines.append(f"Real IV Rank ({real_rank_n} logged snapshots"
                                  f"{', still short of a full year' if real_rank_n < iv_history.FULL_LOOKBACK else ''}"
                                  f"): {real_rank:.0f}%")

            if len(df) >= 60:
                iv_rank = calc_iv_rank(df["Close"], window=30, lookback=252)
                if iv_rank is not None:
                    # This is a REALIZED-volatility percentile, not a true options-market
                    # IV rank (which needs actual implied vol data this tool doesn't have).
                    # It can diverge sharply from real IV, especially around earnings
                    # (IV rises pre-earnings while realized vol stays low -- this reads
                    # "low" exactly when real premium is richest) or right after a shock
                    # (realized vol stays elevated for weeks after IV has already
                    # mean-reverted). Labeled honestly rather than as an options signal.
                    if iv_rank < 30:
                        iv_label = "Low"
                    elif iv_rank < 50:
                        iv_label = "Moderate"
                    elif iv_rank < 70:
                        iv_label = "Elevated"
                    else:
                        iv_label = "Very high"
                    lines.append(f"HV percentile (realized vol, NOT implied — see Trade Idea "
                                  f"caveat): {iv_rank:.1f}% ({iv_label})")
                else:
                    lines.append("IV Rank: Insufficient data")
            else:
                lines.append("IV Rank: Insufficient data (need 60+ days)")
        else:
            lines.append("HV30: Insufficient data")
    else:
        lines.append("HV30 / IV Rank: Insufficient data")

    # ---------------- SECTION 7: Plain English Summary ----------------
    # This section doesn't just restate each indicator -- it weighs every
    # directional signal (trend structure, MACD, ADX/DI, momentum, price vs
    # 200 SMA, and now the detected chart patterns) into one net read, then
    # explains the conclusion using only the signals that actually drove it.
    lines.append("")
    lines.append("--- PLAIN ENGLISH SUMMARY ---")
    mode_label = mode.capitalize()
    date_str = report_date.strftime("%Y-%m-%d")
    lines.append(f"{ticker} — {date_str} {mode_label} Report:")

    nearest_support = None
    nearest_resistance = None
    candidates_below = [v for v in [pd_low, pw_low, low52] if v is not None and v < current_price]
    candidates_above = [v for v in [pd_high, pw_high, high52] if v is not None and v > current_price]
    if candidates_below:
        nearest_support = max(candidates_below)
    if candidates_above:
        nearest_resistance = min(candidates_above)

    # Confluence numbers come from compute_confluence (see its docstring) --
    # a self-contained function shared with the historical backtest, so
    # both are guaranteed to run the exact same logic rather than the
    # backtest hand-reimplementing it and risking silent drift.
    confluence = compute_confluence(df, pattern_matches)
    signals = confluence["signals"]
    net = confluence["net"]
    bull_weight = confluence["bull_weight"]
    bear_weight = confluence["bear_weight"]
    total_weight = confluence["total_weight"]
    sufficient_evidence = confluence["sufficient_evidence"]

    winning_dir = 1 if bull_weight >= bear_weight else -1
    supporting = sorted([s for s in signals if s[1] == winning_dir], key=lambda s: -s[2])[:2]
    opposing = sorted([s for s in signals if s[1] == -winning_dir], key=lambda s: -s[2])[:1]
    support_phrase = "; ".join(note for _, _, _, note in supporting)
    # The caution clause used to be suppressed above |net|=0.85 -- exactly
    # backwards, since a maxed-out net is what happens when opposing
    # evidence gets excluded rather than when it's genuinely absent. Any
    # opposing signal that exists is now always surfaced.
    oppose_bit = f" Caution: {opposing[0][3]}." if opposing else ""

    if total_weight == 0:
        net_label = "No clear directional edge"
        bottom_line = "Bottom line: No clear directional edge — no usable signals for this ticker today."
    elif not sufficient_evidence:
        net_label = "Insufficient evidence"
        bottom_line = (f"Bottom line: Insufficient evidence for a directional call — only "
                        f"{support_phrase or 'thin signal coverage'} (total weight {total_weight:.1f}, "
                        f"need {MIN_CONFLUENCE_WEIGHT:.1f}+ from both the trend/momentum and pattern "
                        "families combined). Treat as no edge.")
    elif abs(net) < 0.15:
        net_label = "Conflicting signals"
        bottom_line = f"Bottom line: Conflicting signals — {support_phrase}.{oppose_bit} Wait for clarity."
    else:
        strength = "confluence" if total_weight >= STRONG_CONFLUENCE_WEIGHT else "lean"
        net_label = f"{'Bullish' if winning_dir > 0 else 'Bearish'} {strength}"
        bottom_line = (f"Bottom line: {net_label} (total weight {total_weight:.1f}) — "
                        f"{support_phrase}.{oppose_bit}")
    lines.append(bottom_line)

    if has_min_data and trend == "RANGE":
        lines.append("Structure: range-bound — no clean higher-timeframe trend to lean on, "
                      "so treat the signals above as tactical, not structural.")

    levels_sent = "Key levels: "
    if nearest_support is not None and nearest_resistance is not None:
        levels_sent += f"support near {fmt_price(nearest_support)}, resistance near {fmt_price(nearest_resistance)}."
    elif nearest_support is not None:
        levels_sent += f"support near {fmt_price(nearest_support)}; no clear resistance identified."
    elif nearest_resistance is not None:
        levels_sent += f"resistance near {fmt_price(nearest_resistance)}; no clear support identified."
    else:
        levels_sent += "insufficient data to identify key levels."
    lines.append(levels_sent)

    # Invalidated patterns are excluded here -- they're already counted as
    # contrarian evidence in the confluence above; presenting one as a live
    # setup with a forward-looking target would contradict that reasoning.
    live_pattern = next((pm for pm in pattern_matches if pm.status != "Invalidated"), None)
    if live_pattern:
        pattern_bit = (f"Pattern in play: {live_pattern.name} "
                        f"({live_pattern.status.lower()}, {quality_text(live_pattern.confidence)} quality)")
        if live_pattern.price_target != "N/A":
            pattern_bit += f", projecting toward {live_pattern.price_target} if it plays out"
        lines.append(pattern_bit + ".")

    if divergence:
        direction_word = "bullish" if any("Bullish" in d for d in divergence) else "bearish"
        lines.append(f"Caution: {direction_word} RSI divergence — momentum is quietly disagreeing with price.")

    vol_bits = []
    if iv_rank is not None:
        vol_bits.append(f"IV rank is {iv_rank:.0f}%")
    if atr_val is not None:
        vol_bits.append(f"ATR is {fmt_price(atr_val)}")
    if vol_bits:
        lines.append("Volatility: " + ", ".join(vol_bits) + ".")

    watch_sent = "Watch for: "
    if live_pattern and live_pattern.status == "Forming" and live_pattern.category != "Candlestick":
        watch_sent += f"confirmation of the {live_pattern.name} — a decisive close beyond its boundary."
    elif live_pattern and live_pattern.status == "Forming":
        watch_sent += f"follow-through on the {live_pattern.name} over the next session or two."
    elif nearest_resistance is not None and nearest_support is not None:
        watch_sent += (f"a close above {fmt_price(nearest_resistance)} to confirm strength, "
                        f"or a break below {fmt_price(nearest_support)} to signal weakness.")
    else:
        watch_sent += "confirmation of the current structure via a decisive close through the nearest key level."
    lines.append(watch_sent)

    # ---------------- SECTION 8: Trade Idea ----------------
    # This reports a DIRECTIONAL STANCE + VOLATILITY READ, and -- only when
    # a local same-day/recent option-chain snapshot exists for this ticker
    # (option_chain.py) -- a named structure with real strikes/credit/width
    # pulled from that chain and its own defining constraint actually
    # checked (e.g. a "Jade Lizard" is only named as such once the credit
    # collected is verified to exceed the call-spread width; otherwise it's
    # reported as the honest Bull Put Spread it actually is). Without a
    # chain for this ticker, structure selection is left to you.
    lines.append("")
    lines.append("--- TRADE IDEA ---")
    if real_iv_context is not None:
        lines.append("(Directional bias + volatility read below; any structure/strikes shown "
                      f"are verified against the {real_iv_context['snapshot_date']} option chain, "
                      "not fabricated.)")
    else:
        lines.append("(Directional bias + volatility read only. No options-chain data -- "
                      "structure selection and strike/credit verification are yours.)")

    trending_market = adx_val is not None and adx_val > 25

    # Prefer a REAL variance-risk-premium read (real ATM IV vs. realized
    # HV30, from a locally downloaded chain -- see fetch_real_iv_context)
    # over the HV-percentile proxy below whenever a chain is available:
    # realized-vol-only readings can only describe what already happened,
    # not what the market is pricing, and have already been caught
    # disagreeing with a real chain (SPCX showed HV-percentile "Elevated"
    # while real IV was routinely rich a different amount than that implied).
    if real_iv_context is not None and real_iv_context["iv_ratio"] is not None:
        iv_ratio = real_iv_context["iv_ratio"]
        # A RATIO, not the raw point spread -- a 2nd audit noted the point
        # spread isn't scale-invariant: 5 points on a 15%-IV name is a 33%
        # relative premium, 5 points on SPCX's 95% IV is barely 5%. 1.15x
        # is inside the audit's suggested 1.15-1.25x "rich" range (chosen
        # at the inclusive end deliberately: SPCX's own live ratio sits at
        # ~1.19x, and this tool has already relied on catching that case).
        IV_RICH_RATIO = 1.15
        IV_CHEAP_RATIO = 1 / IV_RICH_RATIO  # ~0.87x, symmetric in log-space
        iv_high = iv_ratio > IV_RICH_RATIO
        iv_low = iv_ratio < IV_CHEAP_RATIO
        iv_rich_fallback = False
    else:
        iv_high = iv_rank is not None and iv_rank > 50
        iv_low = iv_rank is not None and iv_rank < 30
        # IV rank needs 60+ days of history to compute (see calc_iv_rank); a
        # young ticker with none available previously fell through to "IV too
        # low" by default, which is backwards -- unknown premium richness is not
        # the same as cheap premium. Fall back to raw HV30 as a rough read
        # instead of silently assuming "low" when we simply don't know. Note
        # this fallback is itself weak: high realized vol means the stock moves
        # a lot, which is a reason options SHOULD be expensive, not proof they
        # are overpriced relative to what's coming (the actual edge in selling
        # premium is IV exceeding subsequent realized vol -- a relationship this
        # tool cannot measure with realized vol alone).
        iv_rich_fallback = iv_rank is None and hv30_val is not None and hv30_val > 0.50
    sell_premium_ok = iv_high or iv_rich_fallback

    # Earnings/FOMC dates are needed here (not just for the warning text
    # further down) so they can actually GATE sell_premium_ok. A 2nd audit
    # caught that the EARNINGS WARNING text said "do not treat this as a
    # premium-selling signal" while sell_premium_ok stayed unchanged and
    # the chain block below it printed verified strikes anyway -- advisory
    # text next to actionable strikes is the weaker half of "suppressed or
    # hard-flagged." This makes it suppressed.
    earnings_date = fetch_next_earnings_date(ticker)
    days_to_earnings = (earnings_date - report_date).days if earnings_date is not None else None
    earnings_imminent = days_to_earnings is not None and 0 <= days_to_earnings <= 7

    fomc_date = next_fomc_date(report_date)
    days_to_fomc = (fomc_date - report_date).days if fomc_date is not None else None
    fomc_imminent = days_to_fomc is not None and 0 <= days_to_fomc <= 7

    sell_premium_blocked_reason = None
    if sell_premium_ok and earnings_imminent:
        sell_premium_blocked_reason = (f"earnings on {earnings_date} "
                                        f"({days_to_earnings}d away) make the volatility read unreliable")
    elif sell_premium_ok and fomc_imminent:
        sell_premium_blocked_reason = (f"FOMC decision on {fomc_date} "
                                        f"({days_to_fomc}d away) makes the volatility read unreliable")
    if sell_premium_blocked_reason is not None:
        sell_premium_ok = False

    # A directional call additionally requires the same minimum-evidence
    # floor as the confluence verdict above -- net can cross +/-0.15 on a
    # single thin signal, and Trade Idea must not act more confident than
    # the Bottom Line it's built from.
    bullish_edge = sufficient_evidence and net >= 0.15
    bearish_edge = sufficient_evidence and net <= -0.15

    bullish_pattern = next((pm for pm in pattern_matches
                             if pm.bias == "Bullish" and pm.status != "Invalidated"), None)
    bearish_pattern = next((pm for pm in pattern_matches
                             if pm.bias == "Bearish" and pm.status != "Invalidated"), None)

    # net can be maxed (+/-1.00) purely from which side happened to have
    # SOME signal, independent of whether the evidence floor was cleared
    # -- a 2nd audit caught this printing "Insufficient evidence (net
    # +1.00)" live, which reads as high confidence right next to a "no
    # edge" verdict. Worse, with only 2 confluence families net is close
    # to degenerate even when evidence IS sufficient (68% of directional
    # calls land at exactly +/-1.00 the moment both families simply
    # agree, regardless of whether total_weight is barely over the floor
    # or twice that) -- so total_weight, not net, is always the headline
    # number here. net is still shown alongside once there's sufficient
    # evidence, since it does carry real information in the case the two
    # families partially disagree (net isn't +/-1.00 there).
    net_bit = (f"(total weight {total_weight:.1f}, net {net:+.2f})" if sufficient_evidence
               else f"(total weight {total_weight:.1f})")
    lines.append(f"Directional bias: {net_label} {net_bit}")

    if real_iv_context is not None and real_iv_context["iv_ratio"] is not None:
        iv_pct = real_iv_context["iv"] * 100
        ratio = real_iv_context["iv_ratio"]
        exp_bit = f"{real_iv_context['expiration']} exp"
        if iv_high:
            vol_text = (f"Real ATM IV {iv_pct:.0f}% ({exp_bit}) is {ratio:.2f}x HV30 "
                        f"— premium genuinely rich, per the {real_iv_context['snapshot_date']} chain snapshot")
        elif iv_low:
            vol_text = (f"Real ATM IV {iv_pct:.0f}% ({exp_bit}) is only {ratio:.2f}x HV30 "
                        "— premium not rich; selling here has little edge")
        else:
            vol_text = (f"Real ATM IV {iv_pct:.0f}% ({exp_bit}) roughly matches HV30 ({ratio:.2f}x) "
                        "— no clear edge buying or selling premium")
    elif iv_high:
        vol_text = f"Elevated IV rank ({iv_rank:.0f}%) — premium may be rich; verify against a real chain"
    elif iv_rich_fallback:
        vol_text = (f"High realized volatility (HV30 {hv30_val * 100:.0f}%), IV rank unavailable "
                     "(insufficient history) — premium richness unconfirmed, not a sell signal by itself")
    elif iv_low:
        vol_text = f"Low IV rank ({iv_rank:.0f}%) — premium likely cheap to sell, directional exposure favored"
    elif iv_rank is not None:
        vol_text = f"Moderate IV rank ({iv_rank:.0f}%) — no strong edge buying or selling premium"
    else:
        vol_text = "IV rank unavailable and volatility not clearly elevated"
    lines.append(f"Volatility read: {vol_text}")
    if sell_premium_blocked_reason is not None:
        lines.append(f"Premium-selling structures suppressed below: {sell_premium_blocked_reason}.")

    trend_text = (f"Trending (ADX {adx_val:.0f})" if trending_market
                  else f"Ranging (ADX {adx_val:.0f})" if adx_val is not None
                  else "Insufficient data")
    lines.append(f"Trend regime: {trend_text}")

    if earnings_imminent:
        lines.append(f"EARNINGS WARNING: {ticker} reports on {earnings_date} "
                      f"({days_to_earnings} day{'s' if days_to_earnings != 1 else ''} away). "
                      "The volatility read above is unreliable heading into a binary event -- "
                      "IV typically rises ahead of earnings while realized vol (what this tool "
                      "measures) does not. Premium-selling structures are suppressed until "
                      "after the event.")
    elif earnings_date is not None:
        lines.append(f"Next earnings: {earnings_date}.")

    dividend_date = fetch_next_dividend_date(ticker)
    days_to_dividend = (dividend_date - report_date).days if dividend_date is not None else None
    if days_to_dividend is not None and 0 <= days_to_dividend <= 7:
        lines.append(f"DIVIDEND WARNING: {ticker} goes ex-dividend on {dividend_date} "
                      f"({days_to_dividend} day{'s' if days_to_dividend != 1 else ''} away) -- a short "
                      "call carries early-assignment risk into this date if its extrinsic value drops "
                      "below the dividend. Check assignment risk before holding a short call through it.")

    ref_level = None
    ref_target_raw = None
    ref_label = None
    has_directional_setup = sufficient_evidence and (bullish_edge or bearish_edge)
    if has_directional_setup:
        if bullish_edge:
            ref_pattern = bullish_pattern
            ref_level = pd_low if pd_low is not None else nearest_support
            ref_label = "support"
            ref_target_raw = (parse_price(ref_pattern.price_target) if ref_pattern and ref_pattern.price_target != "N/A"
                               else nearest_resistance)
        else:
            ref_pattern = bearish_pattern
            ref_level = pd_high if pd_high is not None else nearest_resistance
            ref_label = "resistance"
            ref_target_raw = (parse_price(ref_pattern.price_target) if ref_pattern and ref_pattern.price_target != "N/A"
                               else nearest_support)

    # Risk/reward: reward = distance from spot to the reference target,
    # risk = the 1-ATR stop distance already shown above it. The tool
    # previously never computed this at all -- a setup with a $7.77 stop
    # and a $3.23 target printed identically to one with the reverse. A
    # ratio below MIN_RR is still shown (hiding the numbers outright would
    # remove information a trader might still want), but is explicitly
    # flagged rather than presented as an equally actionable idea.
    MIN_RR = 1.5
    rr = None
    if ref_target_raw is not None and atr_val is not None and atr_val > 0:
        rr = abs(ref_target_raw - current_price) / atr_val

    if has_directional_setup and ref_level is not None:
        lines.append(f"Reference level: {ref_label} near {fmt_price(ref_level)}")
    if has_directional_setup and atr_val is not None:
        lines.append(f"Reference stop distance: {fmt_price(atr_val)} (1 ATR — a distance, not a "
                      "risk-management rule; size and manage per the actual structure chosen)")

    if has_directional_setup and rr is not None:
        weak_bit = f" (below the {MIN_RR:.1f}:1 this tool treats as worth calling a real setup)" if rr < MIN_RR else ""
        lines.append(f"Reference target: {fmt_price(ref_target_raw)} — risk/reward {rr:.1f}:1{weak_bit} "
                      "(directional reference only -- meaningless for a credit structure, whose max "
                      "profit is the credit received)")
        if rr < MIN_RR:
            lines.append(f"CAUTION: weak risk/reward — a directional edge exists ({net_label.lower()}) "
                          "but the reference target doesn't clear this tool's risk/reward bar against a "
                          "1-ATR stop. Treat this as a weak setup even though a directional edge exists.")
        else:
            risk_dollars = ACCOUNT_SIZE * RISK_PCT_PER_TRADE
            risk_shares = int(risk_dollars // atr_val)
            max_notional = ACCOUNT_SIZE * MAX_NOTIONAL_PCT_PER_TRADE
            notional_shares = int(max_notional // current_price) if current_price > 0 else risk_shares
            shares = min(risk_shares, notional_shares)
            capped_bit = ""
            if notional_shares < risk_shares:
                capped_bit = (f" (capped down from {risk_shares} — the 1%-risk/{fmt_price(atr_val)}-stop math "
                               f"alone would size {fmt_price(risk_shares * current_price)} notional, over the "
                               f"{MAX_NOTIONAL_PCT_PER_TRADE * 100:.0f}% of account this tool caps a single "
                               "name at regardless of stop distance)")
            lines.append(f"Position size (equity/stock only): ~{shares} shares{capped_bit} — risking "
                          f"{fmt_price(min(shares * atr_val, risk_dollars))} "
                          f"({RISK_PCT_PER_TRADE * 100:.1f}% of a {fmt_price(ACCOUNT_SIZE)} account) at the "
                          f"{fmt_price(atr_val)} stop distance, {fmt_price(shares * current_price)} notional. "
                          "For an options structure, size by the structure's own max loss instead of "
                          "share count -- this number assumes a straight equity position.")
    elif has_directional_setup:
        lines.append("Reference target: unavailable — risk/reward cannot be assessed.")
    elif not trending_market and sell_premium_ok:
        chain_note = (f"a real strike/credit breakdown is below, from the {real_iv_context['snapshot_date']} chain"
                      if real_iv_context is not None else
                      "no local option chain for this ticker -- structure selection is yours, against a real chain")
        lines.append("Setup shape: ranging tape with volatility read favoring premium sale over "
                      f"direction -- a non-directional structure may fit better than a directional one here ({chain_note}).")
    else:
        lines.append("No actionable setup: insufficient directional evidence and no clear "
                      "volatility edge either way.")

    # ---- Real-chain structure verification -- only runs when a local
    # option-chain snapshot exists for this ticker; every strike/credit/
    # width number below is read from that chain, never guessed or
    # theoretical. ----
    if real_iv_context is not None and sell_premium_ok:
        chain, expiration = real_iv_context["chain"], real_iv_context["expiration"]
        chain_bit = f"{expiration} exp, {real_iv_context['snapshot_date']} chain snapshot"
        credit_spread_target_width = (CREDIT_SPREAD_WIDTH_ATR_MULT * atr_val
                                       if atr_val is not None else None)
        if bullish_edge:
            spread = build_credit_spread(chain, expiration, "put", target_short_delta=0.25,
                                          target_width=credit_spread_target_width)
            if spread is not None and spread["max_loss"] > 0:
                rr_credit = spread["credit"] / spread["max_loss"]
                delta_bit = f" (short delta {spread['short_delta']:.2f}" if spread["short_delta"] is not None else ""
                if delta_bit and spread["short_iv"] is not None:
                    delta_bit += f", short-strike IV {spread['short_iv'] * 100:.0f}%"
                delta_bit += ")" if delta_bit else ""
                lines.append(f"Bull Put Spread ({chain_bit}): "
                              f"sell {fmt_price(spread['short_strike'])}P / buy {fmt_price(spread['long_strike'])}P "
                              f"— credit {fmt_price(spread['credit'])}, width {fmt_price(spread['width'])}, "
                              f"max loss {fmt_price(spread['max_loss'])}, R:R {rr_credit:.2f}:1{delta_bit}")
                p_itm = bs_true_p_itm(spread["short_delta"], spread["short_iv"],
                                       real_iv_context["dte"], "put")
                ev = (estimate_credit_structure_ev(1 - p_itm, spread["haircut_credit"],
                                                    spread["haircut_max_loss"], num_legs=2)
                      if p_itm is not None else None)
                for caveat in credit_structure_caveats(ev, spread["credit"], spread["width"]):
                    lines.append(f"  CAVEAT: {caveat}")
                jade = check_jade_lizard(chain, expiration, spread, target_width=credit_spread_target_width)
                if jade is not None:
                    if jade["verified"]:
                        assignment_exposure = spread["short_strike"] * 100
                        stance_bit = ("" if ALLOW_NAKED_STRUCTURES else
                                      "  NOT shown as a recommendation (ALLOW_NAKED_STRUCTURES=False -- "
                                      "this account's rule is defined-risk only); shown as information.")
                        lines.append(f"  -> Verified as a Jade Lizard: naked {fmt_price(spread['short_strike'])}P "
                                      f"+ sell {fmt_price(jade['call_short_strike'])}C / buy "
                                      f"{fmt_price(jade['call_long_strike'])}C — total credit "
                                      f"{fmt_price(jade['total_credit'])} >= call spread width "
                                      f"{fmt_price(jade['call_width'])}, so upside risk is fully covered by credit. "
                                      f"DOWNSIDE IS UNDEFINED: the naked put means assignment exposure of "
                                      f"{fmt_price(assignment_exposure)}/contract (strike x 100) if it finishes "
                                      f"ITM, not the put spread's defined max loss.{stance_bit}")
                    else:
                        lines.append(f"  (Adding a {fmt_price(jade['call_short_strike'])}C/"
                                      f"{fmt_price(jade['call_long_strike'])}C call spread would NOT make this a "
                                      f"true Jade Lizard: total credit {fmt_price(jade['total_credit'])} falls "
                                      f"short of the {fmt_price(jade['call_width'])} width needed to cover upside "
                                      "risk — stick with the put spread alone.)")
            else:
                lines.append("Bull Put Spread: today's chain doesn't have a strike that clears this tool's "
                              "delta/liquidity/width bar — verify manually.")
        elif bearish_edge:
            spread = build_credit_spread(chain, expiration, "call", target_short_delta=0.25,
                                          target_width=credit_spread_target_width)
            if spread is not None and spread["max_loss"] > 0:
                rr_credit = spread["credit"] / spread["max_loss"]
                delta_bit = f" (short delta {spread['short_delta']:.2f}" if spread["short_delta"] is not None else ""
                if delta_bit and spread["short_iv"] is not None:
                    delta_bit += f", short-strike IV {spread['short_iv'] * 100:.0f}%"
                delta_bit += ")" if delta_bit else ""
                lines.append(f"Bear Call Spread ({chain_bit}): "
                              f"sell {fmt_price(spread['short_strike'])}C / buy {fmt_price(spread['long_strike'])}C "
                              f"— credit {fmt_price(spread['credit'])}, width {fmt_price(spread['width'])}, "
                              f"max loss {fmt_price(spread['max_loss'])}, R:R {rr_credit:.2f}:1{delta_bit}")
                p_itm = bs_true_p_itm(spread["short_delta"], spread["short_iv"],
                                       real_iv_context["dte"], "call")
                ev = (estimate_credit_structure_ev(1 - p_itm, spread["haircut_credit"],
                                                    spread["haircut_max_loss"], num_legs=2)
                      if p_itm is not None else None)
                for caveat in credit_structure_caveats(ev, spread["credit"], spread["width"]):
                    lines.append(f"  CAVEAT: {caveat}")
            else:
                lines.append("Bear Call Spread: today's chain doesn't have a strike that clears this tool's "
                              "delta/liquidity/width bar — verify manually.")
        elif not trending_market:
            put_spread = build_credit_spread(chain, expiration, "put", target_short_delta=0.16,
                                              target_width=credit_spread_target_width)
            call_spread = build_credit_spread(chain, expiration, "call", target_short_delta=0.16,
                                               target_width=credit_spread_target_width)
            if put_spread is not None and call_spread is not None:
                total_credit = put_spread["credit"] + call_spread["credit"]
                total_width = max(put_spread["width"], call_spread["width"])
                max_loss = total_width - total_credit
                total_haircut_credit = put_spread["haircut_credit"] + call_spread["haircut_credit"]
                haircut_max_loss = total_width - total_haircut_credit
                delta_bit = ""
                if put_spread["short_delta"] is not None and call_spread["short_delta"] is not None:
                    iv_bit = ""
                    if put_spread["short_iv"] is not None and call_spread["short_iv"] is not None:
                        iv_bit = (f", short-strike IV {put_spread['short_iv'] * 100:.0f}%P / "
                                  f"{call_spread['short_iv'] * 100:.0f}%C")
                    delta_bit = (f" (short deltas {put_spread['short_delta']:.2f}P / "
                                  f"{call_spread['short_delta']:.2f}C{iv_bit})")
                if max_loss > 0:
                    rr_bit = f", R:R {total_credit / max_loss:.2f}:1"
                    lines.append(f"Iron Condor ({chain_bit}): "
                                  f"sell {fmt_price(put_spread['short_strike'])}P/buy {fmt_price(put_spread['long_strike'])}P "
                                  f"+ sell {fmt_price(call_spread['short_strike'])}C/buy {fmt_price(call_spread['long_strike'])}C "
                                  f"— total credit {fmt_price(total_credit)}, max loss {fmt_price(max_loss)}{rr_bit}{delta_bit}")
                    p_itm_put = bs_true_p_itm(put_spread["short_delta"], put_spread["short_iv"],
                                               real_iv_context["dte"], "put")
                    p_itm_call = bs_true_p_itm(call_spread["short_delta"], call_spread["short_iv"],
                                                real_iv_context["dte"], "call")
                    if p_itm_put is not None and p_itm_call is not None:
                        # P(price stays between the two short strikes) = 1 - P(below put
                        # strike) - P(above call strike) -- these are mutually exclusive
                        # outcomes, not independent events, so this is NOT the product
                        # (1-P_itm_put)*(1-P_itm_call) the code used before. The product
                        # form overstates p_win: verified at matched deltas (0.16/0.16 ->
                        # product 0.706 vs correct 0.680; 0.35/0.35 -> product 0.423 vs
                        # correct 0.300), and the gap widens as either short leg gets
                        # closer to the money.
                        p_win = max(0.0, 1 - p_itm_put - p_itm_call)
                        ev = estimate_credit_structure_ev(p_win, total_haircut_credit, haircut_max_loss, num_legs=4)
                        for caveat in credit_structure_caveats(ev, total_credit, total_width):
                            lines.append(f"  CAVEAT: {caveat}")
                else:
                    lines.append(f"Iron Condor: today's chain prices this combination at credit "
                                  f"{fmt_price(total_credit)} >= max width {fmt_price(total_width)} -- "
                                  "not a valid credit structure at these strikes, skipping.")
            else:
                lines.append("Iron Condor: today's chain doesn't have strikes that clear this tool's "
                              "delta/liquidity/width bar — verify manually.")

    if divergence:
        direction_word = "bullish" if any("Bullish" in d for d in divergence) else "bearish"
        lines.append(f"Caution: {direction_word} RSI divergence detected — consider reduced size "
                      "or waiting for it to resolve before entering.")

    prediction_log.log_run(
        ticker=ticker, report_date=report_date, mode=mode, spot=current_price,
        pattern_matches=pattern_matches,
        confluence={
            "net": net, "net_label": net_label, "bull_weight": bull_weight,
            "bear_weight": bear_weight, "total_weight": total_weight,
            "sufficient_evidence": sufficient_evidence,
            "signals": [[label, direction, weight, note] for label, direction, weight, note in signals],
        },
        trade_idea={
            "directional_bias": net_label, "net": net, "iv_rank": iv_rank, "hv30": hv30_val,
            "adx": adx_val, "trending": trending_market, "earnings_date": earnings_date,
            "days_to_earnings": days_to_earnings, "reference_level": ref_level,
            "reference_stop_distance": atr_val, "reference_target": fmt_price(ref_target_raw) if ref_target_raw is not None else None,
        },
    )

    if iv_high or iv_rich_fallback:
        vol_stance = "sell premium"
    elif iv_low:
        vol_stance = "buy premium"
    else:
        vol_stance = "neutral/unknown"

    direction = 1 if net_label.startswith("Bullish") else -1 if net_label.startswith("Bearish") else 0
    portfolio_info = {
        "ticker": ticker,
        "net_label": net_label,
        "direction": direction,
        "sufficient_evidence": sufficient_evidence,
        "vol_stance": vol_stance,
        "earnings_soon": days_to_earnings is not None and 0 <= days_to_earnings <= 7,
        "spy_correlation": beta_ctx["correlation"] if beta_ctx is not None else None,
        "spy_beta": beta_ctx["beta"] if beta_ctx is not None else None,
    }

    return "\n".join(lines), portfolio_info


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    mode, now_et, is_trading_day = get_run_mode()
    mode_label = mode.capitalize()

    safe_print(f"Fetching data — run mode: {mode}")
    if not is_trading_day:
        safe_print(f"NOTE: {now_et.date()} is not a trading day (weekend/holiday) — "
                    "the report will reflect the last available session.")

    data = {}
    for ticker in ALL_TICKERS:
        df = fetch_history(ticker)
        if df is None:
            safe_print(f"WARNING: No data available for {ticker} — it will be skipped in the report.")
        data[ticker] = df

    # Stamp the report with the date of the most recent actual bar, not
    # blindly with "today" -- these differ on weekends/holidays, or if a
    # session's data hasn't posted yet, and a header claiming a date with no
    # underlying bar is worse than an honest "as of" the last real one.
    last_bar_dates = [df.index[-1].date() for df in data.values() if df is not None and not df.empty]
    report_date = max(last_bar_dates) if last_bar_dates else (
        now_et.date() if hasattr(now_et, "date") else date.today())

    vix_value, vix_timestamp, vix_is_stale = fetch_vix()

    spy_line = analyze_market_ticker("SPY", data.get("SPY"))
    qqq_line = analyze_market_ticker("QQQ", data.get("QQQ"))

    report_lines = []
    report_lines.append("=" * 40)
    report_lines.append("DAILY TECHNICAL ANALYSIS REPORT")
    report_lines.append(f"{report_date.strftime('%Y-%m-%d')} — {mode_label} Session")
    report_lines.append("")
    report_lines.append("MARKET CONTEXT")
    for line in build_market_context_section(vix_value, vix_timestamp, vix_is_stale, spy_line, qqq_line,
                                              report_date=report_date):
        report_lines.append(line)
    report_lines.append("=" * 40)
    report_lines.append("")

    portfolio_infos = []
    for ticker in WATCHLIST:
        df = data.get(ticker)
        premarket_data = None
        if mode == "morning" and df is not None:
            premarket_data = fetch_premarket(ticker)
        # A 2nd audit found analyze_ticker unguarded here: one bad quote in
        # one local chain file (or any other single-ticker failure) raised
        # all the way out and killed the ENTIRE report, for every ticker,
        # including the ones that had nothing wrong with them. The new
        # guards in build_credit_spread close the specific crash that was
        # found, but this is the backstop for whatever wasn't found.
        try:
            section, portfolio_info = analyze_ticker(ticker, df, mode, report_date, premarket_data,
                                                       spy_df=data.get("SPY"))
        except Exception as e:
            safe_print(f"WARNING: {ticker} failed to analyze ({e}) -- skipping it, "
                       "continuing with the rest of the report.")
            section = (f"{'=' * 60}\n{ticker}\n{'=' * 60}\n"
                       f"ERROR: analysis failed for this ticker ({e}) -- see console/logs. "
                       "Every other ticker in this report is unaffected.")
            portfolio_info = None
        report_lines.append(section)
        report_lines.append("")
        portfolio_infos.append(portfolio_info)

    report_lines.append("=" * 40)
    report_lines.append("PORTFOLIO VIEW")
    for line in build_portfolio_section(portfolio_infos):
        report_lines.append(line)
    report_lines.append("=" * 40)
    report_lines.append("")

    full_report = "\n".join(report_lines)

    print(full_report)

    if not os.path.isdir(SCRIPT_DIR):
        os.makedirs(SCRIPT_DIR, exist_ok=True)

    filename = f"daily_ta_report_{report_date.strftime('%Y-%m-%d')}_{mode}.txt"
    filepath = os.path.join(SCRIPT_DIR, filename)

    try:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(full_report)
    except Exception as e:
        safe_print(f"ERROR: Failed to save report: {e}")
        sys.exit(1)

    print(f"Report saved to {filepath}")


if __name__ == "__main__":
    main()
