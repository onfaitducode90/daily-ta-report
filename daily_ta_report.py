#!/usr/bin/env python3
"""
Daily Technical Analysis Report Generator
Generates a morning/evening/intraday technical analysis report for a
defined watchlist using yfinance data only.
"""

import os
import sys
import warnings
from datetime import datetime, timedelta, date

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

warnings.filterwarnings("ignore")

try:
    import yfinance as yf
except ImportError:
    print("ERROR: yfinance is required. Install with: pip install yfinance")
    sys.exit(1)

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


# ---------------------------------------------------------------------------
# Utility / mode helpers
# ---------------------------------------------------------------------------

def get_run_mode():
    """Determine morning/evening/intraday mode based on current ET time."""
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

    return mode, now_et


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
    """Best-effort attempt to fetch pre-market price data. Returns dict or None."""
    try:
        t = yf.Ticker(ticker)
        df = t.history(period="1d", interval="1m", prepost=True)
        if df is None or df.empty:
            return None
        df.index = pd.to_datetime(df.index)
        if ET is not None:
            try:
                df.index = df.index.tz_convert(ET)
            except Exception:
                pass
        # Filter to pre-market window (before 9:30 ET) if tz info available
        try:
            premkt = df[df.index.time < datetime.strptime("09:30", "%H:%M").time()]
        except Exception:
            premkt = df
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


def fetch_vix():
    try:
        t = yf.Ticker("^VIX")
        df = t.history(period="10d", interval="1d")
        if df is None or df.empty:
            return None
        return float(df["Close"].iloc[-1])
    except Exception as e:
        safe_print(f"WARNING: Failed to fetch VIX: {e}")
        return None


# ---------------------------------------------------------------------------
# Swing structure
# ---------------------------------------------------------------------------

def find_swings(df, lookback=3):
    """Identify swing highs/lows: bar's high/low is higher/lower than
    `lookback` bars before and after it."""
    highs = df["High"].values
    lows = df["Low"].values
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


# ---------------------------------------------------------------------------
# Indicators
# ---------------------------------------------------------------------------

def ema(series, span):
    return series.ewm(span=span, adjust=False).mean()


def sma(series, window):
    return series.rolling(window=window).mean()


def wilders_smooth(series, period):
    """Wilder's smoothing: first value = simple average of first `period`
    values, subsequent = (prior*(period-1) + current) / period."""
    values = series.values.astype(float)
    n = len(values)
    result = np.full(n, np.nan)
    if n < period:
        return pd.Series(result, index=series.index)
    first_avg = np.nanmean(values[:period])
    result[period - 1] = first_avg
    for i in range(period, n):
        result[i] = (result[i - 1] * (period - 1) + values[i]) / period
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
    close = df["Close"]

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
    adx = wilders_smooth(dx.fillna(0), period)

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


def calc_volume_poc(df, window=20, bins=20):
    sub = df.tail(window)
    if sub.empty:
        return None
    price_min = sub["Low"].min()
    price_max = sub["High"].max()
    if price_max <= price_min:
        return float(price_max)
    bin_edges = np.linspace(price_min, price_max, bins + 1)
    bin_volumes = np.zeros(bins)

    for _, row in sub.iterrows():
        # distribute bar's volume across bins it overlaps (typical price approx)
        typical = (row["High"] + row["Low"] + row["Close"]) / 3
        bin_idx = np.searchsorted(bin_edges, typical, side="right") - 1
        bin_idx = min(max(bin_idx, 0), bins - 1)
        bin_volumes[bin_idx] += row["Volume"]

    max_bin = int(np.argmax(bin_volumes))
    poc = (bin_edges[max_bin] + bin_edges[max_bin + 1]) / 2
    return float(poc)


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


def days_until_next_friday(from_date):
    days_ahead = (4 - from_date.weekday()) % 7  # Friday = weekday 4
    if days_ahead == 0:
        days_ahead = 7  # next Friday, not today
    return days_ahead


# ---------------------------------------------------------------------------
# Candle pattern recognition
# ---------------------------------------------------------------------------

def detect_engulfing(df):
    if len(df) < 2:
        return None
    today = df.iloc[-1]
    yday = df.iloc[-2]
    if today["Open"] < yday["Close"] and today["Close"] > yday["Open"]:
        return "Bullish engulfing"
    if today["Open"] > yday["Close"] and today["Close"] < yday["Open"]:
        return "Bearish engulfing"
    return None


def detect_pin_bar(df):
    if len(df) < 1:
        return None
    row = df.iloc[-1]
    body = abs(row["Close"] - row["Open"])
    lower_wick = min(row["Open"], row["Close"]) - row["Low"]
    upper_wick = row["High"] - max(row["Open"], row["Close"])
    if body == 0:
        body = 1e-9
    if lower_wick > 2 * body and upper_wick < 0.5 * body:
        return "Hammer"
    if upper_wick > 2 * body and lower_wick < 0.5 * body:
        return "Shooting star"
    return None


def detect_doji(df):
    if len(df) < 1:
        return None
    row = df.iloc[-1]
    body = abs(row["Close"] - row["Open"])
    total_range = row["High"] - row["Low"]
    if total_range == 0:
        return None
    if body < 0.10 * total_range:
        return "Doji"
    return None


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


def rsi_divergence(df, rsi_series, lookback=10):
    if len(df) < lookback or rsi_series.dropna().shape[0] < lookback:
        return None
    sub_close = df["Close"].tail(lookback)
    sub_high = df["High"].tail(lookback)
    sub_low = df["Low"].tail(lookback)
    sub_rsi = rsi_series.tail(lookback)

    price_hh_idx = sub_high.values.argmax()
    price_ll_idx = sub_low.values.argmin()

    # Compare most recent high/low vs the max/min excluding last bar for simple divergence check
    last_price_high = sub_high.iloc[-1]
    last_rsi_at_high = sub_rsi.iloc[-1]
    prior_max_high = sub_high.iloc[:-1].max()
    prior_max_high_idx = sub_high.iloc[:-1].values.argmax()
    prior_rsi_at_high = sub_rsi.iloc[:-1].iloc[prior_max_high_idx]

    last_price_low = sub_low.iloc[-1]
    last_rsi_at_low = sub_rsi.iloc[-1]
    prior_min_low = sub_low.iloc[:-1].min()
    prior_min_low_idx = sub_low.iloc[:-1].values.argmin()
    prior_rsi_at_low = sub_rsi.iloc[:-1].iloc[prior_min_low_idx]

    findings = []
    if last_price_high > prior_max_high and last_rsi_at_high < prior_rsi_at_high:
        findings.append("Bearish divergence — price higher high, RSI lower high")
    if last_price_low < prior_min_low and last_rsi_at_low > prior_rsi_at_low:
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


def build_market_context_section(vix_value, spy_trend_line, qqq_trend_line):
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
        lines.append(f"VIX: {vix_value:.1f} — {regime}")
    else:
        lines.append("VIX: N/A — data unavailable")
    lines.append(spy_trend_line)
    lines.append(qqq_trend_line)
    return lines


def analyze_market_ticker(ticker, df):
    """For SPY/QQQ market context — returns a one-line trend summary."""
    if df is None or len(df) < 20:
        return f"{ticker}: Insufficient data for structure analysis"
    trend, detail, _, _ = classify_structure(df, lookback_bars=len(df), swing_lookback=3)
    return f"{ticker}: {trend} — {detail}"


def analyze_ticker(ticker, df, mode, report_date, premarket_data=None):
    """Build the full multi-section report text for a single watchlist ticker."""
    lines = []
    lines.append("=" * 60)
    lines.append(f"{ticker}")
    lines.append("=" * 60)

    if df is None or df.empty:
        lines.append("WARNING: No data available for this ticker — skipping.")
        return "\n".join(lines)

    min_days_required = 30
    has_min_data = len(df) >= min_days_required
    has_full_data = len(df) >= 60

    if ticker == "SPCX" and len(df) < 60:
        lines.append(f"NOTE: SPCX has only {len(df)} trading days of history — "
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
        trend, detail, swing_highs, swing_lows = classify_structure(
            df, lookback_bars=min(60, len(df)), swing_lookback=3)
        lines.append(f"HTF Trend: {trend} — {detail}")
        if trend == "UPTREND":
            bias = "Long bias"
        elif trend == "DOWNTREND":
            bias = "Short bias"
        else:
            bias = "Neutral"
        lines.append(f"Market bias: {bias}")
    else:
        trend, bias = "RANGE", "Neutral"
        lines.append("HTF Trend: Insufficient data")
        lines.append("Market bias: Neutral")

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
        val21 = None
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

    # Rolling VWAP
    if len(df) >= 5:
        vwap5 = calc_rolling_vwap(df, window=5)
        lines.append(f"5-day VWAP: {fmt_price(vwap5)} — price "
                      f"{'above' if current_price > vwap5 else 'below'} "
                      f"({fmt_pct((current_price - vwap5) / vwap5 * 100)})")
    else:
        lines.append("5-day VWAP: Insufficient data")

    # Volume POC
    if len(df) >= 20:
        poc = calc_volume_poc(df, window=20, bins=20)
        lines.append(f"Volume POC: {fmt_price(poc)} — highest volume concentration")
    else:
        poc = None
        lines.append("Volume POC: Insufficient data")

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
    lines.append("--- PRICE ACTION (last 10 candles) ---")
    recent10 = df.tail(10)

    engulf = detect_engulfing(recent10)
    if engulf:
        lines.append(f"Candle pattern: {engulf}")

    pin = detect_pin_bar(recent10)
    if pin:
        lines.append(f"Candle pattern: {pin}")

    doji = detect_doji(recent10)
    if doji:
        lines.append(f"Candle pattern: {doji}")

    inside_count = detect_inside_bars(recent10)
    if inside_count > 0:
        lines.append(f"Inside bar(s): {inside_count} consecutive — compression signal")

    if not engulf and not pin and not doji and inside_count == 0:
        lines.append("Candle pattern: None detected")

    wick_result = wick_analysis(df, n=3)
    if wick_result:
        wick_label, avg_up, avg_lo = wick_result
        lines.append(f"Wick analysis: {wick_label}")

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

    squeeze_fired = False
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
            squeeze_fired = True
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
            divergence = rsi_divergence(df, rsi_series, lookback=10)
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
    if len(df) >= 30:
        hv30_val = calc_hv(df["Close"], window=30)
        if hv30_val is not None:
            lines.append(f"HV30: {hv30_val * 100:.1f}%")
            if len(df) >= 60:
                iv_rank = calc_iv_rank(df["Close"], window=30, lookback=252)
                if iv_rank is not None:
                    if iv_rank < 30:
                        iv_label = "Low IV — avoid selling premium, wait for expansion"
                    elif iv_rank < 50:
                        iv_label = "Moderate IV — selective premium selling"
                    elif iv_rank < 70:
                        iv_label = "Elevated IV — iron condor and jade lizard conditions favorable"
                    else:
                        iv_label = "Very high IV — premium selling highly favorable, size conservatively"
                    lines.append(f"IV Rank (HV-based): {iv_rank:.1f}% — {iv_label}")
                else:
                    lines.append("IV Rank: Insufficient data")
            else:
                lines.append("IV Rank: Insufficient data (need 60+ days)")
        else:
            lines.append("HV30: Insufficient data")
    else:
        lines.append("HV30 / IV Rank: Insufficient data")

    # ---------------- SECTION 7: Plain English Summary ----------------
    lines.append("")
    lines.append("--- PLAIN ENGLISH SUMMARY ---")
    mode_label = mode.capitalize()
    date_str = report_date.strftime("%Y-%m-%d")
    lines.append(f"{ticker} — {date_str} {mode_label} Report:")

    trend_sent = f"Trend: {'The higher timeframe structure is ' + trend.lower() + ' (' + bias.lower() + ')' if has_min_data else 'Insufficient data to determine trend'}."

    nearest_support = None
    nearest_resistance = None
    candidates_below = [v for v in [pd_low, pw_low, low52] if v is not None and v < current_price]
    candidates_above = [v for v in [pd_high, pw_high, high52] if v is not None and v > current_price]
    if candidates_below:
        nearest_support = max(candidates_below)
    if candidates_above:
        nearest_resistance = min(candidates_above)

    levels_sent = "Key levels: "
    if nearest_support is not None and nearest_resistance is not None:
        levels_sent += f"Nearest support sits near {fmt_price(nearest_support)} with resistance near {fmt_price(nearest_resistance)}."
    elif nearest_support is not None:
        levels_sent += f"Nearest support sits near {fmt_price(nearest_support)}, no clear resistance level identified."
    elif nearest_resistance is not None:
        levels_sent += f"Nearest resistance sits near {fmt_price(nearest_resistance)}, no clear support level identified."
    else:
        levels_sent += "Insufficient data to identify key levels."

    mom_bits = []
    if squeeze_data is not None:
        if current_squeeze_on:
            mom_bits.append("squeeze is compressed")
        elif fired_today:
            mom_bits.append("squeeze just fired")
        else:
            mom_bits.append("squeeze is off")
    if len(df) >= 35:
        mom_bits.append(f"MACD is {macd_status.lower()}")
    if adx_val is not None:
        mom_bits.append(f"ADX at {adx_val:.0f} ({'trending' if adx_val > 25 else 'ranging'})")
    momentum_sent = "Momentum: " + (", ".join(mom_bits) + "." if mom_bits else "Insufficient data.")

    vol_bits = []
    if iv_rank is not None:
        vol_bits.append(f"IV rank is {iv_rank:.0f}%")
    if atr_val is not None:
        vol_bits.append(f"ATR is {fmt_price(atr_val)}")
    vol_sent = "Volatility: " + (", ".join(vol_bits) + "." if vol_bits else "Insufficient data.")

    watch_sent = "Watch for: "
    if nearest_resistance is not None and nearest_support is not None:
        watch_sent += (f"a close above {fmt_price(nearest_resistance)} to confirm strength, "
                        f"or a break below {fmt_price(nearest_support)} to signal weakness.")
    else:
        watch_sent += "confirmation of the current structure via a decisive close through the nearest key level."

    lines.append(trend_sent)
    lines.append(levels_sent)
    lines.append(momentum_sent)
    lines.append(vol_sent)
    lines.append(watch_sent)

    # ---------------- SECTION 8: Trade Idea ----------------
    lines.append("")
    lines.append("--- TRADE IDEA ---")

    trend_bullish = trend == "UPTREND"
    trend_bearish = trend == "DOWNTREND"

    # Derive momentum state directly from the momentum series here so the
    # trade logic never confuses "insufficient data" (default False upstream)
    # with a genuine negative/falling reading.
    mom_current = None
    mom_prev = None
    if squeeze_data is not None:
        momentum_series_s8 = squeeze_data["momentum"]
        if not momentum_series_s8.dropna().empty:
            mc = momentum_series_s8.iloc[-1]
            mp = momentum_series_s8.iloc[-2] if len(momentum_series_s8) >= 2 else np.nan
            if not np.isnan(mc):
                mom_current = float(mc)
            if not np.isnan(mp):
                mom_prev = float(mp)

    momentum_negative_s8 = mom_current is not None and mom_current < 0
    momentum_positive_s8 = mom_current is not None and mom_current > 0
    momentum_falling_s8 = mom_current is not None and mom_prev is not None and mom_current < mom_prev
    momentum_rising_s8 = mom_current is not None and mom_prev is not None and mom_current > mom_prev

    # "Not strongly directional" = absolute momentum value sits in the lowest
    # 30% of its own 20-period range.
    momentum_low_directional = False
    if squeeze_data is not None and mom_current is not None:
        last20_mom = squeeze_data["momentum"].tail(20).dropna()
        if len(last20_mom) >= 10:
            pct = scipy_stats.percentileofscore(last20_mom.abs().values, abs(mom_current))
            momentum_low_directional = pct <= 30

    # Squeeze fired within the last 3 bars, not just the most recent bar.
    squeeze_fired_last3 = False
    if squeeze_data is not None:
        squeeze_on_series_s8 = squeeze_data["squeeze_on"]
        consec_series_s8 = squeeze_data["consec"]
        n_bars = len(squeeze_on_series_s8)
        for i in range(max(1, n_bars - 3), n_bars):
            cur_on = squeeze_on_series_s8.iloc[i]
            prev_on = squeeze_on_series_s8.iloc[i - 1]
            prev_cnt = consec_series_s8.iloc[i - 1]
            if pd.isna(cur_on) or pd.isna(prev_on):
                continue
            if (not bool(cur_on)) and bool(prev_on) and prev_cnt >= 6:
                squeeze_fired_last3 = True
                break

    price_above_200sma = len(df) >= 200 and not np.isnan(sma200.iloc[-1]) and current_price > sma200.iloc[-1]
    bullish_engulfing_last = engulf == "Bullish engulfing"

    trade_idea = None
    entry_zone = None
    stop_level = None
    target_level = None

    # Condition 1 — Bearish breakdown
    if trend_bearish and momentum_negative_s8 and momentum_falling_s8 and adx_val is not None and adx_val > 25:
        resistance_ref = pd_high if pd_high is not None else nearest_resistance
        trade_idea = ("BEARISH CALL SPREAD candidate — confirmed downtrend with strong momentum. "
                       f"Consider selling call spread above resistance at {fmt_price(resistance_ref)}.")
        if resistance_ref is not None:
            entry_zone = f"Short near {fmt_price(resistance_ref)} (prior day high)"
            if atr_val is not None:
                stop_level = f"{fmt_price(resistance_ref + atr_val)} (1 ATR above entry)"
            target_level = fmt_price(nearest_support) if nearest_support is not None else "Next support level (insufficient data)"

    # Condition 2 — Bullish breakout
    elif squeeze_fired_last3 and momentum_positive_s8 and trend_bullish:
        trade_idea = ("BULLISH PUT SPREAD or SWING LONG candidate — squeeze breakout with bullish momentum. "
                       f"Consider entry above {fmt_price(pd_high)} with stop at 1 ATR below entry.")
        if pd_high is not None:
            entry_zone = f"Above {fmt_price(pd_high)} (prior day high breakout)"
            if atr_val is not None:
                stop_level = f"{fmt_price(pd_high - atr_val)} (1 ATR below entry)"
            target_level = fmt_price(nearest_resistance) if nearest_resistance is not None else "Next resistance level (insufficient data)"

    # Condition 3 — Jade lizard / bullish put spread
    elif bullish_engulfing_last and momentum_rising_s8 and iv_rank is not None and iv_rank > 50 and price_above_200sma:
        trade_idea = ("JADE LIZARD candidate — bullish engulfing at support with elevated IV and bullish structure. "
                       f"Consider jade lizard with short put below {fmt_price(pd_low)}.")
        if pd_low is not None:
            entry_zone = f"Short put below {fmt_price(pd_low)} (prior day low)"
            if atr_val is not None:
                stop_level = f"{fmt_price(pd_low - atr_val)} (1 ATR below entry)"
            target_level = fmt_price(nearest_resistance) if nearest_resistance is not None else "Next resistance level (insufficient data)"

    # Condition 4 — Iron condor
    elif iv_rank is not None and iv_rank > 50 and adx_val is not None and adx_val < 25:
        expected_move_s8 = None
        if hv30_val is not None:
            days_to_friday_s8 = days_until_next_friday(report_date)
            expected_move_s8 = current_price * (hv30_val / np.sqrt(252)) * np.sqrt(days_to_friday_s8)
        if expected_move_s8 is not None:
            put_strike = current_price - expected_move_s8
            call_strike = current_price + expected_move_s8
            trade_idea = (f"IRON CONDOR candidate — IV rank {iv_rank:.0f}% with ranging ADX {adx_val:.0f}. "
                           f"Price expected to stay within {fmt_price(put_strike)} to {fmt_price(call_strike)} "
                           "this week. Short strikes beyond this range.")
            entry_zone = f"Sell condor at current price {fmt_price(current_price)}"
            if atr_val is not None:
                stop_level = f"Adjust if price closes beyond {fmt_price(current_price - atr_val)} / {fmt_price(current_price + atr_val)} (1 ATR)"
            target_level = f"Max profit if price stays between {fmt_price(put_strike)} and {fmt_price(call_strike)} through expiration"
        else:
            trade_idea = (f"IRON CONDOR candidate — IV rank {iv_rank:.0f}% with ranging ADX {adx_val:.0f}. "
                           "Insufficient data to calculate expected move for strike selection.")

    # Condition 5 — Caution (RSI divergence)
    elif divergence:
        direction = "bullish" if any("Bullish" in d for d in divergence) else "bearish"
        trade_idea = (f"CAUTION — {direction} divergence detected. "
                       "Wait for confirmation before entering any position.")

    # Condition 6 — Low IV
    elif iv_rank is not None and iv_rank < 30:
        trade_idea = ("NO TRADE — IV too low for premium selling. "
                       "Wait for volatility expansion.")

    # Condition 7 — Default
    else:
        trade_idea = "WAIT — no high-confidence setup present today."

    lines.append(f"TRADE IDEA: {trade_idea}")
    if entry_zone is not None:
        lines.append(f"Entry zone: {entry_zone}")
    if stop_level is not None:
        lines.append(f"Stop (1 ATR): {stop_level}")
    if target_level is not None:
        lines.append(f"Target: {target_level}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    mode, now_et = get_run_mode()
    report_date = now_et.date() if hasattr(now_et, "date") else date.today()
    mode_label = mode.capitalize()

    safe_print(f"Fetching data — run mode: {mode}")

    data = {}
    for ticker in ALL_TICKERS:
        df = fetch_history(ticker)
        if df is None:
            safe_print(f"WARNING: No data available for {ticker} — it will be skipped in the report.")
        data[ticker] = df

    vix_value = fetch_vix()

    spy_line = analyze_market_ticker("SPY", data.get("SPY"))
    qqq_line = analyze_market_ticker("QQQ", data.get("QQQ"))

    report_lines = []
    report_lines.append("=" * 40)
    report_lines.append("DAILY TECHNICAL ANALYSIS REPORT")
    report_lines.append(f"{report_date.strftime('%Y-%m-%d')} — {mode_label} Session")
    report_lines.append("")
    report_lines.append("MARKET CONTEXT")
    for line in build_market_context_section(vix_value, spy_line, qqq_line):
        report_lines.append(line)
    report_lines.append("=" * 40)
    report_lines.append("")

    for ticker in WATCHLIST:
        df = data.get(ticker)
        premarket_data = None
        if mode == "morning" and df is not None:
            premarket_data = fetch_premarket(ticker)
        section = analyze_ticker(ticker, df, mode, report_date, premarket_data)
        report_lines.append(section)
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
