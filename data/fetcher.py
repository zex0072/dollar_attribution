"""
Unified data fetching: yfinance (market) + optional FRED (macro rates/SOFR).
All prices returned as daily close DataFrames indexed by date.
"""

import os
import ssl
import logging
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf

try:
    import certifi
    ssl._create_default_https_context = lambda: ssl.create_default_context(
        cafile=certifi.where()
    )
except ImportError:
    pass

from config import MARKET_TICKERS, FRED_SERIES, LOOKBACK_DAYS

log = logging.getLogger(__name__)


def _date_range():
    end   = datetime.today()
    start = end - timedelta(days=int(LOOKBACK_DAYS * 1.5))  # extra buffer for weekends
    return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")


def fetch_market_data() -> pd.DataFrame:
    """
    Pull daily close prices for all yfinance tickers.
    Returns a DataFrame with ticker keys as columns.
    """
    start, end = _date_range()
    tickers = list(MARKET_TICKERS.values())

    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)

    if isinstance(raw.columns, pd.MultiIndex):
        closes = raw["Close"]
    else:
        closes = raw[["Close"]].rename(columns={"Close": tickers[0]})

    # Rename columns from yfinance ticker to our friendly key
    inv = {v: k for k, v in MARKET_TICKERS.items()}
    closes.columns = [inv.get(c, c) for c in closes.columns]

    # yfinance already returns Treasury yields in percent (e.g. ^TNX = 4.32 means 4.32%)
    # No scaling needed — values are used directly as percentage points

    closes.index = pd.to_datetime(closes.index)
    closes = closes.sort_index().dropna(how="all")
    return closes


def fetch_fred_data() -> pd.DataFrame:
    """
    Fetch SOFR, Fed Funds, and Treasury yields from FRED.
    Requires FRED_API_KEY environment variable.
    Returns empty DataFrame on failure (app degrades gracefully).
    """
    api_key = os.environ.get("FRED_API_KEY", "")
    if not api_key:
        log.warning("FRED_API_KEY not set — SOFR/exact 2Y yield unavailable")
        return pd.DataFrame()

    try:
        from fredapi import Fred
        fred = Fred(api_key=api_key)
        start, end = _date_range()

        frames = {}
        for key, series_id in FRED_SERIES.items():
            try:
                s = fred.get_series(series_id, observation_start=start,
                                    observation_end=end)
                frames[key] = s
            except Exception as e:
                log.warning("FRED series %s failed: %s", series_id, e)

        if not frames:
            return pd.DataFrame()

        df = pd.DataFrame(frames)
        df.index = pd.to_datetime(df.index)
        df = df.sort_index().ffill()
        return df

    except ImportError:
        log.warning("fredapi not installed — pip install fredapi")
        return pd.DataFrame()
    except Exception as e:
        log.warning("FRED fetch failed: %s", e)
        return pd.DataFrame()


def build_master_frame() -> pd.DataFrame:
    """
    Merge market + FRED data into one aligned daily DataFrame.
    Derived columns added:
      - yield_curve   : 10Y − 2Y spread
      - funding_stress: 3M T-bill − Fed Funds (FRA-OIS proxy)
      - sofr_basis    : 3M avg SOFR − overnight SOFR (if FRED available)
    """
    mkt  = fetch_market_data()
    fred = fetch_fred_data()

    df = mkt.copy()

    if not fred.empty:
        fred = fred.reindex(df.index, method="ffill")
        for col in fred.columns:
            df[col] = fred[col]

        # Prefer FRED's exact 2-year yield
        if "ty2" in df.columns:
            df["ty2_use"] = df["ty2"]
        else:
            df["ty2_use"] = df.get("ty5", np.nan)

        # SOFR funding stress: overnight SOFR spread vs FF
        if "sofr" in df.columns and "fed_funds" in df.columns:
            df["sofr_spread"] = df["sofr"] - df["fed_funds"]
        else:
            df["sofr_spread"] = np.nan

        # Yield curve: 10Y − 2Y
        df["yield_curve"] = df.get("ty10_fred", df["ty10"]) - df["ty2_use"]

        # FRA-OIS proxy: 3M T-bill − Fed Funds
        ty3m = df.get("ty3m_fred", df.get("ty3m", pd.Series(np.nan, index=df.index)))
        ff   = df.get("fed_funds", pd.Series(np.nan, index=df.index))
        df["fra_ois_proxy"] = ty3m - ff

    else:
        # Fallback: derive from yfinance yields (all in %)
        df["ty2_use"]      = df.get("ty5", np.nan)    # 5Y as 2Y proxy
        df["yield_curve"]  = df["ty10"] - df["ty2_use"]
        df["sofr_spread"]  = np.nan
        # FRA-OIS proxy without FRED: 3M T-bill − 5Y yield (front-end curve slope)
        # Positive → T-bill trades above medium-term rates (inversion / stress signal)
        df["fra_ois_proxy"] = df.get("ty3m", np.nan) - df["ty2_use"]

    # Keep only trading days (drop rows where DXY is missing)
    df = df[df["dxy"].notna()].copy()
    df = df.tail(LOOKBACK_DAYS)
    return df
