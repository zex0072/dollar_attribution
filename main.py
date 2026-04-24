"""
Dollar Volatility Attribution Model — entry point.

Usage:
    python main.py [--port 8050] [--debug]

Optional env vars:
    FRED_API_KEY=<your_key>   enables true 2Y yield, SOFR, Fed Funds from FRED
"""

import sys
import os
import argparse
import logging

# Ensure package imports resolve correctly
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("usd_model")


def main():
    parser = argparse.ArgumentParser(description="USD Attribution Monitor")
    parser.add_argument("--port",  type=int, default=8050)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--host",  default="127.0.0.1")
    args = parser.parse_args()

    fred_key = os.environ.get("FRED_API_KEY", "")
    if fred_key:
        log.info("FRED API key detected — will fetch SOFR, 2Y yield, Fed Funds")
    else:
        log.warning("No FRED_API_KEY — using yfinance proxies for yields/SOFR")
        log.warning("  → Set FRED_API_KEY for full SOFR / exact 2Y data")
        log.warning("  → Free key: https://fredaccount.stlouisfed.org/apikey")

    log.info("Pre-fetching market data…")
    try:
        from data.fetcher import build_master_frame
        df = build_master_frame()
        log.info("Loaded %d trading days × %d columns", len(df), len(df.columns))
        log.info("Date range: %s → %s", df.index[0].date(), df.index[-1].date())
    except Exception as e:
        log.error("Data fetch failed: %s", e)
        sys.exit(1)

    from viz.dashboard import app
    log.info("Dashboard starting at http://%s:%d", args.host, args.port)
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
