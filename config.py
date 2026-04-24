"""
Dollar Volatility Attribution Model — configuration constants.
"""

# ── Lookback & rolling window ─────────────────────────────────────────────────
LOOKBACK_DAYS   = 365        # trading-day history to pull
ROLLING_WINDOW  = 60         # rolling-OLS regression window (trading days)
REFRESH_SECONDS = 300        # dashboard auto-refresh interval

# ── DXY geometric-basket weights ─────────────────────────────────────────────
# DXY = 50.14348112 × EUR^-0.576 × JPY^0.136 × GBP^-0.119 × CAD^0.091
#                   × SEK^0.042 × CHF^0.036
# Sign: positive weight → quoted USD/FX (USD numerator, stronger dollar ↑ DXY)
#       negative weight → quoted FX/USD (dollar is denominator, stronger $ ↓ pair)
DXY_COMPONENTS = {
    "EURUSD=X": {"weight": -0.576, "label": "EUR", "color": "#2196F3"},
    "USDJPY=X": {"weight":  0.136, "label": "JPY", "color": "#FF9800"},
    "GBPUSD=X": {"weight": -0.119, "label": "GBP", "color": "#9C27B0"},
    "USDCAD=X": {"weight":  0.091, "label": "CAD", "color": "#F44336"},
    "USDSEK=X": {"weight":  0.042, "label": "SEK", "color": "#009688"},
    "USDCHF=X": {"weight":  0.036, "label": "CHF", "color": "#795548"},
}

# ── yfinance market tickers ───────────────────────────────────────────────────
MARKET_TICKERS = {
    "dxy":    "DX-Y.NYB",
    "eur":    "EURUSD=X",
    "jpy":    "USDJPY=X",
    "gbp":    "GBPUSD=X",
    "cad":    "USDCAD=X",
    "sek":    "USDSEK=X",
    "chf":    "USDCHF=X",
    "gold":   "GC=F",
    "vix":    "^VIX",
    "ty10":   "^TNX",       # 10-year Treasury yield (×0.1 = %)
    "ty5":    "^FVX",       # 5-year (used as 2Y proxy when FRED unavailable)
    "ty3m":   "^IRX",       # 13-week T-bill
}

# ── FRED series (optional — requires FRED_API_KEY env var) ───────────────────
FRED_SERIES = {
    "sofr":      "SOFR",       # Secured Overnight Financing Rate
    "fed_funds": "DFF",        # Effective Federal Funds Rate
    "ty2":       "DGS2",       # 2-Year Treasury yield
    "ty10_fred": "DGS10",      # 10-Year Treasury yield (backup)
    "ty3m_fred": "DGS3MO",     # 3-Month Treasury yield
}

# ── Macro factor labels for the OLS attribution ──────────────────────────────
FACTOR_COLORS = {
    "Δ 2Y Yield":      "#2196F3",
    "Δ Yield Curve":   "#FF9800",   # 10Y - 2Y spread
    "Δ ln(VIX)":       "#F44336",
    "Δ ln(Gold)":      "#FFC107",
    "Δ Funding Stress":"#9C27B0",   # 3M T-bill − Fed Funds proxy
    "Residual":        "#607D8B",
}

# ── Regime thresholds ─────────────────────────────────────────────────────────
REGIME_RULES = {
    "Risk-Off":        {"vix_z": 1.5},   # VIX z-score above threshold
    "Rate-Driven":     {"ry_r2": 0.5},   # rate factors explain > 50 % of R²
    "Funding Stress":  {"fra_ois_z": 2}, # FRA-OIS z-score
    "EUR Dominance":   {"eur_share": 0.6},  # EUR contribution share
}
