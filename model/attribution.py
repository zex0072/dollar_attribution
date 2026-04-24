"""
Dollar Volatility Attribution Model — two complementary decompositions:

1. Currency Attribution (exact geometric)
   Δln(DXY) ≈ Σ wᵢ·Δln(FXᵢ)
   Tells us WHICH currency drove the move.

2. Macro Factor Attribution (rolling OLS)
   Δln(DXY) ~ β₁·Δ(2Y) + β₂·Δ(curve) + β₃·Δln(VIX) + β₄·Δln(Gold) + β₅·Δ(funding)
   Tells us WHY the dollar moved (rates / risk-off / funding stress).
"""

import numpy as np
import pandas as pd
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant

from config import DXY_COMPONENTS, ROLLING_WINDOW


# ── 1. Currency Attribution ───────────────────────────────────────────────────

def currency_attribution(df: pd.DataFrame) -> pd.DataFrame:
    """
    Exact log-return decomposition of daily DXY changes.

    Returns DataFrame with one column per DXY component (label name),
    plus 'residual', indexed same as df.
    Each value = contribution in log-return points (multiply by 100 for bps).
    """
    log_ret_dxy = np.log(df["dxy"]).diff()

    parts = {}
    for ticker, meta in DXY_COMPONENTS.items():
        key   = _ticker_to_key(ticker)
        label = meta["label"]
        w     = meta["weight"]
        if key in df.columns:
            lr = np.log(df[key]).diff()
            parts[label] = w * lr
        else:
            parts[label] = pd.Series(np.nan, index=df.index)

    attr = pd.DataFrame(parts, index=df.index)
    attr["residual"] = log_ret_dxy - attr.sum(axis=1)
    return attr


def _ticker_to_key(ticker: str) -> str:
    """Map yfinance ticker back to column name in master frame."""
    mapping = {
        "EURUSD=X": "eur",
        "USDJPY=X": "jpy",
        "GBPUSD=X": "gbp",
        "USDCAD=X": "cad",
        "USDSEK=X": "sek",
        "USDCHF=X": "chf",
    }
    return mapping.get(ticker, ticker)


# ── 2. Macro Factor Attribution (Rolling OLS) ─────────────────────────────────

def _build_factors(df: pd.DataFrame) -> pd.DataFrame:
    """Construct factor change series for the OLS model."""
    f = pd.DataFrame(index=df.index)

    # Rate level: 2-year yield change (rate expectations)
    f["Δ 2Y Yield"]     = df["ty2_use"].diff()

    # Term structure: yield curve steepening/flattening
    f["Δ Yield Curve"]  = df["yield_curve"].diff()

    # Risk sentiment: log change in VIX
    f["Δ ln(VIX)"]      = np.log(df["vix"]).diff()

    # Real-asset / safe-haven: log change in Gold
    f["Δ ln(Gold)"]     = np.log(df["gold"]).diff()

    # Funding stress: FRA-OIS proxy change
    f["Δ Funding Stress"] = df["fra_ois_proxy"].diff()

    return f.dropna(how="all")


def rolling_ols_attribution(df: pd.DataFrame,
                             window: int = ROLLING_WINDOW
                             ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Rolling OLS attribution of DXY log-returns onto macro factors.

    Returns:
      contributions : DataFrame — each factor's contribution per day
                      (β̂ᵢ_rolling × Δxᵢ), plus 'Residual'
      betas         : DataFrame — rolling β coefficients per factor
      r2_series     : Series — rolling R²
    """
    y_all = np.log(df["dxy"]).diff().dropna()
    X_all = _build_factors(df).reindex(y_all.index)

    contributions = pd.DataFrame(index=y_all.index, columns=list(X_all.columns) + ["Residual"],
                                  dtype=float)
    betas         = pd.DataFrame(index=y_all.index, columns=list(X_all.columns), dtype=float)
    r2_series     = pd.Series(np.nan, index=y_all.index)

    idx = y_all.index

    for i in range(window, len(idx)):
        win_idx = idx[i - window: i]
        y_win   = y_all.loc[win_idx]
        X_win   = X_all.loc[win_idx].copy()

        # Drop columns with too many NaNs in this window
        valid_cols = X_win.columns[X_win.notna().mean() > 0.8].tolist()
        X_win = X_win[valid_cols].dropna()
        y_win = y_win.reindex(X_win.index)

        if len(y_win) < window // 2 or X_win.empty:
            continue

        try:
            res   = OLS(y_win, add_constant(X_win)).fit()
            today = idx[i]
            x_t   = X_all.loc[today, valid_cols]
            b     = res.params[valid_cols]  # exclude const

            for col in valid_cols:
                contributions.at[today, col] = b[col] * x_t[col] if pd.notna(x_t[col]) else 0.0
                betas.at[today, col]         = b[col]

            fitted    = (b * x_t.fillna(0)).sum() + res.params.get("const", 0)
            contributions.at[today, "Residual"] = y_all.loc[today] - fitted
            r2_series.loc[today] = res.rsquared

        except Exception:
            continue

    contributions = contributions.astype(float)
    return contributions, betas, r2_series


# ── 3. Summary Statistics ──────────────────────────────────────────────────────

def factor_share(contributions: pd.DataFrame, window: int = 20) -> pd.Series:
    """
    Absolute contribution share for each factor over the last `window` days.
    Useful for identifying the dominant driver in the recent period.
    """
    recent = contributions.tail(window).abs().mean()
    total  = recent.sum()
    return (recent / total).sort_values(ascending=False) if total > 0 else recent


def dxy_z_score(df: pd.DataFrame, window: int = 60) -> float:
    """Current DXY level expressed as z-score over rolling `window` days."""
    s = df["dxy"].dropna()
    if len(s) < window:
        return 0.0
    mu  = s.rolling(window).mean().iloc[-1]
    sig = s.rolling(window).std().iloc[-1]
    return float((s.iloc[-1] - mu) / sig) if sig > 0 else 0.0


def vix_z_score(df: pd.DataFrame, window: int = 60) -> float:
    s = df["vix"].dropna()
    if len(s) < window:
        return 0.0
    mu  = s.rolling(window).mean().iloc[-1]
    sig = s.rolling(window).std().iloc[-1]
    return float((s.iloc[-1] - mu) / sig) if sig > 0 else 0.0
