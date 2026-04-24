"""
Market regime classification and signal generation.
Reads from attribution outputs to label the current DXY environment.
"""

from typing import Optional

import pandas as pd
import numpy as np

from model.attribution import factor_share, vix_z_score, dxy_z_score


def classify_regime(df: pd.DataFrame, contributions: pd.DataFrame,
                    ccy_attr: Optional[pd.DataFrame] = None) -> dict:
    """
    Return a dict with:
      regime      : dominant label string
      drivers     : ordered list of (factor, share)
      flags       : dict of boolean condition flags
      dxy_z       : current DXY z-score
      vix_z       : current VIX z-score
    """
    shares = factor_share(contributions, window=10)
    vz     = vix_z_score(df)
    dz     = dxy_z_score(df)

    flags = {
        "risk_off":       vz > 1.5,
        "risk_on":        vz < -1.0,
        "rate_driven":    shares.get("Δ 2Y Yield", 0) + shares.get("Δ Yield Curve", 0) > 0.40,
        "funding_stress": shares.get("Δ Funding Stress", 0) > 0.25,
        "eur_dominated":  _eur_share(ccy_attr) > 0.55,
        "gold_driven":    shares.get("Δ ln(Gold)", 0) > 0.25,
        "strong_usd":     dz > 1.0,
        "weak_usd":       dz < -1.0,
    }

    # Priority order for regime label
    if flags["funding_stress"] and flags["risk_off"]:
        regime = "Funding Stress / Risk-Off"
    elif flags["risk_off"] and flags["rate_driven"]:
        regime = "Risk-Off + Rate Bid"
    elif flags["risk_off"]:
        regime = "Risk-Off"
    elif flags["funding_stress"]:
        regime = "Funding Stress"
    elif flags["rate_driven"] and flags["strong_usd"]:
        regime = "Rate-Driven USD Strength"
    elif flags["rate_driven"] and flags["weak_usd"]:
        regime = "Rate-Driven USD Weakness"
    elif flags["rate_driven"]:
        regime = "Rate-Driven"
    elif flags["eur_dominated"] and flags["strong_usd"]:
        regime = "EUR-Led USD Strength"
    elif flags["eur_dominated"]:
        regime = "EUR-Led Move"
    elif flags["gold_driven"]:
        regime = "Real-Rate / Gold Driven"
    elif flags["strong_usd"]:
        regime = "Broad USD Strength"
    elif flags["weak_usd"]:
        regime = "Broad USD Weakness"
    else:
        regime = "Consolidation"

    drivers = list(shares.items())[:4]

    return {
        "regime":  regime,
        "drivers": drivers,
        "flags":   flags,
        "dxy_z":   round(dz, 2),
        "vix_z":   round(vz, 2),
    }


def _eur_share(ccy_attr: Optional[pd.DataFrame], window: int = 10) -> float:
    """
    EUR's share of total absolute currency attribution over last N days.
    Requires the geometric currency attribution DataFrame (from currency_attribution()).
    Returns 0 if data unavailable.
    """
    if ccy_attr is None or "EUR" not in ccy_attr.columns:
        return 0.0
    recent = ccy_attr.tail(window).abs()
    total  = recent.sum(axis=1).sum()
    if total == 0:
        return 0.0
    return float(recent["EUR"].sum() / total)


def momentum_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Simple momentum / mean-reversion signals for each monitored asset.
    Returns a DataFrame: asset × {level, chg_1d, chg_5d, chg_20d, z60}.
    """
    assets = {
        "DXY":         "dxy",
        "EUR/USD":     "eur",
        "USD/JPY":     "jpy",
        "Gold (GC)":   "gold",
        "VIX":         "vix",
        "10Y Yield":   "ty10",
        "2Y Yield":    "ty2_use",
        "Yield Curve": "yield_curve",
        "FRA-OIS":     "fra_ois_proxy",
    }

    rows = []
    for label, col in assets.items():
        if col not in df.columns:
            continue
        s = df[col].dropna()
        if len(s) < 5:
            continue

        lvl    = s.iloc[-1]
        chg_1d = s.pct_change(1).iloc[-1] * 100  if col not in ("ty10","ty2_use","yield_curve","fra_ois_proxy","sofr_spread") else s.diff(1).iloc[-1]
        chg_5d = s.pct_change(5).iloc[-1] * 100  if col not in ("ty10","ty2_use","yield_curve","fra_ois_proxy","sofr_spread") else s.diff(5).iloc[-1]
        chg_20d= s.pct_change(20).iloc[-1] * 100 if col not in ("ty10","ty2_use","yield_curve","fra_ois_proxy","sofr_spread") else s.diff(20).iloc[-1]
        mu60   = s.tail(60).mean()
        sig60  = s.tail(60).std()
        z60    = (lvl - mu60) / sig60 if sig60 > 0 else 0.0

        rows.append({
            "Asset":   label,
            "Level":   lvl,
            "1D Chg":  chg_1d,
            "5D Chg":  chg_5d,
            "20D Chg": chg_20d,
            "Z-Score (60d)": z60,
        })

    return pd.DataFrame(rows).set_index("Asset")
