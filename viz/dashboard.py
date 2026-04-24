"""
Dollar Volatility Attribution Dashboard — Plotly Dash application.

Panels:
  1  Header bar      — DXY level, regime badge, VIX z-score
  2  DXY price chart — with 20/60-day MAs
  3  Currency attribution — stacked bar (geometric decomposition)
  4  Macro factor attribution — rolling OLS stacked bar
  5  Rolling betas   — how each factor's sensitivity evolves
  6  Signal table    — current levels + z-scores for all monitored assets
  7  Correlation matrix — rolling 60-day pairwise correlations
  8  Scatter: VIX vs DXY change, Gold vs DXY change
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc

from config import DXY_COMPONENTS, FACTOR_COLORS, REFRESH_SECONDS
from data.fetcher import build_master_frame
from model.attribution import (
    currency_attribution,
    rolling_ols_attribution,
    factor_share,
)
from model.signals import classify_regime, momentum_signals


# ── App init ──────────────────────────────────────────────────────────────────

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.CYBORG],
    title="USD Attribution Monitor",
)
app.config.suppress_callback_exceptions = True

ACCENT   = "#00E5FF"
BG       = "#0d1117"
CARD_BG  = "#161b22"
TEXT     = "#e6edf3"
MUTED    = "#8b949e"
POSITIVE = "#3fb950"
NEGATIVE = "#f85149"
WARNING  = "#d29922"


# ── Layout helpers ────────────────────────────────────────────────────────────

def _card(children, **kwargs):
    return dbc.Card(
        dbc.CardBody(children),
        style={"backgroundColor": CARD_BG, "border": f"1px solid #30363d"},
        **kwargs,
        className="mb-3",
    )


def _kpi(label, value_id, sub_id=None):
    return html.Div([
        html.Div(label, style={"color": MUTED, "fontSize": "11px", "textTransform": "uppercase",
                               "letterSpacing": "1px"}),
        html.Div(id=value_id, style={"color": TEXT, "fontSize": "26px", "fontWeight": "700",
                                     "fontFamily": "monospace"}),
        html.Div(id=sub_id, style={"color": MUTED, "fontSize": "12px"}) if sub_id else html.Span(),
    ], style={"padding": "6px 16px"})


# ── Main layout ───────────────────────────────────────────────────────────────

app.layout = dbc.Container(fluid=True, style={"backgroundColor": BG, "minHeight": "100vh",
                                               "padding": "16px"}, children=[

    dcc.Interval(id="refresh", interval=REFRESH_SECONDS * 1000, n_intervals=0),
    dcc.Store(id="store-data"),

    # ── Header ────────────────────────────────────────────────────────────────
    dbc.Row(className="mb-3", children=[
        dbc.Col(html.H4("💵 USD Volatility Attribution Monitor",
                        style={"color": ACCENT, "margin": 0, "fontWeight": 700}), width="auto"),
        dbc.Col(html.Div(id="regime-badge"), width="auto"),
        dbc.Col(html.Div(id="last-update", style={"color": MUTED, "fontSize": "12px",
                                                    "textAlign": "right"})),
    ], align="center"),

    # ── KPI bar ───────────────────────────────────────────────────────────────
    _card(dbc.Row([
        dbc.Col(_kpi("DXY",           "kpi-dxy",    "kpi-dxy-chg"),    width=2),
        dbc.Col(_kpi("EUR/USD",        "kpi-eur",    "kpi-eur-chg"),    width=2),
        dbc.Col(_kpi("10Y Yield",      "kpi-ty10",   "kpi-ty10-chg"),   width=2),
        dbc.Col(_kpi("2Y Yield",       "kpi-ty2",    "kpi-ty2-chg"),    width=2),
        dbc.Col(_kpi("Gold (GC)",      "kpi-gold",   "kpi-gold-chg"),   width=2),
        dbc.Col(_kpi("VIX",            "kpi-vix",    "kpi-vix-chg"),    width=2),
    ])),

    # ── Row 1: DXY chart + Currency attribution ───────────────────────────────
    dbc.Row([
        dbc.Col(_card([
            html.H6("DXY — Price & Moving Averages", style={"color": MUTED}),
            dcc.Graph(id="chart-dxy", style={"height": "300px"}),
        ]), width=6),
        dbc.Col(_card([
            html.H6("Currency Attribution (Geometric Decomposition, daily log-pts)",
                    style={"color": MUTED}),
            dcc.Graph(id="chart-ccy-attr", style={"height": "300px"}),
        ]), width=6),
    ]),

    # ── Row 2: Macro factor attribution + rolling betas ───────────────────────
    dbc.Row([
        dbc.Col(_card([
            html.H6("Macro Factor Attribution — Rolling OLS (daily log-pts)",
                    style={"color": MUTED}),
            dcc.Graph(id="chart-factor-attr", style={"height": "300px"}),
        ]), width=7),
        dbc.Col(_card([
            html.H6("Rolling Factor Betas (60-day window)", style={"color": MUTED}),
            dcc.Graph(id="chart-betas", style={"height": "300px"}),
        ]), width=5),
    ]),

    # ── Row 3: FRA-OIS / SOFR + Yield curve ──────────────────────────────────
    dbc.Row([
        dbc.Col(_card([
            html.H6("SOFR / FRA-OIS Proxy & Funding Stress", style={"color": MUTED}),
            dcc.Graph(id="chart-funding", style={"height": "260px"}),
        ]), width=4),
        dbc.Col(_card([
            html.H6("Yield Curve: 2Y / 10Y", style={"color": MUTED}),
            dcc.Graph(id="chart-yields", style={"height": "260px"}),
        ]), width=4),
        dbc.Col(_card([
            html.H6("VIX vs DXY (60-day rolling)", style={"color": MUTED}),
            dcc.Graph(id="chart-scatter", style={"height": "260px"}),
        ]), width=4),
    ]),

    # ── Row 4: Signal table + Correlation matrix ──────────────────────────────
    dbc.Row([
        dbc.Col(_card([
            html.H6("Asset Snapshot — Levels & Z-Scores", style={"color": MUTED}),
            html.Div(id="table-signals"),
        ]), width=5),
        dbc.Col(_card([
            html.H6("Rolling 60-Day Correlation Matrix", style={"color": MUTED}),
            dcc.Graph(id="chart-corr", style={"height": "320px"}),
        ]), width=7),
    ]),
])


# ── Data loading callback ─────────────────────────────────────────────────────

@app.callback(Output("store-data", "data"), Input("refresh", "n_intervals"))
def load_data(_):
    """Fetch and cache all data as JSON in dcc.Store."""
    try:
        df = build_master_frame()
        # orient="split" preserves DatetimeIndex natively
        return df.to_json(date_format="iso", orient="split")
    except Exception as e:
        return None


# ── Helper: deserialise store ─────────────────────────────────────────────────

def _load(data):
    if not data:
        return None
    df = pd.read_json(data, orient="split")
    df.index = pd.to_datetime(df.index)
    return df.sort_index()


# ── KPI bar ───────────────────────────────────────────────────────────────────

@app.callback(
    [Output("kpi-dxy", "children"),    Output("kpi-dxy-chg", "children"),
     Output("kpi-eur", "children"),    Output("kpi-eur-chg", "children"),
     Output("kpi-ty10", "children"),   Output("kpi-ty10-chg", "children"),
     Output("kpi-ty2", "children"),    Output("kpi-ty2-chg", "children"),
     Output("kpi-gold", "children"),   Output("kpi-gold-chg", "children"),
     Output("kpi-vix", "children"),    Output("kpi-vix-chg", "children"),
     Output("regime-badge", "children"),
     Output("last-update", "children")],
    Input("store-data", "data"),
)
def update_kpis(data):
    df = _load(data)
    if df is None:
        empty = ["—"] * 12
        return *empty, "", "No data"

    from datetime import datetime

    def _fmt_kpi(col, decimals=2, pct=False):
        s = df[col].dropna()
        if s.empty:
            return "—", "—"
        v  = s.iloc[-1]
        v0 = s.iloc[-2] if len(s) > 1 else v
        chg = (v - v0) / v0 * 100 if pct else (v - v0)
        sign = "▲" if chg >= 0 else "▼"
        color = POSITIVE if chg >= 0 else NEGATIVE
        label = f"{v:.{decimals}f}"
        chg_label = html.Span(f"{sign} {abs(chg):.{decimals}f}{'%' if pct else ''}",
                              style={"color": color})
        return label, chg_label

    dxy_v,  dxy_c  = _fmt_kpi("dxy",      2, pct=False)
    eur_v,  eur_c  = _fmt_kpi("eur",      4, pct=False)
    ty10_v, ty10_c = _fmt_kpi("ty10",     2, pct=False)
    ty2_v,  ty2_c  = _fmt_kpi("ty2_use",  2, pct=False)
    gold_v, gold_c = _fmt_kpi("gold",     1, pct=False)
    vix_v,  vix_c  = _fmt_kpi("vix",      2, pct=False)

    # Regime badge
    try:
        ccy_attr = currency_attribution(df)
        factor_contributions, _, _ = rolling_ols_attribution(df)
        sig = classify_regime(df, factor_contributions)
        regime_txt = sig["regime"]
        badge_color = NEGATIVE if "Stress" in regime_txt or "Risk-Off" in regime_txt else (
            POSITIVE if "Strength" in regime_txt else WARNING)
        badge = dbc.Badge(regime_txt, style={"backgroundColor": badge_color,
                                              "fontSize": "13px", "padding": "6px 12px"})
    except Exception:
        badge = dbc.Badge("Computing…", color="secondary")

    ts = datetime.now().strftime("Updated %Y-%m-%d %H:%M:%S")
    return (dxy_v, dxy_c, eur_v, eur_c, ty10_v, ty10_c,
            ty2_v, ty2_c, gold_v, gold_c, vix_v, vix_c,
            badge, ts)


# ── DXY chart ─────────────────────────────────────────────────────────────────

@app.callback(Output("chart-dxy", "figure"), Input("store-data", "data"))
def chart_dxy(data):
    df = _load(data)
    fig = go.Figure()
    if df is None:
        return fig

    s = df["dxy"].dropna()
    fig.add_trace(go.Scatter(x=s.index, y=s, name="DXY",
                             line=dict(color=ACCENT, width=2)))
    fig.add_trace(go.Scatter(x=s.index, y=s.rolling(20).mean(), name="MA20",
                             line=dict(color="#FF9800", width=1, dash="dot")))
    fig.add_trace(go.Scatter(x=s.index, y=s.rolling(60).mean(), name="MA60",
                             line=dict(color="#9C27B0", width=1, dash="dash")))
    _style(fig)
    return fig


# ── Currency attribution chart ────────────────────────────────────────────────

@app.callback(Output("chart-ccy-attr", "figure"), Input("store-data", "data"))
def chart_ccy_attr(data):
    df = _load(data)
    fig = go.Figure()
    if df is None:
        return fig

    attr = currency_attribution(df).tail(90)
    colors = {m["label"]: m["color"] for m in DXY_COMPONENTS.values()}
    colors["residual"] = "#607D8B"

    # Separate positive / negative for stacked bar
    labels = [c for c in attr.columns if c != "residual"] + ["residual"]
    for label in labels:
        col = attr[label].fillna(0) * 100  # to bps / pct-pts
        fig.add_trace(go.Bar(
            x=attr.index, y=col, name=label,
            marker_color=colors.get(label, "#888"),
        ))

    # Overlay DXY log-return as line
    dxy_lr = np.log(df["dxy"]).diff().tail(90) * 100
    fig.add_trace(go.Scatter(x=dxy_lr.index, y=dxy_lr, name="Δln(DXY)",
                             line=dict(color="white", width=1.5), mode="lines"))

    fig.update_layout(barmode="relative")
    _style(fig)
    return fig


# ── Macro factor attribution chart ────────────────────────────────────────────

@app.callback(Output("chart-factor-attr", "figure"), Input("store-data", "data"))
def chart_factor_attr(data):
    df = _load(data)
    fig = go.Figure()
    if df is None:
        return fig

    try:
        contributions, _, _ = rolling_ols_attribution(df)
    except Exception:
        return fig

    attr = contributions.tail(90).fillna(0) * 100

    for col in attr.columns:
        fig.add_trace(go.Bar(
            x=attr.index, y=attr[col], name=col,
            marker_color=FACTOR_COLORS.get(col, "#888"),
        ))

    dxy_lr = np.log(df["dxy"]).diff().tail(90) * 100
    fig.add_trace(go.Scatter(x=dxy_lr.index, y=dxy_lr, name="Δln(DXY)",
                             line=dict(color="white", width=1.5), mode="lines"))

    fig.update_layout(barmode="relative")
    _style(fig)
    return fig


# ── Rolling betas chart ───────────────────────────────────────────────────────

@app.callback(Output("chart-betas", "figure"), Input("store-data", "data"))
def chart_betas(data):
    df = _load(data)
    fig = go.Figure()
    if df is None:
        return fig

    try:
        _, betas, _ = rolling_ols_attribution(df)
    except Exception:
        return fig

    betas = betas.dropna(how="all").tail(120)
    for col in betas.columns:
        fig.add_trace(go.Scatter(
            x=betas.index, y=betas[col].rolling(5).mean(),
            name=col, line=dict(color=FACTOR_COLORS.get(col, "#888"), width=1.5),
        ))

    fig.add_hline(y=0, line_color="#444", line_dash="dot")
    _style(fig)
    return fig


# ── Funding / SOFR chart ──────────────────────────────────────────────────────

@app.callback(Output("chart-funding", "figure"), Input("store-data", "data"))
def chart_funding(data):
    df = _load(data)
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    if df is None:
        return fig

    # FRA-OIS proxy
    if "fra_ois_proxy" in df.columns:
        s = df["fra_ois_proxy"].dropna()
        fig.add_trace(go.Scatter(x=s.index, y=s, name="FRA-OIS proxy",
                                 line=dict(color="#9C27B0", width=2)), secondary_y=False)

    # SOFR if available
    if "sofr" in df.columns:
        s2 = df["sofr"].dropna()
        fig.add_trace(go.Scatter(x=s2.index, y=s2, name="SOFR",
                                 line=dict(color=ACCENT, width=1.5, dash="dot")), secondary_y=True)

    # SOFR spread (overnight)
    if "sofr_spread" in df.columns:
        s3 = df["sofr_spread"].dropna()
        fig.add_trace(go.Bar(x=s3.index, y=s3, name="SOFR-FF spread",
                             marker_color="#F44336", opacity=0.5), secondary_y=False)

    fig.update_yaxes(title_text="Spread (pp)", secondary_y=False,
                     gridcolor="#1f2937", color=TEXT)
    fig.update_yaxes(title_text="SOFR (%)", secondary_y=True, color=TEXT)
    _style(fig, legend=True)
    return fig


# ── Yield curve chart ─────────────────────────────────────────────────────────

@app.callback(Output("chart-yields", "figure"), Input("store-data", "data"))
def chart_yields(data):
    df = _load(data)
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    if df is None:
        return fig

    # 2Y and 10Y
    for col, label, color in [("ty2_use", "2Y", "#2196F3"), ("ty10", "10Y", "#FF9800")]:
        if col in df.columns:
            s = df[col].dropna()
            fig.add_trace(go.Scatter(x=s.index, y=s, name=label,
                                     line=dict(color=color, width=1.5)), secondary_y=False)

    # Yield curve on secondary axis
    if "yield_curve" in df.columns:
        yc = df["yield_curve"].dropna()
        fig.add_trace(go.Scatter(x=yc.index, y=yc, name="10Y-2Y",
                                 line=dict(color="#3fb950", width=2, dash="dash")),
                      secondary_y=True)
        fig.add_hrect(y0=-0.5, y1=0, fillcolor="#f85149",
                      opacity=0.08, secondary_y=True, line_width=0)

    fig.update_yaxes(title_text="Yield (%)", secondary_y=False,
                     gridcolor="#1f2937", color=TEXT)
    fig.update_yaxes(title_text="Curve (pp)", secondary_y=True, color=TEXT)
    _style(fig, legend=True)
    return fig


# ── VIX vs DXY scatter ────────────────────────────────────────────────────────

@app.callback(Output("chart-scatter", "figure"), Input("store-data", "data"))
def chart_scatter(data):
    df = _load(data)
    fig = go.Figure()
    if df is None:
        return fig

    tail = df.tail(60).dropna(subset=["vix", "dxy"])
    dxy_chg = np.log(tail["dxy"]).diff() * 100
    vix_chg = np.log(tail["vix"]).diff() * 100
    dates   = tail.index[1:]

    fig.add_trace(go.Scatter(
        x=vix_chg.iloc[1:], y=dxy_chg.iloc[1:],
        mode="markers+text",
        marker=dict(
            color=list(range(len(dates))),
            colorscale="Viridis",
            size=7, opacity=0.8,
            colorbar=dict(title="Age", thickness=8),
        ),
        text=[d.strftime("%m/%d") for d in dates],
        textposition="top center",
        textfont=dict(size=7, color=MUTED),
        name="VIX vs DXY",
    ))

    # Trend line
    valid = dxy_chg.iloc[1:].notna() & vix_chg.iloc[1:].notna()
    if valid.sum() > 5:
        x_v = vix_chg.iloc[1:][valid]
        y_v = dxy_chg.iloc[1:][valid]
        m, b = np.polyfit(x_v, y_v, 1)
        xl = np.linspace(x_v.min(), x_v.max(), 50)
        fig.add_trace(go.Scatter(x=xl, y=m * xl + b, mode="lines",
                                 line=dict(color=NEGATIVE, dash="dash", width=1),
                                 name=f"β={m:.2f}"))

    fig.update_xaxes(title_text="Δln(VIX) %", gridcolor="#1f2937", color=TEXT)
    fig.update_yaxes(title_text="Δln(DXY) %", gridcolor="#1f2937", color=TEXT)
    _style(fig)
    return fig


# ── Signal table ──────────────────────────────────────────────────────────────

@app.callback(Output("table-signals", "children"), Input("store-data", "data"))
def table_signals(data):
    df = _load(data)
    if df is None:
        return html.Div("Loading…", style={"color": MUTED})

    try:
        sig = momentum_signals(df)
    except Exception:
        return html.Div("Error computing signals", style={"color": NEGATIVE})

    sig = sig.reset_index()
    for col in ["Level", "1D Chg", "5D Chg", "20D Chg", "Z-Score (60d)"]:
        sig[col] = sig[col].apply(lambda v: f"{v:+.2f}" if pd.notna(v) else "—")

    return dash_table.DataTable(
        data=sig.to_dict("records"),
        columns=[{"name": c, "id": c} for c in sig.columns],
        style_table={"overflowX": "auto"},
        style_cell={
            "backgroundColor": CARD_BG,
            "color": TEXT,
            "border": "1px solid #30363d",
            "fontFamily": "monospace",
            "fontSize": "12px",
            "padding": "4px 8px",
        },
        style_header={
            "backgroundColor": "#21262d",
            "color": ACCENT,
            "fontWeight": "bold",
            "border": "1px solid #30363d",
        },
        style_data_conditional=[
            {"if": {"filter_query": "{Z-Score (60d)} contains '+'",
                    "column_id": "Z-Score (60d)"},
             "color": POSITIVE},
            {"if": {"filter_query": "{Z-Score (60d)} contains '-'",
                    "column_id": "Z-Score (60d)"},
             "color": NEGATIVE},
        ],
    )


# ── Correlation matrix ────────────────────────────────────────────────────────

@app.callback(Output("chart-corr", "figure"), Input("store-data", "data"))
def chart_corr(data):
    df = _load(data)
    fig = go.Figure()
    if df is None:
        return fig

    cols = {
        "DXY":       "dxy",
        "EUR/USD":   "eur",
        "USD/JPY":   "jpy",
        "Gold":      "gold",
        "VIX":       "vix",
        "10Y Yld":   "ty10",
        "2Y Yld":    "ty2_use",
        "Yld Curve": "yield_curve",
        "FRA-OIS":   "fra_ois_proxy",
    }
    avail = {k: v for k, v in cols.items() if v in df.columns}
    ret_df = df[list(avail.values())].pct_change(fill_method=None).tail(60)
    ret_df.columns = list(avail.keys())
    corr = ret_df.corr().round(2)

    fig.add_trace(go.Heatmap(
        z=corr.values,
        x=corr.columns.tolist(),
        y=corr.index.tolist(),
        colorscale="RdBu",
        zmid=0, zmin=-1, zmax=1,
        text=corr.values.round(2),
        texttemplate="%{text}",
        textfont={"size": 10},
        colorbar=dict(thickness=10, len=0.8),
    ))

    _style(fig, legend=False)
    fig.update_layout(margin=dict(l=60, r=20, t=20, b=60))
    return fig


# ── Shared layout style ───────────────────────────────────────────────────────

def _style(fig, legend=True):
    fig.update_layout(
        plot_bgcolor=CARD_BG,
        paper_bgcolor=CARD_BG,
        font=dict(color=TEXT, size=11),
        legend=dict(
            bgcolor="rgba(0,0,0,0)", font=dict(size=10),
            orientation="h", yanchor="bottom", y=1.02,
        ) if legend else dict(visible=False),
        margin=dict(l=40, r=20, t=30, b=30),
        xaxis=dict(gridcolor="#1f2937", showgrid=True),
        yaxis=dict(gridcolor="#1f2937", showgrid=True),
    )
