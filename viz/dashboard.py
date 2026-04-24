"""
Dollar Volatility Attribution Dashboard — tabbed interactive layout.

Each chart lives in its own tab; a global lookback slider and per-tab
controls let the user drill into any time window or model parameter.
"""

import sys
import os
import io
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
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


# ── App ───────────────────────────────────────────────────────────────────────

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.CYBORG],
    title="USD Attribution Monitor",
)
app.config.suppress_callback_exceptions = True

# ── Palette ───────────────────────────────────────────────────────────────────
ACCENT   = "#00E5FF"
BG       = "#0d1117"
CARD_BG  = "#161b22"
PANEL    = "#21262d"
TEXT     = "#e6edf3"
MUTED    = "#8b949e"
BORDER   = "#30363d"
POSITIVE = "#3fb950"
NEGATIVE = "#f85149"
WARNING  = "#d29922"

LOOKBACK_OPTIONS = [30, 60, 90, 180, 252, 504, 756]

# ── Shared helpers ────────────────────────────────────────────────────────────

def _style(fig, legend=True, height=440):
    fig.update_layout(
        plot_bgcolor=CARD_BG,
        paper_bgcolor=CARD_BG,
        font=dict(color=TEXT, size=11),
        height=height,
        legend=dict(
            bgcolor="rgba(0,0,0,0)", font=dict(size=10),
            orientation="h", yanchor="bottom", y=1.02,
        ) if legend else dict(visible=False),
        margin=dict(l=48, r=20, t=44, b=36),
        xaxis=dict(gridcolor="#1f2937", showgrid=True,
                   showspikes=True, spikecolor=MUTED, spikethickness=1),
        yaxis=dict(gridcolor="#1f2937", showgrid=True),
        hovermode="x unified",
    )


def _card(*children, padding="16px"):
    return html.Div(list(children), style={
        "backgroundColor": CARD_BG,
        "border": f"1px solid {BORDER}",
        "borderRadius": "8px",
        "padding": padding,
        "marginBottom": "12px",
    })


def _label(text):
    return html.Div(text, style={
        "color": MUTED, "fontSize": "10px",
        "textTransform": "uppercase", "letterSpacing": "1px",
        "marginBottom": "4px",
    })


def _radio(id_, options, value):
    return dcc.RadioItems(
        id=id_,
        options=[{"label": f"  {o}", "value": o} for o in options],
        value=value,
        inline=True,
        inputStyle={"marginRight": "4px"},
        labelStyle={"color": TEXT, "fontSize": "12px",
                    "marginRight": "14px", "cursor": "pointer"},
    )


def _tab_body(*children):
    return html.Div(list(children), style={
        "backgroundColor": PANEL,
        "border": f"1px solid {BORDER}",
        "borderTop": "none",
        "borderRadius": "0 6px 6px 6px",
        "padding": "16px",
        "minHeight": "540px",
    })


TAB_STYLE = {
    "backgroundColor": CARD_BG,
    "color": MUTED,
    "border": f"1px solid {BORDER}",
    "borderBottom": "none",
    "borderRadius": "6px 6px 0 0",
    "padding": "8px 14px",
    "fontSize": "12px",
    "fontWeight": 500,
}
TAB_SELECTED = {
    **TAB_STYLE,
    "backgroundColor": PANEL,
    "color": ACCENT,
    "borderTop": f"2px solid {ACCENT}",
}

# ── KPI bar ───────────────────────────────────────────────────────────────────

def _kpi_cell(label, val_id, chg_id):
    return html.Div([
        html.Div(label, style={"color": MUTED, "fontSize": "10px",
                               "textTransform": "uppercase", "letterSpacing": "1px"}),
        html.Div(id=val_id, style={"color": TEXT, "fontSize": "20px",
                                   "fontWeight": 700, "fontFamily": "monospace",
                                   "lineHeight": "1.2"}),
        html.Div(id=chg_id, style={"fontSize": "11px", "fontFamily": "monospace"}),
    ], style={"padding": "4px 16px", "borderRight": f"1px solid {BORDER}"})


# ── Full layout ───────────────────────────────────────────────────────────────

app.layout = dbc.Container(fluid=True, style={
    "backgroundColor": BG, "minHeight": "100vh", "padding": "14px 18px"
}, children=[

    dcc.Interval(id="refresh", interval=REFRESH_SECONDS * 1000, n_intervals=0),
    dcc.Store(id="store-data"),

    # Header
    dbc.Row([
        dbc.Col(html.H5("💵  美元波动归因监控台",
                        style={"color": ACCENT, "margin": 0, "fontWeight": 700,
                               "letterSpacing": "1px"}), width="auto"),
    ], align="center", className="mb-2"),

    # KPI bar
    _card(
        dbc.Row([
            dbc.Col(_kpi_cell("DXY",       "kpi-dxy",  "kpi-dxy-chg"),  width="auto"),
            dbc.Col(_kpi_cell("EUR/USD",   "kpi-eur",  "kpi-eur-chg"),  width="auto"),
            dbc.Col(_kpi_cell("10Y Yield", "kpi-ty10", "kpi-ty10-chg"), width="auto"),
            dbc.Col(_kpi_cell("2Y Yield",  "kpi-ty2",  "kpi-ty2-chg"),  width="auto"),
            dbc.Col(_kpi_cell("Gold",      "kpi-gold", "kpi-gold-chg"), width="auto"),
            dbc.Col(_kpi_cell("VIX",       "kpi-vix",  "kpi-vix-chg"),  width="auto"),
            dbc.Col(html.Div(id="regime-badge", style={"paddingLeft": "12px"}),
                    width="auto"),
            dbc.Col(html.Div(id="last-update",
                             style={"color": MUTED, "fontSize": "11px",
                                    "textAlign": "right"}),
                    width=True),
        ], align="center", className="g-0"),
        padding="10px 8px",
    ),

    # Judgment panel
    _card(html.Div(id="judgment-panel"), padding="8px 16px"),

    # Global controls — lookback only; OLS window lives inside 宏观因子 tab
    _card(
        dbc.Row([
            dbc.Col([
                _label("回看周期（交易日）— 对所有图表生效"),
                dcc.Slider(
                    id="global-lookback", min=0,
                    max=len(LOOKBACK_OPTIONS) - 1, step=1, value=2,
                    marks={i: str(v) for i, v in enumerate(LOOKBACK_OPTIONS)},
                    tooltip={"always_visible": False},
                ),
            ], width=6),
            dbc.Col(html.Div(id="lookback-hint",
                             style={"color": MUTED, "fontSize": "11px",
                                    "marginTop": "18px"}),
                    width=6),
        ], align="center"),
        padding="12px 20px",
    ),

    # Tabs
    dcc.Tabs(
        id="main-tabs", value="tab-dxy",
        style={"backgroundColor": BG, "borderBottom": f"1px solid {BORDER}"},
        children=[

            # 1. DXY走势
            dcc.Tab(label="📈  DXY 走势", value="tab-dxy",
                    style=TAB_STYLE, selected_style=TAB_SELECTED,
                    children=_tab_body(
                        dbc.Row([
                            dbc.Col([_label("均线"),
                                     _radio("dxy-ma", ["MA20/60", "MA10/30", "隐藏"], "MA20/60")],
                                    width=4),
                            dbc.Col([_label("布林带（20日,2σ）"),
                                     _radio("dxy-bb", ["显示", "隐藏"], "隐藏")],
                                    width=4),
                        ], className="mb-3"),
                        dcc.Graph(id="chart-dxy"),
                    )),

            # 2. 货币归因
            dcc.Tab(label="🌍  货币归因", value="tab-ccy",
                    style=TAB_STYLE, selected_style=TAB_SELECTED,
                    children=_tab_body(
                        dbc.Row([
                            dbc.Col([_label("图表类型"),
                                     _radio("ccy-type", ["柱状图", "折线图"], "柱状图")],
                                    width=4),
                            dbc.Col([_label("叠加 DXY 实际收益"),
                                     _radio("ccy-overlay", ["显示", "隐藏"], "显示")],
                                    width=4),
                        ], className="mb-3"),
                        dcc.Graph(id="chart-ccy-attr"),
                        html.Div(id="ccy-summary", style={"marginTop": "12px"}),
                    )),

            # 3. 宏观因子
            dcc.Tab(label="🔬  宏观因子", value="tab-factor",
                    style=TAB_STYLE, selected_style=TAB_SELECTED,
                    children=_tab_body(
                        dbc.Row([
                            dbc.Col([_label("图表类型"),
                                     _radio("factor-type", ["柱状图", "折线图"], "柱状图")],
                                    width=3),
                            dbc.Col([_label("视图"),
                                     _radio("factor-view",
                                            ["归因贡献", "滚动Beta", "R²"], "归因贡献")],
                                    width=4),
                            dbc.Col([_label("OLS 滚动窗口（交易日）"),
                                     _radio("ols-window", [20, 40, 60, 90], 60)],
                                    width=4),
                        ], className="mb-2"),
                        html.Div(id="ols-warning",
                                 style={"marginBottom": "8px", "fontSize": "12px"}),
                        dcc.Graph(id="chart-factor"),
                    )),

            # 4. 收益率曲线
            dcc.Tab(label="📊  收益率曲线", value="tab-yields",
                    style=TAB_STYLE, selected_style=TAB_SELECTED,
                    children=_tab_body(
                        dbc.Row([
                            dbc.Col([_label("显示系列"),
                                     _radio("yield-series",
                                            ["2Y+10Y+曲线", "仅曲线", "仅绝对值"],
                                            "2Y+10Y+曲线")],
                                    width=6),
                        ], className="mb-3"),
                        dcc.Graph(id="chart-yields"),
                    )),

            # 5. 融资压力
            dcc.Tab(label="💧  融资压力", value="tab-funding",
                    style=TAB_STYLE, selected_style=TAB_SELECTED,
                    children=_tab_body(
                        dbc.Row([
                            dbc.Col([_label("指标"),
                                     _radio("funding-view",
                                            ["FRA-OIS代理", "SOFR利差", "全部"], "全部")],
                                    width=5),
                        ], className="mb-3"),
                        dcc.Graph(id="chart-funding"),
                        _card(html.Div(id="funding-note",
                                       style={"color": MUTED, "fontSize": "12px"}),
                              padding="8px 12px"),
                    )),

            # 6. VIX分析
            dcc.Tab(label="⚡  VIX 分析", value="tab-vix",
                    style=TAB_STYLE, selected_style=TAB_SELECTED,
                    children=_tab_body(
                        dbc.Row([
                            dbc.Col([_label("图形模式"),
                                     _radio("vix-mode",
                                            ["散点回归", "VIX时序", "Gold vs DXY"],
                                            "散点回归")],
                                    width=6),
                        ], className="mb-3"),
                        dcc.Graph(id="chart-vix"),
                    )),

            # 7. 信号快照
            dcc.Tab(label="📋  信号快照", value="tab-signals",
                    style=TAB_STYLE, selected_style=TAB_SELECTED,
                    children=_tab_body(
                        dbc.Row([
                            dbc.Col([_label("排序依据"),
                                     _radio("signal-sort",
                                            ["资产名", "Z-Score", "1D变化"], "资产名")],
                                    width=5),
                        ], className="mb-3"),
                        html.Div(id="table-signals"),
                        html.Div(id="regime-detail", style={"marginTop": "14px"}),
                    )),

            # 8. 相关矩阵
            dcc.Tab(label="🔗  相关矩阵", value="tab-corr",
                    style=TAB_STYLE, selected_style=TAB_SELECTED,
                    children=_tab_body(
                        dbc.Row([
                            dbc.Col([_label("计算方式"),
                                     _radio("corr-type",
                                            ["收益率", "水平值", "变化量"], "收益率")],
                                    width=5),
                        ], className="mb-3"),
                        dcc.Graph(id="chart-corr"),
                    )),
        ],
    ),
])


# ═══════════════════════════════════════════════════════════════════════════════
# Callbacks
# ═══════════════════════════════════════════════════════════════════════════════

@app.callback(Output("store-data", "data"), Input("refresh", "n_intervals"))
def load_data(_):
    try:
        return build_master_frame().to_json(date_format="iso", orient="split")
    except Exception:
        return None


def _load(data, lookback_idx=2):
    if not data:
        return None
    df = pd.read_json(io.StringIO(data), orient="split")
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    n = LOOKBACK_OPTIONS[int(lookback_idx)]
    return df.tail(n)


def _n(lb):
    """Return the number of days for the given lookback index."""
    return LOOKBACK_OPTIONS[int(lb)]


# ── lookback hint ─────────────────────────────────────────────────────────────

@app.callback(Output("lookback-hint", "children"), Input("global-lookback", "value"))
def update_lookback_hint(lb):
    n = _n(lb)
    return f"当前所有图表显示最近 {n} 交易日数据"


# ── KPI bar ───────────────────────────────────────────────────────────────────

@app.callback(
    [Output("kpi-dxy",  "children"), Output("kpi-dxy-chg",  "children"),
     Output("kpi-eur",  "children"), Output("kpi-eur-chg",  "children"),
     Output("kpi-ty10", "children"), Output("kpi-ty10-chg", "children"),
     Output("kpi-ty2",  "children"), Output("kpi-ty2-chg",  "children"),
     Output("kpi-gold", "children"), Output("kpi-gold-chg", "children"),
     Output("kpi-vix",  "children"), Output("kpi-vix-chg",  "children"),
     Output("regime-badge", "children"),
     Output("last-update", "children"),
     Output("judgment-panel", "children")],
    Input("store-data", "data"),
)
def update_kpis(data):
    from datetime import datetime
    if not data:
        return *["—"] * 12, "", "数据加载中…", ""

    df = _load(data, lookback_idx=len(LOOKBACK_OPTIONS) - 1)   # always use full history for KPIs

    def _kpi(col, dec=2, rate=False):
        s = df[col].dropna()
        if s.empty:
            return "—", html.Span("—")
        v, v0 = s.iloc[-1], (s.iloc[-2] if len(s) > 1 else s.iloc[-1])
        chg  = (v - v0) if rate else (v - v0) / v0 * 100
        sign = "▲" if chg >= 0 else "▼"
        c    = POSITIVE if chg >= 0 else NEGATIVE
        unit = "pp" if rate else "%"
        return f"{v:.{dec}f}", html.Span(f"{sign} {abs(chg):.{dec}f}{unit}",
                                          style={"color": c})

    dxy_v,  dxy_c  = _kpi("dxy",      2)
    eur_v,  eur_c  = _kpi("eur",      4)
    ty10_v, ty10_c = _kpi("ty10",     3, rate=True)
    ty2_v,  ty2_c  = _kpi("ty2_use",  3, rate=True)
    gold_v, gold_c = _kpi("gold",     1)
    vix_v,  vix_c  = _kpi("vix",      2)

    try:
        ccy_attr         = currency_attribution(df)
        contribs, _, _   = rolling_ols_attribution(df)
        sig = classify_regime(df, contribs, ccy_attr=ccy_attr)
        r   = sig["regime"]
        bc  = NEGATIVE if any(w in r for w in ("Stress", "Risk-Off")) else (
              POSITIVE if "Strength" in r else WARNING)
        badge = dbc.Badge(r, pill=True,
                          style={"backgroundColor": bc, "fontSize": "12px",
                                 "padding": "5px 12px"})
    except Exception:
        badge = dbc.Badge("计算中…", color="secondary", pill=True)

    judgment = _build_judgment_panel(df)

    ts = f"更新于 {datetime.now().strftime('%Y-%m-%d  %H:%M:%S')}"
    return (dxy_v, dxy_c, eur_v, eur_c, ty10_v, ty10_c,
            ty2_v, ty2_c, gold_v, gold_c, vix_v, vix_c, badge, ts, judgment)


def _build_judgment_panel(df: pd.DataFrame, corr_window: int = 20, dir_window: int = 5):
    """
    Compute 4 structural USD driver signals using recent correlations + directions.
    Returns a Dash row of compact signal cards.
    """
    def _corr(col):
        try:
            s1 = df["dxy"].pct_change().dropna()
            s2 = df[col].pct_change().dropna() if col not in ("ty10", "ty2_use", "fra_ois_proxy") \
                 else df[col].diff().dropna()
            aligned = pd.concat([s1, s2], axis=1).dropna().tail(corr_window)
            if len(aligned) < 10:
                return np.nan
            return float(aligned.iloc[:, 0].corr(aligned.iloc[:, 1]))
        except Exception:
            return np.nan

    def _dir(col):
        """Return +1 / -1 based on net 5-day change."""
        try:
            s = df[col].dropna().tail(dir_window + 1)
            return 1 if s.iloc[-1] > s.iloc[0] else -1
        except Exception:
            return 0

    dxy_dir  = _dir("dxy")
    yld_cor  = _corr("ty10")
    vix_cor  = _corr("vix")
    ois_cor  = _corr("fra_ois_proxy")
    gold_cor = _corr("gold")

    yld_dir  = _dir("ty10")
    vix_dir  = _dir("vix")
    ois_dir  = _dir("fra_ois_proxy")
    gold_dir = _dir("gold")

    # Signal: active when correlation supports the structural story AND recent direction matches
    signals = [
        {
            "label":  "利率驱动",
            "desc":   "DXY↑ + Yield↑",
            "detail": "加息预期推升美元",
            "active": dxy_dir > 0 and yld_dir > 0 and not np.isnan(yld_cor) and yld_cor > 0.2,
            "corr":   yld_cor,
            "corr_label": "DXY/10Y Corr",
        },
        {
            "label":  "避险驱动",
            "desc":   "DXY↑ + VIX↑",
            "detail": "恐慌情绪推升避险买盘",
            "active": dxy_dir > 0 and vix_dir > 0 and not np.isnan(vix_cor) and vix_cor > 0.2,
            "corr":   vix_cor,
            "corr_label": "DXY/VIX Corr",
        },
        {
            "label":  "流动性紧张",
            "desc":   "DXY↑ + SOFR↑",
            "detail": "融资压力导致美元挤压",
            "active": dxy_dir > 0 and ois_dir > 0 and not np.isnan(ois_cor) and ois_cor > 0.15,
            "corr":   ois_cor,
            "corr_label": "DXY/FRA-OIS Corr",
        },
        {
            "label":  "宽松预期",
            "desc":   "DXY↓ + 黄金↑",
            "detail": "实际利率下行，美元走弱",
            "active": dxy_dir < 0 and gold_dir > 0 and not np.isnan(gold_cor) and gold_cor < -0.2,
            "corr":   gold_cor,
            "corr_label": "DXY/Gold Corr",
        },
    ]

    def _card_signal(s):
        active = s["active"]
        border_color = POSITIVE if active else BORDER
        label_color  = POSITIVE if active else MUTED
        indicator    = "●" if active else "○"
        ind_color    = POSITIVE if active else MUTED
        corr_val     = f"{s['corr']:.2f}" if not np.isnan(s["corr"]) else "n/a"

        return html.Div([
            html.Div([
                html.Span(indicator, style={"color": ind_color, "marginRight": "5px",
                                            "fontSize": "14px"}),
                html.Span(s["label"], style={"color": label_color, "fontWeight": 700,
                                             "fontSize": "12px"}),
            ]),
            html.Div(s["desc"], style={"color": TEXT, "fontSize": "11px",
                                       "fontFamily": "monospace", "marginTop": "2px"}),
            html.Div(s["detail"], style={"color": MUTED, "fontSize": "10px",
                                         "marginTop": "1px"}),
            html.Div([
                html.Span(s["corr_label"] + ": ",
                          style={"color": MUTED, "fontSize": "10px"}),
                html.Span(corr_val,
                          style={"color": POSITIVE if (not np.isnan(s["corr"]) and
                                  ((s["corr"] > 0 and s["label"] != "宽松预期") or
                                   (s["corr"] < 0 and s["label"] == "宽松预期")))
                                 else MUTED,
                                 "fontSize": "10px", "fontFamily": "monospace"}),
            ], style={"marginTop": "3px"}),
        ], style={
            "padding": "6px 14px",
            "borderLeft": f"3px solid {border_color}",
            "borderRadius": "4px",
            "backgroundColor": "rgba(255,255,255,0.02)",
            "flex": "1",
            "minWidth": "160px",
        })

    return html.Div([
        html.Div("驱动判断矩阵", style={"color": MUTED, "fontSize": "10px",
                                         "textTransform": "uppercase",
                                         "letterSpacing": "1px", "marginBottom": "6px"}),
        html.Div([_card_signal(s) for s in signals],
                 style={"display": "flex", "gap": "10px", "flexWrap": "wrap"}),
        html.Div(f"基于近 {corr_window} 日相关性 + 近 {dir_window} 日方向",
                 style={"color": MUTED, "fontSize": "10px", "marginTop": "5px"}),
    ])


# ── Tab 1: DXY走势 ────────────────────────────────────────────────────────────

@app.callback(
    Output("chart-dxy", "figure"),
    [Input("store-data", "data"), Input("global-lookback", "value"),
     Input("dxy-ma", "value"),    Input("dxy-bb", "value")],
)
def chart_dxy(data, lb, ma_mode, bb_mode):
    df  = _load(data, lb)
    fig = go.Figure()
    if df is None:
        return fig

    s = df["dxy"].dropna()

    if bb_mode == "显示":
        roll = s.rolling(20)
        mu, sig_ = roll.mean(), roll.std()
        fig.add_trace(go.Scatter(x=s.index, y=mu + 2 * sig_, name="BB上轨",
                                 line=dict(color="#555", width=1, dash="dot"),
                                 showlegend=False))
        fig.add_trace(go.Scatter(x=s.index, y=mu - 2 * sig_, name="BB下轨",
                                 fill="tonexty", fillcolor="rgba(100,100,200,0.07)",
                                 line=dict(color="#555", width=1, dash="dot"),
                                 showlegend=False))

    fig.add_trace(go.Scatter(x=s.index, y=s, name="DXY",
                             line=dict(color=ACCENT, width=2.2)))

    if ma_mode != "隐藏":
        w1, w2 = (20, 60) if ma_mode == "MA20/60" else (10, 30)
        fig.add_trace(go.Scatter(x=s.index, y=s.rolling(w1).mean(), name=f"MA{w1}",
                                 line=dict(color="#FF9800", width=1.2, dash="dot")))
        fig.add_trace(go.Scatter(x=s.index, y=s.rolling(w2).mean(), name=f"MA{w2}",
                                 line=dict(color="#9C27B0", width=1.2, dash="dash")))

    _style(fig)
    fig.update_layout(
        title=dict(text=f"DXY 美元指数走势（最近 {_n(lb)} 交易日）",
                   font=dict(color=MUTED, size=13)),
        yaxis_title="DXY",
    )
    return fig


# ── Tab 2: 货币归因 ───────────────────────────────────────────────────────────

@app.callback(
    [Output("chart-ccy-attr", "figure"), Output("ccy-summary", "children")],
    [Input("store-data", "data"), Input("global-lookback", "value"),
     Input("ccy-type", "value"),  Input("ccy-overlay", "value")],
)
def chart_ccy(data, lb, chart_type, overlay):
    df  = _load(data, lb)
    fig = go.Figure()
    if df is None:
        return fig, ""

    attr   = currency_attribution(df)
    colors = {m["label"]: m["color"] for m in DXY_COMPONENTS.values()}
    colors["residual"] = "#607D8B"
    labels = [c for c in attr.columns if c != "residual"] + ["residual"]
    vals   = attr * 100

    if chart_type == "柱状图":
        for lbl in labels:
            fig.add_trace(go.Bar(x=vals.index, y=vals[lbl].fillna(0),
                                 name=lbl, marker_color=colors.get(lbl, "#888")))
        fig.update_layout(barmode="relative")
    else:
        for lbl in labels:
            fig.add_trace(go.Scatter(x=vals.index, y=vals[lbl].rolling(5).mean(),
                                     name=lbl, mode="lines",
                                     line=dict(color=colors.get(lbl, "#888"), width=1.5)))

    if overlay == "显示":
        dxy_lr = np.log(df["dxy"]).diff() * 100
        fig.add_trace(go.Scatter(x=dxy_lr.index, y=dxy_lr, name="Δln(DXY)",
                                 line=dict(color="white", width=1.8), mode="lines"))

    _style(fig)
    fig.update_layout(
        title=dict(text=f"货币贡献分解（几何权重，最近 {_n(lb)} 日，单位：log-pts × 100）",
                   font=dict(color=MUTED, size=13)),
        yaxis_title="贡献（log-pts × 100）",
    )

    # Recent-contribution mini-table
    recent = attr.tail(5).mean() * 100
    rows   = [html.Tr([
        html.Th("货币", style={"color": MUTED, "fontSize": "11px", "paddingRight": "20px"}),
        html.Th("5日均贡献", style={"color": MUTED, "fontSize": "11px"}),
    ])]
    for k, v in recent.sort_values(key=abs, ascending=False).items():
        c = POSITIVE if v > 0 else NEGATIVE
        rows.append(html.Tr([
            html.Td(k, style={"color": TEXT, "fontFamily": "monospace", "fontSize": "12px"}),
            html.Td(f"{v:+.3f}", style={"color": c, "fontFamily": "monospace",
                                         "fontSize": "12px"}),
        ]))
    summary = _card(
        html.Table(rows, style={"borderCollapse": "collapse"}),
        padding="10px 14px",
    )
    return fig, summary


# ── Tab 3: 宏观因子归因 ───────────────────────────────────────────────────────

@app.callback(
    [Output("chart-factor", "figure"), Output("ols-warning", "children")],
    [Input("store-data", "data"), Input("global-lookback", "value"),
     Input("ols-window", "value"), Input("factor-type", "value"),
     Input("factor-view", "value")],
)
def chart_factor(data, lb, ols_win, chart_type, view):
    df  = _load(data, lb)
    fig = go.Figure()
    if df is None:
        return fig, ""

    n_days      = _n(lb)
    requested   = int(ols_win)
    # Cap OLS window to at most half the available rows; floor at 10
    effective   = min(requested, max(n_days // 2, 10))

    if effective < requested:
        warn = html.Span(
            f"⚠️  回看周期 {n_days} 日 < OLS窗口 {requested} 日，"
            f"已自动截断至 {effective} 日。如需完整窗口请将回看周期调至 {requested*2}+ 日。",
            style={"color": WARNING},
        )
    else:
        warn = html.Span(
            f"✅  OLS窗口 {effective} 日，回看 {n_days} 日",
            style={"color": MUTED},
        )

    try:
        contribs, betas, r2 = rolling_ols_attribution(df, window=effective)
    except Exception:
        return fig, warn

    if view == "归因贡献":
        vals = contribs.fillna(0) * 100
        if chart_type == "柱状图":
            for col in vals.columns:
                fig.add_trace(go.Bar(x=vals.index, y=vals[col], name=col,
                                     marker_color=FACTOR_COLORS.get(col, "#888")))
            fig.update_layout(barmode="relative")
        else:
            for col in vals.columns:
                fig.add_trace(go.Scatter(x=vals.index,
                                         y=vals[col].rolling(5).mean(),
                                         name=col, mode="lines",
                                         line=dict(color=FACTOR_COLORS.get(col, "#888"),
                                                   width=1.5)))
        dxy_lr = np.log(df["dxy"]).diff() * 100
        fig.add_trace(go.Scatter(x=dxy_lr.index, y=dxy_lr, name="Δln(DXY)",
                                 line=dict(color="white", width=1.8)))
        fig.update_layout(
            title=dict(
                text=f"宏观因子归因（OLS {effective}日窗口，最近 {n_days} 日）",
                font=dict(color=MUTED, size=13)),
            yaxis_title="贡献（log-pts × 100）",
        )

    elif view == "滚动Beta":
        b = betas.dropna(how="all")
        for col in b.columns:
            fig.add_trace(go.Scatter(x=b.index, y=b[col].rolling(5).mean(),
                                     name=col, mode="lines",
                                     line=dict(color=FACTOR_COLORS.get(col, "#888"),
                                               width=1.5)))
        fig.add_hline(y=0, line_color="#444", line_dash="dot")
        fig.update_layout(
            title=dict(
                text=f"因子 Beta 系数（{effective}日滚动，最近 {n_days} 日）",
                font=dict(color=MUTED, size=13)),
            yaxis_title="Beta 系数",
        )

    elif view == "R²":
        r2c = r2.dropna()
        fig.add_trace(go.Scatter(x=r2c.index, y=r2c, name="R²",
                                 fill="tozeroy",
                                 fillcolor="rgba(0,229,255,0.10)",
                                 line=dict(color=ACCENT, width=1.8)))
        fig.add_hline(y=0.3, line_color=WARNING, line_dash="dash",
                      annotation_text="R²=0.30", annotation_font_color=WARNING)
        fig.update_layout(
            title=dict(
                text=f"模型拟合度 R²（{effective}日滚动，最近 {n_days} 日）",
                font=dict(color=MUTED, size=13)),
            yaxis=dict(range=[0, 1], gridcolor="#1f2937"),
        )

    _style(fig)
    return fig, warn


# ── Tab 4: 收益率曲线 ─────────────────────────────────────────────────────────

@app.callback(
    Output("chart-yields", "figure"),
    [Input("store-data", "data"), Input("global-lookback", "value"),
     Input("yield-series", "value")],
)
def chart_yields(data, lb, series):
    df  = _load(data, lb)
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    if df is None:
        return fig

    show_abs   = series in ("2Y+10Y+曲线", "仅绝对值")
    show_curve = series in ("2Y+10Y+曲线", "仅曲线")

    if show_abs:
        for col, label, color in [("ty2_use", "2Y", "#2196F3"),
                                   ("ty10",    "10Y", "#FF9800")]:
            if col in df.columns:
                s = df[col].dropna()
                fig.add_trace(go.Scatter(x=s.index, y=s, name=label,
                                         line=dict(color=color, width=1.8)),
                              secondary_y=False)

    if show_curve and "yield_curve" in df.columns:
        yc = df["yield_curve"].dropna()
        fig.add_trace(
            go.Scatter(x=yc.index, y=yc, name="10Y−2Y 曲线",
                       fill="tozeroy", fillcolor="rgba(63,185,80,0.10)",
                       line=dict(color=POSITIVE, width=2, dash="dash")),
            secondary_y=True)
        fig.add_hrect(y0=-2, y1=0, fillcolor=NEGATIVE, opacity=0.06,
                      secondary_y=True, line_width=0,
                      annotation_text="收益率倒挂", annotation_font_color=NEGATIVE,
                      annotation_position="top left")

    fig.update_yaxes(title_text="收益率 (%)", gridcolor="#1f2937",
                     color=TEXT, secondary_y=False)
    fig.update_yaxes(title_text="曲线利差 (pp)", color=TEXT, secondary_y=True)
    fig.update_xaxes(gridcolor="#1f2937")
    _style(fig)
    fig.update_layout(
        title=dict(text=f"美国国债收益率：2Y / 10Y / 曲线斜率（最近 {_n(lb)} 日）",
                   font=dict(color=MUTED, size=13)),
    )
    return fig


# ── Tab 5: 融资压力 ───────────────────────────────────────────────────────────

@app.callback(
    [Output("chart-funding", "figure"), Output("funding-note", "children")],
    [Input("store-data", "data"), Input("global-lookback", "value"),
     Input("funding-view", "value")],
)
def chart_funding(data, lb, view):
    df  = _load(data, lb)
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    if df is None:
        return fig, ""

    show_fra  = view in ("FRA-OIS代理", "全部")
    show_sofr = view in ("SOFR利差", "全部")

    if show_fra and "fra_ois_proxy" in df.columns:
        s = df["fra_ois_proxy"].dropna()
        fig.add_trace(go.Scatter(x=s.index, y=s, name="FRA-OIS代理（3M−5Y）",
                                 fill="tozeroy",
                                 fillcolor="rgba(156,39,176,0.10)",
                                 line=dict(color="#9C27B0", width=2)),
                      secondary_y=False)

    if show_sofr and "sofr" in df.columns:
        s2 = df["sofr"].dropna()
        fig.add_trace(go.Scatter(x=s2.index, y=s2, name="SOFR",
                                 line=dict(color=ACCENT, width=1.5, dash="dot")),
                      secondary_y=True)

    if show_sofr and "sofr_spread" in df.columns:
        s3 = df["sofr_spread"].dropna()
        if s3.notna().any():
            fig.add_trace(go.Bar(x=s3.index, y=s3, name="SOFR−FF利差",
                                 marker_color=NEGATIVE, opacity=0.5),
                          secondary_y=False)

    fig.update_yaxes(title_text="利差 (pp)", gridcolor="#1f2937",
                     color=TEXT, secondary_y=False)
    fig.update_yaxes(title_text="SOFR (%)", color=TEXT, secondary_y=True)
    fig.update_xaxes(gridcolor="#1f2937")
    _style(fig)
    fig.update_layout(
        title=dict(text=f"融资压力：SOFR / FRA-OIS 代理（最近 {_n(lb)} 日）",
                   font=dict(color=MUTED, size=13)),
    )

    note = ""
    if "fra_ois_proxy" in df.columns:
        v    = df["fra_ois_proxy"].dropna().iloc[-1]
        full = _load(data, 4)["fra_ois_proxy"].dropna()
        mu, sig_ = full.tail(60).mean(), full.tail(60).std()
        z = (v - mu) / sig_ if sig_ > 0 else 0
        c = NEGATIVE if z > 1.5 else (POSITIVE if z < 0 else WARNING)
        label = ("⚠️ 融资压力偏高" if z > 1.5
                 else "✅ 融资条件正常" if z < 0
                 else "⚡ 融资压力边际上升")
        note = html.Span([
            "FRA-OIS代理（最新）：",
            html.B(f"{v:.3f} pp", style={"color": c}),
            "  |  60日Z-Score：",
            html.B(f"{z:+.2f}", style={"color": c}),
            f"  |  {label}",
        ])
    return fig, note


# ── Tab 6: VIX分析 ────────────────────────────────────────────────────────────

@app.callback(
    Output("chart-vix", "figure"),
    [Input("store-data", "data"), Input("global-lookback", "value"),
     Input("vix-mode", "value")],
)
def chart_vix(data, lb, mode):
    df  = _load(data, lb)
    fig = go.Figure()
    if df is None:
        return fig

    if mode == "VIX时序":
        s = df["vix"].dropna()
        fig.add_trace(go.Scatter(x=s.index, y=s, name="VIX",
                                 fill="tozeroy",
                                 fillcolor="rgba(248,81,73,0.12)",
                                 line=dict(color=NEGATIVE, width=2)))
        for level, label in [(20, "恐慌线 20"), (30, "危机线 30")]:
            fig.add_hline(y=level, line_color=WARNING, line_dash="dash",
                          annotation_text=label, annotation_font_color=WARNING)
        fig.update_layout(yaxis_title="VIX")

    elif mode == "散点回归":
        tail    = df.dropna(subset=["vix", "dxy"])
        dxy_chg = np.log(tail["dxy"]).diff() * 100
        vix_chg = np.log(tail["vix"]).diff() * 100
        dates   = tail.index[1:]
        fig.add_trace(go.Scatter(
            x=vix_chg.iloc[1:], y=dxy_chg.iloc[1:], mode="markers",
            marker=dict(color=list(range(len(dates))), colorscale="Plasma",
                        size=6, opacity=0.75,
                        colorbar=dict(title="新旧", thickness=8)),
            text=[d.strftime("%m/%d") for d in dates],
            hovertemplate="日期：%{text}<br>ΔVIX：%{x:.2f}%<br>ΔDXY：%{y:.2f}%<extra></extra>",
            name="日度数据",
        ))
        valid = dxy_chg.iloc[1:].notna() & vix_chg.iloc[1:].notna()
        if valid.sum() > 5:
            xv, yv = vix_chg.iloc[1:][valid], dxy_chg.iloc[1:][valid]
            m, b   = np.polyfit(xv, yv, 1)
            xl     = np.linspace(xv.min(), xv.max(), 60)
            fig.add_trace(go.Scatter(x=xl, y=m * xl + b, mode="lines",
                                     line=dict(color=NEGATIVE, dash="dash", width=1.5),
                                     name=f"回归  β={m:.3f}"))
        fig.update_layout(xaxis_title="Δln(VIX) %", yaxis_title="Δln(DXY) %")

    elif mode == "Gold vs DXY":
        tail    = df.dropna(subset=["gold", "dxy"])
        dxy_chg = np.log(tail["dxy"]).diff() * 100
        gld_chg = np.log(tail["gold"]).diff() * 100
        dates   = tail.index[1:]
        fig.add_trace(go.Scatter(
            x=gld_chg.iloc[1:], y=dxy_chg.iloc[1:], mode="markers",
            marker=dict(color=list(range(len(dates))), colorscale="Viridis",
                        size=6, opacity=0.75,
                        colorbar=dict(title="新旧", thickness=8)),
            text=[d.strftime("%m/%d") for d in dates],
            hovertemplate="日期：%{text}<br>ΔGold：%{x:.2f}%<br>ΔDXY：%{y:.2f}%<extra></extra>",
            name="日度数据",
        ))
        valid = dxy_chg.iloc[1:].notna() & gld_chg.iloc[1:].notna()
        if valid.sum() > 5:
            xv, yv = gld_chg.iloc[1:][valid], dxy_chg.iloc[1:][valid]
            m, b   = np.polyfit(xv, yv, 1)
            xl     = np.linspace(xv.min(), xv.max(), 60)
            fig.add_trace(go.Scatter(x=xl, y=m * xl + b, mode="lines",
                                     line=dict(color=WARNING, dash="dash", width=1.5),
                                     name=f"回归  β={m:.3f}"))
        fig.update_layout(xaxis_title="Δln(Gold) %", yaxis_title="Δln(DXY) %")

    _style(fig)
    fig.update_layout(title=dict(
        text={
            "散点回归":    f"VIX 变化 vs DXY 变化（风险情绪驱动分析，最近 {_n(lb)} 日）",
            "VIX时序":     f"VIX 恐慌指数走势（最近 {_n(lb)} 日）",
            "Gold vs DXY": f"黄金变化 vs DXY 变化（实际利率驱动分析，最近 {_n(lb)} 日）",
        }.get(mode, ""),
        font=dict(color=MUTED, size=13),
    ))
    return fig


# ── Tab 7: 信号快照 ───────────────────────────────────────────────────────────

@app.callback(
    [Output("table-signals", "children"), Output("regime-detail", "children")],
    [Input("store-data", "data"), Input("global-lookback", "value"),
     Input("ols-window", "value"),  Input("signal-sort", "value")],
)
def table_signals(data, lb, ols_win, sort_by):
    df = _load(data, lb)
    if df is None:
        return html.Div("加载中…", style={"color": MUTED}), ""

    try:
        sig_df = momentum_signals(df).reset_index()
    except Exception:
        return html.Div("信号计算失败", style={"color": NEGATIVE}), ""

    sort_map = {"资产名": "Asset", "Z-Score": "Z-Score (60d)", "1D变化": "1D Chg"}
    sort_col = sort_map.get(sort_by, "Asset")
    if sort_col != "Asset":
        sig_df = sig_df.iloc[sig_df[sort_col].abs().argsort()[::-1].values]

    for col in ["Level", "1D Chg", "5D Chg", "20D Chg", "Z-Score (60d)"]:
        sig_df[col] = sig_df[col].apply(
            lambda v: f"{v:+.3f}" if pd.notna(v) else "—"
        )

    table = dash_table.DataTable(
        data=sig_df.to_dict("records"),
        columns=[{"name": c, "id": c} for c in sig_df.columns],
        style_table={"overflowX": "auto"},
        style_cell={
            "backgroundColor": CARD_BG, "color": TEXT,
            "border": f"1px solid {BORDER}",
            "fontFamily": "monospace", "fontSize": "12px", "padding": "6px 10px",
        },
        style_header={
            "backgroundColor": PANEL, "color": ACCENT,
            "fontWeight": "bold", "border": f"1px solid {BORDER}",
        },
        style_data_conditional=[
            {"if": {"filter_query": '{Z-Score (60d)} contains "+"',
                    "column_id": "Z-Score (60d)"}, "color": POSITIVE},
            {"if": {"filter_query": '{Z-Score (60d)} contains "-"',
                    "column_id": "Z-Score (60d)"}, "color": NEGATIVE},
            {"if": {"filter_query": '{1D Chg} contains "+"',
                    "column_id": "1D Chg"}, "color": POSITIVE},
            {"if": {"filter_query": '{1D Chg} contains "-"',
                    "column_id": "1D Chg"}, "color": NEGATIVE},
        ],
        sort_action="native",
        page_size=12,
    )

    # Regime detail panel
    try:
        full_df  = _load(data, 4)
        ccy_attr = currency_attribution(full_df)
        n_days   = _n(lb)
        eff_win  = min(int(ols_win), max(n_days // 2, 10))
        contribs, _, _ = rolling_ols_attribution(full_df, window=eff_win)
        sig      = classify_regime(full_df, contribs, ccy_attr=ccy_attr)
        shares   = factor_share(contribs, window=10)

        bc_ = NEGATIVE if any(w in sig["regime"] for w in ("Stress","Risk-Off")) else (
              POSITIVE if "Strength" in sig["regime"] else WARNING)

        driver_rows = [html.Tr([
            html.Th("驱动因子", style={"color": MUTED, "fontSize": "11px",
                                       "textAlign": "left", "paddingRight": "24px"}),
            html.Th("贡献占比", style={"color": MUTED, "fontSize": "11px",
                                       "textAlign": "right"}),
        ])]
        for k, v in list(shares.items())[:5]:
            driver_rows.append(html.Tr([
                html.Td(k, style={"color": TEXT, "fontSize": "12px",
                                  "fontFamily": "monospace", "paddingRight": "24px"}),
                html.Td(f"{v*100:.1f}%", style={"color": ACCENT, "fontSize": "12px",
                                                  "fontFamily": "monospace",
                                                  "textAlign": "right"}),
            ]))

        regime_panel = _card(
            html.Div([
                html.Span("当前市场环境：", style={"color": MUTED, "fontSize": "12px"}),
                html.Span(sig["regime"],
                          style={"color": bc_, "fontWeight": 700,
                                 "fontSize": "14px", "marginLeft": "8px"}),
                html.Span(f"  |  DXY Z={sig['dxy_z']:+.2f}  VIX Z={sig['vix_z']:+.2f}",
                          style={"color": MUTED, "fontSize": "12px"}),
            ], style={"marginBottom": "10px"}),
            html.Table(driver_rows, style={"borderCollapse": "collapse"}),
            padding="12px 16px",
        )
    except Exception:
        regime_panel = ""

    return table, regime_panel


# ── Tab 8: 相关矩阵 ───────────────────────────────────────────────────────────

@app.callback(
    Output("chart-corr", "figure"),
    [Input("store-data", "data"), Input("global-lookback", "value"),
     Input("corr-type", "value")],
)
def chart_corr(data, lb, corr_type):
    df  = _load(data, lb)
    fig = go.Figure()
    if df is None:
        return fig

    cols = {
        "DXY":      "dxy",  "EUR/USD":    "eur",  "USD/JPY":  "jpy",
        "Gold":     "gold", "VIX":        "vix",  "10Y收益率":"ty10",
        "2Y收益率": "ty2_use", "曲线斜率":"yield_curve",
        "FRA-OIS":  "fra_ois_proxy",
    }
    avail  = {k: v for k, v in cols.items() if v in df.columns}
    sub    = df[list(avail.values())].copy()
    sub.columns = list(avail.keys())

    mat = (sub.pct_change(fill_method=None) if corr_type == "收益率"
           else sub.diff() if corr_type == "变化量"
           else sub).corr().round(2)

    fig.add_trace(go.Heatmap(
        z=mat.values, x=mat.columns.tolist(), y=mat.index.tolist(),
        colorscale="RdBu", zmid=0, zmin=-1, zmax=1,
        text=mat.values, texttemplate="%{text:.2f}",
        textfont={"size": 10},
        colorbar=dict(thickness=12, len=0.85, title="ρ"),
    ))

    fig.update_layout(
        plot_bgcolor=CARD_BG, paper_bgcolor=CARD_BG,
        font=dict(color=TEXT, size=11), height=480,
        margin=dict(l=90, r=20, t=50, b=80),
        title=dict(
            text=f"滚动相关矩阵（{LOOKBACK_OPTIONS[int(lb)]}日，{corr_type}）",
            font=dict(color=MUTED, size=13),
        ),
        xaxis=dict(tickangle=-35, gridcolor="#1f2937"),
        yaxis=dict(gridcolor="#1f2937"),
    )
    return fig
