# Dollar Volatility Attribution Model

> **美元波动归因模型** — 实时解析美元指数（DXY）的每日涨跌来自哪里，以及为什么。

---

## 项目初衷

外汇市场中，"美元今天涨了"是一个结果，但背后可以是完全不同的原因：
- 美联储加息预期推升利率差 → **利率驱动**
- 全球风险情绪恶化，资金涌入避险资产 → **避险驱动**
- 银行间融资市场紧张，美元被动走强 → **流动性紧张**
- 实际利率下行，黄金上涨，美元走弱 → **宽松预期**

这四种机制有着截然不同的交易含义和持续时间。本项目的目标是将 DXY 的波动**分解到货币层面**（谁在动）和**归因到宏观因子**（为什么动），提供一个可实时刷新的监控仪表盘。

---

## 功能概览

### 8 个分析标签页

| 标签 | 回答的问题 |
|------|-----------|
| 📈 DXY 走势 | 美元当前处于强势/弱势阶段？趋势是否延续？|
| 🌍 货币归因 | 今天的涨跌，EUR / JPY / GBP / CAD / SEK / CHF 各贡献多少？|
| 🔬 宏观因子归因 | 是利率预期？风险情绪？还是融资压力在主导？|
| 📊 收益率曲线 | 美债曲线形态如何？是否倒挂？|
| 💧 融资压力 | 银行间融资市场（SOFR / FRA-OIS）是否出现异常？|
| ⚡ VIX 分析 | 市场恐慌情绪对美元的影响有多大？|
| 📋 信号快照 | 所有指标的 Z-Score 及当前市场环境分类 |
| 🔗 相关矩阵 | 各资产之间联动关系的滚动热力图 |

### KPI 行 + 驱动判断矩阵

顶部 KPI 栏实时显示 DXY、EUR、JPY、VIX、10Y 收益率、黄金的当日表现。  
其下的**驱动判断矩阵**通过近期滚动相关性 + 方向一致性自动激活最相关的单一信号：

```
利率驱动   避险驱动   流动性紧张   宽松预期
  ●           ○          ○           ○
DXY↑+Yield↑  DXY↑+VIX↑  DXY↑+SOFR↑  DXY↓+黄金↑
```

---

## 技术原理

### 货币归因（精确几何分解）

DXY 是六种货币的几何加权指数：

```
DXY = 50.14348112 × EUR^-0.576 × JPY^0.136 × GBP^-0.119
                  × CAD^0.091  × SEK^0.042  × CHF^0.036
```

对数差分后得到严格恒等式：

```
Δln(DXY) = Σ wᵢ · Δln(FXᵢ) + 残差
```

残差极小（数据对齐误差级别），可验证分解质量。

### 宏观因子归因（滚动 OLS）

以 60 个交易日为窗口，滚动拟合：

```
Δln(DXY) ~ β₁·Δ(2Y Yield)
          + β₂·Δ(10Y−2Y 曲线斜率)
          + β₃·Δln(VIX)
          + β₄·Δln(Gold)
          + β₅·Δ(FRA-OIS 融资利差)
          + ε
```

Beta 系数随时间变化，反映宏观驱动力的结构性转换。

### 驱动判断逻辑

对每个信号计算评分：

```python
score = abs(rolling_corr)   # 当方向一致时
score = 0                   # 当方向不一致时
```

取评分最高者激活，四选一互斥。

---

## 数据来源

| 数据 | 来源 |
|------|------|
| DXY、六大货币对、黄金、VIX、国债收益率 | yfinance（免费，无需 API key）|
| SOFR、联邦基金利率、精确 2Y/10Y 收益率 | FRED（可选，需免费 API key）|

---

## 快速开始

### 1. 安装依赖

```bash
pip install dash plotly pandas numpy yfinance statsmodels python-dotenv
# 可选（精确 SOFR / 2Y 数据）
pip install fredapi
```

### 2. 配置 FRED API Key（可选）

```bash
# 在项目根目录创建 .env 文件
echo "FRED_API_KEY=your_key_here" > .env
# 免费申请：https://fredaccount.stlouisfed.org/apikey
```

### 3. 启动

```bash
python main.py
# 可选参数
python main.py --port 8080 --debug
```

浏览器打开 `http://127.0.0.1:8050`

---

## 项目结构

```
dollar_attribution/
├── config.py              # 全局参数（回看天数、DXY 权重、因子颜色）
├── main.py                # 入口：数据预取 + 启动 Dash 服务器
├── data/
│   └── fetcher.py         # yfinance + FRED 数据拉取与合并
├── model/
│   ├── attribution.py     # 货币归因 & 滚动 OLS 宏观因子归因
│   └── signals.py         # 市场环境分类（Risk-Off / Rate-Driven 等）
└── viz/
    └── dashboard.py       # Dash 应用：布局、回调、8 个标签页
```

---

## 版本历史

| 提交 | 功能 |
|------|------|
| `8ec15e9` | 初始化：货币归因 + 宏观 OLS 模型 |
| `93a9c9a` | 重构为多标签页交互式仪表盘 |
| `de8bd9c` | 支持 `.env` 自动加载 FRED API key |
| `f1f937f` | 修复 OLS 窗口上限、EUR-Led 环境判断、Python 3.13 兼容性 |
| `671aabb` | 修复 pandas 2.x `pd.read_json()` 将 JSON 字符串误作文件路径的问题 |
| `64eee29` | 将历史回看范围扩展至 3 年（800 交易日），新增 504/756 日滑块选项 |
| `9bd2cff` | 新增 KPI 行下方的美元驱动判断矩阵 |
| `2c2aa4f` | 判断矩阵改为单一最强信号激活（四选一互斥） |

---

## 配置参数

`config.py` 中可调整的主要参数：

```python
LOOKBACK_DAYS  = 800   # yfinance 拉取的历史天数（~3 交易年）
ROLLING_WINDOW = 60    # 滚动 OLS 回归窗口（交易日）
REFRESH_SECONDS = 300  # 仪表盘自动刷新间隔（秒）
```

---

## License

MIT
