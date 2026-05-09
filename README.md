# 基金预测项目

## 项目简介

本项目使用LightGBM机器学习模型对47只基金进行涨跌预测，并使用2026年1-4月数据进行模拟实战回测。支持行业轮动分析、动态止损、市场情绪分析等功能。

## 快速开始

### 环境要求

- Python 3.10+
- 依赖包：akshare, pandas, numpy, scikit-learn, lightgbm, joblib

### 安装依赖

```bash
pip install akshare pandas numpy scikit-learn lightgbm joblib
```

### 运行程序

```bash
cd d:/StudyWork/基金分析与预测

# 方式1：交互式运行
python run_fund_analysis.py

# 方式2：一键运行
python run_fund_analysis.py  # 选择选项6
```

## 基金池（50只）

| 类型 | 数量 | 示例 |
|------|------|------|
| ETF/指数 | 20只 | 沪深300ETF、中证500ETF、创业板ETF、医药ETF、新能源ETF、纳指ETF等 |
| 混合型 | 20只 | 易方达蓝筹精选、中欧医疗健康、兴全合润、诺安成长等 |
| 债券型 | 10只 | 招商产业债、易方达增强回报、博时信用债等 |

行业覆盖：宽基指数、消费、医药、新能源、科技、金融、海外、债券

## 回测结果（2026年1-4月）

### 整体表现

| 指标 | 数值 |
|------|------|
| 回测基金数 | 47只 |
| 平均收益率 | +2.88% |
| 平均最大回撤 | -7.22% |
| 盈利基金数 | 28/47 |
| 盈利率 | 60% |

### 收益排名Top 5

| 排名 | 基金代码 | 收益率 | 最大回撤 |
|------|----------|--------|----------|
| 1 | 001156 申万菱信新能源汽车 | +51.44% | -7.41% |
| 2 | 008087 汇添富大盘核心资产 | +22.88% | -5.67% |
| 3 | 008086 汇添富中盘积极成长 | +22.31% | -6.07% |
| 4 | 515030 新能源ETF | +16.76% | -6.68% |
| 5 | 007119 景顺长城绩优成长 | +16.24% | -5.23% |

### 行业轮动建议

**建议关注新能源板块，回避消费板块**

| 排名 | 行业 | 收益率 |
|------|------|--------|
| 1 | 新能源 | +16.31% |
| 2 | 科技 | +13.04% |
| 3 | 宽基指数 | +9.04% |
| 4 | 海外 | +7.48% |
| 5 | 金融 | +3.43% |

## 项目结构

```
基金分析与预测/
├── data/                      # 数据目录
│   ├── raw/                   # 原始数据
│   │   ├── train/             # 2025年训练数据
│   │   └── test/              # 2026年1-4月测试数据
│   ├── processed/             # 处理后的数据
│   └── events/                # 事件数据（市场情绪）
├── models/                    # 模型目录
│   ├── lgbm/                  # LightGBM模型
│   └── lstm/                  # LSTM模型
├── src/                       # 源代码
│   ├── data_loader.py         # 数据加载（50只基金）
│   ├── feature_engine.py      # 特征工程（19个技术指标）
│   ├── model_lgbm.py          # LightGBM模型
│   ├── model_lstm.py          # LSTM时序模型
│   ├── ensemble_model.py      # 模型集成（LSTM + LightGBM）
│   ├── industry_rotation.py   # 行业轮动分析
│   ├── sentiment_scraper.py   # 市场情绪分析
│   ├── risk_control.py        # 风险控制（动态止损）
│   ├── backtest.py            # 基础回测
│   ├── backtest_with_risk.py  # 带风险控制的回测
│   ├── fund_crawler.py        # 基金数据爬虫
│   ├── database.py            # 数据库模块（Supabase + SQLite）
│   ├── train.py               # 训练脚本
│   └── report.py              # 报告生成
├── reports/                   # 报告目录
├── .claude/skills/            # Claude Skill
├── run_fund_analysis.py       # 快速运行脚本
└── README.md                  # 项目说明
```

## 技术栈

- **数据获取**: AKShare（免费、全面的国内基金数据API）
- **特征工程**: 19个技术指标（MA、RSI、MACD、布林带、波动率、动量等）
- **模型**: LightGBM + LSTM集成模型
- **行业轮动**: 基于行业因子的板块轮动策略
- **市场情绪**: 东方财富快讯新闻情感分析
- **风险控制**: 动态止损、仓位管理、分散化配置
- **评估指标**: 准确率、收益率、最大回撤、胜率

## 使用Claude Skill

本项目包含一个Claude Skill，可以通过以下方式触发：

- 基金预测、基金分析、基金推荐
- 基金净值、基金走势
- 投资建议、基金投资

## 已实现功能

1. **行业轮动**: 基于行业因子捕捉板块轮动机会
2. **模型集成**: LSTM + LightGBM集成预测
3. **动态止损**: 根据波动率动态调整止损线
4. **市场情绪**: 从东方财富快讯爬取新闻进行情感分析
5. **风险控制**: 止损机制、仓位管理、分散化配置
6. **数据库支持**: Supabase云端存储 + SQLite本地备选

## 注意事项

1. 本项目仅供学习和研究，不构成投资建议
2. 模型预测结果需结合人工判断
3. 历史数据不代表未来表现
4. 建议定期更新模型

## 参考资源

- [FinRL](https://github.com/AI4Finance-Foundation/FinRL) - 强化学习金融框架
- [Qlib](https://github.com/microsoft/qlib) - 微软量化投资平台
- [AKShare](https://github.com/akfamily/akshare) - 金融数据接口

## 许可证

MIT License
