# 基金分析与预测 - 项目说明

## Python 环境

```bash
PYTHON="/c/Users/Administrator/AppData/Local/Programs/Python/Python313/python.exe"
PYTHONIOENCODING=utf-8
```

所有 Python 命令使用: `PYTHONIOENCODING=utf-8 $PYTHON -m ...`

## 常用命令

```bash
# 全市场筛选入库（支持断点续传，upsert 自动跳过已有）
$PYTHON -m src.fund_crawler screen

# 批量下载净值数据
$PYTHON -m src.fund_crawler

# 模型训练
$PYTHON -m src.train

# 新闻同步
$PYTHON -m src.news_sync

# 每日更新流水线
$PYTHON -m src.daily_update
```

## 项目结构

- `src/config.py` - 集中配置（凭证、限流、关键词）
- `src/database.py` - Supabase 数据库操作
- `src/akshare_client.py` - AKShare 限流封装
- `src/fund_crawler.py` - 基金数据获取与筛选
- `src/news_sync.py` - 新闻同步（COS → Supabase）
- `src/news_analyzer.py` - DeepSeek 情感分析
- `src/train.py` - 模型训练
- `src/daily_update.py` - 每日更新调度

## GitHub 仓库

https://github.com/zz-fish-lzy/fund-analysis
