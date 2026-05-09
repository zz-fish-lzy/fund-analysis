"""
新闻情感分析模块
使用 DeepSeek API 进行情感分析 + 未来本地分类器自动切换
"""

import json
import time
import random
import logging
from datetime import datetime

from openai import OpenAI

from src.config import (
    DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL,
    NEWS_CATEGORIES
)
from src.database import FundDatabase

logger = logging.getLogger(__name__)

# DeepSeek API Prompt 模板
ANALYSIS_PROMPT = """你是一个专业的金融新闻分析师，请分析以下新闻，严格按照JSON格式输出结果：
1. category：新闻分类，只能从以下选项中选一个：
   {categories}
2. sentiment：情感倾向，-1（利空）、0（中性）、1（利好）
3. impact：影响程度，1-5分（1=影响极小，5=影响极大）
4. reason：简短说明原因（不超过20字）

新闻标题：{{title}}
新闻内容：{{content}}""".format(
    categories=json.dumps(NEWS_CATEGORIES, ensure_ascii=False)
)


def _init_deepseek_client():
    """初始化 DeepSeek API 客户端"""
    if not DEEPSEEK_API_KEY:
        logger.error("DEEPSEEK_API_KEY 未配置")
        return None
    return OpenAI(
        api_key=DEEPSEEK_API_KEY,
        base_url=DEEPSEEK_BASE_URL
    )


def analyze_news_api(title, content, client=None):
    """
    用 DeepSeek API 分析单条新闻

    Args:
        title: 新闻标题
        content: 新闻内容
        client: OpenAI 客户端（可选）

    Returns:
        dict: {category, sentiment, impact, reason}
    """
    client = client or _init_deepseek_client()
    if not client:
        return {"category": "其他", "sentiment": 0, "impact": 1, "reason": "API不可用"}

    prompt = ANALYSIS_PROMPT.replace("{{title}}", title[:200]).replace("{{content}}", (content or '')[:500])

    try:
        response = client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.1,
            timeout=30
        )
        result = json.loads(response.choices[0].message.content)

        # 标准化输出
        return {
            'category': result.get('category', '其他'),
            'sentiment': int(result.get('sentiment', 0)),
            'impact': int(result.get('impact', 1)),
            'reason': str(result.get('reason', ''))[:50]
        }

    except json.JSONDecodeError as e:
        logger.warning(f"JSON 解析失败: {e}")
        return {"category": "其他", "sentiment": 0, "impact": 1, "reason": "解析失败"}
    except Exception as e:
        logger.error(f"API 调用失败: {e}")
        return {"category": "其他", "sentiment": 0, "impact": 1, "reason": "API调用失败"}


def process_unlabeled_news(limit=100, db=None):
    """
    批量处理未标注新闻

    Args:
        limit: 每批处理数量
        db: FundDatabase 实例

    Returns:
        dict: {total, success, failed}
    """
    db = db or FundDatabase()
    if not db.client:
        db.connect()

    client = _init_deepseek_client()
    if not client:
        logger.error("DeepSeek 客户端不可用")
        return {'total': 0, 'success': 0, 'failed': 0}

    # 获取未标注新闻
    news_list = db.get_unlabeled_news(limit=limit)
    total = len(news_list)
    logger.info(f"发现 {total} 条未标注新闻")

    success = 0
    failed = 0

    for i, item in enumerate(news_list, 1):
        news_id = item['id']
        title = item.get('title', '')
        content = item.get('content', '')

        logger.info(f"[{i}/{total}] 分析: {title[:30]}...")

        # 调用 API
        analysis = analyze_news_api(title, content, client)

        # 更新数据库
        if db.update_trend_news_analysis(news_id, analysis):
            success += 1
        else:
            failed += 1

        # 安全间隔（0.5-1.5秒随机）
        if i < total:
            time.sleep(random.uniform(0.5, 1.5))

    result = {'total': total, 'success': success, 'failed': failed}
    logger.info(f"情感分析完成: {result}")
    return result


def run_sentiment_pipeline(limit=100, db=None):
    """
    完整情感分析流水线入口

    1. 获取未标注新闻
    2. 调用 DeepSeek API 分析
    3. 更新 trend_news 表
    4. 关联的 fund_news 表也同步更新 sentiment

    Returns:
        dict: 处理结果统计
    """
    db = db or FundDatabase()
    if not db.client:
        db.connect()

    # 处理未标注新闻
    result = process_unlabeled_news(limit=limit, db=db)

    return result


# ── 本地分类器（阶段二/三，暂为占位） ──

def analyze_news_local(title, content):
    """
    本地分类器分析（待实现，需要积累 500+ 标注数据后训练）

    Returns:
        dict: {category, sentiment, impact, reason, sentiment_confidence, category_confidence, source}
    """
    # TODO: 阶段二实现 - FastEmbed + XGBoost
    raise NotImplementedError("本地分类器尚未实现，请使用 API 模式")


def analyze_news(title, content, db=None):
    """
    自动切换分析入口（优先本地模型，置信度低时回退 API）

    当前阶段：直接使用 API
    阶段三实现：本地优先 + API 回退
    """
    # 当前阶段直接用 API
    result = analyze_news_api(title, content)
    result['source'] = 'api'
    return result


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    # 测试单条新闻分析
    test_title = "央行宣布降准0.5个百分点 释放长期资金约1万亿元"
    test_content = "中国人民银行决定于2026年5月15日下调金融机构存款准备金率05个百分点。此次降准共计释放长期资金约1万亿元。"

    print("测试 DeepSeek API 情感分析...")
    result = analyze_news_api(test_title, test_content)
    print(f"结果: {result}")

    # 测试批量处理
    print("\n测试批量处理...")
    result = process_unlabeled_news(limit=5)
    print(f"批量结果: {result}")
