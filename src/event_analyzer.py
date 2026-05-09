"""
事件分析模块
分析财经新闻和宏观经济事件
"""

import re
from datetime import datetime


# 关键词库
POSITIVE_KEYWORDS = [
    '利好', '上涨', '突破', '新高', '增长', '盈利', '降准', '降息',
    '刺激', '扶持', '改革', '创新', '突破', '反弹', '企稳', '回暖'
]

NEGATIVE_KEYWORDS = [
    '利空', '下跌', '暴跌', '风险', '危机', '亏损', '加息', '收紧',
    '监管', '处罚', '违约', '爆雷', '衰退', '通胀', '滞涨', '恐慌'
]

NEUTRAL_KEYWORDS = [
    '震荡', '盘整', '观望', '持平', '稳定', '平稳', '窄幅', '整理'
]


def analyze_news_sentiment(news_text):
    """
    分析新闻情感

    Args:
        news_text: 新闻文本

    Returns:
        dict: 情感分析结果
    """
    # 统计关键词出现次数
    pos_count = sum(1 for word in POSITIVE_KEYWORDS if word in news_text)
    neg_count = sum(1 for word in NEGATIVE_KEYWORDS if word in news_text)
    neu_count = sum(1 for word in NEUTRAL_KEYWORDS if word in news_text)

    total = pos_count + neg_count + neu_count

    if total == 0:
        return {
            'sentiment': 'neutral',
            'score': 0.5,
            'positive': 0,
            'negative': 0,
            'neutral': 0
        }

    # 计算情感得分 (0-1, 越高越正面)
    score = (pos_count + neu_count * 0.5) / total

    # 确定情感类别
    if score > 0.6:
        sentiment = 'positive'
    elif score < 0.4:
        sentiment = 'negative'
    else:
        sentiment = 'neutral'

    return {
        'sentiment': sentiment,
        'score': score,
        'positive': pos_count,
        'negative': neg_count,
        'neutral': neu_count
    }


def extract_event_tags(news_text):
    """
    提取事件标签

    Args:
        news_text: 新闻文本

    Returns:
        list: 事件标签列表
    """
    tags = []

    # 政策事件
    policy_keywords = ['央行', '国务院', '发改委', '财政部', '证监会', '银保监会']
    if any(word in news_text for word in policy_keywords):
        tags.append('政策')

    # 宏观经济
    macro_keywords = ['GDP', 'CPI', 'PMI', 'M2', '利率', '汇率', '通胀']
    if any(word in news_text for word in macro_keywords):
        tags.append('宏观')

    # 行业事件
    industry_keywords = ['科技', '医药', '消费', '金融', '地产', '新能源']
    if any(word in news_text for word in industry_keywords):
        tags.append('行业')

    # 市场事件
    market_keywords = ['A股', '港股', '美股', '大盘', '指数', '成交量']
    if any(word in news_text for word in market_keywords):
        tags.append('市场')

    return tags


def analyze_events_batch(news_list):
    """
    批量分析事件

    Args:
        news_list: 新闻列表 [{'date': ..., 'text': ...}, ...]

    Returns:
        DataFrame: 分析结果
    """
    import pandas as pd

    results = []

    for news in news_list:
        sentiment = analyze_news_sentiment(news['text'])
        tags = extract_event_tags(news['text'])

        results.append({
            'date': news['date'],
            'text': news['text'][:100],  # 截取前100字
            'sentiment': sentiment['sentiment'],
            'sentiment_score': sentiment['score'],
            'tags': ','.join(tags) if tags else '无'
        })

    return pd.DataFrame(results)


def get_market_sentiment(events_df, date):
    """
    获取某日市场情绪

    Args:
        events_df: 事件DataFrame
        date: 日期

    Returns:
        dict: 市场情绪
    """
    # 筛选当日事件
    daily_events = events_df[events_df['date'] == date]

    if daily_events.empty:
        return {
            'sentiment_score': 0.5,
            'event_count': 0,
            'tags': []
        }

    # 计算平均情感得分
    avg_score = daily_events['sentiment_score'].mean()

    # 统计事件标签
    all_tags = []
    for tags in daily_events['tags']:
        if tags != '无':
            all_tags.extend(tags.split(','))

    return {
        'sentiment_score': avg_score,
        'event_count': len(daily_events),
        'tags': list(set(all_tags))
    }


# 使用SnowNLP进行更精确的情感分析（可选）
def analyze_with_snownlp(text):
    """
    使用SnowNLP分析情感

    Args:
        text: 文本

    Returns:
        float: 情感得分 (0-1)
    """
    try:
        from snownlp import SnowNLP
        s = SnowNLP(text)
        return s.sentiments
    except ImportError:
        # 如果SnowNLP未安装，使用关键词方法
        result = analyze_news_sentiment(text)
        return result['score']


if __name__ == '__main__':
    # 测试事件分析
    test_news = [
        {'date': '2026-01-15', 'text': '央行宣布降准0.5个百分点，释放流动性，利好A股市场'},
        {'date': '2026-01-15', 'text': '科技股大幅下跌，市场恐慌情绪蔓延'},
        {'date': '2026-01-16', 'text': '国务院发布新能源扶持政策，相关板块上涨'},
        {'date': '2026-01-16', 'text': 'PMI数据不及预期，经济复苏面临挑战'},
    ]

    print("=" * 60)
    print("事件分析测试")
    print("=" * 60)

    # 批量分析
    results = analyze_events_batch(test_news)
    print("\n分析结果:")
    print(results.to_string())

    # 获取市场情绪
    sentiment = get_market_sentiment(results, '2026-01-15')
    print(f"\n2026-01-15 市场情绪: {sentiment}")
