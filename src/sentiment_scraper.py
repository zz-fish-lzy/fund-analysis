"""
市场情绪分析模块
从东方财富快讯爬取财经新闻，分析市场情绪
"""

import asyncio
import json
import os
import pandas as pd
from datetime import datetime, timedelta
from collections import Counter


# 情感关键词库
POSITIVE_KEYWORDS = [
    '利好', '上涨', '突破', '反弹', '回升', '新高', '增长', '利多',
    '看好', '加仓', '增持', '强势', '放量', '突破', '领涨', '涨停',
    '超预期', '景气', '复苏', '扩张', '盈利', '分红', '回购'
]

NEGATIVE_KEYWORDS = [
    '利空', '下跌', '暴跌', '回调', '风险', '减持', '跌破', '利空',
    '看空', '减仓', '清仓', '弱势', '缩量', '破位', '领跌', '跌停',
    '不及预期', '衰退', '萎缩', '亏损', '违约', '暴雷', '清盘'
]

# 行业板块关键词
SECTOR_KEYWORDS = {
    '消费': ['白酒', '消费', '食品', '饮料', '零售', '家电', '纺织'],
    '医药': ['医药', '医疗', '生物', '疫苗', '创新药', '中药', '器械'],
    '科技': ['科技', '芯片', '半导体', '人工智能', 'AI', '5G', '通信', '软件'],
    '新能源': ['新能源', '光伏', '锂电', '电池', '风电', '储能', '碳中和'],
    '金融': ['银行', '证券', '保险', '金融', '券商', '信托'],
    '地产': ['地产', '房产', '楼市', '房价', '土地', '物业'],
    '制造': ['制造', '机械', '汽车', '军工', '航空', '船舶'],
    '周期': ['钢铁', '煤炭', '有色', '化工', '石油', '黄金', '铜', '铝']
}


class SentimentAnalyzer:
    """市场情绪分析器"""

    def __init__(self):
        self.positive_keywords = POSITIVE_KEYWORDS
        self.negative_keywords = NEGATIVE_KEYWORDS
        self.sector_keywords = SECTOR_KEYWORDS

    def analyze_text(self, text: str) -> dict:
        """
        分析单条文本的情感

        Args:
            text: 新闻文本

        Returns:
            dict: {'score': float, 'positive': int, 'negative': int, 'sectors': list}
        """
        text = text.lower()

        # 统计正面和负面关键词
        positive_count = sum(1 for kw in self.positive_keywords if kw in text)
        negative_count = sum(1 for kw in self.negative_keywords if kw in text)

        # 识别涉及的行业
        sectors = []
        for sector, keywords in self.sector_keywords.items():
            if any(kw in text for kw in keywords):
                sectors.append(sector)

        # 计算情感得分 (-1 到 1)
        total = positive_count + negative_count
        if total == 0:
            score = 0
        else:
            score = (positive_count - negative_count) / total

        return {
            'score': score,
            'positive': positive_count,
            'negative': negative_count,
            'sectors': sectors
        }

    def analyze_news_batch(self, news_list: list) -> dict:
        """
        批量分析新闻情感

        Args:
            news_list: 新闻列表 [{'text': str, 'time': str}, ...]

        Returns:
            dict: 综合情感分析结果
        """
        if not news_list:
            return {
                'overall_score': 0,
                'positive_ratio': 0,
                'negative_ratio': 0,
                'news_count': 0,
                'sector_sentiment': {}
            }

        results = [self.analyze_text(news.get('text', '')) for news in news_list]

        # 综合得分
        scores = [r['score'] for r in results]
        overall_score = sum(scores) / len(scores)

        # 正负面比例
        positive_news = sum(1 for r in results if r['score'] > 0)
        negative_news = sum(1 for r in results if r['score'] < 0)
        total = len(results)

        # 行业情感
        sector_scores = {}
        for r in results:
            for sector in r['sectors']:
                if sector not in sector_scores:
                    sector_scores[sector] = []
                sector_scores[sector].append(r['score'])

        sector_sentiment = {
            sector: sum(scores) / len(scores)
            for sector, scores in sector_scores.items()
            if scores
        }

        return {
            'overall_score': overall_score,
            'positive_ratio': positive_news / total,
            'negative_ratio': negative_news / total,
            'neutral_ratio': (total - positive_news - negative_news) / total,
            'news_count': total,
            'sector_sentiment': sector_sentiment
        }


class EastMoneyNewsScraper:
    """东方财富快讯爬虫"""

    def __init__(self, save_dir='data/events'):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.analyzer = SentimentAnalyzer()

    async def scrape_kuaixun(self, max_pages=5) -> list:
        """
        爬取东方财富基金快讯

        Args:
            max_pages: 最大爬取页数

        Returns:
            list: 新闻列表
        """
        try:
            from playwright.async_api import async_playwright
        except ImportError:
            print("请安装playwright: pip install playwright")
            return []

        news_list = []

        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context()
            page = await context.new_page()

            for page_num in range(1, max_pages + 1):
                try:
                    url = f"https://kuaixun.eastmoney.com/jj_{page_num}.html"
                    await page.goto(url, wait_until='domcontentloaded', timeout=15000)
                    await page.wait_for_timeout(2000)

                    # 提取新闻标题和内容
                    items = await page.query_selector_all('.news_item, .artList li, .list_item, article')

                    if not items:
                        # 尝试其他选择器
                        items = await page.query_selector_all('h2 a, .title a, .news-title a')

                    for item in items:
                        try:
                            text = await item.inner_text()
                            if text and len(text.strip()) > 10:
                                news_list.append({
                                    'text': text.strip(),
                                    'time': datetime.now().strftime('%Y-%m-%d %H:%M'),
                                    'source': 'eastmoney_kuaixun'
                                })
                        except:
                            continue

                    print(f"  爬取第{page_num}页完成，累计{len(news_list)}条")

                    # 避免请求过快
                    await asyncio.sleep(1)

                except Exception as e:
                    print(f"  爬取第{page_num}页失败: {e}")
                    continue

            await browser.close()

        return news_list

    def analyze_market_sentiment(self, news_list: list, save_date: str = None) -> dict:
        """
        分析市场情绪并保存

        Args:
            news_list: 新闻列表
            save_date: 保存日期

        Returns:
            dict: 情感分析结果
        """
        if not save_date:
            save_date = datetime.now().strftime('%Y-%m-%d')

        # 分析情感
        result = self.analyzer.analyze_news_batch(news_list)
        result['date'] = save_date

        # 保存结果
        filename = f"{self.save_dir}/sentiment_{save_date}.json"
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        print(f"情感分析结果已保存: {filename}")
        return result

    def get_sentiment_signal(self, sentiment_result: dict) -> dict:
        """
        根据情感分析生成交易信号

        Args:
            sentiment_result: 情感分析结果

        Returns:
            dict: 交易信号
        """
        score = sentiment_result.get('overall_score', 0)
        positive_ratio = sentiment_result.get('positive_ratio', 0)
        negative_ratio = sentiment_result.get('negative_ratio', 0)

        if score > 0.3:
            signal = 'bullish'
            action = '可适当加仓'
            confidence = min(score * 2, 1.0)
        elif score < -0.3:
            signal = 'bearish'
            action = '建议减仓或观望'
            confidence = min(abs(score) * 2, 1.0)
        else:
            signal = 'neutral'
            action = '保持现有仓位'
            confidence = 0.5

        # 获取行业信号
        sector_sentiment = sentiment_result.get('sector_sentiment', {})
        strong_sectors = [s for s, v in sector_sentiment.items() if v > 0.2]
        weak_sectors = [s for s, v in sector_sentiment.items() if v < -0.2]

        return {
            'signal': signal,
            'action': action,
            'confidence': confidence,
            'overall_score': score,
            'strong_sectors': strong_sectors,
            'weak_sectors': weak_sectors,
            'news_count': sentiment_result.get('news_count', 0)
        }


def load_historical_sentiment(save_dir='data/events') -> pd.DataFrame:
    """
    加载历史情感数据

    Returns:
        DataFrame: 历史情感数据
    """
    files = [f for f in os.listdir(save_dir) if f.startswith('sentiment_') and f.endswith('.json')]

    if not files:
        return pd.DataFrame()

    records = []
    for f in files:
        try:
            with open(os.path.join(save_dir, f), 'r', encoding='utf-8') as fp:
                data = json.load(fp)
                records.append({
                    'date': data.get('date', ''),
                    'overall_score': data.get('overall_score', 0),
                    'positive_ratio': data.get('positive_ratio', 0),
                    'negative_ratio': data.get('negative_ratio', 0),
                    'news_count': data.get('news_count', 0)
                })
        except:
            continue

    if records:
        df = pd.DataFrame(records)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
        return df
    return pd.DataFrame()


async def main():
    """主函数"""
    print("=" * 60)
    print("市场情绪分析 - 东方财富快讯")
    print("=" * 60)

    scraper = EastMoneyNewsScraper()

    # 爬取新闻
    print("\n正在爬取财经快讯...")
    news_list = await scraper.scrape_kuaixun(max_pages=3)

    if news_list:
        print(f"\n爬取完成，共{len(news_list)}条新闻")

        # 分析情感
        result = scraper.analyze_market_sentiment(news_list)

        # 生成信号
        signal = scraper.get_sentiment_signal(result)

        print("\n" + "=" * 60)
        print("市场情绪分析结果")
        print("=" * 60)
        print(f"  情感得分: {result['overall_score']:.4f}")
        print(f"  正面比例: {result['positive_ratio']:.2%}")
        print(f"  负面比例: {result['negative_ratio']:.2%}")
        print(f"  新闻数量: {result['news_count']}")

        print(f"\n交易信号: {signal['signal']}")
        print(f"建议操作: {signal['action']}")
        print(f"信号置信度: {signal['confidence']:.2%}")

        if signal['strong_sectors']:
            print(f"强势行业: {', '.join(signal['strong_sectors'])}")
        if signal['weak_sectors']:
            print(f"弱势行业: {', '.join(signal['weak_sectors'])}")
    else:
        print("未获取到新闻数据")


if __name__ == '__main__':
    asyncio.run(main())
