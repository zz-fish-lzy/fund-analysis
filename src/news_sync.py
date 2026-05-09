"""
新闻同步模块
从腾讯云 COS 读取 TrendRadar 新闻数据，经过滤后写入 Supabase
"""

import json
import logging
from datetime import datetime, timedelta

from src.config import (
    COS_SECRET_ID, COS_SECRET_KEY, COS_BUCKET, COS_REGION,
    NEWS_WHITELIST, NEWS_BLACKLIST
)
from src.database import FundDatabase
from src.industry_rotation import SECTOR_MAPPING

logger = logging.getLogger(__name__)

# 新闻分类 → 行业关键词映射（用于新闻-基金关联）
CATEGORY_SECTOR_MAP = {
    '新能源': ['新能源', '光伏', '锂电', '储能', '风电', '碳中和', '绿色'],
    '医药生物': ['医药', '医疗', '生物', '创新药', '疫苗', '中药'],
    '消费': ['消费', '白酒', '食品', '零售', '家电', '旅游'],
    '科技': ['科技', '半导体', '芯片', '人工智能', 'AI', '算力', '5G'],
    '金融地产': ['金融', '银行', '证券', '保险', '地产', '房地产'],
    '债券': ['债券', '国债', '利率', '降息', '降准', '加息'],
    '宏观经济': ['GDP', 'CPI', 'PMI', '央行', '货币政策', '财政', '经济'],
}


def _init_cos_client():
    """初始化 COS 客户端"""
    try:
        from qcloud_cos import CosConfig, CosS3Client
        config = CosConfig(
            Region=COS_REGION,
            SecretId=COS_SECRET_ID,
            SecretKey=COS_SECRET_KEY,
        )
        return CosS3Client(config)
    except ImportError:
        logger.error("cos-python-sdk-v5 未安装，请运行: pip install cos-python-sdk-v5")
        return None
    except Exception as e:
        logger.error(f"COS 客户端初始化失败: {e}")
        return None


def list_cos_files(client, prefix='', since_date=None):
    """
    列出 COS 桶中的文件

    Args:
        client: CosS3Client
        prefix: 文件前缀过滤
        since_date: 只返回该日期之后的文件

    Returns:
        list of {key, size, last_modified}
    """
    files = []
    marker = ''

    while True:
        try:
            response = client.list_objects(
                Bucket=COS_BUCKET,
                Prefix=prefix,
                Marker=marker,
                MaxKeys=1000
            )
        except Exception as e:
            logger.error(f"列出 COS 文件失败: {e}")
            break

        contents = response.get('Contents', [])
        if not contents:
            break

        for obj in contents:
            key = obj['Key']
            size = int(obj.get('Size', 0))
            last_modified = obj.get('LastModified', '')

            if size == 0:  # 跳过目录
                continue

            if since_date:
                try:
                    mod_date = last_modified[:10]
                    if mod_date < since_date:
                        continue
                except Exception:
                    pass

            files.append({
                'key': key,
                'size': size,
                'last_modified': last_modified
            })

        if response.get('IsTruncated') == 'true':
            marker = response.get('NextMarker', contents[-1]['Key'])
        else:
            break

    logger.info(f"COS 文件列表: {len(files)} 个文件")
    return files


def read_cos_file(client, key):
    """
    读取 COS 单个文件内容（支持 JSON/JSONL/SQLite .db）

    Returns:
        list[dict]: 解析后的新闻数据
    """
    import sqlite3
    import tempfile
    import os

    try:
        response = client.get_object(Bucket=COS_BUCKET, Key=key)
        body = response['Body'].get_raw_stream().read()

        # SQLite .db 文件
        if key.endswith('.db'):
            return _read_sqlite_db(body)

        # JSON/JSONL 文本文件
        content = body.decode('utf-8')
        if key.endswith('.json'):
            data = json.loads(content)
            if isinstance(data, list):
                return data
            return [data]
        elif key.endswith('.jsonl'):
            items = []
            for line in content.strip().split('\n'):
                if line.strip():
                    items.append(json.loads(line))
            return items
        else:
            try:
                data = json.loads(content)
                return data if isinstance(data, list) else [data]
            except json.JSONDecodeError:
                logger.warning(f"无法解析文件: {key}")
                return []

    except Exception as e:
        logger.error(f"读取 COS 文件失败 {key}: {e}")
        return []


def _fetch_article_content(url, timeout=10):
    """
    从 URL 爬取新闻正文内容

    Args:
        url: 新闻链接
        timeout: 超时秒数

    Returns:
        str: 正文内容（失败返回空字符串）
    """
    if not url:
        return ''

    try:
        import requests
        from bs4 import BeautifulSoup

        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        }
        resp = requests.get(url, headers=headers, timeout=timeout)
        resp.encoding = resp.apparent_encoding or 'utf-8'

        soup = BeautifulSoup(resp.text, 'html.parser')

        # 移除 script/style 标签
        for tag in soup(['script', 'style', 'nav', 'header', 'footer', 'aside']):
            tag.decompose()

        # 尝试常见正文容器
        content_selectors = [
            '.article-content', '.article_content', '.content-article',
            '.news-content', '.news_content', '.detail-content',
            '.art_content', '#artibody', '.post-content',
            'article', '.main-content', '.entry-content',
        ]
        for sel in content_selectors:
            el = soup.select_one(sel)
            if el and len(el.get_text(strip=True)) > 50:
                return el.get_text(separator='\n', strip=True)[:3000]

        # 回退：取 body 中最长的 <p> 文本块
        paragraphs = soup.find_all('p')
        if paragraphs:
            texts = [p.get_text(strip=True) for p in paragraphs if len(p.get_text(strip=True)) > 20]
            if texts:
                return '\n'.join(texts)[:3000]

        return ''
    except Exception as e:
        logger.debug(f"爬取正文失败 {url}: {e}")
        return ''


def _read_sqlite_db(body):
    """从 SQLite .db 二进制数据读取 news_items"""
    import sqlite3
    import tempfile
    import os

    tmp = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
    try:
        tmp.write(body)
        tmp.close()

        conn = sqlite3.connect(tmp.name)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # 检查表结构
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {row['name'] for row in cursor.fetchall()}

        if 'news_items' not in tables:
            logger.warning(f"SQLite 文件无 news_items 表，可用表: {tables}")
            conn.close()
            return []

        # 读取 news_items，关联 rank_history 获取最佳排名
        cursor.execute('''
            SELECT n.id, n.title, n.url, n.platform_id, n.rank,
                   n.first_crawl_time, n.last_crawl_time,
                   p.name as platform_name
            FROM news_items n
            LEFT JOIN platforms p ON n.platform_id = p.id
            ORDER BY n.rank ASC
        ''')

        items = []
        for row in cursor.fetchall():
            items.append({
                'title': row['title'] or '',
                'url': row['url'] or '',
                'platform': row['platform_name'] or row['platform_id'] or '',
                'rank': row['rank'],
                'first_crawl_time': row['first_crawl_time'] or '',
                'last_crawl_time': row['last_crawl_time'] or '',
                'source': 'trendradar',
            })

        conn.close()
        return items

    finally:
        try:
            os.unlink(tmp.name)
        except Exception:
            pass


def _is_relevant(news_item):
    """
    内容过滤：白名单匹配 + 黑名单排除

    Returns:
        bool
    """
    text = (news_item.get('title', '') + ' ' + news_item.get('content', '')).lower()

    if not text.strip():
        return False

    # 黑名单优先排除
    for kw in NEWS_BLACKLIST:
        if kw in text:
            return False

    # 白名单包含才保留
    for kw in NEWS_WHITELIST:
        if kw.lower() in text:
            return True

    return False


def _classify_news(news_item):
    """
    基于关键词的初步分类（DeepSeek API 会做精确分类）

    Returns:
        str: 新闻类别
    """
    text = news_item.get('title', '') + ' ' + news_item.get('content', '')

    for category, keywords in CATEGORY_SECTOR_MAP.items():
        for kw in keywords:
            if kw in text:
                return category

    return '其他'


def _associate_funds(news_item):
    """
    新闻-基金关联

    Returns:
        list of fund_code
    """
    text = news_item.get('title', '') + ' ' + news_item.get('content', '')
    associated = []

    # 方法1: 通过行业关键词 → 行业 → 基金
    for sector, keywords in CATEGORY_SECTOR_MAP.items():
        for kw in keywords:
            if kw in text:
                # 在 SECTOR_MAPPING 中找对应行业的基金
                for code, info in SECTOR_MAPPING.items():
                    if info['sector'] == sector or info['sector'] in sector:
                        if code not in associated:
                            associated.append(code)
                break  # 一个行业匹配到就够了

    # 方法2: 宏观经济新闻关联所有宽基指数
    macro_keywords = ['GDP', 'CPI', 'PMI', '央行', '降息', '降准', '加息', '货币政策']
    for kw in macro_keywords:
        if kw in text:
            for code, info in SECTOR_MAPPING.items():
                if info['sector'] == '宽基指数' and code not in associated:
                    associated.append(code)
            break

    return associated


def _normalize_news_item(raw_item, source='trendradar'):
    """
    标准化新闻数据格式

    Returns:
        dict: {title, content, source, url, news_date}
    """
    title = raw_item.get('title', raw_item.get('Title', ''))
    content = raw_item.get('content', raw_item.get('Content', raw_item.get('description', '')))
    url = raw_item.get('url', raw_item.get('Url', raw_item.get('link', '')))
    news_date = raw_item.get('date', raw_item.get('Date', raw_item.get('pubDate', '')))

    # 从 TrendRadar SQLite 格式提取日期
    if not news_date:
        crawl_time = raw_item.get('first_crawl_time', '')
        if crawl_time and len(crawl_time) >= 10:
            # 格式: "05-07" 或 "2026-05-07"
            if '-' in crawl_time:
                parts = crawl_time.split('-')
                if len(parts) == 2:
                    # "05-07" → "2026-05-07"
                    news_date = f"2026-{parts[0]}-{parts[1]}"
                else:
                    news_date = crawl_time[:10]

    if not news_date:
        news_date = datetime.now().strftime('%Y-%m-%d')
    elif len(str(news_date)) > 10:
        try:
            import pandas as pd
            news_date = pd.to_datetime(str(news_date)).strftime('%Y-%m-%d')
        except Exception:
            news_date = datetime.now().strftime('%Y-%m-%d')

    return {
        'title': str(title)[:500],
        'content': str(content)[:5000],
        'source': source,
        'url': str(url)[:500],
        'news_date': news_date,
    }


def _get_existing_titles(db, news_date=None):
    """获取已存在的新闻标题，用于全局去重（不限日期）"""
    if not db.client:
        return set()
    try:
        result = db.client.table('trend_news') \
            .select('title') \
            .execute()
        return {item['title'] for item in result.data}
    except Exception as e:
        logger.warning(f"获取已有标题失败: {e}")
        return set()


def sync_news(since_date=None, db=None):
    """
    主同步流程：COS → 过滤 → 去重 → 关联 → Supabase

    Args:
        since_date: 起始日期（默认3天前）
        db: FundDatabase 实例

    Returns:
        dict: {total_read, total_filtered, total_dedup, total_synced, total_associated}
    """
    db = db or FundDatabase()
    if not db.client:
        db.connect()

    since_date = since_date or (datetime.now() - timedelta(days=3)).strftime('%Y-%m-%d')
    logger.info(f"开始新闻同步，起始日期: {since_date}")

    # 初始化 COS 客户端
    cos_client = _init_cos_client()
    if not cos_client:
        logger.error("COS 客户端不可用，跳过同步")
        return {'total_read': 0, 'total_filtered': 0, 'total_dedup': 0, 'total_synced': 0, 'total_associated': 0}

    # 列出文件
    files = list_cos_files(cos_client, since_date=since_date)
    logger.info(f"发现 {len(files)} 个文件")

    total_read = 0
    total_filtered = 0
    total_dedup = 0
    total_synced = 0
    total_associated = 0

    # 全局缓存已有标题，避免跨日期重复
    existing_titles = _get_existing_titles(db)

    for file_info in files:
        key = file_info['key']
        logger.info(f"处理文件: {key}")

        # 读取文件
        raw_items = read_cos_file(cos_client, key)
        total_read += len(raw_items)

        for raw_item in raw_items:
            # 标准化
            news = _normalize_news_item(raw_item)

            # 内容过滤
            if not _is_relevant(news):
                total_filtered += 1
                continue

            # 内容为空时，尝试从 URL 爬取正文
            if not news['content'] and news['url']:
                logger.info(f"爬取正文: {news['title'][:40]}...")
                news['content'] = _fetch_article_content(news['url'])

            # 去重：全局按标题检查
            news_date = news['news_date']
            if news['title'] in existing_titles:
                total_dedup += 1
                continue

            # 初步分类
            category = _classify_news(news)

            # 写入 trend_news
            trend_record = {
                'news_date': news_date,
                'title': news['title'],
                'content': news['content'],
                'source': news['source'],
                'url': news['url'],
                'category': category,
                'is_labeled_by_api': False,
            }
            if db.insert_trend_news(trend_record):
                total_synced += 1
                existing_titles.add(news['title'])

            # 新闻-基金关联
            fund_codes = _associate_funds(news)
            for code in fund_codes:
                db.insert_fund_news(
                    fund_code=code,
                    news_date=news_date,
                    title=news['title'],
                    content=news['content'],
                    sentiment=0,  # 待 DeepSeek 标注
                    impact=3
                )
                total_associated += 1

    result = {
        'total_read': total_read,
        'total_filtered': total_filtered,
        'total_dedup': total_dedup,
        'total_synced': total_synced,
        'total_associated': total_associated
    }
    logger.info(f"同步完成: {result}")
    return result


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    result = sync_news()
    print(f"同步结果: {result}")
