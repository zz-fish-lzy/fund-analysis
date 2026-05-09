"""
基金数据获取模块
从 AKShare 获取基金数据，支持动态基金池和批量处理
"""

import asyncio
import csv
import os
import logging
import pandas as pd
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.akshare_client import get_fund_nav, get_fund_detail, batch_get_fund_nav, batch_get_fund_nav_fast
from src.config import AKSHARE_BATCH_SIZE, AKSHARE_BATCH_REST

logger = logging.getLogger(__name__)

# 默认基金池文件路径
DEFAULT_POOL_PATH = 'data/fund_pool.csv'


def load_fund_pool(pool_path=None):
    """
    从 CSV 文件加载基金池

    支持多种 CSV 格式，自动检测列名:
    - fund_code/fund_name
    - code/name
    - 基金代码/基金名称
    - 第一列/第二列（兜底）

    Returns:
        dict: {fund_code: fund_name}
    """
    pool_path = pool_path or DEFAULT_POOL_PATH
    if not os.path.exists(pool_path):
        logger.warning(f"基金池文件不存在: {pool_path}，使用默认基金池")
        return get_default_fund_pool()

    try:
        df = pd.read_csv(pool_path, dtype=str)

        # 自动检测列名
        code_col = None
        name_col = None
        for col in df.columns:
            col_lower = col.lower().strip()
            if col_lower in ('fund_code', 'code', '基金代码', '代码'):
                code_col = col
            elif col_lower in ('fund_name', 'name', '基金名称', '名称', '基金简称'):
                name_col = col

        if code_col is None:
            # 假设第一列是代码，第二列是名称
            code_col = df.columns[0]
            name_col = df.columns[1] if len(df.columns) > 1 else None

        pool = {}
        for _, row in df.iterrows():
            code = str(row[code_col]).strip()
            name = str(row[name_col]).strip() if name_col else code
            if code and code != 'nan':
                pool[code] = name

        logger.info(f"从 {pool_path} 加载 {len(pool)} 只基金")
        return pool
    except Exception as e:
        logger.error(f"加载基金池失败: {e}")
        return get_default_fund_pool()


def get_default_fund_pool():
    """默认基金池（49只代表性基金）"""
    return {
        # 指数基金/ETF
        '510300': '沪深300ETF华泰柏瑞',
        '510500': '中证500ETF南方',
        '159915': '创业板ETF易方达',
        '510050': '上证50ETF华夏',
        '512100': '中证1000ETF南方',
        '512010': '医药ETF',
        '515030': '新能源ETF',
        '512690': '酒ETF',
        '512880': '证券ETF',
        '159941': '纳指ETF',
        '513100': '纳指ETF国泰',
        '513050': '中概互联ETF',
        '159920': '恒生ETF',
        # 混合型基金
        '005827': '易方达蓝筹精选混合',
        '003095': '中欧医疗健康混合A',
        '163406': '兴全合润混合A',
        '320007': '诺安成长混合A',
        '001938': '中欧时代先锋股票A',
        '005911': '广发双擎升级混合A',
        '110011': '易方达中小盘混合',
        '000831': '工银瑞信前沿医疗',
        '007119': '景顺长城绩优成长混合',
        '009088': '易方达优质精选混合',
        '010223': '富国天惠成长混合',
        # 债券基金
        '217022': '招商产业债A',
        '110017': '易方达增强回报债券A',
        '050011': '博时信用债券A/B',
    }


def save_fund_pool(pool, pool_path=None):
    """保存基金池到 CSV"""
    pool_path = pool_path or DEFAULT_POOL_PATH
    os.makedirs(os.path.dirname(pool_path), exist_ok=True)

    with open(pool_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(['fund_code', 'fund_name'])
        for code, name in pool.items():
            writer.writerow([code, name])

    logger.info(f"基金池已保存: {pool_path} ({len(pool)} 只)")


def fetch_fund_data(fund_code, start_date=None, end_date=None):
    """
    获取单只基金数据（净值 + 详情）

    Returns:
        dict: {'nav': DataFrame, 'detail': dict} 或 None
    """
    try:
        nav_df = get_fund_nav(fund_code, start_date, end_date)
        detail = get_fund_detail(fund_code)

        return {
            'nav': nav_df,
            'detail': detail,
        }
    except Exception as e:
        logger.error(f"获取 {fund_code} 失败: {e}")
        return None


def batch_fetch_funds(fund_pool=None, start_date=None, end_date=None,
                      save_dir='data/raw', batch_size=None, rest_seconds=None, fast=False):
    """
    批量获取基金数据

    Args:
        fund_pool: 基金池 dict {code: name}，默认从 CSV 加载
        start_date: 开始日期
        end_date: 结束日期
        save_dir: 保存目录
        batch_size: 每批数量（默认从配置读取）
        rest_seconds: 批间休息秒数（默认从配置读取）
        fast: 是否使用快速模式（多线程+低限流，仅限非交易时段）

    Returns:
        dict: {fund_code: DataFrame}
    """
    fund_pool = fund_pool or load_fund_pool()
    batch_size = batch_size or AKSHARE_BATCH_SIZE
    rest_seconds = rest_seconds or AKSHARE_BATCH_REST
    os.makedirs(save_dir, exist_ok=True)

    codes = list(fund_pool.keys())
    logger.info(f"开始批量获取: {len(codes)} 只基金, 快速模式={fast}")

    if fast:
        nav_dict = batch_get_fund_nav_fast(codes, start_date, end_date)
    else:
        nav_dict = batch_get_fund_nav(codes, start_date, end_date,
                                       batch_size=batch_size,
                                       rest_seconds=rest_seconds)

    # 保存到文件
    saved = 0
    for code, df in nav_dict.items():
        if not df.empty:
            filename = os.path.join(save_dir, f'fund_{code}.csv')
            df.to_csv(filename, index=False, encoding='utf-8-sig')
            saved += 1

    logger.info(f"批量获取完成: {saved}/{len(codes)} 只基金已保存到 {save_dir}")
    return nav_dict


def search_funds(keyword, top_n=20):
    """
    按关键词搜索基金

    Args:
        keyword: 搜索关键词（名称或代码）
        top_n: 返回数量

    Returns:
        list: [{'code': str, 'name': str, 'type': str}]
    """
    try:
        import akshare as ak
        df = ak.fund_name_em()

        # 搜索匹配
        mask = df['基金简称'].str.contains(keyword, na=False) | \
               df['基金代码'].str.contains(keyword, na=False)
        results = df[mask].head(top_n)

        return [
            {
                'code': row['基金代码'],
                'name': row['基金简称'],
                'type': row.get('基金类型', ''),
            }
            for _, row in results.iterrows()
        ]
    except Exception as e:
        logger.error(f"搜索基金失败: {e}")
        return []


def add_funds_to_pool(codes, pool_path=None):
    """
    添加基金到基金池

    Args:
        codes: 基金代码列表
        pool_path: 基金池文件路径
    """
    pool = load_fund_pool(pool_path)

    for code in codes:
        if code not in pool:
            try:
                detail = get_fund_detail(code)
                name = detail.get('fund_name', code)
                pool[code] = name
                logger.info(f"添加: {code} - {name}")
            except Exception as e:
                logger.error(f"获取 {code} 信息失败: {e}")
                pool[code] = code

    save_fund_pool(pool, pool_path)
    return pool


# ── 快速批量入库（实施方案：硬指标初筛 → 动态评分 → 分层存储） ──

def quick_screen_and_store(db=None):
    """
    快速全市场筛选并入库

    流程：
    1. 一次 API 获取全市场基金列表（10000+ 只）
    2. 本地硬指标初筛（排除货币/理财/规模小/太新）
    3. 通过筛选的基金写入 funds 表
    4. 返回通过的基金代码列表

    Returns:
        list[dict]: 通过筛选的基金列表
    """
    from src.akshare_client import quick_screen_funds, get_all_fund_daily_nav
    from src.database import FundDatabase

    db = db or FundDatabase()
    if not db.client:
        db.connect()

    # 第一步：全市场初筛（一次 API 调用）
    logger.info("=" * 60)
    logger.info("第一步：全市场硬指标初筛")
    logger.info("=" * 60)
    screened = quick_screen_funds()
    logger.info(f"初筛通过: {len(screened)} 只基金")

    # 第二步：获取全市场当日净值（一次 API 调用）
    logger.info("获取全市场当日净值...")
    try:
        daily_nav = get_all_fund_daily_nav()
        logger.info(f"获取到 {len(daily_nav)} 只基金净值")
    except Exception as e:
        logger.warning(f"获取批量净值失败: {e}，跳过净值关联")
        daily_nav = pd.DataFrame()

    # 第三步：写入 funds 表
    logger.info("写入 funds 表...")
    stored = 0
    for fund in screened:
        code = fund['code']
        name = fund['name']
        fund_type = fund['type']

        fund_data = {
            'fund_code': code,
            'fund_name': name,
            'fund_type': fund_type,
        }
        if db.upsert_fund(fund_data):
            stored += 1

    logger.info(f"入库完成: {stored}/{len(screened)} 只基金")

    return screened


def batch_update_nav_from_daily(fund_pool=None, db=None):
    """
    用全市场批量接口更新当日净值（一次 API 调用更新所有基金）

    比逐只请求快 100 倍以上。

    Returns:
        int: 更新的基金数量
    """
    from src.akshare_client import get_all_fund_daily_nav
    from src.database import FundDatabase

    db = db or FundDatabase()
    if not db.client:
        db.connect()

    fund_pool = fund_pool or load_fund_pool()

    logger.info("获取全市场当日净值...")
    try:
        daily_nav = get_all_fund_daily_nav()
    except Exception as e:
        logger.error(f"获取批量净值失败: {e}")
        return 0

    logger.info(f"获取到 {len(daily_nav)} 只基金净值，开始匹配基金池...")

    today = datetime.now().strftime('%Y-%m-%d')
    updated = 0

    for code in fund_pool:
        nav_row = daily_nav[daily_nav['基金代码'] == code]
        if nav_row.empty:
            continue

        row = nav_row.iloc[0]
        try:
            nav_val = float(row.get('单位净值', 0) or 0)
            if nav_val <= 0:
                continue

            # 写入 fund_nav 表
            import pandas as pd
            nav_df = pd.DataFrame([{
                'date': pd.to_datetime(today),
                'nav': nav_val,
                'daily_return': float(str(row.get('日增长率', '0')).replace('%', '') or 0),
            }])
            db.insert_nav_data(code, nav_df)
            updated += 1
        except Exception as e:
            logger.debug(f"  {code}: 更新失败 - {e}")

    logger.info(f"当日净值更新: {updated}/{len(fund_pool)} 只基金")
    return updated


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

    import sys

    if len(sys.argv) > 1 and sys.argv[1] == 'search':
        keyword = sys.argv[2] if len(sys.argv) > 2 else '沪深300'
        results = search_funds(keyword)
        print(f"搜索 '{keyword}' 结果:")
        for r in results:
            print(f"  {r['code']} - {r['name']} ({r['type']})")

    elif len(sys.argv) > 1 and sys.argv[1] == 'add':
        codes = sys.argv[2:]
        add_funds_to_pool(codes)

    elif len(sys.argv) > 1 and sys.argv[1] == 'screen':
        # 全市场筛选并入库
        quick_screen_and_store()

    elif len(sys.argv) > 1 and sys.argv[1] == 'update-nav':
        # 批量更新当日净值
        batch_update_nav_from_daily()

    elif len(sys.argv) > 1 and sys.argv[1] == 'fast':
        # 快速模式：多线程 + 低限流（仅限周末/非交易时段）
        pool = load_fund_pool()
        print(f"快速模式: {len(pool)} 只基金")
        batch_fetch_funds(pool, start_date='2024-01-01', fast=True)

    else:
        # 默认：批量下载
        pool = load_fund_pool()
        print(f"基金池: {len(pool)} 只基金")
        batch_fetch_funds(pool, start_date='2024-01-01')
