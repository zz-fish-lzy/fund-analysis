"""
AKShare 限流封装
提供随机抖动、指数退避重试、交易时段保护、分批处理
批量接口优先（一次获取全市场数据），单只接口限流保护
"""

import time
import random
import logging
from datetime import datetime
from functools import wraps

import akshare as ak
import pandas as pd

from src.config import (
    AKSHARE_JITTER_MIN, AKSHARE_JITTER_MAX,
    AKSHARE_BATCH_SIZE, AKSHARE_BATCH_REST,
    AKSHARE_MAX_RETRIES
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


def _rate_limited(func):
    """装饰器：每次调用前随机休眠，失败时指数退避重试"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        for attempt in range(AKSHARE_MAX_RETRIES):
            try:
                # 随机抖动间隔
                sleep_time = random.uniform(AKSHARE_JITTER_MIN, AKSHARE_JITTER_MAX)
                time.sleep(sleep_time)
                return func(*args, **kwargs)
            except Exception as e:
                if attempt == AKSHARE_MAX_RETRIES - 1:
                    logger.error(f"{func.__name__} 最终失败: {e}")
                    raise
                wait = 2 ** (attempt + 1) + random.uniform(0, 1)
                logger.warning(f"{func.__name__} 第{attempt+1}次失败，{wait:.1f}秒后重试: {e}")
                time.sleep(wait)
    return wrapper


def is_trading_hours():
    """判断当前是否在交易时段（9:15 ~ 15:05 的工作日）"""
    now = datetime.now()
    if now.weekday() >= 5:  # 周末
        return False
    t = now.hour * 60 + now.minute
    return 9 * 60 + 15 <= t <= 15 * 60 + 5


@_rate_limited
def get_fund_nav(fund_code, start_date=None, end_date=None):
    """
    获取单只基金净值数据（限流版）

    Returns:
        DataFrame: columns = [date, nav, daily_return, fund_code]
    """
    df = ak.fund_open_fund_info_em(symbol=fund_code, indicator="单位净值走势")
    df.columns = ['date', 'nav', 'daily_return']
    df['date'] = pd.to_datetime(df['date'])

    if start_date:
        df = df[df['date'] >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df['date'] <= pd.to_datetime(end_date)]

    df['fund_code'] = fund_code
    df = df.sort_values('date').reset_index(drop=True)
    return df


@_rate_limited
def get_fund_detail(fund_code):
    """
    获取基金详细信息（限流版）

    Returns:
        dict: {fund_name, fund_type, fund_size, establishment_date, fund_manager, ...}
    """
    try:
        # 尝试用雪球接口获取详细信息
        df = ak.fund_individual_basic_info_xq(symbol=fund_code)
        info = {}
        for _, row in df.iterrows():
            info[row.iloc[0]] = row.iloc[1] if len(row) > 1 else ''
        return {
            'fund_name': info.get('基金名称', info.get('name', '')),
            'fund_type': info.get('基金类型', info.get('type', '')),
            'fund_size': info.get('基金规模', info.get('size', '')),
            'establishment_date': info.get('成立日期', info.get('inception_date', '')),
            'fund_manager': info.get('基金经理', info.get('manager', '')),
            'fund_company': info.get('基金公司', info.get('company', '')),
        }
    except Exception:
        # 回退：用 fund_name_em 查询基本信息
        try:
            df = ak.fund_name_em()
            row = df[df['基金代码'] == fund_code]
            if not row.empty:
                return {
                    'fund_name': row.iloc[0].get('基金简称', ''),
                    'fund_type': row.iloc[0].get('基金类型', ''),
                }
        except Exception:
            pass
    return {}


@_rate_limited
def get_fund_daily_em():
    """获取全市场基金当日净值（批量接口，限流版）"""
    return ak.fund_open_fund_daily_em()


def batch_get_fund_nav(fund_codes, start_date=None, end_date=None,
                       batch_size=None, rest_seconds=None):
    """
    分批获取多只基金净值

    Args:
        fund_codes: 基金代码列表
        start_date: 开始日期
        end_date: 结束日期
        batch_size: 每批数量（默认从配置读取）
        rest_seconds: 批间休息秒数（默认从配置读取）

    Returns:
        dict: {fund_code: DataFrame}
    """
    batch_size = batch_size or AKSHARE_BATCH_SIZE
    rest_seconds = rest_seconds or AKSHARE_BATCH_REST

    results = {}
    total = len(fund_codes)

    for i in range(0, total, batch_size):
        batch = fund_codes[i:i + batch_size]
        batch_num = i // batch_size + 1
        total_batches = (total + batch_size - 1) // batch_size

        # 交易时段保护
        if is_trading_hours():
            logger.warning(f"当前处于交易时段，等待至 15:05 后继续...")
            while is_trading_hours():
                time.sleep(60)

        logger.info(f"批次 {batch_num}/{total_batches}: 处理 {len(batch)} 只基金")

        for code in batch:
            try:
                df = get_fund_nav(code, start_date, end_date)
                if not df.empty:
                    results[code] = df
                    logger.info(f"  {code}: {len(df)} 条数据")
                else:
                    logger.warning(f"  {code}: 数据为空")
            except Exception as e:
                logger.error(f"  {code}: 失败 - {e}")

        # 批间休息
        if i + batch_size < total:
            logger.info(f"批间休息 {rest_seconds // 60} 分钟...")
            time.sleep(rest_seconds)

    logger.info(f"批量获取完成: {len(results)}/{total} 只基金")
    return results


def batch_get_fund_detail(fund_codes, batch_size=None, rest_seconds=None):
    """
    分批获取多只基金详细信息

    Returns:
        dict: {fund_code: {fund_name, fund_type, ...}}
    """
    batch_size = batch_size or AKSHARE_BATCH_SIZE
    rest_seconds = rest_seconds or AKSHARE_BATCH_REST

    results = {}
    total = len(fund_codes)

    for i in range(0, total, batch_size):
        batch = fund_codes[i:i + batch_size]
        batch_num = i // batch_size + 1
        total_batches = (total + batch_size - 1) // batch_size

        if is_trading_hours():
            logger.warning(f"当前处于交易时段，等待至 15:05 后继续...")
            while is_trading_hours():
                time.sleep(60)

        logger.info(f"批次 {batch_num}/{total_batches}: 获取 {len(batch)} 只基金信息")

        for code in batch:
            try:
                detail = get_fund_detail(code)
                if detail:
                    results[code] = detail
            except Exception as e:
                logger.error(f"  {code}: 失败 - {e}")

        if i + batch_size < total:
            logger.info(f"批间休息 {rest_seconds // 60} 分钟...")
            time.sleep(rest_seconds)

    return results


# ── 批量接口（一次获取全市场数据，速度极快） ──

@_rate_limited
def get_all_fund_list():
    """
    获取全市场基金列表（一次 API 调用获取 10000+ 只基金）

    Returns:
        DataFrame: columns = [基金代码, 基金简称, 基金类型, ...]
    """
    return ak.fund_name_em()


@_rate_limited
def get_all_fund_daily_nav():
    """
    获取全市场基金当日净值（一次 API 调用）

    Returns:
        DataFrame: columns = [基金代码, 单位净值, 累计净值, 日增长率, ...]
    """
    return ak.fund_open_fund_daily_em()


def quick_screen_funds(min_size=2, min_age_days=365, max_mgmt_fee=1.5):
    """
    快速硬指标初筛（批量接口，一次获取全市场数据后本地筛选）

    Args:
        min_size: 最低规模（亿）
        min_age_days: 最短成立天数
        max_mgmt_fee: 最高管理费率（%）

    Returns:
        list[dict]: 通过筛选的基金列表
    """
    logger.info("获取全市场基金列表...")
    fund_list = get_all_fund_list()
    logger.info(f"全市场共 {len(fund_list)} 只基金")

    # 基本字段筛选
    screened = []
    for _, row in fund_list.iterrows():
        code = str(row.get('基金代码', '')).strip()
        name = str(row.get('基金简称', '')).strip()
        fund_type = str(row.get('基金类型', '')).strip()

        if not code:
            continue

        # 排除货币基金和短期理财
        if '货币' in fund_type or '理财' in fund_type or '短期' in name:
            continue

        screened.append({
            'code': code,
            'name': name,
            'type': fund_type,
        })

    logger.info(f"类型筛选后: {len(screened)} 只基金")
    return screened


if __name__ == '__main__':
    # 测试限流
    print("测试单只基金获取...")
    df = get_fund_nav('510300')
    print(f"510300: {len(df)} 条数据")

    print("\n测试基金详情...")
    detail = get_fund_detail('510300')
    print(f"510300 详情: {detail}")
