"""
每日更新调度模块
统一调度净值更新、评分重算、新闻同步、情感分析、邮箱推送
"""

import logging
from datetime import datetime

from src.database import FundDatabase
from src.data_loader import FUND_POOL
from src.config import (
    AKSHARE_BATCH_SIZE, AKSHARE_BATCH_REST
)

logger = logging.getLogger(__name__)


def is_trading_day():
    """判断今天是否为交易日（简化版：仅排除周末）"""
    now = datetime.now()
    return now.weekday() < 5


def is_weekend():
    """判断今天是否为周末"""
    return datetime.now().weekday() >= 5


def update_nav_data(fund_pool=None, db=None, fast=True):
    """
    每日净值更新

    Args:
        fund_pool: 基金池
        db: 数据库实例
        fast: True=用批量接口一次更新（快100倍），False=逐只获取（获取完整历史）

    增量策略：
    - fast=True：用 ak.fund_open_fund_daily_em() 一次获取全市场当日净值（1次API调用）
    - fast=False：逐只查询最新日期，只获取增量数据（多API调用，适合首次历史补全）
    """
    fund_pool = fund_pool or FUND_POOL
    db = db or FundDatabase()
    if not db.client:
        db.connect()

    if fast:
        # 快速模式：批量接口一次更新所有基金当日净值
        from src.fund_crawler import batch_update_nav_from_daily
        updated = batch_update_nav_from_daily(fund_pool, db)
        return updated

    # 慢速模式：逐只获取完整历史（首次运行或补全数据时用）
    from src.akshare_client import get_fund_nav

    logger.info(f"开始净值更新（逐只模式）: {len(fund_pool)} 只基金")

    codes_to_update = []
    for code in fund_pool:
        latest = db.get_latest_nav_date(code)
        if latest:
            codes_to_update.append((code, latest))
        else:
            codes_to_update.append((code, None))

    updated = 0
    for code, since_date in codes_to_update:
        try:
            df = get_fund_nav(code, start_date=since_date)
            if not df.empty:
                db.insert_nav_data(code, df)
                updated += 1
                logger.info(f"  {code}: 更新 {len(df)} 条")
            else:
                logger.info(f"  {code}: 无新数据")
        except Exception as e:
            logger.error(f"  {code}: 更新失败 - {e}")

    logger.info(f"净值更新完成: {updated}/{len(fund_pool)}")
    return updated


def update_fund_scores(fund_pool=None, db=None):
    """
    重算所有基金评分
    """
    from src.fund_screener import FundScorer

    fund_pool = fund_pool or FUND_POOL
    db = db or FundDatabase()
    if not db.client:
        db.connect()

    logger.info("开始评分更新...")

    # 批量获取净值数据
    nav_dict = {}
    for code in fund_pool:
        try:
            df = db.get_nav_data(code)
            if not df.empty:
                nav_dict[code] = df
        except Exception:
            pass
    logger.info(f"已加载 {len(nav_dict)} 只基金净值数据")

    # 批量获取基金详情
    detail_dict = {}
    for code in fund_pool:
        try:
            fund = db.get_fund(code)
            if fund:
                detail_dict[code] = fund
        except Exception:
            pass
    logger.info(f"已加载 {len(detail_dict)} 只基金详情")

    # 评分
    scorer = FundScorer(db=db)
    results = scorer.score_all_funds(fund_pool, nav_data_dict=nav_dict, detail_dict=detail_dict)

    # 保存
    scorer.save_scores(results)

    # 统计
    focus = sum(1 for _, _, s in results if s['pool_type'] == 'focus')
    obs = sum(1 for _, _, s in results if s['pool_type'] == 'observation')
    elim = sum(1 for _, _, s in results if s['pool_type'] == 'eliminate')
    logger.info(f"评分更新完成: 重点池={focus}, 观察池={obs}, 淘汰池={elim}")

    return results


def update_news(db=None):
    """
    新闻同步 + 情感分析
    """
    db = db or FundDatabase()
    if not db.client:
        db.connect()

    # 同步新闻
    from src.news_sync import sync_news
    sync_result = sync_news(db=db)
    logger.info(f"新闻同步: {sync_result}")

    # 情感分析
    from src.news_analyzer import process_unlabeled_news
    analysis_result = process_unlabeled_news(limit=100, db=db)
    logger.info(f"情感分析: {analysis_result}")

    return sync_result, analysis_result


def send_evening_report(score_results=None, db=None):
    """
    发送晚间报告
    """
    db = db or FundDatabase()
    if not db.client:
        db.connect()

    from src.email_notifier import send_daily_report, send_alert

    # 评分变动（简化：直接取最新评分）
    score_changes = []
    if score_results:
        for code, name, scores in score_results[:20]:
            score_changes.append({
                'fund_code': code,
                'fund_name': name,
                'total_score': scores['total'],
                'score_change': 0,  # 需要与前一天比较
                'pool_type': scores['pool_type']
            })

    # 重要新闻（取影响度最高的）
    news_highlights = []
    try:
        result = db.client.table('trend_news') \
            .select('title, category, sentiment_score, api_analysis') \
            .eq('is_labeled_by_api', True) \
            .order('created_at', desc=True) \
            .limit(15) \
            .execute()
        for item in result.data:
            analysis = item.get('api_analysis', {})
            if isinstance(analysis, str):
                import json
                try:
                    analysis = json.loads(analysis)
                except Exception:
                    analysis = {}
            news_highlights.append({
                'title': item.get('title', ''),
                'category': item.get('category', '其他'),
                'sentiment': item.get('sentiment_score', 0),
                'impact': analysis.get('impact', 3),
            })
    except Exception as e:
        logger.error(f"获取新闻失败: {e}")

    # 预测摘要（暂时为空，待模型训练完成后填充）
    predictions_summary = []

    # 发送
    try:
        send_daily_report(score_changes, news_highlights, predictions_summary)
        logger.info("晚间报告发送成功")
    except Exception as e:
        logger.error(f"报告发送失败: {e}")
        try:
            send_alert("晚间报告发送失败", str(e))
        except Exception:
            pass


def run_daily_update(full=False):
    """
    每日更新主入口

    Args:
        full: True=全量更新（首次运行），False=增量更新（日常运行）

    调度规则：
    - 交易日 18:00 后：完整流水线（净值→评分→新闻→情感→推送）
    - 周末：仅新闻同步 + 情感分析（合并为一次）
    """
    now = datetime.now()
    logger.info(f"每日更新开始: {now.strftime('%Y-%m-%d %H:%M:%S')}")

    db = FundDatabase()
    if not db.connect():
        logger.error("数据库连接失败，退出")
        return

    try:
        if is_trading_day():
            logger.info("交易日模式")

            # 1. 净值更新
            update_nav_data(db=db)

            # 2. 评分更新
            score_results = update_fund_scores(db=db)

            # 3. 新闻同步 + 情感分析
            update_news(db=db)

            # 4. 发送晚间报告
            send_evening_report(score_results, db=db)

        elif is_weekend():
            logger.info("周末模式（合并处理）")

            # 仅新闻同步 + 情感分析
            update_news(db=db)

        logger.info("每日更新完成")

    except Exception as e:
        logger.error(f"每日更新异常: {e}")
        from src.email_notifier import send_alert
        try:
            send_alert("每日更新流程异常", str(e))
        except Exception:
            pass


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(name)s: %(message)s'
    )

    import sys
    full_mode = '--full' in sys.argv
    run_daily_update(full=full_mode)
