"""
首次全量数据加载脚本
执行顺序: 基金信息 → 历史净值 → 评分 → 新闻同步

使用方式:
    python -m scripts.initial_data_load
"""

import sys
import os
import logging
from datetime import datetime

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import load_env
from src.database import FundDatabase
from src.data_loader import FUND_POOL

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)


def step1_insert_fund_info(db):
    """第1步: 将基金池基础信息写入 funds 表"""
    logger.info("=" * 50)
    logger.info("第1步: 写入基金基础信息")
    logger.info("=" * 50)

    records = []
    for code, name in FUND_POOL.items():
        # 根据代码判断基金类型
        if code.startswith('51') or code.startswith('15'):
            fund_type = 'ETF'
        elif code.startswith('00') or code.startswith('01') or code.startswith('16'):
            fund_type = '混合型'
        else:
            fund_type = '其他'

        records.append({
            'fund_code': code,
            'fund_name': name,
            'fund_type': fund_type,
            'pool_type': 'observation',
            'is_focus': False,
            'is_active': True,
        })

    db.upsert_funds_batch(records)
    logger.info(f"写入 {len(records)} 只基金基础信息")


def step2_load_nav_data(db):
    """第2步: 从 AKShare 获取历史净值并写入 fund_nav 表"""
    logger.info("=" * 50)
    logger.info("第2步: 加载历史净值数据")
    logger.info("=" * 50)

    from src.akshare_client import get_fund_nav

    success = 0
    total = len(FUND_POOL)

    for i, (code, name) in enumerate(FUND_POOL.items(), 1):
        logger.info(f"[{i}/{total}] {code} - {name}")

        try:
            # 获取全部历史数据
            df = get_fund_nav(code)
            if not df.empty:
                db.insert_nav_data(code, df)
                success += 1
                logger.info(f"  写入 {len(df)} 条净值数据")
            else:
                logger.warning(f"  数据为空")
        except Exception as e:
            logger.error(f"  失败: {e}")

    logger.info(f"净值加载完成: {success}/{total}")


def step3_score_funds(db):
    """第3步: 对所有基金评分并写入 fund_scores 表"""
    logger.info("=" * 50)
    logger.info("第3步: 基金评分")
    logger.info("=" * 50)

    from src.fund_screener import FundScorer

    scorer = FundScorer(db=db)

    # 从数据库获取净值数据
    nav_dict = {}
    for code in FUND_POOL:
        try:
            df = db.get_nav_data(code)
            if not df.empty:
                nav_dict[code] = df
        except Exception:
            pass

    logger.info(f"已加载 {len(nav_dict)} 只基金净值数据")

    # 评分
    results = scorer.score_all_funds(FUND_POOL, nav_data_dict=nav_dict)

    # 保存
    scorer.save_scores(results)

    # 统计
    focus = sum(1 for _, _, s in results if s['pool_type'] == 'focus')
    obs = sum(1 for _, _, s in results if s['pool_type'] == 'observation')
    elim = sum(1 for _, _, s in results if s['pool_type'] == 'eliminate')
    logger.info(f"评分结果: 重点池={focus}, 观察池={obs}, 淘汰池={elim}")


def step4_sync_news(db):
    """第4步: 从 COS 同步新闻到 trend_news 表"""
    logger.info("=" * 50)
    logger.info("第4步: 新闻同步")
    logger.info("=" * 50)

    try:
        from src.news_sync import sync_news
        result = sync_news(since_date='2026-05-01', db=db)
        logger.info(f"新闻同步结果: {result}")
    except ImportError:
        logger.warning("cos-python-sdk-v5 未安装，跳过新闻同步")
    except Exception as e:
        logger.error(f"新闻同步失败: {e}")


def main():
    """主流程"""
    logger.info("=" * 60)
    logger.info("基金分析与预测 — 首次全量数据加载")
    logger.info(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"基金池: {len(FUND_POOL)} 只基金")
    logger.info("=" * 60)

    # 连接数据库
    db = FundDatabase()
    if not db.connect():
        logger.error("数据库连接失败！")
        logger.error("请检查 .env 中的 Supabase 配置")
        return

    # 按步骤执行
    try:
        step1_insert_fund_info(db)
        step2_load_nav_data(db)
        step3_score_funds(db)
        step4_sync_news(db)

        logger.info("=" * 60)
        logger.info("全部数据加载完成！")
        logger.info("=" * 60)

    except KeyboardInterrupt:
        logger.warning("用户中断执行")
    except Exception as e:
        logger.error(f"执行异常: {e}")
        raise


if __name__ == '__main__':
    main()
