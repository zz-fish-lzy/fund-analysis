"""
基金筛选与评分模块
硬指标初筛 + 动态评分池管理
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from src.config import (
    FOCUS_SCORE_THRESHOLD, OBSERVATION_SCORE_THRESHOLD
)
from src.database import FundDatabase
from src.akshare_client import get_fund_detail, get_fund_nav

logger = logging.getLogger(__name__)


class FundScreener:
    """基金硬指标初筛"""

    def __init__(self, db=None):
        self.db = db or FundDatabase()

    def screen_fund(self, fund_code, nav_data=None, detail=None):
        """
        筛选单只基金

        Args:
            fund_code: 基金代码
            nav_data: 净值 DataFrame（可选，避免重复获取）
            detail: 基金详情 dict（可选）

        Returns:
            dict: {pass: bool, reasons: list, detail: dict}
        """
        reasons = []
        passed = True

        # 获取详情
        if detail is None:
            try:
                detail = get_fund_detail(fund_code)
            except Exception:
                detail = {}

        # 获取净值（用于计算成立时间等）
        if nav_data is None:
            try:
                nav_data = get_fund_nav(fund_code)
            except Exception:
                nav_data = pd.DataFrame()

        # 1. 规模 >= 2亿
        size_str = detail.get('fund_size', '')
        if size_str:
            try:
                # 处理 "123.45亿元" 或 "123.45 亿" 等格式
                size_val = float(''.join(c for c in str(size_str) if c.isdigit() or c == '.'))
                if size_val < 2:
                    reasons.append(f'规模不足2亿: {size_val}亿')
                    passed = False
                if size_val > 200:
                    reasons.append(f'规模超200亿: {size_val}亿')
                    # 不一票否决，仅记录
            except (ValueError, TypeError):
                reasons.append('规模数据缺失')
                # 数据缺失不否决，给通过

        # 2. 成立时间 >= 1年
        if not nav_data.empty:
            first_date = nav_data['date'].min()
            if isinstance(first_date, pd.Timestamp):
                age_days = (datetime.now() - first_date.to_pydatetime()).days
                if age_days < 365:
                    reasons.append(f'成立不足1年: {age_days}天')
                    passed = False

        # 3. 基金经理任职时间（如果有数据）
        manager_start = detail.get('manager_start_date', '')
        if manager_start:
            try:
                start = pd.to_datetime(manager_start)
                tenure_days = (datetime.now() - start).days
                if tenure_days < 365:
                    reasons.append(f'经理任期不足1年: {tenure_days}天')
                    passed = False
            except Exception:
                pass

        # 4. 费率检查
        mgmt_fee = detail.get('management_fee', '')
        if mgmt_fee:
            try:
                fee_val = float(''.join(c for c in str(mgmt_fee) if c.isdigit() or c == '.'))
                if fee_val > 1.5:
                    reasons.append(f'管理费过高: {fee_val}%')
                    passed = False
            except (ValueError, TypeError):
                pass

        return {
            'pass': passed,
            'reasons': reasons,
            'detail': detail
        }

    def screen_pool(self, fund_pool):
        """
        筛选整个基金池

        Args:
            fund_pool: {fund_code: fund_name} dict

        Returns:
            dict: {fund_code: screening_result}
        """
        results = {}
        total = len(fund_pool)

        for i, (code, name) in enumerate(fund_pool.items(), 1):
            logger.info(f"筛选 [{i}/{total}] {code} - {name}")
            try:
                result = self.screen_fund(code)
                results[code] = result
                status = "通过" if result['pass'] else "未通过"
                logger.info(f"  {status}: {result['reasons'] or '无'}")
            except Exception as e:
                logger.error(f"  筛选失败: {e}")
                results[code] = {'pass': False, 'reasons': [str(e)], 'detail': {}}

        passed = sum(1 for r in results.values() if r['pass'])
        logger.info(f"筛选完成: {passed}/{total} 通过")
        return results


class FundScorer:
    """基金动态评分（满分100分）"""

    def __init__(self, db=None):
        self.db = db or FundDatabase()

    def score_performance(self, nav_data):
        """
        业绩评分（满分40）

        评分规则：
        - 近1月收益: 10分（按同类排名分档）
        - 近3月收益: 15分
        - 近1年收益: 15分
        """
        if nav_data.empty or len(nav_data) < 30:
            return 7.0  # 数据不足给中位数

        score = 0.0
        nav_data = nav_data.sort_values('date')

        # 近1月收益
        try:
            recent_1m = nav_data.tail(20)
            ret_1m = (recent_1m['nav'].iloc[-1] / recent_1m['nav'].iloc[0] - 1) * 100
            # 分档: >5%=10, 2-5%=8, 0-2%=6, -2-0%=4, <-2%=2
            if ret_1m > 5:
                score += 10
            elif ret_1m > 2:
                score += 8
            elif ret_1m > 0:
                score += 6
            elif ret_1m > -2:
                score += 4
            else:
                score += 2
        except Exception:
            score += 5

        # 近3月收益
        try:
            recent_3m = nav_data.tail(60)
            ret_3m = (recent_3m['nav'].iloc[-1] / recent_3m['nav'].iloc[0] - 1) * 100
            if ret_3m > 10:
                score += 15
            elif ret_3m > 5:
                score += 12
            elif ret_3m > 0:
                score += 9
            elif ret_3m > -5:
                score += 6
            else:
                score += 3
        except Exception:
            score += 7.5

        # 近1年收益
        try:
            recent_1y = nav_data.tail(250)
            if len(recent_1y) >= 200:
                ret_1y = (recent_1y['nav'].iloc[-1] / recent_1y['nav'].iloc[0] - 1) * 100
                if ret_1y > 20:
                    score += 15
                elif ret_1y > 10:
                    score += 12
                elif ret_1y > 0:
                    score += 9
                elif ret_1y > -10:
                    score += 6
                else:
                    score += 3
            else:
                score += 7.5
        except Exception:
            score += 7.5

        return score

    def score_risk(self, nav_data):
        """
        风险评分（满分25）

        评分规则：
        - 最大回撤: 15分（越小分越高）
        - 波动率: 10分
        """
        if nav_data.empty or len(nav_data) < 30:
            return 5.0

        score = 0.0
        nav_data = nav_data.sort_values('date')

        # 计算收益率序列
        returns = nav_data['nav'].pct_change().dropna()

        # 最大回撤
        try:
            cumulative = (1 + returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            max_dd = abs(drawdown.min()) * 100

            # 分档: <5%=15, 5-10%=12, 10-15%=9, 15-20%=6, >20%=3
            if max_dd < 5:
                score += 15
            elif max_dd < 10:
                score += 12
            elif max_dd < 15:
                score += 9
            elif max_dd < 20:
                score += 6
            else:
                score += 3
        except Exception:
            score += 7.5

        # 波动率（年化）
        try:
            vol = returns.std() * np.sqrt(250) * 100
            # 分档: <10%=10, 10-15%=8, 15-20%=6, 20-30%=4, >30%=2
            if vol < 10:
                score += 10
            elif vol < 15:
                score += 8
            elif vol < 20:
                score += 6
            elif vol < 30:
                score += 4
            else:
                score += 2
        except Exception:
            score += 5

        return score

    def score_manager(self, detail=None):
        """
        经理评分（满分20）

        评分规则：
        - 任职时间: 10分
        - 管理的其他基金业绩: 10分（数据不足时给默认分）
        """
        score = 0.0

        if not detail:
            return 10.0  # 数据不足给中位数

        # 任职时间
        manager_start = detail.get('manager_start_date', '')
        if manager_start:
            try:
                start = pd.to_datetime(manager_start)
                tenure_years = (datetime.now() - start).days / 365
                if tenure_years > 5:
                    score += 10
                elif tenure_years > 3:
                    score += 8
                elif tenure_years > 1:
                    score += 6
                else:
                    score += 4
            except Exception:
                score += 5
        else:
            score += 5

        # 管理其他基金业绩（暂无数据，给默认分）
        score += 5

        return score

    def score_flow(self, detail=None):
        """
        资金流评分（满分15）

        评分规则：
        - 机构持仓变化: 8分
        - 基金规模变化: 7分
        """
        # 暂无精确数据，给默认中位数
        return 7.5

    def score_fund(self, fund_code, nav_data=None, detail=None):
        """
        综合评分

        Returns:
            dict: {total, performance, risk, manager, flow, pool_type}
        """
        if nav_data is None:
            try:
                nav_data = get_fund_nav(fund_code)
            except Exception:
                nav_data = pd.DataFrame()

        if detail is None:
            detail = {}

        performance = self.score_performance(nav_data)
        risk = self.score_risk(nav_data)
        manager = self.score_manager(detail)
        flow = self.score_flow(detail)

        total = performance + risk + manager + flow

        if total >= FOCUS_SCORE_THRESHOLD:
            pool_type = 'focus'
        elif total >= OBSERVATION_SCORE_THRESHOLD:
            pool_type = 'observation'
        else:
            pool_type = 'eliminate'

        return {
            'total': round(total, 2),
            'performance': round(performance, 2),
            'risk': round(risk, 2),
            'manager': round(manager, 2),
            'flow': round(flow, 2),
            'pool_type': pool_type
        }

    def score_all_funds(self, fund_pool, nav_data_dict=None, detail_dict=None):
        """
        评分所有基金

        Args:
            fund_pool: {fund_code: fund_name}
            nav_data_dict: {fund_code: DataFrame}（可选）
            detail_dict: {fund_code: dict}（可选，避免重复获取详情）

        Returns:
            list of (fund_code, fund_name, scores) 按总分降序
        """
        results = []

        for i, (code, name) in enumerate(fund_pool.items(), 1):
            logger.info(f"评分 [{i}/{len(fund_pool)}] {code} - {name}")
            try:
                nav = nav_data_dict.get(code) if nav_data_dict else None
                detail = detail_dict.get(code, {}) if detail_dict else None
                scores = self.score_fund(code, nav_data=nav, detail=detail)
                results.append((code, name, scores))
                logger.info(f"  总分: {scores['total']}, 池: {scores['pool_type']}")
            except Exception as e:
                logger.error(f"  评分失败: {e}")

        results.sort(key=lambda x: x[2]['total'], reverse=True)
        return results

    def save_scores(self, score_results, score_date=None):
        """
        保存评分结果到数据库

        Args:
            score_results: list of (fund_code, fund_name, scores)
            score_date: 评分日期（默认今天）
        """
        score_date = score_date or datetime.now().strftime('%Y-%m-%d')

        records = []
        for code, name, scores in score_results:
            records.append({
                'fund_code': code,
                'score_date': score_date,
                'total_score': scores['total'],
                'performance_score': scores['performance'],
                'risk_score': scores['risk'],
                'manager_score': scores['manager'],
                'flow_score': scores['flow'],
                'pool_type': scores['pool_type']
            })

        if records:
            self.db.insert_scores_batch(records)
            logger.info(f"保存 {len(records)} 条评分记录，日期: {score_date}")

        # 更新 funds 表的 pool_type 和 is_focus
        for code, name, scores in score_results:
            self.db.upsert_fund({
                'fund_code': code,
                'fund_name': name,
                'pool_type': scores['pool_type'],
                'is_focus': scores['pool_type'] == 'focus'
            })


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    from src.data_loader import FUND_POOL

    # 测试筛选
    screener = FundScreener()
    print("开始筛选基金池...")
    # 只测试前3只
    test_pool = dict(list(FUND_POOL.items())[:3])
    results = screener.screen_pool(test_pool)

    # 测试评分
    scorer = FundScorer()
    print("\n开始评分...")
    score_results = scorer.score_all_funds(test_pool)
    for code, name, scores in score_results:
        print(f"  {code} {name}: {scores}")
