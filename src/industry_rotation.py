"""
行业轮动模块
基于行业因子捕捉板块轮动机会
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta


# 行业分类映射（按基金持仓风格分类）
SECTOR_MAPPING = {
    # 宽基指数
    '510300': {'sector': '宽基指数', 'style': '大盘价值'},
    '510500': {'sector': '宽基指数', 'style': '中盘成长'},
    '159915': {'sector': '宽基指数', 'style': '创业板'},
    '510050': {'sector': '宽基指数', 'style': '大盘价值'},
    '512100': {'sector': '宽基指数', 'style': '小盘成长'},
    '159919': {'sector': '宽基指数', 'style': '大盘价值'},
    '000961': {'sector': '宽基指数', 'style': '大盘价值'},

    # 消费
    '012414': {'sector': '消费', 'style': '白酒'},
    '161725': {'sector': '消费', 'style': '白酒'},
    '512690': {'sector': '消费', 'style': '白酒'},
    '005827': {'sector': '消费', 'style': '蓝筹混合'},

    # 医药
    '003095': {'sector': '医药', 'style': '医疗健康'},
    '512010': {'sector': '医药', 'style': '医药行业'},
    '000831': {'sector': '医药', 'style': '医疗前沿'},
    '513060': {'sector': '医药', 'style': '恒生医疗'},

    # 新能源
    '001156': {'sector': '新能源', 'style': '新能源汽车'},
    '515030': {'sector': '新能源', 'style': '新能源'},
    '515790': {'sector': '新能源', 'style': '光伏'},
    '516160': {'sector': '新能源', 'style': '新能源车'},

    # 科技
    '005911': {'sector': '科技', 'style': '半导体'},
    '001938': {'sector': '科技', 'style': '时代先锋'},
    '163406': {'sector': '科技', 'style': '均衡混合'},
    '320007': {'sector': '科技', 'style': '半导体芯片'},

    # 金融
    '512880': {'sector': '金融', 'style': '证券'},
    '001875': {'sector': '金融', 'style': '沪港深'},

    # 海外
    '159941': {'sector': '海外', 'style': '纳斯达克'},
    '513100': {'sector': '海外', 'style': '纳斯达克'},
    '513050': {'sector': '海外', 'style': '中概互联'},
    '159920': {'sector': '海外', 'style': '恒生指数'},

    # 债券
    '217022': {'sector': '债券', 'style': '产业债'},
    '110017': {'sector': '债券', 'style': '增强回报'},
    '050011': {'sector': '债券', 'style': '信用债'},
    '000171': {'sector': '债券', 'style': '灵活配置'},
    '000463': {'sector': '债券', 'style': '双债'},
    '000563': {'sector': '债券', 'style': '定期债'},
    '000628': {'sector': '债券', 'style': '高鑫债'},
    '000789': {'sector': '债券', 'style': '双利债'},
}


class IndustryRotationAnalyzer:
    """行业轮动分析器"""

    def __init__(self, lookback_days=20):
        self.lookback_days = lookback_days

    def get_sector(self, fund_code: str) -> str:
        """获取基金所属行业"""
        info = SECTOR_MAPPING.get(fund_code, {})
        return info.get('sector', '未知')

    def calculate_sector_performance(self, fund_data: dict, current_date: pd.Timestamp) -> dict:
        """
        计算各行业近期表现

        Args:
            fund_data: 所有基金数据 {fund_code: DataFrame}
            current_date: 当前日期

        Returns:
            dict: 各行业表现 {sector: {'return': float, 'momentum': float, 'count': int}}
        """
        sector_returns = {}

        for fund_code, df in fund_data.items():
            sector = self.get_sector(fund_code)
            if sector == '未知' or sector == '债券':
                continue

            # 获取近期数据
            recent = df[df['date'] <= current_date].tail(self.lookback_days)
            if len(recent) < 5:
                continue

            # 计算区间收益率
            period_return = (recent['nav'].iloc[-1] / recent['nav'].iloc[0]) - 1

            # 计算动量（短期vs长期）
            short_return = (recent['nav'].iloc[-1] / recent['nav'].iloc[-5]) - 1 if len(recent) >= 5 else 0
            long_return = period_return

            if sector not in sector_returns:
                sector_returns[sector] = {'returns': [], 'momentums': []}

            sector_returns[sector]['returns'].append(period_return)
            sector_returns[sector]['momentums'].append(short_return - long_return)

        # 汇总各行业
        result = {}
        for sector, data in sector_returns.items():
            result[sector] = {
                'return': np.mean(data['returns']),
                'momentum': np.mean(data['momentums']),
                'count': len(data['returns']),
                'return_std': np.std(data['returns']) if len(data['returns']) > 1 else 0
            }

        return result

    def rank_sectors(self, sector_performance: dict) -> list:
        """
        对行业进行排名

        Args:
            sector_performance: 行业表现数据

        Returns:
            list: 按综合得分排序的行业列表 [(sector, score, details), ...]
        """
        scores = []
        for sector, perf in sector_performance.items():
            # 综合得分 = 收益率权重0.6 + 动量权重0.4
            score = perf['return'] * 0.6 + perf['momentum'] * 0.4
            scores.append((sector, score, perf))

        # 按得分降序排列
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores

    def get_rotation_signal(self, fund_data: dict, current_date: pd.Timestamp) -> dict:
        """
        获取行业轮动信号

        Args:
            fund_data: 所有基金数据
            current_date: 当前日期

        Returns:
            dict: 轮动信号
        """
        # 计算行业表现
        sector_perf = self.calculate_sector_performance(fund_data, current_date)

        if not sector_perf:
            return {'action': 'hold', 'reason': '数据不足', 'rankings': []}

        # 排名
        rankings = self.rank_sectors(sector_perf)

        if len(rankings) < 2:
            return {'action': 'hold', 'reason': '行业数据不足', 'rankings': rankings}

        # 生成信号
        top_sector = rankings[0]
        bottom_sector = rankings[-1]

        signal = {
            'date': current_date,
            'top_sector': top_sector[0],
            'top_score': top_sector[1],
            'bottom_sector': bottom_sector[0],
            'bottom_score': bottom_sector[1],
            'rankings': rankings,
            'action': 'rotate',
            'reason': f"建议关注{top_sector[0]}板块，回避{bottom_sector[0]}板块"
        }

        return signal

    def get_fund_rotation_score(self, fund_code: str, fund_data: dict,
                                 current_date: pd.Timestamp) -> float:
        """
        获取单只基金的轮动得分

        Args:
            fund_code: 基金代码
            fund_data: 所有基金数据
            current_date: 当前日期

        Returns:
            float: 轮动得分 (0-1)
        """
        sector = self.get_sector(fund_code)
        if sector == '未知' or sector == '债券':
            return 0.5

        sector_perf = self.calculate_sector_performance(fund_data, current_date)
        if sector not in sector_perf:
            return 0.5

        # 排名
        rankings = self.rank_sectors(sector_perf)
        sectors = [r[0] for r in rankings]

        if sector in sectors:
            rank = sectors.index(sector)
            # 归一化到0-1，排名越靠前得分越高
            return 1.0 - (rank / len(sectors))
        return 0.5

    def calculate_sector_volatility(self, fund_data: dict, current_date: pd.Timestamp) -> dict:
        """
        计算各行业波动率

        Returns:
            dict: {sector: volatility}
        """
        sector_vols = {}

        for fund_code, df in fund_data.items():
            sector = self.get_sector(fund_code)
            if sector == '未知':
                continue

            recent = df[df['date'] <= current_date].tail(self.lookback_days)
            if len(recent) < 5:
                continue

            vol = recent['daily_return'].std()
            if sector not in sector_vols:
                sector_vols[sector] = []
            sector_vols[sector].append(vol)

        return {sector: np.mean(vols) for sector, vols in sector_vols.items() if vols}


if __name__ == '__main__':
    # 测试行业轮动分析
    from data_loader import load_all_funds

    # 加载测试数据
    test_data = load_all_funds('data/raw/test')

    if test_data:
        analyzer = IndustryRotationAnalyzer(lookback_days=20)

        # 获取最新日期
        all_dates = set()
        for df in test_data.values():
            all_dates.update(df['date'].tolist())
        latest_date = max(all_dates)

        print("=" * 60)
        print("行业轮动分析")
        print("=" * 60)
        print(f"分析日期: {latest_date}")

        signal = analyzer.get_rotation_signal(test_data, latest_date)

        print(f"\n建议: {signal['reason']}")
        print("\n行业排名:")
        for i, (sector, score, perf) in enumerate(signal.get('rankings', []), 1):
            print(f"  {i}. {sector}: 得分={score:.4f}, "
                  f"收益率={perf['return']:.2%}, 动量={perf['momentum']:.4f}")
