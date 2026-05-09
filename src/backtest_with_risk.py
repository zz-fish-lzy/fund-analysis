"""
带风险控制的回测模块
集成止损机制和仓位管理
"""

import numpy as np
import pandas as pd
import os
import joblib
from datetime import datetime
from src.risk_control import RiskController, RiskConfig, PositionSizer
from src.model_lgbm import LGBMPredictor
from src.industry_rotation import IndustryRotationAnalyzer


class RiskManagedBacktester:
    """带风险控制的回测器"""

    def __init__(self, initial_capital=100000, risk_config=None):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.risk_config = risk_config or RiskConfig()
        self.risk_controller = RiskController(self.risk_config)
        self.position_sizer = PositionSizer(self.risk_config)
        self.rotation_analyzer = IndustryRotationAnalyzer(lookback_days=20)

        self.trades = []
        self.daily_values = []
        self.daily_returns = []

    def _extract_features(self, df, idx):
        """提取特征"""
        try:
            if idx < 20:
                return None, None

            recent = df.iloc[idx-19:idx+1]
            navs = recent['nav'].values
            ma5 = np.mean(navs[-5:])
            ma10 = np.mean(navs[-10:])
            ma20 = np.mean(navs)
            nav = navs[-1]

            # 计算近期收益率（用于动态止损）
            returns = np.diff(navs) / navs[:-1]

            features = np.array([[
                ma5, ma10, ma20, ma20,
                (nav - ma5) / ma5 if ma5 > 0 else 0,
                (nav - ma20) / ma20 if ma20 > 0 else 0,
                50, 0, 0, 0,
                0, 0.5,
                np.std(navs[-5:]) if len(navs) >= 5 else 0,
                np.std(navs) if len(navs) > 0 else 0,
                (nav - navs[-5]) / navs[-5] if len(navs) >= 5 and navs[-5] > 0 else 0,
                (nav - navs[-10]) / navs[-10] if len(navs) >= 10 and navs[-10] > 0 else 0,
                (nav - navs[0]) / navs[0] if navs[0] > 0 else 0,
                (nav - min(navs[-5:])) / (max(navs[-5:]) - min(navs[-5:])) if max(navs[-5:]) != min(navs[-5:]) else 0.5,
                (nav - min(navs)) / (max(navs) - min(navs)) if max(navs) != min(navs) else 0.5
            ]])

            return features, returns.tolist()
        except:
            return None, None

    def _execute_buy(self, fund_code, shares, nav):
        """执行买入"""
        cost = shares * nav
        if cost > self.capital:
            shares = self.capital * 0.95 / nav
            cost = shares * nav

        if cost > 0:
            self.capital -= cost
            self.risk_controller.update_position(fund_code, shares, nav, 'buy')
            self.trades.append({
                'date': self.daily_values[-1]['date'] if self.daily_values else datetime.now(),
                'type': 'buy',
                'fund_code': fund_code,
                'shares': shares,
                'nav': nav,
                'cost': cost
            })

    def _execute_sell(self, fund_code, nav, reason=''):
        """执行卖出"""
        if fund_code in self.risk_controller.positions:
            shares = self.risk_controller.positions[fund_code]['shares']
            revenue = shares * nav * 0.999

            self.capital += revenue
            self.risk_controller.update_position(fund_code, shares, nav, 'sell')
            self.trades.append({
                'date': self.daily_values[-1]['date'] if self.daily_values else datetime.now(),
                'type': 'sell',
                'fund_code': fund_code,
                'shares': shares,
                'nav': nav,
                'revenue': revenue,
                'reason': reason
            })

    def _calculate_portfolio_value(self, navs):
        """计算组合总价值"""
        holdings_value = 0
        for fund_code, pos in self.risk_controller.positions.items():
            if fund_code in navs:
                holdings_value += pos['shares'] * navs[fund_code]
        return self.capital + holdings_value

    def run_backtest(self, fund_data, models, scalers, feature_cols=None):
        """运行带风险控制的回测"""
        print("=" * 60)
        print("带风险控制的回测")
        print("=" * 60)

        # 获取所有日期
        all_dates = set()
        for df in fund_data.values():
            all_dates.update(df['date'].tolist())
        all_dates = sorted(all_dates)

        print(f"回测期间: {all_dates[0]} 到 {all_dates[-1]}")
        print(f"交易日数: {len(all_dates)}")

        # 存储近期收益率用于动态止损
        fund_recent_returns = {fc: [] for fc in fund_data.keys()}

        for date in all_dates:
            signals = {}
            navs = {}

            for fund_code, df in fund_data.items():
                if date in df['date'].values:
                    row = df[df['date'] == date].iloc[0]
                    navs[fund_code] = row['nav']

                    if fund_code in models and fund_code in scalers:
                        idx = df[df['date'] == date].index[0]
                        features, recent_returns = self._extract_features(df, idx)
                        if features is not None:
                            features_scaled = scalers[fund_code].transform(features)
                            pred = models[fund_code].predict(features_scaled)
                            signals[fund_code] = pred[0] if len(pred) > 0 else 0
                            # 更新近期收益率
                            if recent_returns:
                                fund_recent_returns[fund_code] = recent_returns

            # 检查止损（支持动态止损）
            positions_to_sell = []
            for fund_code, pos in self.risk_controller.positions.items():
                if fund_code in navs:
                    current_nav = navs[fund_code]
                    cost_nav = pos['cost']
                    recent_rets = fund_recent_returns.get(fund_code, [])
                    stop_result = self.risk_controller.check_stop_loss(
                        fund_code, current_nav, cost_nav, recent_returns=recent_rets)
                    if stop_result['should_stop']:
                        positions_to_sell.append({'fund_code': fund_code, 'reason': stop_result['reason']})

            for item in positions_to_sell:
                if item['fund_code'] in navs:
                    self._execute_sell(item['fund_code'], navs[item['fund_code']], reason=item['reason'])

            # 获取行业轮动得分
            rotation_scores = {}
            for fund_code in signals:
                rotation_scores[fund_code] = self.rotation_analyzer.get_fund_rotation_score(
                    fund_code, fund_data, date)

            # 根据信号调仓（结合行业轮动）
            target_positions = {}
            for fund_code, signal in signals.items():
                if signal > 0.5 and fund_code in navs:
                    # 基础信号权重
                    base_weight = signal - 0.5
                    # 行业轮动调整（0.8-1.2倍）
                    rotation_factor = 0.8 + rotation_scores.get(fund_code, 0.5) * 0.4
                    adjusted_weight = base_weight * rotation_factor
                    target_positions[fund_code] = adjusted_weight

            if target_positions:
                target_positions = self.risk_controller.adjust_for_diversification(
                    target_positions, self._calculate_portfolio_value(navs))

            current_positions = set(self.risk_controller.positions.keys())
            target_funds = set(target_positions.keys())

            for fund_code in current_positions - target_funds:
                if fund_code in navs:
                    self._execute_sell(fund_code, navs[fund_code], reason='信号转弱')

            for fund_code, weight in target_positions.items():
                if fund_code in navs:
                    portfolio_value = self._calculate_portfolio_value(navs)
                    target_value = portfolio_value * weight
                    current_value = 0
                    if fund_code in self.risk_controller.positions:
                        current_value = self.risk_controller.positions[fund_code]['shares'] * navs[fund_code]
                    if target_value > current_value * 1.1:
                        buy_value = target_value - current_value
                        buy_shares = buy_value / navs[fund_code]
                        self._execute_buy(fund_code, buy_shares, navs[fund_code])

            portfolio_value = self._calculate_portfolio_value(navs)
            self.daily_values.append({'date': date, 'portfolio_value': portfolio_value, 'capital': self.capital})

            if len(self.daily_values) > 1:
                prev_value = self.daily_values[-2]['portfolio_value']
                daily_return = (portfolio_value - prev_value) / prev_value
                self.daily_returns.append(daily_return)

        return self._calculate_metrics()

    def _calculate_metrics(self):
        """计算回测指标"""
        if not self.daily_values:
            return {}

        df = pd.DataFrame(self.daily_values)
        total_return = (df['portfolio_value'].iloc[-1] / self.initial_capital) - 1

        days = (df['date'].iloc[-1] - df['date'].iloc[0]).days
        annual_return = (1 + total_return) ** (365 / days) - 1 if days > 0 else 0

        if self.daily_returns:
            returns_array = np.array(self.daily_returns)
            risk_free_rate = 0.03 / 252
            excess_returns = returns_array - risk_free_rate
            sharpe_ratio = np.sqrt(252) * np.mean(excess_returns) / np.std(excess_returns) if np.std(excess_returns) > 0 else 0
        else:
            sharpe_ratio = 0

        values = df['portfolio_value'].values
        running_max = np.maximum.accumulate(values)
        drawdown = (values - running_max) / running_max
        max_drawdown = np.min(drawdown)

        sell_trades = [t for t in self.trades if t['type'] == 'sell']
        if sell_trades:
            winning_trades = sum(1 for t in sell_trades if t.get('revenue', 0) > t.get('cost', 0))
            win_rate = winning_trades / len(sell_trades)
        else:
            win_rate = 0

        risk_assessment = self.risk_controller.check_portfolio_risk(
            df['portfolio_value'].iloc[-1], self.daily_returns)

        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'trade_count': len(self.trades),
            'final_value': df['portfolio_value'].iloc[-1],
            'risk_assessment': risk_assessment
        }


if __name__ == '__main__':
    risk_config = RiskConfig(
        stop_loss_pct=-0.08,
        trailing_stop_pct=-0.05,
        max_position_pct=0.25,
        max_funds=8,
        min_cash_ratio=0.10
    )

    train_dir = 'data/raw/train'
    test_dir = 'data/raw/test'

    fund_data = {}
    models = {}
    scalers = {}

    for filename in os.listdir(test_dir):
        if filename.endswith('.csv'):
            fund_code = filename.replace('fund_', '').replace('.csv', '')
            df = pd.read_csv(os.path.join(test_dir, filename))
            df['date'] = pd.to_datetime(df['date'])
            fund_data[fund_code] = df

            model_path = f'models/lgbm/model_{fund_code}.pkl'
            scaler_path = f'models/lgbm/scaler_{fund_code}.pkl'
            if os.path.exists(model_path) and os.path.exists(scaler_path):
                model = LGBMPredictor(task='classification')
                model.load(model_path)
                models[fund_code] = model
                scalers[fund_code] = joblib.load(scaler_path)

    print(f"加载了 {len(fund_data)} 只基金数据")

    backtester = RiskManagedBacktester(initial_capital=100000, risk_config=risk_config)
    results = backtester.run_backtest(fund_data, models, scalers)

    print("\n" + "=" * 60)
    print("回测结果")
    print("=" * 60)
    print(f"  总收益率: {results['total_return']:+.2%}")
    print(f"  年化收益率: {results['annual_return']:+.2%}")
    print(f"  夏普比率: {results['sharpe_ratio']:.2f}")
    print(f"  最大回撤: {results['max_drawdown']:.2%}")
    print(f"  胜率: {results['win_rate']:.2%}")
    print(f"  交易次数: {results['trade_count']}")
    print(f"  最终资金: {results['final_value']:,.2f}元")
