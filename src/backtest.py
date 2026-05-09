"""
回测模块
模拟实战交易
"""

import numpy as np
import pandas as pd
import os
from datetime import datetime


class Backtester:
    """回测器"""
    def __init__(self, initial_capital=100000, commission=0.001):
        """
        Args:
            initial_capital: 初始资金
            commission: 手续费率
        """
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.commission = commission

        # 持仓
        self.positions = {}  # {fund_code: {'shares': 0, 'cost': 0}}

        # 交易记录
        self.trades = []

        # 每日净值
        self.daily_values = []

    def buy(self, fund_code, price, amount):
        """
        买入基金

        Args:
            fund_code: 基金代码
            price: 价格
            amount: 金额
        """
        # 计算手续费
        commission = amount * self.commission
        total_cost = amount + commission

        # 检查资金
        if total_cost > self.capital:
            print(f"资金不足: 需要 {total_cost:.2f}, 可用 {self.capital:.2f}")
            return False

        # 计算份额
        shares = amount / price

        # 更新持仓
        if fund_code not in self.positions:
            self.positions[fund_code] = {'shares': 0, 'cost': 0}

        self.positions[fund_code]['shares'] += shares
        self.positions[fund_code]['cost'] += amount

        # 更新资金
        self.capital -= total_cost

        # 记录交易
        self.trades.append({
            'date': datetime.now(),
            'type': 'buy',
            'fund_code': fund_code,
            'price': price,
            'shares': shares,
            'amount': amount,
            'commission': commission
        })

        return True

    def sell(self, fund_code, price, shares=None):
        """
        卖出基金

        Args:
            fund_code: 基金代码
            price: 价格
            shares: 份额（None表示全部卖出）
        """
        if fund_code not in self.positions:
            print(f"未持有基金: {fund_code}")
            return False

        position = self.positions[fund_code]

        if shares is None:
            shares = position['shares']

        if shares > position['shares']:
            print(f"份额不足: 持有 {position['shares']:.2f}, 卖出 {shares:.2f}")
            return False

        # 计算金额
        amount = shares * price
        commission = amount * self.commission
        net_amount = amount - commission

        # 更新持仓
        position['shares'] -= shares
        if position['shares'] == 0:
            del self.positions[fund_code]

        # 更新资金
        self.capital += net_amount

        # 记录交易
        self.trades.append({
            'date': datetime.now(),
            'type': 'sell',
            'fund_code': fund_code,
            'price': price,
            'shares': shares,
            'amount': amount,
            'commission': commission
        })

        return True

    def get_portfolio_value(self, prices):
        """
        计算投资组合总价值

        Args:
            prices: 当前价格 {fund_code: price}

        Returns:
            float: 总价值
        """
        holdings_value = 0

        for fund_code, position in self.positions.items():
            if fund_code in prices:
                holdings_value += position['shares'] * prices[fund_code]

        return self.capital + holdings_value

    def record_daily(self, date, prices):
        """
        记录每日净值

        Args:
            date: 日期
            prices: 当前价格 {fund_code: price}
        """
        portfolio_value = self.get_portfolio_value(prices)

        self.daily_values.append({
            'date': date,
            'portfolio_value': portfolio_value,
            'capital': self.capital,
            'holdings_value': portfolio_value - self.capital
        })

    def calculate_metrics(self):
        """
        计算回测指标

        Returns:
            dict: 回测指标
        """
        if not self.daily_values:
            return {}

        df = pd.DataFrame(self.daily_values)

        # 计算收益率
        df['returns'] = df['portfolio_value'].pct_change()

        # 总收益率
        total_return = (df['portfolio_value'].iloc[-1] / self.initial_capital) - 1

        # 年化收益率
        days = (df['date'].iloc[-1] - df['date'].iloc[0]).days
        annual_return = (1 + total_return) ** (365 / days) - 1 if days > 0 else 0

        # 夏普比率（假设无风险利率为3%）
        risk_free_rate = 0.03
        excess_returns = df['returns'] - risk_free_rate / 252
        sharpe_ratio = np.sqrt(252) * excess_returns.mean() / excess_returns.std() if excess_returns.std() > 0 else 0

        # 最大回撤
        cumulative = (1 + df['returns']).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()

        # 胜率
        winning_trades = len([t for t in self.trades if t['type'] == 'sell' and t['amount'] > 0])
        total_trades = len([t for t in self.trades if t['type'] == 'sell'])
        win_rate = winning_trades / total_trades if total_trades > 0 else 0

        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'total_trades': len(self.trades),
            'final_value': df['portfolio_value'].iloc[-1]
        }


def run_backtest_single(fund_code, test_data_dir='data/raw/test',
                        model_dir='models', initial_capital=100000):
    """
    运行单只基金的回测

    Args:
        fund_code: 基金代码
        test_data_dir: 测试数据目录
        model_dir: 模型目录
        initial_capital: 初始资金

    Returns:
        dict: 回测结果
    """
    from src.model_lstm import LSTMPredictor
    from src.model_lgbm import LGBMPredictor
    import joblib

    # 加载测试数据
    test_file = f"{test_data_dir}/features_{fund_code}.csv"
    if not os.path.exists(test_file):
        print(f"测试数据不存在: {test_file}")
        return None

    df = pd.read_csv(test_file)
    df['date'] = pd.to_datetime(df['date'])

    # 特征列
    feature_cols = [
        'MA5', 'MA10', 'MA20', 'MA60',
        'MA5_bias', 'MA20_bias',
        'RSI', 'MACD', 'MACD_signal', 'MACD_hist',
        'BB_width', 'BB_position',
        'volatility_5', 'volatility_20',
        'momentum_5', 'momentum_10', 'momentum_20',
        'position_5', 'position_20'
    ]

    # 加载模型
    lstm_model = LSTMPredictor(input_size=len(feature_cols))
    lstm_model.load(f"{model_dir}/lstm/model_{fund_code}.pt")

    lgbm_model = LGBMPredictor(task='classification')
    lgbm_model.load(f"{model_dir}/lgbm/model_{fund_code}.pkl")

    scaler = joblib.load(f"{model_dir}/lstm/scaler_{fund_code}.pkl")

    # 创建回测器
    backtester = Backtester(initial_capital=initial_capital)

    # 运行回测
    print(f"\n运行回测: {fund_code}")

    for i in range(20, len(df)):
        # 获取特征
        X = df[feature_cols].iloc[i-20:i].values
        X_scaled = scaler.transform(X)

        # LSTM预测
        lstm_pred = lstm_model.predict(X_scaled)
        lstm_signal = 1 if lstm_pred[0] > 0.5 else 0

        # LightGBM预测
        lgbm_pred = lgbm_model.predict_direction(X_scaled[-1:])
        lgbm_signal = lgbm_pred[0]

        # 集成信号（投票）
        signal = 1 if (lstm_signal + lgbm_signal) >= 1 else 0

        # 获取当前价格
        current_price = df['nav'].iloc[i]

        # 执行交易
        if signal == 1 and fund_code not in backtester.positions:
            # 买入信号
            backtester.buy(fund_code, current_price, backtester.capital * 0.9)
        elif signal == 0 and fund_code in backtester.positions:
            # 卖出信号
            backtester.sell(fund_code, current_price)

        # 记录每日净值
        backtester.record_daily(df['date'].iloc[i], {fund_code: current_price})

    # 计算指标
    metrics = backtester.calculate_metrics()

    return metrics


def run_backtest_all(test_data_dir='data/raw/test', model_dir='models',
                     initial_capital=100000):
    """
    运行所有基金的回测

    Args:
        test_data_dir: 测试数据目录
        model_dir: 模型目录
        initial_capital: 初始资金

    Returns:
        dict: 回测结果
    """
    from src.data_loader import FUND_POOL

    all_results = {}

    for fund_code in FUND_POOL.keys():
        try:
            result = run_backtest_single(
                fund_code, test_data_dir, model_dir, initial_capital
            )
            if result:
                all_results[fund_code] = result
        except Exception as e:
            print(f"回测基金 {fund_code} 失败: {e}")

    # 汇总
    print("\n" + "=" * 60)
    print("回测结果汇总")
    print("=" * 60)

    for fund_code, result in all_results.items():
        print(f"\n{fund_code}:")
        print(f"  总收益率: {result['total_return']:.2%}")
        print(f"  夏普比率: {result['sharpe_ratio']:.2f}")
        print(f"  最大回撤: {result['max_drawdown']:.2%}")
        print(f"  胜率: {result['win_rate']:.2%}")

    return all_results


if __name__ == '__main__':
    # 测试回测
    results = run_backtest_all()
