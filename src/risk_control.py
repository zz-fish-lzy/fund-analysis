"""
风险控制模块
包含止损机制、仓位管理、风险评估
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class RiskConfig:
    """风险配置"""
    # 止损配置
    stop_loss_pct: float = -0.08  # 单只基金止损线 (-8%)
    trailing_stop_pct: float = -0.05  # 移动止损 (-5%)
    portfolio_stop_loss: float = -0.15  # 组合止损 (-15%)

    # 动态止损配置
    base_stop_loss: float = -0.08  # 基础止损线
    volatility_multiplier: float = 2.0  # 波动率倍数（止损 = 波动率 * 倍数）
    min_stop_loss: float = -0.05  # 最小止损线
    max_stop_loss: float = -0.15  # 最大止损线

    # 仓位管理
    max_position_pct: float = 0.30  # 单只基金最大仓位 (30%)
    min_position_pct: float = 0.05  # 单只基金最小仓位 (5%)
    max_total_position: float = 0.95  # 最大总仓位 (95%)
    min_cash_ratio: float = 0.05  # 最小现金比例 (5%)

    # 分散化配置
    max_funds: int = 10  # 最大持有基金数量
    min_funds: int = 3  # 最小持有基金数量

    # 波动率控制
    max_volatility: float = 0.25  # 最大允许波动率
    volatility_lookback: int = 20  # 波动率计算回溯期


class RiskController:
    """风险控制器"""

    def __init__(self, config: RiskConfig = None):
        self.config = config or RiskConfig()
        self.positions = {}  # {fund_code: {'shares': 0, 'cost': 0, 'highest_nav': 0}}
        self.trade_history = []
        self.daily_values = []

    def calculate_dynamic_stop_loss(self, fund_code: str, recent_returns: list) -> float:
        """
        计算动态止损线（基于波动率）

        Args:
            fund_code: 基金代码
            recent_returns: 近期收益率列表

        Returns:
            float: 动态止损线（负数）
        """
        if not recent_returns or len(recent_returns) < 5:
            return self.config.base_stop_loss

        # 计算波动率
        volatility = np.std(recent_returns) * np.sqrt(252)

        # 动态止损 = 基础止损 - 波动率 * 倍数
        dynamic_stop = self.config.base_stop_loss - (volatility * self.config.volatility_multiplier)

        # 限制在合理范围内
        dynamic_stop = max(dynamic_stop, self.config.max_stop_loss)  # 不超过最大止损
        dynamic_stop = min(dynamic_stop, self.config.min_stop_loss)  # 不低于最小止损

        return dynamic_stop

    def check_stop_loss(self, fund_code: str, current_nav: float, cost_nav: float,
                        recent_returns: list = None) -> dict:
        """
        检查止损条件（支持动态止损）

        Args:
            fund_code: 基金代码
            current_nav: 当前净值
            cost_nav: 成本净值
            recent_returns: 近期收益率列表（用于动态止损）

        Returns:
            dict: {'should_stop': bool, 'reason': str}
        """
        # 计算收益率
        return_pct = (current_nav - cost_nav) / cost_nav

        # 动态止损（如果有近期收益率数据）
        if recent_returns and len(recent_returns) >= 5:
            stop_loss_line = self.calculate_dynamic_stop_loss(fund_code, recent_returns)
            stop_type = "动态止损"
        else:
            stop_loss_line = self.config.stop_loss_pct
            stop_type = "固定止损"

        if return_pct <= stop_loss_line:
            return {
                'should_stop': True,
                'reason': f'触发{stop_type}: {return_pct:.2%} <= {stop_loss_line:.2%}'
            }

        # 移动止损（如果有持仓记录）
        if fund_code in self.positions:
            highest_nav = self.positions[fund_code].get('highest_nav', cost_nav)
            if highest_nav > 0:
                drawdown = (current_nav - highest_nav) / highest_nav
                if drawdown <= self.config.trailing_stop_pct:
                    return {
                        'should_stop': True,
                        'reason': f'触发移动止损: 从最高点回撤 {drawdown:.2%} <= {self.config.trailing_stop_pct:.2%}'
                    }

        return {'should_stop': False, 'reason': ''}

    def calculate_position_size(self, fund_code: str, signal_strength: float,
                                portfolio_value: float, current_nav: float) -> float:
        """
        计算仓位大小（凯利公式改进版）

        Args:
            fund_code: 基金代码
            signal_strength: 信号强度 (0-1)
            portfolio_value: 组合总价值
            current_nav: 当前净值

        Returns:
            float: 建议买入金额
        """
        # 基础仓位（基于信号强度）
        base_position = signal_strength * self.config.max_position_pct

        # 考虑当前持仓
        current_position_value = 0
        if fund_code in self.positions:
            current_position_value = self.positions[fund_code]['shares'] * current_nav

        # 计算可用仓位
        available_position = portfolio_value * self.config.max_position_pct - current_position_value

        # 考虑总仓位限制
        total_position_value = sum(
            pos['shares'] * current_nav
            for pos in self.positions.values()
        )
        max_additional = portfolio_value * self.config.max_total_position - total_position_value

        # 取较小值
        position_size = min(
            base_position * portfolio_value,
            available_position,
            max_additional
        )

        # 确保不超过最小仓位
        if position_size < portfolio_value * self.config.min_position_pct:
            position_size = 0

        return max(0, position_size)

    def check_portfolio_risk(self, portfolio_value: float, daily_returns: List[float]) -> dict:
        """
        检查组合风险

        Args:
            portfolio_value: 组合总价值
            daily_returns: 每日收益率列表

        Returns:
            dict: 风险评估结果
        """
        if len(daily_returns) < 2:
            return {'risk_level': 'unknown', 'warnings': []}

        returns_array = np.array(daily_returns)

        # 计算波动率
        volatility = np.std(returns_array) * np.sqrt(252)

        # 计算最大回撤
        cumulative = (1 + returns_array).cumprod()
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = np.min(drawdown)

        # 计算夏普比率
        risk_free_rate = 0.03 / 252
        excess_returns = returns_array - risk_free_rate
        sharpe_ratio = np.sqrt(252) * np.mean(excess_returns) / np.std(excess_returns) if np.std(excess_returns) > 0 else 0

        # 风险评级
        warnings = []
        risk_level = 'low'

        if volatility > self.config.max_volatility:
            warnings.append(f'波动率过高: {volatility:.2%} > {self.config.max_volatility:.2%}')
            risk_level = 'high'

        if max_drawdown < self.config.portfolio_stop_loss:
            warnings.append(f'最大回撤超过阈值: {max_drawdown:.2%} < {self.config.portfolio_stop_loss:.2%}')
            risk_level = 'critical'

        if sharpe_ratio < 0.5:
            warnings.append(f'夏普比率过低: {sharpe_ratio:.2f}')
            if risk_level == 'low':
                risk_level = 'medium'

        return {
            'risk_level': risk_level,
            'volatility': volatility,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'warnings': warnings
        }

    def adjust_for_diversification(self, target_positions: Dict[str, float],
                                   portfolio_value: float) -> Dict[str, float]:
        """
        调整仓位以实现分散化

        Args:
            target_positions: 目标仓位 {fund_code: weight}
            portfolio_value: 组合总价值

        Returns:
            dict: 调整后的仓位权重
        """
        # 限制基金数量
        if len(target_positions) > self.config.max_funds:
            # 按权重排序，保留前N个
            sorted_positions = sorted(target_positions.items(),
                                    key=lambda x: x[1], reverse=True)
            target_positions = dict(sorted_positions[:self.config.max_funds])

        # 归一化权重
        total_weight = sum(target_positions.values())
        if total_weight > self.config.max_total_position:
            scale_factor = self.config.max_total_position / total_weight
            target_positions = {k: v * scale_factor for k, v in target_positions.items()}

        # 确保单只基金不超过最大仓位
        for fund_code in target_positions:
            if target_positions[fund_code] > self.config.max_position_pct:
                target_positions[fund_code] = self.config.max_position_pct

        return target_positions

    def update_position(self, fund_code: str, shares: float, nav: float, trade_type: str):
        """
        更新持仓记录

        Args:
            fund_code: 基金代码
            shares: 份额变化
            nav: 当前净值
            trade_type: 交易类型 ('buy' or 'sell')
        """
        if fund_code not in self.positions:
            self.positions[fund_code] = {
                'shares': 0,
                'cost': 0,
                'highest_nav': nav
            }

        pos = self.positions[fund_code]

        if trade_type == 'buy':
            # 买入
            total_cost = pos['shares'] * pos['cost'] + shares * nav
            pos['shares'] += shares
            pos['cost'] = total_cost / pos['shares'] if pos['shares'] > 0 else 0
        else:
            # 卖出
            pos['shares'] -= shares
            if pos['shares'] <= 0:
                del self.positions[fund_code]
                return

        # 更新最高净值
        if nav > pos.get('highest_nav', 0):
            pos['highest_nav'] = nav

        # 记录交易
        self.trade_history.append({
            'fund_code': fund_code,
            'type': trade_type,
            'shares': shares,
            'nav': nav,
            'timestamp': pd.Timestamp.now()
        })

    def get_position_summary(self, current_navs: Dict[str, float]) -> dict:
        """
        获取持仓摘要

        Args:
            current_navs: 当前净值 {fund_code: nav}

        Returns:
            dict: 持仓摘要
        """
        total_value = 0
        position_details = []

        for fund_code, pos in self.positions.items():
            if fund_code in current_navs:
                current_nav = current_navs[fund_code]
                market_value = pos['shares'] * current_nav
                cost_value = pos['shares'] * pos['cost']
                pnl = market_value - cost_value
                pnl_pct = pnl / cost_value if cost_value > 0 else 0

                position_details.append({
                    'fund_code': fund_code,
                    'shares': pos['shares'],
                    'cost': pos['cost'],
                    'current_nav': current_nav,
                    'market_value': market_value,
                    'pnl': pnl,
                    'pnl_pct': pnl_pct
                })

                total_value += market_value

        # 计算仓位权重
        for detail in position_details:
            detail['weight'] = detail['market_value'] / total_value if total_value > 0 else 0

        return {
            'total_value': total_value,
            'position_count': len(position_details),
            'positions': position_details
        }


class PositionSizer:
    """仓位管理器"""

    def __init__(self, risk_config: RiskConfig = None):
        self.config = risk_config or RiskConfig()

    def equal_weight(self, fund_codes: List[str]) -> Dict[str, float]:
        """
        等权重分配

        Args:
            fund_codes: 基金代码列表

        Returns:
            dict: 权重分配
        """
        n = len(fund_codes)
        if n == 0:
            return {}

        weight = min(1.0 / n, self.config.max_position_pct)
        return {code: weight for code in fund_codes}

    def signal_based(self, signals: Dict[str, float]) -> Dict[str, float]:
        """
        基于信号强度分配

        Args:
            signals: 信号强度 {fund_code: strength (0-1)}

        Returns:
            dict: 权重分配
        """
        if not signals:
            return {}

        # 归一化信号
        total_signal = sum(signals.values())
        if total_signal == 0:
            return {}

        weights = {}
        for code, strength in signals.items():
            weight = (strength / total_signal) * self.config.max_total_position
            weights[code] = min(weight, self.config.max_position_pct)

        return weights

    def risk_parity(self, volatilities: Dict[str, float]) -> Dict[str, float]:
        """
        风险平价分配

        Args:
            volatilities: 波动率 {fund_code: vol}

        Returns:
            dict: 权重分配
        """
        if not volatilities:
            return {}

        # 计算逆波动率权重
        inv_vols = {code: 1.0 / vol if vol > 0 else 0 for code, vol in volatilities.items()}
        total_inv_vol = sum(inv_vols.values())

        if total_inv_vol == 0:
            return {}

        weights = {}
        for code, inv_vol in inv_vols.items():
            weight = (inv_vol / total_inv_vol) * self.config.max_total_position
            weights[code] = min(weight, self.config.max_position_pct)

        return weights


# 使用示例
if __name__ == '__main__':
    # 创建风险控制器
    config = RiskConfig(
        stop_loss_pct=-0.08,
        trailing_stop_pct=-0.05,
        max_position_pct=0.30,
        max_funds=10
    )

    controller = RiskController(config)

    # 模拟持仓
    controller.update_position('510300', 1000, 4.5, 'buy')
    controller.update_position('159915', 500, 2.8, 'buy')

    # 检查止损
    result = controller.check_stop_loss('510300', 4.1, 4.5)
    print(f"止损检查: {result}")

    # 获取持仓摘要
    navs = {'510300': 4.2, '159915': 3.0}
    summary = controller.get_position_summary(navs)
    print(f"持仓摘要: {summary}")
