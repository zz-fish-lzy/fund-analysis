"""
LightGBM模型模块
用于基金涨跌预测
"""

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, mean_squared_error
import joblib


def calculate_backtest_metrics(prices, predictions=None, risk_free_rate=0.02):
    """
    计算回测收益指标和风险控制指标（实施方案第二/三维度）

    第二维度：累计收益率、年化收益率、超额收益率、信息比率
    第三维度：最大回撤、夏普比率、卡尔马比率、索提诺比率

    Args:
        prices: 实际净值/价格序列
        predictions: 预测方向（可选，用于计算策略收益）
        risk_free_rate: 无风险利率（年化）

    Returns:
        dict: 回测指标
    """
    prices = np.array(prices, dtype=float)
    returns = np.diff(prices) / prices[:-1]

    # ── 第二维度：收益指标 ──

    # 累计收益率
    total_return = (prices[-1] / prices[0]) - 1

    # 年化收益率
    n_days = len(prices)
    annual_return = (1 + total_return) ** (250 / n_days) - 1 if n_days > 0 else 0

    # 如果有预测，计算策略收益
    strategy_return = total_return
    excess_return = 0
    information_ratio = 0

    if predictions is not None and len(predictions) >= len(returns):
        # 策略：预测涨则持有，预测跌则空仓
        pred_dir = np.sign(predictions[:len(returns)])
        strategy_returns = returns * np.maximum(pred_dir, 0)  # 只做多
        strategy_total = np.prod(1 + strategy_returns) - 1

        # 超额收益率（策略 vs 买入持有）
        excess_return = strategy_total - total_return

        # 信息比率
        excess_daily = strategy_returns - returns
        if excess_daily.std() > 0:
            information_ratio = (excess_daily.mean() * 250) / (excess_daily.std() * np.sqrt(250))

        strategy_return = strategy_total

    # ── 第三维度：风险指标 ──

    # 最大回撤
    cumulative = np.cumprod(1 + returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = abs(drawdown.min())

    # 夏普比率
    excess_returns = returns - risk_free_rate / 250
    if excess_returns.std() > 0:
        sharpe_ratio = (excess_returns.mean() * 250) / (excess_returns.std() * np.sqrt(250))
    else:
        sharpe_ratio = 0

    # 卡尔马比率
    calmar_ratio = annual_return / max_drawdown if max_drawdown > 0 else 0

    # 索提诺比率（只考虑下行风险）
    downside_returns = returns[returns < 0]
    if len(downside_returns) > 0 and downside_returns.std() > 0:
        sortino_ratio = (excess_returns.mean() * 250) / (downside_returns.std() * np.sqrt(250))
    else:
        sortino_ratio = 0

    return {
        'total_return': total_return,
        'annual_return': annual_return,
        'strategy_return': strategy_return,
        'excess_return': excess_return,
        'information_ratio': information_ratio,
        'max_drawdown': max_drawdown,
        'sharpe_ratio': sharpe_ratio,
        'calmar_ratio': calmar_ratio,
        'sortino_ratio': sortino_ratio,
    }


class LGBMPredictor:
    """LightGBM预测器封装类"""
    def __init__(self, task='classification'):
        """
        Args:
            task: 'classification' (涨跌预测) 或 'regression' (收益率预测)
        """
        self.task = task
        self.model = None
        self.feature_importance = None

        # 默认参数
        if task == 'classification':
            self.params = {
                'objective': 'binary',
                'metric': 'binary_logloss',
                'boosting_type': 'gbdt',
                'num_leaves': 31,
                'learning_rate': 0.05,
                'feature_fraction': 0.9,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'verbose': -1,
                'n_jobs': -1,
                'seed': 42
            }
        else:
            self.params = {
                'objective': 'regression',
                'metric': 'rmse',
                'boosting_type': 'gbdt',
                'num_leaves': 31,
                'learning_rate': 0.05,
                'feature_fraction': 0.9,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'verbose': -1,
                'n_jobs': -1,
                'seed': 42
            }

    def train(self, X_train, y_train, X_val=None, y_val=None,
              num_boost_round=1000, early_stopping_rounds=50, verbose=True):
        """
        训练模型

        Args:
            X_train: 训练特征
            y_train: 训练目标
            X_val: 验证特征
            y_val: 验证目标
            num_boost_round: 最大迭代次数
            early_stopping_rounds: 早停轮数
            verbose: 是否显示训练过程
        """
        # 创建数据集
        train_data = lgb.Dataset(X_train, label=y_train)

        valid_sets = [train_data]
        valid_names = ['train']

        if X_val is not None and y_val is not None:
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
            valid_sets.append(val_data)
            valid_names.append('valid')

        # 训练模型
        callbacks = []
        if early_stopping_rounds:
            callbacks.append(lgb.early_stopping(stopping_rounds=early_stopping_rounds))
        if verbose:
            callbacks.append(lgb.log_evaluation(period=50))

        self.model = lgb.train(
            self.params,
            train_data,
            valid_sets=valid_sets,
            valid_names=valid_names,
            num_boost_round=num_boost_round,
            callbacks=callbacks
        )

        # 计算特征重要性
        self.feature_importance = pd.DataFrame({
            'feature': X_train.columns if hasattr(X_train, 'columns') else range(X_train.shape[1]),
            'importance': self.model.feature_importance(importance_type='gain')
        }).sort_values('importance', ascending=False)

        return self.model

    def predict(self, X):
        """
        预测

        Args:
            X: 输入特征

        Returns:
            numpy.ndarray: 预测结果
        """
        if self.model is None:
            raise ValueError("模型未训练")

        predictions = self.model.predict(X)

        if self.task == 'classification':
            # 返回概率
            return predictions
        else:
            return predictions

    def predict_direction(self, X, threshold=0.5):
        """
        预测涨跌方向

        Args:
            X: 输入特征
            threshold: 分类阈值

        Returns:
            numpy.ndarray: 预测方向 (1=涨, 0=跌)
        """
        if self.task != 'classification':
            raise ValueError("predict_direction仅适用于分类任务")

        probabilities = self.predict(X)
        return (probabilities >= threshold).astype(int)

    def evaluate(self, X_test, y_test, actual_prices=None):
        """
        评估模型（实施方案三维度指标体系）

        第一维度：预测性能 - MAE, RMSE, MAPE, 方向准确率
        第二维度：回测收益 - 需要实际价格序列（外部计算）
        第三维度：风险控制 - 需要实际价格序列（外部计算）

        Args:
            X_test: 测试特征
            y_test: 测试目标（方向标签或收益率）
            actual_prices: 实际净值序列（用于计算回测指标）

        Returns:
            dict: 评估指标
        """
        predictions = self.predict(X_test)

        if self.task == 'classification':
            pred_labels = (predictions >= 0.5).astype(int)
            accuracy = accuracy_score(y_test, pred_labels)

            # 方向准确率（分类任务的核心指标）
            direction_accuracy = accuracy

            return {
                'accuracy': accuracy,
                'direction_accuracy': direction_accuracy,
                'predictions': predictions,
                'pred_labels': pred_labels
            }
        else:
            # 回归任务：实施方案要求的预测性能指标
            rmse = np.sqrt(mean_squared_error(y_test, predictions))
            mae = np.mean(np.abs(y_test - predictions))

            # MAPE（平均绝对百分比误差，排除零值）
            non_zero_mask = y_test != 0
            if non_zero_mask.sum() > 0:
                mape = np.mean(np.abs((y_test[non_zero_mask] - predictions[non_zero_mask]) / y_test[non_zero_mask])) * 100
            else:
                mape = float('inf')

            # 方向准确率（预测涨跌方向是否正确）
            if len(y_test) > 1:
                actual_dir = np.sign(y_test)
                pred_dir = np.sign(predictions)
                direction_accuracy = np.mean(actual_dir == pred_dir)
            else:
                direction_accuracy = 0.5

            result = {
                'mae': mae,
                'rmse': rmse,
                'mape': mape,
                'direction_accuracy': direction_accuracy,
                'predictions': predictions,
            }

            # 如果提供了实际价格，计算回测指标
            if actual_prices is not None and len(actual_prices) > 1:
                result.update(calculate_backtest_metrics(actual_prices, predictions))

            return result

    def cross_validate(self, X, y, n_splits=5):
        """
        时间序列交叉验证

        Args:
            X: 特征
            y: 目标
            n_splits: 折数

        Returns:
            dict: 交叉验证结果
        """
        tscv = TimeSeriesSplit(n_splits=n_splits)
        scores = []

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            # 训练模型
            model = LGBMPredictor(task=self.task)
            model.train(X_train, y_train, X_val, y_val, verbose=False)

            # 评估
            result = model.evaluate(X_val, y_val)
            scores.append(result)

            if self.task == 'classification':
                print(f"Fold {fold+1}: Accuracy = {result['accuracy']:.4f}")
            else:
                print(f"Fold {fold+1}: RMSE = {result['rmse']:.6f}")

        # 汇总结果
        if self.task == 'classification':
            avg_score = np.mean([s['accuracy'] for s in scores])
            print(f"\n平均准确率: {avg_score:.4f}")
        else:
            avg_score = np.mean([s['rmse'] for s in scores])
            print(f"\n平均RMSE: {avg_score:.6f}")

        return {
            'scores': scores,
            'avg_score': avg_score
        }

    def get_feature_importance(self, top_n=20):
        """获取特征重要性"""
        if self.feature_importance is None:
            return None

        return self.feature_importance.head(top_n)

    def save(self, path):
        """保存模型"""
        if self.model is None:
            raise ValueError("模型未训练")

        joblib.dump(self.model, path)
        print(f"模型已保存: {path}")

    def load(self, path):
        """加载模型"""
        self.model = joblib.load(path)
        print(f"模型已加载: {path}")


def train_ensemble(X_train, y_train, X_val, y_val, task='classification'):
    """
    训练集成模型（多个LightGBM）

    Args:
        X_train: 训练特征
        y_train: 训练目标
        X_val: 验证特征
        y_val: 验证目标
        task: 任务类型

    Returns:
        list: 模型列表
    """
    models = []

    # 不同参数的模型
    param_sets = [
        {'num_leaves': 31, 'learning_rate': 0.05},
        {'num_leaves': 50, 'learning_rate': 0.03},
        {'num_leaves': 20, 'learning_rate': 0.08},
    ]

    for i, params in enumerate(param_sets):
        print(f"\n训练模型 {i+1}/{len(param_sets)}...")

        predictor = LGBMPredictor(task=task)
        predictor.params.update(params)

        predictor.train(X_train, y_train, X_val, y_val, verbose=False)
        models.append(predictor)

    return models


def ensemble_predict(models, X, task='classification'):
    """
    集成预测

    Args:
        models: 模型列表
        X: 输入特征
        task: 任务类型

    Returns:
        numpy.ndarray: 预测结果
    """
    predictions = []

    for model in models:
        pred = model.predict(X)
        predictions.append(pred)

    predictions = np.array(predictions)

    if task == 'classification':
        # 投票法
        avg_pred = np.mean(predictions, axis=0)
        return (avg_pred >= 0.5).astype(int), avg_pred
    else:
        # 平均法
        return np.mean(predictions, axis=0)


if __name__ == '__main__':
    # 测试LightGBM模型
    print("测试LightGBM模型...")

    # 生成测试数据
    np.random.seed(42)
    n_samples = 1000
    n_features = 19

    X = np.random.randn(n_samples, n_features)
    y = (X[:, 0] + X[:, 1] * 0.5 + np.random.randn(n_samples) * 0.1 > 0).astype(int)

    # 划分训练集和验证集
    split_idx = int(n_samples * 0.8)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    # 训练模型
    predictor = LGBMPredictor(task='classification')
    predictor.train(X_train, y_train, X_val, y_val)

    # 评估
    result = predictor.evaluate(X_val, y_val)
    print(f"\n验证集准确率: {result['accuracy']:.4f}")

    # 特征重要性
    print("\nTop 10 特征重要性:")
    print(predictor.get_feature_importance(10))
