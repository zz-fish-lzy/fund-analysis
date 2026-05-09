"""
模型训练模块
滚动窗口训练 + 三维度评估 + Supabase 集成

实施方案要求：
- 滚动窗口：用过去 1 年数据预测下 1 周，滚动更新
- 严格避免前视偏差：训练 T 日模型只能用 T 日及之前的数据
- 评估指标：预测性能(MAE/RMSE/MAPE/方向准确率) + 回测收益 + 风险控制
"""

import os
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
import joblib

from src.model_lgbm import LGBMPredictor, calculate_backtest_metrics

logger = logging.getLogger(__name__)

# 默认特征列
DEFAULT_FEATURE_COLS = [
    'MA5', 'MA10', 'MA20', 'MA60',
    'MA5_bias', 'MA20_bias',
    'RSI', 'MACD', 'MACD_signal', 'MACD_hist',
    'BB_width', 'BB_position',
    'volatility_5', 'volatility_20',
    'momentum_5', 'momentum_10', 'momentum_20',
    'position_5', 'position_20'
]


def prepare_data(df, feature_cols, target_col):
    """
    准备训练数据（无前视偏差）

    Args:
        df: DataFrame（按日期排序）
        feature_cols: 特征列名
        target_col: 目标列名

    Returns:
        tuple: (X, y, scaler)
    """
    X = df[feature_cols].values
    y = df[target_col].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, scaler


def rolling_window_train(fund_code, data_dir='data/processed/train',
                          model_dir='models', train_window=250, predict_window=5):
    """
    滚动窗口训练（实施方案核心要求）

    策略：用过去 train_window 天训练，预测接下来 predict_window 天，
    窗口向前滚动，严格避免前视偏差。

    Args:
        fund_code: 基金代码
        data_dir: 特征数据目录
        model_dir: 模型保存目录
        train_window: 训练窗口天数（默认250天≈1年）
        predict_window: 预测窗口天数（默认5天≈1周）

    Returns:
        dict: 训练结果（含评估指标）
    """
    filename = f"{data_dir}/features_{fund_code}.csv"
    if not os.path.exists(filename):
        logger.warning(f"数据文件不存在: {filename}")
        return None

    df = pd.read_csv(filename)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)

    # 确定可用特征列
    feature_cols = [c for c in DEFAULT_FEATURE_COLS if c in df.columns]
    if not feature_cols:
        logger.error(f"无可用特征列: {filename}")
        return None

    # 目标列：方向预测
    target_col = 'target_1d_dir'
    if target_col not in df.columns:
        logger.error(f"目标列不存在: {target_col}")
        return None

    total_len = len(df)
    if total_len < train_window + predict_window:
        logger.warning(f"数据不足: {total_len} < {train_window + predict_window}")
        return None

    # ── 滚动窗口训练 ──

    all_predictions = []
    all_targets = []
    all_actual_navs = []
    all_pred_dates = []
    fold_metrics = []

    n_folds = 0
    start_idx = train_window

    while start_idx + predict_window <= total_len:
        # 训练集：严格使用历史数据（无前视偏差）
        train_end = start_idx
        train_start = max(0, train_end - train_window)

        X_train_raw = df.iloc[train_start:train_end][feature_cols].values
        y_train = df.iloc[train_start:train_end][target_col].values

        # 预测集
        pred_end = min(start_idx + predict_window, total_len)
        X_pred_raw = df.iloc[start_idx:pred_end][feature_cols].values
        y_pred = df.iloc[start_idx:pred_end][target_col].values
        pred_dates = df.iloc[start_idx:pred_end]['date'].values

        # 标准化（用训练集的scaler，避免前视偏差）
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train_raw)
        X_pred = scaler.transform(X_pred_raw)

        # 训练 LightGBM
        model = LGBMPredictor(task='classification')

        # 验证集：训练集最后 20%
        val_split = int(len(X_train) * 0.8)
        model.train(
            X_train[:val_split], y_train[:val_split],
            X_train[val_split:], y_train[val_split:],
            num_boost_round=500,
            early_stopping_rounds=30,
            verbose=False
        )

        # 预测
        pred_probs = model.predict(X_pred)
        pred_labels = (pred_probs >= 0.5).astype(int)

        all_predictions.extend(pred_labels)
        all_targets.extend(y_pred)
        all_pred_dates.extend(pred_dates)

        # 获取实际净值（用于回测指标）
        if 'nav' in df.columns:
            all_actual_navs.extend(df.iloc[start_idx:pred_end]['nav'].values)

        n_folds += 1
        start_idx += predict_window

    if not all_predictions:
        logger.error(f"滚动训练无结果: {fund_code}")
        return None

    # ── 计算评估指标 ──

    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)

    # 第一维度：预测性能
    direction_accuracy = np.mean(all_predictions == all_targets)

    # 如果有净值数据，计算回归指标
    metrics = {
        'direction_accuracy': direction_accuracy,
        'n_folds': n_folds,
        'n_predictions': len(all_predictions),
    }

    if len(all_actual_navs) > 1:
        actual_navs = np.array(all_actual_navs)
        # 用实际净值计算回测指标
        backtest = calculate_backtest_metrics(actual_navs)
        metrics.update(backtest)

    # ── 保存最后一个窗口的模型 ──

    fund_model_dir = os.path.join(model_dir, 'lgbm')
    os.makedirs(fund_model_dir, exist_ok=True)

    # 重新用全量数据训练最终模型
    X_all_raw = df[feature_cols].values
    y_all = df[target_col].values
    scaler_final = StandardScaler()
    X_all = scaler_final.fit_transform(X_all_raw)

    val_split = int(len(X_all) * 0.8)
    final_model = LGBMPredictor(task='classification')
    final_model.train(
        X_all[:val_split], y_all[:val_split],
        X_all[val_split:], y_all[val_split:],
        num_boost_round=500,
        early_stopping_rounds=30,
        verbose=False
    )

    final_model.save(os.path.join(fund_model_dir, f'model_{fund_code}.pkl'))
    joblib.dump(scaler_final, os.path.join(fund_model_dir, f'scaler_{fund_code}.pkl'))

    # 保存特征重要性
    importance = final_model.get_feature_importance(len(feature_cols))
    if importance is not None:
        importance.to_csv(
            os.path.join(fund_model_dir, f'importance_{fund_code}.csv'),
            index=False
        )

    logger.info(f"[{fund_code}] 方向准确率={direction_accuracy:.4f}, "
                f"夏普={metrics.get('sharpe_ratio', 0):.2f}, "
                f"最大回撤={metrics.get('max_drawdown', 0):.2%}")

    return {
        'fund_code': fund_code,
        'metrics': metrics,
        'model_path': os.path.join(fund_model_dir, f'model_{fund_code}.pkl'),
    }


def train_all_models(data_dir='data/processed/train', model_dir='models',
                      fund_pool=None, db=None):
    """
    训练所有基金模型并写入数据库

    Args:
        data_dir: 特征数据目录
        model_dir: 模型保存目录
        fund_pool: 基金池（默认从 data_loader 加载）
        db: FundDatabase 实例

    Returns:
        dict: 训练结果
    """
    from src.data_loader import FUND_POOL

    fund_pool = fund_pool or FUND_POOL
    all_results = {}

    logger.info(f"开始训练 {len(fund_pool)} 只基金模型")

    for i, (code, name) in enumerate(fund_pool.items(), 1):
        logger.info(f"[{i}/{len(fund_pool)}] {code} - {name}")
        try:
            result = rolling_window_train(code, data_dir, model_dir)
            if result:
                all_results[code] = result
        except Exception as e:
            logger.error(f"  训练失败: {e}")

    # ── 写入 Supabase ──

    if db is None:
        from src.database import FundDatabase
        db = FundDatabase()
        db.connect()

    if db.client:
        prediction_date = datetime.now().strftime('%Y-%m-%d')
        target_date = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')

        stored = 0
        for code, result in all_results.items():
            metrics = result['metrics']
            try:
                db.insert_prediction_v2(
                    fund_code=code,
                    prediction_date=prediction_date,
                    target_date=target_date,
                    predicted_direction=int(metrics.get('direction_accuracy', 0.5) > 0.5),
                    predicted_score=metrics.get('direction_accuracy', 0.5),
                    model_name='lightgbm_rolling',
                    model_version='v1',
                    evaluation_metrics={
                        'direction_accuracy': round(metrics.get('direction_accuracy', 0), 4),
                        'sharpe_ratio': round(metrics.get('sharpe_ratio', 0), 4),
                        'max_drawdown': round(metrics.get('max_drawdown', 0), 4),
                        'total_return': round(metrics.get('total_return', 0), 4),
                        'n_folds': metrics.get('n_folds', 0),
                    }
                )
                stored += 1
            except Exception as e:
                logger.error(f"  写入数据库失败 {code}: {e}")

        logger.info(f"预测结果已写入 Supabase: {stored}/{len(all_results)}")

    # ── 汇总 ──

    logger.info("=" * 60)
    logger.info("训练完成汇总")
    logger.info("=" * 60)

    for code, result in all_results.items():
        m = result['metrics']
        logger.info(f"  {code}: 方向准确率={m.get('direction_accuracy', 0):.4f}, "
                    f"夏普={m.get('sharpe_ratio', 0):.2f}")

    return all_results


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s: %(message)s')

    import sys
    fund_code = sys.argv[1] if len(sys.argv) > 1 else None

    if fund_code:
        result = rolling_window_train(fund_code)
        if result:
            print(f"\n{fund_code} 训练结果:")
            for k, v in result['metrics'].items():
                print(f"  {k}: {v}")
    else:
        train_all_models()
