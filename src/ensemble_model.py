"""
模型集成模块
将LSTM和LightGBM预测结果进行集成
"""

import numpy as np
import pandas as pd
import os
import joblib
from src.model_lstm import LSTMPredictor
from src.model_lgbm import LGBMPredictor
from src.feature_engine import add_technical_indicators, prepare_features


class EnsemblePredictor:
    """集成预测器（LSTM + LightGBM）"""

    def __init__(self, lstm_weight=0.4, lgbm_weight=0.6):
        """
        Args:
            lstm_weight: LSTM模型权重
            lgbm_weight: LightGBM模型权重
        """
        self.lstm_weight = lstm_weight
        self.lgbm_weight = lgbm_weight
        self.lstm_model = None
        self.lgbm_model = None
        self.scaler = None

    def load_models(self, fund_code: str, model_dir: str = 'models'):
        """
        加载已训练的模型

        Args:
            fund_code: 基金代码
            model_dir: 模型目录
        """
        # 加载LSTM模型
        lstm_path = f"{model_dir}/lstm/model_{fund_code}.pth"
        if os.path.exists(lstm_path):
            self.lstm_model = LSTMPredictor(input_size=19)
            self.lstm_model.load(lstm_path)
            print(f"  LSTM模型已加载: {lstm_path}")

        # 加载LightGBM模型
        lgbm_path = f"{model_dir}/lgbm/model_{fund_code}.pkl"
        if os.path.exists(lgbm_path):
            self.lgbm_model = LGBMPredictor(task='classification')
            self.lgbm_model.load(lgbm_path)
            print(f"  LightGBM模型已加载: {lgbm_path}")

        # 加载Scaler
        scaler_path = f"{model_dir}/lgbm/scaler_{fund_code}.pkl"
        if os.path.exists(scaler_path):
            self.scaler = joblib.load(scaler_path)
            print(f"  Scaler已加载: {scaler_path}")

    def predict(self, features: np.ndarray) -> dict:
        """
        集成预测

        Args:
            features: 特征数据

        Returns:
            dict: 预测结果
        """
        predictions = {}

        # LSTM预测
        if self.lstm_model is not None:
            try:
                lstm_pred = self.lstm_model.predict(features)
                predictions['lstm'] = float(lstm_pred[0]) if len(lstm_pred) > 0 else 0
            except Exception as e:
                print(f"  LSTM预测失败: {e}")
                predictions['lstm'] = 0.5

        # LightGBM预测
        if self.lgbm_model is not None:
            try:
                if self.scaler is not None:
                    features_scaled = self.scaler.transform(features)
                else:
                    features_scaled = features

                lgbm_pred = self.lgbm_model.predict(features_scaled)
                predictions['lgbm'] = float(lgbm_pred[0]) if len(lgbm_pred) > 0 else 0.5
            except Exception as e:
                print(f"  LightGBM预测失败: {e}")
                predictions['lgbm'] = 0.5

        # 集成
        if not predictions:
            return {'signal': 0, 'probability': 0.5, 'method': 'none'}

        # 加权平均
        if 'lstm' in predictions and 'lgbm' in predictions:
            ensemble_prob = (predictions['lstm'] * self.lstm_weight +
                           predictions['lgbm'] * self.lgbm_weight)
            method = 'ensemble'
        elif 'lstm' in predictions:
            ensemble_prob = predictions['lstm']
            method = 'lstm_only'
        else:
            ensemble_prob = predictions['lgbm']
            method = 'lgbm_only'

        # 生成信号
        signal = 1 if ensemble_prob > 0.5 else 0

        return {
            'signal': signal,
            'probability': ensemble_prob,
            'method': method,
            'lstm_pred': predictions.get('lstm'),
            'lgbm_pred': predictions.get('lgbm')
        }

    def predict_batch(self, features_list: list) -> list:
        """
        批量预测

        Args:
            features_list: 特征列表

        Returns:
            list: 预测结果列表
        """
        return [self.predict(features) for features in features_list]

    def evaluate_ensemble(self, test_data: pd.DataFrame, fund_code: str) -> dict:
        """
        评估集成模型效果

        Args:
            test_data: 测试数据
            fund_code: 基金代码

        Returns:
            dict: 评估结果
        """
        feature_cols, _ = prepare_features(test_data)
        X = test_data[feature_cols].values
        y_true = test_data['target_1d_dir'].values

        predictions = []
        for i in range(len(X)):
            features = X[i:i+1]
            pred = self.predict(features)
            predictions.append(pred['probability'])

        predictions = np.array(predictions)
        pred_direction = (predictions > 0.5).astype(int)

        accuracy = np.mean(pred_direction == y_true)

        # 分别计算各模型准确率
        results = {
            'ensemble_accuracy': accuracy,
            'prediction_count': len(predictions),
            'positive_ratio': np.mean(pred_direction)
        }

        return results


def train_and_evaluate_ensemble(fund_code: str, train_data: pd.DataFrame,
                                  test_data: pd.DataFrame, model_dir: str = 'models') -> dict:
    """
    训练并评估集成模型

    Args:
        fund_code: 基金代码
        train_data: 训练数据
        test_data: 测试数据
        model_dir: 模型目录

    Returns:
        dict: 评估结果
    """
    print(f"\n{'='*60}")
    print(f"训练集成模型: {fund_code}")
    print(f"{'='*60}")

    feature_cols, target_cols = prepare_features(train_data)
    target_col = 'target_1d_dir'

    if target_col not in target_cols:
        print(f"目标变量 {target_col} 不存在")
        return {}

    X_train = train_data[feature_cols].values
    y_train = train_data[target_col].values
    X_test = test_data[feature_cols].values
    y_test = test_data[target_col].values

    # 1. 训练LightGBM
    print("\n[1/2] 训练LightGBM模型...")
    lgbm_model = LGBMPredictor(task='classification')
    lgbm_accuracy = lgbm_model.train(X_train, y_train, X_test, y_test)

    # 保存模型
    lgbm_dir = f"{model_dir}/lgbm"
    os.makedirs(lgbm_dir, exist_ok=True)
    lgbm_model.save(f"{lgbm_dir}/model_{fund_code}.pkl")

    # 2. 训练LSTM（简化版，使用较少epoch）
    print("\n[2/2] 训练LSTM模型...")
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    lstm_model = LSTMPredictor(input_size=X_train.shape[1], seq_length=10)
    lstm_model.train(X_train_scaled, y_train, X_test_scaled, y_test, epochs=30, verbose=False)

    # 保存LSTM模型
    lstm_dir = f"{model_dir}/lstm"
    os.makedirs(lstm_dir, exist_ok=True)
    lstm_model.save(f"{lstm_dir}/model_{fund_code}.pth")
    joblib.dump(scaler, f"{lgbm_dir}/scaler_{fund_code}.pkl")

    # 3. 集成评估
    print("\n评估集成效果...")
    ensemble = EnsemblePredictor(lstm_weight=0.4, lgbm_weight=0.6)
    ensemble.lstm_model = lstm_model
    ensemble.lgbm_model = lgbm_model
    ensemble.scaler = scaler

    results = ensemble.evaluate_ensemble(test_data, fund_code)
    results['fund_code'] = fund_code
    results['lgbm_accuracy'] = lgbm_accuracy

    print(f"\n结果:")
    print(f"  LightGBM准确率: {lgbm_accuracy:.2%}")
    print(f"  集成准确率: {results['ensemble_accuracy']:.2%}")

    return results


if __name__ == '__main__':
    from data_loader import load_all_funds
    from feature_engine import process_all_funds

    # 加载数据
    print("加载数据...")
    train_data = load_all_funds('data/raw/train')
    test_data = load_all_funds('data/raw/test')

    # 特征工程
    print("\n特征工程...")
    train_features = process_all_funds(train_data, save_dir='data/processed/train')
    test_features = process_all_funds(test_data, save_dir='data/processed/test')

    # 训练并评估集成模型
    all_results = []
    for fund_code in train_features:
        if fund_code in test_features:
            results = train_and_evaluate_ensemble(
                fund_code,
                train_features[fund_code],
                test_features[fund_code]
            )
            if results:
                all_results.append(results)

    # 汇总结果
    if all_results:
        df_results = pd.DataFrame(all_results)
        print("\n" + "=" * 60)
        print("集成模型汇总")
        print("=" * 60)
        print(df_results[['fund_code', 'lgbm_accuracy', 'ensemble_accuracy']].to_string(index=False))
