"""
特征工程模块
计算技术指标、基金特征、事件特征
"""

import pandas as pd
import numpy as np
import os


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    添加技术指标特征

    Args:
        df: 包含date, nav, acc_nav, daily_return列的DataFrame

    Returns:
        DataFrame: 添加了技术指标的DataFrame
    """
    df = df.copy()

    # 确保按日期排序
    df = df.sort_values('date').reset_index(drop=True)

    # ========== 移动平均线 ==========
    df['MA5'] = df['nav'].rolling(window=5).mean()
    df['MA10'] = df['nav'].rolling(window=10).mean()
    df['MA20'] = df['nav'].rolling(window=20).mean()
    df['MA60'] = df['nav'].rolling(window=60).mean()

    # 均线偏离度
    df['MA5_bias'] = (df['nav'] - df['MA5']) / df['MA5']
    df['MA20_bias'] = (df['nav'] - df['MA20']) / df['MA20']

    # ========== RSI (相对强弱指标) ==========
    delta = df['nav'].diff()
    gain = delta.where(delta > 0, 0)
    loss = (-delta).where(delta < 0, 0)

    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()

    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # ========== MACD ==========
    exp1 = df['nav'].ewm(span=12, adjust=False).mean()
    exp2 = df['nav'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_hist'] = df['MACD'] - df['MACD_signal']

    # ========== 布林带 ==========
    df['BB_middle'] = df['nav'].rolling(window=20).mean()
    bb_std = df['nav'].rolling(window=20).std()
    df['BB_upper'] = df['BB_middle'] + 2 * bb_std
    df['BB_lower'] = df['BB_middle'] - 2 * bb_std
    df['BB_width'] = (df['BB_upper'] - df['BB_lower']) / df['BB_middle']
    df['BB_position'] = (df['nav'] - df['BB_lower']) / (df['BB_upper'] - df['BB_lower'])

    # ========== 波动率 ==========
    df['volatility_5'] = df['daily_return'].rolling(window=5).std()
    df['volatility_20'] = df['daily_return'].rolling(window=20).std()

    # ========== 动量指标 ==========
    df['momentum_5'] = df['nav'].pct_change(periods=5)
    df['momentum_10'] = df['nav'].pct_change(periods=10)
    df['momentum_20'] = df['nav'].pct_change(periods=20)

    # ========== 价格位置 ==========
    df['high_5'] = df['nav'].rolling(window=5).max()
    df['low_5'] = df['nav'].rolling(window=5).min()
    df['high_20'] = df['nav'].rolling(window=20).max()
    df['low_20'] = df['nav'].rolling(window=20).min()

    df['position_5'] = (df['nav'] - df['low_5']) / (df['high_5'] - df['low_5'])
    df['position_20'] = (df['nav'] - df['low_20']) / (df['high_20'] - df['low_20'])

    return df


def add_target_variable(df: pd.DataFrame, periods: list = [1, 5, 10]) -> pd.DataFrame:
    """
    添加目标变量（未来收益率）

    Args:
        df: DataFrame
        periods: 预测周期列表

    Returns:
        DataFrame: 添加了目标变量的DataFrame
    """
    df = df.copy()

    for period in periods:
        # 未来N日收益率
        df[f'target_{period}d'] = df['nav'].pct_change(periods=period).shift(-period)

        # 未来N日涨跌方向 (1=涨, 0=跌)
        df[f'target_{period}d_dir'] = (df[f'target_{period}d'] > 0).astype(int)

    return df


def prepare_features(df: pd.DataFrame) -> tuple:
    """
    准备训练特征和目标变量

    Args:
        df: 完整特征DataFrame

    Returns:
        tuple: (feature_columns, target_columns)
    """
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

    # 目标列
    target_cols = [col for col in df.columns if col.startswith('target_')]

    return feature_cols, target_cols


def process_single_fund(df: pd.DataFrame, fund_code: str) -> pd.DataFrame:
    """
    处理单只基金的特征

    Args:
        df: 原始数据
        fund_code: 基金代码

    Returns:
        DataFrame: 处理后的特征数据
    """
    print(f"处理基金: {fund_code}")

    # 添加技术指标
    df = add_technical_indicators(df)

    # 添加目标变量
    df = add_target_variable(df, periods=[1, 5])

    # 删除包含NaN的行
    df = df.dropna().reset_index(drop=True)

    print(f"  特征数量: {len(df.columns)}, 数据条数: {len(df)}")

    return df


def process_all_funds(data_dict: dict, save_dir: str = 'data/processed') -> dict:
    """
    处理所有基金的特征

    Args:
        data_dict: 所有基金数据 {fund_code: DataFrame}
        save_dir: 保存目录

    Returns:
        dict: 处理后的数据
    """
    os.makedirs(save_dir, exist_ok=True)

    processed_data = {}

    for fund_code, df in data_dict.items():
        # 处理特征
        processed_df = process_single_fund(df, fund_code)

        if not processed_df.empty:
            # 保存处理后的数据
            filename = f"{save_dir}/features_{fund_code}.csv"
            processed_df.to_csv(filename, index=False, encoding='utf-8-sig')

            processed_data[fund_code] = processed_df

    print(f"\n特征工程完成! 处理了 {len(processed_data)} 只基金")

    return processed_data


if __name__ == '__main__':
    # 测试特征工程
    from data_loader import load_all_funds

    # 加载数据
    train_data = load_all_funds('data/raw/train')

    # 处理特征
    processed_data = process_all_funds(train_data, save_dir='data/processed/train')

    # 显示特征列表
    sample_fund = list(processed_data.keys())[0]
    sample_df = processed_data[sample_fund]
    print(f"\n特征列表 ({sample_fund}):")
    print(sample_df.columns.tolist())
    print(f"\n数据样例:")
    print(sample_df.head())
