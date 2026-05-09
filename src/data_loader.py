"""
数据加载模块
负责从AKShare获取基金数据并保存到本地
"""

import akshare as ak
import pandas as pd
import os
from datetime import datetime
from tqdm import tqdm

# 基金池统一从 fund_crawler 加载（支持 CSV 动态配置）
from src.fund_crawler import load_fund_pool
FUND_POOL = load_fund_pool()


def get_fund_nav(fund_code: str, start_date: str = None, end_date: str = None) -> pd.DataFrame:
    """
    获取基金净值数据

    Args:
        fund_code: 基金代码
        start_date: 开始日期，格式：YYYYMMDD
        end_date: 结束日期，格式：YYYYMMDD

    Returns:
        DataFrame: 包含日期、单位净值、日增长率
    """
    try:
        # 获取开放式基金净值
        df = ak.fund_open_fund_info_em(symbol=fund_code, indicator="单位净值走势")

        # 重命名列
        df.columns = ['date', 'nav', 'daily_return']

        # 转换日期格式
        df['date'] = pd.to_datetime(df['date'])

        # 按日期筛选
        if start_date:
            df = df[df['date'] >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df['date'] <= pd.to_datetime(end_date)]

        # 添加基金代码列
        df['fund_code'] = fund_code

        # 按日期排序
        df = df.sort_values('date').reset_index(drop=True)

        return df

    except Exception as e:
        print(f"获取基金 {fund_code} 数据失败: {e}")
        return pd.DataFrame()


def get_fund_info(fund_code: str) -> dict:
    """
    获取基金基本信息

    Args:
        fund_code: 基金代码

    Returns:
        dict: 基金基本信息
    """
    try:
        # 获取基金名称
        fund_name_df = ak.fund_name_em()
        fund_info = fund_name_df[fund_name_df['基金代码'] == fund_code]

        if fund_info.empty:
            return {'code': fund_code, 'name': FUND_POOL.get(fund_code, '未知')}

        return {
            'code': fund_code,
            'name': fund_info.iloc[0]['基金简称'],
            'type': fund_info.iloc[0]['基金类型'],
        }

    except Exception as e:
        print(f"获取基金 {fund_code} 信息失败: {e}")
        return {'code': fund_code, 'name': FUND_POOL.get(fund_code, '未知')}


def download_all_funds(start_date: str = '20250101', end_date: str = None,
                       save_dir: str = 'data/raw') -> dict:
    """
    下载所有基金池中的基金数据

    Args:
        start_date: 开始日期
        end_date: 结束日期
        save_dir: 保存目录

    Returns:
        dict: 所有基金数据 {fund_code: DataFrame}
    """
    os.makedirs(save_dir, exist_ok=True)

    all_data = {}

    print(f"开始下载基金数据，时间范围: {start_date} - {end_date or '至今'}")
    print(f"基金池: {len(FUND_POOL)} 只基金")

    for fund_code, fund_name in tqdm(FUND_POOL.items(), desc="下载进度"):
        print(f"\n正在下载: {fund_code} - {fund_name}")

        # 获取净值数据
        df = get_fund_nav(fund_code, start_date, end_date)

        if not df.empty:
            # 保存到CSV
            filename = f"{save_dir}/fund_{fund_code}.csv"
            df.to_csv(filename, index=False, encoding='utf-8-sig')
            print(f"  保存成功: {filename}, 共 {len(df)} 条数据")

            all_data[fund_code] = df
        else:
            print(f"  警告: {fund_code} 数据为空")

    print(f"\n下载完成! 成功: {len(all_data)}/{len(FUND_POOL)}")

    return all_data


def load_fund_data(fund_code: str, data_dir: str = 'data/raw') -> pd.DataFrame:
    """
    从本地加载基金数据

    Args:
        fund_code: 基金代码
        data_dir: 数据目录

    Returns:
        DataFrame: 基金数据
    """
    filename = f"{data_dir}/fund_{fund_code}.csv"

    if not os.path.exists(filename):
        print(f"文件不存在: {filename}")
        return pd.DataFrame()

    df = pd.read_csv(filename)
    df['date'] = pd.to_datetime(df['date'])

    return df


def load_all_funds(data_dir: str = 'data/raw') -> dict:
    """
    加载所有本地基金数据

    Args:
        data_dir: 数据目录

    Returns:
        dict: 所有基金数据 {fund_code: DataFrame}
    """
    all_data = {}

    for fund_code in FUND_POOL.keys():
        df = load_fund_data(fund_code, data_dir)
        if not df.empty:
            all_data[fund_code] = df

    print(f"加载完成: {len(all_data)} 只基金")

    return all_data


if __name__ == '__main__':
    # 测试数据下载
    print("=" * 50)
    print("基金数据下载测试")
    print("=" * 50)

    # 下载2025年数据（训练集）
    train_data = download_all_funds(
        start_date='20250101',
        end_date='20251231',
        save_dir='data/raw/train'
    )

    # 下载2026年1-4月数据（测试集）
    test_data = download_all_funds(
        start_date='20260101',
        end_date='20260430',
        save_dir='data/raw/test'
    )

    print("\n数据下载完成!")
    print(f"训练集: {len(train_data)} 只基金")
    print(f"测试集: {len(test_data)} 只基金")
