"""
快速启动脚本
一键运行整个流程
"""

import os
import sys

# 设置代理
os.environ['http_proxy'] = 'http://127.0.0.1:7897'
os.environ['https_proxy'] = 'http://127.0.0.1:7897'

def main():
    print("=" * 60)
    print("基金预测项目 - 快速启动")
    print("=" * 60)

    # 检查依赖
    print("\n[1/6] 检查依赖...")
    try:
        import akshare
        import pandas
        import numpy
        import torch
        import lightgbm
        print("✓ 依赖包已安装")
    except ImportError as e:
        print(f"✗ 缺少依赖: {e}")
        print("请先运行 install.bat 安装依赖")
        return

    # 创建目录
    print("\n[2/6] 创建目录结构...")
    dirs = [
        'data/raw/train', 'data/raw/test', 'data/processed/train', 'data/processed/test',
        'data/events', 'models/lstm', 'models/lgbm', 'reports'
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    print("✓ 目录结构已创建")

    # 下载数据
    print("\n[3/6] 下载基金数据...")
    from src.data_loader import download_all_funds

    print("下载2025年训练数据...")
    train_data = download_all_funds(
        start_date='20250101',
        end_date='20251231',
        save_dir='data/raw/train'
    )

    print("\n下载2026年1-4月测试数据...")
    test_data = download_all_funds(
        start_date='20260101',
        end_date='20260430',
        save_dir='data/raw/test'
    )

    # 特征工程
    print("\n[4/6] 运行特征工程...")
    from src.feature_engine import process_all_funds
    from src.data_loader import load_all_funds

    train_data = load_all_funds('data/raw/train')
    processed_data = process_all_funds(train_data, save_dir='data/processed/train')
    print(f"✓ 特征工程完成，处理了 {len(processed_data)} 只基金")

    # 训练模型
    print("\n[5/6] 训练模型...")
    from src.train import train_all_models

    results = train_all_models(
        data_dir='data/processed/train',
        model_dir='models'
    )
    print(f"✓ 模型训练完成")

    # 回测
    print("\n[6/6] 运行回测...")
    from src.backtest import run_backtest_all

    backtest_results = run_backtest_all(
        test_data_dir='data/raw/test',
        model_dir='models',
        initial_capital=100000
    )

    # 生成报告
    print("\n生成报告...")
    from src.report import generate_full_report

    report = generate_full_report(backtest_results, results)

    print("\n" + "=" * 60)
    print("快速启动完成!")
    print("=" * 60)
    print(f"\n报告已保存到: reports/summary_report.md")
    print(f"\n汇总结果:")
    print(f"  平均收益率: {report['total_return']:.2%}")
    print(f"  平均夏普比率: {report['sharpe_ratio']:.2f}")
    print(f"  平均胜率: {report['win_rate']:.2%}")


if __name__ == '__main__':
    main()
