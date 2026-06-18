"""
基金预测项目 - 主运行脚本
一键运行数据下载、特征工程、模型训练、回测
"""

import os
import sys

# 代理自动检测（可用则走代理，否则直连）
from src.config import setup_proxy
setup_proxy()

def main():
    print("=" * 60)
    print("基金预测项目 - 主程序")
    print("=" * 60)

    while True:
        print("\n请选择操作:")
        print("1. 下载基金数据")
        print("2. 运行特征工程")
        print("3. 训练模型")
        print("4. 运行回测")
        print("5. 生成报告")
        print("0. 退出")

        choice = input("\n请输入选项 (0-5): ").strip()

        if choice == '1':
            download_data()
        elif choice == '2':
            run_feature_engineering()
        elif choice == '3':
            train_models()
        elif choice == '4':
            run_backtest()
        elif choice == '5':
            generate_report()
        elif choice == '0':
            print("再见!")
            break
        else:
            print("无效选项，请重新输入")


def download_data():
    """下载基金数据"""
    print("\n" + "=" * 60)
    print("下载基金数据")
    print("=" * 60)

    from src.data_loader import download_all_funds

    # 下载2025年训练数据
    print("\n[1/2] 下载2025年训练数据...")
    train_data = download_all_funds(
        start_date='20250101',
        end_date='20251231',
        save_dir='data/raw/train'
    )

    # 下载2026年测试数据
    print("\n[2/2] 下载2026年1-4月测试数据...")
    test_data = download_all_funds(
        start_date='20260101',
        end_date='20260430',
        save_dir='data/raw/test'
    )

    print("\n数据下载完成!")
    print(f"训练集: {len(train_data)} 只基金")
    print(f"测试集: {len(test_data)} 只基金")


def run_feature_engineering():
    """运行特征工程"""
    print("\n" + "=" * 60)
    print("特征工程")
    print("=" * 60)

    from src.feature_engine import process_all_funds
    from src.data_loader import load_all_funds

    # 加载训练数据
    print("\n加载训练数据...")
    train_data = load_all_funds('data/raw/train')

    # 处理特征
    print("\n计算特征...")
    processed_data = process_all_funds(train_data, save_dir='data/processed/train')

    print(f"\n特征工程完成! 处理了 {len(processed_data)} 只基金")


def train_models():
    """训练模型"""
    print("\n" + "=" * 60)
    print("模型训练")
    print("=" * 60)

    from src.train import train_all_models

    # 训练所有基金的模型
    results = train_all_models(
        data_dir='data/processed/train',
        model_dir='models'
    )

    print("\n模型训练完成!")
    for fund_code, metrics in results.items():
        print(f"  {fund_code}: LSTM准确率={metrics['lstm_acc']:.2%}, "
              f"LightGBM准确率={metrics['lgbm_acc']:.2%}")


def run_backtest():
    """运行回测"""
    print("\n" + "=" * 60)
    print("回测模拟")
    print("=" * 60)

    from src.backtest import run_backtest_all

    # 运行回测
    results = run_backtest_all(
        test_data_dir='data/raw/test',
        model_dir='models',
        initial_capital=100000
    )

    print("\n回测完成!")
    print(f"总收益率: {results['total_return']:.2%}")
    print(f"夏普比率: {results['sharpe_ratio']:.2f}")
    print(f"最大回撤: {results['max_drawdown']:.2%}")


def generate_report():
    """生成报告"""
    print("\n" + "=" * 60)
    print("生成报告")
    print("=" * 60)

    from src.report import generate_full_report

    report = generate_full_report()

    print("\n报告已生成: reports/summary_report.md")
    print("\n报告摘要:")
    print(f"  总收益率: {report['total_return']:.2%}")
    print(f"  夏普比率: {report['sharpe_ratio']:.2f}")
    print(f"  胜率: {report['win_rate']:.2%}")


if __name__ == '__main__':
    main()
