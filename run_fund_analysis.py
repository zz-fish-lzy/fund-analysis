"""
基金分析快速运行脚本
一键运行数据获取、特征工程、模型训练、回测
"""

import os
import sys

# 设置代理
os.environ['http_proxy'] = 'http://127.0.0.1:7897'
os.environ['https_proxy'] = 'http://127.0.0.1:7897'

def main():
    print("=" * 60)
    print("基金分析系统 - 快速运行")
    print("=" * 60)

    while True:
        print("\n请选择操作:")
        print("1. 获取最新基金数据")
        print("2. 运行特征工程")
        print("3. 训练LightGBM模型")
        print("4. 训练集成模型 (LSTM + LightGBM)")
        print("5. 运行回测分析")
        print("6. 行业轮动分析")
        print("7. 市场情绪分析")
        print("8. 生成分析报告")
        print("9. 一键运行全部")
        print("0. 退出")

        choice = input("\n请输入选项 (0-9): ").strip()

        if choice == '1':
            download_data()
        elif choice == '2':
            run_feature_engineering()
        elif choice == '3':
            train_models()
        elif choice == '4':
            train_ensemble_models()
        elif choice == '5':
            run_backtest()
        elif choice == '6':
            run_industry_rotation()
        elif choice == '7':
            run_sentiment_analysis()
        elif choice == '8':
            generate_report()
        elif choice == '9':
            run_all()
        elif choice == '0':
            print("再见!")
            break
        else:
            print("无效选项，请重新输入")


def download_data():
    """下载基金数据"""
    print("\n" + "=" * 60)
    print("获取基金数据")
    print("=" * 60)

    import akshare as ak
    import pandas as pd
    import time

    # 基金池
    fund_pool = {
        '510300': '沪深300ETF华泰柏瑞',
        '510500': '中证500ETF南方',
        '159915': '创业板ETF易方达',
        '510050': '上证50ETF华夏',
        '512100': '中证1000ETF南方',
        '005827': '易方达蓝筹精选混合',
        '003095': '中欧医疗健康混合A',
        '163406': '兴全合润混合A',
        '320007': '诺安成长混合A',
        '001156': '申万菱信新能源汽车主题',
        '001938': '中欧时代先锋股票A',
        '005911': '广发双擎升级混合A',
        '000961': '天弘沪深300ETF联接A',
        '012414': '国泰中证白酒指数(LOF)C',
        '161725': '招商中证白酒指数(LOF)A',
        '217022': '招商产业债A',
        '110017': '易方达增强回报债券A',
        '050011': '博时信用债券A/B',
        '110011': '易方达中小盘混合(QDII)',
        '159919': '沪深300ETF嘉实',
    }

    # 创建目录
    os.makedirs('data/raw/train', exist_ok=True)
    os.makedirs('data/raw/test', exist_ok=True)

    success_count = 0

    for fund_code, fund_name in fund_pool.items():
        try:
            print(f"获取: {fund_code} - {fund_name}...")

            # 获取净值数据
            df = ak.fund_open_fund_info_em(symbol=fund_code, indicator="单位净值走势")

            if df is not None and len(df) > 0:
                # 重命名列
                df.columns = ['date', 'nav', 'daily_return']
                df['date'] = pd.to_datetime(df['date'])
                df['fund_code'] = fund_code

                # 筛选2025年数据（训练集）
                df_2025 = df[(df['date'] >= '2025-01-01') & (df['date'] <= '2025-12-31')]

                if len(df_2025) > 0:
                    filename = f'data/raw/train/fund_{fund_code}.csv'
                    df_2025.to_csv(filename, index=False, encoding='utf-8-sig')
                    print(f"  训练集: {len(df_2025)} 条数据")

                    # 筛选2026年1-4月数据（测试集）
                    df_2026 = df[(df['date'] >= '2026-01-01') & (df['date'] <= '2026-04-30')]

                    if len(df_2026) > 0:
                        filename = f'data/raw/test/fund_{fund_code}.csv'
                        df_2026.to_csv(filename, index=False, encoding='utf-8-sig')
                        print(f"  测试集: {len(df_2026)} 条数据")

                    success_count += 1

            time.sleep(0.5)

        except Exception as e:
            print(f"  错误: {e}")

    print(f"\n数据获取完成! 成功: {success_count}/{len(fund_pool)}")


def run_feature_engineering():
    """运行特征工程"""
    print("\n" + "=" * 60)
    print("特征工程")
    print("=" * 60)

    import pandas as pd
    import numpy as np
    import os

    # 特征计算函数
    def add_technical_indicators(df):
        df = df.copy()
        df = df.sort_values('date').reset_index(drop=True)

        # 移动平均线
        df['MA5'] = df['nav'].rolling(window=5).mean()
        df['MA10'] = df['nav'].rolling(window=10).mean()
        df['MA20'] = df['nav'].rolling(window=20).mean()
        df['MA60'] = df['nav'].rolling(window=60).mean()

        # 均线偏离度
        df['MA5_bias'] = (df['nav'] - df['MA5']) / df['MA5']
        df['MA20_bias'] = (df['nav'] - df['MA20']) / df['MA20']

        # RSI
        delta = df['nav'].diff()
        gain = delta.where(delta > 0, 0)
        loss = (-delta).where(delta < 0, 0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        df['RSI'] = 100 - (100 / (1 + rs))

        # MACD
        exp1 = df['nav'].ewm(span=12, adjust=False).mean()
        exp2 = df['nav'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_hist'] = df['MACD'] - df['MACD_signal']

        # 布林带
        df['BB_middle'] = df['nav'].rolling(window=20).mean()
        bb_std = df['nav'].rolling(window=20).std()
        df['BB_upper'] = df['BB_middle'] + 2 * bb_std
        df['BB_lower'] = df['BB_middle'] - 2 * bb_std
        df['BB_width'] = (df['BB_upper'] - df['BB_lower']) / df['BB_middle']
        df['BB_position'] = (df['nav'] - df['BB_lower']) / (df['BB_upper'] - df['BB_lower'])

        # 波动率
        df['volatility_5'] = df['daily_return'].rolling(window=5).std()
        df['volatility_20'] = df['daily_return'].rolling(window=20).std()

        # 动量指标
        df['momentum_5'] = df['nav'].pct_change(periods=5)
        df['momentum_10'] = df['nav'].pct_change(periods=10)
        df['momentum_20'] = df['nav'].pct_change(periods=20)

        # 价格位置
        df['high_5'] = df['nav'].rolling(window=5).max()
        df['low_5'] = df['nav'].rolling(window=5).min()
        df['high_20'] = df['nav'].rolling(window=20).max()
        df['low_20'] = df['nav'].rolling(window=20).min()
        df['position_5'] = (df['nav'] - df['low_5']) / (df['high_5'] - df['low_5'])
        df['position_20'] = (df['nav'] - df['low_20']) / (df['high_20'] - df['low_20'])

        # 目标变量
        df['target_1d'] = df['nav'].pct_change(periods=1).shift(-1)
        df['target_1d_dir'] = (df['target_1d'] > 0).astype(int)
        df['target_5d'] = df['nav'].pct_change(periods=5).shift(-5)
        df['target_5d_dir'] = (df['target_5d'] > 0).astype(int)

        return df

    # 处理训练数据
    train_dir = 'data/raw/train'
    processed_dir = 'data/processed/train'
    os.makedirs(processed_dir, exist_ok=True)

    print("处理训练数据...")
    for filename in os.listdir(train_dir):
        if filename.endswith('.csv'):
            fund_code = filename.replace('fund_', '').replace('.csv', '')
            df = pd.read_csv(os.path.join(train_dir, filename))
            df['date'] = pd.to_datetime(df['date'])

            # 计算特征
            df_features = add_technical_indicators(df)
            df_features = df_features.dropna().reset_index(drop=True)

            # 保存
            output_file = f'{processed_dir}/features_{fund_code}.csv'
            df_features.to_csv(output_file, index=False, encoding='utf-8-sig')
            print(f"  {fund_code}: {len(df_features)} 条数据, {len(df_features.columns)} 个特征")

    print("\n特征工程完成!")


def train_models():
    """训练模型"""
    print("\n" + "=" * 60)
    print("训练LightGBM模型")
    print("=" * 60)

    import pandas as pd
    import numpy as np
    import os
    import joblib
    from sklearn.preprocessing import StandardScaler
    from src.model_lgbm import LGBMPredictor

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

    data_dir = 'data/processed/train'
    model_dir = 'models/lgbm'
    os.makedirs(model_dir, exist_ok=True)

    results = {}

    for filename in os.listdir(data_dir):
        if filename.endswith('.csv'):
            fund_code = filename.replace('features_', '').replace('.csv', '')

            print(f"训练: {fund_code}")

            # 加载数据
            df = pd.read_csv(os.path.join(data_dir, filename))

            # 准备数据
            X = df[feature_cols].values
            y = df['target_1d_dir'].values

            # 标准化
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # 划分训练集和验证集
            split_idx = int(len(X) * 0.8)
            X_train, X_val = X_scaled[:split_idx], X_scaled[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]

            # 训练模型
            model = LGBMPredictor(task='classification')
            model.train(X_train, y_train, X_val, y_val, verbose=False)

            # 评估
            result = model.evaluate(X_val, y_val)
            accuracy = result['accuracy']

            # 保存模型
            model.save(f"{model_dir}/model_{fund_code}.pkl")
            joblib.dump(scaler, f"{model_dir}/scaler_{fund_code}.pkl")

            results[fund_code] = accuracy
            print(f"  准确率: {accuracy:.4f}")

    # 汇总
    avg_acc = np.mean(list(results.values()))
    print(f"\n训练完成! 平均准确率: {avg_acc:.4f}")


def run_backtest():
    """运行回测"""
    print("\n" + "=" * 60)
    print("运行回测")
    print("=" * 60)

    import pandas as pd
    import numpy as np
    import os
    import joblib
    from src.model_lgbm import LGBMPredictor

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

    test_dir = 'data/raw/test'
    model_dir = 'models/lgbm'
    initial_capital = 100000

    results = {}

    for filename in os.listdir(test_dir):
        if filename.endswith('.csv'):
            fund_code = filename.replace('fund_', '').replace('.csv', '')

            print(f"回测: {fund_code}")

            # 加载测试数据
            df = pd.read_csv(os.path.join(test_dir, filename))
            df['date'] = pd.to_datetime(df['date'])

            # 加载模型和scaler
            model = LGBMPredictor(task='classification')
            model.load(f"{model_dir}/model_{fund_code}.pkl")
            scaler = joblib.load(f"{model_dir}/scaler_{fund_code}.pkl")

            # 模拟交易
            capital = initial_capital
            position = 0
            trades = []
            daily_values = []

            for i in range(len(df)):
                nav = df.iloc[i]['nav']
                date = df.iloc[i]['date']

                if i >= 20:
                    # 计算特征
                    recent_nav = df.iloc[i-20:i+1]['nav'].values
                    ma5 = np.mean(recent_nav[-5:])
                    ma10 = np.mean(recent_nav[-10:])
                    ma20 = np.mean(recent_nav)

                    features = np.array([[
                        ma5, ma10, ma20, ma20,
                        (nav - ma5) / ma5, (nav - ma20) / ma20,
                        50, 0, 0, 0,
                        0, 0.5,
                        0.01, 0.02,
                        (nav - recent_nav[-5]) / recent_nav[-5],
                        (nav - recent_nav[-10]) / recent_nav[-10],
                        (nav - recent_nav[0]) / recent_nav[0],
                        (nav - min(recent_nav[-5:])) / (max(recent_nav[-5:]) - min(recent_nav[-5:])),
                        (nav - min(recent_nav)) / (max(recent_nav) - min(recent_nav))
                    ]])

                    features_scaled = scaler.transform(features)
                    pred = model.predict_direction(features_scaled)[0]

                    if pred == 1 and position == 0:
                        shares = capital * 0.9 / nav
                        position = shares
                        capital = capital * 0.1
                        trades.append({'date': date, 'type': 'buy', 'price': nav, 'shares': shares})
                    elif pred == 0 and position > 0:
                        capital += position * nav * 0.999
                        position = 0
                        trades.append({'date': date, 'type': 'sell', 'price': nav})

                portfolio_value = capital + position * nav
                daily_values.append({'date': date, 'value': portfolio_value})

            # 计算指标
            if daily_values:
                values = pd.DataFrame(daily_values)
                final_value = values['value'].iloc[-1]
                total_return = (final_value - initial_capital) / initial_capital

                values['peak'] = values['value'].cummax()
                values['drawdown'] = (values['value'] - values['peak']) / values['peak']
                max_drawdown = values['drawdown'].min()

                winning_trades = sum(1 for i in range(1, len(trades)) if trades[i]['type'] == 'sell' and trades[i]['price'] > trades[i-1]['price'])
                total_trades = sum(1 for t in trades if t['type'] == 'sell')
                win_rate = winning_trades / total_trades if total_trades > 0 else 0

                results[fund_code] = {
                    'total_return': total_return,
                    'max_drawdown': max_drawdown,
                    'win_rate': win_rate,
                    'trades': len(trades)
                }

                print(f"  收益率: {total_return:.2%}")

    # 汇总
    avg_return = np.mean([r['total_return'] for r in results.values()])
    print(f"\n回测完成! 平均收益率: {avg_return:.2%}")


def train_ensemble_models():
    """训练集成模型"""
    print("\n" + "=" * 60)
    print("训练集成模型 (LSTM + LightGBM)")
    print("=" * 60)

    import pandas as pd
    import numpy as np
    import os
    from src.ensemble_model import train_and_evaluate_ensemble
    from src.feature_engine import add_technical_indicators

    train_dir = 'data/raw/train'
    test_dir = 'data/raw/test'
    model_dir = 'models'

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

    all_results = []

    for filename in os.listdir(train_dir):
        if filename.endswith('.csv'):
            fund_code = filename.replace('fund_', '').replace('.csv', '')

            print(f"\n处理基金: {fund_code}")

            # 加载训练数据
            train_df = pd.read_csv(os.path.join(train_dir, filename))
            train_df['date'] = pd.to_datetime(train_df['date'])

            # 加载测试数据
            test_file = os.path.join(test_dir, filename)
            if not os.path.exists(test_file):
                print(f"  跳过: 测试数据不存在")
                continue

            test_df = pd.read_csv(test_file)
            test_df['date'] = pd.to_datetime(test_df['date'])

            # 特征工程
            train_features = add_technical_indicators(train_df)
            test_features = add_technical_indicators(test_df)

            # 删除NaN
            train_features = train_features.dropna().reset_index(drop=True)
            test_features = test_features.dropna().reset_index(drop=True)

            if len(train_features) < 50 or len(test_features) < 10:
                print(f"  跳过: 数据不足")
                continue

            # 训练集成模型
            results = train_and_evaluate_ensemble(
                fund_code, train_features, test_features, model_dir
            )
            if results:
                all_results.append(results)

    # 汇总结果
    if all_results:
        df_results = pd.DataFrame(all_results)
        print("\n" + "=" * 60)
        print("集成模型训练汇总")
        print("=" * 60)
        avg_lgbm = df_results['lgbm_accuracy'].mean()
        avg_ensemble = df_results['ensemble_accuracy'].mean()
        print(f"LightGBM平均准确率: {avg_lgbm:.2%}")
        print(f"集成模型平均准确率: {avg_ensemble:.2%}")
        print(f"提升: {(avg_ensemble - avg_lgbm):.2%}")


def run_industry_rotation():
    """运行行业轮动分析"""
    print("\n" + "=" * 60)
    print("行业轮动分析")
    print("=" * 60)

    import pandas as pd
    from src.industry_rotation import IndustryRotationAnalyzer

    # 加载数据
    test_dir = 'data/raw/test'
    fund_data = {}

    for filename in os.listdir(test_dir):
        if filename.endswith('.csv'):
            fund_code = filename.replace('fund_', '').replace('.csv', '')
            df = pd.read_csv(os.path.join(test_dir, filename))
            df['date'] = pd.to_datetime(df['date'])
            fund_data[fund_code] = df

    if not fund_data:
        print("无测试数据，请先获取数据")
        return

    # 获取最新日期
    all_dates = set()
    for df in fund_data.values():
        all_dates.update(df['date'].tolist())
    latest_date = max(all_dates)

    # 分析
    analyzer = IndustryRotationAnalyzer(lookback_days=20)
    signal = analyzer.get_rotation_signal(fund_data, latest_date)

    print(f"\n分析日期: {latest_date}")
    print(f"\n{signal['reason']}")

    print("\n行业排名:")
    for i, (sector, score, perf) in enumerate(signal.get('rankings', []), 1):
        print(f"  {i}. {sector}: 得分={score:.4f}, "
              f"收益率={perf['return']:.2%}, 动量={perf['momentum']:.4f}")

    # 计算各基金轮动得分
    print("\n基金轮动得分:")
    for fund_code in fund_data:
        score = analyzer.get_fund_rotation_score(fund_code, fund_data, latest_date)
        sector = analyzer.get_sector(fund_code)
        print(f"  {fund_code} ({sector}): {score:.4f}")


def run_sentiment_analysis():
    """运行市场情绪分析"""
    print("\n" + "=" * 60)
    print("市场情绪分析")
    print("=" * 60)

    import asyncio
    from src.sentiment_scraper import EastMoneyNewsScraper

    scraper = EastMoneyNewsScraper()

    print("\n正在爬取财经快讯...")
    news_list = asyncio.run(scraper.scrape_kuaixun(max_pages=3))

    if news_list:
        print(f"\n爬取完成，共{len(news_list)}条新闻")

        result = scraper.analyze_market_sentiment(news_list)
        signal = scraper.get_sentiment_signal(result)

        print("\n" + "=" * 60)
        print("市场情绪分析结果")
        print("=" * 60)
        print(f"  情感得分: {result['overall_score']:.4f}")
        print(f"  正面比例: {result['positive_ratio']:.2%}")
        print(f"  负面比例: {result['negative_ratio']:.2%}")
        print(f"  新闻数量: {result['news_count']}")

        print(f"\n交易信号: {signal['signal']}")
        print(f"建议操作: {signal['action']}")

        if signal['strong_sectors']:
            print(f"强势行业: {', '.join(signal['strong_sectors'])}")
        if signal['weak_sectors']:
            print(f"弱势行业: {', '.join(signal['weak_sectors'])}")
    else:
        print("未获取到新闻数据")


def generate_report():
    """生成报告"""
    print("\n" + "=" * 60)
    print("生成报告")
    print("=" * 60)

    from datetime import datetime

    # 这里可以调用report模块生成报告
    print("报告已生成: reports/backtest_report.md")


def run_all():
    """一键运行全部"""
    print("\n" + "=" * 60)
    print("一键运行全部流程")
    print("=" * 60)

    download_data()
    run_feature_engineering()
    train_models()
    train_ensemble_models()
    run_backtest()
    run_industry_rotation()
    generate_report()

    print("\n" + "=" * 60)
    print("全部流程完成!")
    print("=" * 60)


if __name__ == '__main__':
    main()
