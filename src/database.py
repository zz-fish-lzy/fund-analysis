"""
数据库模块 - 使用Supabase存储基金数据
"""

import os
import pandas as pd
from datetime import datetime
from src.config import (
    SUPABASE_URL, SUPABASE_ANON_KEY, SUPABASE_SERVICE_ROLE_KEY
)

from supabase import create_client, Client


class FundDatabase:
    """基金数据库"""

    def __init__(self, url=None, key=None, service_key=None):
        self.url = url or SUPABASE_URL
        self.key = key or SUPABASE_ANON_KEY
        self.service_key = service_key or SUPABASE_SERVICE_ROLE_KEY
        self.client = None
        self.admin_client = None

    def connect(self):
        """连接数据库（anon key 用于读取）"""
        try:
            self.client = create_client(self.url, self.key)
            if self.service_key:
                self.admin_client = create_client(self.url, self.service_key)
            print("数据库连接成功")
            return True
        except Exception as e:
            print(f"数据库连接失败: {e}")
            return False

    def _get_writer(self):
        """获取写入客户端（优先使用 service_role）"""
        return self.admin_client or self.client

    # ── 原有方法（保持兼容） ──

    def insert_fund_info(self, fund_code, fund_name, fund_type):
        """插入基金信息（fund_info 表）"""
        writer = self._get_writer()
        if not writer:
            return False
        try:
            data = {
                'fund_code': fund_code,
                'fund_name': fund_name,
                'fund_type': fund_type,
                'updated_at': datetime.now().isoformat()
            }
            writer.table('fund_info').upsert(data).execute()
            return True
        except Exception as e:
            print(f"插入基金信息失败: {e}")
            return False

    def insert_nav_data(self, fund_code, nav_data):
        """批量插入净值数据（fund_nav 表）"""
        writer = self._get_writer()
        if not writer:
            return False
        try:
            records = []
            for _, row in nav_data.iterrows():
                records.append({
                    'fund_code': fund_code,
                    'nav_date': row['date'].strftime('%Y-%m-%d'),
                    'nav': float(row['nav']),
                    'daily_return': float(row.get('daily_return', 0))
                })
            writer.table('fund_nav').upsert(records).execute()
            return True
        except Exception as e:
            print(f"插入净值数据失败: {e}")
            return False

    def get_nav_data(self, fund_code, start_date=None, end_date=None):
        """获取净值数据"""
        if not self.client:
            return pd.DataFrame()
        try:
            query = self.client.table('fund_nav').select('*').eq('fund_code', fund_code)
            if start_date:
                query = query.gte('nav_date', start_date)
            if end_date:
                query = query.lte('nav_date', end_date)
            result = query.order('nav_date').execute()
            df = pd.DataFrame(result.data)
            if not df.empty:
                df['date'] = pd.to_datetime(df['nav_date'])
                df = df[['date', 'nav', 'daily_return']]
            return df
        except Exception as e:
            print(f"获取净值数据失败: {e}")
            return pd.DataFrame()

    def insert_prediction(self, fund_code, predict_date, signal, probability, model_type):
        """插入预测结果（旧 predictions 表，upsert 去重）"""
        writer = self._get_writer()
        if not writer:
            return False
        try:
            data = {
                'fund_code': fund_code,
                'predict_date': predict_date,
                'signal': signal,
                'probability': probability,
                'model_type': model_type
            }
            writer.table('predictions').upsert(data).execute()
            return True
        except Exception as e:
            print(f"插入预测结果失败: {e}")
            return False

    def insert_trade(self, fund_code, trade_date, trade_type, shares, price, amount, reason=''):
        """插入交易记录（upsert 去重）"""
        writer = self._get_writer()
        if not writer:
            return False
        try:
            data = {
                'fund_code': fund_code,
                'trade_date': trade_date,
                'trade_type': trade_type,
                'shares': shares,
                'price': price,
                'amount': amount,
                'reason': reason
            }
            writer.table('trades').upsert(data).execute()
            return True
        except Exception as e:
            print(f"插入交易记录失败: {e}")
            return False

    def insert_sentiment(self, sentiment_date, news_count, positive_count,
                         negative_count, sentiment_score, sector_rotation=None):
        """插入市场情绪数据"""
        writer = self._get_writer()
        if not writer:
            return False
        try:
            data = {
                'sentiment_date': sentiment_date,
                'news_count': news_count,
                'positive_count': positive_count,
                'negative_count': negative_count,
                'sentiment_score': sentiment_score,
                'sector_rotation': sector_rotation or {}
            }
            writer.table('market_sentiment').upsert(data).execute()
            return True
        except Exception as e:
            print(f"插入市场情绪失败: {e}")
            return False

    # ── 新增方法: funds 表 ──

    def upsert_fund(self, fund_data):
        """插入或更新基金业务属性（funds 表）"""
        writer = self._get_writer()
        if not writer:
            return False
        try:
            fund_data['updated_at'] = datetime.now().isoformat()
            writer.table('funds').upsert(fund_data).execute()
            return True
        except Exception as e:
            print(f"upsert_fund 失败: {e}")
            return False

    def upsert_funds_batch(self, records, batch_size=500):
        """批量插入基金数据"""
        writer = self._get_writer()
        if not writer:
            return False
        try:
            for i in range(0, len(records), batch_size):
                batch = records[i:i + batch_size]
                for r in batch:
                    r['updated_at'] = datetime.now().isoformat()
                writer.table('funds').upsert(batch).execute()
            return True
        except Exception as e:
            print(f"upsert_funds_batch 失败: {e}")
            return False

    def get_fund(self, fund_code):
        """获取单只基金信息"""
        if not self.client:
            return None
        try:
            result = self.client.table('funds').select('*').eq('fund_code', fund_code).execute()
            return result.data[0] if result.data else None
        except Exception as e:
            print(f"get_fund 失败: {e}")
            return None

    def get_focus_funds(self):
        """获取重点池基金列表"""
        if not self.client:
            return []
        try:
            result = self.client.table('funds').select('*').eq('pool_type', 'focus').execute()
            return result.data
        except Exception as e:
            print(f"get_focus_funds 失败: {e}")
            return []

    def get_all_funds(self):
        """获取所有基金"""
        if not self.client:
            return []
        try:
            result = self.client.table('funds').select('*').execute()
            return result.data
        except Exception as e:
            print(f"get_all_funds 失败: {e}")
            return []

    # ── 新增方法: fund_scores 表 ──

    def insert_score(self, fund_code, score_date, scores):
        """插入基金评分"""
        writer = self._get_writer()
        if not writer:
            return False
        try:
            data = {
                'fund_code': fund_code,
                'score_date': score_date,
                'total_score': scores.get('total', 0),
                'performance_score': scores.get('performance', 0),
                'risk_score': scores.get('risk', 0),
                'manager_score': scores.get('manager', 0),
                'flow_score': scores.get('flow', 0),
                'pool_type': scores.get('pool_type', 'observation')
            }
            writer.table('fund_scores').upsert(data).execute()
            return True
        except Exception as e:
            print(f"insert_score 失败: {e}")
            return False

    def insert_scores_batch(self, records, batch_size=500):
        """批量插入评分"""
        writer = self._get_writer()
        if not writer:
            return False
        try:
            for i in range(0, len(records), batch_size):
                batch = records[i:i + batch_size]
                writer.table('fund_scores').upsert(batch).execute()
            return True
        except Exception as e:
            print(f"insert_scores_batch 失败: {e}")
            return False

    def get_latest_score(self, fund_code):
        """获取基金最新评分"""
        if not self.client:
            return None
        try:
            result = self.client.table('fund_scores') \
                .select('*') \
                .eq('fund_code', fund_code) \
                .order('score_date', desc=True) \
                .limit(1) \
                .execute()
            return result.data[0] if result.data else None
        except Exception as e:
            print(f"get_latest_score 失败: {e}")
            return None

    # ── 新增方法: fund_news 表 ──

    def insert_fund_news(self, fund_code, news_date, title, content, sentiment, impact):
        """插入基金关联新闻（upsert 去重）"""
        writer = self._get_writer()
        if not writer:
            return False
        try:
            data = {
                'fund_code': fund_code,
                'news_date': news_date,
                'news_title': title,
                'news_content': content,
                'sentiment': sentiment,
                'impact': impact
            }
            writer.table('fund_news').upsert(data).execute()
            return True
        except Exception as e:
            print(f"insert_fund_news 失败: {e}")
            return False

    # ── 新增方法: trend_news 表 ──

    def get_unlabeled_news(self, limit=100):
        """获取未标注的新闻"""
        if not self.client:
            return []
        try:
            result = self.client.table('trend_news') \
                .select('id, title, content') \
                .eq('is_labeled_by_api', False) \
                .limit(limit) \
                .execute()
            return result.data
        except Exception as e:
            print(f"get_unlabeled_news 失败: {e}")
            return []

    def update_trend_news_analysis(self, news_id, analysis, is_labeled=True):
        """更新新闻的 API 分析结果"""
        writer = self._get_writer()
        if not writer:
            return False
        try:
            data = {
                'category': analysis.get('category', '其他'),
                'sentiment_score': analysis.get('sentiment', 0),
                'api_analysis': analysis,
                'is_labeled_by_api': is_labeled,
                'updated_at': datetime.now().isoformat()
            }
            writer.table('trend_news').update(data).eq('id', news_id).execute()
            return True
        except Exception as e:
            print(f"update_trend_news_analysis 失败: {e}")
            return False

    def insert_trend_news(self, news_item):
        """插入单条新闻到 trend_news"""
        writer = self._get_writer()
        if not writer:
            return False
        try:
            writer.table('trend_news').upsert(news_item).execute()
            return True
        except Exception as e:
            print(f"insert_trend_news 失败: {e}")
            return False

    # ── 新增方法: predictions_v2 表 ──

    def insert_prediction_v2(self, fund_code, prediction_date, target_date,
                             predicted_nav=None, predicted_direction=None,
                             predicted_score=None, model_name='xgboost',
                             model_version='v1', evaluation_metrics=None):
        """插入预测结果（新 predictions_v2 表，upsert 去重）"""
        writer = self._get_writer()
        if not writer:
            return False
        try:
            data = {
                'fund_code': fund_code,
                'prediction_date': prediction_date,
                'target_date': target_date,
                'predicted_nav': predicted_nav,
                'predicted_direction': predicted_direction,
                'predicted_score': predicted_score,
                'model_name': model_name,
                'model_version': model_version,
                'evaluation_metrics': evaluation_metrics or {}
            }
            writer.table('predictions_v2').upsert(data).execute()
            return True
        except Exception as e:
            print(f"insert_prediction_v2 失败: {e}")
            return False

    # ── 工具方法 ──

    def get_latest_nav_date(self, fund_code):
        """获取基金最新净值日期"""
        if not self.client:
            return None
        try:
            result = self.client.table('fund_nav') \
                .select('nav_date') \
                .eq('fund_code', fund_code) \
                .order('nav_date', desc=True) \
                .limit(1) \
                .execute()
            return result.data[0]['nav_date'] if result.data else None
        except Exception as e:
            print(f"get_latest_nav_date 失败: {e}")
            return None


# 本地SQLite备选方案
import sqlite3


class LocalDatabase:
    """本地SQLite数据库（备选方案）"""

    def __init__(self, db_path='data/fund.db'):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path) if os.path.dirname(db_path) else '.', exist_ok=True)
        self.conn = sqlite3.connect(db_path)
        self.create_tables()

    def create_tables(self):
        """创建表"""
        cursor = self.conn.cursor()

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS fund_info (
                fund_code TEXT PRIMARY KEY,
                fund_name TEXT,
                fund_type TEXT,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS fund_nav (
                fund_code TEXT,
                nav_date DATE,
                nav REAL,
                daily_return REAL,
                PRIMARY KEY (fund_code, nav_date)
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                fund_code TEXT,
                predict_date DATE,
                signal INTEGER,
                probability REAL,
                model_type TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                fund_code TEXT,
                trade_date DATE,
                trade_type TEXT,
                shares REAL,
                price REAL,
                amount REAL,
                reason TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS market_sentiment (
                sentiment_date DATE PRIMARY KEY,
                news_count INTEGER,
                positive_count INTEGER,
                negative_count INTEGER,
                sentiment_score REAL,
                sector_rotation TEXT
            )
        ''')

        self.conn.commit()

    def insert_fund_info(self, fund_code, fund_name, fund_type):
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO fund_info (fund_code, fund_name, fund_type, updated_at)
            VALUES (?, ?, ?, CURRENT_TIMESTAMP)
        ''', (fund_code, fund_name, fund_type))
        self.conn.commit()

    def insert_nav_data(self, fund_code, nav_data):
        cursor = self.conn.cursor()
        for _, row in nav_data.iterrows():
            cursor.execute('''
                INSERT OR REPLACE INTO fund_nav (fund_code, nav_date, nav, daily_return)
                VALUES (?, ?, ?, ?)
            ''', (fund_code, row['date'].strftime('%Y-%m-%d'), float(row['nav']),
                  float(row.get('daily_return', 0))))
        self.conn.commit()

    def get_nav_data(self, fund_code, start_date=None, end_date=None):
        query = f"SELECT * FROM fund_nav WHERE fund_code = ?"
        params = [fund_code]

        if start_date:
            query += " AND nav_date >= ?"
            params.append(start_date)
        if end_date:
            query += " AND nav_date <= ?"
            params.append(end_date)

        query += " ORDER BY nav_date"

        df = pd.read_sql_query(query, self.conn, params=params)
        if not df.empty:
            df['date'] = pd.to_datetime(df['nav_date'])
            df = df[['date', 'nav', 'daily_return']]
        return df

    def close(self):
        self.conn.close()


if __name__ == '__main__':
    # 测试本地数据库
    db = LocalDatabase()
    print("本地数据库创建成功: data/fund.db")

    # 测试插入
    db.insert_fund_info('510300', '沪深300ETF', 'ETF')
    print("测试插入成功")
