-- ============================================================
-- 基金分析与预测项目 — 第一期数据库迁移脚本
-- 执行方式: 在 Supabase Dashboard → SQL Editor 中粘贴执行
-- ============================================================

-- 1. 新增: funds 基金业务属性表
CREATE TABLE IF NOT EXISTS funds (
    fund_code VARCHAR(10) PRIMARY KEY,
    fund_name VARCHAR(100),
    fund_type VARCHAR(20),
    fund_company VARCHAR(100),
    fund_manager VARCHAR(50),
    manager_start_date DATE,
    fund_size DECIMAL(15,2),           -- 规模(亿元)
    establishment_date DATE,
    management_fee DECIMAL(5,4),       -- 管理费率
    custody_fee DECIMAL(5,4),          -- 托管费率
    is_focus BOOLEAN DEFAULT FALSE,    -- 是否在重点池
    is_active BOOLEAN DEFAULT TRUE,    -- 是否活跃
    pool_type VARCHAR(20) DEFAULT 'observation',  -- focus/observation/eliminate
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- 2. 新增: fund_scores 基金评分表
CREATE TABLE IF NOT EXISTS fund_scores (
    id BIGSERIAL PRIMARY KEY,
    fund_code VARCHAR(10) REFERENCES funds(fund_code) ON DELETE CASCADE,
    score_date DATE NOT NULL,
    total_score DECIMAL(5,2),          -- 总分(0-100)
    performance_score DECIMAL(5,2),    -- 业绩分(0-40)
    risk_score DECIMAL(5,2),           -- 风险分(0-25)
    manager_score DECIMAL(5,2),        -- 经理分(0-20)
    flow_score DECIMAL(5,2),           -- 资金流分(0-15)
    pool_type VARCHAR(20),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(fund_code, score_date)
);

-- 3. 新增: fund_news 基金关联新闻表
CREATE TABLE IF NOT EXISTS fund_news (
    id BIGSERIAL PRIMARY KEY,
    fund_code VARCHAR(10) REFERENCES funds(fund_code) ON DELETE CASCADE,
    news_date DATE NOT NULL,
    news_title TEXT,
    news_content TEXT,
    sentiment INTEGER,          -- -1/0/1
    impact NUMERIC,             -- 1-5
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- 4. 扩展: trend_news 表增加字段
ALTER TABLE trend_news ADD COLUMN IF NOT EXISTS category TEXT;
ALTER TABLE trend_news ADD COLUMN IF NOT EXISTS api_analysis JSONB;
ALTER TABLE trend_news ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ DEFAULT NOW();
ALTER TABLE trend_news ADD COLUMN IF NOT EXISTS is_labeled_by_api BOOLEAN DEFAULT FALSE;

-- 5. 替换: predictions 表（新结构，支持评估指标）
-- 如果旧 predictions 表有数据需要先备份
CREATE TABLE IF NOT EXISTS predictions_v2 (
    id BIGSERIAL PRIMARY KEY,
    fund_code VARCHAR(10) REFERENCES funds(fund_code) ON DELETE CASCADE,
    prediction_date DATE NOT NULL,
    target_date DATE NOT NULL,
    predicted_nav NUMERIC,
    predicted_direction INTEGER,    -- 1=涨, -1=跌, 0=平
    predicted_score NUMERIC,        -- 置信度 0-1
    actual_nav NUMERIC,             -- 真实净值（事后回填）
    actual_direction INTEGER,       -- 真实涨跌方向（事后回填）
    model_name TEXT NOT NULL,
    model_version TEXT,
    evaluation_metrics JSONB,       -- {"mae": 0.02, "direction_acc": 0.58}
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(fund_code, prediction_date, target_date, model_name)
);

-- 6. 索引优化
CREATE INDEX IF NOT EXISTS idx_fund_nav_fund_code ON fund_nav(fund_code);
CREATE INDEX IF NOT EXISTS idx_fund_nav_date ON fund_nav(nav_date);
CREATE INDEX IF NOT EXISTS idx_funds_is_focus ON funds(is_focus) WHERE is_focus = TRUE;
CREATE INDEX IF NOT EXISTS idx_fund_scores_pool_type ON fund_scores(pool_type);
CREATE INDEX IF NOT EXISTS idx_fund_scores_date ON fund_scores(score_date);
CREATE INDEX IF NOT EXISTS idx_fund_news_date ON fund_news(news_date);
CREATE INDEX IF NOT EXISTS idx_fund_news_code ON fund_news(fund_code);
CREATE INDEX IF NOT EXISTS idx_trend_news_date ON trend_news(news_date);
CREATE INDEX IF NOT EXISTS idx_trend_news_category ON trend_news(category);
CREATE INDEX IF NOT EXISTS idx_trend_news_unlabeled ON trend_news(is_labeled_by_api)
    WHERE is_labeled_by_api = FALSE;
CREATE INDEX IF NOT EXISTS idx_predictions_v2_fund_code ON predictions_v2(fund_code);
CREATE INDEX IF NOT EXISTS idx_predictions_v2_date ON predictions_v2(prediction_date);
CREATE INDEX IF NOT EXISTS idx_predictions_v2_model ON predictions_v2(model_name);

-- 7. 视图: 重点基金最新数据
CREATE OR REPLACE VIEW focus_funds_latest AS
SELECT
    f.fund_code, f.fund_name, f.fund_type,
    fn.nav AS unit_nav, fn.daily_return,
    fs.total_score, fs.score_date
FROM funds f
LEFT JOIN LATERAL (
    SELECT nav, daily_return FROM fund_nav
    WHERE fund_code = f.fund_code
    ORDER BY nav_date DESC LIMIT 1
) fn ON TRUE
LEFT JOIN LATERAL (
    SELECT total_score, score_date FROM fund_scores
    WHERE fund_code = f.fund_code
    ORDER BY score_date DESC LIMIT 1
) fs ON TRUE
WHERE f.is_focus = TRUE;

-- 8. 视图: 最新评分
CREATE OR REPLACE VIEW v_latest_scores AS
SELECT DISTINCT ON (fund_code)
    fund_code, score_date, total_score, performance_score,
    risk_score, manager_score, flow_score, pool_type
FROM fund_scores
ORDER BY fund_code, score_date DESC;
