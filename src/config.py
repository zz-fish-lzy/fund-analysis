"""
集中配置模块
管理所有外部服务凭证、限流参数、新闻过滤规则
"""

import os
import socket
from pathlib import Path


def load_env():
    """加载 .env 文件到环境变量"""
    env_path = Path(__file__).parent.parent / '.env'
    if env_path.exists():
        with open(env_path, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, _, value = line.partition('=')
                    os.environ.setdefault(key.strip(), value.strip())


def setup_proxy(proxy_url='http://127.0.0.1:7897'):
    """检测代理是否可用，可用则设置，否则直连"""
    try:
        host, port = proxy_url.split('//')[1].split(':')
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(1)
            s.connect((host, int(port)))
        os.environ['http_proxy'] = proxy_url
        os.environ['https_proxy'] = proxy_url
        print(f"[proxy] 已启用: {proxy_url}")
        return True
    except Exception:
        os.environ.pop('http_proxy', None)
        os.environ.pop('https_proxy', None)
        print("[proxy] 不可用，使用直连")
        return False


# 模块加载时自动读取 .env 并检测代理
load_env()
setup_proxy()

# ── Supabase ──
SUPABASE_URL = os.environ.get('SUPABASE_URL', '')
SUPABASE_ANON_KEY = os.environ.get('SUPABASE_ANON_KEY', '')
SUPABASE_SERVICE_ROLE_KEY = os.environ.get('SUPABASE_SERVICE_ROLE_KEY', '')

# ── 腾讯云 COS ──
COS_SECRET_ID = os.environ.get('COS_SECRET_ID', '')
COS_SECRET_KEY = os.environ.get('COS_SECRET_KEY', '')
COS_BUCKET = os.environ.get('COS_BUCKET', 'trendradar-data-1429116002')
COS_REGION = os.environ.get('COS_REGION', 'ap-chongqing')

# ── DeepSeek API ──
DEEPSEEK_API_KEY = os.environ.get('DEEPSEEK_API_KEY', '')
DEEPSEEK_BASE_URL = 'https://api.deepseek.com'
DEEPSEEK_MODEL = 'deepseek-v4-flash'

# ── 邮箱推送 ──
EMAIL_SMTP_SERVER = 'smtp.qq.com'
EMAIL_SMTP_PORT = 465
EMAIL_USER = '2085549322@qq.com'
EMAIL_AUTH_CODE = 'dkciinxjojwhdiib'

# ── AKShare 限流参数（常规模式） ──
AKSHARE_JITTER_MIN = 1.0    # 2.5 - 1.5 秒
AKSHARE_JITTER_MAX = 4.0    # 2.5 + 1.5 秒
AKSHARE_BATCH_SIZE = 50      # 每批处理基金数
AKSHARE_BATCH_REST = 600     # 批间休息秒数（10 分钟）
AKSHARE_MAX_RETRIES = 3      # 最大重试次数

# ── AKShare 快速模式（周末/非交易时段批量下载） ──
AKSHARE_FAST_JITTER_MIN = 0.3
AKSHARE_FAST_JITTER_MAX = 0.8
AKSHARE_FAST_BATCH_SIZE = 200
AKSHARE_FAST_BATCH_REST = 30
AKSHARE_FAST_WORKERS = 4     # 多线程并发数

# ── 新闻过滤关键词 ──
NEWS_WHITELIST = [
    '基金', '净值', '分红', '限购', '基金经理', '持仓',
    '沪深300', '中证500', '创业板', '科创板',
    '新能源', '医药', '消费', '半导体', '人工智能',
    '降息', '降准', '加息', 'CPI', 'PMI', 'GDP',
    'A股', '央行', '经济', '政策', '利好', '利空',
    '行业', '板块', '利率', '通胀', 'GDP增速',
]

NEWS_BLACKLIST = [
    '招聘', '年会', '辟谣', '致歉', '离职', '任命',
    '专访', '直播', '广告', '推广', '开户', '荐股',
    '骗局', '维权', '炒股软件',
]

# ── 新闻分类 ──
NEWS_CATEGORIES = [
    "宏观经济", "新能源", "医药生物", "消费", "科技",
    "金融地产", "债券", "其他"
]

# ── 基金池阈值 ──
FOCUS_SCORE_THRESHOLD = 70      # >= 70 进重点池
OBSERVATION_SCORE_THRESHOLD = 50  # >= 50 进观察池
# < 50 进淘汰池

# ── 混合存储 ──
COLD_STORAGE_DIR = 'data/cold_nav'  # 本地Parquet冷存储目录
HOT_NAV_MONTHS = 3                  # Supabase保留最近N个月净值
