"""
邮箱推送模块
晚间完整报告推送到 QQ 邮箱
"""

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime

from src.config import (
    EMAIL_SMTP_SERVER, EMAIL_SMTP_PORT,
    EMAIL_USER, EMAIL_AUTH_CODE
)


def send_email(subject, html_body, to_addr=None):
    """
    发送 HTML 邮件

    Args:
        subject: 邮件主题
        html_body: HTML 格式正文
        to_addr: 收件人（默认同发件人）

    Returns:
        bool: 是否成功
    """
    to_addr = to_addr or EMAIL_USER

    msg = MIMEMultipart('alternative')
    msg['Subject'] = subject
    msg['From'] = EMAIL_USER
    msg['To'] = to_addr
    msg.attach(MIMEText(html_body, 'html', 'utf-8'))

    try:
        with smtplib.SMTP_SSL(EMAIL_SMTP_SERVER, EMAIL_SMTP_PORT) as server:
            server.login(EMAIL_USER, EMAIL_AUTH_CODE)
            server.sendmail(EMAIL_USER, to_addr, msg.as_string())
        print(f"邮件发送成功: {subject}")
        return True
    except Exception as e:
        print(f"邮件发送失败: {e}")
        return False


def build_daily_report(score_changes, news_highlights, predictions_summary, report_date=None):
    """
    构建晚间完整报告 HTML

    Args:
        score_changes: list of {fund_code, fund_name, total_score, score_change, pool_type}
        news_highlights: list of {title, category, sentiment, impact, reason}
        predictions_summary: list of {fund_code, fund_name, predicted_direction, predicted_score}
        report_date: 报告日期（默认今天）

    Returns:
        str: HTML 正文
    """
    report_date = report_date or datetime.now().strftime('%Y-%m-%d')

    html = f"""
    <html>
    <head>
    <style>
        body {{ font-family: 'Microsoft YaHei', Arial; margin: 20px; color: #333; }}
        h2 {{ color: #1a5276; border-bottom: 2px solid #2980b9; padding-bottom: 5px; }}
        table {{ border-collapse: collapse; width: 100%; margin: 10px 0; }}
        th {{ background: #2980b9; color: white; padding: 8px 12px; text-align: left; }}
        td {{ padding: 6px 12px; border-bottom: 1px solid #ddd; }}
        tr:hover {{ background: #f5f5f5; }}
        .up {{ color: #e74c3c; font-weight: bold; }}
        .down {{ color: #27ae60; font-weight: bold; }}
        .neutral {{ color: #7f8c8d; }}
        .section {{ margin: 20px 0; }}
        .footer {{ color: #999; font-size: 12px; margin-top: 30px; border-top: 1px solid #eee; padding-top: 10px; }}
    </style>
    </head>
    <body>
    <h2>基金分析日报 — {report_date}</h2>
    """

    # 基金评分变动
    html += '<div class="section"><h3>基金评分变动</h3>'
    if score_changes:
        html += """<table><tr>
            <th>基金代码</th><th>基金名称</th><th>评分</th><th>变动</th><th>池类型</th>
        </tr>"""
        for item in score_changes[:20]:
            change = item.get('score_change', 0)
            cls = 'up' if change > 0 else ('down' if change < 0 else 'neutral')
            sign = '+' if change > 0 else ''
            pool_map = {'focus': '重点', 'observation': '观察', 'eliminate': '淘汰'}
            html += f"""<tr>
                <td>{item['fund_code']}</td>
                <td>{item.get('fund_name', '-')}</td>
                <td>{item.get('total_score', 0):.1f}</td>
                <td class="{cls}">{sign}{change:.1f}</td>
                <td>{pool_map.get(item.get('pool_type', ''), '-')}</td>
            </tr>"""
        html += '</table>'
    else:
        html += '<p>今日无评分变动</p>'
    html += '</div>'

    # 重要新闻
    html += '<div class="section"><h3>重要新闻</h3>'
    if news_highlights:
        html += """<table><tr>
            <th>标题</th><th>分类</th><th>情感</th><th>影响</th>
        </tr>"""
        sentiment_map = {1: ('利好', 'up'), -1: ('利空', 'down'), 0: ('中性', 'neutral')}
        for item in news_highlights[:15]:
            s_text, s_cls = sentiment_map.get(item.get('sentiment', 0), ('中性', 'neutral'))
            html += f"""<tr>
                <td>{item.get('title', '-')[:50]}</td>
                <td>{item.get('category', '-')}</td>
                <td class="{s_cls}">{s_text}</td>
                <td>{item.get('impact', '-')}</td>
            </tr>"""
        html += '</table>'
    else:
        html += '<p>今日无重要新闻</p>'
    html += '</div>'

    # 预测摘要
    html += '<div class="section"><h3>明日预测摘要</h3>'
    if predictions_summary:
        html += """<table><tr>
            <th>基金代码</th><th>基金名称</th><th>预测方向</th><th>置信度</th>
        </tr>"""
        dir_map = {1: ('看涨', 'up'), -1: ('看跌', 'down'), 0: ('持平', 'neutral')}
        for item in predictions_summary[:15]:
            d_text, d_cls = dir_map.get(item.get('predicted_direction', 0), ('持平', 'neutral'))
            html += f"""<tr>
                <td>{item['fund_code']}</td>
                <td>{item.get('fund_name', '-')}</td>
                <td class="{d_cls}">{d_text}</td>
                <td>{item.get('predicted_score', 0):.1%}</td>
            </tr>"""
        html += '</table>'
    else:
        html += '<p>暂无预测数据</p>'
    html += '</div>'

    html += f"""
    <div class="footer">
        本报告由基金分析与预测系统自动生成 | {report_date} 19:00
    </div>
    </body></html>
    """

    return html


def send_daily_report(score_changes, news_highlights, predictions_summary, report_date=None):
    """发送每日晚间报告"""
    report_date = report_date or datetime.now().strftime('%Y-%m-%d')
    subject = f"基金分析日报 — {report_date}"
    html = build_daily_report(score_changes, news_highlights, predictions_summary, report_date)
    return send_email(subject, html)


def send_alert(subject, message):
    """发送告警邮件（失败告警用）"""
    html = f"""
    <html><body>
    <h3 style="color: #e74c3c;">{subject}</h3>
    <p>{message}</p>
    <p style="color: #999;">发送时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </body></html>
    """
    return send_email(f"[告警] {subject}", html)


if __name__ == '__main__':
    # 测试发送
    test_scores = [
        {'fund_code': '510300', 'fund_name': '沪深300ETF', 'total_score': 75.5, 'score_change': 2.3, 'pool_type': 'focus'},
        {'fund_code': '005827', 'fund_name': '易方达蓝筹精选', 'total_score': 68.2, 'score_change': -1.1, 'pool_type': 'observation'},
    ]
    test_news = [
        {'title': '央行宣布降准0.5个百分点', 'category': '宏观经济', 'sentiment': 1, 'impact': 5},
        {'title': '新能源汽车销量创新高', 'category': '新能源', 'sentiment': 1, 'impact': 4},
    ]
    test_predictions = [
        {'fund_code': '510300', 'fund_name': '沪深300ETF', 'predicted_direction': 1, 'predicted_score': 0.62},
    ]
    send_daily_report(test_scores, test_news, test_predictions)
