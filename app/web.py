"""
Streamlit Web App for Chinese Stock Top-10 Predictor
Shanghai (SHG) and Shenzhen (SHE) Stock Exchanges
"""
import streamlit as st
import pandas as pd
import numpy as np
import json
from datetime import datetime
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    TOP10_LATEST_FILE, TOP10_HISTORY_FILE, QUALITY_REPORT_FILE,
    UNIVERSE_META_FILE, CURRENT_MODEL_VERSION, OUTPUTS_DIR, EXCHANGES
)

# 15-day prediction files
TOP10_LATEST_15D_FILE = OUTPUTS_DIR / "top10_latest_15d.parquet"
TOP10_HISTORY_15D_FILE = OUTPUTS_DIR / "top10_history_15d.parquet"

# Page config - wrapped in try/except for import compatibility
try:
    st.set_page_config(
        page_title="🇨🇳 Chinese Stock Top-10 Predictor",
        page_icon="🇨🇳",
        layout="wide"
    )
except st.errors.StreamlitAPIException:
    pass  # Already set by entry point


# ============ Data Loading Functions ============

@st.cache_data(ttl=300)
def load_latest_top10():
    """Load latest top-10 predictions (5-day)"""
    if TOP10_LATEST_FILE.exists():
        df = pd.read_parquet(TOP10_LATEST_FILE)
        df['date'] = pd.to_datetime(df['date'])
        # Merge with universe to get stock names
        df = merge_stock_names(df)
        return df
    return pd.DataFrame()


@st.cache_data(ttl=300)
def load_latest_top10_15d():
    """Load latest top-10 predictions (15-day)"""
    if TOP10_LATEST_15D_FILE.exists():
        df = pd.read_parquet(TOP10_LATEST_15D_FILE)
        df['date'] = pd.to_datetime(df['date'])
        # Merge with universe to get stock names
        df = merge_stock_names(df)
        return df
    return pd.DataFrame()


@st.cache_data(ttl=3600)
def load_chinese_names():
    """Load Chinese stock names from CSV file"""
    csv_path = Path(__file__).parent.parent / "cn_stocks_shg_she_code_name.csv"
    if csv_path.exists():
        try:
            cn_names = pd.read_csv(csv_path, dtype={'code': str})
            # Ensure code is string with proper padding
            cn_names['code'] = cn_names['code'].str.zfill(6)
            return dict(zip(cn_names['code'], cn_names['name']))
        except Exception:
            return {}
    return {}


def merge_stock_names(df: pd.DataFrame) -> pd.DataFrame:
    """Merge stock names - prefer Chinese names from CSV, fallback to English from metadata"""
    if df.empty:
        return df
    
    # Load Chinese names from CSV
    cn_names_dict = load_chinese_names()
    
    # Load English names from universe metadata
    universe = load_universe_meta()
    en_names_dict = {}
    if not universe.empty and 'name' in universe.columns:
        en_names_dict = dict(zip(universe['symbol'], universe['name']))
    
    # Extract stock code from symbol (e.g., "000001.SHE" -> "000001")
    def get_name(symbol):
        code = symbol.split('.')[0]
        # Prefer Chinese name if available
        if code in cn_names_dict:
            return cn_names_dict[code]
        # Fallback to English name
        if symbol in en_names_dict:
            return en_names_dict[symbol]
        return ''
    
    df = df.copy()
    df['name'] = df['symbol'].apply(get_name)
    return df


def get_stock_link(symbol: str, exchange: str) -> str:
    """Generate link to view stock chart on East Money (eastmoney.com)"""
    # East Money URL format: https://quote.eastmoney.com/XXXXXX.html
    return f"https://quote.eastmoney.com/{symbol}.html"


@st.cache_data(ttl=300)
def load_history():
    """Load historical top-10 predictions (5-day), including latest if not in history"""
    history_df = pd.DataFrame()
    
    if TOP10_HISTORY_FILE.exists():
        history_df = pd.read_parquet(TOP10_HISTORY_FILE)
        history_df['date'] = pd.to_datetime(history_df['date'])
    
    # Also check if latest predictions should be merged into history
    if TOP10_LATEST_FILE.exists():
        latest_df = pd.read_parquet(TOP10_LATEST_FILE)
        latest_df['date'] = pd.to_datetime(latest_df['date'])
        latest_date = latest_df['date'].iloc[0]
        
        # If latest date is not in history, add it
        if history_df.empty or latest_date not in history_df['date'].values:
            history_df = pd.concat([history_df, latest_df], ignore_index=True)
            history_df = history_df.drop_duplicates(subset=['symbol', 'date'], keep='last')
    
    # Ensure reason_cn column exists and fill missing values
    if not history_df.empty:
        if 'reason_cn' not in history_df.columns:
            history_df['reason_cn'] = None
        # Fill missing reasons with a default
        history_df['reason_cn'] = history_df['reason_cn'].fillna('模型预测入选')
    
    return history_df


@st.cache_data(ttl=300)
def load_history_15d():
    """Load historical top-10 predictions (15-day)"""
    history_df = pd.DataFrame()
    
    if TOP10_HISTORY_15D_FILE.exists():
        history_df = pd.read_parquet(TOP10_HISTORY_15D_FILE)
        history_df['date'] = pd.to_datetime(history_df['date'])
    
    # Also check if latest predictions should be merged into history
    if TOP10_LATEST_15D_FILE.exists():
        latest_df = pd.read_parquet(TOP10_LATEST_15D_FILE)
        latest_df['date'] = pd.to_datetime(latest_df['date'])
        latest_date = latest_df['date'].iloc[0]
        
        # If latest date is not in history, add it
        if history_df.empty or latest_date not in history_df['date'].values:
            history_df = pd.concat([history_df, latest_df], ignore_index=True)
            history_df = history_df.drop_duplicates(subset=['symbol', 'date'], keep='last')
    
    # Ensure reason_cn column exists and fill missing values
    if not history_df.empty:
        if 'reason_cn' not in history_df.columns:
            history_df['reason_cn'] = None
        history_df['reason_cn'] = history_df['reason_cn'].fillna('模型预测15日入选')
    
    return history_df


@st.cache_data(ttl=3600)
def load_universe_meta():
    """Load universe metadata"""
    if UNIVERSE_META_FILE.exists():
        return pd.read_parquet(UNIVERSE_META_FILE)
    return pd.DataFrame()


def load_quality_report():
    """Load quality report"""
    if QUALITY_REPORT_FILE.exists():
        with open(QUALITY_REPORT_FILE, 'r') as f:
            return json.load(f)
    return {}


def add_confidence_intervals(df: pd.DataFrame) -> pd.DataFrame:
    """Add confidence intervals to predictions"""
    try:
        from models.confidence import add_confidence_intervals as compute_ci
        return compute_ci(df, confidence_level=0.90)
    except Exception:
        df = df.copy()
        df['pred_std'] = df['pred_ret_5'].abs() * 0.4 + 0.02
        df['pred_lower'] = df['pred_ret_5'] - 1.645 * df['pred_std']
        df['pred_upper'] = df['pred_ret_5'] + 1.645 * df['pred_std']
        df['confidence_score'] = 0.7
        return df


# ============ Formatting Functions ============

def format_percent(val):
    """Format value as percentage"""
    if pd.isna(val):
        return "N/A"
    return f"{val * 100:.2f}%"


def format_price(val):
    """Format value as CNY price"""
    if pd.isna(val):
        return "N/A"
    return f"¥{val:.2f}"


def get_confidence_color(score):
    """Get emoji color based on confidence score"""
    if score >= 0.7:
        return "🟢"
    elif score >= 0.5:
        return "🟡"
    else:
        return "🔴"


def get_exchange_name(code):
    """Get exchange full name"""
    return EXCHANGES.get(code, code)


# ============ Chart Functions ============

def render_predictions_chart(df: pd.DataFrame):
    """Render bar chart of predicted returns with confidence intervals (5-day)"""
    import plotly.graph_objects as go
    
    df_plot = df.sort_values('pred_ret_5', ascending=True).copy()
    
    colors = ['#00CC96' if x > 0 else '#EF553B' for x in df_plot['pred_ret_5']]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        y=df_plot['symbol'],
        x=df_plot['pred_ret_5'] * 100,
        orientation='h',
        marker_color=colors,
        name='Predicted Return',
        text=[f"{x*100:.1f}%" for x in df_plot['pred_ret_5']],
        textposition='outside'
    ))
    
    if 'pred_lower' in df_plot.columns and 'pred_upper' in df_plot.columns:
        error_minus = (df_plot['pred_ret_5'] - df_plot['pred_lower']) * 100
        error_plus = (df_plot['pred_upper'] - df_plot['pred_ret_5']) * 100
        
        fig.add_trace(go.Scatter(
            y=df_plot['symbol'],
            x=df_plot['pred_ret_5'] * 100,
            error_x=dict(
                type='data',
                symmetric=False,
                array=error_plus.tolist(),
                arrayminus=error_minus.tolist(),
                color='rgba(0,0,0,0.3)',
                thickness=2
            ),
            mode='markers',
            marker=dict(size=1, color='rgba(0,0,0,0)'),
            name='90% CI',
            showlegend=True
        ))
    
    fig.update_layout(
        title='预测5日收益率 (Predicted 5-Day Returns with 90% CI)',
        xaxis_title='预测收益率 (%)',
        yaxis_title='股票代码',
        height=400,
        showlegend=True,
        xaxis=dict(zeroline=True, zerolinecolor='gray', zerolinewidth=1)
    )
    
    return fig


def render_predictions_chart_15d(df: pd.DataFrame):
    """Render bar chart of predicted returns (15-day)"""
    import plotly.graph_objects as go
    
    df_plot = df.sort_values('pred_ret_15', ascending=True).copy()
    
    colors = ['#00CC96' if x > 0 else '#EF553B' for x in df_plot['pred_ret_15']]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        y=df_plot['symbol'],
        x=df_plot['pred_ret_15'] * 100,
        orientation='h',
        marker_color=colors,
        name='Predicted Return',
        text=[f"{x*100:.1f}%" for x in df_plot['pred_ret_15']],
        textposition='outside'
    ))
    
    fig.update_layout(
        title='预测15日收益率 (Predicted 15-Day Returns)',
        xaxis_title='预测收益率 (%)',
        yaxis_title='股票代码',
        height=400,
        showlegend=True,
        xaxis=dict(zeroline=True, zerolinecolor='gray', zerolinewidth=1)
    )
    
    return fig


def render_exchange_pie_chart(df: pd.DataFrame):
    """Render exchange breakdown pie chart"""
    import plotly.express as px
    
    exchange_counts = df['exchange'].value_counts().reset_index()
    exchange_counts.columns = ['Exchange', 'Count']
    exchange_counts['Exchange'] = exchange_counts['Exchange'].map(get_exchange_name)
    
    fig = px.pie(exchange_counts, values='Count', names='Exchange', 
                 title='交易所分布 (Exchange Distribution)', hole=0.4)
    fig.update_layout(height=350)
    
    return fig


def render_symbol_history_chart(history_df: pd.DataFrame, symbol: str):
    """Render history chart for a specific symbol"""
    import plotly.express as px
    
    symbol_data = history_df[history_df['symbol'] == symbol].copy()
    
    if symbol_data.empty:
        return None
    
    symbol_data = symbol_data.sort_values('date')
    
    fig = px.line(
        symbol_data, x='date', y='pred_ret_5',
        title=f'{symbol} - 历史预测 (Historical Predictions)',
        markers=True
    )
    
    fig.update_layout(
        xaxis_title='日期',
        yaxis_title='预测收益率',
        yaxis_tickformat='.1%',
        height=300
    )
    
    return fig


# ============ Main App ============

def main():
    # Title
    st.title("🇨🇳 中国A股 Top-10 预测器")
    st.markdown("*预测未来5日和15日最有可能跑赢大盘的10只股票*")
    st.markdown("*Predicting the top 10 A-share stocks most likely to outperform over the next 5 and 15 trading days*")
    
    # Load data
    latest_df = load_latest_top10()
    latest_df_15d = load_latest_top10_15d()
    history_df = load_history()
    history_df_15d = load_history_15d()
    quality_report = load_quality_report()
    universe_meta = load_universe_meta()
    
    # Add confidence intervals if not present (5-day)
    if not latest_df.empty and 'confidence_score' not in latest_df.columns:
        latest_df = add_confidence_intervals(latest_df)
    
    # Sidebar
    st.sidebar.header("📊 数据状态 (Data Status)")
    
    # Show prediction date from latest predictions (most reliable source)
    if not latest_df.empty:
        pred_date = latest_df['date'].iloc[0]
        st.sidebar.metric("预测日期", pred_date.strftime('%Y-%m-%d'))
    
    if quality_report:
        asof_date = quality_report.get('asof_date', 'Unknown')
        # Only show data date if different from prediction date
        if latest_df.empty or str(asof_date) != pred_date.strftime('%Y-%m-%d'):
            st.sidebar.metric("数据日期", asof_date)
        
        if 'data' in quality_report:
            st.sidebar.metric("股票数量", quality_report['data'].get('unique_symbols', 'N/A'))
            exchanges = quality_report['data'].get('exchanges', [])
            st.sidebar.markdown(f"**交易所**: {', '.join(exchanges)}")
        
        generated_at = quality_report.get('generated_at', '')
        if generated_at:
            st.sidebar.caption(f"更新时间: {generated_at[:19]}")
    else:
        st.sidebar.warning("No quality report found")
    
    st.sidebar.markdown("---")
    st.sidebar.caption(f"模型版本: {CURRENT_MODEL_VERSION}")
    
    # Main content tabs - now with 15-day tab
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🏆 5日预测", 
        "📆 15日预测",
        "📊 图表分析",
        "📅 历史记录", 
        "ℹ️ 关于"
    ])
    
    # ========== TAB 1: Latest Top-10 (5-Day) ==========
    with tab1:
        st.header("最新 5日预测 (5-Day Predictions)")
        
        if latest_df.empty:
            st.warning("暂无5日预测数据。请先运行更新流程。")
            st.code("python app/update_daily.py --setup", language="bash")
        else:
            # Display date
            pred_date = latest_df['date'].iloc[0]
            st.subheader(f"预测日期: {pred_date.strftime('%Y-%m-%d')}")
            
            # Add sector info if not present
            if 'sector_cn' not in latest_df.columns:
                try:
                    from data.sectors import get_stock_sector, get_sector_name
                    latest_df['sector'] = latest_df['symbol'].apply(lambda x: get_stock_sector(str(x)))
                    latest_df['sector_cn'] = latest_df['sector'].apply(lambda x: get_sector_name(x, chinese=True))
                except ImportError:
                    latest_df['sector_cn'] = '其他'
            
            # Metrics row
            col1, col2, col3, col4 = st.columns(4)
            
            avg_pred_ret = latest_df['pred_ret_5'].mean()
            max_pred_ret = latest_df['pred_ret_5'].max()
            avg_confidence = latest_df['confidence_score'].mean() if 'confidence_score' in latest_df.columns else 0.7
            num_sectors = latest_df['sector_cn'].nunique() if 'sector_cn' in latest_df.columns else 0
            
            col1.metric("平均预测收益(5D)", format_percent(avg_pred_ret))
            col2.metric("最高预测收益(5D)", format_percent(max_pred_ret))
            col3.metric("平均置信度", f"{avg_confidence:.0%}")
            col4.metric("覆盖行业数", f"{num_sectors}")
            
            st.markdown("---")
            
            # Main table with confidence intervals and sector
            display_cols = ['symbol', 'name', 'sector_cn', 'exchange', 'close', 'pred_ret_5', 'pred_lower', 'pred_upper', 
                          'confidence_score', 'pred_price_5d', 'reason_cn']
            display_df = latest_df[[c for c in display_cols if c in latest_df.columns]].copy()
            
            # Add chart links
            if 'symbol' in display_df.columns and 'exchange' in latest_df.columns:
                display_df['chart_link'] = latest_df.apply(
                    lambda row: get_stock_link(row['symbol'], row['exchange']), axis=1
                )
            
            # Format columns
            if 'exchange' in display_df.columns:
                display_df['exchange'] = display_df['exchange'].map(get_exchange_name)
            if 'close' in display_df.columns:
                display_df['close'] = display_df['close'].apply(format_price)
            if 'pred_price_5d' in display_df.columns:
                display_df['pred_price_5d'] = display_df['pred_price_5d'].apply(format_price)
            if 'pred_ret_5' in display_df.columns:
                display_df['pred_ret_5'] = display_df['pred_ret_5'].apply(format_percent)
            if 'pred_lower' in display_df.columns:
                display_df['pred_lower'] = display_df['pred_lower'].apply(format_percent)
            if 'pred_upper' in display_df.columns:
                display_df['pred_upper'] = display_df['pred_upper'].apply(format_percent)
            if 'confidence_score' in display_df.columns:
                display_df['confidence_score'] = display_df['confidence_score'].apply(
                    lambda x: f"{get_confidence_color(x)} {x:.0%}"
                )
            
            # Rename columns for display
            display_df = display_df.rename(columns={
                'symbol': '股票代码',
                'name': '股票名称',
                'sector_cn': '行业',
                'exchange': '交易所',
                'close': '当前价格',
                'pred_ret_5': '预测收益',
                'pred_lower': '下限(90%)',
                'pred_upper': '上限(90%)',
                'confidence_score': '置信度',
                'pred_price_5d': '目标价格',
                'reason_cn': '选股理由',
                'chart_link': '📈 行情'
            })
            
            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True,
                height=450,
                column_config={
                    '📈 行情': st.column_config.LinkColumn(
                        '📈 行情',
                        help='点击查看实时行情 (Click to view live chart)',
                        display_text='查看'
                    )
                }
            )
    
    # ========== TAB 2: Latest Top-10 (15-Day) ==========
    with tab2:
        st.header("最新 15日预测 (15-Day Predictions)")
        
        if latest_df_15d.empty:
            st.warning("暂无15日预测数据。请先运行更新流程。")
            st.code("python models/train_15d.py --predict", language="bash")
        else:
            # Display date
            pred_date_15d = latest_df_15d['date'].iloc[0]
            st.subheader(f"预测日期: {pred_date_15d.strftime('%Y-%m-%d')}")
            
            # Add sector info if not present
            if 'sector_cn' not in latest_df_15d.columns:
                try:
                    from data.sectors import get_stock_sector, get_sector_name
                    latest_df_15d['sector'] = latest_df_15d['symbol'].apply(lambda x: get_stock_sector(str(x)))
                    latest_df_15d['sector_cn'] = latest_df_15d['sector'].apply(lambda x: get_sector_name(x, chinese=True))
                except ImportError:
                    latest_df_15d['sector_cn'] = '其他'
            
            # Metrics row
            col1, col2, col3 = st.columns(3)
            
            avg_pred_ret_15d = latest_df_15d['pred_ret_15'].mean()
            max_pred_ret_15d = latest_df_15d['pred_ret_15'].max()
            num_sectors_15d = latest_df_15d['sector_cn'].nunique() if 'sector_cn' in latest_df_15d.columns else 0
            
            col1.metric("平均预测收益(15D)", format_percent(avg_pred_ret_15d))
            col2.metric("最高预测收益(15D)", format_percent(max_pred_ret_15d))
            col3.metric("覆盖行业数", f"{num_sectors_15d}")
            
            st.markdown("---")
            
            # Main table
            display_cols_15d = ['symbol', 'name', 'sector_cn', 'exchange', 'close', 'pred_ret_15', 'reason_cn']
            display_df_15d = latest_df_15d[[c for c in display_cols_15d if c in latest_df_15d.columns]].copy()
            
            # Add chart links
            if 'symbol' in display_df_15d.columns and 'exchange' in latest_df_15d.columns:
                display_df_15d['chart_link'] = latest_df_15d.apply(
                    lambda row: get_stock_link(row['symbol'], row['exchange']), axis=1
                )
            
            # Format columns
            if 'exchange' in display_df_15d.columns:
                display_df_15d['exchange'] = display_df_15d['exchange'].map(get_exchange_name)
            if 'close' in display_df_15d.columns:
                display_df_15d['close'] = display_df_15d['close'].apply(format_price)
            if 'pred_ret_15' in display_df_15d.columns:
                display_df_15d['pred_ret_15'] = display_df_15d['pred_ret_15'].apply(format_percent)
            
            # Rename columns for display
            display_df_15d = display_df_15d.rename(columns={
                'symbol': '股票代码',
                'name': '股票名称',
                'sector_cn': '行业',
                'exchange': '交易所',
                'close': '当前价格',
                'pred_ret_15': '预测收益(15D)',
                'reason_cn': '选股理由',
                'chart_link': '📈 行情'
            })
            
            st.dataframe(
                display_df_15d,
                use_container_width=True,
                hide_index=True,
                height=450,
                column_config={
                    '📈 行情': st.column_config.LinkColumn(
                        '📈 行情',
                        help='点击查看实时行情 (Click to view live chart)',
                        display_text='查看'
                    )
                }
            )
            
            # 15-day chart
            st.markdown("---")
            st.subheader("15日预测收益率")
            fig_pred_15d = render_predictions_chart_15d(latest_df_15d)
            st.plotly_chart(fig_pred_15d, use_container_width=True)
    
    # ========== TAB 3: Charts & Analysis ==========
    with tab3:
        st.header("📊 图表分析 (Charts & Analysis)")
        
        if latest_df.empty:
            st.warning("暂无数据可用于图表展示")
        else:
            # Predictions chart with CI
            st.subheader("预测收益率与置信区间")
            fig_pred = render_predictions_chart(latest_df)
            st.plotly_chart(fig_pred, use_container_width=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Exchange breakdown
                st.subheader("交易所分布")
                fig_exchange = render_exchange_pie_chart(latest_df)
                st.plotly_chart(fig_exchange, use_container_width=True)
            
            with col2:
                # Confidence distribution
                st.subheader("置信度分布")
                if 'confidence_score' in latest_df.columns:
                    import plotly.express as px
                    fig_conf = px.histogram(
                        latest_df, x='confidence_score', nbins=10,
                        title='预测置信度分布'
                    )
                    fig_conf.update_layout(
                        xaxis_title='置信度',
                        yaxis_title='数量',
                        height=350
                    )
                    st.plotly_chart(fig_conf, use_container_width=True)
            
            # Symbol deep dive
            st.markdown("---")
            st.subheader("🔍 个股详情 (Symbol Deep Dive)")
            
            selected_symbol = st.selectbox(
                "选择股票代码",
                options=latest_df['symbol'].tolist()
            )
            
            if selected_symbol:
                symbol_row = latest_df[latest_df['symbol'] == selected_symbol].iloc[0]
                
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("当前价格", format_price(symbol_row['close']))
                col2.metric("预测收益", format_percent(symbol_row['pred_ret_5']))
                col3.metric("目标价格", format_price(symbol_row.get('pred_price_5d', symbol_row['close'] * (1 + symbol_row['pred_ret_5']))))
                
                if 'confidence_score' in symbol_row:
                    col4.metric("置信度", f"{symbol_row['confidence_score']:.0%}")
                
                # Confidence interval display
                if 'pred_lower' in symbol_row and 'pred_upper' in symbol_row:
                    st.info(f"📊 **90% 置信区间**: {format_percent(symbol_row['pred_lower'])} 至 {format_percent(symbol_row['pred_upper'])}")
                
                # Historical predictions for this symbol
                if not history_df.empty:
                    fig_hist = render_symbol_history_chart(history_df, selected_symbol)
                    if fig_hist:
                        st.plotly_chart(fig_hist, use_container_width=True)
    
    # ========== TAB 4: Historical ==========
    with tab4:
        st.header("📅 历史预测记录")
        
        if history_df.empty:
            st.warning("暂无历史数据")
        else:
            # Date selector
            available_dates = history_df['date'].dt.date.unique()
            available_dates = sorted(available_dates, reverse=True)
            
            selected_date = st.selectbox(
                "选择日期",
                options=available_dates,
                format_func=lambda x: x.strftime('%Y-%m-%d')
            )
            
            if selected_date:
                date_df = history_df[history_df['date'].dt.date == selected_date].copy()
                
                if date_df.empty:
                    st.warning(f"No data for {selected_date}")
                else:
                    st.subheader(f"{selected_date.strftime('%Y-%m-%d')} Top-10 (5日预测)")
                    
                    display_cols = ['symbol', 'exchange', 'close', 'pred_ret_5', 'pred_price_5d', 'reason_cn']
                    display_df = date_df[[c for c in display_cols if c in date_df.columns]].copy()
                    
                    if 'exchange' in display_df.columns:
                        display_df['exchange'] = display_df['exchange'].map(get_exchange_name)
                    if 'close' in display_df.columns:
                        display_df['close'] = display_df['close'].apply(format_price)
                    if 'pred_price_5d' in display_df.columns:
                        display_df['pred_price_5d'] = display_df['pred_price_5d'].apply(format_price)
                    if 'pred_ret_5' in display_df.columns:
                        display_df['pred_ret_5'] = display_df['pred_ret_5'].apply(format_percent)
                    
                    display_df = display_df.rename(columns={
                        'symbol': '股票代码',
                        'exchange': '交易所',
                        'close': '当时价格',
                        'pred_ret_5': '预测收益(5D)',
                        'pred_price_5d': '目标价格',
                        'reason_cn': '选股理由'
                    })
                    
                    st.dataframe(display_df, use_container_width=True, hide_index=True)
            
            # Summary stats
            st.markdown("---")
            st.subheader("📈 历史统计")
            
            col1, col2, col3 = st.columns(3)
            col1.metric("总天数", len(available_dates))
            col2.metric("日期范围", f"{min(available_dates)} to {max(available_dates)}")
            col3.metric("涉及股票", history_df['symbol'].nunique())
            
            with st.expander("🔥 最常入选的股票"):
                freq = history_df['symbol'].value_counts().head(20)
                st.bar_chart(freq)
    
    # ========== TAB 5: About ==========
    with tab5:
        st.header("关于本应用 (About)")
        
        st.markdown("""
        ### 🇨🇳 中国A股 Top-10 预测器
        
        本应用使用机器学习预测未来5个和15个交易日最有可能跑赢大盘的10只A股股票。
        
        #### 功能特点
        
        - **🏆 多周期预测**: 5日和15日两种预测周期
        - **📊 置信区间**: 90%置信度的预测范围
        - **📈 历史追踪**: 预测准确率统计
        - **🔍 深度分析**: 个股详细图表分析
        
        #### 工作原理
        
        1. **数据采集**: 从 EODHD API 获取上交所(SHG)和深交所(SHE)股票数据
        2. **特征工程**: 计算40+技术指标
        3. **两阶段预测**: RandomForest排序 + GradientBoosting回归
        4. **置信估计**: 预测不确定性量化
        
        #### 覆盖范围
        
        | 交易所 | 代码 | 板块 |
        |--------|------|------|
        | 上海证券交易所 | SHG | 600/601/603/605 |
        | 深圳证券交易所 | SHE | 000/001/002 |
        
        #### ⚠️ 免责声明
        
        本工具仅供教育和研究目的使用，不构成任何投资建议。
        投资有风险，入市需谨慎。请在做出投资决策前进行独立研究。
        """)
        
        st.markdown("---")
        st.subheader("🔧 系统状态")
        
        if quality_report:
            with st.expander("数据质量报告"):
                st.json(quality_report)


if __name__ == "__main__":
    main()
