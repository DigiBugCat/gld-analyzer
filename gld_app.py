import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats
import plotly.graph_objects as go
from datetime import datetime, timedelta

st.set_page_config(
    page_title="GLD Price Distribution Analyzer",
    page_icon="📈",
    layout="wide"
)

st.title("🏆 GLD (Gold ETF) Price Distribution Analyzer")
st.markdown("Interactive tool to model your beliefs about GLD price movements using asymmetric fat-tailed distributions")

@st.cache_data(ttl=3600)
def fetch_gld_data(period='2mo', interval='1h'):
    """Fetch GLD price data with caching"""
    ticker = yf.Ticker("GLD")
    df = ticker.history(period=period, interval=interval)
    return df

def fit_initial_params(prices, n_recent=100):
    """Fit asymmetric t-distribution to get reasonable starting values"""
    if len(prices) == 0:
        # Return default values if no data
        return {
            'center': 330.0,
            'spread': 10.0,
            'left_tail': 5.0,
            'right_tail': 5.0,
            'lean': 1.0
        }

    recent = prices[-min(n_recent, len(prices)):]

    center = np.median(recent)
    spread = stats.iqr(recent) / 1.35 if len(recent) > 1 else 10.0

    if len(recent) > 3:
        kurt = stats.kurtosis(recent)
        if kurt > 0:
            base_df = min(10, max(3, 6/kurt + 4))
        else:
            base_df = 8
        skewness = stats.skew(recent)
    else:
        base_df = 8
        skewness = 0

    left_deviations = recent[recent < center] - center
    right_deviations = recent[recent > center] - center

    if len(left_deviations) > 5 and len(right_deviations) > 5:
        left_kurt = stats.kurtosis(left_deviations) if len(left_deviations) > 3 else 0
        right_kurt = stats.kurtosis(right_deviations) if len(right_deviations) > 3 else 0

        left_tail = min(10, max(2.5, base_df - left_kurt))
        right_tail = min(10, max(2.5, base_df - right_kurt))
    else:
        left_tail = base_df
        right_tail = base_df

    lean = 1.0 + np.clip(skewness * 0.2, -0.5, 0.5)

    return {
        'center': round(center, 1),
        'spread': round(spread, 1),
        'left_tail': round(left_tail, 1),
        'right_tail': round(right_tail, 1),
        'lean': round(lean, 2)
    }

def skew_t_pdf_asymmetric(x, center, spread, left_tail, right_tail, lean):
    """Asymmetric fat-tailed distribution"""
    z = (x - center) / spread

    pdf_vals = []
    for zi in z:
        if zi < 0:
            pdf_val = (2 / (lean + 1/lean)) * stats.t.pdf(zi * lean, left_tail) / spread
        else:
            pdf_val = (2 / (lean + 1/lean)) * stats.t.pdf(zi / lean, right_tail) / spread
        pdf_vals.append(pdf_val)

    return np.array(pdf_vals)

# Sidebar for controls
st.sidebar.header("⚙️ Configuration")

# Data period selector
period_options = {
    '1 Week': '1wk',
    '2 Weeks': '2wk',
    '1 Month': '1mo',
    '2 Months': '2mo',
    '3 Months': '3mo',
    '6 Months': '6mo'
}
selected_period = st.sidebar.selectbox(
    "Select Data Period",
    options=list(period_options.keys()),
    index=3  # Default to 2 months
)

# Interval selector
interval_options = {
    '15 minutes': '15m',
    '30 minutes': '30m',
    '1 hour': '1h',
    '1 day': '1d'
}
selected_interval = st.sidebar.selectbox(
    "Select Data Interval",
    options=list(interval_options.keys()),
    index=2  # Default to 1 hour
)

# Fetch data
with st.spinner('Fetching GLD data...'):
    try:
        df = fetch_gld_data(period=period_options[selected_period],
                           interval=interval_options[selected_interval])
        if df.empty:
            st.error("No data received from Yahoo Finance. Please try again later.")
            st.stop()
        prices = df['Close'].values
        dates = df.index
    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
        st.stop()

# Calculate initial parameters
initial_params = fit_initial_params(prices)

# Parameter definitions expander
with st.sidebar.expander("📖 Parameter Definitions", expanded=False):
    st.markdown("""
    **CENTER ($)**: The most likely price level where you think the price normally trades.

    **SPREAD**: How wide the distribution is. Higher = more uncertainty, lower = more concentrated.

    **LEFT TAIL**: Controls probability of extreme DOWN moves. Lower values = fatter tail = higher crash risk.

    **RIGHT TAIL**: Controls probability of extreme UP moves. Lower values = fatter tail = higher moon potential.

    **LEAN**: Makes the distribution lean left or right. <1.0 = bearish tilt, >1.0 = bullish tilt.
    """)

st.sidebar.markdown("---")
st.sidebar.header("📊 Distribution Parameters")

# Reset button
if st.sidebar.button("🔄 Reset to Fitted Values", type="primary", use_container_width=True):
    st.session_state.center = initial_params['center']
    st.session_state.spread = initial_params['spread']
    st.session_state.left_tail = initial_params['left_tail']
    st.session_state.right_tail = initial_params['right_tail']
    st.session_state.lean = initial_params['lean']

# Initialize session state if not exists
if 'center' not in st.session_state:
    st.session_state.center = initial_params['center']
if 'spread' not in st.session_state:
    st.session_state.spread = initial_params['spread']
if 'left_tail' not in st.session_state:
    st.session_state.left_tail = initial_params['left_tail']
if 'right_tail' not in st.session_state:
    st.session_state.right_tail = initial_params['right_tail']
if 'lean' not in st.session_state:
    st.session_state.lean = initial_params['lean']

# Parameter sliders
price_min = float(np.min(prices)) if len(prices) > 0 else 310.0
price_max = float(np.max(prices)) if len(prices) > 0 else 350.0

center = st.sidebar.slider(
    "Center ($)",
    min_value=price_min - 20,
    max_value=price_max + 20,
    value=float(st.session_state.center),
    step=0.5,
    key='center_slider'
)

spread = st.sidebar.slider(
    "Spread",
    min_value=1.0,
    max_value=30.0,
    value=float(st.session_state.spread),
    step=0.5,
    key='spread_slider'
)

left_tail = st.sidebar.slider(
    "Left Tail (Downside Risk)",
    min_value=2.5,
    max_value=20.0,
    value=float(st.session_state.left_tail),
    step=0.5,
    key='left_tail_slider'
)

right_tail = st.sidebar.slider(
    "Right Tail (Upside Potential)",
    min_value=2.5,
    max_value=20.0,
    value=float(st.session_state.right_tail),
    step=0.5,
    key='right_tail_slider'
)

lean = st.sidebar.slider(
    "Lean (Skewness)",
    min_value=0.5,
    max_value=2.0,
    value=float(st.session_state.lean),
    step=0.05,
    key='lean_slider'
)

st.sidebar.markdown("---")
st.sidebar.header("🎯 Probability Threshold")

threshold = st.sidebar.slider(
    "Threshold Price ($)",
    min_value=price_min - 10,
    max_value=price_max + 10,
    value=float(center + 10),
    step=0.5,
    key='threshold_slider'
)

# Data points selector
n_points = st.sidebar.slider(
    "Number of Data Points to Show",
    min_value=20,
    max_value=len(prices),
    value=min(100, len(prices)),
    step=1,
    key='n_points_slider'
)

# Main visualization
subset = prices[-n_points:]
subset_dates = dates[-n_points:]
start_date = subset_dates[0]

# Calculate actual stats from data
actual_mean = np.mean(subset)

# Generate x range with more points for smoother curve
x_range = np.linspace(price_min - 10, price_max + 10, 1000)

# Calculate PDF
pdf_vals = skew_t_pdf_asymmetric(x_range, center, spread, left_tail, right_tail, lean)

# Calculate CDF for each point
cdf_vals = []
for i, x in enumerate(x_range):
    if i == 0:
        cdf_vals.append(0)
    else:
        prob = np.trapz(pdf_vals[:i+1], x_range[:i+1])
        cdf_vals.append(min(prob, 1.0))

cdf_vals = np.array(cdf_vals)

# Calculate probability at threshold
threshold_idx = np.argmin(np.abs(x_range - threshold))
prob_below = cdf_vals[threshold_idx]

# Create plot
fig = go.Figure()

# Histogram of actual data
fig.add_trace(go.Histogram(
    x=subset,
    nbinsx=50,
    histnorm='probability density',
    name='Historical Data',
    opacity=0.6,
    marker_color='lightblue',
    hovertemplate='Price: $%{x:.2f}<extra></extra>'
))

# Manual distribution with hover showing probabilities
fig.add_trace(go.Scatter(
    x=x_range,
    y=pdf_vals,
    mode='lines',
    name='Your Belief Distribution',
    line=dict(color='red', width=3),
    hovertemplate=(
        '<b>Price: $%{x:.2f}</b><br>' +
        'Probability below: %{customdata[0]:.1%}<br>' +
        'Probability above: %{customdata[1]:.1%}<br>' +
        '<extra></extra>'
    ),
    customdata=np.column_stack((cdf_vals, 1-cdf_vals))
))

# Historical mean line
fig.add_trace(go.Scatter(
    x=[actual_mean, actual_mean],
    y=[0, max(pdf_vals)*0.3],
    mode='lines',
    name=f'Historical Mean',
    line=dict(color='blue', width=1, dash='dash'),
    hovertemplate=f'Historical Mean: ${actual_mean:.2f}<extra></extra>'
))

# Your center line
fig.add_trace(go.Scatter(
    x=[center, center],
    y=[0, max(pdf_vals)*0.3],
    mode='lines',
    name=f'Your Center',
    line=dict(color='red', width=1, dash='dash'),
    hovertemplate=f'Your Center: ${center:.2f}<extra></extra>'
))

# Threshold line
fig.add_trace(go.Scatter(
    x=[threshold, threshold],
    y=[0, max(pdf_vals)*1.1],
    mode='lines',
    name=f'Threshold',
    line=dict(color='green', width=2, dash='dot'),
    hovertemplate=(
        f'<b>Threshold: ${threshold:.2f}</b><br>' +
        f'Probability below: {prob_below:.1%}<br>' +
        f'Probability above: {1-prob_below:.1%}<extra></extra>'
    )
))

# Shade area below threshold
x_fill = x_range[x_range <= threshold]
if len(x_fill) > 0:
    y_fill = skew_t_pdf_asymmetric(x_fill, center, spread, left_tail, right_tail, lean)
    fig.add_trace(go.Scatter(
        x=x_fill,
        y=y_fill,
        fill='tozeroy',
        fillcolor='rgba(0,255,0,0.15)',
        line=dict(color='rgba(255,255,255,0)'),
        showlegend=False,
        hoverinfo='skip'
    ))

fig.update_layout(
    title=f'P(Price < ${threshold:.1f}) = {prob_below:.1%} | Data since {start_date.strftime("%Y-%m-%d %H:%M")}',
    xaxis_title="Price ($)",
    yaxis_title="Probability Density",
    height=600,
    hovermode='x unified',
    hoverlabel=dict(
        bgcolor="white",
        font_size=12,
        font_family="Arial"
    ),
    showlegend=True,
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    )
)

st.plotly_chart(fig, use_container_width=True)

# Display interpretation and statistics
col1, col2 = st.columns(2)

with col1:
    st.subheader("🎯 Your Current Belief Settings")

    interpretation = []
    interpretation.append(f"• Price centers around **${center:.1f}**")
    interpretation.append(f"• Typical range: **${center-spread:.1f}** to **${center+spread:.1f}**")

    if left_tail < right_tail:
        interpretation.append("• **Higher crash risk** than moon potential (left tail fatter)")
    elif right_tail < left_tail:
        interpretation.append("• **Higher moon potential** than crash risk (right tail fatter)")
    else:
        interpretation.append("• **Symmetric tail risks**")

    if lean > 1.1:
        interpretation.append("• **Bullish lean** (distribution tilts right)")
    elif lean < 0.9:
        interpretation.append("• **Bearish lean** (distribution tilts left)")
    else:
        interpretation.append("• **Neutral lean**")

    for item in interpretation:
        st.markdown(item)

with col2:
    st.subheader("📊 Key Probability Levels")

    # Calculate probabilities for important levels
    important_levels = sorted([
        np.percentile(prices, 10),
        np.percentile(prices, 25),
        np.percentile(prices, 50),
        np.percentile(prices, 75),
        np.percentile(prices, 90),
        threshold
    ])

    prob_data = []
    for level in important_levels:
        level_idx = np.argmin(np.abs(x_range - level))
        prob = cdf_vals[level_idx]
        prob_above = 1 - prob

        if abs(level - threshold) < 0.1:
            prob_data.append({
                "Price": f"${level:.1f} 🎯",
                "P(Below)": f"{prob:.1%}",
                "P(Above)": f"{prob_above:.1%}"
            })
        else:
            prob_data.append({
                "Price": f"${level:.1f}",
                "P(Below)": f"{prob:.1%}",
                "P(Above)": f"{prob_above:.1%}"
            })

    prob_df = pd.DataFrame(prob_data)
    st.dataframe(prob_df, hide_index=True, use_container_width=True)

# Current price info
st.info(f"**Current GLD Price**: ${prices[-1]:.2f} | **Last Update**: {dates[-1].strftime('%Y-%m-%d %H:%M')}")