import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats
from scipy.optimize import minimize, differential_evolution
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time

st.set_page_config(
    page_title="Stock Price Distribution Analyzer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

def generate_demo_data(period='2mo', interval='1h'):
    """Generate synthetic GLD-like data for demonstration"""
    import datetime as dt

    # Map period to days
    period_map = {'1wk': 7, '2wk': 14, '1mo': 30, '2mo': 60, '3mo': 90, '6mo': 180}
    days = period_map.get(period, 60)

    # Map interval to data points per day
    interval_map = {'15m': 26, '30m': 13, '1h': 7, '1d': 1}
    points_per_day = interval_map.get(interval, 7)

    n_points = days * points_per_day

    # Generate realistic GLD prices around $330 with volatility
    np.random.seed(42)
    base_price = 330
    returns = np.random.normal(0.0001, 0.005, n_points)  # Small positive drift, 0.5% daily vol
    prices = base_price * np.exp(np.cumsum(returns))

    # Add some structure - trending and mean reversion
    trend = np.linspace(0, 0.02, n_points)
    prices = prices * (1 + trend)

    # Create datetime index
    end_date = pd.Timestamp.now()
    if interval == '1d':
        dates = pd.date_range(end=end_date, periods=n_points, freq='D')
    elif interval == '1h':
        dates = pd.date_range(end=end_date, periods=n_points, freq='h')
    elif interval == '30m':
        dates = pd.date_range(end=end_date, periods=n_points, freq='30min')
    else:  # 15m
        dates = pd.date_range(end=end_date, periods=n_points, freq='15min')

    df = pd.DataFrame({
        'Close': prices,
        'Open': prices * np.random.uniform(0.998, 1.002, n_points),
        'High': prices * np.random.uniform(1.001, 1.005, n_points),
        'Low': prices * np.random.uniform(0.995, 0.999, n_points),
        'Volume': np.random.uniform(1e6, 5e6, n_points)
    }, index=dates)

    return df

@st.cache_data(ttl=300)  # Cache for 5 minutes
def fetch_stock_data(ticker='SPY', period='2mo', interval='1h'):
    """Fetch stock price data for any ticker"""
    max_retries = 3
    for attempt in range(max_retries):
        try:
            # Validate ticker and fetch data
            dat = yf.Ticker(ticker)
            df = dat.history(period=period, interval=interval)

            if df is not None and len(df) > 0:
                return df, True  # Return data and success flag

        except Exception as e:
            if attempt == max_retries - 1:
                return pd.DataFrame(), False  # Return empty df and failure flag
            else:
                time.sleep(1)  # Wait before retry
                continue

    return pd.DataFrame(), False  # Return empty df and failure flag

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

def find_quantile(target_prob, center, spread, left_tail, right_tail, lean, is_above=False):
    """Find the price threshold that corresponds to a given probability.
    If is_above=True, finds the threshold where P(X > threshold) = target_prob
    If is_above=False, finds the threshold where P(X < threshold) = target_prob
    """
    from scipy.optimize import minimize_scalar

    # Generate x range for integration
    x_min = center - 10 * spread
    x_max = center + 10 * spread

    def calc_prob_below(threshold):
        x_range = np.linspace(x_min, threshold, 1000)
        pdf_vals = skew_t_pdf_asymmetric(x_range, center, spread, left_tail, right_tail, lean)
        prob = np.trapz(pdf_vals, x_range)
        return min(max(prob, 0), 1)

    def objective(threshold):
        prob_below = calc_prob_below(threshold)
        if is_above:
            actual_prob = 1 - prob_below
        else:
            actual_prob = prob_below
        return abs(actual_prob - target_prob)

    # Use optimization to find the threshold
    result = minimize_scalar(objective, bounds=(x_min, x_max), method='bounded')

    if result.success:
        return result.x
    else:
        # Fallback: simple approximation
        return center + (spread if is_above else -spread)

# Sidebar for controls
st.sidebar.header("🎯 Stock Selection")

# Ticker input
default_ticker = "GLD"
ticker = st.sidebar.text_input(
    "Enter Stock Ticker Symbol",
    value=default_ticker,
    help="Enter any valid stock ticker symbol (e.g., AAPL, MSFT, SPY, GLD)",
    placeholder="Enter ticker..."
).upper().strip()

# Validate ticker is not empty
if not ticker:
    ticker = default_ticker

st.sidebar.markdown("---")
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

# Add dynamic title and description
st.title(f"📈 {ticker} Price Distribution Analyzer")
st.markdown(f"Interactive tool to model your beliefs about **{ticker}** price movements using asymmetric fat-tailed distributions")

# Fetch data
with st.spinner(f'Fetching {ticker} data...'):
    df, success = fetch_stock_data(
        ticker=ticker,
        period=period_options[selected_period],
        interval=interval_options[selected_interval]
    )

    if not success or df.empty:
        st.error(f"❌ Unable to fetch data for ticker '{ticker}'. Please check if the ticker symbol is valid.")
        st.info("💡 Try popular tickers like: AAPL, MSFT, GOOGL, AMZN, TSLA, SPY, QQQ, GLD")
        st.stop()

    prices = df['Close'].values
    dates = df.index

# Calculate initial parameters
initial_params = fit_initial_params(prices)

# Check if ticker or period has changed
data_key = f"{ticker}_{selected_period}_{selected_interval}"
if 'last_data_key' not in st.session_state:
    st.session_state.last_data_key = data_key

# If data source changed, reset to new fitted values
if st.session_state.last_data_key != data_key:
    st.session_state.last_data_key = data_key
    st.session_state.center = initial_params['center']
    st.session_state.spread = initial_params['spread']
    st.session_state.left_tail = initial_params['left_tail']
    st.session_state.right_tail = initial_params['right_tail']
    st.session_state.lean = initial_params['lean']
    st.rerun()  # Force refresh to update sliders

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
st.sidebar.header("🎯 Probability Calculator")

# Probability type selector
prob_type = st.sidebar.radio(
    "Calculate Probability",
    ["Below Threshold", "Above Threshold", "Show Both", "Between Range"],
    index=2,
    help="Choose whether to calculate probability below, above, or between thresholds"
)

if prob_type == "Between Range":
    col_low, col_high = st.sidebar.columns(2)
    with col_low:
        threshold_low = st.number_input(
            "Lower Bound ($)",
            min_value=price_min - 10,
            max_value=price_max + 10,
            value=float(center - 5),
            step=0.5,
            key='threshold_low'
        )
    with col_high:
        threshold_high = st.number_input(
            "Upper Bound ($)",
            min_value=price_min - 10,
            max_value=price_max + 10,
            value=float(center + 5),
            step=0.5,
            key='threshold_high'
        )
    threshold = (threshold_low + threshold_high) / 2  # For compatibility
else:
    threshold = st.sidebar.slider(
        "Threshold Price ($)",
        min_value=price_min - 10,
        max_value=price_max + 10,
        value=float(center + 10),
        step=0.5,
        key='threshold_slider'
    )
    threshold_low = threshold
    threshold_high = threshold

# Data points selector
n_points = st.sidebar.slider(
    "Number of Data Points to Show",
    min_value=20,
    max_value=len(prices),
    value=min(100, len(prices)),
    step=1,
    key='n_points_slider'
)

# Inverse Probability Calculator Section
st.sidebar.markdown("---")
st.sidebar.header("🔮 Inverse Probability Calculator")

with st.sidebar.expander("📖 How it works", expanded=False):
    st.markdown("""
    **Find Threshold**: Given a probability, find the corresponding price threshold.

    **Example**: "What price has only 15% chance of being exceeded?"
    """)

# Mode selector
inverse_mode = st.sidebar.radio(
    "Calculator Mode",
    ["Find Threshold from Probability", "Fit Distribution to Assumption"],
    index=0,
    key="inverse_mode"
)

if inverse_mode == "Find Threshold from Probability":
    # Input desired probability
    col_prob, col_dir = st.sidebar.columns([2, 1])
    with col_prob:
        desired_prob = st.number_input(
            "Desired Probability (%)",
            min_value=0.1,
            max_value=99.9,
            value=15.0,
            step=0.5,
            key='desired_prob'
        ) / 100.0

    with col_dir:
        prob_direction = st.selectbox(
            "Direction",
            ["Above", "Below"],
            index=0,
            key='prob_direction'
        )

    # Calculate button
    if st.sidebar.button("🎯 Calculate Threshold", type="secondary", use_container_width=True):
        is_above = (prob_direction == "Above")
        calculated_threshold = find_quantile(
            desired_prob, center, spread, left_tail, right_tail, lean, is_above
        )
        st.session_state['calculated_threshold'] = calculated_threshold
        st.session_state['calc_prob'] = desired_prob
        st.session_state['calc_dir'] = prob_direction

    # Display result if calculated
    if 'calculated_threshold' in st.session_state:
        st.sidebar.success(
            f"💡 **Result**: {st.session_state['calc_prob']*100:.1f}% chance {st.session_state['calc_dir'].lower()} **${st.session_state['calculated_threshold']:.2f}**"
        )

        # Option to use as threshold
        col1, col2 = st.sidebar.columns(2)
        with col1:
            if st.button("📌 Use as Threshold", use_container_width=True):
                st.session_state['threshold_slider'] = st.session_state['calculated_threshold']
                st.rerun()
        with col2:
            if st.button("🗑 Clear", use_container_width=True):
                del st.session_state['calculated_threshold']
                del st.session_state['calc_prob']
                del st.session_state['calc_dir']
                st.rerun()

else:  # Fit Distribution to Assumption mode
    st.sidebar.markdown("""
    **Set your probability assumptions and fit the distribution to match them.**
    """)

    # Initialize session state for assumptions
    if 'assumptions' not in st.session_state:
        st.session_state.assumptions = []

    # Add new assumption
    st.sidebar.subheader("Add Assumption")
    col_val, col_dir = st.sidebar.columns(2)
    with col_val:
        new_threshold = st.number_input(
            "Price ($)",
            min_value=price_min - 20,
            max_value=price_max + 20,
            value=float(center),
            step=0.5,
            key='new_threshold'
        )
    with col_dir:
        new_direction = st.selectbox(
            "Direction",
            ["Above", "Below"],
            index=0,
            key='new_direction'
        )

    new_prob = st.sidebar.number_input(
        f"Probability (%) {new_direction.lower()} ${new_threshold:.1f}",
        min_value=0.1,
        max_value=99.9,
        value=20.0,
        step=0.5,
        key='new_prob'
    )

    if st.sidebar.button("➕ Add Assumption", use_container_width=True):
        st.session_state.assumptions.append({
            'threshold': new_threshold,
            'probability': new_prob / 100.0,
            'direction': new_direction.lower()
        })

    # Display current assumptions
    if st.session_state.assumptions:
        st.sidebar.subheader("Current Assumptions")
        for i, assumption in enumerate(st.session_state.assumptions):
            col1, col2 = st.sidebar.columns([3, 1])
            with col1:
                st.markdown(f"{assumption['probability']*100:.1f}% {assumption['direction']} ${assumption['threshold']:.1f}")
            with col2:
                if st.button("❌", key=f"del_{i}"):
                    st.session_state.assumptions.pop(i)
                    st.rerun()

        # Parameters to fit
        st.sidebar.subheader("Parameters to Fit")
        fit_center = st.sidebar.checkbox("Center", value=True, key='fit_center')
        fit_spread = st.sidebar.checkbox("Spread", value=True, key='fit_spread')
        fit_tails = st.sidebar.checkbox("Tail Parameters", value=False, key='fit_tails')
        fit_lean = st.sidebar.checkbox("Lean", value=False, key='fit_lean')

        # Fit button
        if st.sidebar.button("🎯 Fit Distribution", type="primary", use_container_width=True):
            with st.spinner("Fitting distribution..."):
                # Tolerance for "close enough" - 3% error is acceptable
                TOLERANCE = 0.03

                # Quick feasibility check
                def quick_check():
                    """Quick check if assumptions are reasonable"""
                    for assumption in st.session_state.assumptions:
                        # Check if threshold is within reasonable range
                        if assumption['threshold'] < center - 5*spread or assumption['threshold'] > center + 5*spread:
                            if assumption['direction'] == 'above' and assumption['probability'] > 0.3:
                                return False
                            if assumption['direction'] == 'below' and assumption['probability'] < 0.7:
                                return False
                        # Check for contradictory assumptions
                        for other in st.session_state.assumptions:
                            if other != assumption and other['direction'] == assumption['direction']:
                                if assumption['direction'] == 'above':
                                    if other['threshold'] > assumption['threshold'] and other['probability'] > assumption['probability']:
                                        return False  # Higher threshold should have lower prob_above
                                else:  # below
                                    if other['threshold'] > assumption['threshold'] and other['probability'] < assumption['probability']:
                                        return False  # Higher threshold should have higher prob_below
                    return True

                if not quick_check():
                    st.sidebar.warning("⚠️ Assumptions might be contradictory or unrealistic. Trying anyway...")

                # Objective function optimized for speed
                def objective(params):
                    idx = 0
                    test_center = params[idx] if fit_center else center
                    idx += 1 if fit_center else 0
                    test_spread = params[idx] if fit_spread else spread
                    idx += 1 if fit_spread else 0
                    test_left_tail = params[idx] if fit_tails else left_tail
                    idx += 1 if fit_tails else 0
                    test_right_tail = params[idx] if fit_tails else right_tail
                    idx += 1 if fit_tails else 0
                    test_lean = params[idx] if fit_lean else lean

                    max_error = 0
                    total_error = 0
                    for assumption in st.session_state.assumptions:
                        # Calculate probability based on direction
                        if assumption['direction'] == 'above':
                            # Use fewer points for faster calculation
                            x_range_test = np.linspace(assumption['threshold'], test_center + 10*test_spread, 200)
                            pdf_vals_test = skew_t_pdf_asymmetric(x_range_test, test_center, test_spread,
                                                                 test_left_tail, test_right_tail, test_lean)
                            actual_prob = np.trapz(pdf_vals_test, x_range_test)
                        else:  # below
                            x_range_test = np.linspace(test_center - 10*test_spread, assumption['threshold'], 200)
                            pdf_vals_test = skew_t_pdf_asymmetric(x_range_test, test_center, test_spread,
                                                                 test_left_tail, test_right_tail, test_lean)
                            actual_prob = np.trapz(pdf_vals_test, x_range_test)

                        actual_prob = min(max(actual_prob, 0), 1)
                        error = abs(actual_prob - assumption['probability'])
                        max_error = max(max_error, error)
                        total_error += error ** 2

                    # Return early if close enough
                    if max_error <= TOLERANCE:
                        return 0  # Signal that we found a good solution

                    return total_error

                # Initial parameters and bounds
                initial_params = []
                bounds = []
                if fit_center:
                    initial_params.append(center)
                    bounds.append((price_min - 30, price_max + 30))
                if fit_spread:
                    initial_params.append(spread)
                    bounds.append((1.0, 50.0))
                if fit_tails:
                    initial_params.append(left_tail)
                    bounds.append((2.5, 20.0))
                    initial_params.append(right_tail)
                    bounds.append((2.5, 20.0))
                if fit_lean:
                    initial_params.append(lean)
                    bounds.append((0.5, 2.0))

                if initial_params:  # Only fit if at least one parameter selected
                    # Try simple optimization first (faster)
                    from scipy.optimize import minimize

                    # Quick attempt with local optimizer
                    quick_result = minimize(objective, initial_params, bounds=bounds,
                                           method='L-BFGS-B', options={'maxiter': 50})

                    # Check if quick solution is good enough
                    final_error = objective(quick_result.x)

                    if final_error <= TOLERANCE**2 * len(st.session_state.assumptions):
                        # Good enough! Use this solution
                        result = quick_result
                        success = True
                    else:
                        # Try harder with global optimization but with fewer iterations
                        result = differential_evolution(objective, bounds, seed=42,
                                                      maxiter=30,  # Reduced from 100
                                                      popsize=10,  # Smaller population
                                                      tol=TOLERANCE,  # Stop when close enough
                                                      atol=TOLERANCE)  # Absolute tolerance
                        success = result.fun <= TOLERANCE**2 * len(st.session_state.assumptions) * 2  # Be more lenient

                    if success:
                        # Extract fitted parameters
                        idx = 0
                        if fit_center:
                            st.session_state.center = result.x[idx]
                            idx += 1
                        if fit_spread:
                            st.session_state.spread = result.x[idx]
                            idx += 1
                        if fit_tails:
                            st.session_state.left_tail = result.x[idx]
                            idx += 1
                            st.session_state.right_tail = result.x[idx]
                            idx += 1
                        if fit_lean:
                            st.session_state.lean = result.x[idx]

                        st.sidebar.success("✓ Distribution fitted successfully!")
                        st.rerun()
                    else:
                        # Check how close we got
                        if result.fun < 0.05**2 * len(st.session_state.assumptions):
                            st.sidebar.info("📊 Got reasonably close! Try adjusting fewer parameters.")
                        else:
                            st.sidebar.warning("⚠️ Couldn't match assumptions exactly. Try different targets or parameters.")
                else:
                    st.sidebar.warning("Please select at least one parameter to fit.")

        # Clear all button
        if st.sidebar.button("🗑 Clear All Assumptions", use_container_width=True):
            st.session_state.assumptions = []
            st.rerun()
    else:
        st.sidebar.info("Add assumptions above to fit the distribution")

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

# Calculate probabilities
if prob_type == "Between Range":
    threshold_low_idx = np.argmin(np.abs(x_range - threshold_low))
    threshold_high_idx = np.argmin(np.abs(x_range - threshold_high))
    prob_low = cdf_vals[threshold_low_idx]
    prob_high = cdf_vals[threshold_high_idx]
    prob_between = prob_high - prob_low
    prob_below = prob_low
    prob_above = 1 - prob_high
else:
    threshold_idx = np.argmin(np.abs(x_range - threshold))
    prob_below = cdf_vals[threshold_idx]
    prob_above = 1 - prob_below
    prob_between = 0

# Create plot
fig = go.Figure()

# Histogram of actual data
fig.add_trace(go.Histogram(
    x=subset,
    nbinsx=50,
    histnorm='probability density',
    name='Historical Data',
    opacity=0.7,
    marker_color='#5DADE2',
    marker_line_color='#3498DB',
    marker_line_width=1,
    hovertemplate='Price: $%{x:.2f}<extra></extra>'
))

# Manual distribution with hover showing probabilities
fig.add_trace(go.Scatter(
    x=x_range,
    y=pdf_vals,
    mode='lines',
    name='Your Belief Distribution',
    line=dict(color='#FF6B6B', width=3),
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
    line=dict(color='#3498DB', width=1, dash='dash'),
    hovertemplate=f'Historical Mean: ${actual_mean:.2f}<extra></extra>'
))

# Your center line
fig.add_trace(go.Scatter(
    x=[center, center],
    y=[0, max(pdf_vals)*0.3],
    mode='lines',
    name=f'Your Center',
    line=dict(color='#FF6B6B', width=1, dash='dash'),
    hovertemplate=f'Your Center: ${center:.2f}<extra></extra>'
))

# Show calculated threshold from inverse probability if it exists
if 'calculated_threshold' in st.session_state:
    calc_threshold = st.session_state['calculated_threshold']
    fig.add_trace(go.Scatter(
        x=[calc_threshold, calc_threshold],
        y=[0, max(pdf_vals)*1.2],
        mode='lines',
        name=f'Calculated: {st.session_state["calc_prob"]*100:.1f}% {st.session_state["calc_dir"].lower()}',
        line=dict(color='#FFD700', width=3, dash='dash'),
        hovertemplate=(
            f'<b>Calculated Threshold: ${calc_threshold:.2f}</b><br>' +
            f'{st.session_state["calc_prob"]*100:.1f}% chance {st.session_state["calc_dir"].lower()}<extra></extra>'
        )
    ))

# Show assumptions if they exist
if 'assumptions' in st.session_state and st.session_state.assumptions:
    for i, assumption in enumerate(st.session_state.assumptions):
        # Calculate actual probability for this assumption
        if assumption.get('direction', 'above') == 'above':
            x_test = np.linspace(assumption['threshold'], center + 10*spread, 500)
        else:  # below
            x_test = np.linspace(center - 10*spread, assumption['threshold'], 500)

        pdf_test = skew_t_pdf_asymmetric(x_test, center, spread, left_tail, right_tail, lean)
        actual_prob = np.trapz(pdf_test, x_test)
        actual_prob = min(max(actual_prob, 0), 1)

        # Color based on how close we are to target
        target_prob = assumption.get('probability', assumption.get('prob_above', 0))
        error = abs(actual_prob - target_prob)
        if error < 0.01:
            color = '#2ECC71'  # Green - good fit
        elif error < 0.05:
            color = '#F39C12'  # Orange - okay fit
        else:
            color = '#E74C3C'  # Red - poor fit

        direction = assumption.get('direction', 'above')
        target_prob = assumption.get('probability', assumption.get('prob_above', 0))
        fig.add_trace(go.Scatter(
            x=[assumption['threshold'], assumption['threshold']],
            y=[0, max(pdf_vals)*0.8],
            mode='lines+text',
            name=f'Target {i+1}',
            line=dict(color=color, width=2, dash='dashdot'),
            text=[f"{target_prob*100:.1f}% (actual: {actual_prob*100:.1f}%)"],
            textposition='top right',
            textfont=dict(size=10, color=color),
            hovertemplate=(
                f'<b>Assumption {i+1}</b><br>' +
                f'Target: {target_prob*100:.1f}% {direction} ${assumption["threshold"]:.2f}<br>' +
                f'Actual: {actual_prob*100:.1f}% {direction}<br>' +
                f'Error: {error*100:.2f}%<extra></extra>'
            )
        ))

# Threshold lines based on type
if prob_type == "Between Range":
    # Lower threshold line
    fig.add_trace(go.Scatter(
        x=[threshold_low, threshold_low],
        y=[0, max(pdf_vals)*1.1],
        mode='lines',
        name=f'Lower Bound',
        line=dict(color='#2ECC71', width=2, dash='dot'),
        hovertemplate=f'<b>Lower: ${threshold_low:.2f}</b><extra></extra>'
    ))
    # Upper threshold line
    fig.add_trace(go.Scatter(
        x=[threshold_high, threshold_high],
        y=[0, max(pdf_vals)*1.1],
        mode='lines',
        name=f'Upper Bound',
        line=dict(color='#2ECC71', width=2, dash='dot'),
        hovertemplate=f'<b>Upper: ${threshold_high:.2f}</b><extra></extra>'
    ))
    # Shade area between thresholds
    x_fill = x_range[(x_range >= threshold_low) & (x_range <= threshold_high)]
    if len(x_fill) > 0:
        y_fill = skew_t_pdf_asymmetric(x_fill, center, spread, left_tail, right_tail, lean)
        fig.add_trace(go.Scatter(
            x=x_fill,
            y=y_fill,
            fill='tozeroy',
            fillcolor='rgba(155, 89, 182, 0.3)',  # Purple for between
            line=dict(color='rgba(255,255,255,0)'),
            showlegend=False,
            hoverinfo='skip',
            name='P(Between)'
        ))
else:
    # Single threshold line
    fig.add_trace(go.Scatter(
        x=[threshold, threshold],
        y=[0, max(pdf_vals)*1.1],
        mode='lines',
        name=f'Threshold',
        line=dict(color='#2ECC71', width=2, dash='dot'),
        hovertemplate=(
            f'<b>Threshold: ${threshold:.2f}</b><br>' +
            f'Probability below: {prob_below:.1%}<br>' +
            f'Probability above: {prob_above:.1%}<extra></extra>'
        )
    ))

    # Shade area based on probability type
    if prob_type == "Below Threshold" or prob_type == "Show Both":
        # Shade area below threshold
        x_fill = x_range[x_range <= threshold]
        if len(x_fill) > 0:
            y_fill = skew_t_pdf_asymmetric(x_fill, center, spread, left_tail, right_tail, lean)
            fig.add_trace(go.Scatter(
                x=x_fill,
                y=y_fill,
                fill='tozeroy',
                fillcolor='rgba(52, 152, 219, 0.3)',  # Blue for below
                line=dict(color='rgba(255,255,255,0)'),
                showlegend=False,
                hoverinfo='skip',
                name='P(Below)'
            ))

    if prob_type == "Above Threshold" or prob_type == "Show Both":
        # Shade area above threshold
        x_fill = x_range[x_range >= threshold]
        if len(x_fill) > 0:
            y_fill = skew_t_pdf_asymmetric(x_fill, center, spread, left_tail, right_tail, lean)
            fig.add_trace(go.Scatter(
                x=x_fill,
                y=y_fill,
                fill='tozeroy',
                fillcolor='rgba(231, 76, 60, 0.3)',  # Red for above
                line=dict(color='rgba(255,255,255,0)'),
                showlegend=False,
                hoverinfo='skip',
                name='P(Above)'
            ))

# Dynamic title based on probability type
if prob_type == "Below Threshold":
    title_text = f'P(Price < ${threshold:.1f}) = {prob_below:.1%} | Data since {start_date.strftime("%Y-%m-%d %H:%M")}'
elif prob_type == "Above Threshold":
    title_text = f'P(Price > ${threshold:.1f}) = {prob_above:.1%} | Data since {start_date.strftime("%Y-%m-%d %H:%M")}'
elif prob_type == "Between Range":
    title_text = f'P(${threshold_low:.1f} < Price < ${threshold_high:.1f}) = {prob_between:.1%} | Since {start_date.strftime("%Y-%m-%d %H:%M")}'
else:  # Show Both
    title_text = f'P < ${threshold:.1f} = {prob_below:.1%} | P > ${threshold:.1f} = {prob_above:.1%} | Since {start_date.strftime("%Y-%m-%d %H:%M")}'

fig.update_layout(
    title=title_text,
    xaxis_title="Price ($)",
    yaxis_title="Probability Density",
    height=600,
    hovermode='x unified',
    hoverlabel=dict(
        bgcolor="rgba(255, 255, 255, 0.9)",
        font_size=12,
        font_family="Arial",
        font_color="black"
    ),
    showlegend=True,
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    ),
    template="plotly_dark",
    xaxis=dict(
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(255, 255, 255, 0.1)',
        zeroline=True,
        zerolinewidth=1,
        zerolinecolor='rgba(255, 255, 255, 0.2)'
    ),
    yaxis=dict(
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(255, 255, 255, 0.1)',
        zeroline=True,
        zerolinewidth=1,
        zerolinecolor='rgba(255, 255, 255, 0.2)'
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

# Display assumption validation if any exist
if 'assumptions' in st.session_state and st.session_state.assumptions:
    st.markdown("---")
    st.subheader("🎯 Assumption Validation")

    validation_data = []
    for i, assumption in enumerate(st.session_state.assumptions):
        # Calculate actual probability
        direction = assumption.get('direction', 'above')
        if direction == 'above':
            x_test = np.linspace(assumption['threshold'], center + 10*spread, 500)
        else:  # below
            x_test = np.linspace(center - 10*spread, assumption['threshold'], 500)

        pdf_test = skew_t_pdf_asymmetric(x_test, center, spread, left_tail, right_tail, lean)
        actual_prob = np.trapz(pdf_test, x_test)
        actual_prob = min(max(actual_prob, 0), 1)

        target_prob = assumption.get('probability', assumption.get('prob_above', 0))
        error = abs(actual_prob - target_prob)
        status = "✅ Good" if error < 0.01 else "⚠️ OK" if error < 0.05 else "❌ Poor"

        # Format display based on direction
        symbol = '>' if direction == 'above' else '<'
        validation_data.append({
            "Assumption": f"#{i+1}",
            "Target": f"{target_prob*100:.1f}% {symbol} ${assumption['threshold']:.1f}",
            "Actual": f"{actual_prob*100:.1f}%",
            "Error": f"{error*100:.2f}%",
            "Status": status
        })

    validation_df = pd.DataFrame(validation_data)
    st.dataframe(validation_df, hide_index=True, use_container_width=True)

    # Suggestion if fit is poor
    max_error = max([abs(float(d['Error'].rstrip('%'))) for d in validation_data], default=0)
    if max_error > 5:
        st.info("💡 **Tip**: Use 'Fit Distribution to Assumption' mode to automatically adjust parameters to match your assumptions.")

# Display key probabilities
if prob_type == "Between Range":
    col3, col4, col5, col6 = st.columns(4)

    with col3:
        st.metric(
            label=f"Current {ticker} Price",
            value=f"${prices[-1]:.2f}",
            delta=f"Last: {dates[-1].strftime('%H:%M')}"
        )

    with col4:
        st.metric(
            label=f"P(< ${threshold_low:.1f})",
            value=f"{prob_below:.1%}",
            delta=None
        )

    with col5:
        st.metric(
            label=f"P(${threshold_low:.1f} - ${threshold_high:.1f})",
            value=f"{prob_between:.1%}",
            delta=None
        )

    with col6:
        st.metric(
            label=f"P(> ${threshold_high:.1f})",
            value=f"{prob_above:.1%}",
            delta=None
        )
else:
    col3, col4, col5 = st.columns(3)

    with col3:
        st.metric(
            label=f"Current {ticker} Price",
            value=f"${prices[-1]:.2f}",
            delta=f"Last: {dates[-1].strftime('%H:%M')}"
        )

    with col4:
        st.metric(
            label=f"P(Price < ${threshold:.1f})",
            value=f"{prob_below:.1%}",
            delta=None
        )

    with col5:
        st.metric(
            label=f"P(Price > ${threshold:.1f})",
            value=f"{prob_above:.1%}",
            delta=None
        )