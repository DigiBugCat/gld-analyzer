# GLD Price Distribution Analyzer

An interactive Streamlit application for analyzing GLD (Gold ETF) price distributions using asymmetric fat-tailed models.

## Features

- **Real-time GLD data**: Fetches historical price data from Yahoo Finance
- **Interactive probability modeling**: Adjust distribution parameters to match your beliefs
- **Asymmetric fat-tailed distributions**: Model crash risks and moon potential separately
- **Visual probability calculator**: See probabilities for any price threshold
- **Multiple timeframes**: Analyze data from 1 week to 6 months
- **Smart parameter fitting**: Automatically fits initial parameters to recent data

## Quick Start

### Option 1: Using the run script
```bash
./run.sh
```

### Option 2: Manual setup
```bash
# Install dependencies
uv pip install -r requirements.txt

# Run the app
streamlit run gld_app.py
```

The app will open in your browser at `http://localhost:8501`

## How to Use

1. **Select Data Period**: Choose how much historical data to analyze (1 week to 6 months)
2. **Adjust Distribution Parameters**:
   - **Center**: Where you think the price normally trades
   - **Spread**: How uncertain you are (wider = more volatile)
   - **Left Tail**: Downside risk (lower = fatter tail = higher crash probability)
   - **Right Tail**: Upside potential (lower = fatter tail = higher moon probability)
   - **Lean**: Skewness (>1 = bullish tilt, <1 = bearish tilt)
3. **Set Threshold**: Pick a price level to calculate probabilities
4. **Analyze**: View the probability distribution and key statistics

## Parameter Definitions

- **CENTER ($)**: The most likely price level where you believe GLD trades
- **SPREAD**: Width of the distribution (higher = more uncertainty)
- **LEFT TAIL**: Controls extreme downside probability (lower = higher crash risk)
- **RIGHT TAIL**: Controls extreme upside probability (lower = higher moon potential)
- **LEAN**: Makes the distribution lean left (<1.0 = bearish) or right (>1.0 = bullish)

## Requirements

- Python 3.8+
- See `requirements.txt` for package dependencies

## Deployment

### Option 1: Streamlit Community Cloud (Recommended - Free)

1. Push your code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Sign in with GitHub
4. Click "New app" and select your repository
5. Set branch as `main` and main file as `gld_app.py`
6. Click Deploy!

Your app will be live at: `https://[your-username]-gld-[random].streamlit.app`

### Option 2: Heroku

1. Install Heroku CLI
2. Run:
```bash
heroku create your-app-name
git push heroku main
```

### Option 3: Railway

1. Connect your GitHub repo to [Railway](https://railway.app)
2. Railway will auto-detect the Streamlit app and deploy

### Option 4: Local Network

Share on your local network:
```bash
streamlit run gld_app.py --server.address 0.0.0.0
```

## Security Note

The app fetches public market data only. No authentication or sensitive data handling is required.