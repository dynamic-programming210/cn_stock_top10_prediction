# Chinese Stock Top-10 Predictor

🇨🇳 中国A股 Top-10 预测器 - Predicting the top 10 A-share stocks most likely to outperform over the next 5 trading days.

## Features

- **Daily Predictions**: Top 10 stocks from Shanghai (SHG) and Shenzhen (SHE) exchanges
- **Machine Learning**: Two-stage model (LightGBM Ranker + XGBoost Regressor)
- **Confidence Intervals**: 90% confidence ranges for predictions
- **Interactive Web App**: Streamlit-based dashboard with charts

## Data Source

Uses [EODHD API](https://eodhd.com) for:
- Historical EOD price data
- Exchange symbol lists
- Fundamental data

## Project Structure

```
cn_stock/
├── config.py              # Configuration and settings
├── data/
│   ├── fetch_eodhd.py     # EODHD API client and data fetching
├── features/
│   ├── build_features.py  # Feature engineering
├── models/
│   ├── train.py           # Two-stage model training
│   ├── confidence.py      # Confidence interval estimation
├── app/
│   ├── update_daily.py    # Daily update pipeline
│   ├── web.py             # Streamlit web app
├── outputs/               # Predictions and reports
└── requirements.txt
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set API Token (Optional)

By default, the API token is configured in `config.py`. To use your own:

```bash
export EODHD_API_TOKEN=your_token_here
```

### 3. Initial Setup

Run the full pipeline with initial data fetch and model training:

```bash
python app/update_daily.py --setup
```

For testing with limited symbols:

```bash
python app/update_daily.py --test
```

### 4. Launch Web App

```bash
streamlit run app/web.py
```

## Usage

### Daily Update

```bash
# Full update (fetch new data + generate predictions)
python app/update_daily.py

# Skip data fetching (use existing data)
python app/update_daily.py --skip-data

# Force full refresh
python app/update_daily.py --full-refresh
```

### Individual Components

```bash
# Fetch data only
python data/fetch_eodhd.py --full-refresh

# Build features only
python features/build_features.py

# Train model only
python models/train.py --retrain
```

## Covered Stocks

| Exchange | Code | Board |
|----------|------|-------|
| Shanghai Stock Exchange | SHG | 600/601/603/605 (Main Board) |
| Shenzhen Stock Exchange | SHE | 000/001/002 (Main Board) |

## API Endpoints Used

- `https://eodhd.com/api/exchange-symbol-list/{exchange}` - Get symbols list
- `https://eodhd.com/api/eod/{symbol}.{exchange}` - Historical EOD data
- `https://eodhd.com/api/fundamentals/{symbol}.{exchange}` - Fundamental data

## Model Details

### Two-Stage Approach

1. **Stage 1 - LightGBM Ranker**: Ranks stocks by likelihood to outperform
2. **Stage 2 - XGBoost Regressor**: Predicts actual 5-day forward returns

### Features (40+)

- Return features (1, 3, 5, 10, 20 days)
- Volatility measures
- Volume signals
- Price momentum
- High/low range indicators
- Candlestick patterns
- Cross-sectional rank features

## ⚠️ Disclaimer

This tool is for educational and research purposes only. It does not constitute investment advice. Always do your own research before making investment decisions.

---

Built with ❤️ using Python, LightGBM, XGBoost, and Streamlit
