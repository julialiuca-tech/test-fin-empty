# SEC Data Exploration and Stock Price Analysis

This repository contains Python tools for downloading, processing, and analyzing SEC XBRL financial data combined with stock price information from Yahoo Finance. The project enables comprehensive financial analysis by correlating SEC filings with stock price trends.

## Overview

The project processes SEC XBRL data to extract financial features and correlates them with stock price movements to build predictive models. It includes data download, feature engineering, stock price analysis, and machine learning components.

## Project Structure

### Data Directories
- `data/` - Raw SEC XBRL data files (organized by quarters)
- `processed_data/` - Featurized and processed financial data
- `stock_data/` - Stock price data and trend labels from Yahoo Finance
- `notebooks/` - Jupyter notebooks for data exploration and analysis

### Python Files

#### Data Download and Population
- **`populate_data.py`** - Downloads SEC XBRL data to local `/data/` directory. This is a sample implementation that downloads only the 4 quarters of 2022 data for demonstration purposes.

- **`Populate_All_Data.py`** - Enhanced version of `populate_data.py` with expanded capabilities to download data from many more quarters. Designed for comprehensive data collection across multiple years.

#### Data Exploration and Analysis
- **`exploreTags.py`** - Explores and analyzes tags within the XBRL data. Helps identify the most relevant financial metrics and tags for feature engineering.

- **`featurize.py`** - Processes top tags from 10-Q and 10-K reports to create numerical features. Performs featurization without stitching data across quarters. Outputs featurized data to the `processed_data/` directory.

- **`validate_featurized_data.py`** - Debugging and validation tool for the featurization process. Ensures data quality and correctness of the feature engineering pipeline.

#### Stock Price Analysis
- **`download_sec_yf.py`** - Downloads stock price data using Yahoo Finance API and computes price trend labels with 1 or 3-month look-ahead horizons. Handles CIK-to-ticker mapping, batch processing, and trend calculation. Outputs stock price data and trend labels to the `stock_data/` directory. Note: Yahoo Finance may impose rate limit. May need to refresh IP address. 

#### Machine Learning
- **`baseline_model.py`** - Implements baseline machine learning models for predicting stock price trends using the featurized financial data.

### Utility Files
- **`utility_data.py`** - Contains utility functions for data processing and CIK-to-ticker mapping.

## Requirements

- Python 3.x
- pandas
- numpy
- matplotlib
- seaborn
- requests
- yfinance
- scikit-learn

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd SEC_data_explore
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

### 1. Download SEC Data
```bash
# For sample data (2022 quarters only)
python populate_data.py

# For comprehensive data collection
python Populate_All_Data.py
```

### 2. Explore and Featurize Data
```bash
# Explore XBRL tags
python exploreTags.py

# Create features from financial data
python featurize.py

# Validate featurization results
python validate_featurized_data.py
```

### 3. Download Stock Prices and Calculate Trends
```bash
# Download stock data and compute trend labels
python download_sec_yf.py
```

### 4. Build Machine Learning Models
```bash
# Train baseline models
python baseline_model.py
```

## Data Flow

1. **Raw Data Collection**: SEC XBRL data is downloaded and stored in `data/`
2. **Feature Engineering**: Financial metrics are extracted and featurized, stored in `processed_data/`
3. **Stock Price Analysis**: Stock prices are downloaded and trend labels are computed, stored in `stock_data/`
4. **Model Training**: Machine learning models are trained using the combined dataset

## Key Features

- **Automated Data Pipeline**: End-to-end processing from raw SEC data to ML-ready features
- **Stock Price Integration**: Correlates financial filings with actual stock performance
- **Trend Analysis**: Computes price trends with configurable look-ahead horizons
- **Data Quality Control**: Includes validation and cleaning steps for robust analysis
- **Scalable Processing**: Handles large datasets with batch processing and rate limiting

## Notes

- The project includes rate limiting for API calls to respect service limits
- Data processing is designed to handle missing or incomplete financial data
- Stock price analysis includes handling of delisted companies and OTC market transitions
- All intermediate results are saved for reproducibility and debugging

## Contributing

Suggestions and improvements are welcome. Please ensure any changes maintain data quality and processing efficiency.