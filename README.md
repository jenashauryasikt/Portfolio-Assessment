# Portfolio Management and Forecasting

**Assumptions:**

- No historical data of stocks available, only user portfolios are available
- Users are intelligent

**Target:**

- Given a portfolio, predict its returns for the next quarter (90 days)

## Data Scraping

- Scrapes data from 2021-01-01 to 2023-12-31 NYSE on a daily scale (exceptions of holidays) for 10 stocks each across 4 sectors - Tech, Finance, Healthcare, Consumer Goods. The columns are:
  - Date
  - Open
  - High
  - Low
  - Close
  - Adjusted Close
  - Volume
  - Dividends
  - Symbol
  - Sector

## Portfolio Maker

- Creates 200 portfolios with division based on sector of focus and frequency of trades
- Each portfolio has following columns:
  - Date: Date on which user executed a trade (buy/sell/both)
  - Symbol: Stock Symbol listed on NYSE
  - Sector: The main industry
  - Buy Price: Random price between low and high
  - Buy Volume: Noisily adjusted for frequency
  - Sell Price: Random price between Buy Price and high, assumes user intelligence
  - Sell Volume: Random volume acceptable within total holdings of current stock
  - Dividends
  - Stock shares: Total holdings of that stock symbol
  - Investment Total: Money currently invested in the market/exchange
  - Returns: Net profit if selling all holdings at date's close

## Pre-processing

- Introduces random NaN values and their handling via:
  - Forward Fill: For expected time-growth attributes
  - Central Value: Mean/Median/Mode of itself/neighbors if uncorrelated and random wrt time
  - Regression: Based on correlated attributes from Pearson Correlation Matrix
- Feature Engineering: Introduce Rolling average returns (over past 90 days)

## Baselines

### Self Projection

Extrapolate own portfolio's most recent quarter returns onto next quarter by using slope of returns/day

### K-Nearest Neighbors

Extrapolate by regression by K-nearest neighbors' most recent quarter growth patterns for current portfolio onto next quarter

## Methods

Based on attributes from K-nearest neighbors

### LSTM (RNN) for Time Series Analysis

- 2-layer LSTM for TSA
- Rudimentary hyperparameter tuning (computational limits)
- Uses L2, dropout, early stopping
- Scope for improvement

### LGBM

- Rudimentary hyperparameter tuning:
  - No. of leaves ~ 1.5 xto 2x (no. of features)
  - Max Depth ~ log_2 (1.5x No. of leaves)
  - N-iterators and learning rate: based on size of dataset

## Performance

| Portfolio type     | Self    | KNN               | LSTM    | LGBM              |
| ------------------ | ------- | ----------------- | ------- | ----------------- |
| Finance/Daily      | 2191.17 | **631.21**  | 1364.18 | 726.28            |
| Tech/Weekly        | 4173.79 | 1396.49           | 2975.63 | **1142.22** |
| Healthcare/Monthly | 1285.52 | 1178.93           | 1402.39 | **1024.11** |
| Uniform/Rarely     | 2100.54 | **1020.93** | 1258.83 | 1263.48           |

These are the RMSE for predicted returns on unscaled values. Each row has a different range and should be considered independently.

LSTM and LGBM underwent basic hyperparameter tuning, extensive grid search or Bayesian Optimization was not performed (time constraints).

- With less frequency, data is greatly reduced and all models converge to a close performance range.
- KNN is simple and elegant. However not expected scale well to real-world data
- LSTM can be robust with tuned architectures and hyperparameters.
- LGBM offers feature understanding, is fast, robust, and will scale well to real-world data.
- The dataset needs better feature engineering to incorporate more features.
- Historical data can greatly aid in improving projections of returns due to more quantity and granularity of data.

# Loss Formulation for Data-to-Text Generation

Better than contrastive loss:

## Reconstruction Loss

1. Data-Text-Data Cycle:
   1. *d*: input triples
   2. *t*: intermediate text
   3. *d_i*: i-th token of reconstructed triple
   4. *|d|*: length of sequence
   5. **L_d = (-1/|d|) x Sum from (i=0) to |d|: log (d_i | d_0, ..., d_{i-1}, t)**
2. Text-Data-Text Cycle:
   1. *t*: Input sentence
   2. *d*: Intermediate generated triples
   3. *t_i*: i-th token of reconstructed sentence
   4. *|t|: length of sequence*
   5. **L_t = (-1/|t|) x Sum from (i=0) to |t|: log (t_i | t_0, ..., t_{i-1}, d)**
